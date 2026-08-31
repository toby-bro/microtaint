/*
 * canary_taint.cpp -- libdft64 DTA pintool for the nft_byteorder evaluation.
 *
 * Uniform detection question (shared across all 7 tools, see common.py):
 *   Does attacker-tainted stdin data reach a WRITE into the canary_after
 *   sentinel region [canary_after, canary_after + size)?
 *
 * Method
 * ------
 *  - Taint source: libdft's built-in file-syscall hooks (hook_file_syscall()).
 *    STDIN_FILENO is treated as a fuzzing fd, so every byte the harness reads
 *    from fd 0 via read()/fread() is byte-precisely tainted in the tagmap.
 *  - Sink: every instruction that writes memory is instrumented. When the
 *    effective address lands inside the baked canary_after range, we emit a
 *    STORE line and record:
 *      * vtainted  -- is the VALUE stored attacker-tainted? Checked at
 *                     IPOINT_AFTER by reading the tagmap at EA, which libdft
 *                     has already populated from the store's source operand.
 *      * atainted  -- were the ADDRESS registers (base/index of the write
 *                     memory operand) attacker-tainted? (data-flow only)
 *
 * The canary_after range is baked from common.recover_layout():
 *   base(L)=0x4b7ac0 -> canary_after = base+256+80 = 0x4b7c10, size 256,
 *   i.e. [0x4b7c10, 0x4b7d10).  Overridable via -lo / -hi knobs.
 *
 * Built -static -no-pie, so these link-time addresses are the runtime
 * addresses under Pin as well; identical for ./harness and ./harness_fixed.
 */

#include <stdio.h>
#include <stdint.h>
#include <string>

#include "pin.H"

#include "branch_pred.h"
#include "libdft_api.h"
#include "libdft_core.h"
#include "syscall_hook.h"
#include "tagmap.h"
#include "ins_helper.h"

/* ---- baked sink range (canary_after) -------------------------------- */
#define CANARY_LO_DFL 0x4b7c10UL
#define CANARY_HI_DFL 0x4b7d10UL /* 0x4b7c10 + 256 */

static KNOB<UINT64> KnobLo(KNOB_MODE_WRITEONCE, "pintool", "canlo",
                           "4946960" /* 0x4b7c10 */, "canary_after low addr");
static KNOB<UINT64> KnobHi(KNOB_MODE_WRITEONCE, "pintool", "canhi",
                           "4947216" /* 0x4b7d10 */, "canary_after high addr");
/* in-bounds register-file range [regs, regs+80); logged alongside the sink so
 * the driver can measure the 20 in-bounds byteswap stores for an under-taint
 * cross-check. Defaults match base(L)=0x4b7ac0 -> regs=0x4b7bc0. */
static KNOB<UINT64> KnobRegLo(KNOB_MODE_WRITEONCE, "pintool", "reglo",
                              "4946880" /* 0x4b7bc0 */, "regs low addr");
static KNOB<UINT64> KnobRegHi(KNOB_MODE_WRITEONCE, "pintool", "reghi",
                              "4946960" /* 0x4b7c10 */, "regs high addr");
static KNOB<std::string> KnobLog(KNOB_MODE_WRITEONCE, "pintool", "canlog",
                                 "libdft_stores.log", "STORE log path");
/* -candbg 1 emits per-instruction register-taint probes along the buggy
 * loop's data-flow path (hardcoded PCs, this harness only). Off by default. */
static KNOB<UINT64> KnobDbg(KNOB_MODE_WRITEONCE, "pintool", "candbg", "0",
                            "emit data-flow probes for the buggy loop");

static ADDRINT g_lo = CANARY_LO_DFL;
static ADDRINT g_hi = CANARY_HI_DFL;
static ADDRINT g_reglo = 0x4b7bc0UL;
static ADDRINT g_reghi = 0x4b7c10UL;

/* ---- per-store scratch (single-threaded harness) -------------------- */
static ADDRINT g_ea = 0;
static UINT32 g_size = 0;
static ADDRINT g_pc = 0;
static bool g_addr_tainted = false;

/* ---- counters ------------------------------------------------------- */
static FILE *g_log = NULL;
static unsigned long g_n_oob = 0;         /* writes landing in canary_after */
static unsigned long g_n_oob_tainted = 0; /* of those, VALUE was tainted    */
static unsigned long g_n_oob_addr = 0;    /* of those, ADDRESS regs tainted */
static unsigned long g_n_ib = 0;          /* writes landing in regs file    */
static unsigned long g_n_ib_tainted = 0;  /* of those, VALUE was tainted    */
static ADDRINT g_first_tainted_pc = 0;

/* IPOINT_BEFORE: capture EA/size/pc and evaluate address-register taint. */
static VOID RecordWrite(ADDRINT ea, UINT32 size, ADDRINT pc, THREADID tid,
                        UINT32 base_indx, UINT32 index_indx, BOOL has_base,
                        BOOL has_index) {
  g_ea = ea;
  g_size = size;
  g_pc = pc;
  bool at = false;
  if (has_base && !tag_is_empty(tagmap_getn_reg(tid, base_indx, 8)))
    at = true;
  if (has_index && !tag_is_empty(tagmap_getn_reg(tid, index_indx, 8)))
    at = true;
  g_addr_tainted = at;
}

/* IPOINT_AFTER: the store has executed and libdft has propagated the
 * source operand's tag into tagmap[EA]; read it back to learn value taint. */
static VOID CheckWrite() {
  bool in_oob = !(g_ea + g_size <= g_lo || g_ea >= g_hi);
  bool in_ib = !(g_ea + g_size <= g_reglo || g_ea >= g_reghi);
  if (!in_oob && !in_ib)
    return; /* neither sentinel nor register file */

  tag_t t = tagmap_getn(g_ea, g_size);
  bool vt = !tag_is_empty(t);

  const char *region;
  long off;
  if (in_oob) {
    region = "canary_after";
    off = (long)(g_ea - g_lo);
    g_n_oob++;
    if (vt) {
      g_n_oob_tainted++;
      if (g_first_tainted_pc == 0)
        g_first_tainted_pc = g_pc;
    }
    if (g_addr_tainted)
      g_n_oob_addr++;
  } else {
    region = "regs";
    off = (long)(g_ea - g_reglo);
    g_n_ib++;
    if (vt)
      g_n_ib_tainted++;
  }

  if (g_log) {
    fprintf(g_log,
            "STORE region=%s ea=0x%lx size=%u vtainted=%d atainted=%d "
            "pc=0x%lx off=%ld\n",
            region, (unsigned long)g_ea, g_size, vt ? 1 : 0,
            g_addr_tainted ? 1 : 0, (unsigned long)g_pc, off);
    fflush(g_log);
  }
}

/* ---- DEBUG: per-PC register-taint probe ----------------------------- */
static VOID Probe(THREADID tid, ADDRINT pc, UINT32 regidx, UINT32 n) {
  tag_t t = tagmap_getn_reg(tid, regidx, n);
  fprintf(stderr, "[probe] pc=0x%lx reg=%u tainted=%d\n", (unsigned long)pc,
          regidx, tag_is_empty(t) ? 0 : 1);
}

static bool g_dbg = false;

static VOID Instruction(INS ins, VOID *v) {
  ADDRINT pc = INS_Address(ins);
  /* DEBUG probes along the tainted value's data-flow path (opt-in) */
  if (g_dbg) {
  if (pc == 0x4034aa) /* movzwl (%rax),%eax  : after load, eax(RAX=10) */
    INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)Probe, IARG_THREAD_ID,
                   IARG_INST_PTR, IARG_UINT32, 10, IARG_UINT32, 2, IARG_END);
  if (pc == 0x4034c3) /* mov %eax,%edi        : after, edi(RDI=3) */
    INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)Probe, IARG_THREAD_ID,
                   IARG_INST_PTR, IARG_UINT32, 3, IARG_UINT32, 2, IARG_END);
  if (pc == 0x4231a4) /* mov %edi,%eax (htons): after, eax */
    INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)Probe, IARG_THREAD_ID,
                   IARG_INST_PTR, IARG_UINT32, 10, IARG_UINT32, 2, IARG_END);
  if (pc == 0x4231a6) /* rol $8,%ax (htons)  : after, eax */
    INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)Probe, IARG_THREAD_ID,
                   IARG_INST_PTR, IARG_UINT32, 10, IARG_UINT32, 2, IARG_END);
  if (pc == 0x4034ca) /* mov %ax,(%rbx)      : before store, ax(RAX=10) */
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)Probe, IARG_THREAD_ID,
                   IARG_INST_PTR, IARG_UINT32, 10, IARG_UINT32, 2, IARG_END);
  }

  if (!INS_IsMemoryWrite(ins))
    return;

  REG base = INS_MemoryBaseReg(ins);
  REG index = INS_MemoryIndexReg(ins);
  BOOL has_base = REG_valid(base);
  BOOL has_index = REG_valid(index);
  UINT32 base_indx = has_base ? (UINT32)REG_INDX(base) : 0;
  UINT32 index_indx = has_index ? (UINT32)REG_INDX(index) : 0;

  INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordWrite, IARG_MEMORYWRITE_EA,
                 IARG_MEMORYWRITE_SIZE, IARG_INST_PTR, IARG_THREAD_ID,
                 IARG_UINT32, base_indx, IARG_UINT32, index_indx, IARG_BOOL,
                 has_base, IARG_BOOL, has_index, IARG_END);

  if (INS_IsValidForIpointAfter(ins))
    INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)CheckWrite, IARG_END);
  else if (INS_IsValidForIpointTakenBranch(ins))
    INS_InsertCall(ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)CheckWrite, IARG_END);
}

static VOID Fini(INT32 code, VOID *v) {
  /* diagnostic: how many bytes of the regs region [g_lo-80, g_lo) are
   * tainted at end of run? tells us whether the source reached the
   * register file at all (i.e. whether fread's copy preserved taint). */
  unsigned regs_tainted = 0;
  for (ADDRINT a = g_lo - 80; a < g_lo; a++)
    if (!tag_is_empty(tagmap_getb(a)))
      regs_tainted++;
  fprintf(stderr,
          "[canary_taint] oob_range=[0x%lx,0x%lx) oob_writes=%lu "
          "oob_value_tainted=%lu oob_addr_tainted=%lu first_tainted_pc=0x%lx "
          "ib_range=[0x%lx,0x%lx) ib_writes=%lu ib_value_tainted=%lu "
          "regs_tainted_bytes=%u/80\n",
          (unsigned long)g_lo, (unsigned long)g_hi, g_n_oob, g_n_oob_tainted,
          g_n_oob_addr, (unsigned long)g_first_tainted_pc,
          (unsigned long)g_reglo, (unsigned long)g_reghi, g_n_ib,
          g_n_ib_tainted, regs_tainted);
  if (g_log)
    fclose(g_log);
}

int main(int argc, char *argv[]) {
  PIN_InitSymbols();

  if (unlikely(PIN_Init(argc, argv))) {
    fprintf(stderr, "[canary_taint] PIN_Init failed\n");
    return -1;
  }

  if (unlikely(libdft_init() != 0)) {
    fprintf(stderr, "[canary_taint] libdft_init failed\n");
    return -1;
  }

  g_lo = (ADDRINT)KnobLo.Value();
  g_hi = (ADDRINT)KnobHi.Value();
  g_reglo = (ADDRINT)KnobRegLo.Value();
  g_reghi = (ADDRINT)KnobRegHi.Value();
  g_dbg = KnobDbg.Value() != 0;
  g_log = fopen(KnobLog.Value().c_str(), "w");

  /* taint-source: stdin (fd 0) is a fuzzing fd inside libdft's file hooks */
  hook_file_syscall();

  INS_AddInstrumentFunction(Instruction, 0);
  PIN_AddFiniFunction(Fini, 0);

  PIN_StartProgram();
  return 0;
}
