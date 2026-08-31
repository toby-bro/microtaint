/*
 * cf_leak.cpp -- libdft64 control-flow-leak detector for the constant-time
 *                square-and-multiply evaluation (RQ7 workload 1).
 *
 * Detection question: how many times does a TAINTED value drive a conditional
 * branch?  i.e. a Jcc whose consumed RFLAGS were last written by an instruction
 * that read tainted operands -- the standard control-flow side-channel signal.
 *
 * libdft64 does not carry a tag on the RFLAGS register itself, so we model
 * "flag taint" explicitly and soundly:
 *   - On every instruction that WRITES RFLAGS (cmp/test/and/sub/...), we OR the
 *     taint of its source operands (register reads + a memory read, if any) and
 *     latch it into g_flags_tainted, remembering the defining PC.
 *   - On every conditional branch (XED_CATEGORY_COND_BR) we read g_flags_tainted
 *     at IPOINT_BEFORE; if set, the branch is driven by tainted data -> a leak.
 *
 * Taint source: stdin (fd 0) via libdft's file-syscall hooks (hook_file_syscall).
 * The harness reads the 4-byte exponent from stdin, so those bytes are the
 * initial taint; base and modulus are compiled-in constants (clean).
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

static KNOB<std::string> KnobLog(KNOB_MODE_WRITEONCE, "pintool", "cflog",
                                 "cf_leak.log", "leak log path");

static FILE *g_log = NULL;

/* latched taint of the live RFLAGS, and the PC that defined them */
static bool g_flags_tainted = false;
static ADDRINT g_flags_def_pc = 0;

/* counters */
static unsigned long g_cond_exec = 0;   /* conditional branches executed      */
static unsigned long g_cond_tainted = 0;/* of those, driven by tainted flags  */
static unsigned long g_flag_writes = 0;  /* flag-writing insns executed        */

/* per-tainted-branch-site execution counts (small map, single-threaded) */
#define MAXSITE 32
static ADDRINT g_site_pc[MAXSITE];
static unsigned long g_site_cnt[MAXSITE];
static unsigned g_nsite = 0;

static VOID note_site(ADDRINT pc) {
    for (unsigned i = 0; i < g_nsite; i++)
        if (g_site_pc[i] == pc) { g_site_cnt[i]++; return; }
    if (g_nsite < MAXSITE) { g_site_pc[g_nsite] = pc; g_site_cnt[g_nsite] = 1; g_nsite++; }
}

/* IPOINT_BEFORE on a flag-writing instruction: OR source-operand taint. */
static VOID FlagWrite(THREADID tid, ADDRINT pc,
                      UINT32 r0, UINT32 r1, UINT32 r2, UINT32 rn0, UINT32 rn1,
                      UINT32 rn2, BOOL hasMem, ADDRINT memEA, UINT32 memSize) {
    g_flag_writes++;
    bool t = false;
    if (r0 != (UINT32)-1 && !tag_is_empty(tagmap_getn_reg(tid, r0, rn0))) t = true;
    if (r1 != (UINT32)-1 && !tag_is_empty(tagmap_getn_reg(tid, r1, rn1))) t = true;
    if (r2 != (UINT32)-1 && !tag_is_empty(tagmap_getn_reg(tid, r2, rn2))) t = true;
    if (hasMem && memSize && !tag_is_empty(tagmap_getn(memEA, memSize))) t = true;
    g_flags_tainted = t;
    g_flags_def_pc = pc;
}

/* IPOINT_BEFORE on a conditional branch: is it driven by tainted flags? */
static VOID CondBranch(ADDRINT pc) {
    g_cond_exec++;
    if (g_flags_tainted) {
        g_cond_tainted++;
        note_site(pc);
        if (g_log) {
            fprintf(g_log,
                    "CF-LEAK jcc_pc=0x%lx flags_def_pc=0x%lx tainted=1\n",
                    (unsigned long)pc, (unsigned long)g_flags_def_pc);
            fflush(g_log);
        }
    }
}

static VOID Instruction(INS ins, VOID *v) {
    ADDRINT pc = INS_Address(ins);

    /* (1) flag-writing instructions: latch source-operand taint into flags */
    if (INS_RegWContain(ins, REG_RFLAGS)) {
        UINT32 rid[3] = {(UINT32)-1, (UINT32)-1, (UINT32)-1};
        UINT32 rn[3]  = {0, 0, 0};
        unsigned k = 0;
        UINT32 nr = INS_MaxNumRRegs(ins);
        for (UINT32 i = 0; i < nr && k < 3; i++) {
            REG r = INS_RegR(ins, i);
            if (!REG_valid(r)) continue;
            REG fr = REG_FullRegName(r);
            if (fr == REG_RFLAGS || fr == REG_RIP) continue;
            if (!REG_is_gr(fr) && !REG_is_xmm(fr) && !REG_is_ymm(fr)) continue;
            rid[k] = (UINT32)REG_INDX(fr);
            rn[k]  = (UINT32)REG_Size(fr);
            if (rn[k] > 8) rn[k] = 8;
            k++;
        }
        BOOL hasMem = INS_IsMemoryRead(ins);
        if (hasMem) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)FlagWrite,
                           IARG_THREAD_ID, IARG_INST_PTR,
                           IARG_UINT32, rid[0], IARG_UINT32, rid[1],
                           IARG_UINT32, rid[2], IARG_UINT32, rn[0],
                           IARG_UINT32, rn[1], IARG_UINT32, rn[2],
                           IARG_BOOL, TRUE, IARG_MEMORYREAD_EA,
                           IARG_MEMORYREAD_SIZE, IARG_END);
        } else {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)FlagWrite,
                           IARG_THREAD_ID, IARG_INST_PTR,
                           IARG_UINT32, rid[0], IARG_UINT32, rid[1],
                           IARG_UINT32, rid[2], IARG_UINT32, rn[0],
                           IARG_UINT32, rn[1], IARG_UINT32, rn[2],
                           IARG_BOOL, FALSE, IARG_ADDRINT, (ADDRINT)0,
                           IARG_UINT32, 0, IARG_END);
        }
    }

    /* (2) conditional branches: is the consumed flag tainted? */
    if (INS_Category(ins) == XED_CATEGORY_COND_BR) {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)CondBranch,
                       IARG_INST_PTR, IARG_END);
    }
}

static VOID Fini(INT32 code, VOID *v) {
    fprintf(stderr,
            "[cf_leak] cond_branches_executed=%lu tainted_cond_branches=%lu "
            "flag_writes=%lu tainted_branch_sites=%u\n",
            g_cond_exec, g_cond_tainted, g_flag_writes, g_nsite);
    for (unsigned i = 0; i < g_nsite; i++)
        fprintf(stderr, "[cf_leak]   site jcc_pc=0x%lx executions_tainted=%lu\n",
                (unsigned long)g_site_pc[i], g_site_cnt[i]);
    if (g_log) fclose(g_log);
}

int main(int argc, char *argv[]) {
    PIN_InitSymbols();
    if (unlikely(PIN_Init(argc, argv))) {
        fprintf(stderr, "[cf_leak] PIN_Init failed\n");
        return -1;
    }
    if (unlikely(libdft_init() != 0)) {
        fprintf(stderr, "[cf_leak] libdft_init failed\n");
        return -1;
    }
    g_log = fopen(KnobLog.Value().c_str(), "w");
    hook_file_syscall();   /* stdin (fd 0) -> fuzzing fd -> tainted bytes */
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
