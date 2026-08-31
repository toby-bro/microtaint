/*
 * dns_taint.cpp -- libdft64 sub-byte granularity probe (RQ7 workload 2).
 *
 * The harness (dns_bitfield) reads ONE byte from stdin and runs the LDNS
 * OPCODE extraction on AL:
 *     0x...: AND AL, 0x78   (24 78)     keep OPCODE bits 6..3, drop QR bit 7
 *     0x...: SHR AL, 3      (C0 E8 03)  right-align OPCODE
 *
 * libdft is BYTE-granular: the finest source it can express is the whole flag
 * byte.  We taint that byte (via stdin file hooks) and read back AL's tag after
 * each of the two instructions.  Because QR and OPCODE live in the SAME byte,
 * libdft taints them together; it cannot express "QR only" vs "OPCODE only".
 *
 * We locate the two instructions by their exact encodings (24 78 ; C0 E8 03)
 * so the probe is independent of link addresses.
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

static KNOB<std::string> KnobLog(KNOB_MODE_WRITEONCE, "pintool", "dnslog",
                                  "dns_taint.log", "granularity log path");
static FILE *g_log = NULL;

static ADDRINT g_and_pc = 0, g_shr_pc = 0;
static int g_al_before = -1, g_al_after_and = -1, g_al_after_shr = -1;

static void logline(const char *s) {
    fprintf(stderr, "[dns_taint] %s\n", s);
    if (g_log) { fprintf(g_log, "%s\n", s); fflush(g_log); }
}

/* AL is byte 0 of RAX; check that single byte's tag. */
static bool al_tainted(THREADID tid) {
    return !tag_is_empty(tagmap_getb_reg(tid, REG_INDX(REG_RAX), 0));
}

static VOID BeforeAnd(THREADID tid, ADDRINT pc) {
    g_al_before = al_tainted(tid) ? 1 : 0;
    char b[128];
    snprintf(b, sizeof b, "AND AL,0x78 @0x%lx : AL tainted BEFORE = %d",
             (unsigned long)pc, g_al_before);
    logline(b);
}
static VOID AfterAnd(THREADID tid, ADDRINT pc) {
    g_al_after_and = al_tainted(tid) ? 1 : 0;
    char b[128];
    snprintf(b, sizeof b, "AND AL,0x78 @0x%lx : AL tainted AFTER  = %d "
             "(QR bit7 masked off; OPCODE bits6..3 kept)",
             (unsigned long)pc, g_al_after_and);
    logline(b);
}
static VOID AfterShr(THREADID tid, ADDRINT pc) {
    g_al_after_shr = al_tainted(tid) ? 1 : 0;
    char b[160];
    snprintf(b, sizeof b, "SHR AL,3   @0x%lx : AL(=extracted OPCODE) tainted "
             "AFTER = %d", (unsigned long)pc, g_al_after_shr);
    logline(b);
}

static VOID Instruction(INS ins, VOID *v) {
    /* match by exact encoding: AND AL,0x78 = 24 78 ; SHR AL,3 = C0 E8 03 */
    ADDRINT pc = INS_Address(ins);
    USIZE sz = INS_Size(ins);
    UINT8 raw[16] = {0};
    if (sz > 0 && sz <= 16)
        PIN_SafeCopy(raw, (const VOID *)pc, sz);

    if (sz == 2 && raw[0] == 0x24 && raw[1] == 0x78) {
        g_and_pc = pc;
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BeforeAnd,
                       IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
        if (INS_IsValidForIpointAfter(ins))
            INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)AfterAnd,
                           IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
    } else if (sz == 3 && raw[0] == 0xC0 && raw[1] == 0xE8 && raw[2] == 0x03) {
        g_shr_pc = pc;
        if (INS_IsValidForIpointAfter(ins))
            INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)AfterShr,
                           IARG_THREAD_ID, IARG_INST_PTR, IARG_END);
    }
}

static VOID Fini(INT32 code, VOID *v) {
    char b[256];
    snprintf(b, sizeof b,
             "SUMMARY and_pc=0x%lx shr_pc=0x%lx al_before=%d al_after_and=%d "
             "al_after_shr=%d",
             (unsigned long)g_and_pc, (unsigned long)g_shr_pc,
             g_al_before, g_al_after_and, g_al_after_shr);
    logline(b);
    if (g_log) fclose(g_log);
}

int main(int argc, char *argv[]) {
    PIN_InitSymbols();
    if (unlikely(PIN_Init(argc, argv))) return -1;
    if (unlikely(libdft_init() != 0)) return -1;
    g_log = fopen(KnobLog.Value().c_str(), "w");
    hook_file_syscall();
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
