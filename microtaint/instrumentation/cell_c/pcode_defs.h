#ifndef PCODE_DEFS_H
#define PCODE_DEFS_H

/*
 * pcode_defs.h — the p-code vocabulary shared by the value interpreter
 * (cell_core.h) and the taint composer (taint_core.h).
 *
 * Split out so the two can be included in either order: taint_core.h needs the
 * opcode enum and the decoded-op record, but must not pull in the frame and
 * execution machinery it does not use.
 */

#include <stdint.h>

#define SP_CONST     0
#define SP_REGISTER  1
#define SP_UNIQUE    2
#define SP_RAM       3
#define SP_OTHER    -1
#define NO_OUT_SPACE -2

/* Opcode enum — must match _OpcodeID in cell.pyx exactly */
typedef enum {
    OP_UNKNOWN=0, OP_COPY, OP_LOAD, OP_STORE, OP_MULTIEQUAL, OP_INDIRECT,
    OP_INT_ADD, OP_INT_SUB, OP_INT_MULT, OP_INT_DIV, OP_INT_SDIV,
    OP_INT_REM, OP_INT_SREM, OP_INT_2COMP, OP_INT_NEGATE,
    OP_INT_AND, OP_INT_OR, OP_INT_XOR,
    OP_INT_LEFT, OP_INT_RIGHT, OP_INT_SRIGHT,
    OP_INT_EQUAL, OP_INT_NOTEQUAL,
    OP_INT_LESS, OP_INT_LESSEQUAL, OP_INT_SLESS, OP_INT_SLESSEQUAL,
    OP_INT_CARRY, OP_INT_SCARRY, OP_INT_SBORROW,
    OP_INT_ZEXT, OP_INT_SEXT, OP_INT_TRUNC, OP_CAST,
    OP_POPCOUNT, OP_LZCOUNT,
    OP_PIECE, OP_SUBPIECE, OP_PTRADD, OP_PTRSUB,
    OP_BOOL_AND, OP_BOOL_OR, OP_BOOL_XOR, OP_BOOL_NEGATE,
    OP_BRANCH, OP_CBRANCH, OP_BRANCHIND, OP_CALL, OP_CALLIND,
    OP_CALLOTHER, OP_RETURN, OP_IMARK, OP_UNIMPLEMENTED,
    OP_SEGMENT, OP_CPOOLREF, OP_NEW, OP_INSERT, OP_EXTRACT,
    OP_FLOAT_ANY, OP_TRUNC_FLOAT
} OpcodeID;

/* Pre-decoded P-code op (mirrors PCodeOp in cell.pyx) */
typedef struct {
    int           oid;
    int           o_sp;
    unsigned long o_off;
    int           o_sz;
    int           callother_out;
    int           n_ins;
    int           i0_sp;  unsigned long i0_off;  int i0_sz;
    int           i1_sp;  unsigned long i1_off;  int i1_sz;
    int           i2_sp;  unsigned long i2_off;  int i2_sz;
} PCOp;

#endif /* PCODE_DEFS_H */
