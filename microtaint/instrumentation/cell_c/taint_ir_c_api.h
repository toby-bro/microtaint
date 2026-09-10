/* The taint-IR interpreter, for other extension modules.
 *
 * The host emitter declines a few opcodes on purpose (division and
 * count-leading-zeros want fixed registers or a CPU feature check out of
 * proportion to how rarely a taint rule reaches them), and the contract has
 * always been that the caller keeps the interpreter for those.  The BLOCK
 * runtime is a different module, so it needs the interpreter through a
 * capsule; without one it turned "run this interpreted" into "skip this
 * block", and a skipped block is taint that is never computed.
 */
#ifndef MICROTAINT_TAINT_IR_C_API_H
#define MICROTAINT_TAINT_IR_C_API_H

#include <stdint.h>

typedef struct {
    /* Same shape as an emitted program: (values, taint, out_taint).  `prog`
     * is the IRProgC the capsule for that program holds. */
    void (*run)(const void *prog, const uint64_t *values,
                const uint64_t *taints, uint64_t *out_taints);
} TaintIrCAPI;

#endif
