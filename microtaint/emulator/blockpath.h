/* Block-at-a-time taint, in C.
 *
 * A UC_HOOK_BLOCK fires once per basic block at 0.67 ns/instruction against
 * ~18 for a per-instruction code hook, and lowering a whole block as one
 * program roughly halves the taint arithmetic because dead-code elimination
 * then sees across the block.  That is the whole reason this exists, so it is
 * C from the start: a Python driver for a hot path cannot collect either win.
 *
 * The runtime here is deliberately free of the hook's context.  It takes an
 * explicit environment, so it can be exercised against a synthetic memory with
 * no emulator at all, and the hook supplies the same fields from its own
 * context.  Planning -- lifting a block and cutting it into regions -- stays in
 * Python: it happens once per distinct block, it is the compiler, and it is not
 * on the hot path.  Everything that runs per block execution is below.
 *
 * Three things shape the design, all of them consequences of the hook firing
 * BEFORE the block runs:
 *
 *   - the emulator cannot be asked for a register value part way through a
 *     block, so the lowering computes them: a region publishes its register
 *     VALUES alongside its taint and the next region reads them;
 *   - a block that faults part way would otherwise commit taint for
 *     instructions that never ran, which is an UNDER-taint, so a block's
 *     result is held pending and committed when the NEXT block proves it
 *     finished;
 *   - a block can store to an address and load it back, which no single
 *     instruction does.  Both the taint AND the value of such a store have to
 *     reach the later load: with the taint alone the load gets the right mask
 *     and a stale value, and any address derived from it points elsewhere.
 */
#ifndef MT_BLOCKPATH_H
#define MT_BLOCKPATH_H

/* No Python.  Not a PyObject, not a refcount, not a GIL acquire anywhere in
 * this file: everything here runs per BLOCK EXECUTION, and the measured cost of
 * touching CPython at that rate is larger than the work being wrapped.  The
 * extension module that wraps this owns the compiled code objects and keeps
 * them alive; the runtime sees only function pointers. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
/* Slot layout                                                            */
/*                                                                        */
/* Distinct from the per-instruction layout in fastpath.h, and            */
/* deliberately so: a block program publishes register VALUES as well as  */
/* taint and needs a stored value per access, neither of which an         */
/* instruction program has.  Keeping the layouts apart means already-     */
/* compiled instruction programs address exactly the words they always    */
/* did.  A program is compiled against one of these and must be run with  */
/* the matching one.                                                      */
/* ---------------------------------------------------------------------- */
#define MT_BLK_VAL_BASE 512 /* register VALUE outputs, slot + this */
#define MT_BLK_MEM_BASE 1024
#define MT_BLK_MAX_ACC 8
#define MT_BLK_PER_ACC 5
#define MT_BLK_SLOTS (MT_BLK_MEM_BASE + MT_BLK_PER_ACC * MT_BLK_MAX_ACC)

#define MT_BLK_A_MEM 0 /* the loaded word, and its shadow */
#define MT_BLK_A_ADDR 1
#define MT_BLK_A_ADDRT 2
#define MT_BLK_A_STTAINT 3
#define MT_BLK_A_STVAL 4

#define MT_BLK_MAX_WRITES 64
#define MT_BLK_MAX_REPORTS 16

/* mt_blk_compute outcomes */
#define MT_BLK_OK 0
#define MT_BLK_DECLINED 1

/* Why a block declined, indexed by reason.  Counted rather than guessed at:
 * a block this path silently skips is unanalysed code, so the share it skips
 * has to be visible. */
#define MT_BLK_MISS_UNPLANNED 0 /* no plan, or a region did not lower */
#define MT_BLK_MISS_SLOTS 1     /* more register slots than the layout holds */
#define MT_BLK_MISS_ACC 2       /* a region makes more accesses than fit */
#define MT_BLK_MISS_WRITES 3    /* more stores than the pending buffer holds */
#define MT_BLK_MISS_MEM 4       /* a guest memory read Unicorn refused */
#define MT_BLK_MISS_REPORTS                                                    \
  5 /* more findings than the pending buffer holds                             \
     */
#define MT_BLK_N_MISS 6

typedef void (*MtBlkFn)(
  const uint64_t *values, const uint64_t *taint, uint64_t *out
);

/* The taint-IR interpreter, for a program the host emitter declined.  It
 * declines division and count-leading-zeros deliberately -- both want fixed
 * registers or a CPU feature check out of proportion to how rarely a taint
 * rule reaches them -- and the contract is that the caller keeps the
 * interpreter for those.  Without one here, a declined program meant a
 * REFUSED block, and a refused block is skipped, so its taint was never
 * computed at all. */
typedef void (*MtBlkInterp)(const void *prog, const uint64_t *values,
                            const uint64_t *taint, uint64_t *out);
static MtBlkInterp mt_blk_interp = NULL;

/* Run one region, however it is carried. */
static inline void mt_blk_call(const void *fn, const void *prog,
                               const uint64_t *values, const uint64_t *taint,
                               uint64_t *out) {
  if (fn) { ((MtBlkFn)fn)(values, taint, out); return; }
  mt_blk_interp(prog, values, taint, out);
}

/* One lowered run of consecutive instructions. */
typedef struct {
  void *fn;      /* MtBlkFn, NULL when the emitter declined this program */
  /* The program itself, for the interpreter.  Non-NULL whenever the region
   * lowered, so `fn == NULL && prog == NULL` is the only unplanned region. */
  void *prog;
  void *addr_prog;   /* the address slice's program, same idea */
  /* The slice of `fn` that computes only the load addresses.  Pass 1 wants
   * nothing else, and the whole program is far larger: measured, 82.7 ops to
   * produce 1.93 addresses.  NULL means run `fn` for pass 1, which is correct
   * and merely slower. */
  void *addr_fn;
  uint64_t addr; /* the region's first guest address */
  uint64_t last; /* ... and its LAST instruction's, which a finding names */
  int n_acc;
  int n_load;  /* of them, how many are loads: pass 1 is for those */
  signed char acc_kind[MT_BLK_MAX_ACC]; /* 0 load, 1 store */
  signed char acc_size[MT_BLK_MAX_ACC];
  signed char acc_needval[MT_BLK_MAX_ACC]; /* the program reads the word */
  /* The register slots whose VALUE this region publishes, and how many.  Only
   * a LATER region of the same block can want one, so it is a handful where
   * the whole register file used to be copied twice: once to seed the value
   * half of the output array so an unwritten slot reads back its current
   * value, and once to thread the published values into the next region.
   * `n_pub < 0` means the plan did not say, and the runtime keeps the old
   * wholesale copies -- a caller that predates this stays correct. */
  int *pub_slots;
  int n_pub;
} MtBlkRegion;

typedef struct {
  int size;      /* the block's byte length: part of its identity */
  int n_regions;
  int handleable; /* every region lowered */
  MtBlkRegion *regions; /* malloc'd */
  /* This block's OWN register read.  Reading the whole file per block is what
   * made the first wired version 2.8x slower than the per-instruction path:
   * measured, 0.392 s of block mode's 0.434 s on bench_untainted went on
   * reading ~88 registers per block, ~37 us each time, while the taint
   * computation the whole thing exists for cost 0.009 s.  A block reads only
   * the registers its regions actually read, which is the same trick
   * `ir_slots` plays per instruction.  The arrays belong to the caller. */
  unsigned long long ids_addr, ptrs_addr, vals_addr;
  int n_calls;   /* uc_reg_read_batch calls; a vector is one call, two slots */
  int n_vals;    /* slots the read fills */
  int *val_slots; /* engine slot for each of those, malloc'd */
  int need_flags; /* the packed flags register is among them */
} MtBlkPlan;

typedef struct {
  uint64_t addr, mask, value;
  int size;
} MtBlkWrite;

/* A block's computed taint, not yet committed.  Nothing here is applied until
 * the block is known to have completed, which is the whole point. */
typedef struct {
  int valid;
  uint64_t address;
  /* The held block's own byte length.  Only here so that a write onto CODE can
   * ask whether it landed on THIS block: a rewrite anywhere else does not make
   * a block that already ran with the bytes it was planned from any less
   * correct, and abandoning it then is an under-taint. */
  int size;
  uint64_t *taint; /* whole post-state, n_taint entries */
  int n_taint;
  int cap_taint;
  MtBlkWrite writes[MT_BLK_MAX_WRITES];
  int n_writes;
  uint64_t rep_addr[MT_BLK_MAX_REPORTS];
  uint64_t rep_mask[MT_BLK_MAX_REPORTS];
  int n_reports;
} MtBlkPending;

/* Everything the runtime needs that is not the plan.  The hook fills this from
 * its own context; a test fills it with a synthetic memory. */
typedef struct {
  int (*mem_read)(void *ctx, uint64_t addr, int size, uint64_t *out);
  void *mem_ctx;
  uint64_t (*shadow_read)(void *shadow, uint64_t addr, int size);
  void (*shadow_write)(void *shadow, uint64_t addr, uint64_t mask, int size);
  void *shadow;
  /* Scratch, MT_BLK_SLOTS entries each.  Owned by the caller so a block does
   * not allocate. */
  uint64_t *sv, *st, *so;
  int pc_slot;         /* -1 when the program counter has no slot */
  int stop_after;      /* diagnostic: 5 stop after pass 1, 6 after the loads */
  unsigned long *miss; /* MT_BLK_N_MISS counters, may be NULL */
  /* The address a declined memory read wanted.  "The read declined" is not
   * actionable; "it declined at this address" says whether the reader is
   * broken or the program computed nonsense. */
  uint64_t last_bad_addr;
  int last_bad_size;
} MtBlkEnv;

static inline void mt_blk_miss(MtBlkEnv *env, int reason) {
  if (env->miss) { env->miss[reason]++; }
}

/* ----------------------------------------------------------------------- */
/* The overlay: what this block has stored but not yet committed.          */
/*                                                                         */
/* Scanned backwards so the most recent store to a byte wins, and byte by  */
/* byte because a store and a later load need not share width or           */
/* alignment.  Linear over the block's own stores, which is a handful:     */
/* a hash would cost more than it saved at this size.                      */
/* ----------------------------------------------------------------------- */
static void mt_blk_overlay(
  const MtBlkWrite *w,
  int n,
  uint64_t addr,
  int size,
  uint64_t *value,
  uint64_t *mask
) {
  for (int i = 0; i < size; i++) {
    const uint64_t byte_addr = addr + (uint64_t)i;
    for (int k = n - 1; k >= 0; k--) {
      if (
        byte_addr < w[k].addr || byte_addr >= w[k].addr + (uint64_t)w[k].size
      ) {
        continue;
      }
      const int off = (int)(byte_addr - w[k].addr);
      const uint64_t vb = (w[k].value >> (8 * off)) & 0xFFu;
      const uint64_t mb = (w[k].mask >> (8 * off)) & 0xFFu;
      *value = (*value & ~(0xFFull << (8 * i))) | (vb << (8 * i));
      *mask = (*mask & ~(0xFFull << (8 * i))) | (mb << (8 * i));
      break;
    }
  }
}

/* ---------------------------------------------------------------------- */
/* Compute one block's taint into `pend`.  Commits nothing.               */
/* ---------------------------------------------------------------------- */
static int mt_blk_compute(
  const MtBlkPlan *plan,
  MtBlkEnv *env,
  const uint64_t *reg_val,
  const uint64_t *reg_taint,
  int n_slots,
  uint64_t address,
  MtBlkPending *pend
) {
  pend->valid = 0;
  pend->n_writes = 0;
  pend->n_reports = 0;
  pend->address = address;
  pend->size = plan ? plan->size : 0;

  if (!plan || !plan->handleable || plan->n_regions <= 0) {
    mt_blk_miss(env, MT_BLK_MISS_UNPLANNED);
    return MT_BLK_DECLINED;
  }
  if (n_slots > MT_BLK_VAL_BASE || n_slots > pend->cap_taint) {
    mt_blk_miss(env, MT_BLK_MISS_SLOTS);
    return MT_BLK_DECLINED;
  }

  const size_t nb = (size_t)n_slots * sizeof(uint64_t);
  /* The working state is `sv` (values) and `st` (taint), held ACROSS the
   * block's regions rather than rebuilt for each one.  Copying the whole slot
   * array per region was over half the cost of computing a block: nine copies
   * of it, where two suffice.
   *
   * `st` doubles as the taint output, which is sound because a compiled
   * program reads every input before it stores any output -- the same
   * guarantee taint_ir_c.run already relies on when it passes one array as
   * both.  So pass 2 runs in place and `st` simply IS the answer. */
  uint64_t *sv = env->sv, *st = env->st, *so = env->so;
  memcpy(st, reg_taint, nb);
  memcpy(sv, reg_val, nb);
  pend->n_taint = n_slots;

  for (int r = 0; r < plan->n_regions; r++) {
    const MtBlkRegion *reg = &plan->regions[r];
    if (!reg->fn && !(reg->prog && mt_blk_interp)) {
      mt_blk_miss(env, MT_BLK_MISS_UNPLANNED);
      return MT_BLK_DECLINED;
    }
    if (reg->n_acc > MT_BLK_MAX_ACC) {
      mt_blk_miss(env, MT_BLK_MISS_ACC);
      return MT_BLK_DECLINED;
    }

    /* A slot the program does not write reads back whatever was seeded in the
     * output array.  Seeding the VALUE half with the register's current value
     * is what makes "not written" and "written zero" distinguishable; without
     * it a register the block legitimately zeroes keeps a stale value.
     *
     * Only the slots this region PUBLISHES are ever read back out of that
     * half, so when the plan says which those are, seeding the rest is work
     * nobody looks at.  A region publishes only what a later region of the
     * same block reads, and the last region of a block publishes nothing. */
    if (reg->n_pub < 0) {
      memcpy(st + MT_BLK_VAL_BASE, sv, nb);
    } else {
      for (int i = 0; i < reg->n_pub; i++) {
        const int sl = reg->pub_slots[i];
        if (sl >= 0 && sl < n_slots) st[MT_BLK_VAL_BASE + sl] = sv[sl];
      }
    }
    const size_t accb = (size_t)MT_BLK_PER_ACC * (size_t)reg->n_acc
                        * sizeof(uint64_t);
    memset(sv + MT_BLK_MEM_BASE, 0, accb);
    memset(st + MT_BLK_MEM_BASE, 0, accb);

    if (env->stop_after == 7) { continue; }

    /* Pass 1 exists only to learn where the LOADS land: a store's address
     * comes out of pass 2 with everything else.  A region that only stores
     * therefore needs one pass, not two. */
    if (reg->n_load > 0) {
      /* `so` receives pass 1's outputs, and the ONLY thing read back out of it
       * is `so[... A_ADDR]` for each load.  Seeding it with the register file
       * the way pass 2's output array is seeded is therefore dead work: that
       * seeding exists so a slot the program does not write reads back its old
       * value, and no such slot is read here.  Zeroing the access area still
       * matters, because an address the slice fails to write must come back 0
       * and fail the guest read rather than carry whatever was there.
       * Measured on bench_dense: 9.8 ns/instr, against 5.7 for the address
       * slice those copies were setting up. */
      memset(so + MT_BLK_MEM_BASE, 0, accb);
      if (env->stop_after == 8) { continue; }
      if (reg->addr_fn || reg->addr_prog) {
        mt_blk_call(reg->addr_fn, reg->addr_prog, sv, st, so);
      } else {
        mt_blk_call(reg->fn, reg->prog, sv, st, so);
      }

      if (env->stop_after == 5) { continue; }
      for (int k = 0; k < reg->n_acc; k++) {
        if (reg->acc_kind[k] != 0) { continue; /* stores read nothing */ }
        const int base = MT_BLK_MEM_BASE + MT_BLK_PER_ACC * k;
        const uint64_t addr = so[base + MT_BLK_A_ADDR];
        const int size = reg->acc_size[k];
        /* Resolving a load is three calls, and 9/10/11 REMOVE one each rather
         * than stopping, because they are interleaved per access and nothing
         * useful can be cut between them.  Measured on bench_dense against
         * stage 4: the guest read is 31.7 ns/instr, the shadow read 3.9, the
         * overlay 1.3.  The read is the single largest thing the engine can
         * still control, and the tests cost nothing measurable: block mode is
         * 66.4-68.8 ms with them and 66.4-68.2 ms without. */
        uint64_t val = 0;
        if (reg->acc_needval[k] && env->mem_read && env->stop_after != 9) {
          if (env->mem_read(env->mem_ctx, addr, size, &val) != 0) {
            mt_blk_miss(env, MT_BLK_MISS_MEM);
            return MT_BLK_DECLINED;
          }
        }
        uint64_t msk =
          (env->shadow_read && env->stop_after != 10)
            ? env->shadow_read(env->shadow, addr, size) : 0;
        /* What this block has already stored wins over both. */
        if (env->stop_after != 11) {
          mt_blk_overlay(pend->writes, pend->n_writes, addr, size, &val, &msk);
        }
        sv[base + MT_BLK_A_MEM] = val;
        st[base + MT_BLK_A_MEM] = msk;
      }
    } else if (env->stop_after == 5 || env->stop_after == 8) {
      /* A region with no loads has to stop where a loading one stops, or the
       * stage measures a mixture: those regions used to run to completion
       * under stage 5 and the rung read as more expensive than it is. */
      continue;
    }

    if (env->stop_after == 6) { continue; }
    /* Pass 2: the answer, in place.  `st` is both the taint input and the
     * output, so afterwards it IS the region's post-state and nothing has to
     * be copied back. */
    mt_blk_call(reg->fn, reg->prog, sv, st, st);

    for (int k = 0; k < reg->n_acc; k++) {
      if (reg->acc_kind[k] != 1) { continue; }
      if (pend->n_writes >= MT_BLK_MAX_WRITES) {
        mt_blk_miss(env, MT_BLK_MISS_WRITES);
        return MT_BLK_DECLINED;
      }
      const int base = MT_BLK_MEM_BASE + MT_BLK_PER_ACC * k;
      MtBlkWrite *w = &pend->writes[pend->n_writes++];
      w->addr = st[base + MT_BLK_A_ADDR];
      w->mask = st[base + MT_BLK_A_STTAINT];
      w->value = st[base + MT_BLK_A_STVAL];
      w->size = reg->acc_size[k];
    }

    /* Thread this region's answer into the next.  `st` already holds the new
     * taint (pass 2 ran in place); the values it published move into `sv`,
     * which is where the next region reads them from.  Only the published
     * slots can have changed, and a slot nobody published keeps the value it
     * already had in `sv`, which is the same answer the wholesale copy gave. */
    if (reg->n_pub < 0) {
      memcpy(sv, st + MT_BLK_VAL_BASE, nb);
    } else {
      for (int i = 0; i < reg->n_pub; i++) {
        const int sl = reg->pub_slots[i];
        if (sl >= 0 && sl < n_slots) sv[sl] = st[MT_BLK_VAL_BASE + sl];
      }
    }

    /* A region that made the program counter secret-dependent.  The whole
     * region has run by the time this is known, so the finest address the
     * runtime can name is the region's LAST instruction -- and a block ends at
     * its branch, so for the case this actually catches that IS the branch,
     * and the same address the per-instruction path reports.  The report is
     * emitted a block late, with the deferred commit, because a block that
     * faults partway through never happened. */
    if (env->pc_slot >= 0 && env->pc_slot < n_slots && st[env->pc_slot]) {
      if (pend->n_reports >= MT_BLK_MAX_REPORTS) {
        mt_blk_miss(env, MT_BLK_MISS_REPORTS);
        return MT_BLK_DECLINED;
      }
      pend->rep_addr[pend->n_reports] = reg->last ? reg->last : reg->addr;
      pend->rep_mask[pend->n_reports] = st[env->pc_slot];
      pend->n_reports++;
      st[env->pc_slot] = 0;
    }
  }

  /* The block's answer, for the commit to apply once the NEXT block proves
   * this one finished.  One copy per block, where there used to be two per
   * region. */
  memcpy(pend->taint, st, nb);
  pend->valid = 1;
  return MT_BLK_OK;
}

/* ----------------------------------------------------------------------- */
/* Put a completed block's answer into effect.                             */
/*                                                                         */
/* Reaching a new block, or the end of the run, is what proves the held    */
/* one completed.  Reports are left in `pend` for the caller: emitting one */
/* is Python, and this runs without the GIL.                               */
/* ----------------------------------------------------------------------- */
static void mt_blk_commit(
  MtBlkPending *pend, MtBlkEnv *env, uint64_t *g_taint, int n_slots
) {
  if (!pend->valid) { return; }
  const int n = pend->n_taint < n_slots ? pend->n_taint : n_slots;
  memcpy(g_taint, pend->taint, (size_t)n * sizeof(uint64_t));
  if (env->shadow_write) {
    for (int i = 0; i < pend->n_writes; i++) {
      env->shadow_write(
        env->shadow,
        pend->writes[i].addr,
        pend->writes[i].mask,
        pend->writes[i].size
      );
    }
  }
  pend->valid = 0;
  pend->n_writes = 0;
}

/* Drop a held block: it did not complete.  Nothing of it was applied, which is
 * the entire reason for holding it. */
static inline void mt_blk_abandon(MtBlkPending *pend) {
  pend->valid = 0;
  pend->n_writes = 0;
  pend->n_reports = 0;
}

/* Does a guest write onto [addr, addr+size) land on the held block's own
 * instructions?
 *
 * The question matters because a write onto code invalidates every cached
 * PLAN, and it used to abandon the held block along with them.  For a rewrite
 * of the held block's own bytes that is right: the block was planned from
 * bytes that are no longer the ones it ran.  For a rewrite anywhere else it is
 * an UNDER-taint, and the two are easy to confuse because the hook tracks the
 * code it has planned as a single [lo, hi) interval -- one JIT page stretches
 * that interval across the whole image, and then every ordinary store to a
 * global in between looks like self-modifying code and silently throws away a
 * block's taint and its findings.
 *
 * A held block with no recorded extent answers yes, so an unknown stays
 * conservative. */
static inline int mt_blk_pending_hit(const MtBlkPending *pend,
                                     uint64_t addr, uint64_t size) {
  if (!pend->valid) return 0;
  if (pend->size <= 0) return 1;
  return addr < pend->address + (uint64_t)pend->size
      && addr + size > pend->address;
}

/* The emitted code a region's `fn` points into is owned by the caller, which
 * must keep it alive for the plan's lifetime.  Nothing here refcounts. */
static void mt_blk_plan_free(MtBlkPlan *p) {
  if (!p) return;
  if (p->regions) {
    for (int i = 0; i < p->n_regions; i++) free(p->regions[i].pub_slots);
  }
  free(p->regions);
  free(p->val_slots);
  free(p);
}

#endif /* MT_BLOCKPATH_H */
