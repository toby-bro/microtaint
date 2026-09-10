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

/* One lowered run of consecutive instructions. */
typedef struct {
  void *fn;      /* MtBlkFn, NULL if this region did not lower */
  uint64_t addr; /* the region's first guest address */
  int n_acc;
  signed char acc_kind[MT_BLK_MAX_ACC]; /* 0 load, 1 store */
  signed char acc_size[MT_BLK_MAX_ACC];
  signed char acc_needval[MT_BLK_MAX_ACC]; /* the program reads the word */
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
  /* The register values threaded from one region to the next, MT_BLK_VAL_BASE
   * entries.  Also caller-owned. */
  uint64_t *cur_val;
  int pc_slot;         /* -1 when the program counter has no slot */
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

  if (!plan || !plan->handleable || plan->n_regions <= 0) {
    mt_blk_miss(env, MT_BLK_MISS_UNPLANNED);
    return MT_BLK_DECLINED;
  }
  if (n_slots > MT_BLK_VAL_BASE || n_slots > pend->cap_taint) {
    mt_blk_miss(env, MT_BLK_MISS_SLOTS);
    return MT_BLK_DECLINED;
  }

  const size_t nb = (size_t)n_slots * sizeof(uint64_t);
  /* pend->taint IS the working taint: it is threaded region to region and
   * ends up being exactly the post-state the commit wants. */
  memcpy(pend->taint, reg_taint, nb);
  memcpy(env->cur_val, reg_val, nb);
  pend->n_taint = n_slots;

  for (int r = 0; r < plan->n_regions; r++) {
    const MtBlkRegion *reg = &plan->regions[r];
    if (!reg->fn) {
      mt_blk_miss(env, MT_BLK_MISS_UNPLANNED);
      return MT_BLK_DECLINED;
    }
    if (reg->n_acc > MT_BLK_MAX_ACC) {
      mt_blk_miss(env, MT_BLK_MISS_ACC);
      return MT_BLK_DECLINED;
    }

    uint64_t *sv = env->sv, *st = env->st, *so = env->so;
    memcpy(sv, env->cur_val, nb);
    memcpy(st, pend->taint, nb);
    /* The program's output array is its taint input array, so a slot the
     * program does not write reads back whatever was seeded there.  Seeding
     * the VALUE slot with the register's current value is what makes "not
     * written" and "written zero" distinguishable; without it the only way
     * to tell them apart is to read an output of zero as "untouched", and
     * then a register the block legitimately zeroes keeps a stale value.
     * Nothing reads these slots as inputs, so seeding them is free. */
    memcpy(st + MT_BLK_VAL_BASE, env->cur_val, nb);
    memset(
      sv + MT_BLK_MEM_BASE,
      0,
      (size_t)MT_BLK_PER_ACC * (size_t)reg->n_acc * sizeof(uint64_t)
    );
    memset(
      st + MT_BLK_MEM_BASE,
      0,
      (size_t)MT_BLK_PER_ACC * (size_t)reg->n_acc * sizeof(uint64_t)
    );

    if (reg->n_acc > 0) {
      /* Pass 1: the addresses.  Outputs land on a copy of the inputs so a
       * slot the program does not write keeps its value. */
      memcpy(so, st, nb);
      memcpy(so + MT_BLK_VAL_BASE, st + MT_BLK_VAL_BASE, nb);
      memset(
        so + MT_BLK_MEM_BASE,
        0,
        (size_t)MT_BLK_PER_ACC * (size_t)reg->n_acc * sizeof(uint64_t)
      );
      ((MtBlkFn)reg->fn)(sv, st, so);

      for (int k = 0; k < reg->n_acc; k++) {
        if (reg->acc_kind[k] != 0) { continue; /* stores read nothing */ }
        const int base = MT_BLK_MEM_BASE + MT_BLK_PER_ACC * k;
        const uint64_t addr = so[base + MT_BLK_A_ADDR];
        const int size = reg->acc_size[k];
        uint64_t val = 0;
        if (reg->acc_needval[k] && env->mem_read) {
          if (env->mem_read(env->mem_ctx, addr, size, &val) != 0) {
            mt_blk_miss(env, MT_BLK_MISS_MEM);
            return MT_BLK_DECLINED;
          }
        }
        uint64_t msk =
          env->shadow_read ? env->shadow_read(env->shadow, addr, size) : 0;
        /* What this block has already stored wins over both. */
        mt_blk_overlay(pend->writes, pend->n_writes, addr, size, &val, &msk);
        sv[base + MT_BLK_A_MEM] = val;
        st[base + MT_BLK_A_MEM] = msk;
      }
    }

    /* Pass 2: the answer, now that the loaded words are known. */
    memcpy(so, st, nb);
    memcpy(so + MT_BLK_VAL_BASE, st + MT_BLK_VAL_BASE, nb);
    ((MtBlkFn)reg->fn)(sv, st, so);

    for (int k = 0; k < reg->n_acc; k++) {
      if (reg->acc_kind[k] != 1) { continue; }
      if (pend->n_writes >= MT_BLK_MAX_WRITES) {
        mt_blk_miss(env, MT_BLK_MISS_WRITES);
        return MT_BLK_DECLINED;
      }
      const int base = MT_BLK_MEM_BASE + MT_BLK_PER_ACC * k;
      MtBlkWrite *w = &pend->writes[pend->n_writes++];
      w->addr = so[base + MT_BLK_A_ADDR];
      w->mask = so[base + MT_BLK_A_STTAINT];
      w->value = so[base + MT_BLK_A_STVAL];
      w->size = reg->acc_size[k];
    }

    /* Thread this region's answer into the next one. */
    memcpy(pend->taint, so, nb);
    memcpy(env->cur_val, so + MT_BLK_VAL_BASE, nb);

    /* A region that made the program counter secret-dependent.  The report
     * carries the REGION's address, so a finding still names where the leak
     * is even though it is emitted a block late. */
    if (
      env->pc_slot >= 0 && env->pc_slot < n_slots && pend->taint[env->pc_slot]
    ) {
      if (pend->n_reports >= MT_BLK_MAX_REPORTS) {
        mt_blk_miss(env, MT_BLK_MISS_REPORTS);
        return MT_BLK_DECLINED;
      }
      pend->rep_addr[pend->n_reports] = reg->addr;
      pend->rep_mask[pend->n_reports] = pend->taint[env->pc_slot];
      pend->n_reports++;
      pend->taint[env->pc_slot] = 0;
    }
  }

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

/* The emitted code a region's `fn` points into is owned by the caller, which
 * must keep it alive for the plan's lifetime.  Nothing here refcounts. */
static void mt_blk_plan_free(MtBlkPlan *p) {
  if (!p) return;
  free(p->regions);
  free(p->val_slots);
  free(p);
}

#endif /* MT_BLOCKPATH_H */
