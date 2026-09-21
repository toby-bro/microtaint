# ruff: noqa: W505, E501, B905, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_table5_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""A ground truth that cannot invent taint.

The shipped `GTSim` reports phantom taint.  It reuses ONE `Uc` across the 2^k
polarity runs of a case and initialises only the four tracked GPRs, SP and the
flags, so every other architectural register carries run i-1's writes into run i.
`shrd rbp, rsp` (SHRD RBP, RSP, CL) is the worst case: RBP is both source and
destination and is untracked, so the second polarity reads back the first
polarity's result and PF moves for a taint bit that is not even in CL.  That one
form produced 2,810 of the 2,875 "under-taints" in the 12 h campaign.

Three defects, three fixes:

  1. registers outside `isa.gprs` are never initialised  ->  restore a pristine
     all-zero CPU context before every run;
  2. mapped memory is never reset between runs           ->  re-zero the stack
     and rewrite the code before every run;
  3. `emu_start` is not checked for completion           ->  verify PC reached
     the end, and invalidate the WHOLE case if any single polarity run fails.

(3) matters in both directions.  The shipped oracle `continue`s past a failing
run and still returns exact=True, computing OR^AND over a subset of the
polarities.  That can only shrink the differing-bit set, so it hides real
under-taints.  Skipping the whole case instead is the only choice that is
neither a false positive nor a silent false negative; `skipped` counts them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unicorn import UC_HOOK_CODE  # noqa: E402

from mt_multiarch import GTSim  # noqa: E402


class CaseInvalid(Exception):
    """A polarity run did not complete, so the case yields no information."""


class HardenedGTSim(GTSim):
    """`GTSim` with the polarity runs made independent, and failures honoured.

    Completion is verified by COUNTING EXECUTED INSTRUCTIONS with a code hook,
    not by reading PC.  Unicorn does not update MIPS's PC after `emu_start`: it
    reads back the start address, so a PC check raises on every MIPS run.
    Combined with the parent's per-run `except Exception: continue` that was
    catastrophic rather than merely wrong: every run of every MIPS case was
    discarded, `outs` came back empty, and the parent's `if not outs` branch
    returned an all-zero mask marked EXACT.  An all-zero ground truth makes
    `GT & ~MT` unconditionally zero, so no under-taint can ever be reported.
    19,085,696 MIPS cases were scored that way in the 2026-09-12 campaign.

    `taint` is overridden rather than delegated, because the parent swallows
    per-run exceptions: raising from `_run` alone cannot give all-or-nothing
    semantics when the caller is the thing discarding the raise.
    """

    def __init__(self, isa):
        super().__init__(isa)
        self.skipped = 0
        self.ran = 0
        self._reached = [0]
        self._end = 0
        self._pristine = None

    def _trace(self, uc, addr, size, data):
        # Where execution actually got to; correct for multi-instruction forms.
        self._reached[0] = addr + size

    def _fresh(self, code):
        uc = super()._fresh(code)
        uc.hook_add(UC_HOOK_CODE, self._trace)
        self._end = self.CODE + len(code)
        self._pristine = uc.context_save()
        return uc

    def _run(self, uc, code, gpr_vals, flag_bits):
        uc.context_restore(self._pristine)
        uc.mem_write(self.STACK, b'\x00' * 0x10000)
        uc.mem_write(self.CODE, code + b'\x00' * 16)
        self._reached[0] = 0
        try:
            out = super()._run(uc, code, gpr_vals, flag_bits)
        except Exception as exc:
            raise CaseInvalid(f'run raised: {exc}') from exc
        if self._reached[0] != self._end:
            raise CaseInvalid(
                f'stopped at {self._reached[0]:#x}, expected {self._end:#x}',
            )
        return out

    def _positions(self, taint):
        isa = self.isa
        # On a sign-extended regime the high half is a FUNCTION of the sign bit,
        # so flipping one of its bits alone is undone by re-canonicalisation and
        # witnesses nothing.  Enumerating only the independent bits keeps the
        # ground truth identical and saves 32 runs per tainted register.
        width = isa.canon if isa.canon and isa.canon < isa.bits else isa.bits
        pos = []
        for name, _ in isa.gprs:
            tm = taint.get(name, 0)
            pos += [(name, b) for b in range(width) if (tm >> b) & 1]
        for f in isa.flags:
            tm = taint.get(f.name, 0)
            pos += [(f.name, b) for b in range(f.width) if (tm >> b) & 1]
        return pos

    def taint_flip_union(self, code, state, taint):
        """Ground truth by flipping each tainted input bit ONCE, independently.

        GT = OR over every tainted bit b of
             out(state) XOR out(state with bit b flipped)

        Cost is k+1 runs for k tainted bits, linear rather than exponential, so
        ANY taint mask is affordable: there is no enumeration budget, no cap on
        how many bits may be tainted, and no separate "sparse" and "dense" case
        kinds.  A case tainting all 64 bits costs 65 runs.

        Run from BOTH POLARITIES: once from the given state, once from the
        state with every tainted bit inverted, unioning the two.  A single flip
        cannot witness a dependence that needs two inputs to move together, and
        for the common families that blind spot sits at a CORNER -- `a AND b`
        hides where both bits are 0, `a OR b` where both are 1 -- so the
        opposite polarity puts those positions where one flip does move the
        output.  Measured against the exact enumeration on overlapping masks,
        this takes and/or from 11.0%/19.9% of real dependences missed to 0.0%,
        `add` from 9.5% to 0.2%, and `imul` from 4.8% to 0.6%.

        Cost is 2(k+1) runs for k tainted bits, still linear rather than
        exponential, so ANY taint mask is affordable: there is no enumeration
        budget, no cap on how many bits may be tainted, and no separate "sparse"
        and "dense" case kinds.  A case tainting all 64 bits costs 130 runs.

        It remains a LOWER BOUND on semantic dependence, just a much tighter
        one.  The residual runs the safe way for soundness: a too-small ground
        truth makes the under-taint test `gt & ~mt` more LENIENT, never
        stricter, so a reported zero is trustworthy.  It runs the unsafe way for
        precision, so measured over-taint is an upper bound on the engine's
        imprecision and partly the oracle's own.

        All-or-nothing, like `taint`: if any run fails the case yields nothing.
        """
        isa = self.isa
        empty = dict.fromkeys(isa.reg_names, 0)
        positions = self._positions(taint)
        if not positions:
            # Nothing is tainted, so there is nothing to witness.  Returning an
            # empty ground truth marked VALID scores the case bit-exact against
            # ANY engine, because `gt & ~mt` is then unconditionally zero.  That
            # is how six MIPS instructions Unicorn cannot even execute came to
            # report checked > 0 at 100% exact: canonicalize() confines MIPS
            # taint to the low 32 bits, emptying masks that the generator's own
            # "is anything tainted" guard had already approved.  A case with
            # nothing to measure is a SKIP, not a pass.
            self.skipped += 1
            return dict(empty), False
        flagset = {f.name for f in isa.flags}

        def split(base_state, flip):
            g = {n: base_state.get(n, 0) & isa.mask for n, _ in isa.gprs}
            fl = {f.name: base_state.get(f.name, 0) & f.mask for f in isa.flags}
            if flip is not None:
                reg, b = flip
                if reg in flagset:
                    fl[reg] = (fl.get(reg, 0) ^ (1 << b)) & next(
                        f for f in isa.flags if f.name == reg).mask
                else:
                    g[reg] = (g[reg] ^ (1 << b)) & isa.mask
                    # Re-project into the architecturally DEFINED regime.  MIPS64
                    # 32-bit ops are only defined on sign-extended words; the
                    # generator canonicalises the base state but nothing
                    # canonicalised the FLIPPED one, so flipping bit 31 left bits
                    # 32-63 carrying the OLD sign and the value's sign could never
                    # change.  `slt`, `sltu`, `dsrl32` and friends were structurally
                    # unmeasurable, ~348k cases all reporting 100% bit-exact.
                    g, _ = isa.canonicalize(g, {})
            return g, fl

        # The opposite polarity: every tainted bit inverted.  See the docstring.
        other = dict(state)
        for r, m in taint.items():
            if r in flagset:
                w = next(f for f in isa.flags if f.name == r)
                other[r] = (other.get(r, 0) ^ (m & w.mask)) & w.mask
            elif r in {n for n, _ in isa.gprs}:
                other[r] = (other.get(r, 0) ^ m) & isa.mask
        other, _ = isa.canonicalize(other, {})

        try:
            uc = self._fresh(code)
            gt = dict(empty)
            for base_state in (state, other):
                base = self._run(uc, code, *split(base_state, None))
                for pos in positions:
                    out = self._run(uc, code, *split(base_state, pos))
                    for r in isa.reg_names:
                        gt[r] |= base[r] ^ out[r]
            self.ran += 1
            return gt, True
        except CaseInvalid:
            self.skipped += 1
            return dict(empty), False

    def taint(self, code, state, taint, budget=13):
        """({reg: mask}, exact), or ({0...}, False) when the case yields nothing.

        All-or-nothing: if ANY run fails the case is discarded and `skipped` is
        incremented.  Reducing over a subset of the polarities produces a mask
        that is neither an upper nor a lower bound, and returning the empty
        accumulator as though it were exact is indistinguishable from "this
        instruction propagates no taint".
        """
        isa = self.isa
        empty = dict.fromkeys(isa.reg_names, 0)
        positions = self._positions(taint)
        k = len(positions)
        if k == 0:
            return dict(empty), True
        flagset = {f.name for f in isa.flags}

        def split(assign_bits):
            g = {n: (state.get(n, 0) & ~taint.get(n, 0)) & isa.mask for n, _ in isa.gprs}
            fl = {f.name: (state.get(f.name, 0) & ~taint.get(f.name, 0)) & f.mask
                  for f in isa.flags}
            for v, (reg, b) in zip(assign_bits, positions):
                if reg in flagset:
                    fl[reg] = fl.get(reg, 0) | (v << b)
                else:
                    g[reg] = (g[reg] | (v << b)) & isa.mask
            return g, fl

        try:
            uc = self._fresh(code)
            if k <= budget:                                  # EXACT
                outs = []
                for a in range(1 << k):
                    g, fl = split([(a >> i) & 1 for i in range(k)])
                    outs.append(self._run(uc, code, g, fl))
                res = {}
                for r in isa.reg_names:
                    orv, andv = 0, None
                    for o in outs:
                        orv |= o[r]
                        andv = o[r] if andv is None else (andv & o[r])
                    res[r] = orv ^ (andv or 0)
                self.ran += 1
                return res, True
            # k > budget: single-bit-flip LOWER bound (soundness only, not exact)
            base = self._run(uc, code, *split([0] * k))
            lb = dict(empty)
            for i in range(k):
                bits = [0] * k
                bits[i] = 1
                o = self._run(uc, code, *split(bits))
                for r in isa.reg_names:
                    lb[r] |= base[r] ^ o[r]
            self.ran += 1
            return lb, False
        except CaseInvalid:
            self.skipped += 1
            return dict(empty), False
