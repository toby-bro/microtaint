"""Static structural analysis of the p-code for the compaction (Phase 3a).

For each bank instruction, classify its lifted p-code to predict how the per-op
model behaves, WITHOUT needing the per-op floors (that is Phase 3b):

  PURE_AFFINE   -- every op affine (routing / xor / const-mask / const-shift /
                   pow2-mult): per-op taint is an exact linear mask map, no
                   differential, trivially collapsible.
  NONAFFINE     -- has >=1 non-affine op (add/sub/mult/cmp/carry/var-shift):
                   a differential (with its per-op floor) is required there.
  RECONVERGENT  -- a subset of NONAFFINE where a register source reaches a
                   non-affine op's inputs via >=2 paths (or an input is used
                   twice): the case where aggressive (value,mask) per-op
                   OVER-taints and the reconvergence window must widen.
  OPAQUE        -- contains CALLOTHER / FLOAT_* / unmodelled: avalanche floor,
                   outside the compaction precision question.

The distribution says how much of the corpus is exact-for-free, how much needs a
differential at all, and how often the window has to fire.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

from collections import Counter
from typing import Any

from pypcode import Context, PcodeOp, Varnode

from microtaint.sleigh.lifter import get_context

_AFFINE_ALWAYS = {'COPY', 'INT_ZEXT', 'INT_SEXT', 'SUBPIECE', 'PIECE', 'EXTRACT',
                  'INSERT', 'INT_NEGATE', 'BOOL_NEGATE', 'INT_XOR', 'BOOL_XOR'}
_SKIP = {'IMARK', 'INDIRECT', 'MULTIEQUAL', 'CPOOLREF', 'NEW', 'CAST', 'SEGMENTOP',
         'BRANCH', 'BRANCHIND', 'CBRANCH', 'CALL', 'CALLIND', 'RETURN', 'STORE', 'LOAD'}
_OPAQUE_PREFIX = ('FLOAT_',)
_OPAQUE = {'CALLOTHER'}


def _const(vn: Varnode) -> bool:
    return bool(vn.space.name == 'const')


def _popcount(x: int) -> int:
    return bin(x).count('1')


def is_affine(op: PcodeOp) -> bool:
    n = op.opcode.name
    if n in _AFFINE_ALWAYS:
        return True
    if n in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'):
        return len(op.inputs) > 1 and _const(op.inputs[1])
    if n in ('INT_AND', 'INT_OR', 'BOOL_AND', 'BOOL_OR'):
        return any(_const(i) for i in op.inputs)
    if n == 'INT_MULT':
        return any(_const(i) and _popcount(i.offset) == 1 for i in op.inputs)
    return False


#: A varnode's identity: (space name, offset, size).
VId = tuple[str, int, int]


def _vid(vn: Varnode) -> VId:
    return (vn.space.name, vn.offset, vn.size)


def classify_instr(ctx: Context, code: bytes) -> tuple[str, int]:
    ops = [o for o in ctx.translate(code, 0x1000).ops if o.opcode.name not in _SKIP]
    if not ops:
        return 'EMPTY', 0
    # opaque?
    for o in ops:
        n = o.opcode.name
        if n in _OPAQUE or n.startswith(_OPAQUE_PREFIX):
            return 'OPAQUE', 0

    producers: dict[VId, PcodeOp] = {}
    for o in ops:
        if o.output is not None:
            producers[_vid(o.output)] = o

    def reg_ancestors(vn: Varnode, seen: set[VId] | None = None) -> set[VId]:
        """Register-space leaf varnodes feeding `vn` (transitively)."""
        if seen is None:
            seen = set()
        vid = _vid(vn)
        if vid in seen:
            return set()
        seen.add(vid)
        if vn.space.name == 'const':
            return set()
        if vn.space.name == 'register' and vid not in producers:
            return {vid}
        prod = producers.get(vid)
        if prod is None:
            return {vid} if vn.space.name == 'register' else set()
        acc = set()
        for i in prod.inputs:
            acc |= reg_ancestors(i, seen)
        return acc

    nonaffine = [o for o in ops if not is_affine(o)]
    if not nonaffine:
        return 'PURE_AFFINE', 0

    reconv = 0
    for o in nonaffine:
        dyn = [i for i in o.inputs if not _const(i)]
        # an input used twice is trivially reconvergent
        vids = [_vid(i) for i in dyn]
        if len(vids) != len(set(vids)):
            reconv += 1
            continue
        anc = [reg_ancestors(i) for i in dyn]
        shared = False
        for a in range(len(anc)):
            for b in range(a + 1, len(anc)):
                if anc[a] & anc[b]:
                    shared = True
                    break
            if shared:
                break
        if shared:
            reconv += 1
    return ('RECONVERGENT' if reconv else 'NONAFFINE'), reconv


def run(isas: list[str] | None = None) -> dict[str, Any]:
    from benchmark.instruction_bank import load_bank
    specs = load_bank(isas=set(isas) if isas else None)
    per_isa: dict[str, Any] = {}
    for name, spec in specs.items():
        key = spec.arch.value if hasattr(spec.arch, 'value') else str(spec.arch)
        try:
            ctx = get_context(key)
        except Exception:
            continue
        cnt: Counter[str] = Counter()
        reconv_ops = 0
        n = 0
        for ins in spec.instructions:
            try:
                cls, rc = classify_instr(ctx, ins.bytes)
            except Exception:
                cls, rc = 'ERROR', 0
            cnt[cls] += 1
            reconv_ops += rc
            n += 1
        per_isa[name] = (n, cnt, reconv_ops)
    return per_isa


if __name__ == '__main__':
    import sys
    isas = sys.argv[1:] or None
    per = run(isas)
    tot = Counter()
    print(f"{'ISA':12}{'n':>5}  {'pure-affine':>12}{'nonaffine':>11}{'reconvergent':>13}{'opaque':>8}{'other':>7}")
    for name, (n, cnt, rc) in per.items():
        tot.update(cnt)
        other = n - cnt['PURE_AFFINE'] - cnt['NONAFFINE'] - cnt['RECONVERGENT'] - cnt['OPAQUE']
        print(f"{name:12}{n:>5}  {cnt['PURE_AFFINE']:>12}{cnt['NONAFFINE']:>11}"
              f"{cnt['RECONVERGENT']:>13}{cnt['OPAQUE']:>8}{other:>7}   (reconv-ops={rc})")
    N = sum(v for v in tot.values())
    print(f"\nTOTAL n={N}: pure-affine={tot['PURE_AFFINE']} ({100*tot['PURE_AFFINE']/N:.0f}%)  "
          f"nonaffine={tot['NONAFFINE']} ({100*tot['NONAFFINE']/N:.0f}%)  "
          f"reconvergent={tot['RECONVERGENT']} ({100*tot['RECONVERGENT']/N:.0f}%)  "
          f"opaque={tot['OPAQUE']} ({100*tot['OPAQUE']/N:.0f}%)")
