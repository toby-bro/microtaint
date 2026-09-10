"""Per-op differential prototype (Phase 3a precision study).

Validates the eval-core redesign BEFORE any C is written, against the Phase-0
oracle.  Three propagation schemes over the lifted p-code, one forward pass:

  Scheme B  (two corner values): carry BOTH polarity corners (hi=V|T, lo=V&~T)
            per varnode through every op; taint = hi XOR lo at the output.  This
            is the BARE two-corner differential, reorganized as one pass.

            FINDING (3a): bare Scheme B UNDER-taints vs the oracle (e.g.
            `add rax,rbx` misses 1 bit of RAX+OF; `and rax,rbx` misses 1 bit of
            PF).  The oracle (circuit.evaluate) is NOT the bare differential --
            it is the differential PLUS the engine's soundness floors
            (transportable carry-union, monotonic, avalanche) that exist because
            two extremal corners coincide on interior bit flips.  So the per-op
            model MUST carry per-op FLOORS (mapper.py categories), not just two
            corners.  Implementing + measuring per-op-with-floors is Phase 3b,
            gated by this harness; the floors map naturally onto p-code ops.

  Scheme C  (compaction): collapse PURE-ROUTING affine prefixes to (value,mask)
            -- where corner reconstruction (value|mask, value&~mask) is exact --
            and only carry two corner values across mixing-affine / non-affine
            ops.  Same answer as B; measures how much of the p-code collapses.
            [prototype: measured structurally here; full impl is Phase 3b.]

  Scheme A  (aggressive): (value,mask) everywhere, reconstruct corners at each
            non-affine op.  Over-taints at mixing-affine + reconvergence; we
            MEASURE that over-taint as the cost of maximal compaction.

Only the register/flag corpus (LE ISAs) is covered here; that is where the
compaction precision question lives.  Memory + BE come with the corpus extension.
"""
from __future__ import annotations

from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF

# Arch -> pypcode key + little-endian? (BE register spaces need byte-order care)
_ARCH_LE = {'AMD64': True, 'ARM64': True, 'RISCV64': True,
            'MIPS64BE': False, 'PPC32BE': False}


class Unsupported(Exception):
    """An op this prototype does not model (opaque/float/etc.)."""


def _mask(nbytes: int) -> int:
    return (1 << (8 * nbytes)) - 1 if nbytes < 8 else MASK64 if nbytes == 8 else (1 << (8 * nbytes)) - 1


def _signed(v: int, nbytes: int) -> int:
    bits = 8 * nbytes
    return v - (1 << bits) if (v >> (bits - 1)) & 1 else v


# ---------------------------------------------------------------------------
# Concrete p-code op semantics (one corner).  in_vals: list of ints already
# read at each input's width; op carries input/output Varnode sizes.
# ---------------------------------------------------------------------------

def _apply(name: str, op, in_vals: list[int]) -> int:
    o_sz = op.output.size
    om = _mask(o_sz)
    a = in_vals[0] if in_vals else 0
    b = in_vals[1] if len(in_vals) > 1 else 0
    isz = op.inputs[0].size if op.inputs else o_sz

    if name == 'COPY':
        return a & om
    if name == 'INT_ADD' or name == 'PTRADD':
        return (a + b) & om
    if name == 'INT_SUB' or name == 'PTRSUB':
        return (a - b) & om
    if name == 'INT_2COMP':
        return (-a) & om
    if name == 'INT_NEGATE':
        return (~a) & om
    if name == 'INT_XOR':
        return (a ^ b) & om
    if name == 'INT_AND':
        return (a & b) & om
    if name == 'INT_OR':
        return (a | b) & om
    if name == 'INT_LEFT':
        return (a << b) & om if b < 128 else 0
    if name == 'INT_RIGHT':
        return (a >> b) & om
    if name == 'INT_SRIGHT':
        return (_signed(a, isz) >> min(b, 8 * isz)) & om
    if name == 'INT_MULT':
        return (a * b) & om
    if name == 'INT_ZEXT':
        return a & om
    if name == 'INT_SEXT':
        return _signed(a, isz) & om
    if name == 'SUBPIECE':
        return (a >> (8 * b)) & om
    if name == 'PIECE':
        return ((a << (8 * op.inputs[1].size)) | b) & om
    if name == 'INT_EQUAL':
        return 1 if a == b else 0
    if name == 'INT_NOTEQUAL':
        return 1 if a != b else 0
    if name == 'INT_LESS':
        return 1 if a < b else 0
    if name == 'INT_LESSEQUAL':
        return 1 if a <= b else 0
    if name == 'INT_SLESS':
        return 1 if _signed(a, isz) < _signed(b, op.inputs[1].size) else 0
    if name == 'INT_SLESSEQUAL':
        return 1 if _signed(a, isz) <= _signed(b, op.inputs[1].size) else 0
    if name == 'INT_CARRY':
        return 1 if (a + b) > _mask(isz) else 0
    if name == 'INT_SCARRY':
        sa, sb = _signed(a, isz), _signed(b, isz)
        return 1 if _signed((a + b) & _mask(isz), isz) != sa + sb else 0
    if name == 'INT_SBORROW':
        sa, sb = _signed(a, isz), _signed(b, isz)
        return 1 if _signed((a - b) & _mask(isz), isz) != sa - sb else 0
    if name == 'POPCOUNT':
        return bin(a).count('1') & om
    if name == 'BOOL_AND':
        return 1 if (a & 1) and (b & 1) else 0
    if name == 'BOOL_OR':
        return 1 if (a & 1) or (b & 1) else 0
    if name == 'BOOL_XOR':
        return (a ^ b) & 1
    if name == 'BOOL_NEGATE':
        return (a ^ 1) & 1
    if name in ('INT_DIV', 'INT_REM', 'INT_SDIV', 'INT_SREM'):
        if b == 0:
            raise Unsupported('div0')
        if name == 'INT_DIV':
            return (a // b) & om
        if name == 'INT_REM':
            return (a % b) & om
        sa, sb = _signed(a, isz), _signed(b, op.inputs[1].size)
        q = int(sa / sb) if name == 'INT_SDIV' else sa - int(sa / sb) * sb
        return q & om
    raise Unsupported(name)


_SKIP = {'IMARK', 'INDIRECT', 'MULTIEQUAL', 'CPOOLREF', 'NEW', 'CAST', 'SEGMENTOP'}
_CTRL = {'BRANCH', 'BRANCHIND', 'CBRANCH', 'CALL', 'CALLIND', 'RETURN'}


# ---------------------------------------------------------------------------
# Scheme B interpreter: two corner values per varnode, one forward pass.
# ---------------------------------------------------------------------------

class TwoCorner:
    def __init__(self, ctx, little_endian: bool):
        self.ctx = ctx
        self.le = little_endian
        self.hi: dict[tuple[str, int], int] = {}
        self.lo: dict[tuple[str, int], int] = {}

    def _rd(self, store, vn) -> int:
        if vn.space.name == 'const':
            return vn.offset & _mask(vn.size)
        v = 0
        for i in range(vn.size):
            byte = store.get((vn.space.name, vn.offset + i), 0)
            sh = i if self.le else (vn.size - 1 - i)
            v |= byte << (8 * sh)
        return v

    def _wr(self, store, vn, val) -> None:
        for i in range(vn.size):
            sh = i if self.le else (vn.size - 1 - i)
            store[(vn.space.name, vn.offset + i)] = (val >> (8 * sh)) & 0xFF

    def init_reg(self, vn, value: int, taint: int) -> None:
        m = _mask(vn.size)
        self._wr(self.hi, vn, (value | taint) & m)
        self._wr(self.lo, vn, (value & ~taint) & m)

    def run(self, ops) -> None:
        for op in ops:
            name = op.opcode.name
            if name in _SKIP or name in _CTRL:
                continue
            if op.output is None:
                # STORE / opaque side effects: unsupported for the register corpus
                if name in ('STORE', 'LOAD', 'CALLOTHER'):
                    raise Unsupported(name)
                continue
            hi_ins = [self._rd(self.hi, v) for v in op.inputs]
            lo_ins = [self._rd(self.lo, v) for v in op.inputs]
            hi_out = _apply(name, op, hi_ins)
            lo_out = _apply(name, op, lo_ins)
            self._wr(self.hi, op.output, hi_out)
            self._wr(self.lo, op.output, lo_out)

    def out_taint(self, vn) -> int:
        return (self._rd(self.hi, vn) ^ self._rd(self.lo, vn)) & _mask(vn.size)


def _arch_key(arch: Architecture) -> str:
    return arch.value if hasattr(arch, 'value') else str(arch)


def scheme_b_taint(arch: Architecture,
                   code: bytes,
                   regs: list[Register],
                   in_taint: dict[str, int],
                   in_values: dict[str, int]):
    """Run Scheme B; return output taint dict over the mappable bank registers.
    Raises Unsupported if any op or register can't be mapped (caller skips)."""
    key = _arch_key(arch)
    if not _ARCH_LE.get(key, False):
        raise Unsupported(f'BE arch {key}')
    ctx = get_context(key)
    reg_vn = ctx.registers
    interp = TwoCorner(ctx, _ARCH_LE[key])
    mapped = []
    for r in regs:
        vn = reg_vn.get(r.name)
        if vn is None:
            continue  # e.g. EFLAGS/RIP not a pypcode register -> not compared
        mapped.append((r.name, vn))
        interp.init_reg(vn, in_values.get(r.name, 0), in_taint.get(r.name, 0))
    ops = ctx.translate(code, 0x1000).ops
    interp.run(ops)
    return {name: interp.out_taint(vn) for name, vn in mapped}


def engine_scheme_b(arch: Architecture,
                    code: bytes,
                    regs: list[Register],
                    in_taint: dict[str, int],
                    in_values: dict[str, int],
                    *,
                    circuit=None):
    """Oracle-harness engine adapter for Scheme B."""
    return scheme_b_taint(arch, code, regs, in_taint, in_values)


if __name__ == '__main__':
    # quick single-instruction sanity vs the differential oracle
    from benchmark.instruction_bank import isa_registers
    from microtaint.types import Architecture as A
    from tests.oracle_harness import classify, reference_taint
    regs = list(isa_registers('AMD64'))
    rn = [r.name for r in regs]
    cases = [
        ('add rax, rbx', bytes.fromhex('4801d8')),
        ('xor rax, rax', bytes.fromhex('4831c0')),
        ('and rax, rbx', bytes.fromhex('4821d8')),
    ]
    for label, code in cases:
        it = {r: (MASK64 if r in ('RAX', 'RBX') else 0) for r in rn}
        iv = {r: 0x1234 * (i + 1) for i, r in enumerate(rn)}
        try:
            got = engine_scheme_b(A.AMD64, code, regs, it, iv)
        except Unsupported as e:
            print(f'{label}: UNSUPPORTED {e}'); continue
        ref = reference_taint(A.AMD64, code, regs, it, iv)
        v = classify(got, ref, [r for r in rn if r in got])
        print(f'{label}: exact={v.exact} under={v.under} over={v.over}')
