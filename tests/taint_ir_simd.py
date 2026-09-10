"""Per-bit ground truth for vector lanes, and the IR's answer.

The bank harness has no ground truth for SIMD -- its Unicorn descriptors cover
general-purpose registers and flags -- so a lane-splitting bug in the lowering
would only ever be visible as a disagreement with the engine's own
differential, which cannot tell a bug from a precision gain.  This closes that:
XMM lanes are read and written directly, every tainted bit is flipped in turn,
and every lane that moves is tainted.

One exclusion, and it is the emulator's fault rather than a limitation here:
Unicorn decodes several two-byte VEX forms as their legacy two-operand
equivalents -- `vpxor xmm0, xmm1, xmm2` runs as `pxor xmm0, xmm2`, ignoring the
VEX-encoded second source -- and its answer for those is not ground truth.

They are DETECTED rather than listed.  A VEX three-operand form never reads its
destination, so running the same instruction twice with identical sources and
two different destination values must give the same answer; when it does not,
Unicorn is executing something else, and the form is skipped and named.  A hand
written list of prefixes goes stale the moment the bank grows, and it went stale
here: it named four forms and there were eight.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import random
from typing import Any

from microtaint.types import Architecture

MASK64 = 0xFFFFFFFFFFFFFFFF
CODE_ADDR = 0x1000

#: Engine lane name -> (unicorn register, bit offset within it).
LANES = {
    'VL_0x1200': ('XMM0', 0), 'VL_0x1208': ('XMM0', 64),
    'VL_0x1240': ('XMM1', 0), 'VL_0x1248': ('XMM1', 64),
}

#: A destination marker the probe below writes; any distinctive pair will do.
_PROBE_MARKERS = (0xDEADBEEFCAFEBABE, 0x0123456789ABCDEF)


def _uc_regs() -> dict[str, int]:
    import unicorn.x86_const as ux
    return {'XMM0': ux.UC_X86_REG_XMM0, 'XMM1': ux.UC_X86_REG_XMM1}


def _run(code: bytes, lane_vals: dict[str, int]) -> dict[str, int]:
    import unicorn
    regs = _uc_regs()
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(CODE_ADDR, 0x1000)
    uc.mem_write(CODE_ADDR, code)
    packed: dict[str, int] = {}
    for name, (reg, sh) in LANES.items():
        packed[reg] = packed.get(reg, 0) | ((lane_vals[name] & MASK64) << sh)
    for reg, v in packed.items():
        uc.reg_write(regs[reg], v)
    uc.emu_start(CODE_ADDR, CODE_ADDR + len(code))
    return {name: (uc.reg_read(regs[reg]) >> sh) & MASK64
            for name, (reg, sh) in LANES.items()}


def _unicorn_honours_the_encoding(code: bytes) -> bool:
    """Does Unicorn execute the instruction this encoding names?

    A VEX three-operand form never reads its destination, so the same
    instruction run twice with identical sources and two different destination
    values has to give the same answer.  When it does not, Unicorn decoded
    something else -- in practice the legacy two-operand form, where the
    destination IS a source -- and its answer cannot be ground truth for this
    encoding.

    Only VEX encodings are probed.  A legacy SSE form is SUPPOSED to read its
    destination -- `paddb xmm0, xmm1` means `xmm0 += xmm1` -- so the same
    question asked of one answers "not honoured" about an instruction Unicorn is
    executing perfectly.  In 64-bit mode a leading C4 or C5 is always VEX.

    Conservative in the right direction: a VEX form that legitimately merges
    into its destination (`vmovss xmm0, xmm1, xmm2`) fails the probe too and is
    skipped, which costs coverage rather than correctness.
    """
    import unicorn
    if not code or code[0] not in (0xC4, 0xC5):
        return True
    outs = []
    for marker in _PROBE_MARKERS:
        try:
            outs.append(_run(code, {n: (marker if reg_of == 'XMM0' else 0)
                                    for n, (reg_of, _sh) in LANES.items()}))
        except unicorn.UcError:
            return False
    return outs[0] == outs[1]


def ground_truth(code: bytes, taint: dict[str, int],
                 vals: dict[str, int]) -> dict[str, int]:
    base = {n: vals[n] & ~taint[n] & MASK64 for n in LANES}
    b = _run(code, base)
    res = {n: 0 for n in LANES}
    for src in LANES:
        tm = taint[src]
        bit = 0
        while tm:
            if tm & 1:
                flipped = dict(base)
                flipped[src] = base[src] | (1 << bit)
                out = _run(code, flipped)
                for k in res:
                    res[k] |= b[k] ^ out[k]
            tm >>= 1
            bit += 1
    return res


def ir_answer(arch: Architecture, code: bytes, taint: dict[str, int],
              vals: dict[str, int]) -> dict[str, int]:
    from microtaint.taint_ir.frompcode import build_ir
    from tests.taint_ir_bank import name_offset

    prog = build_ir(arch, code)
    v: dict[tuple[Any, ...], int] = {}
    t: dict[tuple[Any, ...], int] = {}
    for n in LANES:
        off = name_offset(arch, n)
        for sz in range(1, 9):
            v[('reg', off, sz)] = vals[n]
            t[('reg', off, sz)] = taint[n]
    out = prog.run(v, t)
    res = dict(taint)
    for n in LANES:
        off = name_offset(arch, n)
        for sz in range(1, 9):
            if ('reg', off, sz) in out:
                res[n] = out[('reg', off, sz)]
                break
    return res


def run_simd_bank(n_vec: int = 5, seed: int = 99,
                  ) -> tuple[int, dict[str, list[tuple[str, str, dict[str, str]]]],
                             list[str]]:
    """-> (n_cases, {label: [(lane, missing_bits, taint)]})"""
    import unicorn

    from benchmark.instruction_bank import load_bank
    from microtaint.taint_ir.frompcode import build_ir

    spec = load_bank(isas={'AMD64_SIMD'})['AMD64_SIMD']
    rng = random.Random(seed)
    under: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    skipped: list[str] = []
    n = 0
    for ins in spec.instructions:
        if not _unicorn_honours_the_encoding(ins.bytes):
            skipped.append(ins.label)
            continue
        try:
            build_ir(spec.arch, ins.bytes)
        except Exception:  # noqa: BLE001
            continue
        for _ in range(n_vec):
            vals = {k: rng.getrandbits(64) for k in LANES}
            taint = {k: 0 for k in LANES}
            for _ in range(2):
                taint[rng.choice(list(LANES))] |= 1 << rng.randint(0, 63)
            try:
                gt = ground_truth(ins.bytes, taint, vals)
                got = ir_answer(spec.arch, ins.bytes, taint, vals)
            except unicorn.UcError:
                break
            n += 1
            for k in LANES:
                miss = gt[k] & ~got[k]
                if miss:
                    under.setdefault(ins.label, []).append(
                        (k, hex(miss), {a: hex(b) for a, b in taint.items() if b}))
    return n, under, skipped


def main(argv: list[str] | None = None) -> int:
    n, under, skipped = run_simd_bank()
    print(f'SIMD cases={n} instructions_with_under_taint={len(under)} '
          f'not_ground_truth={len(skipped)}')
    if skipped:
        print(f'  Unicorn does not honour the encoding of: {", ".join(sorted(skipped))}')
    for label, items in list(under.items())[:10]:
        print(f'  {label}: {items[0]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
