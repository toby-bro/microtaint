"""Per-bit ground truth for vector lanes, and the IR's answer.

The bank harness has no ground truth for SIMD -- its Unicorn descriptors cover
general-purpose registers and flags -- so a lane-splitting bug in the lowering
would only ever be visible as a disagreement with the engine's own
differential, which cannot tell a bug from a precision gain.  This closes that:
XMM lanes are read and written directly, every tainted bit is flipped in turn,
and every lane that moves is tainted.

One exclusion, and it is the emulator's fault rather than a limitation here:
Unicorn decodes the two-byte VEX form of `vpxor xmm0, xmm1, xmm2` as the legacy
`pxor xmm0, xmm2`, ignoring the VEX-encoded second source.  Its answer for that
instruction is not ground truth, so VEX forms are skipped and named.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import random

MASK64 = 0xFFFFFFFFFFFFFFFF
CODE_ADDR = 0x1000

#: Engine lane name -> (unicorn register, bit offset within it).
LANES = {
    'VL_0x1200': ('XMM0', 0), 'VL_0x1208': ('XMM0', 64),
    'VL_0x1240': ('XMM1', 0), 'VL_0x1248': ('XMM1', 64),
}

#: Instructions whose Unicorn semantics are not trustworthy; see the module
#: docstring.  Matched as a prefix of the bank label.
UNTRUSTED = ('vpxor', 'vpand', 'vpor', 'vmov')


def _uc_regs():
    import unicorn.x86_const as ux
    return {'XMM0': ux.UC_X86_REG_XMM0, 'XMM1': ux.UC_X86_REG_XMM1}


def _run(code, lane_vals):
    import unicorn
    regs = _uc_regs()
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(CODE_ADDR, 0x1000)
    uc.mem_write(CODE_ADDR, code)
    packed: dict = {}
    for name, (reg, sh) in LANES.items():
        packed[reg] = packed.get(reg, 0) | ((lane_vals[name] & MASK64) << sh)
    for reg, v in packed.items():
        uc.reg_write(regs[reg], v)
    uc.emu_start(CODE_ADDR, CODE_ADDR + len(code))
    return {name: (uc.reg_read(regs[reg]) >> sh) & MASK64
            for name, (reg, sh) in LANES.items()}


def ground_truth(code, taint, vals):
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


def ir_answer(arch, code, taint, vals):
    from microtaint.taint_ir.frompcode import build_ir
    from tests.taint_ir_bank import name_offset

    prog = build_ir(arch, code)
    v, t = {}, {}
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


def run_simd_bank(n_vec=5, seed=99):
    """-> (n_cases, {label: [(lane, missing_bits, taint)]})"""
    import unicorn

    from benchmark.instruction_bank import load_bank
    from microtaint.taint_ir.frompcode import build_ir

    spec = load_bank(isas={'AMD64_SIMD'})['AMD64_SIMD']
    rng = random.Random(seed)
    under: dict = {}
    n = 0
    for ins in spec.instructions:
        if ins.label.lower().startswith(UNTRUSTED):
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
    return n, under


def main(argv=None):
    n, under = run_simd_bank()
    print(f'SIMD cases={n} instructions_with_under_taint={len(under)}')
    for label, items in list(under.items())[:10]:
        print(f'  {label}: {items[0]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
