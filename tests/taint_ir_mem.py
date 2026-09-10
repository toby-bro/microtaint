"""Ground truth for memory taint, and the IR's answer, compared per bit.

The register-only gate cannot say anything about a load or a store, so this
builds the same kind of oracle over guest memory: Unicorn runs the instruction
from a clean base state, every tainted bit -- in a register OR in a memory byte
-- is flipped in turn, and every output that moves, register, flag or memory
byte, is tainted.  That is the definition the whole engine is held to, extended
to the one place the bank harness deliberately skipped.

Against it runs the lowered IR under its two-pass protocol: the program is
evaluated once to obtain the addresses it wants, the caller resolves those
against guest memory and the shadow, and it is evaluated again for the taint.
The protocol is what keeps the emitted program straight-line and call-free, and
it is only valid while no address depends on a value the same instruction
loaded, which the builder checks.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from microtaint.types import Architecture

if TYPE_CHECKING:                    # deferred: unicorn and the engine are
    from tests.oracle_harness import UcDesc  # slow to import

#: The IR keys a register by (kind, byte offset, size).
IRKey = tuple[Any, ...]
IRState = dict[IRKey, int]
#: A ground-truth row: register/flag masks, plus '@mem' as per-byte taint.
MemTruth = dict[str, Any]

MASK64 = 0xFFFFFFFFFFFFFFFF
CODE_ADDR = 0x1000
DATA_ADDR = 0x40000
DATA_LEN = 64


@dataclass
class MemCase:
    label: str
    code: bytes
    isa: str


def _ks(isa: str) -> keystone.Ks:
    import keystone
    if isa == 'AMD64':
        return keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    if isa == 'ARM64':
        return keystone.Ks(keystone.KS_ARCH_ARM64, keystone.KS_MODE_LITTLE_ENDIAN)
    raise ValueError(isa)


AMD64_CASES = [
    'mov rax, [rbx]', 'mov rax, [rbx+8]', 'mov eax, [rbx+4]',
    'mov al, [rbx+1]', 'movzx eax, word ptr [rbx+2]',
    'mov [rbx], rax', 'mov [rbx+8], rcx', 'mov [rbx+4], eax',
    'mov [rbx+1], al', 'add rax, [rbx]', 'add [rbx], rax',
    'xor rax, [rbx+8]', 'and [rbx], rcx', 'or rax, [rbx]',
    'sub rax, [rbx+16]', 'cmp rax, [rbx]', 'inc qword ptr [rbx]',
    'not qword ptr [rbx]', 'neg qword ptr [rbx]', 'shl qword ptr [rbx], 3',
]

ARM64_CASES = [
    'ldr x0, [x1]', 'ldr x0, [x1, #8]', 'ldr w0, [x1, #4]',
    'ldrb w0, [x1, #1]', 'ldrh w0, [x1, #2]',
    'str x0, [x1]', 'str x0, [x1, #8]', 'str w0, [x1, #4]',
    'strb w0, [x1, #1]',
]

#: The register the corpus points at the data page, per ISA.
_PTR = {'AMD64': 'RBX', 'ARM64': 'X1'}


def build_cases(isa: str) -> list[MemCase]:
    src = AMD64_CASES if isa == 'AMD64' else ARM64_CASES
    ks = _ks(isa)
    out: list[MemCase] = []
    for label in src:
        try:
            code = bytes(ks.asm(label)[0])
        except Exception:
            continue
        out.append(MemCase(label, code, isa))
    return out


# ── Unicorn ground truth over registers AND memory ───────────────────

def _uc_desc(isa: str) -> UcDesc:
    from tests import oracle_harness as oh
    desc = oh._uc_desc_amd64() if isa == 'AMD64' else oh._uc_desc_arm64()
    desc.tag = isa
    return desc


def _uc_run_mem(desc: UcDesc, code: bytes, reg_vals: dict[str, int],
                mem_bytes: Sequence[int]) -> MemTruth:
    import unicorn
    uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
    uc.mem_map(CODE_ADDR, 0x1000)
    uc.mem_map(DATA_ADDR, 0x1000)
    uc.mem_write(CODE_ADDR, code)
    uc.mem_write(DATA_ADDR, bytes(mem_bytes))
    for name, const in desc.gp.items():
        uc.reg_write(const, reg_vals.get(name, 0) & desc.mask)
    uc.emu_start(CODE_ADDR, CODE_ADDR + len(code))
    out = {n: uc.reg_read(c) & desc.mask for n, c in desc.gp.items()}
    if desc.eflags_reg is not None:
        ef = uc.reg_read(desc.eflags_reg)
        for fname, bit in desc.flags.items():
            out[fname] = (ef >> bit) & 1
    out['@mem'] = list(uc.mem_read(DATA_ADDR, DATA_LEN))
    return out


def ground_truth_mem(desc: UcDesc, code: bytes, reg_taint: dict[str, int],
                     reg_vals: dict[str, int], mem_taint: Sequence[int],
                     mem_vals: Sequence[int]) -> MemTruth:
    """Per-bit sensitivity over registers and the data page.

    Returns a dict of register/flag masks plus '@mem', a list of per-byte taint.
    """
    base_regs = {n: (reg_vals.get(n, 0) & ~reg_taint.get(n, 0) & desc.mask)
                 for n in desc.gp}
    base_regs[_PTR[desc.tag]] = DATA_ADDR
    base_mem = [mem_vals[i] & ~mem_taint[i] & 0xFF for i in range(DATA_LEN)]
    base = _uc_run_mem(desc, code, base_regs, base_mem)

    result: MemTruth = {k: (0 if k != '@mem' else [0] * DATA_LEN) for k in base}

    def accumulate(out: MemTruth) -> None:
        for k in base:
            if k == '@mem':
                for i in range(DATA_LEN):
                    result['@mem'][i] |= base['@mem'][i] ^ out['@mem'][i]
            else:
                result[k] |= base[k] ^ out[k]

    for src in desc.gp:
        # The pointer is NOT held fixed: a tainted address bit is the case the
        # load rule exists for, and the corpus only taints bits low enough that
        # every reachable address stays inside the mapped window.
        tm = reg_taint.get(src, 0) & desc.mask
        bit = 0
        while tm:
            if tm & 1:
                r = dict(base_regs)
                r[src] = (base_regs[src] | (1 << bit)) & desc.mask
                accumulate(_uc_run_mem(desc, code, r, base_mem))
            tm >>= 1
            bit += 1
    for i in range(DATA_LEN):
        tm = mem_taint[i] & 0xFF
        bit = 0
        while tm:
            if tm & 1:
                m = list(base_mem)
                m[i] = base_mem[i] | (1 << bit)
                accumulate(_uc_run_mem(desc, code, base_regs, m))
            tm >>= 1
            bit += 1
    return result


# ── the IR under its two-pass protocol ───────────────────────────────

def _read_mem(mem: Sequence[int], addr: int, size: int, be: bool) -> int:
    v = 0
    for i in range(size):
        byte = mem[(addr - DATA_ADDR + i) % DATA_LEN]
        v |= byte << (8 * (size - 1 - i) if be else 8 * i)
    return v


def _write_mem(mem: list[int], addr: int, size: int, val: int, be: bool) -> None:
    for i in range(size):
        sh = 8 * (size - 1 - i) if be else 8 * i
        mem[(addr - DATA_ADDR + i) % DATA_LEN] = (val >> sh) & 0xFF


def ir_mem_taint(arch: Architecture, code: bytes, reg_taint: dict[str, int],
                 reg_vals: dict[str, int], mem_taint: Sequence[int],
                 mem_vals: Sequence[int], *, be: bool = False,
                 policy: str = 'avalanche') -> tuple[dict[str, int], list[int]]:
    """Run one instruction's taint program over registers and memory.

    Returns (register taint by name, per-byte memory taint), or raises
    Unsupported when the shape is outside the lowering.
    """
    from microtaint.taint_ir.frompcode import build_ir
    from tests.taint_ir_bank import name_offset

    prog = build_ir(arch, code, pointer_policy=policy)
    # The IR keys registers by byte offset, since one offset carries several
    # names; the caller names them however its own state does.
    vals: IRState = {}
    tnts: IRState = {}
    for n in set(reg_vals) | set(reg_taint):
        off = name_offset(arch, n)
        if off is None:
            continue
        for sz in range(1, 9):
            vals[('reg', off, sz)] = reg_vals.get(n, 0)
            tnts[('reg', off, sz)] = reg_taint.get(n, 0)
    for k in range(len(prog.accesses)):
        vals[('mem', k)] = 0
        tnts[('mem', k)] = 0

    out1 = prog.run(vals, tnts)
    for k, acc in enumerate(prog.accesses):
        addr = out1[('addr', k)]
        if acc['kind'] == 'load':
            vals[('mem', k)] = _read_mem(mem_vals, addr, acc['size'], be)
            tnts[('mem', k)] = _read_mem(mem_taint, addr, acc['size'], be)
    out = prog.run(vals, tnts)

    # A register the instruction does not write keeps its input taint; the
    # program only emits what it changes, so pass-through is reconstructed here.
    reg_out = dict(reg_taint)
    for n in reg_out:
        off = name_offset(arch, n)
        if off is None:
            continue
        for sz in range(1, 9):
            if ('reg', off, sz) in out:
                reg_out[n] = out[('reg', off, sz)]
                break
    mem_out = list(mem_taint)
    for k, acc in enumerate(prog.accesses):
        if acc['kind'] != 'store':
            continue
        if out[('addrt', k)]:
            raise TaintedStoreAddress(acc)
        _write_mem(mem_out, out[('addr', k)], acc['size'],
                   out[('sttaint', k)], be)
    return reg_out, mem_out


class TaintedStoreAddress(Exception):
    """A store through a secret-dependent address: the caller must widen."""


# ── the comparison ───────────────────────────────────────────────────

@dataclass
class MemReport:
    n: int = 0
    exact: int = 0
    over: int = 0
    under: int = 0
    declined: int = 0
    under_examples: list[tuple[str, dict[str, int]]] = field(default_factory=list)

    def summary(self) -> str:
        return (f'cases={self.n} exact={self.exact} over={self.over} '
                f'UNDER={self.under} declined={self.declined}')


def run_mem_bank(isa: str = 'AMD64', n_vec: int = 4, seed: int = 7,
                 policy: str = 'avalanche') -> MemReport:
    from microtaint.taint_ir.frompcode import Unsupported
    from microtaint.types import Architecture
    from tests.perop_c_bank import _engine_names

    arch = getattr(Architecture, isa)
    desc = _uc_desc(isa)          # `_uc_desc` sets desc.tag
    rep = MemReport()
    be = False
    for case in build_cases(isa):
        rng = random.Random(f'mem:{seed}:{case.label}')
        for _ in range(n_vec):
            reg_vals = {n: rng.randint(1, MASK64) for n in desc.gp}
            reg_vals[_PTR[isa]] = DATA_ADDR
            reg_taint = dict.fromkeys(desc.gp, 0)
            # A few tainted bits: on one non-pointer register and in memory.
            others = [n for n in desc.gp if n != _PTR[isa]]
            reg_taint[rng.choice(others)] = 1 << rng.randint(0, 63)
            if policy == 'avalanche' and rng.random() < 0.35:
                # A secret-dependent address: bits 3..5 keep every reachable
                # address inside the 64-byte window, so the oracle can still
                # enumerate it.
                reg_taint[_PTR[isa]] = 1 << rng.randint(3, 5)
            mem_vals = [rng.randint(0, 255) for _ in range(DATA_LEN)]
            mem_taint = [0] * DATA_LEN
            for _ in range(2):
                mem_taint[rng.randint(0, 23)] |= 1 << rng.randint(0, 7)

            names = list(desc.gp) + list(desc.flags)
            alias = _engine_names(arch, names)
            eng_vals = {alias[n]: reg_vals.get(n, 0) for n in names}
            eng_taint = {alias[n]: reg_taint.get(n, 0) for n in names}
            init = _uc_initial(desc)
            eng_vals.update({alias.get(k, k): v for k, v in init.items()})
            try:
                got_reg, got_mem = ir_mem_taint(arch, case.code, eng_taint,
                                                eng_vals, mem_taint, mem_vals,
                                                be=be, policy=policy)
            except (Unsupported, TaintedStoreAddress):
                rep.declined += 1
                continue
            truth = ground_truth_mem(desc, case.code, reg_taint, reg_vals,
                                     mem_taint, mem_vals)
            rep.n += 1
            # Only flags the p-code actually writes are in scope: x86 leaves
            # AF after `cmp` undefined and SLEIGH never computes it, so the
            # hardware moves a bit no engine reading that p-code can know about.
            from tests.perop_c_bank import written_registers
            try:
                written = written_registers(arch, case.code)
            except Exception:
                written = None
            under, over = {}, {}
            for k in list(desc.gp) + list(desc.flags):
                if written is not None and k in desc.flags and k not in written:
                    continue
                g = int(got_reg.get(alias[k], 0))
                r = int(truth.get(k, 0))
                if r & ~g:
                    under[k] = r & ~g
                if g & ~r:
                    over[k] = g & ~r
            for i in range(DATA_LEN):
                g, r = got_mem[i] & 0xFF, truth['@mem'][i] & 0xFF
                if r & ~g:
                    under[f'mem[{i}]'] = r & ~g
                if g & ~r:
                    over[f'mem[{i}]'] = g & ~r
            if under:
                rep.under += 1
                if len(rep.under_examples) < 8:
                    rep.under_examples.append((case.label, under))
            elif over:
                rep.over += 1
            else:
                rep.exact += 1
    return rep


def _uc_initial(desc: UcDesc) -> dict[str, int]:
    from tests.perop_c_bank import uc_initial_state
    return uc_initial_state(desc)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=['AMD64'])
    ap.add_argument('--vectors', type=int, default=4)
    ap.add_argument('--policy', default='avalanche',
                    choices=['avalanche', 'concrete'])
    args = ap.parse_args(argv)
    for isa in args.isas:
        rep = run_mem_bank(isa, n_vec=args.vectors, policy=args.policy)
        print(f'[{isa}] {rep.summary()}')
        for label, under in rep.under_examples:
            print(f'    UNDER {label}: '
                  f'{[(k, hex(v)) for k, v in list(under.items())[:6]]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
