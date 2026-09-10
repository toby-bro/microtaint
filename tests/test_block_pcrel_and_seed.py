"""Block mode must lower a PC-relative operand, and must own the taint it is given.

Three defects, all under-taints, all measured on `benchmark/taint_density/
bench_sparse.elf` before they were fixed:

  * a PC-relative operand lifts to a `ram` varnode -- SLEIGH folds the
    displacement into the program counter and hands back an absolute address --
    and the lowering refused those.  A refused instruction refuses its whole
    block, and an unhandled block is SKIPPED, so nothing computes its taint.
    Two blocks in 10,701 were refused that way, and RBX, R11 and two memory
    words came back clean.
  * block mode never loaded the caller's seeded register taint into the array
    it runs on, so a register tainted before the run started out clean.
  * block mode never synced the array back, so `wrapper.register_taint` --
    the external contract, and what a detector reads -- reported nothing at
    all, whatever the run had computed.

The first is tested at the lowering, where the refusal was; the other two need
a real run, because they are about who owns the state between the wrapper, the
hook and the C runtime.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from typing import TypedDict

import pytest

from microtaint.taint_ir.blocks import plan_block
from microtaint.taint_ir.ir import Access
from microtaint.types import Architecture

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

_ARCH = Architecture.AMD64

# `movzx r11d, byte ptr [rip+0x2095]` at 0x401063 and `mov byte ptr
# [rip+0x1fe6], r11b` at 0x401113: the two instructions from bench_sparse that
# refused, and the address both resolve to.
_PCREL_LOAD = (0x401063, bytes.fromhex('440fb61d95200000'))
_PCREL_STORE = (0x401113, bytes.fromhex('44881de61f0000'))
_RESOLVED = 0x403100
#: A byte nothing in the seed guest reads, used only to arm the hooks.
_ARM_ADDR = 0x7F0000


def _one_region_accesses(base: int, code: bytes, *,
                         abs_ram: bool) -> list[Access] | None:
    """The accesses of a single-instruction lowering, or None if it refused."""
    regions = plan_block(_ARCH, code, base, abs_ram=abs_ram)
    assert len(regions) == 1, f'expected one region, got {len(regions)}'
    prog = regions[0].prog
    if prog is None:
        return None
    return list(prog.accesses)


@pytest.mark.parametrize(('label', 'base', 'code', 'kind'),
                         [('load', *_PCREL_LOAD, 'load'),
                          ('store', *_PCREL_STORE, 'store')])
def test_a_pcrel_operand_becomes_an_access_at_the_resolved_address(
        label: str, base: int, code: bytes, kind: str) -> None:
    """The address is the one SLEIGH resolved, not an offset to compute."""
    accesses = _one_region_accesses(base, code, abs_ram=True)
    assert accesses is not None, f'{label}: the lowering still refuses it'
    assert len(accesses) == 1, f'{label}: {len(accesses)} accesses, expected 1'
    acc = accesses[0]
    assert acc['kind'] == kind, f'{label}: lowered as {acc["kind"]}'
    assert acc['size'] == 1, f'{label}: size {acc["size"]}'

    # And it names the byte the instruction really touches.  The address is a
    # node in the program rather than a number, so it is read back through the
    # program's own constant folding -- which is also what proves it folded to
    # a constant, and so costs the runtime nothing to compute.
    regions = plan_block(_ARCH, code, base, abs_ram=True)
    prog = regions[0].prog
    assert prog is not None
    node = prog.nodes[acc['addr']]
    assert node[0] == 'const', f'{label}: address did not fold to a constant'
    assert node[4] == _RESOLVED, f'{label}: address {node[4]:#x} != {_RESOLVED:#x}'


@pytest.mark.parametrize(('label', 'base', 'code'),
                         [('load', *_PCREL_LOAD), ('store', *_PCREL_STORE)])
def test_a_synthetic_lift_base_still_refuses_a_pcrel_operand(
        label: str, base: int, code: bytes) -> None:
    """`abs_ram=False` must decline, because the address would be wrong.

    The per-instruction path lifts every instruction at one synthetic base, so
    a `ram` varnode's offset there names a byte the instruction never touches.
    Reading the shadow at the wrong address returns clean for a tainted byte,
    which is an under-taint -- so declining is the only sound answer, and this
    pins that the gate is what decides, not the opcode.
    """
    assert _one_region_accesses(base, code, abs_ram=False) is None, (
        f'{label}: lowered against a synthetic base, where the resolved '
        f'address names the wrong byte')


def test_the_block_that_refused_now_lowers_whole() -> None:
    """The real 146-byte block from bench_sparse, end to end.

    A region with `prog=None` is what makes the C hook count the block
    unhandled and skip it, so `every region has a program` is the property that
    matters, not the region count.
    """
    base = 0x40105E
    code = bytes.fromhex(
        '488b7424f8440fb61d952000000fb65c24f741b8000000004c8d15831f00'
        '0049b9b3010000000100004c89e1ba000000000f1f44000066662e0f1f84'
        '000000000066662e0f1f84000000000066662e0f1f84000000000066662e'
        '0f1f8400000000000fb63940c0c70389d04431c001f883c2010fb6fa4132'
        '043a8801490faff10fb6c04801c64883c10181fa0001000075d0')
    regions = plan_block(_ARCH, code, base, abs_ram=True)
    assert regions, 'the planner returned no regions'
    refused = [r for r in regions if r.prog is None]
    assert not refused, (
        f'{len(refused)} of {len(regions)} regions still refuse: '
        f'{[hex(r.addr) for r in refused]}')


# ---------------------------------------------------------------------------
# The run: who owns the register taint.
# ---------------------------------------------------------------------------

#: Taint enters through `read`, goes out through a GLOBAL (so the compiler emits
#: a PC-relative store and load), and comes back into a register that is still
#: live at the exit syscall.  All three parts are needed: a guest whose taint
#: ends in memory only cannot show a lost register, and one that never touches a
#: global cannot show the refused lowering.
_GUEST = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}
volatile unsigned char g_slot;
volatile unsigned long g_out;
void _start(void){
  unsigned char in[32];
  long n = sys_read(0, in, 32);
  if (n <= 0) sys_exit(1);
  unsigned long acc = 0;
  for (long i = 0; i < n; i++) {
    g_slot = in[i];              /* PC-relative store of a tainted byte */
    acc += g_slot;               /* PC-relative load of it back */
  }
  g_out = acc;
  sys_exit((int)(acc & 1));      /* the tainted value is live at the exit */
}
"""


def _build(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=src.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


class RunTaint(TypedDict):
    """One run's answer: register taint by name, and the block counters (None
    when block mode did not install)."""

    regs: dict[str, int]
    stats: dict[str, int] | None


def _run(guest: str, *, block: bool, seed: dict[str, int] | None = None,
         arm_at: int | None = None) -> RunTaint:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(32)))
        w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False,
                                                    stream=io.StringIO()))
        if seed:
            w.register_taint = dict(seed)
        if arm_at is not None:
            # Seeding a register does not arm the hooks -- only injecting taint
            # does -- and block mode installs when they are armed.  One byte
            # nothing reads is enough, and it has to happen AFTER the seed so
            # the load into the array sees it.
            w.taint_bit(arm_at, 0)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:  # a guest that faults is still a comparison
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        return RunTaint(regs={k: v for k, v in w.register_taint.items() if v},
                        stats=w.block_mode_stats())
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


@pytest.fixture(scope='module')
def guest() -> Iterator[str]:
    path = _build(_GUEST)
    yield path
    os.unlink(path)


@pytest.fixture(scope='module')
def both(guest: str) -> tuple[RunTaint, RunTaint]:
    return _run(guest, block=False), _run(guest, block=True)


def test_the_guest_leaves_a_tainted_register(
        both: tuple[RunTaint, RunTaint]) -> None:
    """The premise.  A guest that ends with every register clean makes the
    comparison below pass against anything, including a block mode that
    publishes nothing at all -- which is exactly the defect it is here for."""
    instr, _block = both
    assert instr['regs'], (
        'the instruction path ended with no tainted register, so there is '
        'nothing for the block path to lose')


def test_block_mode_handles_every_block(
        both: tuple[RunTaint, RunTaint]) -> None:
    """An unhandled block is unanalysed code, and it is how the PC-relative
    refusal turned into lost taint in the first place."""
    _instr, block = both
    stats = block['stats']
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 20, f'the hook barely fired: {stats}'
    assert stats['unhandled'] == 0, (
        f'{stats["unhandled"]} of {stats["blocks"]} blocks were skipped, so '
        f'their taint was never computed: {stats}')


#: Outputs Unicorn per-bit truth says DO depend on the tainted input for this
#: guest: the accumulator, the byte the loop last loaded, and the two flags of
#: the closing `and $0x1,%edi` that a tainted low bit can move.  Measured, not
#: asserted -- see test_the_sign_flag_difference_is_a_precision_gain.
_MUST_BE_TAINTED = ('RDI', 'RCX', 'PF', 'ZF')


@pytest.mark.parametrize('name', _MUST_BE_TAINTED)
def test_block_mode_publishes_the_register_taint_it_computed(
        name: str, both: tuple[RunTaint, RunTaint]) -> None:
    """`wrapper.register_taint` is the external contract, and block mode has to
    answer through it: the array it runs on is the state, and nothing else
    syncs it back.

    Scored against what must be tainted rather than against the instruction
    path's answer, because the two may legitimately differ where the block
    lowering is TIGHTER, and comparing them directly reports that as a loss.
    """
    instr, block = both
    assert instr['regs'].get(name), f'the premise: {name} is not tainted at all'
    assert block['regs'].get(name), (
        f'block mode lost {name}: the instruction path has '
        f'{instr["regs"][name]:#x} and block mode has '
        f'{block["regs"].get(name, 0):#x}')


def test_the_sign_flag_difference_is_a_precision_gain(
        both: tuple[RunTaint, RunTaint]) -> None:
    """The one register the two paths disagree on, checked against hardware.

    The guest closes with `and $0x1,%edi`, whose 32-bit result is 0 or 1, so
    SF is bit 31 of a value that is provably zero there.  The block lowering
    computes that; the whole-instruction differential does not and reports SF
    as tainted.  Unicorn is asked rather than believed: flipping every bit of
    RDI in turn moves PF and ZF and never SF.
    """
    unicorn = pytest.importorskip('unicorn')
    import unicorn.x86_const as ux

    def eflags(rdi: int) -> int:
        uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        uc.mem_map(0x1000, 0x1000)
        uc.mem_write(0x1000, bytes.fromhex('83e701'))   # and $0x1,%edi
        uc.reg_write(ux.UC_X86_REG_RDI, rdi)
        uc.reg_write(ux.UC_X86_REG_EFLAGS, 0x202)
        uc.emu_start(0x1000, 0x1000 + 3)
        got: int = uc.reg_read(ux.UC_X86_REG_EFLAGS)
        return got

    base = eflags(0)
    moved = 0
    for bit in range(64):
        moved |= base ^ eflags(1 << bit)
    assert (moved >> 6) & 1, 'ZF does not move, so the probe measured nothing'
    assert not (moved >> 7) & 1, 'SF DOES move, so tainting it is not an over-taint'

    instr, block = both
    assert instr['regs'].get('SF'), 'the instruction path no longer taints SF'
    assert not block['regs'].get('SF'), (
        'block mode now taints SF too, which the hardware says it need not')


#: Moves the seeded register somewhere else and exits.  The claim has to be
#: about a register the RUN wrote: asserting that the seeded one is still
#: tainted passes even when the seed never left the dict, which is the defect.
_SEED_GUEST = r"""
void _start(void){
  __asm__ volatile("mov %%rbx, %%r12\n\t"
                   "mov $60, %%eax\n\t"
                   "mov $0, %%edi\n\t"
                   "syscall" ::: "rax", "rdi", "r12");
  __builtin_unreachable();
}
"""


@pytest.fixture(scope='module')
def seed_guest() -> Iterator[str]:
    path = _build(_SEED_GUEST)
    yield path
    os.unlink(path)


def test_block_mode_picks_up_seeded_register_taint(seed_guest: str) -> None:
    """Taint seeded before the run must reach the array block mode runs on.

    The per-instruction path loads the dict into the array on its first call,
    and block mode never makes that call, so the seed was dropped: a register
    the caller declared secret started the run clean.  The guest copies RBX
    into R12, so what is checked is a register the run itself wrote.
    """
    seeded = _run(seed_guest, block=True, seed={'RBX': 0xFF00},
                  arm_at=_ARM_ADDR)
    assert seeded['stats'] is not None, 'block mode did not install'
    assert seeded['stats']['blocks'] > 0, 'the block hook never fired'
    assert seeded['regs'].get('R12') == 0xFF00, (
        f"the seed never reached the run: R12 is "
        f"{seeded['regs'].get('R12', 0):#x}, expected 0xff00 copied from RBX")
