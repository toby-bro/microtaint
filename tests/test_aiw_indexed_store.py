"""An arbitrary indexed write must be reported when the attacker owns the INDEX.

The detector compared the written address against every tainted register's value
and reported when the two were within 4096 bytes.  That answers the right
question only when the tainted register IS the pointer.  The shape the detector
is actually named after is `table[idx] = v` with a tainted `idx`, which gcc
compiles to

    mov %cl,(%rdx,%rax,1)        rdx = idx (tainted), rax = &table

where the address is 0x403000+idx and the tainted register holds idx: the two are
a whole table apart, so the proximity test does not match.

It nevertheless passed for years, because `resolve_ptr_with_offset` dropped the
second register of a two-register address and resolved the store to its BASE --
which here is the tainted index itself.  The detector was reading an address the
engine had got wrong, and the wrong address happened to equal the tainted
register's value.  Fixing the addressing made the proximity test stop matching
and the detector went silent, with the whole suite still green.

So this pins the ANSWER (the finding is reported) and the SIGNAL it now comes
from (which registers decide where a store lands), not the arithmetic that used
to stand in for it.
"""
from __future__ import annotations

import platform

import pytest

from tests.test_detectors_on_later_visits import (
    _SYSCALLS,
    _compile_freestanding,
    _run_freestanding,
)

keystone = pytest.importorskip('keystone')

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

#: `table[idx & 0xfff] = idx` with a tainted idx: the attacker chooses where the
#: write lands, which is the definition of the finding.
_AIW_SRC = _SYSCALLS + r"""
static char table[4096];
void _start(void){
    unsigned long idx = 0;
    sys_read(0, &idx, 8);
    table[idx & 0xfff] = (char)idx;
    sys_exit(0);
}
"""


def test_tainted_index_into_a_table_is_reported() -> None:
    """End to end: a tainted INDEX, an untainted base, one [AIW]."""
    binary = _compile_freestanding(_AIW_SRC)
    logs = _run_freestanding(binary, b'\x41' * 8, aiw=True)
    assert any('[AIW]' in line or 'arbitrary_indexed_write' in line for line in logs), (
        f'a store to table[tainted idx] was not reported. The address is '
        f'base+index with the taint on the index, so the tainted register holds '
        f'a small offset and sits nowhere near the address it produces: a check '
        f'that compares the two cannot see it. Logs: {logs}'
    )


# ---------------------------------------------------------------------------
# The signal itself: which registers decide where a store lands.
# ---------------------------------------------------------------------------

def _addr_regs(asm: str) -> list[str]:
    from microtaint.sleigh.engine import generate_static_rule, store_address_registers
    from microtaint.types import Architecture, Register

    names = ['RAX', 'RBX', 'RCX', 'RDX', 'RDI', 'CF', 'PF', 'ZF', 'SF', 'OF']
    fmt = ([Register(n, 64) for n in names[:5]]
           + [Register(f, 1) for f in names[5:]] + [Register('RSP', 64)])
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(Architecture.AMD64, bytes(ks.asm(asm)[0]), fmt)
    return sorted(store_address_registers(rule))


def test_a_base_index_store_names_both_registers() -> None:
    """Both halves of `[base+index]` decide where the store lands."""
    assert _addr_regs('mov byte ptr [rdx + rax], cl') == ['RAX', 'RDX']
    assert _addr_regs('mov dword ptr [rdi + rax*4], ecx') == ['RAX', 'RDI']


def test_a_single_register_store_names_it() -> None:
    assert _addr_regs('mov dword ptr [rdi], ecx') == ['RDI']


def test_a_load_is_not_a_store() -> None:
    """Reading through a tainted pointer is not an arbitrary WRITE."""
    assert _addr_regs('mov ecx, dword ptr [rdi]') == []
    assert _addr_regs('mov ecx, dword ptr [rdi + rax*4]') == []


def test_the_stack_pointer_is_excluded() -> None:
    """`leave` propagates T_RBP -> T_RSP, so every later push would report."""
    assert _addr_regs('push rax') == []
    assert _addr_regs('mov dword ptr [rsp + 8], ecx') == []
