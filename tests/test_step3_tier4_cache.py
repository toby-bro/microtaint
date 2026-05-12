"""
Regression test for step 3 — the Tier-4 (version-keyed) instruction cache.

Fully self-contained: the C reproducer is embedded as a string, compiled
into a temporary directory by a session-scoped pytest fixture.  No pre-built
files or hard-coded addresses are required.  All loop addresses are discovered
from the compiled binary at runtime via capstone disassembly.

Background
----------
`hook_core.pyx` keeps two parallel instruction caches per address:

- Tier-3 (legacy): keyed by ``frozenset(register_taint.items())``.
  Correct but pays a frozenset construction on every visit.

- Tier-4 (version): keyed by ``self.taint_version`` — a single integer.
  Stores ``(in_version, out_version, output_state)``.  On lookup, only
  ``in_version == self.taint_version`` is checked; no comparison of
  actual register_taint contents is performed.

Problem
-------
``taint_version`` is updated via three independent code paths:
  1. Cacheable slow path  → set to ``hash(frozenset(output_state.items()))``
  2. Has-mem-ops slow path → ``self.taint_version += 1``
  3. Tier-4 hit           → set to the stored ``out_version``

Mixing increments and content hashes means ``taint_version`` is NOT a
reliable fingerprint of register_taint contents.  Two distinct states
can share a taint_version; Tier-4 then replays output computed for the
wrong state.

Fix
---
Store a fourth element ``input_snapshot`` (copy of register_taint at
store time) in every Tier-4 entry.  On a version hit, additionally verify
``register_taint == input_snapshot`` before adopting the cached output.
"""

# ruff: noqa: S603,PLW1510,PLC0415,C901,S607,ARG001,S110
# mypy: disable-error-code="no-untyped-def,import-untyped,attr-defined,call-overload,no-untyped-call"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import io
import logging
import subprocess
import textwrap

import pytest
from qiling import Qiling
from qiling.const import QL_INTERCEPT, QL_VERBOSE
from unicorn import UC_HOOK_CODE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper
from microtaint.sleigh.engine import _cached_generate_static_rule

logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Embedded C reproducer
# ---------------------------------------------------------------------------

_MIN_LOOP_C = r"""
#include <stdint.h>
#include <unistd.h>
int main(void) {
    unsigned char buf[16];
    read(0, buf, 16);
    uint64_t state = 0;
    const unsigned char *p = buf;
    const unsigned char *end = buf + 16;
    for (; p != end; p++) state ^= (uint64_t)(*p);
    write(1, &state, 8);
    return 0;
}
"""
# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def min_loop_binary(tmp_path_factory):
    """Compile the embedded C source; return path to the binary."""
    d = tmp_path_factory.mktemp('min_loop')
    src = d / 'min_loop.c'
    binary = d / 'min_loop'
    src.write_text(textwrap.dedent(_MIN_LOOP_C))
    result = subprocess.run(
        ['gcc', '-O0', '-g', '-static', str(src), '-o', str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f'gcc failed to compile the embedded C source:\n{result.stderr}'
    return str(binary)


@pytest.fixture(scope='session')
def loop_addrs(min_loop_binary):
    """Disassemble the binary and locate the loop body instructions.

    Returns a dict with keys:
        'loop_head'     — address of  mov rax, [rbp-N]   (load ptr, loop top)
        'load_byte'     — address of  movzx eax, byte [rax]
        'movzx_edx_al'  — address of  movzx edx, al
        'observe'       — address of instruction after movzx_edx_al
                          (observation point: hooking here gives us
                          register_taint AFTER microtaint processed
                          movzx_edx_al, because microtaint's hook
                          registered first fires first)

    Skips the session if capstone is unavailable.
    Fails loudly if the loop pattern cannot be found.
    """
    try:
        from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    except ImportError:
        pytest.skip('capstone not available — needed for address discovery')

    r = subprocess.run(
        ['objdump', '-d', '-M', 'intel', min_loop_binary],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f'objdump failed:\n{r.stderr}'

    # Collect raw bytes for <main>.
    in_main = False
    raw_bytes = bytearray()
    base_addr = None
    for line in r.stdout.splitlines():
        if '<main>:' in line:
            in_main = True
            continue
        if in_main:
            if line and not line.startswith(' ') and '<' in line:
                break
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    addr = int(parts[0].strip().rstrip(':'), 16)
                except ValueError:
                    continue
                if base_addr is None:
                    base_addr = addr
                for b in parts[1].strip().split():
                    try:
                        raw_bytes.append(int(b, 16))
                    except ValueError:
                        pass

    assert base_addr is not None, 'could not locate <main> in objdump output'

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = False
    insns = list(md.disasm(bytes(raw_bytes), base_addr))

    # Pattern: movzx eax, byte [rax]  preceded by  mov rax, [rbp…]
    #          and followed by  movzx edx, al
    addrs = {}
    for i, ins in enumerate(insns):
        if (
            ins.mnemonic == 'movzx'
            and 'eax' in ins.op_str
            and 'byte ptr [rax]' in ins.op_str
            and i > 0
            and insns[i - 1].mnemonic == 'mov'
            and 'rax' in insns[i - 1].op_str
            and i + 1 < len(insns)
            and insns[i + 1].mnemonic == 'movzx'
            and 'edx' in insns[i + 1].op_str
            and 'al' in insns[i + 1].op_str
            and i + 2 < len(insns)
        ):
            addrs['loop_head'] = insns[i - 1].address
            addrs['load_byte'] = ins.address
            addrs['movzx_edx_al'] = insns[i + 1].address
            addrs['observe'] = insns[i + 2].address
            break

    missing = [k for k in ('loop_head', 'load_byte', 'movzx_edx_al', 'observe') if k not in addrs]
    assert not missing, (
        f'Could not find loop instruction(s) {missing} in {min_loop_binary}.\n'
        'Full <main> disassembly:\n' + '\n'.join(f'  0x{i.address:x}  {i.mnemonic} {i.op_str}' for i in insns)
    )
    return addrs


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def _gather_observations(binary, addrs, taint_offset):
    """Emulate ``binary`` with bit-0 of input byte ``taint_offset`` tainted.

    Returns ``(final_shadow, tainted_iter, observations)`` where:
      - final_shadow   : 64-bit shadow of the 8-byte stdout write
      - tainted_iter   : 1-based iteration index in which the tainted byte
                         was loaded (detected by shadow_mem at load time)
      - observations   : list of (iter_index, register_taint_snapshot)
                         captured at addrs['observe'], i.e. immediately
                         after microtaint processed movzx_edx_al

    Iteration index is incremented each time addrs['loop_head'] is entered.
    Because the loop in min_loop is a do-while-style (entry via jmp to
    the condition), the first byte (index 0) is processed in iter 1.
    """
    msg = bytes(range(16))
    output_shadow = []
    wrapper_ref = [None]
    observations = []
    tainted_iter_found = [None]

    def write_hook(ql, fd, buf, count, *_):
        if fd == 1 and count == 8:
            output_shadow.append(wrapper_ref[0].shadow_mem.read_mask(buf, 8))
        return count

    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)

    class _Stdin:
        def read(self, n):
            return msg[:n]

    ql.os.stdin = _Stdin()
    ql.os.set_syscall(1, write_hook, QL_INTERCEPT.CALL)

    r = Reporter(json_mode=True, stream=io.StringIO())
    w = MicrotaintWrapper(
        ql,
        check_sc=False,
        check_bof=False,
        check_uaf=False,
        check_aiw=False,
        reporter=r,
    )
    wrapper_ref[0] = w

    def read_hook(ql, fd, buf, count):
        if fd != 0:
            return 0
        ql.mem.write(buf, msg[:count])
        w.taint_bit(buf + taint_offset, 0)
        return count

    ql.os.set_syscall(0, read_hook, QL_INTERCEPT.CALL)

    loop_head_addr = addrs['loop_head']
    observe_addr = addrs['observe']
    state = {'iter': 0}

    def watcher(uc, address, size, user_data):
        if address == loop_head_addr:
            state['iter'] += 1

        if address == observe_addr:
            rt = {k: v for k, v in w.register_taint.items() if v != 0}
            observations.append((state['iter'], rt))
            # The tainted-byte iteration is the first one where register_taint
            # is non-empty at this point (movzx edx,al propagated the taint).
            if rt and tainted_iter_found[0] is None:
                tainted_iter_found[0] = state['iter']

    ql.uc.hook_add(UC_HOOK_CODE, watcher, None, 1, 0)

    try:
        ql.run()
    except Exception:
        pass

    return (
        output_shadow[0] if output_shadow else 0,
        tainted_iter_found[0],
        observations,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tier4_cache_does_not_corrupt_register_taint(min_loop_binary, loop_addrs):
    """The Tier-4 cache must not replay output computed for a different state.

    We observe register_taint at loop_addrs['observe'] — the instruction
    immediately after  movzx edx, al — in the iteration that loaded the
    tainted byte.  This is AFTER microtaint has processed movzx_edx_al.

    Hand-computed expectation for that iteration (taint_offset=5, bit 0):

        mov rax, [rbp-N]        ; load pointer p (untainted)
          → register_taint = {}
        movzx eax, byte [rax]   ; *p → EAX; byte[5] carries taint bit 0
          → register_taint = {RAX: 0x1}
        movzx edx, al           ; zero-extend AL → EDX
          → register_taint = {RAX: 0x1, RDX: 0x1}   ← observed here

    On the unpatched Tier-4 cache, taint_version at this point collides
    with a value stored from an earlier (untainted) iteration.  Tier-4
    hits and replays output_state={}, wiping both RAX and RDX taint.
    """
    _cached_generate_static_rule.cache_clear()
    _, tainted_iter, obs = _gather_observations(
        min_loop_binary,
        loop_addrs,
        taint_offset=5,
    )

    assert tainted_iter is not None, (
        'Could not detect which iteration loaded the tainted byte.  '
        'Check that shadow memory is readable at the load_byte address.'
    )

    hits = [(rt,) for (it, rt) in obs if it == tainted_iter]
    assert hits, (
        f'No observation captured for iteration {tainted_iter} '
        f'(the tainted-byte iteration).  '
        f'Verify that loop_addrs are correct for this binary.'
    )
    rt = hits[0][0]
    assert rt == {'RAX': 0x1, 'RDX': 0x1}, (
        f'After movzx edx, al in the tainted-byte iteration (iter {tainted_iter}),\n'
        f'register_taint should be {{RAX: 0x1, RDX: 0x1}}.\n'
        f'Got: {rt}\n\n'
        f'This means the Tier-4 cache replayed output_state={{}} from an\n'
        f'earlier iteration where taint_version collided but register_taint\n'
        f'was empty — wiping the byte taint on RAX and its copy in RDX,\n'
        f'so no taint ever reaches the XOR accumulator in memory.'
    )


def test_min_loop_end_to_end(min_loop_binary, loop_addrs):
    """Tainting any single byte (bit 0) must produce a non-zero output shadow.

    On the unpatched Tier-4 cache, bytes at offsets 2-15 produce shadow=0
    because the cache drops taint before it can reach the state accumulator.
    """
    _cached_generate_static_rule.cache_clear()
    failures = []
    for offset in range(16):
        final, _, _ = _gather_observations(min_loop_binary, loop_addrs, taint_offset=offset)
        if final == 0:
            failures.append(offset)

    assert not failures, (
        f'min_loop produced shadow=0 for byte offset(s): {failures}\n\n'
        f'Tainting bit 0 of any single input byte must produce at least\n'
        f'one tainted bit in the 8-byte XOR output written to stdout.\n'
        f'shadow=0 means the Tier-4 instruction cache dropped the taint\n'
        f'before it could propagate through the state accumulator.'
    )
