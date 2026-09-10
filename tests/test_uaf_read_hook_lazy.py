# ruff: noqa: S603, S607, S110, PLC0415
"""The UAF read callback is armed by the first poison, not at startup.

Unicorn calls a UC_HOOK_MEM_READ callback on every guest load.  Registering one
for the whole run cost 97 ns per instruction on bench_dense (measured by
running the same workload with check_uaf on and off), and until something has
been freed that callback provably has nothing to report: is_poisoned is false
everywhere because nothing is poisoned.

So the wrapper does not register it up front.  BitPreciseShadowMemory.poison
fires a one-shot callback the first time it runs, and the wrapper registers the
hook there, before returning to the guest.  Four places poison, and all of them
go through poison(), which is why the choke point is in the shadow rather than
at the call sites.

This is the property that has to hold: a read from poisoned memory is still
reported.  It was not covered before -- the CLI's UAF tests write after free
rather than reading, and the heap tracker's read-after-free test skips wherever
libc symbol hooking does not take effect (it does not, here).  A read after
munmap cannot stand in for it either: Qiling really unmaps the page, so that
faults rather than reaching the poison check.

The test therefore drives the mechanism the heap tracker uses: poison a region
that is still mapped, from between two guest instructions, exactly as free()
does, and let the guest read it.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator

import pytest
from qiling import Qiling

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

# Freestanding, and it reads its own stack frame constantly, so poisoning the
# current frame is certain to be read before the program can exit.
_SRC = r"""
long sys_exit(int s){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(s):"rcx","r11","memory");return r;}
int work(int a){ volatile int b[8]; for (int i=0;i<8;i++) b[i]=a+i;
                 int s=0; for (int i=0;i<8;i++) s+=b[i]; return s; }
void _start(void){ int s=0; for (int i=0;i<64;i++) s+=work(i); sys_exit(s & 1); }
"""

#: Poison at this instruction: late enough that the program is inside work()
#: with a live frame, early enough that plenty of reads follow.
_POISON_AT = 200


@pytest.fixture(scope='module')
def binary() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(
        ['gcc', '-nostdlib', '-O0', '-fno-stack-protector', '-static',
         '-o', path, '-x', 'c', '-'],
        input=_SRC.encode(), check=True)
    yield path
    os.unlink(path)


def _run(binary: str, poison: bool,
         ) -> tuple[MicrotaintWrapper, Reporter, dict[str, object]]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(b'')
    reporter = Reporter(json_mode=False, stream=io.StringIO())
    wrapper = MicrotaintWrapper(ql, check_uaf=True, check_bof=False,
                                check_sc=False, reporter=reporter)
    #: 'before' is the read hook as it stood before the run, 'poisoned'
    #: what it became, and 'n' counts the callbacks.
    seen: dict[str, object] = {'before': wrapper._mem_read_hook,
                               'poisoned': None, 'n': 0}

    if poison:
        def poison_once(_ql: Qiling, _addr: int, _size: int,
                        _ud: object = None) -> None:
            n = seen['n']
            assert isinstance(n, int)
            seen['n'] = n = n + 1
            if n != _POISON_AT:
                return
            sp = _ql.arch.regs.arch_sp
            wrapper.shadow_mem.poison(sp, 0x40)
            seen['poisoned'] = sp
            seen['armed_right_after'] = wrapper._mem_read_hook
        ql.hook_code(poison_once)

    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        ql.run()
    except Exception:
        pass
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)
    return wrapper, reporter, seen


def test_read_hook_is_not_registered_until_something_is_poisoned(
        binary: str) -> None:
    """The optimisation: no poison in the run, so no read callback at all."""
    wrapper, reporter, seen = _run(binary, poison=False)
    assert seen['before'] is None, \
        'the UAF read hook was registered at startup, before any poison'
    assert wrapper._mem_read_hook is None, (
        'nothing was ever poisoned, so the read hook should never have been '
        'armed -- Unicorn would be calling it on every guest load for nothing'
    )
    assert not [f for f in reporter.findings if f.kind.name == 'UAF']


def test_first_poison_arms_the_read_hook_and_the_read_is_reported(
        binary: str) -> None:
    """The safety property: poisoning still detects a subsequent read."""
    wrapper, reporter, seen = _run(binary, poison=True)
    assert seen['poisoned'] is not None, 'the run ended before poisoning'
    assert seen['armed_right_after'] is not None, (
        'poison() returned without arming the read hook -- a read from freed '
        'memory between here and the next arming point would go unreported'
    )
    assert wrapper._mem_read_hook is not None
    uaf = [f for f in reporter.findings if f.kind.name == 'UAF']
    assert uaf, (
        'a read from poisoned memory was not reported; findings were '
        f'{[f.kind.name for f in reporter.findings]}'
    )
