# ruff: noqa: PLC0415, S110
"""Run one guest under the engine and print its taint outcome as JSON.

A separate process on purpose: MICROTAINT_ARR_HOOK is read once, at import, so
the two paths cannot be compared inside a single interpreter.  Invoked by
tests/test_arr_hook_parity.py as

    python -m tests.arr_hook_probe <guest.elf> <stdin-hex>

Taint is read back through shadow_mem.read_mask and the register_taint property.
The memory read is the important one: it does not disturb the array path, while
register_taint hands authority back to the dict and makes the next instruction
reload, so it is taken once, at the end.
"""
from __future__ import annotations

import io
import json
import os
import sys


def main() -> int:
    guest, stdin_hex = sys.argv[1], sys.argv[2]
    stdin_data = bytes.fromhex(stdin_hex)

    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    import microtaint
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(stdin_data)
    reporter = Reporter(json_mode=False, stream=io.StringIO())
    wrapper = MicrotaintWrapper(ql, check_bof=True, check_uaf=True, check_sc=True,
                                check_aiw=True, reporter=reporter)
    os.dup2(devnull, 1)
    try:
        ql.run()
    except Exception:  # noqa: BLE001 - guests exit via a syscall
        pass
    os.dup2(saved, 1)

    mem = {}
    for region in ql.mem.map_info:
        lo, hi = region[0], region[1]
        if hi - lo > (1 << 20):     # skip regions too large to walk here
            continue
        for addr in range(lo, hi, 8):
            try:
                mask = wrapper.shadow_mem.read_mask(addr, 8)
            except Exception:
                mask = 0
            if mask:
                mem[str(addr)] = int(mask)

    json.dump({
        'engine': microtaint.__file__,
        'arr_hook': os.environ.get('MICROTAINT_ARR_HOOK', '1'),
        'findings': sorted(
            (str(getattr(f, 'kind', '?')), int(getattr(f, 'address', 0) or 0))
            for f in getattr(reporter, 'findings', [])),
        'regs': {k: int(v) for k, v in wrapper.register_taint.items() if v},
        'mem': mem,
    }, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
