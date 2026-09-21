#!/usr/bin/env python3
"""detect_taintgrind.py -- the TaintGrind baseline (results/taintgrind.json).

Builds the two harnesses, runs them under the pinned taintgrind container, and
parses what they emit.  Before this existed, `taintgrind.json` was typed by
hand from the trace and a re-run could not check it.

Two things are read off each run:

  * the CT metric is the number of trace lines tagged `IfGoto`, which is an
    Ist_Exit whose guard is tainted, i.e. a conditional jump whose outcome
    depends on the secret.  One per loop iteration on the vulnerable variant.
  * the DNS harness asks taintgrind directly, with TNT_IS_TAINTED on the
    extracted opcode byte, and prints `out_al_tainted=<mask>`.

Run:  python3 detect_taintgrind.py
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
CONTAINER = 'taintgrind:latest'

#: The harnesses mark their own input with TNT_TAINT, so they are compiled
#: against valgrind's headers on the host.
GCC = ['gcc', '-O0', '-g', '-static', '-no-pie', '-fno-stack-protector',
       '-I.', '-I/usr/include/valgrind']
VALGRIND_H = Path('/usr/include/valgrind/valgrind.h')

CT_SRC, CT_BIN = HERE / 'ct_tg.c', HERE / 'ct_tg'
DNS_SRC, DNS_BIN = HERE / 'dns_tg.c', HERE / 'dns_tg'

CT_STDIN = (5).to_bytes(4, 'little')   # the secret exponent, value 5
DNS_STDIN = bytes([0x88])              # QR(bit7)=1, OPCODE(bits6..3)=0001

MT_VULN_MIN = 1
MT_CT = 0


def preflight() -> str:
    if shutil.which('docker') is None:
        return 'docker CLI not found on PATH'
    imgs = subprocess.run(['docker', 'images', '-q', CONTAINER],
                          capture_output=True, text=True)
    if not imgs.stdout.strip():
        return f'container image {CONTAINER} not present; run setup_envs.sh'
    if not VALGRIND_H.exists():
        return (f'{VALGRIND_H} is missing: the harnesses mark their own input '
                f'with TNT_TAINT and are compiled on the host')
    for src, dst in ((CT_SRC, CT_BIN), (DNS_SRC, DNS_BIN)):
        if dst.exists():
            continue
        p = subprocess.run([*GCC, '-o', str(dst), str(src)],
                           cwd=HERE, capture_output=True, text=True)
        if p.returncode != 0:
            return f'building {dst.name} failed: {p.stderr.strip()[-300:]}'
    return ''


def run_guest(binary: Path, argv: list[str], stdin: bytes) -> str:
    """The container's ENTRYPOINT already runs taintgrind, so pass the guest."""
    p = subprocess.run(
        ['docker', 'run', '-i', '--rm', '-v', f'{HERE}:/pwd', CONTAINER,
         f'/pwd/{binary.name}', *argv],
        input=stdin, capture_output=True, timeout=1800)
    # The trace carries the occasional stray non-UTF-8 byte, which is enough to
    # make a naive text read fail or a grep treat the whole file as binary and
    # report nothing.  Decode permissively.
    return (p.stdout + p.stderr).decode('utf-8', 'replace')


def count_ifgoto(text: str) -> tuple[int, dict]:
    """Tainted conditional jumps, and how many times each site fired."""
    sites: dict[str, int] = {}
    n = 0
    for line in text.splitlines():
        if '| IfGoto |' not in line:
            continue
        n += 1
        m = re.match(r'(0x[0-9A-Fa-f]+):', line)
        if m:
            pc = m.group(1).lower()
            sites[pc] = sites.get(pc, 0) + 1
    return n, sites


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    blocker = preflight()
    if blocker:
        print(f'[taintgrind] blocker: {blocker}', file=sys.stderr)
        return 2

    vuln_txt = run_guest(CT_BIN, ['vuln'], CT_STDIN)
    ct_txt = run_guest(CT_BIN, ['ct'], CT_STDIN)
    dns_txt = run_guest(DNS_BIN, [], DNS_STDIN)

    vuln_n, vuln_sites = count_ifgoto(vuln_txt)
    ct_n, ct_sites = count_ifgoto(ct_txt)

    m = re.search(r'opcode_value=(0x[0-9a-f]+) out_al_tainted=(-?\d+)', dns_txt)
    if m is None:
        raise SystemExit('the DNS harness printed no verdict line, so nothing '
                         f'was measured. Its output was:\n{dns_txt[-1500:]}')
    # TNT_IS_TAINTED hands back a mask; the low byte is AL's.
    mask = int(m.group(2)) & 0xFFFFFFFF
    (RESULTS / '_taintgrind_raw.json').write_text(json.dumps({
        'vuln_ifgoto': vuln_n, 'vuln_sites': vuln_sites,
        'ct_ifgoto': ct_n, 'ct_sites': ct_sites,
        'dns_opcode_value': m.group(1), 'dns_taint_mask': hex(mask),
    }, indent=2) + '\n')

    out = {
        'tool': 'taintgrind',
        'config': {
            'docker_image': CONTAINER,
            'invocation': f'docker run -i --rm -v <dir>:/pwd {CONTAINER} '
                          f'/pwd/<binary> [argv]',
            'taint_source': 'explicit TNT_TAINT(addr,size) from taintgrind.h, '
                            'applied in-harness right after the read()',
            'build_flags': ' '.join(GCC),
            'branch_leak_metric': "count of trace lines tagged 'IfGoto' "
                                  '(an Ist_Exit whose guard is tainted, i.e. a '
                                  'tainted conditional jump)',
        },
        'ct': {
            'workload': 'test_constant_time.c (square-and-multiply modexp)',
            'vuln_branch_leaks': vuln_n,
            'vuln_branch_leak_sites': sorted(vuln_sites),
            'ct_branch_leaks': ct_n,
            'ct_branch_leak_sites': sorted(ct_sites),
            'matches_microtaint': vuln_n >= MT_VULN_MIN and ct_n == MT_CT,
            'ran': True,
        },
        'dns': {
            'workload': 'test_bitpacked_dns.c LDNS_OPCODE_WIRE: '
                        'AND AL,0x78 then SHR AL,3',
            'granularity': 'byte / register (VEX Ity_I8 shadow); taintgrind '
                           'tracks taint per byte, not per bit',
            'can_taint_single_bit': False,
            'input_flag_byte': hex(DNS_STDIN[0]),
            'opcode_value': m.group(1),
            'taint_mask': hex(mask),
            'byte_taint_output_al': f'0x{mask & 0xFF:02x}',
            'output_tainted': mask != 0,
            # One tag covers the whole flag byte, so "taint QR only" is not
            # expressible and the masked-away bit still reaches the output.
            'can_discriminate_qr_vs_opcode': False,
            'matches_microtaint': False,
            'ran': True,
        },
    }
    (RESULTS / 'taintgrind.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'[taintgrind] CT vuln={vuln_n} ct={ct_n} | '
          f'DNS out_al taint mask={hex(mask & 0xFF)} '
          f'tainted={mask != 0} -> {RESULTS / "taintgrind.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
