#!/usr/bin/env python3
"""detect_libdft64.py -- the libdft64 per-tool baseline (results/libdft64.json).

Builds the two Pin tools and the two guests, runs them, and parses the summary
lines they print into the shared schema.  Before this existed, `libdft64.json`
was typed by hand from those lines and a re-run could not check it.

The tools print one machine-readable summary each:

  [cf_leak] cond_branches_executed=N tainted_cond_branches=N flag_writes=N
            tainted_branch_sites=N
  [cf_leak]   site jcc_pc=0xPC executions_tainted=N
  [dns_taint] SUMMARY and_pc=0xPC shr_pc=0xPC al_before=B al_after_and=B
              al_after_shr=B

Run:  python3 detect_libdft64.py
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
BASELINES = HERE.parent.parent / 'rq2-comparison' / 'external'

GCC = ['gcc', '-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']
CT_SRC = HERE.parent / 'crypto' / 'square_and_multiply' / 'test_constant_time.c'
CT_BIN = HERE / 'ct_test'
DNS_SRC = HERE / 'dns_bitfield.c'
DNS_BIN = HERE / 'dns_bitfield'

#: The secret is read from stdin: the 4-byte little-endian exponent, value 5.
CT_STDIN = (5).to_bytes(4, 'little')
#: QR(bit7)=1, OPCODE(bits6..3)=0001.  The same byte every other engine gets.
DNS_STDIN = bytes([0x88])

#: What microtaint reports on the same binaries.
MT_VULN_MIN = 1
MT_CT = 0

#: libdft's read() hook taints the syscall's return value, so the two length
#: checks in libc and in main come back tainted in BOTH variants.  They are
#: shared plumbing, not the crypto side channel, and the algorithm-level
#: differential is what the table reports.  Their PCs are not fixed across
#: builds, so they are identified by being tainted in the ct variant too.
PLUMBING_NOTE = (
    "libdft's read hook taints RAX (the read() byte count), so the length "
    'checks in __libc_read and in main count as tainted branches in both '
    'variants. They fire identically in vuln and ct, so the sites common to '
    'both are excluded and the differential is what is reported.')


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build() -> str:
    """Compile the guests and the Pin tools.  '' on success."""
    for src, dst in ((CT_SRC, CT_BIN), (DNS_SRC, DNS_BIN)):
        if dst.exists():
            continue
        if not src.exists():
            return f'{src} is missing'
        p = _sh([*GCC, '-o', str(dst), str(src)])
        if p.returncode != 0:
            return f'building {dst.name} failed: {p.stderr.strip()[-300:]}'
    if not list(HERE.glob('obj-intel64/*.so')):
        p = _sh(['make', 'tools'], cwd=HERE)
        if p.returncode != 0:
            return f'make tools failed: {p.stderr.strip()[-400:]}'
    return ''


def preflight() -> str:
    if shutil.which('make') is None:
        return 'make not found on PATH'
    if not glob.glob(str(BASELINES / 'pin-*' / 'pin')):
        return f'Pin not found under {BASELINES}; run rq2-comparison/setup_envs.sh'
    if not (BASELINES / 'libdft64' / 'src' / 'obj-intel64' / 'libdft.a').exists():
        return 'libdft64 is not built; run rq2-comparison/setup_envs.sh'
    return ''


def run_tool(tool: str, binary: Path, argv: list[str], stdin: bytes) -> str:
    pin = glob.glob(str(BASELINES / 'pin-*' / 'pin'))[0]
    so = HERE / 'obj-intel64' / f'{tool}.so'
    p = subprocess.run([pin, '-t', str(so), '--', str(binary), *argv],
                       input=stdin, capture_output=True, cwd=HERE, timeout=900)
    # Both tools log to stderr; the guest's own output goes to stdout.
    return p.stderr.decode('utf-8', 'replace')


def parse_cf_leak(text: str) -> dict:
    m = re.search(r'\[cf_leak\] cond_branches_executed=(\d+) '
                  r'tainted_cond_branches=(\d+) flag_writes=(\d+) '
                  r'tainted_branch_sites=(\d+)', text)
    if not m:
        raise SystemExit(f'cf_leak printed no summary line, so nothing was '
                         f'measured. Its output was:\n{text[-1500:]}')
    sites = {pc: int(n) for pc, n in
             re.findall(r'\[cf_leak\]\s+site jcc_pc=(0x[0-9a-f]+) '
                        r'executions_tainted=(\d+)', text)}
    return {
        'cond_branches_executed': int(m.group(1)),
        'tainted_cond_branches': int(m.group(2)),
        'flag_writes': int(m.group(3)),
        'tainted_branch_sites': int(m.group(4)),
        'sites': sites,
    }


def parse_dns(text: str) -> dict:
    m = re.search(r'SUMMARY and_pc=(0x[0-9a-f]+) shr_pc=(0x[0-9a-f]+) '
                  r'al_before=(-?\d+) al_after_and=(-?\d+) al_after_shr=(-?\d+)',
                  text)
    if not m:
        raise SystemExit(f'dns_taint printed no SUMMARY line, so nothing was '
                         f'measured. Its output was:\n{text[-1500:]}')
    return {
        'and_pc': m.group(1), 'shr_pc': m.group(2),
        'al_tainted_before_and': bool(int(m.group(3))),
        'al_tainted_after_and': bool(int(m.group(4))),
        'al_tainted_after_shr': bool(int(m.group(5))),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    blocker = preflight() or build()
    if blocker:
        print(f'[libdft64] blocker: {blocker}', file=sys.stderr)
        return 2

    vuln = parse_cf_leak(run_tool('cf_leak', CT_BIN, ['vuln'], CT_STDIN))
    ct = parse_cf_leak(run_tool('cf_leak', CT_BIN, ['ct'], CT_STDIN))
    dns = parse_dns(run_tool('dns_taint', DNS_BIN, [], DNS_STDIN))
    (RESULTS / '_libdft64_raw.json').write_text(
        json.dumps({'vuln': vuln, 'ct': ct, 'dns': dns}, indent=2) + '\n')

    # The algorithm-level differential: sites tainted in vuln and not in ct.
    shared = set(vuln['sites']) & set(ct['sites'])
    vuln_sites = {pc: n for pc, n in vuln['sites'].items() if pc not in shared}
    ct_sites = {pc: n for pc, n in ct['sites'].items() if pc not in shared}
    vuln_leaks = sum(vuln_sites.values())
    ct_leaks = sum(ct_sites.values())

    out_tainted = dns['al_tainted_after_shr']
    out = {
        'tool': 'libdft64 (Intel Pin DTA, byte-granular tag map)',
        'ct': {
            'workload': 'test_constant_time.c (square-and-multiply modexp)',
            'vuln_branch_leaks': vuln_leaks,
            'vuln_branch_leak_sites': sorted(vuln_sites),
            'ct_branch_leaks': ct_leaks,
            'raw_total_tainted_cond_branches': {
                'vuln': vuln['tainted_cond_branches'],
                'ct': ct['tainted_cond_branches'],
            },
            'shared_plumbing_sites': sorted(shared),
            'note_on_raw_totals': PLUMBING_NOTE,
            'matches_microtaint': vuln_leaks >= MT_VULN_MIN and ct_leaks == MT_CT,
            'ran': True,
        },
        'dns': {
            'workload': 'test_bitpacked_dns.c LDNS_OPCODE_WIRE: '
                        'AND AL,0x78 then SHR AL,3',
            'granularity': 'byte',
            'can_taint_single_bit': False,
            'input_flag_byte': hex(DNS_STDIN[0]),
            'al_tainted_before_and': dns['al_tainted_before_and'],
            'al_tainted_after_and': dns['al_tainted_after_and'],
            'al_tainted_after_shr': dns['al_tainted_after_shr'],
            # The whole byte carries one tag, so tainting QR alone is not
            # expressible and the masked-away bit still reaches the output.
            'output_tainted': out_tainted,
            'can_discriminate_qr_vs_opcode': False,
            'matches_microtaint': False,
            'ran': True,
        },
    }
    (RESULTS / 'libdft64.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"[libdft64] CT vuln={vuln_leaks} ct={ct_leaks} "
          f"(raw {vuln['tainted_cond_branches']}/{ct['tainted_cond_branches']}, "
          f"{len(shared)} shared plumbing site(s)) | "
          f'DNS output_tainted={out_tainted} '
          f"-> {RESULTS / 'libdft64.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
