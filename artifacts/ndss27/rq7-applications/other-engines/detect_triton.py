#!/usr/bin/env python3
"""detect_triton.py -- assemble the Triton per-tool baseline (results/triton.json).

Runs the CT and DNS drivers and folds their raw traces into the shared schema,
the way detect_maat.py does for Maat.  Before this existed, `triton.json` was
the only part of Triton's row that no script produced: the two drivers wrote
`_ct_raw.json` and `_dns_raw.json` and somebody read them and typed the
verdict file, so a re-run could not check it.

Run:
  ../../rq2-comparison/.venv_triton/bin/python detect_triton.py
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

import detect_triton_ct
import detect_triton_dns

#: What microtaint reports on the same two binaries, which is what "matches"
#: is measured against rather than asserted.
MT_VULN_MIN = 1
MT_CT = 0

#: What Triton's taint API is.  That is a property of the tool, not something
#: a run measures; the measurable part, whether any bit-level entry point
#: exists at all, is probed and lands in `can_taint_single_bit` and
#: `subbit_api_probe`.  Do not shorten this to "byte": Triton taints NAMED
#: REGISTERS, the smallest of which happens to be eight bits wide, and the
#: paper reports it as register-granular on that basis.
GRANULARITY = ("register/byte-level. Triton's taint engine exposes "
               'taintRegister / taintMemory / setTaintRegister / '
               'setTaintMemory only; the smallest addressable taint unit is '
               'one byte (an 8-bit sub-register such as AL). No bit-level '
               'taint API exists (no taintRegisterBit / taintBit / bit index).')

METHOD = ('Triton ELF load (dependency-free ELF64 loader) + native x86-64 '
          'instruction emulation; target routine driven directly (RIP := '
          'function entry, SysV args in registers) to bypass static-glibc '
          'startup. Taint engine enabled by default.')


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    detect_triton_ct.main()
    detect_triton_dns.main()

    ct_raw = json.loads((RESULTS / '_ct_raw.json').read_text())
    dns_raw = json.loads((RESULTS / '_dns_raw.json').read_text())

    vuln, ct = ct_raw['vuln'], ct_raw['ct']
    # A driver that errored measured nothing, and its zero is not a clean
    # trace.  Say so rather than folding it into a verdict.
    errs = [v['error'] for v in (vuln, ct) if v.get('error')]
    if errs:
        raise SystemExit(f'the CT driver reported an error, so its counts are '
                         f'not a measurement: {errs[0]}')

    vuln_leaks = vuln['leak_count_static']
    ct_leaks = ct['leak_count_static']
    qr, op = dns_raw['qr_intent'], dns_raw['opcode_intent']

    ct_out: dict[str, object] = {
            'workload': 'test_constant_time.c (square-and-multiply modexp)',
            'vuln_branch_leaks': vuln_leaks,
            'vuln_branch_leak_sites': vuln['leak_pcs'],
            'vuln_branch_leak_execs': vuln['n_tainted_branch_execs'],
            'vuln_cond_branches_total': vuln['cond_branches_executed'],
            'ct_branch_leaks': ct_leaks,
            'ct_branch_leak_execs': ct['n_tainted_branch_execs'],
            'ct_cond_branches_total': ct['cond_branches_executed'],
            'matches_microtaint': vuln_leaks >= MT_VULN_MIN and ct_leaks == MT_CT,
            'ran': True,
    }
    dns_out: dict[str, object] = {
            'workload': 'test_bitpacked_dns.c LDNS_OPCODE_WIRE: '
                        'AND AL,0x78 (24 78) then SHR AL,3 (C0 E8 03)',
            'granularity': 'bit' if dns_raw['can_taint_single_bit'] else GRANULARITY,
            'can_taint_single_bit': dns_raw['can_taint_single_bit'],
            'subbit_api_probe': dns_raw['api'],
            'qr_reaches_output': bool(qr['output_al_tainted']),
            'opcode_reaches_output': bool(op['output_al_tainted']),
            'input_flag_byte': qr['al_in'],
            'byte_taint_output_al': dns_raw['byte_taint_output_al'],
            'byte_taint_output_al_tainted': bool(qr['output_al_tainted']),
            # microtaint separates the two fields, so Triton matches it on this
            # workload exactly when it can too.
            'matches_microtaint': dns_raw['can_discriminate_qr_vs_opcode'],
            'control_untainted_output_al_tainted':
                bool(dns_raw['control_untainted']['output_al_tainted']),
            'can_discriminate_qr_vs_opcode':
                dns_raw['can_discriminate_qr_vs_opcode'],
            'ran': True,
    }
    out = {'tool': 'triton', 'method': METHOD, 'ct': ct_out, 'dns': dns_out}
    # The control run exists to catch a pipeline that taints everything.  If it
    # comes back tainted, every other verdict here is meaningless.
    if dns_out['control_untainted_output_al_tainted']:
        raise SystemExit('the untainted control run reported a tainted output, '
                         'so this pipeline taints regardless of its input and '
                         'none of its verdicts mean anything.')

    (RESULTS / 'triton.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"[triton] CT vuln={vuln_leaks} site(s), "
          f"{vuln['n_tainted_branch_execs']} exec(s); ct={ct_leaks} | "
          f"DNS single-bit={dns_out['can_taint_single_bit']}, "
          f"separates QR/OPCODE={dns_out['can_discriminate_qr_vs_opcode']} "
          f"-> {RESULTS / 'triton.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
