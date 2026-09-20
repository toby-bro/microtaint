"""The two RQ7 application tables, from the JSON the experiments emit.

Both tables were transcribed into LaTeX by hand, which means a re-run could
change a number without the table moving.  Every value in them already exists
in a result file, so they can be generated instead.

    python gen_apps_tables.py --results-dir ../results/<stamp> --tex apps.tex
    python gen_apps_tables.py --results-dir ../results/<stamp> --md apps.md

Baseline rows come from other-engines/results/<tool>.json, written by the
detect_* scripts in that directory.  The rows for this engine come from the
run directory given, so the table describes the run it is generated from.
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(HERE, 'other-engines', 'results')

#: Row order, and the display name for each.  This engine goes last, as in the
#: paper.
TOOLS = [('angr', 'angr'), ('maat', 'Maat'), ('libdft64', 'libdft64'),
         ('taintgrind', 'TaintGrind'), ('triton', 'Triton')]

#: The square-and-multiply ladder consumes one exponent bit per iteration.
LADDER_STEPS = 32


def _granularity(tool, dns):
    """Bit, Byte or Register, from what the tool reported about itself.

    `can_taint_single_bit` is the authoritative flag: it is the property the
    table's first column separates on.  For the engines that cannot, the
    granularity string is prose, so we take whichever of byte and register it
    names first ("byte / register" is byte-granular, "register/byte-level" is
    register-granular).
    """
    if dns.get('can_taint_single_bit'):
        return 'Bit'
    s = str(dns.get('granularity') or '').lower()
    hits = sorted((s.index(w), label)
                  for w, label in (('byte', 'Byte'), ('register', 'Register'))
                  if w in s)
    if not hits:
        raise SystemExit(
            f'{tool}: granularity {s!r} names neither byte nor register, so the '
            f'table cannot say how coarse it is.',
        )
    return hits[0][1]


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError:
        return None


def collect(results_dir):
    """(ct_rows, dns_rows), each a list of tuples ready to format."""
    ct, dns = [], []
    for key, name in TOOLS:
        d = _load(os.path.join(BASELINE_DIR, f'{key}.json'))
        if d is None:
            print(f'[!] missing {key}.json, its rows are omitted', file=sys.stderr)
            continue
        gran = _granularity(key, d.get('dns', {}))

        # Attribution: with a single exponent bit as the secret, a bit-granular
        # engine flags the one step that consumes it.  An engine with no
        # sub-byte source taints the whole exponent and flags every step.
        localises = gran == 'Bit'
        ct.append((name, gran, '1' if localises else f'All {LADDER_STEPS}',
                   'Localises' if localises else 'Cannot localise'))

        separated = bool(d.get('dns', {}).get('can_discriminate_qr_vs_opcode'))
        dns.append((name, gran, 'Yes' if separated else 'No',
                    'Correct' if separated else 'False positive'))

    # This engine, from the run being reported on rather than a checked-in file.
    loc = _load(os.path.join(results_dir, 'rq7_ct_localise.json'))
    dnsr = _load(os.path.join(results_dir, 'rq7_dns.json'))
    if loc is None or dnsr is None:
        raise SystemExit(
            f'{results_dir}: rq7_ct_localise.json and rq7_dns.json are both '
            f'needed for our own rows.  Run RQ7 first.',
        )
    # Every bit fired at its own step and nowhere else: that is one flagged step
    # per secret bit, which is what the table claims.
    exact = loc.get('n_exact') == loc.get('n_bits') and loc.get('n_bits')
    ct.append(('microtaint', 'Bit', '1' if exact else loc.get('localisation', '?'),
               'Localises' if exact else 'Cannot localise'))
    separated = bool(dnsr.get('passed')) and dnsr.get('n_passed') == dnsr.get('n_runs')
    dns.append(('microtaint', 'Bit', 'Yes' if separated else 'No',
                'Correct' if separated else 'False positive'))
    return ct, dns


def _tex_table(caption, label, header, rows):
    out = [r'\begin{table}[t]', '    ' + r'\centering',
           f'    \\caption{{{caption}}}', f'    \\label{{{label}}}',
           '    ' + r'\footnotesize', '    ' + r'\begin{tabular}{llcc}',
           '        ' + r'\hline', '        ' + header + r' \\',
           '        ' + r'\hline']
    for row in rows:
        name = r'\ourname' if row[0] == 'microtaint' else row[0]
        if name == r'\ourname':
            out.append('        ' + r'\hline')
        out.append('        ' + ' & '.join([name, *row[1:]]) + r' \\')
    out += ['        ' + r'\hline', '    ' + r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def as_tex(ct, dns):
    head = '% Generated by gen_apps_tables.py.  Do not edit by hand.'
    return '\n\n'.join((
        head,
        _tex_table('Constant-time leak attribution', 'tab:apps-ct',
                   'Engine & Granularity & Steps flagged & Verdict', ct),
        _tex_table('DNS bit-field verdict per engine', 'tab:apps-dns',
                   r'Engine & Granularity & \texttt{QR}/\texttt{OPCODE} separated'
                   ' & Verdict', dns),
    )) + '\n'


def as_md(ct, dns):
    out = ['<!-- Generated by gen_apps_tables.py. Do not edit by hand. -->', '',
           '### Constant-time leak attribution', '',
           '| Engine | Granularity | Steps flagged | Verdict |',
           '| --- | --- | --- | --- |']
    out += [f'| {n} | {g} | {s} | {v} |' for n, g, s, v in ct]
    out += ['', '### DNS bit-field verdict per engine', '',
            '| Engine | Granularity | `QR`/`OPCODE` separated | Verdict |',
            '| --- | --- | --- | --- |']
    out += [f'| {n} | {g} | {s} | {v} |' for n, g, s, v in dns]
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', required=True,
                    help="a run directory, for this engine's own rows")
    ap.add_argument('--tex', metavar='PATH')
    ap.add_argument('--md', metavar='PATH')
    args = ap.parse_args()

    ct, dns = collect(args.results_dir)
    # A table missing a row is a table that quietly answers a different
    # question, so refuse rather than emit a short one.
    if len(ct) != len(TOOLS) + 1:
        raise SystemExit(
            f'only {len(ct)} of {len(TOOLS) + 1} engines have results, so these '
            f'tables would be incomplete.  Run the detect_* scripts in '
            f'other-engines/ first.',
        )
    if args.tex:
        with open(args.tex, 'w') as fh:
            fh.write(as_tex(ct, dns))
        print(f'[+] wrote {args.tex}')
    if args.md:
        with open(args.md, 'w') as fh:
            fh.write(as_md(ct, dns))
        print(f'[+] wrote {args.md}')
    if not (args.tex or args.md):
        print(as_md(ct, dns))
    return 0


if __name__ == '__main__':
    sys.exit(main())
