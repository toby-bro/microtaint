#!/usr/bin/env python3
"""Flag runs that were NOT measured under steady multi-worker load.

The sweep measures with N workers in parallel. A run that happened while the
other workers were idle saw different cache/DRAM conditions than the rest of the
series, so it should be re-measured (delete its log and re-run sweep.sh) if you
want every point taken under identical conditions.

    find_solo.py <log-dir> <work-dir>

Measured bias is small (a control test put paired-vs-solo within +-3.4% with no
consistent sign), so this is a diagnostic, not a hard failure.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path


def main(out: Path, work: Path) -> int:
    log = work / 'bench.log'
    durations: dict[str, int] = {}
    if log.exists():
        # the drive script prints "logged: .../<stamp>-<sha>-timing.json" per run
        for line in log.read_text(errors='replace').splitlines():
            m = re.search(r'-([0-9a-f]{7,})-timing\.json', line)
            if m:
                durations.setdefault(m.group(1), 0)

    runs = []
    for f in sorted(out.glob('*-timing.json')):
        d = json.loads(f.read_text())
        end = datetime.fromisoformat(d['ts'])
        # duration is not recorded in the log itself; approximate the measured
        # window from the run's own aggregate timing (sum of per-instruction ns).
        secs = sum(r.get('ns', 0) for r in d.get('instr', [])) * 1900 / 1e9
        runs.append({'sha': d['git'], 'start': end - timedelta(seconds=secs), 'end': end})

    solo = []
    for r in runs:
        span = (r['end'] - r['start']).total_seconds()
        overlap = sum(
            max(timedelta(0), min(r['end'], o['end']) - max(r['start'], o['start'])).total_seconds()
            for o in runs if o is not r
        )
        if span and overlap / span < 0.5:
            solo.append((r['sha'], overlap / span))

    if solo:
        print(f'{len(solo)} of {len(runs)} run(s) measured with little or no parallel load:')
        for sha, frac in solo:
            print(f'  {sha}  overlapped {frac * 100:.0f}% of its measurement')
        print('  (delete those logs and re-run sweep.sh to redo them alongside others)')
    else:
        print(f'all {len(runs)} runs measured under comparable parallel load')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
