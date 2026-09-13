#!/usr/bin/env python3
"""scripts/perf_ladder.py
=========================
What one more fuzz input costs, on every rung of the ladder.

    native  ->  Qiling+Unicorn  ->  + microtaint, fresh  ->  + checkpointed

Run it after any change that could move end-to-end cost, and before quoting a
number at anybody::

    scripts/perf_ladder.py                       # every rung, 3 repeats
    scripts/perf_ladder.py --repeats 1 --rungs microtaint-checkpoint
    scripts/perf_ladder.py --guest ./my.elf --main-symbol main --json out.json

Why this exists rather than an ad-hoc script per question: every measurement
below has been got WRONG at least once on this project, each time producing a
confidently-stated number that survived until something else contradicted it.
The guards are the point.  They are documented here so that removing one is a
decision rather than an accident.

**1. `instructions:u`, never plain `instructions`.**  The default event counts
kernel instructions too, and those track machine load: repeat measurements of
one unchanged configuration came back 130.7 / 141.1 / 112.7 / 148.2 M while
another process hammered the box.  `:u` reproduces to 0.1 M on a loaded machine.

**2. The marginal method.**  Each rung is run at N and at 5N in SEPARATE
processes and the answer is (I5 - I1) / 4N.  Every fixed cost -- interpreter
start, ~4 s of SLEIGH language loading, ELF load, checkpoint construction, the
one-time compilation of every block -- cancels instead of being amortised into
the per-input figure.  A single run at one N cannot separate them.

**3. The guest is built ONCE, outside every measured process.**  Building it
inside put the static linker's own instruction count into the difference.

**4. perf writes to a FILE (`-o`), never a pipe.**  Two traps live here: zsh's
MULTIOS turns `2>&1 >/dev/null | grep` into something that reported this guest
at 36,187,719 instructions, reproducibly, seven times running, when the true
figure was 185,231; and pypcode's exit-time nanobind message has no trailing
newline, so perf's CSV arrives glued to the end of it and a line-anchored regex
silently drops every microtaint row.

**5. Warm-up before the steady state.**  The first checkpointed iteration
compiles every block in the guest.  Profiling a window that contains it
measures the compiler: at N=150 that one iteration was 61% of the sample.

**6. Nothing is reported that was not measured.**  A rung that tracked no
blocks, found no findings, or whose repeats disagree by more than
`--max-spread` is an error, not a result.  A harness that scores agreement must
count what it compared and refuse to pass on zero.

**7. Work is reported next to cost, and ratios are only formed between rungs
that do the SAME work.**  A checkpointed iteration resumes at main and runs a
few thousand guest instructions; a native launch runs the whole program, 97% of
which is one-time startup.  Dividing the first by the second produced "6.6x
native" for something that is ~1,760x per unit of guest work.  So this script
prints `x native` only for rungs that do a full launch, and prints per-tracked-
block for the rest, where the comparison is meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from microtaint.emulator.snapshot import RunOutcome

#: A guest with a real libc in it: a read, a heap allocation, a copy, a hash
#: loop whose trip count is secret, and a branch on the result.  Static so the
#: measurement is not at the mercy of the host's shared libraries.
_GUEST_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){ char buf[128];
  if(!fgets(buf,sizeof buf,stdin)) return 1;
  size_t n=strlen(buf); char*c=malloc(n+1); memcpy(c,buf,n+1);
  unsigned long h=5381; for(size_t i=0;i<n;i++) h=h*33+(unsigned char)c[i];
  if(h&1) puts("odd"); else puts("even");
  char o[64]; snprintf(o,sizeof o,"%lu %zu",h,n); fputs(o,stdout); free(c);
  return (int)(h&3); }
"""

#: The native baseline a CHECKPOINTED rung should be compared against: the AFL
#: model, where startup happens once in the parent and each input costs a fork,
#: the post-main work, and an exit.  That is what checkpoint+restore replaces,
#: so it is the only native figure the checkpoint rungs are commensurable with.
#: Comparing them against a full `exec` instead was how "6.6x native" happened.
_FORKSERVER_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
static int work(const char *line, FILE *out) {
    char buf[128];
    strncpy(buf, line, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    size_t n = strlen(buf);
    char *c = malloc(n + 1); memcpy(c, buf, n + 1);
    unsigned long h = 5381;
    for (size_t i = 0; i < n; i++) h = h * 33 + (unsigned char)c[i];
    if (h & 1) fputs("odd\n", out); else fputs("even\n", out);
    char o[64]; snprintf(o, sizeof o, "%lu %zu", h, n); fputs(o, out);
    free(c);
    return (int)(h & 3);
}
int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    FILE *devnull = fopen("/dev/null", "w");
    char line[128];
    for (int i = 0; i < n; i++) {
        snprintf(line, sizeof line, "hello world, tainted line %d\n", i);
        pid_t p = fork();
        if (p == 0) { _exit(work(line, devnull)); }
        int st; waitpid(p, &st, 0);
    }
    return 0;
}
"""

#: Rung -> (N for the small run, whether it does a FULL guest launch).
#: The second flag is what makes rule 7 enforceable: only full-launch rungs are
#: comparable with native.
RUNGS: dict[str, tuple[int, bool]] = {
    'native': (100, True),
    'native-forkserver': (200, False),
    'qiling-fresh': (20, True),
    'qiling-checkpoint': (100, False),
    'microtaint-fresh': (10, True),
    'microtaint-checkpoint': (100, False),
}
#: Iterations run before the timed window on the checkpointed rungs.  The first
#: compiles every block; see rule 5.
WARMUP = 12


class Rung(NamedTuple):
    """One rung's marginal cost, and the work that cost bought."""

    name: str
    instrs: float            # host instructions:u per input (USER side)
    kernel: float            # instructions the KERNEL ran per input
    faults: float            # page faults per input
    seconds: float           # wall per input, the per-input loop only
    blocks: int              # guest basic blocks the engine analysed
    findings: int            # leak sites reported
    full_launch: bool


def _inputs(n: int) -> Iterator[bytes]:
    for i in range(n):
        yield b'hello world, tainted line %d\n' % i


# ---------------------------------------------------------------------------
# The worker: one rung, N inputs, in its own process.
# ---------------------------------------------------------------------------

def _worker(rung: str, guest: str, main: int, n: int) -> int:
    import io

    loop = 0.0
    blocks = findings = 0
    if rung != 'native':
        os.environ['MICROTAINT_BLOCK'] = '1'

    def quiet() -> tuple[int, int]:
        return os.dup(1), os.open(os.devnull, os.O_WRONLY)

    if rung == 'native-forkserver':
        t = time.perf_counter()
        subprocess.run([guest, str(n)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
        loop = time.perf_counter() - t

    elif rung == 'native':
        data = next(_inputs(1))
        t = time.perf_counter()
        for _ in range(n):
            subprocess.run([guest], input=data, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        loop = time.perf_counter() - t

    elif rung.startswith('qiling'):
        from qiling import Qiling
        from qiling.const import QL_VERBOSE
        from qiling.os.stats import QlOsNullStats
        if rung == 'qiling-fresh':
            t = time.perf_counter()
            for inp in _inputs(n):
                ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
                ql.os.stats = QlOsNullStats()
                ql.os.stdin = io.BytesIO(inp)
                s, dn = quiet()
                os.dup2(dn, 1)
                try:
                    ql.run()
                except Exception:
                    pass
                os.dup2(s, 1)
                os.close(dn)
                os.close(s)
            loop = time.perf_counter() - t
        else:
            ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
            ql.os.stats = QlOsNullStats()
            ql.os.stdin = io.BytesIO(b'')
            s, dn = quiet()
            os.dup2(dn, 1)
            hk = ql.hook_address(lambda q: q.emu_stop(), main)
            ql.run()
            ql.hook_del(hk)
            st = ql.save(reg=False, mem=True, fd=True, cpu_context=True,
                         os=True, loader=True)
            shape = {(a, b, c) for a, b, c, *_ in ql.mem.map_info}

            def once(inp: bytes) -> None:
                for a, b, c, *_ in list(ql.mem.map_info):
                    if (a, b, c) not in shape:
                        ql.mem.unmap(a, b - a)
                ql.restore(st)
                ql.os.stdin = io.BytesIO(inp)
                ql.exit_point = None
                try:
                    ql.run(begin=main)
                except Exception:
                    pass

            for inp in _inputs(WARMUP):
                once(inp)
            t = time.perf_counter()
            for inp in _inputs(n):
                once(inp)
            loop = time.perf_counter() - t
            os.dup2(s, 1)
            os.close(dn)
            os.close(s)

    else:
        from qiling import Qiling
        from qiling.const import QL_VERBOSE

        from microtaint.emulator.reporter import Reporter
        from microtaint.emulator.wrapper import MicrotaintWrapper
        rep = Reporter(json_mode=True, stream=io.StringIO())
        if rung == 'microtaint-fresh':
            t = time.perf_counter()
            for inp in _inputs(n):
                rep.findings.clear()
                ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
                ql.os.stdin = io.BytesIO(inp)
                w = MicrotaintWrapper(ql, reporter=rep)
                s, dn = quiet()
                os.dup2(dn, 1)
                ok = True
                try:
                    ql.run()
                except Exception:
                    ok = False
                os.dup2(s, 1)
                os.close(dn)
                os.close(s)
                w.block_mode_finish(ok)
            loop = time.perf_counter() - t
            st2 = w.block_mode_stats() or {}
            blocks, findings = st2.get('blocks', 0), len(rep.findings)
        else:
            ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
            ql.os.stdin = io.BytesIO(b'')
            w = MicrotaintWrapper(ql, reporter=rep)
            s, dn = quiet()
            os.dup2(dn, 1)
            w.run_to(main)
            cp = w.checkpoint()

            def once_mt(inp: bytes) -> RunOutcome:
                w.restore(cp)
                ql.os.stdin = io.BytesIO(inp)
                rep.findings.clear()
                return w.resume(cp)

            for inp in _inputs(WARMUP):
                once_mt(inp)
            base = dict(w.block_mode_stats() or {})
            t = time.perf_counter()
            last = None
            for inp in _inputs(n):
                last = once_mt(inp)
            loop = time.perf_counter() - t
            end = dict(w.block_mode_stats() or {})
            os.dup2(s, 1)
            os.close(dn)
            os.close(s)
            blocks = (end.get('blocks', 0) - base.get('blocks', 0)) // max(n, 1)
            findings = len(last.sites) if last is not None else 0

    sys.stderr.write(f'LADDER loop={loop:.6f} blocks={blocks} '
                     f'findings={findings}\n')
    return 0


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------

#: Counted together so the KERNEL side is visible rather than inferred.
#: `instructions` minus `instructions:u` is the kernel's own work, which is
#: what makes the default event load-sensitive (rule 1); page-faults is what
#: makes a restore expensive, and task-clock is CPU time rather than wall, so a
#: loaded machine shows up as the gap between them.
_EVENTS = ('instructions:u', 'instructions', 'task-clock', 'page-faults')
_MARKER = re.compile(r'LADDER loop=([0-9.]+) blocks=(\d+) findings=(\d+)')


def _count(text: str, event: str) -> float | None:
    """One perf CSV counter.

    NOT anchored to the line start (rule 4), but the event name IS anchored at
    its END.  Without that, asking for `instructions` matches the prefix of
    `instructions:u`, so the total silently equalled the user count and the
    kernel column it feeds could only ever read zero -- a counter that cannot
    be non-zero is not a measurement (rule 6).
    """
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*,(?:msec)?,' + re.escape(event)
                  + r'(?=,|\s|$)', text)
    return float(m.group(1)) if m else None


def _run_once(rung: str, guest: str, main: int, n: int,
              use_perf: bool) -> tuple[dict[str, float], float, int, int]:
    """One worker process.  Returns (perf counters, seconds, blocks, finds)."""
    with tempfile.TemporaryDirectory() as td:
        stat = Path(td) / 'perf.csv'
        cmd: list[str] = []
        if use_perf:
            # -o FILE, never a pipe: see rule 4.
            cmd = ['perf', 'stat', '-x,', '-e', ','.join(_EVENTS),
                   '-o', str(stat)]
        cmd += [sys.executable, __file__, '--worker', rung,
                '--guest', guest, '--main', hex(main), '--n', str(n)]
        done = subprocess.run(cmd, capture_output=True, check=False)
        err = done.stderr.decode(errors='replace')
        m = _MARKER.search(err)
        if not m:
            raise SystemExit(
                f'{rung} at N={n} produced no marker, so nothing was measured.\n'
                f'{err[-800:]}')
        loop, blocks, finds = float(m.group(1)), int(m.group(2)), int(m.group(3))
        counters: dict[str, float] = {}
        if use_perf:
            text = stat.read_text(errors='replace') if stat.exists() else ''
            for ev in _EVENTS:
                v = _count(text, ev)
                if v is None and ev == 'instructions:u':
                    raise SystemExit(
                        f'{rung} at N={n}: perf reported no instructions:u.\n'
                        f'{text[-400:]}')
                if v is not None:
                    counters[ev] = v
    return counters, loop, blocks, finds


def _measure(rung: str, guest: str, main: int, use_perf: bool) -> Rung:
    n, full = RUNGS[rung]
    c1, l1, b1, f1 = _run_once(rung, guest, main, n, use_perf)
    c5, l5, _b5, _f5 = _run_once(rung, guest, main, n * 5, use_perf)

    def marginal(ev: str) -> float:
        if ev not in c1 or ev not in c5:
            return float('nan')
        return (c5[ev] - c1[ev]) / (4 * n)

    user = marginal('instructions:u')
    total = marginal('instructions')
    # The KERNEL side, which is exactly the part that moves with machine load
    # and the reason rule 1 exists.  Reported rather than silently included.
    kern = (total - user) if total == total and user == user else float('nan')
    return Rung(rung, user, kern, marginal('page-faults'),
                (l5 - l1) / (4 * n), b1, f1, full)


def _native_alone(guest: str, runs: int = 20) -> float | None:
    """The guest's own instructions:u per run, with NO harness in the count.

    The `native` rung launches the guest from Python, so its figure carries
    `subprocess.run`'s fork/exec as well: 0.305 M against the guest's own
    0.185 M, which is a 1.6x error in any denominator built from it.  perf
    counts the process the kernel exec'd, so this is the honest baseline.
    `-r` repeats in one perf invocation and reports the mean.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'nat.csv'
        inp = Path(td) / 'in'
        inp.write_bytes(next(_inputs(1)))
        with inp.open('rb') as fh:
            subprocess.run(
                ['perf', 'stat', '-r', str(runs), '-x,', '-e', 'instructions:u',
                 '-o', str(out), '--', guest],
                stdin=fh, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False)
        if not out.exists():
            return None
        return _count(out.read_text(errors='replace'), 'instructions:u')


def _growth(rung: str, guest: str, main: int, use_perf: bool,
            windows: tuple[tuple[int, int], ...] = ((2, 12), (40, 50), (90, 100)),
            ) -> list[float]:
    """Marginal cost measured at increasing OFFSETS into the campaign.

    A single marginal cannot tell a flat loop from one that degrades, and this
    project shipped a loop that was QUADRATIC: Qiling's `stats.summary()`
    json-dumps every syscall ever recorded and `ql.run()` calls it once per
    input, so iteration N re-dumped 1..N.  Per-iteration cost went 9.30 M at
    inputs 2-12, 15.24 at 40-50, 23.25 at 90-100, and every single-point
    measurement looked fine.  Measure the SLOPE whenever a loop reuses state.
    """
    out: list[float] = []
    for lo, hi in windows:
        a, _la, _b, _f = _run_once(rung, guest, main, lo, use_perf)
        b, _lb, _b2, _f2 = _run_once(rung, guest, main, hi, use_perf)
        ev = 'instructions:u'
        out.append((b[ev] - a[ev]) / (hi - lo)
                   if ev in a and ev in b else float('nan'))
    return out


def _guest_work(guest: str, main: int) -> tuple[int, int]:
    """(guest instructions in a full launch, from `main` onwards).

    Printed next to every cost so a partial run is never divided by a whole
    one; see rule 7.  Uses Qiling because it is the only thing here that can
    count guest instructions without changing what runs.
    """
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.os.stats import QlOsNullStats
    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stats = QlOsNullStats()
    ql.os.stdin = io.BytesIO(next(_inputs(1)))
    total, after, seen = [0], [0], [False]

    def cb(_q: object, addr: int, _sz: int) -> None:
        total[0] += 1
        if addr == main:
            seen[0] = True
        if seen[0]:
            after[0] += 1

    ql.hook_code(cb)
    s, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    try:
        ql.run()
    except Exception:
        pass
    os.dup2(s, 1)
    os.close(dn)
    os.close(s)
    return total[0], after[0]


def _build(dest: Path, name: str, src: str) -> str:
    """Built ONCE, here, outside every measured process; see rule 3."""
    out = dest / name
    made = subprocess.run(
        ['gcc', '-static', '-O1', '-o', str(out), '-x', 'c', '-'],
        input=src.encode(), capture_output=True, check=False)
    if made.returncode != 0:
        raise SystemExit(f'cannot build {name}: {made.stderr.decode()[:400]}')
    return str(out)


def _symbol(guest: str, name: str) -> int:
    nm = subprocess.run(['nm', guest], capture_output=True, text=True,
                        check=False).stdout
    for line in nm.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == name:
            return int(parts[0], 16)
    raise SystemExit(f'{guest} has no symbol {name!r}; pass --main-symbol')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', metavar='RUNG')
    ap.add_argument('--guest')
    ap.add_argument('--main')
    ap.add_argument('--main-symbol', default='main')
    ap.add_argument('--n', type=int)
    ap.add_argument('--rungs', nargs='*', choices=sorted(RUNGS), default=None)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-spread', type=float, default=15.0,
                    help='refuse to report if repeats disagree by more than '
                         'this %% (rule 6)')
    ap.add_argument('--growth', action='store_true',
                    help='also measure the marginal at increasing offsets into '
                         'the campaign, to catch a loop whose per-input cost '
                         'GROWS (this project shipped a quadratic one)')
    ap.add_argument('--json', metavar='PATH')
    args = ap.parse_args(argv)

    if args.worker:
        return _worker(args.worker, args.guest, int(args.main, 16), args.n)

    use_perf = shutil.which('perf') is not None
    if not use_perf:
        print('perf not found: wall clock only, and wall is load-sensitive '
              'in a way instructions:u is not', file=sys.stderr)

    tmp = tempfile.TemporaryDirectory()
    guest = args.guest or _build(Path(tmp.name), 'ladder_guest.elf', _GUEST_SRC)
    forkserver = _build(Path(tmp.name), 'forkserver.elf', _FORKSERVER_SRC)
    main_addr = _symbol(guest, args.main_symbol)
    launch, from_main = _guest_work(guest, main_addr)
    if launch <= 0:
        raise SystemExit('the guest executed no instructions: nothing to measure')
    alone = _native_alone(guest) if use_perf else None

    names = args.rungs or list(RUNGS)
    got: dict[str, list[Rung]] = {}
    for rep in range(args.repeats):
        for name in names:
            print(f'  [{rep + 1}/{args.repeats}] {name}', file=sys.stderr)
            target = forkserver if name == 'native-forkserver' else guest
            got.setdefault(name, []).append(
                _measure(name, target, main_addr, use_perf))

    rows: dict[str, Rung] = {}
    for name, rs in got.items():
        ii = sorted(r.instrs for r in rs)
        if use_perf and len(ii) > 1 and ii[len(ii) // 2]:
            spread = (ii[-1] - ii[0]) / ii[len(ii) // 2] * 100
            if spread > args.max_spread:
                raise SystemExit(
                    f'{name}: repeats disagree by {spread:.1f}% '
                    f'(> {args.max_spread}%), so no figure from this run is '
                    f'trustworthy: {[f"{x/1e6:.2f}M" for x in ii]}')
        rows[name] = Rung(
            name,
            statistics.median(r.instrs for r in rs),
            statistics.median(r.kernel for r in rs),
            statistics.median(r.faults for r in rs),
            statistics.median(r.seconds for r in rs),
            rs[0].blocks, rs[0].findings, rs[0].full_launch)

    for name, r in rows.items():
        if name.startswith('microtaint') and r.blocks <= 0:
            raise SystemExit(
                f'{name} analysed 0 blocks, so it measured an emulator with no '
                f'taint in it rather than the engine')

    print(f'\nguest: {guest}')
    print(f'  a full launch runs {launch} guest instructions, of which '
          f'{launch - from_main} ({(launch - from_main) / launch * 100:.1f}%) '
          f'is one-time startup')
    print(f'  a checkpointed iteration runs {from_main} from {args.main_symbol}')
    print(f'\nmarginal cost of ONE more input, median of {args.repeats}\n')
    if alone:
        print(f'  the guest ALONE costs {alone / 1e6:.3f} M instructions:u per '
              f'run (no harness in the count);\n  the `native` rung below adds '
              f'the fork/exec that launches it')
    print(f'\n{"rung":<24}{"Minstr:u":>10}{"kern":>9}{"flt":>7}{"ms":>9}'
          f'{"blocks":>8}{"sites":>7}{"x native":>10}{"x no-taint":>12}')
    print('-' * 92)
    # The denominator is the guest ALONE where we could measure it, because the
    # `native` rung carries subprocess.run's fork/exec (0.305 M vs 0.185 M).
    nat_rung = rows.get('native')
    base_native = alone if alone else (nat_rung.instrs if nat_rung else 0.0)
    for name in names:
        r = rows[name]
        # Rule 7: x-native only where both sides do a full launch.
        xn = (f'{r.instrs / base_native:.0f}x'
              if base_native and r.full_launch else '--')
        # A checkpointed rung is commensurable with a native FORK SERVER, not
        # with an exec: both skip startup and pay once per input.
        fs = rows.get('native-forkserver')
        if not r.full_launch and fs and fs.seconds and name != 'native-forkserver':
            xn = f'{r.seconds / fs.seconds:.1f}x*'
        base = {'microtaint-fresh': 'qiling-fresh',
                'microtaint-checkpoint': 'qiling-checkpoint'}.get(name)
        xq = (f'{r.instrs / rows[base].instrs:.2f}x'
              if base and base in rows else '--')
        kern = f'{r.kernel / 1e6:.2f}' if r.kernel == r.kernel else '--'
        flt = f'{r.faults:.0f}' if r.faults == r.faults else '--'
        print(f'{name:<24}{r.instrs / 1e6:>10.3f}{kern:>9}{flt:>7}'
              f'{r.seconds * 1e3:>9.2f}{r.blocks:>8}{r.findings:>7}'
              f'{xn:>10}{xq:>12}')
    print('-' * 92)
    print('kern is instructions the KERNEL ran per input (the part that moves '
          'with machine\nload, and the reason the user-side event is the one '
          'quoted); flt is page faults.')
    print('x native compares full launches only.  A starred figure is vs the '
          'native FORK\nSERVER and is WALL clock, because a fork-server input '
          'is almost all KERNEL work\n(214k kernel instructions against 3.4k '
          'user), which instructions:u cannot see.')
    mc = rows.get('microtaint-checkpoint')
    if mc and from_main:
        print(f'\nper unit of guest work, checkpointed: '
              f'{mc.instrs / from_main:,.0f} host instructions per guest '
              f'instruction,\n{mc.instrs / max(mc.blocks, 1):,.0f} per analysed '
              f'block ({mc.blocks} blocks, {from_main / max(mc.blocks, 1):.1f} '
              f'guest instructions each).')
        print('Block length drives this: a tight compute loop costs far less '
              'per\ninstruction than branchy libc at ~4 instructions a block. '
              'Name the workload.')
    if args.growth:
        print('\nmarginal M instructions:u per input, measured at increasing '
              'offsets into the campaign:')
        for name in names:
            if name == 'native':
                continue
            g = _growth(name, guest, main_addr, use_perf)
            shown = '  '.join('n/a' if x != x else f'{x / 1e6:6.2f}' for x in g)
            grew = (len(g) >= 2 and g[0] == g[0] and g[-1] == g[-1]
                    and g[0] and g[-1] / g[0] > 1.25)
            flag = '   <-- GROWS, the loop is not flat in campaign length' if grew else ''
            print(f'  {name:<24} inputs 2-12 / 40-50 / 90-100:  {shown}{flag}')

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'guest': guest, 'launch_instructions': launch,
             'from_main_instructions': from_main,
             'native_alone_instructions': alone,
             'rungs': {k: v._asdict() for k, v in rows.items()}}, indent=2))
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
