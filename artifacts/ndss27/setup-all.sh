#!/usr/bin/env bash
# Install everything the artifacts need, in one command.
#
#   ./setup-all.sh                 microtaint, the six compared engines, TaintInduce
#   ./setup-all.sh --no-baselines  microtaint only, enough for ./run-all.sh --no-baselines
#
# Every step is idempotent, which matters because most of the runtime here is
# network (about 5GB, dominated by PANDA's guest image).  If a download fails
# part way through, run this again: it picks up where it stopped rather than
# starting over.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

. "$HERE/find_uv.sh"
find_uv || exit 1
. "$HERE/check_deps.sh"
. "$HERE/ensure_git.sh"

BASELINES=1
for a in "$@"; do
  case "$a" in
    --no-baselines) BASELINES=0 ;;
    -h|--help)      sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

check_deps "$BASELINES" || exit 1

step() {  # step <label> <dir> <cmd...>
  local label=$1 dir=$2; shift 2
  echo
  echo "=== $label ==="
  if ( cd "$dir" && "$@" ); then
    echo "[+] $label: done"
  else
    echo "[!] $label: FAILED" >&2
    return 1
  fi
}

# An unpacked archive has no .git, and the version comes from git.
ensure_git "$REPO" || exit 1

# microtaint itself: builds the C and Cython extensions.
#
# CFLAGS=-march=native, because this build is for THIS machine and the paper
# measures performance on it.  The engine's own defaults no longer pass -march:
# they also build the wheels published to PyPI, and a wheel tuned for its build
# machine crashes on any CPU with fewer instructions (v0.7.2 shipped AVX-512
# and died with SIGILL on every CPU without it).  A local build has no such
# problem, and the reference runs in reference-runs/ were produced this way, so
# asking for it here is what keeps timings comparable to them.
step 'microtaint' "$REPO" env CFLAGS="-march=native" uv sync --locked --all-extras || exit 1

if [ "$BASELINES" -eq 0 ]; then
  echo
  echo '[+] Ready.  Next: ./run-all.sh --no-baselines'
  exit 0
fi

# The six compared engines, then TaintInduce.  Each is left to report its own
# progress; both are safe to re-run.
step 'the six compared engines' "$HERE/rq2-comparison"            ./setup_envs.sh       || exit 1
step 'TaintInduce'              "$HERE/rq1-synthesis_vs_inference" ./setup_taintinduce.sh || exit 1

echo
echo '[+] Ready.  Next: ./run-all.sh'
