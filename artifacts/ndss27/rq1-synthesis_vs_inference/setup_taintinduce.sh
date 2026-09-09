#!/bin/bash
# RQ1 compares our offline rule synthesis against TaintInduce's observation-based
# inference.  Getting a TaintInduce to compare against is the whole difficulty of
# this experiment, so this script does exactly that and nothing else.
#
# The original artifact (github.com/melynx/taintinduce) does not run any more:
# it targets Python 3.6, and its serialization layer imports squirrelflowdb,
# which has not existed on PyPI for years.  We therefore maintain a fork that we
# spent several months repairing, and it is that fork we evaluate.
#
#   ./setup_taintinduce.sh          clone, pin, install, smoke-test
#   ./setup_taintinduce.sh --force  wipe external/taintinduce and start over
#
# Takes about two minutes, almost all of it downloading wheels.
set -e

REPO_URL='https://github.com/toby-bro/taintinduce'

# We pin a commit, not the branch tip.  Everything after this commit on master is
# the work that grew into microtaint (a CellIFT-style classifier, an
# instrumentation pass, an m-replica set), and the default entry point there
# stops running the DNF inference at all.  Comparing that against microtaint
# would be comparing microtaint against an early microtaint.  64a703a is the last
# state of the tree that is still TaintInduce: repaired, but the same algorithm.
TAINTINDUCE_COMMIT='64a703a2ce539d725bc4dbf3d90394fb8c9443eb'

here=$(cd "$(dirname "$0")" && pwd)
dest="$here/external/taintinduce"

if ! command -v uv &> /dev/null; then
    echo '[!] uv not found, aborting... check https://docs.astral.sh/uv/ for installation instructions.'
    exit 1
fi

# The DNF minimiser is a prebuilt espresso binary committed into the repository,
# and it is an x86-64 ELF.  On any other host the inference dies at its first
# minimisation call, so say so now rather than twenty minutes into a run.
case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "[!] $(uname -m) host: TaintInduce ships espresso as an x86-64 binary and will not run here." ;;
esac

if [ "${1:-}" = '--force' ]; then
    echo '[*] --force given, removing the previous checkout...'
    rm -rf "$dest"
fi

if [ -d "$dest/.git" ]; then
    echo "[*] $dest already exists, reusing it (pass --force to start over)."
else
    echo "[*] Cloning our TaintInduce fork into external/taintinduce..."
    mkdir -p "$here/external"
    git clone --quiet "$REPO_URL" "$dest"
fi

cd "$dest"
echo "[*] Pinning to $TAINTINDUCE_COMMIT ..."
git checkout --quiet "$TAINTINDUCE_COMMIT"

# uv reads requires-python and the lockfile, fetches an interpreter if the host
# has none that fits, and builds capstone/keystone/unicorn.  There is nothing
# else to compile: espresso is committed prebuilt (see above).
echo '[*] Installing dependencies (uv builds capstone, keystone and unicorn here)...'
uv sync --locked

# Smoke test.  JN ("Just Nibbles") is the fork's 4-bit toy ISA: two 4-bit
# registers and a 4-bit flag register, so the whole state space is 2^12 and the
# observation engine enumerates it instead of sampling.  That makes it both the
# fastest thing to run and, as RQ1 shows, the only width at which the inference
# gets arithmetic right.  About fifteen seconds.
echo '[*] Smoke test: inferring a rule for JN ADD R1, R2 ...'
uv run python -m taintinduce.taintinduce 0 JN --output-dir smoke_test > /dev/null 2>&1
if [ -f smoke_test/0_JN_rule.json ]; then
    echo '[+] TaintInduce runs, and wrote a rule.'
else
    echo '[!] TaintInduce ran but produced no rule; the experiment will not work.'
    exit 1
fi

echo '[+] Ready.  Next: ./run_rq1.sh --quick'
