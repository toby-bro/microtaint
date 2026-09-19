#!/bin/bash
set -e

og_dir=$(pwd)

echo "[*] Setting up virtual environments with uv..."

# Install uv if not present
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/find_uv.sh"
find_uv || exit 1

# 1. Master Orchestrator
#
# --allow-existing on every venv below, for the same reason the Pin download and
# the clones are guarded: this script is mostly network, so a partial run is
# normal and the remedy is to run it again.  Without it `uv venv` refuses with
# "a virtual environment already exists" and `set -e` aborts the whole script on
# its FIRST step, so a re-run could never get past what it had already done.
echo "[*] Building Master Env..."
uv venv --allow-existing .venv_master
uv pip install --python .venv_master keystone-engine unicorn

# 2. Triton
echo "[*] Building Triton Env..."
uv venv --allow-existing .venv_triton
uv pip install --python .venv_triton triton-library

# 3. Angr
echo "[*] Building Angr Env..."
uv venv --allow-existing .venv_angr
uv pip install --python .venv_angr angr

# 4. Maat
echo "[*] Building Maat Env..."
uv venv --allow-existing .venv_maat --python=3.11
uv pip install --python .venv_maat pymaat

# 5. Microtaint -- THIS repository, not PyPI.
#
# This used to pin 'microtaint==0.6.15' from PyPI, so the comparison measured a
# published release rather than the tree it ships with: an artifact evaluator
# running these scripts would have scored an engine that is not the one under
# review, and nothing in the output said so.
echo "[*] Building Microtaint Env..."
repo_root=$(cd "$og_dir/../../.." && pwd)
uv venv --allow-existing .venv_microtaint
uv pip install --python .venv_microtaint "$repo_root"
.venv_microtaint/bin/python -c "
import importlib.metadata as m
print('[+] microtaint', m.version('microtaint'), 'from', '$repo_root')"

echo '[+] Making libdft64...'
mkdir -p external
# Idempotent: `set -e` turns a second run into an abort otherwise.
[ -d external/libdft64 ] && echo "[=] external/libdft64 present" || git clone https://github.com/AngoraFuzzer/libdft64 external/libdft64
cd external/
#PIN_VERSION='external-3.31-98869-gfa6f126a8'
PIN_VERSION='3.20-98437-gf02b61307'
PIN_TGZ="pin-${PIN_VERSION}-gcc-linux.tar.gz"
PIN_DIR="pin-${PIN_VERSION}-gcc-linux"
PIN_URL="https://software.intel.com/sites/landingpage/pintool/downloads/${PIN_TGZ}"

# Idempotent, like the clones above.  This script has a lot of network in it
# (five package environments, a 35MB Pin tarball, a ~1GB container image and
# two container builds), so a transient failure part way through is normal, and
# the fix for one is to run it again.  Unconditional `wget` turned that into
# another download whose only effect was to leave a `.tar.gz.1` file
# beside the real one.
[ -f "$PIN_TGZ" ]     && echo "[=] $PIN_TGZ present"     || wget "$PIN_URL"
[ -f "$PIN_TGZ.sig" ] && echo "[=] $PIN_TGZ.sig present" || wget "$PIN_URL.sig"
openssl cms -verify -binary -in "$PIN_TGZ.sig" -inform DER -content "$PIN_TGZ" -out /dev/null -noverify

# Unpack and patch together: the patches must be applied exactly once per
# extraction.  Guarding them separately would either re-patch an already
# patched tree or leave a freshly extracted one unpatched, depending on which
# guard fired -- and the second sed matches its own output, so re-running it
# appends the flag again every time.
if [ -d "$PIN_DIR" ]; then
    echo "[=] $PIN_DIR present, already unpacked and patched"
else
    tar -xzf "$PIN_TGZ"
    echo '[+] Patching Pin for libdft64...'
    sed -i 's/range\.m_base/range\._base/' "$PIN_DIR/extras/components/include/util/range.hpp"

    # -Wno-error=non-c-typedef-for-linkage cannot be applied unconditionally.
    # libdft64 builds with -Werror, so a compiler that HAS that warning fails on
    # Pin 3.20's headers, which predate it.  But a compiler that does NOT have
    # it rejects the suppression itself, as a hard error:
    #
    #   cc1plus: error: '-Wno-error=non-c-typedef-for-linkage':
    #            no option '-Wnon-c-typedef-for-linkage'
    #
    # Measured: g++ 16.2 accepts the flag, g++ 12.2 (Debian 12) rejects it, so
    # hardcoding it builds on the machine it was written on and breaks
    # elsewhere.  Ask the compiler instead.
    if echo 'int main(){}' | g++ -x c++ -Wno-error=non-c-typedef-for-linkage -fsyntax-only - 2>/dev/null; then
        sed -i 's/-Wall -Werror -Wno-unknown-pragmas/-Wall -Werror -Wno-unknown-pragmas -Wno-error=non-c-typedef-for-linkage/' "$PIN_DIR/source/tools/Config/makefile.unix.config"
        echo '[+] this g++ has -Wnon-c-typedef-for-linkage, suppressed it'
    else
        echo '[=] this g++ has no -Wnon-c-typedef-for-linkage, nothing to suppress'
    fi
fi
export PIN_ROOT=$(pwd)/$PIN_DIR
cd libdft64/
git checkout 20804d5bae5d8aed31a71761b1a1149e35a0da95
# or simply 
docker build -t libdft64:latest . #just in case
make
cd ../../

echo '[+] Pulling PANDA...'
# PINNED BY DIGEST, not by :latest.  A floating tag means a reviewer running
# this next year compares against a different PANDA than the paper measured,
# and nothing in the output would say so.  This digest is the manifest list, so
# it is immutable and still resolves per architecture.
#
# pandare/panda:latest as of 2026-06-09.  To move it deliberately:
#   docker buildx imagetools inspect pandare/panda:latest   # read the digest
PANDA_DIGEST='sha256:e4bb0346e9f9cd9f0b4a2f75b353cbf041005687b4af47d5705cfee054aec71b'
docker pull "pandare/panda@${PANDA_DIGEST}"
# Local alias so the workers can name one stable thing instead of repeating the
# digest; `:pinned` is deliberately not `:latest`, so it cannot be silently
# replaced by a later `docker pull`.
docker tag "pandare/panda@${PANDA_DIGEST}" pandare/panda:pinned
echo "[*] Building PANDA Env..."
uv venv --allow-existing .venv_panda
uv pip install --python .venv_panda pandare

echo '[+] Setting up Taintgrind...'
# Idempotent: `set -e` turns a second run into an abort otherwise.
[ -d external/taintgrind ] && echo "[=] external/taintgrind present" || git clone https://github.com/wmkhoo/taintgrind external/taintgrind
cd external/taintgrind/
git checkout 4a59adff7e67ad6793bb362746bc05352bb4e795
docker build -t taintgrind:latest .
cd ../../

echo "[+] All Python environments ready!"
