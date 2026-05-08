#!/bin/bash
set -e

og_dir=$(pwd)

echo "[*] Setting up virtual environments with uv..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo '[!] uv not found, aborting... check https://docs.astral.sh/uv/ for installation instructions.'
    exit 1
fi

# 1. Master Orchestrator
echo "[*] Building Master Env..."
uv venv .venv_master
uv pip install --python .venv_master keystone-engine

# 2. Triton
echo "[*] Building Triton Env..."
uv venv .venv_triton
uv pip install --python .venv_triton triton-library

# 3. Angr
echo "[*] Building Angr Env..."
uv venv .venv_angr
uv pip install --python .venv_angr angr

# 4. Maat
echo "[*] Building Maat Env..."
uv venv .venv_maat --python=3.11
uv pip install --python .venv_maat pymaat

# 5. Microtaint (Assuming local path, adjust if necessary)
echo "[*] Building Microtaint Env..."
uv venv .venv_microtaint
uv pip install --python .venv_microtaint microtaint

echo '[+] Making libdft64...'
mkdir -p external
git clone https://github.com/AngoraFuzzer/libdft64 external/libdft64
cd external/
#PIN_VERSION='external-3.31-98869-gfa6f126a8'
PIN_VERSION='3.20-98437-gf02b61307'
wget "https://software.intel.com/sites/landingpage/pintool/downloads/pin-${PIN_VERSION}-gcc-linux.tar.gz"
wget "https://software.intel.com/sites/landingpage/pintool/downloads/pin-${PIN_VERSION}-gcc-linux.tar.gz.sig"
openssl cms -verify -binary -in pin-${PIN_VERSION}-gcc-linux.tar.gz.sig -inform DER -content pin-${PIN_VERSION}-gcc-linux.tar.gz -out /dev/null -noverify
tar -xzf pin-${PIN_VERSION}-gcc-linux.tar.gz
echo '[+] Patching Pin for libdft64...'
sed -i 's/range\.m_base/range\._base/' pin-${PIN_VERSION}-gcc-linux/extras/components/include/util/range.hpp
sed -i 's/-Wall -Werror -Wno-unknown-pragmas/-Wall -Werror -Wno-unknown-pragmas -Wno-error=non-c-typedef-for-linkage/' pin-${PIN_VERSION}-gcc-linux/source/tools/Config/makefile.unix.config
export PIN_ROOT=$(pwd)/pin-${PIN_VERSION}-gcc-linux
cd libdft64/
git checkout 20804d5bae5d8aed31a71761b1a1149e35a0da95
# or simply 
docker build -t libdft64:latest . #just in case
make
cd ../../

echo '[+] Pulling PANDA...'
docker pull pandare/panda:latest
echo "[*] Building PANDA Env..."
uv venv .venv_panda
uv pip install --python .venv_panda pandare

echo '[+] Setting up Taintgrind...'
git clone https://github.com/wmkhoo/taintgrind external/taintgrind
cd external/taintgrind/
git checkout 4a59adff7e67ad6793bb362746bc05352bb4e795
docker build -t taintgrind:latest .
cd ../../

echo "[+] All Python environments ready!"
