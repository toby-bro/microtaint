#!/usr/bin/env python3
"""Container-side smoke: instantiate Panda(generic=x86_64), revert, run a cmd."""
import sys
from pandare import Panda
from pandare.utils import find_build_dir
print("[ctest] builddir:", find_build_dir("x86_64"), flush=True)
panda = Panda(generic="x86_64")
print("[ctest] Panda constructed", flush=True)

@panda.queue_blocking
def run():
    panda.revert_sync("root")
    print("[ctest] uname:", panda.run_serial_cmd("uname -r"), flush=True)
    panda.end_analysis()

panda.run()
print("[ctest] done", flush=True)
