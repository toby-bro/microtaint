#!/usr/bin/env python3
"""PANDA smoke test: boot generic x86_64, revert to root, run a serial cmd,
copy the harness in and execute it with the trigger payload on stdin."""
import base64, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFT = HERE.parent
sys.path.insert(0, str(NFT))
import common

from pandare import Panda

panda = Panda(generic="x86_64")

# base64 the payload so we can materialise it in-guest without file transfer races
PAYLOAD_B64 = base64.b64encode(common.PAYLOAD).decode()

@panda.queue_blocking
def run():
    panda.revert_sync("root")
    print("[smoke] uname:", panda.run_serial_cmd("uname -a"))
    # copy the harness binary into the guest
    share = HERE / "guest_share"
    share.mkdir(exist_ok=True)
    import shutil
    shutil.copy(NFT / "harness", share / "harness")
    panda.copy_to_guest(str(share))
    print("[smoke] ls copied:", panda.run_serial_cmd("ls -la /mnt/guest_share* 2>/dev/null; ls -la ~ 2>/dev/null | head"))
    # locate the copied harness
    where = panda.run_serial_cmd("find / -name harness -type f 2>/dev/null | head")
    print("[smoke] harness at:", where)
    hpath = where.strip().splitlines()[0] if where.strip() else "/mnt/guest_share/harness"
    panda.run_serial_cmd(f"cp {hpath} /root/harness && chmod +x /root/harness")
    panda.run_serial_cmd(f"echo {PAYLOAD_B64} | base64 -d > /root/payload")
    out = panda.run_serial_cmd("cd /root && ./harness < payload", timeout=60)
    print("[smoke] HARNESS OUTPUT:\n", out)
    panda.end_analysis()

panda.run()
print("[smoke] done")
