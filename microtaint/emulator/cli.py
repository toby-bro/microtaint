import argparse
import logging
import platform
import sys

from qiling import Qiling

from microtaint.emulator.wrapper import MicrotaintWrapper

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_default_rootfs() -> str:
    """Returns the host rootfs if on Linux, otherwise aborts."""
    if platform.system() == 'Linux':
        return '/'

    logging.error('[!] Non-Linux host detected.')
    logging.error(
        '[!] To emulate Linux ELF binaries on Windows/macOS, you must provide a cross-architecture rootfs via the --rootfs flag.'
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Microtaint Execution Wrapper')
    parser.add_argument('binary', help='Target ELF binary to analyze')
    parser.add_argument('binary_args', nargs=argparse.REMAINDER, help='Arguments for the binary')
    parser.add_argument('--check-bof', action='store_true', help='Detect Buffer Overflows')
    parser.add_argument('--check-uaf', action='store_true', help='Detect Use-After-Free')
    parser.add_argument('--check-sc', action='store_true', help='Detect Side Channels via Implicit Taint')
    parser.add_argument('--rootfs', default=None, help='Custom Qiling rootfs path')

    args = parser.parse_args()

    # Use user-provided rootfs, or fallback to the host's /
    rootfs = args.rootfs if args.rootfs else get_default_rootfs()

    argv = [args.binary] + args.binary_args
    ql = Qiling(argv, rootfs, verbose=0)

    # Attach our engine
    wrapper = MicrotaintWrapper(ql, check_bof=args.check_bof, check_uaf=args.check_uaf, check_sc=args.check_sc)

    logging.info('[*] Starting Tainted Execution...')
    try:
        ql.run()
    except Exception as e:
        logging.error(f'[*] Execution halted: {e}')


if __name__ == '__main__':
    main()
