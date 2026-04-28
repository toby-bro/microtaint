"""
microtaint — bit-precise taint analysis CLI

Usage
-----
  # Detect everything, feed stdin from the terminal
  uv run microtaint --check-all -- ./binary arg1 arg2

  # Read taint source from a file instead of stdin
  uv run microtaint --check-bof --input payload.bin -- ./binary

  # Pipe directly
  python -c "print('A'*64, end='')" | uv run microtaint --check-bof -- ./binary

  # Machine-readable JSON output (findings go to stdout, progress to stderr)
  uv run microtaint --check-all --json -- ./binary 2>/dev/null

  # Custom rootfs for cross-architecture analysis
  uv run microtaint --check-all --rootfs ./sysroot_arm -- ./arm_binary

  # Silence the Qiling + Unicorn noise, keep only findings
  uv run microtaint --check-all --quiet -- ./binary
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.heap import HeapTracker
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='microtaint',
        description='Bit-precise dynamic taint analysis — detects BOF, UAF, and side-channels.',
        epilog='Separate microtaint flags from the target binary with --',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    checks = parser.add_argument_group('detection modes')
    checks.add_argument('--check-bof', action='store_true', help='Detect buffer overflows (tainted RIP)')
    checks.add_argument('--check-uaf', action='store_true', help='Detect use-after-free (poisoned memory access)')
    checks.add_argument(
        '--check-sc',
        action='store_true',
        help='Detect side-channels via implicit taint (tainted branch conditions)',
    )
    checks.add_argument(
        '--check-aiw',
        action='store_true',
        help='Detect arbitrary indexed writes (STORE to a tainted pointer)',
    )
    checks.add_argument('--check-all', action='store_true', help='Enable all detection modes')

    inp = parser.add_argument_group('input')
    inp.add_argument(
        '--input',
        '-i',
        metavar='FILE',
        default=None,
        help="Feed FILE as stdin to the target binary (tainted). Defaults to the process's own stdin.",
    )

    out = parser.add_argument_group('output')
    out.add_argument(
        '--json',
        action='store_true',
        help='Emit findings as a JSON document to stdout. Progress messages go to stderr.',
    )
    out.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress Qiling/Unicorn informational noise. Findings are always shown.',
    )

    emu = parser.add_argument_group('emulator')
    emu.add_argument(
        '--rootfs',
        default=None,
        metavar='PATH',
        help='Qiling rootfs. Defaults to "/" on Linux; required on macOS/Windows.',
    )
    emu.add_argument(
        '--heap-hooks',
        action='store_true',
        default=True,
        help='Hook malloc/free for heap UAF detection (default: on). Use --no-heap-hooks to disable.',
    )
    emu.add_argument('--no-heap-hooks', dest='heap_hooks', action='store_false', help='Disable malloc/free hooking.')

    parser.add_argument('binary', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('binary_args', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    return parser


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split sys.argv on '--', returning (our_flags, [binary, *binary_args])."""
    try:
        sep = argv.index('--')
        return argv[:sep], argv[sep + 1 :]
    except ValueError:
        return argv, []


# ---------------------------------------------------------------------------
# Rootfs resolution
# ---------------------------------------------------------------------------


def _resolve_rootfs(rootfs: str | None, quiet: bool) -> str:  # noqa: ARG001
    if rootfs:
        return rootfs
    if platform.system() == 'Linux':
        return '/'
    msg = '[!] Non-Linux host detected. Provide a Linux sysroot via --rootfs PATH to emulate ELF binaries.'
    print(msg, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stdin wiring
# ---------------------------------------------------------------------------


class InteractiveStream:
    """
    A stdin stream that relays each read() call to the user's terminal live.

    When the emulated binary calls read(0, buf, n), Qiling calls our
    _sys_read_hook which calls f.read(count) on this object.
    We then print '[INPUT] >>> ' to stderr and read one response from the
    real terminal — exactly mirroring real interactive behaviour.

    This is the correct architecture for interactive programs: instead of
    collecting all input upfront, we block at each read() call and let the
    user respond to the program's actual prompts.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def write(self, data: bytes) -> None:
        """Pre-load data (used for --input FILE and pipe modes)."""
        self._buf.extend(data)

    def read(self, count: int) -> bytes:
        """
        Called by the sys_read hook for each read(0, ...) the binary makes.
        If the buffer has data (--input / pipe mode), drain it first.
        Otherwise prompt the user interactively.
        """
        if self._buf:
            chunk = bytes(self._buf[:count])
            del self._buf[:count]
            return chunk

        # Interactive: prompt on stderr so it appears alongside program output
        try:

            print('[INPUT] >>> ', end='', flush=True, file=sys.stderr)
            line = sys.stdin.readline()
            if not line:  # EOF (Ctrl-D)
                return b''
            return line.encode() if isinstance(line, str) else line
        except (EOFError, KeyboardInterrupt):
            return b''


def _make_stdin_stream(input_file: str | None, quiet: bool) -> tuple[InteractiveStream, bytes]:
    """
    Return an (InteractiveStream, pre_loaded_data) pair.

    --input FILE   → load file into stream buffer; no prompting at runtime.
    piped stdin    → read all data now, load into buffer.
    tty (default)  → return an empty InteractiveStream; each read() call by
                     the binary will prompt the user live via [INPUT] >>> .
    """
    stream = InteractiveStream()

    if input_file is not None:
        with open(input_file, 'rb') as fh:
            data = fh.read()
        stream.write(data)
        if not quiet:
            print(f'[*] Input file: {input_file!r} ({len(data)} bytes — tainted)', file=sys.stderr)
        return stream, data

    if not sys.stdin.isatty():
        data = sys.stdin.buffer.read()
        if data:
            stream.write(data)
        if not quiet and data:
            print(f'[*] Piped {len(data)} bytes from stdin (tainted)', file=sys.stderr)
        return stream, data

    # Interactive tty — stream starts empty; prompts appear per read() call
    if not quiet:
        print(
            '[*] Interactive mode — respond to each [INPUT] >>> prompt as the program runs.\n'
            '[*] Press Ctrl-D on an empty line to send EOF to the program.\n'
            '[*] Use --input FILE for binary/non-printable payloads.',
            file=sys.stderr,
        )
    return stream, b''


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(quiet: bool, _json_mode: bool) -> None:
    """
    Route microtaint's own logger to stderr at ERROR level so findings
    always appear. Suppress everything else when --quiet is set.
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING if quiet else logging.INFO)

    # Microtaint wrapper logs findings at ERROR — keep those regardless.
    mt_logger = logging.getLogger('microtaint')
    mt_logger.setLevel(logging.ERROR)

    # Strip any existing handlers to avoid double-printing in pytest.
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    our_argv, target_argv = _split_argv(sys.argv[1:])

    parser = _build_parser()
    args = parser.parse_args(our_argv)

    # --check-all expands to all three individual flags
    if args.check_all:
        args.check_bof = True
        args.check_uaf = True
        args.check_sc = True
        args.check_aiw = True

    # Must have at least one check enabled
    if not (args.check_bof or args.check_uaf or args.check_sc or args.check_aiw):
        parser.error(
            'Specify at least one detection mode: --check-bof, --check-uaf, --check-sc, --check-aiw, or --check-all',
        )

    # Must have a binary
    if not target_argv:
        # Accept legacy positional form: microtaint FLAGS binary [binary_args]
        # (i.e. the user forgot the --)
        if args.binary:
            target_argv = [args.binary, *list(args.binary_args)]
        else:
            parser.error('No target binary specified. Use: microtaint [FLAGS] -- /path/to/binary [args]')

    binary = target_argv[0]
    binary_args = target_argv[1:]

    if not os.path.isfile(binary):
        print(f'[!] Binary not found: {binary!r}', file=sys.stderr)
        sys.exit(2)

    _configure_logging(quiet=args.quiet, _json_mode=args.json)

    rootfs = _resolve_rootfs(args.rootfs, quiet=args.quiet)

    reporter = Reporter(
        json_mode=args.json,
        stream=sys.stdout if args.json else sys.stderr,
    )

    if not args.quiet and not args.json:
        modes = []
        if args.check_bof:
            modes.append('BOF')
        if args.check_uaf:
            modes.append('UAF')
        if args.check_sc:
            modes.append('side-channel')
        if getattr(args, 'check_aiw', False):
            modes.append('arbitrary-write')
        print(f'[*] microtaint — checking: {", ".join(modes)}', file=sys.stderr)
        print(f'[*] Target: {binary} {" ".join(binary_args)}', file=sys.stderr)

    # Prepare stdin data before creating Qiling so we can wire it up immediately.
    stdin_stream, _stdin_data = _make_stdin_stream(args.input, quiet=args.quiet)

    verbosity = QL_VERBOSE.OFF
    ql = Qiling([binary, *binary_args], rootfs, verbose=verbosity)
    ql.os.stdin = stdin_stream

    wrapper = MicrotaintWrapper(
        ql,
        check_bof=args.check_bof,
        check_uaf=args.check_uaf,
        check_sc=args.check_sc,
        check_aiw=getattr(args, 'check_aiw', False),
        reporter=reporter,
    )

    # Heap-level UAF tracking (malloc/free symbol hooks)
    if args.check_uaf and args.heap_hooks:
        heap_tracker = HeapTracker(ql, wrapper.shadow_mem)
        heap_tracker.install()

    if not args.quiet and not args.json:
        print('[*] Starting tainted execution…', file=sys.stderr)

    try:
        ql.run()
    except KeyboardInterrupt:
        print('\n[!] Interrupted by user.', file=sys.stderr)
    except Exception as exc:
        if not args.quiet:
            print(f'[!] Execution halted: {exc}', file=sys.stderr)

    exit_code = reporter.finalize()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
