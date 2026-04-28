from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TextIO


class FindingKind(StrEnum):
    BOF = 'buffer_overflow'
    UAF = 'use_after_free'
    SIDE_CHANNEL = 'side_channel'
    TAINT_SOURCE = 'taint_source'
    AIW = 'arbitrary_indexed_write'  # STORE to a tainted pointer


@dataclass
class Finding:
    kind: FindingKind
    address: int
    description: str
    instruction: str = ''
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            'kind': str(self.kind),
            'address': hex(self.address),
            'description': self.description,
        }
        if self.instruction:
            d['instruction'] = self.instruction
        if self.extra:
            d.update(self.extra)
        return d


# ANSI colour codes — disabled automatically when stdout is not a tty
def _supports_colour(stream: TextIO) -> bool:
    return hasattr(stream, 'isatty') and stream.isatty()


_RESET = '\033[0m'
_BOLD = '\033[1m'
_RED = '\033[31m'
_YELLOW = '\033[33m'
_CYAN = '\033[36m'
_GREEN = '\033[32m'
_DIM = '\033[2m'

_KIND_COLOUR = {
    FindingKind.BOF: _RED,
    FindingKind.UAF: _RED,
    FindingKind.AIW: _RED,
    FindingKind.SIDE_CHANNEL: _YELLOW,
    FindingKind.TAINT_SOURCE: _CYAN,
}

_KIND_LABEL = {
    FindingKind.BOF: '[BOF]',
    FindingKind.UAF: '[UAF]',
    FindingKind.AIW: '[AIW]',
    FindingKind.SIDE_CHANNEL: '[SC] ',
    FindingKind.TAINT_SOURCE: '[SRC]',
}


class Reporter:
    """
    Collects findings during emulation and renders them as either
    human-readable text or machine-readable JSON at the end of a run.
    """

    def __init__(self, *, json_mode: bool = False, stream: TextIO | None = None) -> None:
        self.json_mode = json_mode
        self.stream: TextIO = stream or sys.stderr
        self._colour = not json_mode and _supports_colour(self.stream)
        self.findings: list[Finding] = []

    # ------------------------------------------------------------------
    # Public API called by MicrotaintWrapper
    # ------------------------------------------------------------------

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if not self.json_mode:
            self._print_finding(finding)

    def bof(self, address: int, instruction: str = '') -> None:
        self.add(
            Finding(
                kind=FindingKind.BOF,
                address=address,
                description=f'Buffer overflow: tainted data hijacked control flow at {hex(address)}',
                instruction=instruction,
            ),
        )

    def uaf(self, address: int, size: int = 0) -> None:
        self.add(
            Finding(
                kind=FindingKind.UAF,
                address=address,
                description=f'Use-after-free: access to poisoned (freed) memory at {hex(address)}',
                extra={'access_size': size} if size else {},
            ),
        )

    def side_channel(self, address: int, instruction: str = '', taint_mask: int = 0) -> None:
        self.add(
            Finding(
                kind=FindingKind.SIDE_CHANNEL,
                address=address,
                description=f'Potential crypto side-channel: branch at {hex(address)} depends on tainted data',
                instruction=instruction,
                extra={'taint_mask': hex(taint_mask)} if taint_mask else {},
            ),
        )

    def taint_source(self, address: int, size: int, fd: int = 0) -> None:
        source = 'stdin' if fd == 0 else f'fd={fd}'
        self.add(
            Finding(
                kind=FindingKind.TAINT_SOURCE,
                address=address,
                description=f'Taint introduced: {size} bytes from {source} at {hex(address)}',
                extra={'size': size, 'source': source},
            ),
        )

    def aiw(self, address: int, pointer_taint: int, instruction: str = '') -> None:
        """Arbitrary Indexed Write: STORE instruction with a tainted destination pointer."""
        self.add(
            Finding(
                kind=FindingKind.AIW,
                address=address,
                description=f'Arbitrary write: tainted pointer used as store destination at {hex(address)}',
                instruction=instruction,
                extra={'pointer_taint': hex(pointer_taint)},
            ),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _c(self, code: str, text: str) -> str:
        return f'{code}{text}{_RESET}' if self._colour else text

    def _print_finding(self, f: Finding) -> None:
        colour = _KIND_COLOUR.get(f.kind, '')
        label = _KIND_LABEL.get(f.kind, '[???]')
        addr = hex(f.address)
        line = f'{self._c(colour + _BOLD, label)} {self._c(colour, f.description)}'
        if f.instruction:
            line += f'  {self._c(_DIM, f.instruction)}'
        print(line, file=self.stream)
        _ = addr  # addr already embedded in description; available for future structured lines

    def finalize(self) -> int:
        """
        Print the summary (or emit JSON). Returns an exit code:
          0 — no security findings (taint_source alone does not count)
          1 — at least one BOF, UAF, or side-channel finding
        """
        bof_count = sum(1 for f in self.findings if f.kind == FindingKind.BOF)
        uaf_count = sum(1 for f in self.findings if f.kind == FindingKind.UAF)
        sc_count = sum(1 for f in self.findings if f.kind == FindingKind.SIDE_CHANNEL)
        aiw_count = sum(1 for f in self.findings if f.kind == FindingKind.AIW)
        security_total = bof_count + uaf_count + sc_count + aiw_count

        if self.json_mode:
            output = {
                'findings': [f.to_dict() for f in self.findings],
                'summary': {
                    'total': security_total,
                    'bof': bof_count,
                    'uaf': uaf_count,
                    'side_channel': sc_count,
                    'aiw': aiw_count,
                },
            }
            print(json.dumps(output, indent=2), file=self.stream)
        else:
            self._print_summary()

        return 1 if security_total > 0 else 0

    def _print_summary(self) -> None:
        print('', file=self.stream)
        if not self.findings:
            print(self._c(_GREEN + _BOLD, '[✓] No findings detected.'), file=self.stream)
            return

        bof_count = sum(1 for f in self.findings if f.kind == FindingKind.BOF)
        uaf_count = sum(1 for f in self.findings if f.kind == FindingKind.UAF)
        sc_count = sum(1 for f in self.findings if f.kind == FindingKind.SIDE_CHANNEL)
        aiw_count = sum(1 for f in self.findings if f.kind == FindingKind.AIW)

        sep = self._c(_DIM, '─' * 50)
        print(sep, file=self.stream)
        print(self._c(_BOLD, 'Microtaint analysis summary'), file=self.stream)
        print(sep, file=self.stream)

        if bof_count:
            print(self._c(_RED, f'  Buffer overflows  : {bof_count}'), file=self.stream)
        if uaf_count:
            print(self._c(_RED, f'  Use-after-free    : {uaf_count}'), file=self.stream)
        if aiw_count:
            print(self._c(_RED, f'  Arbitrary writes  : {aiw_count}'), file=self.stream)
        if sc_count:
            print(self._c(_YELLOW, f'  Side channels     : {sc_count}'), file=self.stream)

        total = bof_count + uaf_count + sc_count + aiw_count
        print(sep, file=self.stream)
        print(self._c(_RED + _BOLD, f'  Total findings    : {total}'), file=self.stream)
        print(sep, file=self.stream)
