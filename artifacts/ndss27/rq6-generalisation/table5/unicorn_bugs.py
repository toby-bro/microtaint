# ruff: noqa: W505, E501, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Outputs where Unicorn, not MicroTaint, is wrong.

The campaign's ground truth is Unicorn.  Where Unicorn computes a flag
incorrectly, its value moves with inputs the real instruction does not depend on
(or fails to move where it should), and the engine's correct answer reads as an
under-taint.  Those outputs must be dropped from the comparison, exactly as
`exclude_flags` drops architecturally undefined ones.

Each entry cites the evidence.  Adding one requires showing Unicorn disagrees
with the ISA manual on CONCRETE values, not merely that a taint comparison
failed: `verify.py` in this directory reproduces each claim.

Keyed by ISA; `match` is a substring test against the corpus asm text.
"""
from __future__ import annotations

KNOWN_UNICORN_BUGS = [
    {
        'isa': 'x86_64',
        'match': 'bzhi',
        'flags': ['CF'],
        'quarantine': True,
        'why': ('Unicorn implements BZHI with the wrong threshold, and it is not only '
                'the flag.  Intel: N := SRC2[7:0]; if N < 64 then DEST[63:N] := 0; '
                'CF := (N > 63).  Measured, Unicorn flips CF at N=63 (should be 64) '
                'AND keeps masking bit 63 for every N >= 64, returning 0x7fff... where '
                'the manual says DEST is copied unchanged as 0xffff....  Both the flag '
                'and the result are wrong, so the whole entry is quarantined rather '
                'than just CF.'),
        'evidence': 'ISA_UNDEFINED_FLAGS.md "Ground-truth (Unicorn) BUGS" (CF only); '
                    'campaign_ae/verify_unicorn_bugs.py shows the CF flip at N=63 and '
                    'the wrong result for N >= 64',
    },
]


def quarantined(isa_key: str, asm: str) -> str | None:
    """Instructions whose ground truth is wrong outright, flags and results alike."""
    for b in KNOWN_UNICORN_BUGS:
        if b.get('quarantine') and b['isa'] == isa_key and b['match'] in asm:
            return b['why']
    return None


def suppressed_flags(isa_key: str, asm: str) -> set[str]:
    """Flags whose ground truth is untrustworthy for this instruction."""
    out: set[str] = set()
    for b in KNOWN_UNICORN_BUGS:
        if b['isa'] == isa_key and b['match'] in asm:
            out.update(b['flags'])
    return out


def reason(isa_key: str, asm: str, flag: str) -> str | None:
    for b in KNOWN_UNICORN_BUGS:
        if b['isa'] == isa_key and b['match'] in asm and flag in b['flags']:
            return b['why']
    return None
