"""A load through a tainted address must not taint more bits than it loads.

When the address of a LOAD is tainted the engine cannot know which bytes are
read, so it avalanches the loaded value.  That is sound, but the avalanche must
still respect the WIDTH of the load: a byte load zero-extended into a 64-bit
register can only ever set bits 0-7, whatever the memory holds, because bits
8-63 are architecturally zero.

Measured on base64's alphabet lookup (`movzx r32, byte ptr [r8+idx]`, the
instruction that accounts for that workload's entire data-path
over-approximation):

    mov   al,  byte ptr [r8+rax]   ->  0x00000000000000ff   correct
    movzx eax, byte ptr [r8+rax]   ->  0xffffffffffffffff   56 bits too many

The plain `mov` already bounds it; only the extending form loses the width.

Sign extension is NOT the same case and must keep its wide taint: `movsx` copies
the sign bit into every upper bit, so those bits genuinely depend on the loaded
byte.  The last test pins that, so a fix cannot narrow the bound by making
sign-extension unsound.
"""
from __future__ import annotations

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

NAMES = ['RAX', 'RSI', 'R8', 'RDX', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('RSP', 64)])


def _rax_taint(code_hex: str) -> int:
    """Taint of RAX after the instruction, with the load INDEX tainted."""
    rule = generate_static_rule(Architecture.AMD64, bytes.fromhex(code_hex), FMT)
    vals = dict.fromkeys(NAMES, 0)
    vals.update({'R8': 0x404040, 'RAX': 5, 'RSP': 0x204000})
    tnt = dict.fromkeys(NAMES, 0)
    tnt['RAX'] = 0x3F                      # a 6-bit table index, as base64 uses
    ctx = EvalContext(input_taint=tnt, input_values=vals,
                      simulator=CellSimulator(Architecture.AMD64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return int(rule.evaluate(ctx).get('RAX', 0) or 0)


def test_byte_load_without_extension_is_bounded() -> None:
    """`mov al, [r8+rax]` -- the control: this form is already correct."""
    assert _rax_taint('418a0400') & ~0xFF == 0


def test_zero_extended_byte_load_is_bounded() -> None:
    """`movzx eax, byte ptr [r8+rax]` -- bits 8-63 are zero, so untainted."""
    extra = _rax_taint('410fb60400') & ~0xFF
    assert extra == 0, (
        f'tainted-address byte load tainted {bin(extra).count("1")} bits above '
        f'bit 7 (mask {extra:#018x}); a zero-extended byte load can only set '
        'bits 0-7 whatever memory holds'
    )


def test_zero_extended_word_load_is_bounded() -> None:
    """`movzx eax, word ptr [r8+rax]` -- bits 16-63 are zero."""
    extra = _rax_taint('410fb70400') & ~0xFFFF
    assert extra == 0, f'word load tainted above bit 15 (mask {extra:#018x})'


def test_sign_extended_byte_load_keeps_wide_taint() -> None:
    """`movsx rax, byte ptr [r8+rax]` -- the sign bit reaches every upper bit.

    Guards the fix from over-narrowing: here the upper bits really do depend on
    the loaded byte, so bounding them to 8 would UNDER-taint.
    """
    t = _rax_taint('4c0fbe0400' if False else '490fbe0400')
    assert t & ~0xFF != 0, (
        'sign-extended load lost its upper-bit taint; the sign bit propagates '
        'into bits 8-63, so bounding this to the load width is unsound'
    )
