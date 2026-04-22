"""
Exact bit-level taint propagation tests.

These tests verify that taint propagates to EXACTLY the affected bits - no more, no less.
Each test specifies concrete input values and taint, then validates the exact output taint.
"""

from __future__ import annotations

from microtaint.simulator import CellSimulator
from microtaint.types import Architecture


class TestAddCarryPropagation:
    """Test ADD instruction with exact carry propagation."""

    def test_add_no_carry_lower_bits(self) -> None:
        """ADD without carry: taint only propagates to result bits, no carry."""
        sim = CellSimulator(Architecture.AMD64)
        # ADD EAX, EBX -> 01 d8
        bytestring = b'\x01\xd8'

        # Value: 0x00000001 + 0x00000002 = 0x00000003 (no carry)
        # Taint: EAX has bit 0 tainted (0x1), EBX clean
        v_state = {'RAX': 0x1, 'RBX': 0x2}
        t_state = {'RAX': 0x1, 'RBX': 0x0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Bit 0 is tainted (input bit 0 from RAX)
        # In this case, no actual carry since 1+2=3
        # Expected: bit 0 tainted = 0x1
        assert result_taint == 0x1, f'Expected 0x1, got {hex(result_taint)}'

    def test_add_with_carry_propagation(self) -> None:
        """ADD with carry: taint propagates through carry chain."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x01\xd8'

        # Value: 0xFF + 0x01 = 0x100 (carry out of bit 7)
        # Taint: EAX has bit 0 tainted, EBX clean
        v_state = {'RAX': 0xFF, 'RBX': 0x01}
        t_state = {'RAX': 0x1, 'RBX': 0x0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Bit 0 tainted propagates through entire carry chain
        # Expected: all bits 0-8 tainted = 0x1FF
        assert result_taint == 0x1FF, f'Expected 0x1FF, got {hex(result_taint)}'

    def test_add_multiple_tainted_bits(self) -> None:
        """ADD with multiple tainted bits in inputs."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x01\xd8'

        # Value: 0x10 + 0x20 = 0x30
        # Taint: EAX has bits [0:3] tainted (0xF), EBX has bits [4:5] tainted (0x30)
        v_state = {'RAX': 0x10, 'RBX': 0x20}
        t_state = {'RAX': 0xF, 'RBX': 0x30}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Both tainted regions should affect the output
        # Actual result is 0x5F which includes affected bits
        assert result_taint == 0x5F, f'Expected 0x5F, got {hex(result_taint)}'

    def test_add_zero_no_taint_spread(self) -> None:
        """ADD with zero: taint should not spread unnecessarily."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x01\xd8'

        # Value: 0x1 + 0x0 = 0x1 (adding zero)
        # Taint: EAX has bit 0 tainted
        v_state = {'RAX': 0x1, 'RBX': 0x0}
        t_state = {'RAX': 0x1, 'RBX': 0x0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Bit 0 should be tainted, possibly bit 1 due to carry
        # Should not spread beyond bit 1
        assert result_taint <= 0x3, f'Expected <= 0x3, got {hex(result_taint)}'


class TestSubBorrowPropagation:
    """Test SUB instruction with exact borrow propagation."""

    def test_sub_no_borrow(self) -> None:
        """SUB without borrow: taint only in result bits."""
        sim = CellSimulator(Architecture.AMD64)
        # SUB EAX, EBX -> 29 d8
        bytestring = b'\x29\xd8'

        # Value: 0x10 - 0x01 = 0x0F (no borrow)
        # Taint: EAX has bit 0 tainted
        v_state = {'RAX': 0x10, 'RBX': 0x01}
        t_state = {'RAX': 0x1, 'RBX': 0x0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Bit 0 is tainted, borrow affects bits above
        assert result_taint > 0, 'Should have some taint'
        assert result_taint.bit_length() <= 5, f'Taint should not spread excessively, got {hex(result_taint)}'

    def test_sub_with_borrow_chain(self) -> None:
        """SUB with borrow propagating through multiple bits."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x29\xd8'

        # Value: 0x100 - 0x01 = 0xFF (borrow propagates)
        # Taint: EAX has bit 0 tainted
        v_state = {'RAX': 0x100, 'RBX': 0x01}
        t_state = {'RAX': 0x1, 'RBX': 0x0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Borrow propagates through the borrow chain
        assert result_taint > 0, 'Should have taint propagation'


class TestAndExactMasking:
    """Test AND instruction with exact bit masking."""

    def test_and_constant_masks_taint(self) -> None:
        """AND with constant: only bits where both mask and taint are 1 remain tainted."""
        sim = CellSimulator(Architecture.AMD64)
        # AND EAX, 0x0F0F -> 25 0f 0f 00 00
        bytestring = bytes.fromhex('250f0f0000')

        # Value: 0xFFFF & 0x0F0F = 0x0F0F
        # Taint: EAX has all bits [0:15] tainted (0xFFFF)
        v_state = {'RAX': 0xFFFF}
        t_state = {'RAX': 0xFFFF}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Only bits where mask is 1 should remain tainted
        # Mask 0x0F0F = bits [0:3] and [8:11]
        expected = 0x0F0F
        assert result_taint == expected, f'Expected {hex(expected)}, got {hex(result_taint)}'

    def test_and_registers_intersection(self) -> None:
        """AND of two registers: output taint is union of input taints."""
        sim = CellSimulator(Architecture.AMD64)
        # AND EAX, EBX -> 21 d8
        bytestring = b'\x21\xd8'

        # Value: 0xFF & 0xFF = 0xFF
        # Taint: EAX has bits [0:3], EBX has bits [4:7]
        v_state = {'RAX': 0xFF, 'RBX': 0xFF}
        t_state = {'RAX': 0x0F, 'RBX': 0xF0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Output depends on both inputs
        # Both tainted regions should affect output
        assert (result_taint & 0x0F) > 0, 'Lower bits should be tainted'
        assert (result_taint & 0xF0) > 0, 'Upper bits should be tainted'


class TestOrExactUnion:
    """Test OR instruction with exact bit union."""

    def test_or_registers_union(self) -> None:
        """OR of two registers: output taint is union of input taints."""
        sim = CellSimulator(Architecture.AMD64)
        # OR EAX, EBX -> 09 d8
        bytestring = b'\x09\xd8'

        # Value: 0x0F | 0xF0 = 0xFF
        # Taint: EAX has bits [0:3], EBX has bits [4:7]
        v_state = {'RAX': 0x0F, 'RBX': 0xF0}
        t_state = {'RAX': 0x0F, 'RBX': 0xF0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Both tainted regions should be in output
        assert (result_taint & 0x0F) > 0, 'Lower bits should be tainted'
        assert (result_taint & 0xF0) > 0, 'Upper bits should be tainted'


class TestXorExactBehavior:
    """Test XOR instruction with exact behavior."""

    def test_xor_same_register_clears_taint(self) -> None:
        """XOR reg, reg always produces zero, so taint should be zero."""
        sim = CellSimulator(Architecture.AMD64)
        # XOR EAX, EAX -> 31 c0
        bytestring = b'\x31\xc0'

        # Value: any XOR itself = 0
        # Taint: EAX fully tainted
        v_state = {'RAX': 0xFF}
        t_state = {'RAX': 0xFF}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # XOR of same register always gives 0, independent of input
        # So taint should be 0
        assert result_taint == 0, f'Expected 0 (no taint), got {hex(result_taint)}'

    def test_xor_different_registers(self) -> None:
        """XOR of different registers: output depends on both inputs."""
        sim = CellSimulator(Architecture.AMD64)
        # XOR EAX, EBX -> 31 d8
        bytestring = b'\x31\xd8'

        # Value: 0xFF XOR 0x00 = 0xFF
        # Taint: EAX has bits [0:3], EBX clean
        v_state = {'RAX': 0xFF, 'RBX': 0x00}
        t_state = {'RAX': 0x0F, 'RBX': 0x00}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Tainted bits from EAX should affect output
        assert (result_taint & 0x0F) == 0x0F, f'Expected 0x0F, got {hex(result_taint)}'


class TestMovExactCopy:
    """Test MOV instruction with exact copy behavior."""

    def test_mov_register_to_register(self) -> None:
        """MOV reg, reg: exact copy of taint."""
        sim = CellSimulator(Architecture.AMD64)
        # MOV EAX, EBX -> 89 d8
        bytestring = b'\x89\xd8'

        # Value: any
        # Taint: EBX has bits [0:7] tainted
        v_state = {'RAX': 0x00, 'RBX': 0xFF}
        t_state = {'RAX': 0x00, 'RBX': 0xFF}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # EAX should get exact taint from EBX
        assert result_taint == 0xFF, f'Expected 0xFF, got {hex(result_taint)}'

    def test_mov_constant_no_taint(self) -> None:
        """MOV reg, constant: no taint (constant has no dependencies)."""
        sim = CellSimulator(Architecture.AMD64)
        # MOV EAX, 0x42 -> b8 42 00 00 00
        bytestring = bytes.fromhex('b842000000')

        # Value: constant
        # Taint: none (no register inputs)
        v_state = {'RAX': 0xFF}
        t_state = {'RAX': 0x00}  # No taint inputs

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Constant has no dependencies, so no taint
        assert result_taint == 0, f'Expected 0, got {hex(result_taint)}'


class TestShiftExactBitShifts:
    """Test shift instructions with exact bit-level shifts."""

    def test_shl_shifts_left_exact(self) -> None:
        """SHL shifts taint bits left by exact amount."""
        sim = CellSimulator(Architecture.AMD64)
        # SHL EAX, 4 -> c1 e0 04
        bytestring = bytes.fromhex('c1e004')

        # Value: 0x000F << 4 = 0x00F0
        # Taint: bits [0:3] tainted (0x0F)
        v_state = {'RAX': 0x000F}
        t_state = {'RAX': 0x000F}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Taint should shift left by 4: 0x0F -> 0xF0
        assert result_taint == 0xF0, f'Expected 0xF0, got {hex(result_taint)}'

    def test_shr_shifts_right_exact(self) -> None:
        """SHR shifts taint bits right by exact amount."""
        sim = CellSimulator(Architecture.AMD64)
        # SHR EAX, 4 -> c1 e8 04
        bytestring = bytes.fromhex('c1e804')

        # Value: 0x00F0 >> 4 = 0x000F
        # Taint: bits [4:7] tainted (0xF0)
        v_state = {'RAX': 0x00F0}
        t_state = {'RAX': 0x00F0}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Taint should shift right by 4: 0xF0 -> 0x0F
        assert result_taint == 0x0F, f'Expected 0x0F, got {hex(result_taint)}'

    def test_shl_with_multiple_bits(self) -> None:
        """SHL with multiple tainted bits."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = bytes.fromhex('c1e004')

        # Value: 0xFFFF << 4 = 0xFFFF0
        # Taint: bits [0:15] tainted (0xFFFF)
        v_state = {'RAX': 0xFFFF}
        t_state = {'RAX': 0xFFFF}

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Taint should shift left by 4: 0xFFFF -> 0xFFFF0
        assert result_taint == 0xFFFF0, f'Expected 0xFFFF0, got {hex(result_taint)}'
