from __future__ import annotations

from typing import Any

from pytest_mock import MockerFixture

from microtaint.sleigh.polarity import compute_polarity


# Utility to create a mocked varnode
def create_varnode(mocker: MockerFixture, space_name: str, offset: int, size: int) -> Any:
    vn = mocker.MagicMock()
    vn.space.name = space_name
    vn.offset = offset
    vn.size = size
    return vn


# Utility to create a mocked PcodeOp
def create_pcode_op(
    mocker: MockerFixture,
    opcode_name: str,
    output_vn: Any | None,
    input_vns: list[Any],
) -> Any:
    op = mocker.MagicMock()
    op.opcode.name = opcode_name
    op.output = output_vn
    op.inputs = input_vns
    return op


def test_compute_polarity_empty_slice() -> None:
    assert compute_polarity([]) == {}


def test_compute_polarity_single_op(mocker: MockerFixture) -> None:
    # Example: out = in1 + in2 (INT_ADD)
    out_vn = create_varnode(mocker, 'register', 0, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)
    in2_vn = create_varnode(mocker, 'register', 8, 4)

    op = create_pcode_op(mocker, 'INT_ADD', out_vn, [in1_vn, in2_vn])

    result = compute_polarity([op])

    assert result == {
        'register:4:4': 1,
        'register:8:4': 1,
    }


def test_compute_polarity_int_sub(mocker: MockerFixture) -> None:
    # Example: out = in1 - in2 (INT_SUB)
    out_vn = create_varnode(mocker, 'register', 0, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)
    in2_vn = create_varnode(mocker, 'register', 8, 4)

    op = create_pcode_op(mocker, 'INT_SUB', out_vn, [in1_vn, in2_vn])

    result = compute_polarity([op])

    assert result == {
        # lhs
        'register:4:4': 1,
        # rhs is inverted
        'register:8:4': 0,
    }


def test_compute_polarity_int_negate(mocker: MockerFixture) -> None:
    # Example: out = ~in1 (INT_NEGATE)
    out_vn = create_varnode(mocker, 'register', 0, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)

    op = create_pcode_op(mocker, 'INT_NEGATE', out_vn, [in1_vn])

    result = compute_polarity([op])

    assert result == {
        'register:4:4': 0,
    }


def test_compute_polarity_chained_inversion(mocker: MockerFixture) -> None:
    # out = in1 - (~in2)
    # op1: tmp = ~in2 (INT_NEGATE)
    # op2: out = in1 - tmp (INT_SUB)

    out_vn = create_varnode(mocker, 'register', 0, 4)
    tmp_vn = create_varnode(mocker, 'unique', 10, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)
    in2_vn = create_varnode(mocker, 'register', 8, 4)

    op1 = create_pcode_op(mocker, 'INT_NEGATE', tmp_vn, [in2_vn])
    op2 = create_pcode_op(mocker, 'INT_SUB', out_vn, [in1_vn, tmp_vn])

    # Slice is in executed order, so reversed in compute_polarity: [op1, op2]
    # polarity_map is evaluated backwards: op2 then op1
    result = compute_polarity([op1, op2])

    assert result == {
        'register:4:4': 1,  # lhs of sub -> 1
        'register:8:4': 1,  # in2 -> negate of rhs of sub -> negate of 0 -> 1
        'unique:10:4': 0,  # tmp -> rh of sub
    }


def test_compute_polarity_ignores_consts(mocker: MockerFixture) -> None:
    # out = in1 + 5
    out_vn = create_varnode(mocker, 'register', 0, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)
    const_vn = create_varnode(mocker, 'const', 5, 4)

    op = create_pcode_op(mocker, 'INT_ADD', out_vn, [in1_vn, const_vn])

    result = compute_polarity([op])

    assert result == {
        'register:4:4': 1,
    }


def test_compute_polarity_fallback(mocker: MockerFixture) -> None:
    # Test an unhandled opcode like INT_DIV, should pass through polarity
    out_vn = create_varnode(mocker, 'register', 0, 4)
    in1_vn = create_varnode(mocker, 'register', 4, 4)

    op = create_pcode_op(mocker, 'INT_DIV', out_vn, [in1_vn])

    result = compute_polarity([op])

    assert result == {
        'register:4:4': 1,
    }
