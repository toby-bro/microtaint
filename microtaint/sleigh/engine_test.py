from __future__ import annotations

from pytest_mock import MockerFixture

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import BinaryExpr, InstructionCellExpr, LogicCircuit, Op
from microtaint.sleigh.engine import _map_sleigh_to_state, generate_static_rule
from microtaint.types import Architecture, Register


class DummyRegister(Register):
    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size
        self.uc_const = 0
        self.bits = size * 8
        self.structure: list[int] = []
        self.value = None
        self.address = None


def test_map_sleigh_to_state_x86_flags(mocker: MockerFixture) -> None:
    ctx = mocker.MagicMock()
    state_format: list[Register] = [DummyRegister('FLAGS', size=8)]
    res = _map_sleigh_to_state(ctx, 'X86', state_format, 514, 1)
    assert res == ('FLAGS', 2, 2)


def test_map_sleigh_to_state_register(mocker: MockerFixture) -> None:
    ctx = mocker.MagicMock()
    reg_mock = mocker.MagicMock()
    reg_mock.offset = 16
    reg_mock.size = 8

    ctx.registers.get.side_effect = lambda name: reg_mock if name == 'RAX' else None
    state_format: list[Register] = [DummyRegister('RAX', size=8)]

    res = _map_sleigh_to_state(ctx, 'X86', state_format, 16, 8)
    assert res == ('RAX', 0, 63)


def test_map_sleigh_to_state_partial_register(mocker: MockerFixture) -> None:
    ctx = mocker.MagicMock()
    reg_mock = mocker.MagicMock()
    reg_mock.offset = 16
    reg_mock.size = 8

    ctx.registers.get.side_effect = lambda name: reg_mock if name == 'RAX' else None

    state_format: list[Register] = [DummyRegister('RAX', size=8)]

    res = _map_sleigh_to_state(ctx, 'X86', state_format, 16, 4)
    assert res == ('RAX', 0, 31)

    res2 = _map_sleigh_to_state(ctx, 'X86', state_format, 18, 2)
    assert res2 == ('RAX', 16, 31)


def test_map_sleigh_to_state_not_found(mocker: MockerFixture) -> None:
    ctx = mocker.MagicMock()
    ctx.registers.get.return_value = None
    state_format: list[Register] = [DummyRegister('RAX', size=8)]
    res = _map_sleigh_to_state(ctx, 'X86', state_format, 128, 4)
    assert res is None


def test_generate_static_rule_mapped(mocker: MockerFixture) -> None:
    mock_get_context = mocker.patch('microtaint.sleigh.engine.get_context')
    mock_slice_backward = mocker.patch('microtaint.sleigh.engine.slice_backward')
    mock_determine_category = mocker.patch('microtaint.sleigh.engine.determine_category')
    mock_compute_polarity = mocker.patch('microtaint.sleigh.engine.compute_polarity')

    ctx = mocker.MagicMock()
    mock_get_context.return_value = ctx

    reg_mock1 = mocker.MagicMock()
    reg_mock1.offset = 16
    reg_mock1.size = 8
    reg_mock2 = mocker.MagicMock()
    reg_mock2.offset = 32
    reg_mock2.size = 8

    def mock_get(name: str) -> object:
        if name == 'RAX':
            return reg_mock1
        if name == 'RBX':
            return reg_mock2
        return None

    ctx.registers.get.side_effect = mock_get

    translation = mocker.MagicMock()
    mock_op = mocker.MagicMock()
    mock_op.output.space.name = 'register'
    mock_op.output.offset = 16
    mock_op.output.size = 8

    translation.ops = [mock_op]
    ctx.translate.return_value = translation

    mock_slice_backward.return_value = [mock_op]
    mock_determine_category.return_value = InstructionCategory.MAPPED
    mock_compute_polarity.return_value = {'register:32:8': 1}

    state_format: list[Register] = [DummyRegister('RAX', size=8), DummyRegister('RBX', size=8)]
    bytestring = b'\x00'

    rule = generate_static_rule(Architecture.X86, bytestring, state_format)

    assert isinstance(rule, LogicCircuit)
    assert len(rule.assignments) == 1
    assignment = rule.assignments[0]

    assert assignment.target.name == 'RAX'
    assert len(assignment.dependencies) == 1
    assert assignment.dependencies[0].name == 'RBX'
    assert 'OR' not in str(assignment.expression)


def test_generate_static_rule_transportable(mocker: MockerFixture) -> None:
    mock_get_context = mocker.patch('microtaint.sleigh.engine.get_context')
    mock_slice_backward = mocker.patch('microtaint.sleigh.engine.slice_backward')
    mock_determine_category = mocker.patch('microtaint.sleigh.engine.determine_category')
    mock_compute_polarity = mocker.patch('microtaint.sleigh.engine.compute_polarity')

    ctx = mocker.MagicMock()
    mock_get_context.return_value = ctx
    reg_mock1 = mocker.MagicMock()
    reg_mock1.offset = 16
    reg_mock1.size = 8
    reg_mock2 = mocker.MagicMock()
    reg_mock2.offset = 32
    reg_mock2.size = 8

    ctx.registers.get.side_effect = lambda name: reg_mock1 if name == 'RAX' else (reg_mock2 if name == 'RBX' else None)

    translation = mocker.MagicMock()
    mock_op = mocker.MagicMock()
    mock_op.output.space.name = 'register'
    mock_op.output.offset = 16
    mock_op.output.size = 8

    translation.ops = [mock_op]
    ctx.translate.return_value = translation
    mock_slice_backward.return_value = [mock_op]
    mock_determine_category.return_value = InstructionCategory.TRANSPORTABLE
    mock_compute_polarity.return_value = {'register:32:8': 0}

    state_format: list[Register] = [DummyRegister('RAX', size=8), DummyRegister('RBX', size=8)]

    rule = generate_static_rule(Architecture.X86, b'\x00', state_format)
    assert len(rule.assignments) == 1
    expr = rule.assignments[0].expression
    assert isinstance(expr, BinaryExpr)
    assert expr.op == Op.OR


def test_generate_static_rule_unknown(mocker: MockerFixture) -> None:
    mock_get_context = mocker.patch('microtaint.sleigh.engine.get_context')
    mock_slice_backward = mocker.patch('microtaint.sleigh.engine.slice_backward')
    mock_determine_category = mocker.patch('microtaint.sleigh.engine.determine_category')
    mock_compute_polarity = mocker.patch('microtaint.sleigh.engine.compute_polarity')

    ctx = mocker.MagicMock()
    mock_get_context.return_value = ctx
    reg_mock1 = mocker.MagicMock()
    reg_mock1.offset = 16
    reg_mock1.size = 8
    reg_mock2 = mocker.MagicMock()
    reg_mock2.offset = 32
    reg_mock2.size = 8

    ctx.registers.get.side_effect = lambda name: reg_mock1 if name == 'RAX' else (reg_mock2 if name == 'RBX' else None)

    translation = mocker.MagicMock()
    mock_op = mocker.MagicMock()
    mock_op.output.space.name = 'register'
    mock_op.output.offset = 16
    mock_op.output.size = 8

    translation.ops = [mock_op]
    ctx.translate.return_value = translation
    mock_slice_backward.return_value = [mock_op]
    mock_determine_category.return_value = InstructionCategory.UNKNOWN
    mock_compute_polarity.return_value = {'register:32:8': 1}

    state_format: list[Register] = [DummyRegister('RAX', size=8), DummyRegister('RBX', size=8)]

    rule = generate_static_rule(Architecture.X86, b'\x00', state_format)
    assert len(rule.assignments) == 1
    expr = rule.assignments[0].expression
    assert isinstance(expr, InstructionCellExpr)
