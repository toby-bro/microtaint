import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintError, ImplicitTaintPolicy, Register


def test_implicit_taint_policies(capsys) -> None:  # type: ignore[no-untyped-def]
    """
    Tests the engine's ability to intercept, warn, or stop when
    evaluating conditional branches based on tainted flags.
    """
    # test rdi, 1   (Sets Zero Flag based on RDI)
    # jz skip       (Jumps based on Zero Flag)
    bytestring = b'\x48\xf7\xc7\x01\x00\x00\x00\x74\x03'

    amd64_registers = [
        Register(name='RDI', bits=64),
        Register(name='RIP', bits=64),
    ]

    sim = CellSimulator(Architecture.AMD64)
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # Base Context State
    state = {'RDI': 1, 'RIP': 0x400000}
    taint = {'RDI': 0xFFFFFFFFFFFFFFFF, 'RIP': 0}

    # --- Test 1: IGNORE (Default) ---
    ctx_ignore = EvalContext(
        input_values=state,
        input_taint=taint,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    out_ignore = circuit.evaluate(ctx_ignore)

    # Prove that the PC taint was successfully stripped to prevent path explosion!
    assert 'RIP' not in out_ignore

    # Clear stdout buffer before the next test
    capsys.readouterr()

    # --- Test 2: WARN ---
    ctx_warn = EvalContext(
        input_values=state,
        input_taint=taint,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.WARN,
    )
    out_warn = circuit.evaluate(ctx_warn)

    assert 'RIP' not in out_warn

    # Capture the print() output from stdout
    captured = capsys.readouterr()
    assert 'Implicit Taint Detected' in captured.out

    # --- Test 3: STOP ---
    ctx_stop = EvalContext(
        input_values=state,
        input_taint=taint,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.STOP,
    )

    # Prove that the engine throws the detailed Exception and halts execution
    with pytest.raises(ImplicitTaintError) as exc_info:
        circuit.evaluate(ctx_stop)

    assert 'FATAL: Implicit Taint Detected' in str(exc_info.value)
    assert '48f7c7010000007403' in str(exc_info.value)
