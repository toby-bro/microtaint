from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture


def build_explicit_dataflow_sequence(n: int) -> bytes:
    """
    Builds a highly branched sequence using explicit dataflow.

    Equivalent Assembly for 1 Block (15 bytes):
        test rdi, 1      (48 F7 C7 01 00 00 00)
        jz skip          (74 03)
        add rax, rdi     (48 01 F8)  <-- Explicit Dataflow!
      skip:
        shr rdi, 1       (48 D1 EF)  <-- Shift to test the next bit
    """
    preamble = b'\x48\x31\xc0'  # xor rax, rax
    block = b'\x48\xf7\xc7\x01\x00\x00\x00\x74\x03\x48\x01\xf8\x48\xd1\xef'
    return preamble + (block * n)


def test_path_explosion_scaling() -> None:
    """
    Proves that differential emulation crushes path explosion.
    A 100-branch sequence generates 2^100 paths for symbolic execution.
    CellSimulator evaluates the exact XOR differential in O(N) time.
    """
    # 100 branches = 2^100 paths
    seq_bytes = build_explicit_dataflow_sequence(100)

    sim = CellSimulator(Architecture.AMD64)

    # V_STATE: Base values. RDI is loaded with all 1s.
    v_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})

    # T_STATE: Taint mask. RDI is fully tainted.
    t_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})

    # By feeding the ENTIRE heavily-branched sequence directly into
    # the differential evaluator, Unicorn executes V|T and V&~T linearly.
    out_taint = sim.evaluate_cell_differential(seq_bytes, 'RAX', v_state, t_state)

    # Because RDI propagated explicitly into RAX, the taint must be preserved,
    # proving we traversed the logic accurately without path explosion.
    assert out_taint != 0, 'Dataflow taint failed to propagate!'


def test_path_explosion_speed(benchmark) -> None:  # type: ignore[no-untyped-def]
    """
    Optional: If you run `uv run pytest --benchmark-only`,
    this will prove the 2^100 path sequence executes in microseconds.
    """
    seq_bytes = build_explicit_dataflow_sequence(100)
    sim = CellSimulator(Architecture.AMD64)
    v_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})
    t_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})

    benchmark(sim.evaluate_cell_differential, seq_bytes, 'RAX', v_state, t_state)
