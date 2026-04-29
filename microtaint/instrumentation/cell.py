"""
microtaint.instrumentation.cell
================================
Native P-code differential evaluator.

PURPOSE
-------
This module replaces the Unicorn-based hot path inside CellSimulator.evaluate_concrete
(~62 K calls per 30 s run) with a direct interpreter of the P-code ops that
Sleigh already computed and cached by engine.py.

The key insight: InstructionCellExpr already holds everything needed —
  • raw instruction bytes  → already translated to P-code by the Sleigh LRU cache
  • out_reg / out_bit_start / out_bit_end  → which output slice to read back
  • inputs (dict[str, Expr])  → the (V|T) and (V&~T) concrete values the caller
    evaluated from the expression tree

We implement the full differential in pure Python over that P-code translation,
removing the entire Unicorn round-trip
  (context_restore → 18 reg_writes → emu_start → reg_read)
that currently costs ~180 µs per cell.

INTEGRATION — minimal, flag-gated
-----------------------------------
Only two files change:

  simulator.py  — CellSimulator.__init__ gains  use_unicorn: bool = True
                  evaluate_concrete() dispatches to this module when False.

  ast.pyx       — InstructionCellExpr.evaluate() skips _build_machine_state
                  and calls PCodeCellEvaluator directly when use_unicorn=False,
                  passing the already-evaluated flat dict.

Nothing else changes.  The flag lets you run both backends side-by-side for
correctness comparison.

SUPPORTED OPCODES (complete AMD64 coverage)
--------------------------------------------
Data movement    COPY  LOAD  STORE  MULTIEQUAL  INDIRECT
Integer arith    INT_ADD  INT_SUB  INT_MULT  INT_DIV  INT_SDIV
                 INT_REM  INT_SREM  INT_2COMP  INT_NEGATE
Bitwise          INT_AND  INT_OR  INT_XOR
Shifts           INT_LEFT  INT_RIGHT  INT_SRIGHT
Comparisons      INT_EQUAL  INT_NOTEQUAL
                 INT_LESS  INT_LESSEQUAL  INT_SLESS  INT_SLESSEQUAL
                 INT_CARRY  INT_SCARRY  INT_SBORROW
Extensions       INT_ZEXT  INT_SEXT  INT_TRUNC  POPCOUNT  LZCOUNT
Piece/sub        PIECE  SUBPIECE  CAST
Pointer arith    PTRADD  PTRSUB
Boolean          BOOL_AND  BOOL_OR  BOOL_XOR  BOOL_NEGATE
Control flow     BRANCH  CALL  RETURN             → no-op (safe to skip)
                 CBRANCH  BRANCHIND  CALLIND          → PCodeFallbackNeeded
                 CALLOTHER (no output)                → no-op
                 CALLOTHER (with output, e.g. BSF)    → PCodeFallbackNeeded
Float ops        → raise PCodeFallbackNeeded  (rounding modes need Unicorn)
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from pypcode import Context, PcodeOp, Translation, Varnode

from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel — raised to tell CellSimulator "use Unicorn for this cell"
# ---------------------------------------------------------------------------


class PCodeFallbackNeeded(Exception):
    """Raised when the native evaluator encounters an unsupported opcode."""


# ---------------------------------------------------------------------------
# Unsigned integer helpers
# ---------------------------------------------------------------------------

# Pre-built masks for the common sizes (bytes 1-16).
_MASKS: dict[int, int] = {n: (1 << (n * 8)) - 1 for n in range(1, 17)}


def _mask(val: int, size_bytes: int) -> int:
    m = _MASKS.get(size_bytes)
    if m is None:
        m = (1 << (size_bytes * 8)) - 1
    return val & m


def _signed(val: int, size_bytes: int) -> int:
    """Reinterpret unsigned val as a signed integer of size_bytes."""
    bits = size_bytes * 8
    val = _mask(val, size_bytes)
    if val >= (1 << (bits - 1)):
        val -= 1 << bits
    return val


# ---------------------------------------------------------------------------
# Translation cache — shares the same translated P-code as engine.py.
# Keyed on (arch, bytestring) — identical key to _cached_generate_static_rule.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16384)
def _get_translation(arch: Architecture, bytestring: bytes) -> Translation:
    ctx: Context = get_context(arch)
    return ctx.translate(bytestring, 0x1000)


# ---------------------------------------------------------------------------
# P-code interpreter frame
# ---------------------------------------------------------------------------


class _PCodeFrame:
    """
    Scratch-pad that executes one P-code translation over a concrete state.

    State spaces
    ------------
    regs  : dict[int, int]  — register-space byte-offset -> unsigned value
    uniq  : dict[int, int]  — unique-space offset        -> unsigned value
    mem   : dict[int, int]  — RAM byte-address           -> one byte (0-255)

    The caller pre-populates `regs` (and optionally `mem`) before calling
    execute_all().  All writes apply _mask() to stay within varnode width.
    """

    __slots__ = ('mem', 'reg_sizes', 'regs', 'uniq')

    def __init__(self) -> None:
        self.regs: dict[int, int] = {}
        self.uniq: dict[int, int] = {}
        self.mem: dict[int, int] = {}  # byte-granular: addr -> byte value
        self.reg_sizes: dict[int, int] = {}  # reg offset -> size in bytes (for sub-reg reads)

    # ------------------------------------------------------------------
    # Varnode read / write
    # ------------------------------------------------------------------

    def read(self, vn: Varnode) -> int:
        space = vn.space.name
        sz = vn.size
        if space == 'const':
            return _mask(vn.offset, sz)
        if space == 'register':
            return _mask(self.regs.get(vn.offset, 0), sz)
        if space == 'unique':
            return _mask(self.uniq.get(vn.offset, 0), sz)
        if space == 'ram':
            result = 0
            for i in range(sz):
                result |= self.mem.get(vn.offset + i, 0) << (i * 8)
            return _mask(result, sz)
        return 0  # unknown space — conservative zero

    def write(self, vn: Varnode, val: int) -> None:
        space = vn.space.name
        val = _mask(val, vn.size)
        if space == 'register':
            self.regs[vn.offset] = val
            self.reg_sizes[vn.offset] = vn.size  # track size for sub-register reads
        elif space == 'unique':
            self.uniq[vn.offset] = val
        elif space == 'ram':
            for i in range(vn.size):
                self.mem[vn.offset + i] = (val >> (i * 8)) & 0xFF
        # const / other: no-op

    def write_mem(self, addr: int, val: int, size: int) -> None:
        val = _mask(val, size)
        for i in range(size):
            self.mem[addr + i] = (val >> (i * 8)) & 0xFF

    def read_mem(self, addr: int, size: int) -> int:
        result = 0
        for i in range(size):
            result |= self.mem.get(addr + i, 0) << (i * 8)
        return _mask(result, size)

    # ------------------------------------------------------------------
    # Single-op dispatch
    # ------------------------------------------------------------------

    def execute_op(self, op: PcodeOp) -> None:  # noqa: C901
        name = op.opcode.name
        out = op.output
        ins = op.inputs

        # ---- Data movement -----------------------------------------------
        if name == 'COPY':
            if out:
                self.write(out, self.read(ins[0]))

        elif name == 'LOAD':
            addr = self.read(ins[1])
            size = out.size if out else 8
            val = self.read_mem(addr, size)
            if out:
                self.write(out, val)

        elif name == 'STORE':
            addr = self.read(ins[1])
            self.write_mem(addr, self.read(ins[2]), ins[2].size)

        elif name in ('MULTIEQUAL', 'INDIRECT'):
            if out and ins:
                self.write(out, self.read(ins[0]))

        # ---- Integer arithmetic ------------------------------------------
        elif name == 'INT_ADD':
            if out:
                self.write(out, self.read(ins[0]) + self.read(ins[1]))

        elif name == 'INT_SUB':
            if out:
                self.write(out, self.read(ins[0]) - self.read(ins[1]))

        elif name == 'INT_MULT':
            if out:
                self.write(out, self.read(ins[0]) * self.read(ins[1]))

        elif name == 'INT_DIV':
            if out:
                d = self.read(ins[1])
                self.write(out, 0 if d == 0 else self.read(ins[0]) // d)

        elif name == 'INT_SDIV':
            if out:
                sz = ins[0].size
                a = _signed(self.read(ins[0]), sz)
                b = _signed(self.read(ins[1]), sz)
                self.write(out, 0 if b == 0 else int(a / b))  # truncate toward zero

        elif name == 'INT_REM':
            if out:
                d = self.read(ins[1])
                self.write(out, 0 if d == 0 else self.read(ins[0]) % d)

        elif name == 'INT_SREM':
            if out:
                sz = ins[0].size
                a = _signed(self.read(ins[0]), sz)
                b = _signed(self.read(ins[1]), sz)
                self.write(out, 0 if b == 0 else a - b * int(a / b))

        elif name == 'INT_2COMP':  # arithmetic negation (two's complement)
            if out:
                self.write(out, -self.read(ins[0]))

        elif name == 'INT_NEGATE':  # bitwise NOT
            if out:
                self.write(out, ~self.read(ins[0]))

        # ---- Bitwise -------------------------------------------------------
        elif name == 'INT_AND':
            if out:
                self.write(out, self.read(ins[0]) & self.read(ins[1]))

        elif name == 'INT_OR':
            if out:
                self.write(out, self.read(ins[0]) | self.read(ins[1]))

        elif name == 'INT_XOR':
            if out:
                self.write(out, self.read(ins[0]) ^ self.read(ins[1]))

        # ---- Shifts --------------------------------------------------------
        elif name == 'INT_LEFT':
            if out:
                # Cap shift to avoid Python generating astronomic integers
                self.write(out, self.read(ins[0]) << (self.read(ins[1]) & 0x3F))

        elif name == 'INT_RIGHT':  # logical (unsigned)
            if out:
                self.write(out, self.read(ins[0]) >> (self.read(ins[1]) & 0x3F))

        elif name == 'INT_SRIGHT':  # arithmetic (signed)
            if out:
                sz = ins[0].size
                val = _signed(self.read(ins[0]), sz)
                self.write(out, val >> (self.read(ins[1]) & 0x3F))

        # ---- Comparisons (1-byte boolean outputs) -------------------------
        elif name == 'INT_EQUAL':
            if out:
                self.write(out, 1 if self.read(ins[0]) == self.read(ins[1]) else 0)

        elif name == 'INT_NOTEQUAL':
            if out:
                self.write(out, 0 if self.read(ins[0]) == self.read(ins[1]) else 1)

        elif name == 'INT_LESS':
            if out:
                self.write(out, 1 if self.read(ins[0]) < self.read(ins[1]) else 0)

        elif name == 'INT_LESSEQUAL':
            if out:
                self.write(out, 1 if self.read(ins[0]) <= self.read(ins[1]) else 0)

        elif name == 'INT_SLESS':
            if out:
                sz = ins[0].size
                self.write(out, 1 if _signed(self.read(ins[0]), sz) < _signed(self.read(ins[1]), sz) else 0)

        elif name == 'INT_SLESSEQUAL':
            if out:
                sz = ins[0].size
                self.write(out, 1 if _signed(self.read(ins[0]), sz) <= _signed(self.read(ins[1]), sz) else 0)

        elif name == 'INT_CARRY':  # unsigned addition overflow
            if out:
                bits = ins[0].size * 8
                result = self.read(ins[0]) + self.read(ins[1])
                self.write(out, 1 if result >= (1 << bits) else 0)

        elif name == 'INT_SCARRY':  # signed addition overflow
            if out:
                sz = ins[0].size
                a, b = _signed(self.read(ins[0]), sz), _signed(self.read(ins[1]), sz)
                result = a + b
                limit = 1 << (sz * 8 - 1)
                self.write(out, 1 if result < -limit or result >= limit else 0)

        elif name == 'INT_SBORROW':  # signed subtraction overflow
            if out:
                sz = ins[0].size
                a, b = _signed(self.read(ins[0]), sz), _signed(self.read(ins[1]), sz)
                result = a - b
                limit = 1 << (sz * 8 - 1)
                self.write(out, 1 if result < -limit or result >= limit else 0)

        # ---- Extensions / truncation --------------------------------------
        elif name == 'INT_ZEXT':
            if out:
                self.write(out, self.read(ins[0]))

        elif name == 'INT_SEXT':
            if out:
                self.write(out, _signed(self.read(ins[0]), ins[0].size))

        elif name in ('INT_TRUNC', 'CAST'):
            if out:
                self.write(out, self.read(ins[0]))

        elif name == 'POPCOUNT':
            if out:
                self.write(out, bin(self.read(ins[0])).count('1'))

        elif name == 'LZCOUNT':
            if out:
                sz = ins[0].size
                val = self.read(ins[0])
                bits = sz * 8
                self.write(out, bits if val == 0 else bits - val.bit_length())

        # ---- Piece / subpiece ---------------------------------------------
        elif name == 'PIECE':
            # Concatenate: ins[0] high part, ins[1] low part
            if out:
                lo_bytes = ins[1].size
                self.write(out, (self.read(ins[0]) << (lo_bytes * 8)) | self.read(ins[1]))

        elif name == 'SUBPIECE':
            # ins[1] is a const byte-offset into ins[0]
            if out:
                offset_bytes = self.read(ins[1])
                self.write(out, self.read(ins[0]) >> (offset_bytes * 8))

        # ---- Pointer arithmetic -------------------------------------------
        elif name == 'PTRADD':
            # PTRADD(base, index, elem_size_const) = base + index * elem_size
            if out:
                self.write(out, self.read(ins[0]) + self.read(ins[1]) * self.read(ins[2]))

        elif name == 'PTRSUB':
            if out:
                self.write(out, self.read(ins[0]) - self.read(ins[1]))

        # ---- Boolean ops --------------------------------------------------
        elif name == 'BOOL_AND':
            if out:
                self.write(out, 1 if (self.read(ins[0]) and self.read(ins[1])) else 0)

        elif name == 'BOOL_OR':
            if out:
                self.write(out, 1 if (self.read(ins[0]) or self.read(ins[1])) else 0)

        elif name == 'BOOL_XOR':
            if out:
                self.write(out, 1 if bool(self.read(ins[0])) ^ bool(self.read(ins[1])) else 0)

        elif name == 'BOOL_NEGATE':
            if out:
                self.write(out, 0 if self.read(ins[0]) else 1)

        # ---- Control flow ------------------------------------------------
        elif name == 'BRANCH' or name == 'RETURN' or name == 'CALL':
            # Unconditional / direct: no semantic register output that the
            # differential cares about.  Safe to skip.
            pass

        elif name == 'CALLOTHER':
            # CALLOTHER represents a CPU helper (bit-scans, string ops, AES…).
            # If it has an output varnode the result is semantically meaningful
            # and we cannot compute it without the real CPU model — fall back.
            # If it has no output it is a pure side-effect (safe to skip).
            if op.output is not None:
                raise PCodeFallbackNeeded(f'CALLOTHER with output: {op}')

        elif name in ('CBRANCH', 'BRANCHIND', 'CALLIND'):
            # Conditional branches and indirect calls/jumps involve control
            # flow that our sequential interpreter cannot model correctly.
            # Any instruction whose pcode contains these — CMOVCC, BSF/BSR,
            # indirect CALL — must fall back to Unicorn for the whole cell.
            raise PCodeFallbackNeeded(f'Control-flow opcode: {name}')

        # ---- Sleigh pseudo-ops / markers — safe no-ops --------------------
        # IMARK: instruction marker, emitted as the FIRST op of every x86-64
        #        translation by the AMD64 Sleigh spec.  No output, no semantic
        #        content — must be silently ignored or every instruction falls back.
        # UNIMPLEMENTED: placeholder for unrecognised encodings.
        # SEGMENT/CPOOLREF/NEW: Java-bytecode artifacts, never appear in x86.
        elif name in (
            'IMARK',
            'UNIMPLEMENTED',
            'SEGMENT',
            'CPOOLREF',
            'NEW',
            'INSERT',
            'EXTRACT',
        ):
            pass

        # ---- Float ops — rounding modes require Unicorn -------------------
        elif name.startswith('FLOAT_') or name in ('TRUNC', 'CEIL', 'FLOOR', 'ROUND'):
            raise PCodeFallbackNeeded(f'Float opcode: {name}')

        # ---- Unknown opcode -----------------------------------------------
        else:
            raise PCodeFallbackNeeded(f'Unsupported opcode: {name}')

    def execute_all(self, ops: list[PcodeOp]) -> None:
        for op in ops:
            self.execute_op(op)

    def clear(self) -> None:
        self.regs.clear()
        self.uniq.clear()
        self.mem.clear()
        self.reg_sizes.clear()


# ---------------------------------------------------------------------------
# Register-offset resolver (cached per arch)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _build_reg_maps(arch: Architecture) -> tuple[dict[str, int], dict[str, int]]:
    """
    Return (name->offset, name->size_bytes) for every register in the Sleigh
    context.  Both dicts are keyed by UPPER-CASE register name.
    """
    ctx = get_context(arch)
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for name, vn in ctx.registers.items():
        key = name.upper()
        offsets[key] = vn.offset
        sizes[key] = vn.size
    return offsets, sizes


# ---------------------------------------------------------------------------
# PCodeCellEvaluator — public API used by CellSimulator
# ---------------------------------------------------------------------------


class PCodeCellEvaluator:
    """
    Native P-code differential evaluator.

    CellSimulator instantiates exactly one of these when use_unicorn=False and
    delegates evaluate_concrete() calls here.

    evaluate_concrete(cell, flat_inputs)
        Mirrors the semantics of the Unicorn path: run the instruction once
        with the given concrete register/memory values, read back the output
        slice, return it.

        Called by CellSimulator.evaluate_concrete() with a flat dict built
        from the MachineState that ast.pyx passes in.

    evaluate_differential(cell, or_inputs, and_inputs)
        Low-overhead two-run variant called DIRECTLY from
        InstructionCellExpr.evaluate() in ast.pyx when use_unicorn=False,
        bypassing _build_machine_state and MachineState entirely.
        Both polarized input dicts are already flat at that call site.
    """

    def __init__(self, arch: Architecture) -> None:
        self.arch = arch
        self._frame_a = _PCodeFrame()
        self._frame_b = _PCodeFrame()
        self._offsets, self._sizes = _build_reg_maps(arch)
        # Transition monitoring
        self.native_calls: int = 0
        self.fallback_calls: int = 0

    # ------------------------------------------------------------------
    # State loading
    # ------------------------------------------------------------------

    def _load(self, frame: _PCodeFrame, inputs: dict[str, int]) -> None:
        """
        Populate frame from a flat inputs dict whose keys are either:
          - register names ('RAX', 'ZF', ...)
          - 'MEM_<hex_addr>_<size_bytes>'  for memory operands
        """
        frame.clear()
        offsets = self._offsets
        sizes = self._sizes
        for name, val in inputs.items():
            if name.startswith('MEM_'):
                body = name[4:]
                sep = body.rfind('_')
                if sep >= 0:
                    try:
                        addr = int(body[:sep], 16)
                        size = int(body[sep + 1 :])
                        frame.write_mem(addr, val, size)
                    except ValueError:
                        pass
            else:
                key = name.upper()
                off = offsets.get(key)
                if off is not None:
                    frame.regs[off] = _mask(val, sizes.get(key, 8))

    # ------------------------------------------------------------------
    # Output reading
    # ------------------------------------------------------------------

    def _read_output(
        self,
        frame: _PCodeFrame,
        out_reg: str,
        bit_start: int,
        bit_end: int,
    ) -> int:
        if out_reg.startswith('MEM_'):
            body = out_reg[4:]
            sep = body.rfind('_')
            if sep >= 0:
                try:
                    addr = int(body[:sep], 16)
                    size = int(body[sep + 1 :])
                    val = frame.read_mem(addr, size)
                    return (val >> bit_start) & ((1 << (bit_end - bit_start + 1)) - 1)
                except ValueError:
                    return 0
            return 0

        key = out_reg.upper()
        off = self._offsets.get(key)
        if off is None:
            return 0
        sz = self._sizes.get(key, 8)

        # Fast path: direct hit (most registers, e.g. RAX, RBX, ZF, CF)
        if off in frame.regs:
            val = _mask(frame.regs[off], sz)
            mask = (1 << (bit_end - bit_start + 1)) - 1
            return (val >> bit_start) & mask

        # Sub-register fallback (e.g. AH at offset 1, BH at offset 25).
        # AH lives at byte 1 inside the 8-byte RAX slot stored at offset 0.
        # Find any stored entry whose byte range contains [off .. off+sz-1].
        for k, k_sz in frame.reg_sizes.items():
            if k <= off < k + k_sz:
                # Extract the relevant bytes from the parent entry
                byte_offset = off - k
                val = _mask(frame.regs[k] >> (byte_offset * 8), sz)
                mask = (1 << (bit_end - bit_start + 1)) - 1
                return (val >> bit_start) & mask

        return 0

    # ------------------------------------------------------------------
    # Public evaluate interface
    # ------------------------------------------------------------------

    def evaluate_concrete(self, cell: Any, flat_inputs: dict[str, int]) -> int:
        """
        Single concrete execution.

        Used by CellSimulator.evaluate_concrete() (path: ast.pyx calls
        simulator.evaluate_concrete(cell, machine_state) → we flatten and
        call here).

        Raises PCodeFallbackNeeded on unsupported opcodes so that
        CellSimulator can transparently re-run the same cell via Unicorn.
        """
        translation = _get_translation(self.arch, bytes.fromhex(cell.instruction))
        self._load(self._frame_a, flat_inputs)
        self._frame_a.execute_all(translation.ops)
        self.native_calls += 1
        return self._read_output(
            self._frame_a,
            cell.out_reg,
            cell.out_bit_start,
            cell.out_bit_end,
        )

    def evaluate_differential(
        self,
        cell: Any,
        or_inputs: dict[str, int],
        and_inputs: dict[str, int],
    ) -> int:
        """
        Two-run differential: f(V|T)[out] XOR f(V&~T)[out].

        Called directly from InstructionCellExpr.evaluate() in ast.pyx
        (use_unicorn=False path), bypassing _build_machine_state and
        MachineState entirely.

        Raises PCodeFallbackNeeded on unsupported opcodes.
        """
        translation = _get_translation(self.arch, bytes.fromhex(cell.instruction))
        ops = translation.ops

        self._load(self._frame_a, or_inputs)
        self._frame_a.execute_all(ops)
        out_or = self._read_output(
            self._frame_a,
            cell.out_reg,
            cell.out_bit_start,
            cell.out_bit_end,
        )

        self._load(self._frame_b, and_inputs)
        self._frame_b.execute_all(ops)
        out_and = self._read_output(
            self._frame_b,
            cell.out_reg,
            cell.out_bit_start,
            cell.out_bit_end,
        )

        self.native_calls += 1
        return out_or ^ out_and

    @property
    def fallback_rate(self) -> float:
        total = self.native_calls + self.fallback_calls
        return self.fallback_calls / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            'native_calls': self.native_calls,
            'fallback_calls': self.fallback_calls,
            'fallback_rate': self.fallback_rate,
        }
