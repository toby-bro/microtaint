"""Guard: the compiled C circuit (circuit_c) must handle EVERY Expr type.

The C bytecode compiler (circuit_c.compile_expr) is the production fast path. Any
Expr class it cannot emit forces that assignment to recross the Python boundary
(the AST evaluator) mid-evaluate -- slow, and it defeats the C-side frame cache.

This test enumerates every Expr subclass in microtaint.instrumentation.ast and
asserts the C compiler declares support for it (circuit_c.supported_expr_types()).
So if someone adds a new Expr and forgets to teach compile_expr about it, this
fails -- protecting the "never leave C" invariant of the fast path.

The Cython AST walker (LogicCircuit.evaluate's fallback) needs no such guard: it
evaluates any Expr via polymorphic .evaluate(), so it inherently supports all.
"""

from __future__ import annotations

import microtaint.instrumentation.ast as ast
from microtaint.instrumentation.cell_c import circuit_c


def _all_expr_subclasses() -> set[str]:
    out: set[str] = set()

    def rec(cls: type) -> None:
        for s in cls.__subclasses__():
            out.add(s.__name__)
            rec(s)

    rec(ast.Expr)
    return out


def test_c_compiler_handles_every_expr_type() -> None:
    all_exprs = _all_expr_subclasses()
    supported = set(circuit_c.supported_expr_types())
    missing = sorted(all_exprs - supported)
    assert not missing, (
        f'circuit_c.compile_expr does not compile these {len(missing)} Expr '
        'type(s), so assignments using them fall back to the Python evaluator '
        'inside the C fast path (recrossing the boundary):\n  '
        + '\n  '.join(missing)
        + '\nAdd each to compile_expr (a new opcode) AND to SUPPORTED_EXPR_TYPES '
        'in circuit_c.c.'
    )


def test_supported_list_has_no_unknown_types() -> None:
    """The C supported list must not name a type that isn't an Expr subclass
    (catches typos / stale entries that would make the guard vacuously pass)."""
    all_exprs = _all_expr_subclasses()
    supported = set(circuit_c.supported_expr_types())
    stale = sorted(supported - all_exprs)
    assert not stale, f'circuit_c.supported_expr_types() names non-Expr types: {stale}'
