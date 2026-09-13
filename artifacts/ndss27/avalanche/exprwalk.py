"""Generic, engine-version-stable traversal of taint expression trees.

The previous harness reimplemented `Expr.evaluate` so it could re-evaluate a tree
with the avalanche nodes forced to zero.  That copy had to be updated for every
node type the engine ever gained, and when it was not it degraded SILENTLY: an
unrecognised node was treated as an opaque leaf with "no avalanche inside", so
recursion stopped and any avalanche beneath it vanished.  Measured, it was blind
to 7 of the engine's 11 node types, including `InstructionCellExpr`, whose
`inputs` dict holds whole sub-expressions.

That bias is one-directional.  Under-counting avalanche makes bit precision look
BETTER than it is, i.e. the instrument fails in the direction that flatters the
claim it exists to test.

This module never evaluates anything.  It only answers "what are this node's
children", by reflection over the public attributes every `Expr` exposes, so a
node type the engine adds tomorrow is traversed today.  Neutralising an avalanche
node is then a temporary in-place swap, and the ENGINE's own `evaluate()` runs on
the modified tree, so there is exactly one implementation of taint semantics.
"""
from __future__ import annotations

from typing import Callable, Iterator

from microtaint.instrumentation import ast as _ast

Expr = _ast.Expr
Constant = _ast.Constant
EvalContext = _ast.EvalContext

#: Node types that OVER-APPROXIMATE, by the engine's own documentation.
#:
#: This used to be just the two avalanche nodes, which is why Table 6 drifted:
#: two commits (4731f08, f3f52bb) moved multiply and variable-shift OUT of the
#: blanket fallback into dedicated terms, so their over-approximation stopped
#: being counted while `determine_category` still called them Avalanche.  The
#: honest accounting includes every node that admits to approximating:
#:
#:   AvalancheExpr             the conservative fallback itself
#:   FullMaskAvalancheExpr     the 1-bit flag soundness floor
#:   VariableMultiplyTaintExpr "a sound fill, not an avalanche ... cannot be
#:                             made exact cheaply"  (its own docstring)
#:
#: These claim EXACT and are therefore NOT counted -- but the claim is checked
#: against ground truth rather than trusted (see the over-taint cross-check):
#:   VariableShiftTaintExpr, VariableBitSelectTaintExpr, ComparisonTaintExpr,
#:   EqualityTaintExpr, SignedOverflowTaintExpr
APPROXIMATING_TYPES = frozenset({
    'AvalancheExpr', 'FullMaskAvalancheExpr', 'VariableMultiplyTaintExpr',
})

#: Every node type must be on exactly one side of the line, so that a new one
#: cannot silently join the "not counted" side by default.
CLAIMS_EXACT_TYPES = frozenset({
    'BinaryExpr', 'UnaryExpr', 'Expr', 'InstructionCellExpr',
    'ComparisonTaintExpr', 'EqualityTaintExpr', 'SignedOverflowTaintExpr',
    'VariableShiftTaintExpr', 'VariableBitSelectTaintExpr',
})

#: Kept for callers that specifically want the old, narrower question.
AVALANCHE_TYPES = frozenset({'AvalancheExpr', 'FullMaskAvalancheExpr'})

#: Every expression node type the engine defined when this harness was written.
#: The gate below fails on anything new, because a new node type is exactly the
#: event that silently broke the previous harness.  Reflection will almost
#: certainly traverse it correctly; the gate exists so a human confirms that
#: rather than assuming it.
KNOWN_EXPR_TYPES = frozenset({
    'AvalancheExpr', 'BinaryExpr', 'ComparisonTaintExpr', 'EqualityTaintExpr',
    'Expr', 'FullMaskAvalancheExpr', 'InstructionCellExpr',
    'SignedOverflowTaintExpr', 'UnaryExpr', 'VariableBitSelectTaintExpr',
    'VariableMultiplyTaintExpr', 'VariableShiftTaintExpr',
})


class EngineDrift(RuntimeError):
    """The engine grew an expression node type this harness has not seen."""


def check_engine_node_types() -> None:
    """Fail loudly if the engine's node-type set has changed.

    This is the gate the old harness lacked.  It turns "the number quietly moved"
    into "the experiment refuses to run".
    """
    live = {n for n in dir(_ast)
            if n.endswith('Expr') and not n.startswith('__pyx_unpickle')}
    unclassified = live - APPROXIMATING_TYPES - CLAIMS_EXACT_TYPES
    if unclassified:
        raise EngineDrift(
            f'node type(s) with no exactness verdict: {sorted(unclassified)}.\n'
            'Every node must be declared APPROXIMATING or CLAIMS_EXACT.  Defaulting '
            'to "exact" is how multiply left the avalanche accounting unnoticed and '
            'made the engine look more precise than it is.')
    new = live - KNOWN_EXPR_TYPES
    if new:
        raise EngineDrift(
            f'unrecognised expression node type(s): {sorted(new)}.\n'
            'Traversal is reflective so these are probably handled already, but '
            'confirm they expose their sub-expressions as public attributes (or '
            'inside a list/dict/tuple) and then add them to KNOWN_EXPR_TYPES.\n'
            'Do NOT skip this: a node type that hides children somewhere '
            'reflection cannot see would under-count avalanche, which biases the '
            'result in favour of the paper.')


def _public_attrs(e: object) -> list[str]:
    return [a for a in dir(e)
            if not a.startswith('_') and not callable(getattr(e, a, None))]


def _attr_setter(obj: object, name: str) -> Callable[[Expr], None]:
    """Replace `obj.name`."""
    def set_it(new: Expr) -> None:
        setattr(obj, name, new)
    return set_it


def _list_setter(items: list[Expr], index: int) -> Callable[[Expr], None]:
    """Replace `items[index]`.

    A named factory rather than an immediately-invoked lambda: both bind the
    loop variable correctly, but only one of them can be read at a glance, and
    the lambda form trips two linters that cannot see through it.
    """
    def set_it(new: Expr) -> None:
        items[index] = new
    return set_it


def _dict_setter(items: dict[str, Expr], key: str) -> Callable[[Expr], None]:
    """Replace `items[key]`."""
    def set_it(new: Expr) -> None:
        items[key] = new
    return set_it


def child_slots(e: object) -> list[tuple[Callable[[Expr], None], Expr]]:
    """[(replace_fn, child)] for every Expr one level below `e`.

    Reflective on purpose: anything that IS an Expr, or sits in a list/dict/tuple
    attribute, counts as a child.  A tuple slot raises, because a child we cannot
    replace is a child we cannot neutralise, and silently skipping it is how the
    old harness lost bits.
    """
    out: list[tuple[Callable[[Expr], None], Expr]] = []
    for name in _public_attrs(e):
        val = getattr(e, name, None)
        if isinstance(val, Expr):
            out.append((_attr_setter(e, name), val))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, Expr):
                    out.append((_list_setter(val, i), item))
        elif isinstance(val, dict):
            for k, item in val.items():
                if isinstance(item, Expr):
                    out.append((_dict_setter(val, k), item))
        elif isinstance(val, tuple) and any(isinstance(x, Expr) for x in val):
            raise EngineDrift(
                f'{type(e).__name__}.{name} holds sub-expressions in an immutable '
                'tuple, so they cannot be neutralised in place.  Handle this node '
                'type explicitly rather than letting its avalanche go uncounted.')
    return out


def walk(e: object) -> Iterator[Expr]:
    """Every node in the tree, `e` first."""
    stack: list[Expr] = [e]  # type: ignore[list-item]  # the root may be any node
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:      # trees can share sub-expressions
            continue
        seen.add(id(node))
        yield node
        stack.extend(child for _, child in child_slots(node))


def contains_approximation(e: object, types: frozenset[str] = APPROXIMATING_TYPES) -> bool:
    return any(type(n).__name__ in types for n in walk(e))


def approximating_nodes(e: object) -> list[str]:
    """Which approximating node types this tree actually uses."""
    return sorted({type(n).__name__ for n in walk(e)
                   if type(n).__name__ in APPROXIMATING_TYPES})


def contains_avalanche(e: object) -> bool:
    return any(type(n).__name__ in AVALANCHE_TYPES for n in walk(e))


def count_avalanche_nodes(e: object) -> int:
    return sum(1 for n in walk(e) if type(n).__name__ in AVALANCHE_TYPES)


class neutralised:
    """Context manager: every avalanche node reads as 0 for the duration.

    In-place swap with guaranteed restore, so the ENGINE evaluates the tree.  The
    circuits are cached and shared, so the restore is not optional -- it runs on
    the exception path too.
    """

    def __init__(self, root: Expr, types: frozenset[str] = APPROXIMATING_TYPES) -> None:
        self.root = root
        self.types = types
        self._undo: list[tuple[Callable[[Expr], None], Expr]] = []

    def __enter__(self) -> Expr:
        for node in list(walk(self.root)):
            for replace, child in child_slots(node):
                if type(child).__name__ in self.types:
                    size = getattr(child, 'size_bits', None) or 8
                    replace(Constant(0, max(1, (int(size) + 7) // 8)))
                    self._undo.append((replace, child))
        return self.root

    def __exit__(self, *exc: object) -> None:
        for replace, original in reversed(self._undo):
            replace(original)
        self._undo.clear()


def evaluate_precise(e: Expr, ctx: EvalContext,
                     types: frozenset[str] = APPROXIMATING_TYPES) -> int:
    """Evaluate `e` with every approximating node forced to 0.

    The root needs its own case.  `neutralised` rewrites a node's CHILDREN, so a
    tree whose ROOT is the approximation has nothing to rewrite and evaluates
    unchanged -- which reports 0% approximation for exactly the instructions that
    are entirely approximation.  `imul rax, rbx` is the whole of its own
    `VariableMultiplyTaintExpr`, so it read 0% until this was handled; the
    calibration gate is what caught it.
    """
    if type(e).__name__ in types:
        return 0
    with neutralised(e, types):
        return int(e.evaluate(ctx))


def root_is_approximation(e: object,
                          types: frozenset[str] = APPROXIMATING_TYPES) -> bool:
    return type(e).__name__ in types
