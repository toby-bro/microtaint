"""Every `.pyi` in `microtaint/` must agree with the module it describes.

The compiled modules are Cython and hand-written C, so their `.pyi` files are
maintained by hand, and a hand-maintained stub drifts silently.  Both
directions of drift have been found here, and both cost real time:

  * The stub claims something the runtime does not have.  `shadow.pyi`
    declared `taint_pages` and `state_pages` as visible dicts (they are
    `cdef`, so Python cannot see them at all) and `hook_core.pyi` declared
    `LiveMemReader.uc_handle` and two more C-level addresses the same way.
    A type checker then green-lights an attribute read that raises.
  * The runtime has something the stub does not.  `InstructionHook`'s
    `express_done`, `fast_done`, `instr_total`, the `fb_*` reasons and
    `TaintAssignment.is_mem_target` were all missing.  Here the checker
    rejects correct code, which is how a stub teaches people to write
    `# type: ignore` over a fact that is simply undeclared.

The second kind is the one that recurs, because adding a `cdef public` field
is a one-line change in a `.pyx` and nothing points at the stub.

Both tests count what they compared and fail on zero: a stub walk that
silently matched nothing would agree with anything, which is the failure mode
this suite has been bitten by before.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent / 'microtaint'

#: A stub class whose name starts with `_` is deliberate type-checking fiction
#: -- `_Capsule` names an opaque PyCapsule handle so signatures can talk about
#: it -- and has no runtime counterpart by design.
def _is_fiction(name: str) -> bool:
    return name.startswith('_')


def _stubs() -> list[tuple[pathlib.Path, types.ModuleType, ast.Module]]:
    out = []
    for pyi in sorted(ROOT.rglob('*.pyi')):
        mod_name = str(pyi.relative_to(ROOT.parent).with_suffix('')).replace('/', '.')
        out.append((pyi, importlib.import_module(mod_name), ast.parse(pyi.read_text())))
    return out


def _declared(node: ast.ClassDef) -> set[str]:
    """Names a stub class declares: annotations, assignments and defs."""
    names: set[str] = set()
    for st in node.body:
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names.add(st.target.id)
        elif isinstance(st, ast.Assign):
            names.update(t.id for t in st.targets if isinstance(t, ast.Name))
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(st.name)
    return names


def _classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in tree.body
            if isinstance(n, ast.ClassDef) and not _is_fiction(n.name)]


def test_stubs_declare_nothing_the_runtime_lacks() -> None:
    """A name in a stub must exist on the object it describes."""
    missing, compared = [], 0
    for pyi, mod, tree in _stubs():
        for node in _classes(tree):
            cls = getattr(mod, node.name, None)
            if cls is None:
                missing.append(f'{pyi}: class {node.name} does not exist')
                continue
            for name in sorted(_declared(node)):
                compared += 1
                if not hasattr(cls, name):
                    missing.append(f'{pyi}: {node.name}.{name} does not exist '
                                   f'(a `cdef` field is invisible to Python; '
                                   f'only `cdef public` reaches it)')
    assert compared > 20, f'compared only {compared} names: the walk found nothing'
    assert not missing, 'stub declares what the runtime does not have:\n  ' + \
                        '\n  '.join(missing)


def test_stubs_declare_everything_the_runtime_exposes() -> None:
    """A public runtime attribute must be declared in the stub.

    This is the direction that keeps drifting: `cdef public` is a one-line
    addition in a `.pyx` and nothing else points at the `.pyi`.
    """
    undeclared, compared = [], 0
    for pyi, mod, tree in _stubs():
        for node in _classes(tree):
            cls = getattr(mod, node.name, None)
            if cls is None:
                continue                      # the other test reports this
            declared = _declared(node)
            for name in sorted(vars(cls)):
                if name.startswith('_'):      # private, and Cython's own noise
                    continue
                compared += 1
                if name not in declared:
                    undeclared.append(f'{pyi}: {node.name}.{name} is public at '
                                      f'runtime but the stub does not declare it')
    assert compared > 20, f'compared only {compared} names: the walk found nothing'
    assert not undeclared, 'runtime exposes what the stub does not declare:\n  ' + \
                           '\n  '.join(undeclared)


def test_every_stub_was_actually_walked() -> None:
    """The two tests above are only worth as much as the file list they walk."""
    found = _stubs()
    assert len(found) >= 8, f'only {len(found)} stubs found under {ROOT}'
    for pyi, mod, tree in found:
        has_fn = any(isinstance(n, ast.FunctionDef) for n in tree.body)
        assert _classes(tree) or has_fn, \
            f'{pyi}: parsed but declares neither a class nor a function'
        assert mod is not None
