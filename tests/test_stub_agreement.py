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

def _is_fiction(node: ast.ClassDef) -> bool:
    """Is this stub class deliberately without a runtime counterpart?

    Two kinds are:

      * an underscore-prefixed name -- `_Capsule` gives signatures a word
        for an opaque PyCapsule handle;
      * a Protocol -- it describes a SHAPE that several unrelated classes
        satisfy, which is the entire point of writing one, so demanding a
        class of that name at runtime asks for the opposite.
    """
    if node.name.startswith('_'):
        return True
    return any((isinstance(b, ast.Name) and b.id == 'Protocol')
               or (isinstance(b, ast.Attribute) and b.attr == 'Protocol')
               for b in node.bases)


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
            if isinstance(n, ast.ClassDef) and not _is_fiction(n)]


def _module_names(tree: ast.Module) -> set[str]:
    """Module-level names a stub declares, ignoring what it imports.

    An import in a stub is there to spell a type, not to claim the module
    re-exports it.
    """
    names: set[str] = set()
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(st.name)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names.add(st.target.id)
        elif isinstance(st, ast.Assign):
            names.update(t.id for t in st.targets if isinstance(t, ast.Name))
    return names


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


def test_stubs_declare_the_module_level_names_the_runtime_exposes() -> None:
    """The same check, one level up: module functions, not just methods.

    This is the hole the first version of this test had.  It walked classes
    only, so eleven module-level entry points were missing from two stubs and
    nothing said so: hook_core's five C trampoline `_ptr` / `_ud` pairs and
    blockpath_c's five `hook_*` functions, which are the entire live
    UC_HOOK_BLOCK path.  wrapper.py carried a
    `# mypy: disable-error-code="attr-defined"` header, so the ten errors they
    caused were muted at the one place that would have reported them.
    """
    undeclared, compared = [], 0
    for pyi, mod, tree in _stubs():
        declared = _module_names(tree)
        public = getattr(mod, '__all__', None)
        live = (set(public) if public is not None
                else {n for n in vars(mod) if not n.startswith('_')})
        for name in sorted(live):
            obj = getattr(mod, name, None)
            # Only what the module DEFINES.  A compiled module's namespace also
            # holds whatever it imported, which the stub has no reason to
            # redeclare: the modules themselves (Cython leaves `ctypes`, `os`,
            # `functools`, `logging` sitting there), and anything whose
            # __module__ names somewhere else.
            if isinstance(obj, types.ModuleType):
                continue
            if getattr(obj, '__module__', mod.__name__) != mod.__name__:
                continue
            compared += 1
            if name not in declared:
                undeclared.append(f'{pyi}: module-level {name} is public at '
                                  f'runtime but the stub does not declare it')
    assert compared > 10, f'compared only {compared} names: the walk found nothing'
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
