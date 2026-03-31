from __future__ import annotations

from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


def main() -> None:
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
    ]
    # MOV RAX, RBX -> \x48\x89\xd8
    byte_string = b'\x48\x89\xd8'

    rule = generate_static_rule(arch, byte_string, state_format)

    assert len(rule.assignments) > 0, 'No assignments generated!'
    assignment = rule.assignments[0]

    assert assignment.target.name == 'RAX', f'Expected RAX, got {assignment.target.name}'
    deps = [dep.name for dep in assignment.dependencies]
    assert 'RBX' in deps, f'Expected RBX dependency, got {deps}'

    print('Smoke test passed successfully!')


if __name__ == '__main__':
    main()
