"""SLEIGH contexts, made once per architecture and kept.

Building one costs about 150 ms (it loads the architecture's .sla), so they are
cached for the life of the process and every caller shares them.
"""
from __future__ import annotations

import pypcode

_pypcode_contexts: dict[str, pypcode.Context] = {}


def clear_contexts() -> None:
    """Drop every cached SLEIGH context.

    Two reasons to want this, and one reason it is NOT done automatically.

    A context holds a large amount of nanobind-managed state -- measured, 1441
    instances for x86-64 alone -- and nanobind reports it at interpreter
    shutdown as leaked.  It is not a leak, it is a cache that was never
    emptied, but the message is indistinguishable from a real refcount bug and
    drowns one out.  A long-running process that has finished with an
    architecture may also simply want the memory back.

    It is not registered with `atexit` here on purpose.  Running those
    destructors costs about 16 ms for one architecture and 57 ms for all
    seven, and a process that is about to exit would otherwise have that done
    for free when the kernel reclaims its address space.  For anything that
    runs the engine once -- a fuzzer, a concolic search -- that is pure loss
    repeated per execution, so the choice belongs to the caller.  `tests/`
    registers it, because there the noise costs more than the milliseconds.

    The next `get_context` rebuilds, at the full ~150 ms.
    """
    _pypcode_contexts.clear()


def get_context(arch: str) -> pypcode.Context:
    if arch not in _pypcode_contexts:
        # Map our architectures to pypcode architectures
        arch_map = {
            'X86': 'x86:LE:32:default',
            'AMD64': 'x86:LE:64:default',
            'ARM64': 'AARCH64:LE:64:v8A',
            'RISCV64': 'RISCV:LE:64:default',
            'MIPS64BE': 'MIPS:BE:64:default',
            'PPC32BE': 'PowerPC:BE:32:default',
            'SPARC32BE': 'sparc:BE:32:default',
        }

        if arch not in arch_map:
            raise ValueError(f'Unsupported architecture for lifting: {arch}')

        _pypcode_contexts[arch] = pypcode.Context(arch_map[arch])

    return _pypcode_contexts[arch]
