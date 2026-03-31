from __future__ import annotations

import pypcode

_pypcode_contexts: dict[str, pypcode.Context] = {}


def get_context(arch: str) -> pypcode.Context:
    if arch not in _pypcode_contexts:
        # Map our architectures to pypcode architectures
        arch_map = {
            'X86': 'x86:LE:32:default',
            'AMD64': 'x86:LE:64:default',
            'ARM64': 'AARCH64:LE:64:v8A',
        }

        if arch not in arch_map:
            raise ValueError(f'Unsupported architecture for lifting: {arch}')

        _pypcode_contexts[arch] = pypcode.Context(arch_map[arch])

    return _pypcode_contexts[arch]
