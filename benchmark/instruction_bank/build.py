"""Generator for the unified instruction bank (instructions.jsonl).

Materializes the multi-ISA source corpus -- the ~1,500 verified-sound
instruction forms accumulated by the soundness campaign, historically living in
the deletable scratchpad module ``corpora.py`` -- into a single durable,
keystone-free data file the engine repo owns. Every form is assembled to raw
bytes here (once), so downstream consumers (the perf ratchet, benchmark scripts)
load the bank with NO keystone dependency and NO per-run assembly.

Each form is validated by actually building its static rule with the canonical
per-ISA register format from ``instruction_bank/__init__.py``; a form whose rule
cannot be generated (e.g. an opcode the lifter does not categorise yet) is
skipped and reported rather than silently emitted broken.

Run (from the engine repo)::

    uv run --project . python benchmark/instruction_bank/build.py \
        [--corpora /path/to/scratchpad/corpora.py] [--cell-corpus]

Sources folded in:
  * corpora.py            -- the 5-ISA campaign corpus (asm; riscv is hex).
  * tests/test_cell_benchmark.py::CORPUS  -- 98 AMD64 forms carrying a rich
    value/taint oracle (--cell-corpus; oracle kept in the entry's "oracle" key).

Output line schema (instructions.jsonl), one JSON object per instruction::

    {"isa": "AMD64", "label": "adc rax, rbx", "asm": "adc rax, rbx",
     "bytes": "4811d8", "srcs": ["RAX","RBX","CF"], "constraints": {},
     "categories": ["arith","flag"], "source": "corpora",
     "oracle": {...optional...}}

``asm`` is null for RISC-V (hand-assembled, no textual form stored).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENGINE = _HERE.parent.parent
sys.path.insert(0, str(_ENGINE))

from microtaint.sleigh.engine import generate_static_rule  # noqa: E402
from microtaint.types import Architecture  # noqa: E402

# Loader constants: canonical per-ISA formats + keystone params live next door.
from __init__ import ISA_FORMATS, isa_registers, keystone_for  # type: ignore  # noqa: E402

DEFAULT_CORPORA = Path('/home/jns/Documents/Telecom/PRIM/benchmark/corpora.py')
OUT = _HERE / 'instructions.jsonl'

# corpora.py key -> engine Architecture name
_ISA_KEY = {'x86_64': 'AMD64', 'arm64': 'ARM64', 'mips': 'MIPS64BE',
            'ppc': 'PPC32BE', 'riscv': 'RISCV64'}


# ---------------------------------------------------------------------------
# Category tagging (best-effort, from the mnemonic; multiple tags allowed).
# ---------------------------------------------------------------------------

# Each category's keys are matched against the MNEMONIC (first token) by prefix,
# so 'lh' can't match inside 'mulh'. Memory is detected separately via a '['
# operand or a load/store mnemonic.
_CAT = [
    ('ext', ('movsx', 'movzx', 'sxt', 'uxt', 'sext', 'zext')),
    ('move', ('mov', 'mvn', 'mv', 'li', 'lui', 'auipc', 'mr', 'mflr', 'cpy', 'copy')),
    ('rotate', ('rol', 'ror', 'rcl', 'rcr', 'rlw', 'extr')),
    ('shift', ('shl', 'shr', 'sar', 'sal', 'shld', 'shrd', 'lsl', 'lsr', 'asr',
               'sll', 'srl', 'sra', 'slw', 'srw')),
    ('bitfield', ('ubfx', 'sbfx', 'bfi', 'bfxil', 'bfm', 'ubfm', 'sbfm', 'bfc')),
    ('bitscan', ('bsf', 'bsr', 'clz', 'cls', 'cntlz', 'cnttz', 'popcnt',
                 'lzcnt', 'tzcnt', 'ctz', 'rbit', 'rev', 'bswap', 'bt')),
    ('mul', ('mul', 'imul', 'umul', 'smul', 'madd', 'msub', 'mult', 'maddwd')),
    ('div', ('div', 'idiv', 'sdiv', 'udiv', 'rem', 'mod')),
    ('cmp', ('cmp', 'cmn', 'test', 'tst', 'slt', 'ccmp')),
    ('flag', ('adc', 'sbb', 'sbc', 'cset', 'csel', 'csinc', 'csinv', 'csneg',
              'set', 'cmov', 'adde', 'addc', 'subfe', 'subfc')),
    ('logic', ('and', 'or', 'orr', 'xor', 'eor', 'not', 'nor', 'bic', 'andn',
               'orn', 'nand', 'nots')),
    ('arith', ('add', 'sub', 'neg', 'inc', 'dec', 'subf')),
]

_MEM_MNEM = frozenset((
    'push', 'pop', 'ldr', 'str', 'ldp', 'stp', 'ldur', 'stur', 'lwz', 'stw',
    'lw', 'sw', 'ld', 'sd', 'lb', 'sb', 'lh', 'sh', 'lbu', 'lhu', 'lwu',
    'lbz', 'stb', 'lhz', 'sth', 'ldrb', 'strb', 'ldrh', 'strh'))


def categorize(asm: str) -> list[str]:
    a = asm.lower().strip()
    mnem = a.split()[0] if a else ''
    tags = []
    if '[' in a or ' ptr ' in a or mnem in _MEM_MNEM:
        tags.append('mem')
    for tag, keys in _CAT:
        if any(mnem == k or mnem.startswith(k) for k in keys):
            tags.append(tag)
    return tags or ['other']


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _load_corpora(path: Path):
    spec = importlib.util.spec_from_file_location('corpora', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.CORPORA


def _assemble(isa_key: str, code_spec: str) -> bytes:
    ks = keystone_for(isa_key)
    if ks is None:  # riscv: code_spec is hex machine code
        return bytes.fromhex(code_spec)
    enc, _ = ks.asm(code_spec, 0x1000)
    return bytes(enc)


def _cell_corpus_entries():
    """AMD64 forms from tests/test_cell_benchmark.py::CORPUS (hex + oracle)."""
    sys.path.insert(0, str(_ENGINE / 'tests'))
    try:
        from test_cell_benchmark import CORPUS  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f'  (skip --cell-corpus: {type(e).__name__} {e})')
        return
    for mnem, hexb, invals, intaint, check, expected, desc in CORPUS:
        yield {
            'isa': 'AMD64', 'label': mnem, 'asm': mnem, 'bytes': hexb.lower(),
            'srcs': sorted(intaint), 'constraints': {},
            'categories': categorize(mnem), 'source': 'cell_corpus',
            'oracle': {'values': {k: hex(v) for k, v in invals.items()},
                       'taint': {k: hex(v) for k, v in intaint.items()},
                       'check': check,
                       'expected': (hex(expected) if expected is not None else None),
                       'desc': desc},
        }


# ---------------------------------------------------------------------------
# SIMD / vector corpus. Representative SSE2 (xmm) + NEON (v) forms so the perf
# ratchet + benchmark scripts cover the wide-vector path (the scalar corpus has
# none). Each form is tagged with a distinct PROFILE group key ('AMD64_SIMD' /
# 'ARM64_SIMD') that carries vector lanes in its state format, WITHOUT touching
# the scalar corpus or its committed baselines; the real Architecture rides in
# the 'arch' field. The high per-instruction node/cell counts these show are the
# lane-splitting cost the future circuit_c wide path is meant to drive down.
# ---------------------------------------------------------------------------

_SIMD_CORPUS: dict[str, tuple[str, list[str]]] = {
    'AMD64_SIMD': ('x86_64', [
        'pxor xmm0, xmm1', 'pand xmm0, xmm1', 'por xmm0, xmm1',
        'paddd xmm0, xmm1', 'paddb xmm0, xmm1', 'psubd xmm0, xmm1',
        'punpcklbw xmm0, xmm1', 'pshufd xmm0, xmm1, 0x1b',
        'movaps xmm0, xmm1', 'movdqa xmm0, xmm1', 'psllq xmm0, 7',
        'pcmpeqd xmm0, xmm1', 'pmullw xmm0, xmm1', 'packuswb xmm0, xmm1',
    ]),
    'ARM64_SIMD': ('arm64', [
        'and v0.16b, v1.16b, v2.16b', 'orr v0.16b, v1.16b, v2.16b',
        'eor v0.16b, v1.16b, v2.16b', 'add v0.4s, v1.4s, v2.4s',
        'add v0.16b, v1.16b, v2.16b', 'sub v0.4s, v1.4s, v2.4s',
        'mul v0.4s, v1.4s, v2.4s', 'zip1 v0.16b, v1.16b, v2.16b',
        'mov v0.16b, v1.16b', 'shl v0.4s, v1.4s, #3',
    ]),
}

# SIMD profile group key -> real engine Architecture name.
_SIMD_ARCH = {'AMD64_SIMD': 'AMD64', 'ARM64_SIMD': 'ARM64'}


def _simd_entries():
    """Assemble the SIMD/vector corpus into bank entries (one per form)."""
    for profile, (ks_key, forms) in _SIMD_CORPUS.items():
        for asm in forms:
            try:
                code = _assemble(ks_key, asm)
            except Exception as ex:  # noqa: BLE001
                print(f'  (simd asm skip) {profile:11s} {asm!r}: '
                      f'{type(ex).__name__} {str(ex)[:50]}')
                continue
            yield {'isa': profile, 'arch': _SIMD_ARCH[profile], 'label': asm,
                   'asm': asm, 'bytes': code.hex(), 'srcs': [],
                   'constraints': {}, 'categories': [*categorize(asm), 'simd'],
                   'source': 'simd'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpora', type=Path, default=DEFAULT_CORPORA)
    ap.add_argument('--cell-corpus', action='store_true', default=True)
    ap.add_argument('--no-cell-corpus', dest='cell_corpus', action='store_false')
    args = ap.parse_args()

    entries: list[dict] = []
    built, skipped = 0, []
    seen: set[tuple[str, str]] = set()  # (isa, bytes) dedupe

    def emit(e: dict) -> None:
        nonlocal built
        key = (e['isa'], e['bytes'])
        if key in seen:
            return
        # validate: the static rule must actually build. The group key e['isa']
        # selects the register format (scalar or a SIMD profile); the real
        # Architecture is e['arch'] when present (SIMD), else the group key.
        arch = Architecture(e.get('arch', e['isa']))
        regs = list(isa_registers(e['isa']))
        try:
            generate_static_rule(arch, bytes.fromhex(e['bytes']), regs)
        except Exception as ex:  # noqa: BLE001
            skipped.append(f'{e["isa"]:9s} {e["label"]!r}: {type(ex).__name__} {str(ex)[:60]}')
            return
        seen.add(key)
        entries.append(e)
        built += 1

    if args.corpora.exists():
        corpora = _load_corpora(args.corpora)
        for isa_key, fn in corpora.items():
            arch_name = _ISA_KEY[isa_key]
            if arch_name not in ISA_FORMATS:
                continue
            for label, code_spec, srcs, cons in ((t + ({},) * (4 - len(t)))[:4]
                                                 for t in fn(isa_key)):
                try:
                    code = _assemble(isa_key, code_spec)
                except Exception as ex:  # noqa: BLE001
                    skipped.append(f'{arch_name:9s} {label!r}: asm {type(ex).__name__} {str(ex)[:50]}')
                    continue
                asm = None if isa_key == 'riscv' else code_spec
                # riscv code_spec is hex; its mnemonic lives in the label.
                emit({'isa': arch_name, 'label': label, 'asm': asm,
                      'bytes': code.hex(), 'srcs': list(srcs),
                      'constraints': dict(cons),
                      'categories': categorize(label if asm is None else code_spec),
                      'source': 'corpora'})
    else:
        print(f'  (corpora source not found at {args.corpora}; skipping)')

    if args.cell_corpus:
        for e in _cell_corpus_entries():
            emit(e)

    for e in _simd_entries():
        emit(e)

    entries.sort(key=lambda e: (e['isa'], e['label']))
    with OUT.open('w') as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + '\n')

    by_isa: dict[str, int] = {}
    for e in entries:
        by_isa[e['isa']] = by_isa.get(e['isa'], 0) + 1
    print(f'\nwrote {OUT}  ({built} instructions)')
    for isa in sorted(by_isa):
        print(f'  {isa:9s} {by_isa[isa]:4d}')
    if skipped:
        print(f'\nskipped {len(skipped)} form(s):')
        for s in skipped[:40]:
            print('  ' + s)
        if len(skipped) > 40:
            print(f'  ... and {len(skipped) - 40} more')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
