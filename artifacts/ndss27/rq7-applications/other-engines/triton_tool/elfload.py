#!/usr/bin/env python3
"""elfload.py -- minimal ELF64 program-header parser.

No dependency on `lief` (not installed in .venv_triton and no pip there).
We only need the PT_LOAD segments so we can copy the file image into
Triton's flat concrete memory.  The harness is `-static -no-pie`, so the
p_vaddr fields are the runtime addresses verbatim.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

PT_LOAD = 1


@dataclass
class Segment:
    vaddr: int
    filesz: int
    memsz: int
    flags: int
    data: bytes  # exactly filesz bytes from the file image


def load_segments(path: Path) -> tuple[int, list[Segment]]:
    """Return (entry_point, [PT_LOAD segments]) for an ELF64 little-endian file."""
    blob = Path(path).read_bytes()
    if blob[:4] != b"\x7fELF":
        raise ValueError(f"{path}: not an ELF file")
    if blob[4] != 2:  # EI_CLASS == ELFCLASS64
        raise ValueError(f"{path}: not ELFCLASS64")

    e_entry = struct.unpack_from("<Q", blob, 0x18)[0]
    e_phoff = struct.unpack_from("<Q", blob, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", blob, 0x36)[0]
    e_phnum = struct.unpack_from("<H", blob, 0x38)[0]

    segs: list[Segment] = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_flags = struct.unpack_from("<II", blob, off)
        p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz = struct.unpack_from(
            "<QQQQQ", blob, off + 8
        )
        if p_type != PT_LOAD:
            continue
        data = blob[p_offset:p_offset + p_filesz]
        segs.append(Segment(vaddr=p_vaddr, filesz=p_filesz, memsz=p_memsz,
                            flags=p_flags, data=data))
    return e_entry, segs
