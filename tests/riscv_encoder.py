"""
riscv_encoder.py
================
Minimal RV64I/M assembler covering exactly the instruction forms used in the
microtaint RISC-V test corpus.  Returns 4-byte little-endian instruction words.

Why a custom encoder
--------------------
- ``keystone-engine`` on PyPI ships without the RISC-V backend.
- ``riscv-assembler`` 1.2.1 has buggy SRAI/SRLI/SRLW handling and cannot encode
  branches, loads, stores, LUI/AUIPC, or system instructions.
- Hand-encoding from the RISC-V ISA manual (User-Level ISA v2.2 / Unprivileged
  v20191213) is straightforward and lets us pin every test byte deterministically.

This module is intentionally narrow: it covers the R/I/S/B/U/J formats and the
specific opcodes the corpus uses.  Anything outside its remit raises ValueError.
"""

from __future__ import annotations

# ABI-name → register number (x0 .. x31).  Matches the RISC-V calling
# convention (Volume I, ABI Appendix).
_ABI_REGS: dict[str, int] = {
    'zero': 0,
    'ra': 1,
    'sp': 2,
    'gp': 3,
    'tp': 4,
    't0': 5,
    't1': 6,
    't2': 7,
    's0': 8,
    'fp': 8,
    's1': 9,
    'a0': 10,
    'a1': 11,
    'a2': 12,
    'a3': 13,
    'a4': 14,
    'a5': 15,
    'a6': 16,
    'a7': 17,
    's2': 18,
    's3': 19,
    's4': 20,
    's5': 21,
    's6': 22,
    's7': 23,
    's8': 24,
    's9': 25,
    's10': 26,
    's11': 27,
    't3': 28,
    't4': 29,
    't5': 30,
    't6': 31,
}
# Numeric x0 .. x31
for _i in range(32):
    _ABI_REGS[f'x{_i}'] = _i


def _r(name: str) -> int:
    """Resolve a register name (case-insensitive) to its 5-bit number."""
    n = _ABI_REGS.get(name.lower())
    if n is None:
        raise ValueError(f'unknown register: {name!r}')
    return n


def _imm(val: int, bits: int) -> int:
    """Validate immediate fits in `bits` (signed) and return its bit representation."""
    lo = -(1 << (bits - 1))
    hi = 1 << (bits - 1)  # exclusive
    hi_unsigned = 1 << bits
    if not (lo <= val < hi_unsigned):
        raise ValueError(
            f'immediate {val} does not fit in {bits} bits (range {lo}..{hi-1} signed or 0..{hi_unsigned-1} unsigned)',
        )
    return val & ((1 << bits) - 1)


def _word_to_le(word: int) -> bytes:
    """Convert a 32-bit instruction word to 4 little-endian bytes."""
    return (word & 0xFFFFFFFF).to_bytes(4, 'little')


# ---------------------------------------------------------------------------
# Format encoders
# ---------------------------------------------------------------------------


def _R(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _I(imm12: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _S(imm12: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    imm = imm12 & 0xFFF
    imm_hi = (imm >> 5) & 0x7F
    imm_lo = imm & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | opcode


def _B(imm13: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    """B-type immediate is 13 bits, bit 0 always 0 (2-byte aligned)."""
    if imm13 & 1:
        raise ValueError(f'B-type immediate {imm13} is not 2-byte aligned')
    imm = imm13 & 0x1FFE  # 13-bit field, low bit is implicit zero
    bit12 = (imm >> 12) & 1
    bit11 = (imm >> 11) & 1
    bits10_5 = (imm >> 5) & 0x3F
    bits4_1 = (imm >> 1) & 0xF
    high = (bit12 << 6) | bits10_5
    low = (bits4_1 << 1) | bit11
    return (high << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (low << 7) | opcode


def _U(imm20: int, rd: int, opcode: int) -> int:
    """U-type: imm[31:12] in upper 20 bits."""
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def _J(imm21: int, rd: int, opcode: int) -> int:
    """J-type immediate is 21 bits, bit 0 always 0."""
    if imm21 & 1:
        raise ValueError(f'J-type immediate {imm21} is not 2-byte aligned')
    imm = imm21 & 0x1FFFFE
    bit20 = (imm >> 20) & 1
    bits10_1 = (imm >> 1) & 0x3FF
    bit11 = (imm >> 11) & 1
    bits19_12 = (imm >> 12) & 0xFF
    field = (bit20 << 19) | (bits10_1 << 9) | (bit11 << 8) | bits19_12
    return (field << 12) | (rd << 7) | opcode


# ---------------------------------------------------------------------------
# Public encode() entry point
# ---------------------------------------------------------------------------


def encode(asm_line: str) -> bytes:  # noqa: C901
    """
    Encode one RISC-V assembly line and return its 4-byte LE bytes.

    Operand forms accepted:
      'add t0, t1, t2'   or   'add t0 t1 t2'         (R-type)
      'addi t0, t1, 100'                              (I-type imm)
      'slli t0, t1, 4'                                (I-type shamt)
      'lw   t0, 8(t1)'                                (I-type load)
      'sw   t2, 8(t1)'                                (S-type store)
      'beq  t1, t2, 16'                               (B-type)
      'lui  t0, 0x12345'                              (U-type)
      'jal  t0, 8'                                    (J-type)
    """
    line = asm_line.strip()
    if not line:
        raise ValueError('empty asm line')

    # Tokenise by whitespace and commas
    tokens = [t for t in line.replace(',', ' ').split() if t]
    if not tokens:
        raise ValueError(f'empty asm line: {asm_line!r}')

    mnem = tokens[0].lower()
    args = tokens[1:]

    # ---- R-type ALU (RV64I + RV64M) ----
    R_TYPE = {
        # mnem:        (funct7, funct3, opcode)
        'add': (0x00, 0b000, 0b0110011),
        'sub': (0x20, 0b000, 0b0110011),
        'sll': (0x00, 0b001, 0b0110011),
        'slt': (0x00, 0b010, 0b0110011),
        'sltu': (0x00, 0b011, 0b0110011),
        'xor': (0x00, 0b100, 0b0110011),
        'srl': (0x00, 0b101, 0b0110011),
        'sra': (0x20, 0b101, 0b0110011),
        'or': (0x00, 0b110, 0b0110011),
        'and': (0x00, 0b111, 0b0110011),
        'mul': (0x01, 0b000, 0b0110011),
        'mulh': (0x01, 0b001, 0b0110011),
        'mulhsu': (0x01, 0b010, 0b0110011),
        'mulhu': (0x01, 0b011, 0b0110011),
        'div': (0x01, 0b100, 0b0110011),
        'divu': (0x01, 0b101, 0b0110011),
        'rem': (0x01, 0b110, 0b0110011),
        'remu': (0x01, 0b111, 0b0110011),
        # 32-bit RV64 word-ops (.W suffix)
        'addw': (0x00, 0b000, 0b0111011),
        'subw': (0x20, 0b000, 0b0111011),
        'sllw': (0x00, 0b001, 0b0111011),
        'srlw': (0x00, 0b101, 0b0111011),
        'sraw': (0x20, 0b101, 0b0111011),
        'mulw': (0x01, 0b000, 0b0111011),
        'divw': (0x01, 0b100, 0b0111011),
        'divuw': (0x01, 0b101, 0b0111011),
        'remw': (0x01, 0b110, 0b0111011),
        'remuw': (0x01, 0b111, 0b0111011),
    }
    if mnem in R_TYPE:
        if len(args) != 3:
            raise ValueError(f'{mnem}: expected 3 register operands, got {len(args)}')
        f7, f3, op = R_TYPE[mnem]
        rd, rs1, rs2 = _r(args[0]), _r(args[1]), _r(args[2])
        return _word_to_le(_R(f7, rs2, rs1, f3, rd, op))

    # ---- I-type ALU (immediate) ----
    I_TYPE_ALU = {
        'addi': (0b000, 0b0010011),
        'slti': (0b010, 0b0010011),
        'sltiu': (0b011, 0b0010011),
        'xori': (0b100, 0b0010011),
        'ori': (0b110, 0b0010011),
        'andi': (0b111, 0b0010011),
        'addiw': (0b000, 0b0011011),
    }
    if mnem in I_TYPE_ALU:
        if len(args) != 3:
            raise ValueError(f'{mnem}: expected rd, rs1, imm')
        f3, op = I_TYPE_ALU[mnem]
        rd, rs1, imm = _r(args[0]), _r(args[1]), int(args[2], 0)
        return _word_to_le(_I(_imm(imm, 12), rs1, f3, rd, op))

    # ---- I-type shift-immediate (shamt) ----
    # SLLI/SRLI/SRAI use 6-bit shamt on RV64; func7-equivalent encoded in upper bits.
    SHIFT_I = {
        # mnem:   (funct6, funct3, opcode)
        'slli': (0b000000, 0b001, 0b0010011),
        'srli': (0b000000, 0b101, 0b0010011),
        'srai': (0b010000, 0b101, 0b0010011),
        # 32-bit word shifts use 5-bit shamt + funct7
        'slliw': (0b0000000, 0b001, 0b0011011),
        'srliw': (0b0000000, 0b101, 0b0011011),
        'sraiw': (0b0100000, 0b101, 0b0011011),
    }
    if mnem in SHIFT_I:
        if len(args) != 3:
            raise ValueError(f'{mnem}: expected rd, rs1, shamt')
        f_top, f3, op = SHIFT_I[mnem]
        rd, rs1, shamt = _r(args[0]), _r(args[1]), int(args[2], 0)
        if mnem.endswith('w'):
            if not (0 <= shamt < 32):
                raise ValueError(f'{mnem}: shamt {shamt} out of range [0,32)')
            imm12 = (f_top << 5) | shamt
        else:
            if not (0 <= shamt < 64):
                raise ValueError(f'{mnem}: shamt {shamt} out of range [0,64)')
            # RV64: funct6 (6 bits) | shamt (6 bits) = 12-bit imm field
            imm12 = (f_top << 6) | shamt
        return _word_to_le(_I(imm12, rs1, f3, rd, op))

    # ---- I-type loads (offset-form: 'lw t0, 8(t1)') ----
    LOADS = {
        'lb': (0b000, 0b0000011),
        'lh': (0b001, 0b0000011),
        'lw': (0b010, 0b0000011),
        'ld': (0b011, 0b0000011),
        'lbu': (0b100, 0b0000011),
        'lhu': (0b101, 0b0000011),
        'lwu': (0b110, 0b0000011),
    }
    if mnem in LOADS:
        # Forms: 'lw rd, offset(rs1)'  or  'lw rd, offset, rs1'  (relaxed)
        rd, imm, rs1 = _parse_offset_form(mnem, args)
        f3, op = LOADS[mnem]
        return _word_to_le(_I(_imm(imm, 12), rs1, f3, rd, op))

    # ---- S-type stores ----
    STORES = {
        'sb': (0b000, 0b0100011),
        'sh': (0b001, 0b0100011),
        'sw': (0b010, 0b0100011),
        'sd': (0b011, 0b0100011),
    }
    if mnem in STORES:
        # 'sw rs2, offset(rs1)'
        rs2, imm, rs1 = _parse_offset_form(mnem, args)
        f3, op = STORES[mnem]
        return _word_to_le(_S(_imm(imm, 12), rs2, rs1, f3, op))

    # ---- B-type branches ----
    BRANCHES = {
        'beq': (0b000, 0b1100011),
        'bne': (0b001, 0b1100011),
        'blt': (0b100, 0b1100011),
        'bge': (0b101, 0b1100011),
        'bltu': (0b110, 0b1100011),
        'bgeu': (0b111, 0b1100011),
    }
    if mnem in BRANCHES:
        if len(args) != 3:
            raise ValueError(f'{mnem}: expected rs1, rs2, imm')
        f3, op = BRANCHES[mnem]
        rs1, rs2, imm = _r(args[0]), _r(args[1]), int(args[2], 0)
        return _word_to_le(_B(imm, rs2, rs1, f3, op))

    # ---- U-type ----
    if mnem == 'lui':
        rd, imm = _r(args[0]), int(args[1], 0)
        return _word_to_le(_U(imm & 0xFFFFF, rd, 0b0110111))
    if mnem == 'auipc':
        rd, imm = _r(args[0]), int(args[1], 0)
        return _word_to_le(_U(imm & 0xFFFFF, rd, 0b0010111))

    # ---- J-type ----
    if mnem == 'jal':
        if len(args) != 2:
            raise ValueError('jal: expected rd, imm')
        rd, imm = _r(args[0]), int(args[1], 0)
        return _word_to_le(_J(imm, rd, 0b1101111))

    # JALR is I-type
    if mnem == 'jalr':
        # 'jalr rd, rs1, imm'  (canonical)
        if len(args) == 3:
            rd, rs1, imm = _r(args[0]), _r(args[1]), int(args[2], 0)
        elif len(args) == 2:
            # 'jalr rd, rs1' implies imm=0
            rd, rs1, imm = _r(args[0]), _r(args[1]), 0
        else:
            raise ValueError('jalr: expected rd, rs1[, imm]')
        return _word_to_le(_I(_imm(imm, 12), rs1, 0b000, rd, 0b1100111))

    # ---- System ----
    if mnem == 'ecall':
        return _word_to_le(_I(0, 0, 0b000, 0, 0b1110011))
    if mnem == 'ebreak':
        return _word_to_le(_I(1, 0, 0b000, 0, 0b1110011))
    if mnem == 'fence':
        # Plain `fence` with default pred/succ = 0b1111 (rw, rw)
        return _word_to_le(_I((0b1111 << 4) | 0b1111, 0, 0b000, 0, 0b0001111))
    if mnem == 'nop':
        return encode('addi zero, zero, 0')

    raise ValueError(f'unknown / unsupported mnemonic: {mnem!r}')


def _parse_offset_form(mnem: str, args: list[str]) -> tuple[int, int, int]:
    """Parse 'rd, offset(rs1)' or 'rd, offset, rs1' (relaxed) → (reg_dst_or_src, imm, rs1)."""
    if len(args) == 2 and '(' in args[1]:
        rd_or_rs2 = _r(args[0])
        offset_str, rest = args[1].split('(', 1)
        if not rest.endswith(')'):
            raise ValueError(f'{mnem}: malformed offset operand {args[1]!r}')
        rs1 = _r(rest[:-1])
        imm = int(offset_str, 0)
        return rd_or_rs2, imm, rs1
    if len(args) == 3:
        # Relaxed form: 'lw t0, 8, t1'
        rd_or_rs2 = _r(args[0])
        imm = int(args[1], 0)
        rs1 = _r(args[2])
        return rd_or_rs2, imm, rs1
    raise ValueError(f'{mnem}: expected offset(rs1) form, got args={args}')


def assemble(asm_text: str) -> bytes:
    """
    Encode multiple lines (separated by newlines or ';').  Returns concatenated
    little-endian instruction bytes.  Empty lines and lines starting with '#'
    are ignored.
    """
    out = b''
    for raw in asm_text.replace(';', '\n').split('\n'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        out += encode(line)
    return out
