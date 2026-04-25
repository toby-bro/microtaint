import pypcode


def translate_instruction(arch_string: str, instruction_bytes: bytes, base_address: int = 0x1000) -> None:
    """
    Translates raw instruction bytes into Ghidra's P-code (SLEIGH formula)
    using the pypcode v3.x API.
    """
    try:
        # Initialize the architecture context
        ctx = pypcode.Context(arch_string)
    except Exception as e:
        print(f"Error loading architecture '{arch_string}': {e}")
        return

    print(f'Architecture: {arch_string}')
    print(f'Instruction Bytes: {instruction_bytes.hex()}')
    print('-' * 50)

    try:
        # Translate the bytes into P-code
        translation = ctx.translate(instruction_bytes, base_address=base_address)
    except Exception as e:
        print(f'Translation failed: {e}')
        return

    if not translation.ops:
        print('Failed to decode any P-code operations.')
        return

    # In pypcode 3.x, the translation returns a flat sequence of ops.
    # We can use the built-in PcodePrettyPrinter to format them cleanly.
    for i, op in enumerate(translation.ops):
        formatted_op = pypcode.PcodePrettyPrinter.fmt_op(op)
        print(f'  [{i:02d}] {formatted_op}')


if __name__ == '__main__':
    arm64arch = 'AARCH64:LE:64:v8A'
    for arm64_inst in [
        bytes.fromhex('007C1053'),
        bytes.fromhex('000040F9'),
        bytes.fromhex('0000018B'),
        bytes.fromhex('000001CB'),
    ]:
        translate_instruction(arm64arch, arm64_inst)

        print('\n' + '=' * 50 + '\n')

    # Example 1: AMD64 'add rax, rbx' -> 48 01 d8
    # Architecture strings format: processor:endian:size:variant
    amd64_arch = 'x86:LE:64:default'
    for amd64_inst in [
        b'\x48\x91',
        b'\x91',
        b'\x48\x29\xd8',
        b'\x29\xd8',
        b'\xd3\xe0',
        b'\x39\xd8',
        bytes.fromhex('d3e0'),
        bytes.fromhex('4831c0'),
        b'\x48\x01\xd8',
        b'\x48\x0f\xc8',
        b'\x0f\xc8',
    ]:
        translate_instruction(amd64_arch, amd64_inst)

        print('\n' + '=' * 50 + '\n')

    # Example 2: x86 (32-bit) 'push ebp' -> 55
    x86_arch = 'x86:LE:32:default'
    for x86_inst in [b'\x0f\xc8', bytes.fromhex('6621d8'), b'\x91']:
        translate_instruction(x86_arch, x86_inst)
