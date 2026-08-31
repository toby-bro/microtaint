/*
 * dns_bitfield.c -- sub-byte granularity probe for the LDNS OPCODE extraction.
 *
 * Reproduces the LDNS_OPCODE_WIRE bit-field extract on the RFC1035 flag byte:
 *     AND AL, 0x78   (bytes: 24 78)   -- keep bits 6..3 (OPCODE), drop QR(bit7)
 *     SHR AL, 3      (bytes: C0 E8 03) -- right-align OPCODE into bits 3..0
 *
 * The flag byte packs QR at bit 7 (benign) and OPCODE at bits 6..3
 * (security-relevant).  We read ONE byte from stdin (libdft taints it, the
 * finest granularity libdft can express) and run exactly those two
 * instructions on AL, then store AL into `out`.
 *
 * Ground truth (bit-level): tainting only QR(bit7) -> out clean (masked away);
 * tainting only OPCODE(bits6..3) -> out tainted.  libdft cannot express either
 * in isolation: it taints the whole byte, so it must report out tainted.
 */
#include <stdint.h>
#include <unistd.h>

volatile unsigned char out = 0;

int main(void) {
    unsigned char flag = 0;
    if (read(0, &flag, 1) != 1) return 1;

    unsigned char al = flag;
    __asm__ volatile(
        "movb %1, %%al\n\t"
        "andb $0x78, %%al\n\t"   /* 24 78 : AND AL, 0x78  */
        "shrb $3, %%al\n\t"      /* C0 E8 03 : SHR AL, 3  */
        "movb %%al, %0\n\t"
        : "=m"(out)
        : "r"(al)
        : "al");

    /* keep the byte observable */
    write(1, (const void *)&out, 1);
    return 0;
}
