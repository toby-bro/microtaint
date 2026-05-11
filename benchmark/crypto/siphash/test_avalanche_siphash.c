/*
 * test_avalanche_siphash.c
 *
 * Harness around the upstream SipHash reference implementation
 * (siphash_ref.c) for the microtaint avalanche / precision test.
 *
 *   The crypto primitive itself is the verbatim reference C
 *   implementation by Aumasson & Bernstein, taken from
 *   https://github.com/veorq/SipHash and shipped here as
 *   siphash_ref.c with its CC0 licence intact.  No modifications.
 *
 *   The only original code in this test is the I/O harness in main()
 *   below: it reads 16 bytes of message from stdin, calls the
 *   reference siphash() with a fixed key, and writes the 8-byte
 *   output to stdout.
 *
 * Why SipHash for an avalanche test
 * ---------------------------------
 *   The compression function uses only ADD, XOR, and rotate on 64-bit
 *   words.  Every operation lowers to a pcode opcode the engine handles
 *   natively (INT_ADD, INT_XOR, INT_LEFT/RIGHT/OR for ROL) - no Unicorn
 *   fallback, no FPU/SIMD, no CALLOTHER.  SipHash's PRF security claim
 *   guarantees strong avalanche, so the empirical "did flip" mask on a
 *   single-bit input change should cover roughly half the output bits.
 *
 * What this test demonstrates
 * ---------------------------
 *   We taint exactly ONE bit of the 16-byte input message.  We run
 *   the engine and read the predicted taint mask on the 64-bit output.
 *   Two ground-truth comparisons are done outside the engine:
 *
 *     SOUNDNESS (no under-tainting):
 *       For the same key, run the hash twice with the chosen input
 *       bit clear and set respectively.  XOR the two outputs to get
 *       the empirical "did flip" mask.  This must be a SUBSET of the
 *       engine's predicted mask.  Any bit set empirically but not by
 *       the engine is a missed dependency = under-tainting bug.
 *
 *     PRECISION (no over-tainting, statistical):
 *       Repeat the above with N (~256) random keys.  OR all the
 *       empirical masks together; this converges to the true
 *       "could flip" set.  If the engine's predicted mask still
 *       strictly exceeds the union after N trials, that's
 *       over-tainting.
 *
 *     AVALANCHE (the cryptographic property of SipHash):
 *       For each random key, the empirical mask should have ~32 bits
 *       set on average.  This is a property of SipHash itself, not of
 *       the engine - but it makes the soundness/precision claims
 *       meaningful: the engine must mark approximately half the output
 *       bits as tainted on a single-bit input flip, and that mark must
 *       be tight.
 *
 * Build
 * -----
 *   gcc -O0 -g -static -no-pie -fno-stack-protector \
 *       -o test_avalanche_siphash \
 *       test_avalanche_siphash.c siphash_ref.c
 *
 * Running under microtaint
 * ------------------------
 *   The harness reads exactly 16 bytes from stdin (the message).  Mark
 *   them all tainted, or use a finer-grained taint mask to study how
 *   individual input bits propagate into the 8-byte output.  The hash
 *   output is written to stdout immediately before exit so the engine
 *   can inspect the taint on the output buffer at the syscall.
 */

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/* Provided by siphash_ref.c (upstream reference implementation). */
int siphash(const void *in, size_t inlen, const void *k,
            uint8_t *out, size_t outlen);

int main(void)
{
    /* Fixed key.  The avalanche test is about message-bit propagation,
     * so we hold the key constant.  A complete evaluation would also
     * vary the key across runs; that's done by the harness around this
     * binary, not by recompiling. */
    static const uint8_t key[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    };

    uint8_t msg[16];
    if (read(0, msg, sizeof msg) != (ssize_t)sizeof msg) {
        return 1;
    }

    uint8_t out[8];
    siphash(msg, sizeof msg, key, out, sizeof out);
    write(1, out, sizeof out);
    return 0;
}
