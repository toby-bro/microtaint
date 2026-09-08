/*
 * test_constant_time.c
 *
 * Constant-time / side-channel evaluation for the microtaint engine.
 *
 *   Two implementations of the same primitive — small (32-bit, 32-iter)
 *   modular exponentiation.  One branches on a key bit; one selects with
 *   a bitmask.  The engine should flag the first and clear the second.
 *
 * Provenance of the constructs
 * ----------------------------
 *   pow_branch (vulnerable form):
 *     This is the textbook left-to-right square-and-multiply ladder
 *     described in Knuth, TAOCP volume 2, algorithm 4.6.3.A, and in
 *     Menezes/van Oorschot/Vanstone, Handbook of Applied Cryptography,
 *     algorithm 14.79.  The if-statement on the exponent bit is the
 *     canonical pedagogical example of a key-dependent branch and
 *     appears unchanged in countless cryptography textbooks as the
 *     "naive" form that must be avoided in real implementations.
 *
 *   pow_ct (constant-time form):
 *     The mask-select idiom `result = (mul & mask) | (result & ~mask)`
 *     where `mask = -bit` is the standard branch-free conditional
 *     select.  It is documented as the canonical fix in:
 *       - Thomas Pornin, BearSSL constant-time programming guide:
 *           https://www.bearssl.org/constanttime.html
 *       - the libsodium implementation notes,
 *       - Ferguson, Schneier and Kohno, "Cryptography Engineering",
 *         chapter 16.
 *     Using `0u - bit` rather than a comparison ensures the negation
 *     itself is branch-free on every compiler we are aware of.
 *
 *   Both forms compute the same mathematical function (verified by the
 *   smoke test in the harness building this file) but compile to
 *   fundamentally different control-flow shapes: pow_branch contains
 *   one conditional jump per loop iteration whose taken-or-not is
 *   determined by a bit of the exponent; pow_ct contains no
 *   exponent-dependent jumps at all.
 *
 * What this test demonstrates
 * ---------------------------
 *   The harness runs the same binary twice, with a command-line
 *   argument selecting which routine to call:
 *
 *     ./test_constant_time vuln  -> pow_branch
 *         Expected: a side-channel finding at the je after
 *         test eax,eax (the `if (e & 1)`), repeated up to 32 times.
 *
 *     ./test_constant_time ct    -> pow_ct
 *         Expected: zero side-channel findings.  The mask flows
 *         through arithmetic into result; no tainted flag drives a
 *         conditional jump.
 *
 *   This is the false-positive / false-negative complement to the
 *   avalanche test: the avalanche test certifies the engine does not
 *   miss dependencies; this test certifies the engine does not
 *   hallucinate control-flow leaks where there are none.
 *
 * A note on multiplication
 * ------------------------
 *   The 64-bit multiplication used for modular reduction goes through
 *   the IMUL instruction, which has a CALLOTHER-with-output pcode form
 *   that the engine handles via Unicorn fallback.  Both implementations
 *   pay the same fallback cost on those instructions, so the relative
 *   comparison is fair.  If the fallback affected the comparison, we
 *   would replace the 32-bit modexp with an additive analogue.
 *
 * Threat model and taint scope
 * ----------------------------
 *   Only the EXPONENT is secret (read from stdin, fully tainted).
 *   The base and modulus are PUBLIC constants compiled into the binary.
 *   This matches the RSA/DH threat model exactly: an attacker knows the
 *   base and modulus; only the private key (exponent) is secret.
 *
 *   Keeping base and modulus as constants also avoids a subtlety: if
 *   they were read from stdin they would be tainted too, and any
 *   runtime sanity check on them (e.g. "if (mod < 2)") would itself
 *   constitute a tainted branch and fire a false SC finding on both
 *   variants.
 *
 * Build
 * -----
 *   gcc -O0 -g -static -no-pie -fno-stack-protector \
 *       -o test_constant_time test_constant_time.c
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Public parameters — compiled in, not tainted. */
#define BASE ((uint32_t)7)
#define MOD ((uint32_t)101)

/* --------------------------------------------------------------------- *
 * VULNERABLE square-and-multiply.
 * Knuth TAOCP vol. 2, algorithm 4.6.3.A.  HAC algorithm 14.79.
 * --------------------------------------------------------------------- */
__attribute__((noinline)) static uint32_t pow_branch(
  uint32_t base, uint32_t e, uint32_t mod
) {
  uint32_t result = 1;
  uint32_t b = base % mod;

  for (int i = 0; i < 32; i++) {
    if (e & 1) { /* key-dependent branch */
      result = (uint32_t)((uint64_t)result * b % mod);
    }
    b = (uint32_t)((uint64_t)b * b % mod);
    e >>= 1;
  }
  return result;
}

/* --------------------------------------------------------------------- *
 * CONSTANT-TIME square-and-multiply.
 * Branch-free conditional select per Pornin's BearSSL guide.
 * --------------------------------------------------------------------- */
__attribute__((noinline)) static uint32_t pow_ct(
  uint32_t base, uint32_t e, uint32_t mod
) {
  uint32_t result = 1;
  uint32_t b = base % mod;

  for (int i = 0; i < 32; i++) {
    uint32_t bit = e & 1;
    uint32_t mask = (uint32_t)(0u - bit); /* 0 or 0xFFFFFFFF */
    uint32_t mul = (uint32_t)((uint64_t)result * b % mod);
    result = (mul & mask) | (result & ~mask); /* CT select */
    b = (uint32_t)((uint64_t)b * b % mod);
    e >>= 1;
  }
  return result;
}

int main(int argc, char **argv) {
  /* Read 4 bytes from stdin: the exponent only (the secret).
   * Base and modulus are public constants compiled into the binary.
   * This matches the RSA/DH threat model: attacker knows base and mod,
   * only the private exponent is secret. */
  uint8_t buf[4];
  if (read(0, buf, sizeof buf) != (ssize_t)sizeof buf) { return 1; }
  uint32_t e;
  memcpy(&e, buf, 4);

  uint32_t out;
  if (argc >= 2 && argv[1][0] == 'c') {
    out = pow_ct(BASE, e, MOD); /* expected: no SC findings */
  } else {
    out = pow_branch(BASE, e, MOD); /* expected: SC at the je   */
  }
  write(1, &out, sizeof out);
  return 0;
}
