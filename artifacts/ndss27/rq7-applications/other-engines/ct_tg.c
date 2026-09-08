/*
 * ct_tg.c — taintgrind harness for the constant-time / side-channel test.
 *
 * Byte-for-byte identical to
 *   ../crypto/square_and_multiply/test_constant_time.c
 * EXCEPT it #includes "taintgrind.h" and calls TNT_TAINT() on the 4 stdin
 * bytes (the secret exponent) right after read(), defining that memory as
 * the taint source. The two pow_ functions are copied verbatim.
 *
 * Build (matches the original test's flags plus taintgrind include path):
 *   gcc -O0 -g -static -no-pie -fno-stack-protector -I. -I/usr/include/valgrind
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "taintgrind.h"   /* TNT_TAINT client request (taint source) */

/* Public parameters — compiled in, not tainted. */
#define BASE ((uint32_t)7)
#define MOD ((uint32_t)101)

/* --------------------------------------------------------------------- *
 * VULNERABLE square-and-multiply.  (verbatim)
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
 * CONSTANT-TIME square-and-multiply.  (verbatim)
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
  uint8_t buf[4];
  if (read(0, buf, sizeof buf) != (ssize_t)sizeof buf) { return 1; }

  /* Mark the 4-byte secret exponent as the taint source. */
  TNT_TAINT(buf, sizeof buf);

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
