/*
 * dns_tg.c — taintgrind harness for the DNS bit-field (sub-byte) test.
 *
 * Reproduces the LDNS_OPCODE_WIRE extraction on the RFC1035 flag byte:
 *     AND AL, 0x78   (24 78)   ; keep bits 6..3 (the OPCODE field)
 *     SHR AL, 3      (C0 E8 03); right-align OPCODE into bits 3..0
 * QR is bit 7 (benign); OPCODE is bits 6..3.
 *
 * taintgrind is byte/register granular: it cannot mark a single bit.
 * We mark the whole flag byte tainted (the finest taintgrind can do) and
 * report whether the extracted OPCODE output (AL) is tainted.
 *
 * Build:
 *   gcc -O0 -g -static -no-pie -fno-stack-protector -I. -I/usr/include/valgrind
 */

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "taintgrind.h"

int main(void) {
  /* RFC1035 flag byte. Read 1 byte from stdin so the value is runtime
   * data; default to 0x85 (QR=1, OPCODE=0, ...) if none supplied. */
  volatile uint8_t flag = 0x85;
  uint8_t rd;
  if (read(0, &rd, 1) == 1) flag = rd;

  /* Mark the entire flag byte as tainted — taintgrind's finest granularity.
   * There is no way to taint only bit 7 (QR) or only bits 6..3 (OPCODE). */
  TNT_TAINT((void *)&flag, 1);

  uint8_t opcode = flag;
  /* The exact two instructions from LDNS_OPCODE_WIRE. */
  __asm__ volatile(
      "andb $0x78, %0\n\t"   /* AND AL, 0x78 */
      "shrb $3, %0\n\t"      /* SHR AL, 3   */
      : "+r"(opcode)
      :
      :);

  int out_tainted = 0;
  TNT_IS_TAINTED(out_tainted, &opcode, 1);

  /* Emit the extracted OPCODE so the store/exit shows in the log. */
  write(1, &opcode, 1);
  fprintf(stderr, "opcode_value=0x%02x out_al_tainted=%d\n",
          opcode, out_tainted);
  return 0;
}
