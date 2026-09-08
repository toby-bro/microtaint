/*
 * ARM64 twin of bench_untainted.c: the SAME workload body, with AArch64
 * syscall stubs, so a per-ISA number compares the engine rather than two
 * different programs.  Linux/AArch64 syscall numbers differ from x86-64
 * (read 63, write 64, exit 93).
 *
 * Build: aarch64-linux-gnu-gcc -O1 -static -nostdlib -fno-stack-protector
 */
static long sys_read(int fd, void *buf, unsigned long n) {
  register long x8 __asm__("x8") = 63;
  register long x0 __asm__("x0") = fd;
  register long x1 __asm__("x1") = (long)buf;
  register long x2 __asm__("x2") = (long)n;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x1), "r"(x2), "r"(x8) : "memory");
  return x0;
}
static void sys_write(int fd, const void *buf, unsigned long n) {
  register long x8 __asm__("x8") = 64;
  register long x0 __asm__("x0") = fd;
  register long x1 __asm__("x1") = (long)buf;
  register long x2 __asm__("x2") = (long)n;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x1), "r"(x2), "r"(x8) : "memory");
}
static void sys_exit(int code) {
  register long x8 __asm__("x8") = 93;
  register long x0 __asm__("x0") = code;
  __asm__ volatile("svc #0" :: "r"(x0), "r"(x8));
  __builtin_unreachable();
}

#define INPUT_SIZE 64
#define CLEAN_SIZE 256
#define ROUNDS 40
#define TAINT_EVERY 1000000  /* never: no tainted instruction at all */

static unsigned char tainted[INPUT_SIZE];
static unsigned char clean[CLEAN_SIZE];

void _start(void) {
  int i, r;
  unsigned long acc = 0;

  /* The ONLY tainted bytes in the program. */
  long n = sys_read(0, tainted, INPUT_SIZE);
  if (n <= 0) sys_exit(1);

  /* Untainted working set, derived from constants only. */
  for (i = 0; i < CLEAN_SIZE; i++) clean[i] = (unsigned char)(i * 7 + 13);

  for (r = 0; r < ROUNDS; r++) {
    /* The bulk: pure untainted mixing.  A taint engine should be able to
     * dismiss every one of these instructions with a cheap check. */
    for (i = 0; i < CLEAN_SIZE; i++) {
      unsigned char v = clean[i];
      v = (unsigned char)((v << 3) | (v >> 5));
      v = (unsigned char)(v + (unsigned char)(i ^ r));
      v ^= clean[(i + 1) & (CLEAN_SIZE - 1)];
      clean[i] = v;
      acc = acc * 1099511628211UL + v;
    }
    /* No tainted slice at all: this binary reads stdin (so taint IS injected
     * and live in the shadow) and then never touches those bytes again. An
     * engine that dismisses instructions with no tainted input should approach
     * native speed here. Reading even ONE tainted byte and folding it into acc
     * would poison the accumulator and, through it, the whole mixing loop --
     * which is exactly what this binary must not do. */
  }

  sys_write(1, &acc, sizeof(acc));
  sys_write(1, tainted, INPUT_SIZE);
  sys_exit(0);
}
