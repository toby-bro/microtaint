/*
 * microtaint benchmark: SPARSELY-tainted workload.
 *
 * bench_r10 taints essentially every value, which is the worst case and tells
 * us nothing about the common one.  Real targets read a little untrusted input
 * and then spend most of their instructions on untainted work: loop counters,
 * table setup, bookkeeping, pointer arithmetic on clean data.
 *
 * This does the same total amount of work as a mixing benchmark, but only a
 * small, controllable slice of it touches the tainted bytes:
 *
 *   TAINT_FRAC = 1/64  -> ~1.5% of the mixing loops read tainted state
 *
 * Everything else operates on a second, untainted buffer that is never derived
 * from stdin.  A taint engine that pre-filters on "are any inputs tainted"
 * should approach native speed here; one that evaluates unconditionally pays
 * the same price as bench_r10.
 *
 * Build: gcc -O1 -static -nostdlib -fno-stack-protector -o bench_sparse.elf bench_sparse.c
 */

static long sys_read(int fd, void *buf, unsigned long n) {
  long r;
  __asm__ volatile("syscall" : "=a"(r) : "0"(0), "D"((long)fd), "S"(buf), "d"(n)
                   : "rcx", "r11", "memory");
  return r;
}
static void sys_write(int fd, const void *buf, unsigned long n) {
  long r;
  __asm__ volatile("syscall" : "=a"(r) : "0"(1), "D"((long)fd), "S"(buf), "d"(n)
                   : "rcx", "r11", "memory");
}
static void sys_exit(int code) {
  __asm__ volatile("syscall" ::"a"(60), "D"((long)code) : "rcx", "r11");
  __builtin_unreachable();
}

#define INPUT_SIZE 64
#define CLEAN_SIZE 256
#define ROUNDS 40
#define TAINT_EVERY 64   /* 1 in 64 iterations touches tainted data */

static unsigned char tainted[INPUT_SIZE];
static unsigned char clean[CLEAN_SIZE];

void _start(void) {
  int i, r;
  unsigned long acc = 0;
  unsigned char tsink = 0;

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
    /* The sparse tainted slice. */
    for (i = 0; i < CLEAN_SIZE; i += TAINT_EVERY) {
      unsigned char t = tainted[i & (INPUT_SIZE - 1)];
      t = (unsigned char)((t << 1) | (t >> 7));
      t ^= clean[i];
      tainted[i & (INPUT_SIZE - 1)] = t;
      tsink ^= t;   /* deliberately NOT acc: folding tainted data into the
                     * clean accumulator would poison the whole mixing loop and
                     * turn this into a dense benchmark wearing a sparse name. */
    }
  }

  sys_write(1, &acc, sizeof(acc));
  sys_write(1, &tsink, 1);
  sys_write(1, tainted, INPUT_SIZE);
  sys_exit(0);
}
