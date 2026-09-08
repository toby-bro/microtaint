/*
 * microtaint benchmark: heavy taint propagation stress test
 *
 * Design goals:
 *   - 256 bytes of tainted stdin input
 *   - ~100 rounds of mixed arithmetic on tainted data (XOR, ADD, ROL, multiply)
 *   - Array indexing with tainted indices  → tainted memory addresses
 * (LOAD/STORE)
 *   - Flag taint:  comparisons on tainted values  → ZF/SF tainted
 *   - Eventual unchecked memcpy → stack BOF (tainted return address)
 *
 * Pure syscalls, zero libc, so nearly every instruction is in the main binary
 * and our ranged hook sees the full taint propagation chain.
 */

/* ── syscall wrappers ─────────────────────────────────────────────────── */
static long sys_read(int fd, void *buf, unsigned long n) {
  long r;
  __asm__ volatile("syscall"
                   : "=a"(r)
                   : "0"(0), "D"((long)fd), "S"(buf), "d"(n)
                   : "rcx", "r11", "memory");
  return r;
}

static void sys_write(int fd, const void *buf, unsigned long n) {
  long r;
  __asm__ volatile("syscall"
                   : "=a"(r)
                   : "0"(1), "D"((long)fd), "S"(buf), "d"(n)
                   : "rcx", "r11", "memory");
}

static void sys_exit(int code) {
  __asm__ volatile("syscall" ::"a"(60), "D"((long)code) : "rcx", "r11");
  __builtin_unreachable();
}

/* ── constants ────────────────────────────────────────────────────────── */
#define INPUT_SIZE 256
#define ROUNDS 100
#define SBOX_SIZE 256

/* A fixed S-box (non-tainted) — but indexing into it with a tainted index
 * makes the OUTPUT tainted via a memory LOAD with tainted address. */
static const unsigned char SBOX[SBOX_SIZE] = {
  99,  124, 119, 123, 242, 107, 111, 197, 48,  1,   103, 43,  254, 215, 171,
  118, 202, 130, 201, 125, 250, 89,  71,  240, 173, 212, 162, 175, 156, 164,
  114, 192, 183, 253, 147, 38,  54,  63,  247, 204, 52,  165, 229, 241, 113,
  216, 49,  21,  4,   199, 35,  195, 24,  150, 5,   154, 7,   18,  128, 226,
  235, 39,  178, 117, 9,   131, 44,  26,  27,  110, 90,  160, 82,  59,  214,
  179, 41,  227, 47,  132, 83,  209, 0,   237, 32,  252, 177, 91,  106, 203,
  190, 57,  74,  76,  88,  207, 208, 239, 170, 251, 67,  77,  51,  133, 69,
  249, 2,   127, 80,  60,  159, 168, 81,  163, 64,  143, 146, 157, 56,  245,
  188, 182, 218, 33,  16,  255, 243, 210, 205, 12,  19,  236, 95,  151, 68,
  23,  196, 167, 126, 61,  100, 93,  25,  115, 96,  129, 79,  220, 34,  42,
  144, 136, 70,  238, 184, 20,  222, 94,  11,  219, 224, 50,  58,  10,  73,
  6,   36,  92,  194, 211, 172, 98,  145, 149, 228, 121, 231, 200, 55,  109,
  141, 213, 78,  169, 108, 86,  244, 234, 101, 122, 174, 8,   186, 120, 37,
  46,  28,  166, 180, 198, 232, 221, 116, 31,  75,  189, 139, 138, 112, 62,
  181, 102, 72,  3,   246, 14,  97,  53,  87,  185, 134, 193, 29,  158, 225,
  248, 152, 17,  105, 217, 142, 148, 155, 30,  135, 233, 206, 85,  40,  223,
  140, 161, 137, 13,  191, 230, 66,  104, 65,  153, 45,  15,  176, 84,  187,
  22,
};

/* ── helper: rotate left 8-bit ───────────────────────────────────────── */
static inline unsigned char rol8(unsigned char v, int n) {
  return (unsigned char)((v << n) | (v >> (8 - n)));
}

/* ── single round: mixes the 256-byte state buffer heavily ───────────── */
/*
 * Each round does:
 *   1. XOR-diffusion:  state[i] ^= state[(i+1) % 256]  (taint spreads right)
 *   2. S-box sub:      state[i]  = SBOX[state[i]]       (tainted index →
 * tainted load)
 *   3. ROL + ADD:      state[i]  = rol8(state[i], 3) + (i & 0xFF)
 *   4. Reverse sweep:  state[i] ^= state[(i+255) % 256] (taint spreads left)
 *   5. Accumulate hash into a 64-bit running total
 *
 * Per round: 256 iterations × ~15 tainted ops = ~3840 tainted instructions.
 * × 100 rounds = 384,000 tainted instructions just from mixing.
 */
static unsigned long mix_round(
  unsigned char *state, unsigned long acc, int round
) {
  int i;

  /* Forward XOR diffusion */
  for (i = 0; i < INPUT_SIZE; i++) {
    state[i] ^= state[(i + 1) & (INPUT_SIZE - 1)];
  }

  /* S-box substitution — tainted index causes tainted LOAD */
  for (i = 0; i < INPUT_SIZE; i++) { state[i] = SBOX[state[i]]; }

  /* ROL + add round constant — keeps every byte tainted */
  for (i = 0; i < INPUT_SIZE; i++) {
    state[i] = rol8(state[i], 3) + (unsigned char)(i ^ round);
  }

  /* Reverse XOR diffusion */
  for (i = INPUT_SIZE - 1; i >= 0; i--) {
    state[i] ^= state[(i + INPUT_SIZE - 1) & (INPUT_SIZE - 1)];
  }

  /* Accumulate: fold all 256 tainted bytes into acc */
  for (i = 0; i < INPUT_SIZE; i++) {
    acc = (acc * 6364136223846793005ULL + state[i]) ^ (acc >> 33);
  }

  return acc;
}

/* ── unsafe memcpy (the vulnerable sink) ─────────────────────────────── */
/* Destination is a small stack buffer; no length check → BOF.
 * The tainted source overwrites the return address.                       */
static void __attribute__((noinline)) unsafe_copy(
  const unsigned char *src, int len
) {
  /* Fixed 64-byte stack buffer — BOF if len > 64 */
  unsigned char local_buf[64];
  int i;
  for (i = 0; i < len; i++) {
    local_buf[i] = src[i]; /* tainted write → eventually overwrites RIP */
  }
  /* Force a load from local_buf so the compiler keeps it */
  sys_write(1, local_buf, 8);
}

/* ── entry point ─────────────────────────────────────────────────────── */
void _start(void) {
  unsigned char state[INPUT_SIZE];
  unsigned long acc = 0xdeadbeefcafe1234ULL;
  int i, round;

  /* ── Taint source: read 256 bytes from stdin ── */
  long n = sys_read(0, state, INPUT_SIZE);
  if (n <= 0) { sys_exit(1); }

  /* ── Heavy taint propagation: 100 mix rounds ── */
  for (round = 0; round < ROUNDS; round++) {
    acc = mix_round(state, acc, round);

    /*
     * Interleave: scatter state bytes using tainted acc as seed.
     * This creates tainted STORE addresses → harder differential evaluation.
     */
    for (i = 0; i < INPUT_SIZE; i++) {
      unsigned char idx = (unsigned char)((acc >> (i & 7)) ^ i);
      unsigned char tmp = state[i];
      state[i] = state[idx];
      state[idx] = tmp;
    }
  }

  /*
   * Write out the hash (8 bytes) so the compiler can't optimise away
   * the mix_round calls.
   */
  sys_write(1, &acc, 8);

  /* ── BOF sink: copy 256-byte tainted state into 64-byte local buffer ── */
  unsafe_copy(state, (int)n); /* n = 256 → overflows by 192 bytes */

  sys_exit(0);
}
