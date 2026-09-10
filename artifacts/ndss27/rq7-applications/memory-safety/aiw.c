/* Arbitrary indexed write: the store address carries tainted bits, so the
 * attacker chooses where the write lands. */
#include "syscalls.h"

static char table[4096];

void _start(void) {
    setup_stdin();
    unsigned long idx = 0;
    sys_read(0, &idx, 8);
    table[idx & 0xfff] = (char)idx;
    sys_exit(0);
}
