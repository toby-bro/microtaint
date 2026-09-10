/* Stack buffer overflow: 32 tainted bytes into a 16-byte buffer overwrite the
 * saved return address, so taint reaches RIP at the `ret`. */
#include "syscalls.h"

static void vulnerable(void) {
    char buf[16];
    sys_read(0, buf, 32);
}

void _start(void) {
    setup_stdin();
    vulnerable();
    sys_exit(0);
}
