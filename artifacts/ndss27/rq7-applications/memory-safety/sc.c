/* Side channel: the branch condition is a tainted byte, so taint reaches the
 * program counter without ever being written to it. */
#include "syscalls.h"

void _start(void) {
    setup_stdin();
    char key[8];
    sys_read(0, key, 8);
    if (key[0] == 'X') sys_exit(1);
    sys_exit(0);
}
