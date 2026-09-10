/* Use after free: the region is unmapped, then written through the stale
 * pointer.  microtaint poisons the freed range and reports the access. */
#include "syscalls.h"

static long sys_mmap(void *addr, unsigned long len, int prot, int flags, int fd, long off) {
    long ret;
    register long r10 __asm__("r10") = flags;
    register long r8 __asm__("r8") = fd;
    register long r9 __asm__("r9") = off;
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "0"(9), "D"(addr), "S"(len), "d"(prot), "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    return ret;
}

static long sys_munmap(void *addr, unsigned long len) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(11), "D"(addr), "S"(len) : "rcx", "r11", "memory");
    return ret;
}

void _start(void) {
    char *p = (char *)sys_mmap(0, 4096, 3, 34, -1, 0);
    sys_munmap(p, 4096);
    p[0] = 'A';
    sys_exit(0);
}
