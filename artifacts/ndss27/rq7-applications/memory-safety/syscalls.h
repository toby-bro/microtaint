/* Freestanding syscall stubs.  The detectors are about taint, not libc, so the
 * targets are built with -nostdlib: no dynamic loader, no libc init, and the
 * only instructions Qiling executes are the ones written here. */
#ifndef SYSCALLS_H
#define SYSCALLS_H

static long sys_read(int fd, void *buf, unsigned long count) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(0), "D"(fd), "S"(buf), "d"(count) : "rcx", "r11", "memory");
    return ret;
}

static long sys_open(const char *path, int flags) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(2), "D"(path), "S"(flags), "d"(0) : "rcx", "r11", "memory");
    return ret;
}

static long sys_dup2(int oldfd, int newfd) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(33), "D"(oldfd), "S"(newfd) : "rcx", "r11", "memory");
    return ret;
}

static long sys_exit(int status) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(60), "D"(status) : "rcx", "r11", "memory");
    return ret;
}

/* If argv[1] names a file, use it as stdin.  microtaint taints whatever the
 * program reads from fd 0, so this is what makes the input the taint source. */
static void setup_stdin(void) {
    long argc;
    char **argv;
    __asm__ volatile("mov %%rsp, %%rax\n mov (%%rax), %0\n lea 8(%%rax), %1\n" : "=r"(argc), "=r"(argv)::"rax");
    if (argc > 1) {
        long fd = sys_open(argv[1], 0);
        if (fd >= 0) sys_dup2(fd, 0);
    }
}

#endif
