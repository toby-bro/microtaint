import os
import platform
import subprocess
import tempfile

import pytest
from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.wrapper import MicrotaintWrapper

pytestmark = pytest.mark.skipif(platform.system() != 'Linux', reason='Emulator compilation tests require Linux')


def compile_c_code(source_code: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    # -nostdlib removes ALL glibc overhead and crashes
    cmd = ['gcc', '-nostdlib', '-O0', '-fno-stack-protector', '-o', path, '-x', 'c', '-']
    subprocess.run(cmd, input=source_code.encode(), check=True)
    return path


def run_wrapper(binary_path: str, stdin_data: bytes, check_bof: bool, check_uaf: bool, check_sc: bool) -> list[str]:
    from io import StringIO

    from microtaint.sleigh.engine import _cached_generate_static_rule

    _cached_generate_static_rule.cache_clear()

    from qiling.extensions import pipe

    ql = Qiling([binary_path], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = pipe.SimpleInStream(0)
    ql.os.stdin.write(stdin_data)

    # Capture reporter output in a StringIO so tests can assert on finding labels
    from microtaint.emulator.reporter import Reporter

    report_stream = StringIO()
    reporter = Reporter(json_mode=False, stream=report_stream)

    wrapper = MicrotaintWrapper(
        ql,
        check_bof=check_bof,
        check_uaf=check_uaf,
        check_sc=check_sc,
        reporter=reporter,
    )
    try:
        ql.run()
    except Exception:
        pass

    reporter.finalize()
    return report_stream.getvalue().splitlines()


# Baremetal syscall definitions for our test environment
SYSCALLS = """
long sys_read(int fd, void *buf, unsigned long count) {
    long ret;
    __asm__ volatile ("syscall" : "=a" (ret) : "0"(0), "D"(fd), "S"(buf), "d"(count) : "rcx", "r11", "memory");
    return ret;
}
long sys_open(const char *filename, int flags) {
    long ret;
    __asm__ volatile ("syscall" : "=a" (ret) : "0"(2), "D"(filename), "S"(flags), "d"(0) : "rcx", "r11", "memory");
    return ret;
}
long sys_dup2(int oldfd, int newfd) {
    long ret;
    __asm__ volatile ("syscall" : "=a" (ret) : "0"(33), "D"(oldfd), "S"(newfd) : "rcx", "r11", "memory");
    return ret;
}
long sys_exit(int status) {
    long ret;
    __asm__ volatile ("syscall" : "=a" (ret) : "0"(60), "D"(status) : "rcx", "r11", "memory");
    return ret;
}
void setup_stdin() {
    long argc;
    char **argv;
    __asm__ volatile ("mov %%rsp, %%rax\\n mov (%%rax), %0\\n lea 8(%%rax), %1\\n" : "=r"(argc), "=r"(argv) :: "rax");
    if (argc > 1) {
        long fd = sys_open(argv[1], 0);
        if (fd >= 0) sys_dup2(fd, 0);
    }
}
"""


def test_detects_buffer_overflow() -> None:
    source = (
        SYSCALLS
        + """
    void vulnerable() {
        char buf[16];
        sys_read(0, buf, 32); // BOF
    }
    void _start() {
        setup_stdin();
        vulnerable();
        sys_exit(0);
    }
    """
    )
    binary = compile_c_code(source)
    payload = b'A' * 32

    logs = run_wrapper(binary, payload, check_bof=True, check_uaf=False, check_sc=False)
    assert any('[BOF]' in line or 'buffer_overflow' in line for line in logs)


def test_detects_side_channel() -> None:
    source = (
        SYSCALLS
        + """
    void _start() {
        setup_stdin();
        char key[8];
        sys_read(0, key, 8);
        
        if (key[0] == 'X') {
            sys_exit(1);
        }
        sys_exit(0);
    }
    """
    )
    binary = compile_c_code(source)
    payload = b'X0000000'

    logs = run_wrapper(binary, payload, check_bof=False, check_uaf=False, check_sc=True)
    assert any('[SC]' in line or 'side_channel' in line for line in logs)


def test_detects_use_after_free() -> None:
    source = (
        SYSCALLS
        + """
    long sys_mmap(void *addr, unsigned long length, int prot, int flags, int fd, long offset) {
        long ret;
        register long r10 asm("r10") = flags;
        register long r8 asm("r8") = fd;
        register long r9 asm("r9") = offset;
        __asm__ volatile ("syscall" : "=a" (ret) : "0"(9), "D"(addr), "S"(length), "d"(prot), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
        return ret;
    }
    long sys_munmap(void *addr, unsigned long length) {
        long ret;
        __asm__ volatile ("syscall" : "=a" (ret) : "0"(11), "D"(addr), "S"(length) : "rcx", "r11", "memory");
        return ret;
    }
    void _start() {
        char *ptr = (char*)sys_mmap(0, 4096, 3, 34, -1, 0);
        sys_munmap(ptr, 4096);
        ptr[0] = 'A'; // UAF Access
        sys_exit(0);
    }
    """
    )
    binary = compile_c_code(source)
    logs = run_wrapper(binary, b'', check_bof=False, check_uaf=True, check_sc=False)
    assert any('[UAF]' in line or 'use_after_free' in line for line in logs)
