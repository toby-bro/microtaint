#ifndef MT_CODEBUF_H
#define MT_CODEBUF_H

/*
 * mt_codebuf — the one platform-dependent part of a JIT backend.
 * =============================================================
 *
 * Emitting machine code is the portable half of a JIT: the bytes for an ADD
 * depend on the ISA and on nothing else.  What is not portable is getting a
 * page the CPU will execute, and that is the whole of this file, so the
 * backends stay pure emitters and the platform question is answered once.
 *
 * Three steps, in this order, because W^X systems refuse a page that is
 * writable and executable at the same time:
 *
 *   alloc          writable, not yet executable
 *   protect_exec   executable, no longer writable, instruction cache flushed
 *   free
 *
 * The cache flush is not optional on AArch64, where the instruction and data
 * caches are not coherent and freshly written code is not guaranteed to be
 * visible to the fetcher without it.  On x86-64 it is a no-op, and cheap
 * enough that it is not worth making conditional.
 */

#include <stddef.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

/* Writable memory for `sz` bytes of code, or NULL.  `sz` should be a whole
 * number of pages: both backends round it up before calling. */
static inline void *mt_code_alloc(size_t sz) {
#if defined(_WIN32)
    return VirtualAlloc(NULL, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void *m = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return m == MAP_FAILED ? NULL : m;
#endif
}

/* Make the page executable and publish the `used` bytes actually written to
 * the instruction fetcher.  0 on success, -1 on failure, and on failure the
 * caller still owns the page and must free it. */
static inline int mt_code_protect_exec(void *p, size_t sz, size_t used) {
#if defined(_WIN32)
    DWORD old = 0;
    if (!VirtualProtect(p, sz, PAGE_EXECUTE_READ, &old)) return -1;
    /* Windows' own barrier.  __builtin___clear_cache is a GCC/clang builtin
     * and MSVC has no equivalent, so the platform call is what works under
     * every compiler that can build this file. */
    FlushInstructionCache(GetCurrentProcess(), p, used);
    return 0;
#else
    if (mprotect(p, sz, PROT_READ | PROT_EXEC) != 0) return -1;
    __builtin___clear_cache((char *)p, (char *)p + used);
    return 0;
#endif
}

static inline void mt_code_free(void *p, size_t sz) {
#if defined(_WIN32)
    /* MEM_RELEASE frees the whole reservation and requires a size of 0; the
     * size is carried anyway so the POSIX side can use it. */
    (void)sz;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, sz);
#endif
}

#endif /* MT_CODEBUF_H */
