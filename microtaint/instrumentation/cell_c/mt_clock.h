#ifndef MT_CLOCK_H
#define MT_CLOCK_H

/*
 * mt_clock — a monotonic nanosecond counter, wherever this builds.
 *
 * `clock_gettime(CLOCK_MONOTONIC)` is POSIX.  MinGW does not reliably declare
 * it (it depends on the runtime version and on whether winpthread is linked),
 * and MSVC does not have it at all, which is what broke the Windows wheel.
 *
 * What a benchmark needs from a clock is that it be monotonic and fine
 * grained; it never needs a wall date.  QueryPerformanceCounter is exactly
 * that on Windows, and its frequency is fixed for the lifetime of the
 * process, so it is read once.
 */

#include <stdint.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <time.h>
#endif

/* Nanoseconds from some unspecified fixed point.  Only differences are
 * meaningful, which is all a benchmark uses. */
static inline uint64_t mt_now_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    /* Split the division so a long-running counter cannot overflow the
     * multiply: whole seconds first, then the remainder scaled. */
    return (uint64_t)(now.QuadPart / freq.QuadPart) * 1000000000ull
         + (uint64_t)(now.QuadPart % freq.QuadPart) * 1000000000ull
           / (uint64_t)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

#endif /* MT_CLOCK_H */
