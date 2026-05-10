/*
 * harness.c — minimal userland driver that reproduces the nftables
 * nft_byteorder alignment/stride bug under microtaint.
 *
 * The function nft_byteorder_eval is copied verbatim from the unpatched
 * net/netfilter/nft_byteorder.c. Only the surrounding type definitions
 * and macros are stubbed out so the function compiles in userland.
 *
 * Layout reasoning
 * ----------------
 *   - struct nft_regs in the kernel is `u32 data[NFT_REG32_NUM]` with
 *     NFT_REG32_NUM = 20  →  80 bytes total.
 *   - Each "register" priv->sreg / priv->dreg is a byte offset into
 *     that array (must be a multiple of 4 by validator).
 *   - We pick sreg = 0  and  dreg = 0 (writing in place is fine; the
 *     bug is about overshooting the END of dreg+len, not source/dest
 *     overlap).
 *   - We pick priv->len = 4  →  the loop runs 4/2 = 2 iterations,
 *     but with stride 4 (sizeof of the union), which means the SECOND
 *     iteration writes at offset +4 — STILL inside the register file.
 *     That is harmless. To make the bug observable we pick priv->len
 *     such that the buggy stride walks past dreg + 80 (the end of
 *     the register file) into the canary placed AFTER it on the stack.
 *
 *     With dreg = 0 and priv->len = N, the loop iterates N/2 times
 *     with stride 4 and writes 2 bytes per iteration. The highest
 *     written byte is at offset 4*(N/2 - 1) + 1 = 2N - 3.
 *     For the validator: it accepts only N such that dreg + N <= 80,
 *     i.e. N <= 80. We pick N = 80, the maximum allowed.
 *     Buggy access reaches offset 2*80 - 3 = 157 — 77 bytes past
 *     the end of the register file. That overshoots the canary
 *     immediately after it.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>   /* htons, ntohs, htonl, ntohl */

/* -------------------------------------------------------------------------- */
/* Kernel-type stubs: enough to compile nft_byteorder_eval verbatim.          */
/* -------------------------------------------------------------------------- */

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u16 __be16;
typedef u32 __be32;
typedef u64 __be64;
typedef u16 __u16;
typedef u32 __u32;

#define __force
#define ___constant_swab64(x) \
    ((u64)((((u64)(x) & 0x00000000000000ffull) << 56) | \
           (((u64)(x) & 0x000000000000ff00ull) << 40) | \
           (((u64)(x) & 0x0000000000ff0000ull) << 24) | \
           (((u64)(x) & 0x00000000ff000000ull) <<  8) | \
           (((u64)(x) & 0x000000ff00000000ull) >>  8) | \
           (((u64)(x) & 0x0000ff0000000000ull) >> 24) | \
           (((u64)(x) & 0x00ff000000000000ull) >> 40) | \
           (((u64)(x) & 0xff00000000000000ull) >> 56)))

/* Mirror the kernel's NFT_REG32_NUM = 20  →  80 bytes of register space. */
#define NFT_REG32_NUM 20

struct nft_regs {
    u32 data[NFT_REG32_NUM];
};

/* nft_expr — vtable + inline data. Only `data` is touched here. */
struct nft_expr_ops { int dummy; };
struct nft_expr {
    const struct nft_expr_ops *ops;
    unsigned char data[] __attribute__((aligned(__alignof__(u64))));
};

/* nft_byteorder_eval reads priv via nft_expr_priv(expr). */
static inline void *nft_expr_priv(const struct nft_expr *expr)
{
    return (void *)expr->data;
}

/* nft_pktinfo — never read by nft_byteorder_eval but in the prototype. */
struct nft_pktinfo { int dummy; };

/* Operation codes. */
enum nft_byteorder_ops {
    NFT_BYTEORDER_NTOH,
    NFT_BYTEORDER_HTON,
};

/* The expression-private struct. Field layout taken from
 * include/net/netfilter/nf_tables.h. */
struct nft_byteorder {
    u8  sreg;     /* byte offset into nft_regs.data */
    u8  dreg;     /* byte offset into nft_regs.data */
    u8  op;       /* enum nft_byteorder_ops */
    u8  len;      /* total bytes to operate on */
    u8  size;     /* element size: 2, 4, or 8 */
};

/* ---------------------------------------------------------------------- */
/* nft_byteorder_eval — verbatim from net/netfilter/nft_byteorder.c       */
/* (unpatched). Whitespace preserved.                                     */
/* ---------------------------------------------------------------------- */

void nft_byteorder_eval(const struct nft_expr *expr,
            struct nft_regs *regs,
            const struct nft_pktinfo *pkt)
{
    const struct nft_byteorder *priv = nft_expr_priv(expr);
    u32 *src = &regs->data[priv->sreg];
    u32 *dst = &regs->data[priv->dreg];
    union { u32 u32; u16 u16; } *s, *d;
    unsigned int i;

    s = (void *)src;
    d = (void *)dst;

    switch (priv->size) {
    case 8: {
        u64 *_s = (u64 *)src;
        u64 *_d = (u64 *)dst;

        switch (priv->op) {
        case NFT_BYTEORDER_NTOH:
            for (i = 0; i < priv->len / 8; i++)
                _d[i] = ___constant_swab64(_s[i]);
            break;
        case NFT_BYTEORDER_HTON:
            for (i = 0; i < priv->len / 8; i++)
                _d[i] = ___constant_swab64(_s[i]);
            break;
        }
        break;
    }
    case 4:
        switch (priv->op) {
        case NFT_BYTEORDER_NTOH:
            for (i = 0; i < priv->len / 4; i++)
                d[i].u32 = ntohl((__force __be32)s[i].u32);
            break;
        case NFT_BYTEORDER_HTON:
            for (i = 0; i < priv->len / 4; i++)
                d[i].u32 = (__force __u32)htonl(s[i].u32);
            break;
        }
        break;
    case 2:
        switch (priv->op) {
        case NFT_BYTEORDER_NTOH:
            for (i = 0; i < priv->len / 2; i++)
                d[i].u16 = ntohs((__force __be16)s[i].u16);
            break;
        case NFT_BYTEORDER_HTON:
            for (i = 0; i < priv->len / 2; i++)
                d[i].u16 = (__force __u16)htons(s[i].u16);
            break;
        }
        break;
    }
}

/* ---------------------------------------------------------------------- */
/* Driver                                                                 */
/* ---------------------------------------------------------------------- */

/*
 * The harness lays out, in this exact order in BSS, three back-to-back
 * 256-byte regions, all 16-byte aligned. The middle region IS the
 * register file; the outer two are sentinel canaries filled with a
 * known pattern. The python harness will:
 *
 *   - taint REGS (color A) so that any value the buggy loop writes
 *     also paints color-A taint at its destination — proving that the
 *     attacker-controlled source data physically reached out-of-bounds.
 *   - mark CANARY_AFTER (color B) as sentinel territory in a second,
 *     parallel BitPreciseShadowMemory.
 *
 * On every guest write, the python hook checks color B; an overlap is
 * the bug. Optionally it also checks color A at that address — if the
 * bytes coming in are themselves color-A tainted, that proves the
 * specific cross-color "mixing" the user asked for.
 *
 * Each region is 256 bytes (one cache line plus padding) which is
 * comfortably larger than the 80-byte register file and the 77-byte
 * overshoot, so the canary fully covers the buggy stride.
 */
#define REGION_SIZE 256

/*
 * One contiguous struct so the linker cannot insert padding between
 * the register file and the trailing canary. Layout:
 *
 *   [ canary_before : 256 B ][ regs : 80 B ][ canary_after : 256 B ]
 *
 * The canary_after directly follows regs_storage with no gap; that's
 * the bytes the buggy stride is going to stomp.
 */
struct layout {
    unsigned char canary_before[REGION_SIZE];
    struct nft_regs regs;
    unsigned char canary_after[REGION_SIZE];
} __attribute__((aligned(64)));

static struct layout L;
#define regs_storage  (L.regs)
#define canary_before (L.canary_before)
#define canary_after  (L.canary_after)

int main(void)
{
    /*
     * Read three numbers from stdin, in this order:
     *   1) priv->len    (one byte; valid range 0..80 to pass the
     *                    configuration validator)
     *   2) priv->size   (one byte; we pick 2 to trigger the bug)
     *   3) priv->op     (one byte; 0 = NTOH, 1 = HTON)
     *
     * In the real kernel these come from netlink as u8 fields of the
     * compiled rule. The microtaint harness will inject all three via
     * the read() syscall hook so they enter the run as color-A
     * tainted bytes.
     *
     * The harness ALSO populates regs_storage from stdin (one byte
     * per byte of the 80-byte register file) so the source of the
     * htons/ntohs is attacker-controlled — that's how a real exploit
     * gets data into the register file in the first place.
     */
    unsigned char ctl[3];
    if (fread(ctl, 1, sizeof(ctl), stdin) != sizeof(ctl)) return 1;

    if (fread(&regs_storage, 1, sizeof(regs_storage), stdin)
        != sizeof(regs_storage))
        return 1;

    /* Fill canaries with a recognisable, NON-zero, NON-PALINDROMIC pattern.
     * We alternate so any 2-byte word is 0xCDEF (non-palindromic, easy to
     * spot post-swap). Without this, the byteswap of 0xCDCD reads back as
     * 0xCDCD and the corruption is silent. */
    for (size_t i = 0; i < sizeof(canary_before); i++)
        canary_before[i] = (i & 1) ? 0xAB : 0x89;
    for (size_t i = 0; i < sizeof(canary_after); i++)
        canary_after[i]  = (i & 1) ? 0xCD : 0xEF;

    /* Build the expression. We allocate it on the stack as a u64-
     * aligned buffer big enough to hold struct nft_byteorder. */
    union {
        struct nft_expr expr;
        unsigned char raw[sizeof(struct nft_expr) + sizeof(struct nft_byteorder)];
    } expr_buf;
    expr_buf.expr.ops = NULL;

    struct nft_byteorder *priv = (struct nft_byteorder *)expr_buf.expr.data;
    priv->sreg = 0;
    priv->dreg = 0;
    priv->op   = ctl[2];
    priv->len  = ctl[0];
    priv->size = ctl[1];

    struct nft_pktinfo pkt = { 0 };

    /* Print the canary BEFORE the call so we have a baseline. */
    fprintf(stdout, "BEFORE: canary_after[0..7] = "
                    "%02x %02x %02x %02x %02x %02x %02x %02x\n",
        canary_after[0], canary_after[1], canary_after[2], canary_after[3],
        canary_after[4], canary_after[5], canary_after[6], canary_after[7]);
    fflush(stdout);

    nft_byteorder_eval(&expr_buf.expr, &regs_storage, &pkt);

    fprintf(stdout, "AFTER:  canary_after[0..7] = "
                    "%02x %02x %02x %02x %02x %02x %02x %02x\n",
        canary_after[0], canary_after[1], canary_after[2], canary_after[3],
        canary_after[4], canary_after[5], canary_after[6], canary_after[7]);
    fflush(stdout);

    /* Print the addresses so the python side can find the canary. We
     * use a marker prefix the python harness can grep for. */
    fprintf(stdout, "ADDR regs_storage  = %p size = %zu\n",
            (void *)&regs_storage, sizeof(regs_storage));
    fprintf(stdout, "ADDR canary_after  = %p size = %zu\n",
            (void *)canary_after, sizeof(canary_after));
    fprintf(stdout, "ADDR canary_before = %p size = %zu\n",
            (void *)canary_before, sizeof(canary_before));
    fflush(stdout);

    return 0;
}
