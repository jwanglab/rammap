# Undefined Behavior in minimap2's `ksw_ll_i16` and `mm_align1_inv`

## Overview

minimap2's lightweight Smith-Waterman function `ksw_ll_i16` can return a
`query_end` position that lies beyond the actual query length. When this
value is consumed by `mm_align1_inv` (inversion alignment), it produces a
negative buffer offset, causing a read from memory before the start of the
query buffer. This is undefined behavior in C.

The bug is triggered when `qlen % 8 != 0` and the maximum DP score happens
to land on a SIMD padding position.

## SIMD segmented layout

`ksw_ll_i16` uses a striped (interleaved) SIMD layout. Each `__m128i`
register holds 8 `int16` values. The query is divided into `slen =
ceil(qlen / 8)` segments of 8 elements each, for a total of `qlen8 = slen
* 8` positions. When `qlen` is not a multiple of 8, the extra positions are
initialized with zero scores:

```c
// ksw2_ll_sse.c, ksw_ll_qinit()
for (t = 0; t < m; ++t)
    for (i = 0, k = t; i < slen; ++i, k += p)
        qp[t][i] = k < qlen ? mat[t * m + query[k]] : 0;
//                   ^^^^^^^^^                        ^^^
//                   real positions                   padding
```

Padding positions don't contribute substitution scores, but they *can*
accumulate non-zero values during the DP through gap-extension propagation
from neighboring real positions.

## The bug: unfiltered `query_end` scan

After the DP loop, the function scans the stored `Hmax` row to find which
query position achieved the global maximum score `gmax`:

```c
// ksw2_ll_sse.c:149-150
for (i = 0, H8 = (uint16_t*)Hmax; i < qlen8; ++i)
    if ((int)H8[i] == gmax) *qe = i / 8 + i % 8 * slen;
```

The scan iterates over all `qlen8` positions — including padding. Nothing
prevents `*qe` from being set to a position `>= qlen`. The position
mapping `i / 8 + (i % 8) * slen` translates correctly from striped to
linear order, but padding positions map to values between `qlen` and
`qlen8 - 1`, which are not valid query indices.

## How it reaches `mm_align1_inv`

`mm_align1_inv` (align.c) calls `ksw_ll_i16` to find the alignment
endpoint in a reversed-complement context, then adjusts the offset to
forward coordinates:

```c
// align.c:940-941
qp = ksw_ll_qinit(km, 2, ql, qseq, 5, mat);
score = ksw_ll_i16(qp, tl, tseq, opt->q, opt->e, &q_off, &t_off);

// align.c:946 — offset adjustment
q_off = ql - (q_off + 1);
t_off = tl - (t_off + 1);
```

If `q_off` was returned as a padding position (e.g., `q_off = 471` for
`qlen = 469`), the subtraction underflows:

    q_off = 469 - (471 + 1) = -3

The code then passes this negative offset directly to `mm_align_pair`:

```c
// align.c:947
mm_align_pair(km, opt, ql - q_off, qseq + q_off, ...);
//                                  ^^^^^^^^^^^^^
//                                  reads 3 bytes before buffer start
```

`qseq + q_off` with `q_off = -3` is pointer arithmetic before the
allocated object — undefined behavior per C11 §6.5.6¶8.

## Concrete example

Read `SRR8858432.799`, mapped against GRCh38 chr20 in `map-pb` preset:

| Value | Detail |
|-------|--------|
| `qlen` | 469 |
| `slen` | `ceil(469/8) = 59` |
| `qlen8` | `59 * 8 = 472` |
| raw `q_off` from `ksw_ll_i16` | 471 (padding position) |
| adjusted `q_off` | `469 - 472 = -3` |
| behavior | reads 3 bytes before `qseq` buffer |

minimap2 produces a secondary inversion record (`tp:A:i`) from this. The
record has `cm:i:0, s1:i:0` — no chaining support, consistent with being
bootstrapped from garbage data.

## Proposed fix (not applied)

The fix is a one-line filter in the `query_end` scan in `ksw2_ll_sse.c`.
Reject any position that maps beyond the actual query length:

```c
// ksw2_ll_sse.c:149-150, proposed fix
for (i = 0, H8 = (uint16_t*)Hmax; i < qlen8; ++i) {
    if ((int)H8[i] == gmax) {
        int pos = i / 8 + i % 8 * slen;
        if (pos < qlen) *qe = pos;     // ← reject padding positions
    }
}
```

When the maximum score exists *only* at padding positions (no real query
position achieves `gmax`), `*qe` stays at its initial value of -1 and the
caller should treat the alignment as not found. The equivalent change in
`ksw2_ll_neon.c` and any other SIMD variant is identical.

## rammap's handling

rammap applies the clamp above in its Rust reimplementation of
`ksw_ll_i16` (commit `0ed9642`). Additionally, `try_align_inversion`
includes an explicit bounds check that minimap2 lacks:

```rust
// pipeline.rs — after offset adjustment
if q_off < 0 || q_off >= ql || t_off < 0 || t_off >= tl {
    return None;
}
```

This means rammap never produces the 6 inversion records that minimap2
generates via UB. These records all have zero chaining support (`cm:i:0,
s1:i:0`) and are low-confidence secondary alignments.
