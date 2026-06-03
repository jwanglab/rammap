// Lightweight Smith-Waterman and global alignment kernels

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use super::common::simd_compat::*;

use super::common::*;
use super::single::extend_single_affine;

// ============================================================================
// Lightweight Smith-Waterman
// Used for quick inversion scoring.
// ============================================================================

/// Query profile for lightweight i16 Smith-Waterman (lightweight_profile_init, size=2 only)
pub struct LightweightProfile {
    pub qlen: i32,
    pub segment_len: i32,     // segmented length = ceil(qlen / 8)
    pub query_profile: Vec<i16>,  // query profile: m * segment_len * 8 values
    pub h0: Vec<i16>,  // segment_len * 8
    pub h1: Vec<i16>,  // segment_len * 8
    pub e: Vec<i16>,   // segment_len * 8
    pub hmax: Vec<i16>, // segment_len * 8
}

/// Initialize query profile for lightweight i16 Smith-Waterman.
/// Initialize lightweight query profile (size=2 / int16 only).
pub fn lightweight_profile_init(qlen: i32, query: &[u8], alphabet_size: i32, score_matrix: &[i8]) -> LightweightProfile {
    let p = 8i32; // 8 int16 values per __m128i
    let slen = (qlen + p - 1) / p;
    let m_usize = alphabet_size as usize;

    // Build segmented query profile (int16)
    // Layout: for each alphabet char a (0..m), for each segment i (0..slen),
    // store p values at positions k = i, i+slen, i+2*slen, ...
    let mut qp = vec![0i16; m_usize * slen as usize * p as usize];
    {
        let mut t = 0usize;
        for a in 0..m_usize {
            let nlen = (slen * p) as usize;
            let ma = &score_matrix[a * m_usize..a * m_usize + m_usize];
            for i in 0..slen as usize {
                let mut k = i;
                while k < nlen {
                    qp[t] = if (k as i32) >= qlen {
                        -1
                    } else {
                        ma[query[k] as usize] as i16
                    };
                    t += 1;
                    k += slen as usize;
                }
            }
        }
    }

    let sz = slen as usize * p as usize;
    LightweightProfile {
        qlen,
        segment_len: slen,
        query_profile: qp,
        h0: vec![0i16; sz],
        h1: vec![0i16; sz],
        e: vec![0i16; sz],
        hmax: vec![0i16; sz],
    }
}

/// Lightweight i16 Smith-Waterman local alignment.
/// Lightweight i16 Smith-Waterman local alignment.
/// Returns (score, query_end, target_end). query_end/target_end are -1 if no alignment found.
#[cfg(target_arch = "x86_64")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_sse2(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(target_arch = "aarch64")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_neon(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(target_arch = "wasm32")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_wasm(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    lightweight_align_i16_scalar(qp, target_len, target, gap_open, gap_extend)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn lightweight_align_i16_sse2(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) { unsafe {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    let zero = _mm_set1_epi32(0);
    let gapoe = _mm_set1_epi16((gap_open + gap_extend) as i16);
    let gape_v = _mm_set1_epi16(gap_extend as i16);

    // Zero out working arrays
    for v in qp.e.iter_mut() { *v = 0; }
    for v in qp.h0.iter_mut() { *v = 0; }
    for v in qp.hmax.iter_mut() { *v = 0; }

    for i in 0..target_len {
        let mut f = zero;
        let mut max = zero;
        let s_offset = target[i as usize] as usize * slen as usize * 8;

        // h = H0[slen-1] shifted left by 2 bytes
        let h0_last_idx = (slen as usize - 1) * 8;
        let mut h = _mm_loadu_si128(qp.h0[h0_last_idx..].as_ptr() as *const __m128i);
        h = _mm_slli_si128(h, 2);

        for j in 0..slen as usize {
            let s = _mm_loadu_si128(qp.query_profile[s_offset + j * 8..].as_ptr() as *const __m128i);
            h = _mm_adds_epi16(h, s);
            let e = _mm_loadu_si128(qp.e[j * 8..].as_ptr() as *const __m128i);
            h = _mm_max_epi16(h, e);
            h = _mm_max_epi16(h, f);
            max = _mm_max_epi16(max, h);
            _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h);
            let h_sub = _mm_subs_epu16(h, gapoe);
            let e_sub = _mm_subs_epu16(e, gape_v);
            let e_new = _mm_max_epi16(e_sub, h_sub);
            _mm_storeu_si128(qp.e[j * 8..].as_mut_ptr() as *mut __m128i, e_new);
            f = _mm_subs_epu16(f, gape_v);
            f = _mm_max_epi16(f, h_sub);
            h = _mm_loadu_si128(qp.h0[j * 8..].as_ptr() as *const __m128i);
        }

        // F-wave propagation
        for _k in 0..8 {
            f = _mm_slli_si128(f, 2);
            let mut did_break = false;
            for j in 0..slen as usize {
                let mut h1 = _mm_loadu_si128(qp.h1[j * 8..].as_ptr() as *const __m128i);
                h1 = _mm_max_epi16(h1, f);
                _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h1);
                let h1_sub = _mm_subs_epu16(h1, gapoe);
                f = _mm_subs_epu16(f, gape_v);
                if _mm_movemask_epi8(_mm_cmpgt_epi16(f, h1_sub)) == 0 {
                    did_break = true;
                    break;
                }
            }
            if did_break { break; }
        }

        // __max_8: find max of 8 int16 values
        let mut imax_v = max;
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 8));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 4));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 2));
        let imax = _mm_extract_epi16(imax_v, 0) as i16 as i32;

        if imax >= gmax {
            gmax = imax;
            target_end = i;
            qp.hmax.copy_from_slice(&qp.h1);
        }

        // Swap H0 and H1
        std::mem::swap(&mut qp.h0, &mut qp.h1);
    }

    // Find query_end from Hmax. Prefer an in-bounds position with value == gmax;
    // fall back to the highest-scoring in-bounds position when gmax was achieved
    // only at SIMD padding cells. The score returned is still gmax.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen { query_end = pos; }
        }
    }
    if query_end < 0 {
        let mut best_val: i32 = i32::MIN;
        for i in 0..qlen8 {
            let val = qp.hmax[i as usize] as i32;
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen && val > best_val {
                best_val = val;
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}}

#[cfg(target_arch = "aarch64")]
unsafe fn lightweight_align_i16_neon(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) { unsafe {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    let zero = vdupq_n_s16(0);
    let gapoe = vdupq_n_u16((gap_open + gap_extend) as u16);
    let gape_v = vdupq_n_u16(gap_extend as u16);

    for v in qp.e.iter_mut() { *v = 0; }
    for v in qp.h0.iter_mut() { *v = 0; }
    for v in qp.hmax.iter_mut() { *v = 0; }

    for i in 0..target_len {
        let mut f = zero;
        let mut max = zero;
        let s_offset = target[i as usize] as usize * slen as usize * 8;

        let h0_last_idx = (slen as usize - 1) * 8;
        let mut h = vld1q_s16(qp.h0[h0_last_idx..].as_ptr());
        h = vextq_s16(vdupq_n_s16(0), h, 7);

        for j in 0..slen as usize {
            let s = vld1q_s16(qp.query_profile[s_offset + j * 8..].as_ptr());
            h = vqaddq_s16(h, s);
            let e = vld1q_s16(qp.e[j * 8..].as_ptr());
            h = vmaxq_s16(h, e);
            h = vmaxq_s16(h, f);
            max = vmaxq_s16(max, h);
            vst1q_s16(qp.h1[j * 8..].as_mut_ptr(), h);
            let h_sub = vqsubq_u16(vreinterpretq_u16_s16(h), gapoe);
            let e_sub = vqsubq_u16(vreinterpretq_u16_s16(e), gape_v);
            let e_new = vmaxq_s16(vreinterpretq_s16_u16(e_sub), vreinterpretq_s16_u16(h_sub));
            vst1q_s16(qp.e[j * 8..].as_mut_ptr(), e_new);
            f = vreinterpretq_s16_u16(vqsubq_u16(vreinterpretq_u16_s16(f), gape_v));
            f = vmaxq_s16(f, vreinterpretq_s16_u16(h_sub));
            h = vld1q_s16(qp.h0[j * 8..].as_ptr());
        }

        // F-wave propagation
        for _k in 0..8 {
            f = vextq_s16(vdupq_n_s16(0), f, 7);
            let mut did_break = false;
            for j in 0..slen as usize {
                let mut h1 = vld1q_s16(qp.h1[j * 8..].as_ptr());
                h1 = vmaxq_s16(h1, f);
                vst1q_s16(qp.h1[j * 8..].as_mut_ptr(), h1);
                let h1_sub = vqsubq_u16(vreinterpretq_u16_s16(h1), gapoe);
                f = vreinterpretq_s16_u16(vqsubq_u16(vreinterpretq_u16_s16(f), gape_v));
                let cmp = vcgtq_s16(f, vreinterpretq_s16_u16(h1_sub));
                // Check if all lanes are zero (no f > h1_sub)
                let any_set = vmaxvq_u16(cmp);
                if any_set == 0 {
                    did_break = true;
                    break;
                }
            }
            if did_break { break; }
        }

        // Find max of 8 int16 values
        let imax = vmaxvq_s16(max) as i32;

        if imax >= gmax {
            gmax = imax;
            target_end = i;
            qp.hmax.copy_from_slice(&qp.h1);
        }

        std::mem::swap(&mut qp.h0, &mut qp.h1);
    }

    // Find query_end from Hmax. Prefer an in-bounds position with value == gmax;
    // fall back to the highest-scoring in-bounds position when gmax was achieved
    // only at SIMD padding cells. The score returned is still gmax.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen { query_end = pos; }
        }
    }
    if query_end < 0 {
        let mut best_val: i32 = i32::MIN;
        for i in 0..qlen8 {
            let val = qp.hmax[i as usize] as i32;
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen && val > best_val {
                best_val = val;
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}}

/// WASM SIMD128 implementation of lightweight_align_i16
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn lightweight_align_i16_wasm(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    // SIMD region: every intrinsic call below is an `unsafe fn` (the
    // simd_compat shims wrap raw-pointer v128 loads/stores), so the whole
    // computation runs inside one inner `unsafe` block per Rust 2024's
    // unsafe_op_in_unsafe_fn. Nothing declared here escapes the block.
    unsafe {
        let zero = _mm_set1_epi32(0);
        let gapoe = _mm_set1_epi16((gap_open + gap_extend) as i16);
        let gape_v = _mm_set1_epi16(gap_extend as i16);

        for v in qp.e.iter_mut() { *v = 0; }
        for v in qp.h0.iter_mut() { *v = 0; }
        for v in qp.hmax.iter_mut() { *v = 0; }

        for i in 0..target_len {
            let mut f = zero;
            let mut max = zero;
            let s_offset = target[i as usize] as usize * slen as usize * 8;
            let h0_last_idx = (slen as usize - 1) * 8;
            let mut h = _mm_loadu_si128(qp.h0[h0_last_idx..].as_ptr() as *const __m128i);
            h = _mm_slli_si128(h, 2);

            for j in 0..slen as usize {
                let s = _mm_loadu_si128(qp.query_profile[s_offset + j * 8..].as_ptr() as *const __m128i);
                h = _mm_adds_epi16(h, s);
                let e = _mm_loadu_si128(qp.e[j * 8..].as_ptr() as *const __m128i);
                h = _mm_max_epi16(h, e);
                h = _mm_max_epi16(h, f);
                max = _mm_max_epi16(max, h);
                _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h);
                let h_sub = _mm_subs_epu16(h, gapoe);
                let e_sub = _mm_subs_epu16(e, gape_v);
                let e_new = _mm_max_epi16(e_sub, h_sub);
                _mm_storeu_si128(qp.e[j * 8..].as_mut_ptr() as *mut __m128i, e_new);
                f = _mm_subs_epu16(f, gape_v);
                f = _mm_max_epi16(f, h_sub);
                h = _mm_loadu_si128(qp.h0[j * 8..].as_ptr() as *const __m128i);
            }

            for _k in 0..8 {
                f = _mm_slli_si128(f, 2);
                let mut did_break = false;
                for j in 0..slen as usize {
                    let mut h1 = _mm_loadu_si128(qp.h1[j * 8..].as_ptr() as *const __m128i);
                    h1 = _mm_max_epi16(h1, f);
                    _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h1);
                    let h1_sub = _mm_subs_epu16(h1, gapoe);
                    f = _mm_subs_epu16(f, gape_v);
                    if _mm_movemask_epi8(_mm_cmpgt_epi16(f, h1_sub)) == 0 {
                        did_break = true;
                        break;
                    }
                }
                if did_break { break; }
            }

            let mut imax_v = max;
            imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 8));
            imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 4));
            imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 2));
            let imax = _mm_extract_epi16::<0>(imax_v) as i16 as i32;

            if imax >= gmax {
                gmax = imax;
                target_end = i;
                qp.hmax.copy_from_slice(&qp.h1);
            }

            std::mem::swap(&mut qp.h0, &mut qp.h1);
        }
    }

    // Find query_end from Hmax. Prefer an in-bounds position with value == gmax;
    // fall back to the highest-scoring in-bounds position when gmax was achieved
    // only at SIMD padding cells. The score returned is still gmax.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen { query_end = pos; }
        }
    }
    if query_end < 0 {
        let mut best_val: i32 = i32::MIN;
        for i in 0..qlen8 {
            let val = qp.hmax[i as usize] as i32;
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen && val > best_val {
                best_val = val;
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}

/// Scalar fallback for lightweight_align_i16
pub fn lightweight_align_i16_scalar(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    // Simple scalar Smith-Waterman for non-SIMD platforms
    let qlen = qp.qlen as usize;
    let target_len = target_len as usize;
    let _m = 5usize; // alphabet size

    // Reconstruct scoring from query profile
    // qp.query_profile is segmented; we need to unsegment to get mat scores
    let slen = qp.segment_len as usize;
    let p = 8usize;

    let mut h_prev = vec![0i32; qlen + 1];
    let mut h_curr = vec![0i32; qlen + 1];
    let mut e_arr = vec![0i32; qlen + 1];
    let mut gmax: i32 = 0;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    for (i, &t_base) in target[..target_len].iter().enumerate() {
        let tc = t_base as usize;
        let mut f: i32 = 0;
        for j in 1..=qlen {
            // Get score from query profile: unsegment index
            // segmented index for query position (j-1): seg = (j-1) % slen, lane = (j-1) / slen
            // offset in qp.query_profile: tc * slen * p + seg * p + lane
            let seg = (j - 1) % slen;
            let lane = (j - 1) / slen;
            let score = if lane < p {
                qp.query_profile[tc * slen * p + seg * p + lane] as i32
            } else {
                0
            };

            let mut h = h_prev[j - 1] + score;
            if h < 0 { h = 0; }

            e_arr[j] = std::cmp::max(e_arr[j].saturating_sub(gap_extend), h_curr.get(j).copied().unwrap_or(0).saturating_sub(gap_open + gap_extend));
            // Wait, this is wrong - we need unsigned saturation semantics
            // Let's just do it simply
            let e_val = std::cmp::max(
                (e_arr[j] as i64 - gap_extend as i64).max(0) as i32,
                (h as i64 - (gap_open + gap_extend) as i64).max(0) as i32,
            );
            e_arr[j] = e_val;

            f = std::cmp::max(
                (f as i64 - gap_extend as i64).max(0) as i32,
                (h as i64 - (gap_open + gap_extend) as i64).max(0) as i32,
            );

            h = std::cmp::max(h, std::cmp::max(e_arr[j], f));
            h_curr[j] = h;

            if h >= gmax {
                gmax = h;
                target_end = i as i32;
                query_end = (j - 1) as i32;
            }
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
        for v in h_curr.iter_mut() { *v = 0; }
    }

    (gmax, query_end, target_end)
}


/// Global alignment (Needleman-Wunsch) with CIGAR traceback.
///
/// Dispatches to the best available implementation:
/// - SIMD targets (x86_64/aarch64/wasm32): SIMD extension with end_bonus
///   (AVX512 > AVX2 > SSE2 > NEON > WASM SIMD128)
/// - Non-SIMD targets: row-by-row Gotoh NW (scalar fallback)
///
/// Both produce correct end-to-end CIGARs. The SIMD path recomputes
/// the score from the CIGAR (stripping the end_bonus inflation).
pub fn global_align(
    qseq: &[u8], tseq: &[u8],
    alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let qlen = qseq.len();
    let tlen = tseq.len();
    if qlen == 0 || tlen == 0 {
        if qlen > 0 { result.cigar = vec![(qlen as u32) << 4 | 1]; result.score = -(gap_open + gap_extend * qlen as i32); }
        if tlen > 0 { result.cigar = vec![(tlen as u32) << 4 | 2]; result.score = -(gap_open + gap_extend * tlen as i32); }
        return;
    }

    // Use SIMD extension on SIMD-capable targets (faster at all lengths).
    // Gotoh scalar NW is the fallback for non-SIMD architectures.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
    let has_simd = !*crate::align::env_flags::FORCE_SCALAR;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    let has_simd = false;

    if has_simd {
        global_align_simd(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, result);
    } else {
        global_align_gotoh(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, result);
    }
}

/// SIMD-accelerated global alignment via extension DP with end_bonus.
///
/// Uses the same SIMD kernels as the mapper (AVX512/AVX2/SSE2). The large
/// end_bonus forces the alignment to cover both sequences end-to-end.
/// Score is recomputed from CIGAR for correctness (end_bonus inflates ez.score).
fn global_align_simd(
    qseq: &[u8], tseq: &[u8],
    alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let end_bonus = (score_matrix[0] as i32 * std::cmp::max(qseq.len(), tseq.len()) as i32).max(1000);
    let bw = if bandwidth > 0 { bandwidth } else { -1 };
    extend_single_affine(
        qseq, tseq, alphabet_size, score_matrix,
        gap_open as i8, gap_extend as i8,
        bw, -1, end_bonus, APPROX_MAX, result,
    );
    // Recompute score from CIGAR (strip end_bonus inflation)
    let m = alphabet_size as usize;
    let mut score = 0i32;
    let mut qi = 0usize;
    let mut ti = 0usize;
    for &c in &result.cigar {
        let len = (c >> 4) as usize;
        match c & 0xf {
            0 => {
                for _ in 0..len {
                    if qi < qseq.len() && ti < tseq.len() {
                        score += score_matrix[tseq[ti].min(4) as usize * m + qseq[qi].min(4) as usize] as i32;
                    }
                    qi += 1; ti += 1;
                }
            }
            1 => { score -= gap_open + gap_extend * len as i32; qi += len; }
            2 => { score -= gap_open + gap_extend * len as i32; ti += len; }
            _ => {}
        }
    }
    result.score = score;
}

/// Row-by-row Gotoh NW with banded backtrack matrix.
///
/// Row-by-row Gotoh NW (scalar fallback for non-SIMD architectures).
/// Simple H[i][j], E[j], F recurrence with banded backtrack matrix.
fn global_align_gotoh(
    qseq: &[u8], tseq: &[u8],
    _alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let qlen = qseq.len();
    let tlen = tseq.len();
    let m = 5usize; // nt4 alphabet
    let gapoe = gap_open + gap_extend;
    let w = if bandwidth > 0 { bandwidth as usize } else { qlen + tlen };

    // DP arrays (current and previous row of H, plus E for query gaps)
    let mut h_prev = vec![NEG_INF; tlen + 1];
    let mut h_curr = vec![NEG_INF; tlen + 1];
    let mut e = vec![NEG_INF; tlen + 1]; // E[j]: best score ending with query gap at column j

    // Initialize first row: H[0][j] = gap penalties
    h_prev[0] = 0;
    for j in 1..=tlen {
        h_prev[j] = -(gapoe + gap_extend * (j as i32 - 1));
        e[j] = NEG_INF;
    }

    // Backtrack matrix: 2 bits per cell (0=diag, 1=up/I, 2=left/D)
    let mut bt = vec![0u8; qlen * tlen];

    for i in 1..=qlen {
        let mut f = NEG_INF; // F: best score ending with target gap (current row)
        h_curr[0] = -(gapoe + gap_extend * (i as i32 - 1));

        // Band boundaries
        let j_start = if w < i { i - w } else { 1 };
        let j_end = std::cmp::min(tlen, i + w);

        for j in j_start..=j_end {
            // Match/mismatch from diagonal
            let s = score_matrix[tseq[j - 1] as usize * m + qseq[i - 1] as usize] as i32;
            let diag = h_prev[j - 1] + s;

            // Query gap (insertion): extend or open from H
            let e_ext = e[j] - gap_extend;
            let e_open = h_prev[j] - gapoe;
            e[j] = std::cmp::max(e_ext, e_open);

            // Target gap (deletion): extend or open from H
            let f_ext = f - gap_extend;
            let f_open = h_curr[j - 1] - gapoe;
            f = std::cmp::max(f_ext, f_open);

            // Best of three
            let h = std::cmp::max(diag, std::cmp::max(e[j], f));
            h_curr[j] = h;

            // Backtrack direction
            let d = if h == diag { 0u8 }
                    else if h == e[j] { 1u8 }
                    else { 2u8 };
            bt[(i - 1) * tlen + (j - 1)] = d;
        }

        std::mem::swap(&mut h_prev, &mut h_curr);
        h_curr.fill(NEG_INF);
    }

    result.score = h_prev[tlen];

    // Traceback from (qlen, tlen)
    let mut cigar = Vec::new();
    let mut i = qlen;
    let mut j = tlen;

    while i > 0 && j > 0 {
        let d = bt[(i - 1) * tlen + (j - 1)];
        match d {
            0 => { push_cigar(&mut cigar, 0, 1); i -= 1; j -= 1; } // M
            1 => { push_cigar(&mut cigar, 1, 1); i -= 1; }          // I (consume query)
            _ => { push_cigar(&mut cigar, 2, 1); j -= 1; }          // D (consume target)
        }
    }
    if i > 0 { push_cigar(&mut cigar, 1, i as u32); }
    if j > 0 { push_cigar(&mut cigar, 2, j as u32); }

    cigar.reverse();
    result.cigar = cigar;
}
