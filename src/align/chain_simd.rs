//! SIMD-optimized chaining for x86_64 (AVX2), aarch64 (NEON), and wasm32 (SIMD128).
//!
//! Two-phase approach:
//! - Phase 1 (SIMD): Batch-compute chain scores for all predecessors using SIMD
//! - Phase 2 (Scalar): Apply max_chain_skip / visited logic over precomputed scores
//!
//! This guarantees bit-exact output with the scalar implementation while
//! getting 3-5x speedup on the score computation hot path.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use crate::align::chain::{chain_backtrack, compute_chain_score, fast_log2};
use crate::align::map::{ChainingParams, ChainingBuffers};
use crate::align::sketch::Minimizer;
use crate::align::sort::radix_sort_128x;

/// Vectorized fast_log2 for 8 floats (AVX2).
/// Matches the scalar `fast_log2` exactly (same polynomial, same rounding).
/// MUST NOT use FMA — use explicit mul+add to match scalar rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_fast_log2_avx2(x: __m256) -> __m256 {
    let z = _mm256_castps_si256(x);

    // exponent = ((z >> 23) & 255) - 128
    let exp_raw = _mm256_and_si256(_mm256_srli_epi32(z, 23), _mm256_set1_epi32(255));
    let exp = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(128));
    let log_2 = _mm256_cvtepi32_ps(exp);

    // mantissa: clear exponent bits, set exponent to 127 (1.0 bias)
    let mant_bits = _mm256_or_si256(
        _mm256_and_si256(z, _mm256_set1_epi32(!(255i32 << 23))),
        _mm256_set1_epi32(127i32 << 23),
    );
    let z_f = _mm256_castsi256_ps(mant_bits);

    // Horner's: ((-0.34484843 * z_f) + 2.0246658) * z_f + (-0.6748776)
    let c0 = _mm256_set1_ps(-0.34484843f32);
    let c1 = _mm256_set1_ps(2.024_665_8f32);
    let c2 = _mm256_set1_ps(-0.674_877_6f32);

    let t1 = _mm256_mul_ps(c0, z_f);        // c0 * z_f
    let t2 = _mm256_add_ps(t1, c1);          // c0 * z_f + c1
    let t3 = _mm256_mul_ps(t2, z_f);         // (c0 * z_f + c1) * z_f
    let poly = _mm256_add_ps(t3, c2);        // ... + c2

    _mm256_add_ps(log_2, poly)
}

/// Batch-compute chain scores for predecessors j in [start, end) against anchor i.
/// Writes scores to sc_buf[start..end]. Invalid pairs get i32::MIN.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_chain_scores_batch_avx2(
    qi: i32,
    ri: i32,
    rid_strand_i: u32,
    soa_ref_pos: &[i32],
    soa_query_pos: &[i32],
    soa_query_span: &[i32],
    soa_rid_strand: &[u32],
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    sc_buf: &mut [i32],
    start: usize,
    end: usize,
) { unsafe {
    if start >= end {
        return;
    }

    let qi_v = _mm256_set1_epi32(qi);
    let ri_v = _mm256_set1_epi32(ri);
    let rid_i_v = _mm256_set1_epi32(rid_strand_i as i32);
    let max_dx_v = _mm256_set1_epi32(max_dist_x);
    let max_dy_v = _mm256_set1_epi32(max_dist_y);
    let bw_v = _mm256_set1_epi32(bw);
    let zero_v = _mm256_setzero_si256();
    let one_v = _mm256_set1_epi32(1);
    let min_val_v = _mm256_set1_epi32(i32::MIN);
    let pen_gap_v = _mm256_set1_ps(chn_pen_gap);
    let pen_skip_v = _mm256_set1_ps(chn_pen_skip);
    let half_v = _mm256_set1_ps(0.5f32);

    // Process 8 predecessors at a time
    let aligned_end = start + ((end - start) / 8) * 8;
    let mut j = start;

    while j < aligned_end {
        // Load 8 predecessors' SoA data
        let qj_v = _mm256_loadu_si256(soa_query_pos.as_ptr().add(j) as *const __m256i);
        let rj_v = _mm256_loadu_si256(soa_ref_pos.as_ptr().add(j) as *const __m256i);
        let span_v = _mm256_loadu_si256(soa_query_span.as_ptr().add(j) as *const __m256i);
        let rid_j_v = _mm256_loadu_si256(soa_rid_strand.as_ptr().add(j) as *const __m256i);

        // query_diff = qi - qj (wrapping subtraction, matches Rust wrapping_sub)
        let qdiff = _mm256_sub_epi32(qi_v, qj_v);
        // ref_diff = ri - rj
        let rdiff = _mm256_sub_epi32(ri_v, rj_v);

        // --- Build validity mask ---
        // 1. query_diff > 0
        let mut valid = _mm256_cmpgt_epi32(qdiff, zero_v);

        // 2. query_diff <= max_dist_x  (i.e., NOT (query_diff > max_dist_x))
        let qdiff_gt_max_dx = _mm256_cmpgt_epi32(qdiff, max_dx_v);
        valid = _mm256_andnot_si256(qdiff_gt_max_dx, valid);

        // 3. Same ref_id_strand
        let same_rid = _mm256_cmpeq_epi32(rid_i_v, rid_j_v);
        valid = _mm256_and_si256(valid, same_rid);

        // 4. ref_diff != 0 (same_rid already ensures same target)
        let rdiff_zero = _mm256_cmpeq_epi32(rdiff, zero_v);
        valid = _mm256_andnot_si256(rdiff_zero, valid);

        // 5. query_diff <= max_dist_y  (NOT (qdiff > max_dy))
        let qdiff_gt_max_dy = _mm256_cmpgt_epi32(qdiff, max_dy_v);
        valid = _mm256_andnot_si256(qdiff_gt_max_dy, valid);

        // 6. gap_width = abs(ref_diff - query_diff); gap_width <= bw
        let diff = _mm256_sub_epi32(rdiff, qdiff);
        let gap_w = _mm256_abs_epi32(diff);
        let gw_gt_bw = _mm256_cmpgt_epi32(gap_w, bw_v);
        valid = _mm256_andnot_si256(gw_gt_bw, valid);

        // Also check: n_seg > 1 case — ref_diff > max_dist_y
        // Not needed here since we only enter SIMD for n_seg==1, !is_cdna

        // --- Score computation ---
        // min_diff = min(ref_diff, query_diff)
        let min_diff = _mm256_min_epi32(rdiff, qdiff);

        // score = min(q_span_j, min_diff)
        let mut score = _mm256_min_epi32(span_v, min_diff);

        // --- Penalty computation ---
        // needs_penalty = (gap_width > 0) || (min_diff > q_span)
        let gw_gt_zero = _mm256_cmpgt_epi32(gap_w, zero_v);
        let md_gt_span = _mm256_cmpgt_epi32(min_diff, span_v);
        let needs_pen = _mm256_or_si256(gw_gt_zero, md_gt_span);

        // lin_pen = chn_pen_gap * gap_width + chn_pen_skip * min_diff
        let gap_w_f = _mm256_cvtepi32_ps(gap_w);
        let min_diff_f = _mm256_cvtepi32_ps(min_diff);
        let mul1 = _mm256_mul_ps(pen_gap_v, gap_w_f);
        let mul2 = _mm256_mul_ps(pen_skip_v, min_diff_f);
        let lin_pen = _mm256_add_ps(mul1, mul2);

        // log_pen = fast_log2(gap_width + 1) if gap_width >= 1, else 0
        let gw_plus1 = _mm256_add_epi32(gap_w, one_v);
        let gw_plus1_f = _mm256_cvtepi32_ps(gw_plus1);
        let log2_val = simd_fast_log2_avx2(gw_plus1_f);
        // Mask: zero out log_pen where gap_width == 0
        let log_pen = _mm256_and_ps(_mm256_castsi256_ps(gw_gt_zero), log2_val);

        // total_pen = lin_pen + 0.5 * log_pen
        let half_log = _mm256_mul_ps(half_v, log_pen);
        let total_pen = _mm256_add_ps(lin_pen, half_log);

        // Convert to i32 (truncation toward zero, matches Rust `as i32`)
        let pen_i32 = _mm256_cvttps_epi32(total_pen);

        // Apply penalty only where needs_pen is set
        let pen_masked = _mm256_and_si256(needs_pen, pen_i32);
        score = _mm256_sub_epi32(score, pen_masked);

        // Mask invalid scores to i32::MIN
        score = _mm256_blendv_epi8(min_val_v, score, valid);

        // Store 8 scores
        _mm256_storeu_si256(sc_buf.as_mut_ptr().add(j) as *mut __m256i, score);

        j += 8;
    }

    // Scalar remainder
    while j < end {
        // We can't call compute_chain_score directly because it takes Minimizer refs,
        // but we have SoA data. Recompute from SoA fields to match exactly.
        let qj = soa_query_pos[j];
        let rj = soa_ref_pos[j];
        let span_j = soa_query_span[j];
        let rid_j = soa_rid_strand[j];

        let query_diff = qi.wrapping_sub(qj);
        if query_diff <= 0 || query_diff > max_dist_x {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let ref_diff = ri.wrapping_sub(rj);
        if rid_strand_i == rid_j && (ref_diff == 0 || query_diff > max_dist_y) {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let gap_width = (ref_diff - query_diff).abs();
        if rid_strand_i == rid_j && gap_width > bw {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let min_diff = ref_diff.min(query_diff);
        let mut sc = span_j.min(min_diff);

        if gap_width > 0 || min_diff > span_j {
            let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
            let log_pen = if gap_width >= 1 {
                fast_log2((gap_width + 1) as f32)
            } else {
                0.0f32
            };
            sc -= (lin_pen + 0.5f32 * log_pen) as i32;
        }

        sc_buf[j] = sc;
        j += 1;
    }
}}

/// AVX2 SIMD-optimized chaining.
/// Phase 1: Batch-compute scores with SIMD. Phase 2: Scalar skip/visited logic.
/// Produces bit-exact output matching `chain_anchors_scalar`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn chain_anchors_avx2(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { unsafe {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw {
        max_dist_x = bw;
    }
    if max_dist_y < bw {
        max_dist_y = bw;
    }

    let real_max_drop = bw; // non-cdna path
    let no_chain_skip = opt.max_chain_skip <= 0;

    // --- Extract SoA fields from AoS Minimizer array ---
    let mut soa_ref_pos = std::mem::take(&mut ctx.soa_ref_pos);
    let mut soa_query_pos = std::mem::take(&mut ctx.soa_query_pos);
    let mut soa_query_span = std::mem::take(&mut ctx.soa_query_span);
    let mut soa_rid_strand = std::mem::take(&mut ctx.soa_ref_id_strand);
    let mut soa_sc_buf = std::mem::take(&mut ctx.soa_scores_buf);

    soa_ref_pos.resize(n, 0);
    soa_query_pos.resize(n, 0);
    soa_query_span.resize(n, 0);
    soa_rid_strand.resize(n, 0);
    soa_sc_buf.resize(n, 0);

    for j in 0..n {
        soa_ref_pos[j] = a[j].ref_pos();
        soa_query_pos[j] = a[j].query_pos();
        soa_query_span[j] = a[j].query_span();
        soa_rid_strand[j] = (a[j].x >> 32) as u32;
    }

    // --- DP buffers ---
    let mut predecessors = std::mem::take(&mut ctx.predecessors); let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    visited.clear();
    visited.resize(n, 0i32);

    let mut global_max_score = 0;
    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span();
        let mut skip_count: i32 = 0;

        while window_start < i
            && (a[i].ref_id_strand() != a[window_start].ref_id_strand()
                || (a[i].x > a[window_start].x + max_dist_x as u64))
        {
            window_start += 1;
        }

        if !no_chain_skip && (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j: i64 = if window_start > 0 {
            window_start as i64 - 1
        } else {
            -1
        };

        if no_chain_skip {
            // No early termination — compute all scores, then scan
            if i > window_start {
                compute_chain_scores_batch_avx2(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, window_start, i,
                );
            }
            for j in window_start..i {
                let sc = soa_sc_buf[j];
                if sc == i32::MIN { continue; }
                let total_sc = sc + scores[j];
                if total_sc > best_score { best_score = total_sc; best_predecessor = j as i64; }
            }
        } else {
            // Chunked reverse scan with early termination
            const CHUNK_SIZE: usize = 64;
            let mut chunk_end = i;
            let mut skip_fired = false;
            while chunk_end > window_start {
                let chunk_start = if chunk_end >= window_start + CHUNK_SIZE { chunk_end - CHUNK_SIZE } else { window_start };

                compute_chain_scores_batch_avx2(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, chunk_start, chunk_end,
                );

                for j in (chunk_start..chunk_end).rev() {
                    let sc = soa_sc_buf[j];
                    if sc == i32::MIN { continue; }
                    let total_sc = sc + scores[j];
                    if total_sc > best_score {
                        best_score = total_sc; best_predecessor = j as i64;
                        if skip_count > 0 { skip_count -= 1; }
                    } else if visited[j] == i as i32 {
                        skip_count += 1;
                        if skip_count > opt.max_chain_skip { end_j = j as i64; skip_fired = true; break; }
                    }
                    if predecessors[j] >= 0 { visited[predecessors[j] as usize] = i as i32; }
                }

                if skip_fired { break; }
                chunk_end = chunk_start;
            }

            if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
                let mut best = i32::MIN; best_anchor_idx = -1;
                for j in (window_start..i).rev() {
                    if best < scores[j] { best = scores[j]; best_anchor_idx = j as i64; }
                }
            }

            if best_anchor_idx >= 0 && best_anchor_idx < end_j {
                let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize],
                    max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, false, 1);
                if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                    best_score = sc + scores[best_anchor_idx as usize];
                    best_predecessor = best_anchor_idx;
                }
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0
            && peak_scores[best_predecessor as usize] > best_score
        {
            peak_scores[best_predecessor as usize]
        } else {
            best_score
        };

        if best_anchor_idx < 0
            || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64
                && scores[best_anchor_idx as usize] < scores[i])
        {
            best_anchor_idx = i as i64;
        }

        if global_max_score < best_score {
            global_max_score = best_score;
        }
    }


    // --- Backtrack and compact (identical to scalar) ---
    let (u, n_u, n_v) = chain_backtrack(
        n,
        &scores,
        &predecessors,
        &mut peak_scores,
        &mut visited,
        &mut bt_candidates,
        opt.min_cnt,
        opt.min_chain_score,
        real_max_drop,
    );

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates);
        ctx.scores = scores;
        ctx.peak_scores = peak_scores;
        ctx.visited = visited;
        ctx.soa_ref_pos = soa_ref_pos;
        ctx.soa_query_pos = soa_query_pos;
        ctx.soa_query_span = soa_query_span;
        ctx.soa_ref_id_strand = soa_rid_strand;
        ctx.soa_scores_buf = soa_sc_buf;
        return (Vec::new(), Vec::new());
    }

    // compact_a logic — Step 1: Write chain anchors to b[] in forward order
    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0usize;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni {
            let idx = peak_scores[k0 + (ni - j - 1)] as usize;
            b.push(a[idx]);
            k += 1;
        }
    }

    // Step 2: Sort chains by target position of first anchor
    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer {
            x: b[k_pos].x,
            y: ((k_pos as u64) << 32) | (i as u64),
        });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);

    // Step 3: Reorder u[] and anchors according to sorted order
    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize;
        let offset = (w_val.y >> 32) as usize;
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni {
            b2.push(b[offset + idx]);
        }
    }

    // Return buffers to context for reuse
    ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates);
    ctx.scores = scores;
    ctx.peak_scores = peak_scores;
    ctx.visited = visited;
    ctx.soa_ref_pos = soa_ref_pos;
    ctx.soa_query_pos = soa_query_pos;
    ctx.soa_query_span = soa_query_span;
    ctx.soa_ref_id_strand = soa_rid_strand;
    ctx.soa_scores_buf = soa_sc_buf;

    (u2, b2)
}}

// =============================================================================
// SSE (x86_64) implementation — 4-wide, for CPUs without AVX2
// =============================================================================

/// Vectorized fast_log2 for 4 floats (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn simd_fast_log2_sse(x: __m128) -> __m128 {
    let z = _mm_castps_si128(x);
    let exp_raw = _mm_and_si128(_mm_srli_epi32(z, 23), _mm_set1_epi32(255));
    let exp = _mm_sub_epi32(exp_raw, _mm_set1_epi32(128));
    let log_2 = _mm_cvtepi32_ps(exp);
    let mant_bits = _mm_or_si128(
        _mm_and_si128(z, _mm_set1_epi32(!(255i32 << 23))),
        _mm_set1_epi32(127i32 << 23),
    );
    let z_f = _mm_castsi128_ps(mant_bits);
    let c0 = _mm_set1_ps(-0.34484843f32);
    let c1 = _mm_set1_ps(2.024_665_8f32);
    let c2 = _mm_set1_ps(-0.674_877_6f32);
    let t1 = _mm_mul_ps(c0, z_f);
    let t2 = _mm_add_ps(t1, c1);
    let t3 = _mm_mul_ps(t2, z_f);
    let poly = _mm_add_ps(t3, c2);
    _mm_add_ps(log_2, poly)
}

/// SSE2 helper: absolute value of i32x4 (no _mm_abs_epi32 in SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn sse2_abs_epi32(v: __m128i) -> __m128i {
    let neg = _mm_sub_epi32(_mm_setzero_si128(), v);
    let mask = _mm_cmpgt_epi32(v, _mm_setzero_si128());
    _mm_or_si128(_mm_and_si128(mask, v), _mm_andnot_si128(mask, neg))
}

/// SSE2 helper: min of i32x4 (no _mm_min_epi32 in SSE2, added in SSE4.1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn sse2_min_epi32(a: __m128i, b: __m128i) -> __m128i {
    let mask = _mm_cmpgt_epi32(a, b); // mask = a > b
    _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
}

/// Batch-compute chain scores for 4 predecessors at a time (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn compute_chain_scores_batch_sse(
    qi: i32, ri: i32, rid_strand_i: u32,
    soa_ref_pos: &[i32], soa_query_pos: &[i32],
    soa_query_span: &[i32], soa_rid_strand: &[u32],
    max_dist_x: i32, max_dist_y: i32, bw: i32,
    chn_pen_gap: f32, chn_pen_skip: f32,
    sc_buf: &mut [i32], start: usize, end: usize,
) { unsafe {
    if start >= end { return; }

    let qi_v = _mm_set1_epi32(qi);
    let ri_v = _mm_set1_epi32(ri);
    let rid_i_v = _mm_set1_epi32(rid_strand_i as i32);
    let max_dx_v = _mm_set1_epi32(max_dist_x);
    let max_dy_v = _mm_set1_epi32(max_dist_y);
    let bw_v = _mm_set1_epi32(bw);
    let zero_v = _mm_setzero_si128();
    let one_v = _mm_set1_epi32(1);
    let min_val_v = _mm_set1_epi32(i32::MIN);
    let pen_gap_v = _mm_set1_ps(chn_pen_gap);
    let pen_skip_v = _mm_set1_ps(chn_pen_skip);
    let half_v = _mm_set1_ps(0.5f32);

    let aligned_end = start + ((end - start) / 4) * 4;
    let mut j = start;

    while j < aligned_end {
        let qj_v = _mm_loadu_si128(soa_query_pos.as_ptr().add(j) as *const __m128i);
        let rj_v = _mm_loadu_si128(soa_ref_pos.as_ptr().add(j) as *const __m128i);
        let span_v = _mm_loadu_si128(soa_query_span.as_ptr().add(j) as *const __m128i);
        let rid_j_v = _mm_loadu_si128(soa_rid_strand.as_ptr().add(j) as *const __m128i);

        let qdiff = _mm_sub_epi32(qi_v, qj_v);
        let rdiff = _mm_sub_epi32(ri_v, rj_v);

        // Build validity mask (SSE2: vector masks, same as AVX2 style)
        let mut valid = _mm_cmpgt_epi32(qdiff, zero_v);
        valid = _mm_andnot_si128(_mm_cmpgt_epi32(qdiff, max_dx_v), valid);
        valid = _mm_and_si128(valid, _mm_cmpeq_epi32(rid_i_v, rid_j_v));
        valid = _mm_andnot_si128(_mm_cmpeq_epi32(rdiff, zero_v), valid);
        valid = _mm_andnot_si128(_mm_cmpgt_epi32(qdiff, max_dy_v), valid);

        let diff = _mm_sub_epi32(rdiff, qdiff);
        let gap_w = sse2_abs_epi32(diff);
        valid = _mm_andnot_si128(_mm_cmpgt_epi32(gap_w, bw_v), valid);

        let min_diff = sse2_min_epi32(rdiff, qdiff);
        let mut score = sse2_min_epi32(span_v, min_diff);

        // Penalty
        let gw_gt_zero = _mm_cmpgt_epi32(gap_w, zero_v);
        let md_gt_span = _mm_cmpgt_epi32(min_diff, span_v);
        let needs_pen = _mm_or_si128(gw_gt_zero, md_gt_span);

        let gap_w_f = _mm_cvtepi32_ps(gap_w);
        let min_diff_f = _mm_cvtepi32_ps(min_diff);
        let lin_pen = _mm_add_ps(_mm_mul_ps(pen_gap_v, gap_w_f), _mm_mul_ps(pen_skip_v, min_diff_f));

        let gw_plus1_f = _mm_cvtepi32_ps(_mm_add_epi32(gap_w, one_v));
        let log2_val = simd_fast_log2_sse(gw_plus1_f);
        let log_pen = _mm_and_ps(_mm_castsi128_ps(gw_gt_zero), log2_val);

        let total_pen = _mm_add_ps(lin_pen, _mm_mul_ps(half_v, log_pen));
        let pen_i32 = _mm_cvttps_epi32(total_pen);

        let pen_masked = _mm_and_si128(needs_pen, pen_i32);
        score = _mm_sub_epi32(score, pen_masked);

        // Blend: valid ? score : MIN — SSE2 doesn't have _mm_blendv_epi8,
        // so use and/andnot/or
        score = _mm_or_si128(_mm_and_si128(valid, score), _mm_andnot_si128(valid, min_val_v));

        _mm_storeu_si128(sc_buf.as_mut_ptr().add(j) as *mut __m128i, score);
        j += 4;
    }

    // Scalar remainder
    while j < end {
        let qj = soa_query_pos[j];
        let rj = soa_ref_pos[j];
        let span_j = soa_query_span[j];
        let rid_j = soa_rid_strand[j];
        let query_diff = qi.wrapping_sub(qj);
        if query_diff <= 0 || query_diff > max_dist_x {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let ref_diff = ri.wrapping_sub(rj);
        if rid_strand_i == rid_j && (ref_diff == 0 || query_diff > max_dist_y) {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let gap_width = (ref_diff - query_diff).abs();
        if rid_strand_i == rid_j && gap_width > bw {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let min_diff = ref_diff.min(query_diff);
        let mut sc = span_j.min(min_diff);
        if gap_width > 0 || min_diff > span_j {
            let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
            let log_pen = if gap_width >= 1 { fast_log2((gap_width + 1) as f32) } else { 0.0f32 };
            sc -= (lin_pen + 0.5f32 * log_pen) as i32;
        }
        sc_buf[j] = sc;
        j += 1;
    }
}}

/// SSE2 SIMD-optimized chaining — 4-wide, for x86_64 CPUs without AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn chain_anchors_sse(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { unsafe {
    let n = a.len();
    if n == 0 { return (Vec::new(), Vec::new()); }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw { max_dist_x = bw; }
    if max_dist_y < bw { max_dist_y = bw; }

    let real_max_drop = bw;
    let no_chain_skip = opt.max_chain_skip <= 0;

    let mut soa_ref_pos = std::mem::take(&mut ctx.soa_ref_pos);
    let mut soa_query_pos = std::mem::take(&mut ctx.soa_query_pos);
    let mut soa_query_span = std::mem::take(&mut ctx.soa_query_span);
    let mut soa_rid_strand = std::mem::take(&mut ctx.soa_ref_id_strand);
    let mut soa_sc_buf = std::mem::take(&mut ctx.soa_scores_buf);

    soa_ref_pos.resize(n, 0); soa_query_pos.resize(n, 0);
    soa_query_span.resize(n, 0); soa_rid_strand.resize(n, 0);
    soa_sc_buf.resize(n, 0);

    for j in 0..n {
        soa_ref_pos[j] = a[j].ref_pos();
        soa_query_pos[j] = a[j].query_pos();
        soa_query_span[j] = a[j].query_span();
        soa_rid_strand[j] = (a[j].x >> 32) as u32;
    }

    let mut predecessors = std::mem::take(&mut ctx.predecessors); let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32); scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32); visited.clear(); visited.resize(n, 0i32);

    let mut global_max_score = 0;
    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span();
        let mut skip_count: i32 = 0;

        while window_start < i
            && (a[i].ref_id_strand() != a[window_start].ref_id_strand()
                || (a[i].x > a[window_start].x + max_dist_x as u64))
        { window_start += 1; }

        if !no_chain_skip && (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j: i64 = if window_start > 0 { window_start as i64 - 1 } else { -1 };

        if no_chain_skip {
            // No early termination possible — compute all scores, then scan
            if i > window_start {
                compute_chain_scores_batch_sse(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, window_start, i,
                );
            }
            for j in window_start..i {
                let sc = soa_sc_buf[j];
                if sc == i32::MIN { continue; }
                let total_sc = sc + scores[j];
                if total_sc > best_score { best_score = total_sc; best_predecessor = j as i64; }
            }
        } else {
            // Chunked reverse scan: compute batch scores for each chunk, then
            // scan with max_chain_skip early termination. Only compute the next
            // chunk if skip hasn't fired — avoids wasting SIMD on predecessors
            // that will never be examined.
            const CHUNK_SIZE: usize = 64;
            let mut chunk_end = i;
            let mut skip_fired = false;
            while chunk_end > window_start {
                let chunk_start = if chunk_end >= window_start + CHUNK_SIZE {
                    chunk_end - CHUNK_SIZE
                } else {
                    window_start
                };

                compute_chain_scores_batch_sse(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, chunk_start, chunk_end,
                );

                for j in (chunk_start..chunk_end).rev() {
                    let sc = soa_sc_buf[j];
                    if sc == i32::MIN { continue; }
                    let total_sc = sc + scores[j];
                    if total_sc > best_score {
                        best_score = total_sc; best_predecessor = j as i64;
                        if skip_count > 0 { skip_count -= 1; }
                    } else if visited[j] == i as i32 {
                        skip_count += 1;
                        if skip_count > opt.max_chain_skip {
                            end_j = j as i64;
                            skip_fired = true;
                            break;
                        }
                    }
                    if predecessors[j] >= 0 { visited[predecessors[j] as usize] = i as i32; }
                }

                if skip_fired { break; }
                chunk_end = chunk_start;
            }

            if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
                let mut best = i32::MIN; best_anchor_idx = -1;
                for j in (window_start..i).rev() {
                    if best < scores[j] { best = scores[j]; best_anchor_idx = j as i64; }
                }
            }

            if best_anchor_idx >= 0 && best_anchor_idx < end_j {
                let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize],
                    max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, false, 1);
                if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                    best_score = sc + scores[best_anchor_idx as usize];
                    best_predecessor = best_anchor_idx;
                }
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0
            && peak_scores[best_predecessor as usize] > best_score
        { peak_scores[best_predecessor as usize] } else { best_score };

        if best_anchor_idx < 0
            || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64
                && scores[best_anchor_idx as usize] < scores[i])
        { best_anchor_idx = i as i64; }

        if global_max_score < best_score { global_max_score = best_score; }
    }

    let (u, n_u, n_v) = chain_backtrack(n, &scores, &predecessors, &mut peak_scores,
        &mut visited, &mut bt_candidates, opt.min_cnt, opt.min_chain_score, real_max_drop);

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
        ctx.peak_scores = peak_scores; ctx.visited = visited;
        ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
        ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
        ctx.soa_scores_buf = soa_sc_buf;
        return (Vec::new(), Vec::new());
    }

    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0usize;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni { b.push(a[peak_scores[k0 + (ni - j - 1)] as usize]); k += 1; }
    }
    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer { x: b[k_pos].x, y: ((k_pos as u64) << 32) | (i as u64) });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);
    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize;
        let offset = (w_val.y >> 32) as usize;
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni { b2.push(b[offset + idx]); }
    }

    ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
    ctx.peak_scores = peak_scores; ctx.visited = visited;
    ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
    ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
    ctx.soa_scores_buf = soa_sc_buf;
    (u2, b2)
}}

// =============================================================================
// AVX-512 (x86_64) implementation — 16-wide, mirrors AVX2 logic
// =============================================================================

/// Vectorized fast_log2 for 16 floats (AVX-512).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn simd_fast_log2_avx512(x: __m512) -> __m512 {
    let z = _mm512_castps_si512(x);
    let exp_raw = _mm512_and_si512(_mm512_srli_epi32(z, 23), _mm512_set1_epi32(255));
    let exp = _mm512_sub_epi32(exp_raw, _mm512_set1_epi32(128));
    let log_2 = _mm512_cvtepi32_ps(exp);
    let mant_bits = _mm512_or_si512(
        _mm512_and_si512(z, _mm512_set1_epi32(!(255i32 << 23))),
        _mm512_set1_epi32(127i32 << 23),
    );
    let z_f = _mm512_castsi512_ps(mant_bits);
    let c0 = _mm512_set1_ps(-0.34484843f32);
    let c1 = _mm512_set1_ps(2.024_665_8f32);
    let c2 = _mm512_set1_ps(-0.674_877_6f32);
    let t1 = _mm512_mul_ps(c0, z_f);
    let t2 = _mm512_add_ps(t1, c1);
    let t3 = _mm512_mul_ps(t2, z_f);
    let poly = _mm512_add_ps(t3, c2);
    _mm512_add_ps(log_2, poly)
}

/// Batch-compute chain scores for 16 predecessors at a time (AVX-512).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn compute_chain_scores_batch_avx512(
    qi: i32, ri: i32, rid_strand_i: u32,
    soa_ref_pos: &[i32], soa_query_pos: &[i32],
    soa_query_span: &[i32], soa_rid_strand: &[u32],
    max_dist_x: i32, max_dist_y: i32, bw: i32,
    chn_pen_gap: f32, chn_pen_skip: f32,
    sc_buf: &mut [i32], start: usize, end: usize,
) { unsafe {
    if start >= end { return; }

    let qi_v = _mm512_set1_epi32(qi);
    let ri_v = _mm512_set1_epi32(ri);
    let rid_i_v = _mm512_set1_epi32(rid_strand_i as i32);
    let max_dx_v = _mm512_set1_epi32(max_dist_x);
    let max_dy_v = _mm512_set1_epi32(max_dist_y);
    let bw_v = _mm512_set1_epi32(bw);
    let zero_v = _mm512_setzero_si512();
    let one_v = _mm512_set1_epi32(1);
    let min_val_v = _mm512_set1_epi32(i32::MIN);
    let pen_gap_v = _mm512_set1_ps(chn_pen_gap);
    let pen_skip_v = _mm512_set1_ps(chn_pen_skip);
    let half_v = _mm512_set1_ps(0.5f32);

    // Process 16 predecessors at a time
    let aligned_end = start + ((end - start) / 16) * 16;
    let mut j = start;

    while j < aligned_end {
        let qj_v = _mm512_loadu_si512(soa_query_pos.as_ptr().add(j) as *const __m512i);
        let rj_v = _mm512_loadu_si512(soa_ref_pos.as_ptr().add(j) as *const __m512i);
        let span_v = _mm512_loadu_si512(soa_query_span.as_ptr().add(j) as *const __m512i);
        let rid_j_v = _mm512_loadu_si512(soa_rid_strand.as_ptr().add(j) as *const __m512i);

        let qdiff = _mm512_sub_epi32(qi_v, qj_v);
        let rdiff = _mm512_sub_epi32(ri_v, rj_v);

        // AVX-512 comparisons return __mmask16 instead of vector masks
        let mut valid: __mmask16 = _mm512_cmpgt_epi32_mask(qdiff, zero_v);
        valid &= !_mm512_cmpgt_epi32_mask(qdiff, max_dx_v);
        valid &= _mm512_cmpeq_epi32_mask(rid_i_v, rid_j_v);
        valid &= !_mm512_cmpeq_epi32_mask(rdiff, zero_v);
        valid &= !_mm512_cmpgt_epi32_mask(qdiff, max_dy_v);

        let diff = _mm512_sub_epi32(rdiff, qdiff);
        let gap_w = _mm512_abs_epi32(diff);
        valid &= !_mm512_cmpgt_epi32_mask(gap_w, bw_v);

        let min_diff = _mm512_min_epi32(rdiff, qdiff);
        let mut score = _mm512_min_epi32(span_v, min_diff);

        // Penalty computation
        let needs_pen: __mmask16 = _mm512_cmpgt_epi32_mask(gap_w, zero_v)
            | _mm512_cmpgt_epi32_mask(min_diff, span_v);

        let gap_w_f = _mm512_cvtepi32_ps(gap_w);
        let min_diff_f = _mm512_cvtepi32_ps(min_diff);
        let lin_pen = _mm512_add_ps(
            _mm512_mul_ps(pen_gap_v, gap_w_f),
            _mm512_mul_ps(pen_skip_v, min_diff_f),
        );

        let gw_plus1_f = _mm512_cvtepi32_ps(_mm512_add_epi32(gap_w, one_v));
        let log2_val = simd_fast_log2_avx512(gw_plus1_f);
        // Zero out log_pen where gap_width == 0
        let gw_gt_zero = _mm512_cmpgt_epi32_mask(gap_w, zero_v);
        let log_pen = _mm512_maskz_mov_ps(gw_gt_zero, log2_val);

        let total_pen = _mm512_add_ps(lin_pen, _mm512_mul_ps(half_v, log_pen));
        let pen_i32 = _mm512_cvttps_epi32(total_pen);

        // Apply penalty only where needs_pen
        let pen_masked = _mm512_maskz_mov_epi32(needs_pen, pen_i32);
        score = _mm512_sub_epi32(score, pen_masked);

        // Mask invalid scores to i32::MIN
        score = _mm512_mask_blend_epi32(valid, min_val_v, score);

        _mm512_storeu_si512(sc_buf.as_mut_ptr().add(j) as *mut __m512i, score);
        j += 16;
    }

    // Scalar remainder
    while j < end {
        let qj = soa_query_pos[j];
        let rj = soa_ref_pos[j];
        let span_j = soa_query_span[j];
        let rid_j = soa_rid_strand[j];
        let query_diff = qi.wrapping_sub(qj);
        if query_diff <= 0 || query_diff > max_dist_x {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let ref_diff = ri.wrapping_sub(rj);
        if rid_strand_i == rid_j && (ref_diff == 0 || query_diff > max_dist_y) {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let gap_width = (ref_diff - query_diff).abs();
        if rid_strand_i == rid_j && gap_width > bw {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let min_diff = ref_diff.min(query_diff);
        let mut sc = span_j.min(min_diff);
        if gap_width > 0 || min_diff > span_j {
            let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
            let log_pen = if gap_width >= 1 { fast_log2((gap_width + 1) as f32) } else { 0.0f32 };
            sc -= (lin_pen + 0.5f32 * log_pen) as i32;
        }
        sc_buf[j] = sc;
        j += 1;
    }
}}

/// AVX-512 SIMD-optimized chaining — 16-wide, same structure as AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(crate) unsafe fn chain_anchors_avx512(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { unsafe {
    let n = a.len();
    if n == 0 { return (Vec::new(), Vec::new()); }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw { max_dist_x = bw; }
    if max_dist_y < bw { max_dist_y = bw; }

    let real_max_drop = bw;
    let no_chain_skip = opt.max_chain_skip <= 0;

    let mut soa_ref_pos = std::mem::take(&mut ctx.soa_ref_pos);
    let mut soa_query_pos = std::mem::take(&mut ctx.soa_query_pos);
    let mut soa_query_span = std::mem::take(&mut ctx.soa_query_span);
    let mut soa_rid_strand = std::mem::take(&mut ctx.soa_ref_id_strand);
    let mut soa_sc_buf = std::mem::take(&mut ctx.soa_scores_buf);

    soa_ref_pos.resize(n, 0);
    soa_query_pos.resize(n, 0);
    soa_query_span.resize(n, 0);
    soa_rid_strand.resize(n, 0);
    soa_sc_buf.resize(n, 0);

    for j in 0..n {
        soa_ref_pos[j] = a[j].ref_pos();
        soa_query_pos[j] = a[j].query_pos();
        soa_query_span[j] = a[j].query_span();
        soa_rid_strand[j] = (a[j].x >> 32) as u32;
    }

    let mut predecessors = std::mem::take(&mut ctx.predecessors); let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    visited.clear();
    visited.resize(n, 0i32);

    let mut global_max_score = 0;
    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span();
        let mut skip_count: i32 = 0;

        while window_start < i
            && (a[i].ref_id_strand() != a[window_start].ref_id_strand()
                || (a[i].x > a[window_start].x + max_dist_x as u64))
        { window_start += 1; }

        if !no_chain_skip && (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j: i64 = if window_start > 0 { window_start as i64 - 1 } else { -1 };

        if no_chain_skip {
            if i > window_start {
                compute_chain_scores_batch_avx512(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, window_start, i,
                );
            }
            for j in window_start..i {
                let sc = soa_sc_buf[j];
                if sc == i32::MIN { continue; }
                let total_sc = sc + scores[j];
                if total_sc > best_score { best_score = total_sc; best_predecessor = j as i64; }
            }
        } else {
            const CHUNK_SIZE: usize = 64;
            let mut chunk_end = i;
            let mut skip_fired = false;
            while chunk_end > window_start {
                let chunk_start = if chunk_end >= window_start + CHUNK_SIZE { chunk_end - CHUNK_SIZE } else { window_start };

                compute_chain_scores_batch_avx512(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, chunk_start, chunk_end,
                );

                for j in (chunk_start..chunk_end).rev() {
                    let sc = soa_sc_buf[j];
                    if sc == i32::MIN { continue; }
                    let total_sc = sc + scores[j];
                    if total_sc > best_score {
                        best_score = total_sc; best_predecessor = j as i64;
                        if skip_count > 0 { skip_count -= 1; }
                    } else if visited[j] == i as i32 {
                        skip_count += 1;
                        if skip_count > opt.max_chain_skip { end_j = j as i64; skip_fired = true; break; }
                    }
                    if predecessors[j] >= 0 { visited[predecessors[j] as usize] = i as i32; }
                }

                if skip_fired { break; }
                chunk_end = chunk_start;
            }

            if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
                let mut best = i32::MIN; best_anchor_idx = -1;
                for j in (window_start..i).rev() {
                    if best < scores[j] { best = scores[j]; best_anchor_idx = j as i64; }
                }
            }

            if best_anchor_idx >= 0 && best_anchor_idx < end_j {
                let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize],
                    max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, false, 1);
                if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                    best_score = sc + scores[best_anchor_idx as usize];
                    best_predecessor = best_anchor_idx;
                }
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0
            && peak_scores[best_predecessor as usize] > best_score
        { peak_scores[best_predecessor as usize] } else { best_score };

        if best_anchor_idx < 0
            || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64
                && scores[best_anchor_idx as usize] < scores[i])
        { best_anchor_idx = i as i64; }

        if global_max_score < best_score { global_max_score = best_score; }
    }

    let (u, n_u, n_v) = chain_backtrack(
        n, &scores, &predecessors, &mut peak_scores, &mut visited,
        &mut bt_candidates,
        opt.min_cnt, opt.min_chain_score, real_max_drop,
    );

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
        ctx.peak_scores = peak_scores; ctx.visited = visited;
        ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
        ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
        ctx.soa_scores_buf = soa_sc_buf;
        return (Vec::new(), Vec::new());
    }

    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0usize;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni {
            let idx = peak_scores[k0 + (ni - j - 1)] as usize;
            b.push(a[idx]);
            k += 1;
        }
    }

    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer { x: b[k_pos].x, y: ((k_pos as u64) << 32) | (i as u64) });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);

    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize;
        let offset = (w_val.y >> 32) as usize;
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni { b2.push(b[offset + idx]); }
    }

    ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
    ctx.peak_scores = peak_scores; ctx.visited = visited;
    ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
    ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
    ctx.soa_scores_buf = soa_sc_buf;

    (u2, b2)
}}

// =============================================================================
// NEON (aarch64) implementation — 4-wide, mirrors AVX2 logic exactly
// =============================================================================

/// Vectorized fast_log2 for 4 floats (NEON).
/// Matches the scalar `fast_log2` exactly (same polynomial, same rounding).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn simd_fast_log2_neon(x: float32x4_t) -> float32x4_t {
    let z = vreinterpretq_s32_f32(x);

    // exponent = ((z >> 23) & 255) - 128
    let exp_raw = vandq_s32(vshrq_n_s32(z, 23), vdupq_n_s32(255));
    let exp = vsubq_s32(exp_raw, vdupq_n_s32(128));
    let log_2 = vcvtq_f32_s32(exp);

    // mantissa: clear exponent bits, set exponent to 127 (1.0 bias)
    let mant_bits = vorrq_s32(
        vandq_s32(z, vdupq_n_s32(!(255i32 << 23))),
        vdupq_n_s32(127i32 << 23),
    );
    let z_f = vreinterpretq_f32_s32(mant_bits);

    // Horner's: ((-0.34484843 * z_f) + 2.0246658) * z_f + (-0.6748776)
    // MUST NOT use fused multiply-add — use explicit mul+add to match scalar rounding.
    let c0 = vdupq_n_f32(-0.34484843f32);
    let c1 = vdupq_n_f32(2.024_665_8f32);
    let c2 = vdupq_n_f32(-0.674_877_6f32);

    let t1 = vmulq_f32(c0, z_f);       // c0 * z_f
    let t2 = vaddq_f32(t1, c1);         // c0 * z_f + c1
    let t3 = vmulq_f32(t2, z_f);        // (c0 * z_f + c1) * z_f
    let poly = vaddq_f32(t3, c2);       // ... + c2

    vaddq_f32(log_2, poly)
}

/// Batch-compute chain scores for predecessors j in [start, end) against anchor i.
/// Writes scores to sc_buf[start..end]. Invalid pairs get i32::MIN.
/// 4-wide NEON version — mirrors `compute_chain_scores_batch_avx2` exactly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn compute_chain_scores_batch_neon(
    qi: i32,
    ri: i32,
    rid_strand_i: u32,
    soa_ref_pos: &[i32],
    soa_query_pos: &[i32],
    soa_query_span: &[i32],
    soa_rid_strand: &[u32],
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    sc_buf: &mut [i32],
    start: usize,
    end: usize,
) { unsafe {
    if start >= end {
        return;
    }

    let qi_v = vdupq_n_s32(qi);
    let ri_v = vdupq_n_s32(ri);
    let rid_i_v = vdupq_n_s32(rid_strand_i as i32);
    let max_dx_v = vdupq_n_s32(max_dist_x);
    let max_dy_v = vdupq_n_s32(max_dist_y);
    let bw_v = vdupq_n_s32(bw);
    let zero_v = vdupq_n_s32(0);
    let one_v = vdupq_n_s32(1);
    let min_val_v = vdupq_n_s32(i32::MIN);
    let pen_gap_v = vdupq_n_f32(chn_pen_gap);
    let pen_skip_v = vdupq_n_f32(chn_pen_skip);
    let half_v = vdupq_n_f32(0.5f32);

    // Process 4 predecessors at a time
    let aligned_end = start + ((end - start) / 4) * 4;
    let mut j = start;

    while j < aligned_end {
        // Load 4 predecessors' SoA data
        let qj_v = vld1q_s32(soa_query_pos.as_ptr().add(j));
        let rj_v = vld1q_s32(soa_ref_pos.as_ptr().add(j));
        let span_v = vld1q_s32(soa_query_span.as_ptr().add(j));
        let rid_j_v = vld1q_s32(soa_rid_strand.as_ptr().add(j) as *const i32);

        // query_diff = qi - qj
        let qdiff = vsubq_s32(qi_v, qj_v);
        // ref_diff = ri - rj
        let rdiff = vsubq_s32(ri_v, rj_v);

        // --- Build validity mask ---
        // 1. query_diff > 0
        let mut valid = vcgtq_s32(qdiff, zero_v);

        // 2. query_diff <= max_dist_x  (NOT (query_diff > max_dist_x))
        let qdiff_gt_max_dx = vcgtq_s32(qdiff, max_dx_v);
        valid = vbicq_u32(valid, qdiff_gt_max_dx);

        // 3. Same ref_id_strand
        let same_rid = vceqq_s32(rid_i_v, rid_j_v);
        valid = vandq_u32(valid, same_rid);

        // 4. ref_diff != 0
        let rdiff_zero = vceqq_s32(rdiff, zero_v);
        valid = vbicq_u32(valid, rdiff_zero);

        // 5. query_diff <= max_dist_y
        let qdiff_gt_max_dy = vcgtq_s32(qdiff, max_dy_v);
        valid = vbicq_u32(valid, qdiff_gt_max_dy);

        // 6. gap_width = abs(ref_diff - query_diff); gap_width <= bw
        let diff = vsubq_s32(rdiff, qdiff);
        let gap_w = vabsq_s32(diff);
        let gw_gt_bw = vcgtq_s32(gap_w, bw_v);
        valid = vbicq_u32(valid, gw_gt_bw);

        // --- Score computation ---
        let min_diff = vminq_s32(rdiff, qdiff);
        let mut score = vminq_s32(span_v, min_diff);

        // --- Penalty computation ---
        let gw_gt_zero = vcgtq_s32(gap_w, zero_v);
        let md_gt_span = vcgtq_s32(min_diff, span_v);
        let needs_pen = vorrq_u32(gw_gt_zero, md_gt_span);

        // lin_pen = chn_pen_gap * gap_width + chn_pen_skip * min_diff
        let gap_w_f = vcvtq_f32_s32(gap_w);
        let min_diff_f = vcvtq_f32_s32(min_diff);
        let mul1 = vmulq_f32(pen_gap_v, gap_w_f);
        let mul2 = vmulq_f32(pen_skip_v, min_diff_f);
        let lin_pen = vaddq_f32(mul1, mul2);

        // log_pen = fast_log2(gap_width + 1) if gap_width >= 1, else 0
        let gw_plus1 = vaddq_s32(gap_w, one_v);
        let gw_plus1_f = vcvtq_f32_s32(gw_plus1);
        let log2_val = simd_fast_log2_neon(gw_plus1_f);
        // Mask: zero out log_pen where gap_width == 0
        let log_pen = vreinterpretq_f32_u32(vandq_u32(gw_gt_zero, vreinterpretq_u32_f32(log2_val)));

        // total_pen = lin_pen + 0.5 * log_pen
        let half_log = vmulq_f32(half_v, log_pen);
        let total_pen = vaddq_f32(lin_pen, half_log);

        // Convert to i32 (truncation toward zero, matches Rust `as i32`)
        let pen_i32 = vcvtq_s32_f32(total_pen);

        // Apply penalty only where needs_pen is set
        let pen_masked = vandq_s32(vreinterpretq_s32_u32(needs_pen), pen_i32);
        score = vsubq_s32(score, pen_masked);

        // Mask invalid scores to i32::MIN
        score = vbslq_s32(valid, score, min_val_v);

        // Store 4 scores
        vst1q_s32(sc_buf.as_mut_ptr().add(j), score);

        j += 4;
    }

    // Scalar remainder
    while j < end {
        let qj = soa_query_pos[j];
        let rj = soa_ref_pos[j];
        let span_j = soa_query_span[j];
        let rid_j = soa_rid_strand[j];

        let query_diff = qi.wrapping_sub(qj);
        if query_diff <= 0 || query_diff > max_dist_x {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let ref_diff = ri.wrapping_sub(rj);
        if rid_strand_i == rid_j && (ref_diff == 0 || query_diff > max_dist_y) {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let gap_width = (ref_diff - query_diff).abs();
        if rid_strand_i == rid_j && gap_width > bw {
            sc_buf[j] = i32::MIN;
            j += 1;
            continue;
        }

        let min_diff = ref_diff.min(query_diff);
        let mut sc = span_j.min(min_diff);

        if gap_width > 0 || min_diff > span_j {
            let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
            let log_pen = if gap_width >= 1 {
                fast_log2((gap_width + 1) as f32)
            } else {
                0.0f32
            };
            sc -= (lin_pen + 0.5f32 * log_pen) as i32;
        }

        sc_buf[j] = sc;
        j += 1;
    }
}}

/// NEON SIMD-optimized chaining for aarch64.
/// Phase 1: Batch-compute scores with NEON (4-wide). Phase 2: Scalar skip/visited logic.
/// Produces bit-exact output matching `chain_anchors_scalar`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn chain_anchors_neon(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { unsafe {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw {
        max_dist_x = bw;
    }
    if max_dist_y < bw {
        max_dist_y = bw;
    }

    let real_max_drop = bw;
    let no_chain_skip = opt.max_chain_skip <= 0;

    // --- Extract SoA fields from AoS Minimizer array ---
    let mut soa_ref_pos = std::mem::take(&mut ctx.soa_ref_pos);
    let mut soa_query_pos = std::mem::take(&mut ctx.soa_query_pos);
    let mut soa_query_span = std::mem::take(&mut ctx.soa_query_span);
    let mut soa_rid_strand = std::mem::take(&mut ctx.soa_ref_id_strand);
    let mut soa_sc_buf = std::mem::take(&mut ctx.soa_scores_buf);

    soa_ref_pos.resize(n, 0);
    soa_query_pos.resize(n, 0);
    soa_query_span.resize(n, 0);
    soa_rid_strand.resize(n, 0);
    soa_sc_buf.resize(n, 0);

    for j in 0..n {
        soa_ref_pos[j] = a[j].ref_pos();
        soa_query_pos[j] = a[j].query_pos();
        soa_query_span[j] = a[j].query_span();
        soa_rid_strand[j] = (a[j].x >> 32) as u32;
    }

    // --- DP buffers ---
    let mut predecessors = std::mem::take(&mut ctx.predecessors); let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    visited.clear();
    visited.resize(n, 0i32);

    let mut global_max_score = 0;
    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span();
        let mut skip_count: i32 = 0;

        while window_start < i
            && (a[i].ref_id_strand() != a[window_start].ref_id_strand()
                || (a[i].x > a[window_start].x + max_dist_x as u64))
        {
            window_start += 1;
        }

        if !no_chain_skip && (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j: i64 = if window_start > 0 {
            window_start as i64 - 1
        } else {
            -1
        };

        if no_chain_skip {
            if i > window_start {
                compute_chain_scores_batch_neon(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, window_start, i,
                );
            }
            for j in window_start..i {
                let sc = soa_sc_buf[j];
                if sc == i32::MIN { continue; }
                let total_sc = sc + scores[j];
                if total_sc > best_score { best_score = total_sc; best_predecessor = j as i64; }
            }
        } else {
            const CHUNK_SIZE: usize = 64;
            let mut chunk_end = i;
            let mut skip_fired = false;
            while chunk_end > window_start {
                let chunk_start = if chunk_end >= window_start + CHUNK_SIZE { chunk_end - CHUNK_SIZE } else { window_start };

                compute_chain_scores_batch_neon(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, chunk_start, chunk_end,
                );

                for j in (chunk_start..chunk_end).rev() {
                    let sc = soa_sc_buf[j];
                    if sc == i32::MIN { continue; }
                    let total_sc = sc + scores[j];
                    if total_sc > best_score {
                        best_score = total_sc; best_predecessor = j as i64;
                        if skip_count > 0 { skip_count -= 1; }
                    } else if visited[j] == i as i32 {
                        skip_count += 1;
                        if skip_count > opt.max_chain_skip { end_j = j as i64; skip_fired = true; break; }
                    }
                    if predecessors[j] >= 0 { visited[predecessors[j] as usize] = i as i32; }
                }

                if skip_fired { break; }
                chunk_end = chunk_start;
            }

            if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
                let mut best = i32::MIN; best_anchor_idx = -1;
                for j in (window_start..i).rev() {
                    if best < scores[j] { best = scores[j]; best_anchor_idx = j as i64; }
                }
            }

            if best_anchor_idx >= 0 && best_anchor_idx < end_j {
                let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize],
                    max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, false, 1);
                if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                    best_score = sc + scores[best_anchor_idx as usize];
                    best_predecessor = best_anchor_idx;
                }
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0
            && peak_scores[best_predecessor as usize] > best_score
        {
            peak_scores[best_predecessor as usize]
        } else {
            best_score
        };

        if best_anchor_idx < 0
            || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64
                && scores[best_anchor_idx as usize] < scores[i])
        {
            best_anchor_idx = i as i64;
        }

        if global_max_score < best_score {
            global_max_score = best_score;
        }
    }

    // --- Backtrack and compact (identical to AVX2/scalar) ---
    let (u, n_u, n_v) = chain_backtrack(
        n,
        &scores,
        &predecessors,
        &mut peak_scores,
        &mut visited,
        &mut bt_candidates,
        opt.min_cnt,
        opt.min_chain_score,
        real_max_drop,
    );

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates);
        ctx.scores = scores;
        ctx.peak_scores = peak_scores;
        ctx.visited = visited;
        ctx.soa_ref_pos = soa_ref_pos;
        ctx.soa_query_pos = soa_query_pos;
        ctx.soa_query_span = soa_query_span;
        ctx.soa_ref_id_strand = soa_rid_strand;
        ctx.soa_scores_buf = soa_sc_buf;
        return (Vec::new(), Vec::new());
    }

    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0usize;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni {
            let idx = peak_scores[k0 + (ni - j - 1)] as usize;
            b.push(a[idx]);
            k += 1;
        }
    }

    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer {
            x: b[k_pos].x,
            y: ((k_pos as u64) << 32) | (i as u64),
        });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);

    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize;
        let offset = (w_val.y >> 32) as usize;
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni {
            b2.push(b[offset + idx]);
        }
    }

    ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates);
    ctx.scores = scores;
    ctx.peak_scores = peak_scores;
    ctx.visited = visited;
    ctx.soa_ref_pos = soa_ref_pos;
    ctx.soa_query_pos = soa_query_pos;
    ctx.soa_query_span = soa_query_span;
    ctx.soa_ref_id_strand = soa_rid_strand;
    ctx.soa_scores_buf = soa_sc_buf;

    (u2, b2)
}}

// =============================================================================
// WASM SIMD128 implementation — 4-wide, mirrors NEON logic exactly
// =============================================================================

/// Vectorized fast_log2 for 4 floats (WASM SIMD128).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn simd_fast_log2_wasm(x: v128) -> v128 {
    let z = x; // reinterpret as i32x4 via bit ops

    // exponent = ((z >> 23) & 255) - 128
    let exp_raw = v128_and(i32x4_shr(z, 23), i32x4_splat(255));
    let exp = i32x4_sub(exp_raw, i32x4_splat(128));
    let log_2 = f32x4_convert_i32x4(exp);

    // mantissa: clear exponent bits, set exponent to 127
    let mant_bits = v128_or(
        v128_and(z, i32x4_splat(!(255i32 << 23))),
        i32x4_splat(127i32 << 23),
    );
    let z_f = mant_bits; // reinterpret as f32x4

    // Horner's: ((-0.34484843 * z_f) + 2.0246658) * z_f + (-0.6748776)
    let c0 = f32x4_splat(-0.34484843f32);
    let c1 = f32x4_splat(2.024_665_8f32);
    let c2 = f32x4_splat(-0.674_877_6f32);

    let t1 = f32x4_mul(c0, z_f);
    let t2 = f32x4_add(t1, c1);
    let t3 = f32x4_mul(t2, z_f);
    let poly = f32x4_add(t3, c2);

    f32x4_add(log_2, poly)
}

/// Batch-compute chain scores for 4 predecessors at a time (WASM SIMD128).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn compute_chain_scores_batch_wasm(
    qi: i32,
    ri: i32,
    rid_strand_i: u32,
    soa_ref_pos: &[i32],
    soa_query_pos: &[i32],
    soa_query_span: &[i32],
    soa_rid_strand: &[u32],
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    sc_buf: &mut [i32],
    start: usize,
    end: usize,
) { unsafe {
    if start >= end { return; }

    let qi_v = i32x4_splat(qi);
    let ri_v = i32x4_splat(ri);
    let rid_i_v = i32x4_splat(rid_strand_i as i32);
    let max_dx_v = i32x4_splat(max_dist_x);
    let max_dy_v = i32x4_splat(max_dist_y);
    let bw_v = i32x4_splat(bw);
    let zero_v = i32x4_splat(0);
    let one_v = i32x4_splat(1);
    let min_val_v = i32x4_splat(i32::MIN);
    let pen_gap_v = f32x4_splat(chn_pen_gap);
    let pen_skip_v = f32x4_splat(chn_pen_skip);
    let half_v = f32x4_splat(0.5f32);

    let aligned_end = start + ((end - start) / 4) * 4;
    let mut j = start;

    while j < aligned_end {
        let qj_v = v128_load(soa_query_pos.as_ptr().add(j) as *const v128);
        let rj_v = v128_load(soa_ref_pos.as_ptr().add(j) as *const v128);
        let span_v = v128_load(soa_query_span.as_ptr().add(j) as *const v128);
        let rid_j_v = v128_load(soa_rid_strand.as_ptr().add(j) as *const v128);

        let qdiff = i32x4_sub(qi_v, qj_v);
        let rdiff = i32x4_sub(ri_v, rj_v);

        // Build validity mask
        let mut valid = i32x4_gt(qdiff, zero_v);
        let qdiff_gt_max_dx = i32x4_gt(qdiff, max_dx_v);
        valid = v128_andnot(valid, qdiff_gt_max_dx);  // valid & ~gt_max_dx
        let same_rid = i32x4_eq(rid_i_v, rid_j_v);
        valid = v128_and(valid, same_rid);
        let rdiff_zero = i32x4_eq(rdiff, zero_v);
        valid = v128_andnot(valid, rdiff_zero);
        let qdiff_gt_max_dy = i32x4_gt(qdiff, max_dy_v);
        valid = v128_andnot(valid, qdiff_gt_max_dy);

        let diff = i32x4_sub(rdiff, qdiff);
        let gap_w = i32x4_abs(diff);
        let gw_gt_bw = i32x4_gt(gap_w, bw_v);
        valid = v128_andnot(valid, gw_gt_bw);

        // Score computation
        let min_diff = i32x4_min(rdiff, qdiff);
        let mut score = i32x4_min(span_v, min_diff);

        // Penalty
        let gw_gt_zero = i32x4_gt(gap_w, zero_v);
        let md_gt_span = i32x4_gt(min_diff, span_v);
        let needs_pen = v128_or(gw_gt_zero, md_gt_span);

        let gap_w_f = f32x4_convert_i32x4(gap_w);
        let min_diff_f = f32x4_convert_i32x4(min_diff);
        let mul1 = f32x4_mul(pen_gap_v, gap_w_f);
        let mul2 = f32x4_mul(pen_skip_v, min_diff_f);
        let lin_pen = f32x4_add(mul1, mul2);

        let gw_plus1 = i32x4_add(gap_w, one_v);
        let gw_plus1_f = f32x4_convert_i32x4(gw_plus1);
        let log2_val = simd_fast_log2_wasm(gw_plus1_f);
        let log_pen = v128_and(gw_gt_zero, log2_val);

        let half_log = f32x4_mul(half_v, log_pen);
        let total_pen = f32x4_add(lin_pen, half_log);

        let pen_i32 = i32x4_trunc_sat_f32x4(total_pen);

        let pen_masked = v128_and(needs_pen, pen_i32);
        score = i32x4_sub(score, pen_masked);

        // Mask invalid to i32::MIN
        score = v128_bitselect(score, min_val_v, valid);

        v128_store(sc_buf.as_mut_ptr().add(j) as *mut v128, score);
        j += 4;
    }

    // Scalar remainder
    while j < end {
        let qj = soa_query_pos[j];
        let rj = soa_ref_pos[j];
        let span_j = soa_query_span[j];
        let rid_j = soa_rid_strand[j];

        let query_diff = qi.wrapping_sub(qj);
        if query_diff <= 0 || query_diff > max_dist_x {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let ref_diff = ri.wrapping_sub(rj);
        if rid_strand_i == rid_j && (ref_diff == 0 || query_diff > max_dist_y) {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let gap_width = (ref_diff - query_diff).abs();
        if rid_strand_i == rid_j && gap_width > bw {
            sc_buf[j] = i32::MIN; j += 1; continue;
        }
        let min_diff = ref_diff.min(query_diff);
        let mut sc = span_j.min(min_diff);
        if gap_width > 0 || min_diff > span_j {
            let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
            let log_pen = if gap_width >= 1 { fast_log2((gap_width + 1) as f32) } else { 0.0f32 };
            sc -= (lin_pen + 0.5f32 * log_pen) as i32;
        }
        sc_buf[j] = sc;
        j += 1;
    }
}}

/// WASM SIMD128 chaining — 4-wide, same structure as NEON/AVX2.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn chain_anchors_wasm(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { unsafe {
    let n = a.len();
    if n == 0 { return (Vec::new(), Vec::new()); }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw { max_dist_x = bw; }
    if max_dist_y < bw { max_dist_y = bw; }

    let real_max_drop = bw;
    let no_chain_skip = opt.max_chain_skip <= 0;

    let mut soa_ref_pos = std::mem::take(&mut ctx.soa_ref_pos);
    let mut soa_query_pos = std::mem::take(&mut ctx.soa_query_pos);
    let mut soa_query_span = std::mem::take(&mut ctx.soa_query_span);
    let mut soa_rid_strand = std::mem::take(&mut ctx.soa_ref_id_strand);
    let mut soa_sc_buf = std::mem::take(&mut ctx.soa_scores_buf);

    soa_ref_pos.resize(n, 0);
    soa_query_pos.resize(n, 0);
    soa_query_span.resize(n, 0);
    soa_rid_strand.resize(n, 0);
    soa_sc_buf.resize(n, 0);

    for j in 0..n {
        soa_ref_pos[j] = a[j].ref_pos();
        soa_query_pos[j] = a[j].query_pos();
        soa_query_span[j] = a[j].query_span();
        soa_rid_strand[j] = (a[j].x >> 32) as u32;
    }

    let mut predecessors = std::mem::take(&mut ctx.predecessors); let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    visited.clear();
    visited.resize(n, 0i32);

    let mut global_max_score = 0;
    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span();
        let mut skip_count: i32 = 0;

        while window_start < i
            && (a[i].ref_id_strand() != a[window_start].ref_id_strand()
                || (a[i].x > a[window_start].x + max_dist_x as u64))
        { window_start += 1; }

        if !no_chain_skip && (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j: i64 = if window_start > 0 { window_start as i64 - 1 } else { -1 };

        if no_chain_skip {
            if i > window_start {
                compute_chain_scores_batch_wasm(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, window_start, i,
                );
            }
            for j in window_start..i {
                let sc = soa_sc_buf[j];
                if sc == i32::MIN { continue; }
                let total_sc = sc + scores[j];
                if total_sc > best_score { best_score = total_sc; best_predecessor = j as i64; }
            }
        } else {
            const CHUNK_SIZE: usize = 64;
            let mut chunk_end = i;
            let mut skip_fired = false;
            while chunk_end > window_start {
                let chunk_start = if chunk_end >= window_start + CHUNK_SIZE { chunk_end - CHUNK_SIZE } else { window_start };

                compute_chain_scores_batch_wasm(
                    soa_query_pos[i], soa_ref_pos[i], soa_rid_strand[i],
                    &soa_ref_pos, &soa_query_pos, &soa_query_span, &soa_rid_strand,
                    max_dist_x, max_dist_y, bw,
                    opt.chn_pen_gap, opt.chn_pen_skip,
                    &mut soa_sc_buf, chunk_start, chunk_end,
                );

                for j in (chunk_start..chunk_end).rev() {
                    let sc = soa_sc_buf[j];
                    if sc == i32::MIN { continue; }
                    let total_sc = sc + scores[j];
                    if total_sc > best_score {
                        best_score = total_sc; best_predecessor = j as i64;
                        if skip_count > 0 { skip_count -= 1; }
                    } else if visited[j] == i as i32 {
                        skip_count += 1;
                        if skip_count > opt.max_chain_skip { end_j = j as i64; skip_fired = true; break; }
                    }
                    if predecessors[j] >= 0 { visited[predecessors[j] as usize] = i as i32; }
                }

                if skip_fired { break; }
                chunk_end = chunk_start;
            }

            if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
                let mut best = i32::MIN; best_anchor_idx = -1;
                for j in (window_start..i).rev() {
                    if best < scores[j] { best = scores[j]; best_anchor_idx = j as i64; }
                }
            }

            if best_anchor_idx >= 0 && best_anchor_idx < end_j {
                let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize],
                    max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, false, 1);
                if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                    best_score = sc + scores[best_anchor_idx as usize];
                    best_predecessor = best_anchor_idx;
                }
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0
            && peak_scores[best_predecessor as usize] > best_score
        { peak_scores[best_predecessor as usize] } else { best_score };

        if best_anchor_idx < 0
            || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64
                && scores[best_anchor_idx as usize] < scores[i])
        { best_anchor_idx = i as i64; }

        if global_max_score < best_score { global_max_score = best_score; }
    }

    let (u, n_u, n_v) = chain_backtrack(
        n, &scores, &predecessors, &mut peak_scores, &mut visited,
        &mut bt_candidates,
        opt.min_cnt, opt.min_chain_score, real_max_drop,
    );

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
        ctx.peak_scores = peak_scores; ctx.visited = visited;
        ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
        ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
        ctx.soa_scores_buf = soa_sc_buf;
        return (Vec::new(), Vec::new());
    }

    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0usize;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni {
            let idx = peak_scores[k0 + (ni - j - 1)] as usize;
            b.push(a[idx]);
            k += 1;
        }
    }

    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer { x: b[k_pos].x, y: ((k_pos as u64) << 32) | (i as u64) });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);

    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize;
        let offset = (w_val.y >> 32) as usize;
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni { b2.push(b[offset + idx]); }
    }

    ctx.predecessors = predecessors; ctx.bt_candidates = std::mem::take(&mut bt_candidates); ctx.scores = scores;
    ctx.peak_scores = peak_scores; ctx.visited = visited;
    ctx.soa_ref_pos = soa_ref_pos; ctx.soa_query_pos = soa_query_pos;
    ctx.soa_query_span = soa_query_span; ctx.soa_ref_id_strand = soa_rid_strand;
    ctx.soa_scores_buf = soa_sc_buf;

    (u2, b2)
}}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::align::chain::{chain_anchors_scalar, fast_log2};
    use crate::align::map::ChainingParams;

    #[test]
    fn test_simd_log2_vs_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let test_values: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0, 0.5, 0.001, 1.5, 7.7, 50.3, 999.9,
            2.5, 123.456,
        ];

        unsafe {
            // Process 8 at a time
            for chunk in test_values.chunks(8) {
                let mut input = [1.0f32; 8];
                for (i, &v) in chunk.iter().enumerate() {
                    input[i] = v;
                }
                let v = _mm256_loadu_ps(input.as_ptr());
                let result = simd_fast_log2_avx2(v);
                let mut output = [0.0f32; 8];
                _mm256_storeu_ps(output.as_mut_ptr(), result);

                for (i, &v) in chunk.iter().enumerate() {
                    let scalar = fast_log2(v);
                    assert_eq!(
                        output[i].to_bits(),
                        scalar.to_bits(),
                        "log2 mismatch for {}: simd={} scalar={}",
                        v,
                        output[i],
                        scalar
                    );
                }
            }
        }
    }

    #[test]
    fn test_chain_simd_vs_scalar_synthetic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let k = 15;
        let span_mask = (k as u64) << 32;
        let mut anchors_simd: Vec<Minimizer> = Vec::new();

        // Chain 1: Linear perfect match
        let chain1 = vec![
            Minimizer { x: 100, y: span_mask | 100 },
            Minimizer { x: 120, y: span_mask | 120 },
            Minimizer { x: 150, y: span_mask | 150 },
        ];
        // Chain 2: Isolated
        let chain2 = vec![Minimizer { x: 500, y: span_mask | 500 }];
        // Chain 3: Indel
        let chain3 = vec![
            Minimizer { x: 1000, y: span_mask | 1000 },
            Minimizer { x: 1030, y: span_mask | 1020 },
        ];

        // Add enough anchors to exceed the 32-anchor threshold
        for i in 0..30 {
            anchors_simd.push(Minimizer {
                x: 2000 + i * 20,
                y: span_mask | (2000 + i * 20),
            });
        }

        for a in &chain1 {
            anchors_simd.push(*a);
        }
        for a in &chain2 {
            anchors_simd.push(*a);
        }
        for a in &chain3 {
            anchors_simd.push(*a);
        }
        let mut anchors_scalar: Vec<Minimizer> = anchors_simd.clone();

        // Sort by x (ref position) as required
        anchors_simd.sort_by_key(|a| a.x);
        anchors_scalar.sort_by_key(|a| a.x);

        let opt = ChainingParams {
            min_cnt: 1,
            min_chain_score: 10,
            max_gap: 500,
            max_gap_ref: -1,
            max_dist_x: 500,
            max_dist_y: 500,
            bandwidth: 500,
            bandwidth_long: 500,
            max_chain_skip: 25,
            max_chain_iter: 50,
            chn_pen_gap: 0.5,
            chn_pen_skip: 0.5,
            chain_gap_scale: 0.8,
            rmq_rescue_size: 1000,
            rmq_rescue_ratio: 0.1,
            rmq_inner_dist: 1000,
            rmq_size_cap: 100000,
        };

        let mut ctx_simd = ChainingBuffers::new();
        let mut ctx_scalar = ChainingBuffers::new();

        let (u_simd, chains_simd) = unsafe {
            chain_anchors_avx2(&opt, 500, 500, &mut anchors_simd, &mut ctx_simd)
        };
        let (u_scalar, chains_scalar) =
            chain_anchors_scalar(&opt, false, 1, 500, 500, &mut anchors_scalar, &mut ctx_scalar);

        assert_eq!(u_simd.len(), u_scalar.len(), "chain count mismatch");
        for (i, (us, uc)) in u_simd.iter().zip(u_scalar.iter()).enumerate() {
            assert_eq!(us, uc, "chain {} u mismatch: simd={} scalar={}", i, us, uc);
        }
        assert_eq!(
            chains_simd.len(),
            chains_scalar.len(),
            "anchor count mismatch"
        );
        for (i, (as_, ac)) in chains_simd.iter().zip(chains_scalar.iter()).enumerate() {
            assert_eq!(
                as_.x, ac.x,
                "anchor {} x mismatch: simd={} scalar={}",
                i, as_.x, ac.x
            );
            assert_eq!(
                as_.y, ac.y,
                "anchor {} y mismatch: simd={} scalar={}",
                i, as_.y, ac.y
            );
        }
    }

    #[test]
    fn test_chain_simd_larger_window() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Create a long collinear chain with 200 anchors to stress SIMD
        let k = 15;
        let span_mask = (k as u64) << 32;
        let mut anchors_simd: Vec<Minimizer> = Vec::new();

        for i in 0..200 {
            anchors_simd.push(Minimizer {
                x: (i * 10) as u64,
                y: span_mask | ((i * 10) as u64),
            });
        }
        let mut anchors_scalar: Vec<Minimizer> = anchors_simd.clone();

        let opt = ChainingParams {
            min_cnt: 1,
            min_chain_score: 10,
            max_gap: 5000,
            max_gap_ref: -1,
            max_dist_x: 5000,
            max_dist_y: 5000,
            bandwidth: 5000,
            bandwidth_long: 5000,
            max_chain_skip: 25,
            max_chain_iter: 5000,
            chn_pen_gap: 0.5,
            chn_pen_skip: 0.5,
            chain_gap_scale: 0.8,
            rmq_rescue_size: 1000,
            rmq_rescue_ratio: 0.1,
            rmq_inner_dist: 1000,
            rmq_size_cap: 100000,
        };

        let mut ctx_simd = ChainingBuffers::new();
        let mut ctx_scalar = ChainingBuffers::new();

        let (u_simd, chains_simd) = unsafe {
            chain_anchors_avx2(&opt, 5000, 5000, &mut anchors_simd, &mut ctx_simd)
        };
        let (u_scalar, chains_scalar) =
            chain_anchors_scalar(&opt, false, 1, 5000, 5000, &mut anchors_scalar, &mut ctx_scalar);

        assert_eq!(u_simd.len(), u_scalar.len(), "chain count mismatch");
        for (i, (us, uc)) in u_simd.iter().zip(u_scalar.iter()).enumerate() {
            assert_eq!(us, uc, "chain {} u mismatch", i);
        }
        assert_eq!(chains_simd.len(), chains_scalar.len(), "anchor count mismatch");
        for (i, (as_, ac)) in chains_simd.iter().zip(chains_scalar.iter()).enumerate() {
            assert_eq!(as_.x, ac.x, "anchor {} x mismatch", i);
            assert_eq!(as_.y, ac.y, "anchor {} y mismatch", i);
        }
    }

    #[test]
    fn test_chain_no_skip_fast_path() {
        // Tests the --max-chain-skip=0 fast path (no skip heuristic).
        // This produces DIFFERENT output from the skip path, so we just verify
        // it runs without panics and produces valid chains.
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let k = 15;
        let span_mask = (k as u64) << 32;

        // Create anchors with gaps that would trigger skip heuristic
        let mut anchors: Vec<Minimizer> = Vec::new();
        // Main chain: evenly spaced
        for i in 0..50 {
            anchors.push(Minimizer {
                x: (i * 20) as u64,
                y: span_mask | ((i * 20) as u64),
            });
        }
        // Noise anchors interleaved (different ref_id to be filtered)
        for i in 0..20 {
            anchors.push(Minimizer {
                x: (1u64 << 33) | ((i * 30) as u64), // different ref_id
                y: span_mask | ((i * 30 + 5) as u64),
            });
        }
        anchors.sort_by_key(|a| a.x);

        let opt_skip = ChainingParams {
            min_cnt: 1,
            min_chain_score: 10,
            max_gap: 5000,
            max_gap_ref: -1,
            max_dist_x: 5000,
            max_dist_y: 5000,
            bandwidth: 5000,
            bandwidth_long: 5000,
            max_chain_skip: 25,
            max_chain_iter: 5000,
            chn_pen_gap: 0.5,
            chn_pen_skip: 0.5,
            chain_gap_scale: 0.8,
            rmq_rescue_size: 1000,
            rmq_rescue_ratio: 0.1,
            rmq_inner_dist: 1000,
            rmq_size_cap: 100000,
        };

        let mut opt_noskip = opt_skip.clone();
        opt_noskip.max_chain_skip = 0;

        let mut anchors_skip = anchors.clone();
        let mut anchors_noskip = anchors;

        let mut ctx_skip = ChainingBuffers::new();
        let mut ctx_noskip = ChainingBuffers::new();

        let (u_skip, chains_skip) = unsafe {
            chain_anchors_avx2(&opt_skip, 5000, 5000, &mut anchors_skip, &mut ctx_skip)
        };
        let (u_noskip, chains_noskip) = unsafe {
            chain_anchors_avx2(&opt_noskip, 5000, 5000, &mut anchors_noskip, &mut ctx_noskip)
        };

        // Both should produce valid output (non-empty for this input)
        assert!(!u_skip.is_empty(), "skip path should produce chains");
        assert!(!u_noskip.is_empty(), "no-skip path should produce chains");

        // Verify chain scores are positive
        for (i, u) in u_skip.iter().enumerate() {
            let score = (*u >> 32) as i32;
            assert!(score > 0, "skip chain {} has non-positive score {}", i, score);
        }
        for (i, u) in u_noskip.iter().enumerate() {
            let score = (*u >> 32) as i32;
            assert!(score > 0, "no-skip chain {} has non-positive score {}", i, score);
        }

        // Verify anchor counts are consistent
        let skip_anchor_count: usize = u_skip.iter().map(|u| (u & 0xFFFFFFFF) as usize).sum();
        assert_eq!(skip_anchor_count, chains_skip.len(),
            "skip: u anchor count doesn't match chains length");
        let noskip_anchor_count: usize = u_noskip.iter().map(|u| (u & 0xFFFFFFFF) as usize).sum();
        assert_eq!(noskip_anchor_count, chains_noskip.len(),
            "no-skip: u anchor count doesn't match chains length");
    }
}
