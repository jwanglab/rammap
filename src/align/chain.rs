//! DP-based anchor chaining
//!
//! Takes a reference-sorted array of [`Minimizer`] anchors and computes optimal
//! chains using dynamic programming. For each anchor `i`, the algorithm scans
//! backwards within a distance window to find the predecessor `j` that maximizes
//! `score[j] + gap_score(i,j)`. Gap penalties combine linear (`chn_pen_gap`,
//! `chn_pen_skip`) and logarithmic terms.
//!
//! Key parameters from [`ChainingParams`]: `max_gap`/`bandwidth` control the
//! distance window, `max_chain_skip` limits backtrack iterations, and
//! `min_chain_score`/`min_cnt` filter weak chains.
//!
//! Output: a `Vec<u64>` of chain descriptors (upper 32 bits = score, lower 32
//! bits = anchor count) plus a reordered anchor array. Chains are sorted by
//! reference position of their first anchor. On x86_64 with AVX2, dispatches
//! to a SIMD-accelerated implementation for genomic (non-cDNA) single-segment
//! inputs with 32+ anchors.

use crate::align::sketch::Minimizer;
use crate::align::sort::radix_sort_128x;
use crate::align::map::{ChainingParams, ChainingBuffers};

#[inline(always)]
pub(crate) fn fast_log2(x: f32) -> f32 {
    let z = x.to_bits();
    let log_2 = (((z >> 23) & 255) as i32 - 128) as f32;
    let z_f = f32::from_bits(z & !(255 << 23) | (127 << 23));
    log_2 + ((-0.34484843f32 * z_f + 2.024_665_8_f32) * z_f - 0.674_877_6_f32)
}

#[inline(always)]
pub(crate) fn compute_chain_score(
    ai: &Minimizer,
    aj: &Minimizer,
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
    is_cdna: bool,
    n_seg: i32,
) -> i32 {
    let query_diff = ai.query_pos().wrapping_sub(aj.query_pos()); // q_pos diff

    let seg_id_i = ai.segment_id() as u64;
    let seg_id_j = aj.segment_id() as u64;

    if query_diff <= 0 || query_diff > max_dist_x {
        return i32::MIN;
    }

    let ref_diff = ai.ref_pos().wrapping_sub(aj.ref_pos());

    if seg_id_i == seg_id_j && (ref_diff == 0 || query_diff > max_dist_y) {
        return i32::MIN;
    }

    let gap_width = if ref_diff > query_diff { ref_diff - query_diff } else { query_diff - ref_diff };
    if seg_id_i == seg_id_j && gap_width > bw {
        return i32::MIN;
    }
    if n_seg > 1 && !is_cdna && seg_id_i == seg_id_j && ref_diff > max_dist_y {
        return i32::MIN;
    }

    let min_diff = if ref_diff < query_diff { ref_diff } else { query_diff };
    let q_span = aj.query_span(); // q_span from earlier anchor (aj), not ai

    let mut score = if q_span < min_diff { q_span } else { min_diff };

    if gap_width > 0 || min_diff > q_span {
        let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
        let log_pen = if gap_width >= 1 { fast_log2((gap_width + 1) as f32) } else { 0.0f32 };

        if is_cdna || seg_id_i != seg_id_j {
            if seg_id_i != seg_id_j && ref_diff == 0 {
                score += 1;
            } else if ref_diff > query_diff || seg_id_i != seg_id_j {
                let pen = if lin_pen < log_pen { lin_pen } else { log_pen };
                score -= pen as i32;
            } else {
                score -= (lin_pen + 0.5f32 * log_pen) as i32;
            }
        } else {
            score -= (lin_pen + 0.5f32 * log_pen) as i32;
        }
    }

    score
}


/// Walk back through predecessors from candidate `k` to find the chain's
/// start anchor, applying a per-step z-drop (`max_drop`) cutoff. Marks
/// visited anchors with sentinel `2` and resets them before returning.
pub(crate) fn chain_backtrack_end(max_drop: i32, candidates: &[Minimizer], scores: &[i32], predecessors: &[i32], visited: &mut [i32], k: i64) -> i64 {
    let mut i = candidates[k as usize].y as i64;

    let mut end_i;
    let mut max_i = i;
    let mut max_s = 0;

    if i < 0 || i as usize >= visited.len() {
        return i;
    }
    if visited[i as usize] != 0 {
        return i;
    }

    loop {
        visited[i as usize] = 2;
        // end_i = i = p[i]; both are set to p[i]
        i = predecessors[i as usize] as i64;
        end_i = i;

        let f_curr = candidates[k as usize].x as i32;

        let s = if i < 0 {
            f_curr
        } else {
            f_curr - scores[i as usize]
        };

        if s > max_s {
            max_s = s;
            max_i = i;
        } else if max_s - s > max_drop {
            break;
        }

        if i < 0 || visited[i as usize] != 0 {
            break;
        }
    }

    // Reset visited[]
    let mut curr = candidates[k as usize].y as i64;
    while curr >= 0 && curr != end_i {
        visited[curr as usize] = 0;
        curr = predecessors[curr as usize] as i64;
    }

    max_i
}

/// Reconstruct chains from the DP scores+predecessors arrays. Returns
/// `(u, n_u, n_v)` where `u[i]` packs `(score, anchor_count)` for chain `i`,
/// `n_u` is the chain count, and `n_v` is the total anchor count across chains.
/// Anchors are emitted into `v` in backtrack (reverse) order.
pub(crate) fn chain_backtrack(
    // km: void* - allocator, redundant in Rust
    n: usize,
    scores: &[i32],
    predecessors: &[i32],
    v: &mut [i32], // modified in place to store anchor indices; C: v[n_v++] = i
    visited: &mut [i32],
    candidates: &mut Vec<Minimizer>, // pooled scratch
    min_cnt: i32,
    min_sc: i32,
    max_drop: i32,
) -> (Vec<u64>, usize, usize) { // returns (u, n_u, n_v)
    candidates.clear();
    candidates.reserve(n);
    let mut n_z = 0;

    for (i, &score) in scores[..n].iter().enumerate() {
        if score >= min_sc {
            n_z += 1;
            candidates.push(Minimizer { x: score as u64, y: i as u64 });
        }
    }

    if n_z == 0 {
        return (Vec::new(), 0, 0);
    }

    // Sort candidates by score (x) using MSD radix sort
    radix_sort_128x(candidates);

    for x in visited.iter_mut() { *x = 0; }

    let mut n_v = 0;
    let mut n_u = 0;

    let mut u: Vec<u64> = Vec::new();

    for k in (0..n_z).rev() {
        let z_k = &candidates[k];
        let z_k_y_idx = z_k.y as usize;

        if visited[z_k_y_idx] == 0 {
            let n_v0 = n_v;
            let end_i = chain_backtrack_end(max_drop, candidates, scores, predecessors, visited, k as i64);

            let mut i = z_k_y_idx as i64;
            while i != end_i {
                if n_v < v.len() {
                    v[n_v] = i as i32;
                }
                n_v += 1;
                visited[i as usize] = 1;
                i = predecessors[i as usize] as i64;
            }

            let sc = if i < 0 {
                z_k.x as i32
            } else {
                (z_k.x as i32) - scores[i as usize]
            };

            if sc >= min_sc && n_v > n_v0 && (n_v - n_v0) as i32 >= min_cnt {
                n_u += 1;
                u.push(((sc as u64) << 32) | ((n_v - n_v0) as u64));
            } else {
                n_v = n_v0;
            }
        }
    }

    (u, n_u, n_v)
}

// Dispatch: SIMD or scalar chaining based on CPU features and mode
pub fn chain_anchors(
    opt: &ChainingParams,
    is_cdna: bool,
    n_seg: i32,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) {
    // Fall back to scalar for cdna/multi-seg or small inputs
    if is_cdna || n_seg > 1 || a.len() < 32 {
        return chain_anchors_scalar(opt, is_cdna, n_seg, max_dist_x, max_dist_y, a, ctx);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let force_scalar = *crate::align::env_flags::FORCE_SCALAR_CHAIN;
        if !force_scalar && crate::align::dp::use_avx512() && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe {
                super::chain_simd::chain_anchors_avx512(opt, max_dist_x, max_dist_y, a, ctx)
            };
        }
        if !force_scalar && crate::align::dp::use_avx2() {
            return unsafe {
                super::chain_simd::chain_anchors_avx2(opt, max_dist_x, max_dist_y, a, ctx)
            };
        }
        if !force_scalar {
            return unsafe {
                super::chain_simd::chain_anchors_sse(opt, max_dist_x, max_dist_y, a, ctx)
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let force_scalar = *crate::align::env_flags::FORCE_SCALAR_CHAIN;
        if !force_scalar {
            return unsafe {
                super::chain_simd::chain_anchors_neon(opt, max_dist_x, max_dist_y, a, ctx)
            };
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        return unsafe {
            super::chain_simd::chain_anchors_wasm(opt, max_dist_x, max_dist_y, a, ctx)
        };
    }

    // wasm32 the block above always returns, so this is unreachable for wasm32
    #[cfg(not(target_arch = "wasm32"))]
    return chain_anchors_scalar(opt, is_cdna, n_seg, max_dist_x, max_dist_y, a, ctx);
}

/// Partition anchors by (ref_id, strand) and chain each partition independently
/// in parallel via rayon. Since anchors on different reference sequences cannot
/// form chains, partitions are completely independent.
///
/// Returns merged (u, chains) identical to what `chain_anchors` would produce,
/// but with partitions processed in parallel for better throughput on reads
/// with anchors spanning many reference sequences.
#[cfg(feature = "parallel")]
pub fn chain_anchors_partitioned(
    opt: &ChainingParams,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer],
) -> (Vec<u64>, Vec<Minimizer>) {
    use rayon::prelude::*;

    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Find partition boundaries by ref_id (NOT ref_id_strand).
    // Both strands of the same ref must stay together because chaining
    // allows cross-strand connections (inversions). Anchors are sorted by
    // x = (ref_id << 33 | strand << 32 | ref_pos), so both strands of the same
    // ref_id are contiguous.
    let mut boundaries: Vec<usize> = vec![0];
    for i in 1..n {
        if a[i].ref_id() != a[i - 1].ref_id() {
            boundaries.push(i);
        }
    }
    boundaries.push(n);

    let n_partitions = boundaries.len() - 1;

    // If only one partition (common for short reads mapping to one chromosome),
    // skip the partitioning overhead and chain directly.
    if n_partitions <= 1 {
        let mut ctx = ChainingBuffers::new();
        return chain_anchors(opt, false, 1, max_dist_x, max_dist_y, a, &mut ctx);
    }

    // Split anchors into non-overlapping mutable slices for each partition.
    // We use raw pointer arithmetic because Rust can't prove non-overlapping
    // borrows from a single &mut [Minimizer].
    let base = a.as_mut_ptr() as usize;

    let results: Vec<(Vec<u64>, Vec<Minimizer>)> = (0..n_partitions)
        .into_par_iter()
        .map(|pi| {
            let start = boundaries[pi];
            let end = boundaries[pi + 1];
            let len = end - start;
            let slice = unsafe {
                std::slice::from_raw_parts_mut((base as *mut Minimizer).add(start), len)
            };
            let mut ctx = ChainingBuffers::new();
            chain_anchors(opt, false, 1, max_dist_x, max_dist_y, slice, &mut ctx)
        })
        .collect();

    // Merge: concatenate all u vectors and chain anchor arrays.
    let total_u: usize = results.iter().map(|(u, _)| u.len()).sum();
    let total_chains: usize = results.iter().map(|(_, c)| c.len()).sum();
    let mut merged_u = Vec::with_capacity(total_u);
    let mut merged_chains = Vec::with_capacity(total_chains);

    for (u, chains) in results {
        merged_u.extend_from_slice(&u);
        merged_chains.extend_from_slice(&chains);
    }

    (merged_u, merged_chains)
}

// Scalar DP chaining over a reference-sorted anchor array. For each anchor i,
// scan back within a distance window for the best-scoring predecessor; record
// score + predecessor; then backtrack into compacted chains.
pub(crate) fn chain_anchors_scalar(
    opt: &ChainingParams,
    is_cdna: bool,
    n_seg: i32,
    max_dist_x: i32,
    max_dist_y: i32,
    a: &mut [Minimizer], // Input anchors
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) { // Returns (u, a_new)
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let bw = opt.bandwidth;
    let mut max_dist_x = max_dist_x;
    let mut max_dist_y = max_dist_y;
    if max_dist_x < bw { max_dist_x = bw; }
    if max_dist_y < bw && !is_cdna { max_dist_y = bw; }

    let real_max_drop = if is_cdna { i32::MAX } else { bw };

    let mut predecessors = std::mem::take(&mut ctx.predecessors);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    // visited uses sentinel comparison (visited[j] == i) so must be zeroed
    visited.clear(); visited.resize(n, 0i32);

    let mut global_max_score = 0;

    let mut window_start: usize = 0;
    let mut best_anchor_idx: i64 = -1; // Tracks best-scoring anchor within distance window

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let mut best_score = a[i].query_span(); // q_span
        let mut skip_count = 0;

        while window_start < i && ( a[i].ref_id_strand() != a[window_start].ref_id_strand() || (a[i].x > a[window_start].x + max_dist_x as u64) ) {
            window_start += 1;
        }

        // max_iter limiting
        if (i - window_start) > opt.max_chain_iter as usize {
            window_start = i - opt.max_chain_iter as usize;
        }

        let mut end_j = if window_start > 0 { window_start as i64 - 1 } else { -1 };

        for j in (window_start..i).rev() {
            let sc = compute_chain_score(&a[i], &a[j], max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, is_cdna, n_seg);

            if sc == i32::MIN { continue; }

            let total_sc = sc + scores[j];
            if total_sc > best_score {
                best_score = total_sc;
                best_predecessor = j as i64;
                if skip_count > 0 { skip_count -= 1; }
            } else if visited[j] == i as i32 {
                skip_count += 1;
                if skip_count > opt.max_chain_skip {
                    end_j = j as i64;
                    break;
                }
            }
            if predecessors[j] >= 0 {
                visited[predecessors[j] as usize] = i as i32;
            }
        }

        // best_anchor_idx optimization: reset if too far away
        // a[] is sorted by x, so a[i].x >= a[best_anchor_idx].x — subtraction cannot underflow
        if best_anchor_idx < 0 || a[i].x - a[best_anchor_idx as usize].x > max_dist_x as u64 {
            let mut best = i32::MIN;
            best_anchor_idx = -1;
            for j in (window_start..i).rev() {
                if best < scores[j] {
                    best = scores[j];
                    best_anchor_idx = j as i64;
                }
            }
        }

        // Check if best_anchor_idx provides a better path than what main loop found
        if best_anchor_idx >= 0 && best_anchor_idx < end_j {
            let sc = compute_chain_score(&a[i], &a[best_anchor_idx as usize], max_dist_x, max_dist_y, bw, opt.chn_pen_gap, opt.chn_pen_skip, is_cdna, n_seg);
            if sc != i32::MIN && best_score < sc + scores[best_anchor_idx as usize] {
                best_score = sc + scores[best_anchor_idx as usize];
                best_predecessor = best_anchor_idx;
            }
        }

        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0 && peak_scores[best_predecessor as usize] > best_score { peak_scores[best_predecessor as usize] } else { best_score };

        // Update best_anchor_idx if current anchor has better score
        if best_anchor_idx < 0 || (a[i].x - a[best_anchor_idx as usize].x <= max_dist_x as u64 && scores[best_anchor_idx as usize] < scores[i]) {
            best_anchor_idx = i as i64;
        }

        if global_max_score < best_score { global_max_score = best_score; }
    }

    let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let (u, n_u, n_v) = chain_backtrack(n, &scores, &predecessors, &mut peak_scores, &mut visited, &mut bt_candidates, opt.min_cnt, opt.min_chain_score, real_max_drop);

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited; ctx.bt_candidates = bt_candidates;
        return (Vec::new(), Vec::new());
    }

    // compact_a logic
    // Step 1: Write chain anchors to b[] in forward order
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

    // Step 2: Sort chains by target position of their first anchor.
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
        let j = (w_val.y & 0xFFFFFFFF) as usize; // original chain index
        let offset = (w_val.y >> 32) as usize;    // offset in b[]
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni {
            b2.push(b[offset + idx]);
        }
    }

    ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited; ctx.bt_candidates = bt_candidates;
    (u2, b2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_synthetic() {
        let k = 15;
        let span_mask = (k as u64) << 32;
        let mut anchors: Vec<Minimizer> = Vec::new();

        // Chain 1: Linear perfect match
        anchors.push(Minimizer { x: 100, y: span_mask | 100 });
        anchors.push(Minimizer { x: 120, y: span_mask | 120 });
        anchors.push(Minimizer { x: 150, y: span_mask | 150 });
        
        // Chain 2: Isolated
        anchors.push(Minimizer { x: 500, y: span_mask | 500 });
        
        // Chain 3: Indel
        anchors.push(Minimizer { x: 1000, y: span_mask | 1000 });
        anchors.push(Minimizer { x: 1030, y: span_mask | 1020 });
    
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
        let mut bufs = ChainingBuffers::new();
        let (u, _chains) = chain_anchors(
            &opt, false, 1, 500, 500,
            &mut anchors, &mut bufs,
        );
        
        // Original manual verification result was 4 chains
        assert_eq!(u.len(), 4);
        
        // Verify scores
        // Chain 0 (Main chain)
        let score0 = u[0] >> 32;
        assert_eq!(score0, 20); // 15 + 5 + 0
    }
}
