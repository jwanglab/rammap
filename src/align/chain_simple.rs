//! Greedy single-linkage chainer — a simple proof-of-concept.
//!
//! Demonstrates the [`Chainer`] trait with a minimal O(n) algorithm:
//! scan reference-sorted anchors left to right, extending the current chain
//! while consecutive anchors are within `max_gap` on both axes and roughly
//! colinear. Start a new chain when a break is detected.
//!
//! This is much simpler than the DP chainer (`chain.rs`) which considers
//! all pairwise predecessor relationships with gap penalties. The greedy
//! approach produces fewer and lower-quality chains but runs in linear time
//! and is easy to understand.
//!
//! # Trade-offs vs DP chaining
//!
//! | Property           | DP chainer          | Greedy chainer      |
//! |--------------------|---------------------|---------------------|
//! | Time complexity    | O(n × max_skip)     | O(n)                |
//! | Gap handling       | Penalized, optimal  | Binary (in/out)     |
//! | Overlapping chains | Yes (backtracking)  | No (partition)      |
//! | Chain quality      | High                | Adequate            |

use crate::align::sketch::Minimizer;
use crate::align::map::{ChainingParams, ChainingBuffers, Chainer};

/// Greedy single-linkage chainer.
///
/// Chains are formed by scanning anchors left to right and extending
/// whenever the next anchor is within `max_gap` on both reference and
/// query axes, and the diagonal difference is within `bandwidth`.
pub struct GreedyChainer;

impl Chainer for GreedyChainer {
    fn chain(
        &self,
        params: &ChainingParams,
        anchors: &mut [Minimizer],
        _bufs: &mut ChainingBuffers,
    ) -> (Vec<u64>, Vec<Minimizer>) {
        let n = anchors.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        // Anchors should already be sorted by (ref_id_strand, ref_pos).
        // We scan left to right, building chains greedily.
        let max_gap = params.max_gap as i64;
        let bandwidth = params.bandwidth as i64;

        let mut chains_out: Vec<Minimizer> = Vec::with_capacity(n);
        let mut u: Vec<u64> = Vec::new();

        // Current chain state
        let mut chain_start = 0usize;
        let mut prev_rpos = anchors[0].ref_pos() as i64;
        let mut prev_qpos = anchors[0].query_pos() as i64;
        let prev_rid_strand = anchors[0].ref_id_strand();

        for i in 1..n {
            let rpos = anchors[i].ref_pos() as i64;
            let qpos = anchors[i].query_pos() as i64;
            let rid_strand = anchors[i].ref_id_strand();

            // Check chain-break conditions
            let ref_gap = rpos - prev_rpos;
            let qry_gap = qpos - prev_qpos;
            let diag_diff = (ref_gap - qry_gap).abs();

            let is_break = rid_strand != prev_rid_strand  // different chromosome/strand
                || ref_gap < 0                             // non-monotonic ref (shouldn't happen if sorted)
                || ref_gap > max_gap                       // too far on reference
                || qry_gap < 0                             // query goes backward
                || qry_gap > max_gap                       // too far on query
                || diag_diff > bandwidth;                  // off-diagonal

            if is_break {
                // Emit current chain
                emit_chain(&anchors[chain_start..i], &mut chains_out, &mut u);
                chain_start = i;
            }

            prev_rpos = rpos;
            prev_qpos = qpos;
        }

        // Emit final chain
        emit_chain(&anchors[chain_start..n], &mut chains_out, &mut u);

        // Sort chains by score descending (best first), as the pipeline expects
        let mut chain_indices: Vec<usize> = (0..u.len()).collect();
        chain_indices.sort_unstable_by(|&a, &b| (u[b] >> 32).cmp(&(u[a] >> 32)));

        let mut sorted_u = Vec::with_capacity(u.len());
        let mut sorted_chains = Vec::with_capacity(chains_out.len());
        let mut offsets: Vec<usize> = vec![0; u.len()];
        {
            let mut off = 0;
            for i in 0..u.len() {
                offsets[i] = off;
                off += (u[i] & 0xFFFFFFFF) as usize;
            }
        }
        for &idx in &chain_indices {
            sorted_u.push(u[idx]);
            let cnt = (u[idx] & 0xFFFFFFFF) as usize;
            let start = offsets[idx];
            sorted_chains.extend_from_slice(&chains_out[start..start + cnt]);
        }

        (sorted_u, sorted_chains)
    }
}

/// Emit a chain: compute score as sum of anchor spans, push to output.
fn emit_chain(anchors: &[Minimizer], chains_out: &mut Vec<Minimizer>, u: &mut Vec<u64>) {
    if anchors.is_empty() {
        return;
    }
    let cnt = anchors.len() as u64;
    // Simple score: sum of k-mer spans (each anchor contributes its k-mer length)
    let score: i64 = anchors.iter().map(|a| a.query_span() as i64).sum();
    let u_val = ((score as u64) << 32) | cnt;
    u.push(u_val);
    chains_out.extend_from_slice(anchors);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> ChainingParams {
        ChainingParams {
            min_cnt: 3, min_chain_score: 40, max_gap: 5000,
            max_gap_ref: 5000, max_dist_x: 5000, max_dist_y: 5000,
            bandwidth: 500, bandwidth_long: 20000, max_chain_skip: 25,
            max_chain_iter: 5000, chn_pen_gap: 1.0, chn_pen_skip: 0.05,
            chain_gap_scale: 0.8, rmq_rescue_size: 1000, rmq_rescue_ratio: 0.1,
            rmq_inner_dist: 1000, rmq_size_cap: 100000,
        }
    }

    fn make_anchor(ref_pos: i32, query_pos: i32, span: i32) -> Minimizer {
        // Pack ref_pos into lower 32 of x, ref_id=0 strand=0 in upper 32
        let x = ref_pos as u32 as u64;
        // Pack query_pos into lower 32 of y, span into bits 32-39
        let y = (query_pos as u32 as u64) | ((span as u64) << 32);
        Minimizer { x, y }
    }

    #[test]
    fn test_greedy_single_chain() {
        // Anchors on a clean diagonal — should form one chain
        let mut anchors = vec![
            make_anchor(100, 10, 15),
            make_anchor(120, 30, 15),
            make_anchor(140, 50, 15),
            make_anchor(160, 70, 15),
        ];
        let params = test_params();
        let mut bufs = ChainingBuffers::new();
        let chainer = GreedyChainer;
        let (u, chains) = chainer.chain(&params, &mut anchors, &mut bufs);

        assert_eq!(u.len(), 1, "should produce one chain");
        assert_eq!((u[0] & 0xFFFFFFFF), 4, "chain should have 4 anchors");
        assert_eq!(chains.len(), 4);
    }

    #[test]
    fn test_greedy_chain_break() {
        // Two groups separated by a large gap
        let mut anchors = vec![
            make_anchor(100, 10, 15),
            make_anchor(120, 30, 15),
            make_anchor(50000, 40010, 15), // big gap
            make_anchor(50020, 40030, 15),
        ];
        let params = test_params();
        let mut bufs = ChainingBuffers::new();
        let chainer = GreedyChainer;
        let (u, chains) = chainer.chain(&params, &mut anchors, &mut bufs);

        assert_eq!(u.len(), 2, "should produce two chains");
        assert_eq!(chains.len(), 4);
    }

    #[test]
    fn test_greedy_empty() {
        let mut anchors: Vec<Minimizer> = Vec::new();
        let params = test_params();
        let mut bufs = ChainingBuffers::new();
        let chainer = GreedyChainer;
        let (u, chains) = chainer.chain(&params, &mut anchors, &mut bufs);
        assert!(u.is_empty());
        assert!(chains.is_empty());
    }

    #[test]
    fn test_greedy_sorted_by_score() {
        // Second group is longer (higher score) — should come first in output
        let mut anchors = vec![
            make_anchor(100, 10, 15),          // chain 1: 1 anchor
            make_anchor(50000, 40000, 15),     // chain 2: 3 anchors
            make_anchor(50020, 40020, 15),
            make_anchor(50040, 40040, 15),
        ];
        let params = test_params();
        let mut bufs = ChainingBuffers::new();
        let chainer = GreedyChainer;
        let (u, _chains) = chainer.chain(&params, &mut anchors, &mut bufs);

        assert_eq!(u.len(), 2);
        // Best chain (higher score) should be first
        assert!((u[0] >> 32) >= (u[1] >> 32), "chains should be sorted by score descending");
    }
}
