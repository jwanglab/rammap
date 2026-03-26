//! Alignment statistics tracking.
//!
//! `AlignmentStats` accumulates per-thread timing and count metrics (sketch,
//! seed, chain, align, post-processing) that are summed across threads for
//! the final summary report.

use std::time::Duration;
use std::ops::Add;

#[derive(Debug, Default, Clone, Copy)]
pub struct AlignmentStats {
    pub t_sketch: Duration,
    pub t_seed: Duration,
    pub t_chain: Duration,
    pub t_align: Duration,
    pub t_post: Duration,
    pub n_reads: usize,
    pub n_seeds: usize,
    pub n_anchors: usize,
    pub n_chains: usize,
}

impl Add for AlignmentStats {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            t_sketch: self.t_sketch + other.t_sketch,
            t_seed: self.t_seed + other.t_seed,
            t_chain: self.t_chain + other.t_chain,
            t_align: self.t_align + other.t_align,
            t_post: self.t_post + other.t_post,
            n_reads: self.n_reads + other.n_reads,
            n_seeds: self.n_seeds + other.n_seeds,
            n_anchors: self.n_anchors + other.n_anchors,
            n_chains: self.n_chains + other.n_chains,
        }
    }
}
