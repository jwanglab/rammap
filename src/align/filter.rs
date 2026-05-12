//! Parent assignment and secondary filtering
//!
//! Operates on anything implementing the `Filterable` trait, so the same logic
//! applies to both pre-alignment chains (in `map.rs`) and post-alignment results
//! (in `pipeline.rs`). Parent assignment computes query-coordinate overlap between
//! all pairs of hits; a hit whose overlap with a higher-scoring hit exceeds
//! `mask_level` is marked as a child of that parent. Secondary selection then
//! keeps at most `best_n` hits, discarding children whose score falls below
//! `pri_ratio * parent_score` (or `pri_ratio * 0.85` for non-primary parents).
//!
//! Key inputs: `FilterParams` (thresholds derived from `MapOptions` and `Index`).
//! Key output: each item's `parent`, `subsc`, and `is_primary` fields are set
//! in-place; items marked for removal have their `is_primary` cleared.

use crate::align::map::MapOptions;
use crate::align::index::Index;

/// Parameters for filtering, derived from MapOptions and Index
#[derive(Debug, Clone)]
pub struct FilterParams {
    pub pri_ratio: f32,
    pub best_n: i32,
    pub min_diff: i32,
    pub min_strand_sc: i32,
    pub mask_level: f32,
    pub mask_len: i32,
    pub hard_mask_level: bool,
}

impl FilterParams {
    /// Create filter params from MapOptions and Index
    pub fn new(opt: &MapOptions, mi: &Index) -> Self {
        use crate::align::map::AlignFlags;
        FilterParams {
            pri_ratio: opt.filtering.pri_ratio,
            best_n: opt.filtering.best_n,
            min_diff: (mi.kmer_size * 2) as i32,        // k * 2
            min_strand_sc: (opt.chaining.max_gap as f32 * 0.8) as i32,
            mask_level: opt.filtering.mask_level,
            mask_len: opt.filtering.mask_len,
            hard_mask_level: opt.flags.contains(AlignFlags::HARD_MASK_LEVEL),
        }
    }
}

/// Trait for items that can be filtered (chains or alignments)
pub trait Filterable {
    fn query_start(&self) -> usize;
    fn query_end(&self) -> usize;
    fn score(&self) -> i32;
    fn is_reverse(&self) -> bool;
    fn is_alt(&self) -> bool { false }
}

/// Scale score for ALT contigs (hit.c:99-104)
#[inline]
pub fn scale_alt_score(score: i32, alt_diff_frac: f32) -> i32 {
    if score < 0 { return score; }
    let s = (score as f32 * (1.0 - alt_diff_frac) + 0.499) as i32;
    if s > 0 { s } else { 1 }
}

/// Tracks parent assignment state during filtering
pub struct ParentState {
    /// Index of parent for each item (i if item i is its own parent/primary)
    pub parent: Vec<usize>,
    /// Score of the parent for each item
    pub parent_score: Vec<i32>,
    /// Strand of the parent for each item
    pub parent_rev: Vec<bool>,
    /// Primary regions as (index, qs, qe)
    primaries: Vec<(usize, usize, usize)>,
    mask_level: f32,
    mask_len: i32,
    hard_mask_level: bool,
}

impl ParentState {
    /// Create new parent state for n items
    pub fn new(n: usize, mask_level: f32, mask_len: i32, hard_mask_level: bool) -> Self {
        ParentState {
            parent: (0..n).collect(),
            parent_score: Vec::with_capacity(n),
            parent_rev: Vec::with_capacity(n),
            primaries: Vec::new(),
            mask_level,
            mask_len,
            hard_mask_level,
        }
    }

    /// Initialize with scores and strands from items
    pub fn init_from_items<T: Filterable>(&mut self, items: &[T]) {
        self.parent_score = items.iter().map(|item| item.score()).collect();
        self.parent_rev = items.iter().map(|item| item.is_reverse()).collect();
    }

    /// Assign parents based on query overlap (implements mm_set_parent logic from hit.c:125-186)
    pub fn assign_parents<T: Filterable>(&mut self, items: &[T]) {
        let n = items.len();
        if n == 0 { return; }

        // First item is always primary
        self.primaries.push((0, items[0].query_start(), items[0].query_end()));

        for (i, item) in items[1..].iter().enumerate() {
            let i = i + 1; // offset since we started from index 1
            let si = item.query_start();
            let ei = item.query_end();
            let len_i = ei.saturating_sub(si);
            if len_i == 0 {
                continue;
            }

            // Phase 1: Collect coverage intervals from ALL overlapping primaries (hit.c:138-144)
            // hard_mask_level: skip uncov_len calculation (hit.c:137 goto skip_uncov), uncov_len=0
            let uncov_len = if self.hard_mask_level {
                0usize
            } else {
                let mut cov: Vec<(usize, usize)> = Vec::new();
                for &(_, qs_p, qe_p) in &self.primaries {
                    if qe_p <= si || qs_p >= ei { continue; }
                    let sj = qs_p.max(si);
                    let ej = qe_p.min(ei);
                    cov.push((sj, ej));
                }

                // Phase 2: Compute uncov_len using sorted interval union (hit.c:146-156)
                if cov.is_empty() {
                    self.primaries.push((i, si, ei));
                    continue;
                }
                cov.sort_unstable();
                let mut x = si;
                let mut uncov = 0usize;
                for &(cs, ce) in &cov {
                    if cs > x { uncov += cs - x; }
                    if ce > x { x = ce; }
                }
                if ei > x { uncov += ei - x; }
                uncov
            };

            // Phase 3: Check each overlapping primary for mask condition (hit.c:158-179)
            let mut found_parent = false;
            for &(p_idx, qs_p, qe_p) in &self.primaries {
                let len_p = qe_p.saturating_sub(qs_p);
                if qe_p <= si || qs_p >= ei { continue; } // no overlap
                let min_len = len_i.min(len_p);
                let max_len = len_i.max(len_p);
                // Compute overlap (hit.c:164)
                let ol = if si < qs_p {
                    if ei <= qs_p { 0 } else if ei < qe_p { ei - qs_p } else { qe_p - qs_p }
                } else if qe_p <= si { 0 } else if qe_p < ei { qe_p - si } else { ei - si };

                if min_len > 0 && max_len > 0 {
                    // Overlap test:
                    let ratio = (ol as f32 / min_len as f32)
                        - (uncov_len as f32 / max_len as f32);
                    if ratio > self.mask_level && (uncov_len as i32) <= self.mask_len {
                        // Secondary of this primary's parent (hit.c:167)
                        let actual_parent = self.parent[p_idx]; // rp->parent
                        self.parent[i] = actual_parent;
                        self.parent_score[i] = self.parent_score[actual_parent];
                        self.parent_rev[i] = self.parent_rev[actual_parent];
                        found_parent = true;
                        break; // hit.c:178
                    }
                }
            }

            // If no parent found → new primary (hit.c:182)
            if !found_parent {
                self.primaries.push((i, si, ei));
            }
        }
    }

    /// Check if item at index i is an independent primary (its own parent)
    #[inline]
    pub fn is_primary(&self, i: usize) -> bool {
        self.parent[i] == i
    }
}

/// Simple filterable item for cases where we need to construct from raw data
#[derive(Debug, Clone, Copy)]
pub struct FilterableItem {
    pub query_start: usize,
    pub query_end: usize,
    pub score: i32,
    pub is_reverse: bool,
    pub is_alt: bool,
}

impl Filterable for FilterableItem {
    fn query_start(&self) -> usize { self.query_start }
    fn query_end(&self) -> usize { self.query_end }
    fn score(&self) -> i32 { self.score }
    fn is_reverse(&self) -> bool { self.is_reverse }
    fn is_alt(&self) -> bool { self.is_alt }
}

/// Result of checking if an item passes filtering
#[derive(Debug, Clone, Copy)]
pub struct FilterResult {
    pub passes: bool,
    pub passes_ratio: bool,
    pub passes_min_diff: bool,
    pub passes_strand: bool,
}

/// Check if a secondary item passes the secondary-overlap / score-ratio
/// filtering criteria (run after parent assignment).
/// `check_strand`: strand retention is enabled for pre-alignment filtering
/// but disabled for post-alignment filtering.
pub fn check_secondary_filter(
    score: i32,
    rev: bool,
    parent_score: i32,
    parent_rev: bool,
    params: &FilterParams,
    check_strand: bool,
) -> FilterResult {
    // pri_ratio check: score >= parent_score * pri_ratio
    let passes_ratio = (score as f32) >= (parent_score as f32 * params.pri_ratio);

    // min_diff check: score + min_diff >= parent_score
    let passes_min_diff = score + params.min_diff >= parent_score;

    // Strand-based retention: keep alignments on opposite strand if score > min_strand_sc
    // (only when check_strand=true)
    let diff_strand = rev != parent_rev;
    let passes_strand = check_strand && diff_strand && score > params.min_strand_sc;

    FilterResult {
        passes: passes_ratio || passes_min_diff || passes_strand,
        passes_ratio,
        passes_min_diff,
        passes_strand,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockItem {
        qs: usize,
        qe: usize,
        score: i32,
        rev: bool,
    }

    impl Filterable for MockItem {
        fn query_start(&self) -> usize {
            self.qs
        }
        fn query_end(&self) -> usize {
            self.qe
        }
        fn score(&self) -> i32 {
            self.score
        }
        fn is_reverse(&self) -> bool {
            self.rev
        }
    }

    #[test]
    fn test_parent_assignment_no_overlap() {
        let items = vec![
            MockItem { qs: 0, qe: 100, score: 200, rev: false },
            MockItem { qs: 200, qe: 300, score: 150, rev: false },
        ];

        let mut state = ParentState::new(items.len(), 0.5, i32::MAX, false);
        state.init_from_items(&items);
        state.assign_parents(&items);

        // Both should be their own parent (no overlap)
        assert!(state.is_primary(0));
        assert!(state.is_primary(1));
    }

    #[test]
    fn test_parent_assignment_with_overlap() {
        let items = vec![
            MockItem { qs: 0, qe: 100, score: 200, rev: false },
            MockItem { qs: 10, qe: 90, score: 150, rev: false }, // Fully contained
        ];

        let mut state = ParentState::new(items.len(), 0.5, i32::MAX, false);
        state.init_from_items(&items);
        state.assign_parents(&items);

        assert!(state.is_primary(0));
        assert!(!state.is_primary(1)); // Secondary of 0
        assert_eq!(state.parent[1], 0);
    }

    #[test]
    fn test_secondary_filter_passes_ratio() {
        let params = FilterParams {
            pri_ratio: 0.8,
            best_n: 5,
            min_diff: 30,
            min_strand_sc: 4000,
            mask_level: 0.5,
            mask_len: i32::MAX,
            hard_mask_level: false,
        };

        // Score = 85, parent = 100, ratio = 0.85 >= 0.8 -> passes
        let result = check_secondary_filter(85, false, 100, false, &params, true);
        assert!(result.passes);
        assert!(result.passes_ratio);
    }

    #[test]
    fn test_secondary_filter_passes_min_diff() {
        let params = FilterParams {
            pri_ratio: 0.8,
            best_n: 5,
            min_diff: 30,
            min_strand_sc: 4000,
            mask_level: 0.5,
            mask_len: i32::MAX,
            hard_mask_level: false,
        };

        // Score = 75, parent = 100, ratio = 0.75 < 0.8, but 75 + 30 >= 100 -> passes
        let result = check_secondary_filter(75, false, 100, false, &params, true);
        assert!(result.passes);
        assert!(!result.passes_ratio);
        assert!(result.passes_min_diff);
    }

    #[test]
    fn test_secondary_filter_passes_strand() {
        let params = FilterParams {
            pri_ratio: 0.8,
            best_n: 5,
            min_diff: 30,
            min_strand_sc: 4000,
            mask_level: 0.5,
            mask_len: i32::MAX,
            hard_mask_level: false,
        };

        // Score = 5000, parent = 10000, ratio = 0.5 < 0.8, min_diff fails too
        // But different strand and score > 4000 -> passes (with check_strand=true)
        let result = check_secondary_filter(5000, true, 10000, false, &params, true);
        assert!(result.passes);
        assert!(!result.passes_ratio);
        assert!(!result.passes_min_diff);
        assert!(result.passes_strand);
    }

    #[test]
    fn test_secondary_filter_fails() {
        let params = FilterParams {
            pri_ratio: 0.8,
            best_n: 5,
            min_diff: 30,
            min_strand_sc: 4000,
            mask_level: 0.5,
            mask_len: i32::MAX,
            hard_mask_level: false,
        };

        // Score = 50, parent = 100, ratio = 0.5 < 0.8, 50 + 30 = 80 < 100, same strand
        let result = check_secondary_filter(50, false, 100, false, &params, true);
        assert!(!result.passes);
    }
}
