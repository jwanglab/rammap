//! Example: low-level pairwise DP alignment using rammap's SIMD engine.
//!
//! Demonstrates global, semi-global, and extension alignment modes with
//! the same SIMD-optimized DP kernels used internally for read mapping.
//!
//! Usage:
//!   cargo run --release --example dp_align

use rammap::{dp_align, dp_global, dp_local, dp_extension, DpScoring, encode_nt4};

fn main() {
    // Two similar sequences with a mismatch and a small insertion
    let query  = b"ACGTACGTAAACGTACGTACGT";
    let target = b"ACGTACCTACGTACGTACGT";
    //                   ^  mismatch      ^^^ query has "AAA" insertion

    let q = encode_nt4(query);
    let t = encode_nt4(target);
    let scoring = DpScoring::default();

    println!("Query:  {} ({} bp)", std::str::from_utf8(query).unwrap(), query.len());
    println!("Target: {} ({} bp)", std::str::from_utf8(target).unwrap(), target.len());
    println!("Scoring: match={}, mismatch={}, gap_open={}, gap_extend={}",
        scoring.match_score, scoring.mismatch, scoring.gap_open, scoring.gap_extend);
    println!();

    // Semi-global alignment (default)
    let result = dp_align(&q, &t, &scoring, -1);
    println!("Semi-global: score={:3}  cigar={}  q[{}..{}] t[{}..{}]",
        result.score, result.cigar,
        result.query_start, result.query_end,
        result.target_start, result.target_end);

    // Global alignment (Needleman-Wunsch: must cover both sequences end-to-end)
    let result = dp_global(&q, &t, &scoring, -1);
    println!("Global:      score={:3}  cigar={}  q[{}..{}] t[{}..{}]",
        result.score, result.cigar,
        result.query_start, result.query_end,
        result.target_start, result.target_end);

    // Local alignment (Smith-Waterman: find best local region)
    let result = dp_local(&q, &t, &scoring);
    println!("Local:       score={:3}  cigar={}  q[{}..{}] t[{}..{}]",
        result.score, result.cigar,
        result.query_start, result.query_end,
        result.target_start, result.target_end);

    // Extension alignment (stops at best score — for seed extension)
    let result = dp_extension(&q, &t, &scoring, 100);
    println!("Extension:   score={:3}  cigar={}  q[{}..{}] t[{}..{}]",
        result.score, result.cigar,
        result.query_start, result.query_end,
        result.target_start, result.target_end);

    // Dual-affine gap model (separate penalties for short and long gaps)
    println!();
    let dual = DpScoring {
        match_score: 1, mismatch: 4,
        gap_open: 6, gap_extend: 2,
        gap_open2: 26, gap_extend2: 1,
    };
    let result = dp_align(&q, &t, &dual, -1);
    println!("Dual-affine: score={:3}  cigar={}  (O={},{} E={},{})",
        result.score, result.cigar,
        dual.gap_open, dual.gap_open2, dual.gap_extend, dual.gap_extend2);
}
