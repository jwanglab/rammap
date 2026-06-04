//! Dynamic-programming kernels
//!
//! Provides three kernel types — single-affine, dual-affine, and splice-aware — each
//! with SIMD specializations for SSE2, AVX2, AVX512BW, NEON, WASM SIMD128, and a
//! scalar fallback. The public entry points are `extend_single_affine`,
//! `extend_dual_affine`, and `extend_splice`; each dispatches to the best available
//! SIMD variant at runtime, optionally capped per-preset (e.g., `sr`/`ava-*` cap
//! at AVX2 to avoid AVX-512 license throttling). Override-able via env vars:
//! `RAMMAP_FORCE_SCALAR`, `RAMMAP_FORCE_SSE`, `RAMMAP_FORCE_AVX2`, `RAMMAP_FORCE_AVX512`.
//!
//! All kernels return a `DpResult` containing the alignment score, the query/target
//! end positions, the number of columns computed, and an optional CIGAR. Callers in
//! `extend.rs` invoke these kernels for left-extension, gap-fill, and right-extension.

mod common;
mod single;
mod dual;
mod splice;
mod lw;

use std::sync::atomic::{AtomicU8, Ordering};

/// Per-process cap on SIMD width.
///
/// `sr`/`ava-*` presets set this to `Avx2` because their workload is dominated by
/// chaining/seeding/output rather than DP — a workload class where AVX-512's
/// per-core license throttling (200–600 MHz drop on Skylake-X / Cascade Lake / Cooper Lake
/// for ~2 ms after any AVX-512 instruction) costs more than the DP speedup wins.
/// Other presets leave this `Auto` and use AVX-512 when available.
///
/// Override with env vars (highest priority first):
/// - `RAMMAP_FORCE_SCALAR=1`
/// - `RAMMAP_FORCE_SSE=1`
/// - `RAMMAP_FORCE_AVX2=1`
/// - `RAMMAP_FORCE_AVX512=1` (overrides preset cap; opt back into AVX-512)
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SimdCap {
    Auto = 0,
    Avx2 = 1,
    Sse = 2,
    Scalar = 3,
}

static SIMD_CAP: AtomicU8 = AtomicU8::new(SimdCap::Auto as u8);

/// Set the per-process SIMD cap. Typically called from `apply_preset_str` once
/// during startup. Library users with multiple Aligners should be aware this is
/// process-global; reset to `Auto` if switching back to a DP-heavy preset.
pub fn set_simd_cap(cap: SimdCap) {
    SIMD_CAP.store(cap as u8, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_cap() -> SimdCap {
    match SIMD_CAP.load(Ordering::Relaxed) {
        1 => SimdCap::Avx2,
        2 => SimdCap::Sse,
        3 => SimdCap::Scalar,
        _ => SimdCap::Auto,
    }
}

/// Combines env-var overrides, preset cap, and runtime CPU detection.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn use_avx512() -> bool {
    use crate::align::env_flags::*;
    if *FORCE_SCALAR { return false; }
    if *FORCE_SSE { return false; }
    if *FORCE_AVX2 { return false; }
    // Preset cap applies unless user explicitly opts back into AVX-512.
    if !*FORCE_AVX512 && !matches!(simd_cap(), SimdCap::Auto) {
        return false;
    }
    is_x86_feature_detected!("avx512bw")
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn use_avx2() -> bool {
    use crate::align::env_flags::*;
    if *FORCE_SCALAR { return false; }
    if *FORCE_SSE { return false; }
    if matches!(simd_cap(), SimdCap::Sse | SimdCap::Scalar) { return false; }
    is_x86_feature_detected!("avx2")
}

// Re-export public API from common
pub use common::{
    DpResult, NEG_INF,
    SCORE_ONLY, RIGHT_ALIGN, GENERIC_SCORING, APPROX_MAX, APPROX_DROP,
    EXTENSION_ONLY, REV_CIGAR,
    SPLICE_FORWARD, SPLICE_REVERSE, SPLICE_FLANK, SPLICE_COMPLEX, SPLICE_SCORE,
    CIGAR_MATCH, CIGAR_INS, CIGAR_DEL, CIGAR_N_SKIP, SPSC_OFFSET,
};

// Re-export public API from single-affine
pub use single::{extend_single_affine, extend_single_affine_scalar};

// Re-export public API from dual-affine
pub use dual::{extend_dual_affine, extend_dual_affine_scalar};

// Re-export public API from splice
pub use splice::{extend_splice, extend_splice_scalar};

// Re-export public API from lightweight/global
pub use lw::{
    LightweightProfile,
    lightweight_profile_init, lightweight_align_i16, lightweight_align_i16_scalar,
    global_align,
};

#[cfg(test)]
mod tests {
    use super::*;
    // Also import internal SIMD functions needed by concordance tests
    #[cfg(target_arch = "x86_64")]
    use super::single::{extend_single_affine41_impl, extend_single_affine_avx512_fn};
    #[cfg(target_arch = "x86_64")]
    use super::dual::{extend_dual_affine2_impl, extend_dual_affine41_impl, extend_dual_affine_avx2_fn, extend_dual_affine_avx512_fn};
    #[cfg(target_arch = "aarch64")]
    use super::dual::extend_dual_affine_neon_impl;

    #[test]
    fn test_scalar_dual_affine() {
        // Test the scalar implementation
        // Query: 50 A's, Target: 20 A's
        // Requires 30bp insertion to align all

        let qseq = vec![0u8; 50];
        let tseq = vec![0u8; 20];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // Single-affine: gap_open=4, gap_extend=2
        let mut ez_single = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 4, 2, -1, -1, 0, 0, &mut ez_single);

        // Dual-affine: gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        let mut ez_dual = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_dual);

        println!("Scalar single-affine: score={}", ez_single.score);
        println!("Scalar dual-affine: score={}", ez_dual.score);

        // Print CIGARs
        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Single CIGAR: {}", cigar_str(&ez_single.cigar));
        println!("Dual CIGAR: {}", cigar_str(&ez_dual.cigar));

        // Expected scores:
        // Single: 20*2 - (4 + 30*2) = 40 - 64 = -24
        // Dual: 20*2 - (24 + 30*1) = 40 - 54 = -14
        assert_eq!(ez_single.score, -24, "Single-affine score should be -24");
        assert_eq!(ez_dual.score, -14, "Dual-affine score should be -14");
        assert!(ez_dual.score > ez_single.score, "Dual should beat single");
    }

    #[test]
    fn test_simple_match() {
        let qseq = [0u8, 1, 2, 3]; // ACGT encoded as 0, 1, 2, 3
        let tseq = [0u8, 1, 2, 3];
        let alphabet_size = 5;
        // Simple matrix: match=2, mismatch=-4
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // gap_open=4, gap_extend=2, bandwidth=10, z_drop=-1
        let gap_open = 4;
        let gap_extend = 2;
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX; // Use approx max path we implemented

        let mut result = DpResult::default();

        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);

        println!("Score: {}, Max: {}", result.score, result.max);
        assert_eq!(result.score, 8);
    }

    #[test]
    fn test_1_mismatch() {
        let qseq = [0u8, 1, 2, 3]; // ACGT
        let tseq = [0u8, 1, 0, 3]; // ACAT (G->A mismatch)
        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let gap_open = 4;
        let gap_extend = 2;
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX;

        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);

        // Score: 2+2-4+2 = 2
        println!("Score: {}", result.score);
        assert_eq!(result.score, 2);
    }

    #[test]
    fn test_gap() {
        let qseq = [0u8, 1, 2, 3]; // ACGT
        let tseq = [0u8, 1, 3];    // ACT (G deleted)
        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        // High match score to ensure extension wins
        for i in 0..25 { score_matrix[i] = -10; }
        for i in 0..4 { score_matrix[i*5 + i] = 10; }

        let gap_open = 4;
        let gap_extend = 1; // Gap cost 4+1=5
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX;

        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);

        // Score: AC matches (20) - Gap (5) + T match (10) = 25.
        println!("Score: {}", result.score);
        assert_eq!(result.score, 25);
    }

    #[test]
    fn test_dual_affine_long_gap() {
        // Test case where dual-affine should outperform single-affine
        // Query: 10 A's + 30 gap + 10 A's = effectively 20 A's with 30bp insertion
        // Target: 20 A's
        // This way the alignment must include the gap to cover the full query

        let qseq = vec![0u8; 50]; // 50 A's (query)
        let tseq = vec![0u8; 20];     // 20 A's (target)
        // This forces a 30bp insertion in query to align

        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // Gap penalties: gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        // For a 30bp insertion:
        // Single-affine: 4 + 30*2 = 64
        // Dual-affine: min(64, 24 + 30*1) = min(64, 54) = 54
        // Dual-affine saves 10 points

        let gap_open = 4i8;
        let gap_extend = 2i8;
        let gap_open2 = 24i8;
        let gap_extend2 = 1i8;
        let bandwidth = 100;
        let z_drop = -1;
        // Use both APPROX_MAX and RIGHT flags
        let flags = APPROX_MAX | RIGHT_ALIGN;

        let mut ez_single = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open, gap_extend, bandwidth, z_drop, 0, flags, &mut ez_single);

        let mut ez_dual = DpResult::default();
        extend_dual_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_dual);

        println!("Single-affine: score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_single.score, ez_single.max, ez_single.max_score_query_pos, ez_single.max_score_target_pos, ez_single.cigar.len());
        println!("Dual-affine: score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_dual.score, ez_dual.max, ez_dual.max_score_query_pos, ez_dual.max_score_target_pos, ez_dual.cigar.len());

        // Print CIGAR if available
        if !ez_single.cigar.is_empty() {
            let cigar_str: String = ez_single.cigar.iter()
                .map(|&c| {
                    let len = c >> 4;
                    let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                    format!("{}{}", len, op)
                }).collect();
            println!("Single CIGAR: {}", cigar_str);
        }
        if !ez_dual.cigar.is_empty() {
            let cigar_str: String = ez_dual.cigar.iter()
                .map(|&c| {
                    let len = c >> 4;
                    let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                    format!("{}{}", len, op)
                }).collect();
            println!("Dual CIGAR: {}", cigar_str);
        }

        // Dual-affine should produce a higher (better) score for long gaps
        // Expected: 20 matches, 30bp insertion in query
        // Single: 20*2 - (4 + 30*2) = 40 - 64 = -24
        // Dual: 20*2 - (24 + 30*1) = 40 - 54 = -14
        // Dual-affine should be 10 points better
        assert!(ez_dual.score > ez_single.score,
            "Dual-affine ({}) should be > single-affine ({}) for long gaps",
            ez_dual.score, ez_single.score);
    }

    #[test]
    fn test_neon_vs_scalar_dual_affine() {
        // Compare NEON and scalar dual-affine implementations
        let qseq = vec![0u8; 50]; // 50 A's
        let tseq = vec![0u8; 20]; // 20 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let gap_open = 4i8;
        let gap_extend = 2i8;
        let gap_open2 = 24i8;
        let gap_extend2 = 1i8;
        let bandwidth = 100;
        let z_drop = -1;
        let flags = APPROX_MAX | RIGHT_ALIGN;

        // Run scalar
        let mut ez_scalar = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, gap_open as i32, gap_extend as i32, gap_open2 as i32, gap_extend2 as i32, -1, -1, 0, 0, &mut ez_scalar);

        // Run SIMD directly (NEON on aarch64, SSE2 on x86_64)
        let mut ez_simd = DpResult::default();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_simd);
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_simd);
        }

        println!("=== SIMD vs Scalar Dual-Affine Comparison ===");
        println!("Scalar: score={}, cigar_len={}", ez_scalar.score, ez_scalar.cigar.len());
        println!("SIMD:   score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_simd.score, ez_simd.max, ez_simd.max_score_query_pos, ez_simd.max_score_target_pos, ez_simd.cigar.len());

        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Scalar CIGAR: {}", cigar_str(&ez_scalar.cigar));
        println!("SIMD CIGAR:   {}", cigar_str(&ez_simd.cigar));

        // Expected: score=-14 (20*2 - (24+30*1))
        println!("Expected score: -14");

        assert_eq!(ez_scalar.score, -14, "Scalar should produce -14");
        assert_eq!(ez_simd.score, -14, "SIMD should produce -14");
        assert_eq!(ez_scalar.score, ez_simd.score, "Scores should match");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dual_affine_tiebreak_regression() {
        // Regression: the NEON exact-max scan used a plain linear scan and so
        // picked a different equal-score endpoint than the scalar/SSE/AVX
        // kernels (which use a 4-lane max reduction), producing a different but
        // equal-score CIGAR. This minimal case (delta-debugged from real
        // map-hifi HiFi reads) reproduced the divergence; NEON must now match
        // scalar exactly. See extend_dual_affine_neon_impl max tracking.
        let q: Vec<u8> = "12130130110130130310130110130130110130110130130330110130130130130110130110110110130110110110110130130310130110130130110130130130330110132130110130130130130130310130110110130110130110110130110130110110130130310130110130110130130110130110130130330110130130110130110130130110110130110130130".bytes().map(|b| b - b'0').collect();
        let t: Vec<u8> = "1301101101101101101301101101301301301101101101101101101101301101101301101101301101101101101301101101101101301301101101101101101301301301301101101101101301101101301101101301101101101101301101101301101101101101301101101101101103301301101101101101301101101101101101301101101".bytes().map(|b| b - b'0').collect();
        let mut sm = [0i8; 25];
        for i in 0..4 { for j in 0..4 { sm[i*5+j] = if i==j {1} else {-4}; } sm[i*5+4] = -1; }
        for j in 0..5 { sm[4*5+j] = -1; }
        let (go, ge, go2, ge2) = (6i8, 2i8, 26i8, 1i8);
        let (bw, zd, eb, flags) = (751i32, 400i32, -1i32, 0xc2i32);
        let mut es = DpResult::default();
        extend_dual_affine_scalar(&q, &t, 5, &sm, go as i32, ge as i32, go2 as i32, ge2 as i32, bw, zd, eb, flags, &mut es);
        let mut en = DpResult::default();
        unsafe { extend_dual_affine_neon_impl(&q, &t, 5, &sm, go, ge, go2, ge2, bw, zd, eb, flags, &mut en); }
        assert_eq!(en.score, es.score, "NEON score must match scalar");
        assert_eq!(en.cigar, es.cigar, "NEON CIGAR must match scalar (max-position tie-break parity)");
    }

    #[test]
    fn test_neon_dual_affine_small() {
        // Small test case for easier debugging
        // Query: 10 A's, Target: 5 A's (need 5bp insertion)
        let qseq = vec![0u8; 10];
        let tseq = vec![0u8; 5];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        // For 5bp insertion: single = 4+5*2 = 14, dual = min(14, 24+5*1) = min(14, 29) = 14
        // Single-affine is cheaper for short gaps, so scores should be similar

        let mut ez_scalar = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_scalar);

        let mut ez_simd = DpResult::default();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 50, -1, 0, APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 50, -1, 0, APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
        }

        println!("=== Small Test (10 vs 5 bp) ===");
        println!("Scalar: score={}", ez_scalar.score);
        println!("SIMD:   score={}", ez_simd.score);

        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Scalar CIGAR: {}", cigar_str(&ez_scalar.cigar));
        println!("SIMD CIGAR:   {}", cigar_str(&ez_simd.cigar));

        // Expected: 5 matches (5*2=10), 5bp insertion (4+5*2=14)
        // Score = 10 - 14 = -4
        println!("Expected score: -4");

        assert_eq!(ez_scalar.score, -4, "Scalar score should be -4");
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_sequences() {
        let qseq: Vec<u8> = vec![];
        let tseq: Vec<u8> = vec![];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // Empty sequences should return default/no alignment
        assert_eq!(result.cigar.len(), 0);
    }

    #[test]
    fn test_single_base_match() {
        let qseq = vec![0u8]; // A
        let tseq = vec![0u8]; // A

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        assert_eq!(result.score, 2, "Single match should score 2");
    }

    #[test]
    fn test_single_base_mismatch() {
        let qseq = vec![0u8]; // A
        let tseq = vec![1u8]; // C

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        assert_eq!(result.score, -4, "Single mismatch should score -4");
    }

    // ========================================================================
    // Gap Size Tests
    // ========================================================================

    #[test]
    fn test_short_gap_uses_first_penalty() {
        // 10bp gap: first penalty (4+10*2=24) < second penalty (24+10*1=34)
        let qseq = vec![0u8; 15]; // 15 A's
        let tseq = vec![0u8; 5];  // 5 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 5 matches (5*2=10), 10bp insertion with first penalty (4+10*2=24)
        // Score = 10 - 24 = -14
        assert_eq!(result.score, -14, "Short gap should use first penalty");
    }

    #[test]
    fn test_medium_gap_at_crossover() {
        // 20bp gap: first penalty (4+20*2=44) == second penalty (24+20*1=44)
        // At crossover point, both should give same cost
        let qseq = vec![0u8; 30]; // 30 A's
        let tseq = vec![0u8; 10]; // 10 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 10 matches (10*2=20), 20bp insertion at crossover (cost=44)
        // Score = 20 - 44 = -24
        assert_eq!(result.score, -24, "Gap at crossover should score -24");
    }

    #[test]
    fn test_long_gap_uses_second_penalty() {
        // Already tested in test_dual_affine_long_gap, but verify again
        // 30bp gap: first penalty (4+30*2=64) > second penalty (24+30*1=54)
        let qseq = vec![0u8; 50]; // 50 A's
        let tseq = vec![0u8; 20]; // 20 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 20 matches (20*2=40), 30bp insertion with second penalty (24+30*1=54)
        // Score = 40 - 54 = -14
        assert_eq!(result.score, -14, "Long gap should use second penalty");
    }

    // ========================================================================
    // CIGAR Validation Tests
    // ========================================================================

    fn cigar_to_string(cigar: &[u32]) -> String {
        cigar.iter().map(|&c| {
            let len = c >> 4;
            let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
            format!("{}{}", len, op)
        }).collect()
    }

    // Only used by the x86_64-gated SSE-vs-AVX concordance tests below; gate it
    // to match, so it isn't dead code on aarch64/wasm.
    #[cfg(target_arch = "x86_64")]
    fn cigar_consumed(cigar: &[u32]) -> (usize, usize) {
        let mut q = 0usize;
        let mut t = 0usize;
        for &c in cigar {
            let len = (c >> 4) as usize;
            match c & 0xf { 0 => { q += len; t += len; }, 1 => { q += len; }, 2 => { t += len; }, _ => {} }
        }
        (q, t)
    }

    #[test]
    fn test_cigar_all_matches() {
        let qseq = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let tseq = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        let cigar = cigar_to_string(&result.cigar);
        assert_eq!(cigar, "8M", "Perfect match should be 8M");
        assert_eq!(result.score, 16, "8 matches should score 16");
    }

    #[test]
    fn test_cigar_with_insertion() {
        // Query has extra bases in the middle
        let qseq = vec![0u8, 1, 2, 3, 3, 3, 0, 1]; // ACGTTTAC (extra TTT)
        let tseq = vec![0u8, 1, 0, 1];             // ACAC

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        println!("Insertion test CIGAR: {}", cigar_to_string(&result.cigar));
        println!("Score: {}", result.score);
        // Should align as: 2M + 4I + 2M or similar
        assert!(!result.cigar.is_empty(), "Should produce CIGAR");
    }

    #[test]
    fn test_cigar_with_deletion() {
        // Target has extra bases
        let qseq = vec![0u8, 1];                   // AC
        let tseq = vec![0u8, 1, 2, 3, 0, 1];       // ACGTAC

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        println!("Deletion test CIGAR: {}", cigar_to_string(&result.cigar));
        println!("Score: {}", result.score);
        // Should align as: 2M + 4D or similar
        assert!(!result.cigar.is_empty(), "Should produce CIGAR");
    }

    // ========================================================================
    // Scalar vs SIMD Consistency Tests
    // ========================================================================

    #[test]
    fn test_scalar_neon_consistency_various_sizes() {
        let sizes = [(5, 5), (10, 8), (20, 15), (30, 25), (50, 40)];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        for (qlen, tlen) in sizes {
            let qseq = vec![0u8; qlen];
            let tseq = vec![0u8; tlen];

            let mut ez_scalar = DpResult::default();
            extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_scalar);

            let mut ez_simd = DpResult::default();
            #[cfg(target_arch = "aarch64")]
            unsafe {
                extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 100, -1, 0,
                    APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 100, -1, 0,
                    APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
            }

            println!("Size {}x{}: scalar={}, simd={}", qlen, tlen, ez_scalar.score, ez_simd.score);
            assert_eq!(ez_scalar.score, ez_simd.score,
                "Scalar and SIMD should produce same score for {}x{}", qlen, tlen);
        }
    }

    #[test]
    fn test_single_affine_basic() {
        // Test the single-affine function
        let qseq = vec![0u8, 1, 2, 3]; // ACGT
        let tseq = vec![0u8, 1, 2, 3]; // ACGT

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 100, -1, 0,
            APPROX_MAX | RIGHT_ALIGN, &mut result);

        assert_eq!(result.score, 8, "4 matches should score 8");
    }

    #[test]
    fn test_extd2() {
        // target: ACGTACGTACGTACGT (16 bases)
        // query:  ACGTACGT (8 bases)
        // map-ont: a=2, b=4, gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }

        let mut result = DpResult::default();
        extend_dual_affine(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, 0, &mut result);

        // expect: score=-4 max=16 max_q=7 max_t=7 mqe=16 mqe_t=7
        assert_eq!(result.score, -4, "score mismatch");
        assert_eq!(result.max, 16, "max mismatch");
        assert_eq!(result.max_score_query_pos, 7, "max_q mismatch");
        assert_eq!(result.max_score_target_pos, 7, "max_t mismatch");
        assert_eq!(result.max_query_end_score, 16, "mqe mismatch");
        assert_eq!(result.max_query_end_target_pos, 7, "mqe_t mismatch");
    }

    #[test]
    fn test_extd2_score_only() {
        // Same test but with SCORE_ONLY (no CIGAR traceback)
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }
        let mut result = DpResult::default();
        extend_dual_affine(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, SCORE_ONLY, &mut result);
        println!("score_only extd2: score={} max={} mqe={} mte={}",
            result.score, result.max, result.max_query_end_score, result.max_target_end_score);
        assert_eq!(result.score, -4, "score_only score mismatch");
    }

    #[test]
    fn test_extd2_scalar() {
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, 0, &mut result);
        println!("scalar extd2: score={} max={} max_q={} max_t={} mqe={} mqe_t={} mte={} mte_q={} zd={}",
            result.score, result.max, result.max_score_query_pos, result.max_score_target_pos, result.max_query_end_score, result.max_query_end_target_pos, result.max_target_end_score, result.max_target_end_query_pos, result.zdropped);
        assert_eq!(result.score, -4, "scalar score mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_vs_sse_cigar_concordance() {
        // Test with sequences long enough to exercise multiple SIMD registers (>64 bytes)
        // Use a realistic scoring scheme
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i * 5 + j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i * 5 + 4] = 0;
        }
        for j in 0..5usize { score_matrix[4 * 5 + j] = 0; }

        // Create 100bp sequences with some mismatches to exercise gap logic
        let mut qseq: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        let mut tseq: Vec<u8> = (0..120).map(|i| (i % 4) as u8).collect();
        // Add some mismatches
        qseq[30] = 3;
        qseq[60] = 3;
        tseq[35] = 3;
        tseq[70] = 3;

        // Call SSE41 (CIGAR mode, flags=0 means no SCORE_ONLY)
        let mut result_sse = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut result_sse,
            );
        }

        // Call AVX2 (same params)
        let mut result_avx2 = DpResult::default();
        unsafe {
            extend_dual_affine_avx2_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut result_avx2,
            );
        }

        println!("SSE41: score={} max={} cigar={}", result_sse.score, result_sse.max, cigar_to_string(&result_sse.cigar));
        println!("AVX2:  score={} max={} cigar={}", result_avx2.score, result_avx2.max, cigar_to_string(&result_avx2.cigar));

        // Also compare score-only mode
        let mut result_sse_so = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut result_sse_so,
            );
        }
        let mut result_avx2_so = DpResult::default();
        unsafe {
            extend_dual_affine_avx2_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut result_avx2_so,
            );
        }
        println!("SSE41 score-only: score={} max={}", result_sse_so.score, result_sse_so.max);
        println!("AVX2  score-only: score={} max={}", result_avx2_so.score, result_avx2_so.max);

        assert_eq!(result_sse_so.score, result_avx2_so.score,
            "Score-only: SSE41 and AVX2 should match");

        assert_eq!(result_sse.score, result_avx2.score,
            "SSE41 and AVX2 should produce same score");
        assert_eq!(result_sse.max, result_avx2.max,
            "SSE41 and AVX2 should produce same max");
        assert_eq!(cigar_to_string(&result_sse.cigar), cigar_to_string(&result_avx2.cigar),
            "SSE41 and AVX2 should produce same CIGAR");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_vs_sse_concordance() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping AVX512 test: avx512bw not available");
            return;
        }

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i * 5 + j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i * 5 + 4] = 0;
        }
        for j in 0..5usize { score_matrix[4 * 5 + j] = 0; }

        // 200bp sequences with scattered mismatches to exercise gaps and CIGAR
        let mut qseq: Vec<u8> = (0..200).map(|i| (i % 4) as u8).collect();
        let mut tseq: Vec<u8> = (0..220).map(|i| (i % 4) as u8).collect();
        qseq[30] = 3; qseq[60] = 3; qseq[90] = 3; qseq[150] = 3;
        tseq[35] = 3; tseq[70] = 3; tseq[100] = 3; tseq[180] = 3;

        // --- Single-affine ---
        let mut sse_sa = DpResult::default();
        let mut avx512_sa = DpResult::default();
        unsafe {
            extend_single_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, -1, 400, 0, 0, &mut sse_sa,
            );
            extend_single_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, -1, 400, 0, 0, &mut avx512_sa,
            );
        }
        assert_eq!(sse_sa.score, avx512_sa.score, "single-affine score mismatch");
        assert_eq!(sse_sa.max, avx512_sa.max, "single-affine max mismatch");
        // CIGARs may differ between SIMD widths at low scores (tie-breaking
        // depends on lane processing order). Verify consumed lengths match.
        let (sq, st) = cigar_consumed(&sse_sa.cigar);
        let (aq, at) = cigar_consumed(&avx512_sa.cigar);
        assert_eq!((sq, st), (aq, at), "single-affine CIGAR consumed lengths differ: SSE={} AVX512={}",
            cigar_to_string(&sse_sa.cigar), cigar_to_string(&avx512_sa.cigar));

        // --- Dual-affine ---
        let mut sse_da = DpResult::default();
        let mut avx512_da = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut sse_da,
            );
            extend_dual_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut avx512_da,
            );
        }
        assert_eq!(sse_da.score, avx512_da.score, "dual-affine score mismatch");
        assert_eq!(sse_da.max, avx512_da.max, "dual-affine max mismatch");
        let (dq1, dt1) = cigar_consumed(&sse_da.cigar);
        let (dq2, dt2) = cigar_consumed(&avx512_da.cigar);
        assert_eq!((dq1, dt1), (dq2, dt2), "dual-affine CIGAR consumed lengths differ");

        // --- Score-only mode ---
        let mut sse_so = DpResult::default();
        let mut avx512_so = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut sse_so,
            );
            extend_dual_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut avx512_so,
            );
        }
        assert_eq!(sse_so.score, avx512_so.score, "score-only score mismatch");
        assert_eq!(sse_so.max, avx512_so.max, "score-only max mismatch");
    }
}
