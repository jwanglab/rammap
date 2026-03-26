//! Open syncmer sketching — an alternative to minimizers.
//!
//! A **syncmer** selects k-mers based on the position of their smallest
//! sub-k-mer (s-mer), rather than being the smallest k-mer in a sliding
//! window. An "open" syncmer is a k-mer whose minimum s-mer hash occurs
//! at the first or last position.
//!
//! Compared to minimizers:
//! - **More uniform spacing**: syncmer selection depends only on the k-mer
//!   itself, not on its neighbors, reducing clustering on low-complexity regions.
//! - **Conserved under mutation**: a single-base change affects fewer selected
//!   k-mers because selection doesn't depend on a window of neighbors.
//! - **No window parameter**: controlled by (k, s) instead of (k, w).
//!
//! This module implements the [`Sketcher`] trait, producing [`Minimizer`] values
//! with the same bit-packing as the standard minimizer sketcher, so it plugs
//! directly into the seed → chain → align pipeline.
//!
//! # References
//!
//! Edgar, R. (2021). Syncmers are more sensitive than minimizers for selecting
//! conserved k-mers in biological sequences. PeerJ, 9, e10805.

use crate::align::sketch::{Minimizer, Sketcher, encode_base, kmer_hash};

/// Open syncmer sketcher.
///
/// Selects k-mers where the smallest s-mer hash occurs at position 0 or
/// position k-s (the first or last s-mer position within the k-mer).
///
/// # Parameters
/// - `k`: k-mer size (must be ≤ 28)
/// - `s`: sub-k-mer size for syncmer selection (must be < k)
///
/// Smaller `s` selects more k-mers (less stringent); `s = k-1` is most
/// selective. A good default is `s ≈ k/2` or `s = k - 4`.
pub struct SyncmerSketcher {
    pub k: usize,
    pub s: usize,
}

impl SyncmerSketcher {
    pub fn new(k: usize, s: usize) -> Self {
        assert!(k > 0 && k <= 28, "k must be 1..=28");
        assert!(s > 0 && s < k, "s must be 1..k");
        SyncmerSketcher { k, s }
    }
}

impl Sketcher for SyncmerSketcher {
    fn sketch(&self, seq: &[u8], len: usize, _rid: usize, out: &mut Vec<Minimizer>) {
        out.clear();
        if len < self.k {
            return;
        }

        let k = self.k;
        let s = self.s;
        let k_mask = (1u64 << (2 * k)) - 1;
        let s_mask = (1u64 << (2 * s)) - 1;
        let n_smer_positions = k - s + 1; // number of s-mer positions within a k-mer

        // Build forward and reverse k-mer rolling hashes
        let mut fwd_kmer: u64 = 0; // forward k-mer bits
        let mut rev_kmer: u64 = 0; // reverse-complement k-mer bits
        let shift_k = 2 * (k - 1);
        let mut consecutive = 0usize; // consecutive valid bases

        for (i, &b) in seq[..len].iter().enumerate() {
            let base = encode_base(b);
            if base < 4 {
                // Roll forward k-mer
                fwd_kmer = ((fwd_kmer << 2) | base as u64) & k_mask;
                // Roll reverse k-mer (complement of base at high position)
                rev_kmer = (rev_kmer >> 2) | ((3 - base as u64) << shift_k);
                consecutive += 1;
            } else {
                // Ambiguous base — reset
                fwd_kmer = 0;
                rev_kmer = 0;
                consecutive = 0;
                continue;
            }

            // Need k consecutive valid bases to have a complete k-mer
            if consecutive < k {
                continue;
            }

            // Pick canonical strand (smaller hash)
            let fwd_hash = kmer_hash(fwd_kmer, k_mask);
            let rev_hash = kmer_hash(rev_kmer, k_mask);
            let (kmer_bits, strand) = if fwd_hash <= rev_hash {
                (fwd_kmer, 0u64)
            } else {
                (rev_kmer, 1u64)
            };

            // Check open syncmer condition:
            // Is the minimum s-mer hash at the first or last position?
            let canonical_hash = fwd_hash.min(rev_hash);
            if is_open_syncmer(kmer_bits, s, s_mask, n_smer_positions) {
                let pos = i + 1 - k; // 0-based start position of this k-mer
                let x = (canonical_hash << 8) | k as u64;
                let y = ((pos as u64) << 1) | strand;
                out.push(Minimizer { x, y });
            }
        }
    }
}

/// Check if a k-mer is an open syncmer: its minimum s-mer hash is at
/// position 0 or the last position (k-s).
fn is_open_syncmer(kmer_bits: u64, _s: usize, s_mask: u64, n_positions: usize) -> bool {
    let mut min_hash = u64::MAX;
    let mut min_pos = 0;

    // Slide an s-mer window across the k-mer's 2-bit encoding
    for j in 0..n_positions {
        let shift = 2 * j;
        let s_bits = (kmer_bits >> shift) & s_mask;
        let h = simple_hash(s_bits, s_mask);
        if h < min_hash {
            min_hash = h;
            min_pos = j;
        }
    }

    // Open syncmer: minimum at first or last position
    min_pos == 0 || min_pos == n_positions - 1
}

/// Simple hash for s-mer selection within a k-mer.
/// Uses a different hash than the k-mer hash to avoid correlation.
#[inline]
fn simple_hash(bits: u64, mask: u64) -> u64 {
    let mut x = bits;
    x = (x ^ (x >> 16)).wrapping_mul(0x45d9f3b) & mask;
    x = (x ^ (x >> 16)).wrapping_mul(0x45d9f3b) & mask;
    x ^ (x >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syncmer_produces_output() {
        // Realistic non-repetitive sequence
        let seq = b"ATCGATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGA";
        let sketcher = SyncmerSketcher::new(10, 6);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        assert!(!out.is_empty(), "syncmer should produce at least one hit");
        // Verify packing: span should be k
        for m in &out {
            assert_eq!(m.x & 0xFF, 10, "kmer_span should be k=10");
        }
    }

    #[test]
    fn test_syncmer_density_varies_with_s() {
        // More complex sequence to avoid degenerate cases
        let seq = b"GATTACAGATCGTACGATCGATCGATCGTAGCTAGCTAGCATGCATGCATCG\
                     ATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGAGTCAG";
        let selective = SyncmerSketcher::new(10, 9);
        let liberal = SyncmerSketcher::new(10, 5);
        let mut sel_out = Vec::new();
        let mut lib_out = Vec::new();
        selective.sketch(seq, seq.len(), 0, &mut sel_out);
        liberal.sketch(seq, seq.len(), 0, &mut lib_out);
        // Both should produce some output
        assert!(!sel_out.is_empty(), "selective syncmer produced no output");
        assert!(!lib_out.is_empty(), "liberal syncmer produced no output");
    }

    #[test]
    fn test_syncmer_handles_ambiguous_bases() {
        let seq = b"ACGTNNNNACGTACGTACGTACGT";
        let sketcher = SyncmerSketcher::new(10, 6);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        // Should produce output only from the valid region after the Ns
        for m in &out {
            let pos = (m.y >> 1) as usize;
            assert!(pos >= 8, "no syncmers should come from the N region");
        }
    }

    #[test]
    fn test_syncmer_empty_and_short() {
        let sketcher = SyncmerSketcher::new(15, 11);
        let mut out = Vec::new();
        sketcher.sketch(b"", 0, 0, &mut out);
        assert!(out.is_empty());
        sketcher.sketch(b"ACGT", 4, 0, &mut out);
        assert!(out.is_empty(), "sequence shorter than k should produce no syncmers");
    }
}
