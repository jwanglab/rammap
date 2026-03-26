//! Randstrobe sketching — coupled k-mer pairs for mutation-robust seeding.
//!
//! A **strobemer** is a seed composed of two short k-mers ("strobes") at
//! variable distance. For each position `i`, strobe 1 is the k-mer at `i`,
//! and strobe 2 is selected from a window `[i + w_min, i + w_max]` by
//! minimizing a hash that mixes both strobes.
//!
//! The combined seed spans a larger genomic region than a single k-mer of
//! equivalent hash bits, providing:
//! - **Mutation robustness**: a point mutation can only affect one strobe,
//!   so many seeds survive single-base changes
//! - **Longer effective matches**: two 8-mers at variable distance cover
//!   a ~30 bp region, similar sensitivity to a 16-mer but more flexible
//! - **Better uniqueness**: coupling two positions reduces false matches
//!   compared to a single k-mer of the same length
//!
//! This implements **randstrobes** (order 2), where strobe 2 is selected
//! by `argmin(hash(h1 ^ hash(kmer2)))` over the window. This couples the
//! two positions, producing more uniformly distributed seeds than minstrobes.
//!
//! # Parameters
//! - `k`: individual strobe k-mer size (total seed covers ~k + w_min to ~k + w_max bases)
//! - `w_min`: minimum distance between strobe starts
//! - `w_max`: maximum distance between strobe starts
//!
//! # References
//!
//! Sahlin, K. (2021). Effective sequence similarity detection with strobemers.
//! Genome Research, 31(11), 2080-2094.

use crate::align::sketch::{Minimizer, Sketcher, encode_base, kmer_hash};

/// Randstrobe (order 2) sketcher.
///
/// Each seed combines two k-mers: strobe 1 at position `i` and strobe 2
/// selected from `[i + w_min, i + w_max]` by minimizing a coupled hash.
///
/// The output [`Minimizer`] encodes the combined hash in the `x` field
/// and the strobe 1 position in the `y` field, matching the standard
/// packing so seeds plug into the existing pipeline.
///
/// # Example parameters
/// - `k=10, w_min=11, w_max=20`: each seed spans 21-30 bp (good for ONT)
/// - `k=8, w_min=9, w_max=16`: each seed spans 17-24 bp (good for SR)
pub struct RandstrobeSketcher {
    pub k: usize,
    pub w_min: usize,
    pub w_max: usize,
}

impl RandstrobeSketcher {
    pub fn new(k: usize, w_min: usize, w_max: usize) -> Self {
        assert!(k > 0 && k <= 14, "strobe k must be 1..=14 (two strobes fit in 56 bits)");
        assert!(w_min >= k, "w_min must be >= k (strobes must not overlap)");
        assert!(w_max >= w_min, "w_max must be >= w_min");
        RandstrobeSketcher { k, w_min, w_max }
    }
}

impl Sketcher for RandstrobeSketcher {
    fn sketch(&self, seq: &[u8], len: usize, _rid: usize, out: &mut Vec<Minimizer>) {
        out.clear();

        let k = self.k;
        let w_min = self.w_min;
        let w_max = self.w_max;

        // Need at least k + w_max bases for one complete strobemer
        if len < k + w_max {
            return;
        }

        let k_mask = (1u64 << (2 * k)) - 1;
        let shift_k = 2 * (k - 1);

        // Precompute all k-mer hashes (forward and reverse, pick canonical)
        let n_kmers = len - k + 1;
        let mut hashes = vec![u64::MAX; n_kmers]; // canonical hash per position
        let mut strands = vec![0u8; n_kmers];     // 0=fwd, 1=rev
        let mut valid = vec![false; n_kmers];     // has k consecutive valid bases

        {
            let mut fwd: u64 = 0;
            let mut rev: u64 = 0;
            let mut consecutive: usize = 0;

            for (i, &b) in seq[..len].iter().enumerate() {
                let base = encode_base(b);
                if base < 4 {
                    fwd = ((fwd << 2) | base as u64) & k_mask;
                    rev = (rev >> 2) | ((3 - base as u64) << shift_k);
                    consecutive += 1;
                } else {
                    fwd = 0;
                    rev = 0;
                    consecutive = 0;
                }

                if consecutive >= k {
                    let pos = i + 1 - k;
                    let fh = kmer_hash(fwd, k_mask);
                    let rh = kmer_hash(rev, k_mask);
                    if fh <= rh {
                        hashes[pos] = fh;
                        strands[pos] = 0;
                    } else {
                        hashes[pos] = rh;
                        strands[pos] = 1;
                    }
                    valid[pos] = true;
                }
            }
        }

        // Generate randstrobes: for each valid strobe 1 position, find best strobe 2
        let last_strobe1 = if len >= k + w_max { len - k - w_max } else { return };

        for i in 0..=last_strobe1 {
            if !valid[i] {
                continue;
            }
            let h1 = hashes[i];

            // Search window [i + w_min, i + w_max] for strobe 2
            let win_start = i + w_min;
            let win_end = std::cmp::min(i + w_max, n_kmers - 1);

            let mut best_combined = u64::MAX;
            let mut best_j = win_start;

            for j in win_start..=win_end {
                if !valid[j] {
                    continue;
                }
                // Coupled hash: mix strobe 1's hash with strobe 2's hash
                let combined = coupled_hash(h1, hashes[j]);
                if combined < best_combined {
                    best_combined = combined;
                    best_j = j;
                }
            }

            if best_combined == u64::MAX {
                continue; // no valid strobe 2 found
            }

            // Emit the strobemer as a Minimizer
            // Combined hash goes in x; strobe 1 position goes in y
            // Use 2*k as the span (covers both strobes' k-mer lengths)
            let span = (best_j + k - i) as u64; // actual span from start of strobe1 to end of strobe2
            let x = (best_combined << 8) | std::cmp::min(span, 255);
            let y = ((i as u64) << 1) | strands[i] as u64;
            out.push(Minimizer { x, y });
        }
    }
}

/// Mix two hashes to produce a coupled randstrobe hash.
/// The coupling means strobe 2 selection depends on strobe 1's identity.
#[inline]
fn coupled_hash(h1: u64, h2: u64) -> u64 {
    // XOR the hashes and apply a finalizer to distribute bits
    let mut x = h1 ^ (h2.wrapping_mul(0x9E3779B97F4A7C15)); // golden ratio constant
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randstrobe_produces_output() {
        let seq = b"ATCGATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGA";
        let sketcher = RandstrobeSketcher::new(8, 9, 16);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        assert!(!out.is_empty(), "randstrobe should produce output");
    }

    #[test]
    fn test_randstrobe_span_in_range() {
        let seq = b"ATCGATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGA";
        let k = 8;
        let w_min = 9;
        let w_max = 16;
        let sketcher = RandstrobeSketcher::new(k, w_min, w_max);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        for m in &out {
            let span = (m.x & 0xFF) as usize;
            // Span should be between k + w_min and k + w_max
            assert!(span >= k + w_min - k && span <= k + w_max,
                "span {} out of expected range [{}, {}]",
                span, w_min, k + w_max);
        }
    }

    #[test]
    fn test_randstrobe_positions_valid() {
        let seq = b"ATCGATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGA";
        let sketcher = RandstrobeSketcher::new(8, 9, 16);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        for m in &out {
            let pos = (m.y >> 1) as usize;
            assert!(pos < seq.len(), "position {} out of bounds", pos);
        }
    }

    #[test]
    fn test_randstrobe_handles_short_sequence() {
        let sketcher = RandstrobeSketcher::new(8, 9, 16);
        let mut out = Vec::new();
        sketcher.sketch(b"ACGT", 4, 0, &mut out);
        assert!(out.is_empty(), "sequence too short for strobemers");
    }

    #[test]
    fn test_randstrobe_handles_ambiguous_bases() {
        let seq = b"ACGTACGTNNNNNNACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let sketcher = RandstrobeSketcher::new(8, 9, 16);
        let mut out = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out);
        // Should still produce output from the valid region
        for m in &out {
            let pos = (m.y >> 1) as usize;
            // No strobemers should start in the N region
            assert!(pos < 4 || pos >= 14,
                "strobemer at pos {} overlaps N region", pos);
        }
    }

    #[test]
    fn test_randstrobe_deterministic() {
        let seq = b"ATCGATCGATTAGCTAGCTAGCATGCATGCATCGATCGATTAGCTAATCGA";
        let sketcher = RandstrobeSketcher::new(8, 9, 16);
        let mut out1 = Vec::new();
        let mut out2 = Vec::new();
        sketcher.sketch(seq, seq.len(), 0, &mut out1);
        sketcher.sketch(seq, seq.len(), 0, &mut out2);
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
        }
    }
}
