//! Minimizer sketching for DNA sequences.
//!
//! Computes (w,k)-minimizers: for each window of `w` consecutive k-mers, the
//! k-mer with the smallest hash is selected as a minimizer. Both forward and
//! reverse-complement k-mers are hashed; the canonical (smaller) strand is kept.
//!
//! Each minimizer is a 128-bit [`Minimizer`] with two packed fields:
//! - **x** (reference/hash): bits 0-7 kmer_span, bits 8+ minimizer hash.
//!   After seed collection, repacked as: bits 0-31 ref_pos, bit 32+ ref_id|strand.
//! - **y** (query): bits 0-31 query_pos (bit 0 = strand), bits 32-39 kmer_span,
//!   bits 48-55 segment ID, bit 43 self-mapping flag (ava mode).
//!
//! [`sketch_sequence`] clears and fills an output `Vec<Minimizer>`;
//! [`sketch_sequence_append`] appends without clearing (for multi-segment reads).
//! Homopolymer-compressed (HPC) mode collapses runs before hashing.

use std::collections::VecDeque;

/// Anchor bit-field encoding constants for the `y` field of `Minimizer`.
/// Segment ID occupies bits 48-55 of y.
pub const SEED_SEG_SHIFT: u64 = 48;
pub const SEED_SEG_MASK: u64  = 0xFFu64 << SEED_SEG_SHIFT;
/// Flag for self-mapping anchors in overlap (ava) mode (bit 43 of y).
pub const SEED_SELF: u64      = 1u64 << 43;
/// Flag: anchor marked as long-join bridge between distant seeds (bit 40 of y).
pub const SEED_LONG_JOIN: u64 = 1u64 << 40;
/// Flag: anchor ignored during gap-fill alignment (bit 41 of y).
pub const SEED_IGNORE: u64    = 1u64 << 41;
/// Flag: anchor is part of a tandem repeat region (bit 42 of y).
pub const SEED_TANDEM: u64    = 1u64 << 42;

/// A minimizer anchor: packed 128-bit representation of a (reference, query) seed match.
///
/// **x field** (reference): `ref_id << 33 | ref_pos << 1 | strand` (upper) | position (lower 32)
/// **y field** (query): `seg_id << 48 | ... | kmer_span << 32 | query_pos << 1 | strand` (lower 32)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Minimizer {
  pub x: u64,
  pub y: u64,
}

pub type Mm128 = Minimizer;

impl Minimizer {
    /// Reference position (lower 32 bits of x, as signed integer).
    #[inline(always)]
    pub fn ref_pos(&self) -> i32 {
        self.x as u32 as i32
    }

    /// Set reference position (lower 32 bits of x), preserving upper bits (rid/strand).
    #[inline(always)]
    pub fn set_ref_pos(&mut self, pos: i32) {
        self.x = (self.x & 0xFFFFFFFF_00000000) | (pos as u32 as u64);
    }

    /// Reference ID and strand bits combined (upper 32 bits of x).
    /// Used for same-target comparisons without separating rid/strand.
    #[inline(always)]
    pub fn ref_id_strand(&self) -> u64 {
        self.x >> 32
    }

    /// Reference sequence ID (bits 33-62 of x, without strand bit).
    #[inline(always)]
    pub fn ref_id(&self) -> usize {
        ((self.x >> 32) & 0x7FFFFFFF) as usize
    }

    /// Query position (lower 32 bits of y, as signed integer).
    #[inline(always)]
    pub fn query_pos(&self) -> i32 {
        self.y as i32
    }

    /// K-mer span (bits 32-39 of y).
    #[inline(always)]
    pub fn query_span(&self) -> i32 {
        ((self.y >> 32) & 0xff) as i32
    }

    /// Segment ID for multi-segment reads (bits 48-55 of y).
    #[inline(always)]
    pub fn segment_id(&self) -> usize {
        ((self.y & SEED_SEG_MASK) >> SEED_SEG_SHIFT) as usize
    }

    /// Whether this is a self-mapping anchor in overlap (ava) mode.
    #[inline(always)]
    pub fn is_self_anchor(&self) -> bool {
        (self.y & SEED_SELF) != 0
    }

    /// Whether this anchor is ignored during gap-fill alignment.
    #[inline(always)]
    pub fn is_ignored(&self) -> bool {
        (self.y & SEED_IGNORE) != 0
    }

    /// Whether this anchor is part of a tandem repeat region.
    #[inline(always)]
    pub fn is_tandem(&self) -> bool {
        (self.y & SEED_TANDEM) != 0
    }

    /// Whether this anchor is a long-join bridge between distant seeds.
    #[inline(always)]
    pub fn is_long_join(&self) -> bool {
        (self.y & SEED_LONG_JOIN) != 0
    }
}

/// Trait for sequence sketching strategies.
///
/// A sketcher converts a DNA sequence into a set of [`Minimizer`] entries
/// that will be used for seed lookup in the index. Different strategies
/// select different subsets of k-mers (e.g., minimizers, syncmers).
///
/// The output `Minimizer` values must use the standard packing:
/// - `x`: `hash << 8 | kmer_span` (hash from [`kmer_hash`])
/// - `y`: `position << 1 | strand` (strand: 0=forward, 1=reverse)
pub trait Sketcher {
    /// Sketch a sequence, clearing and filling the output vector.
    fn sketch(&self, seq: &[u8], len: usize, rid: usize, out: &mut Vec<Minimizer>);
}

/// Lookup table for base encoding: A=0, C=1, G=2, T=3, other=4
/// Using a 256-byte table for branchless lookup (cache-friendly)
static BASE_ENCODING: [u8; 256] = {
    let mut table = [4u8; 256];
    table[b'A' as usize] = 0;
    table[b'a' as usize] = 0;
    table[b'C' as usize] = 1;
    table[b'c' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'g' as usize] = 2;
    table[b'T' as usize] = 3;
    table[b't' as usize] = 3;
    table
};

#[inline(always)]
pub fn encode_base(b: u8) -> u8 {
    BASE_ENCODING[b as usize]
}

/// Invertible 64-bit hash for k-mer bits. Reduces nucleotide composition bias.
pub fn kmer_hash(mut bits: u64, mask: u64) -> u64 {
  bits = (!bits).wrapping_add(bits << 21) & mask; // bits = (bits << 21) - bits - 1;
  bits = bits ^ (bits >> 24);
  bits = ((bits + (bits << 3)).wrapping_add(bits << 8)) & mask; // bits * 265
  bits = bits ^ (bits >> 14);
  bits = ((bits + (bits << 2)).wrapping_add(bits << 4)) & mask; // bits * 21
  bits = bits ^ (bits >> 28);
  (bits + (bits << 31)) & mask
}

/// Standard (w,k)-minimizer sketcher. Wraps [`sketch_sequence`] to implement
/// the [`Sketcher`] trait.
pub struct MinimizerSketcher {
    pub w: usize,
    pub k: usize,
    pub is_hpc: bool,
}

impl Sketcher for MinimizerSketcher {
    fn sketch(&self, seq: &[u8], len: usize, rid: usize, out: &mut Vec<Minimizer>) {
        sketch_sequence(seq, len, self.w, self.k, rid, self.is_hpc, out);
    }
}

/// Compute minimizers for a DNA sequence and write them to the output vector.
/// Clears `p` before writing.
pub fn sketch_sequence(
  seq: &[u8],
  len: usize,
  w: usize,
  k: usize,
  rid: usize,
  is_hpc: bool,
  p: &mut Vec<Minimizer>,
) {
    p.clear();
    sketch_sequence_impl(seq, len, w, k, rid, is_hpc, p);
}

/// Like sketch_sequence but appends to existing vector (does not clear).
/// Used by collect_minimizers_multi for combined multi-segment sketching.
pub fn sketch_sequence_append(
  seq: &[u8],
  len: usize,
  w: usize,
  k: usize,
  rid: usize,
  is_hpc: bool,
  p: &mut Vec<Minimizer>,
) {
    sketch_sequence_impl(seq, len, w, k, rid, is_hpc, p);
}

/// Core minimizer sketching algorithm. Appends minimizers to `p`.
/// Both sketch_sequence (clear + sketch) and sketch_sequence_append (append-only) delegate here.
fn sketch_sequence_impl(
  seq: &[u8],
  len: usize,
  w: usize,
  k: usize,
  rid: usize,
  is_hpc: bool,
  p: &mut Vec<Minimizer>,
) {
    if len == 0 {
        return;
    }
    assert!(w > 0 && w < 256 && k > 0 && k <= 28);

    let shift1 = 2 * (k - 1);
    let mask = (1u64 << (2 * k)) - 1;
    let mut kmer: [u64; 2] = [0, 0];
    let mut i: usize = 0;
    let mut l: usize = 0;
    let mut buf_pos: usize = 0;
    let mut min_pos: usize = 0;
    let mut kmer_span: usize = 0;
    let mut buf: Vec<Minimizer> = vec![Minimizer { x: 0, y: 0 }; 256];
    let mut min: Minimizer = Minimizer { x: u64::MAX, y: u64::MAX };
    let mut tq = VecDeque::new();

    while i < len {
        let c = encode_base(seq[i]);
        let mut info = Minimizer { x: u64::MAX, y: u64::MAX };
        if c < 4 { // not an ambiguous base
            if is_hpc {
                let mut skip_len = 1;
                if i + 1 < len && seq[i + 1] == seq[i] {
                    skip_len = 2;
                    while i + skip_len < len {
                        if seq[i + skip_len] != seq[i] { break; }
                        skip_len += 1;
                    }
                    i += skip_len - 1;
                }
                tq.push_back(skip_len);
                kmer_span += skip_len;
                if tq.len() > k {
                    kmer_span -= tq.pop_front().unwrap_or(0);
                }
            } else {
                kmer_span = if l + 1 < k { l + 1 } else { k };
            }
            kmer[0] = (kmer[0] << 2 | c as u64) & mask;           // forward k-mer
            kmer[1] = (kmer[1] >> 2) | (3u64 ^ c as u64) << shift1; // reverse k-mer
            if kmer[0] == kmer[1] { // skip symmetric k-mers
                i += 1;
                continue;
            }
            let z: usize = if kmer[0] < kmer[1] { 0 } else { 1 }; // strand
            l += 1;
            if l >= k && kmer_span < 256 {
                info.x = kmer_hash(kmer[z], mask) << 8 | kmer_span as u64;
                info.y = (rid as u64) << 32 | (i as u64) << 1 | z as u64;
            }
        } else {
            l = 0;
            kmer_span = 0;
            tq.clear();
        }

        buf[buf_pos] = info;

        // Special case for the first window — push identical k-mers
        if l == (w + k - 1) && min.x != u64::MAX {
            for item in &buf[(buf_pos + 1)..w] {
                if min.x == item.x && item.y != min.y { p.push(*item); }
            }
            for item in &buf[..buf_pos] {
                if min.x == item.x && item.y != min.y { p.push(*item); }
            }
        }

        if info.x <= min.x { // a new minimum; write the old min
            if l >= (w + k) && min.x != u64::MAX { p.push(min); }
            min = info;
            min_pos = buf_pos;
        } else if buf_pos == min_pos { // old min has moved outside the window
            if l >= (w + k - 1) && min.x != u64::MAX { p.push(min); }
            min.x = u64::MAX;
            for (j, item) in buf[(buf_pos + 1)..w].iter().enumerate() {
                if min.x == u64::MAX || min.x >= item.x { min = *item; min_pos = buf_pos + 1 + j; }
            }
            for (j, item) in buf[..=buf_pos].iter().enumerate() {
                if min.x == u64::MAX || min.x >= item.x { min = *item; min_pos = j; }
            }
            if l >= (w + k - 1) && min.x != u64::MAX { // write identical k-mers
                for item in &buf[(buf_pos + 1)..w] {
                    if min.x == item.x && min.y != item.y { p.push(*item); }
                }
                for item in &buf[..=buf_pos] {
                    if min.x == item.x && min.y != item.y { p.push(*item); }
                }
            }
        }

        if buf_pos + 1 == w { buf_pos = 0; } else { buf_pos += 1; }
        i += 1;
    }
    if min.x != u64::MAX {
        p.push(min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sketch_basic() {
        let seq = "GTTGATAATCACTCACTGAGTGACATCCAAATCATGGCGTCCCAAGGCACCAAACGGTCTTATGAACAGATGGAAACTGATGGAGATCGCCAGAATGCAACTGAGATTAGGGCATCCGTCGGAAAGATGATTGATGGAATTGGGAGATTCTACATCCAAATGTGCACTGAACTTAAACTCAGTGATCATGAAGGACGGTTGATCCAAAACAGCTTGACAATAGAGAAAATGGTGCTTTCTGCTTTTGATGAAAGAAGGAATAAATACCTGGAAGAACACCCCAGCGCGGGGAAAGATCCCAAGAAAACCGGGGGGCCCATATACAGGAGAGTCGATGGGAAATGGATGAGAGAACTCGTCCTTTATGACAAAGAAGAAATAAGGCGAATCTGGCGCCAAGCCAACAATGGTGAGGATGCTACATCTGGTCTAACCCACCTAATGATTTGGCATTCCAATTTGAATGATGCAACATACCAAAGGACAAGAGCTCTTGTTCGGACTGGAATGGACCCCAGAATGTGCTCTCTGATGCAGGGCTCGACTCTCCCTAGAAGGTCCGGAGCTGCCGGTGCTGCAGTCAAAGGAATCGGAACAATGGTGATGGAACTGATCAGAATGATCAAACGGGGGATCAACGATCGAAATTTTTGGAGAGGTGAGAATGGGCGGAAAACAAGAAGTGCTTATGAGAGAATGTGCAACATTCTCAAAGGAAAATTTCAAACAGCTGCACAAAAAGCAATGGTGGATCAAGTTAGAGAAAGCCGGAATCCAGGAAACGCTGAGATCGAAGATCTCATATTTTTAGCAAGATCTGCACTGATATTGAGAGGATCAGTTGCTCACAAATCTTGCCTACCTGCCTGTGCATATGGACCTGCAGTATCCAGTGGTTATGACTTTGAAAAAGAGGGATATTCCTTGGTGGGAATAGACCCTTTCAAACTACTTCAAAATAGCCAAATATACAGCTTAATCAGACCTAATGAGAATCCAGCACACAAGAGTCAGCTGGTGTGGATGGCATGTCATTCTGCTGCATTTGAAGATTTAAGATTGTTAAGCTTCATCAGAGGAACAAAAGTATCTCCTCGGGGGAAACTGTCAACTAGAGGAGTACAAATTGCTTCAAATGAGAACATGGATAATATGGGATCAAGCACTCTTGAACTGAGAAGCGGGTACTGGGCCATAAGGACCAGGAGTGGAGGAAACACTAATCAGCAGAGGGCCTCCGCAGGCCAAACCAGTGTGCAACCAACGTTTTCTGTACAAAGAAACCTCCCATTTGAAAAGTCAACCATCATGGCAGCATTCACTGGAAATACGGAAGGAAGAACTTCAGACATGAGGGCAGAAATTATAAGGATGATGGAAGGTGCAAAACCAGAAGAAGTGTCATTCCGGGGGAGGGGAGTTTTCGAGCTCTCTGACGAGAAGGCAGCGAACCCGATCGTGCCCTCTTTTGATATGAGTAACGAAGGATCTTATTTCTTCGGAGACAATGCAGAAGAATACGACAATTAAGAAAAAANNNN";
        let mut mins = Vec::new();
        sketch_sequence(seq.as_bytes(), seq.len(), 10, 15, 0, false, &mut mins);
        assert!(mins.len() > 0);
    }

    #[test]
    fn test_sketch_empty() {
        let seq = "";
        let mut mins = Vec::new();
        sketch_sequence(seq.as_bytes(), seq.len(), 10, 15, 0, false, &mut mins);
        assert_eq!(mins.len(), 0);
    }

}
