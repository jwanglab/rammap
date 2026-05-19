//! Alignment extension stage: converts chained anchors into base-level alignments.
//!
//! Entry point is `align_anchors`, which takes region-relative nt4-encoded query/target
//! sequences and a set of anchors, and produces an `AlignResult` containing CIGAR,
//! score, and coordinate mappings. The core flow for each anchor chain is:
//! left-extension, inter-anchor gap-fill, right-extension — all delegated to the
//! DP kernels in `dp.rs`. After extension, z-drop splitting (`split_zdrop`) may
//! break a single alignment into multiple sub-alignments at score dips.
//!
//! Key types: `AlignAnchorContext` bundles per-region inputs (sequences, anchors,
//! options); `AlignResult` carries the final CIGAR, scores, and query/target ranges.
//! CS, MD, and DS tag formatting functions convert CIGAR + sequences into SAM tags.

use crate::align::sketch::Minimizer;
use crate::align::dp::{self, DpResult, APPROX_MAX, RIGHT_ALIGN, EXTENSION_ONLY, REV_CIGAR,
    GENERIC_SCORING, SPLICE_FORWARD, SPLICE_REVERSE, SPLICE_FLANK, SPLICE_COMPLEX, SPLICE_SCORE};
use crate::align::index::Index;
use crate::align::junc;
use crate::align::map::{AlignFlags, MapOptions};
use std::cmp;

/// Reverse complement a DNA sequence (ASCII encoding).
pub fn rev_comp(seq: &[u8]) -> Vec<u8> {
    let mut rc = Vec::with_capacity(seq.len());
    for b in seq.iter().rev() {
        let c = match b {
            b'A' => b'T',
            b'a' => b'T',
            b'C' => b'G',
            b'c' => b'G',
            b'G' => b'C',
            b'g' => b'C',
            b'T' => b'A',
            b't' => b'A',
            b'U' => b'A',
            b'u' => b'A',
            _ => b'N',
        };
        rc.push(c);
    }
    rc
}

/// Reverse complement a nt4-encoded sequence (0=A,1=C,2=G,3=T,4=N).
pub fn rev_comp_nt4(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| if b < 4 { 3 - b } else { 4 }).collect()
}

/// Convert ASCII sequence to nt4 + reverse complement in one pass.
pub fn encode_nt4_rc(ascii_seq: &[u8]) -> Vec<u8> {
    ascii_seq.iter().rev().map(|&b| {
        let nt4 = match b { b'A'|b'a'=>0, b'C'|b'c'=>1, b'G'|b'g'=>2, b'T'|b't'=>3, _=>4 };
        if nt4 < 4 { 3 - nt4 } else { 4 }
    }).collect()
}

/// nt4 value to lowercase ASCII base.
const NT4_TO_LOWER: [u8; 5] = [b'a', b'c', b'g', b't', b'n'];

/// nt4 value to uppercase ASCII base.
const NT4_TO_UPPER: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];

// Re-export anchor flags from sketch.rs (canonical home for Minimizer bit-fields)
pub use crate::align::sketch::{SEED_IGNORE, SEED_LONG_JOIN, SEED_TANDEM};


/// Maximum query length for short-read RNA heuristic.
const MAX_QLEN_FLANK: usize = 100;

use super::dp::{CIGAR_MATCH, CIGAR_INS, CIGAR_N_SKIP};

/// Heuristic alignment for short-read RNA.
///
/// For gap-fill segments where the reference is longer than the query (potential intron),
/// constructs a synthetic target: left_flank + N_padding + right_flank, aligns with exts2,
/// and adjusts the intron length if exactly one intron is found.
/// Returns true if the heuristic succeeded (ez is filled), false to fall through to normal alignment.
fn align_short_read_rna(
    q_sub: &[u8], t_sub: &[u8], mat: &[i8],
    opt: &MapOptions, end_bonus: i32, dp_flag: i32,
    junc: Option<&[u8]>, junc_bonus: i8, junc_pen: i8,
    ez: &mut DpResult,
) -> bool {
    let q = opt.scoring.gap_open;
    let e = opt.scoring.gap_extend;
    let q2 = opt.scoring.gap_open2;
    let noncan = opt.scoring.noncanon_penalty;
    let zdrop = opt.alignment.zdrop;
    let qlen = q_sub.len();
    let tlen = t_sub.len();
    let ilen = (q2 * 2) as usize;
    let tlen2 = qlen * 2 + ilen;

    // Guard conditions
    if qlen > MAX_QLEN_FLANK || tlen2 > tlen { return false; }

    // Count exact matches from left and right
    // Note: counts ALL matching positions, not just consecutive from start
    let mut ll = 0usize;
    for i in 0..qlen {
        if q_sub[i] == t_sub[i] && q_sub[i] < 4 { ll += 1; }
    }
    let mut lr = 0usize;
    for i in 0..qlen {
        if q_sub[qlen - 1 - i] == t_sub[tlen - 1 - i] && q_sub[qlen - 1 - i] < 4 { lr += 1; }
    }
    // Too many mismatches for this heuristic
    if qlen as i32 - (ll as i32 + lr as i32) > 9 { return false; }

    // Construct synthetic target: left_qlen_bases + N*ilen + right_qlen_bases
    let mut tseq2 = vec![0u8; tlen2];
    tseq2[..qlen].copy_from_slice(&t_sub[..qlen]);
    tseq2[qlen..qlen + ilen].fill(4); // N padding
    tseq2[qlen + ilen..].copy_from_slice(&t_sub[tlen - qlen..]);

    // Construct synthetic junction array
    let junc2 = if let Some(j) = junc {
        let mut junc2 = vec![0u8; tlen2];
        junc2[..qlen].copy_from_slice(&j[..qlen]);
        // middle ilen bytes stay 0
        junc2[qlen + ilen..].copy_from_slice(&j[tlen - qlen..]);
        Some(junc2)
    } else {
        None
    };

    // Align against synthetic target
    // Note: dp_flag already includes SPLICE_COMPLEX (or not) from the caller's
    // splice_dp_flag already includes SPLICE_COMPLEX (or not) from the caller.
    dp::extend_splice(q_sub, &tseq2, 5, mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop, end_bonus, junc_bonus, junc_pen, dp_flag, junc2.as_deref(), ez);

    // Validate result
    if ez.zdropped != 0 { return false; }
    if ez.cigar.is_empty() { return false; }
    if (ez.cigar[0] & 0xf) != CIGAR_MATCH { return false; }
    if (ez.cigar[ez.cigar.len() - 1] & 0xf) != CIGAR_MATCH { return false; }

    let mut nn = 0;
    let mut n_ins = 0;
    for &op in &ez.cigar {
        if (op & 0xf) == CIGAR_N_SKIP { nn += 1; }
        else if (op & 0xf) == CIGAR_INS { n_ins += 1; }
    }
    if nn != 1 || n_ins > 0 { return false; }

    // Adjust intron length in CIGAR
    let intron_adj = (tlen as i32 - tlen2 as i32) as u32;
    for op in &mut ez.cigar {
        if (*op & 0xf) == CIGAR_N_SKIP {
            *op += intron_adj << 4;
        }
    }
    true
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32"))]
/// Encode a single ASCII base to nt4.
#[inline]
pub fn encode_nt4_byte(b: u8) -> u8 {
    match b { b'A'|b'a'=>0, b'C'|b'c'=>1, b'G'|b'g'=>2, b'T'|b't'=>3, _=>4 }
}

/// Find the longest stretch of consecutively matching anchors.
/// For SR mode, replaces fix_bad_ends + filter_bad_seeds + adjust_minier.
/// Returns (as1_offset, cnt1) into the anchors slice.
pub fn max_stretch(anchors: &[Minimizer]) -> (usize, usize) {
    let n = anchors.len();
    if n < 2 {
        return (0, n);
    }

    let mut max_score: i32 = -1;
    let mut max_i: usize = 0;
    let mut max_len: usize = 0;

    let mut score = anchors[0].query_span();
    let mut len: usize = 1;

    for i in 1..n {
        let q_span = anchors[i].query_span();
        let lr = anchors[i].ref_pos() - anchors[i - 1].ref_pos();
        let lq = anchors[i].query_pos() - anchors[i - 1].query_pos();
        if lq == lr {
            score += std::cmp::min(lq, q_span);
            len += 1;
        } else {
            if score > max_score {
                max_score = score;
                max_len = len;
                max_i = i - len;
            }
            score = q_span;
            len = 1;
        }
    }
    if score > max_score {
        max_len = len;
        max_i = n - len;
    }
    (max_i, max_len)
}

/// Compute seed extension score.
/// Extends anchor region by ext_len, aligns with lightweight DP to score match quality.
fn compute_seed_extension_score(
    qseq: &[u8],  // ASCII query on correct strand
    tseq: &[u8],  // ASCII reference (region-relative)
    anchor: &Minimizer,
    ext_len: i32,
    mat: &[i8],
    gapo: i32,
    gape: i32,
    // (index, rid, rgn_start): when Some, fetch fresh forward-strand bytes from
    // the index instead of slicing `tseq`. This matches mm2's
    // `mm_seed_ext_score` behavior and avoids clipping the extension range to
    // the chain's pre-extracted region. None for the qstrand+reverse path,
    // where `tseq` already contains the full rev-comp'd chromosome.
    extract: Option<(&crate::align::index::Index, usize, usize)>,
) -> i32 {
    let q_span = anchor.query_span();
    let re_a = anchor.ref_pos() + 1;
    let rs_a = re_a - q_span;
    let qe_a = anchor.query_pos() + 1;
    let qs_a = qe_a - q_span;

    let qlen = qseq.len() as i32;

    let qs = std::cmp::max(0, qs_a - ext_len) as usize;
    let qe = std::cmp::min(qlen, qe_a + ext_len) as usize;

    // Reference range: prefer fresh fetch from index (forward strand) so the
    // window isn't bounded by the chain's pre-extracted region.
    let mut owned_t: Vec<u8> = Vec::new();
    let t_slice: &[u8] = if let Some((mi, rid, rgn_start)) = extract {
        let chrom_len = mi.seqs[rid].len as i32;
        // Anchor coords stored in `anchor` are region-relative; rebase to
        // absolute chromosome coordinates for the index fetch.
        let abs_re_a = re_a + rgn_start as i32;
        let abs_rs_a = rs_a + rgn_start as i32;
        let abs_rs = std::cmp::max(0, abs_rs_a - ext_len) as usize;
        let abs_re = std::cmp::min(chrom_len, abs_re_a + ext_len) as usize;
        if abs_re <= abs_rs { return 0; }
        owned_t.resize(abs_re - abs_rs, 0);
        mi.extract_nt4_into(rid, abs_rs, abs_re, &mut owned_t);
        &owned_t[..]
    } else {
        // qstrand+reverse: `tseq` is the full rev-comp'd chromosome. Slicing
        // by region-relative coords here is correct and not bounded by a
        // smaller region.
        let tlen = tseq.len() as i32;
        let rs = std::cmp::max(0, rs_a - ext_len) as usize;
        let re = std::cmp::min(tlen, re_a + ext_len) as usize;
        if re <= rs { return 0; }
        &tseq[rs..re]
    };
    if qe <= qs { return 0; }

    let mut qp = dp::lightweight_profile_init((qe - qs) as i32, &qseq[qs..qe], 5, mat);
    let (score, _q_off, _t_off) = dp::lightweight_align_i16(&mut qp, t_slice.len() as i32, t_slice, gapo, gape);
    score
}

/// Fix bad boundary exons in splice chains.
/// Trims first/last anchors if they are weakly supported across large gaps
fn fix_bad_ends_splice(
    anchors: &[Minimizer],
    qseq: &[u8],
    tseq: &[u8],
    mat: &[i8],
    anchor_ext_len: i32,
    anchor_ext_shift: i32,
    gapo: i32,
    gape: i32,
    extract: Option<(&crate::align::index::Index, usize, usize)>,
) -> (usize, usize) {
    let mut as1: usize = 0;
    let mut cnt1 = anchors.len();
    if anchors.len() < 3 { return (as1, cnt1); }

    // Check first anchor
    let gap_left = anchors[1].ref_pos() - anchors[0].ref_pos();
    if gap_left > 0 {
        let log_gap = (gap_left as f64).ln();
        let span = anchors[0].query_span() as f64;
        if span < log_gap + anchor_ext_shift as f64 {
            let score = compute_seed_extension_score(qseq, tseq, &anchors[0], anchor_ext_len, mat, gapo, gape, extract);
            if (score as f64) / (mat[0] as f64) < log_gap + anchor_ext_shift as f64 {
                as1 += 1;
                cnt1 -= 1;
            }
        }
    }

    // Check last anchor (using original indices)
    let n = anchors.len();
    let gap_right = anchors[n - 1].ref_pos() - anchors[n - 2].ref_pos();
    if gap_right > 0 {
        let log_gap = (gap_right as f64).ln();
        let span = anchors[n - 1].query_span() as f64;
        if span < log_gap + anchor_ext_shift as f64 {
            let score = compute_seed_extension_score(qseq, tseq, &anchors[n - 1], anchor_ext_len, mat, gapo, gape, extract);
            if (score as f64) / (mat[0] as f64) < log_gap + anchor_ext_shift as f64 {
                cnt1 -= 1;
            }
        }
    }

    (as1, cnt1)
}

#[derive(Debug, Clone)]
pub struct CigarOp {
    pub op: char,
    pub len: u32,
}

impl std::fmt::Display for CigarOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.len, self.op)
    }
}

pub struct AlignmentContext {
    /// Reusable buffer for the extracted target region.
    /// Allocated once per thread, grown as needed, never shrunk —
    /// avoids per-mapping `malloc`/`free` churn for the multi-megabyte
    /// regions produced by chromosome-scale chains (asm presets).
    /// Filled via `set_len + extract_nt4_into` to skip the zero-fill
    /// that `vec![0u8; N]` would otherwise perform.
    pub target_buf: Vec<u8>,
}

impl Default for AlignmentContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AlignmentContext {
    pub fn new() -> Self {
        AlignmentContext { target_buf: Vec::new() }
    }
}

/// Per-call context for align_anchors (values that vary per mapping).
/// Context for a single [`align_anchors`] call. Carries per-chain metadata
/// that doesn't belong in [`MapOptions`] (which is shared across all chains).
pub struct AlignAnchorContext<'a> {
    /// (rs1, qs1, re1, qe1): nearby-seed bounds from the squeezed array.
    /// Used to compute left/right extension boundaries. Region-relative.
    pub seed_bounds: (i32, i32, i32, i32),
    /// Whether this chain is on the reverse strand.
    pub rev: bool,
    /// Reference sequence ID (for junction db lookups).
    pub rid: usize,
    /// Splice flags (SPLICE_FOR, SPLICE_REV) controlling intron orientation.
    pub splice_flag: AlignFlags,
    /// Whether this chain was produced by a z-drop split (affects zdrop threshold).
    pub split_inv: bool,
    /// Whether the index was built with homopolymer compression.
    pub is_hpc: bool,
    /// K-mer size from the index.
    pub k: usize,
    /// Optional junction database for splice-aware alignment.
    pub junc_db: Option<&'a crate::align::junc::JunctionDb>,
    /// Offset from region-relative to absolute chromosome coordinates.
    /// Added to all positions when calling junction db (which uses absolute coords).
    /// Set to `rgn_start` when tseq is a region extraction; 0 for full chromosome.
    pub ref_offset: usize,
}


/// Build 5x5 scoring matrix.
/// mat[i*5+j] = a if i==j (match), -b if i!=j (mismatch), -sc_ambi for N
pub fn build_scoring_matrix(a: i32, b: i32) -> [i8; 25] {
    build_scoring_matrix_full(a, b, 0, 1)
}

/// Build 5x5 scoring matrix with optional transition scoring.
/// Encoding: A=0, C=1, G=2, T=3, N=4
/// Transitions: A↔G (0↔2), C↔T (1↔3) — get -transition instead of -b when transition > 0
/// sc_ambi: ambiguity (N) penalty (default: 1, stored as positive, negated here)
pub fn build_scoring_matrix_full(a: i32, b: i32, transition: i32, sc_ambi: i32) -> [i8; 25] {
    let mut mat = [0i8; 25];
    let ambi = -(sc_ambi as i8);
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                mat[i * 5 + j] = a as i8;
            } else if transition > 0 && ((i == 0 && j == 2) || (i == 2 && j == 0) || (i == 1 && j == 3) || (i == 3 && j == 1)) {
                mat[i * 5 + j] = -(transition as i8);
            } else {
                mat[i * 5 + j] = -(b as i8);
            }
        }
        mat[i * 5 + 4] = ambi;
    }
    for j in 0..5 {
        mat[4 * 5 + j] = ambi;
    }
    mat
}

/// Collect positions of gaps longer than min_gap between consecutive anchors.
fn collect_long_gaps(anchors: &[Minimizer], min_gap: i32) -> Vec<usize> {
    let mut gaps = Vec::new();
    for i in 1..anchors.len() {
        let gap = (anchors[i].query_pos() - anchors[i-1].query_pos())
                - (anchors[i].ref_pos() - anchors[i-1].ref_pos());
        if gap < -min_gap || gap > min_gap {
            gaps.push(i);
        }
    }
    gaps
}

/// Filter bad seeds between anchors.
/// Marks seeds with SEED_IGNORE flag.
fn filter_bad_seeds(anchors: &mut [Minimizer], min_gap: i32, diff_thres: i32, max_ext_len: i32, max_ext_cnt: usize) {
    let k_vec = collect_long_gaps(anchors, min_gap);
    let n = k_vec.len();
    if n <= 1 { return; }

    let mut max_val = 0i32;
    let mut max_st: isize = -1;
    let mut max_en: isize = -1;
    let mut k = 0;
    loop {
        if k == n || k as isize >= max_en {
            if max_en > 0 {
                for anchor in &mut anchors[k_vec[max_st as usize]..k_vec[max_en as usize]] {
                    anchor.y |= SEED_IGNORE;
                }
            }
            max_val = 0;
            max_st = -1;
            max_en = -1;
            if k == n { break; }
        }
        let i = k_vec[k];
        let gap = (anchors[i].query_pos() - anchors[i-1].query_pos())
                - (anchors[i].ref_pos() - anchors[i-1].ref_pos());
        let mut n_ins = if gap > 0 { gap } else { 0 };
        let mut n_del = if gap < 0 { -gap } else { 0 };
        let qs_start = anchors[i-1].query_pos();
        let rs_start = anchors[i-1].ref_pos();
        let mut max_diff = 0i32;
        let mut max_diff_l: isize = -1;
        let end_l = cmp::min(n, k + 1 + max_ext_cnt);
        for (l, &j) in k_vec.iter().enumerate().take(end_l).skip(k+1) {
            if anchors[j].query_pos() - qs_start > max_ext_len || anchors[j].ref_pos() - rs_start > max_ext_len { break; }
            let gap2 = (anchors[j].query_pos() - anchors[j-1].query_pos())
                     - (anchors[j].ref_pos() - anchors[j-1].ref_pos());
            if gap2 > 0 { n_ins += gap2; } else { n_del += -gap2; }
            let diff = n_ins + n_del - (n_ins - n_del).abs();
            if max_diff < diff {
                max_diff = diff;
                max_diff_l = l as isize;
            }
        }
        if max_diff > diff_thres && max_diff > max_val {
            max_val = max_diff;
            max_st = k as isize;
            max_en = max_diff_l;
        }
        k += 1;
    }
}

/// Alternative bad seed filtering.
/// Marks seeds with SEED_IGNORE and SEED_LONG_JOIN flags.
fn filter_bad_seeds_alt(anchors: &mut [Minimizer], min_gap: i32, max_ext: i32) {
    let k_vec = collect_long_gaps(anchors, min_gap);
    let n = k_vec.len();
    if n == 0 { return; }

    let mut k = 0;
    while k < n {
        let i = k_vec[k];
        let gap1_raw = (anchors[i].query_pos() - anchors[i-1].query_pos())
                     - (anchors[i].ref_pos() - anchors[i-1].ref_pos());
        let mut gap1 = gap1_raw.abs();
        let mut re1 = anchors[i].ref_pos();
        let mut qe1 = anchors[i].query_pos();
        let mut l = k + 1;
        while l < n {
            let j = k_vec[l];
            if anchors[j].query_pos() - qe1 > max_ext || anchors[j].ref_pos() - re1 > max_ext { break; }
            let gap2_raw = (anchors[j].query_pos() - anchors[j-1].query_pos())
                         - (anchors[j].ref_pos() - anchors[j-1].ref_pos());
            let q_span_pre = anchors[j-1].query_span();
            let rs2 = anchors[j-1].ref_pos() + q_span_pre;
            let qs2 = anchors[j-1].query_pos() + q_span_pre;
            let m = cmp::min(rs2 - re1, qs2 - qe1);
            let gap2 = gap2_raw.abs();
            if m > gap1 + gap2 { break; }
            re1 = anchors[j].ref_pos();
            qe1 = anchors[j].query_pos();
            gap1 = gap2;
            l += 1;
        }
        if l > k + 1 {
            let end = k_vec[l - 1];
            for anchor in &mut anchors[k_vec[k]..end] {
                anchor.y |= SEED_IGNORE;
            }
            anchors[end].y |= SEED_LONG_JOIN;
        }
        k = l;
    }
}

/// Fix bad chain endpoints by trimming weak boundary alignments.
pub fn fix_bad_ends(anchors: &[Minimizer], bw: i32, min_match: i32) -> (usize, usize) {
    let cnt = anchors.len();
    let mut as_idx = 0usize;
    let mut cnt_out = cnt;

    if cnt < 3 { return (as_idx, cnt_out); }

    // Compute mlen for the chain (fuzzy matched length)
    let mlen = {
        let mut ml = anchors[0].query_span();
        for i in 1..cnt {
            let span = anchors[i].query_span();
            let tl = anchors[i].ref_pos() - anchors[i-1].ref_pos();
            let ql = anchors[i].query_pos() - anchors[i-1].query_pos();
            // Fuzzy length: tl>span && ql>span ? span : tl<ql ? tl : ql
            ml += if tl > span && ql > span { span } else if tl < ql { tl } else { ql };
        }
        ml
    };

    // Trim from left
    let mut m = anchors[0].query_span();
    let mut l = m;
    for i in 1..(cnt - 1) {
        let q_span = anchors[i].query_span();
        if anchors[i].y & SEED_LONG_JOIN != 0 { break; }
        let lr = anchors[i].ref_pos() - anchors[i-1].ref_pos();
        let lq = anchors[i].query_pos() - anchors[i-1].query_pos();
        let min_lrlq = cmp::min(lr, lq);
        let max_lrlq = cmp::max(lr, lq);
        if max_lrlq - min_lrlq > l >> 1 { as_idx = i; }
        l += min_lrlq;
        m += cmp::min(min_lrlq, q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 { break; }
    }
    cnt_out = cnt - as_idx;

    // Trim from right
    m = anchors[cnt - 1].query_span();
    l = m;
    for i in (as_idx+1..(cnt - 1)).rev() {
        let q_span = anchors[i+1].query_span();
        if anchors[i+1].y & SEED_LONG_JOIN != 0 { break; }
        let lr = anchors[i+1].ref_pos() - anchors[i].ref_pos();
        let lq = anchors[i+1].query_pos() - anchors[i].query_pos();
        let min_lrlq = cmp::min(lr, lq);
        let max_lrlq = cmp::max(lr, lq);
        if max_lrlq - min_lrlq > l >> 1 { cnt_out = i + 1 - as_idx; }
        l += min_lrlq;
        m += cmp::min(min_lrlq, q_span);
        if l >= bw << 1 || (m >= min_match && m >= bw) || m >= mlen >> 1 { break; }
    }

    (as_idx, cnt_out)
}

/// CIGAR post-processing: merge ops, fix boundaries, apply right-alignment.
/// Returns (qshift, tshift) for coordinate adjustment
fn fix_cigar(cigar: &mut Vec<u32>, qseq: &[u8], tseq: &[u8], qs: i32, rs: i32) -> (i32, i32) {
    let (mut qshift, mut tshift) = (0i32, 0i32);
    if cigar.len() <= 1 { return (qshift, tshift); }

    // Phase 1: Indel left-alignment
    let mut toff = 0usize;
    let mut qoff = 0usize;
    let mut to_shrink = false;
    let n = cigar.len();
    for k in 0..n {
        let op = cigar[k] & 0xf;
        let len = (cigar[k] >> 4) as usize;
        if len == 0 { to_shrink = true; }
        if op == 0 { // M
            toff += len;
            qoff += len;
        } else if op == 1 || op == 2 { // I or D
            if k > 0 && k < n - 1 && (cigar[k-1] & 0xf) == 0 && (cigar[k+1] & 0xf) == 0 {
                let prev_len = (cigar[k-1] >> 4) as usize;
                let mut l = 0usize;
                if op == 1 { // INS: compare query bases
                    let qi = qs as usize + qoff;
                    while l < prev_len {
                        if qi < l + 1 || qi + len < l + 1 { break; }
                        if qseq[qi - 1 - l] != qseq[qi + len - 1 - l] { break; }
                        l += 1;
                    }
                } else { // DEL: compare target bases
                    let ti = rs as usize + toff;
                    while l < prev_len {
                        if ti < l + 1 || ti + len < l + 1 { break; }
                        if tseq[ti - 1 - l] != tseq[ti + len - 1 - l] { break; }
                        l += 1;
                    }
                }
                if l > 0 {
                    cigar[k-1] -= (l as u32) << 4;
                    cigar[k+1] += (l as u32) << 4;
                    qoff -= l;
                    toff -= l;
                }
                if l == prev_len { to_shrink = true; }
            }
            if op == 1 { qoff += len; }
            else { toff += len; }
        } else if op == 3 { // N_SKIP
            toff += len;
        }
    }

    // Phase 2: Fix adjacent I/D sequences like 5I6D7I
    // Uses unconditional k += 1 to match C for-loop semantics (++k always runs)
    if cigar.len() >= 3 {
        let mut k = 0;
        while k + 2 < cigar.len() {
            let op_k = cigar[k] & 0xf;
            let op_k1 = cigar[k+1] & 0xf;
            if op_k > 0 && op_k + op_k1 == 3 { // I+D or D+I
                let mut s = [0u32; 3]; // s[0]=M, s[1]=I, s[2]=D
                let mut l = k;
                while l < cigar.len() {
                    let op = cigar[l] & 0xf;
                    if op == 1 || op == 2 || (cigar[l] >> 4) == 0 {
                        s[op as usize] += cigar[l] >> 4;
                        l += 1;
                    } else {
                        break;
                    }
                }
                if s[1] > 0 && s[2] > 0 && l - k > 2 {
                    cigar[k] = (s[1] << 4) | 1; // combined I
                    cigar[k+1] = (s[2] << 4) | 2; // combined D
                    for c in &mut cigar[k+2..l] {
                        *c &= 0xf; // zero out length
                    }
                    to_shrink = true;
                }
                k = l;
            }
            k += 1;
        }
    }

    // Phase 3: Squeeze zero-length ops and merge adjacent same ops
    if to_shrink {
        // Remove zero-length ops
        cigar.retain(|&c| (c >> 4) != 0);
        // Merge adjacent same-op entries
        let mut merged: Vec<u32> = Vec::with_capacity(cigar.len());
        for &c in cigar.iter() {
            if let Some(last) = merged.last_mut() {
                if (*last & 0xf) == (c & 0xf) {
                    *last += (c >> 4) << 4;
                } else {
                    merged.push(c);
                }
            } else {
                merged.push(c);
            }
        }
        *cigar = merged;
    }

    // Phase 4: Remove leading I or D
    if !cigar.is_empty() {
        let first_op = cigar[0] & 0xf;
        if first_op == 1 || first_op == 2 { // leading I or D
            let l = (cigar[0] >> 4) as i32;
            if first_op == 1 { // INS
                qshift = l;
            } else { // DEL
                tshift = l;
            }
            cigar.remove(0);
        }
    }

    (qshift, tshift)
}

/// Test z-drop on a CIGAR string.
/// Returns 0 if no zdrop, 1 if zdrop detected, 2 if inversion zdrop detected
fn test_zdrop(
    qseq: &[u8], tseq: &[u8], cigar: &[u32], mat: &[i8; 25],
    opt: &MapOptions,
) -> i32 {
    let q_gap = opt.scoring.gap_open;
    let e_gap = opt.scoring.gap_extend;
    let zdrop = opt.alignment.zdrop;
    let zdrop_inv = opt.alignment.zdrop_inv;
    let max_gap = opt.chaining.max_gap;
    let min_chain_score = opt.chaining.min_chain_score;
    let min_dp_max = opt.alignment.min_dp_max;
    let a = opt.scoring.match_score;
    let flags = opt.flags;
    let mut score: i32 = 0;
    let mut max: i32 = i32::MIN;
    let mut max_i: i32 = -1;
    let mut max_j: i32 = -1;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut max_zdrop: i32 = 0;
    // pos[0] = (t_start, t_end), pos[1] = (q_start, q_end) of worst z-drop region
    let mut pos: [[i32; 2]; 2] = [[-1, -1], [-1, -1]];

    for &c in cigar {
        let op = c & 0xf;
        let len = (c >> 4) as i32;
        if op == 0 { // M
            for _l in 0..len {
                let ti = i as usize;
                let qi = j as usize;
                if ti < tseq.len() && qi < qseq.len() {
                    score += mat[tseq[ti] as usize * 5 + qseq[qi] as usize] as i32;
                }
                // update_max_zdrop inline
                if score < max {
                    let li = i - max_i;
                    let lj = j - max_j;
                    let diff = (li - lj).abs();
                    let z = max - score - diff * e_gap;
                    if z > max_zdrop {
                        max_zdrop = z;
                        pos[0][0] = max_i; pos[0][1] = i;
                        pos[1][0] = max_j; pos[1][1] = j;
                    }
                } else {
                    max = score;
                    max_i = i;
                    max_j = j;
                }
                i += 1;
                j += 1;
            }
        } else if op == 1 || op == 2 || op == 3 { // I, D, N
            score -= q_gap + e_gap * len;
            if op == 1 { j += len; }
            else { i += len; }
            // update_max_zdrop inline
            if score < max {
                let li = i - max_i;
                let lj = j - max_j;
                let diff = (li - lj).abs();
                let z = max - score - diff * e_gap;
                if z > max_zdrop {
                    max_zdrop = z;
                    pos[0][0] = max_i; pos[0][1] = i;
                    pos[1][0] = max_j; pos[1][1] = j;
                }
            } else {
                max = score;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Test if there is an inversion in the most dropped region
    let q_len = pos[1][1] - pos[1][0];
    let t_len = pos[0][1] - pos[0][0];
    if !flags.intersects(AlignFlags::SPLICE | AlignFlags::SHORT_READ | AlignFlags::FOR_ONLY | AlignFlags::REV_ONLY)
        && max_zdrop > zdrop_inv
        && q_len < max_gap
        && t_len < max_gap
    {
        // Reverse complement the query region
        let mut qseq2 = vec![0u8; q_len as usize];
        for k in 0..q_len as usize {
            let c = qseq[pos[1][1] as usize - k - 1];
            qseq2[k] = if c >= 4 { 4 } else { 3 - c };
        }
        let mut qp = dp::lightweight_profile_init(q_len, &qseq2, 5, mat);
        let (inv_score, _q_off, _t_off) = dp::lightweight_align_i16(
            &mut qp,
            t_len,
            &tseq[pos[0][0] as usize..],
            q_gap,
            e_gap,
        );
        if inv_score >= min_chain_score * a && inv_score >= min_dp_max {
            return 2; // potential inversion
        }
    }

    if max_zdrop > zdrop { 1 } else { 0 }
}

/// Append CIGAR operations with merging.
fn append_cigar(cigar: &mut Vec<u32>, new_ops: &[u32]) {
    if new_ops.is_empty() { return; }
    if !cigar.is_empty() && (cigar.last().unwrap() & 0xf) == (new_ops[0] & 0xf) {
        // Same op at boundary — merge
        *cigar.last_mut().unwrap() += (new_ops[0] >> 4) << 4;
        cigar.extend_from_slice(&new_ops[1..]);
    } else {
        cigar.extend_from_slice(new_ops);
    }
}

pub fn fmt_cigar(ops: &[CigarOp], eqx: bool) -> String {
    if eqx {
        ops.iter().map(|op| op.to_string()).collect()
    } else {
        // Merge = and X into M
        let mut min_ops: Vec<CigarOp> = Vec::new();
        for op in ops {
            let ch = if op.op == '=' || op.op == 'X' { 'M' } else { op.op };
            if let Some(last) = min_ops.last_mut() && last.op == ch {
                last.len += op.len;
                continue;
            }
            min_ops.push(CigarOp { op: ch, len: op.len });
        }
        min_ops.iter().map(|op| op.to_string()).collect()
    }
}

pub fn calculate_alignment_score(ops: &[CigarOp], a: i32, b: i32, q: i32, e: i32, q2: i32, e2: i32) -> i32 {
    let mut score = 0;
    for op in ops {
        match op.op {
            '=' => score += a * op.len as i32,
            'X' => score -= b * op.len as i32,
            'M' => {
                 score += a * op.len as i32; 
            },
            'I' | 'D' => {
                let cost1 = q + e * op.len as i32;
                let cost2 = q2 + e2 * op.len as i32;
                score -= std::cmp::min(cost1, cost2);
            },
            _ => {} 
        }
    }
    score
}

/// Visitor pattern for CIGAR-based tag formatting.
/// Each visitor accumulates formatted output as `walk_cigar_ops` dispatches CIGAR operations.
trait CigarTagVisitor {
    /// Called for consecutive match/mismatch bases. `qi` and `ti` are 0-based offsets
    /// into the aligned query and target regions respectively.
    fn on_aligned(&mut self, op: char, len: usize, qi: usize, ti: usize);
    /// Called for insertion (query gap) operations.
    fn on_insertion(&mut self, len: usize, qi: usize);
    /// Called for deletion operations.
    fn on_deletion(&mut self, len: usize, ti: usize);
    /// Called for intron skip (N) operations.
    fn on_intron(&mut self, len: usize, ti: usize);
    /// Flush any accumulated state and return the formatted string.
    fn finish(&mut self) -> String;
}

/// Walk CIGAR ops, dispatching to the visitor for each operation type.
fn walk_cigar_ops(ops: &[CigarOp], visitor: &mut impl CigarTagVisitor) -> String {
    let mut qi = 0usize;
    let mut ti = 0usize;
    for op in ops {
        let len = op.len as usize;
        match op.op {
            '=' | 'X' | 'M' => {
                visitor.on_aligned(op.op, len, qi, ti);
                qi += len;
                ti += len;
            }
            'I' => {
                visitor.on_insertion(len, qi);
                qi += len;
            }
            'D' => {
                visitor.on_deletion(len, ti);
                ti += len;
            }
            'N' => {
                visitor.on_intron(len, ti);
                ti += len;
            }
            _ => {}
        }
    }
    visitor.finish()
}

// ---------------------------------------------------------------------------
// MdVisitor — produces MD:Z: tag
// ---------------------------------------------------------------------------

struct MdVisitor<'a> {
    tseq: &'a [u8],
    rs: usize,
    md: String,
    match_len: u32,
}

impl<'a> MdVisitor<'a> {
    fn new(tseq: &'a [u8], rs: usize) -> Self {
        Self { tseq, rs, md: String::new(), match_len: 0 }
    }
}

impl CigarTagVisitor for MdVisitor<'_> {
    #[inline]
    fn on_aligned(&mut self, op: char, len: usize, _qi: usize, ti: usize) {
        match op {
            '=' => {
                self.match_len += len as u32;
            }
            'X' => {
                let r_idx = self.rs + ti;
                for i in 0..len {
                    self.md.push_str(&self.match_len.to_string());
                    self.match_len = 0;
                    if r_idx + i < self.tseq.len() {
                        self.md.push(Index::NT4_TO_ASCII[self.tseq[r_idx + i].min(4) as usize] as char);
                    }
                }
            }
            'M' => {
                // Should not happen if coming from align_anchors producing =/X, but handle safely
            }
            _ => {}
        }
    }

    #[inline]
    fn on_insertion(&mut self, _len: usize, _qi: usize) {
        // Ignored in MD. Ref index does not move.
    }

    #[inline]
    fn on_deletion(&mut self, len: usize, ti: usize) {
        let r_idx = self.rs + ti;
        // Deletion from Reference
        // (match_len even when 0, e.g., after a mismatch: "0^ACGT")
        self.md.push_str(&self.match_len.to_string());
        self.match_len = 0;
        self.md.push('^');
        for i in 0..len {
            if r_idx + i < self.tseq.len() {
                self.md.push(Index::NT4_TO_ASCII[self.tseq[r_idx + i].min(4) as usize] as char);
            }
        }
    }

    #[inline]
    fn on_intron(&mut self, _len: usize, _ti: usize) {
        // Intron skip: not included in MD (ref index advanced by walk_cigar_ops)
    }

    fn finish(&mut self) -> String {
        if self.match_len > 0 {
            self.md.push_str(&self.match_len.to_string());
        }
        std::mem::take(&mut self.md)
    }
}

// ---------------------------------------------------------------------------
// CsVisitor — produces cs:Z: tag
// ---------------------------------------------------------------------------

struct CsVisitor<'a> {
    qseq: &'a [u8],
    tseq: &'a [u8],
    qs: usize,
    rs: usize,
    long: bool,
    cs: String,
}

impl<'a> CsVisitor<'a> {
    fn new(qseq: &'a [u8], tseq: &'a [u8], qs: usize, rs: usize, long: bool) -> Self {
        Self { qseq, tseq, qs, rs, long, cs: String::new() }
    }

    #[inline]
    fn flush_match(&mut self, run: &mut Vec<u8>) {
        if run.is_empty() { return; }
        if self.long {
            self.cs.push('=');
            self.cs.extend(run.iter().map(|&b| b as char));
        } else {
            self.cs.push_str(&format!(":{}", run.len()));
        }
        run.clear();
    }
}

impl CigarTagVisitor for CsVisitor<'_> {
    #[inline]
    fn on_aligned(&mut self, op: char, len: usize, qi: usize, ti: usize) {
        let q_idx = self.qs + qi;
        let r_idx = self.rs + ti;
        match op {
            '=' => {
                if self.long {
                    self.cs.push('=');
                    for j in 0..len {
                        if q_idx + j < self.qseq.len() {
                            self.cs.push(NT4_TO_UPPER[self.qseq[q_idx + j].min(4) as usize] as char);
                        }
                    }
                } else {
                    self.cs.push_str(&format!(":{}", len));
                }
            }
            'X' => {
                for j in 0..len {
                    let qb = NT4_TO_LOWER[self.qseq[q_idx + j].min(4) as usize];
                    let rb = NT4_TO_LOWER[self.tseq[r_idx + j].min(4) as usize];
                    self.cs.push_str(&format!("*{}{}", rb as char, qb as char));
                }
            }
            'M' => {
                let mut run: Vec<u8> = Vec::new();
                for j in 0..len {
                    if (q_idx + j) < self.qseq.len() && (r_idx + j) < self.tseq.len() {
                        let qb = self.qseq[q_idx + j];
                        let rb = self.tseq[r_idx + j];
                        if qb == rb {
                            run.push(NT4_TO_UPPER[qb.min(4) as usize]);
                        } else {
                            self.flush_match(&mut run);
                            self.cs.push_str(&format!(
                                "*{}{}",
                                NT4_TO_LOWER[rb.min(4) as usize] as char,
                                NT4_TO_LOWER[qb.min(4) as usize] as char
                            ));
                        }
                    }
                }
                self.flush_match(&mut run);
            }
            _ => {}
        }
    }

    #[inline]
    fn on_insertion(&mut self, len: usize, qi: usize) {
        let q_idx = self.qs + qi;
        self.cs.push('+');
        for i in 0..len {
            if q_idx + i < self.qseq.len() {
                self.cs.push(NT4_TO_LOWER[self.qseq[q_idx + i].min(4) as usize] as char);
            }
        }
    }

    #[inline]
    fn on_deletion(&mut self, len: usize, ti: usize) {
        let r_idx = self.rs + ti;
        self.cs.push('-');
        for i in 0..len {
            if r_idx + i < self.tseq.len() {
                self.cs.push(NT4_TO_LOWER[self.tseq[r_idx + i].min(4) as usize] as char);
            }
        }
    }

    #[inline]
    fn on_intron(&mut self, len: usize, ti: usize) {
        let r_idx = self.rs + ti;
        if len >= 2 && r_idx + len <= self.tseq.len() {
            let d1 = NT4_TO_LOWER[self.tseq[r_idx].min(4) as usize] as char;
            let d2 = NT4_TO_LOWER[self.tseq[r_idx + 1].min(4) as usize] as char;
            let a1 = NT4_TO_LOWER[self.tseq[r_idx + len - 2].min(4) as usize] as char;
            let a2 = NT4_TO_LOWER[self.tseq[r_idx + len - 1].min(4) as usize] as char;
            self.cs.push_str(&format!("~{}{}{}{}{}", d1, d2, len, a1, a2));
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.cs)
    }
}

// ---------------------------------------------------------------------------
// DsVisitor — produces ds:Z: tag (homology-aware indel brackets)
// ---------------------------------------------------------------------------

struct DsVisitor<'a> {
    qseq: &'a [u8],
    tseq: &'a [u8],
    qs: usize,
    rs: usize,
    q_len: usize,
    t_len: usize,
    ds: String,
}

impl<'a> DsVisitor<'a> {
    fn new(qseq: &'a [u8], tseq: &'a [u8], qs: usize, rs: usize, q_len: usize, t_len: usize) -> Self {
        Self { qseq, tseq, qs, rs, q_len, t_len, ds: String::new() }
    }
}

impl CigarTagVisitor for DsVisitor<'_> {
    #[inline]
    fn on_aligned(&mut self, op: char, len: usize, qi: usize, ti: usize) {
        let q_idx = self.qs + qi;
        let r_idx = self.rs + ti;
        match op {
            '=' => {
                self.ds.push_str(&format!(":{}", len));
            }
            'X' => {
                for j in 0..len {
                    let qb = NT4_TO_LOWER[self.qseq[q_idx + j].min(4) as usize];
                    let rb = NT4_TO_LOWER[self.tseq[r_idx + j].min(4) as usize];
                    self.ds.push_str(&format!("*{}{}", rb as char, qb as char));
                }
            }
            'M' => {
                let mut match_len = 0u32;
                for j in 0..len {
                    if (q_idx + j) < self.qseq.len() && (r_idx + j) < self.tseq.len() {
                        let qb = self.qseq[q_idx + j];
                        let rb = self.tseq[r_idx + j];
                        if qb == rb {
                            match_len += 1;
                        } else {
                            if match_len > 0 {
                                self.ds.push_str(&format!(":{}", match_len));
                                match_len = 0;
                            }
                            self.ds.push_str(&format!(
                                "*{}{}",
                                NT4_TO_LOWER[rb.min(4) as usize] as char,
                                NT4_TO_LOWER[qb.min(4) as usize] as char
                            ));
                        }
                    }
                }
                if match_len > 0 {
                    self.ds.push_str(&format!(":{}", match_len));
                }
            }
            _ => {}
        }
    }

    #[inline]
    fn on_insertion(&mut self, len: usize, qi: usize) {
        // Homology-aware insertion formatting
        let y = qi; // current query offset (relative to aligned region start)
        // Right homology: count z from 1..=len where qseq[y+len-z] == qseq[y-z]
        let mut lr: usize = 0;
        for z in 1..=len {
            if y < z { break; } // y - z < 0
            if self.qseq[self.qs + y + len - z] != self.qseq[self.qs + y - z] { break; }
            lr += 1;
        }
        // Left homology: count z from 0..len where qseq[y+len+z] == qseq[y+z]
        let mut ll: usize = 0;
        for z in 0..len {
            if y + len + z >= self.q_len { break; }
            if self.qseq[self.qs + y + len + z] != self.qseq[self.qs + y + z] { break; }
            ll += 1;
        }
        self.ds.push('+');
        write_indel_ds(&mut self.ds, len, &self.qseq[self.qs + y..], ll, lr);
    }

    #[inline]
    fn on_deletion(&mut self, len: usize, ti: usize) {
        // Homology-aware deletion formatting
        let x = ti; // current target offset (relative to aligned region start)
        // Right homology: count z from 1..=len where tseq[x+len-z] == tseq[x-z]
        let mut lr: usize = 0;
        for z in 1..=len {
            if x < z { break; } // x - z < 0
            if self.tseq[self.rs + x + len - z] != self.tseq[self.rs + x - z] { break; }
            lr += 1;
        }
        // Left homology: count z from 0..len where tseq[x+len+z] == tseq[x+z]
        let mut ll: usize = 0;
        for z in 0..len {
            if x + len + z >= self.t_len { break; }
            if self.tseq[self.rs + x + z] != self.tseq[self.rs + x + len + z] { break; }
            ll += 1;
        }
        self.ds.push('-');
        write_indel_ds(&mut self.ds, len, &self.tseq[self.rs + x..], ll, lr);
    }

    #[inline]
    fn on_intron(&mut self, len: usize, ti: usize) {
        let r_idx = self.rs + ti;
        if len >= 2 && r_idx + len <= self.tseq.len() {
            let d1 = NT4_TO_LOWER[self.tseq[r_idx].min(4) as usize] as char;
            let d2 = NT4_TO_LOWER[self.tseq[r_idx + 1].min(4) as usize] as char;
            let a1 = NT4_TO_LOWER[self.tseq[r_idx + len - 2].min(4) as usize] as char;
            let a2 = NT4_TO_LOWER[self.tseq[r_idx + len - 1].min(4) as usize] as char;
            self.ds.push_str(&format!("~{}{}{}{}{}", d1, d2, len, a1, a2));
        }
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.ds)
    }
}

// ---------------------------------------------------------------------------
// Public formatting functions — thin wrappers over walk_cigar_ops
// ---------------------------------------------------------------------------

pub fn fmt_md(
    ops: &[CigarOp],
    tseq: &[u8],
    rs: usize,
) -> String {
    let mut visitor = MdVisitor::new(tseq, rs);
    walk_cigar_ops(ops, &mut visitor)
}

/// Alignment result with additional statistics.
/// Output of [`align_anchors`]. All coordinates are region-relative (the caller
/// adds `ref_offset` to convert ref coords back to absolute chromosome positions).
///
/// If z-drop splitting occurred, `split_right_anchors` contains the right-side
/// anchors (also region-relative) for a separate alignment pass.
pub struct AlignResult {
    /// CIGAR operations in =/X/I/D/N format (M ops are split into =/X).
    pub cigar_ops: Vec<CigarOp>,
    /// Aligned query start (region-relative for query).
    pub query_start: usize,
    /// Aligned query end (exclusive).
    pub query_end: usize,
    /// Aligned reference start (region-relative).
    pub ref_start: usize,
    /// Aligned reference end (exclusive, region-relative).
    pub ref_end: usize,
    /// Sum of DP scores from all alignment segments (left ext + gap fills + right ext).
    pub dp_score: i32,
    /// Right-side anchors from z-drop split (region-relative ref_pos). Caller must
    /// restore to absolute coords before using in a subsequent alignment.
    pub split_right_anchors: Option<Vec<Minimizer>>,
    /// Offset of the split point within the chain's anchor array.
    pub split_offset_in_orig: Option<usize>,
    /// True if z-drop split detected inversion (zdrop_code == 2).
    pub split_inv: bool,
}

/// HPC-adjusted coordinate for a position in a sequence.
/// Walks backward from `pos` while the base matches, returning the adjusted start.
/// Adjust anchor offset to chain start index.
fn hpc_adjust_ref(seq: &[u8], pos: i32) -> i32 {
    if pos <= 0 || pos as usize >= seq.len() { return pos; }
    let c = seq[pos as usize];
    let mut i = pos - 1;
    while i > 0 && seq[i as usize] == c { i -= 1; }
    if seq[i as usize] != c { i + 1 } else { i }
}

/// HPC-adjusted coordinate for query.
fn hpc_adjust_query(seq: &[u8], pos: i32) -> i32 {
    if pos <= 0 || pos as usize >= seq.len() { return pos; }
    let c = seq[pos as usize];
    let mut i = pos - 1;
    while i > 0 && seq[i as usize] == c { i -= 1; }
    i + 1
}

/// Result of anchor trimming and coordinate setup.
struct TrimResult {
    as1_offset: usize,
    cnt1: usize,
    work_anchors: Vec<Minimizer>,
    rs: i32,
    qs: i32,
    re: i32,
    qe: i32,
}

/// Trim anchors and compute alignment coordinates.
/// SR mode uses max_stretch; non-SR uses fix_bad_ends + filter_bad_seeds + hpc_adjust.
/// Returns None if no valid anchors remain after trimming.
fn trim_and_prepare_anchors(
    anchors: &mut [Minimizer],
    qseq: &[u8],
    tseq: &[u8],
    mat: &[i8; 25],
    opt: &MapOptions,
    is_hpc: bool,
    splice_flag: AlignFlags,
    k_half: i32,
    extract: Option<(&crate::align::index::Index, usize, usize)>,
) -> Option<TrimResult> {
    let w = opt.chaining.bandwidth;
    let min_chain_score = opt.chaining.min_chain_score;
    let max_gap = opt.chaining.max_gap;
    let is_splice = opt.filtering.is_splice;
    let is_sr = opt.flags.contains(AlignFlags::SHORT_READ);
    let anchor_ext_len = opt.alignment.anchor_ext_len;
    let anchor_ext_shift = opt.alignment.anchor_ext_shift;
    let q = opt.scoring.gap_open;
    let e = opt.scoring.gap_extend;
    if is_sr && !is_hpc {
        // SR path: max_stretch replaces fix_bad_ends + filter + adjust
        let (off, cnt) = max_stretch(anchors);
        if cnt == 0 { return None; }
        let work = anchors[off..off + cnt].to_vec();
        let first = &work[0];
        let last = &work[cnt - 1];
        let q_span_first = first.query_span();
        Some(TrimResult {
            as1_offset: off,
            cnt1: cnt,
            rs: first.ref_pos() + 1 - q_span_first,
            qs: first.query_pos() + 1 - q_span_first,
            re: last.ref_pos() + 1,
            qe: last.query_pos() + 1,
            work_anchors: work,
        })
    } else {
        // Non-SR path
        let no_end_flt = splice_flag.contains(AlignFlags::NO_END_FLT);
        let (off, cnt) = if no_end_flt {
            (0, anchors.len())
        } else if !is_splice {
            fix_bad_ends(anchors, w, min_chain_score * 2)
        } else {
            fix_bad_ends_splice(anchors, qseq, tseq, mat, anchor_ext_len, anchor_ext_shift, q, e, extract)
        };
        if cnt == 0 { return None; }

        let mut wa = anchors[off..off + cnt].to_vec();

        // Filter bad seeds
        filter_bad_seeds(&mut wa, 10, 40, max_gap >> 1, 10);
        filter_bad_seeds_alt(&mut wa, 30, max_gap >> 1);

        // Propagate SEED_IGNORE and SEED_LONG_JOIN flags back to original anchors
        for i in 0..cnt {
            let fl = wa[i].y & (SEED_IGNORE | SEED_LONG_JOIN);
            if fl != 0 {
                anchors[off + i].y |= fl;
            }
        }

        // Adjust anchor coordinates (HPC or k-mer offset)
        let first = &wa[0];
        let last = &wa[cnt - 1];
        let (rs, qs, re, qe) = if is_hpc {
            (hpc_adjust_ref(tseq, first.ref_pos()),
             hpc_adjust_query(qseq, first.query_pos()),
             hpc_adjust_ref(tseq, last.ref_pos()),
             hpc_adjust_query(qseq, last.query_pos()))
        } else {
            (first.ref_pos() - k_half,
             first.query_pos() - k_half,
             last.ref_pos() - k_half,
             last.query_pos() - k_half)
        };

        Some(TrimResult {
            as1_offset: off,
            cnt1: cnt,
            work_anchors: wa,
            rs, qs, re, qe,
        })
    }
}

/// Compute left extension boundary (rs0, qs0) for non-SR mode.
/// `orig_anchor_rpos`/`orig_anchor_qpos`: raw coordinates of original chain start anchor.
/// `seed_r_bound`/`seed_q_bound`: rs1/qs1 from nearby-seed inspection.
/// Returns (rs0, qs0).
fn compute_left_boundary(
    rs: i32, qs: i32,
    orig_anchor_rpos: i32, orig_anchor_qpos: i32, orig_q_span: i32,
    seed_r_bound: i32, seed_q_bound: i32,
    opt: &MapOptions,
) -> (i32, i32) {
    let max_gap = opt.chaining.max_gap;
    let a = opt.scoring.match_score;
    let q = opt.scoring.gap_open;
    let e = opt.scoring.gap_extend;
    let mut rs0 = cmp::max(0, orig_anchor_rpos + 1 - orig_q_span);
    let mut qs0 = orig_anchor_qpos + 1 - orig_q_span;
    let (mut rs1_bound, mut qs1_bound) = if seed_r_bound > 0 || seed_q_bound > 0 {
        (seed_r_bound, seed_q_bound)
    } else {
        (0i32, 0i32)
    };

    if qs > 0 && rs > 0 {
        // Extend boundary by gap allowance
        let mut l = cmp::min(qs, max_gap);
        qs1_bound = cmp::max(qs1_bound, qs - l);
        qs0 = cmp::min(qs0, qs1_bound);
        l += if l * a > q { (l * a - q) / e } else { 0 };
        l = cmp::min(l, max_gap);
        l = cmp::min(l, rs);
        rs1_bound = cmp::max(rs1_bound, rs - l);
        rs0 = cmp::min(rs0, rs1_bound);
        rs0 = cmp::min(rs0, rs);
    } else {
        rs0 = rs;
        qs0 = qs;
    }
    (cmp::max(0, rs0), cmp::max(0, qs0))
}

/// Compute right extension boundary (re0, qe0) for non-SR mode.
/// `orig_anchor_rpos`/`orig_anchor_qpos`: raw coordinates of original chain end anchor.
/// `seed_r_bound`/`seed_q_bound`: re1/qe1 from nearby-seed inspection.
/// Returns (re0, qe0).
fn compute_right_boundary(
    re: i32, qe: i32,
    orig_anchor_rend: i32, orig_anchor_qend: i32,
    seed_r_bound: i32, seed_q_bound: i32,
    opt: &MapOptions,
    qlen: i32, tlen: i32,
    _seed_bounds: (i32, i32, i32, i32),
) -> (i32, i32) {
    let max_gap = opt.chaining.max_gap;
    let a = opt.scoring.match_score;
    let q = opt.scoring.gap_open;
    let e = opt.scoring.gap_extend;
    let mut re0 = orig_anchor_rend;
    let mut qe0 = orig_anchor_qend;
    let (mut re1_bound, mut qe1_bound) = if seed_r_bound < tlen || seed_q_bound < qlen {
        (seed_r_bound, seed_q_bound)
    } else {
        (tlen, qlen)
    };

    let mut l = cmp::min(qlen - qe, max_gap);
    qe1_bound = cmp::min(qe1_bound, qe + l);
    qe0 = cmp::max(qe0, qe1_bound);
    l += if l * a > q { (l * a - q) / e } else { 0 };
    l = cmp::min(l, max_gap);
    l = cmp::min(l, tlen - re);
    re1_bound = cmp::min(re1_bound, re + l);
    re0 = cmp::max(re0, re1_bound);

    (cmp::min(re0, tlen), cmp::min(qe0, qlen))
}

/// CIGAR post-processing: fix_cigar + convert M→=/X + condense adjacent ops.
/// Returns (final_cigar_ops, adjusted_qs, adjusted_rs).
fn finalize_cigar(
    mut raw_cigar: Vec<u32>,
    qseq: &[u8],
    tseq: &[u8],
    qs: i32,
    rs: i32,
) -> (Vec<CigarOp>, i32, i32) {
    let mut final_qs = qs;
    let mut final_rs = rs;
    let (qshift, tshift) = fix_cigar(&mut raw_cigar, qseq, tseq, final_qs, final_rs);
    if qshift > 0 {
        final_qs += qshift;
    }
    if tshift > 0 {
        final_rs += tshift;
    }

    let cigar_ops = convert_cigar_to_eqx(&raw_cigar, qseq, tseq, final_qs as usize, final_rs as usize);

    // Condense adjacent same-type ops
    let mut condensed: Vec<CigarOp> = Vec::new();
    for op in cigar_ops {
        if let Some(last) = condensed.last_mut() {
            if last.op == op.op {
                last.len += op.len;
            } else {
                condensed.push(op);
            }
        } else {
            condensed.push(op);
        }
    }

    (condensed, final_qs, final_rs)
}

/// Align a chain of anchors to produce CIGAR operations.
///
/// This is the monolithic alignment entry point.
/// It performs left extension, per-gap fill, and right extension using DP
/// kernels, with z-drop splitting and anchor trimming.
///
/// # Inputs
/// - `anchors`: sorted by ref_pos, **region-relative** coordinates (caller subtracts
///   `rgn_start` before calling). May be mutated in place (SEED_IGNORE/SEED_LONG_JOIN
///   flags set during anchor trimming).
/// - `qseq`: nt4-encoded query sequence (0=A,1=C,2=G,3=T,4=N), on the correct strand.
/// - `tseq`: nt4-encoded target region (region-relative, typically 10-200 KB).
/// - `opt`: scoring/chaining/alignment parameters.
/// - `call`: per-chain context (seed_bounds, junction db, ref_offset for absolute coords).
///
/// # Outputs
/// [`AlignResult`] with region-relative coordinates. The caller adds `ref_offset` to
/// convert `ref_start`/`ref_end` back to absolute chromosome positions. Split anchors
/// (if any) also have region-relative ref_pos.
/// `lazy_extract` parameter (`Option<(mi, rid, rgn_start)>`): when `Some`,
/// segments of `tseq` are filled on demand from `mi` (each DP call fills its
/// own segment; post-DP processing fills the final alignment range). When
/// `None`, `tseq` is assumed pre-filled (qstrand+reverse path, where the
/// buffer holds a rev-comp'd full chromosome).
pub fn align_anchors(
    anchors: &mut [Minimizer],
    qseq: &[u8],
    tseq: &mut [u8],
    lazy_extract: Option<(&crate::align::index::Index, usize, usize)>,
    opt: &MapOptions,
    call: &AlignAnchorContext,
) -> AlignResult {
    // Local bindings from opt/call to avoid modifying the function body
    let a = opt.scoring.match_score;
    let b = opt.scoring.mismatch_penalty;
    let q = opt.scoring.gap_open;
    let e = opt.scoring.gap_extend;
    let q2 = opt.scoring.gap_open2;
    let e2 = opt.scoring.gap_extend2;
    let w = opt.chaining.bandwidth;
    let w_long = opt.chaining.bandwidth_long;
    let zdrop = opt.alignment.zdrop;
    let zdrop_inv = opt.alignment.zdrop_inv;
    let end_bonus = opt.alignment.end_bonus;
    let is_splice = opt.filtering.is_splice;
    let _k = call.k;
    let seed_bounds = call.seed_bounds;
    let min_cnt = opt.chaining.min_cnt;
    let max_sw_mat = opt.alignment.max_sw_mat;
    let transition = opt.scoring.transition;
    let sc_ambi = opt.scoring.ambig_penalty;
    let is_hpc = call.is_hpc;
    let noncan = opt.scoring.noncanon_penalty;
    let splice_flag = call.splice_flag;
    let rev = call.rev;
    let flags = opt.flags;
    let min_dp_len = opt.alignment.min_dp_len;
    let junc_db = call.junc_db;
    let rid = call.rid;
    let ref_offset = call.ref_offset as i32;
    let junc_bonus = opt.scoring.junc_bonus;
    let junc_pen = opt.scoring.junc_pen;
    let split_inv = call.split_inv;

    // --- Function body (formerly align_anchors_full) ---
    let empty = AlignResult { cigar_ops: Vec::new(), query_start: 0, query_end: 0, ref_start: 0, ref_end: 0, dp_score: 0, split_right_anchors: None, split_offset_in_orig: None, split_inv: false };
    if anchors.is_empty() { return empty; }


    let mat = build_scoring_matrix_full(a, b, transition, sc_ambi);

    // Bandwidth scaling
    let bw = (w as f64 * 1.5 + 1.0) as i32;
    let mut bw_long = (w_long as f64 * 1.5 + 1.0) as i32;
    if bw_long < bw { bw_long = bw; }

    let qlen = qseq.len() as i32;
    let tlen = tseq.len() as i32;
    let k_half = (_k >> 1) as i32;

    // Generic scoring flag: needed when transition != b (non-uniform mismatch penalties)
    let generic_sc_flag = if transition != 0 && transition != b { GENERIC_SCORING } else { 0 };

    // Splice DP flags
    let mut splice_dp_flag: i32 = 0;
    if is_splice {
        if splice_flag.contains(AlignFlags::SPLICE_FOR) {
            splice_dp_flag |= if rev { SPLICE_REVERSE } else { SPLICE_FORWARD };
        }
        if splice_flag.contains(AlignFlags::SPLICE_REV) {
            splice_dp_flag |= if rev { SPLICE_FORWARD } else { SPLICE_REVERSE };
        }
        if splice_flag.contains(AlignFlags::SPLICE_FLANK) {
            splice_dp_flag |= SPLICE_FLANK;
        }
        if !flags.contains(AlignFlags::SPLICE_OLD) {
            splice_dp_flag |= SPLICE_COMPLEX;
        }
        // Add SPLICE_SCORE flag for SPSC mode
        if let Some(junc::JunctionDb::Spsc(_)) = junc_db {
            splice_dp_flag |= SPLICE_SCORE;
        }
    }
    // Compute is_rev_splice for junction lookup strand
    let is_rev_splice = (splice_dp_flag & SPLICE_REVERSE) != 0;

    // --- Steps 1-3: Anchor trimming + coordinate computation ---
    let is_sr = flags.contains(AlignFlags::SHORT_READ);

    // Lazy fill: for non-HPC non-splice modes (e.g. asm/map-ont), `tseq`
    // arrives uninitialized and we fill segments on demand before each DP
    // call + once for the final alignment range before CIGAR finalization.
    // HPC/splice/SR need `tseq` filled up front because the trim path reads
    // it. When `lazy_extract` is `None` (qstrand+reverse path), the caller
    // has already filled `tseq`.
    let use_lazy = lazy_extract.is_some() && !is_hpc && !is_splice && !is_sr;
    if let Some((mi, rid, rgn_start)) = lazy_extract && !use_lazy {
        let len = tseq.len();
        mi.extract_nt4_into(rid, rgn_start, rgn_start + len, tseq);
    }

    let trim = match trim_and_prepare_anchors(
        anchors, qseq, tseq, &mat, opt,
        is_hpc, splice_flag, k_half, lazy_extract,
    ) {
        Some(t) => t,
        None => return empty,
    };
    let TrimResult { as1_offset, cnt1, work_anchors, rs, qs, re, qe } = trim;



    // Raw CIGAR accumulator (u32 format: len<<4 | op)
    let mut raw_cigar: Vec<u32> = Vec::new();
    let mut dp_score: i32 = 0;

    // is_sr_rna = (flag & SR_RNA) && is_splice
    let is_sr_rna = flags.contains(AlignFlags::SR_RNA) && is_splice;

    // --- Step 4: Compute extension boundaries ---
    let (rs0, qs0);
    if is_sr {
        // SR mode: force full query coverage
        qs0 = 0;
        let mut l = qs;
        l += if l * a + end_bonus > q { (l * a + end_bonus - q) / e } else { 0 };
        rs0 = cmp::max(0, rs - l);
    } else {
        // Non-SR mode
        let orig_first = &anchors[0];
        let (rs0_tmp, qs0_tmp) = compute_left_boundary(
            rs, qs,
            orig_first.ref_pos(), orig_first.query_pos(), orig_first.query_span(),
            seed_bounds.0, seed_bounds.1,
            opt,
        );
        rs0 = rs0_tmp;
        qs0 = qs0_tmp;
    }

    // SEED_SELF: clamp leftward extension distance to |q-r| of the chain's
    // FIRST original anchor (pre-trim) in absolute coords. `pre_qs` is in
    // absolute alignment-space query coords. `pre_rs` (after the earlier
    // rebase) is in region-relative ref coords; shift by `ref_offset` so
    // max_ext is computed in matching coord systems.
    let (mut rs0, mut qs0) = (rs0, qs0);
    if !anchors.is_empty() && (anchors[0].y & crate::align::map::SEED_SELF) != 0 {
        let pre_q_span = anchors[0].query_span();
        let pre_qs = anchors[0].query_pos() + 1 - pre_q_span;
        let pre_rs = anchors[0].ref_pos() + 1 - pre_q_span;
        let pre_rs_abs = pre_rs + ref_offset;
        let max_ext = (pre_qs - pre_rs_abs).unsigned_abs() as i32;
        if pre_rs - rs0 > max_ext { rs0 = pre_rs - max_ext; }
        if pre_qs - qs0 > max_ext { qs0 = pre_qs - max_ext; }
    }


    // --- Step 5: Left extension ---
    let (mut rs1, mut qs1) = (rs, qs);
    if qs > 0 && rs > 0 && qs > qs0 && rs > rs0 {
        let q_ext_len = (qs - qs0) as usize;

        // Lazy fill the left-extension segment [rs0, rs) in tseq.
        if use_lazy && let Some((mi, rid, rgn_start)) = lazy_extract {
            let a = rs0 as usize;
            let b = rs as usize;
            mi.extract_nt4_into(rid, rgn_start + a, rgn_start + b, &mut tseq[a..b]);
        }

        // Reverse sequences for left extension
        let mut q_rev: Vec<u8> = qseq[qs0 as usize..qs as usize].to_vec();
        let mut t_rev: Vec<u8> = tseq[rs0 as usize..rs as usize].to_vec();
        q_rev.reverse();
        t_rev.reverse();

        // Junction buffer for left extension: get, then reverse
        let junc_left = if let Some(db) = junc_db {
            let mut jbuf = vec![0u8; (rs - rs0) as usize];
            junc::get_junc(db, rid, rs0 + ref_offset, rs + ref_offset, is_rev_splice, &mut jbuf);
            jbuf.reverse();
            Some(jbuf)
        } else {
            None
        };

        // Left extension flags: EXTZ_ONLY | RIGHT | REV_CIGAR
        let flag = EXTENSION_ONLY | RIGHT_ALIGN | REV_CIGAR | generic_sc_flag;

        // Left extension zdrop: use zdrop_inv for split_inv chains
        let left_zdrop = if split_inv { zdrop_inv } else { zdrop };

        let mut ez = DpResult::default();
        // max_sw_mat check: skip alignment if matrix too large
        let sw_mat_size = q_rev.len() as i64 * t_rev.len() as i64;
        if max_sw_mat > 0 && sw_mat_size > max_sw_mat {
            ez.max_score_query_pos = -1;
            ez.max_score_target_pos = -1;
            ez.max_query_end_target_pos = -1;
            ez.zdropped = 1;
        } else if is_splice {
            dp::extend_splice(&q_rev, &t_rev, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, left_zdrop, end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | flag, junc_left.as_deref(), &mut ez);
        } else if q == q2 && e == e2 {
            dp::extend_single_affine(&q_rev, &t_rev, 5, &mat, q as i8, e as i8, bw, left_zdrop, end_bonus, flag, &mut ez);
        } else {
            dp::extend_dual_affine(&q_rev, &t_rev, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw, left_zdrop, end_bonus, flag, &mut ez);
        }

        if !ez.cigar.is_empty() {
            append_cigar(&mut raw_cigar, &ez.cigar);
            dp_score += ez.max;
        }
        // Compute consumed bases
        rs1 = rs - if ez.reach_end != 0 { ez.max_query_end_target_pos + 1 } else { ez.max_score_target_pos + 1 };
        qs1 = qs - if ez.reach_end != 0 { q_ext_len as i32 } else { ez.max_score_query_pos + 1 };
    }
    let (mut re1, mut qe1) = (rs, qs);

    // Gap filling: iterate seeds, accumulate, align when appropriate
    let mut gap_rs = rs; // current gap-fill start (reference)
    let mut gap_qs = qs; // current gap-fill start (query)
    let mut dropped = false;
    let mut split_right_anchors: Option<Vec<Minimizer>> = None;
    let mut split_offset_in_orig: Option<usize> = None;
    let mut split_inv_flag = false;

    // SR gap-fill: single iteration from first to last anchor
    let gap_fill_start = if is_sr { cnt1 - 1 } else { 1 };
    for i in gap_fill_start..cnt1 {
        if (work_anchors[i].y & (SEED_IGNORE | SEED_TANDEM)) != 0 && i != cnt1 - 1 {
            continue;
        }

        // Anchor coordinate computation
        let (anchor_re, anchor_qe) = if is_sr && !is_hpc {
            // SR non-HPC: pos + 1
            (work_anchors[i].ref_pos() + 1,
             work_anchors[i].query_pos() + 1)
        } else if is_hpc {
            (hpc_adjust_ref(tseq, work_anchors[i].ref_pos()),
             hpc_adjust_query(qseq, work_anchors[i].query_pos()))
        } else {
            (work_anchors[i].ref_pos() - k_half,
             work_anchors[i].query_pos() - k_half)
        };
        re1 = anchor_re;
        qe1 = anchor_qe;

        let is_last = i == cnt1 - 1;
        let is_long_join = (work_anchors[i].y & SEED_LONG_JOIN) != 0;
        let should_align = is_last || is_long_join
            || (anchor_qe - gap_qs >= min_dp_len && anchor_re - gap_rs >= min_dp_len);
        // gap-fill always uses end_bonus = -1
        let gap_end_bonus: i32 = -1;

        if should_align {
            let mut bw1 = bw_long;
            if is_long_join {
                bw1 = cmp::max(anchor_qe - gap_qs, anchor_re - gap_rs);
            }

            let q_start = cmp::max(0, gap_qs) as usize;
            let q_end = cmp::min(qlen, anchor_qe) as usize;
            let t_start = cmp::max(0, gap_rs) as usize;
            let t_end = cmp::min(tlen, anchor_re) as usize;

            // Lazy fill the gap segment [t_start, t_end) in tseq.
            if use_lazy && t_end > t_start && let Some((mi, rid, rgn_start)) = lazy_extract {
                mi.extract_nt4_into(rid, rgn_start + t_start, rgn_start + t_end, &mut tseq[t_start..t_end]);
            }

            if q_end > q_start && t_end > t_start {
                let q_sub = &qseq[q_start..q_end];
                let t_sub = &tseq[t_start..t_end];

                // Junction buffer for gap-fill
                let junc_gap = if let Some(db) = junc_db {
                    let mut jbuf = vec![0u8; t_sub.len()];
                    junc::get_junc(db, rid, gap_rs + ref_offset, anchor_re + ref_offset, is_rev_splice, &mut jbuf);
                    Some(jbuf)
                } else {
                    None
                };

                let mut ez = DpResult::default();
                let mut zdrop_code = 0i32;
                // max_sw_mat check: skip alignment if matrix too large
                // Reset: max_q=max_t=mqe_t=mte_q=-1, max=0, zdropped=1
                let sw_mat_size = q_sub.len() as i64 * t_sub.len() as i64;
                if max_sw_mat > 0 && sw_mat_size > max_sw_mat {
                    ez.max_score_query_pos = -1;
                    ez.max_score_target_pos = -1;
                    ez.max_query_end_target_pos = -1;
                    ez.zdropped = 1;
                } else if (is_sr || is_sr_rna) && q_sub.len() == t_sub.len() {
                    // SR/SR-RNA ungapped alignment
                    // Only when query and reference gap lengths match (no indels)
                    // Try ungapped score first; only use gapped alignment if ungapped is poor
                    let gap_len = q_sub.len() as i32;
                    let max_gapped_score = (gap_len - 2) * a - 2 * (q + e);
                    let mut ungapped_score = 0i32;
                    for j in 0..q_sub.len() {
                        if q_sub[j] >= 4 || t_sub[j] >= 4 {
                            ungapped_score += if sc_ambi > 0 { -sc_ambi } else { sc_ambi };
                        } else if q_sub[j] == t_sub[j] {
                            ungapped_score += a;
                        } else {
                            ungapped_score -= b;
                        }
                    }
                    if ungapped_score > max_gapped_score {
                        // Emit single M op (will be converted to =/X later)
                        ez.score = ungapped_score;
                        ez.cigar = vec![((gap_len as u32) << 4)]; // M op
                    } else {
                        // Fall through to gapped alignment
                        if is_splice {
                            dp::extend_splice(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop, gap_end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | APPROX_MAX | generic_sc_flag, junc_gap.as_deref(), &mut ez);
                        } else if q == q2 && e == e2 {
                            dp::extend_single_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, bw1, zdrop, gap_end_bonus, APPROX_MAX | generic_sc_flag, &mut ez);
                        } else {
                            dp::extend_dual_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw1, zdrop, gap_end_bonus, APPROX_MAX | generic_sc_flag, &mut ez);
                        }
                    }
                    // Z-drop test applies to both paths
                    zdrop_code = if !ez.cigar.is_empty() {
                        test_zdrop(q_sub, t_sub, &ez.cigar, &mat, opt)
                    } else { 0 };
                    if zdrop_code != 0 {
                        ez = DpResult::default();
                        let zdrop2 = if zdrop_code == 2 { zdrop_inv } else { zdrop };
                        if is_splice {
                            dp::extend_splice(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop2, gap_end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | generic_sc_flag, junc_gap.as_deref(), &mut ez);
                        } else if q == q2 && e == e2 {
                            dp::extend_single_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, bw1, zdrop2, gap_end_bonus, generic_sc_flag, &mut ez);
                        } else {
                            dp::extend_dual_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw1, zdrop2, gap_end_bonus, generic_sc_flag, &mut ez);
                        }
                    }
                } else {
                    // For is_sr_rna, try heuristic alignment first
                    let mut skip_full = false;
                    if is_sr_rna {
                        skip_full = align_short_read_rna(q_sub, t_sub, &mat, opt, gap_end_bonus, splice_dp_flag | APPROX_MAX | generic_sc_flag, junc_gap.as_deref(), junc_bonus as i8, junc_pen as i8, &mut ez);
                    }
                    // First pass: APPROX_MAX
                    if !skip_full {
                        if is_splice {
                            dp::extend_splice(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop, gap_end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | APPROX_MAX | generic_sc_flag, junc_gap.as_deref(), &mut ez);
                        } else if q == q2 && e == e2 {
                            dp::extend_single_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, bw1, zdrop, gap_end_bonus, APPROX_MAX | generic_sc_flag, &mut ez);
                        } else {
                            dp::extend_dual_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw1, zdrop, gap_end_bonus, APPROX_MAX | generic_sc_flag, &mut ez);
                        }
                    }

                    // Z-drop test
                    zdrop_code = if !ez.cigar.is_empty() {
                        test_zdrop(q_sub, t_sub, &ez.cigar, &mat, opt)
                    } else { 0 };
                    if zdrop_code != 0 {
                        ez = DpResult::default();
                        let zdrop2 = if zdrop_code == 2 { zdrop_inv } else { zdrop };
                        if is_splice {
                            dp::extend_splice(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop2, gap_end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | generic_sc_flag, junc_gap.as_deref(), &mut ez);
                        } else if q == q2 && e == e2 {
                            dp::extend_single_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, bw1, zdrop2, gap_end_bonus, generic_sc_flag, &mut ez);
                        } else {
                            dp::extend_dual_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw1, zdrop2, gap_end_bonus, generic_sc_flag, &mut ez);
                        }
                    }
                }

                if !ez.cigar.is_empty() {
                    append_cigar(&mut raw_cigar, &ez.cigar);
                }

                if ez.zdropped != 0 {
                    dp_score += ez.max;
                    re1 = gap_rs + ez.max_score_target_pos + 1;
                    qe1 = gap_qs + ez.max_score_query_pos + 1;
                    // Find last anchor before z-drop truncation point
                    let max_re_split = gap_rs + ez.max_score_target_pos;
                    let mut j_split = i as i32 - 1;
                    while j_split >= 0 {
                        if work_anchors[j_split as usize].ref_pos() <= max_re_split {
                            break;
                        }
                        j_split -= 1;
                    }
                    if j_split < 0 { j_split = 0; }
                    // Check if remaining anchors >= min_cnt
                    let remaining = cnt1 as i32 - (j_split + 1);
                    if remaining >= min_cnt {
                        // Split: save right part anchors from original chain
                        // In our terms: split_offset = as1_offset + (j_split + 1) as usize
                        let split_off = as1_offset + (j_split + 1) as usize;
                        if split_off < anchors.len() {
                            split_right_anchors = Some(anchors[split_off..].to_vec());
                            split_offset_in_orig = Some(split_off);
                            // if zdrop_code == 2, mark split as inversion
                            if zdrop_code == 2 {
                                split_inv_flag = true;
                            }
                        }
                    }
                    dropped = true;
                    break;
                } else {
                    dp_score += ez.score;
                }
            }

            // Update gap-fill start for next segment
            gap_rs = anchor_re;
            gap_qs = anchor_qe;
        }
    }

    // --- Step 7: Right extension ---
    let mut final_re = re1;
    let mut final_qe = qe1;

    if !dropped && qe < qlen && re < tlen {
        // Compute right extension boundary
        let (mut qe0, mut re0);
        if is_sr {
            // SR mode: force full query coverage
            qe0 = qlen;
            let mut l = qlen - qe;
            l += if l * a + end_bonus > q { (l * a + end_bonus - q) / e } else { 0 };
            re0 = cmp::min(tlen, re + l);
        } else {
            // Non-SR mode
            let orig_last = &anchors[anchors.len() - 1];
            let (re0_tmp, qe0_tmp) = compute_right_boundary(
                re, qe,
                orig_last.ref_pos() + 1, orig_last.query_pos() + 1,
                seed_bounds.2, seed_bounds.3,
                opt,
                qlen, tlen,
                seed_bounds,
            );
            re0 = re0_tmp;
            qe0 = qe0_tmp;
        }

        // SEED_SELF: clamp rightward extension to |q-r| of LAST pre-trim
        // anchor. Same coord-system issue as the left clamp: `pre_qe` is
        // absolute, `pre_re` is region-relative after the anchor rebase;
        // compute max_ext in absolute coords.
        if !anchors.is_empty() && (anchors[0].y & crate::align::map::SEED_SELF) != 0 {
            let last_idx = anchors.len() - 1;
            let pre_qe = anchors[last_idx].query_pos() + 1;
            let pre_re = anchors[last_idx].ref_pos() + 1;
            let pre_re_abs = pre_re + ref_offset;
            let max_ext = (pre_qe - pre_re_abs).unsigned_abs() as i32;
            if re0 - pre_re > max_ext { re0 = pre_re + max_ext; }
            if qe0 - pre_qe > max_ext { qe0 = pre_qe + max_ext; }
        }

        if qe0 > qe && re0 > re {
            // Lazy fill the right-extension segment [re, re0) in tseq.
            if use_lazy && let Some((mi, rid, rgn_start)) = lazy_extract {
                let a = re as usize;
                let b = re0 as usize;
                mi.extract_nt4_into(rid, rgn_start + a, rgn_start + b, &mut tseq[a..b]);
            }
            let q_sub = &qseq[qe as usize..qe0 as usize];
            let t_sub = &tseq[re as usize..re0 as usize];

            // Junction buffer for right extension: no reversal needed
            let junc_right = if let Some(db) = junc_db {
                let mut jbuf = vec![0u8; (re0 - re) as usize];
                junc::get_junc(db, rid, re + ref_offset, re0 + ref_offset, is_rev_splice, &mut jbuf);
                Some(jbuf)
            } else {
                None
            };

            // Right extension flags: EXTZ_ONLY
            let flag = EXTENSION_ONLY | generic_sc_flag;
            let mut ez = DpResult::default();
            // max_sw_mat check: skip alignment if matrix too large
            let sw_mat_size = q_sub.len() as i64 * t_sub.len() as i64;
            if max_sw_mat > 0 && sw_mat_size > max_sw_mat {
                ez.max_score_query_pos = -1;
                ez.max_score_target_pos = -1;
                ez.max_query_end_target_pos = -1;
                ez.zdropped = 1;
            } else if is_splice {
                dp::extend_splice(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, noncan as i8, zdrop, end_bonus, junc_bonus as i8, junc_pen as i8, splice_dp_flag | flag, junc_right.as_deref(), &mut ez);
            } else if q == q2 && e == e2 {
                dp::extend_single_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, bw, zdrop, end_bonus, flag, &mut ez);
            } else {
                dp::extend_dual_affine(q_sub, t_sub, 5, &mat, q as i8, e as i8, q2 as i8, e2 as i8, bw, zdrop, end_bonus, flag, &mut ez);
            }

            if !ez.cigar.is_empty() {
                append_cigar(&mut raw_cigar, &ez.cigar);
                dp_score += ez.max;
            }

            // Consumed bases
            final_re = re + if ez.reach_end != 0 { ez.max_query_end_target_pos + 1 } else { ez.max_score_target_pos + 1 };
            final_qe = qe + if ez.reach_end != 0 { qe0 - qe } else { ez.max_score_query_pos + 1 };
        }
    }

    if final_re <= 0 { final_re = re1; }
    if final_qe <= 0 { final_qe = qe1; }

    // Lazy mode: fill the final alignment range [rs1, final_re) so that
    // `finalize_cigar` (indel right-alignment, op merging, =/X conversion)
    // and any downstream MD/cs tag generators can read tseq there.
    if use_lazy && let Some((mi, rid, rgn_start)) = lazy_extract {
        let a = cmp::max(0, rs1) as usize;
        let b = cmp::max(a as i32, final_re) as usize;
        if b > a {
            mi.extract_nt4_into(rid, rgn_start + a, rgn_start + b, &mut tseq[a..b]);
        }
    }

    // --- Steps 8-9: fix_cigar + convert M→=/X + condense ---
    let (condensed, final_qs, final_rs) = finalize_cigar(raw_cigar, qseq, tseq, qs1, rs1);


    AlignResult {
        cigar_ops: condensed,
        query_start: cmp::max(0, final_qs) as usize,
        query_end: cmp::max(0, final_qe) as usize,
        ref_start: cmp::max(0, final_rs) as usize,
        ref_end: cmp::max(0, final_re) as usize,
        dp_score,
        split_right_anchors,
        split_offset_in_orig,
        split_inv: split_inv_flag,
    }
}

/// Public wrapper for convert_cigar_to_eqx (used by pipeline.rs for inversion alignment)
pub fn convert_cigar_to_eqx_pub(raw_cigar: &[u32], qseq: &[u8], tseq: &[u8], qs: usize, rs: usize) -> Vec<CigarOp> {
    convert_cigar_to_eqx(raw_cigar, qseq, tseq, qs, rs)
}

/// Public wrapper for fix_cigar (used by pipeline.rs for inversion alignment)
pub fn fix_cigar_pub(cigar: &mut Vec<u32>, qseq: &[u8], tseq: &[u8], qs: i32, rs: i32) -> (i32, i32) {
    fix_cigar(cigar, qseq, tseq, qs, rs)
}

/// Convert M ops to =/X ops (EQX mode).
fn convert_cigar_to_eqx(raw_cigar: &[u32], qseq: &[u8], tseq: &[u8], qs: usize, rs: usize) -> Vec<CigarOp> {
    let mut ops = Vec::new();
    let mut qi = qs;
    let mut ti = rs;

    for &c in raw_cigar {
        let len = (c >> 4) as usize;
        let op = c & 0xf;

        match op {
            0 => { // M -> split into = and X
                let mut pos = 0;
                while pos < len {
                    // Run of matches
                    let start = pos;
                    while pos < len && qi + pos < qseq.len() && ti + pos < tseq.len()
                        && qseq[qi + pos] == tseq[ti + pos] {
                        pos += 1;
                    }
                    if pos > start {
                        ops.push(CigarOp { op: '=', len: (pos - start) as u32 });
                    }
                    // Run of mismatches
                    let start = pos;
                    while pos < len && qi + pos < qseq.len() && ti + pos < tseq.len()
                        && qseq[qi + pos] != tseq[ti + pos] {
                        pos += 1;
                    }
                    if pos > start {
                        ops.push(CigarOp { op: 'X', len: (pos - start) as u32 });
                    }
                }
                qi += len;
                ti += len;
            },
            1 => { // I
                ops.push(CigarOp { op: 'I', len: len as u32 });
                qi += len;
            },
            2 => { // D
                ops.push(CigarOp { op: 'D', len: len as u32 });
                ti += len;
            },
            3 => { // N
                ops.push(CigarOp { op: 'N', len: len as u32 });
                ti += len;
            },
            7 => { // = (already EQX)
                ops.push(CigarOp { op: '=', len: len as u32 });
                qi += len;
                ti += len;
            },
            8 => { // X (already EQX)
                ops.push(CigarOp { op: 'X', len: len as u32 });
                qi += len;
                ti += len;
            },
            _ => {
                ops.push(CigarOp { op: 'M', len: len as u32 });
                qi += len;
                ti += len;
            }
        }
    }
    ops
}

pub fn fmt_cs(
    ops: &[CigarOp],
    qseq: &[u8],
    tseq: &[u8],
    qs: usize,
    rs: usize,
    long: bool,
) -> String {
    let mut visitor = CsVisitor::new(qseq, tseq, qs, rs, long);
    walk_cigar_ops(ops, &mut visitor)
}

/// Generate ds:Z difference string from CIGAR and sequences.
pub fn fmt_ds(
    ops: &[CigarOp],
    qseq: &[u8],
    tseq: &[u8],
    qs: usize,
    rs: usize,
) -> String {
    // First pass: compute q_len and t_len (total aligned lengths for bounds checking)
    let mut q_len: usize = 0;
    let mut t_len: usize = 0;
    for op in ops {
        match op.op {
            '=' | 'X' | 'M' => { q_len += op.len as usize; t_len += op.len as usize; }
            'I' => { q_len += op.len as usize; }
            'D' | 'N' => { t_len += op.len as usize; }
            _ => {}
        }
    }
    let mut visitor = DsVisitor::new(qseq, tseq, qs, rs, q_len, t_len);
    walk_cigar_ops(ops, &mut visitor)
}

/// Format indel portion of ds:Z string.
fn write_indel_ds(out: &mut String, len: usize, seq: &[u8], ll: usize, lr: usize) {
    if ll + lr >= len {
        // Entire indel is ambiguous
        out.push('[');
        for &b in &seq[..len] {
            out.push(NT4_TO_LOWER[b.min(4) as usize] as char);
        }
        out.push(']');
    } else {
        let mut k: usize = 0;
        if ll > 0 {
            out.push('[');
            for i in 0..ll {
                out.push(NT4_TO_LOWER[seq[k + i].min(4) as usize] as char);
            }
            out.push(']');
            k += ll;
        }
        for i in 0..(len - lr - ll) {
            out.push(NT4_TO_LOWER[seq[k + i].min(4) as usize] as char);
        }
        k += len - lr - ll;
        if lr > 0 {
            out.push('[');
            for i in 0..lr {
                out.push(NT4_TO_LOWER[seq[k + i].min(4) as usize] as char);
            }
            out.push(']');
        }
    }
}
