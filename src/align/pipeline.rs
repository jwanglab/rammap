//! Top-level alignment pipeline orchestration.
//!
//! `process_query` is the main entry point for each read: it drives the full
//! map -> align -> filter -> MAPQ -> format pipeline. `align_single_mapping`
//! handles one chain: extracts the target region from the packed index,
//! adjusts coordinates, and invokes the DP kernel. Key types are
//! `ProcessedQuery` (per-read output bundle) and `AlnResult` (one alignment's
//! score, CIGAR, coordinates, and optional tags). Final output is emitted as
//! PAF or SAM via `format_output`.

use crate::align::map::{MapOptions, map_query, map_query_multi, MapContext, Mapping, AlignFlags, compute_bounds_from_squeezed};
use crate::align::pair::{PeReg, pair_alignments};
use crate::align::sketch::Minimizer;
use crate::align::index::Index;
use crate::align::extend::{AlignmentContext, AlignAnchorContext, align_anchors, build_scoring_matrix_full, convert_cigar_to_eqx_pub, fix_cigar_pub, fmt_cigar, fmt_cs, fmt_ds, fmt_md, CigarOp};
use crate::align::filter::{FilterParams, ParentState, FilterableItem, check_secondary_filter, scale_alt_score};
use crate::align::sort::radix_sort_128x_pair;
use crate::align::extend::{rev_comp, rev_comp_nt4, encode_nt4_rc};
use std::fmt::Write;
use crate::align::stats::AlignmentStats;
use crate::align::junc::JunctionDb;
use crate::align::jump::JumpDb;
use serde::{Serialize, Deserialize};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Encode ASCII base to 0-4 (A=0,C=1,G=2,T=3,N=4)
#[inline]
fn encode_nt4(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}

#[inline]
fn fast_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let log_2 = ((bits >> 23) & 255) as f32 - 128.0;
    let mut z_bits = bits;
    z_bits &= !(255 << 23);
    z_bits += 127 << 23;
    let z = f32::from_bits(z_bits);
    log_2 + ((-0.34484843 * z + 2.024_665_8) * z - 0.674_877_6)
}

// Compute dp_max (maximum segment alignment score) with log-gap scoring
/// Compute dp_max by walking the CIGAR base-by-base.
/// Uses the full scoring matrix for per-base scoring (important for transition scoring).
/// qseq/tseq are the aligned regions (0-indexed, encoded as 0=A,1=C,2=G,3=T,4=N).
fn compute_alignment_score_max(ops: &[CigarOp], mat: &[i8; 25], q: i32, e: i32, qseq: &[u8], tseq: &[u8], log_gap: bool) -> i32 {
    let mut s: f64 = 0.0;
    let mut max: f64 = 0.0;
    let mut qoff: usize = 0;
    let mut toff: usize = 0;
    for op in ops {
        let len = op.len as usize;
        match op.op {
            '=' | 'X' | 'M' => {
                // Walk base-by-base, look up score from matrix
                for l in 0..len {
                    let cq = if qoff + l < qseq.len() { qseq[qoff + l] as usize } else { 4 };
                    let ct = if toff + l < tseq.len() { tseq[toff + l] as usize } else { 4 };
                    s += mat[ct * 5 + cq] as f64;
                    if s < 0.0 { s = 0.0; }
                    else if s > max { max = s; }
                }
                qoff += len;
                toff += len;
            }
            'I' => {
                // log_gap uses logarithmic penalty, else flat q+e
                if log_gap {
                    s -= (q as f64) + (e as f64) * fast_log2(1.0 + len as f32) as f64;
                } else {
                    s -= (q + e) as f64;
                }
                if s < 0.0 { s = 0.0; }
                qoff += len;
            }
            'D' => {
                // same log_gap vs flat distinction
                if log_gap {
                    s -= (q as f64) + (e as f64) * fast_log2(1.0 + len as f32) as f64;
                } else {
                    s -= (q + e) as f64;
                }
                if s < 0.0 { s = 0.0; }
                toff += len;
            }
            'N' => {
                toff += len;
            }
            _ => {}
        }
    }
    (max + 0.499) as i32
}

/// CIGAR-derived alignment statistics: matches, edit distance, divergence, etc.
/// Shared by align_single_mapping and try_align_inversion to avoid code duplication.
pub struct CigarStats {
    pub matches: usize,
    pub edit_distance: u32,
    pub block_len: usize,
    pub num_ambiguous: usize,
    pub divergence: f64,
    pub has_n_skip: bool,
    pub gap_opens: u32,
}

/// Output format configuration: controls which tags/features to generate.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub do_cigar: bool,
    pub do_cs: bool,
    pub cs_long: bool,
    pub do_md: bool,
    pub do_ds: bool,
    pub eqx: bool,
    pub output_sam: bool,
    pub rg_id: Option<String>,
    pub split_mode: bool,
}

/// Per-read metadata for output formatting.
pub struct ReadInfo<'a> {
    pub qname: &'a str,
    pub qseq: &'a [u8],
    pub qual: Option<&'a str>,
    pub comment: Option<&'a str>,
    pub n_seg: usize,
    pub seg_idx: usize,
}

/// Intermediate mapping results passed to process_query_core.
pub(crate) struct MapResult {
    pub regs: Vec<Mapping>,
    pub rep_len: usize,
    pub stats: AlignmentStats,
    pub squeezed: Vec<Minimizer>,
}

impl CigarStats {
    /// Compute alignment statistics by walking CIGAR ops and aligned sequences.
    ///
    /// `aln_qseq` and `aln_tseq` are nt4-encoded aligned regions (0=A,1=C,2=G,3=T,4+=N).
    pub fn from_cigar(ops: &[CigarOp], aln_qseq: &[u8], aln_tseq: &[u8]) -> Self {
        // Phase 1: Count CIGAR operations
        let mut m_cnt: u32 = 0;
        let mut x_cnt: u32 = 0;
        let mut i_cnt: u32 = 0;
        let mut d_cnt: u32 = 0;
        let mut n_gapo: u32 = 0;
        let mut has_n_skip = false;
        for op in ops {
            match op.op {
                '=' => m_cnt += op.len,
                'X' => x_cnt += op.len,
                'I' => { i_cnt += op.len; n_gapo += 1; }
                'D' => { d_cnt += op.len; n_gapo += 1; }
                'M' => m_cnt += op.len,
                'N' => { has_n_skip = true; }
                _ => {}
            }
        }

        // Phase 2: Count ambiguous bases (N) by walking CIGAR + sequences
        let mut n_ambi: u32 = 0;
        let mut n_ambi_match: u32 = 0;
        {
            let mut qi: usize = 0;
            let mut ti: usize = 0;
            for op in ops {
                let len = op.len as usize;
                match op.op {
                    '=' | 'X' | 'M' => {
                        for l in 0..len {
                            if (qi + l < aln_qseq.len() && aln_qseq[qi + l] > 3)
                                || (ti + l < aln_tseq.len() && aln_tseq[ti + l] > 3)
                            {
                                n_ambi += 1;
                                if op.op == '=' || op.op == 'M' {
                                    n_ambi_match += 1;
                                }
                            }
                        }
                        qi += len;
                        ti += len;
                    }
                    'I' => {
                        for l in 0..len {
                            if qi + l < aln_qseq.len() && aln_qseq[qi + l] > 3 {
                                n_ambi += 1;
                            }
                        }
                        qi += len;
                    }
                    'D' => {
                        for l in 0..len {
                            if ti + l < aln_tseq.len() && aln_tseq[ti + l] > 3 {
                                n_ambi += 1;
                            }
                        }
                        ti += len;
                    }
                    'N' => { ti += len; }
                    _ => {}
                }
            }
        }

        // Phase 3: Compute derived statistics
        let matches = (m_cnt - n_ambi_match) as usize;
        let edit_distance = x_cnt + i_cnt + d_cnt + n_ambi_match;
        let block_len = (m_cnt + x_cnt + i_cnt + d_cnt - n_ambi) as usize;
        let nn = n_ambi as usize;

        let n_gap = i_cnt + d_cnt;
        let event_denom = block_len as f64 + nn as f64 - n_gap as f64 + n_gapo as f64;
        let divergence = if event_denom > 0.0 {
            1.0 - matches as f64 / event_denom
        } else {
            0.0
        };

        CigarStats { matches, edit_distance, block_len, num_ambiguous: nn, divergence, has_n_skip, gap_opens: n_gapo }
    }
}

// Stats needed for dp_max recalculation
pub struct DpRecalcInfo {
    pub match_len: i32,
    pub block_len: i32,
    pub num_ambiguous: i32,
    pub gap_bases: i32,
    pub gap_opens: i32,
    pub sum_log_gap: f64,
}

impl DpRecalcInfo {
    fn from_ops(ops: &[CigarOp]) -> Self {
        let mut mlen = 0i32;
        let mut blen = 0i32;
        let mut n_gap = 0i32;
        let mut n_gapo = 0i32;
        let mut sum_log_gap = 0.0f64;
        for op in ops {
            match op.op {
                '=' | 'M' => { mlen += op.len as i32; blen += op.len as i32; }
                'X' => { blen += op.len as i32; }
                'I' | 'D' => {
                    blen += op.len as i32;
                    n_gapo += 1;
                    n_gap += op.len as i32;
                    sum_log_gap += fast_log2(1.0 + op.len as f32) as f64;
                }
                _ => {}
            }
        }
        DpRecalcInfo { match_len: mlen, block_len: blen, num_ambiguous: 0, gap_bases: n_gap, gap_opens: n_gapo, sum_log_gap }
    }

    /// Reconstruct DpRecalcInfo from a CIGAR string (e.g. "10M2I5M3D8M").
    pub fn from_cigar_str(cigar: &str) -> Self {
        let mut mlen = 0i32;
        let mut blen = 0i32;
        let mut n_gap = 0i32;
        let mut n_gapo = 0i32;
        let mut sum_log_gap = 0.0f64;
        let mut num = 0u32;
        for c in cigar.chars() {
            if c.is_ascii_digit() {
                num = num * 10 + (c as u32 - '0' as u32);
            } else {
                let len = num;
                num = 0;
                match c {
                    '=' | 'M' => { mlen += len as i32; blen += len as i32; }
                    'X' => { blen += len as i32; }
                    'I' | 'D' => {
                        blen += len as i32;
                        n_gapo += 1;
                        n_gap += len as i32;
                        sum_log_gap += fast_log2(1.0 + len as f32) as f64;
                    }
                    _ => {} // N, S, H, etc.
                }
            }
        }
        DpRecalcInfo { match_len: mlen, block_len: blen, num_ambiguous: 0, gap_bases: n_gap, gap_opens: n_gapo, sum_log_gap }
    }

    fn event_identity(&self) -> f64 {
        let denom = self.block_len + self.num_ambiguous - self.gap_bases + self.gap_opens;
        if denom <= 0 { return 0.0; }
        self.match_len as f64 / denom as f64
    }

    fn recalc_dp_max(&self, b2: f64, match_sc: i32) -> i32 {
        let n_mis = self.block_len + self.num_ambiguous - self.match_len - self.gap_bases;
        let gap_cost = self.gap_opens as f64 * b2 + self.sum_log_gap;
        let result = match_sc as f64 * (self.match_len as f64 - b2 * n_mis as f64 - gap_cost) + 0.499;
        if result < 0.0 { 0 } else { result as i32 }
    }
}

// Recompute dp_max values using divergence-scaled scoring
pub fn update_dp_max(
    dp_max_vals: &mut [i32],
    recalc_infos: &[DpRecalcInfo],
    qs_vals: &[usize],
    qe_vals: &[usize],
    qlen: usize,
    rank_frac: f32,
    a: i32,
    b: i32,
) {
    if dp_max_vals.len() < 2 { return; }

    let mut max = -1i32;
    let mut max2 = -1i32;
    let mut max_i: Option<usize> = None;

    for (i, &val) in dp_max_vals.iter().enumerate() {
        if val > max {
            max2 = max;
            max = val;
            max_i = Some(i);
        } else if val > max2 {
            max2 = val;
        }
    }

    let max_i = match max_i {
        Some(i) => i,
        None => return,
    };
    if max < 0 || max2 < 0 { return; }

    let qcov = qe_vals[max_i].saturating_sub(qs_vals[max_i]);
    if (qcov as f64) < (qlen as f64 * rank_frac as f64) { return; }
    if (max2 as f64) < (max as f64 * rank_frac as f64) { return; }

    let mut div = 1.0 - recalc_infos[max_i].event_identity();
    if div < 0.02 { div = 0.02; }
    let mut b2 = 0.5 / div;
    if b2 * (a as f64) < b as f64 { b2 = a as f64 / b as f64; }


    for i in 0..dp_max_vals.len() {
        dp_max_vals[i] = recalc_infos[i].recalc_dp_max(b2, a);
    }
}

// Alignment result struct - replaces the unwieldy 18-field tuple
#[derive(Clone, Serialize, Deserialize)]
pub struct AlnResult {
    // Mapping metadata (owned, so splits don't need references)
    pub ref_id: usize,
    pub is_reverse: bool,
    pub chain_score: i32,    // s1 tag (may be reduced after z-drop split)
    pub initial_chain_score: i32,         // initial chain score before splitting
    pub anchor_count: usize, // cm tag (anchor count)
    pub s2_score: Option<i32>,
    pub hash: u32,
    // Alignment results
    pub align_score: i32,
    pub matches: usize,
    pub block_len: usize,
    pub cigar_str: String,
    pub cs_str: String,
    pub ds_str: String,
    pub md_str: String,
    pub query_start: usize,
    pub query_end: usize,
    pub ref_start: usize,
    pub ref_end: usize,
    pub edit_distance: u32,
    pub num_ambiguous: usize,
    pub divergence: f64,
    pub is_secondary: bool,
    pub split: u8,           // z-drop split flag: 1=left part, 2=right part
    pub split_depth: u8,     // 0=original chain, 1=direct split-child, 2+=nested split-of-split
    pub dp_score: i32,       // for sorting (may be recalculated)
    pub dp_score_original: i32, // for ms:i: tag (original)
    pub effective_cnt: i32,  // for post-alignment filtering
    // Pre-alignment n_sub (from first parent assignment call, persists for r[0])
    pub pre_num_suboptimal: i32,
    pub is_spliced: bool,    // true if alignment contains N_SKIP operations (introns)
    pub trans_strand: u8,    // 0=unknown, 1=+strand, 2=-strand, 3=ambiguous
    // Post-alignment parent assignment results (for MAPQ, set during secondary selection)
    pub dp_score_secondary: i32, // best dp_max among children
    pub secondary_chain_score: i32, // best chain_score among children
    pub num_suboptimal: i32, // count of children with similar dp_max
    pub split_inv: bool,     // true if this split part should trigger inversion alignment
    pub inv: bool,           // true if this is an inversion alignment result
    pub proper_frag: bool,   // true if this alignment is part of a concordant PE pair
    pub seg_split: bool,     // true if from seg_gen splitting (post-alignment filter skips cnt check)
    pub div: f32,            // estimated divergence (for dv:f: tag in non-CIGAR PAF)
    pub is_alt: bool,        // true if mapped to an ALT contig
    pub is_root_chain: bool, // true if from a root chain (parent==id), for align1_inv parent check
}

/// Compute fuzzy mlen/blen from anchor array.
fn estimate_match_block_lengths(anchors: &[Minimizer]) -> (usize, usize) {
    if anchors.is_empty() { return (0, 0); }
    let first_span = anchors[0].query_span();
    let mut mlen = first_span;
    let mut blen = first_span;
    for i in 1..anchors.len() {
        let span = anchors[i].query_span();
        let tl = anchors[i].ref_pos().wrapping_sub(anchors[i-1].ref_pos());
        let ql = anchors[i].query_pos().wrapping_sub(anchors[i-1].query_pos());
        blen += if tl > ql { tl } else { ql };
        mlen += if tl > span && ql > span { span } else if tl < ql { tl } else { ql };
    }
    (mlen.max(0) as usize, blen.max(0) as usize)
}

// Helper: align a mapping and produce an AlnResult + DpRecalcInfo + optional split
fn align_single_mapping(
    mapping: &mut Mapping,
    opt: &MapOptions,
    mi: &Index,
    _qseq: &[u8],
    qseq_fwd_nt4: &[u8],
    qseq_rc_nt4: &[u8],
    qlen: usize,
    ctx: &mut AlignmentContext,
    _map_ctx: &mut MapContext,
    out: &OutputConfig,
    squeezed: &[Minimizer],
    splice_flag: AlignFlags,
    junc_db: Option<&JunctionDb>,
) -> (AlnResult, DpRecalcInfo, Option<Mapping>) {
    let mut align_score = mapping.score;
    let mut dp_max = mapping.score;
    let mut recalc_info = DpRecalcInfo { match_len: 0, block_len: 0, num_ambiguous: 0, gap_bases: 0, gap_opens: 0, sum_log_gap: 0.0 };
    let (fuzzy_mlen, fuzzy_blen) = estimate_match_block_lengths(&mapping.anchors);
    let mut matches = fuzzy_mlen;
    let mut block_len = fuzzy_blen;
    let mut cigar_str = String::new();
    let mut cs_str = String::new();
    let mut ds_str = String::new();
    let mut md_str = String::new();
    let mut nm: u32 = 0;
    let mut nn: usize = 0;
    let mut de: f64 = 0.0;
    let mut has_n_skip = false;
    let mut qs = mapping.query_start;
    let mut qe = mapping.query_end;
    let mut rs = mapping.ref_start;
    let mut re = mapping.ref_end;
    let mut effective_cnt = mapping.anchor_count;
    let mut chain_score_adj = mapping.score; // may be reduced after z-drop split
    let mut new_split: Option<Mapping> = None;

    if out.do_cigar {
        let is_qstrand = opt.flags.contains(AlignFlags::QSTRAND);
        let rid = mapping.ref_id;

        let tlen = mi.seqs[rid].len;

        // Compute the [rs0, re0] target-sequence extraction window so the
        // downstream extension-alignment sees the same bytes as a reference
        // bandwidth-bounded extension would. Three coordinate adjustments
        // apply:
        //
        // 1. Bad-diagonal anchors at chain ends are trimmed first; the window
        //    derives from POST-trim chain extents. NO_END_FLT and splice mode
        //    skip the trim and use pre-trim coords.
        //
        // 2. Chain qs/qe are in alignment-space (strand-flipped on the "-"
        //    strand). `mapping.query_start/end` is alignment-space on "+" and
        //    forward-space on "-"; convert before use.
        //
        // 3. Shift chain start/end coords from "start/end+1 of kmer" to
        //    "center of kmer" (non-HPC: subtract `k/2`). HPC uses a
        //    sequence-dependent shift handled inside `align_anchors`.
        let max_gap = opt.chaining.max_gap;
        let a = opt.scoring.match_score;
        let q = opt.scoring.gap_open;
        let e = opt.scoring.gap_extend;
        let end_bonus = opt.alignment.end_bonus;
        let is_sr_mode = opt.flags.contains(AlignFlags::SHORT_READ);
        let is_splice = opt.filtering.is_splice;
        let no_end_flt = opt.flags.contains(AlignFlags::NO_END_FLT);
        let is_hpc = mi.homopolymer_compressed;
        let qlen_i32 = qlen as i32;
        let tlen_i32 = tlen as i32;
        let k_half = (mi.kmer_size as i32) >> 1;

        // The region formula uses two coord conventions:
        //
        //   - rs0_init / re0_init: PRE-trim chain extents in "start-of-kmer"
        //     / "end+1-of-kmer" convention. Used as the initial extraction
        //     bound, then shrunk against the max_gap-bounded window.
        //
        //   - chain_rs / chain_re / chain_qs / chain_qe: POST-trim chain
        //     extents in "center-of-kmer" convention. Used in the
        //     max_gap-bounded computation below.
        //
        // `mapping.{ref,query}_{start,end}` already store PRE-trim
        // start-of-kmer / end+1-of-kmer. On "-" strand, `mapping.query_*`
        // is forward-space; convert to alignment-space before use.
        let rs0_init = mapping.ref_start as i32;
        let re0_init = mapping.ref_end as i32;

        // Compute POST-trim chain extents. The trim depends on mode:
        //   - SR non-HPC: max-stretch trim, result in start-of-kmer /
        //     end+1-of-kmer convention (no center shift).
        //   - Splice: DP-scoring-based trim — fall back to pre-trim extents
        //     here; the wider region is safe.
        //   - Otherwise: bad-end trim, then shift to center-of-kmer.
        //
        // `align_anchors` re-runs the same trim on the same input (position
        // differences are invariant to the constant rgn_off rebase applied
        // below), so the trim is identical.
        let (chain_rs, chain_qs, chain_re, chain_qe) = if mapping.anchors.is_empty() {
            // No anchors (e.g. inversion placeholder). Apply the
            // center-of-kmer shift to PRE-trim chain extents.
            let start_shift = (mi.kmer_size as i32 - 1) - k_half;
            let end_shift = 1 + k_half;
            let (qs0_pre, qe0_pre) = if mapping.is_reverse && !is_qstrand {
                (qlen_i32 - mapping.query_end as i32,
                 qlen_i32 - mapping.query_start as i32)
            } else {
                (mapping.query_start as i32, mapping.query_end as i32)
            };
            (
                rs0_init + start_shift,
                qs0_pre + start_shift,
                re0_init - end_shift,
                qe0_pre - end_shift,
            )
        } else if is_sr_mode {
            // SR: max-stretch trim, start-of-kmer convention (no center shift).
            let (off, cnt) = crate::align::extend::max_stretch(&mapping.anchors);
            let (trim_first_idx, trim_last_idx) = if cnt == 0 {
                (0usize, mapping.anchors.len() - 1)
            } else {
                (off, off + cnt - 1)
            };
            let trim_first = &mapping.anchors[trim_first_idx];
            let trim_last = &mapping.anchors[trim_last_idx];
            let first_q_span = trim_first.query_span();
            (
                trim_first.ref_pos() + 1 - first_q_span,
                trim_first.query_pos() + 1 - first_q_span,
                trim_last.ref_pos() + 1,
                trim_last.query_pos() + 1,
            )
        } else {
            let (trim_first_idx, trim_last_idx) = if !is_splice && !no_end_flt {
                let (off, cnt) = crate::align::extend::fix_bad_ends(
                    &mapping.anchors,
                    opt.chaining.bandwidth,
                    opt.chaining.min_chain_score * 2,
                );
                if cnt == 0 {
                    (0usize, mapping.anchors.len() - 1)
                } else {
                    (off, off + cnt - 1)
                }
            } else {
                (0usize, mapping.anchors.len() - 1)
            };
            let trim_first = &mapping.anchors[trim_first_idx];
            let trim_last = &mapping.anchors[trim_last_idx];
            // Center-of-kmer shift (non-HPC): r = a.x - k/2, q = a.y - k/2.
            (
                trim_first.ref_pos() - k_half,
                trim_first.query_pos() - k_half,
                trim_last.ref_pos() - k_half,
                trim_last.query_pos() - k_half,
            )
        };

        let (rs0, re0) = if is_hpc {
            // HPC mode: the center-of-kmer shift is sequence-dependent
            // (depends on homopolymer runs in qseq/tseq) and can't be
            // computed here without those sequences. The shift is bounded
            // by the kmer span, and the underlying region formula caps the
            // bandwidth contribution at `max_gap` per side; `pad = max_gap *
            // 2` (twice that algorithmic bound) safely contains the exact
            // region. `align_anchors` does its own HPC-aware boundary calc
            // and converges to the precise window.
            let pad = max_gap as usize * 2;
            (
                std::cmp::max(0, mapping.ref_start as i32 - pad as i32),
                std::cmp::min(tlen_i32, mapping.ref_end as i32 + pad as i32),
            )
        } else if is_sr_mode {
            // SR-mode region: post-trim chain (max-stretch).
            let mut l = chain_qs;
            l += if l * a + end_bonus > q { (l * a + end_bonus - q) / e } else { 0 };
            let rs0 = std::cmp::max(0, chain_rs - l);
            let mut l = qlen_i32 - chain_qe;
            l += if l * a + end_bonus > q { (l * a + end_bonus - q) / e } else { 0 };
            let re0 = std::cmp::min(tlen_i32, chain_re + l);
            (rs0, re0)
        } else {
            // Non-SR region. Initial rs0/re0 use PRE-trim chain extents;
            // the max_gap-bounded branch uses POST-trim chain coords.
            let mut rs0 = rs0_init;
            if chain_qs > 0 && chain_rs > 0 {
                let mut l = std::cmp::min(chain_qs, max_gap);
                l += if l * a > q { (l * a - q) / e } else { 0 };
                l = std::cmp::min(l, max_gap);
                l = std::cmp::min(l, chain_rs);
                let rs1 = std::cmp::max(mapping.left_bound_rs1, chain_rs - l);
                rs0 = std::cmp::min(rs0, rs1);
                rs0 = std::cmp::min(rs0, chain_rs);
            }
            let mut re0 = re0_init;
            if chain_qe < qlen_i32 && chain_re < tlen_i32 {
                let mut l = std::cmp::min(qlen_i32 - chain_qe, max_gap);
                l += if l * a > q { (l * a - q) / e } else { 0 };
                l = std::cmp::min(l, max_gap);
                l = std::cmp::min(l, tlen_i32 - chain_re);
                let re1 = std::cmp::min(mapping.right_bound_re1, chain_re + l);
                re0 = std::cmp::max(re0, re1);
            }
            (rs0, re0)
        };

        // SEED_SELF clamp. For diagonal self-mapping chains, restrict the
        // extracted region to within `|chain_qs - chain_rs|` of the chain
        // extents. Uses PRE-trim `mapping.{ref,query}_*`. SEED_SELF chains
        // are always on the "+" strand.
        let (rs0, re0) = if !mapping.anchors.is_empty()
            && (mapping.anchors[0].y & crate::align::map::SEED_SELF) != 0 {
            let r_qs = mapping.query_start as i32;
            let r_rs = mapping.ref_start as i32;
            let r_qe = mapping.query_end as i32;
            let r_re = mapping.ref_end as i32;
            let max_ext_left = (r_qs - r_rs).abs();
            let max_ext_right = (r_qe - r_re).abs();
            let mut rs0 = rs0;
            let mut re0 = re0;
            if r_rs - rs0 > max_ext_left { rs0 = r_rs - max_ext_left; }
            if re0 - r_re > max_ext_right { re0 = r_re + max_ext_right; }
            (rs0, re0)
        } else {
            (rs0, re0)
        };

        let rgn_start = std::cmp::max(0, rs0) as usize;
        let rgn_end = std::cmp::min(tlen_i32, re0) as usize;

        // Extract target region from packed index.
        //
        // Common case (forward strand or non-qstrand): use a thread-local
        // reusable buffer (`ctx.target_buf`), growing with `resize` only
        // when too small. The buffer is left UNINITIALIZED here;
        // `align_anchors` lazy-extracts each DP segment from the index on
        // demand, keeping the active working set in cache rather than
        // streaming a megabyte-scale region through memory upfront. This
        // avoids both the per-mapping malloc/free and the redundant memset
        // that `vec![0u8; N]` would do — the two biggest costs on
        // chromosome-scale chains where region sizes hit megabytes.
        //
        // The rare qstrand+reverse path needs a full-chromosome rev-comp,
        // which doesn't fit the reusable-buffer pattern; eager-fill there
        // and signal `align_anchors` to skip its lazy fills.
        let mut qstrand_owned: Vec<u8> = Vec::new();
        let rgn_start_final: usize;
        let use_qstrand_rev = is_qstrand && mapping.is_reverse;
        if use_qstrand_rev {
            let mut full = vec![0u8; tlen];
            mi.extract_nt4_into(rid, 0, tlen, &mut full);
            qstrand_owned = rev_comp_nt4(&full);
            rgn_start_final = 0; // full chromosome
        } else {
            let size = rgn_end - rgn_start;
            if ctx.target_buf.len() < size {
                ctx.target_buf.resize(size, 0);
            }
            // Note: NO `extract_nt4_into` here. `align_anchors` fills tseq
            // lazily per DP segment + once for the alignment range before
            // CIGAR finalization.
            rgn_start_final = rgn_start;
        };

        // Use the caller-provided nt4-encoded query (forward or reverse-
        // complement). Previously this re-encoded `qseq` on every chain;
        // for chromosome-scale qlens with many chains that was the
        // dominant cost. The caller (`align_and_format`) encodes once per
        // outer call and passes references in.
        let query_seq_for_aln: &[u8] = if mapping.is_reverse && !is_qstrand {
            qseq_rc_nt4
        } else {
            qseq_fwd_nt4
        };

        // Adjust anchor ref positions to region-relative coordinates
        let rgn_off = rgn_start_final as i32;
        for anchor in mapping.anchors.iter_mut() {
            anchor.set_ref_pos(anchor.ref_pos() - rgn_off);
        }

        // Adjust seed_bounds to region-relative. The target slice we'll pass
        // to align_anchors is either the rev-comp'd qstrand buffer or a
        // slice of `ctx.target_buf[..rgn_size]`. Either way the length is
        // `rgn_end - rgn_start` for non-qstrand and `tlen` for qstrand+rev.
        let target_len_for_bounds: i32 = if use_qstrand_rev {
            qstrand_owned.len() as i32
        } else {
            (rgn_end - rgn_start) as i32
        };
        let adj_seed_bounds = (
            (mapping.left_bound_rs1 - rgn_off).max(0),
            mapping.left_bound_qs1,
            (mapping.right_bound_re1 - rgn_off).min(target_len_for_bounds),
            mapping.right_bound_qe1,
        );

        let call_ctx = AlignAnchorContext {
            seed_bounds: adj_seed_bounds,
            rev: mapping.is_reverse,
            rid: mapping.ref_id,
            splice_flag,
            split_inv: mapping.split_inv,
            is_hpc: mi.homopolymer_compressed,
            k: mi.kmer_size,
            junc_db,
            ref_offset: rgn_start_final,
        };
        // Pick the target buffer and the lazy-extract context. For
        // qstrand+reverse the buffer is pre-filled with rev-comp'd full
        // chromosome bytes, so `lazy_extract = None` (skip per-segment
        // fills). For the common path, the buffer is uninitialized and
        // `align_anchors` fills segments on demand.
        let region_size = if use_qstrand_rev { qstrand_owned.len() } else { rgn_end - rgn_start };
        let aln_result = if use_qstrand_rev {
            align_anchors(
                &mut mapping.anchors,
                query_seq_for_aln,
                &mut qstrand_owned,
                None,
                opt,
                &call_ctx,
            )
        } else {
            align_anchors(
                &mut mapping.anchors,
                query_seq_for_aln,
                &mut ctx.target_buf[..region_size],
                Some((mi, rid, rgn_start)),
                opt,
                &call_ctx,
            )
        };

        // After align_anchors, the alignment range [rs1..final_re) of the
        // active buffer has been filled (the lazy-fill pass before
        // finalize_cigar) — safe for downstream tag generation.
        let target_region: &[u8] = if use_qstrand_rev {
            &qstrand_owned
        } else {
            &ctx.target_buf[..region_size]
        };

        // Restore anchor ref positions to absolute (for split mapping reuse)
        for anchor in mapping.anchors.iter_mut() {
            anchor.set_ref_pos(anchor.ref_pos() + rgn_off);
        }

        let ops = aln_result.cigar_ops;
        let new_qs = aln_result.query_start;
        let new_qe = aln_result.query_end;
        // Convert region-relative results back to absolute chromosome coords
        let new_rs = aln_result.ref_start + rgn_start_final;
        let new_re = aln_result.ref_end + rgn_start_final;

        // Handle split: if z-drop triggered, save right part for later alignment
        if let Some(mut split_anchors) = aln_result.split_right_anchors {
            // Restore split anchor ref positions to absolute chromosome coords
            for a in split_anchors.iter_mut() {
                a.set_ref_pos(a.ref_pos() + rgn_off);
            }
            let split_cnt = split_anchors.len() as i32;
            let orig_cnt = mapping.anchors.len() as i32;
            effective_cnt = orig_cnt - split_cnt; // r->cnt -= r2->cnt
            let split_score = ((mapping.score as f32 * (split_cnt as f32 / orig_cnt as f32)) as f64 + 0.499) as i32;
            chain_score_adj = mapping.score - split_score; // r->score -= r2->score

            // Compute split's position in the squeezed array for nearby-seed bounds
            let split_sq_start = mapping.sq_start + aln_result.split_offset_in_orig.unwrap_or(0);
            let split_sq_cnt = split_cnt as usize;
            let tlen_i32 = mi.seqs[mapping.ref_id].len as i32;
            let qlen_i32 = qlen as i32;
            let (split_rs1, split_qs1, split_re1, split_qe1) =
                compute_bounds_from_squeezed(squeezed, split_sq_start, split_sq_cnt, tlen_i32, qlen_i32, opt.chaining.min_cnt);

            let mut split_mapping = Mapping {
                ref_id: mapping.ref_id,
                query_start: 0, query_end: 0, ref_start: 0, ref_end: 0,
                score: split_score,
                initial_chain_score: mapping.initial_chain_score, // split inherits initial_chain_score
                anchor_count: split_cnt,
                anchors: split_anchors,
                is_reverse: mapping.is_reverse,
                s2_score: mapping.s2_score, // split inherits subsc
                is_secondary: mapping.is_secondary,
                hash: mapping.hash,
                pre_num_suboptimal: mapping.pre_num_suboptimal, // split inherits n_sub
                left_bound_rs1: split_rs1,
                left_bound_qs1: split_qs1,
                right_bound_re1: split_re1,
                right_bound_qe1: split_qe1,
                original_as: mapping.original_as, // split inherits from parent
                sq_start: split_sq_start,
                div: mapping.div,
                strand_retained: mapping.strand_retained,
                compact_parent: mapping.compact_parent,
                split_inv: aln_result.split_inv,
                seg_split: mapping.seg_split,
                seg_id: mapping.seg_id,
                is_alt: mapping.is_alt,
            };
            // Set coordinates from anchors
            if !split_mapping.anchors.is_empty() {
                let first = &split_mapping.anchors[0];
                let last = &split_mapping.anchors[split_mapping.anchors.len() - 1];
                let q_span = first.query_span() as usize;
                let first_r_end = first.ref_pos() as usize;
                let first_q_end = first.query_pos() as usize;
                let last_r_end = last.ref_pos() as usize;
                let last_q_end = last.query_pos() as usize;
                split_mapping.ref_start = first_r_end + 1 - q_span;
                split_mapping.ref_end = last_r_end + 1;
                if !mapping.is_reverse {
                    split_mapping.query_start = first_q_end + 1 - q_span;
                    split_mapping.query_end = last_q_end + 1;
                } else {
                    split_mapping.query_start = qlen.saturating_sub(last_q_end + 1);
                    split_mapping.query_end = qlen.saturating_sub(first_q_end + 1 - q_span);
                }
            }
            new_split = Some(split_mapping);
        }

        // Use raw dp_score from DP segments (for AS tag)
        align_score = aln_result.dp_score;
        // Build scoring matrix — use region-relative coords for sequence access
        let mat = crate::align::extend::build_scoring_matrix_full(opt.scoring.match_score, opt.scoring.mismatch_penalty, opt.scoring.transition, opt.scoring.ambig_penalty);
        let rgn_rs = aln_result.ref_start; // region-relative
        let rgn_re = aln_result.ref_end;   // region-relative
        let aln_qseq = &query_seq_for_aln[new_qs..new_qe];
        let aln_tseq = &target_region[rgn_rs..rgn_re];
        // SR uses flat q+e penalty; non-SR uses logarithmic gap penalty
        let log_gap = !opt.flags.intersects(AlignFlags::SHORT_READ | AlignFlags::SR_RNA);
        dp_max = compute_alignment_score_max(&ops, &mat, opt.scoring.gap_open, opt.scoring.gap_extend, aln_qseq, aln_tseq, log_gap);
        recalc_info = DpRecalcInfo::from_ops(&ops);

        if mapping.is_reverse && !is_qstrand {
            qs = qlen - new_qe;
            qe = qlen - new_qs;
        } else {
            qs = new_qs;
            qe = new_qe;
        }
        rs = new_rs;
        re = new_re;

        let stats = CigarStats::from_cigar(&ops, aln_qseq, aln_tseq);
        matches = stats.matches;
        nm = stats.edit_distance;
        block_len = stats.block_len;
        nn = stats.num_ambiguous;
        de = stats.divergence;
        has_n_skip = stats.has_n_skip;

        cigar_str = fmt_cigar(&ops, out.eqx);
        if out.do_cs {
            cs_str = fmt_cs(&ops, query_seq_for_aln, target_region, new_qs, rgn_rs, out.cs_long);
        }
        if out.do_md {
            md_str = fmt_md(&ops, target_region, rgn_rs);
        }
        if out.do_ds {
            ds_str = fmt_ds(&ops, query_seq_for_aln, target_region, new_qs, rgn_rs);
        }
    }

    let is_spliced = has_n_skip;

    let result = AlnResult {
        ref_id: mapping.ref_id,
        is_reverse: mapping.is_reverse,
        chain_score: chain_score_adj,  // reduced after z-drop split
        initial_chain_score: mapping.initial_chain_score,       // never modified
        anchor_count: effective_cnt as usize,   // reduced after z-drop split
        s2_score: mapping.s2_score,
        hash: mapping.hash,
        align_score,
        matches,
        block_len,
        cigar_str,
        cs_str,
        ds_str,
        md_str,
        query_start: qs, query_end: qe, ref_start: rs, ref_end: re,
        edit_distance: nm, num_ambiguous: nn, divergence: de,
        is_secondary: mapping.is_secondary,
        split: 0,
        split_depth: 0,
        dp_score: dp_max,
        dp_score_original: dp_max,
        effective_cnt,
        pre_num_suboptimal: mapping.pre_num_suboptimal,
        is_spliced,
        trans_strand: 0,
        dp_score_secondary: 0,
        secondary_chain_score: 0,
        num_suboptimal: 0,
        split_inv: false,
        inv: false,
        proper_frag: false,
        seg_split: mapping.seg_split,
        div: mapping.div,
        is_alt: false,
        is_root_chain: !mapping.is_secondary,
    };

    (result, recalc_info, new_split)
}


/// Attempt inversion alignment from z-drop split positions.
/// Returns Some(AlnResult) if an inversion alignment was produced.
fn try_align_inversion(
    opt: &MapOptions,
    mi: &Index,
    qlen: usize,
    qseq_fwd_nt4: &[u8],   // forward query, nt4-encoded
    qseq_rc_nt4: &[u8],    // reverse-complement query, nt4-encoded
    r1: &AlnResult,         // left split part (split & 1)
    r2: &AlnResult,         // right split part (split & 2)
    out: &OutputConfig,
) -> Option<AlnResult> {
    // Preconditions
    if (r1.split & 1) == 0 || (r2.split & 2) == 0 { return None; }
    // Parent gate. In ALL_CHAINS mode parent assignment is skipped, so
    // "primary" reduces to "this chain is itself a split-child".
    if opt.flags.contains(AlignFlags::ALL_CHAINS) {
        if (r1.split & 2) == 0 || (r2.split & 2) == 0 { return None; }
    } else if !r1.is_root_chain || !r2.is_root_chain {
        return None;
    }
    if r1.ref_id != r2.ref_id || r1.is_reverse != r2.is_reverse { return None; }

    let ql = if r1.is_reverse {
        if r1.query_start < r2.query_end { return None; }
        (r1.query_start - r2.query_end) as i32
    } else {
        if r2.query_start < r1.query_end { return None; }
        (r2.query_start - r1.query_end) as i32
    };
    let tl = if r2.ref_start < r1.ref_end { return None; } else { (r2.ref_start - r1.ref_end) as i32 };

    if ql < opt.chaining.min_chain_score || ql > opt.chaining.max_gap { return None; }
    if tl < opt.chaining.min_chain_score || tl > opt.chaining.max_gap { return None; }

    let mat = build_scoring_matrix_full(opt.scoring.match_score, opt.scoring.mismatch_penalty, opt.scoring.transition, opt.scoring.ambig_penalty);

    // Get target sequence
    let mut tseq: Vec<u8> = mi.get_region_nt4(r1.ref_id, r1.ref_end, r2.ref_start);

    // Get query sequence (forward or RC, nt4-encoded)
    let mut qseq: Vec<u8> = if r1.is_reverse {
        // qseq = &qseq0[0][r2->qe]  (length = ql)
        qseq_fwd_nt4[r2.query_end..r2.query_end + ql as usize].to_vec()
    } else {
        // qseq = &qseq0[1][qlen - r2->qs]  (length = ql)
        let start = qlen - r2.query_start;
        qseq_rc_nt4[start..start + ql as usize].to_vec()
    };

    // Reverse both for lightweight alignment
    qseq.reverse();
    tseq.reverse();

    // Lightweight alignment
    let mut qp = crate::align::dp::lightweight_profile_init(ql, &qseq, 5, &mat);
    let (score, mut q_off, mut t_off) = crate::align::dp::lightweight_align_i16(&mut qp, tl, &tseq, opt.scoring.gap_open, opt.scoring.gap_extend);

    // Reverse back
    qseq.reverse();
    tseq.reverse();

    if score < opt.alignment.min_dp_max { return None; }
    if q_off < 0 || t_off < 0 { return None; }

    // Adjust offsets. Lightweight guarantees in-bounds positions, so post-transform
    // q_off in [0, ql-1] and t_off in [0, tl-1].
    q_off = ql - (q_off + 1);
    t_off = tl - (t_off + 1);


    // Full alignment
    let bw = (opt.chaining.bandwidth as f64 * 1.5) as i32;
    let q_sub = &qseq[q_off as usize..];
    let t_sub = &tseq[t_off as usize..];

    let mut ez = crate::align::dp::DpResult::default();
    let mut dp_flag = crate::align::dp::EXTENSION_ONLY;
    // Add GENERIC_SC when transition scoring is active
    if opt.scoring.transition != 0 && opt.scoring.mismatch_penalty != opt.scoring.transition {
        dp_flag |= crate::align::dp::GENERIC_SCORING;
    }
    if opt.scoring.gap_open == opt.scoring.gap_open2 && opt.scoring.gap_extend == opt.scoring.gap_extend2 {
        crate::align::dp::extend_single_affine(q_sub, t_sub, 5, &mat, opt.scoring.gap_open as i8, opt.scoring.gap_extend as i8, bw, opt.alignment.zdrop, -1, dp_flag, &mut ez);
    } else {
        crate::align::dp::extend_dual_affine(q_sub, t_sub, 5, &mat, opt.scoring.gap_open as i8, opt.scoring.gap_extend as i8, opt.scoring.gap_open2 as i8, opt.scoring.gap_extend2 as i8, bw, opt.alignment.zdrop, -1, dp_flag, &mut ez);
    }
    if ez.cigar.is_empty() { return None; }

    // Normalize CIGAR (left-align indels, merge ops, strip leading I/D). Returns
    // (qshift, tshift) — bases removed from the alignment start. The alignment END
    // position is unchanged; only the START moves forward by the shift.
    let (qshift, tshift) = fix_cigar_pub(&mut ez.cigar, &qseq, &tseq, q_off, t_off);

    // Build inversion result. End coords use the original (pre-shift) offsets
    // because ez.max_score_*_pos is relative to the DP's original starting point.
    let inv_rev = !r1.is_reverse;
    let rid = r1.ref_id;

    let (inv_qs, inv_qe) = if !inv_rev {
        let qs = r2.query_end + (q_off + qshift) as usize;
        let qe = r2.query_end + q_off as usize + ez.max_score_query_pos as usize + 1;
        (qs, qe)
    } else {
        let qe = r2.query_start - (q_off + qshift) as usize;
        let qs = (r2.query_start - q_off as usize) - (ez.max_score_query_pos as usize + 1);
        (qs, qe)
    };
    let inv_rs = r1.ref_end + (t_off + tshift) as usize;
    let inv_re = r1.ref_end + t_off as usize + ez.max_score_target_pos as usize + 1;

    // CIGAR processing below uses post-shift offsets (the leading op was stripped,
    // so the remaining CIGAR aligns starting at q_off+qshift / t_off+tshift).
    q_off += qshift;
    t_off += tshift;

    // Convert CIGAR to =/X ops
    let raw_cigar = &ez.cigar;
    let ops = convert_cigar_to_eqx_pub(raw_cigar, &qseq, &tseq, q_off as usize, t_off as usize);
    let mut condensed: Vec<CigarOp> = Vec::new();
    for op in &ops {
        if let Some(last) = condensed.last_mut() && last.op == op.op {
            last.len += op.len;
            continue;
        }
        condensed.push(op.clone());
    }

    // Compute stats
    let log_gap = !opt.flags.intersects(AlignFlags::SHORT_READ | AlignFlags::SR_RNA);
    let dp_max = compute_alignment_score_max(&condensed, &mat, opt.scoring.gap_open, opt.scoring.gap_extend, &qseq[q_off as usize..], &tseq[t_off as usize..], log_gap);
    // Add back the cost of the stripped leading gap
    let stripped_gap_len = qshift + tshift; // exactly one of qshift/tshift is non-zero
    let align_score = if stripped_gap_len > 0 {
        ez.max + opt.scoring.gap_open + opt.scoring.gap_extend * stripped_gap_len
    } else {
        ez.max
    };

    let cigar_str = fmt_cigar(&condensed, out.eqx);
    // fmt_cs/md/ds now accept nt4-encoded sequences directly
    let cs_str = if out.do_cs {
        fmt_cs(&condensed, &qseq, &tseq, q_off as usize, t_off as usize, out.cs_long)
    } else {
        String::new()
    };
    let md_str = if out.do_md {
        fmt_md(&condensed, &tseq, t_off as usize)
    } else {
        String::new()
    };
    let ds_str = if out.do_ds {
        fmt_ds(&condensed, &qseq, &tseq, q_off as usize, t_off as usize)
    } else {
        String::new()
    };

    let aln_qseq = &qseq[q_off as usize..];
    let aln_tseq = &tseq[t_off as usize..];
    let stats = CigarStats::from_cigar(&condensed, aln_qseq, aln_tseq);
    let matches = stats.matches;
    let nm = stats.edit_distance;
    let block_len = stats.block_len;
    let nn = stats.num_ambiguous;
    let de = stats.divergence;

    Some(AlnResult {
        ref_id: rid,
        is_reverse: inv_rev,
        chain_score: 0,    // s1:i:0 for inversions (cm:i:0 too)
        initial_chain_score: 0,
        anchor_count: 0,
        s2_score: None,
        hash: 0,
        align_score,
        matches,
        block_len,
        cigar_str,
        cs_str,
        ds_str,
        md_str,
        query_start: inv_qs,
        query_end: inv_qe,
        ref_start: inv_rs,
        ref_end: inv_re,
        edit_distance: nm,
        num_ambiguous: nn,
        divergence: de,
        is_secondary: false,
        split: 0,
        split_depth: 0,
        dp_score: dp_max,
        dp_score_original: dp_max,
        effective_cnt: 0,
        pre_num_suboptimal: 0,
        is_spliced: false,
        trans_strand: 0,
        dp_score_secondary: 0,
        secondary_chain_score: 0,
        num_suboptimal: 0,
        split_inv: false,
        inv: true,
        proper_frag: false,
        seg_split: false,
        div: -1.0,  // inversions don't have estimated divergence
        is_alt: false,
        is_root_chain: true, // inversions only produced from root chains (checked in preconditions)
    })
}

/// Results from processing a single query through the alignment pipeline.
/// Contains everything needed for SAM/PAF output.
pub struct ProcessedQuery {
    pub results: Vec<AlnResult>,
    pub mapqs: Vec<i32>,
    pub sam_pri: Vec<bool>,
    pub parent_indices: Vec<usize>, // parent index in results array (self for primaries)
    pub rep_len: i32,
    pub stats: AlignmentStats,
}

/// Core alignment pipeline: map → align → filter → MAPQ.
/// Returns ProcessedQuery suitable for output formatting.
pub fn process_query(
    opt: &MapOptions,
    mi: &Index,
    qname: &str,
    qseq: &[u8],
    ctx: &mut AlignmentContext,
    map_ctx: &mut MapContext,
    junc_db: Option<&JunctionDb>,
    out: &OutputConfig,
) -> ProcessedQuery {
    // max_qlen check: skip mapping if query length exceeds limit
    if opt.filtering.max_qlen > 0 && qseq.len() > opt.filtering.max_qlen as usize {
        return ProcessedQuery {
            results: vec![], mapqs: vec![], sam_pri: vec![], parent_indices: vec![],
            rep_len: 0, stats: AlignmentStats::default(),
        };
    }
    let (regs, rep_len, stats, squeezed) = map_query(opt, mi, qname, qseq, map_ctx);
    let map_result = MapResult { regs, rep_len, stats, squeezed };
    process_query_core(opt, mi, qseq, ctx, map_ctx, junc_db, out, map_result)
}

/// Process pre-built registrations from seg_gen (strong pairing path).
/// Builds the squeezed array from regs' anchors and runs the alignment/MAPQ pipeline.
pub fn process_query_from_regs(
    opt: &MapOptions,
    mi: &Index,
    qseq: &[u8],
    ctx: &mut AlignmentContext,
    map_ctx: &mut MapContext,
    junc_db: Option<&JunctionDb>,
    out: &OutputConfig,
    mut regs: Vec<Mapping>,
    rep_len: usize,
) -> ProcessedQuery {
    let stats = AlignmentStats::default();

    // Build squeezed array from regs' anchors (same logic as end of map_query)
    let mut pos_order: Vec<usize> = (0..regs.len()).collect();
    pos_order.sort_by_key(|&i| regs[i].original_as);
    let mut squeezed: Vec<Minimizer> = Vec::new();
    for &pi in &pos_order {
        let start = squeezed.len();
        squeezed.extend_from_slice(&regs[pi].anchors);
        regs[pi].sq_start = start;

        // Compute bounds
        let sq_cnt = regs[pi].anchors.len();
        if sq_cnt > 0 {
            let rid = regs[pi].ref_id;
            let tlen_i32 = mi.seqs[rid].len as i32;
            let qlen_i32 = qseq.len() as i32;
            let (left_rs1, left_qs1, right_re1, right_qe1) =
                compute_bounds_from_squeezed(&squeezed, start, sq_cnt, tlen_i32, qlen_i32, opt.chaining.min_cnt);
            regs[pi].left_bound_rs1 = left_rs1;
            regs[pi].left_bound_qs1 = left_qs1;
            regs[pi].right_bound_re1 = right_re1;
            regs[pi].right_bound_qe1 = right_qe1;
        }
    }

    // process_query_from_regs is only called from the strong pairing path, never in split mode
    let map_result = MapResult { regs, rep_len, stats, squeezed };
    process_query_core(opt, mi, qseq, ctx, map_ctx, junc_db, out, map_result)
}

/// Compute per-read MAPQs from alignment results.
/// Runs AFTER pair assignment so parent assignments reflect any PE promotions.
fn compute_mapping_qualities(
    results: &[AlnResult],
    parent_indices: &[usize],
    opt: &MapOptions,
    rep_len: f32,
) -> Vec<i32> {
    let is_sr_mapq = opt.flags.intersects(AlignFlags::SHORT_READ | AlignFlags::SR_RNA);
    let is_splice_mapq = opt.flags.contains(AlignFlags::SPLICE);

    // uniq_ratio = sum_sc / (sum_sc + rep_len)
    let mut uniq_ratio = 1.0f32;
    if !results.is_empty() {
        let sum_sc: i64 = results.iter().enumerate()
            .filter(|(i, _)| parent_indices[*i] == *i)
            .map(|(_, x)| x.chain_score as i64)
            .sum();
        if sum_sc > 0 {
            uniq_ratio = sum_sc as f32 / (sum_sc as f32 + rep_len);
        }
    }

    // Count non-root spliced alignments
    let n_2nd_splice: i32 = results.iter().enumerate()
        .filter(|(i, r)| parent_indices[*i] != *i && r.is_spliced)
        .count() as i32;

    // Compute MAPQ for each root (parent==id), children get 0
    // Inversions always get mapq=0, then inv_mapq updates them
    let mut mapqs: Vec<i32> = results.iter().enumerate().map(|(idx, r)| {
        let is_root = parent_indices[idx] == idx;
        if r.inv {
            0
        } else if is_root {
            let initial_chain_score = r.initial_chain_score as f32;
            let score = r.chain_score as f32;
            let dp_max = r.dp_score as f32;
            let dp_max2 = r.dp_score_secondary as f32;
            let identity = if r.block_len > 0 { r.matches as f32 / r.block_len as f32 } else { 1.0 };
            let pen_s1 = (if r.chain_score > 100 { 1.0f32 } else { 0.01 * score }) * uniq_ratio;
            let pen_cm = if r.anchor_count > 10 { 1.0f32 } else { 0.1 * r.anchor_count as f32 };
            let pen_cm = pen_s1.min(pen_cm);
            let match_sc = opt.scoring.match_score as f32;
            let q_coef = 40.0f32;
            let subsc = (r.secondary_chain_score.max(opt.chaining.min_chain_score)) as f32;

            // cigar_str is empty when no alignment was done (non-CIGAR PAF mode).
            let has_p = !r.cigar_str.is_empty();

            let mut mapq;
            if has_p && r.dp_score_secondary > 0 && r.dp_score > 0 {
                let x = if is_sr_mapq && is_splice_mapq {
                    dp_max2 / dp_max
                } else {
                    dp_max2 * subsc / dp_max / initial_chain_score
                };
                mapq = (identity * pen_cm * q_coef * (1.0 - x * x) * (dp_max / match_sc).ln()) as i32;
                if !is_sr_mapq {
                    let mapq_alt = (6.02f32 * identity * identity * (dp_max - dp_max2) / match_sc + 0.499) as i32;
                    mapq = mapq.min(mapq_alt);
                }
                if is_splice_mapq && is_sr_mapq && r.is_spliced && n_2nd_splice == 0 {
                    mapq += 10;
                }
            } else {
                let x = subsc / initial_chain_score;
                if has_p {
                    mapq = (identity * pen_cm * q_coef * (1.0 - x) * (dp_max / match_sc).ln()) as i32;
                } else {
                    mapq = (pen_cm * q_coef * (1.0 - x) * score.ln()) as i32;
                }
            }

            let n_sub_penalty = (4.343f32 * (r.num_suboptimal as f32 + 1.0).ln() + 0.499) as i32;
            mapq -= n_sub_penalty;
            mapq = mapq.clamp(0, 60);
            if has_p && r.dp_score > r.dp_score_secondary && mapq == 0 {
                mapq = 1;
            }
            mapq
        } else {
            0
        }
    }).collect();

    // Set inversion MAPQ from flanking alignments. ALL_CHAINS mode includes
    // every chain in the aux pool; other modes exclude secondaries.
    if mapqs.len() >= 3 {
        let all_chains_mode = opt.flags.contains(AlignFlags::ALL_CHAINS);
        let mut aux: Vec<(u64, u64)> = Vec::new();
        for (i, r) in results.iter().enumerate() {
            let include = if all_chains_mode { true } else { !r.is_secondary };
            if include {
                aux.push((((r.ref_id as u64) << 32) | r.ref_start as u64, i as u64));
            }
        }
        crate::align::sort::radix_sort_128x_pair(&mut aux);
        for k in 1..aux.len().saturating_sub(1) {
            let idx = aux[k].1 as usize;
            if results[idx].inv {
                let left_mapq = mapqs[aux[k - 1].1 as usize];
                let right_mapq = mapqs[aux[k + 1].1 as usize];
                mapqs[idx] = left_mapq.min(right_mapq);
            }
        }
    }

    mapqs
}

/// Post-alignment filtering.
/// Removes results that fail min_cnt, min_chain_score, min_dp_max, or max_clip_ratio thresholds.
/// When `recalc_infos` is Some, filters both results and recalc_infos in parallel.
fn filter_alignment_results(
    results: &mut Vec<AlnResult>,
    recalc_infos: Option<&mut Vec<DpRecalcInfo>>,
    min_cnt: i32,
    min_chain_score: i32,
    min_dp_max: i32,
    max_clip_ratio: f32,
    qlen: usize,
) {
    let clip_thresh = qlen as f32 * max_clip_ratio;
    if let Some(recalc) = recalc_infos {
        // Filter both results and recalc_infos together
        let mut keep = vec![true; results.len()];
        for (i, r) in results.iter().enumerate() {
            if !r.inv && !r.seg_split && r.effective_cnt < min_cnt {
                keep[i] = false;
            }
            if (r.matches as i32) < min_chain_score {
                keep[i] = false;
            }
            if r.dp_score < min_dp_max {
                keep[i] = false;
            }
            if (r.query_start as f32) > clip_thresh
                && ((qlen.saturating_sub(r.query_end)) as f32) > clip_thresh
            {
                keep[i] = false;
            }
        }
        let mut new_results = Vec::with_capacity(results.len());
        let mut new_recalc = Vec::with_capacity(recalc.len());
        let old_results = std::mem::take(results);
        let old_recalc = std::mem::take(recalc);
        for (i, (r, rc)) in old_results.into_iter().zip(old_recalc.into_iter()).enumerate() {
            if keep[i] {
                new_results.push(r);
                new_recalc.push(rc);
            }
        }
        *results = new_results;
        *recalc = new_recalc;
    } else {
        // Filter results only (retain path)
        results.retain(|r| {
            let flt = (!r.inv && !r.seg_split && r.effective_cnt < min_cnt)
                || (r.matches as i32) < min_chain_score
                || r.dp_score < min_dp_max
                || ((r.query_start as f32) > clip_thresh
                    && ((qlen.saturating_sub(r.query_end)) as f32) > clip_thresh);
            !flt
        });
    }
}

/// Post-alignment parent assignment, secondary selection, and dp_max2 tracking.
/// Sorts results, assigns parents, selects secondaries, and counts suboptimal hits.
/// Handles ALL_CHAINS bypass, CIGAR vs non-CIGAR modes.
/// Returns (filtered results, parent_indices).
fn assign_parents_and_select(
    mut results: Vec<AlnResult>,
    recalc_infos: &[DpRecalcInfo],
    opt: &MapOptions,
    mi: &Index,
    out: &OutputConfig,
    split_mode: bool,
    qlen: usize,
) -> (Vec<AlnResult>, Vec<usize>) {
    let mut parent_indices: Vec<usize> = (0..results.len()).collect();

    // ALL_CHAINS mode (ava-ont, ava-pb): only direct (depth==1) split-children
    // and inversion alignments count as primaries; originals and deeper splits
    // are secondary.
    if !results.is_empty() && opt.flags.contains(AlignFlags::ALL_CHAINS) {
        for r in results.iter_mut() {
            r.is_secondary = !(r.split_depth == 1 || r.inv);
        }
        parent_indices = results.iter().enumerate()
            .map(|(i, r)| if r.is_secondary { usize::MAX } else { i })
            .collect();
    }

    // Non-CIGAR mode: preserve pre-alignment parent structure from chain_post.
    // Parent assignment and secondary selection run during chain_post, then
    // MAPQ computation uses that parent structure. We need parent_indices to reflect
    // roots (is_sec=false) vs children (is_sec=true) for correct MAPQ and s2 output.
    if !results.is_empty() && !out.do_cigar && !opt.flags.contains(AlignFlags::ALL_CHAINS) {
        // Set parent_indices: roots (is_sec=false) point to self, secondaries point elsewhere
        // Find first root for secondaries to reference
        let first_root = results.iter().position(|r| !r.is_secondary).unwrap_or(0);
        for (i, r) in results.iter().enumerate() {
            if r.is_secondary {
                parent_indices[i] = first_root;
            }
        }
        // Preserve pre-alignment subsc (s2) and n_sub from chain_post
        // Parent assignment sets both subsc and n_sub during chain_post,
        // and MAPQ computation uses them directly (no recomputation in non-CIGAR mode)
        for r in results.iter_mut() {
            r.secondary_chain_score = r.s2_score.unwrap_or(0);
            r.num_suboptimal = r.pre_num_suboptimal;
        }
    }

    if !results.is_empty() && out.do_cigar {
        // Skip update_dp_max for SR, SR_RNA, ALL_CHAINS, split_prefix; require qlen >= rank_min_len (500)
        let do_update_dp = !split_mode && !opt.flags.intersects(AlignFlags::SHORT_READ | AlignFlags::SR_RNA | AlignFlags::ALL_CHAINS) && qlen >= 500;
        if do_update_dp {
            let mut dp_max_vals: Vec<i32> = results.iter().map(|r| r.dp_score).collect();
            let qs_vals: Vec<usize> = results.iter().map(|r| r.query_start).collect();
            let qe_vals: Vec<usize> = results.iter().map(|r| r.query_end).collect();
            update_dp_max(&mut dp_max_vals, recalc_infos, &qs_vals, &qe_vals, qlen, 0.9, opt.scoring.match_score, opt.scoring.mismatch_penalty);
            for (i, r) in results.iter_mut().enumerate() {
                r.dp_score = dp_max_vals[i];
            }
        }

        if out.do_cigar {
            filter_alignment_results(&mut results, None, opt.chaining.min_cnt, opt.chaining.min_chain_score, opt.alignment.min_dp_max, opt.alignment.max_clip_ratio, qlen);
        }

        // Propagate is_alt from index sequences to results
        for r in results.iter_mut() {
            if mi.seqs[r.ref_id].is_alt {
                r.is_alt = true;
            }
        }

        if opt.flags.contains(AlignFlags::ALL_CHAINS) {
            // Parent assignment was done above. Still re-sort by (dp_max, hash)
            // descending so the inv-mapq aux ordering uses the same regs order
            // as if parent_select had run.
            let alt_drop = opt.filtering.alt_drop;
            let mut aux: Vec<(u64, u64)> = results.iter().enumerate().map(|(i, r)| {
                let score = if r.is_alt { scale_alt_score(r.dp_score, alt_drop) } else { r.dp_score };
                let key = ((score as u64) << 32) | (r.hash as u64);
                (key, i as u64)
            }).collect();
            radix_sort_128x_pair(&mut aux);
            let mut opt_results: Vec<Option<AlnResult>> = results.into_iter().map(Some).collect();
            results = aux.iter().rev().map(|&(_, i)| opt_results[i as usize].take().unwrap()).collect();
            // Recompute parent_indices to track new positions of primaries.
            parent_indices = results.iter().enumerate()
                .map(|(i, r)| if r.is_secondary { usize::MAX } else { i })
                .collect();
        } else {

        // Sort by (dp_max, hash) descending with unstable but deterministic paired radix sort
        {
            let alt_drop = opt.filtering.alt_drop;
            let mut aux: Vec<(u64, u64)> = results.iter().enumerate().map(|(i, r)| {
                let score = if r.is_alt { scale_alt_score(r.dp_score, alt_drop) } else { r.dp_score };
                let key = ((score as u64) << 32) | (r.hash as u64);
                (key, i as u64)
            }).collect();
            radix_sort_128x_pair(&mut aux);
            let mut opt_results: Vec<Option<AlnResult>> = results.into_iter().map(Some).collect();
            results = aux.iter().rev().map(|&(_, i)| opt_results[i as usize].take().unwrap()).collect();
        }

        let filter_items: Vec<FilterableItem> = results
            .iter()
            .map(|r| FilterableItem {
                query_start: r.query_start,
                query_end: r.query_end,
                score: r.chain_score,
                is_reverse: r.is_reverse,
                is_alt: r.is_alt,
            })
            .collect();

        let filter_params = FilterParams::new(opt, mi);
        let mut parent_state = ParentState::new(filter_items.len(), filter_params.mask_level, filter_params.mask_len, filter_params.hard_mask_level);
        parent_state.init_from_items(&filter_items);
        parent_state.assign_parents(&filter_items);

        let sub_diff = opt.scoring.match_score * 2 + opt.scoring.mismatch_penalty;
        let mut dp_max2 = vec![0i32; results.len()];
        let mut subsc_vec: Vec<i32> = results.iter()
            .map(|r| r.s2_score.unwrap_or(0))
            .collect();
        let mut n_sub_post = vec![0i32; results.len()];
        if !results.is_empty() {
            n_sub_post[0] = results[0].pre_num_suboptimal;
        }
        let alt_drop = opt.filtering.alt_drop;
        for i in 0..results.len() {
            let pi = parent_state.parent[i];
            if pi != i {
                // ALT penalty for subsc (chain score comparison)
                let sci = if !results[pi].is_alt && results[i].is_alt {
                    scale_alt_score(results[i].chain_score, alt_drop)
                } else {
                    results[i].chain_score
                };
                if sci > subsc_vec[pi] {
                    subsc_vec[pi] = sci;
                }
                let mut cnt_sub = false;
                if results[i].anchor_count >= results[pi].anchor_count {
                    cnt_sub = true;
                }
                let (si, ei) = (results[i].query_start, results[i].query_end);
                let (sj, ej) = (results[pi].query_start, results[pi].query_end);
                let ol = {
                    let start = si.max(sj);
                    let end = ei.min(ej);
                    end.saturating_sub(start)
                };
                let min_len = (ei - si).min(ej - sj);
                let not_identical = results[i].ref_id != results[pi].ref_id
                    || results[i].ref_start != results[pi].ref_start
                    || results[i].ref_end != results[pi].ref_end
                    || ol != min_len;
                if not_identical {
                    // ALT penalty for dp_max2 comparison
                    let dp_sci = if !results[pi].is_alt && results[i].is_alt {
                        scale_alt_score(results[i].dp_score, alt_drop)
                    } else {
                        results[i].dp_score
                    };
                    if dp_sci > dp_max2[pi] {
                        dp_max2[pi] = dp_sci;
                    }
                    if results[pi].dp_score - results[i].dp_score <= sub_diff {
                        cnt_sub = true;
                    }
                }
                if cnt_sub {
                    n_sub_post[pi] += 1;
                }
            }
        }

        for i in 0..results.len() {
            results[i].dp_score_secondary = dp_max2[i];
            results[i].secondary_chain_score = subsc_vec[i];
            results[i].num_suboptimal = n_sub_post[i];
        }

        // Select secondaries
        let n = results.len();
        let score_at: Vec<(i32, bool)> = results.iter().map(|r| (r.chain_score, r.is_reverse)).collect();
        let mut keep = vec![false; n];
        let mut is_sec = vec![false; n];
        let mut k = 0usize;
        let mut n_second = 0i32;

        // Two-pass: mark `keep[i]` using ORIGINAL parent scores. Compaction in the
        // second pass (via filtered_results below) so parent lookups never read a
        // slot that was overwritten by earlier iterations.
        for i in 0..n {
            let p = parent_state.parent[i];
            if p == i || results[i].inv {
                k += 1;
                keep[i] = true;
                if results[i].inv && p != i {
                    is_sec[i] = true;
                }
            } else {
                let (p_score, p_rev) = score_at[p];
                let filter_result = check_secondary_filter(
                    results[i].chain_score,
                    results[i].is_reverse,
                    p_score,
                    p_rev,
                    &filter_params,
                    false,
                );

                if filter_result.passes && n_second < opt.filtering.best_n {
                    let identical = results[i].query_start == results[p].query_start && results[i].query_end == results[p].query_end
                        && results[i].ref_id == results[p].ref_id && results[i].ref_start == results[p].ref_start && results[i].ref_end == results[p].ref_end;
                    if !identical {
                        k += 1;
                        n_second += 1;
                        keep[i] = true;
                        is_sec[i] = true;
                    }
                }
            }
        }

        // Build old-to-new index mapping for parent tracking
        let mut old_to_new: Vec<usize> = vec![0; n];
        {
            let mut new_idx = 0usize;
            for i in 0..n {
                if keep[i] {
                    old_to_new[i] = new_idx;
                    new_idx += 1;
                }
            }
        }
        parent_indices = Vec::with_capacity(k);
        for i in 0..n {
            if keep[i] {
                let old_parent = parent_state.parent[i];
                if keep[old_parent] {
                    parent_indices.push(old_to_new[old_parent]);
                } else {
                    parent_indices.push(old_to_new[i]); // self
                }
            }
        }

        let mut filtered_results = Vec::with_capacity(k);
        for (i, mut res) in results.into_iter().enumerate() {
            if keep[i] {
                res.is_secondary = is_sec[i];
                filtered_results.push(res);
            }
        }
        results = filtered_results;
        } // end else (non-ALL_CHAINS path)
    }

    (results, parent_indices)
}

/// Shared core of alignment pipeline: sort → align → filter → MAPQ.
fn process_query_core(
    opt: &MapOptions,
    mi: &Index,
    qseq: &[u8],
    ctx: &mut AlignmentContext,
    map_ctx: &mut MapContext,
    junc_db: Option<&JunctionDb>,
    out: &OutputConfig,
    map_result: MapResult,
) -> ProcessedQuery {
    let mut regs = map_result.regs;
    let rep_len = map_result.rep_len;
    let mut stats = map_result.stats;
    let squeezed = map_result.squeezed;
    let split_mode = out.split_mode;
    // Mark ALT contigs on chains before sorting
    let n_alt = mi.seqs.iter().filter(|s| s.is_alt).count();
    if n_alt > 0 {
        for r in regs.iter_mut() {
            if mi.seqs[r.ref_id].is_alt { r.is_alt = true; }
        }
    }

    // Sort by (score, hash) descending
    // When ALT contigs present, sort by alt-penalized scores
    if !regs.is_empty() {
        if n_alt > 0 {
            let alt_drop = opt.filtering.alt_drop;
            regs.sort_by(|a, b| {
                let sa = if a.is_alt { scale_alt_score(a.score, alt_drop) } else { a.score };
                let sb = if b.is_alt { scale_alt_score(b.score, alt_drop) } else { b.score };
                sb.cmp(&sa).then_with(|| b.hash.cmp(&a.hash))
            });
        } else {
            regs.sort_by(|a, b| b.score.cmp(&a.score)
                .then_with(|| b.hash.cmp(&a.hash)));
        }
    }

    let qlen = qseq.len();

    let mut results: Vec<AlnResult> = Vec::with_capacity(regs.len());
    let mut recalc_infos: Vec<DpRecalcInfo> = Vec::with_capacity(regs.len());

    // 1. Perform Extension and Calculation for all candidates.
    // Process splits inline to maintain adjacency.
    // Each reg is processed, its split is inserted right after it, then the loop
    // naturally processes the split next. After each right-split, inversion detection
    // checks the adjacent pair (left part at i-1, right part at i).
    let t_aln_start = Instant::now();
    let is_splice = opt.flags.contains(AlignFlags::SPLICE);
    let is_dual_splice = is_splice && opt.flags.contains(AlignFlags::SPLICE_FOR) && opt.flags.contains(AlignFlags::SPLICE_REV);

    // Build work queue: original regs first, splits get inserted at front.
    // The u8 is split-depth: 0 = original chain, 1 = direct child of an original,
    // 2+ = deeper-nested split-of-a-split. Used downstream to distinguish
    // primary split-children from secondary deep splits.
    let mut work_queue: std::collections::VecDeque<(Mapping, u8)> = std::collections::VecDeque::new();
    for r in regs.into_iter() {
        work_queue.push_back((r, 0));
    }

    // Encode the query once per outer call (forward + reverse-complement)
    // and pass references into `align_single_mapping`. Previously each
    // chain re-encoded the query inside `align_single_mapping` — catastrophic
    // at chromosome-scale qlens (a 248 Mb chr1 query × 679k chains ≈ 170 TB
    // of encoding work).
    let qseq_fwd_nt4: Vec<u8> = qseq.iter().map(|&b| encode_nt4(b)).collect();
    let qseq_rc_nt4: Vec<u8> = encode_nt4_rc(qseq);

    while let Some((mut r, parent_depth)) = work_queue.pop_front() {
        let is_split_part = parent_depth > 0;
        let (mut result, recalc_info, new_split) = if is_dual_splice {
            let base_flags = opt.flags & !(AlignFlags::SPLICE_FOR | AlignFlags::SPLICE_REV);
            let mut r_for = r.clone();
            let (result_for, recalc_for, split_for) = align_single_mapping(
                &mut r_for, opt, mi, qseq, &qseq_fwd_nt4, &qseq_rc_nt4, qlen, ctx, map_ctx, out, &squeezed, base_flags | AlignFlags::SPLICE_FOR, junc_db,
            );

            let is_sr_rna = opt.flags.contains(AlignFlags::SR_RNA);
            let sr_shortcut = is_sr_rna
                && (r.query_end - r.query_start) == (r.ref_end - r.ref_start)
                && (result_for.query_end - result_for.query_start) == (result_for.ref_end - result_for.ref_start)
                && result_for.query_start == 0 && result_for.query_end == qlen;
            if sr_shortcut {
                let mut res = result_for;
                res.trans_strand = 0;
                r = r_for;
                (res, recalc_for, split_for)
            } else {
                let mut r_rev = r.clone();
                let (result_rev, recalc_rev, split_rev) = align_single_mapping(
                    &mut r_rev, opt, mi, qseq, &qseq_fwd_nt4, &qseq_rc_nt4, qlen, ctx, map_ctx, out, &squeezed, base_flags | AlignFlags::SPLICE_REV, junc_db,
                );

                let (mut chosen_result, chosen_recalc, chosen_split, trans_strand) =
                    if result_for.align_score > result_rev.align_score {
                        (result_for, recalc_for, split_for, 1u8)
                    } else if result_for.align_score < result_rev.align_score {
                        (result_rev, recalc_rev, split_rev, 2u8)
                    } else {
                        let which = (qlen as i32 + result_for.align_score) & 1;
                        if which == 0 {
                            (result_for, recalc_for, split_for, 3u8)
                        } else {
                            (result_rev, recalc_rev, split_rev, 3u8)
                        }
                    };

                chosen_result.trans_strand = trans_strand;

                if chosen_result.is_spliced {
                    let bonus = opt.scoring.match_score + opt.scoring.mismatch_penalty;
                    if trans_strand == 1 || trans_strand == 2 {
                        chosen_result.dp_score += bonus + (bonus >> 1);
                    } else if trans_strand == 3 {
                        chosen_result.dp_score -= bonus;
                    }
                }

                r = if trans_strand == 2 || (trans_strand == 3 && (qlen as i32 + chosen_result.align_score) & 1 != 0) {
                    r_rev
                } else {
                    r_for
                };

                (chosen_result, chosen_recalc, chosen_split)
            }
        } else {
            let splice_f = opt.flags;
            let (mut res, rec, spl) = align_single_mapping(
                &mut r, opt, mi, qseq, &qseq_fwd_nt4, &qseq_rc_nt4, qlen, ctx, map_ctx, out, &squeezed, splice_f, junc_db,
            );
            if is_splice {
                res.trans_strand = if opt.flags.contains(AlignFlags::SPLICE_FOR) { 1 } else { 2 };
            }
            (res, rec, spl)
        };

        // Mark split parts; split_depth carries the chain's nesting level.
        if is_split_part {
            result.split |= 2;
            result.split_inv = r.split_inv;
        }
        result.split_depth = parent_depth;

        // If this result has a new split, mark it as left part and queue the split.
        // The new split's depth is parent_depth + 1 (one level deeper).
        if let Some(split) = new_split {
            result.split |= 1;
            let child_depth = parent_depth.saturating_add(1);
            work_queue.push_front((split, child_depth));
        }

        results.push(result);
        recalc_infos.push(recalc_info);

        // Inline inversion detection:
        // After inserting a right-split result, check if it and its predecessor
        // form a valid inversion pair. Parent check is inside try_align_inversion() itself,
        // not at the call site.
        if results.len() >= 2 && out.do_cigar && !opt.flags.contains(AlignFlags::NO_INV) {
            let idx = results.len() - 1;
            if results[idx].split_inv && let Some(inv_result) = try_align_inversion(opt, mi, qlen, &qseq_fwd_nt4, &qseq_rc_nt4, &results[idx - 1], &results[idx], out) {
                // Build recalc_info from the inversion alignment stats (mlen/blen/n_ambi come from per-base comparison in CigarStats; gap stats come from CIGAR ops)
                let mut inv_recalc = DpRecalcInfo::from_cigar_str(&inv_result.cigar_str);
                inv_recalc.match_len = inv_result.matches as i32;
                inv_recalc.block_len = inv_result.block_len as i32;
                inv_recalc.num_ambiguous = inv_result.num_ambiguous as i32;
                results.push(inv_result);
                recalc_infos.push(inv_recalc);
            }
        }
    }

    stats.t_align = t_aln_start.elapsed();

    // Post-alignment filtering
    if out.do_cigar {
        filter_alignment_results(&mut results, Some(&mut recalc_infos), opt.chaining.min_cnt, opt.chaining.min_chain_score, opt.alignment.min_dp_max, opt.alignment.max_clip_ratio, qlen);
    }

    // Parent assignment, secondary selection, dp_max ranking
    let (results, parent_indices) = assign_parents_and_select(results, &recalc_infos, opt, mi, out, split_mode, qlen);

    // Set SAM primary flag (first non-secondary result)
    let mut sam_pri = vec![false; results.len()];
    {
        let mut n_pri = 0;
        for (i, r) in results.iter().enumerate() {
            if !r.is_secondary {
                n_pri += 1;
                sam_pri[i] = n_pri == 1;
            }
        }
    }

    let mapqs = compute_mapping_qualities(&results, &parent_indices, opt, rep_len as f32);

    ProcessedQuery { results, mapqs, sam_pri, parent_indices, rep_len: rep_len as i32, stats }
}

/// Re-run post-alignment filtering and MAPQ on merged results from split index.
/// Pipeline: update_dp_max -> reset dp_max2/subsc/n_sub -> sort -> assign parents ->
/// select secondaries -> set SAM primary -> compute MAPQ.
pub fn refilter_merged_results(
    mut results: Vec<AlnResult>,
    opt: &MapOptions,
    k: usize,
    qlen: usize,
    rep_len: i32,
    do_cigar: bool,
) -> ProcessedQuery {
    let stats = AlignmentStats::default();

    if results.is_empty() {
        return ProcessedQuery { results, mapqs: vec![], sam_pri: vec![], parent_indices: vec![], rep_len, stats };
    }

    // 1. update_dp_max
    let is_sr = opt.flags.intersects(AlignFlags::SHORT_READ | AlignFlags::SR_RNA);
    if do_cigar && !is_sr && qlen >= 500 {
        let recalc_infos: Vec<DpRecalcInfo> = results.iter()
            .map(|r| DpRecalcInfo::from_cigar_str(&r.cigar_str))
            .collect();
        let mut dp_max_vals: Vec<i32> = results.iter().map(|r| r.dp_score).collect();
        let qs_vals: Vec<usize> = results.iter().map(|r| r.query_start).collect();
        let qe_vals: Vec<usize> = results.iter().map(|r| r.query_end).collect();
        update_dp_max(&mut dp_max_vals, &recalc_infos, &qs_vals, &qe_vals, qlen, 0.9, opt.scoring.match_score, opt.scoring.mismatch_penalty);
        for (i, r) in results.iter_mut().enumerate() {
            r.dp_score = dp_max_vals[i];
        }
    }

    // 2. Reset dp_max2/subsc/n_sub
    for r in results.iter_mut() {
        r.dp_score_secondary = 0;
        r.secondary_chain_score = 0;
        r.num_suboptimal = 0;
    }

    // 3. Sort by (dp_max, hash) descending
    // Note: merge does NOT re-run post-alignment filtering. The per-part
    // filtering already ran during alignment. Sort only filters cnt==0.
    // Our per-part results already survived filtering, so no cnt==0 filter needed.
    let mut parent_indices: Vec<usize> = (0..results.len()).collect();

    if !results.is_empty() {
        if opt.flags.contains(AlignFlags::ALL_CHAINS) {
            for r in results.iter_mut() {
                r.is_secondary = true;
            }
            parent_indices = vec![usize::MAX; results.len()];
        } else {
            // Sort by (dp_score, hash) descending — same hit-sort routine as
            // the earlier site. See comment there for the unstable-tie behavior.
            {
                let alt_drop = opt.filtering.alt_drop;
                let mut aux: Vec<(u64, u64)> = results.iter().enumerate().map(|(i, r)| {
                    let score = if r.is_alt { scale_alt_score(r.dp_score, alt_drop) } else { r.dp_score };
                    let key = ((score as u64) << 32) | (r.hash as u64);
                    (key, i as u64)
                }).collect();
                radix_sort_128x_pair(&mut aux);
                let mut opt_results: Vec<Option<AlnResult>> = results.into_iter().map(Some).collect();
                results = aux.iter().rev().map(|&(_, i)| opt_results[i as usize].take().unwrap()).collect();
            }

            // 5. Assign parents
            let filter_items: Vec<FilterableItem> = results.iter()
                .map(|r| FilterableItem {
                    query_start: r.query_start,
                    query_end: r.query_end,
                    score: r.chain_score,
                    is_reverse: r.is_reverse,
                    is_alt: r.is_alt,
                })
                .collect();

            let filter_params = FilterParams::new(opt, &Index::header_only(k, 0, false, vec![]));
            let mut parent_state = ParentState::new(filter_items.len(), filter_params.mask_level, filter_params.mask_len, filter_params.hard_mask_level);
            parent_state.init_from_items(&filter_items);
            parent_state.assign_parents(&filter_items);

            let sub_diff = opt.scoring.match_score * 2 + opt.scoring.mismatch_penalty;
            let mut dp_max2 = vec![0i32; results.len()];
            let mut subsc_vec: Vec<i32> = results.iter()
                .map(|r| r.s2_score.unwrap_or(0))
                .collect();
            let mut n_sub_post = vec![0i32; results.len()];
            // In merge, n_sub_post[0] starts at 0 (reset above), NOT pre_num_suboptimal
            let alt_drop = opt.filtering.alt_drop;
            for i in 0..results.len() {
                let pi = parent_state.parent[i];
                if pi != i {
                    // ALT penalty for subsc
                    let sci = if !results[pi].is_alt && results[i].is_alt {
                        scale_alt_score(results[i].chain_score, alt_drop)
                    } else {
                        results[i].chain_score
                    };
                    if sci > subsc_vec[pi] {
                        subsc_vec[pi] = sci;
                    }
                    let mut cnt_sub = false;
                    if results[i].anchor_count >= results[pi].anchor_count {
                        cnt_sub = true;
                    }
                    let (si, ei) = (results[i].query_start, results[i].query_end);
                    let (sj, ej) = (results[pi].query_start, results[pi].query_end);
                    let ol = {
                        let start = si.max(sj);
                        let end = ei.min(ej);
                        end.saturating_sub(start)
                    };
                    let min_len = (ei - si).min(ej - sj);
                    let not_identical = results[i].ref_id != results[pi].ref_id
                        || results[i].ref_start != results[pi].ref_start
                        || results[i].ref_end != results[pi].ref_end
                        || ol != min_len;
                    if not_identical {
                        // ALT penalty for dp_max2
                        let dp_sci = if !results[pi].is_alt && results[i].is_alt {
                            scale_alt_score(results[i].dp_score, alt_drop)
                        } else {
                            results[i].dp_score
                        };
                        if dp_sci > dp_max2[pi] {
                            dp_max2[pi] = dp_sci;
                        }
                        if results[pi].dp_score - results[i].dp_score <= sub_diff {
                            cnt_sub = true;
                        }
                    }
                    if cnt_sub {
                        n_sub_post[pi] += 1;
                    }
                }
            }

            for i in 0..results.len() {
                results[i].dp_score_secondary = dp_max2[i];
                results[i].secondary_chain_score = subsc_vec[i];
                results[i].num_suboptimal = n_sub_post[i];
            }

            // 6. Select secondaries
            let n = results.len();
            let mut keep = vec![false; n];
            let mut is_sec = vec![false; n];
            let mut k = 0usize;
            let mut n_second = 0i32;

            for i in 0..n {
                let p = parent_state.parent[i];
                if p == i || results[i].inv {
                    // Primary or inversion: always keep
                    k += 1;
                    keep[i] = true;
                    if results[i].inv && p != i {
                        is_sec[i] = true;
                    }
                } else {
                    let p_score = results[p].chain_score;
                    let p_rev = results[p].is_reverse;
                    let filter_result = check_secondary_filter(
                        results[i].chain_score,
                        results[i].is_reverse,
                        p_score,
                        p_rev,
                        &filter_params,
                        false,
                    );
                    if filter_result.passes && n_second < opt.filtering.best_n {
                        let identical = results[i].query_start == results[p].query_start && results[i].query_end == results[p].query_end
                            && results[i].ref_id == results[p].ref_id && results[i].ref_start == results[p].ref_start && results[i].ref_end == results[p].ref_end;
                        if !identical {
                            k += 1;
                            n_second += 1;
                            keep[i] = true;
                            is_sec[i] = true;
                        }
                    }
                }
            }

            // Build old-to-new index mapping
            let mut old_to_new: Vec<usize> = vec![0; n];
            {
                let mut new_idx = 0usize;
                for i in 0..n {
                    if keep[i] {
                        old_to_new[i] = new_idx;
                        new_idx += 1;
                    }
                }
            }
            parent_indices = Vec::with_capacity(k);
            for i in 0..n {
                if keep[i] {
                    let old_parent = parent_state.parent[i];
                    if keep[old_parent] {
                        parent_indices.push(old_to_new[old_parent]);
                    } else {
                        parent_indices.push(old_to_new[i]);
                    }
                }
            }

            let mut filtered_results = Vec::with_capacity(k);
            for (i, mut res) in results.into_iter().enumerate() {
                if keep[i] {
                    res.is_secondary = is_sec[i];
                    filtered_results.push(res);
                }
            }
            results = filtered_results;
        }
    }

    // 7. Set SAM primary flag
    let mut sam_pri = vec![false; results.len()];
    {
        let mut n_pri = 0;
        for (i, r) in results.iter().enumerate() {
            if !r.is_secondary {
                n_pri += 1;
                sam_pri[i] = n_pri == 1;
            }
        }
    }

    // 8. Compute MAPQ
    let mapqs = compute_mapping_qualities(&results, &parent_indices, opt, rep_len as f32);

    ProcessedQuery { results, mapqs, sam_pri, parent_indices, rep_len, stats }
}

pub fn align_and_format_query(
    opt: &MapOptions,
    mi: &Index,
    read: &ReadInfo,
    ctx: &mut AlignmentContext,
    map_ctx: &mut MapContext,
    junc_db: Option<&JunctionDb>,
    jump_db: Option<&JumpDb>,
    out: &OutputConfig,
) -> (String, AlignmentStats) {
    let qname = read.qname;
    let qseq = read.qseq;
    let mut pq = process_query(opt, mi, qname, qseq, ctx, map_ctx, junc_db, out);

    // Jump splice extension: after alignment, for single-segment splice mode
    if let Some(jdb) = jump_db {
        let is_splice = opt.flags.contains(AlignFlags::SPLICE);
        if is_splice {
            let qlen = qseq.len();
            for r in pq.results.iter_mut() {
                crate::align::jump::jump_split(mi, opt, qlen, qseq, r, jdb);
            }
        }
    }

    let mut output_buffer = String::new();
    format_output(&mut output_buffer, opt, mi, read, &pq, out, None);
    (output_buffer, pq.stats)
}

/// Mate info for PE SAM output.
pub struct MateInfo {
    pub ref_id: Option<usize>,  // None if mate unmapped
    pub ref_start: usize,
    pub ref_end: usize,
    pub is_reverse: bool,
}

/// Trim /1 or /2 suffix from QNAME.
fn trim_read_name_suffix(name: &str) -> &str {
    let b = name.as_bytes();
    let l = b.len();
    if l >= 3 && b[l - 1].is_ascii_digit() && b[l - 2] == b'/' {
        &name[..l - 2]
    } else {
        name
    }
}

/// Compare read names ignoring /1 /2 suffix.
pub fn qname_same(s1: &str, s2: &str) -> bool {
    let t1 = trim_read_name_suffix(s1);
    let t2 = trim_read_name_suffix(s2);
    t1 == t2
}

/// Write junction BED output for a single alignment result.
/// Outputs one line per N_SKIP (intron) in the CIGAR: chr\tstart\tend\tqname\tscore\tstrand
fn write_junction_bed(output_buffer: &mut String, mi: &Index, qname: &str, r: &AlnResult) {
    if !r.is_spliced || r.cigar_str.is_empty() { return; }
    if r.trans_strand != 1 && r.trans_strand != 2 { return; }
    // Parse CIGAR string for N_SKIP operations
    let mut t_off = r.ref_start;
    let rid = r.ref_id;
    let tname = &mi.seqs[rid].name;
    let mut first = true;
    let cigar_bytes = r.cigar_str.as_bytes();
    let mut num_start = 0;
    for (i, &b) in cigar_bytes.iter().enumerate() {
        if b.is_ascii_digit() { continue; }
        let len: usize = std::str::from_utf8(&cigar_bytes[num_start..i]).unwrap_or("0").parse().unwrap_or(0);
        num_start = i + 1;
        match b {
            b'M' | b'=' | b'X' | b'D' => { t_off += len; }
            b'N' => {
                // Intron: score donor/acceptor splice sites
                if len < 2 { t_off += len; continue; }
                let rev = (r.trans_strand == 2) ^ r.is_reverse;
                let (d0, d1, a0, a1) = if !rev {
                    // Forward: donor = start, acceptor = end
                    let d0 = mi.get_nt4(rid, t_off);
                    let d1 = mi.get_nt4(rid, t_off + 1);
                    let a0 = mi.get_nt4(rid, t_off + len - 2);
                    let a1 = mi.get_nt4(rid, t_off + len - 1);
                    (d0, d1, a0, a1)
                } else {
                    // Reverse: swap + revcomp
                    let ra0 = mi.get_nt4(rid, t_off);
                    let ra1 = mi.get_nt4(rid, t_off + 1);
                    let rd0 = mi.get_nt4(rid, t_off + len - 2);
                    let rd1 = mi.get_nt4(rid, t_off + len - 1);
                    // revcomp_splice: swap + complement
                    let d0 = if rd1 < 4 { 3 - rd1 } else { 4 };
                    let d1 = if rd0 < 4 { 3 - rd0 } else { 4 };
                    let a0 = if ra1 < 4 { 3 - ra1 } else { 4 };
                    let a1 = if ra0 < 4 { 3 - ra0 } else { 4 };
                    (d0, d1, a0, a1)
                };
                let mut score1 = 0i32;
                let mut score2 = 0i32;
                // GT=3, GC=2, AT=1 for donor
                if d0 == 2 && d1 == 3 { score1 = 3; }
                else if d0 == 2 && d1 == 1 { score1 = 2; }
                else if d0 == 0 && d1 == 3 { score1 = 1; }
                // AG=3, AC=1 for acceptor
                if a0 == 0 && a1 == 2 { score2 = 3; }
                else if a0 == 0 && a1 == 1 { score2 = 1; }
                if !first { output_buffer.push('\n'); } else { first = false; }
                use std::fmt::Write;
                write!(output_buffer, "{}\t{}\t{}\t{}\t{}\t{}", tname, t_off, t_off + len, qname, score1 + score2, if rev { '-' } else { '+' }).ok();
                t_off += len;
            }
            b'I' | b'S' | b'H' => { /* query-consuming or hard clip, no ref movement */ }
            _ => {}
        }
    }
    if !first { output_buffer.push('\n'); }
}

/// Format an unmapped record (SAM unmapped or PAF no-hit) when results are empty.
fn format_unmapped_record(
    output_buffer: &mut String,
    opt: &MapOptions,
    mi: &Index,
    out: &OutputConfig,
    read: &ReadInfo,
    rep_len: i32,
    mate_info: Option<&MateInfo>,
) {
    let qname = read.qname;
    let qseq = read.qseq;
    let qual = read.qual;
    let qlen = qseq.len();
    let n_seg = read.n_seg;
    let seg_idx = read.seg_idx;
    let comment = read.comment;
    // Derive display names from read info (same logic as format_output)
    let out_qname: &str = if n_seg > 1 { trim_read_name_suffix(qname) } else { qname };
    let paf_qname_buf: String = if n_seg >= 2 && opt.flags.contains(AlignFlags::FRAG_MODE) {
        format!("{}/{}", qname, seg_idx + 1)
    } else {
        String::new()
    };
    let paf_qname: &str = if !paf_qname_buf.is_empty() { &paf_qname_buf } else { qname };
    if out.output_sam && !opt.flags.contains(AlignFlags::SAM_HIT_ONLY) {
        let qual_str = if opt.flags.contains(AlignFlags::NO_QUAL) { "*" } else { qual.unwrap_or("*") };
        // PE unmapped: set PE flags, and if mate mapped, use mate's position
        if n_seg > 1 {
            let mut flag: u32 = 0x5; // 0x1 (paired) | 0x4 (unmapped)
            if seg_idx == 0 { flag |= 0x40; }
            else if seg_idx == n_seg - 1 { flag |= 0x80; }
            if let Some(mi_info) = mate_info {
                if mi_info.is_reverse { flag |= 0x20; }
                if let Some(mate_rid) = mi_info.ref_id {
                    // Place unmapped read at mate's position
                    let mate_tname = &mi.seqs[mate_rid].name;
                    write!(output_buffer, "{}\t{}\t{}\t{}\t0\t*\t", out_qname, flag, mate_tname, mi_info.ref_start + 1).ok();
                    // RNEXT/PNEXT for mate
                    write!(output_buffer, "=\t{}\t0\t", mi_info.ref_start + 1).ok();
                } else {
                    flag |= 0x8; // mate unmapped
                    write!(output_buffer, "{}\t{}\t*\t0\t0\t*\t*\t0\t0\t", out_qname, flag).ok();
                }
            } else {
                flag |= 0x8;
                write!(output_buffer, "{}\t{}\t*\t0\t0\t*\t*\t0\t0\t", out_qname, flag).ok();
            }
            let seq_str = String::from_utf8_lossy(qseq);
            write!(output_buffer, "{}\t{}\trl:i:{}", seq_str, qual_str, rep_len).ok();
            if let Some(c) = comment { write!(output_buffer, "\t{}", c).ok(); }
            writeln!(output_buffer).ok();
        } else {
            write!(output_buffer, "{}\t4\t*\t0\t0\t*\t*\t0\t0\t{}\t{}\trl:i:{}", out_qname, String::from_utf8_lossy(qseq), qual_str, rep_len).ok();
            if let Some(c) = comment { write!(output_buffer, "\t{}", c).ok(); }
            writeln!(output_buffer).ok();
        }
    } else if !out.output_sam && opt.flags.contains(AlignFlags::PAF_NO_HIT) {
        writeln!(output_buffer, "{}\t{}\t0\t0\t*\t*\t0\t0\t0\t0\t0\t0", paf_qname, qlen).ok();
    }
}

/// Format a single mapped SAM record.
/// `idx` is the index of this result in the results array (needed for SA tag).
/// Count CIGAR operations in a run-length-encoded CIGAR string like "10M2I5D".
/// Counts one op per run (each `<digits><letter>` chunk).
fn count_cigar_ops(cigar: &str) -> usize {
    cigar.bytes().filter(|&b| !b.is_ascii_digit()).count()
}

/// Map a CIGAR op letter to its BAM op code.
#[inline]
fn bam_op_code(op: u8) -> u32 {
    match op {
        b'M' => 0, b'I' => 1, b'D' => 2, b'N' => 3,
        b'S' => 4, b'H' => 5, b'P' => 6, b'=' => 7, b'X' => 8,
        _ => 0,
    }
}

/// Encode a run-length CIGAR string as comma-separated BAM uint32 values
/// (`,<len<<4 | op>,...`) appended to `out`. Does NOT emit soft/hard clip ops —
/// the caller adds those.
fn write_cigar_as_bam_uints(out: &mut String, cigar: &str) {
    let bytes = cigar.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut len: u32 = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            len = len * 10 + (bytes[i] - b'0') as u32;
            i += 1;
        }
        if i >= bytes.len() { break; }
        let op = bytes[i];
        i += 1;
        write!(out, ",{}", (len << 4) | bam_op_code(op)).ok();
    }
}

fn format_sam_record(
    output_buffer: &mut String,
    r: &AlnResult,
    idx: usize,
    results: &[AlnResult],
    mapqs: &[i32],
    sam_pri_i: bool,
    tp_tag: char,
    s1_val: i32,
    cm_val: usize,
    opt: &MapOptions,
    mi: &Index,
    out: &OutputConfig,
    out_qname: &str,
    qseq: &[u8],
    qual: Option<&str>,
    qlen: usize,
    rep_len: i32,
    n_seg: usize,
    seg_idx: usize,
    mate_info: Option<&MateInfo>,
    rg_id: Option<&str>,
    comment: Option<&str>,
) {
    let tname = &mi.seqs[r.ref_id].name;

    // Flag assignment
    let mut flag: u32 = if r.is_reverse { 16 } else { 0 };
    if r.is_secondary {
        flag |= 0x100; // secondary
    } else if !sam_pri_i {
        flag |= 0x800; // supplementary
    }
    // PE flags
    if n_seg > 1 {
        flag |= 0x1; // paired
        if r.proper_frag { flag |= 0x2; } // proper pair
        if seg_idx == 0 { flag |= 0x40; } // first in pair
        else if seg_idx == n_seg - 1 { flag |= 0x80; } // last in pair
        if let Some(mi_info) = mate_info {
            if mi_info.ref_id.is_none() { flag |= 0x8; } // mate unmapped
            if mi_info.is_reverse { flag |= 0x20; } // mate reverse strand
        } else {
            flag |= 0x8; // no mate = mate unmapped
        }
    }

    let mapq = mapqs[idx];
    let mut full_cigar = String::new();

    let (clip_head, clip_tail) = if r.is_reverse {
        (qlen.saturating_sub(r.query_end), r.query_start)
    } else {
        (r.query_start, qlen.saturating_sub(r.query_end))
    };

    let clip_char = if (flag & 0x800 != 0 && !opt.flags.contains(AlignFlags::SOFTCLIP))
        || (flag & 0x100 != 0 && opt.flags.contains(AlignFlags::SECONDARY_SEQ)) { 'H' } else { 'S' };
    if clip_head > 0 { full_cigar.push_str(&format!("{}{}", clip_head, clip_char)); }
    full_cigar.push_str(&r.cigar_str);
    if clip_tail > 0 { full_cigar.push_str(&format!("{}{}", clip_tail, clip_char)); }

    // BAM n_cigar_op is a uint16. When -L (LONG_CIGAR) is set and the CIGAR would
    // exceed 65535 ops, move it into a CG:B:I tag and emit a placeholder <slen>S<rlen>N
    // in the CIGAR column so samtools view/convert can round-trip.
    const MAX_BAM_CIGAR_OP: usize = 65535;
    let long_cigar_enabled = opt.flags.contains(AlignFlags::LONG_CIGAR);
    let mid_op_count = count_cigar_ops(&r.cigar_str);
    let total_cigar_ops = mid_op_count
        + if clip_head > 0 { 1 } else { 0 }
        + if clip_tail > 0 { 1 } else { 0 };
    let cigar_in_tag = long_cigar_enabled
        && !r.cigar_str.is_empty()
        && total_cigar_ops > MAX_BAM_CIGAR_OP;

    let placeholder_cigar = if cigar_in_tag {
        let slen = if flag & 0x900 == 0 || opt.flags.contains(AlignFlags::SOFTCLIP) {
            qlen
        } else if flag & 0x100 != 0 && !opt.flags.contains(AlignFlags::SECONDARY_SEQ) {
            0
        } else {
            r.query_end - r.query_start
        };
        let rlen = r.ref_end - r.ref_start;
        Some(format!("{}S{}N", slen, rlen))
    } else {
        None
    };

    let no_qual = opt.flags.contains(AlignFlags::NO_QUAL);
    let sec_seq = opt.flags.contains(AlignFlags::SECONDARY_SEQ);
    let soft_supp = opt.flags.contains(AlignFlags::SOFTCLIP);
    let (out_seq, out_qual) = if flag & 0x900 == 0 || (flag & 0x800 != 0 && soft_supp) {
        let seq = if r.is_reverse {
            String::from_utf8(rev_comp(qseq)).unwrap_or_else(|_| "INVALID_UTF8".to_string())
        } else {
            String::from_utf8_lossy(qseq).to_string()
        };
        let q = if no_qual {
            "*".to_string()
        } else if let Some(qs) = qual {
            if r.is_reverse { qs.chars().rev().collect::<String>() } else { qs.to_string() }
        } else {
            "*".to_string()
        };
        (seq, q)
    } else if flag & 0x100 != 0 && !sec_seq {
        ("*".to_string(), "*".to_string())
    } else {
        let partial_seq = &qseq[r.query_start..r.query_end];
        let seq = if r.is_reverse {
            String::from_utf8(rev_comp(partial_seq)).unwrap_or_else(|_| "INVALID_UTF8".to_string())
        } else {
            String::from_utf8_lossy(partial_seq).to_string()
        };
        let q = if no_qual {
            "*".to_string()
        } else if let Some(qs) = qual {
            let partial_qual = &qs[r.query_start..r.query_end];
            if r.is_reverse { partial_qual.chars().rev().collect::<String>() } else { partial_qual.to_string() }
        } else {
            "*".to_string()
        };
        (seq, q)
    };

    // Write QNAME, FLAG, RNAME, POS, MAPQ, CIGAR
    let cigar_col: &str = placeholder_cigar.as_deref().unwrap_or(&full_cigar);
    write!(output_buffer, "{}\t{}\t{}\t{}\t{}\t{}\t",
        out_qname, flag, tname, r.ref_start + 1, mapq, cigar_col).ok();

    // Write RNEXT/PNEXT/TLEN
    if n_seg > 1 {
        let this_rid = r.ref_id as i64;
        let this_pos = r.ref_start;
        let mut tlen: i64 = 0;
        if let Some(mi_info) = mate_info {
            if let Some(mate_rid) = mi_info.ref_id {
                if this_rid == mate_rid as i64 {
                    // Same chromosome
                    let this_pos5 = if r.is_reverse { r.ref_end as i64 - 1 } else { this_pos as i64 };
                    let next_pos5 = if mi_info.is_reverse { mi_info.ref_end as i64 - 1 } else { mi_info.ref_start as i64 };
                    tlen = next_pos5 - this_pos5;
                    write!(output_buffer, "=\t").ok();
                } else {
                    write!(output_buffer, "{}\t", mi.seqs[mate_rid].name).ok();
                }
                write!(output_buffer, "{}\t", mi_info.ref_start + 1).ok();
            } else {
                // Mate unmapped but this is mapped: use own position
                write!(output_buffer, "=\t{}\t", this_pos + 1).ok();
            }
        } else {
            // Mate unmapped, this mapped: use own position
            write!(output_buffer, "=\t{}\t", this_pos + 1).ok();
        }
        if tlen > 0 { tlen += 1; } else if tlen < 0 { tlen -= 1; }
        write!(output_buffer, "{}\t", tlen).ok();
    } else {
        write!(output_buffer, "*\t0\t0\t").ok();
    }

    // Write SEQ/QUAL
    write!(output_buffer, "{}\t{}", out_seq, out_qual).ok();

    write!(output_buffer, "\tNM:i:{}", r.edit_distance).ok();
    write!(output_buffer, "\tms:i:{}", r.dp_score_original).ok();
    write!(output_buffer, "\tAS:i:{}", r.align_score).ok();
    write!(output_buffer, "\tnn:i:{}", r.num_ambiguous).ok();
    if r.trans_strand == 1 {
        write!(output_buffer, "\tts:A:+").ok();
    } else if r.trans_strand == 2 {
        write!(output_buffer, "\tts:A:-").ok();
    }
    write!(output_buffer, "\ttp:A:{}", tp_tag).ok();
    write!(output_buffer, "\tcm:i:{}", cm_val).ok();
    write!(output_buffer, "\ts1:i:{}", s1_val).ok();
    if !r.is_secondary {
        write!(output_buffer, "\ts2:i:{}", r.secondary_chain_score).ok();
    }
    if r.divergence == 0.0 {
        write!(output_buffer, "\tde:f:0").ok();
    } else {
        write!(output_buffer, "\tde:f:{:.4}", r.divergence).ok();
    }
    if r.split > 0 {
        write!(output_buffer, "\tzd:i:{}", r.split).ok();
    }
    // SA tag
    if !r.is_secondary && results.len() > 1 {
        let n_sa = results.iter().enumerate()
            .filter(|&(j, rj)| j != idx && !rj.is_secondary && !rj.cigar_str.is_empty())
            .count();
        if n_sa > 0 {
            write!(output_buffer, "\tSA:Z:").ok();
            for (j, rj) in results.iter().enumerate() {
                if j == idx || rj.is_secondary || rj.cigar_str.is_empty() { continue; }
                let l_m;
                let mut l_i = 0usize;
                let mut l_d = 0usize;
                let qspan = rj.query_end - rj.query_start;
                let rspan = rj.ref_end - rj.ref_start;
                if qspan < rspan {
                    l_m = qspan;
                    l_d = rspan - l_m;
                } else {
                    l_m = rspan;
                    l_i = qspan - l_m;
                }
                let clip5 = if rj.is_reverse { qlen - rj.query_end } else { rj.query_start };
                let clip3 = if rj.is_reverse { rj.query_start } else { qlen - rj.query_end };
                let strand = if rj.is_reverse { '-' } else { '+' };
                let sa_nm = rj.edit_distance;
                let sa_mapq = mapqs[j];

                write!(output_buffer, "{},{},{},", mi.seqs[rj.ref_id].name, rj.ref_start + 1, strand).ok();
                if clip5 > 0 { write!(output_buffer, "{}S", clip5).ok(); }
                if l_m > 0 { write!(output_buffer, "{}M", l_m).ok(); }
                if l_i > 0 { write!(output_buffer, "{}I", l_i).ok(); }
                if l_d > 0 { write!(output_buffer, "{}D", l_d).ok(); }
                if clip3 > 0 { write!(output_buffer, "{}S", clip3).ok(); }
                write!(output_buffer, ",{},{};", sa_mapq, sa_nm).ok();
            }
        }
    }

    if let Some(id) = rg_id {
        write!(output_buffer, "\tRG:Z:{}", id).ok();
    }
    // MD takes priority over CS/DS
    if out.do_md { write!(output_buffer, "\tMD:Z:{}", r.md_str).ok(); }
    else if out.do_ds { write!(output_buffer, "\tds:Z:{}", r.ds_str).ok(); }
    else if out.do_cs { write!(output_buffer, "\tcs:Z:{}", r.cs_str).ok(); }
    if cigar_in_tag {
        let clip_op_code: u32 = if clip_char == 'H' { 5 } else { 4 };
        write!(output_buffer, "\tCG:B:I").ok();
        if clip_head > 0 {
            write!(output_buffer, ",{}", ((clip_head as u32) << 4) | clip_op_code).ok();
        }
        write_cigar_as_bam_uints(output_buffer, &r.cigar_str);
        if clip_tail > 0 {
            write!(output_buffer, ",{}", ((clip_tail as u32) << 4) | clip_op_code).ok();
        }
    }
    write!(output_buffer, "\trl:i:{}", rep_len).ok();
    if let Some(c) = comment { write!(output_buffer, "\t{}", c).ok(); }
    writeln!(output_buffer).ok();
}

/// Format a single mapped PAF record.
fn format_paf_record(
    output_buffer: &mut String,
    r: &AlnResult,
    mapq: i32,
    tp_tag: char,
    s1_val: i32,
    cm_val: usize,
    opt: &MapOptions,
    mi: &Index,
    out: &OutputConfig,
    paf_qname: &str,
    qlen: usize,
    rep_len: i32,
    comment: Option<&str>,
) {
    let tname = &mi.seqs[r.ref_id].name;
    let tlen = mi.seqs[r.ref_id].len;
    let strand_char = if r.is_reverse { '-' } else { '+' };
    // qstrand mode: flip ref coords for rev-strand
    let (paf_rs, paf_re) = if opt.flags.contains(AlignFlags::QSTRAND) && r.is_reverse {
        (tlen - r.ref_end, tlen - r.ref_start)
    } else {
        (r.ref_start, r.ref_end)
    };
    write!(output_buffer, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        paf_qname, qlen, r.query_start, r.query_end, strand_char,
        tname, tlen, paf_rs, paf_re,
        r.matches, r.block_len, mapq).ok();

    if out.do_cigar {
        write!(output_buffer, "\tNM:i:{}", r.edit_distance).ok();
        write!(output_buffer, "\tms:i:{}", r.dp_score_original).ok();
        write!(output_buffer, "\tAS:i:{}", r.align_score).ok();
        write!(output_buffer, "\tnn:i:{}", r.num_ambiguous).ok();
        if r.trans_strand == 1 {
            write!(output_buffer, "\tts:A:+").ok();
        } else if r.trans_strand == 2 {
            write!(output_buffer, "\tts:A:-").ok();
        }
    }
    write!(output_buffer, "\ttp:A:{}", tp_tag).ok();
    write!(output_buffer, "\tcm:i:{}", cm_val).ok();
    write!(output_buffer, "\ts1:i:{}", s1_val).ok();
    if !r.is_secondary {
        write!(output_buffer, "\ts2:i:{}", r.secondary_chain_score).ok();
    }
    if out.do_cigar {
        if r.divergence == 0.0 {
            write!(output_buffer, "\tde:f:0").ok();
        } else {
            write!(output_buffer, "\tde:f:{:.4}", r.divergence).ok();
        }
    } else if r.div >= 0.0 && r.div <= 1.0 {
        if r.div == 0.0 {
            write!(output_buffer, "\tdv:f:0").ok();
        } else {
            write!(output_buffer, "\tdv:f:{:.4}", r.div).ok();
        }
    }
    if r.split > 0 {
        write!(output_buffer, "\tzd:i:{}", r.split).ok();
    }

    write!(output_buffer, "\trl:i:{}", rep_len).ok();
    if opt.flags.contains(AlignFlags::OUT_CIGAR) && !r.cigar_str.is_empty() {
        write!(output_buffer, "\tcg:Z:{}", r.cigar_str).ok();
    }
    // MD takes priority over CS/DS
    if out.do_md { write!(output_buffer, "\tMD:Z:{}", r.md_str).ok(); }
    else if out.do_ds { write!(output_buffer, "\tds:Z:{}", r.ds_str).ok(); }
    else if out.do_cs { write!(output_buffer, "\tcs:Z:{}", r.cs_str).ok(); }
    if let Some(c) = comment { write!(output_buffer, "\t{}", c).ok(); }
    writeln!(output_buffer).ok();
}

/// Format SAM/PAF output for a single segment.
/// n_seg=1 for single-read, n_seg=2 for PE. seg_idx=0/1 for PE read1/read2.
/// mate_info is the SAM primary of the other segment (if n_seg > 1).
pub fn format_output(
    output_buffer: &mut String,
    opt: &MapOptions,
    mi: &Index,
    read: &ReadInfo,
    pq: &ProcessedQuery,
    out: &OutputConfig,
    mate_info: Option<&MateInfo>,
) {
    let qname = read.qname;
    let qseq = read.qseq;
    let qual = read.qual;
    let comment = read.comment;
    let n_seg = read.n_seg;
    let seg_idx = read.seg_idx;
    let rg_id = out.rg_id.as_deref();
    let results = &pq.results;
    let mapqs = &pq.mapqs;
    let sam_pri = &pq.sam_pri;
    let rep_len = pq.rep_len;
    let qlen = qseq.len();
    // SAM: trim /1 /2 suffix. PAF: keep original name + append /{seg_idx+1} for PE.
    let sam_qname: &str = if n_seg > 1 { trim_read_name_suffix(qname) } else { qname };
    let paf_qname_buf: String = if n_seg >= 2 && opt.flags.contains(AlignFlags::FRAG_MODE) {
        format!("{}/{}", qname, seg_idx + 1)
    } else {
        String::new()
    };
    let paf_qname: &str = if !paf_qname_buf.is_empty() { &paf_qname_buf } else { qname };
    let out_qname = sam_qname; // default for SAM paths

    // Output unmapped record if no results
    if results.is_empty() {
        format_unmapped_record(output_buffer, opt, mi, out, read, rep_len, mate_info);
    }
    // --write-junc mode: output junction BED instead of SAM/PAF
    if opt.flags.contains(AlignFlags::OUT_JUNC) {
        let parent_indices = &pq.parent_indices;
        for (i, r) in results.iter().enumerate() {
            // Only root alignments (parent == self) with mapq >= 10
            if parent_indices[i] != i || mapqs[i] < 10 { continue; }
            write_junction_bed(output_buffer, mi, paf_qname, r);
        }
        return;
    }

    let skip_sec = opt.flags.contains(AlignFlags::NO_PRINT_2ND);

    for (i, r) in results.iter().enumerate() {
        if skip_sec && r.is_secondary { continue; }
        let tp_tag = if r.inv {
            if !r.is_secondary { 'I' } else { 'i' }
        } else if !r.is_secondary { 'P' } else { 'S' };
        let s1_val = r.chain_score;
        let cm_val = r.anchor_count;

        if out.output_sam {
            format_sam_record(
                output_buffer, r, i, results, mapqs, sam_pri[i],
                tp_tag, s1_val, cm_val,
                opt, mi, out, out_qname, qseq, qual, qlen, rep_len,
                n_seg, seg_idx, mate_info, rg_id, comment,
            );
        } else {
            format_paf_record(
                output_buffer, r, mapqs[i],
                tp_tag, s1_val, cm_val,
                opt, mi, out, paf_qname, qlen, rep_len, comment,
            );
        }
    }
}

/// Align and format a paired-end read pair (weak pairing path).
/// Pre-flips read2 per pe_ori, maps each independently, optionally runs pair assignment,
/// post-flips results, then formats PE SAM output for both reads.
pub fn align_and_format_pair(
    opt: &MapOptions,
    mi: &Index,
    read1: &ReadInfo,
    read2: &ReadInfo,
    ctx: &mut AlignmentContext,
    map_ctx: &mut MapContext,
    junc_db: Option<&JunctionDb>,
    out: &OutputConfig,
) -> (String, AlignmentStats) {
    // Group consecutive reads by base name (qname).
    // If names differ, each read is an independent n_seg=1 fragment.
    if !qname_same(read1.qname, read2.qname) {
        let s1 = ReadInfo { n_seg: 1, seg_idx: 0, ..*read1 };
        let s2 = ReadInfo { n_seg: 1, seg_idx: 0, ..*read2 };
        let pq1 = process_query(opt, mi, s1.qname, s1.qseq, ctx, map_ctx, junc_db, out);
        let pq2 = process_query(opt, mi, s2.qname, s2.qseq, ctx, map_ctx, junc_db, out);
        let mut buf = String::new();
        format_output(&mut buf, opt, mi, &s1, &pq1, out, None);
        format_output(&mut buf, opt, mi, &s2, &pq2, out, None);
        return (buf, pq1.stats + pq2.stats);
    }

    let qname1 = read1.qname;
    let qseq1 = read1.qseq;
    let qual1 = read1.qual;
    let comment1 = read1.comment;
    let qname2 = read2.qname;
    let qseq2 = read2.qseq;
    let qual2 = read2.qual;
    let comment2 = read2.comment;
    let pe_ori = opt.pairing.pe_ori;
    let mut output_buffer = String::new();

    // pe_ori flipping: rev-comp read2 before alignment
    // For FR (pe_ori=1): read2 (j=1) is rev-comped. pe_ori&1 = 1 for j==1.
    // For read1: j==0, pe_ori>>1&1 = 0 for pe_ori=1, so read1 is NOT flipped.
    let flip_r1 = (pe_ori >> 1) & 1 != 0;
    let flip_r2 = pe_ori & 1 != 0;

    // Flip sequence for alignment; qual is only used in output (original orientation)
    let qseq1_work = if flip_r1 { rev_comp(qseq1) } else { qseq1.to_vec() };
    let qseq2_work = if flip_r2 { rev_comp(qseq2) } else { qseq2.to_vec() };

    // Weak pairing per-segment chaining is only activated when CIGAR is requested;
    // without CIGAR it decays into joint (strong) chaining.
    let is_weak = opt.flags.contains(AlignFlags::WEAK_PAIRING) && out.do_cigar;
    let is_independ = opt.flags.contains(AlignFlags::INDEPEND_SEG);

    // frag_gap: gap parameter for pair assignment (strong path uses computed max_chain_gap_ref,
    // weak path uses opt.chaining.max_gap_ref)
    let (mut pq1, mut pq2, frag_gap, multi_stats) = if is_independ {
        // Fully independent mapping: each segment treated as a standalone single-end read.
        // No joint chaining, no pair rescoring — each read keeps its own rep_len and hash seed.
        let pq1 = process_query(opt, mi, qname1, &qseq1_work, ctx, map_ctx, junc_db, out);
        let pq2 = process_query(opt, mi, qname2, &qseq2_work, ctx, map_ctx, junc_db, out);
        (pq1, pq2, opt.chaining.max_gap_ref, AlignmentStats::default())
    } else if is_weak {
        // Weak pairing: map each read independently
        // The FIRST segment's qname is used for ALL segments. This affects read_hash
        // (via compute_read_hash), which propagates to chain hashes and pair tiebreaking.
        // Use qname1 for both to match.
        let mut pq1 = process_query(opt, mi, qname1, &qseq1_work, ctx, map_ctx, junc_db, out);
        let pq2 = process_query(opt, mi, qname1, &qseq2_work, ctx, map_ctx, junc_db, out);
        // Quirk: the mapping wrapper processes segments sequentially. rep_len is overwritten
        // each time, so after it returns, rep_len = seg1's (R2's) rep_len. The caller then
        // copies this value to ALL segments. MAPQ is already computed with per-read rep_len,
        // so this only affects the rl:i output tag.
        pq1.rep_len = pq2.rep_len;
        (pq1, pq2, opt.chaining.max_gap_ref, AlignmentStats::default())
    } else {
        // Strong pairing: combined minimizers, chaining, seg_gen, per-segment alignment
        let seqs: Vec<&[u8]> = vec![&qseq1_work, &qseq2_work];
        let qlens = vec![qseq1_work.len(), qseq2_work.len()];

        let multi = map_query_multi(opt, mi, qname1, &seqs, &qlens, map_ctx);
        let multi_stats = multi.stats;
        let shared_rep_len = multi.rep_len;
        let frag_gap = multi.frag_gap;

        let mut per_seg = multi.per_seg;
        let (regs2, _anchors2) = if per_seg.len() > 1 { per_seg.remove(1) } else { (Vec::new(), Vec::new()) };
        let (regs1, _anchors1) = if !per_seg.is_empty() { per_seg.remove(0) } else { (Vec::new(), Vec::new()) };

        let pq1 = process_query_from_regs(opt, mi, &qseq1_work, ctx, map_ctx, junc_db, out, regs1, shared_rep_len);
        let pq2 = process_query_from_regs(opt, mi, &qseq2_work, ctx, map_ctx, junc_db, out, regs2, shared_rep_len);
        (pq1, pq2, frag_gap, multi_stats)
    };

    // Call pair assignment when PE with CIGAR enabled
    // Always call pair for n_segs==2 && pe_ori>=0 && CIGAR mode.
    // Skipped under INDEPEND_SEG (--pairing no): reads are treated as unrelated.
    if out.do_cigar && opt.pairing.pe_ori >= 0 && !is_independ {
        let qlens = [qseq1_work.len() as i32, qseq2_work.len() as i32];
        let sub_diff = opt.scoring.match_score * 2 + opt.scoring.mismatch_penalty;

        // Convert results to PeReg arrays
        let mut pe_regs: [Vec<PeReg>; 2] = [Vec::new(), Vec::new()];
        for (s, pq) in [&pq1, &pq2].iter().enumerate() {
            for (i, r) in pq.results.iter().enumerate() {
                pe_regs[s].push(PeReg {
                    dp_score: r.dp_score,
                    ref_id: r.ref_id,
                    ref_start: r.ref_start,
                    ref_end: r.ref_end,
                    is_reverse: r.is_reverse,
                    hash: r.hash,
                    mapq: pq.mapqs[i],
                    id: i,
                    parent: pq.parent_indices[i],
                    sam_pri: pq.sam_pri[i],
                    proper_frag: r.proper_frag,
                });
            }
        }

        pair_alignments(frag_gap, opt.pairing.pe_bonus, sub_diff, opt.scoring.match_score, &qlens, &mut pe_regs);

        // Write back changes from pair assignment
        for (s, pq) in [&mut pq1, &mut pq2].iter_mut().enumerate() {
            for pr in &pe_regs[s] {
                let i = pr.id;
                if i < pq.results.len() {
                    pq.results[i].proper_frag = pr.proper_frag;
                    pq.mapqs[i] = pr.mapq;
                    pq.sam_pri[i] = pr.sam_pri;
                    // Update is_sec based on pair assignment's parent reassignment
                    pq.results[i].is_secondary = pr.parent != pr.id;
                }
            }
        }
    }

    // Post-flip: reverse coordinates back for flipped reads
    if flip_r1 {
        let qlen1 = qseq1.len();
        for r in pq1.results.iter_mut() {
            let t = r.query_start;
            r.query_start = qlen1 - r.query_end;
            r.query_end = qlen1 - t;
            r.is_reverse = !r.is_reverse;
            if r.trans_strand == 1 { r.trans_strand = 2; }
            else if r.trans_strand == 2 { r.trans_strand = 1; }
        }
    }
    if flip_r2 {
        let qlen2 = qseq2.len();
        for r in pq2.results.iter_mut() {
            let t = r.query_start;
            r.query_start = qlen2 - r.query_end;
            r.query_end = qlen2 - t;
            r.is_reverse = !r.is_reverse;
            if r.trans_strand == 1 { r.trans_strand = 2; }
            else if r.trans_strand == 2 { r.trans_strand = 1; }
        }
    }

    // Build mate info from SAM primary of each segment
    let mate1 = get_mate_info(&pq1);
    let mate2 = get_mate_info(&pq2);

    // Format output for both segments (using original sequences, not flipped)
    let out_read1 = ReadInfo { qname: qname1, qseq: qseq1, qual: qual1, comment: comment1, n_seg: 2, seg_idx: 0 };
    let out_read2 = ReadInfo { qname: qname2, qseq: qseq2, qual: qual2, comment: comment2, n_seg: 2, seg_idx: 1 };
    format_output(&mut output_buffer, opt, mi, &out_read1, &pq1, out, mate2.as_ref());
    format_output(&mut output_buffer, opt, mi, &out_read2, &pq2, out, mate1.as_ref());

    let stats = multi_stats + pq1.stats + pq2.stats;
    (output_buffer, stats)
}

/// Extract mate info from the SAM primary of a processed query.
pub fn get_mate_info(pq: &ProcessedQuery) -> Option<MateInfo> {
    for (i, sp) in pq.sam_pri.iter().enumerate() {
        if *sp {
            return Some(MateInfo {
                ref_id: Some(pq.results[i].ref_id),
                ref_start: pq.results[i].ref_start,
                ref_end: pq.results[i].ref_end,
                is_reverse: pq.results[i].is_reverse,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::index::Index;
    use crate::align::extend::AlignmentContext;

    #[test]
    fn test_count_cigar_ops() {
        assert_eq!(count_cigar_ops(""), 0);
        assert_eq!(count_cigar_ops("10M"), 1);
        assert_eq!(count_cigar_ops("10M2I5D"), 3);
        assert_eq!(count_cigar_ops("3=1X2=1I4="), 5);
        assert_eq!(count_cigar_ops("100M"), 1);
    }

    #[test]
    fn test_write_cigar_as_bam_uints() {
        let mut s = String::new();
        write_cigar_as_bam_uints(&mut s, "10M2I5D3N1S7=2X");
        // op codes: M=0, I=1, D=2, N=3, S=4, ==7, X=8
        // 10<<4|0=160, 2<<4|1=33, 5<<4|2=82, 3<<4|3=51, 1<<4|4=20, 7<<4|7=119, 2<<4|8=40
        assert_eq!(s, ",160,33,82,51,20,119,40");
    }

    #[test]
    fn test_align_and_format_simple() {
        let seq = "ACGTACGT";
        let idx = Index::build(vec![("ref1".to_string(), seq.as_bytes().to_vec())], 4, 3, false, 50000);

        let mut opt = MapOptions::default();
        opt.chaining.min_cnt = 1;
        opt.chaining.min_chain_score = 0;
        opt.alignment.min_dp_max = 0;
        let mut map_ctx = MapContext::new();
        let mut aln_ctx = AlignmentContext::new();

        let out_cfg = OutputConfig {
            do_cigar: true,
            do_cs: false,
            cs_long: false,
            do_md: false,
            do_ds: false,
            eqx: false,
            output_sam: true,
            rg_id: None,
            split_mode: false,
        };
        let read = ReadInfo {
            qname: "read1",
            qseq: seq.as_bytes(),
            qual: Some("ABCDEFGH"),
            comment: None,
            n_seg: 1,
            seg_idx: 0,
        };
        let (output, stats) = align_and_format_query(
            &opt,
            &idx,
            &read,
            &mut aln_ctx,
            &mut map_ctx,
            None, // junc_db
            None, // jump_db
            &out_cfg,
        );

        if !output.contains("ref1") {
            eprintln!("Stats: seeds={}, anchors={}, chains={}", stats.n_seeds, stats.n_anchors, stats.n_chains);
            eprintln!("Time: sketch={:?}, seed={:?}, chain={:?}", stats.t_sketch, stats.t_seed, stats.t_chain);
        }
        assert!(output.contains("read1"), "Output should contain qname: {}", output);
        assert!(output.contains("ref1"), "Output should contain rname: {}", output);
    }
}
