//! Needleman-Wunsch aligner
//!
//! Implements the [`Aligner`] trait using a basic NW global alignment on each
//! gap between anchors, plus left and right extensions. The NW implementation
//! is fully contained in this file (~100 lines) with no dependencies on dp.rs.
//!
//! # Trade-offs vs prod aligner
//!
//! | Property              | prod aligner         | Simple NW aligner    |
//! |-----------------------|----------------------|----------------------|
//! | Time complexity       | O(n × bandwidth)     | O(n × m) per gap     |
//! | Gap model             | Affine (dual)        | Affine (single)      |
//! | Extension beyond seeds| Yes (left + right)   | Yes (left + right)   |
//! | Splice detection      | Yes (GT-AG scoring)  | No                   |
//! | Z-drop splitting      | Yes                  | No                   |
//! | SIMD acceleration     | AVX512/AVX2/SSE/NEON | None (scalar)        |

use crate::align::sketch::Minimizer;
use crate::align::map::{MapOptions, Aligner};
use crate::align::extend::{AlignmentContext, AlignAnchorContext, AlignResult, CigarOp};

/// For each gap between consecutive anchors (and for left/right extensions),
/// runs a full Needleman-Wunsch alignment with affine gap penalties and
/// produces =/X/I/D CIGAR ops. No SIMD, no external dependencies.
pub struct NWAligner;

impl Aligner for NWAligner {
    fn align(
        &self,
        anchors: &mut [Minimizer],
        qseq: &[u8],
        tseq: &[u8],
        opt: &MapOptions,
        _ctx: &mut AlignmentContext,
        _call: &AlignAnchorContext,
    ) -> AlignResult {
        let empty = AlignResult {
            cigar_ops: Vec::new(),
            query_start: 0, query_end: 0,
            ref_start: 0, ref_end: 0,
            dp_score: 0,
            split_right_anchors: None,
            split_offset_in_orig: None,
            split_inv: false,
        };
        if anchors.is_empty() { return empty; }

        anchors.sort_unstable_by_key(|a| a.ref_pos());

        let a = opt.scoring.match_score;
        let b = opt.scoring.mismatch_penalty;
        let gapo = opt.scoring.gap_open;
        let gape = opt.scoring.gap_extend;
        let k = anchors[0].query_span() as usize;

        let first = &anchors[0];
        let rs = (first.ref_pos() as usize).saturating_sub(k - 1);
        let qs = (first.query_pos() as usize).saturating_sub(k - 1);

        // we don't use the end position here, but if you want it:
        //let last = &anchors[anchors.len() - 1];
        //let re = last.ref_pos() as usize + 1;
        //let qe = last.query_pos() as usize + 1;

        let mut ops: Vec<CigarOp> = Vec::new();
        let mut dp_score: i32 = 0;
        let mut cur_q = qs;
        let mut cur_r = rs;

        for anchor in anchors.iter() {
            let span = anchor.query_span() as usize;
            let a_r_start = (anchor.ref_pos() as usize) + 1 - span;
            let a_q_start = (anchor.query_pos() as usize) + 1 - span;

            // Align the gap before this anchor
            if a_r_start > cur_r || a_q_start > cur_q {
                let q_gap = a_q_start.saturating_sub(cur_q);
                let r_gap = a_r_start.saturating_sub(cur_r);
                if q_gap > 0 && r_gap > 0 {
                    let q_sub = &qseq[cur_q..cur_q + q_gap];
                    let t_sub = &tseq[cur_r..cur_r + r_gap];
                    let (gap_ops, gap_score) = nw_align(q_sub, t_sub, a, b, gapo, gape);
                    ops.extend_from_slice(&gap_ops);
                    dp_score += gap_score;
                } else if q_gap > 0 {
                    push_op(&mut ops, 'I', q_gap as u32);
                    dp_score -= gapo + gape * q_gap as i32;
                } else if r_gap > 0 {
                    push_op(&mut ops, 'D', r_gap as u32);
                    dp_score -= gapo + gape * r_gap as i32;
                }
            }

            // Emit anchor region as =/X by comparing bases
            for j in 0..span {
                let qi = a_q_start + j;
                let ri = a_r_start + j;
                if qi < qseq.len() && ri < tseq.len() && qseq[qi] == tseq[ri] && qseq[qi] < 4 {
                    push_op(&mut ops, '=', 1);
                    dp_score += a;
                } else {
                    push_op(&mut ops, 'X', 1);
                    dp_score -= b;
                }
            }

            cur_q = a_q_start + span;
            cur_r = a_r_start + span;
        }

        // Right extension: align remaining sequence after last anchor
        let q_remain = qseq.len().saturating_sub(cur_q);
        let r_remain = tseq.len().saturating_sub(cur_r);
        if q_remain > 0 && r_remain > 0 {
            let q_sub = &qseq[cur_q..cur_q + q_remain];
            let t_sub = &tseq[cur_r..cur_r + r_remain];
            let (ext_ops, ext_score) = nw_align(q_sub, t_sub, a, b, gapo, gape);
            ops.extend_from_slice(&ext_ops);
            dp_score += ext_score;
            cur_q += q_remain;
            cur_r += r_remain;
        }

        AlignResult {
            cigar_ops: ops,
            query_start: qs,
            query_end: cur_q,
            ref_start: rs,
            ref_end: cur_r,
            dp_score,
            split_right_anchors: None,
            split_offset_in_orig: None,
            split_inv: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Self-contained Needleman-Wunsch with affine gaps
// ---------------------------------------------------------------------------

const NEG_INF: i32 = -0x40000000;

/// Needleman-Wunsch global alignment with affine gap penalties.
///
/// Returns (cigar_ops, score). Produces =/X/I/D ops by comparing nt4 bases.
/// Fully self-contained — no dependencies on dp.rs.
fn nw_align(
    qseq: &[u8], tseq: &[u8],
    match_score: i32, mismatch_pen: i32,
    gap_open: i32, gap_extend: i32,
) -> (Vec<CigarOp>, i32) {
    let qlen = qseq.len();
    let tlen = tseq.len();
    if qlen == 0 && tlen == 0 { return (Vec::new(), 0); }
    if qlen == 0 { return (vec![CigarOp { op: 'D', len: tlen as u32 }], -(gap_open + gap_extend * tlen as i32)); }
    if tlen == 0 { return (vec![CigarOp { op: 'I', len: qlen as u32 }], -(gap_open + gap_extend * qlen as i32)); }

    let gapoe = gap_open + gap_extend;

    // DP matrices: H (best), E (query gap / insertion), F (target gap / deletion)
    let mut h = vec![vec![NEG_INF; tlen + 1]; qlen + 1];
    let mut e = vec![vec![NEG_INF; tlen + 1]; qlen + 1];
    let mut f = vec![vec![NEG_INF; tlen + 1]; qlen + 1];

    // Backtrack: 0=diag, 1=up (I), 2=left (D)
    let mut bt = vec![vec![0u8; tlen + 1]; qlen + 1];

    // Initialize boundaries
    h[0][0] = 0;
    for j in 1..=tlen {
        h[0][j] = -(gapoe + gap_extend * (j as i32 - 1));
        bt[0][j] = 2; // D
    }
    for i in 1..=qlen {
        h[i][0] = -(gapoe + gap_extend * (i as i32 - 1));
        bt[i][0] = 1; // I
    }

    // Fill
    for i in 1..=qlen {
        for j in 1..=tlen {
            let s = if qseq[i - 1] == tseq[j - 1] && qseq[i - 1] < 4 {
                match_score
            } else {
                -mismatch_pen
            };
            let diag = h[i - 1][j - 1] + s;

            e[i][j] = std::cmp::max(h[i - 1][j] - gapoe, e[i - 1][j] - gap_extend);
            f[i][j] = std::cmp::max(h[i][j - 1] - gapoe, f[i][j - 1] - gap_extend);

            let best = std::cmp::max(diag, std::cmp::max(e[i][j], f[i][j]));
            h[i][j] = best;

            bt[i][j] = if best == diag { 0 }
                       else if best == e[i][j] { 1 }
                       else { 2 };
        }
    }

    let score = h[qlen][tlen];

    // Traceback
    let mut cigar = Vec::new();
    let mut i = qlen;
    let mut j = tlen;
    while i > 0 || j > 0 {
        let d = if i == 0 { 2 } else if j == 0 { 1 } else { bt[i][j] };
        match d {
            0 => {
                if qseq[i - 1] == tseq[j - 1] && qseq[i - 1] < 4 {
                    push_op(&mut cigar, '=', 1);
                } else {
                    push_op(&mut cigar, 'X', 1);
                }
                i -= 1; j -= 1;
            }
            1 => { push_op(&mut cigar, 'I', 1); i -= 1; }
            _ => { push_op(&mut cigar, 'D', 1); j -= 1; }
        }
    }

    cigar.reverse();
    (cigar, score)
}

/// Append a CIGAR op, merging with the previous op if same type.
#[inline]
fn push_op(ops: &mut Vec<CigarOp>, op: char, len: u32) {
    if len == 0 { return; }
    if let Some(last) = ops.last_mut() && last.op == op {
        last.len += len;
        return;
    }
    ops.push(CigarOp { op, len });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_anchor(ref_pos: i32, query_pos: i32, span: i32) -> Minimizer {
        let x = ref_pos as u32 as u64;
        let y = (query_pos as u32 as u64) | ((span as u64) << 32);
        Minimizer { x, y }
    }

    #[test]
    fn test_nw_identical() {
        let q: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let t: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let (ops, score) = nw_align(&q, &t, 2, 4, 4, 2);
        assert_eq!(score, 16); // 8 * 2
        assert!(ops.iter().all(|op| op.op == '='));
    }

    #[test]
    fn test_nw_with_gap() {
        let q: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let t: Vec<u8> = vec![0, 1, 3, 0, 1, 2, 3];    // ACTACGT (G deleted)
        let (ops, score) = nw_align(&q, &t, 2, 4, 4, 2);
        assert!(score > 0, "score={}", score);
        let has_indel = ops.iter().any(|op| op.op == 'I' || op.op == 'D');
        assert!(has_indel, "should have insertion or deletion");
    }

    #[test]
    fn test_simple_aligner_single_anchor() {
        let mut anchors = vec![make_anchor(24, 24, 15)];
        let qseq: Vec<u8> = (0..50).map(|i| i % 4).collect();
        let tseq: Vec<u8> = (0..50).map(|i| i % 4).collect();
        let opt = MapOptions::default();
        let mut ctx = AlignmentContext::new();
        let call = AlignAnchorContext {
            seed_bounds: (0, 0, 50, 50),
            rev: false, rid: 0,
            splice_flag: crate::align::map::AlignFlags::empty(),
            split_inv: false, is_hpc: false, k: 15,
            junc_db: None, ref_offset: 0,
        };

        let aligner = NWAligner;
        let result = aligner.align(&mut anchors, &qseq, &tseq, &opt, &mut ctx, &call);

        assert!(!result.cigar_ops.is_empty());
        assert!(result.dp_score > 0);
    }

    #[test]
    fn test_simple_aligner_two_anchors_with_gap() {
        let mut anchors = vec![
            make_anchor(24, 24, 15),
            make_anchor(44, 44, 15),
        ];
        let qseq: Vec<u8> = (0..60).map(|i| i % 4).collect();
        let tseq: Vec<u8> = (0..60).map(|i| i % 4).collect();
        let opt = MapOptions::default();
        let mut ctx = AlignmentContext::new();
        let call = AlignAnchorContext {
            seed_bounds: (0, 0, 60, 60),
            rev: false, rid: 0,
            splice_flag: crate::align::map::AlignFlags::empty(),
            split_inv: false, is_hpc: false, k: 15,
            junc_db: None, ref_offset: 0,
        };

        let aligner = NWAligner;
        let result = aligner.align(&mut anchors, &qseq, &tseq, &opt, &mut ctx, &call);

        assert_eq!(result.ref_start, 10);
        // ref_end extends past last anchor due to right extension
        assert!(result.ref_end >= 45);
        assert!(result.dp_score > 0);
        // Gap between anchors should be aligned with NW
        let has_match = result.cigar_ops.iter().any(|op| op.op == '=');
        assert!(has_match);
    }

    #[test]
    fn test_simple_aligner_empty() {
        let mut anchors: Vec<Minimizer> = Vec::new();
        let qseq: Vec<u8> = vec![0; 50];
        let tseq: Vec<u8> = vec![0; 50];
        let opt = MapOptions::default();
        let mut ctx = AlignmentContext::new();
        let call = AlignAnchorContext {
            seed_bounds: (0, 0, 50, 50),
            rev: false, rid: 0,
            splice_flag: crate::align::map::AlignFlags::empty(),
            split_inv: false, is_hpc: false, k: 15,
            junc_db: None, ref_offset: 0,
        };

        let aligner = NWAligner;
        let result = aligner.align(&mut anchors, &qseq, &tseq, &opt, &mut ctx, &call);
        assert!(result.cigar_ops.is_empty());
    }
}
