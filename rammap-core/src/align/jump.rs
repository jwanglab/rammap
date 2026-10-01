//! Jump-chain splice extension for long-intron detection.
//!
//! Extends clipped alignment ends into annotated exon junctions using a
//! `JumpDb` built from BED12 annotations. Used during splice alignment to
//! resolve long introns that the standard chaining step cannot bridge.

use crate::align::index::Index;
use crate::align::map::MapOptions;
use crate::align::pipeline::AlnResult;

const MIN_EXON_LEN: usize = 20;
const JUNC_ANNO: u16 = 0x1;
// const JUNC_MISC: u16 = 0x2;

/// A single junction record in the jump database.
#[derive(Clone, Debug)]
pub struct JumpJunc {
    pub left_pos: i32,      // left/start position
    pub right_pos: i32,     // right/end position
    pub count: i32,      // supporting read count
    pub strand: i16,   // +1 forward, -1 reverse, 0 unknown
    pub flag: u16,     // JUNC_ANNO or JUNC_MISC
}

/// Per-reference jump database (sorted by off, then off2).
pub struct JumpDb {
    // junctions[rid] = sorted Vec of JumpJunc
    pub junctions: Vec<Vec<JumpJunc>>,
}

impl JumpDb {
    /// Load a BED12 file into a jump database. Each intron between exon blocks
    /// generates two entries: (off=left, off2=right) and (off=right, off2=left).
    /// flag: JUNC_ANNO (0x1) for -j, JUNC_MISC (0x2) for --pass1
    /// min_sc: minimum score threshold (-1 to disable)
    pub fn load(mi: &Index, path: &str, flag: u16, min_sc: i32) -> std::io::Result<Self> {
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let n_seq = mi.seqs.len();
        let mut junctions: Vec<Vec<JumpJunc>> = (0..n_seq).map(|_| Vec::new()).collect();

        // Build name→rid lookup
        let name_to_rid: std::collections::HashMap<&str, usize> = mi.seqs.iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect();

        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') || line.is_empty() { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 12 { continue; }

            let chr = fields[0];
            let rid = match name_to_rid.get(chr) {
                Some(&r) => r,
                None => continue,
            };

            let start: i32 = match fields[1].parse() { Ok(v) => v, Err(_) => continue };
            let _end: i32 = match fields[2].parse() { Ok(v) => v, Err(_) => continue };

            // Score filtering (field 4)
            let score: i32 = fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(-1);
            if min_sc >= 0 && score >= 0 && score < min_sc { continue; }

            // Strand
            let strand: i16 = match fields.get(5).map(|s| s.as_bytes().first()) {
                Some(Some(b'+')) => 1,
                Some(Some(b'-')) => -1,
                _ => 0,
            };

            // Parse BED12 blocks to extract introns
            let n_blocks: usize = match fields[9].parse() { Ok(v) => v, Err(_) => continue };
            if n_blocks < 2 { continue; }

            let sizes: Vec<i32> = fields[10].split(',').filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            let starts: Vec<i32> = fields[11].split(',').filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();

            if sizes.len() < n_blocks || starts.len() < n_blocks { continue; }

            // Each gap between consecutive blocks is an intron
            let cnt = if score >= 0 { score } else { 1 };
            for b in 0..n_blocks - 1 {
                let intron_start = start + starts[b] + sizes[b]; // end of block b
                let intron_end = start + starts[b + 1]; // start of block b+1
                if intron_end <= intron_start { continue; }

                // Two entries per intron: left→right and right→left
                junctions[rid].push(JumpJunc {
                    left_pos: intron_start, right_pos: intron_end, count: cnt, strand, flag,
                });
                junctions[rid].push(JumpJunc {
                    left_pos: intron_end, right_pos: intron_start, count: cnt, strand, flag,
                });
            }
        }

        // Sort each per-ref array by (off, off2)
        for juncs in &mut junctions {
            juncs.sort_by(|a, b| a.left_pos.cmp(&b.left_pos).then(a.right_pos.cmp(&b.right_pos)));
        }

        Ok(JumpDb { junctions })
    }

    /// Merge another JumpDb into this one.
    pub fn merge(&mut self, other: &JumpDb) {
        for (rid, juncs) in self.junctions.iter_mut().enumerate() {
            if rid < other.junctions.len() {
                juncs.extend_from_slice(&other.junctions[rid]);
                juncs.sort_by(|a, b| a.left_pos.cmp(&b.left_pos).then(a.right_pos.cmp(&b.right_pos)));
            }
        }
    }

    /// Binary search: find index of last element with off <= x.
    fn get_core(juncs: &[JumpJunc], x: i32) -> Option<usize> {
        if juncs.is_empty() || x < juncs[0].left_pos { return None; }
        let mut s: usize = 0;
        let mut e: usize = juncs.len();
        while s < e {
            let mid = s + (e - s) / 2;
            if x >= juncs[mid].left_pos && (mid + 1 >= juncs.len() || x < juncs[mid + 1].left_pos) {
                return Some(mid);
            } else if x < juncs[mid].left_pos {
                e = mid;
            } else {
                s = mid + 1;
            }
        }
        None // should not happen for valid input
    }

    /// Get all junctions with off in [st, en) for reference rid.
    pub fn get(&self, rid: usize, st: i32, en: i32) -> &[JumpJunc] {
        if rid >= self.junctions.len() { return &[]; }
        let juncs = &self.junctions[rid];
        if juncs.is_empty() { return &[]; }
        // en used directly (already clamped by caller)
        let l = match Self::get_core(juncs, st) { Some(v) => v, None => return &[] };
        let r = match Self::get_core(juncs, en) { Some(v) => v, None => return &[] };
        if r < l { return &[]; }
        &juncs[l + 1..=r]  // return a[l+1..r] inclusive
    }
}

/// Encode ASCII base to nt4 (A=0,C=1,G=2,T=3,N=4).
#[inline]
fn seq_nt4(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0, b'C' | b'c' => 1, b'G' | b'g' => 2, b'T' | b't' => 3,
        _ => 4,
    }
}

/// Get nt4-encoded query subsequence for jump extension.
/// For left extension: first `ql` bases (or their revcomp from end).
/// For right extension: last `ql` bases (or their revcomp from start).
fn get_qseq(qseq_ascii: &[u8], qlen: usize, rev: bool, is_left: bool, ql: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(ql);
    if !rev {
        if is_left {
            for &b in &qseq_ascii[..ql] { out.push(seq_nt4(b)); }
        } else {
            for &b in &qseq_ascii[qlen - ql..qlen] { out.push(seq_nt4(b)); }
        }
    } else if is_left {
        for i in (qlen - ql..qlen).rev() {
            let c = seq_nt4(qseq_ascii[i]);
            out.push(if c < 4 { 3 - c } else { c });
        }
    } else {
        for i in (0..ql).rev() {
            let c = seq_nt4(qseq_ascii[i]);
            out.push(if c < 4 { 3 - c } else { c });
        }
    }
    out
}

/// Get nt4-encoded reference subsequence (forward strand).
fn get_tseq(mi: &Index, rid: usize, st: usize, en: usize) -> Vec<u8> {
    mi.get_region_nt4(rid, st, en)
}

/// Apply jump splice extension to a single alignment result.
/// Split alignment at exon boundaries using junction annotation.
pub fn jump_split(
    mi: &Index,
    opt: &MapOptions,
    qlen: usize,
    qseq: &[u8],  // ASCII query
    r: &mut AlnResult,
    jump_db: &JumpDb,
) {
    // EQX mode not supported for jump split
    jump_split_left(mi, opt, qlen, qseq, r, jump_db, 0);
    jump_split_right(mi, opt, qlen, qseq, r, jump_db, 0);
}

/// Check if alignment is eligible for jump extension.
fn jump_check(mi: &Index, r: &AlnResult, qlen: usize, ext: usize, is_left: bool) -> bool {
    if r.cigar_str.is_empty() { return false; }
    // Parse first/last CIGAR op
    let cigar_bytes = r.cigar_str.as_bytes();
    let (op, clen) = if is_left {
        parse_first_cigar_op(cigar_bytes)
    } else {
        parse_last_cigar_op(cigar_bytes)
    };
    if op != b'M' { return false; }
    if clen <= ext { return false; }

    let e = (r.is_reverse as usize) ^ (is_left as usize); // 0 for left side, 1 for right
    let clip = if e == 0 { r.query_start } else { qlen - r.query_end };
    if is_left {
        if clip >= r.ref_start { return false; }
    } else if clip >= mi.seqs[r.ref_id].len - r.ref_end { return false; }
    true
}

fn parse_first_cigar_op(cigar: &[u8]) -> (u8, usize) {
    let mut i = 0;
    while i < cigar.len() && cigar[i].is_ascii_digit() { i += 1; }
    if i == 0 || i >= cigar.len() { return (0, 0); }
    let len: usize = std::str::from_utf8(&cigar[..i]).unwrap_or("0").parse().unwrap_or(0);
    (cigar[i], len)
}

fn parse_last_cigar_op(cigar: &[u8]) -> (u8, usize) {
    if cigar.is_empty() { return (0, 0); }
    let op = cigar[cigar.len() - 1];
    // Find start of last number
    let mut i = cigar.len() - 2;
    while i > 0 && cigar[i].is_ascii_digit() { i -= 1; }
    if !cigar[i].is_ascii_digit() { i += 1; }
    let len: usize = std::str::from_utf8(&cigar[i..cigar.len() - 1]).unwrap_or("0").parse().unwrap_or(0);
    (op, len)
}

fn jump_split_left(
    mi: &Index,
    opt: &MapOptions,
    qlen: usize,
    qseq_ascii: &[u8],
    r: &mut AlnResult,
    jump_db: &JumpDb,
    ts_strand: i16,
) {
    let ext = 1 + (opt.scoring.mismatch_penalty + opt.scoring.match_score - 1) / opt.scoring.match_score + 1;
    let clip = if !r.is_reverse { r.query_start } else { qlen - r.query_end };
    let extt = clip.min(ext as usize);

    if !jump_check(mi, r, qlen, (ext + MIN_EXON_LEN as i32) as usize, true) { return; }

    let st = (r.ref_start as i32 - extt as i32).max(0);
    let en = r.ref_start as i32 + ext;
    let candidates = jump_db.get(r.ref_id, st, en);
    if candidates.is_empty() { return; }

    let mut i0_anno: Option<usize> = None;
    let mut n_anno = 0;
    let mut mm0_anno = 0;
    let mut i0_misc: Option<usize> = None;
    let mut n_misc = 0;
    let mut mm0_misc = 0;
    let mut qseq_buf: Option<Vec<u8>> = None;

    for (i, ai) in candidates.iter().enumerate() {
        if (ts_strand as i32) * (ai.strand as i32) < 0 { continue; }
        if ai.right_pos >= ai.left_pos { continue; } // wrong direction for left
        if ai.left_pos - ai.right_pos < 6 { continue; } // intron too small
        if (ai.right_pos as usize) < clip + ext as usize { continue; }

        // Lazy init query sequence
        if qseq_buf.is_none() {
            qseq_buf = Some(get_qseq(qseq_ascii, qlen, r.is_reverse, true, clip + ext as usize));
        }
        let qseq_nt4 = qseq_buf.as_ref().unwrap();

        let tl1 = clip as i32 + (ai.left_pos - r.ref_start as i32);
        // Get reference: two segments joined
        let tseq_right = get_tseq(mi, r.ref_id, ai.left_pos as usize, r.ref_start + ext as usize);
        let tseq_left = get_tseq(mi, r.ref_id, (ai.right_pos - tl1) as usize, ai.right_pos as usize);

        // Build combined tseq
        let mut tseq = Vec::with_capacity(clip + ext as usize);
        tseq.extend_from_slice(&tseq_left);
        tseq.extend_from_slice(&tseq_right);

        // Count mismatches
        let mut mm1 = 0;
        for j in 0..tl1 as usize {
            if j >= qseq_nt4.len() || j >= tseq.len() { break; }
            if qseq_nt4[j] != tseq[j] || qseq_nt4[j] > 3 || tseq[j] > 3 { mm1 += 1; }
        }
        let mut mm2 = 0;
        for j in tl1 as usize..clip + ext as usize {
            if j >= qseq_nt4.len() || j >= tseq.len() { break; }
            if qseq_nt4[j] != tseq[j] || qseq_nt4[j] > 3 || tseq[j] > 3 { mm2 += 1; }
        }

        if mm1 == 0 && mm2 <= 1 {
            if ai.flag & JUNC_ANNO != 0 {
                i0_anno = Some(i);
                mm0_anno = mm1 + mm2;
                n_anno += 1;
            } else {
                i0_misc = Some(i);
                mm0_misc = mm1 + mm2;
                n_misc += 1;
            }
        }
    }

    let (m, i0, mm0) = if n_anno > 0 {
        (n_anno, i0_anno.unwrap(), mm0_anno)
    } else if n_misc > 0 {
        (n_misc, i0_misc.unwrap(), mm0_misc)
    } else {
        return;
    };

    let ai = &candidates[i0];
    // Signed offset from the original alignment start to the new post-intron boundary.
    let junction_shift = ai.left_pos - r.ref_start as i32;

    if m == 1 && clip as i32 + junction_shift >= opt.filtering.jump_min_match {
        // Add one more exon: prepend match + intron to CIGAR
        let new_match_len = clip as i32 + junction_shift;
        let intron_len = ai.left_pos - ai.right_pos;

        // Modify existing first CIGAR op (reduce its length by junction_shift)
        let first_op_new_len = get_first_cigar_len(&r.cigar_str) as i32 - junction_shift;
        let rest_cigar = trim_first_cigar_op(&r.cigar_str, first_op_new_len as usize);

        // Prepend MD units for new pre-intron (`mm1`) and post-intron (`mm2`) regions if MD tracking is enabled.
        if !r.md_str.is_empty() {
            let tseq_right = get_tseq(mi, r.ref_id, ai.left_pos as usize, r.ref_start + ext as usize);
            let tseq_left = get_tseq(mi, r.ref_id, (ai.right_pos - new_match_len) as usize, ai.right_pos as usize);
            let mut tseq = Vec::with_capacity(clip + ext as usize);
            tseq.extend_from_slice(&tseq_left);
            tseq.extend_from_slice(&tseq_right);
            let qseq_nt4 = get_qseq(qseq_ascii, qlen, r.is_reverse, true, clip + ext as usize);
            let mm1_units: Vec<MdUnit> = (0..new_match_len as usize)
                .map(|j| md_unit_at(&qseq_nt4, &tseq, j)).collect();
            let mm2_units: Vec<MdUnit> = (new_match_len as usize..clip + ext as usize)
                .map(|j| md_unit_at(&qseq_nt4, &tseq, j)).collect();

            r.md_str = splice_md_head(&r.md_str, &mm1_units, &mm2_units, junction_shift);
        }

        r.cigar_str = format!("{}M{}N{}", new_match_len, intron_len, rest_cigar);
        r.ref_start = (ai.right_pos - new_match_len) as usize;
        if !r.is_reverse { r.query_start = 0; } else { r.query_end = qlen; }
        let matches_before = r.matches;
        r.block_len += clip;
        r.matches += clip - mm0;
        extend_divergence(r, matches_before, clip);
        r.edit_distance += mm0 as u32;
        r.dp_score_original += (clip as i32 - mm0 as i32) * opt.scoring.match_score - mm0 as i32 * opt.scoring.mismatch_penalty;
        r.dp_score += (clip as i32 - mm0 as i32) * opt.scoring.match_score - mm0 as i32 * opt.scoring.mismatch_penalty;
        if !r.is_spliced {
            r.is_spliced = true;
            let bonus = (opt.scoring.match_score + opt.scoring.mismatch_penalty) + ((opt.scoring.match_score + opt.scoring.mismatch_penalty) >> 1);
            r.dp_score += bonus;
        }
    } else if m > 0 && ai.left_pos > r.ref_start as i32 {
        // Shrink md_str by the same number of bases removed from the CIGAR.
        let first_new_len = get_first_cigar_len(&r.cigar_str) as i32 - junction_shift;
        r.cigar_str = replace_first_cigar_len(&r.cigar_str, first_new_len as usize);
        if !r.md_str.is_empty() {
            r.md_str = render_md(&md_trim_head(&parse_md(&r.md_str), junction_shift as usize));
        }
        r.ref_start += junction_shift as usize;
        if !r.is_reverse { r.query_start += junction_shift as usize; } else { r.query_end -= junction_shift as usize; }
    }
}

fn jump_split_right(
    mi: &Index,
    opt: &MapOptions,
    qlen: usize,
    qseq_ascii: &[u8],
    r: &mut AlnResult,
    jump_db: &JumpDb,
    ts_strand: i16,
) {
    let ext = 1 + (opt.scoring.mismatch_penalty + opt.scoring.match_score - 1) / opt.scoring.match_score + 1;
    let clip = if !r.is_reverse { qlen - r.query_end } else { r.query_start };
    let extt = clip.min(ext as usize);
    let tlen = mi.seqs[r.ref_id].len;

    if !jump_check(mi, r, qlen, (ext + MIN_EXON_LEN as i32) as usize, false) { return; }

    let st = r.ref_end as i32 - ext;
    let en = r.ref_end as i32 + extt as i32;
    let candidates = jump_db.get(r.ref_id, st, en);
    if candidates.is_empty() { return; }

    let mut i0_anno: Option<usize> = None;
    let mut n_anno = 0;
    let mut mm0_anno = 0;
    let mut i0_misc: Option<usize> = None;
    let mut n_misc = 0;
    let mut mm0_misc = 0;
    let mut qseq_buf: Option<Vec<u8>> = None;

    for (i, ai) in candidates.iter().enumerate() {
        if (ts_strand as i32) * (ai.strand as i32) < 0 { continue; }
        if ai.right_pos <= ai.left_pos { continue; } // wrong direction for right
        if ai.right_pos - ai.left_pos < 6 { continue; }
        if ai.right_pos as usize + clip + ext as usize > tlen { continue; }

        if qseq_buf.is_none() {
            qseq_buf = Some(get_qseq(qseq_ascii, qlen, r.is_reverse, false, clip + ext as usize));
        }
        let qseq_nt4 = qseq_buf.as_ref().unwrap();

        let tl1 = clip as i32 + (r.ref_end as i32 - ai.left_pos);
        // Get reference: two segments
        let tseq_left = get_tseq(mi, r.ref_id, (r.ref_end as i32 - ext) as usize, ai.left_pos as usize);
        let tseq_right = get_tseq(mi, r.ref_id, ai.right_pos as usize, (ai.right_pos + tl1) as usize);

        // Build combined tseq: left part then right part
        let mut tseq = Vec::with_capacity(clip + ext as usize);
        tseq.extend_from_slice(&tseq_left);
        tseq.extend_from_slice(&tseq_right);

        // Count mismatches
        let split_point = (clip as i32 + ext - tl1) as usize;
        let mut mm2 = 0;
        for j in 0..split_point {
            if j >= qseq_nt4.len() || j >= tseq.len() { break; }
            if qseq_nt4[j] != tseq[j] || qseq_nt4[j] > 3 || tseq[j] > 3 { mm2 += 1; }
        }
        let mut mm1 = 0;
        for j in split_point..clip + ext as usize {
            if j >= qseq_nt4.len() || j >= tseq.len() { break; }
            if qseq_nt4[j] != tseq[j] || qseq_nt4[j] > 3 || tseq[j] > 3 { mm1 += 1; }
        }

        if mm1 == 0 && mm2 <= 1 {
            if ai.flag & JUNC_ANNO != 0 {
                if i0_anno.is_none() { i0_anno = Some(i); mm0_anno = mm1 + mm2; }
                n_anno += 1;
            } else {
                if i0_misc.is_none() { i0_misc = Some(i); mm0_misc = mm1 + mm2; }
                n_misc += 1;
            }
        }
    }

    let (m, i0, mm0) = if n_anno > 0 {
        (n_anno, i0_anno.unwrap(), mm0_anno)
    } else if n_misc > 0 {
        (n_misc, i0_misc.unwrap(), mm0_misc)
    } else {
        return;
    };

    let ai = &candidates[i0];
    // Signed offset from the original alignment end to the new pre-intron boundary.
    let junction_shift = r.ref_end as i32 - ai.left_pos;

    if m == 1 && clip as i32 + junction_shift >= opt.filtering.jump_min_match {
        // Add one more exon: append intron + match to CIGAR
        let new_match_len = clip as i32 + junction_shift;
        let intron_len = ai.right_pos - ai.left_pos;

        // Modify existing last CIGAR op (reduce its length by junction_shift)
        let last_new_len = get_last_cigar_len(&r.cigar_str) as i32 - junction_shift;
        let prefix_cigar = trim_last_cigar_op(&r.cigar_str, last_new_len as usize);

        // Append MD units for new post-intron (`mm1`) and pre-intron (`mm2`) regions if MD tracking is enabled.
        if !r.md_str.is_empty() {
            let split_point = (clip as i32 + ext - new_match_len) as usize;
            let tseq_left = get_tseq(mi, r.ref_id, (r.ref_end as i32 - ext) as usize, ai.left_pos as usize);
            let tseq_right = get_tseq(mi, r.ref_id, ai.right_pos as usize, (ai.right_pos + new_match_len) as usize);
            let mut tseq = Vec::with_capacity(clip + ext as usize);
            tseq.extend_from_slice(&tseq_left);
            tseq.extend_from_slice(&tseq_right);
            let qseq_nt4 = get_qseq(qseq_ascii, qlen, r.is_reverse, false, clip + ext as usize);
            let mm2_units: Vec<MdUnit> = (0..split_point)
                .map(|j| md_unit_at(&qseq_nt4, &tseq, j)).collect();
            let mm1_units: Vec<MdUnit> = (split_point..clip + ext as usize)
                .map(|j| md_unit_at(&qseq_nt4, &tseq, j)).collect();

            r.md_str = splice_md_tail(&r.md_str, &mm2_units, &mm1_units, junction_shift);
        }

        r.cigar_str = format!("{}{}N{}M", prefix_cigar, intron_len, new_match_len);
        r.ref_end = (ai.right_pos + new_match_len) as usize;
        if !r.is_reverse { r.query_end = qlen; } else { r.query_start = 0; }
        let matches_before = r.matches;
        r.block_len += clip;
        r.matches += clip - mm0;
        extend_divergence(r, matches_before, clip);
        r.edit_distance += mm0 as u32;
        r.dp_score_original += (clip as i32 - mm0 as i32) * opt.scoring.match_score - mm0 as i32 * opt.scoring.mismatch_penalty;
        r.dp_score += (clip as i32 - mm0 as i32) * opt.scoring.match_score - mm0 as i32 * opt.scoring.mismatch_penalty;
        if !r.is_spliced {
            r.is_spliced = true;
            let bonus = (opt.scoring.match_score + opt.scoring.mismatch_penalty) + ((opt.scoring.match_score + opt.scoring.mismatch_penalty) >> 1);
            r.dp_score += bonus;
        }
    } else if m > 0 && r.ref_end as i32 > ai.left_pos {
        // Shrink md_str by the same number of bases removed from the CIGAR.
        let last_new_len = get_last_cigar_len(&r.cigar_str) as i32 - junction_shift;
        r.cigar_str = replace_last_cigar_len(&r.cigar_str, last_new_len as usize);
        if !r.md_str.is_empty() {
            r.md_str = render_md(&md_trim_tail(&parse_md(&r.md_str), junction_shift as usize));
        }
        r.ref_end -= junction_shift as usize;
        if !r.is_reverse { r.query_end -= junction_shift as usize; } else { r.query_start += junction_shift as usize; }
    }
}

// CIGAR string helpers

fn get_first_cigar_len(cigar: &str) -> usize {
    let bytes = cigar.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
    cigar[..i].parse().unwrap_or(0)
}

fn get_last_cigar_len(cigar: &str) -> usize {
    let bytes = cigar.as_bytes();
    if bytes.is_empty() { return 0; }
    let mut i = bytes.len() - 2;
    while i > 0 && bytes[i].is_ascii_digit() { i -= 1; }
    if !bytes[i].is_ascii_digit() { i += 1; }
    cigar[i..bytes.len() - 1].parse().unwrap_or(0)
}

/// Replace first CIGAR op length, return new CIGAR string with replaced first op.
fn trim_first_cigar_op(cigar: &str, new_len: usize) -> String {
    let bytes = cigar.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
    // i now points to the op character
    format!("{}{}", new_len, &cigar[i..])
}

fn replace_first_cigar_len(cigar: &str, new_len: usize) -> String {
    trim_first_cigar_op(cigar, new_len)
}

/// Replace last CIGAR op length, return prefix + new last op.
fn trim_last_cigar_op(cigar: &str, new_len: usize) -> String {
    let bytes = cigar.as_bytes();
    if bytes.is_empty() { return String::new(); }
    let op = bytes[bytes.len() - 1];
    let mut i = bytes.len() - 2;
    while i > 0 && bytes[i].is_ascii_digit() { i -= 1; }
    if !bytes[i].is_ascii_digit() { i += 1; }
    format!("{}{}{}", &cigar[..i], new_len, op as char)
}

fn replace_last_cigar_len(cigar: &str, new_len: usize) -> String {
    trim_last_cigar_op(cigar, new_len)
}

/// Recomputes `r.divergence` (the `de:f:` tag) after extending an alignment by `clip` bases.
///
/// Reconstructs the prior divergence denominator from `matches_before` and `r.divergence`, 
/// adjusts it by `clip`, and updates `r.divergence` using the already-updated `r.matches`.
///
/// # Key Assumption
///
/// The newly extended region contains substitutions only (no indels), so the effective 
/// event denominator increases by exactly `clip`.
fn extend_divergence(r: &mut AlnResult, matches_before: usize, clip: usize) {
    if r.divergence >= 1.0 || matches_before == 0 { return; }
    let event_denom_old = matches_before as f64 / (1.0 - r.divergence);
    let event_denom_new = event_denom_old + clip as f64;
    r.divergence = if event_denom_new > 0.0 { 1.0 - r.matches as f64 / event_denom_new } else { 0.0 };
}

// MD string helpers for keeping `AlnResult.md_str` in sync with CIGAR extensions and trims.

/// Represents a single token in a parsed SAM MD string (match run, single mismatch, or deletion).
///
/// Stores reference bases for mismatches and deletions per the SAM spec grammar:
/// `[0-9]+(([A-Z]|\^[A-Z]+)[0-9]+)*`.
#[derive(Clone, Debug, PartialEq)]
enum MdTok {
    Match(usize),
    Mismatch(u8),
    Del(Vec<u8>),
}

fn parse_md(md: &str) -> Vec<MdTok> {
    let b = md.as_bytes();
    let mut i = 0;
    let mut toks = Vec::new();
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            let n: usize = std::str::from_utf8(&b[start..i]).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            if n > 0 { toks.push(MdTok::Match(n)); }
        } else if b[i] == b'^' {
            i += 1;
            let start = i;
            while i < b.len() && b[i].is_ascii_alphabetic() { i += 1; }
            toks.push(MdTok::Del(b[start..i].to_vec()));
        } else {
            toks.push(MdTok::Mismatch(b[i]));
            i += 1;
        }
    }
    toks
}

/// Render tokens back to a canonical MD string (adjacent `Match` runs merge automatically,
/// and the result always starts/ends with a number, per the MD grammar).
fn render_md(toks: &[MdTok]) -> String {
    let mut out = String::new();
    let mut pending_match = 0usize;
    for t in toks {
        match t {
            MdTok::Match(n) => pending_match += n,
            MdTok::Mismatch(c) => {
                out.push_str(&pending_match.to_string());
                pending_match = 0;
                out.push(*c as char);
            }
            MdTok::Del(bases) => {
                out.push_str(&pending_match.to_string());
                pending_match = 0;
                out.push('^');
                for b in bases { out.push(*b as char); }
            }
        }
    }
    out.push_str(&pending_match.to_string());
    out
}

/// Drop the first `n` reference-consumed positions (matches/mismatches count 1 each,
/// deletions count their full length), splitting a token if the cut lands inside it.
fn md_trim_head(toks: &[MdTok], mut n: usize) -> Vec<MdTok> {
    let mut i = 0;
    let mut partial: Option<MdTok> = None;
    while n > 0 && i < toks.len() {
        match &toks[i] {
            MdTok::Match(len) => {
                if *len <= n { n -= len; } else { partial = Some(MdTok::Match(len - n)); n = 0; }
            }
            MdTok::Mismatch(_) => { n -= 1; }
            MdTok::Del(bases) => {
                let len = bases.len();
                if len <= n { n -= len; } else { partial = Some(MdTok::Del(bases[n..].to_vec())); n = 0; }
            }
        }
        i += 1;
    }
    let mut result = Vec::new();
    if let Some(p) = partial { result.push(p); }
    result.extend_from_slice(&toks[i..]);
    result
}

/// Drop the last `n` reference-consumed positions, mirroring [`md_trim_head`].
fn md_trim_tail(toks: &[MdTok], mut n: usize) -> Vec<MdTok> {
    let mut i = toks.len();
    let mut partial: Option<MdTok> = None;
    while n > 0 && i > 0 {
        i -= 1;
        match &toks[i] {
            MdTok::Match(len) => {
                if *len <= n { n -= len; } else { partial = Some(MdTok::Match(len - n)); n = 0; }
            }
            MdTok::Mismatch(_) => { n -= 1; }
            MdTok::Del(bases) => {
                let len = bases.len();
                if len <= n { n -= len; } else { partial = Some(MdTok::Del(bases[..len - n].to_vec())); n = 0; }
            }
        }
    }
    let mut result = toks[..i].to_vec();
    if let Some(p) = partial { result.push(p); }
    result
}

/// One newly-compared reference position: a match, or a mismatch against the given
/// (ASCII) reference base. Built directly from the same nt4 query/reference comparison
/// `jump_split_left`/`jump_split_right` already do for candidate scoring -- never a
/// deletion, since that comparison is a fixed-length substitution-only walk.
enum MdUnit {
    Match,
    Mismatch(u8),
}

fn md_unit_at(qseq_nt4: &[u8], tseq: &[u8], j: usize) -> MdUnit {
    let q = qseq_nt4[j];
    let t = tseq[j];
    if q != t || q > 3 || t > 3 {
        MdUnit::Mismatch(Index::NT4_TO_ASCII[t.min(4) as usize])
    } else {
        MdUnit::Match
    }
}

/// Splice MD tag units for a newly discovered upstream exon onto the front of `old_md`.
///
/// Used by `jump_split_left` to update MD tags across a 5' splice junction.
///
/// # Arguments
///
/// * `old_md` - Original alignment MD string.
/// * `mm1_units` - MD units for the new pre-intron exon (always prepended to the front).
/// * `mm2_units` - MD units covering the overlap/growth window at the old alignment start.
/// * `junction_shift` - Coordinate shift (`ai.left_pos - r.ref_start`):
///   * `>= 0`: Overlap. Trims `junction_shift` bases from the front of `old_md`.
///   * `< 0`: Growth. Prepends `-junction_shift` units from `mm2_units` ahead of `old_md`.
fn splice_md_head(old_md: &str, mm1_units: &[MdUnit], mm2_units: &[MdUnit], junction_shift: i32) -> String {
    let old_toks = parse_md(old_md);
    let old_part = if junction_shift >= 0 {
        md_trim_head(&old_toks, junction_shift as usize)
    } else {
        let growth_end = ((-junction_shift) as usize).min(mm2_units.len());
        let mut combined = md_units_to_toks(&mm2_units[..growth_end]);
        combined.extend(old_toks);
        combined
    };
    let mut final_toks = md_units_to_toks(mm1_units);
    final_toks.extend(old_part);
    render_md(&final_toks)
}

/// Splice MD tag units for a newly discovered downstream exon onto the end of `old_md`.
///
/// Used by `jump_split_right` to update MD tags across a 3' splice junction.
///
/// # Arguments
///
/// * `old_md` - Original alignment MD string.
/// * `mm2_units` - MD units covering the overlap/growth window at the old alignment end.
/// * `mm1_units` - MD units for the new post-intron exon (always appended to the end).
/// * `junction_shift` - Coordinate shift (`r.ref_end - ai.left_pos`):
///   * `>= 0`: Overlap. Trims `junction_shift` bases from the tail of `old_md`.
///   * `< 0`: Growth. Appends `-junction_shift` units from `mm2_units` after `old_md`.
fn splice_md_tail(old_md: &str, mm2_units: &[MdUnit], mm1_units: &[MdUnit], junction_shift: i32) -> String {
    let old_toks = parse_md(old_md);
    let mut final_toks = if junction_shift >= 0 {
        md_trim_tail(&old_toks, junction_shift as usize)
    } else {
        let growth_start = mm2_units.len().saturating_sub((-junction_shift) as usize);
        let mut combined = old_toks;
        combined.extend(md_units_to_toks(&mm2_units[growth_start..]));
        combined
    };
    final_toks.extend(md_units_to_toks(mm1_units));
    render_md(&final_toks)
}

fn md_units_to_toks(units: &[MdUnit]) -> Vec<MdTok> {
    let mut toks = Vec::new();
    let mut run = 0usize;
    for u in units {
        match u {
            MdUnit::Match => run += 1,
            MdUnit::Mismatch(refbase) => {
                toks.push(MdTok::Match(run));
                run = 0;
                toks.push(MdTok::Mismatch(*refbase));
            }
        }
    }
    toks.push(MdTok::Match(run));
    toks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::index::Index;

    // Deterministic pseudo-random ACGT sequence, no homopolymer runs (LCG).
    fn gen_dna(state: &mut u64, n: usize) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut seq: Vec<u8> = Vec::new();
        while seq.len() < n {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = bases[(*state >> 33) as usize % 4];
            if seq.last() != Some(&b) { seq.push(b); }
        }
        seq
    }

    #[allow(clippy::too_many_arguments)]
    fn make_aln_result(
        ref_id: usize, cigar_str: &str, md_str: &str, query_start: usize, query_end: usize,
        ref_start: usize, ref_end: usize, matches: usize, block_len: usize, divergence: f64,
    ) -> AlnResult {
        AlnResult {
            ref_id, is_reverse: false, chain_score: 0, initial_chain_score: 0, anchor_count: 0, s2_score: None, hash: 0,
            align_score: 0, matches, block_len, cigar_str: cigar_str.to_string(), cs_str: String::new(), ds_str: String::new(),
            md_str: md_str.to_string(), query_start, query_end, ref_start, ref_end, edit_distance: 0, num_ambiguous: 0, divergence,
            is_secondary: false, split: 0, split_depth: 0, dp_score: 0, dp_score_original: 0, effective_cnt: 0, pre_num_suboptimal: 0,
            is_spliced: false, trans_strand: 0, dp_score_secondary: 0, secondary_chain_score: 0, num_suboptimal: 0, split_inv: false,
            inv: false, proper_frag: false, seg_split: false, div: 0.0, is_alt: false, is_root_chain: false,
        }
    }

    /// Verifies that `jump_split` updates derived alignment fields (`md_str`, `divergence`,
    /// and `edit_distance`) alongside `cigar_str` when extending an alignment.
    ///
    /// Specifically tests boundary expansion ("growth") where a mismatch falls within
    /// the confirmation/growth window (`mm2`) prior to the new exon junction.
    #[test]
    fn test_jump_split_right_extends_derived_fields_through_a_growth_mismatch() {
        let mut st = 0xC0FFEE_u64;
        let mut exon1 = gen_dna(&mut st, 100); // true first exon, full length
        let intron = gen_dna(&mut st, 500);
        let exon2 = gen_dna(&mut st, 10);
        let padding = gen_dna(&mut st, 30); // room for jump_split_right's own bounds check

        // Hardcode an explicit C -> A SNP at position 98
        let mismatch_pos = 98;
        let ref_base_at_mismatch = b'C';
        exon1[mismatch_pos] = ref_base_at_mismatch;
        let reference = [&exon1[..], &intron[..], &exon2[..], &padding[..]].concat();

        let mut query_exon1 = exon1.clone();
        query_exon1[mismatch_pos] = b'A'; // Mismatch in query sequence
        let query = [&query_exon1[..], &exon2[..]].concat();

        let index = Index::build(vec![("chr1".to_string(), reference)], 5, 15, false, usize::MAX);

        // Build candidate splice database entry for the intron
        let intron_start = exon1.len() as i32;
        let intron_end = (exon1.len() + intron.len()) as i32;
        let jump_db = JumpDb {
            junctions: vec![vec![
                // Decoy anchors required by JumpDb binary search
                JumpJunc { left_pos: 10, right_pos: 20, count: 1, strand: 1, flag: JUNC_ANNO },
                JumpJunc { left_pos: 20, right_pos: 10, count: 1, strand: 1, flag: JUNC_ANNO },
                // Target intron boundaries
                JumpJunc { left_pos: intron_start, right_pos: intron_end, count: 1, strand: 1, flag: JUNC_ANNO },
                JumpJunc { left_pos: intron_end, right_pos: intron_start, count: 1, strand: 1, flag: JUNC_ANNO },
            ]],
        };

        let opt = MapOptions::default();
        // Baseline state before jump_split: stopped short at pos 98 (98 matches, MD: "98")
        let mut r = make_aln_result(0, "98M", "98", 0, 98, 0, 98, 98, 98, 0.0);

        jump_split(&index, &opt, query.len(), &query, &mut r, &jump_db);

        // Expected state after growing through C->A mismatch and splicing 10bp exon 2:
        let expected_divergence = 1.0 - (109.0 / 110.0);

        assert_eq!(r.cigar_str, format!("{}M{}N{}M", exon1.len(), intron.len(), exon2.len()));
        assert_eq!(r.matches, 109);
        assert_eq!(r.edit_distance, 1);
        assert_eq!(r.md_str, "98C11");
        assert!((r.divergence - expected_divergence).abs() < 1e-9);
    }
}
