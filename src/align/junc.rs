//! Junction database for known splice sites, used during splice alignment.
//!
//! Supports two modes:
//! - BED mode (`--junc-bed`): boolean flags at known splice donor/acceptor sites
//! - SPSC mode (`--spsc`): per-position numerical scores
//!
//! Loaded from BED-format annotation files and queried by the DP splice kernel
//! to bias scoring toward known junctions.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader};
use std::fs::File;

use super::dp::SPSC_OFFSET;

/// Per-interval data for BED mode: a single annotated interval on a target
/// sequence with strand and optional thick/itemRgb extras from the BED row.
#[derive(Debug, Clone)]
pub struct BedInterval {
    pub start: i32,
    pub end: i32,
    pub strand: i32, // +1, -1, or 0
}

/// Per-(contig, strand) sorted entries for SPSC mode
#[derive(Debug, Clone, Default)]
pub struct SpscContig {
    pub entries: Vec<u64>, // sorted by pos; encoding: pos<<8 | (score+64)<<1 | type
}

/// Junction database — separate from Index, not serialized
#[derive(Debug)]
pub enum JunctionDb {
    /// BED mode: per-contig sorted intervals, indexed by rid
    Bed(Vec<Vec<BedInterval>>),
    /// SPSC mode: indexed by cid<<1|strand_bit, length = n_seq*2
    Spsc(Vec<SpscContig>),
}


/// Compute max SPSC bonus
pub fn max_spsc_bonus(q2: i32, q: i32) -> i32 {
    let max_sc = (q2 + 1) / 2 - 1;
    max_sc.max(q2 - q)
}

/// Load BED junction annotations
///
/// Supports BED6 (direct intervals) and BED12 (block-based intron extraction).
/// `read_junc=true` enables BED12 intron extraction mode.
pub fn load_bed_junctions(
    path: &str,
    names: &HashMap<String, usize>,
    n_seq: usize,
) -> io::Result<JunctionDb> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut intervals: Vec<Vec<BedInterval>> = vec![Vec::new(); n_seq];

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') || line.starts_with("track") || line.starts_with("browser") {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 { continue; }

        let rid = match names.get(fields[0]) {
            Some(&id) => id,
            None => continue,
        };
        let st: i32 = match fields[1].parse() { Ok(v) => v, Err(_) => continue };
        let en: i32 = match fields[2].parse() { Ok(v) => v, Err(_) => continue };
        if st < 0 || st >= en { continue; }

        let strand = if fields.len() > 5 {
            match fields[5] {
                "+" => 1i32,
                "-" => -1i32,
                _ => 0i32,
            }
        } else {
            0
        };

        // BED12: extract introns from block structure
        if fields.len() >= 12 {
            let n_blk: usize = match fields[9].parse() { Ok(v) => v, Err(_) => continue };
            if n_blk < 2 { continue; }
            let sizes: Vec<i32> = fields[10].split(',').filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            let starts: Vec<i32> = fields[11].split(',').filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            if sizes.len() < n_blk || starts.len() < n_blk { continue; }

            let mut prev_en = st + starts[0] + sizes[0];
            for i in 1..n_blk {
                let intron_st = prev_en;
                let intron_en = st + starts[i];
                prev_en = st + starts[i] + sizes[i];
                if intron_en > intron_st {
                    intervals[rid].push(BedInterval { start: intron_st, end: intron_en, strand });
                }
            }
        } else {
            intervals[rid].push(BedInterval { start: st, end: en, strand });
        }
    }

    // Sort and merge duplicates (index.c:762-794)
    let mut total = 0usize;
    for intv in &mut intervals {
        if intv.is_empty() { continue; }
        // Sort by st, then by en within same st
        intv.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
        // Merge intervals with identical (start, end)
        let mut merged = Vec::with_capacity(intv.len());
        let mut i = 0;
        while i < intv.len() {
            let mut j = i + 1;
            while j < intv.len() && intv[j].start == intv[i].start && intv[j].end == intv[i].end {
                j += 1;
            }
            merged.push(intv[i].clone());
            i = j;
        }
        total += merged.len();
        *intv = merged;
    }
    eprintln!("[M::load_bed_junctions] loaded {} non-redundant junctions", total);
    Ok(JunctionDb::Bed(intervals))
}

/// Load SPSC splice scores
pub fn load_spsc_scores(
    path: &str,
    names: &HashMap<String, usize>,
    n_seq: usize,
    seq_lens: &[usize],
    max_sc: i32,
    scale: f32,
) -> io::Result<JunctionDb> {
    let max_sc = max_sc.min(63); // SPSC_OFFSET = 64, must fit in u8
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut contigs: Vec<SpscContig> = (0..n_seq * 2).map(|_| SpscContig::default()).collect();
    let mut n_read: usize = 0;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 { continue; }

        let cid = match names.get(fields[0]) {
            Some(&id) => id,
            None => continue,
        };
        let pos: i64 = match fields[1].parse() { Ok(v) => v, Err(_) => continue };
        let strand = match fields[2] {
            "+" => 1i32,
            "-" => -1i32,
            _ => continue,
        };
        let type_bit: u64 = match fields[3] {
            "D" => 0,
            "A" => 1,
            _ => continue,
        };
        let mut score: i32 = match fields[4].parse() { Ok(v) => v, Err(_) => continue };

        // Apply scale (index.c:1010-1011)
        if scale > 0.0 && scale < 1.0 {
            score = if score > 0 {
                (score as f64 * scale as f64 + 0.499) as i32
            } else {
                (score as f64 * scale as f64 - 0.499) as i32
            };
        }
        // Clamp (index.c:1012-1013)
        if score > max_sc { score = max_sc; }
        if score < -max_sc { score = -max_sc; }

        if pos <= 0 || pos >= seq_lens[cid] as i64 { continue; }

        let strand_bit = if strand > 0 { 0usize } else { 1usize };
        let s = &mut contigs[cid * 2 + strand_bit];
        s.entries.push((pos as u64) << 8 | ((score + SPSC_OFFSET) as u64) << 1 | type_bit);
        n_read += 1;
    }

    // Sort each contig's entries by position
    for s in &mut contigs {
        if !s.entries.is_empty() {
            s.entries.sort_unstable();
        }
    }
    eprintln!("[M::load_spsc_scores] loaded {} splice scores", n_read);
    Ok(JunctionDb::Spsc(contigs))
}

/// Fill junction array for reference region [st, en)
///
/// Dispatches to BED or SPSC lookup depending on JunctionDb variant.
/// `rev` selects minus-strand SPSC data when true.
pub fn get_junc(db: &JunctionDb, ctg: usize, st: i32, en: i32, rev: bool, junc: &mut [u8]) {
    match db {
        JunctionDb::Bed(intervals) => bed_junc(intervals, ctg, st, en, junc),
        JunctionDb::Spsc(contigs) => spsc_get(contigs, ctg, st as i64, en as i64, rev, junc),
    }
}

/// BED junction lookup
fn bed_junc(intervals: &[Vec<BedInterval>], ctg: usize, st: i32, en: i32, s: &mut [u8]) {
    let len = (en - st) as usize;
    for b in s[..len].iter_mut() { *b = 0; }
    if ctg >= intervals.len() { return; }
    let intv = &intervals[ctg];
    if intv.is_empty() { return; }

    // Binary search for first interval with st >= query start
    let mut left = 0usize;
    let mut right = intv.len();
    while right > left {
        let mid = left + ((right - left) >> 1);
        if intv[mid].start >= st { right = mid; }
        else { left = mid + 1; }
    }

    for iv in &intv[left..] {
        if st <= iv.start && en >= iv.end && iv.strand != 0 {
            let st_off = (iv.start - st) as usize;
            let en_off = (iv.end - 1 - st) as usize;
            if st_off < len && en_off < len {
                if iv.strand > 0 {
                    s[st_off] |= 1;  // forward donor
                    s[en_off] |= 2;  // forward acceptor
                } else {
                    s[st_off] |= 8;  // reverse donor
                    s[en_off] |= 4;  // reverse acceptor
                }
            }
        }
    }
}

/// Binary search for interval containing position x
fn find_intv(entries: &[u64], x: i64) -> i32 {
    let n = entries.len() as i32;
    if n == 0 { return -1; }
    if x < (entries[0] >> 8) as i64 { return -1; }
    let mut s = 0i32;
    let mut e = n;
    while s < e {
        let mid = s + (e - s) / 2;
        let pos_mid = (entries[mid as usize] >> 8) as i64;
        let pos_next = if mid + 1 < n { (entries[(mid + 1) as usize] >> 8) as i64 } else { i64::MAX };
        if x >= pos_mid && x < pos_next {
            return mid;
        } else if x < pos_mid {
            e = mid;
        } else {
            s = mid + 1;
        }
    }
    // Should not reach here if input is valid
    n - 1
}

/// SPSC splice score lookup
fn spsc_get(contigs: &[SpscContig], cid: usize, st: i64, en: i64, rev: bool, sc: &mut [u8]) {
    let len = (en - st) as usize;
    for b in sc[..len].iter_mut() { *b = 0xff; }
    let strand_bit = if rev { 1usize } else { 0usize };
    let idx = cid * 2 + strand_bit;
    if idx >= contigs.len() { return; }
    let s = &contigs[idx];
    if s.entries.is_empty() { return; }

    let l = find_intv(&s.entries, st);
    let r = find_intv(&s.entries, en);
    for j in (l + 1)..=r {
        if j < 0 { continue; }
        let j = j as usize;
        if j >= s.entries.len() { break; }
        let x = (s.entries[j] >> 8) as i64 - st;
        let score = (s.entries[j] & 0xff) as u8;
        if x == en - st { continue; }
        if x >= 0 && (x as usize) < len {
            let xu = x as usize;
            if sc[xu] == 0xff || sc[xu] < score {
                sc[xu] = score;
            }
        }
    }
}
