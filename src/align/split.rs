//! Split-index mode: multi-part reference processing and merge.
//!
//! When the reference is too large for a single index, it is split into parts.
//! Each part is aligned independently, results are written to temporary files,
//! and `split_merge` combines them with re-filtering and re-pairing.

use std::io::{self, BufReader, BufWriter, Read, Write};
use std::fs::File;
use crate::align::index::{Index, TargetSequence};
use crate::align::pipeline::{AlnResult, OutputConfig, ReadInfo, refilter_merged_results, format_output, get_mate_info};
use crate::align::pair::{PeReg, pair_alignments};
use crate::align::map::MapOptions;
use crate::align::stats::AlignmentStats;

/// Create a temp file for one index part, write the header.
/// Header format:
///   k: u32, n_seq: u32, then per-seq: name_len(u32), name([u8]), seq_len(u32)
pub fn split_init(prefix: &str, part_index: usize, mi: &Index) -> io::Result<BufWriter<File>> {
    let path = format!("{}.{:04}.tmp", prefix, part_index);
    let f = File::create(&path).map_err(|e| {
        io::Error::new(e.kind(), format!("failed to write to temporary file '{}': {}", path, e))
    })?;
    let mut w = BufWriter::new(f);
    w.write_all(&(mi.kmer_size as u32).to_le_bytes())?;
    w.write_all(&(mi.seqs.len() as u32).to_le_bytes())?;
    for s in &mi.seqs {
        let name_bytes = s.name.as_bytes();
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&(s.len as u32).to_le_bytes())?;
    }
    Ok(w)
}

/// Write one query's results to a temp file.
/// Per-query format: n_results(i32), rep_len(i32), frag_gap(i32),
/// then for each result: bincode-serialized AlnResult.
pub fn split_write_query(
    w: &mut impl Write,
    results: &[AlnResult],
    rep_len: i32,
    frag_gap: i32,
) -> io::Result<()> {
    w.write_all(&(results.len() as i32).to_le_bytes())?;
    w.write_all(&rep_len.to_le_bytes())?;
    w.write_all(&frag_gap.to_le_bytes())?;
    for r in results {
        let encoded = bincode::serialize(r).map_err(|e| {
            io::Error::other(format!("bincode serialize error: {}", e))
        })?;
        w.write_all(&(encoded.len() as u32).to_le_bytes())?;
        w.write_all(&encoded)?;
    }
    Ok(())
}

/// Read headers from all part files, build global seqs list + rid_shifts.
/// Returns (k, all_seqs, rid_shifts, readers).
pub fn split_merge_prep(
    prefix: &str,
    n_parts: usize,
) -> io::Result<(usize, Vec<TargetSequence>, Vec<usize>, Vec<BufReader<File>>)> {
    let mut readers = Vec::with_capacity(n_parts);
    let mut n_seq_parts = Vec::with_capacity(n_parts);
    let mut k_val = 0usize;

    for i in 0..n_parts {
        let path = format!("{}.{:04}.tmp", prefix, i);
        let f = File::open(&path).map_err(|e| {
            io::Error::new(e.kind(), format!("failed to open temporary file '{}': {}", path, e))
        })?;
        let mut r = BufReader::new(f);

        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let k = u32::from_le_bytes(buf4) as usize;
        if i == 0 { k_val = k; }

        r.read_exact(&mut buf4)?;
        let n_seq = u32::from_le_bytes(buf4) as usize;
        n_seq_parts.push(n_seq);
        readers.push(r);
    }

    // Compute rid_shifts
    let mut rid_shifts = vec![0usize; n_parts];
    for i in 1..n_parts {
        rid_shifts[i] = rid_shifts[i - 1] + n_seq_parts[i - 1];
    }

    // Read all sequence headers
    let total_seqs: usize = n_seq_parts.iter().sum();
    let mut all_seqs = Vec::with_capacity(total_seqs);
    for (i, reader) in readers.iter_mut().enumerate() {
        for _ in 0..n_seq_parts[i] {
            let mut buf4 = [0u8; 4];
            reader.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8 in seq name: {}", e))
            })?;
            reader.read_exact(&mut buf4)?;
            let seq_len = u32::from_le_bytes(buf4) as usize;
            all_seqs.push(TargetSequence {
                name,
                len: seq_len,
                offset: 0,
                is_alt: false,
            });
        }
    }

    Ok((k_val, all_seqs, rid_shifts, readers))
}

/// Read one query's results from a part file. Returns None on EOF.
pub fn split_read_query(r: &mut impl Read) -> io::Result<Option<(Vec<AlnResult>, i32, i32)>> {
    let mut buf4 = [0u8; 4];
    match r.read_exact(&mut buf4) {
        Ok(()) => {},
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n_results = i32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?;
    let rep_len = i32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?;
    let frag_gap = i32::from_le_bytes(buf4);

    let mut results = Vec::with_capacity(n_results as usize);
    for _ in 0..n_results {
        r.read_exact(&mut buf4)?;
        let enc_len = u32::from_le_bytes(buf4) as usize;
        let mut enc_buf = vec![0u8; enc_len];
        r.read_exact(&mut enc_buf)?;
        let aln: AlnResult = bincode::deserialize(&enc_buf).map_err(|e| {
            io::Error::other(format!("bincode deserialize error: {}", e))
        })?;
        results.push(aln);
    }

    Ok(Some((results, rep_len, frag_gap)))
}

/// Delete temp files.
pub fn split_rm_tmp(prefix: &str, n_parts: usize) {
    for i in 0..n_parts {
        let path = format!("{}.{:04}.tmp", prefix, i);
        let _ = std::fs::remove_file(&path);
    }
}

/// Configuration for split-index merge.
pub struct MergeConfig<'a> {
    pub prefix: &'a str,
    pub n_parts: usize,
    pub k: usize,
    pub w: usize,
    pub is_hpc: bool,
    pub pe_mode: bool,
    pub pe_flip: Option<(bool, bool)>,
    pub query_path: &'a str,
    pub query2_path: Option<&'a str>,
    pub two_file_pe: bool,
    pub sam_rg: Option<&'a str>,
    pub copy_comment: bool,
}

/// Merge results from all split parts and output.
/// Merge alignment results from multiple index parts and output.
pub fn merge_split_results(
    config: &MergeConfig,
    opt: &MapOptions,
    out_cfg: &OutputConfig,
    handle: &mut dyn Write,
) -> anyhow::Result<AlignmentStats> {
    let prefix = config.prefix;
    let n_parts = config.n_parts;
    let k = config.k;
    let w = config.w;
    let is_hpc = config.is_hpc;
    let pe_mode = config.pe_mode;
    let pe_flip = config.pe_flip;
    let query_path = config.query_path;
    let query2_path = config.query2_path;
    let two_file_pe = config.two_file_pe;
    let sam_rg = config.sam_rg;
    let copy_comment = config.copy_comment;
    // 1. Read all temp file headers, build global seqs + rid_shifts
    let (k_from_file, all_seqs, rid_shifts, mut readers) = split_merge_prep(prefix, n_parts)
        .map_err(|e| anyhow::anyhow!("Split merge prep failed: {}", e))?;

    let _ = k_from_file; // k already known from caller

    // 2. Build header-only index for output formatting
    let global_mi = Index::header_only(k, w, is_hpc, all_seqs);

    // 3. Write SAM header if SAM output
    if out_cfg.output_sam {
        writeln!(handle, "@HD\tVN:1.6\tSO:unsorted\tGO:query")?;
        for s in &global_mi.seqs {
            writeln!(handle, "@SQ\tSN:{}\tLN:{}", s.name, s.len)?;
        }
        let version = env!("CARGO_PKG_VERSION");
        let cmd_line = std::env::args().collect::<Vec<String>>().join(" ");
        writeln!(handle, "@PG\tID:rammap\tPN:rammap\tVN:{}\tCL:{}", version, cmd_line)?;
        if let Some(rg_line) = sam_rg {
            let expanded = rg_line.replace("\\t", "\t");
            writeln!(handle, "{}", expanded)?;
        }
    }

    // 4. Re-open query files for output
    let query_reader = crate::fasta::open(query_path)
        .map_err(|e| anyhow::anyhow!("Error re-opening query for merge: {}", e))?;
    let query_reader2 = if two_file_pe {
        let q2 = query2_path.ok_or_else(|| anyhow::anyhow!("PE merge requires query2 path"))?;
        Some(crate::fasta::open(q2)
            .map_err(|e| anyhow::anyhow!("Error re-opening query2 for merge: {}", e))?)
    } else {
        None
    };

    let mut total_stats = AlignmentStats::default();
    let mut record_iter = query_reader.records();
    let mut record_iter2 = query_reader2.map(|r| r.records());
    let mut output_buffer = String::new();

    if pe_mode {
        let (flip_r1, flip_r2) = pe_flip.unwrap_or((false, false));
        loop {
            // Read R1
            let rec1 = match record_iter.next() {
                Some(Ok(r)) => r,
                Some(Err(e)) => { eprintln!("Warning: Error reading record: {}", e); continue; },
                None => break,
            };
            // Read R2
            let rec2 = if two_file_pe {
                match record_iter2.as_mut().unwrap().next() {
                    Some(Ok(r)) => r,
                    Some(Err(e)) => { eprintln!("Warning: Error reading R2 record: {}", e); continue; },
                    None => break,
                }
            } else {
                match record_iter.next() {
                    Some(Ok(r)) => r,
                    Some(Err(e)) => { eprintln!("Warning: Error reading R2 record: {}", e); continue; },
                    None => break,
                }
            };

            let qname1 = rec1.name().to_string();
            let qseq1 = rec1.sequence().to_vec();
            let qual1 = rec1.quality().map(|qs| String::from_utf8_lossy(qs).to_string());
            let comment1 = if copy_comment { rec1.description().map(|s| s.to_string()) } else { None };
            let qname2 = rec2.name().to_string();
            let qseq2 = rec2.sequence().to_vec();
            let qual2 = rec2.quality().map(|qs| String::from_utf8_lossy(qs).to_string());
            let comment2 = if copy_comment { rec2.description().map(|s| s.to_string()) } else { None };

            // Read results from all parts for both segments
            let mut all_results1: Vec<AlnResult> = Vec::new();
            let mut all_results2: Vec<AlnResult> = Vec::new();
            let mut max_rep_len = 0i32;
            let mut frag_gap = opt.chaining.max_gap_ref;

            for (j, reader) in readers.iter_mut().enumerate() {
                // Segment 1
                if let Some((mut results, rep_len, fg)) = split_read_query(reader)
                    .map_err(|e| anyhow::anyhow!("Error reading split part {}: {}", j, e))? {
                    for r in results.iter_mut() { r.ref_id += rid_shifts[j]; }
                    all_results1.extend(results);
                    if rep_len > max_rep_len { max_rep_len = rep_len; }
                    if j == 0 { frag_gap = fg; }
                }
                // Segment 2
                if let Some((mut results, rep_len, _fg)) = split_read_query(reader)
                    .map_err(|e| anyhow::anyhow!("Error reading split part {} seg2: {}", j, e))? {
                    for r in results.iter_mut() { r.ref_id += rid_shifts[j]; }
                    all_results2.extend(results);
                    if rep_len > max_rep_len { max_rep_len = rep_len; }
                }
            }

            // Refilter each segment
            let qseq1_work = if flip_r1 { crate::align::extend::rev_comp(&qseq1) } else { qseq1.clone() };
            let qseq2_work = if flip_r2 { crate::align::extend::rev_comp(&qseq2) } else { qseq2.clone() };
            let mut pq1 = refilter_merged_results(all_results1, opt, k, qseq1_work.len(), max_rep_len, out_cfg.do_cigar);
            let mut pq2 = refilter_merged_results(all_results2, opt, k, qseq2_work.len(), max_rep_len, out_cfg.do_cigar);
            // merge doesn't update rep_len, so rl:i: tag is 0
            pq1.rep_len = 0;
            pq2.rep_len = 0;

            // Run pair-rescue / TLEN inference across the two mates.
            if out_cfg.do_cigar && opt.pairing.pe_ori >= 0 {
                let qlens = [qseq1_work.len() as i32, qseq2_work.len() as i32];
                let sub_diff = opt.scoring.match_score * 2 + opt.scoring.mismatch_penalty;
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

                for (s, seg_pe_regs) in pe_regs.iter().enumerate() {
                    let pq = if s == 0 { &mut pq1 } else { &mut pq2 };
                    for pr in seg_pe_regs {
                        let i = pr.id;
                        if i < pq.results.len() {
                            pq.results[i].proper_frag = pr.proper_frag;
                            pq.mapqs[i] = pr.mapq;
                            pq.sam_pri[i] = pr.sam_pri;
                            pq.results[i].is_secondary = pr.parent != pr.id;
                        }
                    }
                }
            }

            // Post-flip coordinates
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

            // Format output
            let mate1 = get_mate_info(&pq1);
            let mate2 = get_mate_info(&pq2);
            output_buffer.clear();
            let ri1 = ReadInfo { qname: &qname1, qseq: &qseq1, qual: qual1.as_deref(), comment: comment1.as_deref(), n_seg: 2, seg_idx: 0 };
            let ri2 = ReadInfo { qname: &qname2, qseq: &qseq2, qual: qual2.as_deref(), comment: comment2.as_deref(), n_seg: 2, seg_idx: 1 };
            format_output(&mut output_buffer, opt, &global_mi, &ri1, &pq1, out_cfg, mate2.as_ref());
            format_output(&mut output_buffer, opt, &global_mi, &ri2, &pq2, out_cfg, mate1.as_ref());
            write!(handle, "{}", output_buffer)?;
            total_stats = total_stats + pq1.stats + pq2.stats;
        }
    } else {
        // Single-end merge
        loop {
            let rec = match record_iter.next() {
                Some(Ok(r)) => r,
                Some(Err(e)) => { eprintln!("Warning: Error reading record: {}", e); continue; },
                None => break,
            };

            let qname = rec.name().to_string();
            let qseq = rec.sequence().to_vec();
            let qual = rec.quality().map(|qs| String::from_utf8_lossy(qs).to_string());
            let comment = if copy_comment { rec.description().map(|s| s.to_string()) } else { None };
            let qlen = qseq.len();

            // Read results from all parts
            let mut all_results: Vec<AlnResult> = Vec::new();
            let mut max_rep_len = 0i32;

            for (j, reader) in readers.iter_mut().enumerate() {
                if let Some((mut results, rep_len, _frag_gap)) = split_read_query(reader)
                    .map_err(|e| anyhow::anyhow!("Error reading split part {}: {}", j, e))? {
                    for r in results.iter_mut() {
                        r.ref_id += rid_shifts[j];
                    }
                    all_results.extend(results);
                    if rep_len > max_rep_len { max_rep_len = rep_len; }
                }
            }

            // Refilter (uses max_rep_len for MAPQ, matching mm_set_mapq2 in merge_hits)
            let mut pq = refilter_merged_results(all_results, opt, k, qlen, max_rep_len, out_cfg.do_cigar);
            // merge doesn't update rep_len, so rl:i: tag is 0
            pq.rep_len = 0;

            // Format output
            output_buffer.clear();
            let ri = ReadInfo { qname: &qname, qseq: &qseq, qual: qual.as_deref(), comment: comment.as_deref(), n_seg: 1, seg_idx: 0 };
            format_output(&mut output_buffer, opt, &global_mi, &ri, &pq, out_cfg, None);
            write!(handle, "{}", output_buffer)?;
            total_stats = total_stats + pq.stats;
        }
    }

    // Clean up temp files
    split_rm_tmp(prefix, n_parts);

    Ok(total_stats)
}
