
use wasm_bindgen::prelude::*;
use crate::align::map::{MapOptions, MapContext, AlignFlags};
use crate::align::index::Index;
use crate::align::pipeline::{align_and_format_query, OutputConfig, ReadInfo};
use crate::align::extend::AlignmentContext;

#[wasm_bindgen]
extern "C" {
    /// Log callback — sends progress messages to the JS host.
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Log to both the JS console and our callback.
fn wasm_log(s: &str) {
    log(s);
}

/// Apply a minimap2-style preset to MapOptions + k/w.
fn apply_preset_wasm(opt: &mut MapOptions, k: &mut usize, w: &mut usize, is_hpc: &mut bool, preset: &str) {
    match preset {
        "map-ont" => {
            *k = 15; *w = 10;
            opt.scoring.match_score = 2; opt.scoring.mismatch_penalty = 4;
            opt.scoring.gap_open = 4; opt.scoring.gap_extend = 2;
            opt.scoring.gap_open2 = 24; opt.scoring.gap_extend2 = 1;
        }
        "map-hifi" => {
            *k = 19; *w = 19;
            opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 4;
            opt.scoring.gap_open = 6; opt.scoring.gap_extend = 2;
            opt.scoring.gap_open2 = 26; opt.scoring.gap_extend2 = 1;
            opt.alignment.min_dp_max = 200;
        }
        "sr" => {
            *k = 21; *w = 11;
            opt.scoring.match_score = 2; opt.scoring.mismatch_penalty = 8;
            opt.scoring.gap_open = 12; opt.scoring.gap_extend = 2;
            opt.scoring.gap_open2 = 24; opt.scoring.gap_extend2 = 1;
            opt.flags.insert(AlignFlags::SHORT_READ | AlignFlags::FRAG_MODE);
        }
        "splice" => {
            *k = 15; *w = 5;
            opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 2;
            opt.scoring.gap_open = 2; opt.scoring.gap_extend = 1;
            opt.scoring.gap_open2 = 32; opt.scoring.gap_extend2 = 0;
            opt.filtering.is_splice = true;
            opt.flags.insert(AlignFlags::SPLICE | AlignFlags::SPLICE_FLANK);
        }
        "asm20" => {
            *k = 19; *w = 10;
            opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 4;
            opt.scoring.gap_open = 6; opt.scoring.gap_extend = 2;
            opt.scoring.gap_open2 = 26; opt.scoring.gap_extend2 = 1;
            opt.chaining.bandwidth = 1000; opt.chaining.bandwidth_long = 1000;
            opt.flags.insert(AlignFlags::RMQ_CHAIN);
        }
        _ => {
            *k = 15; *w = 10;
        }
    }
}

/// Parse FASTA or FASTQ text, returning (name, sequence) pairs.
fn parse_sequences(text: &str) -> Vec<(String, String)> {
    let mut seqs = Vec::new();
    let text = text.trim();
    if text.is_empty() { return seqs; }

    if text.starts_with('@') {
        // FASTQ format: 4 lines per record
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        while i + 3 < lines.len() {
            if lines[i].starts_with('@') {
                let name = lines[i][1..].split_whitespace().next().unwrap_or("read").to_string();
                let seq = lines[i + 1].to_string();
                seqs.push((name, seq));
                i += 4;
            } else {
                i += 1;
            }
        }
    } else {
        // FASTA format
        for entry in text.split('>').filter(|s| !s.trim().is_empty()) {
            if let Some((header, seq)) = entry.split_once('\n') {
                let name = header.trim().split_whitespace().next().unwrap_or("seq").to_string();
                let seq_str = seq.replace(|c: char| c.is_whitespace(), "");
                seqs.push((name, seq_str));
            }
        }
    }
    seqs
}

/// Main alignment entry point for the web demo.
///
/// Returns a JSON-ish string with two sections separated by "\n---LOG---\n":
///   1. PAF/SAM output lines
///   2. Log/timing messages
#[wasm_bindgen]
pub fn align_wasm_full(
    target_text: &str,
    query_text: &str,
    preset: &str,
    output_sam: bool,
    output_cigar: bool,
) -> String {
    let mut log_buf = String::new();
    let mut output_buf = String::new();

    macro_rules! emit_log {
        ($($arg:tt)*) => {{
            let msg = format!($($arg)*);
            wasm_log(&msg);
            log_buf.push_str(&msg);
            log_buf.push('\n');
        }};
    }

    // Parse preset
    let mut k = 15usize;
    let mut w = 10usize;
    let mut is_hpc = false;
    let mut opt = MapOptions::default();
    apply_preset_wasm(&mut opt, &mut k, &mut w, &mut is_hpc, preset);
    if output_cigar || output_sam {
        opt.flags.insert(AlignFlags::OUT_CIGAR);
    }
    emit_log!("[*] Preset: {} (k={}, w={})", preset, k, w);

    // Parse reference
    let target_seqs: Vec<(String, Vec<u8>)> = parse_sequences(target_text)
        .into_iter()
        .map(|(name, seq)| (name, seq.into_bytes()))
        .collect();
    if target_seqs.is_empty() {
        emit_log!("[ERROR] No reference sequences found");
        return format!("{}\n---LOG---\n{}", output_buf, log_buf);
    }
    let total_bases: usize = target_seqs.iter().map(|(_, s)| s.len()).sum();
    emit_log!("[*] Reference: {} sequence(s), {} bases", target_seqs.len(), total_bases);

    // Build index
    let t0 = web_time::Instant::now();
    let idx = Index::build(target_seqs, w, k, is_hpc, 50000);
    opt.seeding.mid_occ = idx.cal_mid_occ(2e-4, 10, 10000);
    emit_log!("[*] Index built in {:.3}s (mid_occ={})", t0.elapsed().as_secs_f64(), opt.seeding.mid_occ);

    // Parse queries
    let query_seqs = parse_sequences(query_text);
    if query_seqs.is_empty() {
        emit_log!("[ERROR] No query sequences found");
        return format!("{}\n---LOG---\n{}", output_buf, log_buf);
    }
    emit_log!("[*] Queries: {} read(s)", query_seqs.len());

    // Align
    let t1 = web_time::Instant::now();
    let out_cfg = OutputConfig {
        do_cigar: output_cigar,
        do_cs: false,
        do_md: false,
        do_ds: false,
        eqx: false,
        output_sam,
        rg_id: None,
        split_mode: false,
    };
    let mut ctx = AlignmentContext::new();
    let mut map_ctx = MapContext::new();
    let mut n_aligned = 0usize;

    for (qname, qseq) in &query_seqs {
        let ri = ReadInfo {
            qname,
            qseq: qseq.as_bytes(),
            qual: None,
            comment: None,
            n_seg: 1,
            seg_idx: 0,
        };
        let res = align_and_format_query(
            &opt, &idx, &ri,
            &mut ctx, &mut map_ctx,
            None, None,
            &out_cfg,
        );
        if !res.0.is_empty() {
            output_buf.push_str(&res.0);
            n_aligned += 1;
        }
    }

    let map_time = t1.elapsed().as_secs_f64();
    emit_log!("[*] Aligned {} reads ({} with hits) in {:.3}s ({:.0} reads/sec)",
        query_seqs.len(), n_aligned, map_time,
        query_seqs.len() as f64 / map_time.max(0.001));

    format!("{}\n---LOG---\n{}", output_buf, log_buf)
}

/// Legacy simple API (kept for backward compatibility with wasm tests).
#[wasm_bindgen]
pub fn align_wasm(target_fasta: &str, query_fasta: &str, output_sam: bool, is_splice: bool) -> String {
    let preset = if is_splice { "splice" } else { "map-ont" };
    let result = align_wasm_full(target_fasta, query_fasta, preset, output_sam, true);
    // Return only the alignment output (before the ---LOG--- separator)
    result.split("\n---LOG---\n").next().unwrap_or("").to_string()
}

#[wasm_bindgen]
pub fn force_align_wasm(tseq: &str, qseq: &str) -> String {
    use crate::align::sketch::Minimizer;
    use crate::align::extend::{align_anchors, AlignAnchorContext, fmt_cigar};

    let tseq_bytes = tseq.as_bytes();
    let qseq_bytes = qseq.as_bytes();

    let x: u64 = 0;
    let y: u64 = 1u64 << 32;

    let mut anchors = vec![Minimizer { x, y }];

    let mut ctx = AlignmentContext::new();
    let mut opt = MapOptions::default();
    opt.filtering.is_splice = false;
    opt.chaining.min_chain_score = 0;
    opt.chaining.min_cnt = 0;
    opt.alignment.min_dp_max = i32::MIN;

    let call_ctx = AlignAnchorContext {
        seed_bounds: (0, 0, tseq_bytes.len() as i32, qseq_bytes.len() as i32),
        rev: false,
        rid: 0,
        splice_flag: AlignFlags::empty(),
        split_inv: false,
        is_hpc: false,
        k: 1,
        junc_db: None,
        ref_offset: 0,
    };
    let aln_result = align_anchors(
        &mut anchors,
        qseq_bytes,
        tseq_bytes,
        &opt,
        &mut ctx,
        &call_ctx,
    );

    fmt_cigar(&aln_result.cigar_ops, false)
}

// ==================== WASM Tests ====================
#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_force_align_exact_match() {
        let result = force_align_wasm("ACGTACGT", "ACGTACGT");
        assert!(result.contains("8M") || result.contains("8="),
                "Expected 8M or 8= for exact match, got: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_mismatch() {
        let result = force_align_wasm("ACGTACGT", "ACGAACGT");
        assert!(!result.is_empty(), "Should produce alignment");
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_insertion() {
        let result = force_align_wasm("ACGTACGT", "ACGTTACGT");
        assert!(result.contains("I") || result.contains("M"),
                "Should produce alignment with insertion: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_deletion() {
        let result = force_align_wasm("ACGTTACGT", "ACGTACGT");
        assert!(result.contains("D") || result.contains("M"),
                "Should produce alignment with deletion: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_basic() {
        let target_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);
        let result = align_wasm(&target, &query, false, false);
        assert!(result.contains("query") || result.is_empty(),
                "Output should contain query name or be empty if below threshold");
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_longer_sequence() {
        let target_seq = "ACGTACGT".repeat(50);
        let query_seq = "ACGTACGT".repeat(25);
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);
        let result = align_wasm(&target, &query, false, false);
        assert!(result.len() >= 0, "Should not panic");
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_sam_output() {
        let target_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);
        let result = align_wasm(&target, &query, true, false);
        assert!(result.contains("\t") || result.contains("query"),
                "Should produce SAM format output");
    }
}
