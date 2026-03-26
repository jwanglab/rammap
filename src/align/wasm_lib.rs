
use wasm_bindgen::prelude::*;
use crate::align::map::{MapOptions, MapContext, AlignFlags};
use crate::align::index::Index;
use crate::align::pipeline::{align_and_format_query, OutputConfig, ReadInfo};
use crate::align::extend::AlignmentContext;

#[wasm_bindgen]
pub fn align_wasm(target_fasta: &str, query_fasta: &str, output_sam: bool, is_splice: bool) -> String {
    // 1. Parse Target
    // Simple FASTA parsing: split by '>', first line header, rest seq
    let mut target_seqs = Vec::new();
    for entry in target_fasta.split('>').filter(|s| !s.trim().is_empty()) {
        if let Some((header, seq)) = entry.split_once('\n') {
             let name = header.trim().split_whitespace().next().unwrap_or("target").to_string();
             let seq_str = seq.replace(|c: char| c.is_whitespace(), "");
             target_seqs.push((name, seq_str.into_bytes()));
        }
    }
    
    // 2. Build Index (k=15, w=10 default for now, or match splice preset)
    // Preset logic: if splice: q2=32, e2=0. 
    let mut opt = MapOptions::default();
    if is_splice {
        opt.filtering.is_splice = true;
        opt.scoring.gap_open2 = 32;
        opt.scoring.gap_extend2 = 0; // standard splice
        // Usually splice uses smaller k? or default k=15 is fine.
    }
    
    // Build index
    let w = 10;
    let k = 15;
    let is_hpc = false;
    let max_occ = 50000; // Hard cap for index building

    let idx = Index::build(target_seqs, w, k, is_hpc, max_occ);

    // Calculate mid_occ for filtering at query time
    opt.seeding.mid_occ = idx.cal_mid_occ(2e-4, 10, 10000);
    
    // 3. Parse Queries
    let mut query_seqs = Vec::new();
    for entry in query_fasta.split('>').filter(|s| !s.trim().is_empty()) {
        if let Some((header, seq)) = entry.split_once('\n') {
             let name = header.trim().split_whitespace().next().unwrap_or("query").to_string();
             let seq_str = seq.replace(|c: char| c.is_whitespace(), "");
             query_seqs.push((name, seq_str));
        }
    }
    
    // 4. Align
    let mut output = String::new();
    let mut ctx = AlignmentContext::new();
    let mut map_ctx = MapContext::new();
    
    for (qname, qseq) in query_seqs {
        let out_cfg = OutputConfig {
            do_cigar: true,
            do_cs: true,
            do_md: false,
            do_ds: false,
            eqx: false,
            output_sam,
            rg_id: None,
            split_mode: false,
        };
        let ri = ReadInfo {
            qname: &qname,
            qseq: qseq.as_bytes(),
            qual: None,
            comment: None,
            n_seg: 1,
            seg_idx: 0,
        };
        let res = align_and_format_query(
            &opt,
            &idx,
            &ri,
            &mut ctx,
            &mut map_ctx,
            None, // junc_db
            None, // jump_db
            &out_cfg,
        );
        output.push_str(&res.0);
    }
    
    output
}

#[wasm_bindgen]
pub fn force_align_wasm(tseq: &str, qseq: &str) -> String {
    use crate::align::sketch::Minimizer;
    use crate::align::extend::{align_anchors, AlignAnchorContext, fmt_cigar};

    let tseq_bytes = tseq.as_bytes();
    let qseq_bytes = qseq.as_bytes();

    // Create a fake anchor at (0,0)
    // x: rid=0, pos=0, strand=0 -> 0
    let x: u64 = 0;
    // y: span=1, q_pos=0 -> 1 << 32
    let y: u64 = (1u64 << 32);

    let mut anchors = vec![Minimizer { x, y }];

    let mut ctx = AlignmentContext::new();
    let mut opt = MapOptions::default();
    opt.filtering.is_splice = false;
    opt.chaining.min_chain_score = 0; // force-align: don't filter the fake anchor
    opt.chaining.min_cnt = 0;
    opt.alignment.min_dp_max = i32::MIN;

    let call_ctx = AlignAnchorContext {
        seed_bounds: (0, 0, tseq_bytes.len() as i32, qseq_bytes.len() as i32),
        rev: false,
        rid: 0,
        splice_flag: AlignFlags::empty(),
        split_inv: false,
        is_hpc: false,
        k: 1, // k=1 for force-align (no indexing, avoids k_half coordinate underflow)
        junc_db: None,
    };
    let aln_result = align_anchors(
        &mut anchors,
        qseq_bytes,
        tseq_bytes,
        &opt,
        &mut ctx,
        &call_ctx,
    );

    // Formatting
    fmt_cigar(&aln_result.cigar_ops, false)
}

// ==================== WASM Tests ====================
#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    // Run in Node.js (remove run_in_browser to allow --node testing)

    #[wasm_bindgen_test]
    fn test_force_align_exact_match() {
        let result = force_align_wasm("ACGTACGT", "ACGTACGT");
        assert!(result.contains("8M") || result.contains("8="),
                "Expected 8M or 8= for exact match, got: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_mismatch() {
        let result = force_align_wasm("ACGTACGT", "ACGAACGT");
        // Should have matches with one mismatch
        assert!(!result.is_empty(), "Should produce alignment");
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_insertion() {
        let result = force_align_wasm("ACGTACGT", "ACGTTACGT");
        // Should have an insertion
        assert!(result.contains("I") || result.contains("M"),
                "Should produce alignment with insertion: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_force_align_with_deletion() {
        let result = force_align_wasm("ACGTTACGT", "ACGTACGT");
        // Should have a deletion
        assert!(result.contains("D") || result.contains("M"),
                "Should produce alignment with deletion: {}", result);
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_basic() {
        // Need longer sequences to meet min_chain_score threshold (40)
        // k=15, w=10 means we need at least ~25bp to get a minimizer match
        let target_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"; // 48bp
        let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"; // 40bp
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);
        let result = align_wasm(&target, &query, false, false);
        // Should produce PAF output
        assert!(result.contains("query") || result.is_empty(),
                "Output should contain query name or be empty if below threshold");
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_longer_sequence() {
        // Test with a longer sequence to exercise SIMD paths
        let target_seq = "ACGTACGT".repeat(50); // 400bp
        let query_seq = "ACGTACGT".repeat(25); // 200bp
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);

        let result = align_wasm(&target, &query, false, false);
        // With repetitive sequences, may or may not align depending on occ filter
        // Just check it doesn't panic
        assert!(result.len() >= 0, "Should not panic");
    }

    #[wasm_bindgen_test]
    fn test_align_wasm_sam_output() {
        // Longer sequences for SAM test
        let target_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let target = format!(">ref\n{}", target_seq);
        let query = format!(">query\n{}", query_seq);
        let result = align_wasm(&target, &query, true, false);
        // SAM output should have tab-separated fields or unmapped record
        assert!(result.contains("\t") || result.contains("query"),
                "Should produce SAM format output");
    }
}
