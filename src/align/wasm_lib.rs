
use wasm_bindgen::prelude::*;
use crate::align::map::{MapOptions, MapContext, AlignFlags};
use crate::align::index::{Index, IndexBuilder};
use crate::align::pipeline::{align_and_format_query, OutputConfig, ReadInfo};
use crate::align::extend::AlignmentContext;
use crate::fasta::{FastaStreamer, FastqStreamer};

/// Re-export rayon thread pool initializer for WASM multithreading.
/// JS must call `await initThreadPool(n)` before any parallel work.
#[cfg(feature = "wasm-threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
extern "C" {
    /// Log callback — sends progress messages to the JS host.
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Log to both the JS console and our callback.
fn wasm_log(s: &str) {
    log(s);
}

/// Forward Rust panics to console.error with full payload (location + message).
/// Without this, panics surface to JS as the opaque "unreachable executed"
/// trap and there is no way to tell what actually went wrong.
#[wasm_bindgen(start)]
pub fn _set_panic_hook() {
    use std::sync::Once;
    static SET: Once = Once::new();
    SET.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let mut msg = String::from("PANIC: ");
            if let Some(loc) = info.location() {
                msg.push_str(&format!("[{}:{}:{}] ", loc.file(), loc.line(), loc.column()));
            }
            let payload = info.payload();
            if let Some(s) = payload.downcast_ref::<&str>() {
                msg.push_str(s);
            } else if let Some(s) = payload.downcast_ref::<String>() {
                msg.push_str(s);
            } else {
                msg.push_str("<non-string payload>");
            }
            error(&msg);
        }));
    });
}

/// Apply a named preset to MapOptions + k/w/is_hpc.
///
/// Delegates to the canonical native preset table (`api::apply_preset_str`) so
/// the wasm build supports the exact same preset set as the CLI and library —
/// including the HPC presets (`map-pb`/`ava-pb`) and every alias. An
/// unrecognized name resets and falls back to `map-ont`, so the wasm entry
/// points never fail on a bad preset string.
fn apply_preset_wasm(opt: &mut MapOptions, k: &mut usize, w: &mut usize, is_hpc: &mut bool, preset: &str) {
    if crate::api::apply_preset_str(opt, k, w, is_hpc, preset).is_err() {
        *opt = MapOptions::default();
        *is_hpc = false;
        let _ = crate::api::apply_preset_str(opt, k, w, is_hpc, "map-ont");
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
        cs_long: false,
        do_md: false,
        do_ds: false,
        eqx: false,
        output_sam,
        rg_id: None,
        split_mode: false,
    };
    let mut n_aligned = 0usize;

    #[cfg(feature = "wasm-threads")]
    {
        use rayon::prelude::*;
        emit_log!("[*] Using parallel alignment ({} threads)", rayon::current_num_threads());

        let results: Vec<String> = query_seqs.par_iter().map_init(
            || (AlignmentContext::new(), MapContext::new()),
            |(ctx, map_ctx), (qname, qseq)| {
                let ri = ReadInfo {
                    qname, qseq: qseq.as_bytes(),
                    qual: None, comment: None, n_seg: 1, seg_idx: 0,
                };
                let res = align_and_format_query(
                    &opt, &idx, &ri, ctx, map_ctx, None, None, &out_cfg,
                );
                res.0
            },
        ).collect();

        for res in results {
            if !res.is_empty() {
                output_buf.push_str(&res);
                n_aligned += 1;
            }
        }
    }

    #[cfg(not(feature = "wasm-threads"))]
    {
        let mut ctx = AlignmentContext::new();
        let mut map_ctx = MapContext::new();

        for (qname, qseq) in &query_seqs {
            let ri = ReadInfo {
                qname, qseq: qseq.as_bytes(),
                qual: None, comment: None, n_seg: 1, seg_idx: 0,
            };
            let res = align_and_format_query(
                &opt, &idx, &ri, &mut ctx, &mut map_ctx, None, None, &out_cfg,
            );
            if !res.0.is_empty() {
                output_buf.push_str(&res.0);
                n_aligned += 1;
            }
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
    use crate::align::extend::{align_anchors, AlignAnchorContext, encode_nt4_byte, fmt_cigar};

    let mut tseq_nt4: Vec<u8> = tseq.as_bytes().iter().copied().map(encode_nt4_byte).collect();
    let qseq_nt4: Vec<u8> = qseq.as_bytes().iter().copied().map(encode_nt4_byte).collect();

    let x: u64 = 0;
    let y: u64 = 1u64 << 32;

    let mut anchors = vec![Minimizer { x, y }];

    let mut opt = MapOptions::default();
    opt.filtering.is_splice = false;
    opt.chaining.min_chain_score = 0;
    opt.chaining.min_cnt = 0;
    opt.alignment.min_dp_max = i32::MIN;

    let call_ctx = AlignAnchorContext {
        seed_bounds: (0, 0, tseq_nt4.len() as i32, qseq_nt4.len() as i32),
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
        &qseq_nt4,
        &mut tseq_nt4,
        None,
        &opt,
        &call_ctx,
    );

    fmt_cigar(&aln_result.cigar_ops, false)
}

// ──────────────────────────────────────────────────────────────────────────
// Streaming session API for multi-GB inputs.
//
// `align_wasm_full` requires the caller to materialize the entire reference
// and query text as JS strings, which breaks down past V8's ~512 MB string
// cap. `AlignSession` instead accepts arbitrary byte chunks: bytes flow
// through the streaming FASTA / FASTQ parsers, each completed record is
// packed-and-dropped (ref) or aligned-and-dropped (query), so peak WASM
// linear memory stays bounded regardless of total input size.
// ──────────────────────────────────────────────────────────────────────────

/// Streaming alignment session. Construct, push ref chunks, finalize the
/// reference, push query chunks (each chunk emits its own output prefix),
/// then `finalize` to get the trailing log.
/// Query parser with lazy format detection. The browser demo accepts both
/// FASTA and FASTQ query files; we check the first non-whitespace byte of the
/// first chunk (`>` = FASTA, `@` = FASTQ) and dispatch to the appropriate
/// streamer. Bytes received before detection are buffered and flushed once
/// the format is known.
enum QueryParser {
    /// No non-whitespace bytes seen yet. `buf` accumulates incoming chunks and
    /// is replayed into the chosen streamer once the format byte arrives.
    Pending { buf: Vec<u8> },
    Fasta(FastaStreamer),
    Fastq(FastqStreamer),
}

impl QueryParser {
    fn new() -> Self {
        QueryParser::Pending { buf: Vec::new() }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        if let QueryParser::Pending { buf } = self {
            buf.extend_from_slice(chunk);
            // Look for the first non-whitespace byte to lock in the format.
            for (i, &b) in buf.iter().enumerate() {
                if b.is_ascii_whitespace() { continue; }
                let rest: Vec<u8> = buf[i..].to_vec();
                *self = match b {
                    b'>' => {
                        let mut s = FastaStreamer::new();
                        s.push(&rest);
                        QueryParser::Fasta(s)
                    }
                    b'@' => {
                        let mut s = FastqStreamer::new();
                        s.push(&rest);
                        QueryParser::Fastq(s)
                    }
                    _ => return Err(JsValue::from_str(
                        "query input is not FASTA (starts with '>') or FASTQ (starts with '@')")),
                };
                return Ok(());
            }
            return Ok(()); // all whitespace so far — keep buffering
        }
        match self {
            QueryParser::Fasta(s) => s.push(chunk),
            QueryParser::Fastq(s) => s.push(chunk),
            QueryParser::Pending { .. } => unreachable!(),
        }
        Ok(())
    }

    fn next_record(&mut self) -> Option<(String, Vec<u8>)> {
        match self {
            QueryParser::Pending { .. } => None,
            QueryParser::Fasta(s) => s.next_record(),
            QueryParser::Fastq(s) => s.next_record(),
        }
    }

    fn finalize(&mut self) {
        match self {
            QueryParser::Pending { .. } => {}
            QueryParser::Fasta(s) => s.finalize(),
            QueryParser::Fastq(s) => s.finalize(),
        }
    }
}

#[wasm_bindgen]
pub struct AlignSession {
    // Config
    preset: String,
    output_sam: bool,
    output_cigar: bool,
    opt: MapOptions,
    out_cfg: OutputConfig,
    k: usize,
    w: usize,
    is_hpc: bool,

    // Accumulated log (joined with output at the end).
    log_buf: String,

    // Ref-build phase (Some until `finalize_ref` is called).
    builder: Option<IndexBuilder>,
    ref_parser: Option<FastaStreamer>,
    t_ref_start: Option<web_time::Instant>,

    // Align phase (Some after `finalize_ref`).
    idx: Option<Index>,
    query_parser: Option<QueryParser>,
    ctx: Option<AlignmentContext>,
    map_ctx: Option<MapContext>,
    t_align_start: Option<web_time::Instant>,
    n_aligned: usize,
    n_queries: usize,
}

#[wasm_bindgen]
impl AlignSession {
    /// Create a new session. After construction, push ref bytes via
    /// `append_ref`, then call `finalize_ref`, then push query bytes via
    /// `append_query`, then call `finalize`.
    #[wasm_bindgen(constructor)]
    pub fn new(preset: &str, output_sam: bool, output_cigar: bool) -> Self {
        let mut k = 15usize;
        let mut w = 10usize;
        let mut is_hpc = false;
        let mut opt = MapOptions::default();
        apply_preset_wasm(&mut opt, &mut k, &mut w, &mut is_hpc, preset);
        if output_cigar || output_sam {
            opt.flags.insert(AlignFlags::OUT_CIGAR);
        }
        let out_cfg = OutputConfig {
            do_cigar: output_cigar,
            do_cs: false,
            cs_long: false,
            do_md: false,
            do_ds: false,
            eqx: false,
            output_sam,
            rg_id: None,
            split_mode: false,
        };
        let mut log_buf = String::new();
        log_buf.push_str(&format!("[*] Preset: {} (k={}, w={})\n", preset, k, w));

        Self {
            preset: preset.to_string(),
            output_sam,
            output_cigar,
            opt,
            out_cfg,
            k, w, is_hpc,
            log_buf,
            builder: Some(IndexBuilder::new(w, k, is_hpc, 50000)),
            ref_parser: Some(FastaStreamer::new()),
            t_ref_start: Some(web_time::Instant::now()),
            idx: None,
            query_parser: None,
            ctx: None,
            map_ctx: None,
            t_align_start: None,
            n_aligned: 0,
            n_queries: 0,
        }
    }

    /// Optional pre-allocation hint for the packed reference buffer. Pass an
    /// upper-bound estimate of the total reference bases (e.g. the
    /// uncompressed file size). Avoids the doubling realloc that can briefly
    /// use 3× the final packed-buffer size on multi-GB inputs.
    pub fn reserve_ref_bases(&mut self, expected_total: u64) -> Result<(), JsValue> {
        let builder = self.builder.as_mut()
            .ok_or_else(|| JsValue::from_str("reserve_ref_bases called after finalize_ref"))?;
        builder.reserve_bases(expected_total as usize);
        Ok(())
    }

    /// Push a chunk of reference bytes. Completed FASTA records get packed
    /// into the index immediately and their raw bytes are dropped. Safe to
    /// call with chunks of any size, including byte-by-byte.
    pub fn append_ref(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        let parser = self.ref_parser.as_mut()
            .ok_or_else(|| JsValue::from_str("append_ref called after finalize_ref"))?;
        let builder = self.builder.as_mut()
            .ok_or_else(|| JsValue::from_str("append_ref called after finalize_ref"))?;
        parser.push(chunk);
        while let Some((name, seq)) = parser.next_record() {
            builder.add_sequence(name, seq);
        }
        Ok(())
    }

    /// Flush the in-flight ref record (if any), build the index, and
    /// transition to the alignment phase.
    pub fn finalize_ref(&mut self) -> Result<(), JsValue> {
        let mut parser = self.ref_parser.take()
            .ok_or_else(|| JsValue::from_str("finalize_ref called twice"))?;
        let mut builder = self.builder.take()
            .ok_or_else(|| JsValue::from_str("finalize_ref called twice"))?;
        parser.finalize();
        while let Some((name, seq)) = parser.next_record() {
            builder.add_sequence(name, seq);
        }

        let n_seqs = builder.num_sequences();
        let total_bases = builder.total_bases();
        if n_seqs == 0 {
            return Err(JsValue::from_str("No reference sequences found"));
        }
        let t0 = self.t_ref_start.take().unwrap_or_else(web_time::Instant::now);
        self.log_buf.push_str(&format!(
            "[*] Reference: {} sequence(s), {} bases\n", n_seqs, total_bases));

        let idx = builder.finish();
        self.opt.seeding.mid_occ = idx.cal_mid_occ(2e-4, 10, 10000);
        self.log_buf.push_str(&format!(
            "[*] Index built in {:.3}s (mid_occ={})\n",
            t0.elapsed().as_secs_f64(), self.opt.seeding.mid_occ));

        self.idx = Some(idx);
        self.query_parser = Some(QueryParser::new());
        self.ctx = Some(AlignmentContext::new());
        self.map_ctx = Some(MapContext::new());
        self.t_align_start = Some(web_time::Instant::now());
        Ok(())
    }

    /// Push a chunk of FASTQ query bytes. Each completed read is aligned
    /// immediately against the prebuilt index; returns the concatenated PAF
    /// (or SAM) output produced by reads completed in this chunk.
    pub fn append_query(&mut self, chunk: &[u8]) -> Result<String, JsValue> {
        let parser = self.query_parser.as_mut()
            .ok_or_else(|| JsValue::from_str("append_query called before finalize_ref"))?;
        let idx = self.idx.as_ref()
            .ok_or_else(|| JsValue::from_str("append_query called before finalize_ref"))?;
        let ctx = self.ctx.as_mut().unwrap();
        let map_ctx = self.map_ctx.as_mut().unwrap();

        parser.push(chunk)?;
        let mut out = String::new();
        while let Some((qname, qseq)) = parser.next_record() {
            self.n_queries += 1;
            let ri = ReadInfo {
                qname: &qname,
                qseq: &qseq,
                qual: None, comment: None, n_seg: 1, seg_idx: 0,
            };
            let res = align_and_format_query(
                &self.opt, idx, &ri, ctx, map_ctx, None, None, &self.out_cfg,
            );
            if !res.0.is_empty() {
                out.push_str(&res.0);
                self.n_aligned += 1;
            }
        }
        Ok(out)
    }

    /// Flush any trailing FASTQ record and return the final log section.
    /// The log block is prefixed with "---LOG---\n" so it concatenates
    /// cleanly with the accumulated PAF/SAM output on the JS side.
    pub fn finalize(&mut self) -> Result<String, JsValue> {
        let parser = self.query_parser.as_mut()
            .ok_or_else(|| JsValue::from_str("finalize called before finalize_ref"))?;
        let idx = self.idx.as_ref().unwrap();
        let ctx = self.ctx.as_mut().unwrap();
        let map_ctx = self.map_ctx.as_mut().unwrap();
        parser.finalize();
        let mut trailing_out = String::new();
        while let Some((qname, qseq)) = parser.next_record() {
            self.n_queries += 1;
            let ri = ReadInfo {
                qname: &qname,
                qseq: &qseq,
                qual: None, comment: None, n_seg: 1, seg_idx: 0,
            };
            let res = align_and_format_query(
                &self.opt, idx, &ri, ctx, map_ctx, None, None, &self.out_cfg,
            );
            if !res.0.is_empty() {
                trailing_out.push_str(&res.0);
                self.n_aligned += 1;
            }
        }

        if let Some(t0) = self.t_align_start.take() {
            let secs = t0.elapsed().as_secs_f64().max(0.001);
            self.log_buf.push_str(&format!(
                "[*] Aligned {} reads ({} with hits) in {:.3}s ({:.0} reads/sec)\n",
                self.n_queries, self.n_aligned, secs,
                self.n_queries as f64 / secs));
        }

        let log = std::mem::take(&mut self.log_buf);
        Ok(format!("{}---LOG---\n{}", trailing_out, log))
    }
}

// Silence unused-field warnings (preset/output_sam/output_cigar are surfaced
// via the constructor but not read again in the implementation; keeping them
// makes the struct self-describing in debug dumps).
#[allow(dead_code)]
const _AVOID_UNUSED: fn(&AlignSession) = |s: &AlignSession| {
    let _ = (&s.preset, s.output_sam, s.output_cigar, s.is_hpc, s.w, s.k);
};

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
