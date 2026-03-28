//! High-level alignment API for library consumers.
//!
//! This module provides a simple interface for aligning sequences against a
//! reference, hiding the internal pipeline details (MapContext, AlignmentContext,
//! coordinate adjustments, etc.).
//!
//! # Quick Start
//!
//! ```no_run
//! use rammap::api::{Aligner, Preset};
//! use rammap::Strand;
//!
//! let aligner = Aligner::from_index("reference.mmi", Preset::MapOnt).unwrap();
//! let results = aligner.map_seq("read1", b"ACGTACGTACGT...");
//! for aln in &results.alignments {
//!     println!("{}\t{}\t{}\t{}\tMAPQ={}",
//!         aln.target_name, aln.target_start, aln.target_end,
//!         if aln.strand == Strand::Forward { "+" } else { "-" },
//!         aln.mapq);
//! }
//! ```
//!
//! # Thread Safety
//!
//! `Aligner` is `Send + Sync` — wrap it in `Arc<Aligner>` to share across threads.
//! Each call to `map_seq`/`map_pair` allocates lightweight per-call buffers internally.

use std::io;
use crate::align::index::Index;
use crate::align::map::{MapOptions, MapContext, AlignFlags};
use crate::align::extend::AlignmentContext;
use crate::align::pipeline::{self, OutputConfig, ReadInfo};

/// Alignment preset matching minimap2's `-x` presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    MapOnt,
    MapPb,
    MapHifi,
    LrHq,
    LrHqae,
    MapIclr,
    Sr,
    Splice,
    SpliceHq,
    SpliceSr,
    Cdna,
    Asm5,
    Asm10,
    Asm20,
    AvaOnt,
    AvaPb,
}

impl Preset {
    /// Convert to the string form
    pub fn as_str(&self) -> &'static str {
        match self {
            Preset::MapOnt => "map-ont",
            Preset::MapPb => "map-pb",
            Preset::MapHifi => "map-hifi",
            Preset::LrHq => "lr:hq",
            Preset::LrHqae => "lr:hqae",
            Preset::MapIclr => "map-iclr",
            Preset::Sr => "sr",
            Preset::Splice => "splice",
            Preset::SpliceHq => "splice:hq",
            Preset::SpliceSr => "splice:sr",
            Preset::Cdna => "cdna",
            Preset::Asm5 => "asm5",
            Preset::Asm10 => "asm10",
            Preset::Asm20 => "asm20",
            Preset::AvaOnt => "ava-ont",
            Preset::AvaPb => "ava-pb",
        }
    }
}

/// Strand orientation of an alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    Reverse,
}

/// A single alignment result.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Target sequence name.
    pub target_name: String,
    /// Target sequence length.
    pub target_len: usize,
    /// Aligned region start on target (0-based).
    pub target_start: usize,
    /// Aligned region end on target (exclusive).
    pub target_end: usize,
    /// Aligned region start on query (0-based).
    pub query_start: usize,
    /// Aligned region end on query (exclusive).
    pub query_end: usize,
    /// Alignment strand.
    pub strand: Strand,
    /// Mapping quality (0-60).
    pub mapq: i32,
    /// Whether this is a primary alignment.
    pub is_primary: bool,
    /// Number of matching bases.
    pub matches: usize,
    /// Alignment block length.
    pub block_len: usize,
    /// Edit distance (NM tag).
    pub edit_distance: u32,
    /// CIGAR string (if CIGAR output enabled).
    pub cigar: Option<String>,
    /// CS tag string (if requested).
    pub cs: Option<String>,
    /// Alignment score (AS tag).
    pub score: i32,
    /// Sequence divergence (0.0 = identical).
    pub divergence: f64,
}

/// Result of aligning one read (or read pair).
#[derive(Debug, Clone)]
pub struct MapResult {
    /// All alignments for this read, ordered by score (primary first).
    pub alignments: Vec<Alignment>,
}

/// High-level aligner wrapping an index and alignment parameters.
///
/// Construct via [`Aligner::from_index`] or [`Aligner::from_fasta`], then call
/// [`Aligner::map_seq`] or [`Aligner::map_pair`] per read.
pub struct Aligner {
    index: Index,
    options: MapOptions,
    out_cfg: OutputConfig,
}

impl Aligner {
    /// Load an aligner from a pre-built minimap2 `.mmi` index file.
    pub fn from_index(path: &str, preset: Preset) -> io::Result<Self> {
        let index = Index::load(path)?;
        let (options, out_cfg) = build_options(preset, index.kmer_size, index.window_size);
        Ok(Aligner { index, options, out_cfg })
    }

    /// Build an aligner from a FASTA reference file.
    ///
    /// This builds the index in memory (may take several seconds for large genomes).
    pub fn from_fasta(path: &str, preset: Preset) -> io::Result<Self> {
        let mut k = 15usize;
        let mut w = 10usize;
        let mut is_hpc = false;
        let mut opt = MapOptions::default();
        apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, preset.as_str());

        let seqs = crate::fasta::read_fasta(path)?;
        let index = Index::build(seqs, w, k, is_hpc, usize::MAX);
        let out_cfg = OutputConfig {
            do_cigar: true,
            do_cs: false,
            do_md: false,
            do_ds: false,
            eqx: false,
            output_sam: false,
            rg_id: None,
            split_mode: false,
        };
        Ok(Aligner { index, options: opt, out_cfg })
    }

    /// Align a single-end read against the reference.
    pub fn map_seq(&self, name: &str, seq: &[u8]) -> MapResult {
        let mut ctx = AlignmentContext::new();
        let mut map_ctx = MapContext::new();
        let pq = pipeline::process_query(
            &self.options, &self.index, name, seq,
            &mut ctx, &mut map_ctx, None, &self.out_cfg,
        );
        to_map_result(&pq, &self.index, &self.out_cfg)
    }

    /// Align a paired-end read pair against the reference.
    pub fn map_pair(&self, name: &str, seq1: &[u8], seq2: &[u8]) -> MapResult {
        let mut ctx = AlignmentContext::new();
        let mut map_ctx = MapContext::new();
        let read1 = ReadInfo { qname: name, qseq: seq1, qual: None, comment: None, n_seg: 2, seg_idx: 0 };
        let read2 = ReadInfo { qname: name, qseq: seq2, qual: None, comment: None, n_seg: 2, seg_idx: 1 };
        let (output, _stats) = pipeline::align_and_format_pair(
            &self.options, &self.index, &read1, &read2,
            &mut ctx, &mut map_ctx, None, &self.out_cfg,
        );
        // Parse PAF output back into MapResult for the pair case
        // (align_and_format_pair produces formatted output, not ProcessedQuery)
        parse_paf_to_map_result(&output, &self.index)
    }

    /// Format a pre-computed MapResult as a PAF string.
    ///
    /// Useful when you want to compute alignments and format separately.
    pub fn format_paf(&self, name: &str, query_len: usize, result: &MapResult) -> String {
        let mut lines = Vec::new();
        for aln in &result.alignments {
            let strand = if aln.strand == Strand::Forward { '+' } else { '-' };
            let mut line = format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                name, query_len, aln.query_start, aln.query_end,
                strand, aln.target_name, aln.target_len,
                aln.target_start, aln.target_end,
                aln.matches, aln.block_len, aln.mapq,
            );
            if let Some(ref cigar) = aln.cigar {
                line.push_str(&format!("\tcg:Z:{}", cigar));
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Access the underlying index (for advanced use).
    pub fn index(&self) -> &Index { &self.index }

    /// Access the current MapOptions (for advanced use).
    pub fn options(&self) -> &MapOptions { &self.options }

    /// Mutably access MapOptions for fine-tuning after construction.
    pub fn options_mut(&mut self) -> &mut MapOptions { &mut self.options }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_options(preset: Preset, index_k: usize, index_w: usize) -> (MapOptions, OutputConfig) {
    let mut k = index_k;
    let mut w = index_w;
    let mut is_hpc = false;
    let mut opt = MapOptions::default();
    apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, preset.as_str());

    let out_cfg = OutputConfig {
        do_cigar: true,
        do_cs: false,
        do_md: false,
        do_ds: false,
        eqx: false,
        output_sam: false,
        rg_id: None,
        split_mode: false,
    };
    (opt, out_cfg)
}

fn to_map_result(
    pq: &pipeline::ProcessedQuery,
    mi: &Index,
    out: &OutputConfig,
) -> MapResult {
    let alignments = pq.results.iter().zip(pq.mapqs.iter()).map(|(r, &mapq)| {
        Alignment {
            target_name: mi.seqs[r.ref_id].name.clone(),
            target_len: mi.seqs[r.ref_id].len,
            target_start: r.ref_start,
            target_end: r.ref_end,
            query_start: r.query_start,
            query_end: r.query_end,
            strand: if r.is_reverse { Strand::Reverse } else { Strand::Forward },
            mapq,
            is_primary: !r.is_secondary,
            matches: r.matches,
            block_len: r.block_len,
            edit_distance: r.edit_distance,
            cigar: if out.do_cigar && !r.cigar_str.is_empty() { Some(r.cigar_str.clone()) } else { None },
            cs: if out.do_cs && !r.cs_str.is_empty() { Some(r.cs_str.clone()) } else { None },
            score: r.align_score,
            divergence: r.divergence,
        }
    }).collect();
    MapResult { alignments }
}

/// Parse PAF-formatted output back into a MapResult (for paired-end path).
fn parse_paf_to_map_result(paf: &str, _mi: &Index) -> MapResult {
    let mut alignments = Vec::new();
    for line in paf.lines() {
        if line.is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 { continue; }
        let target_name = fields[5].to_string();
        let target_len = fields[6].parse().unwrap_or(0);
        let target_start: usize = fields[7].parse().unwrap_or(0);
        let target_end: usize = fields[8].parse().unwrap_or(0);
        let matches: usize = fields[9].parse().unwrap_or(0);
        let block_len: usize = fields[10].parse().unwrap_or(0);
        let mapq: i32 = fields[11].parse().unwrap_or(0);
        let query_start: usize = fields[2].parse().unwrap_or(0);
        let query_end: usize = fields[3].parse().unwrap_or(0);
        let strand = if fields[4] == "+" { Strand::Forward } else { Strand::Reverse };

        // Extract optional tags
        let mut cigar = None;
        let mut cs = None;
        let mut score = 0i32;
        let mut edit_distance = 0u32;
        let mut divergence = 0.0f64;
        let mut is_secondary = false;
        for &tag in &fields[12..] {
            if let Some(val) = tag.strip_prefix("cg:Z:") { cigar = Some(val.to_string()); }
            else if let Some(val) = tag.strip_prefix("cs:Z:") { cs = Some(val.to_string()); }
            else if let Some(val) = tag.strip_prefix("AS:i:") { score = val.parse().unwrap_or(0); }
            else if let Some(val) = tag.strip_prefix("NM:i:") { edit_distance = val.parse().unwrap_or(0); }
            else if let Some(val) = tag.strip_prefix("de:f:") { divergence = val.parse().unwrap_or(0.0); }
            else if tag.starts_with("tp:A:S") || tag.starts_with("tp:A:s") { is_secondary = true; }
        }

        alignments.push(Alignment {
            target_name, target_len, target_start, target_end,
            query_start, query_end, strand, mapq,
            is_primary: !is_secondary, matches, block_len,
            edit_distance, cigar, cs, score, divergence,
        });
    }
    MapResult { alignments }
}

// ---------------------------------------------------------------------------
// Preset application (moved from main.rs)
// ---------------------------------------------------------------------------

/// Apply a preset by string name (e.g., "map-ont", "sr", "splice").
///
/// This is the same function used by the CLI's `-x` flag. Modifies `opt`,
/// `k`, `w`, and `is_hpc` in place.
pub fn apply_preset_str(opt: &mut MapOptions, k: &mut usize, w: &mut usize, is_hpc: &mut bool, preset: &str) {
    match preset {
        "lr" | "map-ont" => {
            *k = 15; *w = 10;
        },
        "map10k" | "map-pb" => {
            *is_hpc = true; *k = 19;
        },
        "lr:hq" => {
            *k = 19; *w = 19;
            opt.chaining.max_gap = 10000;
            opt.seeding.min_mid_occ = 50; opt.seeding.max_mid_occ = 500;
        },
        "map-hifi" | "map-ccs" => {
            *k = 19; *w = 19;
            opt.chaining.max_gap = 10000;
            opt.seeding.min_mid_occ = 50; opt.seeding.max_mid_occ = 500;
            opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 4; opt.scoring.gap_open = 6; opt.scoring.gap_open2 = 26; opt.scoring.gap_extend = 2; opt.scoring.gap_extend2 = 1;
            opt.alignment.min_dp_max = 200;
        },
        "lr:hqae" => {
            *k = 25; *w = 51;
            opt.flags.insert(AlignFlags::RMQ_CHAIN);
            opt.seeding.min_mid_occ = 50; opt.seeding.max_mid_occ = 500;
            opt.chaining.rmq_inner_dist = 5000;
            opt.seeding.occ_dist = 200;
            opt.filtering.best_n = 100;
            opt.chaining.chain_gap_scale = 5.0;
        },
        "map-iclr-prerender" => {
            *k = 15;
            opt.scoring.mismatch_penalty = 6; opt.scoring.transition = 1;
            opt.scoring.gap_open = 10; opt.scoring.gap_open2 = 50;
        },
        "map-iclr" => {
            *k = 19;
            opt.scoring.mismatch_penalty = 6; opt.scoring.transition = 4;
            opt.scoring.gap_open = 10; opt.scoring.gap_open2 = 50;
        },
        p if p.starts_with("asm") => {
            *k = 19; *w = 19;
            opt.chaining.bandwidth = 1000; opt.chaining.bandwidth_long = 100000;
            opt.chaining.max_gap = 10000;
            opt.flags.insert(AlignFlags::RMQ_CHAIN);
            opt.seeding.min_mid_occ = 50; opt.seeding.max_mid_occ = 500;
            opt.alignment.min_dp_max = 200;
            opt.filtering.best_n = 50;
            match p {
                "asm5" => {
                    opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 19; opt.scoring.gap_open = 39; opt.scoring.gap_open2 = 81; opt.scoring.gap_extend = 3; opt.scoring.gap_extend2 = 1;
                    opt.alignment.zdrop = 200; opt.alignment.zdrop_inv = 200;
                },
                "asm10" => {
                    opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 9; opt.scoring.gap_open = 16; opt.scoring.gap_open2 = 41; opt.scoring.gap_extend = 2; opt.scoring.gap_extend2 = 1;
                    opt.alignment.zdrop = 200; opt.alignment.zdrop_inv = 200;
                },
                "asm20" => {
                    opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 4; opt.scoring.gap_open = 6; opt.scoring.gap_open2 = 26; opt.scoring.gap_extend = 2; opt.scoring.gap_extend2 = 1;
                    opt.alignment.zdrop = 200; opt.alignment.zdrop_inv = 200;
                    *w = 10;
                },
                _ => {
                    eprintln!("Warning: Unknown asm preset '{}', using asm20 defaults", p);
                }
            }
        },
        "short" | "sr" => {
            *k = 21; *w = 11;
            opt.flags.insert(AlignFlags::SHORT_READ | AlignFlags::FRAG_MODE | AlignFlags::NO_PRINT_2ND | AlignFlags::HEAP_SORT);
            opt.pairing.pe_ori = 1;
            opt.scoring.match_score = 2; opt.scoring.mismatch_penalty = 8; opt.scoring.gap_open = 12; opt.scoring.gap_extend = 2; opt.scoring.gap_open2 = 24; opt.scoring.gap_extend2 = 1;
            opt.alignment.zdrop = 100; opt.alignment.zdrop_inv = 100;
            opt.alignment.end_bonus = 10;
            opt.pairing.max_frag_len = 800;
            opt.chaining.max_gap = 100;
            opt.chaining.bandwidth = 100; opt.chaining.bandwidth_long = 100;
            opt.filtering.pri_ratio = 0.5;
            opt.chaining.min_cnt = 2;
            opt.chaining.min_chain_score = 25;
            opt.alignment.min_dp_max = 40;
            opt.filtering.best_n = 20;
            opt.seeding.mid_occ = 1000;
            opt.seeding.max_occ = 5000;
        },
        p if p.starts_with("splice") || p == "cdna" => {
            *k = 15; *w = 5;
            opt.flags.insert(AlignFlags::SPLICE | AlignFlags::SPLICE_FOR | AlignFlags::SPLICE_REV | AlignFlags::SPLICE_FLANK);
            opt.alignment.max_sw_mat = 0;
            opt.chaining.max_gap = 2000;
            opt.chaining.max_gap_ref = 200000;
            opt.chaining.bandwidth = 200000; opt.chaining.bandwidth_long = 200000;
            opt.scoring.match_score = 1; opt.scoring.mismatch_penalty = 2; opt.scoring.gap_open = 2; opt.scoring.gap_extend = 1; opt.scoring.gap_open2 = 32; opt.scoring.gap_extend2 = 0;
            opt.scoring.noncanon_penalty = 9;
            opt.scoring.junc_bonus = 9;
            opt.scoring.junc_pen = 5;
            opt.alignment.zdrop = 200; opt.alignment.zdrop_inv = 100;
            opt.filtering.is_splice = true;
            if p == "splice:hq" {
                opt.scoring.noncanon_penalty = 5; opt.scoring.mismatch_penalty = 4; opt.scoring.gap_open = 6; opt.scoring.gap_open2 = 24;
            } else if p == "splice:sr" {
                opt.flags.insert(AlignFlags::NO_PRINT_2ND | AlignFlags::HEAP_SORT | AlignFlags::FRAG_MODE | AlignFlags::WEAK_PAIRING | AlignFlags::SR_RNA);
                opt.scoring.noncanon_penalty = 5; opt.scoring.mismatch_penalty = 4; opt.scoring.gap_open = 6; opt.scoring.gap_open2 = 24;
                opt.chaining.min_chain_score = 25;
                opt.alignment.min_dp_max = 40;
                opt.alignment.min_dp_len = 20;
                opt.pairing.pe_ori = 1;
                opt.filtering.best_n = 10;
            }
        },
        "ava-ont" => {
            *k = 15; *w = 5;
            opt.flags.insert(AlignFlags::ALL_CHAINS | AlignFlags::NO_DIAG | AlignFlags::NO_DUAL | AlignFlags::NO_LJOIN);
            opt.chaining.min_chain_score = 100; opt.filtering.pri_ratio = 0.0;
            opt.chaining.max_chain_skip = 25;
            opt.chaining.bandwidth = 2000; opt.chaining.bandwidth_long = 2000;
            opt.seeding.occ_dist = 0;
        },
        "ava-pb" => {
            *is_hpc = true; *k = 19; *w = 5;
            opt.flags.insert(AlignFlags::ALL_CHAINS | AlignFlags::NO_DIAG | AlignFlags::NO_DUAL | AlignFlags::NO_LJOIN);
            opt.chaining.min_chain_score = 100; opt.filtering.pri_ratio = 0.0;
            opt.chaining.max_chain_skip = 25;
            opt.chaining.bandwidth_long = opt.chaining.bandwidth;
            opt.seeding.occ_dist = 0;
        },
        _ => {
            eprintln!("Warning: Unknown preset '{}', using default", preset);
        }
    }
    opt.chaining.chn_pen_gap = (opt.chaining.chain_gap_scale as f64 * 0.01 * (*k as f64)) as f32;
    opt.chaining.chn_pen_skip = 0.0;
}

// ---------------------------------------------------------------------------
// Low-level pairwise DP alignment API
// ---------------------------------------------------------------------------

/// Scoring parameters for pairwise DP alignment.
#[derive(Debug, Clone)]
pub struct DpScoring {
    /// Match score (positive, e.g., 2).
    pub match_score: i32,
    /// Mismatch penalty (positive, applied as negative, e.g., 4).
    pub mismatch: i32,
    /// Gap open penalty (positive, e.g., 4).
    pub gap_open: i32,
    /// Gap extend penalty (positive, e.g., 2).
    pub gap_extend: i32,
    /// Second gap open penalty for dual-affine model (0 to disable).
    pub gap_open2: i32,
    /// Second gap extend penalty for dual-affine model (0 to disable).
    pub gap_extend2: i32,
}

impl Default for DpScoring {
    fn default() -> Self {
        DpScoring { match_score: 2, mismatch: 4, gap_open: 4, gap_extend: 2, gap_open2: 0, gap_extend2: 0 }
    }
}

/// Result of a pairwise DP alignment.
#[derive(Debug, Clone)]
pub struct DpAlignment {
    /// Alignment score.
    pub score: i32,
    /// CIGAR string (M/I/D operations).
    pub cigar: String,
    /// Query start position (0-based). For extension mode, may be > 0.
    pub query_start: usize,
    /// Query end position (exclusive).
    pub query_end: usize,
    /// Target start position (0-based).
    pub target_start: usize,
    /// Target end position (exclusive).
    pub target_end: usize,
}

/// Align two nt4-encoded sequences using the SIMD-optimized DP engine.
///
/// Performs semi-global extension alignment (like minimap2's gap-fill): aligns
/// `query` against `target` with affine gap penalties, returning the best-scoring
/// alignment with CIGAR traceback.
///
/// Sequences must be nt4-encoded (0=A, 1=C, 2=G, 3=T, 4=N). Use
/// [`encode_nt4`] to convert from ASCII.
///
/// # Parameters
/// - `query`: nt4-encoded query sequence
/// - `target`: nt4-encoded target sequence
/// - `scoring`: gap penalties and match/mismatch scores
/// - `bandwidth`: band width for banded alignment (-1 for full matrix)
///
/// # Example
///
/// ```
/// use rammap::api::{dp_align, DpScoring, encode_nt4};
///
/// let query = encode_nt4(b"ACGTACGTACGT");
/// let target = encode_nt4(b"ACGTACCTACGT");
/// let result = dp_align(&query, &target, &DpScoring::default(), -1);
/// println!("score={} cigar={}", result.score, result.cigar);
/// ```
pub fn dp_align(query: &[u8], target: &[u8], scoring: &DpScoring, bandwidth: i32) -> DpAlignment {
    use crate::align::dp;
    use crate::align::extend::build_scoring_matrix;

    let mat = build_scoring_matrix(scoring.match_score, scoring.mismatch);
    let mut ez = dp::DpResult::default();

    if scoring.gap_open2 > 0 || scoring.gap_extend2 > 0 {
        dp::extend_dual_affine(
            query, target, 5, &mat,
            scoring.gap_open as i8, scoring.gap_extend as i8,
            scoring.gap_open2 as i8, scoring.gap_extend2 as i8,
            bandwidth, -1, 0, dp::APPROX_MAX, &mut ez,
        );
    } else {
        dp::extend_single_affine(
            query, target, 5, &mat,
            scoring.gap_open as i8, scoring.gap_extend as i8,
            bandwidth, -1, 0, dp::APPROX_MAX, &mut ez,
        );
    }

    let cigar_raw = &ez.cigar;
    let cigar = raw_cigar_to_string(cigar_raw);
    let (qs, qe, ts, te) = cigar_bounds(cigar_raw);
    // Recompute score from CIGAR for consistency across SIMD variants
    let score = compute_score_from_cigar(
        cigar_raw, query, target, &mat,
        scoring.gap_open, scoring.gap_extend, scoring.gap_open2, scoring.gap_extend2,
    );
    DpAlignment {
        score,
        cigar,
        query_start: qs,
        query_end: qe,
        target_start: ts,
        target_end: te,
    }
}

/// Global alignment (Needleman-Wunsch): align full query to full target.
///
/// True banded NW using the Suzuki-Kasahara formulation
/// Produces a CIGAR spanning both sequences end-to-end with
/// proper NW boundary conditions (gap penalties on first row/column).
///
/// Uses single-affine gap model (gap_open2/gap_extend2 are ignored).
///
/// # Parameters
/// - `bandwidth`: band width for banded alignment (-1 for full matrix)
pub fn dp_global(query: &[u8], target: &[u8], scoring: &DpScoring, bandwidth: i32) -> DpAlignment {
    use crate::align::dp;
    use crate::align::extend::build_scoring_matrix;

    let mat = build_scoring_matrix(scoring.match_score, scoring.mismatch);
    let mut ez = dp::DpResult::default();
    dp::global_align(
        query, target, 5, &mat,
        scoring.gap_open, scoring.gap_extend,
        bandwidth, &mut ez,
    );

    let cigar = raw_cigar_to_string(&ez.cigar);
    let (qs, qe, ts, te) = cigar_bounds(&ez.cigar);
    DpAlignment { score: ez.score, cigar, query_start: qs, query_end: qe, target_start: ts, target_end: te }
}

/// Local alignment (Smith-Waterman): find the best-scoring local region.
///
/// Two-phase approach:
/// 1. Forward Smith-Waterman to find the optimal endpoint (score, qe, te)
/// 2. Reverse extension from endpoint to find start position + CIGAR
///
/// The reverse extension uses `EXTENSION_ONLY | REV_CIGAR` on reversed
/// prefixes — it extends backward from the endpoint, stopping at the
/// optimal start, and outputs the CIGAR in forward order. Both phases
/// use SIMD-accelerated kernels.
pub fn dp_local(query: &[u8], target: &[u8], scoring: &DpScoring) -> DpAlignment {
    use crate::align::dp;
    use crate::align::extend::build_scoring_matrix;

    let mat = build_scoring_matrix(scoring.match_score, scoring.mismatch);

    // Phase 1: Forward Smith-Waterman to find the best endpoint
    let mut qp = dp::lightweight_profile_init(query.len() as i32, query, 5, &mat);
    let (score, q_end, t_end) = dp::lightweight_align_i16(
        &mut qp, target.len() as i32, target, scoring.gap_open, scoring.gap_extend,
    );
    if score <= 0 || q_end < 0 || t_end < 0 {
        return DpAlignment { score: 0, cigar: String::new(), query_start: 0, query_end: 0, target_start: 0, target_end: 0 };
    }

    let qe = q_end as usize + 1;
    let te = t_end as usize + 1;

    // Phase 2: Reverse extension → start position + CIGAR in one pass
    let q_rev: Vec<u8> = query[..qe].iter().rev().copied().collect();
    let t_rev: Vec<u8> = target[..te].iter().rev().copied().collect();
    let mut ez = dp::DpResult::default();
    dp::extend_single_affine(
        &q_rev, &t_rev, 5, &mat,
        scoring.gap_open as i8, scoring.gap_extend as i8,
        -1, score, 0,
        dp::EXTENSION_ONLY | dp::REV_CIGAR | dp::APPROX_MAX,
        &mut ez,
    );

    let cigar = raw_cigar_to_string(&ez.cigar);
    let (_, cig_qlen, _, cig_tlen) = cigar_bounds(&ez.cigar);
    let qs = qe - cig_qlen;
    let ts = te - cig_tlen;

    DpAlignment {
        score,
        cigar,
        query_start: qs,
        query_end: qe,
        target_start: ts,
        target_end: te,
    }
}

/// Extension alignment: align from position 0 outward, stopping at the best score.
///
/// Useful for extending a seed match left or right. Returns the optimal-scoring
/// prefix alignment (may not reach the end of either sequence).
pub fn dp_extension(query: &[u8], target: &[u8], scoring: &DpScoring, bandwidth: i32) -> DpAlignment {
    use crate::align::dp::{self, EXTENSION_ONLY};
    use crate::align::extend::build_scoring_matrix;

    let mat = build_scoring_matrix(scoring.match_score, scoring.mismatch);
    let mut ez = dp::DpResult::default();

    let zdrop = 400; // default z-drop threshold for extension
    if scoring.gap_open2 > 0 || scoring.gap_extend2 > 0 {
        dp::extend_dual_affine(
            query, target, 5, &mat,
            scoring.gap_open as i8, scoring.gap_extend as i8,
            scoring.gap_open2 as i8, scoring.gap_extend2 as i8,
            bandwidth, zdrop, 0, EXTENSION_ONLY | dp::APPROX_MAX, &mut ez,
        );
    } else {
        dp::extend_single_affine(
            query, target, 5, &mat,
            scoring.gap_open as i8, scoring.gap_extend as i8,
            bandwidth, zdrop, 0, EXTENSION_ONLY | dp::APPROX_MAX, &mut ez,
        );
    }

    let cigar = raw_cigar_to_string(&ez.cigar);
    let (qs, qe, ts, te) = cigar_bounds(&ez.cigar);
    let ext_qe = if ez.max_score_query_pos >= 0 { std::cmp::min(qe, ez.max_score_query_pos as usize + 1) } else { qe };
    let ext_te = if ez.max_score_target_pos >= 0 { std::cmp::min(te, ez.max_score_target_pos as usize + 1) } else { te };
    DpAlignment {
        score: ez.score,
        cigar,
        query_start: qs,
        query_end: ext_qe,
        target_start: ts,
        target_end: ext_te,
    }
}

/// Encode an ASCII DNA sequence to nt4 (A=0, C=1, G=2, T=3, N=4).
///
/// Required for DP alignment functions which operate on nt4-encoded sequences.
pub fn encode_nt4(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| crate::align::extend::encode_nt4_byte(b)).collect()
}

/// Compute true alignment score from raw CIGAR + sequences.
/// Walks the CIGAR, scoring matches/mismatches from the matrix and gaps with
/// affine penalties. If gap_open2/gap_extend2 are nonzero, uses dual-affine
/// model (min of both gap costs).
fn compute_score_from_cigar(
    cigar: &[u32], query: &[u8], target: &[u8],
    mat: &[i8], gapo: i32, gape: i32, gapo2: i32, gape2: i32,
) -> i32 {
    let dual = gapo2 > 0 || gape2 > 0;
    let mut score = 0i32;
    let mut qi = 0usize;
    let mut ti = 0usize;
    for &c in cigar {
        let len = (c >> 4) as usize;
        match c & 0xf {
            0 => { // M: match/mismatch
                for _ in 0..len {
                    if qi < query.len() && ti < target.len() {
                        score += mat[target[ti].min(4) as usize * 5 + query[qi].min(4) as usize] as i32;
                    }
                    qi += 1;
                    ti += 1;
                }
            }
            1 => { // I: insertion (query gap)
                let cost = gapo + gape * len as i32;
                if dual {
                    let cost2 = gapo2 + gape2 * len as i32;
                    score -= cost.min(cost2);
                } else {
                    score -= cost;
                }
                qi += len;
            }
            2 => { // D: deletion (target gap)
                let cost = gapo + gape * len as i32;
                if dual {
                    let cost2 = gapo2 + gape2 * len as i32;
                    score -= cost.min(cost2);
                } else {
                    score -= cost;
                }
                ti += len;
            }
            _ => {}
        }
    }
    score
}

/// Convert raw u32 CIGAR (len<<4 | op) to a human-readable string.
fn raw_cigar_to_string(cigar: &[u32]) -> String {
    let mut s = String::new();
    for &c in cigar {
        let len = c >> 4;
        let op = match c & 0xf {
            0 => 'M',
            1 => 'I',
            2 => 'D',
            _ => '?',
        };
        s.push_str(&format!("{}{}", len, op));
    }
    s
}

/// Compute aligned region bounds from raw CIGAR.
fn cigar_bounds(cigar: &[u32]) -> (usize, usize, usize, usize) {
    let mut qlen = 0usize;
    let mut tlen = 0usize;
    for &c in cigar {
        let len = (c >> 4) as usize;
        match c & 0xf {
            0 => { qlen += len; tlen += len; } // M
            1 => { qlen += len; }               // I
            2 => { tlen += len; }               // D
            _ => {}
        }
    }
    (0, qlen, 0, tlen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_as_str_roundtrip() {
        let presets = [
            Preset::MapOnt, Preset::MapPb, Preset::MapHifi, Preset::Sr,
            Preset::Splice, Preset::Asm5, Preset::AvaOnt,
        ];
        for p in &presets {
            let s = p.as_str();
            assert!(!s.is_empty(), "preset {:?} should have a string form", p);
        }
    }

    #[test]
    fn test_apply_preset_sr() {
        let mut opt = MapOptions::default();
        let mut k = 15; let mut w = 10; let mut is_hpc = false;
        apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, "sr");
        assert_eq!(k, 21);
        assert_eq!(w, 11);
        assert!(opt.flags.contains(AlignFlags::SHORT_READ));
    }

    #[test]
    fn test_map_result_empty() {
        let result = MapResult { alignments: Vec::new() };
        assert!(result.alignments.is_empty());
    }

    #[test]
    fn test_dp_align_identical() {
        let seq = encode_nt4(b"ACGTACGTACGT");
        let result = dp_align(&seq, &seq, &DpScoring::default(), -1);
        assert!(result.score > 0, "identical seqs should have positive score");
        assert_eq!(result.cigar, "12M");
    }

    #[test]
    fn test_dp_align_with_mismatch() {
        let query = encode_nt4(b"ACGTACGTACGT");
        let target = encode_nt4(b"ACGTACCTACGT");
        //                              ^ mismatch
        let result = dp_align(&query, &target, &DpScoring::default(), -1);
        assert!(result.score > 0);
        assert!(result.cigar.contains('M'), "should contain M ops");
    }

    #[test]
    fn test_dp_align_with_insertion() {
        let query  = encode_nt4(b"ACGTAAAACGT");
        let target = encode_nt4(b"ACGTACGT");
        let result = dp_align(&query, &target, &DpScoring::default(), -1);
        assert!(result.cigar.contains('I'), "should contain insertion: {}", result.cigar);
    }

    #[test]
    fn test_dp_global_identical() {
        let query = encode_nt4(b"ACGTACGT");
        let target = encode_nt4(b"ACGTACGT");
        let result = dp_global(&query, &target, &DpScoring::default(), -1);
        assert_eq!(result.query_end, 8);
        assert_eq!(result.target_end, 8);
        assert_eq!(result.cigar, "8M");
        assert_eq!(result.score, 16); // 8 * 2
    }

    #[test]
    fn test_dp_global_with_indel() {
        let query  = encode_nt4(b"ACGTAAACGT");
        let target = encode_nt4(b"ACGTACGT");
        let result = dp_global(&query, &target, &DpScoring::default(), -1);
        // Must cover both sequences end-to-end
        let total_q: usize = result.cigar.chars().filter(|c| !c.is_ascii_digit())
            .zip(result.cigar.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()))
            .map(|(op, len)| { let l: usize = len.parse().unwrap(); if op == 'D' { 0 } else { l } }).sum();
        let total_t: usize = result.cigar.chars().filter(|c| !c.is_ascii_digit())
            .zip(result.cigar.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()))
            .map(|(op, len)| { let l: usize = len.parse().unwrap(); if op == 'I' { 0 } else { l } }).sum();
        assert_eq!(total_q, 10, "CIGAR should consume full query (10bp): cigar={}", result.cigar);
        assert_eq!(total_t, 8, "CIGAR should consume full target (8bp): cigar={}", result.cigar);
    }

    #[test]
    fn test_dp_local_finds_best_region() {
        // Embed a matching region in noise
        let query  = encode_nt4(b"TTTTACGTACGTACGTTTTT");
        let target = encode_nt4(b"GGGGACGTACGTACGTGGGG");
        let result = dp_local(&query, &target, &DpScoring::default());
        assert!(result.score > 0);
        // Local alignment should find the ACGTACGTACGT match, not the flanking noise
        assert!(result.query_start >= 3, "should skip noise prefix: qs={}", result.query_start);
        assert!(result.query_end <= 16, "should skip noise suffix: qe={}", result.query_end);
    }

    #[test]
    fn test_dp_extension() {
        // Extension aligns from position 0, stopping at the best score
        let query  = encode_nt4(b"ACGTACGTACGTACGTACGTACGTACGTACGTCGATCGATCGATCGATCG");
        let target = encode_nt4(b"ACGTACGTACGTACGTACGTACGTACGTACGTTTTTTTTTTTTTTTTTTTT");
        let result = dp_extension(&query, &target, &DpScoring::default(), -1);
        // Should produce a valid alignment with CIGAR
        assert!(!result.cigar.is_empty(), "extension should produce CIGAR");
    }

    #[test]
    fn test_dp_dual_affine() {
        let query = encode_nt4(b"ACGTACGT");
        let target = encode_nt4(b"ACGTACGT");
        let scoring = DpScoring {
            match_score: 2, mismatch: 4,
            gap_open: 4, gap_extend: 2,
            gap_open2: 24, gap_extend2: 1,
        };
        let result = dp_align(&query, &target, &scoring, -1);
        assert_eq!(result.score, 16); // 8 * 2
        assert_eq!(result.cigar, "8M");
    }

    #[test]
    fn test_encode_nt4() {
        let encoded = encode_nt4(b"ACGTNacgtn");
        assert_eq!(encoded, vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
    }
}
