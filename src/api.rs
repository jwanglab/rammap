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
//! for m in &results.mappings {
//!     println!("{}\t{}\t{}\t{}\tMAPQ={}",
//!         m.target_name, m.target_start, m.target_end,
//!         if m.strand == Strand::Forward { "+" } else { "-" },
//!         m.mapq);
//! }
//! ```
//!
//! # Thread Safety
//!
//! `Aligner` is `Send + Sync` — wrap it in `Arc<Aligner>` to share across threads.
//! Each call to `map_seq`/`map_pair` allocates lightweight per-call buffers internally.

use std::io;
use std::sync::Arc;
use crate::align::index::Index;
use crate::align::map::{MapOptions, MapContext, AlignFlags};
use crate::align::extend::AlignmentContext;
use crate::align::pipeline::{self, OutputConfig, ReadInfo};
use crate::align::junc::{self, JunctionDb};
use std::collections::HashMap;

/// Alignment preset (selected via the `-x` flag).
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

/// Structured CIGAR operation (length + op type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CigarOp {
    /// Operation length.
    pub len: u32,
    /// Operation type: 0=M, 1=I, 2=D, 3=N (intron skip).
    pub op: u8,
}

impl CigarOp {
    /// Operation type as a character (M, I, D, N, S, H, =, X).
    pub fn op_char(&self) -> char {
        match self.op { 0 => 'M', 1 => 'I', 2 => 'D', 3 => 'N', 4 => 'S', 5 => 'H', 7 => '=', 8 => 'X', _ => '?' }
    }
}

/// Per-call options for CS/MD tag generation, overriding the aligner default.
#[derive(Debug, Clone, Copy, Default)]
pub struct MapOpts {
    /// Generate CS tag for this call (overrides aligner setting if Some).
    pub cs: Option<bool>,
    /// Generate MD tag for this call (overrides aligner setting if Some).
    pub md: Option<bool>,
}

/// A single alignment result.
#[derive(Debug, Clone)]
pub struct Mapping {
    /// Target sequence name (shared via Arc to reduce allocation when many
    /// alignments reference the same target).
    pub target_name: Arc<str>,
    /// Target sequence numeric index in the reference.
    pub target_id: usize,
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
    /// Whether this is a supplementary alignment.
    pub is_supplementary: bool,
    /// Whether this alignment contains splice junctions (N_SKIP operations).
    pub is_spliced: bool,
    /// Transcript strand for splice alignments: None=unknown, Some(Forward)=+, Some(Reverse)=-.
    pub trans_strand: Option<Strand>,
    /// Number of matching bases.
    pub matches: usize,
    /// Alignment block length.
    pub block_len: usize,
    /// Edit distance (NM tag).
    pub edit_distance: u32,
    /// CIGAR string (if CIGAR output enabled).
    pub cigar: Option<String>,
    /// Structured CIGAR (if CIGAR output enabled). Each element is (len, op).
    pub cigar_ops: Option<Vec<CigarOp>>,
    /// CS tag string (if requested).
    pub cs: Option<String>,
    /// MD tag string (if requested).
    pub md: Option<String>,
    /// Alignment score (AS tag).
    pub score: i32,
    /// Sequence divergence (0.0 = identical).
    pub divergence: f64,
}

/// Result of aligning one read (or read pair).
#[derive(Debug, Clone)]
pub struct MapResult {
    /// All alignments for this read, ordered by score (primary first).
    pub mappings: Vec<Mapping>,
}

/// High-level aligner wrapping an index and alignment parameters.
///
/// Construct via [`Aligner::from_index`], [`Aligner::from_fasta`], or
/// [`Aligner::from_seqs`], then call [`Aligner::map_seq`] per read.
///
/// # Builder-Style Preset Methods
///
/// For minimap2-rs compatibility, preset methods are also available:
///
/// ```no_run
/// use rammap::api::Aligner;
/// let aligner = Aligner::from_fasta("ref.fa", Aligner::map_ont()).unwrap();
/// ```
pub struct Aligner {
    index: Index,
    options: MapOptions,
    out_cfg: OutputConfig,
    /// Optional known-junction database (minimap2 `--junc-bed` equivalent),
    /// biasing splice scoring toward annotated junctions when set.
    junc_db: Option<JunctionDb>,
}

impl Aligner {
    // --- Preset convenience methods (minimap2-rs compatibility) ---

    /// ONT long read preset.
    pub fn map_ont() -> Preset { Preset::MapOnt }
    /// PacBio CLR preset.
    pub fn map_pb() -> Preset { Preset::MapPb }
    /// PacBio HiFi preset.
    pub fn map_hifi() -> Preset { Preset::MapHifi }
    /// Long read high-quality preset.
    pub fn lr_hq() -> Preset { Preset::LrHq }
    /// Long read HQ all-extension preset.
    pub fn lr_hqae() -> Preset { Preset::LrHqae }
    /// ICLR preset.
    pub fn map_iclr() -> Preset { Preset::MapIclr }
    /// Short read preset.
    pub fn sr() -> Preset { Preset::Sr }
    /// Splice (RNA) preset.
    pub fn splice() -> Preset { Preset::Splice }
    /// Splice high-quality preset.
    pub fn splice_hq() -> Preset { Preset::SpliceHq }
    /// Splice short-read preset.
    pub fn splice_sr() -> Preset { Preset::SpliceSr }
    /// cDNA preset.
    pub fn cdna() -> Preset { Preset::Cdna }
    /// Assembly 5% divergence preset.
    pub fn asm5() -> Preset { Preset::Asm5 }
    /// Assembly 10% divergence preset.
    pub fn asm10() -> Preset { Preset::Asm10 }
    /// Assembly 20% divergence preset.
    pub fn asm20() -> Preset { Preset::Asm20 }
    /// All-vs-all ONT overlap preset.
    pub fn ava_ont() -> Preset { Preset::AvaOnt }
    /// All-vs-all PacBio overlap preset.
    pub fn ava_pb() -> Preset { Preset::AvaPb }

    // --- Constructors ---

    /// Load an aligner from a pre-built `.mmi` index file.
    pub fn from_index(path: &str, preset: Preset) -> io::Result<Self> {
        let index = Index::load(path)?;
        let (mut options, out_cfg) = build_options(preset, index.kmer_size, index.window_size);
        // Calibrate the seed occurrence threshold using the loaded index — matches
        // what from_fasta/from_seqs do. Without this, mid_occ stays at its default
        // (unset) and all seeds are rejected, producing zero alignments.
        options.seeding.mid_occ = index.cal_mid_occ(
            2e-4,
            options.seeding.min_mid_occ,
            options.seeding.max_mid_occ,
        );
        Ok(Aligner { index, options, out_cfg, junc_db: None })
    }

    /// Build an aligner from a FASTA reference file.
    pub fn from_fasta(path: &str, preset: Preset) -> io::Result<Self> {
        let mut k = 15usize;
        let mut w = 10usize;
        let mut is_hpc = false;
        let mut opt = MapOptions::default();
        apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, preset.as_str()).expect("Preset enum variants are all recognized");

        let seqs = crate::fasta::read_fasta(path)?;
        let index = Index::build(seqs, w, k, is_hpc, usize::MAX);
        let out_cfg = OutputConfig {
            do_cigar: true, do_cs: false, cs_long: false, do_md: false, do_ds: false,
            eqx: false, output_sam: false, rg_id: None, split_mode: false,
        };
        opt.seeding.mid_occ = index.cal_mid_occ(2e-4, opt.seeding.min_mid_occ, opt.seeding.max_mid_occ);
        Ok(Aligner { index, options: opt, out_cfg, junc_db: None })
    }

    /// Build an aligner from in-memory sequences (name, sequence pairs).
    ///
    /// ```no_run
    /// use rammap::api::{Aligner, Preset};
    /// let seqs = vec![
    ///     ("chr1".to_string(), b"ACGTACGT...".to_vec()),
    ///     ("chr2".to_string(), b"TGCATGCA...".to_vec()),
    /// ];
    /// let aligner = Aligner::from_seqs(seqs, Preset::MapOnt);
    /// ```
    pub fn from_seqs(seqs: Vec<(String, Vec<u8>)>, preset: Preset) -> Self {
        let mut k = 15usize;
        let mut w = 10usize;
        let mut is_hpc = false;
        let mut opt = MapOptions::default();
        apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, preset.as_str()).expect("Preset enum variants are all recognized");
        opt.flags.insert(AlignFlags::OUT_CIGAR);

        let index = Index::build(seqs, w, k, is_hpc, usize::MAX);
        opt.seeding.mid_occ = index.cal_mid_occ(2e-4, opt.seeding.min_mid_occ, opt.seeding.max_mid_occ);
        let out_cfg = OutputConfig {
            do_cigar: true, do_cs: false, cs_long: false, do_md: false, do_ds: false,
            eqx: false, output_sam: false, rg_id: None, split_mode: false,
        };
        Aligner { index, options: opt, out_cfg, junc_db: None }
    }

    // --- Mapping ---

    /// Align a single-end read against the reference.
    pub fn map_seq(&self, name: &str, seq: &[u8]) -> MapResult {
        self.map_seq_with(name, seq, MapOpts::default())
    }

    /// Align a single-end read with per-call options (CS/MD toggles).
    pub fn map_seq_with(&self, name: &str, seq: &[u8], opts: MapOpts) -> MapResult {
        let mut ctx = AlignmentContext::new();
        let mut map_ctx = MapContext::new();
        let out = self.resolve_out_cfg(&opts);
        let pq = pipeline::process_query(
            &self.options, &self.index, name, seq,
            &mut ctx, &mut map_ctx, self.junc_db.as_ref(), &out,
        );
        to_map_result(&pq, &self.index, &out)
    }

    /// Align a paired-end read pair against the reference.
    pub fn map_pair(&self, name: &str, seq1: &[u8], seq2: &[u8]) -> MapResult {
        self.map_pair_with(name, seq1, seq2, MapOpts::default())
    }

    /// Align a paired-end read pair with per-call options.
    pub fn map_pair_with(&self, name: &str, seq1: &[u8], seq2: &[u8], opts: MapOpts) -> MapResult {
        let mut ctx = AlignmentContext::new();
        let mut map_ctx = MapContext::new();
        let out = self.resolve_out_cfg(&opts);
        let read1 = ReadInfo { qname: name, qseq: seq1, qual: None, comment: None, n_seg: 2, seg_idx: 0 };
        let read2 = ReadInfo { qname: name, qseq: seq2, qual: None, comment: None, n_seg: 2, seg_idx: 1 };
        let (output, _stats) = pipeline::align_and_format_pair(
            &self.options, &self.index, &read1, &read2,
            &mut ctx, &mut map_ctx, self.junc_db.as_ref(), &out,
        );
        parse_paf_to_map_result(&output, &self.index)
    }

    /// Load known splice junctions from a BED file (BED6/BED12), enabling
    /// junction-aware splice scoring on subsequent `map_seq` calls. This is the
    /// rammap equivalent of minimap2's `--junc-bed` (and minimap2-rs's
    /// `read_junction_lr`). Reference names in the BED are resolved against the
    /// built index, so this must be called after construction.
    pub fn load_junctions_bed(&mut self, path: &str) -> io::Result<()> {
        let name_to_rid: HashMap<String, usize> = self
            .index
            .seqs
            .iter()
            .enumerate()
            .map(|(i, sq)| (sq.name.clone(), i))
            .collect();
        let db = junc::load_bed_junctions(path, &name_to_rid, self.index.seqs.len())?;
        self.junc_db = Some(db);
        Ok(())
    }

    /// Load per-position splice-site (SPSC) scores from a file, enabling
    /// score-weighted junction-aware splice alignment on subsequent `map_seq`
    /// calls. This is the library equivalent of the CLI's `--spsc` (the
    /// `--junc-bed` alternative that uses graded donor/acceptor scores rather
    /// than boolean known-junction flags). `scale` defaults to `0.7` (the CLI
    /// default for `--spsc-scale`) when `None`. Must be called after construction
    /// so contig names resolve against the built index.
    pub fn load_junctions_spsc(&mut self, path: &str, scale: Option<f32>) -> io::Result<()> {
        let name_to_rid: HashMap<String, usize> = self
            .index
            .seqs
            .iter()
            .enumerate()
            .map(|(i, sq)| (sq.name.clone(), i))
            .collect();
        let max_sc = junc::max_spsc_bonus(self.options.scoring.gap_open2, self.options.scoring.gap_open);
        let scale = scale.unwrap_or(0.7);
        let seq_lens: Vec<usize> = self.index.seqs.iter().map(|s| s.len).collect();
        let db = junc::load_spsc_scores(
            path,
            &name_to_rid,
            self.index.seqs.len(),
            &seq_lens,
            max_sc,
            scale,
        )?;
        self.junc_db = Some(db);
        Ok(())
    }

    // --- Output ---

    /// Format a pre-computed MapResult as a PAF string.
    pub fn format_paf(&self, name: &str, query_len: usize, result: &MapResult) -> String {
        let mut lines = Vec::new();
        for aln in &result.mappings {
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

    // --- Index management ---

    /// Save the index to a file (RMMI format).
    pub fn save_index(&self, path: &str) -> io::Result<()> {
        self.index.save(path).map_err(io::Error::other)
    }

    /// Access the underlying index.
    pub fn index(&self) -> &Index { &self.index }

    /// Access the current MapOptions.
    pub fn options(&self) -> &MapOptions { &self.options }

    /// Mutably access MapOptions for fine-tuning after construction.
    pub fn options_mut(&mut self) -> &mut MapOptions { &mut self.options }

    /// Mutably access OutputConfig for toggling CIGAR/CS/MD/SAM defaults.
    pub fn output_config_mut(&mut self) -> &mut OutputConfig { &mut self.out_cfg }

    // --- Internal ---

    fn resolve_out_cfg(&self, opts: &MapOpts) -> OutputConfig {
        let mut out = self.out_cfg.clone();
        if let Some(cs) = opts.cs { out.do_cs = cs; }
        if let Some(md) = opts.md { out.do_md = md; }
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_options(preset: Preset, index_k: usize, index_w: usize) -> (MapOptions, OutputConfig) {
    let mut k = index_k;
    let mut w = index_w;
    let mut is_hpc = false;
    let mut opt = MapOptions::default();
    apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, preset.as_str()).expect("Preset enum variants are all recognized");

    let out_cfg = OutputConfig {
        do_cigar: true,
        do_cs: false,
        cs_long: false,
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
    // Cache Arc<str> per target to avoid cloning the name for each alignment
    let mut name_cache: Vec<Option<Arc<str>>> = vec![None; mi.seqs.len()];

    let mappings = pq.results.iter().zip(pq.mapqs.iter()).map(|(r, &mapq)| {
        let cigar_str = if out.do_cigar && !r.cigar_str.is_empty() { Some(r.cigar_str.clone()) } else { None };
        let cigar_ops = cigar_str.as_ref().map(|s| parse_cigar_string(s));
        let trans_strand = match r.trans_strand {
            1 => Some(Strand::Forward),
            2 => Some(Strand::Reverse),
            _ => None,
        };
        let target_name = name_cache[r.ref_id].get_or_insert_with(|| {
            Arc::from(mi.seqs[r.ref_id].name.as_str())
        }).clone();
        Mapping {
            target_name,
            target_id: r.ref_id,
            target_len: mi.seqs[r.ref_id].len,
            target_start: r.ref_start,
            target_end: r.ref_end,
            query_start: r.query_start,
            query_end: r.query_end,
            strand: if r.is_reverse { Strand::Reverse } else { Strand::Forward },
            mapq,
            is_primary: !r.is_secondary,
            is_supplementary: r.split != 0,
            is_spliced: r.is_spliced,
            trans_strand,
            matches: r.matches,
            block_len: r.block_len,
            edit_distance: r.edit_distance,
            cigar: cigar_str,
            cigar_ops,
            cs: if out.do_cs && !r.cs_str.is_empty() { Some(r.cs_str.clone()) } else { None },
            md: if out.do_md && !r.md_str.is_empty() { Some(r.md_str.clone()) } else { None },
            score: r.align_score,
            divergence: r.divergence,
        }
    }).collect();
    MapResult { mappings }
}

/// Parse a CIGAR string like "10M2I5M3D8M" into structured CigarOps.
fn parse_cigar_string(s: &str) -> Vec<CigarOp> {
    let mut ops = Vec::new();
    let mut num = 0u32;
    for c in s.chars() {
        if c.is_ascii_digit() {
            num = num * 10 + (c as u32 - '0' as u32);
        } else {
            let op = match c {
                'M' => 0, 'I' => 1, 'D' => 2, 'N' => 3, 'S' => 4, 'H' => 5, '=' => 7, 'X' => 8,
                _ => continue,
            };
            if num > 0 { ops.push(CigarOp { len: num, op }); }
            num = 0;
        }
    }
    ops
}

/// Parse PAF-formatted output back into a MapResult (for paired-end path).
fn parse_paf_to_map_result(paf: &str, _mi: &Index) -> MapResult {
    let mut mappings = Vec::new();
    for line in paf.lines() {
        if line.is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 { continue; }
        let target_name: Arc<str> = Arc::from(fields[5]);
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

        let cigar_ops = cigar.as_ref().map(|s| parse_cigar_string(s));
        let is_spliced = cigar_ops.as_ref().is_some_and(|ops| ops.iter().any(|op| op.op == 3));

        // Extract md tag
        let mut md = None;
        for &tag in &fields[12..] {
            if let Some(val) = tag.strip_prefix("MD:Z:") { md = Some(val.to_string()); }
        }

        // Find target_id by name
        let target_id = _mi.seqs.iter().position(|s| s.name.as_str() == &*target_name).unwrap_or(0);

        mappings.push(Mapping {
            target_name, target_id, target_len, target_start, target_end,
            query_start, query_end, strand, mapq,
            is_primary: !is_secondary, is_supplementary: false,
            is_spliced, trans_strand: None,
            matches, block_len,
            edit_distance, cigar, cigar_ops, cs, md, score, divergence,
        });
    }
    MapResult { mappings }
}

// ---------------------------------------------------------------------------
// Preset application (moved from main.rs)
// ---------------------------------------------------------------------------

/// Apply a preset by string name (e.g., "map-ont", "sr", "splice").
///
/// This is the same function used by the CLI's `-x` flag. Modifies `opt`,
/// `k`, `w`, and `is_hpc` in place. Returns an error if the preset name
/// is not recognized so the caller can abort rather than silently using
/// the default configuration.
pub fn apply_preset_str(opt: &mut MapOptions, k: &mut usize, w: &mut usize, is_hpc: &mut bool, preset: &str) -> Result<(), String> {
    // Reset SIMD cap each time so switching presets has predictable behavior.
    // Specific presets below opt back into a cap if their workload is hurt by
    // AVX-512's per-core license throttling (chain/seed-heavy presets where DP
    // is a small fraction of CPU time).
    crate::align::dp::set_simd_cap(crate::align::dp::SimdCap::Auto);
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
                    return Err(format!("unrecognized asm preset '{}' (expected asm5, asm10, or asm20)", p));
                }
            }
        },
        "short" | "sr" => {
            // Cap SIMD at AVX2: SR mapping spends ~7-10% of CPU in DP and the rest
            // in chaining/seeding/output, so AVX-512's all-core license throttle
            // (200-600 MHz drop on Skylake-X / Cascade / Cooper Lake) costs more
            // than the DP speedup wins. RAMMAP_FORCE_AVX512=1 overrides this.
            crate::align::dp::set_simd_cap(crate::align::dp::SimdCap::Avx2);
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
            opt.mini_batch_size = 50_000_000;
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
                opt.mini_batch_size = 100_000_000;
            }
        },
        "ava-ont" => {
            // Cap SIMD at AVX2 — AVA workload is dominated by chaining (ALL_CHAINS)
            // with no base-level alignment, so AVX-512 throttling pure-loses here.
            crate::align::dp::set_simd_cap(crate::align::dp::SimdCap::Avx2);
            *k = 15; *w = 5;
            opt.flags.insert(AlignFlags::ALL_CHAINS | AlignFlags::NO_DIAG | AlignFlags::NO_DUAL | AlignFlags::NO_LJOIN);
            opt.chaining.min_chain_score = 100; opt.filtering.pri_ratio = 0.0;
            opt.chaining.max_chain_skip = 25;
            opt.chaining.bandwidth = 2000; opt.chaining.bandwidth_long = 2000;
            opt.seeding.occ_dist = 0;
        },
        "ava-pb" => {
            // Cap SIMD at AVX2 — same reasoning as ava-ont.
            crate::align::dp::set_simd_cap(crate::align::dp::SimdCap::Avx2);
            *is_hpc = true; *k = 19; *w = 5;
            opt.flags.insert(AlignFlags::ALL_CHAINS | AlignFlags::NO_DIAG | AlignFlags::NO_DUAL | AlignFlags::NO_LJOIN);
            opt.chaining.min_chain_score = 100; opt.filtering.pri_ratio = 0.0;
            opt.chaining.max_chain_skip = 25;
            opt.chaining.bandwidth_long = opt.chaining.bandwidth;
            opt.seeding.occ_dist = 0;
        },
        _ => {
            return Err(format!("unrecognized preset '{}'", preset));
        }
    }
    opt.chaining.chn_pen_gap = (opt.chaining.chain_gap_scale as f64 * 0.01 * (*k as f64)) as f32;
    opt.chaining.chn_pen_skip = 0.0;
    Ok(())
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
/// Performs semi-global extension alignment used to fill the gap between two
/// adjacent chained anchors: aligns `query` against `target` with affine gap
/// penalties, returning the best-scoring alignment with CIGAR traceback.
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
        dp::EXTENSION_ONLY | dp::REV_CIGAR,
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
            bandwidth, zdrop, 0, EXTENSION_ONLY, &mut ez,
        );
    } else {
        dp::extend_single_affine(
            query, target, 5, &mat,
            scoring.gap_open as i8, scoring.gap_extend as i8,
            bandwidth, zdrop, 0, EXTENSION_ONLY, &mut ez,
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
    fn test_load_junctions_bed_populates_db() {
        // Build a tiny in-memory index, then load a BED6 junction referencing
        // that contig and confirm the junction DB is populated and wired in.
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec();
        let mut aligner = Aligner::from_seqs(vec![("ctg1".to_string(), seq)], Preset::Splice);
        assert!(aligner.junc_db.is_none(), "junctions should be unset before loading");

        // Write a small BED6 file (contig, start, end, name, score, strand).
        let mut path = std::env::temp_dir();
        path.push(format!("rammap_junc_test_{}.bed", std::process::id()));
        std::fs::write(&path, "ctg1\t5\t25\t.\t0\t+\n").expect("write test BED");

        aligner
            .load_junctions_bed(path.to_str().unwrap())
            .expect("load_junctions_bed should succeed");
        assert!(aligner.junc_db.is_some(), "junctions should be set after loading");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_junctions_spsc_populates_db() {
        // SPSC file: contig, pos, strand, type(D/A), score (tab-separated).
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".to_vec();
        let mut aligner = Aligner::from_seqs(vec![("ctg1".to_string(), seq)], Preset::Splice);
        assert!(aligner.junc_db.is_none());

        let mut path = std::env::temp_dir();
        path.push(format!("rammap_spsc_test_{}.tsv", std::process::id()));
        std::fs::write(&path, "ctg1\t10\t+\tD\t5\nctg1\t20\t+\tA\t5\n")
            .expect("write test SPSC");

        aligner
            .load_junctions_spsc(path.to_str().unwrap(), None)
            .expect("load_junctions_spsc should succeed");
        assert!(aligner.junc_db.is_some(), "junctions should be set after loading SPSC");

        let _ = std::fs::remove_file(&path);
    }

    // Deterministic DNA generator (xorshift) for synthetic splice scenarios.
    fn gen_dna(state: &mut u64, n: usize) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..n).map(|_| {
            *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
            bases[(*state % 4) as usize]
        }).collect()
    }

    // Target offset of the first intron (N) in a CIGAR = sum of target-consuming
    // ops (M/D/=/X) before it. None if the CIGAR has no intron.
    fn intron_donor_offset(cigar: &str) -> Option<i32> {
        let mut consumed = 0i32;
        let mut num = String::new();
        for c in cigar.chars() {
            if c.is_ascii_digit() { num.push(c); continue; }
            let len: i32 = num.parse().unwrap_or(0);
            num.clear();
            match c {
                'N' => return Some(consumed),
                'M' | 'D' | '=' | 'X' => consumed += len,
                _ => {}
            }
        }
        None
    }

    /// Behavioral test: a junction BED must actually change the alignment, not
    /// just populate a field. Build a two-exon reference with an intron and
    /// align the spliced transcript. Without an annotation the splice site
    /// lands a few bp off; a BED at the true intron boundary snaps the intron
    /// to the annotated position and improves the score — proving the junction
    /// DB reaches the splice DP through the Aligner.
    #[test]
    fn test_junctions_reposition_splice_site() {
        let mut st = 0xC0FFEE_u64;
        let exon1 = gen_dna(&mut st, 100);
        let intron = gen_dna(&mut st, 100);
        let exon2 = gen_dna(&mut st, 100);
        let reference: Vec<u8> = exon1.iter().chain(&intron).chain(&exon2).copied().collect();
        let query: Vec<u8> = exon1.iter().chain(&exon2).copied().collect();
        let intron_start = exon1.len() as i32;
        let intron_end = exon1.len() + intron.len();

        // Baseline: no junction annotation.
        let a0 = Aligner::from_seqs(vec![("chr1".to_string(), reference.clone())], Preset::Splice);
        let r0 = a0.map_seq("read", &query);
        assert_eq!(r0.mappings.len(), 1, "expected one mapping without junctions");
        let m0 = &r0.mappings[0];
        let c0 = m0.cigar.clone().expect("cigar present");

        // Same alignment, with the true intron annotated as a junction.
        let mut a1 = Aligner::from_seqs(vec![("chr1".to_string(), reference)], Preset::Splice);
        let mut path = std::env::temp_dir();
        path.push(format!("rammap_junc_behavior_{}.bed", std::process::id()));
        std::fs::write(&path, format!("chr1\t{}\t{}\t.\t0\t+\n", intron_start, intron_end))
            .expect("write test BED");
        a1.load_junctions_bed(path.to_str().unwrap()).expect("load_junctions_bed");
        let _ = std::fs::remove_file(&path);
        let r1 = a1.map_seq("read", &query);
        assert_eq!(r1.mappings.len(), 1, "expected one mapping with junctions");
        let m1 = &r1.mappings[0];
        let c1 = m1.cigar.clone().expect("cigar present");

        assert!(m0.is_spliced && m1.is_spliced, "both alignments should splice (c0={c0}, c1={c1})");
        // The annotation changed the alignment and improved its score.
        assert_ne!(c0, c1, "junction BED should change the CIGAR (both = {c0})");
        assert!(m1.score > m0.score, "junction bonus should raise the score: with={} without={}", m1.score, m0.score);
        // With the annotation the intron snaps exactly to the annotated donor;
        // without it, the splice site is mis-placed.
        assert_eq!(intron_donor_offset(&c1), Some(intron_start),
            "annotated splice should sit at the junction (cigar {c1})");
        assert_ne!(intron_donor_offset(&c0), Some(intron_start),
            "unannotated splice should be mis-placed (cigar {c0})");
    }

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
        apply_preset_str(&mut opt, &mut k, &mut w, &mut is_hpc, "sr").unwrap();
        assert_eq!(k, 21);
        assert_eq!(w, 11);
        assert!(opt.flags.contains(AlignFlags::SHORT_READ));
    }

    #[test]
    fn test_map_result_empty() {
        let result = MapResult { mappings: Vec::new() };
        assert!(result.mappings.is_empty());
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
