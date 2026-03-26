# rammap

A pure-Rust extensible sequence aligner and mapper intended to mirror the interface and produce identical output to [minimap2](https://github.com/lh3/minimap2).

Supports all major minimap2 presets (map-ont, map-hifi, sr, splice, asm, ava)
with full CIGAR, CS/MD tags, SAM, and PAF output. SIMD-optimized DP kernels
(SSE2, AVX2, AVX512BW) with scalar fallback.

## Quick Start (CLI)

```bash
cargo build --release
./target/release/rammap -x map-ont -c reference.fa reads.fq > alignments.paf
./target/release/rammap -x sr -a reference.fa reads_R1.fq reads_R2.fq > alignments.sam
```

### Presets

| Preset | Data type |
|--------|-----------|
| `map-ont` | Oxford Nanopore reads |
| `map-pb` | PacBio CLR reads |
| `map-hifi` | PacBio HiFi reads |
| `lr:hq` | Accurate long reads |
| `sr` | Short reads (Illumina) |
| `splice` | Long-read RNA-seq |
| `splice:hq` | High-quality RNA-seq |
| `cdna` | cDNA-seq |
| `asm5` / `asm10` / `asm20` | Assembly-to-reference (1-5% / 10% / 20% divergence) |
| `ava-ont` / `ava-pb` | All-vs-all overlap |

## Library API

rammap can be used as a Rust library for programmatic alignment.

### Add to your project

```toml
[dependencies]
rammap = { path = "../rammap", default-features = false, features = ["parallel"] }
```

### Example

```rust
use rammap::{Aligner, Preset, Strand};

fn main() -> std::io::Result<()> {
    // Load a reference (from FASTA or pre-built .mmi index)
    let aligner = Aligner::from_fasta("reference.fa", Preset::MapOnt)?;

    // Align a read
    let result = aligner.map_seq("read1", b"ACGTACGTACGT...");

    for aln in &result.alignments {
        let strand = match aln.strand {
            Strand::Forward => "+",
            Strand::Reverse => "-",
        };
        println!("{} {}:{}-{} strand={} mapq={} score={}",
            aln.target_name, aln.target_start, aln.target_end,
            strand, aln.mapq, aln.score);

        if let Some(ref cigar) = aln.cigar {
            println!("  CIGAR: {}", cigar);
        }
    }
    Ok(())
}
```

### API Reference

#### `Aligner`

```rust
// Load from pre-built minimap2 index (fast)
let aligner = Aligner::from_index("reference.mmi", Preset::Sr)?;

// Build index from FASTA (slower, builds in memory)
let aligner = Aligner::from_fasta("reference.fa", Preset::MapOnt)?;

// Align single-end read
let result = aligner.map_seq("read_name", sequence_bytes);

// Align paired-end reads
let result = aligner.map_pair("read_name", seq_r1, seq_r2);

// Fine-tune parameters after construction
aligner.options_mut().chaining.max_gap = 10000;
aligner.options_mut().filtering.best_n = 50;
```

#### `MapResult` / `Alignment`

```rust
let result = aligner.map_seq("read1", seq);

for aln in &result.alignments {
    aln.target_name     // "chr1", "chr20", etc.
    aln.target_start    // 0-based start on reference
    aln.target_end      // exclusive end on reference
    aln.query_start     // 0-based start on query
    aln.query_end       // exclusive end on query
    aln.strand          // Strand::Forward or Strand::Reverse
    aln.mapq            // mapping quality (0-60)
    aln.is_primary      // true for primary alignment
    aln.matches         // number of matching bases
    aln.block_len       // alignment block length
    aln.edit_distance   // NM tag value
    aln.cigar           // Option<String> — CIGAR string
    aln.score           // alignment score (AS tag)
    aln.divergence      // sequence divergence (0.0 = identical)
}
```

#### Thread Safety

`Aligner` is `Send + Sync`. For multi-threaded use:

```rust
use std::sync::Arc;

let aligner = Arc::new(Aligner::from_index("ref.mmi", Preset::MapOnt)?);

// Each thread gets a clone of the Arc
let aligner_clone = aligner.clone();
std::thread::spawn(move || {
    let result = aligner_clone.map_seq("read", seq);
});
```

### Low-Level DP Alignment

The SIMD-optimized DP engine can be used directly for pairwise sequence
alignment, without the full mapping pipeline:

```rust
use rammap::{dp_align, dp_global, dp_local, dp_extension, DpScoring, encode_nt4};

// Encode sequences to nt4 (A=0, C=1, G=2, T=3, N=4)
let query  = encode_nt4(b"ACGTACGTACGT");
let target = encode_nt4(b"ACGTACCTACGT");

// Semi-global alignment (default — no end penalty)
let result = dp_align(&query, &target, &DpScoring::default(), -1);
println!("score={} cigar={}", result.score, result.cigar);

// Global alignment (Needleman-Wunsch — covers both sequences end-to-end)
let result = dp_global(&query, &target, &DpScoring::default(), -1);

// Local alignment (Smith-Waterman — finds best-scoring local region)
let result = dp_local(&query, &target, &DpScoring::default());

// Extension alignment (stops at best score — for extending seed matches)
let result = dp_extension(&query, &target, &DpScoring::default(), 100);

// Custom scoring
let scoring = DpScoring {
    match_score: 1, mismatch: 4,
    gap_open: 6, gap_extend: 2,
    gap_open2: 26, gap_extend2: 1,  // dual-affine (long gaps)
};
let result = dp_align(&query, &target, &scoring, -1);
```

The DP engine automatically selects the best SIMD backend (AVX512 > AVX2 > SSE2 > scalar).

### Running the Examples

```bash
# Map reads against a reference
cargo run --release --example simple_align -- reference.fa reads.fq

# Pairwise DP alignment
cargo run --release --example dp_align
```

## Building

```bash
# Full build (CLI + library)
cargo build --release

# Library only (no CLI dependencies)
cargo build --lib --release --no-default-features --features parallel

# Run tests
cargo test

# Run integration tests (requires test data)
bash tests/integration_test.sh
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a detailed guide to the
codebase, and [`docs/extensibility.md`](docs/extensibility.md) for the educational
pipeline module implementations.

## Performance

See [`docs/performance.md`](docs/performance.md) for detailed benchmarks against minimap2.

Highlights (GRCh38, 8 threads, 20K reads):
- **map-ont**: 22% faster wall time, byte-identical output
- **sr**: parity on speed, 3.5 GB less memory
- **100% concordance** on 29/30 test presets (1 minimap2 undefined behavior)

## License

MIT
