# rammap

A pure-Rust extensible sequence aligner and mapper mirroring the interface and producing identical output to [minimap2](https://github.com/lh3/minimap2).

Supports all minimap2 presets (map-ont, map-hifi, sr, splice, asm, ava) with full CIGAR, CS/MD tags, SAM, and PAF output. SIMD-optimized DP kernels
(NEON, SSE2, SSE4.1, AVX2, AVX512BW, WASM v128) with scalar memory safe fallback.

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
| `map-pb` | PacBio CLR reads (HPC seeds) |
| `map-hifi` | PacBio HiFi reads |
| `map-iclr` | Illumina Complete Long Reads |
| `lr:hq` | Accurate long reads |
| `lr:hqae` | HQ long-read assembly eval |
| `sr` | Short reads (Illumina) |
| `splice` | Long-read RNA-seq |
| `splice:hq` | High-quality RNA-seq |
| `splice:sr` | Spliced short RNA-seq |
| `cdna` | cDNA / splice alias |
| `asm5` / `asm10` / `asm20` | Assembly-to-reference (~0.1% / 1% / 5% divergence) |
| `ava-ont` / `ava-pb` | All-vs-all overlap |

Run `rammap -x --help` for the full list of accepted preset names and aliases.

## Library API

rammap can be used as a Rust library for programmatic alignment, with an API
compatible with [minimap2-rs](https://github.com/jguhlin/minimap2-rs).

### Add to your project

```toml
[dependencies]
# The library is published as `rammap-core` but imported in code as `rammap`
# (e.g. `use rammap::api::Aligner;`).
rammap-core = { version = "1.1.0", default-features = false, features = ["parallel"] }
```

Library features (on `rammap-core`): `parallel` (rayon-based threading), `wasm-threads`
(browser-side rayon for WASM builds). The CLI is a separate crate (`rammap`, which
depends on `rammap-core`) and provides the binary plus the `jemalloc` / `mimalloc`
opt-in global allocators (off by default).

### Example

```rust
use rammap::{Aligner, Preset, Strand};

fn main() -> std::io::Result<()> {
    // Load a reference (from FASTA or pre-built .mmi index)
    let aligner = Aligner::from_fasta("reference.fa", Preset::MapOnt)?;

    // Or use builder-style preset methods:
    // let aligner = Aligner::from_fasta("reference.fa", Aligner::map_ont())?;

    // Align a read
    let result = aligner.map_seq("read1", b"ACGTACGTACGT...");

    for m in &result.mappings {
        let strand = match m.strand {
            Strand::Forward => "+",
            Strand::Reverse => "-",
        };
        println!("{} {}:{}-{} strand={} mapq={} score={}",
            m.target_name, m.target_start, m.target_end,
            strand, m.mapq, m.score);

        if let Some(ref cigar) = m.cigar {
            println!("  CIGAR: {}", cigar);
        }

        // Structured CIGAR also available:
        if let Some(ref ops) = m.cigar_ops {
            for op in ops {
                print!("{}{}", op.len, op.op_char());
            }
            println!();
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

// Build index from in-memory sequences
let seqs = vec![("chr1".to_string(), b"ACGT...".to_vec())];
let aligner = Aligner::from_seqs(seqs, Preset::MapOnt);

// Builder-style preset methods (minimap2-rs compatible)
let aligner = Aligner::from_fasta("ref.fa", Aligner::map_ont())?;
let aligner = Aligner::from_fasta("ref.fa", Aligner::splice())?;

// Align single-end read
let result = aligner.map_seq("read_name", sequence_bytes);

// Align paired-end reads
let result = aligner.map_pair("read_name", seq_r1, seq_r2);

// Per-call CS/MD tag toggle
use rammap::MapOpts;
let result = aligner.map_seq_with("read1", seq, MapOpts { cs: Some(true), md: Some(true) });
assert!(result.mappings[0].cs.is_some());

// Save index for reuse
aligner.save_index("output.rmmi")?;

// Fine-tune parameters after construction
aligner.options_mut().chaining.max_gap = 10000;
aligner.options_mut().filtering.best_n = 50;

// Toggle output options
aligner.output_config_mut().eqx = true;  // =/X CIGAR
```

#### `MapResult` / `Mapping`

```rust
let result = aligner.map_seq("read1", seq);

for m in &result.mappings {
    m.target_name       // Arc<str> — "chr1", "chr20", etc. (shared across mappings)
    m.target_id         // numeric index into reference
    m.target_start      // 0-based start on reference
    m.target_end        // exclusive end on reference
    m.query_start       // 0-based start on query
    m.query_end         // exclusive end on query
    m.strand            // Strand::Forward or Strand::Reverse
    m.mapq              // mapping quality (0-60)
    m.is_primary        // true for primary alignment
    m.is_supplementary  // true for supplementary alignment
    m.is_spliced        // true if contains intron (N_SKIP) operations
    m.trans_strand      // transcript strand for splice alignments
    m.matches           // number of matching bases
    m.block_len         // alignment block length
    m.edit_distance     // NM tag value
    m.cigar             // Option<String> — CIGAR string
    m.cigar_ops         // Option<Vec<CigarOp>> — structured CIGAR
    m.cs                // Option<String> — CS tag (if requested)
    m.md                // Option<String> — MD tag (if requested)
    m.score             // alignment score (AS tag)
    m.divergence        // sequence divergence (0.0 = identical)
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

The DP engine automatically selects the best SIMD backend (AVX512 > AVX2 > SSE4.1 > SSE2 > NEON > WASM > scalar).

## Building

```bash
# Full build (CLI + library) — builds both workspace crates
cargo build --release

# Library only (no CLI dependencies)
cargo build -p rammap-core --release --no-default-features --features parallel

# Run tests
cargo test

# Run integration tests (requires test data)
bash tests/integration_test.sh
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a detailed guide to the codebase.

## Extensibility

See [`docs/extensibility.md`](docs/extensibility.md) for documentation on building and replacing
modular components (seeding, chaining, alignment, etc).

## Performance

See [`docs/performance.md`](docs/performance.md) for detailed benchmarks against minimap2.

## WASM

See [`docs/wasm.md`](docs/wasm.md) for WASM build instructions, web demo, and benchmarks.

## Python

[`rammappy`](https://github.com/tomdstanton/rammappy) is an externally maintained module that provides Python bindings directly to the `rammap-core` API via PyO3.

## License

MIT
