# Extensible Pipeline Architecture

rammap's alignment pipeline is built around trait-based interfaces at each
major stage. Each stage has a production implementation and one or more
alternative implementations that can be swapped in at compile time — no
runtime overhead, no code changes to the rest of the pipeline.

## Pipeline Stages and Traits

```
Query FASTQ
  → Sketcher      (sketch.rs)       produces Vec<Minimizer>
  → Index lookup  (seed.rs)         hash table, not trait-based
  → Chainer       (chain.rs)        produces (Vec<u64>, Vec<Minimizer>)
  → Filter        (filter.rs)       not trait-based
  → Aligner       (extend.rs)       produces AlignResult
  → Output        (pipeline.rs)     PAF/SAM formatting
```

Three stages expose swappable trait interfaces:

| Trait | Defined in | Input | Output |
|-------|-----------|-------|--------|
| `Sketcher` | `sketch.rs` | `&[u8]` sequence | `Vec<Minimizer>` seeds |
| `Chainer` | `map.rs` | `&mut [Minimizer]` anchors | `(Vec<u64>, Vec<Minimizer>)` chains |
| `Aligner` | `map.rs` | anchors + nt4 sequences | `AlignResult` (CIGAR, coords, score) |

All implementations produce the same output types, so downstream stages
don't need to know which implementation was used.

## Available Implementations

### Sketching

| Implementation | File | Algorithm | Use case |
|---------------|------|-----------|----------|
| `MinimizerSketcher` | sketch.rs | (w,k)-minimizer: smallest hash in sliding window | Production (default) |
| `SyncmerSketcher` | syncmer.rs | Open syncmer: min s-mer at first/last position | More uniform spacing, mutation-robust |
| `RandstrobeSketcher` | strobemer.rs | Coupled k-mer pairs at variable distance | Wider span, survives point mutations |

**Trait interface:**

```rust
pub trait Sketcher {
    fn sketch(&self, seq: &[u8], len: usize, rid: usize, out: &mut Vec<Minimizer>);
}
```

**Example — syncmer sketcher:**

```rust
use rammap::align::syncmer::SyncmerSketcher;
use rammap::align::sketch::Sketcher;

let sketcher = SyncmerSketcher::new(15, 11); // k=15, s=11
let mut seeds = Vec::new();
sketcher.sketch(sequence, sequence.len(), 0, &mut seeds);
// seeds: Vec<Minimizer> — same type as MinimizerSketcher produces
```

### Chaining

| Implementation | File | Algorithm | Use case |
|---------------|------|-----------|----------|
| `DpChainer` | chain.rs | DP with gap penalties, O(n * max\_skip) | Production (default) |
| `RmqChainer` | chain\_rmq.rs | Range-minimum-query chaining | Assembly presets (asm5/10/20) |
| `GreedyChainer` | chain\_simple.rs | Single-linkage scan, O(n) | Fast prototyping, simple baselines |

**Trait interface:**

```rust
pub trait Chainer {
    fn chain(
        &self,
        params: &ChainingParams,
        anchors: &mut [Minimizer],
        bufs: &mut ChainingBuffers,
    ) -> (Vec<u64>, Vec<Minimizer>);
}
```

The output `Vec<u64>` encodes `(score << 32 | anchor_count)` per chain.
Anchors are packed contiguously in the `Vec<Minimizer>`, with each chain's
anchors forming a consecutive slice.

**Example — greedy chainer:**

```rust
use rammap::align::chain_simple::GreedyChainer;
use rammap::align::map::{Chainer, ChainingBuffers};

let chainer = GreedyChainer;
let mut bufs = ChainingBuffers::new();
let (chain_scores, chained_anchors) = chainer.chain(&params, &mut anchors, &mut bufs);
// Same output format as DpChainer — downstream code works unchanged
```

### Alignment

| Implementation | File | Algorithm | Use case |
|---------------|------|-----------|----------|
| `RMAligner` | extend.rs + dp.rs | SIMD ksw2 with z-drop, splice, dual-affine | Production (default) |
| `NWAligner` | align\_simple.rs | Self-contained NW per gap, affine gaps | Simple baseline, no SIMD dependency |

**Trait interface:**

```rust
pub trait Aligner {
    fn align(
        &self,
        anchors: &mut [Minimizer],
        qseq: &[u8],          // nt4-encoded query
        tseq: &[u8],          // nt4-encoded target region
        opt: &MapOptions,
        ctx: &mut AlignmentContext,
        call: &AlignAnchorContext,
    ) -> AlignResult;
}
```

`AlignResult` contains CIGAR ops, coordinates, score, and optional
z-drop split anchors. All coordinates are region-relative (the caller
manages the offset to absolute chromosome positions).

**Example — NW aligner:**

```rust
use rammap::align::align_simple::NWAligner;
use rammap::align::map::Aligner;

let aligner = NWAligner;
let result = aligner.align(&mut anchors, &qseq, &tseq, &opt, &mut ctx, &call);
// result.cigar_ops, result.ref_start, result.score — same struct as RMAligner
// Uses self-contained Needleman-Wunsch with affine gaps for each inter-anchor gap
```

## Swapping Implementations via Cargo Features

To swap an implementation at compile time, add feature flags to `Cargo.toml`
and use `#[cfg(feature = "...")]` in the pipeline code. This is zero-cost —
the unused implementation is not compiled.

### Step 1: Add features to Cargo.toml

```toml
[features]
default = ["parallel", "cli"]
simple-chain = []
simple-align = []
```

### Step 2: Swap chaining in map.rs

In `map_query()`, replace the chaining call:

```rust
// Production (default)
#[cfg(not(feature = "simple-chain"))]
let (u, chains) = if use_rmq {
    chain_anchors_rmq(&opt.chaining, &mut anchors, &mut ctx.chain_bufs)
} else {
    chain_anchors(&opt.chaining, is_cdna, n_segs, max_x, max_y,
                  &mut anchors, &mut ctx.chain_bufs)
};

// Alternative: greedy O(n) chainer
#[cfg(feature = "simple-chain")]
let (u, chains) = {
    use crate::align::chain_simple::GreedyChainer;
    use crate::align::map::Chainer;
    GreedyChainer.chain(&opt.chaining, &mut anchors, &mut ctx.chain_bufs)
};
```

### Step 3: Swap alignment in pipeline.rs

In `align_single_mapping()`, replace the alignment call:

```rust
// Production (default)
#[cfg(not(feature = "simple-align"))]
let aln_result = align_anchors(
    &mut mapping.anchors, &query_seq_for_aln,
    target_region, opt, ctx, &call_ctx,
);

// Alternative: NW aligner (self-contained, no SIMD)
#[cfg(feature = "simple-align")]
let aln_result = {
    use crate::align::align_simple::NWAligner;
    use crate::align::map::Aligner;
    NWAligner.align(
        &mut mapping.anchors, &query_seq_for_aln,
        target_region, opt, ctx, &call_ctx,
    )
};
```

### Step 4: Build

```bash
# Production (default):
cargo build --release

# Greedy chainer:
cargo build --release --features simple-chain

# Anchor-only aligner:
cargo build --release --features simple-align

# Both:
cargo build --release --features simple-chain,simple-align
```

The binary is identical in usage — same CLI, same output format (PAF/SAM).
Only the internal algorithm changes.

## Writing Your Own Implementation

To add a new sketching, chaining, or alignment strategy:

### 1. Create a new module

```rust
// src/align/my_sketcher.rs
use crate::align::sketch::{Minimizer, Sketcher, encode_base, kmer_hash};

pub struct MyCustomSketcher {
    pub k: usize,
    // your parameters
}

impl Sketcher for MyCustomSketcher {
    fn sketch(&self, seq: &[u8], len: usize, _rid: usize, out: &mut Vec<Minimizer>) {
        out.clear();
        // Your seed selection logic here.
        // For each selected k-mer, push a Minimizer with:
        //   x = (hash << 8) | kmer_span
        //   y = (position << 1) | strand
    }
}
```

### 2. Register the module

Add to `src/align/mod.rs`:

```rust
pub mod my_sketcher;
```

### 3. Wire it in

Use the Cargo feature pattern shown above, or call it directly via the
library API for experimentation:

```rust
use rammap::align::my_sketcher::MyCustomSketcher;
use rammap::align::sketch::Sketcher;

let sketcher = MyCustomSketcher { k: 15 };
let mut seeds = Vec::new();
sketcher.sketch(seq, seq.len(), 0, &mut seeds);
```

### Key constraints

- **Sketcher**: Output `Minimizer` values must use the standard packing
  (`hash << 8 | span` in x, `pos << 1 | strand` in y) for index lookup
  compatibility.

- **Chainer**: Output `Vec<u64>` must encode `(score << 32 | count)` per
  chain, with anchors packed contiguously in the output `Vec<Minimizer>`.

- **Aligner**: Input anchors have region-relative `ref_pos` (caller
  subtracts `rgn_start`). Output `AlignResult` coordinates are also
  region-relative. CIGAR ops must be in =/X/I/D/N format.

## File Locations

```
src/align/
  sketch.rs          Sketcher trait + MinimizerSketcher (production)
  syncmer.rs         SyncmerSketcher (alternative)
  strobemer.rs       RandstrobeSketcher (alternative)
  chain.rs           DpChainer (production)
  chain_rmq.rs       RmqChainer (production, assembly presets)
  chain_simple.rs    GreedyChainer (alternative)
  extend.rs + dp.rs  RMAligner (production)
  align_simple.rs    NWAligner (alternative)
  map.rs             Chainer + Aligner trait definitions
```
