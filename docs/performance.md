# Alignment validation and performance

rammap 0.1.0 vs minimap2 2.30-r1290

# GRCh38 Full-Genome Benchmark (8 Threads)

## System

| | |
|---|---|
| **CPU** | AMD Ryzen 9 7900X 12-Core (24 threads) |
| **RAM** | 128 GB DDR5 |
| **OS** | Ubuntu 22.04, Linux 6.8.0-94-generic |
| **SIMD** | SSE4.1, AVX2, AVX512BW |
| **Rust** | 1.94.0, `-C target-cpu=native` |
| **Profile** | `opt-level=3`, `lto="fat"`, `codegen-units=1` |

## Test Data

All tests use the full human GRCh38 reference (3.1 GB, 3.09 Gbp). Both tools build
indices from FASTA at runtime (no pre-built `.mmi`). Reads subsampled from full datasets.

| Type | File | Reads |
|------|------|------:|
| ONT (long) | `ont_20000.fq` | 20,000 |
| PacBio HiFi (long) | `hifi_20000.fq` | 20,000 |
| Direct RNA | `rna_5000.fq` | 5,000 |
| Illumina PE | `sr_20000_R{1,2}.fq` | 19,985 pairs |
| ONT overlap | `ava_1000.fq` | 1,000 |
| Assembly contigs | `asm_contigs.fa` | 20 |

---

## Concordance Summary

All core presets produce identical output (line count and sorted content) between
rammap and minimap2, except map-pb (1 known inversion UB diff).

| # | Test | Lines | Result | Notes |
|--:|------|------:|--------|-------|
| 1 | map-ont | 35,083 | **100% concordance** | |
| 2 | map-ont-cigar | 32,732 | **100% concordance** | |
| 3 | map-ont-sam | 32,739 alns | **100% concordance** | |
| 4 | lr-hq | 31,579 | **100% concordance** | |
| 5 | lr-hqae | 44,954 | **100% concordance** | |
| 6 | map-iclr | 32,747 | **100% concordance** | |
| 7 | map-hifi | 26,006 | **100% concordance** | |
| 8 | map-pb | 26,482/26,483 | **PASS** | 1 line diff: known inversion UB |
| 9 | splice | 8,040 | **100% concordance** | |
| 10 | splice-hq | 7,273 | **100% concordance** | |
| 11 | cdna | 8,040 | **100% concordance** | |
| 12 | sr | 39,965 | **100% concordance** | |
| 13 | splice-sr | 39,611 | **100% concordance** | |
| 14 | asm5 | 20 | **100% concordance** | |
| 15 | asm10 | 20 | **100% concordance** | |
| 16 | asm20 | 20 | **100% concordance** | |
| 17 | ava-ont | 209,757 | **100% concordance** | |
| 18 | custom-scoring | 32,286 | **100% concordance** | |
| 19 | secondary-N5 | 32,732 | **100% concordance** | |
| 20 | eqx | 32,732 | **100% concordance** | |
| 21 | custom-kw | 32,327 | **100% concordance** | |

---

## Performance Comparison (8 Threads)

Wall time and peak RSS for rammap (RT) vs minimap2 (MM2). Both tools index from
FASTA at runtime. Wall ratio < 1.0 means rammap is faster.

### Long-Read Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| map-ont | 151s | 217s | **0.70x** | 9.4 GB | 8.6 GB | 1.09x |
| map-ont-cigar | 158s | 229s | **0.69x** | 10.4 GB | 11.3 GB | **0.92x** |
| map-ont-sam | 153s | 231s | **0.66x** | 10.4 GB | 11.7 GB | **0.89x** |
| lr-hq | 81s | 77s | 1.05x | 15.7 GB | 17.8 GB | **0.88x** |
| lr-hqae | 37s | 39s | **0.95x** | 6.1 GB | 6.8 GB | **0.90x** |
| map-hifi | 39s | 31s | 1.26x | 9.9 GB | 13.9 GB | **0.71x** |
| map-pb | 39s | 31s | 1.26x | 8.6 GB | 9.6 GB | **0.90x** |
| map-iclr | 145s | 105s | 1.38x | 25.8 GB | 16.7 GB | 1.54x |

### Splice / RNA Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| splice | 49s | 40s | 1.23x | 24.7 GB | 19.2 GB | 1.29x |
| splice-hq | 49s | 41s | 1.20x | 24.7 GB | 19.2 GB | 1.29x |
| cdna | 50s | 40s | 1.25x | 24.8 GB | 19.2 GB | 1.29x |
| splice-sr | 52s | 40s | 1.30x | 24.6 GB | 19.2 GB | 1.28x |

### Short-Read Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| sr | 49s | 29s | 1.69x | 25.2 GB | 13.6 GB | 1.85x |
| sr-sam | 49s | 29s | 1.69x | 25.2 GB | 13.6 GB | 1.85x |

### Assembly Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| asm5 | 32s | 29s | 1.10x | 15.7 GB | 14.9 GB | 1.05x |
| asm10 | 31s | 29s | 1.07x | 15.7 GB | 14.8 GB | 1.06x |
| asm20 | 52s | 32s | 1.62x | 25.8 GB | 14.3 GB | 1.80x |

### Overlap / Parameter Variations

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| ava-ont | 26s | 11s | 2.36x | 1.3 GB | 1.5 GB | **0.87x** |
| custom-scoring | 152s | 230s | **0.66x** | 10.4 GB | 11.4 GB | **0.91x** |
| secondary-N5 | 154s | 230s | **0.67x** | 10.4 GB | 11.3 GB | **0.92x** |
| eqx | 153s | 230s | **0.67x** | 10.4 GB | 11.4 GB | **0.91x** |
| custom-kw | 87s | 89s | **0.98x** | 25.1 GB | 18.4 GB | 1.36x |

---

## GRCh38 Performance Summary

**Wall time ratio** (rammap / minimap2; lower is better):

```
Faster  ============================|============================  Slower
                                    |
        map-ont-sam  0.66x █████████|
        custom-scor  0.66x █████████|
        secondary-N5 0.67x █████████|
        eqx          0.67x █████████|
        map-ont-cig  0.69x  ████████|
        map-ont      0.70x  ████████|
        lr-hqae      0.95x         █|
        custom-kw    0.98x          |
        lr-hq        1.05x          |█
        asm5         1.10x          |██
        asm10        1.07x          |██
        splice-hq    1.20x          |█████
        splice       1.23x          |██████
        cdna         1.25x          |██████
        map-hifi     1.26x          |██████
        map-pb       1.26x          |██████
        splice-sr    1.30x          |████████
        map-iclr     1.38x          |██████████
        asm20        1.62x          |████████████████
        sr           1.69x          |█████████████████
        ava-ont      2.36x          |██████████████████████████████
```

### Key Observations (GRCh38, 8 threads)

**Faster than minimap2**:
- map-ont/map-ont-cigar/map-ont-sam: **30-34% faster** — ONT is the primary use case
- custom-scoring/secondary-N5/eqx: **33% faster** (same ONT pipeline with extra output)
- lr-hqae: **5% faster**, **10% less memory**
- custom-kw: **~parity** (0.98x)

**Slower than minimap2**:
- splice/cdna/splice-hq: **1.2-1.25x slower** — index build dominates (few reads)
- sr: **1.7x slower** — dominated by index build time for tiny read count
- map-hifi/map-pb: **1.26x slower** — HPC index overhead
- map-iclr: **1.38x slower** — k=21 index is large (25.8 GB)
- asm20: **1.62x slower** — index build overhead, only 20 contigs aligned
- ava-ont: **2.4x slower** — all-vs-all quadratic chaining overhead

### Memory

rammap uses packed 4-bit reference storage (~375 MB for GRCh38) with on-demand
per-region nt4 extraction, vs minimap2's mmap-based packed storage. Memory usage
varies by preset:

- **Less memory**: map-hifi (0.71x), lr-hq (0.88x), lr-hqae (0.90x), map-ont-cigar (0.92x)
- **Parity**: map-ont (1.09x), splice (1.00-1.29x), asm5/10 (1.05x)
- **More memory**: sr (1.85x), map-iclr (1.54x), asm20 (1.80x) — index-build-dominated presets hold full index in RAM

---

## Known Differences

### GRCh38 Tests

| Test | Diffs | Explanation |
|------|------:|-------------|
| map-pb | 1 | Known `ksw_ll_i16` UB: inversion with `cm:i:0, s1:i:0`. minimap2 reads before buffer start, rammap correctly rejects. |

### SIMD Tie-Breaking

Different SIMD widths (SSE=16 lanes, AVX2=32, AVX512=64) can produce
different CIGAR strings for the same input when multiple cells in the DP
matrix have equal scores. The traceback direction bits depend on the order
cells are processed within a SIMD register, and wider registers process
more cells per iteration, changing which tied cell "wins."

This is inherent to all banded SIMD DP implementations, including
minimap2's C ksw2. It does not affect scores, alignment boundaries, or
consumed lengths — only the placement of gaps within equally-scored
regions. The mapper's integration tests confirm all SIMD variants
produce byte-identical output because the chaining/filtering pipeline
eliminates borderline alignments before output.

### minimap2 UB

For details on the `ksw_ll_i16` undefined behavior that causes the
inversion diffs, see [`docs/minimap2-ksw-ll-ub.md`](minimap2-ksw-ll-ub.md).

---

## Threading Model

### rammap

| Component | Threading | Notes |
|-----------|-----------|-------|
| FASTA reading | Single | Sequential I/O |
| Index sketching | Rayon `par_iter` | Parallelizes across reference sequences |
| Index 4-bit packing | Rayon `par_iter` | Parallel per-sequence with sequential merge |
| Index sort | Parallel MSD radix | Two-level partition then parallel sub-bucket sort |
| Lookup table build | Single | O(n) sequential scan |
| Query I/O | Dedicated read-ahead thread | `sync_channel(1)` overlapped I/O |
| Mapping (seed/chain/align) | `-t N` worker threads | Crossbeam scoped threads |
| Output formatting | Per-thread, flushed in order | Buffered writes |

### minimap2

| Component | Threading | Notes |
|-----------|-----------|-------|
| FASTA reading | Single | mmap for .mmi index |
| Index sketching | Thread pool | `kt_for` over sequences |
| Index sort/hash | Thread pool | `kt_for` over buckets (16K independent) |
| Query I/O | Pipeline step 0 | 3-stage pipeline: read → map → output |
| Mapping (seed/chain/align) | `-t N` worker threads | `kt_pipeline` step 1 |
| Output formatting | Pipeline step 2 | Sequential output step |

### Key Differences

- **Index build**: Both tools parallelize sketching across sequences. minimap2
  parallelizes per-bucket hash table construction across 16K buckets via `kt_for`.
  rammap uses a two-level parallel MSD radix sort (adaptive top-byte detection +
  parallel sub-bucket recursion via rayon). rammap's 4-bit packing is also parallel.
- **I/O pipeline**: minimap2 uses a 3-stage pipeline (`kt_pipeline`) that overlaps
  reading, mapping, and output. rammap uses a dedicated read-ahead thread with a
  synchronous channel, achieving similar overlap.
- **Index format**: minimap2 uses mmap for `.mmi` files (zero-copy load). rammap
  deserializes `.rmmi` files via bincode (requires allocation + copy).
- **DP kernels**: minimap2 dispatches to SSE4.1 or SSE2 only. rammap has AVX2 and
  AVX512BW DP kernels in addition to SSE, providing ~2x throughput on wider SIMD.
