# Alignment validation and performance

rammap 0.1.0 vs minimap2 2.30-r1287+

# GRCh38 Full-Genome Benchmark (8 Threads)

## System

| | |
|---|---|
| **CPU** | AMD Ryzen 9 7900X 12-Core (24 threads) |
| **RAM** | 128 GB DDR5 |
| **OS** | Ubuntu 22.04, Linux 6.8.0-94-generic |
| **SIMD** | SSE4.1, AVX2, AVX512BW |
| **Rust** | 1.90.0, `-C target-cpu=native` |
| **Profile** | `opt-level=3`, `lto="fat"`, `codegen-units=1` |

## Test Data

All tests use the full human GRCh38 reference (3.1 GB, 3.09 Gbp). Pre-built `.mmi`
indices per unique (k, w, HPC) group. Reads subsampled from full datasets.

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

30 tests (17 core presets, 5 parameter variations, 5 SIMD concordance, 3 scalar DP concordance).

### Phase 1: Core Preset Concordance (rammap vs minimap2)

| # | Test | Lines | Result | Notes |
|--:|------|------:|--------|-------|
| 1 | map-ont | 32,732 | **100% concordance** | Identical MD5 |
| 2 | map-ont-sam | 32,739 alns | **100% concordance** | |
| 3 | lr-hq | 31,579 | **100% concordance** | Identical MD5 |
| 4 | lr-hqae | 44,954 | **100% concordance** | Identical MD5 |
| 5 | map-iclr | 32,747 | **100% concordance** | Identical MD5 |
| 6 | map-hifi | 26,006 | **100% concordance** | Identical MD5 |
| 7 | map-pb | 26,482/26,483 | **PASS** | 1 line diff: known inversion UB (see [Known Differences](#known-differences-1)) |
| 8 | splice | 8,040 | **100% concordance** | Identical MD5 |
| 9 | splice-hq | 7,273 | **100% concordance** | Identical MD5 |
| 10 | cdna | 8,040 | **100% concordance** | Identical MD5 |
| 11 | sr | 39,965 | **100% concordance** | Identical MD5 |
| 12 | sr-sam | 39,970 alns | **100% concordance** | |
| 13 | splice-sr | 19,798/19,799 | **PASS** | 1 line diff, 0 after normalization |
| 14 | asm5 | 20 | **100% concordance** | Identical MD5 |
| 15 | asm10 | 20 | **100% concordance** | Identical MD5 |
| 16 | asm20 | 20 | **100% concordance** | Identical MD5 |
| 17 | ava-ont | 209,757 | **100% concordance** | Identical MD5 |

### Phase 2: Parameter Variations

| # | Test | Lines | Result | Notes |
|--:|------|------:|--------|-------|
| 18 | custom-scoring (-A1 -B2 -O2,12 -E2,1) | 32,286 | **100% concordance** | Identical MD5 |
| 19 | secondary-N5 | 32,732 | **100% concordance** | Identical MD5 |
| 20 | eqx (--eqx) | 32,732 | **100% concordance** | Identical MD5 |
| 21 | custom-kw (-k17 -w11) | 32,327 | **100% concordance** | Identical MD5 |
| 22 | split-30M (-I 30M) | 3,407,634/3,407,761 | **FAIL** | 79 raw diffs (0.002%), all inversions from `ksw_ll_i16` UB |

### Phase 3: SIMD Concordance (rammap SSE vs AVX2 vs AVX512 vs scalar-chain)

| # | Test | Variants | Result |
|--:|------|:--------:|--------|
| 23 | simd-ont | 3 | **0 diffs** across all variants |
| 24 | simd-hifi | 3 | **0 diffs** across all variants |
| 25 | simd-splice | 3 | **0 diffs** across all variants |
| 26 | simd-sr | 3 | **0 diffs** across all variants |
| 27 | simd-lr-hqae | 3 | **0 diffs** across all variants |

### Phase 4: Scalar DP Concordance (SIMD DP vs scalar fallback)

| # | Test | Lines | Result |
|--:|------|------:|--------|
| 28 | scalar-ont | 9,796 | **Exact match** |
| 29 | scalar-hifi | 2,607 | **Exact match** |
| 30 | scalar-splice | 1,554 | **Exact match** |

---

## Performance Comparison (8 Threads)

Wall time and peak RSS for rammap (RT) vs minimap2 (MM2). Both tools load from
pre-built `.mmi` indices (except custom-kw and split-30M which index from FASTA).

### Long-Read Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| map-ont | 199s | 231s | **0.86x** | 12.7 GB | 11.4 GB | 1.12x |
| map-ont-sam | 198s | 230s | **0.86x** | 12.8 GB | 11.5 GB | 1.11x |
| lr-hq | 44s | 51s | **0.86x** | 8.2 GB | 14.4 GB | **0.57x** |
| lr-hqae | 35s | 39s | **0.89x** | 4.9 GB | 6.8 GB | **0.72x** |
| map-iclr | 150s | 92s | 1.63x | 13.4 GB | 16.7 GB | **0.80x** |
| map-hifi | 29s | 31s | **0.94x** | 8.0 GB | 13.9 GB | **0.57x** |
| map-pb | 43s | 31s | 1.42x | 9.5 GB | 9.5 GB | 1.00x |

### Splice / RNA Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| splice | 23s | 12s | 1.84x | 19.1 GB | 15.6 GB | 1.23x |
| splice-hq | 23s | 12s | 1.83x | 19.0 GB | 15.6 GB | 1.22x |
| cdna | 23s | 12s | 1.81x | 18.9 GB | 15.6 GB | 1.21x |
| splice-sr | 21s | 11s | 1.95x | 18.6 GB | 13.6 GB | 1.36x |

### Short-Read Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| sr | 10.4s | 12.2s | **0.86x** | 10.6 GB | 10.9 GB | **0.97x** |
| sr-sam | 10.4s | 12.1s | **0.86x** | 10.6 GB | 10.9 GB | **0.97x** |

### Assembly Presets

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| asm5 | 7.6s | 13.5s | **0.56x** | 7.6 GB | 11.3 GB | **0.67x** |
| asm10 | 7.6s | 13.4s | **0.56x** | 7.6 GB | 11.3 GB | **0.67x** |
| asm20 | 12.8s | 14.5s | **0.88x** | 11.3 GB | 12.7 GB | **0.89x** |

### Overlap / Parameter Variations

| Preset | RT Wall | MM2 Wall | Wall Ratio | RT Mem | MM2 Mem | Mem Ratio |
|--------|--------:|---------:|-----------:|-------:|--------:|----------:|
| ava-ont | 28s | 11s | 2.54x | 1.4 GB | 1.5 GB | 0.94x |
| custom-scoring | 199s | 230s | **0.87x** | 12.7 GB | 11.2 GB | 1.13x |
| secondary-N5 | 197s | 231s | **0.85x** | 12.7 GB | 11.2 GB | 1.13x |
| eqx | 202s | 230s | **0.88x** | 12.7 GB | 11.3 GB | 1.12x |
| custom-kw | 108s | 89s | 1.22x | 12.7 GB | 18.0 GB | **0.71x** |
| split-30M | 713s | 839s | **0.85x** | 6.0 GB | 6.2 GB | 0.97x |

---

## GRCh38 Performance Summary

**Wall time ratio** (rammap / minimap2; lower is better):

```
Faster  ============================|============================  Slower
                                    |
        asm5         0.56x █████████|
        asm10        0.56x █████████|
        secondary-N5 0.85x      ████|
        split-30M    0.85x      ████|
        map-ont      0.86x      ████|
        sr           0.86x      ████|
        custom-scor  0.87x       ███|
        asm20        0.88x       ███|
        eqx          0.88x       ███|
        lr-hqae      0.89x        ██|
        map-hifi     0.94x         █|
        custom-kw    1.22x          |█████
        map-pb       1.42x          |██████████
        map-iclr     1.63x          |████████████████
        cdna         1.81x          |████████████████████
        splice-hq    1.83x          |████████████████████
        splice       1.84x          |█████████████████████
        splice-sr    1.95x          |██████████████████████
        ava-ont      2.54x          |███████████████████████████████
```

### Key Observations (GRCh38, 8 threads)

**Faster than minimap2**:
- asm5/asm10: **44% faster** wall, **33% less memory**
- map-ont/sr/secondary-N5: **14-15% faster** wall time
- lr-hq: **14% faster**, **43% less memory**
- map-hifi: **6% faster**, **43% less memory**
- split-30M: **15% faster** (multi-part index processing)

**Slower than minimap2**:
- splice/cdna/splice-sr/splice-hq: **1.8-2.0x slower** — index loading dominates
- ava-ont: **2.5x slower** — all-vs-all quadratic self-mapping overhead
- map-pb: **1.4x slower** — HPC index + PacBio chaining overhead
- map-iclr: **1.6x slower**

### Memory

rammap uses packed 4-bit reference storage (~375 MB for GRCh38) with on-demand
per-region nt4 extraction, vs minimap2's mmap-based packed storage. Memory usage
varies by preset:

- **Less memory**: lr-hq (0.57x), map-hifi (0.57x), asm5/10 (0.67x), lr-hqae (0.72x)
- **Parity**: sr (0.97x), map-pb (1.00x), split-30M (0.97x)
- **More memory**: splice presets (1.2-1.4x), map-ont (1.12x) — splice indices are large

---

## Known Differences

### GRCh38 Tests

| Test | Diffs | Explanation |
|------|------:|-------------|
| map-pb | 2 | Known `ksw_ll_i16` UB: inversions with `cm:i:0, s1:i:0`. minimap2 reads before buffer start, rammap correctly rejects. |
| splice-sr | 1 | 1 line count diff, 0 diffs after normalization. |
| split-30M | 79 | All 79 diffs are inversions (`tp:A:I` / `tp:A:i` with `cm:i:0, s1:i:0`). Same `ksw_ll_i16` UB root cause. 72 remain after MAPQ/de:f normalization. Out of 3.4M total lines (0.002%). |

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
| Index sort | Single | MSD radix sort, in-place |
| Bucket offset table | Single | O(n) sequential scan |
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

- **Index build**: minimap2 parallelizes per-bucket hash table construction across 16K
  buckets via `kt_for`. rammap sketches in parallel across sequences but sorts and
  builds the bucket offset table single-threaded. For a single reference sequence
  (chr20), neither tool benefits from parallel sketching.
- **I/O pipeline**: minimap2 uses a 3-stage pipeline (`kt_pipeline`) that overlaps
  reading, mapping, and output. rammap uses a dedicated read-ahead thread with a
  synchronous channel, achieving similar overlap.
- **Index format**: minimap2 uses mmap for `.mmi` files (zero-copy load). rammap
  deserializes `.rmmi` files via bincode (requires allocation + copy).
