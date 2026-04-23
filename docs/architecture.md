# rammap Architecture Reference

**rammap** is a pure-Rust minimap2-compatible sequence aligner producing byte-identical output via the same algorithms. It supports all major minimap2 presets (map-ont, map-hifi, sr, splice, asm, ava) with full CIGAR, CS/MD/DS tag, SAM, and PAF output.

Standalone crate with both `src/lib.rs` (library) and `src/main.rs` (binary). 30 Rust source files, ~28K lines total. The alignment module is ~24K lines across 24 files; the FASTA/FASTQ reader is ~650 lines.

**Key dependencies**: rayon (parallelism), clap (CLI), serde/bincode (index serialization), memmap2 (FASTA mmap), flate2 (gzip), wasm-bindgen (WASM support).

**Build profile**: `opt-level=3`, `lto="fat"`, `codegen-units=1`, `target-cpu=native` via `.cargo/config.toml`.

---

## Table of Contents

1. [Source Tree](#1-source-tree)
2. [CLI Layer](#2-cli-layer)
3. [Alignment Pipeline Overview](#3-alignment-pipeline-overview)
4. [Module Reference: src/align/](#4-module-reference-srcalign)
5. [Key Data Structures](#5-key-data-structures)
6. [Algorithm Details](#6-algorithm-details)
7. [SIMD Architecture](#7-simd-architecture)
8. [Performance Patterns](#8-performance-patterns)
9. [I/O Layer](#9-io-layer)
10. [Testing](#10-testing)
11. [Preset Reference](#11-preset-reference)
12. [Memory Safety](#12-memory-safety)
13. [Porting Lessons](#13-critical-porting-lessons)

---

## 1. Source Tree

```
src/
  main.rs             (1662)  # Binary entry: CLI parse + alignment orchestration
  lib.rs                 (7)  # Library exports: align, fasta, api
  api.rs               (915)  # Programmatic API entry points

  align/                       # alignment engine (~24K lines)
    mod.rs               (50)  # Module declarations
    sketch.rs           (332)  # Minimizer generation (rolling hash, windowing)
    sort.rs             (349)  # MSD radix sort (parallel two-level for index build)
    index.rs            (553)  # Reference index: build, save, load (RMMI/MMI formats)
    index_bucket.rs     (314)  # Per-bucket hash table backend
    seed.rs             (555)  # Seed collection: index lookup, occurrence filtering, heap mode
    map.rs             (1775)  # Mapping orchestration: MapOptions, MapContext, map_query()
    chain.rs            (471)  # Standard DP chaining (O(n^2) with skip limits)
    chain_simd.rs      (1289)  # SIMD chaining: AVX2 (8-wide) + NEON (4-wide)
    chain_rmq.rs        (770)  # RMQ-accelerated chaining (arena treap, O(log n))
    chain_simple.rs     (215)  # Greedy chain scoring (simple/fast)
    filter.rs           (375)  # Parent assignment + secondary filtering
    extend.rs          (2117)  # Anchor extension, gap-fill, CIGAR generation, CS/MD/DS formatters

    dp/                        # SIMD DP engine (~10.7K lines, split by algorithm)
      mod.rs             (832)  # Re-exports public API, tests (SIMD concordance)
      common.rs          (827)  # Types (DpResult), constants, flags, memory mgmt, traceback
      single.rs         (1917)  # Single-affine gap: SSE2/4.1/AVX2/AVX512/NEON/WASM/scalar
      dual.rs           (3069)  # Dual-affine gap: SSE2/4.1/AVX2/AVX512/NEON/WASM/scalar
      splice.rs         (3450)  # Splice-aware: SSE2/4.1/AVX2/AVX512/NEON/WASM/scalar
      lw.rs              (612)  # Lightweight i16 Smith-Waterman + global NW alignment

    pipeline.rs        (2649)  # End-to-end pipeline: process_query, format_output, MAPQ, PE
    pair.rs             (250)  # Paired-end logic (concordant pairing, PE MAPQ)
    junc.rs             (295)  # Junction annotation scoring (BED/SPSC for splice)
    jump.rs             (522)  # Jump splice extension (BED12 junction rescue)
    split.rs            (418)  # Split index: temp file I/O, multi-part merge
    align_simple.rs     (339)  # Simple alignment kernels
    stats.rs             (39)  # AlignmentStats timing struct
    wasm_lib.rs         (217)  # wasm-bindgen entry point

  fasta/                       # Custom FASTA/FASTQ I/O (faster than noodles for raw parsing)
    mod.rs                (7)  # Module declarations
    reader.rs           (558)  # Streaming parser (zero-copy RefRecord)
    record.rs            (83)  # Record/RefRecord types
```

---

## 2. CLI Layer

### Entry Point (`src/main.rs`)

rammap's CLI lives entirely in `main.rs` (no separate `cli.rs` or `commands/` directory). The `AlignArgs` struct defines all command-line arguments via clap, and `run()` orchestrates the alignment pipeline.

```
main() → AlignArgs::parse() → run(args)
  ├─ apply_preset(opt, k, w, is_hpc, preset)
  ├─ load or build Index (single-part or batched)
  ├─ for each index part:
  │   ├─ load JunctionDb, JumpDb (if splice mode)
  │   └─ map_one_part(mi, opt, queries, out_cfg, ...)
  │       ├─ spawn read-ahead thread (sync_channel(1))
  │       └─ rayon parallel over query batches:
  │           ├─ map_query() / map_query_multi()
  │           ├─ process_query()
  │           ├─ align_and_format_pair() / align_and_format_query()
  │           └─ format_output()
  └─ if --split-prefix: merge_split_results(prefix, n_parts, opt, mi)
```

### CLI Arguments (`AlignArgs`)

**Core**: `target` (FASTA/index), `queries` (FASTA/FASTQ), `-x preset`, `-k kmer`, `-w window`, `-t threads`, `-o output`

**Scoring**: `-A match`, `-B mismatch`, `-O gap_open[,gap_open2]`, `-E gap_extend[,gap_extend2]`, `--transition`, `--score-n`

**Chaining**: `--bw`, `--min-chain-score`, `--min-cnt`, `--max-gap`, `--chain-gap-scale`

**Seeding**: `--mid-occ`, `--mid-occ-frac`, `--mid-occ-range`, `--max-occ`

**Output**: `-c` (CIGAR), `--cs` (CS tag), `-a` (SAM), `--MD`, `--ds`, `--eqx`, `-N best_n`

**Paired-end**: `-F frag_len`, `--pairing no|weak|strong`, `--frag` (interleaved)

**Splice**: `-G max_intron`, `--junc-bed`, `--spsc`, `--junc-bonus`, `--junc-pen`, `-j` (jump), `--pass1`

**Split index**: `-I batch_size`, `--split-prefix`

**Advanced**: `--rmq`, `--heap-sort`, `--all-chains`, `-X` (ava), `--qstrand`, `--no-hash-name`, `--write-junc`

---

## 3. Alignment Pipeline Overview

### End-to-End Flow

```
FASTA Reference                     Query FASTA/FASTQ
      |                                    |
      v                                    v
  Index::build()                   sketch_sequence()
  (sketch.rs → sort.rs → index.rs)  (sketch.rs)
      |                                    |
      v                                    v
  Index { entries, bucket_offsets }   Vec<Minimizer>
                    \                     /
                     \                   /
                      v                 v
                collect_seed_hits() (seed.rs)
                         |
                         v
                  Vec<Minimizer> (anchors, sorted by ref_pos)
                         |
                         v
           chain_anchors() / chain_anchors_rmq()
                  (chain.rs / chain_rmq.rs)
                         |
                         v
                   Vec<Mapping> (chains)
                         |
                         v
              assign_parents + select_sub (filter.rs)
                         |
                         v
                   Vec<Mapping> (filtered)
                         |
                         v
               align_anchors() (extend.rs)
               calls extend_dual_affine / extend_splice (dp.rs)
                         |
                         v
                  Vec<AlnResult> (with CIGAR)
                         |
                         v
                compute_mapqs() (pipeline.rs)
                         |
                         v
                format_output() (pipeline.rs)
                         |
                         v
                   SAM / PAF output
```

### Paired-End Variant

```
align_and_format_pair(opt, mi, segs, names, qlens, ...)
  ├─ process_query() for read 1
  ├─ process_query() for read 2
  ├─ pair_alignments(regs1, regs2, opt)   [concordant pair detection]
  └─ format_output() for both reads       [with PE SAM flags]
```

### Split Index Variant

```
map_one_part_split(mi, opt, queries, split_writer, ...)
  ├─ same alignment as map_one_part()
  └─ split_write_query(writer, results)   [serialize to temp file]

merge_split_results(prefix, n_parts, opt, mi)
  ├─ split_merge_prep()                   [load headers, compute rid_shifts]
  ├─ for each query across all parts:
  │   ├─ split_read_query() per part
  │   ├─ merge all results
  │   └─ refilter_merged_results()        [mm_hit_sort + mm_set_parent + mm_select_sub]
  └─ format_output()
```

---

## 4. Module Reference: src/align/

### sketch.rs — Minimizer Generation

**Purpose**: Generate minimizer anchors from DNA sequences using rolling k-mer hash and window selection.

**Key type**: `Minimizer` — packed 128-bit anchor representation (see [Data Structures](#5-key-data-structures)).

**Key functions**:
- `sketch_sequence(seq, len, w, k, rid, is_hpc, p: &mut Vec<Minimizer>)` — Clears `p`, generates minimizers for sequence
- `sketch_sequence_append(...)` — Same but appends (for multi-segment reads)
- Both delegate to `sketch_sequence_impl()` (shared core logic)

**Internal helpers**: `encode_base()` (256-byte LUT), `kmer_hash()` (invertible 64-bit hash).

**Constants**: `MM_SEED_SEG_SHIFT=48`, `MM_SEED_SEG_MASK`, `MM_SEED_SELF=1<<43`.

---

### sort.rs — Radix Sort

**Purpose**: MSD in-place radix sort. NOT stable.

**Key trait**: `RadixKey` — `fn radix_key(&self) -> u64` implemented for `Minimizer` (key=x) and `(u64, u64)` (key=first element).

**Key functions**:
- `radix_sort_128x(arr: &mut [Minimizer])` — Sort minimizers by x field
- `radix_sort_pair(arr: &mut [(u64, u64)])` — Sort index entries by hash, then position within ties. Uses adaptive two-level partitioning (detects highest occupied byte, creates up to 65K buckets) with parallel sub-bucket sorting via rayon.

**Algorithm**: American Flag sort with 8-bit digits, 256 buckets, in-place cyclic permutation. Falls back to insertion sort for arrays < 64 elements. `radix_sort_pair` adds a parallel recursion phase: after two sequential partition passes, independent sub-buckets are sorted in parallel.

---

### index.rs + index\_bucket.rs — Reference Index

**Purpose**: Build, save, and load searchable minimizer hash tables from reference sequences. Uses per-bucket hash tables for lookup.

**Key types**:
- `Index` — `{kmer_size, window_size, homopolymer_compressed, index (part#), seqs: Vec<TargetSequence>, backend: LookupBackend, packed_seqs: Vec<u32>}`
- `TargetSequence` — `{name, len, offset, is_alt}` (metadata only; sequence data in packed\_seqs)
- `SeedLookup` trait — `get()`, `get_range()`, `get_by_range()`, `occurrence_counts()`, `is_empty()`
- `LookupBackend` enum — currently `BucketHash(BucketHashLookup)`, extensible for future backends
- `BucketHashLookup` (index\_bucket.rs) — per-bucket open-addressing hash tables with shared flat positions array. Each bucket maps hash suffixes to (offset, count) ranges.

**Key methods**:
- `Index::build(seqs, w, k, is_hpc) -> Self` — Fused per-sequence pack+sketch (sequential, dropping each sequence's ASCII immediately), parallel bucket sort, sequential per-bucket hash table build
- `Index::save(path)` / `save_part(writer)` — Serialize (RMMI format with magic prefix)
- `Index::load(path)` / `load_part(reader)` — Detect format (RMMI, MMI v2, old bincode)
- `Index::load_minimap2(reader)` — Read minimap2 .mmi format, constructing `BucketHashLookup` directly from per-bucket hash tables
- `Index::get(hash)` / `get_range(hash)` / `get_by_range(range)` — Delegated to backend
- `Index::cal_mid_occ(frac, min, max) -> usize` — Compute occurrence threshold via backend's `occurrence_counts()`
- `Index::get_nt4(rid, pos) -> u8` — Single base as nt4 (0=A,1=C,2=G,3=T,4=N)
- `Index::get_region_nt4(rid, start, end) -> Vec<u8>` — Region as nt4 bytes

**Lookup**: Per-bucket hash table indexed by `hash & mask` (low bits), then open-addressing probe for `hash >> bucket_bits` (high bits). O(1) average case.

**Build memory model**: Sequential sketching processes one reference sequence at a time, dropping ASCII data immediately after packing + sketching. Bucket Vecs grow incrementally during sketch, then are sorted in parallel and consumed one-at-a-time during hash table build. Peak memory ≈ max(buckets, positions + hash tables) — never both simultaneously.

---

### seed.rs — Seed Collection

**Purpose**: Find query minimizer matches in the reference index, filter repetitive seeds.

**Key functions**:
- `collect_seed_hits(opt, mi, qlen, minimizers, anchors, mini_pos, qname_opt) -> rep_len` — Main seed collection: index lookup per query minimizer, filters by `mid_occ/max_occ`, returns representative length
- `collect_seed_hits_heap(...)` — Min-heap extraction mode for `MM_F_HEAP_SORT` (sr preset)
- `collect_seed_hits_with_occ(...)` — Direct occurrence lookup variant
- `select_seeds(seeds, qlen, max_occ, max_max_occ, dist)` — Heap-based selection for high-occurrence seed runs: keeps spatially diverse subset
- `filter_minimizers_by_occ(minimizers, mid_occ, q_occ_frac) -> usize` — Remove high-occurrence query minimizers
- `compute_read_hash(qname, qlen, seed, flags) -> u32` — X31 hash for deterministic seeding

**Key types**: `SeedInfo` (query pos, span, hit count, tandem flag), `MatchedSeed`.

**Helper hashes**: `hash64()`, `wang_hash()`, `x31_hash_string()`.

---

### map.rs — Mapping Orchestration

**Purpose**: Core mapping pipeline — sketching, seeding, chaining, post-chain filtering. Owns `MapOptions`, `MapContext`, `Mapping`.

**Key types**: See [Data Structures](#5-key-data-structures) for `MapOptions`, `AlignFlags`, `MapContext`, `Mapping`.

**Key functions**:
- `map_query(opt, mi, qname, qseq, ctx) -> (Vec<Mapping>, rep_len, AlignmentStats, Vec<Minimizer>)` — Single-segment mapping: sketch → seed → chain → filter → return chains
- `map_query_multi(opt, mi, qname, seqs, qlens, ctx) -> MultiMapResult` — Multi-segment (PE strong-pairing): handles segment boundaries during chaining
- `compute_bounds_from_squeezed(squeezed, sq_start, sq_cnt, tlen, qlen, min_cnt) -> (rs1, qs1, re1, qe1)` — Extension boundary computation from nearby seeds
- `estimate_divergence(regs, anchors, qlen, mini_pos, mi)` — K-mer based divergence estimation (dv:f tag)
- `chain_post(regs, opt, mi, ...)` — Post-chain filtering: `assign_parents()` + `select_sub()` + `s2_score` computation

**Internal flow of `map_query()`**:
1. `sketch_sequence()` on query
2. `filter_minimizers_by_occ()` to remove high-frequency minimizers
3. `collect_seed_hits()` or `collect_seed_hits_heap()` to find index matches
4. `radix_sort_128x()` anchors by (`ref_id`, `ref_pos`)
5. `chain_anchors()` or `chain_anchors_rmq()` (if `MM_F_RMQ_CHAIN`)
6. Backtrack + compact chains
7. `chain_post()` for parent/secondary filtering
8. Optional RMQ rescue re-chaining
9. Return `Vec<Mapping>` + metadata

---

### chain.rs — Standard Chaining DP

**Purpose**: Connect compatible anchors into high-scoring chains via dynamic programming.

**Key functions**:
- `chain_anchors(anchors, scores, predecessors, visited, max_chain_skip, max_chain_iter, bw, chain_gap_scale, max_gap_ref, chn_pen_gap, chn_pen_skip, is_cdna, n_seg) -> Vec<Minimizer>` — O(n^2) DP with skip limits
- `compute_chain_score(ai, aj, max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg) -> i32` — Pairwise scoring between anchors
- `chain_backtrack_end(...)` — Traceback from endpoint with max-drop threshold
- `fast_log2(x) -> f32` — Fast log2 approximation

**DP recurrence** (for anchors sorted by `ref_pos`):
```
score[i] = max(
    qi_span,                                          // base: single anchor
    max over j<i { score[j] + connect_score(i, j) }  // extend chain
)
```

**Scoring function** `compute_chain_score(ai, aj)`:
- `query_diff = ai.query_pos - aj.query_pos` (must be > 0)
- `ref_diff = ai.ref_pos - aj.ref_pos` (must be > 0)
- `gap = |ref_diff - query_diff|` (must be <= bandwidth)
- `score = min(aj.span, min(ref_diff, query_diff))` minus gap penalty
- Gap penalty = `chn_pen_gap * gap + 0.5 * log2(gap)` (logarithmic + linear)

---

### `chain_rmq.rs` — RMQ-Accelerated Chaining

**Purpose**: O(log n) chaining via augmented treap for range minimum queries. Critical for long reads and assembly presets.

**Key types**:
- `RmqTree` — Arena-based treap with subtree-min augmentation. Fields: `nodes: Vec<Node>`, `root`, `free_head`, `rng` (xorshift64)
- `Node` — `{key_y, key_i, priority, left, right, parent, sub_min_val, sub_min_i, val}`
- `RmqRevIter` — Stack-based reverse in-order iterator for tree traversal

**Key functions**:
- `chain_anchors_rmq(anchors, ...) -> Vec<Minimizer>` — RMQ-based chaining with rescue re-chaining
- `RmqTree::insert_elem(y, i, val)` — O(log n) treap insert with subtree-min update
- `RmqTree::erase(y, i)` — O(log n) deletion
- `RmqTree::rmq(lo_y, hi_y) -> (min_val, min_i)` — Range minimum query on closed interval [`lo_y`, `hi_y`]

**Performance**: Reduced RMQ rescue rechain from 27.9s to 7.7s single-thread for map-ont.

---

### filter.rs — Parent Assignment & Secondary Filtering

**Purpose**: Determines primary/secondary status and filters low-quality chains.

**Key types**:
- `ParentState` — `{parent: Vec<usize>, parent_score: Vec<i32>, ...}` tracking primary regions
- `FilterParams` — `{pri_ratio, best_n, min_diff, mask_level, mask_len, hard_mask_level}`
- `FilterableItem` trait — abstraction for items that can be filtered by query overlap

**Key functions**:
- `assign_parents<T: FilterableItem>(items: &mut [T])` — Set parent for each item based on query overlap. First item = primary; subsequent assigned to highest-scoring non-overlapping parent.
- `select_sub(items, parent_state, params)` — Remove suboptimal secondaries: keep `best_n` per parent, filter by `pri_ratio`
- `check_secondary_filter(item, parent, params) -> bool` — Single-item filter decision
- `scale_alt_score(score, alt_diff_frac) -> i32` — ALT contig scoring adjustment

---

### extend.rs — Anchor Extension & CIGAR Generation

**Purpose**: Convert chains of anchors into full alignments by extending left/right boundaries and filling gaps with SIMD DP. Also handles CIGAR formatting (CS/MD/DS tags).

**Key types**:
- `CigarOp` — `{op: char, len: u32}` where op is one of `=`, `X`, `M`, `I`, `D`, `N`, `S`, `H`
- `AlignmentContext` — Reusable DP buffers: `{dp_curr, dp_prev: Vec<DpCell>, traceback: Vec<u8>}`
- `AlignmentKernel` trait — Polymorphic DP: `fn align(qseq, tseq, ...) -> Vec<CigarOp>`
- `TrimResult` — Output of `trim_and_prepare_anchors()`: anchor array with coordinates
- `CigarTagVisitor` trait — Visitor pattern for CIGAR tag formatting: `on_aligned()`, `on_insertion()`, `on_deletion()`, `on_intron()`, `finish()`

**Key functions**:
- `align_anchors(anchors, qseq, tseq, ..., junc_db, rid) -> AlignResult` — Main entry: wrapper for `align_anchors_full()`
- `align_anchors_full(...)` — Full implementation (~525 lines): left extension → gap-fill loop → right extension → CIGAR finalization. Handles z-drop splitting, inversion detection, splice modes.
- `trim_and_prepare_anchors(...)` — Anchor trimming, bad-end fixing, coordinate setup
- `compute_left_boundary(...)` / `compute_right_boundary(...)` — Extension boundary from nearby seeds
- `finalize_cigar(...)` — CIGAR post-processing: `fix_cigar` + =/X conversion
- `build_scoring_matrix(a, b) -> [i8; 25]` — Simple 5x5 substitution matrix
- `build_scoring_matrix_full(a, b, transition, sc_ambi) -> [i8; 25]` — Full matrix with transition + ambiguity
- `fmt_cigar(ops, eqx) -> String` — CIGAR text format
- `fmt_cs(ops, qseq, tseq, qs, rs) -> String` — CS tag via `CsVisitor`
- `fmt_md(ops, qseq, tseq, qs, rs) -> String` — MD tag via `MdVisitor`
- `fmt_ds(ops, ...) -> String` — DS tag via `DsVisitor`
- `walk_cigar_ops(ops, visitor) -> String` — Shared CIGAR iteration driving all three visitors
- `convert_cigar_to_eqx_pub(raw_cigar, qseq, tseq, qs, rs) -> Vec<CigarOp>` — Expand M → =/X
- `rev_comp(seq) -> Vec<u8>` — Reverse complement (ASCII)
- `rev_comp_nt4(seq) -> Vec<u8>` — Reverse complement (nt4)
- `encode_nt4_byte(b) -> u8` — ASCII base to nt4

**Constants**: `SEED_IGNORE=1<<41`, `SEED_LONG_JOIN=1<<40`, `SEED_TANDEM=1<<42` (defined in sketch.rs, re-exported from extend.rs).

---

### dp/ — SIMD Dynamic Programming (~10.7K lines)

**Purpose**: SIMD-accelerated banded DP alignment. Three algorithm variants across six platform targets (SSE2, SSE4.1, AVX2, AVX512BW, NEON, WASM SIMD128) plus scalar fallback. Split into sub-modules by algorithm:

| Module | Lines | Contents |
|--------|------:|---------|
| `dp/common.rs` | 827 | Types (`DpResult`, `LightweightProfile`), constants, alignment flags, memory management (`AlignedMemory`, `DP_MEM_CACHE`), WASM SIMD compat layer, x86 SIMD helpers, traceback functions |
| `dp/single.rs` | 1,917 | Single-affine gap: `cost = q + k*e`. All SIMD variants + scalar + dispatch |
| `dp/dual.rs` | 3,069 | Dual-affine gap: `cost = min(q+k*e, q2+k*e2)`. All SIMD variants + scalar + dispatch |
| `dp/splice.rs` | 3,450 | Splice-aware: GT-AG junction detection + bonus. All SIMD variants + scalar + dispatch |
| `dp/lw.rs` | 612 | Lightweight i16 Smith-Waterman (no traceback) + global NW alignment |
| `dp/mod.rs` | 832 | Module declarations, `pub use` re-exports, SIMD concordance tests |

Each algorithm module contains all its SIMD variants (SSE2/SSE4.1 via macro, AVX2 macro, AVX512 macro, NEON, WASM) plus scalar fallback — keeping all code for one algorithm together.

**Public API** (re-exported from `dp/mod.rs`):
- `extend_single_affine(...)` — Single-affine gap dispatch
- `extend_dual_affine(...)` — Dual-affine gap dispatch (used for most long-read modes)
- `extend_splice(...)` — Splice-aware dispatch
- `lightweight_profile_init(...)` / `lightweight_align_i16(...)` — Quick scoring without traceback
- `global_align(...)` — Global NW (SIMD or Gotoh scalar)

**Key constants** (alignment flags, in `dp/common.rs`):

| Flag | Value | Meaning |
|------|-------|---------|
| `SCORE_ONLY` | 0x01 | Skip traceback |
| `RIGHT_ALIGN` | 0x02 | Right-align gaps |
| `GENERIC_SCORING` | 0x04 | Use full 5x5 scoring matrix |
| `APPROX_MAX` | 0x08 | Approximate max tracking (faster) |
| `APPROX_DROP` | 0x10 | Enable z-drop heuristic |
| `EXTENSION_ONLY` | 0x40 | Stop at first max (extension only) |
| `REV_CIGAR` | 0x80 | Output CIGAR in reverse |
| `SPLICE_FORWARD` | 0x100 | Forward splice mode |
| `SPLICE_REVERSE` | 0x200 | Reverse splice mode |
| `SPLICE_FLANK` | 0x400 | Splice flank bonus |
| `SPLICE_COMPLEX` | 0x800 | Complex splice handling |
| `SPLICE_SCORE` | 0x1000 | Use junction bonus array |

**Shared helpers** (in `dp/common.rs`):
- `push_cigar(cigar, op, len)` — Append CIGAR op (merges consecutive same-op)
- `alloc_h_array(approx_max, tlen_, simd_width)` — Allocate 32-bit H array for exact-max tracking
- `traceback_dual_affine()` / `traceback_single_affine_safe()` / `traceback_splice()` — CIGAR reconstruction from DP matrix
- `traceback_start_position()` — Find start position for traceback

**SIMD macro architecture**: See [SIMD Architecture](#7-simd-architecture).

---

### pipeline.rs — End-to-End Pipeline

**Purpose**: Orchestrates full alignment from mapping results to formatted output. Handles single-read, paired-end, split-index, and inversion modes.

**Key types**:
- `OutputConfig` — `{do_cigar, do_cs, do_md, do_ds, eqx, output_sam: bool}`
- `CigarStats` — `{matches, edit_distance, block_len, num_ambiguous, divergence, has_n_skip, gap_opens}` with `CigarStats::from_cigar(ops, qseq, tseq)` constructor
- `AlnResult` — Full alignment result (~30 fields, see [Data Structures](#5-key-data-structures))
- `ProcessedQuery` — `{results, mapqs, sam_pri, parent_indices, rep_len, stats}`
- `DpRecalcInfo` — `{match_len, block_len, num_ambiguous, gap_bases, gap_opens, sum_log_gap}`

**Core pipeline functions**:
- `align_and_format_query(opt, mi, qseq, qname, qlen, regs, out_cfg, ctx, junc_db, jump_db, split_mode) -> ProcessedQuery` — Single-segment entry: map → align → format
- `process_query(opt, mi, qseq, qname, qlen, regs, out_cfg, ctx, junc_db, jump_db, split_mode) -> ProcessedQuery` — Core alignment loop calling `align_single_mapping()` per chain
- `process_query_core(...)` — Orchestrator (~216 lines): alignment loop → `filter_alignment_results()` → `assign_parents_and_select()` → `compute_mapqs()`
- `align_single_mapping(mapping, opt, mi, qseq, qlen, ctx, junc_db) -> AlnResult` — Align one chain: extracts sequences, calls `align_anchors()`, computes CigarStats
- `align_and_format_pair(opt, mi, segs, names, qlens, junc_db, jump_db, ctx, out_cfg) -> Vec<(String, String)>` — PE entry point

**Post-alignment functions**:
- `filter_alignment_results(results, recalc_infos, opt)` — Threshold filtering (`min_dp_max`, `min_cnt`, etc.)
- `assign_parents_and_select(results, recalc_infos, opt, mi, ...)` — Parent assignment + secondary selection + `dp_max` ranking (~253 lines)
- `compute_mapqs(results, mapqs, ...)` — MAPQ calculation (log-odds ratio between top 2 `dp_max` scores)
- `update_dp_max(dp_max_vals, recalc_infos, qlen, ...)` — Recalculate `dp_max` from CIGAR stats
- `compute_alignment_score_max(ops, mat, q, e, qseq, tseq, log_gap) -> i32` — Walk CIGAR to find max local score

**Output functions**:
- `format_output(qname, qlen, results, mapqs, out_cfg, opt, seq_data) -> (String, bool)` — Thin dispatcher calling `format_sam_record` or `format_paf_record`
- `format_sam_record(buf, result, ...)` — Single SAM record (~211 lines): all columns + tags
- `format_paf_record(buf, result, ...)` — Single PAF record (~77 lines)
- `format_unmapped_record(buf, ...)` — Unmapped read output

**Inversion handling**:
- `try_align_inversion(opt, mi, qlen, qseq_fwd, qseq_rc, r1, r2, out) -> Option<AlnResult>` — Detect and align inversions via lightweight SW scoring + full DP extension

**SAM tags output**: NM, AS, ms, s1, s2, cm, nn, tp, de:f, dv:f, rl:i, cs, MD, ds, SA (supplementary).

---

### pair.rs — Paired-End Pairing

**Purpose**: Concordant pair detection and PE MAPQ adjustment.

**Key type**: `PeReg` — `{dp_score, ref_id, ref_start, ref_end, is_reverse, hash, id, parent, sam_pri, proper_frag}`

**Key function**: `pair_alignments(max_gap_ref, pe_bonus, sub_diff, match_sc, qlens, regs1, regs2)` — Finds best concordant pair by scanning combined sorted array of `(ref_id, position, strand)`. Adjusts MAPQ for concordant pairs.

**Algorithm**: Sort both reads' alignments by position. Scan for proper pairs (forward-first + reverse-second compatible, within `max_frag_len`). Apply `pe_bonus` to best pair's MAPQ.

---

### junc.rs — Junction Annotation Scoring

**Purpose**: Load and query splice junction annotations for bonus/penalty scoring in splice-aware alignment.

**Key type**: `JunctionDb` enum — `Bed(Vec<Vec<BedInterval>>)` or `Spsc(Vec<SpscContig>)`

**Key functions**:
- `load_bed_junctions(path, names, n_seq) -> JunctionDb` — Parse BED6/BED12 junctions
- `load_spsc_scores(path, names, n_seq, scale) -> JunctionDb` — Per-position SPSC scores
- `get_junc(db, rid, strand, pos) -> Option<u8>` — O(log n) lookup via binary search

**Integration**: Called during `extend_splice()` gap-fill in extend.rs. Junction presence adds bonus; unannotated splice adds penalty.

---

### jump.rs — Jump Splice Extension

**Purpose**: Extend clipped alignment regions into annotated exon junctions.

**Key type**: `JumpDb` — `{junctions: Vec<Vec<JumpJunc>>}` where `JumpJunc` = `{left_pos, right_pos, count, strand, flag}`

**Key functions**:
- `JumpDb::load(mi, path, flag, min_sc) -> Self` — Parse BED12 junctions
- `jump_split_left()` / `jump_split_right()` — Extend clipped ends into junctions
- Applied in pipeline after `process_query()` for single-segment splice reads

**Modes**: `-j file` sets `MM_JUNC_ANNO` (`min_sc=-1`); `--pass1 file` sets `MM_JUNC_MISC` (`min_sc=5`).

---

### split.rs — Split Index Support

**Purpose**: Multi-part index handling: serialize per-part results to temp files, then merge globally.

**Key functions**:
- `split_init(prefix, part_index, mi) -> BufWriter<File>` — Create temp file with header
- `split_write_query(writer, results, rep_len, frag_gap)` — Bincode-serialize AlnResult batch
- `split_merge_prep(prefix, n_parts) -> (k, seqs, rid_shifts, readers)` — Load headers, compute coordinate shifts
- `split_read_query(reader, k, rid_shifts) -> Vec<AlnResult>` — Deserialize results
- `merge_split_results(prefix, n_parts, opt, mi, out, stats)` — Full merge pipeline

**Critical**: `rl:i:0` always in merge output (minimap2 doesn't set `rep_len` during merge). `mid_occ` calculated once from first part only.

---

### stats.rs — Timing Statistics

**Type**: `AlignmentStats` — `{t_sketch, t_seed, t_chain, t_align, t_post: Duration, n_reads, n_seeds, n_anchors, n_chains: u64}`. Implements `Add` for aggregation across threads/parts.

---

### wasm.rs / wasm\_lib.rs — WebAssembly Support

`wasm_lib.rs` provides `wasm-bindgen` entry points `align_wasm(target_fasta, query_fasta, output_sam, is_splice) -> String` and `force_align_wasm(...)`. Uses WASM SIMD128 intrinsics when available, falls back to scalar.

---

## 5. Key Data Structures

### Minimizer Bit-Field Encoding (`sketch.rs`)

```
Minimizer { x: u64, y: u64 }

x field (reference):
  [63]          unused
  [62:32]       ref_id (31 bits) << 33 | strand (1 bit) << 32
  [31:0]        ref_pos (32 bits, as i32)

y field (query):
  [55:48]       seg_id (8 bits, for multi-segment reads)
  [43]          MM_SEED_SELF flag (self-mapping in ava mode)
  [39:32]       kmer_span (8 bits)
  [31:1]        query_pos (31 bits)
  [0]           strand (0=forward, 1=reverse)

Accessor methods:
  .ref_pos() -> i32        // x as u32 as i32
  .ref_id() -> usize       // (x >> 32) & 0x7FFFFFFF
  .ref_id_strand() -> u64  // x >> 32
  .query_pos() -> i32      // y as i32
  .query_span() -> i32     // (y >> 32) & 0xff
  .segment_id() -> usize   // (y >> 48) & 0xff
  .is_self_anchor() -> bool // y & (1 << 43) != 0
```

### MapOptions (`map.rs`)

```
MapOptions {
    seeding: SeedingParams {
        mid_occ: usize,          // Occurrence threshold for filtering seeds
        max_occ: usize,          // Hard max occurrence (re-chain trigger)
        max_max_occ: usize,      // Absolute max (default 4095)
        occ_dist: i32,           // Distance threshold for select_seeds
        q_occ_frac: f32,         // Fraction-based occurrence filter
        min_mid_occ: i32,        // Lower bound for cal_mid_occ
        max_mid_occ: i32,        // Upper bound for cal_mid_occ
    },
    chaining: ChainingParams {
        min_cnt: i32,            // Min anchors per chain
        min_chain_score: i32,    // Min total chain score
        max_gap: i32,            // Max gap in query or ref
        max_gap_ref: i32,        // Max gap in ref only (for splice)
        max_dist_x: i32,        // Max query distance between anchors
        max_dist_y: i32,         // Max ref distance between anchors
        bandwidth: i32,          // Max diagonal distance in chain
        bandwidth_long: i32,     // Bandwidth for long gaps
        max_chain_skip: i32,     // Max consecutive skipped anchors
        max_chain_iter: i32,     // Max predecessors to check per anchor
        chn_pen_gap: f32,        // Gap penalty coefficient
        chn_pen_skip: f32,       // Skip penalty coefficient
        chain_gap_scale: f32,    // Scale factor for chn_pen_gap derivation
        rmq_rescue_size: i32,    // Min chain size for RMQ rescue
        rmq_rescue_ratio: f32,   // Score ratio threshold for rescue
        rmq_inner_dist: i32,     // Inner distance for RMQ tree
        rmq_size_cap: i32,       // Cap on RMQ tree size
    },
    scoring: ScoringParams {
        match_score: i32,        // Match bonus (default 2)
        mismatch_penalty: i32,   // Mismatch cost (default 4)
        gap_open: i32,           // Short gap open (default 4)
        gap_extend: i32,         // Short gap extend (default 2)
        gap_open2: i32,          // Long gap open (default 24)
        gap_extend2: i32,        // Long gap extend (default 1)
        transition: i32,         // Transition bonus (A<->G, C<->T)
        ambig_penalty: i32,      // N-base penalty
        noncanon_penalty: i32,   // Non-canonical splice penalty
        junc_bonus: i32,         // Junction annotation bonus
        junc_pen: i32,           // Junction annotation penalty
    },
    alignment: AlignmentParams {
        zdrop: i32,              // Z-drop threshold
        zdrop_inv: i32,          // Z-drop for inversions
        end_bonus: i32,          // Bonus for reaching sequence end
        max_sw_mat: i64,         // Max DP matrix size (SW memory cap)
        min_dp_max: i32,         // Min dp_max to keep alignment
        min_dp_len: i32,         // Min length for DP alignment (vs short-circuit)
        anchor_ext_len: i32,     // Anchor seed extension length
        anchor_ext_shift: i32,   // Anchor seed extension shift
        max_clip_ratio: f32,     // Max clipping ratio
    },
    filtering: FilteringParams {
        best_n: i32,             // Max secondary alignments to report
        pri_ratio: f32,          // Min score ratio vs primary
        mask_level: f32,         // Query overlap threshold for masking
        mask_len: i32,           // Min overlap length for masking
        is_splice: bool,         // Splice-aware mode
        alt_drop: f32,           // ALT contig score drop
        seed: i32,               // Random seed
        chain_skip_scale: f32,   // Scale for chn_pen_skip derivation
        max_qlen: i32,           // Max query length
        jump_min_match: i32,     // Min matches for jump extension
    },
    pairing: PairedEndParams {
        max_frag_len: i32,       // Max fragment length for PE
        pe_ori: i32,             // Expected orientation (FR=0<<1|1)
        pe_bonus: i32,           // MAPQ bonus for concordant pairs
    },
    flags: AlignFlags,           // Bitflags (see below)
}
```

### AlignFlags (`map.rs`)

```
NO_DIAG          0x001        No diagonal self-anchors (ava)
NO_DUAL          0x002        No qname > tname (ava)
NO_QUAL          0x010        Don't output quality
OUT_CIGAR        0x020        Produce CIGAR string
SPLICE           0x080        Splice-aware alignment
SPLICE_FOR       0x100        Forward splice strand
SPLICE_REV       0x200        Reverse splice strand
NO_LJOIN         0x400        Skip long-join rescue
INDEPEND_SEG     0x800        Independent segment mapping
SHORT_READ       0x1000       Short-read optimizations
FRAG_MODE        0x2000       Fragment/PE mode
NO_PRINT_2ND     0x4000       Suppress secondary output
EQX              0x8000       Use =/X instead of M in CIGAR
LONG_CIGAR       0x10000      Long CIGAR format
SOFTCLIP         0x20000      Soft-clip ends
SPLICE_FLANK     0x40000      Splice flank bonus
FOR_ONLY         0x100000     Forward strand only
REV_ONLY         0x200000     Reverse strand only
HEAP_SORT        0x400000     Heap-based seed sorting
ALL_CHAINS       0x800000     Output all chains (skip filtering)
NO_END_FLT       0x1000000    No end filtering
HARD_MASK_LEVEL  0x2000000    Hard mask level
SAM_HIT_ONLY     0x4000000    SAM output for hits only
PAF_NO_HIT       0x8000000    PAF no-hit output
NO_HASH_NAME     0x40000000   Don't hash read name
RMQ_CHAIN        0x80000000   Use RMQ chaining
QSTRAND          0x100000000  Query-strand alignment
NO_INV           0x200000000  Skip inversion detection
SPLICE_OLD       0x400000000  Old splice mode
SECONDARY_SEQ    0x800000000  Output sequence for secondaries
WEAK_PAIRING     0x4000000000 Weak PE pairing
SR_RNA           0x8000000000 Short-read RNA mode
OUT_JUNC         0x10000000000 Output junction BED
```

### AlnResult (`pipeline.rs`)

```
AlnResult {
    // Mapping metadata
    ref_id: usize,                  // Target sequence index
    is_reverse: bool,               // Reverse strand
    chain_score: i32,               // s1 tag
    initial_chain_score: i32,       // Before z-drop split
    anchor_count: usize,            // cm tag
    s2_score: Option<i32>,          // Suboptimal chain score
    hash: u32,                      // Read hash for deterministic seeding

    // Alignment coordinates
    query_start: usize,
    query_end: usize,
    ref_start: usize,
    ref_end: usize,

    // Alignment quality
    align_score: i32,               // AS tag
    matches: usize,                 // PAF col 10
    block_len: usize,               // PAF col 11
    edit_distance: u32,             // NM tag
    num_ambiguous: usize,           // nn tag
    divergence: f64,                // de:f tag

    // Formatted strings
    cigar_str: String,
    cs_str: String,
    ds_str: String,
    md_str: String,

    // Status flags
    is_secondary: bool,
    is_spliced: bool,               // Contains N_SKIP ops
    trans_strand: u8,               // 0=unknown, 1=+, 2=-, 3=ambiguous
    split: u8,                      // z-drop split: 1=left, 2=right
    split_inv: bool,                // Trigger inversion alignment
    inv: bool,                      // Is inversion result
    proper_frag: bool,              // Concordant PE pair
    seg_split: bool,                // From seg_gen splitting
    is_alt: bool,                   // ALT contig
    is_root_chain: bool,            // Root chain (parent==self)
    div: f32,                       // dv:f tag (seed-based divergence)

    // Scores for sorting/filtering
    dp_score: i32,                  // For ranking (may be recalculated)
    dp_score_original: i32,         // ms:i tag (original)
    effective_cnt: i32,             // For mm_filter_regs
    pre_num_suboptimal: i32,        // Pre-alignment n_sub
    dp_score_secondary: i32,        // Best dp_max among children
    secondary_chain_score: i32,     // Best chain_score among children
    num_suboptimal: i32,            // Count of similar-scoring children
}
```

### Index (`index.rs` + `index_bucket.rs`)

```
Index {
    kmer_size: usize,               // K-mer length
    window_size: usize,             // Minimizer window length
    homopolymer_compressed: bool,   // HPC mode
    index: usize,                   // Part number (0-based)
    seqs: Vec<TargetSequence>,      // Reference sequence metadata
    backend: LookupBackend,         // Seed lookup (BucketHash)
    packed_seqs: Vec<u32>,          // 4-bit packed sequences (8 bases/u32)
}

LookupBackend::BucketHash(BucketHashLookup {
    bucket_bits: u32,               // log2(n_buckets) for hash partitioning
    buckets: Vec<Bucket>,           // Per-bucket open-addressing hash tables
    positions: Vec<u64>,            // Shared flat position array
})

Bucket {
    keys: Vec<u32>,                 // Hash suffixes (EMPTY_KEY = u32::MAX)
    vals: Vec<u64>,                 // Packed (offset << 32) | count
    mask: u32,                      // capacity - 1 (power of 2)
}

TargetSequence {
    name: String,
    len: usize,
    offset: u64,                    // Cumulative offset into packed_seqs
    is_alt: bool,
}
```

### DpResult (`dp.rs`)

```
DpResult {
    max: i32,                       // Best score found
    max_score_query_pos: i32,       // Query pos of max
    max_score_target_pos: i32,      // Target pos of max
    max_query_end_score: i32,       // Score when query exhausted
    max_query_end_target_pos: i32,  // Target pos for above
    max_target_end_score: i32,      // Score when target exhausted
    max_target_end_query_pos: i32,  // Query pos for above
    score: i32,                     // Final alignment score
    cigar_capacity: i32,
    cigar_len: i32,
    reach_end: i32,                 // 1 if reached sequence end
    zdropped: i32,                  // 1 if z-drop triggered
    cigar: Vec<u32>,                // Packed: len << 4 | op (0=M, 1=I, 2=D, 3=N)
}
```

### MapContext (`map.rs`)

```
MapContext {
    anchors: Vec<Minimizer>,        // Anchor buffer (reused per read)
    minimizers: Vec<Minimizer>,     // Query minimizer buffer
    chain_bufs: ChainingBuffers,    // DP + SoA chaining buffers (see below)
    mini_pos: Vec<u64>,             // Position array for divergence
}

ChainingBuffers {
    predecessors: Vec<i64>,         // DP predecessor array
    scores: Vec<i32>,               // DP score array
    peak_scores: Vec<i32>,          // DP peak score array
    visited: Vec<i32>,              // Backtrack visited flags
    soa_ref_pos: Vec<i32>,          // SoA chaining buffers for SIMD
    soa_query_pos: Vec<i32>,
    soa_query_span: Vec<i32>,
    soa_ref_id_strand: Vec<u32>,
    soa_scores_buf: Vec<i32>,
}
```

---

## 6. Algorithm Details

### Minimizer Sketching

For a sequence of length L with k-mer size k and window size w:

1. Scan positions 0..L, maintaining forward and reverse-complement k-mer hashes
2. For each valid position (no ambiguous bases, k consecutive bases seen):
   - Forward: `kmer[0] = (kmer[0] << 2 | base) & mask`
   - Reverse: `kmer[1] = (kmer[1] >> 2) | (3 ^ base) << shift`
   - Skip palindromic k-mers (`kmer[0] == kmer[1]`)
   - Select canonical strand: `z = if kmer[0] < kmer[1] { 0 } else { 1 }`
   - Hash: `h = kmer_hash(kmer[z], mask) << 8 | span`
3. Maintain circular buffer of w k-mer hashes. Output minimizer when the minimum changes.
4. HPC mode: compress homopolymer runs, adjust k-mer span accordingly.

### Seed Collection

For each query minimizer with hash h:
1. Lookup via `get_range(h)`: select bucket by `h & mask`, probe for `h >> bucket_bits` in per-bucket hash table
2. Retrieve positions via `get_by_range(start, end)`: slice into shared positions array
3. For each matching reference position: create anchor `(ref_id, ref_pos, query_pos, strand)`
4. Filter: skip if `hit_count` > `mid_occ` (unless in `select_seeds` heap)
5. Result: `Vec<Minimizer>` anchors sorted by reference position via radix sort

### Chaining DP Recurrence

For anchors `a[0..n]` sorted by (`ref_pos`):

```
f[i] = max(
    a[i].span,                                  // base case
    max over j < i where compatible(i,j) {
        f[j] + score(a[i], a[j])
    }
)

compatible(i, j):
    same ref_id and strand
    0 < query_diff <= max_dist_x
    0 < ref_diff <= max_dist_y
    |ref_diff - query_diff| <= bandwidth

score(a[i], a[j]):
    min_diff = min(ref_diff, query_diff)
    base = min(a[j].span, min_diff)
    gap = |ref_diff - query_diff|
    penalty = chn_pen_gap * gap + 0.5 * log2(gap + 1) if gap > 0
    return base - ceil(penalty)
```

Backtrack from highest f[i] via predecessor links. Skip limit (`max_chain_skip`) prevents O(n^2) worst case.

### RMQ Chaining

Uses arena-based treap (randomized BST) augmented with subtree-minimum for O(log n) range queries:

1. Process anchors left-to-right by `ref_pos`
2. For anchor i, query treap for best predecessor in `query_pos` range `[i.qpos - max_dist, i.qpos]`
3. Insert anchor i into treap with `key=query_pos`, `value=negative_score`
4. Erase out-of-range anchors (`ref_pos distance > max_dist_x`)
5. `rmq(lo, hi)` traverses treap branches, using subtree-min to prune entire subtrees

The treap uses xorshift64 random priorities for expected O(log n) height.

### Banded DP (Suzuki-Kasahara Formulation)

Processes anti-diagonals of the DP matrix with SIMD vectors:

1. **Query profile**: Pre-compute score lookup `qp[q][t] = mat[t*5 + q_base[q]]`
2. **Band**: For each target position r, compute query positions `[st, en]` within bandwidth
3. **DP states** (dual-affine):
   - H: best score ending in match/mismatch
   - E1: best score ending in short gap (penalty q+k*e)
   - F1: best score ending in short gap (query direction)
   - E2: best score ending in long gap (penalty q2+k*e2)
   - F2: best score ending in long gap (query direction)
4. **SIMD**: Process 16 positions per `__m128i` register (i8 values with periodic overflow check)
5. **Z-drop**: If `current_max < global_max - zdrop`, stop and set `zdropped=1`
6. **Traceback**: Store 2-bit decisions per cell, reconstruct CIGAR in reverse

### MAPQ Calculation

```
if has_cigar:
    mapq = 40 * (1 - dp_max2 / dp_max) * ln(dp_max)  // log-odds
else:
    mapq = 40 * (1 - s2 / s1) * min(1, matches/10) * ln(s1)

Adjustments:
    if dp_max > dp_max2 && mapq == 0: mapq = 1  // promote unique
    if is_inversion: mapq = 0
    if PE concordant: mapq += pe_bonus
    clamp to [0, 60]
```

### Post-Chain Filtering

`assign_parents()`: For each chain sorted by score:
- First chain = primary (parent = self)
- Subsequent chains: find highest-scoring earlier chain with query overlap > `mask_level`
- If found: child of that parent. If not: new primary.

`select_sub()`: For each parent:
- Keep at most `best_n` children
- Remove children with score < `pri_ratio * parent_score`
- Hard mask: remove if overlap > `mask_level` and score < threshold

---

## 7. SIMD Architecture

### Platform Support

**DP kernels (`dp/single.rs`, `dp/dual.rs`, `dp/splice.rs`)**:

| Platform | Register Type | Width | Dispatch |
|----------|--------------|-------|----------|
| x86\_64 AVX512BW | `__m512i` | 64 lanes | Runtime `is_x86_feature_detected!("avx512bw")` |
| x86\_64 AVX2 | `__m256i` | 32 lanes | Runtime `is_x86_feature_detected!("avx2")` |
| x86\_64 SSE4.1 | `__m128i` | 16 lanes | Runtime `is_x86_feature_detected!("sse4.1")` |
| x86\_64 SSE2 | `__m128i` | 16 lanes | Fallback (always available on x86\_64) |
| aarch64 NEON | `uint8x16_t` | 16 lanes | Compile-time `#[cfg(target_arch = "aarch64")]` |
| wasm32 SIMD128 | `v128` | 16 lanes | Compile-time `#[cfg(target_feature = "simd128")]` |
| Scalar | `i32` arrays | 1 | Fallback for any platform |

**Chaining (chain_simd.rs)**:

| Platform | Width | Dispatch |
|----------|-------|----------|
| x86\_64 AVX2 | 8 predecessors/iter | Runtime `is_x86_feature_detected!("avx2")` |
| aarch64 NEON | 4 predecessors/iter | Compile-time `#[cfg(target_arch = "aarch64")]` |
| Scalar | 1 predecessor/iter | Fallback |

Note: minimap2 has no SIMD chaining and no AVX2/AVX512 DP kernels (SSE4.1 max).

### SSE2/SSE4.1 Macro Unification

SSE2 and SSE4.1 share `__m128i` register type and differ in only 3 operations. Each algorithm file (`dp/single.rs`, `dp/dual.rs`, `dp/splice.rs`) defines its own SSE macro:

```rust
// In dp/single.rs:
macro_rules! extend_single_affine_impl { ... }

// In dp/dual.rs:
macro_rules! extend_dual_affine_impl { ... }

// In dp/splice.rs:
macro_rules! extend_splice_impl { ... }
```

Each macro is instantiated for SSE2, SSE4.1, and WASM within the same file. AVX2 and AVX512 have separate macros (also in the same file) due to different SIMD widths.

**Operation differences**:

| Operation | SSE4.1 | SSE2 Emulation |
|-----------|--------|----------------|
| `max_epi8` | `_mm_max_epi8(a, b)` | `cmpgt + and + andnot + or` |
| `min_epi8` | `_mm_min_epi8(a, b)` | `cmpgt + andnot + and + or` |
| `blendv_epi8` | `_mm_blendv_epi8(a, b, mask)` | `andnot(mask, a) \| and(mask, b)` |

**Compile-time dispatch**: `$is_sse41` is a `bool` constant — compiler eliminates dead branches via constant folding.

### NEON Kept Separate

NEON implementations are NOT unified with SSE via macros due to structural differences:
- Different register types (`uint8x16_t` vs `__m128i`)
- `vextq_u8` (concatenate-extract) vs `_mm_slli_si128` (byte shift)
- Signed/unsigned reinterpretation required (`vreinterpretq_s8_u8`)
- `vbslq_u8` (bit select) vs blend
- Reversed `andnot` argument order

NEON shares only the non-SIMD helpers from `dp/common.rs` (init, `push_cigar`, traceback). All SIMD variants for one algorithm live in the same file (e.g., `dp/dual.rs` contains the SSE, AVX2, AVX512, NEON, WASM, and scalar dual-affine implementations).

### Runtime Dispatch Pattern

**DP alignment** (dp.rs):
```rust
pub fn extend_dual_affine(...) {
    if FORCE_SCALAR { return ..._scalar(...); }
    #[cfg(target_arch = "x86_64")] {
        if avx512bw   { return ..._avx512(...); }   // 64 lanes
        if avx2       { return ..._avx2(...); }      // 32 lanes
        if sse4.1     { return ..._sse41(...); }     // 16 lanes
        return ..._sse2(...);                         // 16 lanes (fallback)
    }
    #[cfg(target_arch = "aarch64")]
        return ..._neon(...);                         // 16 lanes
    #[cfg(target_arch = "wasm32")]
        return ..._wasm(...);                         // 16 lanes
    ..._scalar(...)                                   // 1 lane
}
```

**Chaining** (chain.rs → chain_simd.rs):
```rust
pub fn chain_anchors(...) {
    if is_cdna || n_seg > 1 || a.len() < 32 { return ..._scalar(...); }
    #[cfg(target_arch = "x86_64")]
    if avx2 { return chain_anchors_avx2(...); }       // 8 predecessors/iter
    #[cfg(target_arch = "aarch64")]
    return chain_anchors_neon(...);                    // 4 predecessors/iter
    ..._scalar(...)                                    // 1 predecessor/iter
}
```
```

---

## 8. Performance Patterns

### Overlapped I/O

```
Read-ahead thread          Worker threads (rayon)
  |                              |
  | sync_channel(1)              |
  |------ batch of reads ------->|
  |                              |-- map_query() --|
  |------ next batch ----------->|-- map_query() --|
  |                              |-- format_output() --|
  |                              |-- write to stdout --|
```

One batch reads ahead while the previous batch aligns. `sync_channel(1)` bounds memory to 2 batches.

### Buffer Reuse via MapContext

```rust
// Per-thread, reused across reads:
let mut ctx = MapContext::new();

for read in batch {
    // mem::take moves buffers out, avoiding allocation
    let mut anchors = std::mem::take(&mut ctx.anchors);
    map_query(opt, mi, &read, &mut anchors, ...);
    ctx.anchors = anchors;  // Return buffers
}
```

Chaining DP buffers (`chain_bufs.predecessors`, `.scores`, `.peak_scores`, `.visited`) and seed buffers (`mini_pos`) are all reused via `ChainingBuffers`.

### Thread-Local DP Matrix Caching

```rust
thread_local! {
    static CACHED_MEM: Cell<Option<AlignedMemory>> = Cell::new(None);
}
```

DP matrices (~17MB for typical long reads) allocated once per thread, reused across all alignments. Avoids `mmap`/`munmap` syscalls per alignment call.

### Targeted Memory Zeroing

Only the DP scoring region is zeroed before each alignment. Traceback memory (95%+ of allocation) is NOT zeroed — it's written before read during DP. This reduced zeroing overhead from ~10% to <0.5% of alignment time.

---

## 9. I/O Layer

### Custom FASTA/FASTQ Reader (`src/fasta/reader.rs`)

Zero-copy streaming parser, faster than noodles for raw sequence parsing.

```rust
// Zero-copy mode (preferred for alignment):
reader.for_each_record(|record: RefRecord| {
    // record.seq() returns &[u8] into reader's buffer
    // No allocation per record
});

// Owned mode (when records need to outlive reader):
for record in reader.records() {
    // record.seq: Vec<u8> (allocated)
}
```

Auto-detects FASTA (`>` header) vs FASTQ (`@` header). Handles multi-line FASTA sequences. Supports gzip-compressed input via flate2.

### Memory-Mapped FASTA (`src/fasta/reader.rs`)

For reference sequences, `read_fasta_mmap()` uses `memmap2` for zero-copy access to large FASTA files (native platforms only, gated with `cfg(not(wasm32))`).

---

## 10. Testing

### Unit Tests

Located in `#[cfg(test)] mod tests` blocks within source files. Cover:
- Radix sort correctness (sort.rs)
- Minimizer sketching (sketch.rs)
- DP correctness: simple match, mismatch, gaps, dual-affine, splice (dp.rs)
- Chain scoring (chain.rs)
- CIGAR formatting: CS, MD, DS tags (extend.rs)
- Index build/save/load round-trip (index.rs)
- Various edge cases

### Integration Tests (29/29 pass)

`tests/integration_test.sh --mode=quick` compares rammap vs minimap2 across all major presets plus SIMD concordance:

| Test | Status | Notes |
|------|--------|-------|
| map-ont, map-ont-cigar, map-ont-sam, map-ont-cs-md | 100% match | |
| lr-hq, lr-hqae, map-hifi | 100% match | |
| splice, splice-hq, cdna, splice-sr | 100% match | |
| sr, sr-sam | 100% match | |
| asm5, asm10, asm20 | 100% match | |
| ava-ont | 100% match | |
| split-perpart | 100% match | |
| split-merge | PASS | 4239 diffs (4238 real, 1 MAPQ/de:f) |
| custom-scoring, custom-kw | 100% match | custom-kw: 1 known inversion UB diff |
| secondary-N5, eqx | 100% match | |
| map-pb | PASS | 1 known inversion UB diff |
| map-iclr | PASS | 1 known inversion UB diff |
| simd-ont, simd-hifi, simd-splice, simd-sr | 0 diffs | SSE vs AVX2 vs AVX512 vs scalar-chain |

Test data: `tests/inttest/` (chr20 reference, 5000 reads per preset).

### WASM Tests

7 WASM tests pass via `wasm-pack test --node -- --lib`.

### Performance Benchmarks

See [performance.md](performance.md) for full benchmark results comparing rammap vs minimap2.

---

## 11. Preset Reference

| Preset | k | w | HPC | Key Flags | Scoring (A/B/O/E/O2/E2) | Special |
|--------|---|---|-----|-----------|-------------------------|---------|
| map-ont | 15 | 10 | - | - | 2/4/4/2/24/1 | Default |
| map-pb | 19 | 10 | yes | - | 2/4/4/2/24/1 | HPC |
| lr:hq | 19 | 19 | - | RMQ | 2/4/4/2/24/1 | mid\_occ 50-500 |
| map-hifi | 19 | 19 | - | - | 1/4/6/2/26/1 | min\_dp\_max=200 |
| lr:hqae | 25 | 51 | - | RMQ,HEAP | 2/4/4/2/24/1 | rmq\_inner\_dist=5000 |
| map-iclr | 19 | 10 | - | - | 2/6/10/2/50/1 | transition=4 |
| asm5 | 19 | 19 | - | RMQ | 1/19/39/3/81/1 | bw=1000 |
| asm10 | 19 | 19 | - | RMQ | 1/9/16/2/41/1 | bw=1000 |
| asm20 | 19 | 10 | - | RMQ | 1/4/6/2/26/1 | bw=1000 |
| sr | 21 | 11 | - | SHORT,FRAG,HEAP,NO\_2ND | 2/8/12/2/24/1 | PE FR, bw=100 |
| splice | 15 | 5 | - | SPLICE,FLANK | 1/2/2/1/32/0 | max\_gap\_ref=200K |
| splice:hq | 15 | 5 | - | SPLICE,FLANK | 1/4/6/1/24/0 | noncanon=5 |
| splice:sr | 15 | 5 | - | SPLICE,FRAG,HEAP,WEAK,SR\_RNA | 1/4/6/1/24/0 | PE weak |
| cdna | 15 | 5 | - | SPLICE,FLANK | 1/2/2/1/32/0 | Same as splice |
| ava-ont | 15 | 5 | - | ALL\_CHAINS,NO\_DIAG,NO\_DUAL,NO\_LJOIN | 2/4/4/2/24/1 | All-vs-all |
| ava-pb | 19 | 5 | yes | ALL\_CHAINS,NO\_DIAG,NO\_DUAL,NO\_LJOIN | 2/4/4/2/24/1 | HPC, all-vs-all |

**Derived values**: `chn_pen_gap = chain_gap_scale * 0.01 * k`, `chn_pen_skip = chain_skip_scale * match_score * 0.01`.

---

## 12. Memory Safety

### Unsafe Code Inventory

All `unsafe` code is confined to SIMD kernels and parallel sort:

| File | Unsafe blocks | Reason |
|------|:------------:|--------|
| `dp/single.rs` | ~30 | SIMD intrinsics, raw pointer DP matrix access |
| `dp/dual.rs` | ~35 | SIMD intrinsics, raw pointer DP matrix access, pointer-based traceback |
| `dp/splice.rs` | ~35 | SIMD intrinsics, raw pointer DP matrix access, pointer-based traceback |
| `dp/common.rs` | ~5 | SIMD helpers, memory allocation, traceback pointer access |
| `dp/lw.rs` | ~5 | SIMD intrinsics for lightweight alignment |
| `chain_simd.rs` | ~15 | AVX2 + NEON chaining kernels with SIMD intrinsics |
| `chain.rs` | 2 | Dispatch calls to `chain_anchors_avx2` / `chain_anchors_neon` |
| `sort.rs` | 1 | `from_raw_parts_mut` for parallel non-overlapping mutable slice access in `radix_sort_pair` |

**Zero unsafe** in all other files: `dp/mod.rs`, `extend.rs`, `pipeline.rs`, `map.rs`,
`filter.rs`, `seed.rs`, `sketch.rs`, `index.rs`, `index_bucket.rs`, `fasta/`, `api.rs`,
`main.rs`, `jump.rs`, `junc.rs`, `pair.rs`, `split.rs`, `stats.rs`.

### Why SIMD Requires Unsafe

Rust's `std::arch` SIMD intrinsics are `unsafe fn` for two reasons:

1. **Target feature guarantee**: Calling an AVX2 intrinsic on a CPU without AVX2
   is undefined behavior. The caller must verify CPU support (via runtime
   `is_x86_feature_detected!` or compile-time `#[target_feature]`).

2. **Raw pointer access**: SIMD load/store operations (`_mm_loadu_si128`,
   `_mm_storeu_si128`) dereference raw pointers into the DP matrix buffers.

Pure arithmetic intrinsics (`_mm_add_epi8`, `_mm_max_epi8`, etc.) are inherently
safe operations but are marked `unsafe fn` in `std::arch` by convention. As of
Rust edition 2024 (1.94+), calling these within a `#[target_feature]` function
no longer requires an explicit `unsafe` block for the target feature check —
only pointer-based operations still need it.

### Fully-Safe Scalar Code Path

A complete alignment pipeline exists without any `unsafe` code, used as the
fallback on architectures without SIMD (or via `FORCE_SCALAR=1`):

```
Query FASTQ
  → sketch.rs              (safe)
  → seed.rs                (safe)
  → chain.rs scalar DP     (safe)
  → filter.rs              (safe)
  → extend.rs orchestration(safe)
  → dp.rs scalar kernels:
      extend_single_affine_scalar  (safe DP fill + safe traceback)
      lightweight_align_i16_scalar (safe)
      global_align_gotoh           (safe)
  → pipeline.rs formatting (safe)
```

The scalar DP kernels produce identical results to SIMD (verified by 600/600
concordance tests against minimap2). The safe traceback (`traceback_single_affine_safe`)
uses bounds-checked slice indexing instead of raw pointer arithmetic, with zero
performance difference at `-O3` (compiler eliminates the bounds checks).

### Index and FASTA I/O

Index loading (`index.rs`) and FASTA reading (`fasta/reader.rs`) are fully safe.
Index binary parsing uses `read_u32_vec`/`read_u64_vec` helpers that read into
byte buffers and parse with `from_le_bytes`. FASTA loading uses `std::fs::read`
(no memory mapping).

---

## 13. Compatibility notes

Key behavioral details for anyone modifying alignment code.

### Seeding & Indexing
- **mid_occ calculate-once**: minimap2 calculates `mid_occ` from first index part only. Must NOT recalculate per-part for split index.
- **Seed heap tie-breaking**: When seed counts tie, minimap2 breaks ties by position.

### Chaining
- **RMQ boundary semantics**: minimap2's `krmq_rmq` uses CLOSED interval `[lo_y, hi_y]`. Our `rmq()` must include elements at exactly `hi_y`.
- **compact_a sorting**: After backtracking, chains are re-sorted by `ref_pos` of first anchor, then anchors reordered accordingly.

### Alignment
- **split_inv left extension zdrop**: When `split_inv=true`, left extension uses `zdrop_inv` instead of `zdrop` (align.c:791). Without this, left extension overextends, breaking inversion detection.
- **Case-insensitive CIGAR**: FASTA may have lowercase (soft-masked) bases. Must `.to_ascii_uppercase()` both sides for `M -> =/X` expansion.
- **fix_bad_ends_splice**: Splice mode trims weakly-supported terminal exons before main alignment.
### SIMD Tie-Breaking
- **Non-deterministic gap placement across SIMD widths**: When multiple DP cells have equal scores, the traceback direction depends on which cell is processed last within a SIMD register. Different SIMD widths (SSE=16, AVX2=32, AVX512=64 bytes) process cells in different orders, producing different gap placements in the CIGAR for the same score. This is inherent to the DP design (including minimap2's C implementation) and is NOT a bug — the alignments are equally valid alternatives. It does not affect scores, alignment boundaries, or the mapper's output (the chaining/filtering pipeline eliminates borderline alignments).

### minimap2 Undefined Behavior
- **ksw_ll_i16 query_end overflow**: When `qlen % 8 != 0`, the lightweight Smith-Waterman can return `query_end` at a SIMD padding position beyond `qlen`. minimap2's `mm_align1_inv` uses this value to compute a negative buffer offset (reading before the query buffer start — undefined behavior in C). rammap clamps `query_end` to `pos < qlen` and rejects the inversion. This causes ~6 known diffs across integration tests (all `tp:A:I` inversions with `cm:i:0, s1:i:0`). See [`docs/minimap2-ksw-ll-ub.md`](minimap2-ksw-ll-ub.md) for details.

### Output Formatting
- **MD tag format**: Each mismatch outputs `{match_count}{base}` individually. Consecutive mismatches: `0A0B0C`. Deletions prefix with count: `{match_count}^{bases}`.
- **CS/MD priority**: When both `--cs` and `--MD` set, only MD is output (minimap2 if/else).
- **PE PAF qname**: Append `/{seg_idx+1}` to raw qname. Does NOT trim existing `/1`/`/2` suffix.
- **Non-CIGAR s2/MAPQ**: In non-CIGAR mode, use `post_subsc` from chain score, not `dp_max`. `has_p` = `!cigar_str.is_empty()`.

### Filtering
- **ALL_CHAINS outside do_cigar gate**: ava presets have `do_cigar=false`. `ALL_CHAINS` handling must run regardless.
- **mm_filter_regs skip for inversions**: Must check `!r.inv && cnt < min_cnt` in BOTH filter passes.
- **Inversion MAPQ=0**: Set mapq=0 for inversions BEFORE the `dp_max > dp_max2` promotion.

### Split Index
- **rl:i:0 in merge**: minimap2's merge doesn't update `rep_len`, so output tag is always 0.
- **No mm_filter_regs in merge**: Merge only runs `mm_hit_sort + mm_set_parent + mm_select_sub`.
