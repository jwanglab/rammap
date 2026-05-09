//! # rammap alignment engine
//!
//! Minimap2-compatible sequence alignment pipeline producing byte-identical output.
//!
//! ## Pipeline stages
//!
//! ```text
//! Query FASTQ
//!   → sketch (sketch.rs)        — minimizer sketching
//!   → seed   (seed.rs)          — index lookup, seed collection
//!   → chain  (chain.rs/rmq)     — anchor chaining (DP or RMQ)
//!   → filter (filter.rs)        — chain filtering, parent assignment
//!   → align  (extend.rs+dp.rs)  — DP extension alignment per chain
//!   → output (pipeline.rs)      — MAPQ, PAF/SAM formatting
//! ```
//!
//! ## Key types crossing stage boundaries
//!
//! - [`sketch::Minimizer`] — 128-bit packed seed/anchor (used in seeding, chaining, alignment)
//! - [`map::Mapping`] — one chain with metadata (produced by chaining, consumed by alignment)
//! - [`extend::AlignResult`] — DP alignment output (CIGAR, coords, score)
//! - [`pipeline::AlnResult`] — final formatted alignment (ready for PAF/SAM output)
//! - [`map::MapContext`] — per-thread reusable buffers for seeding and chaining
//! - [`map::MapOptions`] — all algorithm parameters (scoring, chaining, filtering thresholds)

pub mod align_simple;
pub mod chain;
pub(crate) mod env_flags;
pub mod chain_simple;
pub mod chain_rmq;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
pub(crate) mod chain_simd;
pub mod dp;
pub mod extend;
pub mod filter;
pub mod index_bucket;
pub mod index;
pub mod jump;
pub mod junc;
pub mod map;
pub mod pair;
pub mod pipeline;
pub mod seed;
pub mod sketch;
pub mod sort;
pub mod split;
pub mod strobemer;
pub mod syncmer;
pub mod stats;
#[cfg(target_arch = "wasm32")]
pub mod wasm_lib;
