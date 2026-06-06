pub mod align;
pub mod api;
pub mod fasta;

// Convenience re-exports for library users
pub use api::{Aligner, Mapping, MapResult, MapOpts, Preset, Strand, CigarOp};
pub use api::{dp_align, dp_global, dp_local, dp_extension, DpScoring, DpAlignment, encode_nt4};
// Runtime knob for the per-thread DP scratch-cache retention cap (peak-RSS control).
pub use align::dp::{set_dp_cache_cap_mb, DEFAULT_DP_CACHE_CAP_MB, DP_CACHE_UNBOUNDED};
