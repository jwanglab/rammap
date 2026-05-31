pub mod align;
pub mod api;
pub mod fasta;

// Convenience re-exports for library users
pub use api::{Aligner, Mapping, MapResult, MapOpts, Preset, Strand, CigarOp};
pub use api::{dp_align, dp_global, dp_local, dp_extension, DpScoring, DpAlignment, encode_nt4};
