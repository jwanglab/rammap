pub mod record;
pub mod reader;

pub use record::Record;
pub use reader::{Reader, open, FastxError, parse_fasta_bytes};

pub use reader::read_fasta;
