//! Minimal example: align a sequence against a reference using the rammap API.
//!
//! Usage:
//!   cargo run --release --example simple_align -- <reference.fa> <query.fq>
//!
//! Example:
//!   cargo run --release --example simple_align -- tests/inttest/chr20.fa tests/inttest/ont_5000.fq

use rammap::{Aligner, Preset, Strand};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <reference.fa|.mmi|.idx> <reads.fq>", args[0]);
        process::exit(1);
    }
    let ref_path = &args[1];
    let reads_path = &args[2];

    // Load reference — detects .mmi (pre-built index) vs .fa (build on the fly)
    eprintln!("[*] Loading reference: {}", ref_path);
    let aligner = if ref_path.ends_with(".mmi") || ref_path.ends_with(".idx") {
        Aligner::from_index(ref_path, Preset::MapOnt).expect("Failed to load index")
    } else {
        Aligner::from_fasta(ref_path, Preset::MapOnt).expect("Failed to build index")
    };

    // Open reads and align each one
    eprintln!("[*] Aligning reads from: {}", reads_path);
    let reader = rammap::fasta::open(reads_path).expect("Failed to open reads");
    let mut n_reads = 0;
    let mut n_aligned = 0;

    for record in reader.records() {
        let record = record.expect("Failed to parse read");
        let name = record.name();
        let seq = record.sequence();

        let result = aligner.map_seq(name, seq);

        if !result.alignments.is_empty() {
            n_aligned += 1;
            for aln in &result.alignments {
                let strand = match aln.strand { Strand::Forward => '+', Strand::Reverse => '-' };
                // Print PAF-like output
                println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    name, seq.len(), aln.query_start, aln.query_end,
                    strand, aln.target_name, aln.target_len,
                    aln.target_start, aln.target_end,
                    aln.matches, aln.block_len, aln.mapq,
                );
            }
        }

        n_reads += 1;
        if n_reads >= 10 { break; } // just show first 10 for the example
    }

    eprintln!("[*] Done: {}/{} reads aligned", n_aligned, n_reads);
}
