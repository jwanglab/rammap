use std::io::{self, BufRead};
use std::path::Path;
use std::fs::File;
use flate2::read::MultiGzDecoder;

/// Convert RNA bases U->T and u->t in place
/// ASCII 'U' = 0x55, 'T' = 0x54, 'u' = 0x75, 't' = 0x74
/// so just decrement
#[inline]
fn rna_to_dna_inplace(seq: &mut [u8]) {
    for b in seq.iter_mut() {
        if *b == b'U' || *b == b'u' { *b -= 1; }
    }
}

use thiserror::Error;
use crate::fasta::record::{Record, RefRecord};

#[derive(Error, Debug)]
pub enum FastxError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Unknown file format or empty file")]
    UnknownFormat,
    #[error("Line {line}: Malformed FASTQ: {msg}")]
    MalformedFastq { line: usize, msg: String },
    #[error("Line {line}: Truncated file")]
    Truncated { line: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RecordType {
    Fasta,
    Fastq,
}

pub struct Reader<R> {
    reader: R,
    record_type: Option<RecordType>, // Detected on first read
    buf: Vec<u8>,
    line_number: usize,
    done: bool,
}

impl<R: BufRead> Reader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            record_type: None,
            buf: Vec::with_capacity(64 * 1024),
            line_number: 1,
            done: false,
        }
    }

    /// `Ok(None)` = empty input (EOF before any byte): zero records
    /// `Ok(Some(fmt))` = recognized
    /// `Err(UnknownFormat)` only for non-empty, unrecognized input
    fn detect_format(&mut self) -> Result<Option<RecordType>, FastxError> {
        let buffer = self.reader.fill_buf()?;
        if buffer.is_empty() {
             return Ok(None);
        }
        if buffer[0] == b'>' {
            Ok(Some(RecordType::Fasta))
        } else if buffer[0] == b'@' {
            Ok(Some(RecordType::Fastq))
        } else {
            // maybe skip empty lines? For now strict.
            Err(FastxError::UnknownFormat)
        }
    }

    /// Iterate over records using a callback (Zero-Copy)
    pub fn for_each_record<F>(&mut self, callback: F) -> Result<(), FastxError>
    where
        F: FnMut(RefRecord),
    {
        if self.record_type.is_none() {
            match self.detect_format()? {
                Some(fmt) => self.record_type = Some(fmt),
                None => return Ok(()), // empty input: nothing to do
            }
        }

        match self.record_type.unwrap() {
            RecordType::Fastq => self.parse_fastq(callback),
            RecordType::Fasta => self.parse_fasta(callback),
        }
    }
    
    /// Returns an iterator over owned Records.
    /// Note: This copies data. Use `for_each_record` for maximum performance.
    pub fn records(self) -> RecordIter<R> {
        RecordIter { reader: self }
    }

    fn parse_fastq<F>(&mut self, mut callback: F) -> Result<(), FastxError>
    where
        F: FnMut(RefRecord),
    {
        loop {
            self.buf.clear();
            
            // Line 1: Header
            let n1 = self.reader.read_until(b'\n', &mut self.buf)?;
            if n1 == 0 { break; } // Normal EOF
            if self.buf[0] != b'@' {
                // we quietly skip malformed records...
                // eprintln!("Warning: Line {} does not start with '@'. Skipping.", self.line_number);
                self.line_number += 1;
                continue;
            }
            self.line_number += 1;

            let l1_end = trim_newline(&self.buf);
            let head_range = 1..l1_end; // Skip @
            
            // Line 2: Sequence
            let seq_start = self.buf.len();
            let n2 = self.reader.read_until(b'\n', &mut self.buf)?;
            if n2 == 0 { break; }
            self.line_number += 1;

            let l2_end = seq_start + trim_newline(&self.buf[seq_start..]);
            let seq_range = seq_start..l2_end;
            
            // Line 3: +
            let plus_start = self.buf.len();
            let n3 = self.reader.read_until(b'\n', &mut self.buf)?;
            if n3 == 0 { break; }
            if self.buf[plus_start] != b'+' {
                 self.line_number += 1;
                 continue;
            }
            self.line_number += 1;
            
            // Line 4: Quality
            let qual_start = self.buf.len();
            let n4 = self.reader.read_until(b'\n', &mut self.buf)?;
            if n4 == 0 { break; }
            self.line_number += 1;

            let l4_end = qual_start + trim_newline(&self.buf[qual_start..]);
            let qual_range = qual_start..l4_end;
            
            // Validation
            if (seq_range.end - seq_range.start) != (qual_range.end - qual_range.start) {
                // Length mismatch
                continue;
            }
            
            {
                let r = RefRecord {
                    head: &self.buf[head_range],
                    seq: &self.buf[seq_range],
                    qual: Some(&self.buf[qual_range]),
                };
                callback(r);
            }
        }
        Ok(())
    }

    fn parse_fasta<F>(&mut self, mut callback: F) -> Result<(), FastxError>
    where
        F: FnMut(RefRecord),
    {
        let mut seq_buf = Vec::new(); // Buffer for accumulating sequence lines
        let mut header_buf = Vec::new(); 
        
        self.buf.clear();
        if self.reader.read_until(b'\n', &mut self.buf)? == 0 { return Ok(()); }
        self.line_number += 1;
        
        if self.buf[0] != b'>' { return Err(FastxError::UnknownFormat); }
        
        loop {
            header_buf.clear();
            header_buf.extend_from_slice(&self.buf); 
            
            seq_buf.clear();
            
            loop {
                self.buf.clear();
                let n = self.reader.read_until(b'\n', &mut self.buf)?;
                if n == 0 {
                    break;
                }
                self.line_number += 1;
                
                if self.buf[0] == b'>' {
                    break;
                }
                
                let l_end = trim_newline(&self.buf);
                seq_buf.extend_from_slice(&self.buf[..l_end]);
            }
            
            let l1_end = trim_newline(&header_buf);
            
            let r = RefRecord {
                head: &header_buf[1..l1_end],
                seq: &seq_buf,
                qual: None,
            };
            callback(r);
            
            if self.buf.is_empty() {
                break;
            }
        }
        Ok(())
    }
}

fn trim_newline(buf: &[u8]) -> usize {
    if buf.ends_with(b"\r\n") {
        buf.len().saturating_sub(2)
    } else if buf.ends_with(b"\n") {
        buf.len().saturating_sub(1)
    } else {
        buf.len()
    }
}

pub struct RecordIter<R> {
    reader: Reader<R>,
}

impl<R: BufRead> Iterator for RecordIter<R> {
    type Item = Result<Record, FastxError>;

    fn next(&mut self) -> Option<Self::Item> {
        // This is inefficient because we cannot yield from inside the `for_each_record` loop 
        // without coroutines.
        // We have to reimplement `parse_one` logic or use a state machine.
        // Or adapt `for_each_record` to be re-entrant? No.
        
        // For convenience in high-level code, we just need `next()`.
        // Let's implement a `read_next` method on Reader that returns `Option<Record>`.
        match self.reader.read_next() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<R: BufRead> Reader<R> {
     pub fn read_next(&mut self) -> Result<Option<Record>, FastxError> {
        if self.done {
            return Ok(None);
        }
        if self.record_type.is_none() {
            match self.detect_format() {
                Ok(Some(fmt)) => self.record_type = Some(fmt),
                Ok(None) => { self.done = true; return Ok(None); } // empty input: 0 records
                Err(e) => { self.done = true; return Err(e); }     // report once, then done
            }
        }

        let r = match self.record_type.unwrap() {
             RecordType::Fastq => self.read_next_fastq(),
             RecordType::Fasta => self.read_next_fasta(),
        };
        if r.is_err() {
            self.done = true;
        }
        r
    }
    
    fn read_next_fastq(&mut self) -> Result<Option<Record>, FastxError> {
        loop {
            self.buf.clear();
            let n1 = self.reader.read_until(b'\n', &mut self.buf)?;
            if n1 == 0 { return Ok(None); }
             if self.buf[0] != b'@' {
                self.line_number += 1;
                continue;
            }
            self.line_number += 1;
            let h_end = trim_newline(&self.buf);
            let name_line = String::from_utf8_lossy(&self.buf[1..h_end]).to_string(); // Excluding @
            
            // Parse name/desc
            let mut parts = name_line.splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("").to_string();
            let desc = parts.next().map(|s| s.to_string());
            
            // Seq
            self.buf.clear();
            if self.reader.read_until(b'\n', &mut self.buf)? == 0 { return Ok(None); }
            self.line_number += 1;
            let s_end = trim_newline(&self.buf);
            let mut seq = self.buf[..s_end].to_vec();
            rna_to_dna_inplace(&mut seq);

            // +
            self.buf.clear();
             if self.reader.read_until(b'\n', &mut self.buf)? == 0 { return Ok(None); }
            self.line_number += 1;
             
             // Qual
             self.buf.clear();
             if self.reader.read_until(b'\n', &mut self.buf)? == 0 { return Ok(None); }
            self.line_number += 1;
            let q_end = trim_newline(&self.buf);
            let qual = self.buf[..q_end].to_vec();
            
            return Ok(Some(Record::new(name, desc, seq, Some(qual))));
        }
    }
    
    fn read_next_fasta(&mut self) -> Result<Option<Record>, FastxError> {
        // We need a persistent buffer for the "next header" line in FASTA
        // because we read until '>' to finish current record.
        // Let's assume `self.buf` holds the header line entering this function
        // IF we are in a sequence.
        // But `self.buf` is dynamic.
        
        // Strategy: 
        // 1. If `self.buf` is empty, read header.
        // 2. If `self.buf` is not empty (contains last >Line), use it.
        
        if self.buf.is_empty() {
             if self.reader.read_until(b'\n', &mut self.buf)? == 0 { return Ok(None); }
             self.line_number += 1;
        }
        
        // Now self.buf should have header
        if self.buf[0] != b'>' {
            // maybe we finished? or bad format?
             // try reading one more?
             return Ok(None); 
        }
        
        let h_end = trim_newline(&self.buf);
        let name_line = String::from_utf8_lossy(&self.buf[1..h_end]).to_string();
        let mut parts = name_line.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_string();
        let desc = parts.next().map(|s| s.to_string());
        
        let mut seq = Vec::new();
        loop {
            self.buf.clear();
             let n = self.reader.read_until(b'\n', &mut self.buf)?;
             if n == 0 { 
                 // EOF
                 self.buf.clear(); // Clear to signal EOF for next call
                 break; 
             }
             self.line_number += 1;
             
             if self.buf[0] == b'>' {
                 // Next record found. `self.buf` has >Header.
                 // Break loop, return current record.
                 // Keep `self.buf` populated for next call.
                 break;
             }
             
             let l_end = trim_newline(&self.buf);
             seq.extend_from_slice(&self.buf[..l_end]);
        }
        rna_to_dna_inplace(&mut seq);

        Ok(Some(Record::new(name, desc, seq, None)))
    }

    /// Read sequences until cumulative bases exceed batch_size.
    /// Returns (seqs, is_eof). Caller can call again for the next batch.
    /// Check before the next read,
    /// meaning the last sequence that pushes past the limit IS included.
    pub fn read_batch(&mut self, batch_size: u64) -> Result<(Vec<(String, Vec<u8>)>, bool), FastxError> {
        let mut seqs = Vec::new();
        let mut sum_len: u64 = 0;
        loop {
            if sum_len > batch_size && !seqs.is_empty() {
                return Ok((seqs, false));
            }
            match self.read_next()? {
                Some(r) => {
                    sum_len += r.sequence().len() as u64;
                    seqs.push((r.name().to_string(), r.sequence().to_vec()));
                }
                None => return Ok((seqs, true)),
            }
        }
    }
}

pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Reader<Box<dyn BufRead + Send>>> {
    if path.as_ref().as_os_str() == "-" {
        let stdin = io::stdin();
        return Ok(Reader::new(Box::new(io::BufReader::new(stdin))));
    }

    let file = File::open(path.as_ref()).map_err(|e| io::Error::new(e.kind(), format!("Failed to open FASTA/FASTQ '{}': {}", path.as_ref().display(), e)))?;

    // Detect by content (gzip magic 0x1f 0x8b) rather than trusting the extension.
    // `fill_buf` peeks without consuming, so the magic bytes remain available to
    // whichever decoder we wire up below. An empty file falls through to plain text.
    let mut buf_reader = io::BufReader::new(file);
    let is_gz_content = {
        let peek = buf_reader.fill_buf()?;
        peek.len() >= 2 && peek[0] == 0x1f && peek[1] == 0x8b
    };
    let ext_says_gz = path.as_ref().to_string_lossy().ends_with(".gz");
    if ext_says_gz && !is_gz_content {
        eprintln!("[WARN] '{}' has a .gz extension but is not gzipped; reading as plain text.", path.as_ref().display());
    } else if !ext_says_gz && is_gz_content {
        eprintln!("[WARN] '{}' is gzipped but lacks a .gz extension; decompressing.", path.as_ref().display());
    }

    let reader: Box<dyn BufRead + Send> = if is_gz_content {
        Box::new(io::BufReader::new(MultiGzDecoder::new(buf_reader)))
    } else {
        Box::new(buf_reader)
    };
    Ok(Reader::new(reader))
}

// ------ Memory-Mapped FASTA Reader for Index Building -----
/// Read all sequences from a FASTA file into memory.
/// Returns Vec of (name, sequence) pairs.
/// Used for index building where all sequences are needed at once.
pub fn read_fasta<P: AsRef<Path>>(path: P) -> io::Result<Vec<(String, Vec<u8>)>> {
    let data = std::fs::read(path)?;
    parse_fasta_bytes(&data)
}

/// Parse FASTA from a byte slice (works with mmap or regular buffers)
pub fn parse_fasta_bytes(data: &[u8]) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut sequences = Vec::new();

    if data.is_empty() {
        return Ok(sequences);
    }

    // The streaming reader (Reader::detect_format) requires the first byte to be
    // '>' before accepting input as FASTA. This mmap/batch path used to skip
    // forward to the first '>' *anywhere* in the buffer instead of checking the
    // first byte, which meant any non-FASTA binary file containing a stray 0x3E
    // byte (near-guaranteed for any file of a few hundred bytes or more — e.g. a
    // wrong path, a corrupted download, or an index file with an unrecognized
    // extension that fell through the index/sequence-file dispatch) was silently
    // "parsed" as one bogus sequence with binary garbage as its name, with no
    // error at all. Enforce the same first-byte check here for consistency.
    if data[0] != b'>' {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "not a valid FASTA file: expected '>' as the first byte, found {:?}. \
                 Check that the correct file was provided — this is not a FASTA/FASTQ \
                 file, and if it's meant to be a pre-built index it may have an \
                 unrecognized extension (expected .mmi/.idx/.rmmi).",
                data[0] as char
            ),
        ));
    }

    // Use memchr for fast newline scanning
    let mut pos = 0;
    let len = data.len();

    while pos < len {
        // Skip to next '>'
        while pos < len && data[pos] != b'>' {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        // Parse header line
        let header_start = pos + 1; // Skip '>'

        // Find end of header line using memchr
        let header_end = memchr::memchr(b'\n', &data[header_start..])
            .map(|i| header_start + i)
            .unwrap_or(len);

        // Extract name (first whitespace-delimited token)
        let header_bytes = &data[header_start..header_end];
        let name_end = header_bytes.iter()
            .position(|&b| b == b' ' || b == b'\t')
            .unwrap_or(header_bytes.len());

        let name = String::from_utf8_lossy(&header_bytes[..name_end]).to_string();

        pos = header_end + 1;

        // Collect sequence lines until next '>' or EOF
        // Pre-estimate capacity based on typical sequence density
        let mut seq = Vec::with_capacity(1024);

        while pos < len && data[pos] != b'>' {
            // Find end of line
            let line_end = memchr::memchr(b'\n', &data[pos..])
                .map(|i| pos + i)
                .unwrap_or(len);

            // Copy sequence bytes (skip newlines and handle \r\n)
            let line = &data[pos..line_end];
            let line = if line.ends_with(b"\r") {
                &line[..line.len() - 1]
            } else {
                line
            };

            // Normalize to uppercase while copying
            seq.extend(line.iter().map(|&b| b.to_ascii_uppercase()));

            pos = line_end + 1;
        }

        if !name.is_empty() {
            sequences.push((name, seq));
        }
    }

    Ok(sequences)
}

// ------ Indexed Reader (Keep or minimal refactor) -----
// Needs logic for Indexed Reading.
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_detect_format_fasta() {
        let data = b">seq1\nACGT";
        let cursor = Cursor::new(data);
        let mut reader = Reader::new(cursor);
        let format = reader.detect_format().unwrap();
        assert_eq!(format, Some(RecordType::Fasta));
    }

    #[test]
    fn test_detect_format_fastq() {
        let data = b"@seq1\nACGT\n+\nIIII";
        let cursor = Cursor::new(data);
        let mut reader = Reader::new(cursor);
        let format = reader.detect_format().unwrap();
        assert_eq!(format, Some(RecordType::Fastq));
    }

    #[test]
    fn test_read_fasta_owned() {
        let data = b">seq1\nACGT\n>seq2 desc\nAAAA\nTTTT";
        let cursor = Cursor::new(data);
        let reader = Reader::new(cursor);
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].name(), "seq1");
        assert_eq!(records[0].sequence(), b"ACGT");
        
        assert_eq!(records[1].name(), "seq2");
        assert_eq!(records[1].description(), Some("desc"));
        assert_eq!(records[1].sequence(), b"AAAATTTT");
    }

    #[test]
    fn test_read_fastq_owned() {
        let data = b"@seq1\nACGT\n+\nIIII\n@seq2\nAA\n+\nKK";
        let cursor = Cursor::new(data);
        let reader = Reader::new(cursor);
        let records: Vec<_> = reader.records().map(|r| r.unwrap()).collect();
        
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].name(), "seq1");
        assert_eq!(records[0].sequence(), b"ACGT");
        assert_eq!(records[0].quality().unwrap(), b"IIII");
        
        assert_eq!(records[1].name(), "seq2");
        assert_eq!(records[1].sequence(), b"AA");
        assert_eq!(records[1].quality().unwrap(), b"KK");
    }

    #[test]
    fn test_for_each_record_fasta() {
        let data = b">seq1\nACGT";
        let cursor = Cursor::new(data);
        let mut reader = Reader::new(cursor);
        let mut count = 0;
        reader.for_each_record(|r| {
            assert_eq!(r.head, b"seq1");
            assert_eq!(r.seq, b"ACGT");
            count += 1;
        }).unwrap();
        assert_eq!(count, 1);
    }

     #[test]
    fn test_for_each_record_fastq() {
        let data = b"@s1\nA\n+\nI";
        let cursor = Cursor::new(data);
        let mut reader = Reader::new(cursor);
        let mut count = 0;
        let _ = reader.for_each_record(|r| {
            assert_eq!(r.head, b"s1");
            assert_eq!(r.seq, b"A");
            assert_eq!(r.qual.unwrap(), b"I");
            count += 1;
        });
        assert_eq!(count, 1);
    }

    /// Empty input (e.g. an empty stdin query when only one positional is
    /// given, as in `-x ava-ont reads.fq`) must yield ZERO records and
    /// terminate cleanly — no error
    #[test]
    fn test_empty_input_is_clean_eof() {
        let mut it = Reader::new(Cursor::new(b"" as &[u8])).records();
        assert!(it.next().is_none(), "empty input must yield no records");
        // And it must STAY terminated (fused), not resurrect an error.
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    /// Non-empty but unrecognized input reports the error exactly ONCE
    #[test]
    fn test_bad_input_errors_once_then_fused() {
        let mut it = Reader::new(Cursor::new(b"not a fastx file\nsecond line\n" as &[u8])).records();
        assert!(matches!(it.next(), Some(Err(_))), "first read should surface the error");
        for _ in 0..1000 {
            assert!(it.next().is_none(), "iterator must not re-emit the error (no spin/spam)");
        }
    }

    /// Simulate a log-and-continue consumer (like the CLI read loops) over bad
    /// input: it must terminate with a bounded number of warnings, not forever
    #[test]
    fn test_consumer_loop_terminates_on_bad_input() {
        let it = Reader::new(Cursor::new(b"garbage not fastx\n" as &[u8])).records();
        let mut warnings = 0usize;
        let mut records = 0usize;
        for item in it {
            match item {
                Ok(_) => records += 1,
                Err(_) => { warnings += 1; continue; } // mirrors the CLI's warn+continue
            }
        }
        assert_eq!(records, 0);
        assert!(warnings <= 1, "bad input should warn at most once, got {warnings}");
    }
}
