
use crate::align::sketch::sketch_sequence;
use crate::align::sort::radix_sort_pair;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::io::{self, BufWriter, BufReader, Read, Write, Seek, SeekFrom};

/// Read a Vec<u32> from a binary stream (little-endian, safe).
fn read_u32_vec<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    reader.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Read a Vec<u64> from a binary stream (little-endian, safe).
fn read_u64_vec<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<u64>> {
    let mut buf = vec![0u8; n * 8];
    reader.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
use indicatif::ProgressIterator;
#[cfg(all(not(target_arch = "wasm32"), feature = "parallel"))]
use indicatif::ParallelProgressIterator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSequence {
    pub name: String,
    pub len: usize,
    pub offset: u64,
    #[serde(default)]
    pub is_alt: bool,
}

/// Magic bytes for multi-part index format
const RMMI_MAGIC: &[u8; 4] = b"RMMI";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub kmer_size: usize,
    pub window_size: usize,
    pub homopolymer_compressed: bool,
    pub index: usize, // part number (0-based)
    pub seqs: Vec<TargetSequence>,
    pub entries: Vec<(u64, u64)>, // Sorted by hash
    /// Packed 4-bit reference sequences (8 bases per u32, minimap2 encoding).
    /// Kept at runtime for on-demand per-region nt4 extraction (~375 MB for GRCh38).
    #[serde(default)]
    pub packed_seqs: Vec<u32>,
    #[serde(skip)]
    bucket_offsets: Vec<u32>, // bucket_offsets[i] = start index for bucket i; len = n_buckets + 1
    #[serde(skip)]
    bucket_shift: u32, // hash >> bucket_shift = bucket index
}

impl Index {
    /// Strip target sequences from index (for --idx-no-seq).
    /// Keeps all metadata (name, len, offset) but clears all sequence data.
    pub fn strip_sequences(&mut self) {
        self.packed_seqs = Vec::new();
    }

    /// Returns true if this index has stored sequences.
    pub fn has_sequences(&self) -> bool {
        !self.packed_seqs.is_empty()
    }

    /// nt4 value to uppercase ASCII base.
    pub const NT4_TO_ASCII: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];

    /// Get a single base at position `pos` in sequence `rid` as nt4 (0=A,1=C,2=G,3=T,4=N).
    #[inline]
    pub fn get_nt4(&self, rid: usize, pos: usize) -> u8 {
        let gpos = self.seqs[rid].offset as usize + pos;
        ((self.packed_seqs[gpos >> 3] >> (((gpos & 7) << 2) as u32)) & 0xf).min(4) as u8
    }

    /// Extract a region [start..end) from sequence `rid` as nt4-encoded bytes (allocating).
    pub fn get_region_nt4(&self, rid: usize, start: usize, end: usize) -> Vec<u8> {
        let mut out = vec![0u8; end - start];
        self.extract_nt4_into(rid, start, end, &mut out);
        out
    }

    /// Extract a region [start..end) from sequence `rid` as nt4 bytes into caller buffer.
    #[inline]
    pub fn extract_nt4_into(&self, rid: usize, start: usize, end: usize, buf: &mut [u8]) {
        let gpos_start = self.seqs[rid].offset as usize + start;
        Self::unpack_nt4_into(&self.packed_seqs, gpos_start, &mut buf[..end - start]);
    }

    /// Fast bulk unpack from packed 4-bit to nt4 bytes (0=A,1=C,2=G,3=T,4=N).
    /// Processes 8 bases per u32 word for aligned portions.
    fn unpack_nt4_into(packed: &[u32], gpos_start: usize, out: &mut [u8]) {
        let len = out.len();
        if len == 0 { return; }
        let mut i = 0;
        let mut gpos = gpos_start;

        // Handle unaligned prefix
        while i < len && (gpos & 7) != 0 {
            out[i] = (((packed[gpos >> 3] >> (((gpos & 7) << 2) as u32)) & 0xf) as u8).min(4);
            i += 1;
            gpos += 1;
        }

        // Fast path: extract 8 bases per u32 word
        let word_start = gpos >> 3;
        let full_words = (len - i) >> 3;
        for w in 0..full_words {
            let word = packed[word_start + w];
            let base = i + (w << 3);
            out[base]     = ((word & 0xf) as u8).min(4);
            out[base + 1] = (((word >>  4) & 0xf) as u8).min(4);
            out[base + 2] = (((word >>  8) & 0xf) as u8).min(4);
            out[base + 3] = (((word >> 12) & 0xf) as u8).min(4);
            out[base + 4] = (((word >> 16) & 0xf) as u8).min(4);
            out[base + 5] = (((word >> 20) & 0xf) as u8).min(4);
            out[base + 6] = (((word >> 24) & 0xf) as u8).min(4);
            out[base + 7] = (((word >> 28) & 0xf) as u8).min(4);
        }
        i += full_words << 3;
        gpos = gpos_start + i;

        // Handle unaligned suffix
        while i < len {
            out[i] = (((packed[gpos >> 3] >> (((gpos & 7) << 2) as u32)) & 0xf) as u8).min(4);
            i += 1;
            gpos += 1;
        }
    }

    /// Save a single index to file (backward-compatible single-part format).
    pub fn save(&self, path: &str) -> io::Result<()> {
        let f = File::create(path).map_err(|e| io::Error::new(e.kind(), format!("Failed to create index '{}': {}", path, e)))?;
        let mut writer = BufWriter::new(f);
        self.save_part(&mut writer)
    }

    /// Save one part with RMMI magic prefix.
    pub fn save_part<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(RMMI_MAGIC)?;
        bincode::serialize_into(writer, self).map_err(io::Error::other)
    }

    /// Load a single-part index from file. Handles both old (no magic) and new (RMMI) formats.
    pub fn load(path: &str) -> io::Result<Self> {
        let f = File::open(path).map_err(|e| io::Error::new(e.kind(), format!("Failed to open index '{}': {}", path, e)))?;
        let mut reader = BufReader::new(f);
        match Self::load_part(&mut reader)? {
            Some(idx) => Ok(idx),
            None => Err(io::Error::new(io::ErrorKind::InvalidData, "Empty index file")),
        }
    }

    /// .mmi format magic: "MMI..02"
    const MINIMAP2_INDEX_MAGIC: &'static [u8; 4] = b"MMI\x02";

    /// Load the next index part from a reader. Returns None on EOF.
    /// Detects RMMI (rammap), MMI\2 (minimap2), or old bincode format.
    pub fn load_part<R: Read + Seek>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut magic = [0u8; 4];
        match reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let mut idx: Self = if &magic == RMMI_MAGIC {
            // rammap format: magic already consumed, deserialize the rest
            bincode::deserialize_from(reader)
                .map_err(io::Error::other)?
        } else if &magic == Self::MINIMAP2_INDEX_MAGIC {
            // .mmi format: parse binary layout
            return Self::load_minimap2(reader);
        } else {
            // Old format (no magic): seek back and deserialize from start
            reader.seek(SeekFrom::Current(-4))?;
            bincode::deserialize_from(reader)
                .map_err(io::Error::other)?
        };
        idx.build_hash_lut();
        Ok(Some(idx))
    }

    /// Load one index part from a .mmi file.
    /// The 4-byte magic has already been consumed by load_part.
    fn load_minimap2<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        use std::time::Instant;
        let t_total = Instant::now();

        // MINIMAP2_FLAG_HPC = 0x1, MINIMAP2_FLAG_NO_SEQ = 0x2
        const MINIMAP2_FLAG_HPC: u32 = 0x1;
        const MINIMAP2_FLAG_NO_SEQ: u32 = 0x2;

        // Read header: [w, k, b, n_seq, flag] as 5 × u32
        let hdr_vec = read_u32_vec(reader, 5)?;
        let hdr = [hdr_vec[0], hdr_vec[1], hdr_vec[2], hdr_vec[3], hdr_vec[4]];
        let w = hdr[0] as usize;
        let k = hdr[1] as usize;
        let b = hdr[2] as usize;
        let n_seq = hdr[3] as usize;
        let flag = hdr[4];
        let is_hpc = (flag & MINIMAP2_FLAG_HPC) != 0;
        let no_seq = (flag & MINIMAP2_FLAG_NO_SEQ) != 0;

        // Read sequence metadata
        let mut seqs = Vec::with_capacity(n_seq);
        let mut sum_len: u64 = 0;
        for _ in 0..n_seq {
            let mut name_len = [0u8; 1];
            reader.read_exact(&mut name_len)?;
            let name = if name_len[0] > 0 {
                let mut name_buf = vec![0u8; name_len[0] as usize];
                reader.read_exact(&mut name_buf)?;
                String::from_utf8_lossy(&name_buf).to_string()
            } else {
                String::new()
            };
            let mut seq_len = [0u8; 4];
            reader.read_exact(&mut seq_len)?;
            let len = u32::from_le_bytes(seq_len) as usize;
            seqs.push(TargetSequence {
                name,
                len,
                offset: sum_len,
                is_alt: false,
            });
            sum_len += len as u64;
        }

        // Read per-bucket hash tables and build flat entries
        let t_hash = Instant::now();
        let n_buckets = 1usize << b;
        // Pre-allocate: estimate ~1 minimizer per w bases
        let estimated_entries = (sum_len as usize) / w.max(1);
        let mut entries: Vec<(u64, u64)> = Vec::with_capacity(estimated_entries);

        for bucket_idx in 0..n_buckets {
            // Read n (i32) — size of positions array
            let mut n_buf = [0u8; 4];
            reader.read_exact(&mut n_buf)?;
            let n = i32::from_le_bytes(n_buf) as usize;

            // Read p[0..n] (u64 array) — multi-occurrence positions
            let p = if n > 0 { read_u64_vec(reader, n)? } else { Vec::new() };

            // Read hash_size (u32)
            let mut size_buf = [0u8; 4];
            reader.read_exact(&mut size_buf)?;
            let hash_size = u32::from_le_bytes(size_buf) as usize;

            if hash_size == 0 { continue; }

            // Bulk read all hash entries for this bucket
            let hash_buf = read_u64_vec(reader, hash_size * 2)?;

            for i in 0..hash_size {
                let key = hash_buf[i * 2];
                let value = hash_buf[i * 2 + 1];

                // Reconstruct full minimizer hash:
                // key = (minimizer >> b) << 1, with LSB = singleton flag
                // full_hash = (key >> 1) << b | bucket_idx
                let full_hash = (key >> 1) << b | bucket_idx as u64;

                if key & 1 != 0 {
                    // Singleton: value is the direct position
                    entries.push((full_hash, value));
                } else {
                    // Multi-occurrence: value = (offset_in_p << 32) | count
                    let count = (value & 0xFFFFFFFF) as usize;
                    let start = (value >> 32) as usize;
                    for j in 0..count {
                        entries.push((full_hash, p[start + j]));
                    }
                }
            }
        }
        let hash_elapsed = t_hash.elapsed();

        // Sort entries by (hash, position) — matches our internal format
        let t_sort = Instant::now();
        #[cfg(feature = "parallel")]
        entries.par_sort_unstable();
        #[cfg(not(feature = "parallel"))]
        radix_sort_pair(&mut entries);
        let sort_elapsed = t_sort.elapsed();

        // Read packed 4-bit sequences if present — keep in packed format
        let packed_seqs = if !no_seq {
            let n_u32 = sum_len.div_ceil(8) as usize;
            if n_u32 > 0 { read_u32_vec(reader, n_u32)? } else { Vec::new() }
        } else {
            Vec::new()
        };

        let n_entries = entries.len();
        let mut idx = Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs,
            entries,
            packed_seqs,
            bucket_offsets: Vec::new(),
            bucket_shift: 0,
        };
        idx.build_hash_lut();

        eprintln!("[*] Index loaded: {}M entries, {}M bases in {:.1}s (hash {:.1}s, sort {:.1}s)",
            n_entries / 1_000_000, sum_len / 1_000_000,
            t_total.elapsed().as_secs_f64(),
            hash_elapsed.as_secs_f64(), sort_elapsed.as_secs_f64());
        Ok(Some(idx))
    }

    /// Create a header-only index with sequence metadata but no minimizer data.
    /// Used during merge phase where we only need names/lengths for output formatting.
    pub fn header_only(k: usize, w: usize, is_hpc: bool, seqs: Vec<TargetSequence>) -> Self {
        Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs,
            entries: Vec::new(),
            packed_seqs: Vec::new(),
            bucket_offsets: Vec::new(),
            bucket_shift: 0,
        }
    }

    pub fn new(w: usize, k: usize, is_hpc: bool) -> Self {
        Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs: Vec::new(),
            entries: Vec::new(),
            packed_seqs: Vec::new(),
            bucket_offsets: Vec::new(),
            bucket_shift: 0,
        }
    }

    /// Build an index from target sequences.
    ///
    /// Note: `max_occ` is a hard cap to filter extremely repetitive minimizers during index
    /// construction. The `mid_occ` threshold (calculated via `cal_mid_occ`) should be applied
    /// at query time, not during index building.
    pub fn build(
        seqs: Vec<(String, Vec<u8>)>,
        w: usize,
        k: usize,
        is_hpc: bool,
        max_occ: usize,
    ) -> Self {
        let mut idx = Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs: Vec::new(),
            entries: Vec::new(),
            packed_seqs: Vec::new(),
            bucket_offsets: Vec::new(),
            bucket_shift: 0,
        };

        let mut offset = 0usize;
        let mut ascii_seqs: Vec<Vec<u8>> = Vec::new();

        // Phase 1: Storage and Metadata (Sequential)
        for (name, seq_bytes) in seqs {
            let len = seq_bytes.len();
            let mut normalized = seq_bytes;
            for b in normalized.iter_mut() {
                *b = b.to_ascii_uppercase();
            }

            idx.seqs.push(TargetSequence {
                name,
                len,
                offset: offset as u64,
                is_alt: false,
            });
            ascii_seqs.push(normalized);
            offset += len;
        }

        // Pack ASCII sequences into global 4-bit packed array
        {
            #[inline]
            fn ascii_to_nt4(b: u8) -> u32 {
                match b { b'A' | b'a' => 0, b'C' | b'c' => 1, b'G' | b'g' => 2, b'T' | b't' => 3, _ => 4 }
            }
            let n_u32 = offset.div_ceil(8);
            let mut packed = vec![0u32; n_u32];
            for (seq_info, ascii) in idx.seqs.iter().zip(ascii_seqs.iter()) {
                let goff = seq_info.offset as usize;
                for (i, &b) in ascii.iter().enumerate() {
                    let gpos = goff + i;
                    packed[gpos >> 3] |= ascii_to_nt4(b) << (((gpos & 7) << 2) as u32);
                }
            }
            idx.packed_seqs = packed;
        }

        // Phase 2: Parallel Sketching
        // Helper
        fn sketch_seq_helper(rid: usize, ascii: &[u8], len: usize, w: usize, k: usize, is_hpc: bool) -> Vec<(u64, u64)> {
            let mut minimizers = Vec::new();
            sketch_sequence(ascii, len, w, k, rid, is_hpc, &mut minimizers);
            minimizers.into_iter().map(|m| (m.x >> 8, m.y)).collect()
        }

        // Only show progress bar for large inputs (> 10 sequences) to reduce overhead
        let use_progress = idx.seqs.len() > 10;

        #[cfg(all(not(target_arch = "wasm32"), feature = "parallel"))]
        let entries_per_seq: Vec<Vec<(u64, u64)>> = {
            if use_progress {
                let style = ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("##-");
                let pb = ProgressBar::new(idx.seqs.len() as u64);
                pb.set_style(style);
                pb.set_message("Sketching sequences");
                idx.seqs.par_iter().zip(ascii_seqs.par_iter()).enumerate().progress_with(pb).map(|(rid, (t_seq, ascii))| {
                    sketch_seq_helper(rid, ascii, t_seq.len, w, k, is_hpc)
                }).collect()
            } else {
                idx.seqs.par_iter().zip(ascii_seqs.par_iter()).enumerate().map(|(rid, (t_seq, ascii))| {
                    sketch_seq_helper(rid, ascii, t_seq.len, w, k, is_hpc)
                }).collect()
            }
        };

        #[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
        let entries_per_seq: Vec<Vec<(u64, u64)>> = {
            if use_progress {
                let style = ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("##-");
                let pb = ProgressBar::new(idx.seqs.len() as u64);
                pb.set_style(style);
                pb.set_message("Sketching sequences");
                idx.seqs.iter().zip(ascii_seqs.iter()).enumerate().progress_with(pb).map(|(rid, (t_seq, ascii))| {
                     sketch_seq_helper(rid, ascii, t_seq.len, w, k, is_hpc)
                }).collect()
            } else {
                idx.seqs.iter().zip(ascii_seqs.iter()).enumerate().map(|(rid, (t_seq, ascii))| {
                     sketch_seq_helper(rid, ascii, t_seq.len, w, k, is_hpc)
                }).collect()
            }
        };

        #[cfg(target_arch = "wasm32")]
        let entries_per_seq: Vec<Vec<(u64, u64)>> = {
            idx.seqs.iter().zip(ascii_seqs.iter()).enumerate().map(|(rid, (t_seq, ascii))| {
                sketch_seq_helper(rid, ascii, t_seq.len, w, k, is_hpc)
            }).collect()
        };

        // Phase 3: Flatten and Sort
        let total_entries: usize = entries_per_seq.iter().map(|v| v.len()).sum();
        let mut temp_entries: Vec<(u64, u64)> = Vec::with_capacity(total_entries);
        for mut v in entries_per_seq {
            temp_entries.append(&mut v);
        }

        // Sort by (hash, position)
        radix_sort_pair(&mut temp_entries);

        // Filter extremely repetitive minimizers (hard cap)
        // Note: mid_occ filtering happens at query time, not here
        if !temp_entries.is_empty() && max_occ < usize::MAX {
            let mut keep_idx = 0;
            let mut i = 0;
            while i < temp_entries.len() {
                let start = i;
                let h = temp_entries[i].0;
                while i < temp_entries.len() && temp_entries[i].0 == h {
                    i += 1;
                }
                let count = i - start;
                if count <= max_occ {
                    // Keep this block
                    if keep_idx != start {
                        for k in start..i {
                            temp_entries[keep_idx] = temp_entries[k];
                            keep_idx += 1;
                        }
                    } else {
                        keep_idx = i;
                    }
                }
            }
            temp_entries.truncate(keep_idx);
        }
        
        idx.entries = temp_entries;
        idx.build_hash_lut();

        idx
    }

    /// Build bucket offset table from sorted entries for O(1) bucket + binary search lookup.
    ///
    /// Uses top bits of hash for bucket selection. Since entries are sorted by hash,
    /// all entries in bucket i come before bucket i+1, enabling a simple offset table.
    /// Build: O(n) sequential scan. Lookup: O(1) + O(log bucket_size).
    fn build_hash_lut(&mut self) {
        if self.entries.is_empty() {
            self.bucket_offsets = vec![0; 2];
            self.bucket_shift = 0;
            return;
        }

        // Determine hash bit width from k, then pick bucket count
        let hash_bits = (2 * self.kmer_size) as u32;
        let bucket_bits = 18u32.min(hash_bits); // 2^18 = 256K buckets
        let shift = hash_bits.saturating_sub(bucket_bits);
        let n_buckets = 1usize << bucket_bits;

        let mut offsets = vec![0u32; n_buckets + 1];

        // Count entries per bucket
        for &(hash, _) in &self.entries {
            let bucket = (hash >> shift) as usize;
            // Safety: bucket < n_buckets since hash < 2^hash_bits and shift = hash_bits - bucket_bits
            offsets[bucket] += 1;
        }

        // Prefix sum: offsets[i] = start of bucket i
        let mut acc = 0u32;
        for offset in &mut offsets[..n_buckets] {
            let count = *offset;
            *offset = acc;
            acc += count;
        }
        offsets[n_buckets] = acc;

        self.bucket_offsets = offsets;
        self.bucket_shift = shift;
    }

    pub fn get(&self, hash: u64) -> Option<&[(u64, u64)]> {
        if self.bucket_offsets.len() < 2 { return None; }

        let bucket = (hash >> self.bucket_shift) as usize;
        if bucket + 1 >= self.bucket_offsets.len() { return None; }

        let start = self.bucket_offsets[bucket] as usize;
        let end = self.bucket_offsets[bucket + 1] as usize;
        if start >= end { return None; }

        let slice = &self.entries[start..end];

        // Binary search within bucket for first entry with this hash
        let first = slice.partition_point(|e| e.0 < hash);
        if first >= slice.len() || slice[first].0 != hash { return None; }

        // Find end of group (entries with same hash are contiguous)
        let last = first + slice[first..].partition_point(|e| e.0 == hash);

        Some(&slice[first..last])
    }

    /// Calculate mid_occ threshold to filter top `frac` fraction of repetitive minimizers.
    /// Compute mid-occurrence threshold from seed frequency distribution.
    pub fn cal_mid_occ(&self, frac: f32, min_mid_occ: i32, max_mid_occ: i32) -> usize {
        let min_mid = min_mid_occ.max(1) as usize;
        if frac <= 0.0 { return usize::MAX; }
        if self.entries.is_empty() { return min_mid; }

        // Count occurrences per unique hash by scanning sorted entries
        let mut counts: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            let h = self.entries[i].0;
            let start = i;
            while i < self.entries.len() && self.entries[i].0 == h { i += 1; }
            counts.push((i - start) as u32);
        }

        let n = counts.len();
        if n == 0 { return min_mid; }

        counts.sort_unstable();

        let k = ((1.0f64 - frac as f64) * n as f64) as usize;
        let k = k.min(n - 1);

        let mut threshold = counts[k] as usize + 1;

        // Clamp to [min_mid_occ, max_mid_occ] (matching mm_mapopt_update)
        if threshold < min_mid { threshold = min_mid; }
        if max_mid_occ > min_mid_occ && threshold > max_mid_occ as usize {
            threshold = max_mid_occ as usize;
        }

        threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idx_build() {
        let seq = "GTTGATAATCACTCACTGAGTGACATCCAAATCATGGCGTCCCAAGGCACCAAACGGTCTTATGAACAGATGGAAACTGATGGAGATCGCCAGAATGCAACTGAGATTAGGGCATCCGTCGGAAAGATGATTGATGGAATTGGGAGATTCTACATCCAAATGTGCACTGAACTTAAACTCAGTGATCATGAAGGACGGTTGATCCAAAACAGCTTGACAATAGAGAAAATGGTGCTTTCTGCTTTTGATGAAAGAAGGAATAAATACCTGGAAGAACACCCCAGCGCGGGGAAAGATCCCAAGAAAACCGGGGGGCCCATATACAGGAGAGTCGATGGGAAATGGATGAGAGAACTCGTCCTTTATGACAAAGAAGAAATAAGGCGAATCTGGCGCCAAGCCAACAATGGTGAGGATGCTACATCTGGTCTAACCCACCTAATGATTTGGCATTCCAATTTGAATGATGCAACATACCAAAGGACAAGAGCTCTTGTTCGGACTGGAATGGACCCCAGAATGTGCTCTCTGATGCAGGGCTCGACTCTCCCTAGAAGGTCCGGAGCTGCCGGTGCTGCAGTCAAAGGAATCGGAACAATGGTGATGGAACTGATCAGAATGATCAAACGGGGGATCAACGATCGAAATTTTTGGAGAGGTGAGAATGGGCGGAAAACAAGAAGTGCTTATGAGAGAATGTGCAACATTCTCAAAGGAAAATTTCAAACAGCTGCACAAAAAGCAATGGTGGATCAAGTTAGAGAAAGCCGGAATCCAGGAAACGCTGAGATCGAAGATCTCATATTTTTAGCAAGATCTGCACTGATATTGAGAGGATCAGTTGCTCACAAATCTTGCCTACCTGCCTGTGCATATGGACCTGCAGTATCCAGTGGTTATGACTTTGAAAAAGAGGGATATTCCTTGGTGGGAATAGACCCTTTCAAACTACTTCAAAATAGCCAAATATACAGCTTAATCAGACCTAATGAGAATCCAGCACACAAGAGTCAGCTGGTGTGGATGGCATGTCATTCTGCTGCATTTGAAGATTTAAGATTGTTAAGCTTCATCAGAGGAACAAAAGTATCTCCTCGGGGGAAACTGTCAACTAGAGGAGTACAAATTGCTTCAAATGAGAACATGGATAATATGGGATCAAGCACTCTTGAACTGAGAAGCGGGTACTGGGCCATAAGGACCAGGAGTGGAGGAAACACTAATCAGCAGAGGGCCTCCGCAGGCCAAACCAGTGTGCAACCAACGTTTTCTGTACAAAGAAACCTCCCATTTGAAAAGTCAACCATCATGGCAGCATTCACTGGAAATACGGAAGGAAGAACTTCAGACATGAGGGCAGAAATTATAAGGATGATGGAAGGTGCAAAACCAGAAGAAGTGTCATTCCGGGGGAGGGGAGTTTTCGAGCTCTCTGACGAGAAGGCAGCGAACCCGATCGTGCCCTCTTTTGATATGAGTAACGAAGGATCTTATTTCTTCGGAGACAATGCAGAAGAATACGACAATTAAGAAAAAANNNN";
        let idx = Index::build(vec![("test_seq".to_string(), seq.as_bytes().to_vec())], 10, 15, false, 50000);
        
        let hash = 86616326159 >> 8;
        let positions = idx.get(hash);
        assert!(positions.is_some());
        assert_eq!(positions.unwrap()[0].1, 624);
        assert_eq!(idx.seqs.len(), 1);
    }
    #[test]
    fn test_cal_mid_occ() {
        let mut seqs = Vec::new();
        // Create 100 A's -> 1 k-mer with 100 occurrences (roughly)
        // Create 100 distinct sequences "CG...0", "CG...1" -> 100 singletons
        
        let mut t_seq = String::new();
        for _ in 0..100 { t_seq.push('A'); } 
        seqs.push(("polyA".to_string(), t_seq.into_bytes()));
        
        for i in 0..100 {
            seqs.push((format!("uq{}", i), format!("CGT{}AGCT", i).into_bytes()));
        }

        // w=10, k=5
        let idx = Index::build(seqs, 10, 5, false, 50000);

        // Test cal_mid_occ
        // frac=0.0 -> returns usize::MAX (no filtering)
        let m0 = idx.cal_mid_occ(0.0, 10, 1000000);
        assert_eq!(m0, usize::MAX, "frac=0.0 should return MAX");

        // frac=1.0 -> finds count at position 0 (smallest) + 1
        // Should return at least 10 (min clamp)
        let m1 = idx.cal_mid_occ(1.0, 10, 1000000);
        assert!(m1 >= 10, "frac=1.0 should be at least 10, got {}", m1);

        // frac=0.5 -> finds count at 50th percentile
        let m05 = idx.cal_mid_occ(0.5, 10, 1000000);
        assert!(m05 >= 10, "frac=0.5 should be at least 10, got {}", m05); 
    }
}
