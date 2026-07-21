
use crate::align::sketch::{sketch_sequence, Minimizer};
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

// ─── Seed lookup trait + backends ───

/// Trait for seed lookup backends.
pub trait SeedLookup {
    fn get(&self, hash: u64) -> Option<&[u64]>;
    fn get_range(&self, hash: u64) -> Option<(u32, u32)>;
    fn get_by_range(&self, range: (u32, u32)) -> &[u64];
    /// Iterator over occurrence counts for each unique hash (for cal_mid_occ).
    fn occurrence_counts(&self) -> Box<dyn Iterator<Item = u32> + '_>;
    fn is_empty(&self) -> bool;
}

/// Seed lookup backend. Currently uses BucketHash (per-bucket open-addressing
/// hash tables over a shared flat positions array). The enum + trait are kept
/// for future extensibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LookupBackend {
    BucketHash(super::index_bucket::BucketHashLookup),
}

impl SeedLookup for LookupBackend {
    #[inline]
    fn get(&self, hash: u64) -> Option<&[u64]> {
        match self { LookupBackend::BucketHash(b) => b.get(hash) }
    }
    #[inline]
    fn get_range(&self, hash: u64) -> Option<(u32, u32)> {
        match self { LookupBackend::BucketHash(b) => b.get_range(hash) }
    }
    #[inline]
    fn get_by_range(&self, range: (u32, u32)) -> &[u64] {
        match self { LookupBackend::BucketHash(b) => b.get_by_range(range) }
    }
    fn occurrence_counts(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        match self { LookupBackend::BucketHash(b) => b.occurrence_counts() }
    }
    fn is_empty(&self) -> bool {
        match self { LookupBackend::BucketHash(b) => b.is_empty() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub kmer_size: usize,
    pub window_size: usize,
    pub homopolymer_compressed: bool,
    pub index: usize, // part number (0-based)
    pub seqs: Vec<TargetSequence>,
    /// Seed lookup backend.
    backend: LookupBackend,
    /// Packed 4-bit reference sequences (8 bases per u32; nibbles hold the nt4
    /// codes 0=A, 1=C, 2=G, 3=T, 4=N, low nibble first).
    /// Kept at runtime for on-demand per-region nt4 extraction (~375 MB for GRCh38).
    #[serde(default)]
    pub packed_seqs: Vec<u32>,
    /// Derived (not serialized): dense lexicographic ranks of the target names,
    /// computed on first use for the ava self-skip fast path. See [`name_rank`].
    #[serde(skip)]
    name_rank: std::sync::OnceLock<NameRankData>,
}

/// Dense lexicographic ranking of target-sequence names, used to turn the
/// per-hit `qname.cmp(seq.name)` string comparison (which dominates the ava
/// seeding phase) into a fast integer comparison. Names that are byte-equal
/// share a rank — identical ordering to `str::cmp`.
#[derive(Debug, Clone)]
pub struct NameRankData {
    /// Dense rank per rid.
    pub rank: Vec<u32>,
    /// rids sorted by name, for resolving a query name's rank via binary search.
    order: Vec<u32>,
}

impl NameRankData {
    /// Lexicographic rank of `qname`, iff it exactly matches an indexed name
    /// (always true for ava self-mapping). Returns `None` otherwise so callers
    /// can fall back to a direct string comparison.
    #[inline]
    pub fn rank_of(&self, seqs: &[TargetSequence], qname: &str) -> Option<u32> {
        self.order
            .binary_search_by(|&idx| seqs[idx as usize].name.as_str().cmp(qname))
            .ok()
            .map(|pos| self.rank[self.order[pos] as usize])
    }
}

/// nt4 encoding lookup: A→0, C→1, G→2, T→3, anything else (including N) →4.
static NT4_TABLE: [u32; 256] = {
    let mut t = [4u32; 256];
    t[b'A' as usize] = 0; t[b'a' as usize] = 0;
    t[b'C' as usize] = 1; t[b'c' as usize] = 1;
    t[b'G' as usize] = 2; t[b'g' as usize] = 2;
    t[b'T' as usize] = 3; t[b't' as usize] = 3;
    t
};

/// Incremental index builder. Pack and sketch each sequence as it arrives,
/// drop the bytes after, then call [`finish`](Self::finish) once all
/// sequences have been added.
///
/// Lets WASM consumers stream a multi-GB reference into the index without
/// ever holding the full ASCII bytes alongside the packed representation —
/// peak memory stays ~(packed + buckets) rather than ~(raw + packed + buckets).
pub struct IndexBuilder {
    idx: Index,
    w: usize,
    k: usize,
    is_hpc: bool,
    max_occ: usize,
    bucket_bits: u32,
    bucket_mask: u64,
    packed: Vec<u32>,
    buckets: Vec<Vec<(u64, u64)>>,
    offset: usize,
    /// Reusable per-call minimizer scratch. Allocated once on the first
    /// large sequence; cleared (not freed) between calls so we don't pay the
    /// alloc/free round-trip on every chromosome — important on WASM where
    /// the heap can fragment.
    minis_scratch: Vec<Minimizer>,
}

impl IndexBuilder {
    pub fn new(w: usize, k: usize, is_hpc: bool, max_occ: usize) -> Self {
        let bucket_bits = 10u32.min((2 * k) as u32);
        let n_buckets = 1usize << bucket_bits;
        Self {
            idx: Index {
                kmer_size: k,
                window_size: w,
                homopolymer_compressed: is_hpc,
                index: 0,
                seqs: Vec::new(),
                backend: LookupBackend::BucketHash(super::index_bucket::BucketHashLookup::empty()),
                packed_seqs: Vec::new(),
                name_rank: Default::default(),
            },
            w, k, is_hpc, max_occ,
            bucket_bits,
            bucket_mask: (n_buckets - 1) as u64,
            packed: Vec::new(),
            buckets: vec![Vec::new(); n_buckets],
            offset: 0,
            minis_scratch: Vec::new(),
        }
    }

    /// Number of sequences added so far.
    pub fn num_sequences(&self) -> usize { self.idx.seqs.len() }

    /// Total reference bases packed so far.
    pub fn total_bases(&self) -> usize { self.offset }

    /// Reserve packed-buffer capacity for an expected total reference base count.
    /// Optional but recommended for multi-GB streaming builds — avoids the
    /// repeated doubling reallocation, which can briefly use 3× the final
    /// packed-buffer size on the last grow. The estimate is an upper bound;
    /// final size may be smaller.
    pub fn reserve_bases(&mut self, expected_total: usize) {
        let n_u32 = expected_total.div_ceil(8);
        if n_u32 > self.packed.capacity() {
            self.packed.reserve(n_u32 - self.packed.len());
        }
    }

    /// Add one sequence: pack it into the global 4-bit buffer, sketch its
    /// minimizers, distribute to buckets. The sequence bytes are dropped on
    /// return.
    pub fn add_sequence(&mut self, name: String, mut seq: Vec<u8>) {
        if self.is_hpc {
            for b in seq.iter_mut() { *b = b.to_ascii_uppercase(); }
        }

        let rid = self.idx.seqs.len();
        let len = seq.len();
        let goff = self.offset;
        self.idx.seqs.push(TargetSequence {
            name,
            len,
            offset: goff as u64,
            is_alt: false,
        });

        // Grow the packed buffer for the new bases (8 nibbles per u32).
        let new_offset = goff + len;
        let n_u32_needed = new_offset.div_ceil(8);
        if n_u32_needed > self.packed.len() {
            self.packed.resize(n_u32_needed, 0);
        }

        // Pack into the global buffer. Adjacent sequences may share the
        // boundary u32 at byte offsets not divisible by 8, so this must
        // remain sequential.
        for (j, &b) in seq.iter().enumerate() {
            let gpos = goff + j;
            self.packed[gpos >> 3] |= NT4_TABLE[b as usize] << (((gpos & 7) << 2) as u32);
        }

        // Sketch into the persistent scratch Vec (cleared by sketch_sequence),
        // then distribute. Keeping the Vec alive across calls avoids a
        // ~13M-entry alloc/free per chromosome.
        sketch_sequence(&seq, len, self.w, self.k, rid, self.is_hpc, &mut self.minis_scratch);
        for &m in &self.minis_scratch {
            let hash = m.x >> 8;
            self.buckets[(hash & self.bucket_mask) as usize].push((hash, m.y));
        }

        self.offset = new_offset;
        // seq (Vec<u8>) is dropped here, freeing the raw bytes before the
        // next sequence arrives.
    }

    /// Finalize: sort buckets, build the per-bucket hash backend, attach to
    /// the index, and return it.
    pub fn finish(mut self) -> Index {
        #[cfg(feature = "parallel")]
        self.buckets.par_iter_mut().for_each(|b| {
            if !b.is_empty() { b.sort_unstable(); }
        });
        #[cfg(not(feature = "parallel"))]
        for b in self.buckets.iter_mut() {
            if !b.is_empty() { b.sort_unstable(); }
        }

        self.idx.backend = LookupBackend::BucketHash(
            super::index_bucket::BucketHashLookup::build(self.bucket_bits, &mut self.buckets, self.max_occ)
        );
        self.idx.packed_seqs = self.packed;
        self.idx
    }
}

impl Index {
    /// Dense lexicographic ranking of the target names, computed once on first use and cached
    pub fn name_rank(&self) -> &NameRankData {
        self.name_rank.get_or_init(|| {
            let n = self.seqs.len();
            let mut order: Vec<u32> = (0..n as u32).collect();
            order.sort_by(|&a, &b| self.seqs[a as usize].name.cmp(&self.seqs[b as usize].name));
            let mut rank = vec![0u32; n];
            let mut r = 0u32;
            for i in 0..n {
                if i > 0 && self.seqs[order[i] as usize].name != self.seqs[order[i - 1] as usize].name {
                    r += 1;
                }
                rank[order[i] as usize] = r;
            }
            NameRankData { rank, order }
        })
    }

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
    ///
    /// Each nibble of `packed` is guaranteed by `Index::build` to hold 0-4 only
    /// (the encoding table maps to {0,1,2,3,4} and OR-writes target unique nibble
    /// positions), so `& 0xf` alone is sufficient and the runtime `.min(4)` clamp
    /// is dropped.
    fn unpack_nt4_into(packed: &[u32], gpos_start: usize, out: &mut [u8]) {
        let len = out.len();
        if len == 0 { return; }
        let mut i = 0;
        let mut gpos = gpos_start;

        while i < len && (gpos & 7) != 0 {
            out[i] = ((packed[gpos >> 3] >> (((gpos & 7) << 2) as u32)) & 0xf) as u8;
            i += 1;
            gpos += 1;
        }

        let word_start = gpos >> 3;
        let full_words = (len - i) >> 3;
        if full_words > 0 {
            let words = &packed[word_start..word_start + full_words];
            let main_end = i + (full_words << 3);
            let (chunks, _rem) = out[i..main_end].as_chunks_mut::<8>();
            for (chunk, &word) in chunks.iter_mut().zip(words.iter()) {
                chunk[0] = ( word        & 0xf) as u8;
                chunk[1] = ((word >>  4) & 0xf) as u8;
                chunk[2] = ((word >>  8) & 0xf) as u8;
                chunk[3] = ((word >> 12) & 0xf) as u8;
                chunk[4] = ((word >> 16) & 0xf) as u8;
                chunk[5] = ((word >> 20) & 0xf) as u8;
                chunk[6] = ((word >> 24) & 0xf) as u8;
                chunk[7] = ((word >> 28) & 0xf) as u8;
            }
            i = main_end;
            gpos = gpos_start + i;
        }

        while i < len {
            out[i] = ((packed[gpos >> 3] >> (((gpos & 7) << 2) as u32)) & 0xf) as u8;
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

    /// Peek at a file's leading 4 bytes to detect a binary index (RMMI or minimap2
    /// `.mmi`), independent of file extension. Callers deciding whether to load a
    /// pre-built index or build a new one from a sequence file should not rely on
    /// extension alone: a real minimap2/rammap index saved under a non-standard name
    /// (e.g. `.mm2` instead of `.mmi`) is otherwise silently misread as sequence
    /// input, producing a bogus index built from the raw index bytes with no error.
    /// Returns false on any read error, or for the legacy no-magic bincode format
    /// (which can't be distinguished from arbitrary binary data by sniffing alone —
    /// that format still needs the extension-based check as a fallback).
    pub fn sniff_is_index_file(path: &str) -> bool {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_err() {
            return false;
        }
        &magic == RMMI_MAGIC || &magic == Self::MINIMAP2_INDEX_MAGIC
    }

    /// Load the next index part from a reader. Returns None on EOF.
    /// Detects RMMI (bincode-serialized), MMI\2 (.mmi binary layout), or
    /// legacy no-magic bincode formats.
    pub fn load_part<R: Read + Seek>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut magic = [0u8; 4];
        match reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let idx: Self = if &magic == RMMI_MAGIC {
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

        // Read per-bucket hash tables directly into BucketHashLookup.
        // .mmi format stores per-bucket: n positions (p[]), then hash entries.
        let n_buckets = 1usize << b;
        let mut bucket_data: Vec<(Vec<u64>, Vec<(u64, u64)>)> = Vec::with_capacity(n_buckets);

        for _ in 0..n_buckets {
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

            let hash_entries = if hash_size > 0 {
                let hash_buf = read_u64_vec(reader, hash_size * 2)?;
                (0..hash_size).map(|i| (hash_buf[i * 2], hash_buf[i * 2 + 1])).collect()
            } else {
                Vec::new()
            };

            bucket_data.push((p, hash_entries));
        }

        let backend = LookupBackend::BucketHash(
            super::index_bucket::BucketHashLookup::from_minimap2_buckets(b as u32, bucket_data)
        );

        // Read packed 4-bit sequences if present
        let packed_seqs = if !no_seq {
            let n_u32 = sum_len.div_ceil(8) as usize;
            if n_u32 > 0 { read_u32_vec(reader, n_u32)? } else { Vec::new() }
        } else {
            Vec::new()
        };

        let idx = Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs,
            backend,
            packed_seqs,
            name_rank: Default::default(),
        };

        eprintln!("[*] Index loaded: {} seqs, {}M bases in {:.1}s",
            idx.seqs.len(), sum_len / 1_000_000,
            t_total.elapsed().as_secs_f64());
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
            backend: LookupBackend::BucketHash(super::index_bucket::BucketHashLookup::empty()),
            packed_seqs: Vec::new(),
            name_rank: Default::default(),
        }
    }

    pub fn new(w: usize, k: usize, is_hpc: bool) -> Self {
        Index {
            kmer_size: k,
            window_size: w,
            homopolymer_compressed: is_hpc,
            index: 0,
            seqs: Vec::new(),
            backend: LookupBackend::BucketHash(super::index_bucket::BucketHashLookup::empty()),
            packed_seqs: Vec::new(),
            name_rank: Default::default(),
        }
    }

    /// Build an index from target sequences. Convenience wrapper around
    /// Build an index from target sequences. Uses parallel pack + sketch on
    /// CHUNK_SIZE-sequence batches (rayon-fanned when the `parallel` feature
    /// is enabled), which is materially faster than the sequential
    /// [`IndexBuilder`] for multi-sequence references — that path exists for
    /// WASM streaming, where we can't hold the full `Vec<(String, Vec<u8>)>`
    /// in memory at once.
    ///
    /// `max_occ` is a hard cap to filter extremely repetitive minimizers
    /// during index construction. The `mid_occ` threshold (calculated via
    /// `cal_mid_occ`) should be applied at query time, not during build.
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
            backend: LookupBackend::BucketHash(super::index_bucket::BucketHashLookup::empty()),
            packed_seqs: Vec::new(),
            name_rank: Default::default(),
        };

        let mut offset = 0usize;

        // Collect metadata and total length.
        let mut seq_data: Vec<Vec<u8>> = Vec::with_capacity(seqs.len());
        for (name, seq_bytes) in seqs {
            let len = seq_bytes.len();
            idx.seqs.push(TargetSequence { name, len, offset: offset as u64, is_alt: false });
            offset += len;
            seq_data.push(seq_bytes);
        }

        let bucket_bits = 10u32.min((2 * k) as u32);
        let n_buckets = 1usize << bucket_bits;
        let mask = (n_buckets - 1) as u64;

        let n_u32 = offset.div_ceil(8);
        let mut packed = vec![0u32; n_u32];
        let mut buckets: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_buckets];

        // Process sequences in chunks: drain a chunk, pack sequentially,
        // parallel-sketch into per-sequence minimizer Vecs, sequential
        // distribute into global buckets, drop the chunk. Bounds peak memory
        // to roughly chunk_seqs + chunk_minimizers + global_buckets.
        const CHUNK_SIZE: usize = 32;
        let mut rid_offset = 0usize;
        while !seq_data.is_empty() {
            let take_n = CHUNK_SIZE.min(seq_data.len());
            let chunk: Vec<Vec<u8>> = seq_data.drain(..take_n).map(|mut s| {
                if is_hpc {
                    for b in s.iter_mut() { *b = b.to_ascii_uppercase(); }
                }
                s
            }).collect();

            // Pack chunk sequentially. Adjacent sequences may share u32s at byte
            // boundaries, so concurrent read-modify-write would race.
            for (i, seq) in chunk.iter().enumerate() {
                let rid = rid_offset + i;
                let goff = idx.seqs[rid].offset as usize;
                for (j, &b) in seq.iter().enumerate() {
                    let gpos = goff + j;
                    packed[gpos >> 3] |= NT4_TABLE[b as usize] << (((gpos & 7) << 2) as u32);
                }
            }

            // Parallel-sketch each sequence into its own minimizer Vec.
            #[cfg(feature = "parallel")]
            let chunk_minis: Vec<Vec<crate::align::sketch::Minimizer>> = chunk.par_iter().enumerate()
                .map(|(i, ascii)| {
                    let rid = rid_offset + i;
                    let len = idx.seqs[rid].len;
                    let mut m = Vec::new();
                    sketch_sequence(ascii, len, w, k, rid, is_hpc, &mut m);
                    m
                })
                .collect();
            #[cfg(not(feature = "parallel"))]
            let chunk_minis: Vec<Vec<crate::align::sketch::Minimizer>> = chunk.iter().enumerate()
                .map(|(i, ascii)| {
                    let rid = rid_offset + i;
                    let len = idx.seqs[rid].len;
                    let mut m = Vec::new();
                    sketch_sequence(ascii, len, w, k, rid, is_hpc, &mut m);
                    m
                })
                .collect();

            // Distribute minimizers into global buckets sequentially.
            for minis in chunk_minis {
                for m in minis {
                    let hash = m.x >> 8;
                    buckets[(hash & mask) as usize].push((hash, m.y));
                }
            }

            rid_offset += take_n;
        }
        drop(seq_data);
        idx.packed_seqs = packed;

        // Parallel bucket post-processing: sort each bucket independently.
        #[cfg(feature = "parallel")]
        buckets.par_iter_mut().for_each(|b| {
            if !b.is_empty() { b.sort_unstable(); }
        });
        #[cfg(not(feature = "parallel"))]
        for b in buckets.iter_mut() {
            if !b.is_empty() { b.sort_unstable(); }
        }

        idx.backend = LookupBackend::BucketHash(
            super::index_bucket::BucketHashLookup::build(bucket_bits, &mut buckets, max_occ)
        );
        idx
    }

    pub fn get(&self, hash: u64) -> Option<&[u64]> {
        self.backend.get(hash)
    }

    /// Returns the (start, end) range for a hash, for deferred retrieval via `get_by_range`.
    #[inline]
    pub fn get_range(&self, hash: u64) -> Option<(u32, u32)> {
        self.backend.get_range(hash)
    }

    /// Retrieve a slice of positions from a previously computed range.
    #[inline]
    pub fn get_by_range(&self, range: (u32, u32)) -> &[u64] {
        self.backend.get_by_range(range)
    }

    /// Prefetch the hash table bucket for a hash value into L1 cache.
    /// Call a few iterations ahead of the actual `get_range` for that hash.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        match &self.backend {
            LookupBackend::BucketHash(b) => b.prefetch(hash),
        }
    }

    /// Prefetch a positions range into L1 cache.
    #[inline]
    pub fn prefetch_positions(&self, range: (u32, u32)) {
        match &self.backend {
            LookupBackend::BucketHash(b) => b.prefetch_positions(range),
        }
    }

    /// Calculate mid_occ threshold to filter top `frac` fraction of repetitive minimizers.
    pub fn cal_mid_occ(&self, frac: f32, min_mid_occ: i32, max_mid_occ: i32) -> usize {
        let min_mid = min_mid_occ.max(1) as usize;
        if frac <= 0.0 { return usize::MAX; }
        if self.backend.is_empty() { return min_mid; }

        let mut counts: Vec<u32> = self.backend.occurrence_counts().collect();

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
        assert_eq!(positions.unwrap()[0], 624);
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
