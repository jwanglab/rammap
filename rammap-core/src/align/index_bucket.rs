//! Per-bucket hash table index backend.
//!
//! Each of 2^B buckets has its own open-addressing hash table that maps
//! hash suffixes to `SeedHits`: a single-occurrence position stored inline, or
//! an (offset, count) range in a shared flat `positions` array.
//! During build, each bucket's temporary (hash, position) pairs are processed
//! and freed independently — the key to controlling peak build memory.

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::SeqAccess;
use super::index::{SeedHits, SeedLookup};
use hashbrown::HashTable;

/// A single bucket's hash table — hashbrown SwissTable for ~87.5% load factor.
/// Keys are u64 hash suffixes (hash >> bucket_bits) and already well-distributed
/// minimizer hashes, so we pass them directly as the hash to skip rehashing.
/// Entries are stored as (key, val); val is a raw `SeedHits`.
#[derive(Debug, Default)]
struct Bucket(HashTable<(u64, u64)>);

#[inline]
fn bucket_empty() -> Bucket {
    Bucket(HashTable::new())
}

#[inline]
fn bucket_with_capacity(n: usize) -> Bucket {
    Bucket(HashTable::with_capacity(n))
}

impl Bucket {
    #[inline]
    fn insert(&mut self, key: u64, val: u64) {
        self.0.insert_unique(key, (key, val), |&(k, _)| k);
    }

    #[inline]
    fn get(&self, key: u64) -> Option<&u64> {
        self.0.find(key, |&(k, _)| k == key).map(|(_, v)| v)
    }

    #[inline]
    fn is_empty(&self) -> bool { self.0.is_empty() }

    #[inline]
    fn values_mut(&mut self) -> impl Iterator<Item = &mut u64> + '_ {
        self.0.iter_mut().map(|(_, v)| v)
    }

    #[inline]
    fn values(&self) -> impl Iterator<Item = &u64> + '_ {
        self.0.iter().map(|(_, v)| v)
    }
}

impl Clone for Bucket {
    fn clone(&self) -> Self {
        let mut t = HashTable::with_capacity(self.0.len());
        for &entry in self.0.iter() {
            t.insert_unique(entry.0, entry, |&(k, _)| k);
        }
        Bucket(t)
    }
}

impl Serialize for Bucket {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = ser.serialize_seq(Some(self.0.len()))?;
        for &entry in self.0.iter() {
            seq.serialize_element(&entry)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Bucket {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Bucket;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("seq of (u64,u64)")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<Bucket, A::Error> {
                let n = s.size_hint().unwrap_or(0);
                let mut t = HashTable::with_capacity(n);
                while let Some(entry) = s.next_element::<(u64, u64)>()? {
                    t.insert_unique(entry.0, entry, |&(k, _)| k);
                }
                Ok(Bucket(t))
            }
        }
        de.deserialize_seq(V)
    }
}

/// Hit counts at or above this are stored out of line: the count field is 0 and
/// the true count occupies the positions slot before the hits. Lowered under
/// test so that path is exercised.
const OUT_OF_LINE_COUNT: usize = if cfg!(test) { 16 } else { 1 << SeedHits::COUNT_BITS };

/// Append one multi-occurrence hit list to `positions` and return its range,
/// with the offset relative to the start of `positions`.
fn push_range(positions: &mut Vec<u64>, hits: impl ExactSizeIterator<Item = u64>) -> SeedHits {
    let offset = positions.len() as u64;
    let n = hits.len();
    let count = if n >= OUT_OF_LINE_COUNT { positions.push(n as u64); 0 } else { n as u64 };
    positions.extend(hits);
    SeedHits::range(offset, count)
}

/// Minimap2-style per-bucket hash index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketHashLookup {
    bucket_bits: u32,
    buckets: Vec<Bucket>,
    /// Flat position array for multi-occurrence hashes; singletons are inline.
    positions: Vec<u64>,
}

impl BucketHashLookup {
    /// Create an empty lookup (for placeholder/header-only indices).
    pub fn empty() -> Self {
        Self { bucket_bits: 0, buckets: Vec::new(), positions: Vec::new() }
    }

    /// Start an empty lookup to be filled one bucket at a time with
    /// `push_mmi_bucket`, so a loader never holds the source buckets and the
    /// converted table in memory at once.
    pub fn with_bucket_bits(bucket_bits: u32) -> Self {
        Self { bucket_bits, buckets: Vec::with_capacity(1usize << bucket_bits), positions: Vec::new() }
    }

    /// Append the next bucket in the `.mmi` per-bucket hash table format.
    ///
    /// `p` holds the bucket's multi-occurrence positions and `hash_kv` its hash
    /// entries as flat key/value pairs, where key bit 0 is the singleton flag.
    /// We convert to our own bucket hash format with a unified positions array.
    pub fn push_mmi_bucket(&mut self, p: &[u64], hash_kv: &[u64]) {
        if hash_kv.is_empty() {
            self.buckets.push(bucket_empty());
            return;
        }

        let positions = &mut self.positions;
        let mut ht = bucket_with_capacity(hash_kv.len() / 2);

        for kv in hash_kv.chunks_exact(2) {
            let (key, value) = (kv[0], kv[1]);
            // Source-format key: (hash_suffix << 1) | singleton_flag.
            // Our key: hash_suffix (key >> 1), stored as u32.
            let our_key = key >> 1;

            if key & 1 != 0 {
                // Singleton: value is the position directly
                ht.insert(our_key, SeedHits::single(value).raw());
            } else {
                // Multi-occurrence: value = (offset_in_p << 32) | count
                let count = (value & 0xFFFFFFFF) as usize;
                let start = (value >> 32) as usize;
                ht.insert(our_key, push_range(positions, p[start..start + count].iter().copied()).raw());
            }
        }

        self.buckets.push(ht);
    }

    /// Build from pre-distributed, pre-sorted bucket data.
    ///
    /// Each bucket is processed independently in parallel: build per-bucket
    /// hash table + local positions, consume and free the source bucket. Then
    /// concatenate all per-bucket positions and fix up offsets.
    pub fn build(bucket_bits: u32, sorted_buckets: &mut [Vec<(u64, u64)>], max_occ: usize) -> Self {
        let n_buckets = sorted_buckets.len();

        // Process each bucket independently: sort (already done), build hash table
        // with local position offsets, collect local positions vec.
        // Returns (Bucket, Vec<u64>) per input bucket.
        let process_one = |b: Vec<(u64, u64)>| -> (Bucket, Vec<u64>) {
            if b.is_empty() {
                return (bucket_empty(), Vec::new());
            }
            // Count unique hashes for hash table sizing.
            let mut n_unique = 0usize;
            {
                let mut i = 0;
                while i < b.len() {
                    let h = b[i].0;
                    let start = i;
                    while i < b.len() && b[i].0 == h { i += 1; }
                    if max_occ < usize::MAX && (i - start) > max_occ { continue; }
                    n_unique += 1;
                }
            }

            let mut ht = bucket_with_capacity(n_unique);
            let mut local_pos: Vec<u64> = Vec::new();

            let mut i = 0;
            while i < b.len() {
                let h = b[i].0;
                let start = i;
                while i < b.len() && b[i].0 == h { i += 1; }
                let count = i - start;
                if max_occ < usize::MAX && count > max_occ { continue; }

                let key = h >> bucket_bits;
                if count == 1 {
                    ht.insert(key, SeedHits::single(b[start].1).raw());
                    continue;
                }
                // Local offset; shifted to the global positions array below.
                ht.insert(key, push_range(&mut local_pos, b[start..i].iter().map(|e| e.1)).raw());
            }
            (ht, local_pos)
        };

        // Process each bucket sequentially: build hash table, copy positions
        // into shared array, consume and free the source bucket. This avoids
        // holding all per-bucket results simultaneously.
        // (Bucket sorting is done externally by the caller using par_iter.)
        let mut total_positions = 0usize;
        for b in sorted_buckets.iter() {
            let mut i = 0;
            while i < b.len() {
                let h = b[i].0;
                let start = i;
                while i < b.len() && b[i].0 == h { i += 1; }
                if max_occ < usize::MAX && (i - start) > max_occ { continue; }
                if i - start > 1 { total_positions += i - start + (i - start >= OUT_OF_LINE_COUNT) as usize; }
            }
        }
        // Fail before the build work rather than at the bucket that overflows.
        assert!((total_positions as u64) < SeedHits::MAX_OFFSET,
            "index has too many multi-occurrence positions for one part ({total_positions}); use a smaller -I");

        let mut positions: Vec<u64> = Vec::with_capacity(total_positions);
        let mut buckets: Vec<Bucket> = Vec::with_capacity(n_buckets);

        for bucket_entries in sorted_buckets.iter_mut() {
            let b = std::mem::take(bucket_entries);
            let (mut ht, local_pos) = process_one(b);
            // Turn local (offset, count) pairs into ranges into the global positions array.
            let global_offset = positions.len() as u64;
            if global_offset > 0 {
                for v in ht.values_mut() {
                    if let Some((offset, count)) = SeedHits::from_raw(*v).as_range() {
                        *v = SeedHits::range(offset as u64 + global_offset, count as u64).raw();
                    }
                }
            }
            positions.extend_from_slice(&local_pos);
            buckets.push(ht);
            // b + local_pos dropped here — freed before next bucket
        }

        Self { bucket_bits, buckets, positions }
    }

    /// Convert an index saved before singletons were stored inline, where every
    /// value is an unflagged (offset, count) range into `positions`.
    pub fn upgrade_v1(&mut self) {
        let old = std::mem::take(&mut self.positions);
        let mut positions = Vec::new();
        for b in &mut self.buckets {
            for v in b.values_mut() {
                let (offset, count) = ((*v >> 32) as usize, (*v & 0xFFFFFFFF) as usize);
                *v = if count == 1 {
                    SeedHits::single(old[offset]).raw()
                } else {
                    push_range(&mut positions, old[offset..offset + count].iter().copied()).raw()
                };
            }
        }
        self.positions = positions;
    }
}

impl BucketHashLookup {
    /// Prefetch the bucket for a hash value into L1 cache.
    /// Call this 5-10 iterations ahead of the actual lookup.
    /// With hashbrown's SwissTable internals hidden, we prefetch the bucket struct
    /// itself; the actual probe location is opaque, so this is a coarser hint.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bi = (hash & mask) as usize;
        if bi < self.buckets.len() {
            // aarch64's prefetch intrinsic (`stdarch_aarch64_prefetch`) is still
            // nightly-only as of Rust 1.96, so we skip the hint there to stay on
            // stable. It costs nothing on Apple M2 and 1-2% on a Raspberry Pi.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let bucket_ptr = &self.buckets[bi] as *const _ as *const u8;
                std::arch::x86_64::_mm_prefetch(bucket_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
    }

    /// Prefetch a handle's positions into L1 cache (no-op for an inline singleton).
    #[inline]
    pub fn prefetch_hits(&self, hits: SeedHits) {
        let Some((offset, _)) = hits.as_range() else { return };
        if offset < self.positions.len() {
            unsafe {
                let ptr = self.positions.as_ptr().add(offset) as *const u8;
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                #[cfg(not(target_arch = "x86_64"))]
                { let _ = ptr; }
            }
        }
    }
}

impl BucketHashLookup {
    /// The positions behind a range, resolving an out-of-line count.
    #[inline]
    fn range_hits(&self, offset: usize, count: usize) -> &[u64] {
        if count == 0 {
            let n = self.positions[offset] as usize;
            &self.positions[offset + 1..offset + 1 + n]
        } else {
            &self.positions[offset..offset + count]
        }
    }
}

impl SeedLookup for BucketHashLookup {
    #[inline]
    fn get(&self, hash: u64) -> Option<&[u64]> {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket = &self.buckets[(hash & mask) as usize];
        let key = hash >> self.bucket_bits;
        let raw = bucket.get(key)?;
        // An inline singleton's raw value is the position itself.
        Some(match SeedHits::from_raw(*raw).as_range() {
            Some((offset, count)) => self.range_hits(offset, count),
            None => std::slice::from_ref(raw),
        })
    }

    #[inline]
    fn get_hits(&self, hash: u64) -> Option<SeedHits> {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket = &self.buckets[(hash & mask) as usize];
        let key = hash >> self.bucket_bits;
        bucket.get(key).map(|&raw| SeedHits::from_raw(raw))
    }

    #[inline]
    fn hits<'a>(&'a self, hits: &'a SeedHits) -> &'a [u64] {
        match hits.as_range() {
            Some((offset, count)) => self.range_hits(offset, count),
            // A singleton handle holds the position itself.
            None => std::slice::from_ref(hits.pos_ref()),
        }
    }

    fn occurrence_counts(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.buckets.iter().flat_map(move |b| {
            b.values().map(move |&raw| match SeedHits::from_raw(raw).as_range() {
                Some((offset, count)) => self.range_hits(offset, count).len() as u32,
                None => 1,
            })
        }))
    }

    fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_hash_basic() {
        let bucket_bits = 2u32; // 4 buckets
        let mask = (1u64 << bucket_bits) - 1;

        // Create test entries: 3 hashes, each in a different bucket
        let mut buckets = vec![Vec::new(); 4];
        let entries = vec![
            (100u64, 1000u64), // bucket 100 & 3 = 0
            (101, 2000),       // bucket 101 & 3 = 1
            (101, 2001),       // same hash, bucket 1
            (102, 3000),       // bucket 102 & 3 = 2
        ];
        for &(h, p) in &entries {
            buckets[(h & mask) as usize].push((h, p));
        }
        for b in &mut buckets { b.sort_unstable(); }

        let lut = BucketHashLookup::build(bucket_bits, &mut buckets, usize::MAX);

        // Singleton lookup
        assert_eq!(lut.get(100), Some(&[1000u64][..]));
        // Multi-occurrence lookup
        let r = lut.get(101).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.contains(&2000));
        assert!(r.contains(&2001));
        // Another singleton
        assert_eq!(lut.get(102), Some(&[3000u64][..]));
        // Missing hash
        assert_eq!(lut.get(999), None);

        // get_hits / hits: singletons inline, multi-occurrence as a range
        let h = lut.get_hits(100).unwrap();
        assert_eq!(h.as_range(), None);
        assert_eq!(lut.hits(&h), &[1000u64][..]);
        let h = lut.get_hits(101).unwrap();
        assert_eq!(lut.hits(&h).len(), 2);
        assert!(lut.positions.len() == 2, "only multi-occurrence positions are stored");
    }

    #[test]
    fn test_ranges_in_later_buckets_point_at_their_own_positions() {
        let bucket_bits = 1u32; // 2 buckets
        let mut buckets = vec![
            vec![(2u64, 10u64), (2, 11), (4, 12)],  // bucket 0: multi + singleton
            vec![(3u64, 20u64), (3, 21), (3, 22)],  // bucket 1: multi at global offset 2
        ];
        let lut = BucketHashLookup::build(bucket_bits, &mut buckets, usize::MAX);
        assert_eq!(lut.get(2), Some(&[10u64, 11][..]));
        assert_eq!(lut.get(4), Some(&[12u64][..]));
        assert_eq!(lut.get(3), Some(&[20u64, 21, 22][..]));
        let mut counts: Vec<u32> = lut.occurrence_counts().collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 2, 3]);
    }

    #[test]
    fn test_mmi_bucket_and_v1_upgrade_match() {
        // One .mmi bucket: key bit 0 flags a singleton whose value is the position;
        // otherwise the value is (offset_in_p << 32) | count.
        let p = [500u64, 501, 502];
        let kv = [(7u64 << 1) | 1, 42, 9u64 << 1, 3];
        let mut lut = BucketHashLookup::with_bucket_bits(0);
        lut.push_mmi_bucket(&p, &kv);
        assert_eq!(lut.get(7), Some(&[42u64][..]));
        assert_eq!(lut.get(9), Some(&[500u64, 501, 502][..]));

        // The same table in the v1 layout: every value an unflagged range.
        let mut ht = bucket_with_capacity(2);
        ht.insert(9, 3);         // offset 0, count 3
        ht.insert(7, 3 << 32 | 1); // offset 3, count 1
        let mut v1 = BucketHashLookup { bucket_bits: 0, buckets: vec![ht], positions: vec![500, 501, 502, 42] };
        v1.upgrade_v1();
        assert_eq!(v1.get(7), Some(&[42u64][..]));
        assert_eq!(v1.get(9), Some(&[500u64, 501, 502][..]));
        assert_eq!(v1.positions.len(), 3);
    }

    #[test]
    fn test_out_of_line_counts_in_every_build_path() {
        let n = OUT_OF_LINE_COUNT + 4;
        let big: Vec<u64> = (0..n as u64).map(|i| 1000 + i).collect();
        let check = |lut: &BucketHashLookup, key: u64| {
            assert_eq!(lut.get(key), Some(&big[..]));
            let h = lut.get_hits(key).unwrap();
            assert_eq!(h.as_range().unwrap().1, 0, "count is stored out of line");
            assert_eq!(lut.hits(&h), &big[..]);
        };

        // build: the out-of-line range sits in bucket 1, after bucket 0's range,
        // so its offset is shifted to the global array.
        let mut buckets = vec![
            vec![(2u64, 10u64), (2, 11)],
            big.iter().map(|&p| (3u64, p)).collect(),
        ];
        let lut = BucketHashLookup::build(1, &mut buckets, usize::MAX);
        check(&lut, 3);
        assert_eq!(lut.get(2), Some(&[10u64, 11][..]));
        assert_eq!(lut.positions.len(), 2 + 1 + n);
        let mut counts: Vec<u32> = lut.occurrence_counts().collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![2, n as u32]);

        // .mmi bucket
        let kv = [9u64 << 1, n as u64];
        let mut lut = BucketHashLookup::with_bucket_bits(0);
        lut.push_mmi_bucket(&big, &kv);
        check(&lut, 9);

        // v1 upgrade
        let mut ht = bucket_with_capacity(1);
        ht.insert(9, n as u64);
        let mut v1 = BucketHashLookup { bucket_bits: 0, buckets: vec![ht], positions: big.clone() };
        v1.upgrade_v1();
        check(&v1, 9);
    }
}
