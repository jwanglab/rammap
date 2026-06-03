//! Per-bucket hash table index backend.
//!
//! Each of 2^B buckets has its own open-addressing hash table that maps
//! hash suffixes to (offset, count) ranges in a shared flat `positions` array.
//! During build, each bucket's temporary (hash, position) pairs are processed
//! and freed independently — the key to controlling peak build memory.
//!
//! All positions (including singletons) live in `positions[]` so that the
//! existing `get_range`/`get_by_range` API works without changes.

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::SeqAccess;
use super::index::SeedLookup;
use hashbrown::HashTable;

/// A single bucket's hash table — hashbrown SwissTable for ~87.5% load factor.
/// Keys are u64 hash suffixes (hash >> bucket_bits) and already well-distributed
/// minimizer hashes, so we pass them directly as the hash to skip rehashing.
/// Entries are stored as (key, val); val packs (offset: u32, count: u32).
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
    fn get(&self, key: u64) -> Option<u64> {
        self.0.find(key, |&(k, _)| k == key).map(|&(_, v)| v)
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

/// Minimap2-style per-bucket hash index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketHashLookup {
    bucket_bits: u32,
    buckets: Vec<Bucket>,
    /// Flat position array for ALL hashes (singletons and multi-occurrence).
    positions: Vec<u64>,
}

impl BucketHashLookup {
    /// Create an empty lookup (for placeholder/header-only indices).
    pub fn empty() -> Self {
        Self { bucket_bits: 0, buckets: Vec::new(), positions: Vec::new() }
    }

    /// Load from the `.mmi` per-bucket hash table format.
    ///
    /// Each bucket contributes: `p[]` (multi-occurrence positions) + hash entries
    /// (key/value pairs where key bit 0 = singleton flag). We convert to our
    /// own bucket hash format with a unified positions array.
    pub fn from_minimap2_buckets(
        bucket_bits: u32,
        bucket_data: Vec<(Vec<u64>, Vec<(u64, u64)>)>, // per-bucket: (p_array, hash_entries)
    ) -> Self {
        let n_buckets = bucket_data.len();
        let mut positions: Vec<u64> = Vec::new();
        let mut buckets: Vec<Bucket> = Vec::with_capacity(n_buckets);

        for (p, hash_entries) in &bucket_data {
            if hash_entries.is_empty() {
                buckets.push(bucket_empty());
                continue;
            }

            let mut ht = bucket_with_capacity(hash_entries.len());

            for &(key, value) in hash_entries {
                // Source-format key: (hash_suffix << 1) | singleton_flag.
                // Our key: hash_suffix (key >> 1), stored as u32.
                let our_key = key >> 1;

                if key & 1 != 0 {
                    // Singleton: value is the position directly
                    let offset = positions.len() as u64;
                    positions.push(value);
                    ht.insert(our_key, (offset << 32) | 1);
                } else {
                    // Multi-occurrence: value = (offset_in_p << 32) | count
                    let count = (value & 0xFFFFFFFF) as usize;
                    let start = (value >> 32) as usize;
                    let offset = positions.len() as u64;
                    for j in 0..count {
                        positions.push(p[start + j]);
                    }
                    ht.insert(our_key, (offset << 32) | count as u64);
                }
            }

            buckets.push(ht);
        }

        Self { bucket_bits, buckets, positions }
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
                let offset = local_pos.len() as u32;
                for it in b.iter().take(i).skip(start) {
                    local_pos.push(it.1);
                }
                ht.insert(key, ((offset as u64) << 32) | count as u64);
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
                total_positions += i - start;
            }
        }

        let mut positions: Vec<u64> = Vec::with_capacity(total_positions);
        let mut buckets: Vec<Bucket> = Vec::with_capacity(n_buckets);

        for bucket_entries in sorted_buckets.iter_mut() {
            let b = std::mem::take(bucket_entries);
            let (mut ht, local_pos) = process_one(b);
            // Fix up offsets to global positions array.
            let global_offset = positions.len() as u64;
            if global_offset > 0 {
                for v in ht.values_mut() {
                    let old = *v;
                    *v = ((old >> 32) + global_offset) << 32 | (old & 0xFFFFFFFF);
                }
            }
            positions.extend_from_slice(&local_pos);
            buckets.push(ht);
            // b + local_pos dropped here — freed before next bucket
        }

        Self { bucket_bits, buckets, positions }
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

    /// Prefetch a positions range into L1 cache.
    #[inline]
    pub fn prefetch_positions(&self, range: (u32, u32)) {
        if (range.0 as usize) < self.positions.len() {
            unsafe {
                let ptr = self.positions.as_ptr().add(range.0 as usize) as *const u8;
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                #[cfg(not(target_arch = "x86_64"))]
                { let _ = ptr; }
            }
        }
    }
}

impl SeedLookup for BucketHashLookup {
    #[inline]
    fn get(&self, hash: u64) -> Option<&[u64]> {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket = &self.buckets[(hash & mask) as usize];
        let key = hash >> self.bucket_bits;
        let encoded = bucket.get(key)?;
        let offset = (encoded >> 32) as usize;
        let count = (encoded & 0xFFFFFFFF) as usize;
        Some(&self.positions[offset..offset + count])
    }

    #[inline]
    fn get_range(&self, hash: u64) -> Option<(u32, u32)> {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket = &self.buckets[(hash & mask) as usize];
        let key = hash >> self.bucket_bits;
        let encoded = bucket.get(key)?;
        let offset = (encoded >> 32) as u32;
        let count = (encoded & 0xFFFFFFFF) as u32;
        Some((offset, offset + count))
    }

    #[inline]
    fn get_by_range(&self, range: (u32, u32)) -> &[u64] {
        &self.positions[range.0 as usize..range.1 as usize]
    }

    fn occurrence_counts(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.buckets.iter().flat_map(|b| {
            b.values().map(|val| (*val & 0xFFFFFFFF) as u32)
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

        // get_range / get_by_range
        let (s, e) = lut.get_range(101).unwrap();
        assert_eq!((e - s) as usize, 2);
        let slice = lut.get_by_range((s, e));
        assert_eq!(slice.len(), 2);
    }
}
