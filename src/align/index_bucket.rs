//! Per-bucket hash table index backend.
//!
//! Each of 2^B buckets has its own open-addressing hash table that maps
//! hash suffixes to (offset, count) ranges in a shared flat `positions` array.
//! During build, each bucket's temporary (hash, position) pairs are processed
//! and freed independently, matching minimap2's `worker_post` — the key to
//! controlling peak build memory.
//!
//! All positions (including singletons) live in `positions[]` so that the
//! existing `get_range`/`get_by_range` API works without changes.

use serde::{Serialize, Deserialize};
use super::index::SeedLookup;

/// Sentinel for empty hash table slots.
const EMPTY_KEY: u32 = u32::MAX;

/// A single bucket's hash table. Open-addressing with linear probing.
/// Keys are u32 hash suffixes (hash >> bucket_bits, truncated to 32 bits).
/// Values are packed (offset: u32, count: u32) as a single u64.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Bucket {
    /// Interleaved [key0(u32), pad(u32), val(u64), key1(u32), pad(u32), val(u64), ...]
    /// Using separate key/val arrays for better cache behavior on lookup.
    keys: Vec<u32>,
    vals: Vec<u64>,
    mask: u32, // capacity - 1
}

impl Bucket {
    fn empty() -> Self {
        Self { keys: Vec::new(), vals: Vec::new(), mask: 0 }
    }

    fn with_capacity(n: usize) -> Self {
        if n == 0 { return Self::empty(); }
        let cap = ((n * 4 / 3) + 1).next_power_of_two().max(4);
        let keys = vec![EMPTY_KEY; cap];
        let vals = vec![0u64; cap];
        Self { keys, vals, mask: (cap - 1) as u32 }
    }

    #[inline]
    fn insert(&mut self, key: u32, value: u64) {
        let mut idx = key & self.mask;
        loop {
            let i = idx as usize;
            if self.keys[i] == EMPTY_KEY {
                self.keys[i] = key;
                self.vals[i] = value;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline]
    fn get(&self, key: u32) -> Option<u64> {
        if self.mask == 0 { return None; }
        let mut idx = key & self.mask;
        loop {
            let i = idx as usize;
            let k = self.keys[i];
            if k == EMPTY_KEY { return None; }
            if k == key { return Some(self.vals[i]); }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Iterate over all occupied (key, value) pairs.
    fn iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.keys.iter().copied().zip(self.vals.iter().copied())
            .filter(|(k, _)| *k != EMPTY_KEY)
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

    /// Load from minimap2's per-bucket hash table format.
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
                buckets.push(Bucket::empty());
                continue;
            }

            let mut ht = Bucket::with_capacity(hash_entries.len());

            for &(key, value) in hash_entries {
                // minimap2 key: (hash_suffix << 1) | singleton_flag
                // Our key: hash_suffix (key >> 1), stored as u32
                let our_key = (key >> 1) as u32;

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
    /// Each bucket is processed independently in parallel (like minimap2's
    /// `worker_post` via `kt_for`): build per-bucket hash table + local
    /// positions, consume and free the source bucket. Then concatenate
    /// all per-bucket positions and fix up offsets.
    pub fn build(bucket_bits: u32, sorted_buckets: &mut [Vec<(u64, u64)>], max_occ: usize) -> Self {
        let n_buckets = sorted_buckets.len();

        // Process each bucket independently: sort (already done), build hash table
        // with local position offsets, collect local positions vec.
        // Returns (Bucket, Vec<u64>) per input bucket.
        let process_one = |b: Vec<(u64, u64)>| -> (Bucket, Vec<u64>) {
            if b.is_empty() {
                return (Bucket::empty(), Vec::new());
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

            let mut ht = Bucket::with_capacity(n_unique);
            let mut local_pos: Vec<u64> = Vec::new();

            let mut i = 0;
            while i < b.len() {
                let h = b[i].0;
                let start = i;
                while i < b.len() && b[i].0 == h { i += 1; }
                let count = i - start;
                if max_occ < usize::MAX && count > max_occ { continue; }

                let key = (h >> bucket_bits) as u32;
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
                for i in 0..ht.keys.len() {
                    if ht.keys[i] != EMPTY_KEY {
                        let old = ht.vals[i];
                        ht.vals[i] = ((old >> 32) + global_offset) << 32 | (old & 0xFFFFFFFF);
                    }
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
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bi = (hash & mask) as usize;
        if bi < self.buckets.len() {
            let bucket = &self.buckets[bi];
            if bucket.mask > 0 {
                let key = (hash >> self.bucket_bits) as u32;
                let idx = (key & bucket.mask) as usize;
                // Prefetch the keys and vals arrays at the expected probe position
                unsafe {
                    let keys_ptr = bucket.keys.as_ptr().add(idx) as *const u8;
                    let vals_ptr = bucket.vals.as_ptr().add(idx) as *const u8;
                    #[cfg(target_arch = "x86_64")]
                    {
                        std::arch::x86_64::_mm_prefetch(keys_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                        std::arch::x86_64::_mm_prefetch(vals_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        std::arch::aarch64::_prefetch(keys_ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
                        std::arch::aarch64::_prefetch(vals_ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
                    }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    { let _ = (keys_ptr, vals_ptr); }
                }
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
                #[cfg(target_arch = "aarch64")]
                std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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
        let key = (hash >> self.bucket_bits) as u32;
        let encoded = bucket.get(key)?;
        let offset = (encoded >> 32) as usize;
        let count = (encoded & 0xFFFFFFFF) as usize;
        Some(&self.positions[offset..offset + count])
    }

    #[inline]
    fn get_range(&self, hash: u64) -> Option<(u32, u32)> {
        let mask = (1u64 << self.bucket_bits) - 1;
        let bucket = &self.buckets[(hash & mask) as usize];
        let key = (hash >> self.bucket_bits) as u32;
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
            b.iter().map(|(_, val)| (val & 0xFFFFFFFF) as u32)
        }))
    }

    fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.mask == 0)
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
