//! Radix sort implementations for index building and chain sorting.
//!
//! MSD (Most Significant Digit) in-place radix sort
//! This is an unstable sort: elements with equal keys may be reordered
//! Falls back to insertion sort for small subarrays

use crate::align::sketch::Minimizer;

const RS_MIN_SIZE: usize = 64;
const RS_MAX_BITS: u32 = 8;

/// Trait for types that can be radix-sorted by extracting a u64 key.
pub trait RadixKey: Copy {
    fn radix_key(&self) -> u64;
}

impl RadixKey for Minimizer {
    #[inline(always)]
    fn radix_key(&self) -> u64 { self.x }
}

impl RadixKey for (u64, u64) {
    #[inline(always)]
    fn radix_key(&self) -> u64 { self.0 }
}

/// Insertion sort fallback for small subarrays, sorting by radix key only.
#[inline]
fn rs_insertsort<T: RadixKey>(arr: &mut [T]) {
    rs_insertsort_by(arr, &|e: &T| e.radix_key());
}

/// Insertion sort by `key`: stable, compares keys only.
#[inline]
fn rs_insertsort_by<T: Copy, F: Fn(&T) -> u64>(arr: &mut [T], key: &F) {
    for i in 1..arr.len() {
        if key(&arr[i]) < key(&arr[i - 1]) {
            let tmp = arr[i];
            let mut j = i;
            while j > 0 && key(&tmp) < key(&arr[j - 1]) {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = tmp;
        }
    }
}

/// MSD in-place radix sort
fn rs_sort<T: RadixKey>(arr: &mut [T], n_bits: u32, s: u32) {
    rs_sort_by(arr, n_bits, s, &|e: &T| e.radix_key());
}

/// MSD in-place radix sort keyed by `key`.
fn rs_sort_by<T: Copy, F: Fn(&T) -> u64>(arr: &mut [T], n_bits: u32, s: u32, key: &F) {
    let size = 1usize << n_bits; // 256
    let m = (size - 1) as u64;

    // bb[i] = current write position for bucket i (advances during permutation)
    // be[i] = end boundary for bucket i (fixed after prefix sum)
    let mut bb = [0usize; 256];
    let mut be = [0usize; 256];

    // Count elements per bucket
    for item in arr.iter() {
        let byte = ((key(item) >> s) & m) as usize;
        be[byte] += 1;
    }

    // Prefix sums: be[i] = cumulative end, bb[i] = start of bucket i
    #[allow(clippy::manual_memcpy)] // can't use copy_from_slice: be is modified in the same loop
    for i in 1..size {
        be[i] += be[i - 1];
        bb[i] = be[i - 1];
    }

    // In-place cyclic permutation (American Flag sort)
    let mut k = 0usize;
    while k < size {
        if bb[k] < be[k] {
            let l = ((key(&arr[bb[k]]) >> s) & m) as usize;
            if l != k {
                // Element at k's front belongs to bucket l — start swap cycle
                let mut tmp = arr[bb[k]];
                let mut cur = l;
                loop {
                    // Place tmp into bucket cur, pick up what was there
                    std::mem::swap(&mut tmp, &mut arr[bb[cur]]);
                    bb[cur] += 1;
                    // Where does the picked-up element belong?
                    cur = ((key(&tmp) >> s) & m) as usize;
                    if cur == k {
                        break;
                    }
                }
                // Place the final element (belonging to bucket k) at k's front
                arr[bb[k]] = tmp;
                bb[k] += 1;
            } else {
                // Element already in correct bucket
                bb[k] += 1;
            }
        } else {
            k += 1;
        }
    }

    // Reset bb to proper bucket boundaries for recursion
    bb[0] = 0;
    bb[1..size].copy_from_slice(&be[..(size - 1)]);

    // Recurse on sub-buckets
    if s > 0 {
        let next_s = s.saturating_sub(n_bits);
        for i in 0..size {
            let len = be[i] - bb[i];
            if len > RS_MIN_SIZE {
                rs_sort_by(&mut arr[bb[i]..be[i]], n_bits, next_s, key);
            } else if len > 1 {
                rs_insertsort_by(&mut arr[bb[i]..be[i]], key);
            }
        }
    }
}

/// In-place radix sort for a key `key_bytes` wide. Unstable, but the order of
/// equal keys is deterministic, which matters wherever later steps depend on it.
pub fn radix_sort_by_key<T: Copy, F: Fn(&T) -> u64>(arr: &mut [T], key_bytes: u32, key: F) {
    if arr.len() <= RS_MIN_SIZE {
        rs_insertsort_by(arr, &key);
    } else {
        rs_sort_by(arr, RS_MAX_BITS, (key_bytes - 1) * RS_MAX_BITS, &key);
    }
}

/// MSD in-place radix sort by x field.
/// NOT stable — elements with equal x values may be reordered.
pub fn radix_sort_128x(arr: &mut [Minimizer]) {
    if arr.len() <= RS_MIN_SIZE {
        rs_insertsort(arr);
    } else {
        rs_sort(arr, RS_MAX_BITS, (8 - 1) * RS_MAX_BITS);
    }
}

/// Radix-sort `(key, index)` ascending, then take it in reverse, so items come
/// out by key descending with ties in reverse radix order.
pub fn hit_sort_desc<T, F: Fn(&T) -> u64>(v: &mut Vec<T>, key: F) {
    if v.len() <= 1 { return; }
    let mut aux: Vec<(u64, u64)> = v.iter().enumerate().map(|(i, r)| (key(r), i as u64)).collect();
    radix_sort_128x_pair(&mut aux);
    let mut tmp: Vec<Option<T>> = v.drain(..).map(Some).collect();
    v.extend(aux.iter().rev().map(|&(_, i)| tmp[i as usize].take().unwrap()));
}

/// MSD in-place radix sort for `(u64, u64)` aux entries by `.0`. Cycle-leader
/// bucket-shuffle radix variant with a deterministic ordering of ties; used
/// for the per-read hit sort where output stability matters.
pub fn radix_sort_128x_pair(arr: &mut [(u64, u64)]) {
    if arr.len() <= RS_MIN_SIZE {
        rs_insertsort(arr);
    } else {
        rs_sort(arr, RS_MAX_BITS, (8 - 1) * RS_MAX_BITS);
    }
}

/// Perform the top-level MSD radix partition: count, prefix-sum, and in-place
/// cyclic permutation for the most significant byte. Returns (bb, be) bucket
/// boundaries so the caller can recurse into each bucket independently.
fn rs_partition_top<T: RadixKey>(arr: &mut [T], s: u32) -> ([usize; 256], [usize; 256]) {
    let m = 255u64;
    let mut bb = [0usize; 256];
    let mut be = [0usize; 256];

    for item in arr.iter() {
        let byte = ((item.radix_key() >> s) & m) as usize;
        be[byte] += 1;
    }

    // prefix sum on `be` in place
    for i in 1..256 {
      be[i] += be[i - 1];
    }
    // bucket starts = bucket ends shifted right by one
    bb[1..].copy_from_slice(&be[..255]);

    // In-place cyclic permutation (American Flag sort)
    let mut k = 0usize;
    while k < 256 {
        if bb[k] < be[k] {
            let l = ((arr[bb[k]].radix_key() >> s) & m) as usize;
            if l != k {
                let mut tmp = arr[bb[k]];
                let mut cur = l;
                loop {
                    std::mem::swap(&mut tmp, &mut arr[bb[cur]]);
                    bb[cur] += 1;
                    cur = ((tmp.radix_key() >> s) & m) as usize;
                    if cur == k { break; }
                }
                arr[bb[k]] = tmp;
                bb[k] += 1;
            } else {
                bb[k] += 1;
            }
        } else {
            k += 1;
        }
    }

    // Reset bb to proper bucket starts
    bb[0] = 0;
    bb[1..256].copy_from_slice(&be[..255]);

    (bb, be)
}

/// MSD in-place radix sort for (u64, u64) index entries.
/// Sorts by first element (hash), then by second (position) within ties.
/// The top-level partition is sequential; the 256 sub-bucket sorts run in
/// parallel via rayon. Sub-bucket recursion is sequential. Result is
/// identical to the fully sequential version.
pub fn radix_sort_pair(arr: &mut [(u64, u64)]) {
    if arr.len() <= RS_MIN_SIZE {
        rs_insertsort_pair(arr);
        return;
    }

    // Find the highest occupied byte to start partitioning at the right level.
    // This avoids wasting partition passes on empty top bytes (e.g. 30-bit hashes
    // stored in 64-bit keys have zero upper bytes).
    let mut max_key = 0u64;
    for item in arr.iter() {
        max_key |= item.radix_key();
    }
    let top_byte = if max_key == 0 { 0 } else { (63 - max_key.leading_zeros()) / 8 };
    let top_s = top_byte * RS_MAX_BITS;

    // Two-level partition to get more buckets for better parallelism.
    // Level 1: partition by the highest occupied byte.
    let (bb1, be1) = rs_partition_top(arr, top_s);

    // Level 2: sub-partition each level-1 bucket by the next byte down.
    // This gives up to 65536 independent buckets for parallel recursion.
    let next_s = top_s.saturating_sub(RS_MAX_BITS);
    let mut buckets: Vec<(usize, usize)> = Vec::new();

    if next_s > 0 || top_s == 0 {
        // Can do a second partition level
        for i in 0..256 {
            let len = be1[i] - bb1[i];
            if len <= 1 { continue; }
            let sub = &mut arr[bb1[i]..be1[i]];
            if len <= RS_MIN_SIZE {
                // Too small for radix — just record as a single bucket
                buckets.push((bb1[i], be1[i]));
            } else {
                let (bb2, be2) = rs_partition_top(sub, next_s);
                for j in 0..256 {
                    if be2[j] > bb2[j] {
                        buckets.push((bb1[i] + bb2[j], bb1[i] + be2[j]));
                    }
                }
            }
        }
    } else {
        // Only one byte of key bits — level-1 buckets are final
        for i in 0..256 {
            if be1[i] > bb1[i] {
                buckets.push((bb1[i], be1[i]));
            }
        }
    }
    let recurse_s = next_s.saturating_sub(RS_MAX_BITS);

    // Safety: buckets are non-overlapping slices of arr, so parallel mutable
    // access is safe. We use raw pointer arithmetic to avoid borrow checker issues.
    // Both bindings exist only for the parallel closure (passing the base as a
    // usize keeps it Send+Sync), so they're gated to the parallel feature.
    #[cfg(feature = "parallel")]
    let ptr = arr.as_mut_ptr();

    #[cfg(feature = "parallel")]
    let base = ptr as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        buckets.par_iter().for_each(|&(start, end)| {
            let slice = unsafe { std::slice::from_raw_parts_mut((base as *mut (u64, u64)).add(start), end - start) };
            let len = slice.len();
            if len > RS_MIN_SIZE {
                rs_sort(slice, RS_MAX_BITS, recurse_s);
            } else if len > 1 {
                rs_insertsort(slice);
            }
            // Second pass: sort positions within each hash group
            let mut i = 0;
            while i < slice.len() {
                let h = slice[i].0;
                let s = i;
                while i < slice.len() && slice[i].0 == h { i += 1; }
                if i - s > 1 {
                    slice[s..i].sort_unstable_by_key(|e| e.1);
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for &(start, end) in &buckets {
            let slice = &mut arr[start..end];
            let len = slice.len();
            if len > RS_MIN_SIZE {
                rs_sort(slice, RS_MAX_BITS, recurse_s);
            } else if len > 1 {
                rs_insertsort(slice);
            }
            let mut i = 0;
            while i < slice.len() {
                let h = slice[i].0;
                let s = i;
                while i < slice.len() && slice[i].0 == h { i += 1; }
                if i - s > 1 {
                    slice[s..i].sort_unstable_by_key(|e| e.1);
                }
            }
        }
    }
}

/// Lexicographic insertion sort for (u64, u64) — used only for small arrays in radix_sort_pair.
/// Sorts by first element, then by second on ties.
#[inline]
fn rs_insertsort_pair(arr: &mut [(u64, u64)]) {
    for i in 1..arr.len() {
        if arr[i].0 < arr[i - 1].0 || (arr[i].0 == arr[i - 1].0 && arr[i].1 < arr[i - 1].1) {
            let tmp = arr[i];
            let mut j = i;
            while j > 0 && (tmp.0 < arr[j - 1].0 || (tmp.0 == arr[j - 1].0 && tmp.1 < arr[j - 1].1)) {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = tmp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_basic() {
        let mut anchors = vec![
            Minimizer { x: 300, y: 1 },
            Minimizer { x: 100, y: 2 },
            Minimizer { x: 200, y: 3 },
        ];
        radix_sort_128x(&mut anchors);
        assert_eq!(anchors[0].x, 100);
        assert_eq!(anchors[1].x, 200);
        assert_eq!(anchors[2].x, 300);
    }

    #[test]
    fn test_radix_sort_deterministic() {
        let mut a1 = vec![
            Minimizer { x: 1000, y: 10 },
            Minimizer { x: 500, y: 20 },
            Minimizer { x: 1000, y: 30 },
            Minimizer { x: 200, y: 40 },
        ];
        let mut a2 = a1.clone();
        radix_sort_128x(&mut a1);
        radix_sort_128x(&mut a2);
        for i in 0..a1.len() {
            assert_eq!(a1[i].x, a2[i].x);
            assert_eq!(a1[i].y, a2[i].y);
        }
    }

    #[test]
    fn test_radix_sort_by_key_small_is_stable() {
        // <= RS_MIN_SIZE falls back to insertion sort, which keeps ties in input order
        let mut a: Vec<(u64, u32)> = (0..40u32).map(|i| ((i % 3) as u64, i)).collect();
        radix_sort_by_key(&mut a, 8, |e| e.0);
        let ids: Vec<u32> = a.iter().map(|e| e.1).collect();
        let mut expect: Vec<u32> = (0..40).filter(|i| i % 3 == 0).collect();
        expect.extend((0..40).filter(|i| i % 3 == 1));
        expect.extend((0..40).filter(|i| i % 3 == 2));
        assert_eq!(ids, expect);
    }

    #[test]
    fn test_radix_sort_by_key_tie_order() {
        // Fixed expected order: equal keys come out in the radix sort's in-place order.
        let mut a: Vec<(u64, u32)> = (0..100u32)
            .map(|i| (((i * 7 % 5) as u64) << 33 | ((i % 3) as u64) << 1, i))
            .collect();
        radix_sort_by_key(&mut a, 8, |e| e.0);
        let ids: Vec<u32> = a.iter().map(|e| e.1).collect();
        let expect: Vec<u32> = vec![
            0, 60, 45, 90, 30, 15, 75, 40, 85, 25, 10, 70, 55, 80, 20, 5, 65, 50, 95, 35,
            3, 63, 48, 93, 18, 33, 78, 43, 88, 13, 28, 73, 58, 83, 8, 23, 68, 53, 98, 38,
            6, 81, 21, 66, 51, 36, 96, 1, 61, 46, 16, 91, 31, 76, 41, 11, 86, 26, 71, 56,
            84, 24, 9, 69, 54, 39, 99, 4, 64, 49, 94, 34, 19, 79, 44, 89, 29, 14, 74, 59,
            42, 12, 87, 27, 72, 57, 7, 82, 22, 67, 52, 37, 97, 2, 62, 47, 17, 92, 32, 77,
        ];
        assert_eq!(ids, expect);
    }

    #[test]
    fn test_hit_sort_desc_reverses_ties() {
        // The ascending radix order is taken in reverse, so ties come out reversed
        let mut v: Vec<(u64, u32)> = vec![(5, 0), (9, 1), (5, 2), (9, 3)];
        hit_sort_desc(&mut v, |e| e.0);
        assert_eq!(v, vec![(9, 3), (9, 1), (5, 2), (5, 0)]);
    }

    #[test]
    fn test_radix_sort_large() {
        let mut anchors: Vec<Minimizer> = (0..10000)
            .map(|i| Minimizer { x: (10000 - i) as u64, y: i as u64 })
            .collect();
        radix_sort_128x(&mut anchors);
        for i in 1..anchors.len() {
            assert!(anchors[i].x >= anchors[i - 1].x, "not sorted at index {}", i);
        }
    }
}
