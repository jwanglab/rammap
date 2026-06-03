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
    for i in 1..arr.len() {
        if arr[i].radix_key() < arr[i - 1].radix_key() {
            let tmp = arr[i];
            let mut j = i;
            while j > 0 && tmp.radix_key() < arr[j - 1].radix_key() {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = tmp;
        }
    }
}

/// MSD in-place radix sort
fn rs_sort<T: RadixKey>(arr: &mut [T], n_bits: u32, s: u32) {
    let size = 1usize << n_bits; // 256
    let m = (size - 1) as u64;

    // bb[i] = current write position for bucket i (advances during permutation)
    // be[i] = end boundary for bucket i (fixed after prefix sum)
    let mut bb = [0usize; 256];
    let mut be = [0usize; 256];

    // Count elements per bucket
    for item in arr.iter() {
        let byte = ((item.radix_key() >> s) & m) as usize;
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
            let l = ((arr[bb[k]].radix_key() >> s) & m) as usize;
            if l != k {
                // Element at k's front belongs to bucket l — start swap cycle
                let mut tmp = arr[bb[k]];
                let mut cur = l;
                loop {
                    // Place tmp into bucket cur, pick up what was there
                    std::mem::swap(&mut tmp, &mut arr[bb[cur]]);
                    bb[cur] += 1;
                    // Where does the picked-up element belong?
                    cur = ((tmp.radix_key() >> s) & m) as usize;
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
                rs_sort(&mut arr[bb[i]..be[i]], n_bits, next_s);
            } else if len > 1 {
                rs_insertsort(&mut arr[bb[i]..be[i]]);
            }
        }
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
