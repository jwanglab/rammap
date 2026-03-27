//! Seed collection: maps query minimizers to reference index hits.
//!
//! Given a set of query minimizers (from `sketch`), looks up each minimizer's
//! hash in the [`Index`] to find matching reference positions, then builds a
//! sorted anchor array of `Minimizer` structs with repacked (ref, query) coords.
//!
//! Two collection strategies:
//! - [`collect_seed_hits`] / [`collect_seed_hits_with_occ`]: radix-sort based,
//!   used by most presets.
//! - [`collect_seed_hits_heap`]: min-heap merge producing sorted output directly,
//!   used by short-read (sr) presets.
//!
//! High-frequency seeds are filtered by `mid_occ` (max occurrence threshold).
//! For runs of consecutive high-occ minimizers, [`select_seeds`] retains a
//! density-limited subset with the lowest occurrence counts rather than
//! discarding them all. Filtered seed spans are accumulated into `rep_len`
//! (repetitive length) for mapping quality estimation.

use crate::align::sketch::Minimizer;
use crate::align::index::Index;
use crate::align::map::{MapOptions, AlignFlags};
use crate::align::sketch::{SEED_SEG_SHIFT, SEED_SELF};

// Hash functions
#[inline]
pub(crate) fn hash64(mut key: u64) -> u64 {
    key = (!key).wrapping_add(key << 21); // key = (key << 21) - key - 1
    key = key ^ (key >> 24);
    key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key.wrapping_add(key << 31);
    key
}

#[inline]
fn wang_hash(mut key: u32) -> u32 {
    key = key.wrapping_add(!(key << 15));
    key ^= key >> 10;
    key = key.wrapping_add(key << 3);
    key ^= key >> 6;
    key = key.wrapping_add(!(key << 11));
    key ^= key >> 16;
    key
}

#[inline]
fn x31_hash_string(s: &[u8]) -> u32 {
    if s.is_empty() {
        return 0;
    }
    let mut h = s[0] as u32;
    for &b in &s[1..] {
        h = (h << 5).wrapping_sub(h).wrapping_add(b as u32);
    }
    h
}

pub(crate) fn compute_read_hash(qname: &str, qlen: usize, seed: u32, flags: AlignFlags) -> u32 {
    let mut hash = if flags.contains(AlignFlags::NO_HASH_NAME) { 0 } else { x31_hash_string(qname.as_bytes()) };
    hash ^= wang_hash(qlen as u32).wrapping_add(wang_hash(seed));
    wang_hash(hash)
}

/// Seed info collected for mm_select_seeds filtering
struct SeedInfo {
    query_pos: u32,     // query position (query_pos >> 1)
    q_span: u32,    // k-mer span
    hit_count: usize,       // number of index hits
    is_tandem: bool,
    is_filtered: bool,      // true = filtered out
    mi_idx: usize,  // index into q_minimizers
}

/// Select seeds by frequency threshold
/// For runs of consecutive high-occ minimizers, keeps the ones with lowest occurrence.
fn select_seeds(seeds: &mut [SeedInfo], qlen: usize, max_occ: usize, max_max_occ: usize, dist: i32) {
    let n = seeds.len();
    if n <= 1 { return; }

    // Check if any seeds are high-occ
    let mut has_high = false;
    for s in seeds.iter() {
        if s.hit_count > max_occ { has_high = true; break; }
    }
    if !has_high { return; }

    let mut last0: i64 = -1; // index of last non-high-occ seed

    for i in 0..=n {
        if i == n || seeds[i].hit_count <= max_occ {
            let run_start = (last0 + 1) as usize;
            if i > run_start {
                // High-occ run from run_start..i
                let ps = if last0 < 0 { 0 } else { (seeds[last0 as usize].query_pos >> 1) as i32 };
                let pe = if i == n { qlen as i32 } else { (seeds[i].query_pos >> 1) as i32 };
                let max_high_occ = ((pe - ps) as f64 / dist as f64 + 0.499) as usize;

                if max_high_occ > 0 {
                    const MAX_MAX_HIGH_OCC: usize = 128;
                    let max_high_occ = max_high_occ.min(MAX_MAX_HIGH_OCC);

                    // Use a max-heap to keep the max_high_occ seeds with lowest 'n'
                    // Heap stores (n, index)
                    // comparison: higher n at top, ties broken by higher j (seed index)
                    let mut heap: Vec<(usize, usize)> = Vec::with_capacity(max_high_occ + 1);
                    for (j, seed) in seeds[run_start..i].iter().enumerate() {
                        let j = j + run_start;
                        if heap.len() < max_high_occ {
                            heap.push((seed.hit_count, j));
                            // Sift up — compare full (n, j) tuple
                            let mut k = heap.len() - 1;
                            while k > 0 {
                                let parent = (k - 1) / 2;
                                if heap[k] > heap[parent] {
                                    heap.swap(k, parent);
                                    k = parent;
                                } else { break; }
                            }
                        } else if seed.hit_count < heap[0].0 {
                            // Replace top (largest n) with this seed (smaller n)
                            heap[0] = (seed.hit_count, j);
                            // Sift down — compare full (n, j) tuple
                            let mut k = 0;
                            loop {
                                let left = 2 * k + 1;
                                let right = 2 * k + 2;
                                let mut largest = k;
                                if left < heap.len() && heap[left] > heap[largest] { largest = left; }
                                if right < heap.len() && heap[right] > heap[largest] { largest = right; }
                                if largest != k { heap.swap(k, largest); k = largest; }
                                else { break; }
                            }
                        }
                    }

                    // Mark heap contents as flt=true (will be XORed below)
                    for &(_, idx) in &heap {
                        seeds[idx].is_filtered = true;
                    }
                }

                // XOR all seeds in the run — inverts: selected (flt=true) → false, others → true
                for seed in &mut seeds[run_start..i] {
                    seed.is_filtered = !seed.is_filtered;
                }
                // Seeds with n > max_max_occ are always filtered
                for seed in &mut seeds[run_start..i] {
                    if seed.hit_count > max_max_occ {
                        seed.is_filtered = true;
                    }
                }
            }
            last0 = i as i64;
        }
    }
}

/// Matched seed after filtering
/// Only non-filtered seeds are retained; filtered seeds contribute to rep_len.
struct MatchedSeed {
    q_span: u32,    // k-mer span
    is_tandem: bool,
    mi_idx: usize,  // index into q_minimizers
    seg_id: u64,    // segment index (from minimizer rid field, for multi-segment)
}

/// Common seed collection + filtering logic shared by both heap and non-heap paths.
/// Collect anchor matches from index hits.
///
/// Returns (matched_seeds, n_a, rep_len) where n_a is total anchor count.
fn collect_anchor_matches(
    opt: &MapOptions,
    mi: &Index,
    qlen: usize,
    q_minimizers: &[Minimizer],
    mini_pos: &mut Vec<u64>,
    max_occ: usize,
) -> (Vec<MatchedSeed>, usize, usize) {
    mini_pos.clear();

    // Pre-compute tandem flags
    let n_qm = q_minimizers.len();
    let mut is_tandem = vec![false; n_qm];
    for i in 0..n_qm {
        if i > 0 && (q_minimizers[i].x >> 8) == (q_minimizers[i - 1].x >> 8) {
            is_tandem[i] = true;
        }
        if i + 1 < n_qm && (q_minimizers[i].x >> 8) == (q_minimizers[i + 1].x >> 8) {
            is_tandem[i] = true;
        }
    }

    // Phase 1: Collect seed info (like mm_seed_collect_all)
    let mut seeds: Vec<SeedInfo> = Vec::with_capacity(n_qm);
    for (mi_idx, m) in q_minimizers.iter().enumerate() {
        let hash = m.x >> 8;
        let q_span = (m.x & 0xFF) as u32;
        let q_pos = m.y as u32; // includes strand bit
        let n = mi.get(hash).map_or(0, |h| h.len());
        if n == 0 { continue; }
        seeds.push(SeedInfo {
            query_pos: q_pos,
            q_span,
            hit_count: n,
            is_tandem: is_tandem[mi_idx],
            is_filtered: false,
            mi_idx,
        });
    }

    // Phase 2: Apply mm_select_seeds or simple filtering
    if opt.seeding.occ_dist > 0 && opt.seeding.max_max_occ > max_occ {
        select_seeds(&mut seeds, qlen, max_occ, opt.seeding.max_max_occ, opt.seeding.occ_dist);
    } else {
        for s in seeds.iter_mut() {
            if s.hit_count > max_occ { s.is_filtered = true; }
        }
    }

    // Phase 3: Partition filtered vs non-filtered, compute rep_len and n_a
    let mut matched_seeds: Vec<MatchedSeed> = Vec::with_capacity(seeds.len());
    let mut n_a: usize = 0;
    let mut rep_len: usize = 0;
    let mut rep_st: usize = 0;
    let mut rep_en: usize = 0;

    for seed in &seeds {
        if seed.is_filtered {
            let en = (seed.query_pos >> 1) as usize + 1;
            let st = en.saturating_sub(seed.q_span as usize);
            if st > rep_en {
                rep_len += rep_en - rep_st;
                rep_st = st;
                rep_en = en;
            } else {
                rep_en = en;
            }
            continue;
        }

        n_a += seed.hit_count;
        mini_pos.push(((seed.q_span as u64) << 32) | ((seed.query_pos >> 1) as u64));
        matched_seeds.push(MatchedSeed {
            q_span: seed.q_span,
            is_tandem: seed.is_tandem,
            mi_idx: seed.mi_idx,
            seg_id: (q_minimizers[seed.mi_idx].y >> 32),
        });
    }

    rep_len += rep_en - rep_st;

    (matched_seeds, n_a, rep_len)
}

/// Non-heap seed collection with explicit max_occ parameter (for re-chaining path).
pub(crate) fn collect_seed_hits_with_occ(
    opt: &MapOptions,
    mi: &Index,
    qlen: usize,
    q_minimizers: &[Minimizer],
    anchors: &mut Vec<Minimizer>,
    mini_pos: &mut Vec<u64>,
    max_occ: usize,
    qname: Option<&str>,
) -> usize {
    anchors.clear();
    let (matched_seeds, n_a, rep_len) = collect_anchor_matches(opt, mi, qlen, q_minimizers, mini_pos, max_occ);
    anchors.reserve(n_a);

    let seed_tandem: u64 = crate::align::sketch::SEED_TANDEM;
    let skip_flags = opt.flags & (AlignFlags::NO_DIAG | AlignFlags::NO_DUAL);

    for seed in &matched_seeds {
        let m = &q_minimizers[seed.mi_idx];
        let hash = m.x >> 8;
        let q_span = seed.q_span as usize;
        let q_pos = ((m.y as u32) >> 1) as usize;
        let tandem = seed.is_tandem;

        if let Some(hits) = mi.get(hash) {
            for &(_, r_packed) in hits {
                // skip_seed logic
                let mut is_self = false;
                if let Some(qn) = qname && !skip_flags.is_empty() {
                    let rid = (r_packed >> 32) as usize;
                    let tname = &mi.seqs[rid].name;
                    let tlen = mi.seqs[rid].len;
                    let cmp = qn.cmp(tname.as_str());
                    if opt.flags.contains(AlignFlags::NO_DIAG) && cmp == std::cmp::Ordering::Equal && tlen == qlen {
                        let r_pos_raw = (r_packed as u32) >> 1;
                        let q_pos_raw = (m.y as u32) >> 1;
                        if r_pos_raw == q_pos_raw { continue; } // skip diagonal
                        if (r_packed & 1) == (m.y & 1) { is_self = true; } // same strand
                    }
                    if opt.flags.contains(AlignFlags::NO_DUAL) && cmp == std::cmp::Ordering::Greater { continue; }
                }

                let r_pos = (r_packed as u32) >> 1;
                let r_strand = r_packed & 1;
                let q_strand = m.y & 1;
                let is_rev = q_strand != r_strand;
                let q_pos_u32 = q_pos as u32;

                let x: u64;
                let mut y: u64;

                if !is_rev {
                    x = (r_packed & 0xFFFFFFFF00000000) | (r_pos as u64);
                    y = (q_span as u64) << 32 | (q_pos_u32 as u64);
                } else if opt.flags.contains(AlignFlags::QSTRAND) {
                    // qstrand mode: keep query pos, reverse reference pos (map.c:189-195)
                    let rid = (r_packed >> 32) as usize;
                    let tlen = mi.seqs[rid].len as u64;
                    let rpos_rev = tlen.wrapping_sub(r_pos as u64 + 1).wrapping_sub(1).wrapping_add(q_span as u64);
                    x = (1u64 << 63) | (r_packed & 0xFFFFFFFF00000000) | rpos_rev;
                    y = (q_span as u64) << 32 | (q_pos_u32 as u64);
                } else {
                    x = (1u64 << 63) | (r_packed & 0xFFFFFFFF00000000) | (r_pos as u64);
                    let q_pos_rev = (qlen as u64)
                        .wrapping_sub(q_pos_u32 as u64)
                        .wrapping_add(q_span as u64)
                        .wrapping_sub(2);
                    y = (q_span as u64) << 32 | q_pos_rev;
                }

                if tandem { y |= seed_tandem; }
                if is_self { y |= SEED_SELF; }
                y |= seed.seg_id << SEED_SEG_SHIFT;
                anchors.push(Minimizer { x, y });
            }
        }
    }

    crate::align::sort::radix_sort_128x(anchors);

    rep_len
}

pub fn collect_seed_hits(
    opt: &MapOptions,
    mi: &Index,
    qlen: usize,
    q_minimizers: &[Minimizer],
    anchors: &mut Vec<Minimizer>,
    mini_pos: &mut Vec<u64>,
    qname: Option<&str>,
) -> usize {
    collect_seed_hits_with_occ(opt, mi, qlen, q_minimizers, anchors, mini_pos, opt.seeding.mid_occ, qname)
}

// --- Heap-based seed collection ---

/// Min-heap sift-down. Matches ksort.h with heap_lt(a, b) = a.x > b.x.
/// Smaller x values float to root.
#[inline]
fn anchor_heap_down(heap: &mut [(u64, u64)], start: usize, n: usize) {
    let mut i = start;
    let tmp = heap[i];
    loop {
        let mut k = (i << 1) + 1; // left child
        if k >= n { break; }
        // Pick the child with smaller .0
        if k + 1 < n && heap[k].0 > heap[k + 1].0 { k += 1; }
        // If child has strictly larger .0, stop
        if heap[k].0 > tmp.0 { break; }
        heap[i] = heap[k];
        i = k;
    }
    heap[i] = tmp;
}

/// Build min-heap from unsorted array. Matches ksort.h ks_heapmake.
fn anchor_heap_make(heap: &mut [(u64, u64)]) {
    let n = heap.len();
    if n <= 1 { return; }
    for i in (0..n / 2).rev() {
        anchor_heap_down(heap, i, n);
    }
}

/// Heap-based seed collection.
/// Uses a min-heap to produce anchors in reference position order without radix sort.
/// Used when HEAP_SORT flag is set (splice:sr, sr presets).
pub fn collect_seed_hits_heap(
    opt: &MapOptions,
    mi: &Index,
    qlen: usize,
    q_minimizers: &[Minimizer],
    anchors: &mut Vec<Minimizer>,
    mini_pos: &mut Vec<u64>,
    max_occ: usize,
    qname: Option<&str>,
) -> usize {
    anchors.clear();
    let (matched_seeds, n_a, rep_len) = collect_anchor_matches(opt, mi, qlen, q_minimizers, mini_pos, max_occ);
    if n_a == 0 { return rep_len; }

    anchors.resize(n_a, Minimizer { x: 0, y: 0 });

    // Cache hit slices for each matched seed (equivalent to mm_seed_t.cr pointers)
    let hit_slices: Vec<Option<&[(u64, u64)]>> = matched_seeds.iter().map(|seed| {
        let hash = q_minimizers[seed.mi_idx].x >> 8;
        mi.get(hash)
    }).collect();

    // Initialize heap: (r_packed, seed_idx << 32 | hit_counter)
    let mut heap: Vec<(u64, u64)> = Vec::with_capacity(matched_seeds.len());
    for (i, hits) in hit_slices.iter().enumerate() {
        if let Some(h) = hits && !h.is_empty() {
            heap.push((h[0].1, (i as u64) << 32)); // h[0].1 = r_packed
        }
    }
    anchor_heap_make(&mut heap);

    let mut n_for: usize = 0;
    let mut n_rev: usize = 0;
    let seed_tandem: u64 = crate::align::sketch::SEED_TANDEM;
    let skip_flags = opt.flags & (AlignFlags::NO_DIAG | AlignFlags::NO_DUAL);

    while !heap.is_empty() {
        let r = heap[0].0;
        let seed_idx = (heap[0].1 >> 32) as usize;
        let hit_idx = (heap[0].1 & 0xFFFFFFFF) as usize;
        let seed = &matched_seeds[seed_idx];
        let m = &q_minimizers[seed.mi_idx];

        let r_pos = (r as u32) >> 1;
        let is_rev = (m.y & 1) != (r & 1);

        // skip_seed logic
        let mut is_self = false;
        let mut skip = false;
        if let Some(qn) = qname && !skip_flags.is_empty() {
            let rid = (r >> 32) as usize;
            let tname = &mi.seqs[rid].name;
            let tlen = mi.seqs[rid].len;
            let cmp = qn.cmp(tname.as_str());
            if opt.flags.contains(AlignFlags::NO_DIAG) && cmp == std::cmp::Ordering::Equal && tlen == qlen {
                let r_pos_raw = (r as u32) >> 1;
                let q_pos_raw = (m.y as u32) >> 1;
                if r_pos_raw == q_pos_raw { skip = true; }
                else if (r & 1) == (m.y & 1) { is_self = true; }
            }
            if !skip && opt.flags.contains(AlignFlags::NO_DUAL) && cmp == std::cmp::Ordering::Greater { skip = true; }
        }

        if !skip {
            if !is_rev {
                let p = &mut anchors[n_for];
                p.x = (r & 0xFFFFFFFF00000000) | (r_pos as u64);
                p.y = (seed.q_span as u64) << 32 | ((m.y as u32) >> 1) as u64;
                if seed.is_tandem { p.y |= seed_tandem; }
                if is_self { p.y |= SEED_SELF; }
                p.y |= seed.seg_id << SEED_SEG_SHIFT;
                n_for += 1;
            } else if opt.flags.contains(AlignFlags::QSTRAND) {
                // qstrand mode: keep query pos, reverse reference pos
                n_rev += 1;
                let p = &mut anchors[n_a - n_rev];
                let rid = (r >> 32) as usize;
                let tlen = mi.seqs[rid].len as u64;
                let rpos_rev = tlen.wrapping_sub(r_pos as u64 + 1).wrapping_sub(1).wrapping_add(seed.q_span as u64);
                p.x = (1u64 << 63) | (r & 0xFFFFFFFF00000000) | rpos_rev;
                p.y = (seed.q_span as u64) << 32 | ((m.y as u32) >> 1) as u64;
            } else {
                n_rev += 1;
                let p = &mut anchors[n_a - n_rev];
                p.x = (1u64 << 63) | (r & 0xFFFFFFFF00000000) | (r_pos as u64);
                let q_pos_rev = (qlen as u64)
                    .wrapping_sub(((m.y as u32) >> 1) as u64)
                    .wrapping_add(seed.q_span as u64)
                    .wrapping_sub(2);
                p.y = (seed.q_span as u64) << 32 | q_pos_rev;
                if seed.is_tandem { p.y |= seed_tandem; }
                if is_self { p.y |= SEED_SELF; }
                p.y |= seed.seg_id << SEED_SEG_SHIFT;
            }
        }

        // Update heap: advance to next hit or remove seed
        let hits = hit_slices[seed_idx].unwrap();
        let hs = heap.len();
        if hit_idx < hits.len() - 1 {
            let next_r = hits[hit_idx + 1].1;
            heap[0] = (next_r, ((seed_idx as u64) << 32) | ((hit_idx + 1) as u64));
        } else {
            heap[0] = heap[hs - 1];
            heap.truncate(hs - 1);
        }
        let hn = heap.len();
        if hn > 0 {
            anchor_heap_down(&mut heap, 0, hn);
        }
    }

    // Reverse the reverse-strand section (accumulated in descending order)
    if n_rev > 0 {
        anchors[n_a - n_rev..n_a].reverse();
    }

    // Compact: move reverse section right after forward section
    if n_a > n_for + n_rev {
        for i in 0..n_rev {
            anchors[n_for + i] = anchors[n_a - n_rev + i];
        }
    }
    anchors.truncate(n_for + n_rev);

    // No radix_sort_128x needed — anchors are already in sorted order

    rep_len
}

/// Apply mm_filter_minimizers_by_occ on minimizer vector (shared by single and multi-segment paths).
pub(crate) fn filter_minimizers_by_occ(minimizers: &mut Vec<Minimizer>, mid_occ: usize, q_occ_frac: f32) {
    let n = minimizers.len();
    let q_occ_max = mid_occ;
    let q_frac_thresh = (n as f32 * q_occ_frac) as usize;

    let mut sorted: Vec<(u64, usize)> = minimizers.iter().enumerate()
        .map(|(i, m)| (m.x, i)).collect();
    sorted.sort_unstable();

    let mut to_remove = vec![false; n];
    let mut st = 0;
    for i in 0..=sorted.len() {
        if i == sorted.len() || sorted[i].0 != sorted[st].0 {
            let cnt = i - st;
            if cnt > q_occ_max && cnt > q_frac_thresh {
                for j in st..i {
                    to_remove[sorted[j].1] = true;
                }
            }
            st = i;
        }
    }
    let mut j = 0;
    for i in 0..n {
        if !to_remove[i] {
            minimizers[j] = minimizers[i];
            j += 1;
        }
    }
    minimizers.truncate(j);
}
