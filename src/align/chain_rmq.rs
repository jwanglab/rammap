//! RMQ-based anchor chaining
//!
//! An alternative to DP chaining that uses range-minimum queries on an
//! augmented treap to find optimal predecessors in O(log n) per anchor instead
//! of scanning backwards through a distance window. This is more efficient for
//! presets with large bandwidths such as `lr:hqae` and `asm`.
//!
//! The treap is keyed by query position and augmented with subtree-minimum
//! priority (negative DP score adjusted by position). An optional inner tree
//! with a tighter distance window (`rmq_inner_dist`) handles nearby anchors
//! that the RMQ might miss due to gap penalty interactions.
//!
//! Backtracking, chain extraction, and output format (score|count descriptors
//! plus reordered anchors) are shared with the DP chaining module via
//! [`chain_backtrack`].

use crate::align::sketch::Minimizer;
use crate::align::chain::{fast_log2, chain_backtrack};
use crate::align::sort::radix_sort_128x;
use crate::align::map::{ChainingParams, ChainingBuffers};

const NIL: u32 = u32::MAX;

/// Arena-based treap with augmented subtree-min priority.
/// Gives O(log n) expected for insert, erase, and range-minimum query.
/// Augmented red-black tree with range-minimum query support.
struct RmqTree {
    nodes: Vec<TreapNode>,
    root: u32,
    size: usize,
    rng: u64, // xorshift64 state
}

#[derive(Clone)]
struct TreapNode {
    key_y: i32,
    key_i: usize,
    pri: f64,       // node's priority (negative score, lower = better)
    sub_min: f64,   // minimum pri in subtree rooted at this node
    heap: u32,      // random heap priority for treap balancing
    left: u32,
    right: u32,
}

impl RmqTree {
    fn new() -> Self {
        RmqTree {
            nodes: Vec::with_capacity(256),
            root: NIL,
            size: 0,
            rng: 0x12345678_9abcdef0,
        }
    }

    #[inline]
    fn xorshift(&mut self) -> u32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x as u32
    }

    #[inline]
    fn update_min(&mut self, idx: u32) {
        if idx == NIL { return; }
        let i = idx as usize;
        let mut m = self.nodes[i].pri;
        let l = self.nodes[i].left;
        let r = self.nodes[i].right;
        if l != NIL && self.nodes[l as usize].sub_min < m {
            m = self.nodes[l as usize].sub_min;
        }
        if r != NIL && self.nodes[r as usize].sub_min < m {
            m = self.nodes[r as usize].sub_min;
        }
        self.nodes[i].sub_min = m;
    }

    /// Split tree rooted at `t` into (< key, >= key). Returns (left_root, right_root).
    fn split(&mut self, t: u32, key_y: i32, key_i: usize) -> (u32, u32) {
        if t == NIL { return (NIL, NIL); }
        let ti = t as usize;
        if (self.nodes[ti].key_y, self.nodes[ti].key_i) < (key_y, key_i) {
            let r = self.nodes[ti].right;
            let (lr, rr) = self.split(r, key_y, key_i);
            self.nodes[ti].right = lr;
            self.update_min(t);
            (t, rr)
        } else {
            let l = self.nodes[ti].left;
            let (ll, rl) = self.split(l, key_y, key_i);
            self.nodes[ti].left = rl;
            self.update_min(t);
            (ll, t)
        }
    }

    /// Merge two trees where all keys in `l` < all keys in `r`.
    fn merge(&mut self, l: u32, r: u32) -> u32 {
        if l == NIL { return r; }
        if r == NIL { return l; }
        if self.nodes[l as usize].heap > self.nodes[r as usize].heap {
            let lr = self.nodes[l as usize].right;
            self.nodes[l as usize].right = self.merge(lr, r);
            self.update_min(l);
            l
        } else {
            let rl = self.nodes[r as usize].left;
            self.nodes[r as usize].left = self.merge(l, rl);
            self.update_min(r);
            r
        }
    }

    fn insert_elem(&mut self, y: i32, i: usize, pri: f64) {
        let heap = self.xorshift();
        let idx = self.nodes.len() as u32;
        self.nodes.push(TreapNode {
            key_y: y, key_i: i, pri, sub_min: pri,
            heap, left: NIL, right: NIL,
        });
        let (l, r) = self.split(self.root, y, i);
        let m = self.merge(l, idx);
        self.root = self.merge(m, r);
        self.size += 1;
    }

    fn erase(&mut self, y: i32, i: usize) -> bool {
        self.size -= 1;
        self.root = self.erase_impl(self.root, y, i);
        true
    }

    fn erase_impl(&mut self, t: u32, y: i32, i: usize) -> u32 {
        if t == NIL { self.size += 1; return NIL; } // not found, undo size decrement
        let ti = t as usize;
        let k = (self.nodes[ti].key_y, self.nodes[ti].key_i);
        if k == (y, i) {
            let l = self.nodes[ti].left;
            let r = self.nodes[ti].right;
            return self.merge(l, r);
        }
        if (y, i) < k {
            let l = self.nodes[ti].left;
            self.nodes[ti].left = self.erase_impl(l, y, i);
        } else {
            let r = self.nodes[ti].right;
            self.nodes[ti].right = self.erase_impl(r, y, i);
        }
        self.update_min(t);
        t
    }

    #[inline]
    fn len(&self) -> usize {
        self.size
    }

    /// Range minimum query: O(log n) using subtree-min augmentation.
    /// Finds element with minimum priority in the range
    /// y > lo_y (exclusive), y < hi_y (all), y == hi_y only if i == 0.
    fn rmq(&self, lo_y: i32, hi_y: i32) -> Option<(i32, usize, f64)> {
        let mut best_pri = f64::MAX;
        let mut best: Option<(i32, usize, f64)> = None;
        self.rmq_impl(self.root, lo_y, hi_y, &mut best_pri, &mut best);
        best
    }

    fn rmq_impl(&self, t: u32, lo_y: i32, hi_y: i32, best_pri: &mut f64, best: &mut Option<(i32, usize, f64)>) {
        if t == NIL { return; }
        let ti = t as usize;
        // Prune: if subtree's min priority can't beat current best, skip
        if self.nodes[ti].sub_min >= *best_pri { return; }

        let ky = self.nodes[ti].key_y;
        let ki = self.nodes[ti].key_i;

        // Check if this node is in range: y > lo_y, and (y < hi_y || (y == hi_y && i == 0))
        let in_range = ky > lo_y && (ky < hi_y || (ky == hi_y && ki == 0));
        if in_range && self.nodes[ti].pri < *best_pri {
            *best_pri = self.nodes[ti].pri;
            *best = Some((ky, ki, self.nodes[ti].pri));
        }

        // Search left subtree if it could contain in-range elements
        if ky > lo_y {
            self.rmq_impl(self.nodes[ti].left, lo_y, hi_y, best_pri, best);
        }
        // Search right subtree if it could contain in-range elements
        if ky < hi_y || (ky == hi_y && ki == 0) {
            self.rmq_impl(self.nodes[ti].right, lo_y, hi_y, best_pri, best);
        }
    }

    /// Create a reverse in-order iterator over elements with key_y <= start_y.
    /// Yields elements in descending (key_y, key_i) order. O(log n) init, O(1) per step.
    fn iter_rev_le(&self, start_y: i32) -> RmqRevIter<'_> {
        let mut stack = Vec::with_capacity(32);
        let mut t = self.root;
        while t != NIL {
            let node = &self.nodes[t as usize];
            if node.key_y <= start_y {
                stack.push(t);
                t = node.right;
            } else {
                t = node.left;
            }
        }
        RmqRevIter { tree: self, stack }
    }
}

struct RmqRevIter<'a> {
    tree: &'a RmqTree,
    stack: Vec<u32>,
}

impl<'a> Iterator for RmqRevIter<'a> {
    type Item = (i32, usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let t = self.stack.pop()?;
        let node = &self.tree.nodes[t as usize];
        let result = (node.key_y, node.key_i, node.pri);
        // Explore left subtree: push rightmost path (all < current, hence in range)
        let mut child = node.left;
        while child != NIL {
            self.stack.push(child);
            child = self.tree.nodes[child as usize].right;
        }
        Some(result)
    }
}

/// Simplified score computation for RMQ chaining (matches comput_sc_simple)
/// Returns (score, is_exact, width)
#[inline(always)]
fn compute_chain_score_simple(
    ai: &Minimizer,
    aj: &Minimizer,
    chn_pen_gap: f32,
    chn_pen_skip: f32,
) -> (i32, bool, i32) {
    let query_diff = ai.query_pos().wrapping_sub(aj.query_pos());
    let ref_diff = ai.ref_pos().wrapping_sub(aj.ref_pos());
    let gap_width = if ref_diff > query_diff { ref_diff - query_diff } else { query_diff - ref_diff };
    let min_diff = if ref_diff < query_diff { ref_diff } else { query_diff };
    let q_span = aj.query_span();

    let mut sc = if q_span < min_diff { q_span } else { min_diff };
    let exact = gap_width == 0 && min_diff <= q_span;

    if gap_width > 0 || query_diff > q_span {
        let lin_pen = chn_pen_gap * (gap_width as f32) + chn_pen_skip * (min_diff as f32);
        let log_pen = if gap_width >= 1 { fast_log2((gap_width + 1) as f32) } else { 0.0f32 };
        sc -= (lin_pen + 0.5f32 * log_pen) as i32;
    }

    (sc, exact, gap_width)
}

/// RMQ-based chaining (port of mg_lchain_rmq)
///
/// This is more efficient than DP chaining for large bandwidths because it uses
/// O(log n) range maximum queries instead of O(n) iteration.
///
/// Arguments:
/// - opt: chaining parameters (max_gap, rmq_inner_dist, bandwidth, max_chain_skip, etc.)
/// - a: input anchors (modified in place)
/// - ctx: reusable map context (provides DP buffers)
///
/// Returns: (u, a_new) where u is chain scores/counts and a_new is the reordered anchors
pub fn chain_anchors_rmq(
    opt: &ChainingParams,
    a: &mut [Minimizer],
    ctx: &mut ChainingBuffers,
) -> (Vec<u64>, Vec<Minimizer>) {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let bw = opt.bandwidth;
    let max_dist = if opt.max_gap < bw { bw } else { opt.max_gap };
    let max_dist_inner = if opt.rmq_inner_dist < 0 { 0 }
        else if opt.rmq_inner_dist > max_dist { max_dist }
        else { opt.rmq_inner_dist };
    let max_drop = bw;

    let mut predecessors = std::mem::take(&mut ctx.predecessors);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i64);
    scores.resize(n, 0i32);
    peak_scores.resize(n, 0i32);
    // visited uses sentinel comparison (visited[j] == i) so must be zeroed
    visited.clear(); visited.resize(n, 0i32);

    let mut root = RmqTree::new();
    let mut root_inner = if max_dist_inner > 0 { Some(RmqTree::new()) } else { None };

    let mut window_start: usize = 0;
    let mut inner_window_start: usize = 0;
    let mut insert_from: usize = 0;
    let mut _mmax_f = 0;

    for i in 0..n {
        let mut best_predecessor: i64 = -1;
        let q_span = a[i].query_span();
        let mut best_score = q_span;

        // Add in-range anchors (when position changes)
    
        if insert_from < i && a[insert_from].x != a[i].x {
            for j in insert_from..i {
                let pri = -(scores[j] as f64 + 0.5 * opt.chn_pen_gap as f64 * ((a[j].ref_pos() + a[j].query_pos()) as f64));
                root.insert_elem(a[j].query_pos(), j, pri);
                if let Some(inner) = &mut root_inner {
                    inner.insert_elem(a[j].query_pos(), j, pri);
                }
            }
            insert_from = i;
        }

        // Remove anchors out of range from root
        while window_start < i && (
            a[i].ref_id_strand() != a[window_start].ref_id_strand() ||
            a[i].x > a[window_start].x + max_dist as u64 ||
            root.len() > opt.rmq_size_cap as usize
        ) {
            root.erase(a[window_start].query_pos(), window_start);
            window_start += 1;
        }

        // Remove anchors out of range from root_inner
        if let Some(inner) = &mut root_inner {
            while inner_window_start < i && (
                a[i].ref_id_strand() != a[inner_window_start].ref_id_strand() ||
                a[i].x > a[inner_window_start].x + max_dist_inner as u64 ||
                inner.len() > opt.rmq_size_cap as usize
            ) {
                inner.erase(a[inner_window_start].query_pos(), inner_window_start);
                inner_window_start += 1;
            }
        }

        // RMQ query on root tree
        let lo_y = a[i].query_pos() - max_dist;
        let hi_y = a[i].query_pos();

        if let Some((_, q_i, _)) = root.rmq(lo_y, hi_y) {
            let j = q_i;
            let (sc, exact, width) = compute_chain_score_simple(&a[i], &a[j], opt.chn_pen_gap, opt.chn_pen_skip);
            let total_sc = scores[j] + sc;
            if width <= bw && total_sc > best_score {
                best_score = total_sc;
                best_predecessor = j as i64;
            }

            // If not exact match, also search inner tree for close matches
            if !exact {
                if let Some(ref inner) = root_inner {
                    if a[i].query_pos() > 0 {
                        let mut skip_count = 0;
                        for (q_y, q_i, _q_pri) in inner.iter_rev_le(a[i].query_pos() - 1) {
                            if q_y < a[i].query_pos() - max_dist_inner {
                                break;
                            }
                            let j = q_i;
                            let (sc, _, width) = compute_chain_score_simple(&a[i], &a[j], opt.chn_pen_gap, opt.chn_pen_skip);
                            if width <= bw {
                                let total_sc = scores[j] + sc;
                                if total_sc > best_score {
                                    best_score = total_sc;
                                    best_predecessor = j as i64;
                                    if skip_count > 0 { skip_count -= 1; }
                                } else if visited[j] == i as i32 {
                                    skip_count += 1;
                                    if skip_count > opt.max_chain_skip {
                                        break;
                                    }
                                }
                                if predecessors[j] >= 0 {
                                    visited[predecessors[j] as usize] = i as i32;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Set max
        scores[i] = best_score;
        predecessors[i] = best_predecessor;
        peak_scores[i] = if best_predecessor >= 0 && peak_scores[best_predecessor as usize] > best_score { peak_scores[best_predecessor as usize] } else { best_score };
        if _mmax_f < best_score { _mmax_f = best_score; }

    }

    // Backtrack to extract chains
    let (u, n_u, n_v) = chain_backtrack(n, &scores, &predecessors, &mut peak_scores, &mut visited, opt.min_cnt, opt.min_chain_score, max_drop);

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited;
        return (Vec::new(), Vec::new());
    }

    // compact_a logic: reorder anchors according to chains (matching lchain.c:78-111)
    // Step 1: Write chain anchors to b[] in forward order
    let mut b: Vec<Minimizer> = Vec::with_capacity(n_v);
    let mut k = 0;
    for &u_val in &u[..n_u] {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        let k0 = k;
        for j in 0..ni {
            let idx = peak_scores[k0 + (ni - j - 1)] as usize;
            b.push(a[idx]);
            k += 1;
        }
    }

    // Step 2: Sort chains by target position of first anchor (lchain.c:93-107)
    let mut w: Vec<Minimizer> = Vec::with_capacity(n_u);
    let mut k_pos = 0usize;
    for (i, &u_val) in u[..n_u].iter().enumerate() {
        let ni = (u_val & 0xFFFFFFFF) as usize;
        w.push(Minimizer {
            x: b[k_pos].x,
            y: ((k_pos as u64) << 32) | (i as u64),
        });
        k_pos += ni;
    }
    radix_sort_128x(&mut w);

    // Step 3: Reorder u[] and anchors according to sorted order
    let mut u2: Vec<u64> = Vec::with_capacity(n_u);
    let mut b2: Vec<Minimizer> = Vec::with_capacity(n_v);
    for &w_val in &w[..n_u] {
        let j = (w_val.y & 0xFFFFFFFF) as usize; // original chain index
        let offset = (w_val.y >> 32) as usize;    // offset in b[]
        let ni = (u[j] & 0xFFFFFFFF) as usize;
        u2.push(u[j]);
        for idx in 0..ni {
            b2.push(b[offset + idx]);
        }
    }

    ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited;
    (u2, b2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmq_tree_basic() {
        let mut tree = RmqTree::new();

        // Insert some elements
        tree.insert_elem(100, 0, -10.0);
        tree.insert_elem(200, 1, -20.0);
        tree.insert_elem(150, 2, -15.0);

        assert_eq!(tree.len(), 3);

        // RMQ closed interval [(lo_y, INT32_MAX), (hi_y, 0)]:
        //   y > lo_y (exclusive lower), y < hi_y (all), y == hi_y only if i == 0
        // Range (99, 201) includes y=100,150,200 → should return i=1 (pri=-20.0)
        let result = tree.rmq(99, 201);
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 1); // pri=-20.0 is lowest

        // Range (99, 150) includes only y=100 → should return i=0
        // Element at (150, i=2) excluded because i=2 > 0
        let result = tree.rmq(99, 150);
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 0);

        // Range (100, 200) excludes y=100 (lower bound exclusive), includes y=150 → i=2
        // Element at (200, i=1) excluded because i=1 > 0
        let result = tree.rmq(100, 200);
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 2);

        // Erase and check
        tree.erase(150, 2);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_chain_rmq_simple() {
        let k = 15;
        let span_mask = (k as u64) << 32;
        let mut anchors: Vec<Minimizer> = Vec::new();

        // Linear chain of 3 anchors
        anchors.push(Minimizer { x: 100, y: span_mask | 100 });
        anchors.push(Minimizer { x: 120, y: span_mask | 120 });
        anchors.push(Minimizer { x: 150, y: span_mask | 150 });

        let opt = ChainingParams {
            min_cnt: 1,
            min_chain_score: 10,
            max_gap: 500,
            max_gap_ref: -1,
            max_dist_x: 500,
            max_dist_y: 500,
            bandwidth: 500,
            bandwidth_long: 500,
            max_chain_skip: 25,
            max_chain_iter: 5000,
            chn_pen_gap: 0.5,
            chn_pen_skip: 0.5,
            chain_gap_scale: 0.8,
            rmq_rescue_size: 1000,
            rmq_rescue_ratio: 0.1,
            rmq_inner_dist: 100,
            rmq_size_cap: 500,
        };
        let mut bufs = ChainingBuffers::new();
        let (u, _chains) = chain_anchors_rmq(
            &opt, &mut anchors, &mut bufs,
        );

        // Should produce at least one chain
        assert!(!u.is_empty());
    }
}
