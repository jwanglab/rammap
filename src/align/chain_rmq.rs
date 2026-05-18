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

/// Arena-based AVL tree with subtree-min augmentation for O(log n) RMQ.
/// Replaces the previous treap with iterative insert/erase and compact nodes.
struct RmqTree {
    nodes: Vec<AvlNode>,
    root: u32,
    size: usize,
    free_head: u32, // free list for node reuse
    // Scratch buffers reused across insert_elem calls to avoid per-call allocation.
    insert_path: Vec<u32>,
    insert_stack: Vec<u8>,
}

#[derive(Clone, Copy)]
struct AvlNode {
    pri: f64,           // negative DP score (RMQ value to minimize)
    key_y: i32,         // query position (sort key)
    key_i: u32,         // anchor index (tiebreak key)
    left: u32,
    right: u32,
    parent: u32,
    sub_min_idx: u32,   // arena index of node with min pri in subtree
    balance: i8,        // AVL balance factor: -1, 0, +1
}

impl RmqTree {
    fn new() -> Self {
        RmqTree {
            nodes: Vec::with_capacity(256),
            root: NIL,
            size: 0,
            free_head: NIL,
            insert_path: Vec::with_capacity(64),
            insert_stack: Vec::with_capacity(64),
        }
    }

    #[inline]
    fn len(&self) -> usize { self.size }

    #[inline(always)]
    fn key_lt(ay: i32, ai: u32, by: i32, bi: u32) -> bool {
        ay < by || (ay == by && ai < bi)
    }

    fn alloc_node(&mut self, key_y: i32, key_i: u32, pri: f64) -> u32 {
        let idx;
        if self.free_head != NIL {
            idx = self.free_head;
            self.free_head = self.nodes[idx as usize].right; // free list uses right pointer
            self.nodes[idx as usize] = AvlNode {
                pri, key_y, key_i, left: NIL, right: NIL, parent: NIL,
                sub_min_idx: idx, balance: 0,
            };
        } else {
            idx = self.nodes.len() as u32;
            self.nodes.push(AvlNode {
                pri, key_y, key_i, left: NIL, right: NIL, parent: NIL,
                sub_min_idx: idx, balance: 0,
            });
        }
        idx
    }

    fn free_node(&mut self, idx: u32) {
        self.nodes[idx as usize].right = self.free_head;
        self.free_head = idx;
    }

    #[inline]
    fn update_sub_min(&mut self, idx: u32) {
        let i = idx as usize;
        let l = self.nodes[i].left;
        let r = self.nodes[i].right;
        let cur_pri = self.nodes[i].pri;
        let mut best = if l == NIL {
            idx
        } else {
            let l_min = self.nodes[l as usize].sub_min_idx;
            if cur_pri < self.nodes[l_min as usize].pri { idx } else { l_min }
        };
        if r != NIL {
            let r_min = self.nodes[r as usize].sub_min_idx;
            let best_pri = self.nodes[best as usize].pri;
            if best_pri >= self.nodes[r_min as usize].pri {
                best = r_min;
            }
        }
        self.nodes[i].sub_min_idx = best;
    }

    /// Set child[dir] of p to c, updating parent pointer.
    #[inline]
    fn set_child(&mut self, p: u32, dir: usize, c: u32) {
        if dir == 0 { self.nodes[p as usize].left = c; }
        else { self.nodes[p as usize].right = c; }
        if c != NIL { self.nodes[c as usize].parent = p; }
    }

    #[inline]
    fn child(&self, idx: u32, dir: usize) -> u32 {
        if dir == 0 { self.nodes[idx as usize].left } else { self.nodes[idx as usize].right }
    }

    /// Single rotation: rotate x up over its parent p.
    /// dir=0: left rotation (x is right child of p), dir=1: right rotation.
    ///
    /// `x` inherits p's old sub_min; p's sub_min is recomputed from
    /// (p.p[dir], q.p[dir]) — for dir=1 this is reversed from natural left/right,
    /// which matters for tie-breaking among equal-pri descendants.
    fn rotate(&mut self, p: u32, dir: usize) -> u32 {
        let x = self.child(p, 1 - dir);
        let mid = self.child(x, dir);
        let gp = self.nodes[p as usize].parent;
        let saved_p_sub = self.nodes[p as usize].sub_min_idx;
        let p_dir_child = self.child(p, dir);
        self.set_child(x, dir, p);
        self.set_child(p, 1 - dir, mid);
        self.nodes[x as usize].parent = gp;
        if gp != NIL {
            if self.nodes[gp as usize].left == p { self.nodes[gp as usize].left = x; }
            else { self.nodes[gp as usize].right = x; }
        }
        self.update_sub_min_with_children(p, p_dir_child, mid);
        self.nodes[x as usize].sub_min_idx = saved_p_sub;
        x
    }

    /// Fused double rotation. Layout:
    ///   before: (a, ((b, c)r, d)q)p  with dir=0
    ///   after:  ((a, b)p, (c, d)q)r
    /// `r` inherits p's old sub_min; p and q are recomputed from explicit
    /// post-rotation children using update_sub_min_with_children, since the
    /// arg order to that helper carries tie-break preference.
    fn rotate2(&mut self, p: u32, dir: usize) -> u32 {
        let opp = 1 - dir;
        let q = self.child(p, opp);
        let r = self.child(q, dir);
        let saved_p_sub = self.nodes[p as usize].sub_min_idx;
        let gp = self.nodes[p as usize].parent;
        let p_dir = self.child(p, dir);
        let r_dir = self.child(r, dir);
        let q_opp = self.child(q, opp);
        let r_opp = self.child(r, opp);
        self.update_sub_min_with_children(p, p_dir, r_dir);
        self.update_sub_min_with_children(q, q_opp, r_opp);
        self.nodes[r as usize].sub_min_idx = saved_p_sub;
        self.set_child(p, opp, r_dir);
        self.set_child(r, dir, p);
        self.set_child(q, dir, r_opp);
        self.set_child(r, opp, q);
        self.nodes[r as usize].parent = gp;
        if gp != NIL {
            if self.nodes[gp as usize].left == p { self.nodes[gp as usize].left = r; }
            else { self.nodes[gp as usize].right = r; }
        }
        r
    }

    /// Compute sub_min for `idx` from explicit children rather than reading
    /// `idx`'s current left/right. `q` is preferred over `r` on equal-pri.
    #[inline]
    fn update_sub_min_with_children(&mut self, idx: u32, q: u32, r: u32) {
        let cur_pri = self.nodes[idx as usize].pri;
        let mut best = idx;
        if q != NIL {
            let q_min = self.nodes[q as usize].sub_min_idx;
            if cur_pri >= self.nodes[q_min as usize].pri {
                best = q_min;
            }
        }
        if r != NIL {
            let r_min = self.nodes[r as usize].sub_min_idx;
            if self.nodes[best as usize].pri >= self.nodes[r_min as usize].pri {
                best = r_min;
            }
        }
        self.nodes[idx as usize].sub_min_idx = best;
    }

    /// Insert and rebalance. BST descent records the ancestor path and the
    /// deepest non-zero-balance ancestor `bp`. Then two independent walks:
    /// sub_min leaf->root with early break when the new node is no longer the
    /// subtree min, and balance bp->leaf. One rotation at bp if `|balance|>=2`.
    ///
    /// Sub_min and balance walks must not be interleaved: an interleaved walk
    /// that exits at the rotation point and then re-walks ancestors with
    /// natural-args `update_sub_min` overrides rotation-specific tie-break
    /// results at `new_root.parent` on equal-pri ties.
    fn insert_elem(&mut self, y: i32, i: usize, pri: f64) {
        let key_i = i as u32;
        let new_idx = self.alloc_node(y, key_i, pri);
        self.size += 1;

        if self.root == NIL {
            self.root = new_idx;
            return;
        }

        // BST descent. Record full ancestor path; bp_idx tracks the deepest
        // non-zero-balance ancestor seen so far. Balance walk later uses only
        // stack[bp_idx..]. Buffers are reused across calls to avoid per-call
        // allocation.
        let mut path = std::mem::take(&mut self.insert_path);
        let mut stack = std::mem::take(&mut self.insert_stack);
        path.clear();
        stack.clear();
        let mut bp_idx: usize = 0;
        let mut cur = self.root;
        loop {
            if self.nodes[cur as usize].balance != 0 {
                bp_idx = path.len();
            }
            let n = &self.nodes[cur as usize];
            let go_left = Self::key_lt(y, key_i, n.key_y, n.key_i);
            let dir = if go_left { 0u8 } else { 1u8 };
            let next = if go_left { n.left } else { n.right };
            path.push(cur);
            stack.push(dir);
            if next == NIL {
                self.set_child(cur, dir as usize, new_idx);
                break;
            }
            cur = next;
        }

        // Step 1: sub_min walk from leaf's parent up to root with early break.
        // alloc_node already set new leaf's sub_min_idx to itself.
        for k in (0..path.len()).rev() {
            let anc = path[k];
            self.update_sub_min(anc);
            if self.nodes[anc as usize].sub_min_idx != new_idx {
                break;
            }
        }

        // Step 2: balance walk from bp down to leaf's parent.
        for k in bp_idx..path.len() {
            let anc = path[k];
            let dir = stack[k];
            if dir == 0 {
                self.nodes[anc as usize].balance -= 1;
            } else {
                self.nodes[anc as usize].balance += 1;
            }
        }

        // Step 3: rotate at bp if balance hits +-2.
        let bp = path[bp_idx];
        let bp_balance = self.nodes[bp as usize].balance;
        if bp_balance >= 2 || bp_balance <= -2 {
            let heavy_dir = if bp_balance > 0 { 1usize } else { 0 };
            let heavy_child = self.child(bp, heavy_dir);
            let hb = self.nodes[heavy_child as usize].balance;
            let opposite = 1 - heavy_dir;
            let new_root;
            if (heavy_dir == 1 && hb >= 0) || (heavy_dir == 0 && hb <= 0) {
                new_root = self.rotate(bp, opposite);
                self.nodes[bp as usize].balance = if hb == 0 { if heavy_dir == 1 { 1 } else { -1 } } else { 0 };
                self.nodes[new_root as usize].balance = if hb == 0 { if heavy_dir == 1 { -1 } else { 1 } } else { 0 };
            } else {
                let inner = self.child(heavy_child, opposite);
                let inner_bal = self.nodes[inner as usize].balance;
                new_root = self.rotate2(bp, opposite);
                self.nodes[bp as usize].balance = if (heavy_dir == 1 && inner_bal > 0) || (heavy_dir == 0 && inner_bal < 0) {
                    if heavy_dir == 1 { -1 } else { 1 }
                } else { 0 };
                self.nodes[heavy_child as usize].balance = if (heavy_dir == 1 && inner_bal < 0) || (heavy_dir == 0 && inner_bal > 0) {
                    if heavy_dir == 1 { 1 } else { -1 }
                } else { 0 };
                self.nodes[new_root as usize].balance = 0;
                let _ = inner;
            }
            if self.root == bp { self.root = new_root; }
        }

        // Return scratch buffers to the tree for the next call.
        self.insert_path = path;
        self.insert_stack = stack;
    }

    fn erase(&mut self, y: i32, i: usize) -> bool {
        let key_i = i as u32;
        // Find the node
        let mut cur = self.root;
        while cur != NIL {
            let n = &self.nodes[cur as usize];
            if n.key_y == y && n.key_i == key_i { break; }
            cur = if Self::key_lt(y, key_i, n.key_y, n.key_i) { n.left } else { n.right };
        }
        if cur == NIL { return false; }

        self.size -= 1;

        // If two children, copy in-order successor's data and delete the successor instead
        let victim = cur;
        if self.nodes[cur as usize].left != NIL && self.nodes[cur as usize].right != NIL {
            let mut succ = self.nodes[cur as usize].right;
            while self.nodes[succ as usize].left != NIL {
                succ = self.nodes[succ as usize].left;
            }
            self.nodes[cur as usize].key_y = self.nodes[succ as usize].key_y;
            self.nodes[cur as usize].key_i = self.nodes[succ as usize].key_i;
            self.nodes[cur as usize].pri = self.nodes[succ as usize].pri;
            cur = succ;
        }

        // cur has at most one child. Record which side of parent it was on BEFORE splicing.
        let par = self.nodes[cur as usize].parent;
        let del_was_left = par != NIL && self.nodes[par as usize].left == cur;

        let child = if self.nodes[cur as usize].left != NIL {
            self.nodes[cur as usize].left
        } else {
            self.nodes[cur as usize].right
        };

        // Splice out cur
        if child != NIL { self.nodes[child as usize].parent = par; }
        if par == NIL {
            self.root = child;
        } else if del_was_left {
            self.nodes[par as usize].left = child;
        } else {
            self.nodes[par as usize].right = child;
        }

        self.free_node(cur);

        // Walk up from par rebalancing. Track which side shrunk.
        let mut p = par;
        let mut shrunk_left = del_was_left;
        while p != NIL {
            self.update_sub_min(p);
            let old_balance = self.nodes[p as usize].balance;
            // Adjust balance: left shrunk -> balance increases; right shrunk -> decreases
            let new_balance = if shrunk_left { old_balance + 1 } else { old_balance - 1 };
            self.nodes[p as usize].balance = new_balance;

            if new_balance == 1 || new_balance == -1 {
                // Height didn't change (was 0, now +-1), stop propagating
                break;
            } else if new_balance == 0 {
                // Height decreased, continue propagating up
                let pp = self.nodes[p as usize].parent;
                if pp != NIL {
                    shrunk_left = self.nodes[pp as usize].left == p;
                }
                p = pp;
            } else {
                // |new_balance| == 2, need rotation
                let heavy_dir = if new_balance > 0 { 1usize } else { 0 };
                let heavy_child = self.child(p, heavy_dir);
                let hb = self.nodes[heavy_child as usize].balance;
                let opposite = 1 - heavy_dir;

                if (heavy_dir == 1 && hb >= 0) || (heavy_dir == 0 && hb <= 0) {
                    // Single rotation
                    let new_root = self.rotate(p, opposite);
                    if hb == 0 {
                        // Height didn't change after rotation
                        self.nodes[p as usize].balance = if heavy_dir == 1 { 1 } else { -1 };
                        self.nodes[new_root as usize].balance = if heavy_dir == 1 { -1 } else { 1 };
                        if self.root == p { self.root = new_root; }
                        break; // height unchanged, stop
                    } else {
                        self.nodes[p as usize].balance = 0;
                        self.nodes[new_root as usize].balance = 0;
                        if self.root == p { self.root = new_root; }
                        // Height decreased, continue propagating
                        let pp = self.nodes[new_root as usize].parent;
                        if pp != NIL {
                            shrunk_left = self.nodes[pp as usize].left == new_root;
                        }
                        p = pp;
                    }
                } else {
                    // Double rotation (fused).
                    let inner = self.child(heavy_child, opposite);
                    let inner_bal = self.nodes[inner as usize].balance;
                    let new_root = self.rotate2(p, opposite);
                    self.nodes[p as usize].balance = if (heavy_dir == 1 && inner_bal > 0) || (heavy_dir == 0 && inner_bal < 0) {
                        if heavy_dir == 1 { -1 } else { 1 }
                    } else { 0 };
                    self.nodes[heavy_child as usize].balance = if (heavy_dir == 1 && inner_bal < 0) || (heavy_dir == 0 && inner_bal > 0) {
                        if heavy_dir == 1 { 1 } else { -1 }
                    } else { 0 };
                    self.nodes[new_root as usize].balance = 0;
                    if self.root == p { self.root = new_root; }
                    let _ = inner;
                    // Height decreased, continue propagating
                    let pp = self.nodes[new_root as usize].parent;
                    if pp != NIL {
                        shrunk_left = self.nodes[pp as usize].left == new_root;
                    }
                    p = pp;
                }
            }
        }
        // Continue updating sub_min for remaining ancestors
        while p != NIL {
            self.update_sub_min(p);
            p = self.nodes[p as usize].parent;
        }
        // Update sub_min from victim (whose key/pri was overwritten) up to root
        if victim != cur {
            let mut v = victim;
            while v != NIL {
                self.update_sub_min(v);
                v = self.nodes[v as usize].parent;
            }
        }
        true
    }

    /// Two-path LCA range minimum query: returns the element with minimum pri
    /// in the closed key interval [(lo_y, u32::MAX), (hi_y, 0)] — i.e. y > lo_y
    /// strictly, y < hi_y, or y == hi_y with i == 0.
    fn rmq(&self, lo_y: i32, hi_y: i32) -> Option<(i32, usize, f64)> {
        if self.root == NIL { return None; }

        // Key comparison: (y, i) ordering. lo = (lo_y, MAX), hi = (hi_y, 0).
        let lo_key = (lo_y, u32::MAX);
        let hi_key = (hi_y, 0u32);

        // Trace path from root to lo bound
        let mut path_lo: [(u32, i8); 64] = [(NIL, 0); 64]; // (node, cmp result)
        let mut len_lo = 0usize;
        let mut p = self.root;
        while p != NIL {
            let n = &self.nodes[p as usize];
            let nk = (n.key_y, n.key_i);
            let cmp = if lo_key < nk { -1i8 } else if lo_key > nk { 1 } else { 0 };
            path_lo[len_lo] = (p, cmp);
            len_lo += 1;
            if cmp < 0 { p = n.left; }
            else if cmp > 0 { p = n.right; }
            else { break; }
        }

        // Trace path from root to hi bound
        let mut path_hi: [(u32, i8); 64] = [(NIL, 0); 64];
        let mut len_hi = 0usize;
        p = self.root;
        while p != NIL {
            let n = &self.nodes[p as usize];
            let nk = (n.key_y, n.key_i);
            let cmp = if hi_key < nk { -1i8 } else if hi_key > nk { 1 } else { 0 };
            path_hi[len_hi] = (p, cmp);
            len_hi += 1;
            if cmp < 0 { p = n.left; }
            else if cmp > 0 { p = n.right; }
            else { break; }
        }

        // Find LCA: first shared node where lo goes left/equal and hi goes right/equal
        let mut lca = 0usize;
        let mut found_lca = false;
        for i in 0..len_lo.min(len_hi) {
            if path_lo[i].0 == path_hi[i].0 && path_lo[i].1 <= 0 && path_hi[i].1 >= 0 {
                lca = i;
                found_lca = true;
                break;
            }
            if path_lo[i].0 != path_hi[i].0 { break; }
        }
        if !found_lca { return None; }

        let mut min_idx = path_lo[lca].0; // start with LCA node

        // Scan lo-path below LCA: nodes where we went left/equal are in range.
        // Their right subtrees are fully in range.
        for (node, cmp) in path_lo.iter().take(len_lo).skip(lca + 1) {
            if *cmp <= 0 {
                // This node is in range (>= lo)
                if self.nodes[*node as usize].pri < self.nodes[min_idx as usize].pri {
                    min_idx = *node;
                }
                // Its right subtree is fully in range
                let rc = self.nodes[*node as usize].right;
                if rc != NIL {
                    let rc_min = self.nodes[rc as usize].sub_min_idx;
                    if self.nodes[rc_min as usize].pri < self.nodes[min_idx as usize].pri {
                        min_idx = rc_min;
                    }
                }
            }
        }

        // Scan hi-path below LCA: nodes where we went right/equal are in range.
        // Their left subtrees are fully in range.
        for (node, cmp) in path_hi.iter().take(len_hi).skip(lca + 1) {
            if *cmp >= 0 {
                if self.nodes[*node as usize].pri < self.nodes[min_idx as usize].pri {
                    min_idx = *node;
                }
                let lc = self.nodes[*node as usize].left;
                if lc != NIL {
                    let lc_min = self.nodes[lc as usize].sub_min_idx;
                    if self.nodes[lc_min as usize].pri < self.nodes[min_idx as usize].pri {
                        min_idx = lc_min;
                    }
                }
            }
        }

        let n = &self.nodes[min_idx as usize];
        Some((n.key_y, n.key_i as usize, n.pri))
    }

    /// Reverse in-order iterator over elements with key_y <= start_y.
    ///
    /// The walk stack is bounded by the AVL tree height, which is itself bounded
    /// by ~1.44·log2(`rmq_size_cap`) — well under 64 for any practical cap. We
    /// keep the stack inline on the iterator to avoid a heap allocation per
    /// `iter_rev_le()` call (hot path: fires once per non-exact outer-tree hit,
    /// tens of millions of times per chromosome at asm presets).
    fn iter_rev_le(&self, start_y: i32) -> RmqRevIter<'_> {
        let mut iter = RmqRevIter { tree: self, stack: [0u32; RMQ_ITER_STACK_CAP], stack_len: 0 };
        let mut t = self.root;
        while t != NIL {
            let node = &self.nodes[t as usize];
            if node.key_y <= start_y {
                debug_assert!(iter.stack_len < RMQ_ITER_STACK_CAP, "RMQ iter stack overflow");
                iter.stack[iter.stack_len] = t;
                iter.stack_len += 1;
                t = node.right;
            } else {
                t = node.left;
            }
        }
        iter
    }
}

const RMQ_ITER_STACK_CAP: usize = 64;

struct RmqRevIter<'a> {
    tree: &'a RmqTree,
    stack: [u32; RMQ_ITER_STACK_CAP],
    stack_len: usize,
}

impl<'a> Iterator for RmqRevIter<'a> {
    type Item = (i32, usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.stack_len == 0 { return None; }
        self.stack_len -= 1;
        let t = self.stack[self.stack_len];
        let node = &self.tree.nodes[t as usize];
        let result = (node.key_y, node.key_i as usize, node.pri);
        let mut child = node.left;
        while child != NIL {
            debug_assert!(self.stack_len < RMQ_ITER_STACK_CAP, "RMQ iter stack overflow");
            self.stack[self.stack_len] = child;
            self.stack_len += 1;
            child = self.tree.nodes[child as usize].right;
        }
        Some(result)
    }
}

/// Simplified score computation for RMQ chaining.
/// Returns (score, is_exact, width). `width` is the diagonal deviation
/// `|ref_diff - query_diff|`, used to gate by bandwidth in the chaining loop.
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

/// RMQ-based chaining: uses a range-max query tree over score-by-position to
/// pick the best predecessor in O(log n) per anchor instead of the DP path's
/// O(n) inner scan. Wins on large bandwidths / long-read chains.
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
    let mut bt_candidates = std::mem::take(&mut ctx.bt_candidates);
    let mut scores = std::mem::take(&mut ctx.scores);
    let mut peak_scores = std::mem::take(&mut ctx.peak_scores);
    let mut visited = std::mem::take(&mut ctx.visited);
    predecessors.resize(n, 0i32);
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

        if let Some((_q_y, q_i, _q_pri)) = root.rmq(lo_y, hi_y) {
            let j = q_i;
            let (sc, exact, width) = compute_chain_score_simple(&a[i], &a[j], opt.chn_pen_gap, opt.chn_pen_skip);
            let total_sc = scores[j] + sc;
            if width <= bw && total_sc > best_score {
                best_score = total_sc;
                best_predecessor = j as i64;
            }

            // If not exact match, also search inner tree for close matches
            if !exact && let Some(ref inner) = root_inner && a[i].query_pos() > 0 {
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

        // Set max
        scores[i] = best_score;
        predecessors[i] = best_predecessor as i32;
        peak_scores[i] = if best_predecessor >= 0 && peak_scores[best_predecessor as usize] > best_score { peak_scores[best_predecessor as usize] } else { best_score };
        if _mmax_f < best_score { _mmax_f = best_score; }
    }

    // Backtrack to extract chains
    let (u, n_u, n_v) = chain_backtrack(n, &scores, &predecessors, &mut peak_scores, &mut visited, &mut bt_candidates, opt.min_cnt, opt.min_chain_score, max_drop);

    if n_u == 0 {
        ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited; ctx.bt_candidates = bt_candidates;
        return (Vec::new(), Vec::new());
    }

    // compact_a logic: reorder anchors into per-chain contiguous runs.
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

    // Step 2: Sort chains by target position of their first anchor.
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

    ctx.predecessors = predecessors; ctx.scores = scores; ctx.peak_scores = peak_scores; ctx.visited = visited; ctx.bt_candidates = bt_candidates;
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
