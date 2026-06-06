#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ============================================================================
// WASM SIMD128 Compatibility Layer
// ============================================================================
// Maps SSE intrinsic names to WASM SIMD128 equivalents, allowing the DP
// macros to be instantiated for WASM without modifying their bodies.

#[cfg(target_arch = "wasm32")]
#[allow(non_camel_case_types)]
pub(super) mod simd_compat {
    use core::arch::wasm32::*;

    /// WASM v128 aliased to the SSE type name used throughout dp.rs macros.
    pub type __m128i = v128;

    // --- Load / Store ---
    #[inline(always)]
    pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i { unsafe { v128_load(p as *const v128) } }
    #[inline(always)]
    pub unsafe fn _mm_load_si128(p: *const __m128i) -> __m128i { unsafe { v128_load(p as *const v128) } }
    #[inline(always)]
    pub unsafe fn _mm_storeu_si128(p: *mut __m128i, v: __m128i) { unsafe { v128_store(p as *mut v128, v) } }
    #[inline(always)]
    pub unsafe fn _mm_store_si128(p: *mut __m128i, v: __m128i) { unsafe { v128_store(p as *mut v128, v) } }

    // --- Broadcast / Init ---
    #[inline(always)]
    pub unsafe fn _mm_setzero_si128() -> __m128i { i8x16_splat(0) }
    #[inline(always)]
    pub unsafe fn _mm_set1_epi8(v: i8) -> __m128i { i8x16_splat(v) }
    #[inline(always)]
    pub unsafe fn _mm_set1_epi32(v: i32) -> __m128i { i32x4_splat(v) }
    #[inline(always)]
    pub unsafe fn _mm_setr_epi32(e0: i32, e1: i32, e2: i32, e3: i32) -> __m128i {
        i32x4(e0, e1, e2, e3)
    }
    #[inline(always)]
    pub unsafe fn _mm_set_epi8(
        e15: i8, e14: i8, e13: i8, e12: i8,
        e11: i8, e10: i8, e9: i8, e8: i8,
        e7: i8, e6: i8, e5: i8, e4: i8,
        e3: i8, e2: i8, e1: i8, e0: i8,
    ) -> __m128i {
        i8x16(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)
    }

    // --- i8 Arithmetic ---
    #[inline(always)]
    pub unsafe fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_add(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_sub(a, b) }

    // --- i8 Comparison ---
    #[inline(always)]
    pub unsafe fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_eq(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_gt(a, b) }

    // --- i8 Min/Max (native on WASM, equivalent to SSE4.1) ---
    #[inline(always)]
    pub unsafe fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_max(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i { i8x16_min(a, b) }

    // --- u8 Min/Max ---
    #[inline(always)]
    pub unsafe fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i { u8x16_max(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i { u8x16_min(a, b) }

    // --- Bitwise ---
    #[inline(always)]
    pub unsafe fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i { v128_and(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i { v128_or(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i {
        // SSE: !a & b. WASM v128_andnot(a, b) = a & !b. So swap args.
        v128_andnot(b, a)
    }

    // --- Blend (SSE4.1-equivalent) ---
    #[inline(always)]
    pub unsafe fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
        // SSE4.1: for each byte, if MSB of mask is set, take from b, else from a.
        // WASM v128_bitselect(a, b, mask): for each bit, if mask bit=1, take from a, else from b.
        // To match SSE4.1 blendv semantics (MSB-based), we propagate the MSB to all bits
        // via arithmetic right shift, then use bitselect.
        let sign_mask = i8x16_shr(mask, 7); // propagate MSB to all bits
        v128_bitselect(b, a, sign_mask)
    }

    // --- Insert/Extract (SSE4.1-equivalent) ---
    // These must be macros because lane index must be a compile-time constant.
    macro_rules! _mm_insert_epi8_impl {
        ($vec:expr, $val:expr, 0) => { i8x16_replace_lane::<0>($vec, $val as i8) };
        ($vec:expr, $val:expr, 1) => { i8x16_replace_lane::<1>($vec, $val as i8) };
        ($vec:expr, $val:expr, 15) => { i8x16_replace_lane::<15>($vec, $val as i8) };
    }

    // --- i32 Arithmetic ---
    #[inline(always)]
    pub unsafe fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i { i32x4_add(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i { i32x4_gt(a, b) }

    // --- Byte Shift (via swizzle) ---
    // _mm_slli_si128(v, N): shift left by N bytes, zeros enter at low positions.
    // _mm_srli_si128(v, N): shift right by N bytes, zeros enter at high positions.
    // Using 2-arg form to match how x86_64 intrinsics are called in the DP macros
    // (x86_64 uses #[rustc_legacy_const_generics] to allow both calling conventions).
    // i8x16_swizzle returns 0 for indices >= 16, which gives us zero-fill for free.
    #[inline(always)]
    pub unsafe fn _mm_slli_si128(a: __m128i, imm8: i32) -> __m128i {
        if imm8 <= 0 { return a; }
        if imm8 >= 16 { return unsafe { _mm_setzero_si128() }; }
        let n = imm8 as u8;
        let mut idx = [0x80u8; 16];
        let mut i = n;
        while i < 16 { idx[i as usize] = i - n; i += 1; }
        unsafe { i8x16_swizzle(a, v128_load(idx.as_ptr() as *const v128)) }
    }

    #[inline(always)]
    pub unsafe fn _mm_srli_si128(a: __m128i, imm8: i32) -> __m128i {
        if imm8 <= 0 { return a; }
        if imm8 >= 16 { return unsafe { _mm_setzero_si128() }; }
        let n = imm8 as u8;
        let mut idx = [0x80u8; 16];
        let mut i = 0u8;
        while i + n < 16 { idx[i as usize] = i + n; i += 1; }
        unsafe { i8x16_swizzle(a, v128_load(idx.as_ptr() as *const v128)) }
    }

    // --- sse2_insert_byte0 (used unconditionally in DP macros) ---
    #[inline(always)]
    pub unsafe fn sse2_insert_byte0(vec: __m128i, val: u8) -> __m128i {
        i8x16_replace_lane::<0>(vec, val as i8)
    }

    // --- i16 operations (for lightweight_align_i16) ---
    #[inline(always)]
    pub unsafe fn _mm_set1_epi16(v: i16) -> __m128i { i16x8_splat(v) }
    #[inline(always)]
    pub unsafe fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i { i16x8_add_sat(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i { u16x8_sub_sat(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i { i16x8_max(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i { i16x8_gt(a, b) }
    #[inline(always)]
    pub unsafe fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32 {
        // Rust's SSE returns i32. WASM extract returns the lane value.
        match IMM8 {
            0 => u16x8_extract_lane::<0>(a) as i32,
            1 => u16x8_extract_lane::<1>(a) as i32,
            2 => u16x8_extract_lane::<2>(a) as i32,
            3 => u16x8_extract_lane::<3>(a) as i32,
            4 => u16x8_extract_lane::<4>(a) as i32,
            5 => u16x8_extract_lane::<5>(a) as i32,
            6 => u16x8_extract_lane::<6>(a) as i32,
            _ => u16x8_extract_lane::<7>(a) as i32,
        }
    }
    #[inline(always)]
    pub unsafe fn _mm_movemask_epi8(a: __m128i) -> i32 { u8x16_bitmask(a) as i32 }
}


// ============================================================================
// SSE2 Helper Functions (emulate SSE4.1 operations)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn sse2_max_epi8(a: __m128i, b: __m128i) -> __m128i { unsafe {
    let mask = _mm_cmpgt_epi8(a, b);
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn sse2_min_epi8(a: __m128i, b: __m128i) -> __m128i { unsafe {
    let mask = _mm_cmpgt_epi8(a, b);
    _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
}}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn sse2_insert_byte0(vec: __m128i, val: u8) -> __m128i { unsafe {
    let mask = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1i8);
    let byte_vec = _mm_set1_epi8(val as i8);
    _mm_or_si128(_mm_andnot_si128(mask, vec), _mm_and_si128(mask, byte_vec))
}}

// ============================================================================
// AVX2 Helper Functions
// ============================================================================

/// Cross-lane byte shift left by 1 for AVX2 (256-bit).
///
/// SSE's `_mm_slli_si128(v, 1)` shifts across the full 128-bit register.
/// AVX2's `_mm256_bslli_epi128(v, 1)` only shifts within each 128-bit lane.
/// This function performs a true 256-bit shift-left-by-1-byte, inserting
/// `carry` at byte 0 and returning the displaced byte 31 as carry_out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn avx2_shift_left_1(v: __m256i, carry: __m256i) -> (__m256i, __m256i) {
    // 1. Shift left by 1 within each 128-bit lane
    let shifted = _mm256_bslli_epi128(v, 1);
    // 2. Get byte 15 (last of low lane) into byte 0 of high lane
    let cross = _mm256_permute2x128_si256(v, v, 0x08); // low→high, zero→low
    let cross = _mm256_bsrli_epi128(cross, 15);
    // 3. Combine: shifted | cross | carry
    let result = _mm256_or_si256(_mm256_or_si256(shifted, cross), carry);
    // 4. Extract carry_out = byte 31 → byte 0
    let carry_out = _mm256_bsrli_epi128(
        _mm256_permute2x128_si256(v, v, 0x81), // high→low, zero→high
        15,
    );
    (result, carry_out)
}

/// Insert a byte at position 0 of a 256-bit register, preserving bytes 1-31.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn avx2_insert_byte0(vec: __m256i, val: u8) -> __m256i { unsafe {
    let low = _mm256_castsi256_si128(vec);
    let low = sse2_insert_byte0(low, val);
    _mm256_inserti128_si256(vec, low, 0)
}}

/// Shift a 512-bit register left by 1 byte across all four 128-bit lanes, inserting
/// `carry` at byte 0 and returning the displaced byte 63 as carry_out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[inline]
pub(super) unsafe fn avx512_shift_left_1(v: __m512i, carry: __m512i) -> (__m512i, __m512i) {
    // 1. Shift left by 1 within each 128-bit lane
    let shifted = _mm512_bslli_epi128(v, 1);
    // 2. Rotate lanes left: lane3→0, lane0→1, lane1→2, lane2→3
    let rotated = _mm512_shuffle_i32x4(v, v, 0b10_01_00_11);
    // 3. Extract last byte of each rotated lane → first byte position
    let cross_raw = _mm512_bsrli_epi128(rotated, 15);
    // cross_raw: byte0=byte63(unwanted), byte16=byte15, byte32=byte31, byte48=byte47
    let cross = _mm512_maskz_mov_epi8(0x0001_0001_0001_0000u64, cross_raw);
    // 4. Combine: shifted | cross | carry
    let result = _mm512_or_si512(_mm512_or_si512(shifted, cross), carry);
    // 5. Extract carry_out = byte 63 → byte 0
    let lane3 = _mm512_shuffle_i32x4(v, v, 0xFF); // broadcast lane 3
    let carry_out = _mm512_maskz_mov_epi8(1u64, _mm512_bsrli_epi128(lane3, 15));
    (result, carry_out)
}

/// Insert a byte at position 0 of a 512-bit register, preserving bytes 1-63.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[inline]
pub(super) unsafe fn avx512_insert_byte0(vec: __m512i, val: u8) -> __m512i {
    _mm512_mask_set1_epi8(vec, 1u64, val as i8)
}

// ============================================================================
// Constants
// ============================================================================

/// Negative infinity for DP initialization
pub const NEG_INF: i32 = -0x40000000;

// Alignment flags
/// Only compute score, skip traceback
pub const SCORE_ONLY: i32 = 0x01;
/// Right-align gaps (prefer gaps at end)
pub const RIGHT_ALIGN: i32 = 0x02;
/// Use generic scoring matrix
pub const GENERIC_SCORING: i32 = 0x04;
/// Use approximate max score tracking
pub const APPROX_MAX: i32 = 0x08;
/// Enable z-drop heuristic
pub const APPROX_DROP: i32 = 0x10;
/// Extension-only mode (stop at max score)
pub const EXTENSION_ONLY: i32 = 0x40;
/// Reverse CIGAR output
pub const REV_CIGAR: i32 = 0x80;
/// Splice alignment: forward transcript strand
pub const SPLICE_FORWARD: i32 = 0x100;
/// Splice alignment: reverse transcript strand
pub const SPLICE_REVERSE: i32 = 0x200;
/// Splice alignment: use flank penalties
pub const SPLICE_FLANK: i32 = 0x400;
/// Splice alignment: complex splice model (miniprot-style)
pub const SPLICE_COMPLEX: i32 = 0x800;
/// Splice alignment: use splice score from junc array
pub const SPLICE_SCORE: i32 = 0x1000;

// CIGAR operation codes
pub const CIGAR_MATCH: u32 = 0;
pub const CIGAR_INS: u32 = 1;
pub const CIGAR_DEL: u32 = 2;
pub const CIGAR_N_SKIP: u32 = 3;
// Splice score offset
pub const SPSC_OFFSET: i32 = 64;

// ============================================================================
// Types
// ============================================================================

/// Extension alignment result structure
///
/// Contains alignment score, coordinates, and optional CIGAR string.
#[derive(Debug, Clone, Default)]
pub struct DpResult {
    /// Maximum score found during alignment
    pub max: i32,
    /// Query position of maximum score (0-based)
    pub max_score_query_pos: i32,
    /// Target position of maximum score (0-based)
    pub max_score_target_pos: i32,
    /// Max score when query is exhausted
    pub max_query_end_score: i32,
    /// Target position for max_query_end_score
    pub max_query_end_target_pos: i32,
    /// Max score when target is exhausted
    pub max_target_end_score: i32,
    /// Query position for max_target_end_score
    pub max_target_end_query_pos: i32,
    /// Final alignment score
    pub score: i32,
    /// CIGAR capacity (internal use)
    pub cigar_capacity: i32,
    /// Number of CIGAR operations
    pub cigar_len: i32,
    /// Whether alignment reached sequence end
    pub reach_end: i32,
    /// Whether alignment was z-dropped
    pub zdropped: i32,
    /// CIGAR operations (len << 4 | op), op: 0=M, 1=I, 2=D
    pub cigar: Vec<u32>,
}

// ============================================================================
// Memory Management
// ============================================================================

// Thread-local cache for DP matrix memory. Avoids repeated mmap/munmap
// syscalls for large allocations (~17MB per CIGAR alignment call).
// Keeps the high-water-mark allocation alive per thread.
//
// Safety: DP calls are sequential per thread (never nested), so the cache
// is taken on AlignedMemory::new() and returned on Drop. Only one AlignedMemory
// is alive per thread at any time.
pub(super) use std::cell::Cell;

thread_local! {
    pub(super) static DP_MEM_CACHE: Cell<Option<(*mut u8, std::alloc::Layout)>> = const { Cell::new(None) };
}

// The per-thread DP scratch cache retains each thread's high-water-mark buffer.
// The splice DP is unbanded, so a single large gap-fill segment needs
// `O((qlen+tlen)·min(qlen,tlen))` bytes — hundreds of MB for the rare read with a
// big unaligned gap. Without a cap, that one-off high-water mark is pinned per
// worker thread for the rest of the run; at high thread counts the sum dominates
// peak RSS (the genome-mode regression vs minimap2, whose `km`/`cap_kalloc` arena
// is bounded the same way). The cap (see `super::dp_cache_cap_bytes` /
// `super::set_dp_cache_cap_mb`) frees buffers larger than the bound on drop; the
// common case (almost all buffers well under the cap) stays cached and churn-free.
use super::dp_cache_cap_bytes;

pub(super) struct AlignedMemory {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedMemory {
    pub(super) fn new(size: usize, align: usize) -> Self {
        unsafe {
            // Try to reuse the cached allocation
            let (ptr, layout) = DP_MEM_CACHE.with(|cache| {
                if let Some((cached_ptr, cached_layout)) = cache.take() {
                    if cached_layout.size() >= size && cached_layout.align() >= align {
                        // Reuse without zeroing — DP algorithms initialize
                        // their own boundary conditions before reading.
                        return (cached_ptr, cached_layout);
                    }
                    // Cached allocation too small — free it and allocate larger
                    std::alloc::dealloc(cached_ptr, cached_layout);
                }
                // Allocate fresh
                let layout = std::alloc::Layout::from_size_align(size, align)
                    .unwrap_or_else(|_| panic!("DP: invalid alignment layout (size={}, align={})", size, align));
                let ptr = std::alloc::alloc_zeroed(layout);
                assert!(!ptr.is_null(), "DP: failed to allocate {} bytes (aligned to {})", size, align);
                (ptr, layout)
            });
            Self { ptr, layout }
        }
    }

    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedMemory {
    fn drop(&mut self) {
        let cap = dp_cache_cap_bytes();
        if cap != 0 && self.layout.size() > cap {
            // Oversized one-off buffer (rare large gap-fill): free it back to the
            // OS rather than pinning it per-thread for the rest of the run. This
            // bounds the cached high-water mark, the cap_kalloc analog.
            unsafe { std::alloc::dealloc(self.ptr, self.layout); }
            self.ptr = std::ptr::null_mut();
            return;
        }
        // Return to cache instead of deallocating. Keep the larger allocation
        // if the cache already has one (high-water-mark strategy). If the cache
        // already holds a (cap-bounded) buffer, free this one to avoid leaking
        // the displaced allocation.
        DP_MEM_CACHE.with(|cache| {
            if let Some((old_ptr, old_layout)) = cache.replace(Some((self.ptr, self.layout))) {
                unsafe { std::alloc::dealloc(old_ptr, old_layout); }
            }
        });
        // Null out to prevent use-after-free if Drop is somehow called twice
        self.ptr = std::ptr::null_mut();
    }
}

// ============================================================================
// Shared Helper Functions
// ============================================================================

/// Initialize DpResult fields for the DP loop (extz2/extd2 variant).
/// Sets score tracking fields to initial values before alignment begins.
#[inline(always)]
pub(super) fn init_dp_result(result: &mut DpResult) {
    result.max = 0;
    result.max_score_query_pos = -1;
    result.max_score_target_pos = -1;
    result.max_query_end_score = NEG_INF;
    result.max_target_end_score = NEG_INF;
    result.score = NEG_INF;
}

/// Initialize DpResult fields for the DP loop (exts2 variant).
/// Sets all score tracking fields including endpoint positions, cigar, and status.
#[inline(always)]
pub(super) fn init_dp_result_full(result: &mut DpResult) {
    result.max = 0;
    result.max_score_query_pos = -1;
    result.max_score_target_pos = -1;
    result.max_query_end_score = NEG_INF;
    result.max_target_end_score = NEG_INF;
    result.max_query_end_target_pos = -1;
    result.max_target_end_query_pos = -1;
    result.score = NEG_INF;
    result.cigar.clear();
    result.zdropped = 0;
    result.reach_end = 0;
}

/// Append a CIGAR operation to the CIGAR vector, merging with the last
/// operation if it has the same op code.
///
/// CIGAR encoding: each u32 stores (length << 4 | op), where op is:
/// 0=M, 1=I, 2=D, 3=N_SKIP
#[inline(always)]
pub(super) fn push_cigar(cigar: &mut Vec<u32>, op: u32, len: u32) {
    if let Some(last) = cigar.last_mut() && (*last & 0xf) == op {
        *last += len << 4;
        return;
    }
    cigar.push((len << 4) | op);
}

/// Allocate H[] array for exact max tracking (extd2/exts2 only).
///
/// When approx_max is false, allocates a tlen_*simd_width element i32 array
/// initialized to NEG_INF. When approx_max is true, returns an empty Vec and
/// null pointer.
///
/// simd_width must match the DP kernel's SIMD width (16 for SSE/NEON/scalar,
/// 32 for AVX2, 64 for AVX512) so that tlen_*simd_width >= target_len.
///
/// The caller must keep the returned Vec alive for the duration of the DP loop
/// to ensure the pointer remains valid.
#[inline(always)]
pub(super) fn alloc_h_array(approx_max: bool, tlen_: usize, simd_width: usize) -> (Vec<i32>, *mut i32) {
    if !approx_max {
        let h_vec = vec![NEG_INF; tlen_ * simd_width];
        let h_ptr = h_vec.as_ptr() as *mut i32;
        (h_vec, h_ptr)
    } else {
        (Vec::new(), std::ptr::null_mut())
    }
}

/// Compute the traceback starting position (i, j) from the result state.
///
/// Returns (i, j) where i is the target position and j is the query position
/// to start backtracking from. Returns (-1, -1) if no valid starting position.
///
/// Also sets result.reach_end = 1 if the EXTZ_ONLY condition is met.
#[inline(always)]
pub(super) fn traceback_start_position(
    result: &mut DpResult,
    query_len: usize,
    target_len: usize,
    end_bonus: i32,
    flags: i32,
) -> (i32, i32) {
    if result.zdropped == 0 && (flags & EXTENSION_ONLY) == 0 {
        (target_len as i32 - 1, query_len as i32 - 1)
    } else if result.zdropped == 0 && (flags & EXTENSION_ONLY) != 0 && result.max_query_end_score + end_bonus > result.max {
        result.reach_end = 1;
        (result.max_query_end_target_pos, query_len as i32 - 1)
    } else if result.max_score_target_pos >= 0 && result.max_score_query_pos >= 0 {
        (result.max_score_target_pos, result.max_score_query_pos)
    } else {
        (-1, -1)
    }
}

/// Traceback for dual-affine (extd2) alignment — shared across SSE2/SSE4.1/NEON.
///
/// Walks back through the traceback matrix to reconstruct the CIGAR string.
/// Dual-affine has 5 states: 0=M, 1=D1, 2=I1, 3=D2, 4=I2.
///
/// # Safety
/// p_ptr, band_offset_ptr, band_offset_end_ptr must point to valid memory
/// from the DP traceback allocation.
#[inline(always)]
pub(super) unsafe fn traceback_dual_affine(
    result: &mut DpResult,
    query_len: usize,
    target_len: usize,
    end_bonus: i32,
    flags: i32,
    n_col_: usize,
    simd_width: usize,
    p_ptr: *mut u8,
    band_offset_ptr: *mut i32,
    band_offset_end_ptr: *mut i32,
) { unsafe {
    let (mut i, mut j) = traceback_start_position(result, query_len, target_len, end_bonus, flags);

    if i < 0 || j < 0 {
        return;
    }
    let mut cigar = Vec::new();
    let mut state = 0i32;
    let stride = n_col_ * simd_width;

    while i >= 0 && j >= 0 {
        let r = i + j;
        let off_r = *band_offset_ptr.add(r as usize);
        let off_end_r = *band_offset_end_ptr.add(r as usize);

        let mut force_state = -1i32;
        if i < off_r { force_state = 2; }
        if i > off_end_r { force_state = 1; }

        let tmp = if force_state < 0 {
            let idx = r as usize * stride + (i - off_r) as usize;
            *p_ptr.add(idx)
        } else {
            0
        };

        if state == 0 { state = (tmp & 7) as i32; }
        else if ((tmp >> (state + 2)) & 1) == 0 { state = 0; }

        if state == 0 { state = (tmp & 7) as i32; }
        if force_state >= 0 { state = force_state; }

        let (op, di, dj) = match state {
            0 => (0u32, 1, 1),  // M
            1 => (2u32, 1, 0),  // D1
            2 => (1u32, 0, 1),  // I1
            3 => (2u32, 1, 0),  // D2
            4 => (1u32, 0, 1),  // I2
            _ => (0u32, 1, 1),
        };

        push_cigar(&mut cigar, op, 1);

        i -= di;
        j -= dj;
    }

    // Handle remaining
    if i >= 0 {
        push_cigar(&mut cigar, 2, (i + 1) as u32);
    }
    if j >= 0 {
        push_cigar(&mut cigar, 1, (j + 1) as u32);
    }

    let rev_cigar = (flags & REV_CIGAR) != 0;
    if !rev_cigar {
        cigar.reverse();
    }
    result.cigar = cigar;
}}

/// Traceback for single-affine (extz2) alignment — shared across SSE2/SSE4.1/NEON.
///
/// Walks back through the traceback matrix to reconstruct the CIGAR string.
/// Single-affine has 3 states: 0=M, 1=D, 2=I.
///
/// For a safe alternative using slice indexing, see [`traceback_single_affine_safe`].
///
/// # Safety
/// p_ptr, band_offset_ptr, band_offset_end_ptr must point to valid memory
/// from the DP traceback allocation.
#[inline(always)]
pub(super) unsafe fn traceback_single_affine(
    result: &mut DpResult,
    query_len: usize,
    target_len: usize,
    end_bonus: i32,
    flags: i32,
    n_col_: usize,
    simd_width: usize,
    p_ptr: *mut u8,
    band_offset_ptr: *mut i32,
    band_offset_end_ptr: *mut i32,
) { unsafe {
    let rev_cigar = (flags & REV_CIGAR) != 0;
    let (mut i, mut j) = traceback_start_position(result, query_len, target_len, end_bonus, flags);

    if i >= 0 && j >= 0 {
        let mut cigar = Vec::new();
        let mut state = 0;
        let stride = n_col_ * simd_width;

        while i >= 0 && j >= 0 {
            let mut force_state = -1;
            let r = i + j;
            let off_r = *band_offset_ptr.add(r as usize);
            let off_end_r = *band_offset_end_ptr.add(r as usize);

            if i < off_r { force_state = 2; }
            if i > off_end_r { force_state = 1; }

            let tmp = if force_state < 0 {
                let idx = r as usize * stride + (i - off_r) as usize;
                *p_ptr.add(idx)
            } else {
                0
            };

            if state == 0 { state = (tmp & 7) as i32; }
            else if (tmp >> (state + 2)) & 1 == 0 { state = 0; }

            if state == 0 { state = (tmp & 7) as i32; }
            if force_state >= 0 { state = force_state; }

            if state == 0 {
                push_cigar(&mut cigar, 0, 1);
                i -= 1; j -= 1;
            } else if state == 1 {
                push_cigar(&mut cigar, 2, 1);
                i -= 1;
            } else {
                push_cigar(&mut cigar, 1, 1);
                j -= 1;
            }
        }

        if i >= 0 {
            push_cigar(&mut cigar, 2, (i + 1) as u32);
        }
        if j >= 0 {
            push_cigar(&mut cigar, 1, (j + 1) as u32);
        }

        if !rev_cigar {
            cigar.reverse();
        }

        result.cigar = cigar;
    }
}}

/// Safe traceback for single-affine alignment using slice indexing.
///
/// Equivalent to [`traceback_single_affine`] but uses bounds-checked slice access
/// instead of raw pointer arithmetic. Used by the scalar extz2 implementation
/// to provide a fully-safe code path on non-SIMD targets.
pub(super) fn traceback_single_affine_safe(
    result: &mut DpResult,
    query_len: usize,
    target_len: usize,
    end_bonus: i32,
    flags: i32,
    stride: usize,
    p: &[u8],
    band_off: &[i32],
    band_off_end: &[i32],
) {
    let rev_cigar = (flags & REV_CIGAR) != 0;
    let (mut i, mut j) = traceback_start_position(result, query_len, target_len, end_bonus, flags);

    if i >= 0 && j >= 0 {
        let mut cigar = Vec::new();
        let mut state = 0i32;

        while i >= 0 && j >= 0 {
            let mut force_state = -1i32;
            let r = (i + j) as usize;
            let off_r = band_off[r];
            let off_end_r = band_off_end[r];

            if i < off_r { force_state = 2; }
            if i > off_end_r { force_state = 1; }

            let tmp = if force_state < 0 {
                let idx = r * stride + (i - off_r) as usize;
                p[idx]
            } else {
                0
            };

            if state == 0 { state = (tmp & 7) as i32; }
            else if (tmp >> (state + 2)) & 1 == 0 { state = 0; }

            if state == 0 { state = (tmp & 7) as i32; }
            if force_state >= 0 { state = force_state; }

            if state == 0 {
                push_cigar(&mut cigar, 0, 1);
                i -= 1; j -= 1;
            } else if state == 1 {
                push_cigar(&mut cigar, 2, 1);
                i -= 1;
            } else {
                push_cigar(&mut cigar, 1, 1);
                j -= 1;
            }
        }

        if i >= 0 { push_cigar(&mut cigar, 2, (i + 1) as u32); }
        if j >= 0 { push_cigar(&mut cigar, 1, (j + 1) as u32); }

        if !rev_cigar { cigar.reverse(); }
        result.cigar = cigar;
    }
}

/// Traceback for splice-aware (exts2) alignment — shared across SSE2/SSE4.1/NEON.
///
/// Walks back through the traceback matrix to reconstruct the CIGAR string.
/// Splice has 4 states: 0=M, 1=D, 2=I, 3=N_SKIP (intron) when long_thres > 0.
///
/// # Safety
/// p_ptr, band_offset_ptr, band_offset_end_ptr must point to valid memory
/// from the DP traceback allocation.
#[inline(always)]
pub(super) unsafe fn traceback_splice(
    result: &mut DpResult,
    query_len: usize,
    target_len: usize,
    end_bonus: i32,
    flags: i32,
    n_col_: usize,
    simd_width: usize,
    long_thres: i32,
    p_ptr: *mut u8,
    band_offset_ptr: *mut i32,
    band_offset_end_ptr: *mut i32,
) { unsafe {
    let rev_cigar = (flags & REV_CIGAR) != 0;
    let (mut i, mut j) = traceback_start_position(result, query_len, target_len, end_bonus, flags);

    if i >= 0 && j >= 0 {
        let mut cigar = Vec::new();
        let mut state = 0i32;
        let stride = n_col_ * simd_width;

        while i >= 0 && j >= 0 {
            let r = i + j;
            let off_r = *band_offset_ptr.add(r as usize);
            let off_end_r = *band_offset_end_ptr.add(r as usize);

            let mut force_state = -1i32;
            if i < off_r { force_state = 2; }
            if i > off_end_r { force_state = 1; }

            let tmp = if force_state < 0 {
                let idx = r as usize * stride + (i - off_r) as usize;
                *p_ptr.add(idx)
            } else {
                0
            };

            if state == 0 { state = (tmp & 7) as i32; }
            else if ((tmp >> (state + 2)) & 1) == 0 { state = 0; }
            if state == 0 { state = (tmp & 7) as i32; }
            if force_state >= 0 { state = force_state; }

            let (op, di, dj) = match state {
                0 => (0u32, 1, 1),  // M
                1 => (2u32, 1, 0),  // D
                2 => (1u32, 0, 1),  // I
                3 => {
                    if long_thres > 0 {
                        (CIGAR_N_SKIP, 1, 0) // N_SKIP (intron)
                    } else {
                        (2u32, 1, 0) // D (when long_thres <= 0, treat as normal deletion)
                    }
                },
                _ => (0u32, 1, 1),
            };

            push_cigar(&mut cigar, op, 1);

            i -= di;
            j -= dj;
        }

        // Handle remaining: trailing deletion or N_SKIP
        if i >= 0 {
            let op = if long_thres > 0 && i >= long_thres {
                CIGAR_N_SKIP
            } else {
                2 // DEL
            };
            push_cigar(&mut cigar, op, (i + 1) as u32);
        }
        if j >= 0 {
            push_cigar(&mut cigar, 1, (j + 1) as u32);
        }

        if !rev_cigar {
            cigar.reverse();
        }
        result.cigar = cigar;
    }
}}

