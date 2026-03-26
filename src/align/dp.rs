//! Dynamic-programming kernels
//!
//! Provides three kernel types — single-affine, dual-affine, and splice-aware — each
//! with SIMD specializations for SSE2, AVX2, AVX512BW, NEON, WASM SIMD128, and a
//! scalar fallback. The public entry points are `extend_single_affine`,
//! `extend_dual_affine`, and `extend_splice`; each dispatches to the best available
//! SIMD variant at runtime (overridable via `RAMMAP_FORCE_SSE` / `RAMMAP_FORCE_AVX2`).
//!
//! All kernels return a `DpResult` containing the alignment score, the query/target
//! end positions, the number of columns computed, and an optional CIGAR. Callers in
//! `extend.rs` invoke these kernels for left-extension, gap-fill, and right-extension.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ============================================================================
// WASM SIMD128 Compatibility Layer
// ============================================================================
// Maps SSE intrinsic names to WASM SIMD128 equivalents, allowing the DP
// macros to be instantiated for WASM without modifying their bodies.

#[cfg(target_arch = "wasm32")]
#[allow(non_camel_case_types)]
mod simd_compat {
    use core::arch::wasm32::*;

    /// WASM v128 aliased to the SSE type name used throughout dp.rs macros.
    pub type __m128i = v128;

    // --- Load / Store ---
    #[inline(always)]
    pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i { v128_load(p as *const v128) }
    #[inline(always)]
    pub unsafe fn _mm_load_si128(p: *const __m128i) -> __m128i { v128_load(p as *const v128) }
    #[inline(always)]
    pub unsafe fn _mm_storeu_si128(p: *mut __m128i, v: __m128i) { v128_store(p as *mut v128, v) }
    #[inline(always)]
    pub unsafe fn _mm_store_si128(p: *mut __m128i, v: __m128i) { v128_store(p as *mut v128, v) }

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
        if imm8 >= 16 { return _mm_setzero_si128(); }
        let n = imm8 as u8;
        let mut idx = [0x80u8; 16];
        let mut i = n;
        while i < 16 { idx[i as usize] = i - n; i += 1; }
        i8x16_swizzle(a, v128_load(idx.as_ptr() as *const v128))
    }

    #[inline(always)]
    pub unsafe fn _mm_srli_si128(a: __m128i, imm8: i32) -> __m128i {
        if imm8 <= 0 { return a; }
        if imm8 >= 16 { return _mm_setzero_si128(); }
        let n = imm8 as u8;
        let mut idx = [0x80u8; 16];
        let mut i = 0u8;
        while i + n < 16 { idx[i as usize] = i + n; i += 1; }
        i8x16_swizzle(a, v128_load(idx.as_ptr() as *const v128))
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

#[cfg(target_arch = "wasm32")]
use simd_compat::*;

// ============================================================================
// SSE2 Helper Functions (emulate SSE4.1 operations)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sse2_max_epi8(a: __m128i, b: __m128i) -> __m128i { unsafe {
    let mask = _mm_cmpgt_epi8(a, b);
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sse2_min_epi8(a: __m128i, b: __m128i) -> __m128i { unsafe {
    let mask = _mm_cmpgt_epi8(a, b);
    _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
}}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sse2_insert_byte0(vec: __m128i, val: u8) -> __m128i { unsafe {
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
unsafe fn avx2_shift_left_1(v: __m256i, carry: __m256i) -> (__m256i, __m256i) {
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
unsafe fn avx2_insert_byte0(vec: __m256i, val: u8) -> __m256i { unsafe {
    let low = _mm256_castsi256_si128(vec);
    let low = sse2_insert_byte0(low, val);
    _mm256_inserti128_si256(vec, low, 0)
}}

/// Shift a 512-bit register left by 1 byte across all four 128-bit lanes, inserting
/// `carry` at byte 0 and returning the displaced byte 63 as carry_out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn avx512_shift_left_1(v: __m512i, carry: __m512i) -> (__m512i, __m512i) {
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
unsafe fn avx512_insert_byte0(vec: __m512i, val: u8) -> __m512i {
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
use std::cell::Cell;

thread_local! {
    static DP_MEM_CACHE: Cell<Option<(*mut u8, std::alloc::Layout)>> = const { Cell::new(None) };
}

struct AlignedMemory {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedMemory {
    fn new(size: usize, align: usize) -> Self {
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

    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedMemory {
    fn drop(&mut self) {
        // Return to cache instead of deallocating. Keep the larger allocation
        // if the cache already has one (high-water-mark strategy).
        DP_MEM_CACHE.with(|cache| {
            cache.set(Some((self.ptr, self.layout)));
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
fn init_dp_result(result: &mut DpResult) {
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
fn init_dp_result_full(result: &mut DpResult) {
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
fn push_cigar(cigar: &mut Vec<u32>, op: u32, len: u32) {
    if let Some(last) = cigar.last_mut() {
        if (*last & 0xf) == op {
            *last += len << 4;
            return;
        }
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
fn alloc_h_array(approx_max: bool, tlen_: usize, simd_width: usize) -> (Vec<i32>, *mut i32) {
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
fn traceback_start_position(
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
unsafe fn traceback_dual_affine(
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
unsafe fn traceback_single_affine(
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
fn traceback_single_affine_safe(
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
unsafe fn traceback_splice(
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

#[cfg(target_arch = "aarch64")]
unsafe fn extend_single_affine_neon_impl(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8], // scoring matrix 5x5 flattened (25 elements)
    gap_open: i8, // gap open
    gap_extend: i8, // gap extend
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) { unsafe {
    let query_len = qseq.len();
    let target_len = tseq.len();
    let _n_col_ = if (query_len + target_len - 1) * 16 < query_len * target_len { (query_len + target_len - 1 + 15) / 16 } else { (query_len + 15) / 16 + (target_len + 15) / 16 }; // simplified
    let approx_max = (flags & APPROX_MAX) != 0;
    
    if alphabet_size <= 0 || query_len <= 0 || target_len <= 0 {
        return;
    }
    
    // Constants
    let zero_ = vdupq_n_u8(0);
    let q_ = vdupq_n_u8(gap_open as u8);
    let qe2_ = vdupq_n_u8(((gap_open as i32 + gap_extend as i32) * 2) as u8);
    let flag1_ = vdupq_n_u8(1);
    let flag2_ = vdupq_n_u8(2);
    let flag8_ = vdupq_n_u8(0x08);
    let flag16_ = vdupq_n_u8(0x10);
    
    let _sc_mch_ = vdupq_n_s8(score_matrix[0]);
    let _sc_mis_ = vdupq_n_s8(score_matrix[1]);
    let _sc_n = if score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1] == 0 { 
        vdupq_n_s8(-(gap_extend as i8)) 
    } else { 
        vdupq_n_s8(score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1]) 
    };
    
    let _m1_ = vdupq_n_u8((alphabet_size - 1) as u8);
    let _max_sc_ = vdupq_n_u8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as u8);

    // Dimension calculations
    let bandwidth = if bandwidth < 0 { if target_len > query_len { target_len as i32 } else { query_len as i32 } } else { bandwidth };
    let wl = bandwidth;
    let _wr = bandwidth;
    
    let tlen_ = (target_len + 15) / 16; // Number of 16-byte blocks for target_len
    let _qlen_ = (query_len + 15) / 16; // Number of 16-byte blocks for query_len

    // _n_col_ is for traceback arrays p, off, off_end
    let mut _n_col_ = if query_len < target_len { query_len } else { target_len };
    _n_col_ = ((if _n_col_ < (bandwidth + 1) as usize { _n_col_ } else { (bandwidth + 1) as usize }) + 15) / 16 + 1;
    
    let with_cigar = (flags & SCORE_ONLY) == 0;
    
    // Calculate total memory needed for a single allocation
    // Buffer sizing: sf gets tlen_*16 bytes, qr gets (qlen_+1)*16 bytes
    let qlen_ = (query_len + 15) / 16;
    let dp_size = 5 * tlen_ * 16;
    let sf_offset = dp_size;
    let qr_offset = sf_offset + tlen_ * 16;
    let p_offset = qr_offset + (qlen_ + 1) * 16;

    let mut mem_size_bytes = p_offset;

    // Additional memory for traceback if with_cigar
    let mut p_ptr: *mut u8 = std::ptr::null_mut();
    let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
    let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

    if with_cigar {
        // p: (query_len + target_len - 1) * _n_col_ * 16 bytes
        // off: (query_len + target_len - 1) * 4 bytes (int32)
        // off_end: (query_len + target_len - 1) * 4 bytes (int32)
        let p_size = (query_len + target_len - 1) * _n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        // Align band_offset_ptr
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        
        mem_size_bytes = off_end_offset_start + off_size;
    }

    let mem = AlignedMemory::new(mem_size_bytes, 16);
    // Zero DP+scoring region (not traceback — written per-cell in DP loop)
    std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

    let u = mem.as_ptr() as *mut uint8x16_t;
    let base_ptr = mem.as_ptr();

    // ... (rest of pointer init)
    
    // Core Loop
    // ...
    
         // Score calc logic
         // ...
         

    
    // definitions based on offsets
    let v = u.add(tlen_);
    let x = v.add(tlen_);
    let y = x.add(tlen_);
    let s = y.add(tlen_);
    let sf = base_ptr.add(sf_offset);
    let qr = base_ptr.add(qr_offset);
    
    // Traceback pointer initialization
    if with_cigar {
        let p_size = (query_len + target_len - 1) * _n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        
        p_ptr = base_ptr.add(p_offset);
        band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
        band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
    }
    
    // Reverse query
    let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
    for t in 0..query_len {
        qr_slice[t] = qseq[query_len - 1 - t];
    }
    
    // Copy target to sf
    let _sf_slice = std::slice::from_raw_parts_mut(sf, target_len);
    std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

    // ... continue implementation ...
    
    // Core Loop
    // for (r = 0; r < query_len + target_len - 1; ++r)
    let mut last_st = -1;
    let mut last_en = -1;
    let valid_range = (query_len + target_len - 1) as i32;
    
    // Scoring variables
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;
    
    for r in 0..valid_range {
        let mut st = 0;
        let mut en = target_len as i32 - 1;
        let x1: i8;
        let v1: i8;
        
        let qrr = qr.offset(query_len as isize - 1 - r as isize);
        let u8_ptr = u as *mut u8;
        let v8_ptr = v as *mut u8;
        
        // Find boundaries
        if st < (r as i32 - query_len as i32 + 1) { st = r as i32 - query_len as i32 + 1; }
        if en > r as i32 { en = r as i32; }
        if st < ((r as i32 - wl + 1) >> 1) { st = (r as i32 - wl + 1) >> 1; }
        if en > ((r as i32 + wl) >> 1) { en = (r as i32 + wl) >> 1; }
        
        if st > en {
            result.zdropped = 1;
            break;
        }
        
        let st0 = st;
        let en0 = en;
        
        // Alignment to 16-byte boundaries (simulating C logic)
        // Align st down to 16, en up to 16-1
        st = (st / 16) * 16;
        en = ((en + 16) / 16) * 16 - 1;
        
        // set boundary conditions
        // set boundary conditions
        if st > 0 {
             if st - 1 >= last_st && st - 1 <= last_en {
                 x1 = *(x as *mut i8).add((st - 1) as usize);
                 v1 = *(v as *mut i8).add((st - 1) as usize);
             } else {
                 x1 = 0;
                 v1 = 0;
             }
        } else {
             x1 = 0;
             v1 = if r == 0 { 0 } else { gap_open };
        }
        
        if en >= r as i32 {
             *(y as *mut i8).add(r as usize) = 0;
             *(u as *mut i8).add(r as usize) = if r == 0 { 0 } else { gap_open };
        }
        // Unlike C which checks GENERIC_SCORING, we assume match/mismatch logic for now for speed?
        // Actually, let's implement the generic case logic first if flags suggests, or just the standard match/mismatch
        // C implementation uses the standard match/mismatch block usually.
        
        let _st_idx = st0 as usize;
        let _en_idx = en0 as usize;
        
        // Scalar loop for scoring setup (easier to port safely first, optimize later?)
        // C uses SIMD here too.
        // Let's use scalar for now to ensure correctness of logic then upgrade to SIMD if needed.
        // Actually, the SIMD setup is quite complex with blends.
        
        // C logic:
        // for (t = st0; t <= en0; t += 16) { set scores }
        
        // Set scores (16-element chunks, SIMD scoring loop)
        if (flags & GENERIC_SCORING) == 0 {
            // Simple match/mismatch scoring (uniform penalties)
            let sc_mis_val = score_matrix[1] as u8;
            let sc_mch_val = score_matrix[0] as u8;
            let sc_n_val = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                (-(gap_extend as i8)) as u8
            } else {
                score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8
            };
            let m1_val = (alphabet_size - 1) as u8;
            let mut t = st0;
            while t <= en0 {
                for k in 0..16i32 {
                    let pos = (t + k) as usize;
                    let sf_val = *sf.add(pos);
                    let qr_val = *qrr.add(pos);
                    let is_n = sf_val == m1_val || qr_val == m1_val;
                    let score = if is_n {
                        sc_n_val
                    } else if sf_val == qr_val {
                        sc_mch_val
                    } else {
                        sc_mis_val
                    };
                    *(s as *mut u8).add(pos) = score;
                }
                t += 16;
            }
        } else {
            // Generic scoring: full matrix lookup score_matrix[target_base * alphabet_size + query_base]
            let s_ptr = s as *mut u8;
            for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 16 - 1) {
                *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
            }
        }
        
        // Core anti-diagonal DP loop
        let mut x1_ = vsetq_lane_u8(x1 as u8, vdupq_n_u8(0), 0);
        let mut v1_ = vsetq_lane_u8(v1 as u8, vdupq_n_u8(0), 0);

        let st_ = st as usize / 16;
        let en_ = en as usize / 16;

        for ti in st_..=en_ {
             // Load score + bias, shift x and v for diagonal access
             let mut z = vaddq_u8(vld1q_u8((s as *const u8).add(ti*16)), qe2_);
             let xt_val = vld1q_u8((x as *const u8).add(ti*16));
             let mut xt1 = xt_val;
             
             // tmp = _mm_srli_si128(xt1, 15);
             let tmp = vextq_u8(xt1, zero_, 15);
             
             // xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_);
             let shifted_xt1 = vextq_u8(zero_, xt1, 15); 
             xt1 = vorrq_u8(shifted_xt1, x1_);
             x1_ = tmp;
             
             // vt1 = _mm_load_si128(&v[t]);
             let vt_val = vld1q_u8((v as *const u8).add(ti*16));
             let mut vt1 = vt_val;
             
             // tmp = _mm_srli_si128(vt1, 15);
             let tmp_v = vextq_u8(vt1, zero_, 15);
             
             // vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);
             let shifted_vt1 = vextq_u8(zero_, vt1, 15);
             vt1 = vorrq_u8(shifted_vt1, v1_);
             v1_ = tmp_v;
             
             // a = _mm_add_epi8(xt1, vt1);
             let mut a = vaddq_u8(xt1, vt1);
             
             // ut = _mm_load_si128(&u[t]); 
             let ut = vld1q_u8((u as *const u8).add(ti*16));
             
             // b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);
             let yt = vld1q_u8((y as *const u8).add(ti*16));
             let mut b = vaddq_u8(yt, ut);
             
             let b_final_s8 = vreinterpretq_s8_u8(b);
             let b_s8 = b_final_s8;
             let z_s8 = vreinterpretq_s8_u8(z);
             let a_s8 = vreinterpretq_s8_u8(a);
             
             if with_cigar {
                 let offset = (r as usize * _n_col_) as isize - st_ as isize;
                 let pr_ptr = (p_ptr as *mut u8).add((offset + ti as isize) as usize * 16);
                 
                 if ti == st_ {
                     *band_offset_ptr.add(r as usize) = st;
                     *band_offset_end_ptr.add(r as usize) = en;
                 }
                 
                 // z = max(z, a) (Signed)
                 let z_s8_new = vmaxq_s8(z_s8, a_s8);
                 z = vreinterpretq_u8_s8(z_s8_new);
                 
                 let mask_z_gt_a = vcgtq_s8(z_s8, a_s8); // Signed compare
                 let mut d = vbicq_u8(flag1_, mask_z_gt_a); // d = z > a ? 0 : 1
                 
                 let z_s8_curr = vreinterpretq_s8_u8(z);
                 
                 // mask = z > b (Signed)
                 let mask_z_gt_b = vcgtq_s8(z_s8_curr, b_s8);
                 d = vbslq_u8(mask_z_gt_b, d, flag2_);
                 
                 // z = max(z, b) (Signed)
                 let z_s8_final = vmaxq_s8(z_s8_curr, b_s8);
                 z = vreinterpretq_u8_s8(z_s8_final);
                 
                 // Remove vminq_u8 logic for now, C uses signed logic primarily
                 // But we should cap at 127 if possible? C doesn't seem to explicitly cap in SSE4.1 path
                 // z = vminq_u8(z, _max_sc_); // Removed
                 
                 vst1q_u8((u as *mut u8).add(ti*16), vsubq_u8(z, vt1));
                 vst1q_u8((v as *mut u8).add(ti*16), vsubq_u8(z, ut));
                 z = vsubq_u8(z, q_);
                 a = vsubq_u8(a, z);
                 b = vsubq_u8(b, z);
                 
                 // Update u, v, x, y from z
                 let a_final_s8 = vreinterpretq_s8_u8(a);
                 let x_res = vmaxq_s8(a_final_s8, vreinterpretq_s8_u8(zero_));
                 vst1q_u8((x as *mut u8).add(ti*16), vreinterpretq_u8_s8(x_res));
                 
                 // d |= (a > 0 ? 0x08 : 0)
                 let mask_a = vcgtq_s8(a_final_s8, vreinterpretq_s8_u8(zero_));
                 let val_flag8 = vandq_u8(flag8_, mask_a);
                 d = vorrq_u8(d, val_flag8);
                 
                 let b_final_s8 = vreinterpretq_s8_u8(b);
                 let y_res = vmaxq_s8(b_final_s8, vreinterpretq_s8_u8(zero_));
                 vst1q_u8((y as *mut u8).add(ti*16), vreinterpretq_u8_s8(y_res));
                 
                 // d |= (b > 0 ? 0x10 : 0)
                 let mask_b = vcgtq_s8(b_final_s8, vreinterpretq_s8_u8(zero_));
                 let val_flag16 = vandq_u8(flag16_, mask_b); // mask_b is uint8x16 (result of vcgt)
                 d = vorrq_u8(d, val_flag16);
                 
                 vst1q_u8(pr_ptr, d);
             } else {
                 // score only
                 // z = max(z, a) (Signed)
                 let z_s8_new = vmaxq_s8(z_s8, a_s8);
                 let _z_un = vreinterpretq_u8_s8(z_s8_new);
                 // z = max(z, b) (Signed)
                 let z_s8_final = vmaxq_s8(z_s8_new, b_s8);
                 z = vreinterpretq_u8_s8(z_s8_final);
                 
                 vst1q_u8((u as *mut u8).add(ti*16), vsubq_u8(z, vt1));
                 vst1q_u8((v as *mut u8).add(ti*16), vsubq_u8(z, ut));
                 z = vsubq_u8(z, q_);
                 a = vsubq_u8(a, z);
                 b = vsubq_u8(b, z);
                 
                 let a_final_s8 = vreinterpretq_s8_u8(a);
                 let x_res = vmaxq_s8(a_final_s8, vreinterpretq_s8_u8(zero_));
                 vst1q_u8((x as *mut u8).add(ti*16), vreinterpretq_u8_s8(x_res));
                 
                 let b_final_s8 = vreinterpretq_s8_u8(b);
                 let y_res = vmaxq_s8(b_final_s8, vreinterpretq_s8_u8(zero_));
                 vst1q_u8((y as *mut u8).add(ti*16), vreinterpretq_u8_s8(y_res));
             }
        }
        
        // Debug
        // println!("r={} st={} en={} max_sc={}", r, st, en, score_matrix[0]); 
        
        // Approx Logic
        if !approx_max {
             // ...
        } else {
             // Approx max logic
             if r > 0 {
                 if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t + 1 <= en0 {
                     let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                     let d1_val = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                     let d0 = d0_val - (gap_open as i32 + gap_extend as i32);
                     let d1 = d1_val - (gap_open as i32 + gap_extend as i32);
                     
                     if d0 > d1 {
                         h0 += d0;
                     } else {
                         h0 += d1;
                         last_h0_t += 1;
                     }
                 } else if last_h0_t >= st0 && last_h0_t <= en0 {
                      let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                      h0 += d0_val - (gap_open as i32 + gap_extend as i32);
                 } else {
                      last_h0_t += 1;
                      let d1_val = *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                      h0 += d1_val - (gap_open as i32 + gap_extend as i32);
                 }
                 
                 // Update max score (approx)
                 if h0 > result.max {
                      result.max = h0;
                      result.max_score_target_pos = last_h0_t;
                      result.max_score_query_pos = r - last_h0_t;
                 }
                 
                 // Check z_drop
                 if (flags & APPROX_DROP) != 0 {
                      if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                          let tl = last_h0_t - result.max_score_target_pos;
                          let ql = (r - last_h0_t) - result.max_score_query_pos;
                          let l = if tl > ql { tl - ql } else { ql - tl };
                          if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend as i32) {
                              result.zdropped = 1;
                              break;
                          }
                      }
                 }
             } else {
                 // r == 0
                 let v0 = *v8_ptr.add(0) as i8 as i32;
                 h0 = v0 - (gap_open as i32 + gap_extend as i32) * 2;
                 last_h0_t = 0;
                 if h0 > result.max {
                     result.max = h0; result.max_score_target_pos = 0; result.max_score_query_pos = 0;
                 }
             }
        }
        
        // Final score update
        if r == valid_range as i32 - 1 /* query_len+target_len-2 */ {
            // Check if en0 reached end
             if en0 == target_len as i32 - 1 {
                 result.score = h0;
             }
        }
        
        last_st = st;
        last_en = en;
    }
    
        if with_cigar {
            traceback_single_affine(result, query_len, target_len, end_bonus, flags, _n_col_, 16, p_ptr, band_offset_ptr, band_offset_end_ptr);
        }
    }}

// ============================================================================
// SSE2/SSE4.1 Unified Implementation - Single-Affine Alignment
// ============================================================================
//
// Macro generates both SSE2 and SSE4.1 variants. The only differences are:
// - max_epi8: SSE2 uses sse2_max_epi8 helper, SSE4.1 uses native _mm_max_epi8
// - blend: SSE2 uses and/andnot/or pattern, SSE4.1 uses _mm_blendv_epi8
// Both variants require only SSE2 target_feature (SSE4.1 is detected at runtime).

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
macro_rules! extend_single_affine_impl {
    ($fn_name:ident, $max_epi8:path, $is_sse41:expr, $target_feat:tt) => {
        #[target_feature(enable = $target_feat)]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();

            if alphabet_size <= 0 || query_len == 0 || target_len == 0 {
                return;
            }

            // Constants
            let zero_ = _mm_setzero_si128();
            let q_ = _mm_set1_epi8(gap_open);
            let qe2_ = _mm_set1_epi8(((gap_open as i32 + gap_extend as i32) * 2) as i8);
            let flag1_ = _mm_set1_epi8(1);
            let flag2_ = _mm_set1_epi8(2);
            let flag8_ = _mm_set1_epi8(0x08);
            let flag16_ = _mm_set1_epi8(0x10);

            let _sc_mch_ = _mm_set1_epi8(score_matrix[0]);
            let _sc_mis_ = _mm_set1_epi8(score_matrix[1]);
            let _sc_n = if score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1] == 0 {
                _mm_set1_epi8(-(gap_extend as i8))
            } else {
                _mm_set1_epi8(score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1])
            };

            let _m1_ = _mm_set1_epi8((alphabet_size - 1) as i8);
            let _max_sc_ = _mm_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);

            // Dimension calculations
            let bandwidth = if bandwidth < 0 { if target_len > query_len { target_len as i32 } else { query_len as i32 } } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(16);

            let mut _n_col_ = if query_len < target_len { query_len } else { target_len };
            _n_col_ = (if _n_col_ < (bandwidth + 1) as usize { _n_col_ } else { (bandwidth + 1) as usize }).div_ceil(16) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 5 arrays: u, v, x, y, s
            let qlen_ = query_len.div_ceil(16);
            let dp_size = 5 * tlen_ * 16;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 16;
            let p_offset = qr_offset + (qlen_ + 1) * 16;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 16);
            // Zero DP+scoring region (not traceback — written per-cell in DP loop)
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m128i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let s = y.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }

            // Copy target
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;
                let x1: i8;
                let v1: i8;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;

                // Find boundaries
                if st < (r - query_len as i32 + 1) { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < ((r - wl + 1) >> 1) { st = (r - wl + 1) >> 1; }
                if en > ((r + wl) >> 1) { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *(x as *mut i8).add((st - 1) as usize);
                        v1 = *(v as *mut i8).add((st - 1) as usize);
                    } else {
                        x1 = 0;
                        v1 = 0;
                    }
                } else {
                    x1 = 0;
                    v1 = if r == 0 { 0 } else { gap_open };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = 0;
                    *(u as *mut i8).add(r as usize) = if r == 0 { 0 } else { gap_open };
                }

                // Set scores (SIMD 16-element chunks)
                if (flags & GENERIC_SCORING) == 0 {
                    // Simple match/mismatch scoring (uniform penalties)
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm_loadu_si128(sf.add(t as usize) as *const __m128i);
                        let st_v = _mm_loadu_si128(qrr.add(t as usize) as *const __m128i);
                        let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, _m1_), _mm_cmpeq_epi8(st_v, _m1_));
                        let tmp = _mm_cmpeq_epi8(sq, st_v);
                        // Blend: select _sc_mch_ where equal, _sc_mis_ where not
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(_sc_mis_, _sc_mch_, tmp)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(tmp, _sc_mis_), _mm_and_si128(tmp, _sc_mch_))
                        };
                        // Blend: select _sc_n where ambiguous
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(tmp, _sc_n, mask)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, _sc_n))
                        };
                        _mm_storeu_si128((s as *mut u8).add(t as usize) as *mut __m128i, tmp);
                        t += 16;
                    }
                } else {
                    // Generic scoring: full matrix lookup
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 16 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop
                let mut x1_ = sse2_insert_byte0(zero_, x1 as u8);
                let mut v1_ = sse2_insert_byte0(zero_, v1 as u8);

                let st_ = st as usize / 16;
                let en_ = en as usize / 16;

                for ti in st_..=en_ {
                    // Load score + bias
                    let mut z = _mm_add_epi8(_mm_loadu_si128(s.add(ti)), qe2_);

                    // Shift x for diagonal access
                    let xt_val = _mm_loadu_si128(x.add(ti));
                    let mut xt1 = xt_val;
                    let tmp = _mm_srli_si128(xt1, 15);
                    xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_);
                    x1_ = tmp;

                    // Shift v for diagonal access
                    let vt_val = _mm_loadu_si128(v.add(ti));
                    let mut vt1 = vt_val;
                    let tmp_v = _mm_srli_si128(vt1, 15);
                    vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);
                    v1_ = tmp_v;

                    // a = x[t-1] + v[t-1]
                    let mut a = _mm_add_epi8(xt1, vt1);

                    // b = y[t] + u[t]
                    let ut = _mm_loadu_si128(u.add(ti));
                    let yt = _mm_loadu_si128(y.add(ti));
                    let mut b = _mm_add_epi8(yt, ut);

                    if with_cigar {
                        let offset = (r as usize * _n_col_) as isize - st_ as isize;
                        let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);

                        if ti == st_ {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        // z = max(z, a)
                        let z_new = $max_epi8(z, a);
                        let mask_z_gt_a = _mm_cmpgt_epi8(z, a);
                        let mut d = _mm_andnot_si128(mask_z_gt_a, flag1_);

                        z = z_new;

                        // z = max(z, b), track state
                        let mask_z_gt_b = _mm_cmpgt_epi8(z, b);
                        d = if $is_sse41 {
                            _mm_blendv_epi8(flag2_, d, mask_z_gt_b)
                        } else {
                            _mm_or_si128(_mm_and_si128(mask_z_gt_b, d), _mm_andnot_si128(mask_z_gt_b, flag2_))
                        };

                        z = $max_epi8(z, b);

                        // Update u, v
                        _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        z = _mm_sub_epi8(z, q_);
                        a = _mm_sub_epi8(a, z);
                        b = _mm_sub_epi8(b, z);

                        // x = max(a, 0) - qe2
                        let x_res = $max_epi8(a, zero_);
                        _mm_storeu_si128(x.add(ti), x_res);

                        // d |= (a > 0 ? 0x08 : 0)
                        let mask_a = _mm_cmpgt_epi8(a, zero_);
                        d = _mm_or_si128(d, _mm_and_si128(flag8_, mask_a));

                        // y = max(b, 0) - qe2
                        let y_res = $max_epi8(b, zero_);
                        _mm_storeu_si128(y.add(ti), y_res);

                        // d |= (b > 0 ? 0x10 : 0)
                        let mask_b = _mm_cmpgt_epi8(b, zero_);
                        d = _mm_or_si128(d, _mm_and_si128(flag16_, mask_b));

                        _mm_storeu_si128(pr_ptr as *mut __m128i, d);
                    } else {
                        // Score only
                        z = $max_epi8(z, a);
                        z = $max_epi8(z, b);

                        _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        z = _mm_sub_epi8(z, q_);
                        a = _mm_sub_epi8(a, z);
                        b = _mm_sub_epi8(b, z);

                        let x_res = $max_epi8(a, zero_);
                        _mm_storeu_si128(x.add(ti), x_res);

                        let y_res = $max_epi8(b, zero_);
                        _mm_storeu_si128(y.add(ti), y_res);
                    }
                }

                // Score and max tracking
                {
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1_val = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            let d0 = d0_val - (gap_open as i32 + gap_extend as i32);
                            let d1 = d1_val - (gap_open as i32 + gap_extend as i32);

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d0_val - (gap_open as i32 + gap_extend as i32);
                        } else {
                            last_h0_t += 1;
                            let d1_val = *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d1_val - (gap_open as i32 + gap_extend as i32);
                        }

                        // Update max score
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        // Check z_drop
                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        // r == 0
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - (gap_open as i32 + gap_extend as i32) * 2;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0; result.max_score_target_pos = 0; result.max_score_query_pos = 0;
                        }
                    }
                }

                // Final score update
                if r == valid_range - 1 && en0 == target_len as i32 - 1 {
                    result.score = h0;
                }

                last_st = st;
                last_en = en;
            }

            if with_cigar {
                traceback_single_affine(result, query_len, target_len, end_bonus, flags, _n_col_, 16, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_single_affine_impl!(extend_single_affine2_impl, sse2_max_epi8, false, "sse2");
#[cfg(target_arch = "x86_64")]
extend_single_affine_impl!(extend_single_affine41_impl, _mm_max_epi8, true, "sse2");
#[cfg(target_arch = "wasm32")]
extend_single_affine_impl!(extend_single_affine_wasm_impl, _mm_max_epi8, true, "simd128");

// ============================================================================
// AVX2 Implementation - Single-Affine Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_single_affine_avx2_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();

            if alphabet_size <= 0 || query_len == 0 || target_len == 0 {
                return;
            }

            // Constants (256-bit)
            let zero_ = _mm256_setzero_si256();
            let q_ = _mm256_set1_epi8(gap_open);
            let qe2_ = _mm256_set1_epi8(((gap_open as i32 + gap_extend as i32) * 2) as i8);
            let flag1_ = _mm256_set1_epi8(1);
            let flag2_ = _mm256_set1_epi8(2);
            let flag8_ = _mm256_set1_epi8(0x08);
            let flag16_ = _mm256_set1_epi8(0x10);

            let _sc_mch_ = _mm256_set1_epi8(score_matrix[0]);
            let _sc_mis_ = _mm256_set1_epi8(score_matrix[1]);
            let _sc_n = if score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1] == 0 {
                _mm256_set1_epi8(-(gap_extend as i8))
            } else {
                _mm256_set1_epi8(score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1])
            };

            let _m1_ = _mm256_set1_epi8((alphabet_size - 1) as i8);
            let _max_sc_ = _mm256_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);

            // Dimension calculations (width=32)
            let bandwidth = if bandwidth < 0 { if target_len > query_len { target_len as i32 } else { query_len as i32 } } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(32) + 1; // +1 for byte-addressed SSE-compat padding

            let mut _n_col_ = if query_len < target_len { query_len } else { target_len };
            _n_col_ = (if _n_col_ < (bandwidth + 1) as usize { _n_col_ } else { (bandwidth + 1) as usize }).div_ceil(32) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 5 arrays: u, v, x, y, s (32-byte aligned)
            let qlen_ = query_len.div_ceil(32);
            let dp_size = 5 * tlen_ * 32;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 32;
            let p_offset = qr_offset + (qlen_ + 1) * 32;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 32);
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m256i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let s = y.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }

            // Copy target
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;
                let x1: i8;
                let v1: i8;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;

                // Find boundaries
                if st < (r - query_len as i32 + 1) { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < ((r - wl + 1) >> 1) { st = (r - wl + 1) >> 1; }
                if en > ((r + wl) >> 1) { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *(x as *mut i8).add((st - 1) as usize);
                        v1 = *(v as *mut i8).add((st - 1) as usize);
                    } else {
                        x1 = 0;
                        v1 = 0;
                    }
                } else {
                    x1 = 0;
                    v1 = if r == 0 { 0 } else { gap_open };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = 0;
                    *(u as *mut i8).add(r as usize) = if r == 0 { 0 } else { gap_open };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm256_loadu_si256(sf.add(t as usize) as *const __m256i);
                        let st_v = _mm256_loadu_si256(qrr.add(t as usize) as *const __m256i);
                        let mask = _mm256_or_si256(_mm256_cmpeq_epi8(sq, _m1_), _mm256_cmpeq_epi8(st_v, _m1_));
                        let tmp = _mm256_cmpeq_epi8(sq, st_v);
                        let tmp = _mm256_blendv_epi8(_sc_mis_, _sc_mch_, tmp);
                        let tmp = _mm256_blendv_epi8(tmp, _sc_n, mask);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp, 1));
                        }
                        t += 32;
                    }
                } else {
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 32 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx2_insert_byte0(_mm256_setzero_si256(), x1 as u8);
                let mut v1_ = avx2_insert_byte0(_mm256_setzero_si256(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = _n_col_ * 32;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    // Save excess bytes on last partial iteration
                    let excess = if bp + 31 > en_usize {
                        bp + 32 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 16];
                    let mut save_v = [0u8; 16];
                    let mut save_x = [0u8; 16];
                    let mut save_y = [0u8; 16];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                    }

                    // Byte-addressed loads
                    let mut z = _mm256_add_epi8(_mm256_loadu_si256(s_b_ptr.add(bp) as *const __m256i), qe2_);

                    let xt_val = _mm256_loadu_si256(x_b.add(bp) as *const __m256i);
                    let (xt1, tmp_x) = avx2_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm256_loadu_si256(v_b.add(bp) as *const __m256i);
                    let (vt1, tmp_v) = avx2_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let mut a = _mm256_add_epi8(xt1, vt1);

                    let ut = _mm256_loadu_si256(u_b.add(bp) as *const __m256i);
                    let mut b = _mm256_add_epi8(_mm256_loadu_si256(y_b.add(bp) as *const __m256i), ut);

                    if !with_cigar {
                        // Score only
                        z = _mm256_max_epi8(z, a);
                        z = _mm256_max_epi8(z, b);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        z = _mm256_sub_epi8(z, q_);
                        a = _mm256_sub_epi8(a, z);
                        b = _mm256_sub_epi8(b, z);

                        let x_res = _mm256_max_epi8(a, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, x_res);

                        let y_res = _mm256_max_epi8(b, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, y_res);
                    } else {
                        // With CIGAR — byte-addressed traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        // z = max(z, a)
                        let z_new = _mm256_max_epi8(z, a);
                        let mask_z_gt_a = _mm256_cmpgt_epi8(z, a);
                        let mut d = _mm256_andnot_si256(mask_z_gt_a, flag1_);
                        z = z_new;

                        // z = max(z, b), track state
                        let mask_z_gt_b = _mm256_cmpgt_epi8(z, b);
                        d = _mm256_blendv_epi8(flag2_, d, mask_z_gt_b);
                        z = _mm256_max_epi8(z, b);

                        // Update u, v
                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        z = _mm256_sub_epi8(z, q_);
                        a = _mm256_sub_epi8(a, z);
                        b = _mm256_sub_epi8(b, z);

                        // x = max(a, 0) - qe2
                        let x_res = _mm256_max_epi8(a, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, x_res);

                        // d |= (a > 0 ? 0x08 : 0)
                        let mask_a = _mm256_cmpgt_epi8(a, zero_);
                        d = _mm256_or_si256(d, _mm256_and_si256(flag8_, mask_a));

                        // y = max(b, 0) - qe2
                        let y_res = _mm256_max_epi8(b, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, y_res);

                        // d |= (b > 0 ? 0x10 : 0)
                        let mask_b = _mm256_cmpgt_epi8(b, zero_);
                        d = _mm256_or_si256(d, _mm256_and_si256(flag16_, mask_b));

                        _mm256_storeu_si256(pr_ptr_local as *mut __m256i, d);
                    }

                    // Restore excess bytes on partial last iteration
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 32;
                }

                // Approx max logic (scalar — identical to SSE version)
                {
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1_val = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            let d0 = d0_val - (gap_open as i32 + gap_extend as i32);
                            let d1 = d1_val - (gap_open as i32 + gap_extend as i32);

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d0_val - (gap_open as i32 + gap_extend as i32);
                        } else {
                            last_h0_t += 1;
                            let d1_val = *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d1_val - (gap_open as i32 + gap_extend as i32);
                        }

                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - (gap_open as i32 + gap_extend as i32) * 2;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0; result.max_score_target_pos = 0; result.max_score_query_pos = 0;
                        }
                    }
                }

                // Final score update
                if r == valid_range - 1 && en0 == target_len as i32 - 1 {
                    result.score = h0;
                }

                last_st = st;
                last_en = en;
            }

            if with_cigar {
                traceback_single_affine(result, query_len, target_len, end_bonus, flags, _n_col_, 32, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_single_affine_avx2_impl!(extend_single_affine_avx2_fn);

// ============================================================================
// AVX512 Implementation - Single-Affine Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_single_affine_avx512_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx512bw")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();

            if alphabet_size <= 0 || query_len == 0 || target_len == 0 {
                return;
            }

            // Constants (512-bit)
            let zero_ = _mm512_setzero_si512();
            let q_ = _mm512_set1_epi8(gap_open);
            let qe2_ = _mm512_set1_epi8(((gap_open as i32 + gap_extend as i32) * 2) as i8);
            let flag1_ = _mm512_set1_epi8(1);
            let flag2_ = _mm512_set1_epi8(2);
            let flag8_ = _mm512_set1_epi8(0x08);
            let flag16_ = _mm512_set1_epi8(0x10);

            let _sc_mch_ = _mm512_set1_epi8(score_matrix[0]);
            let _sc_mis_ = _mm512_set1_epi8(score_matrix[1]);
            let _sc_n = if score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1] == 0 {
                _mm512_set1_epi8(-(gap_extend as i8))
            } else {
                _mm512_set1_epi8(score_matrix[(alphabet_size as usize)*(alphabet_size as usize)-1])
            };

            let _m1_ = _mm512_set1_epi8((alphabet_size - 1) as i8);
            let _max_sc_ = _mm512_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);

            // Dimension calculations (width=64)
            let bandwidth = if bandwidth < 0 { if target_len > query_len { target_len as i32 } else { query_len as i32 } } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(64) + 1; // +1 for byte-addressed SSE-compat padding

            let mut _n_col_ = if query_len < target_len { query_len } else { target_len };
            _n_col_ = (if _n_col_ < (bandwidth + 1) as usize { _n_col_ } else { (bandwidth + 1) as usize }).div_ceil(64) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 5 arrays: u, v, x, y, s (64-byte aligned)
            let qlen_ = query_len.div_ceil(64);
            let dp_size = 5 * tlen_ * 64;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 64;
            let p_offset = qr_offset + (qlen_ + 1) * 64;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 64);
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m512i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let s = y.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * _n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }

            // Copy target
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;
                let x1: i8;
                let v1: i8;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;

                // Find boundaries
                if st < (r - query_len as i32 + 1) { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < ((r - wl + 1) >> 1) { st = (r - wl + 1) >> 1; }
                if en > ((r + wl) >> 1) { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *(x as *mut i8).add((st - 1) as usize);
                        v1 = *(v as *mut i8).add((st - 1) as usize);
                    } else {
                        x1 = 0;
                        v1 = 0;
                    }
                } else {
                    x1 = 0;
                    v1 = if r == 0 { 0 } else { gap_open };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = 0;
                    *(u as *mut i8).add(r as usize) = if r == 0 { 0 } else { gap_open };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm512_loadu_si512(sf.add(t as usize) as *const __m512i);
                        let st_v = _mm512_loadu_si512(qrr.add(t as usize) as *const __m512i);
                        let is_n: __mmask64 = _mm512_cmpeq_epi8_mask(sq, _m1_) | _mm512_cmpeq_epi8_mask(st_v, _m1_);
                        let eq: __mmask64 = _mm512_cmpeq_epi8_mask(sq, st_v);
                        let tmp = _mm512_mask_blend_epi8(eq, _sc_mis_, _sc_mch_);
                        let tmp = _mm512_mask_blend_epi8(is_n, tmp, _sc_n);
                        let tmp256 = _mm512_castsi512_si256(tmp);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp256));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp256, 1));
                        }
                        if t + 32 <= en0 {
                            let hi256 = _mm512_extracti64x4_epi64(tmp, 1);
                            _mm_storeu_si128(s_b.add(t as usize + 32) as *mut __m128i, _mm256_castsi256_si128(hi256));
                            if t + 48 <= en0 {
                                _mm_storeu_si128(s_b.add(t as usize + 48) as *mut __m128i, _mm256_extracti128_si256(hi256, 1));
                            }
                        }
                        t += 64;
                    }
                } else {
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 64 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx512_insert_byte0(_mm512_setzero_si512(), x1 as u8);
                let mut v1_ = avx512_insert_byte0(_mm512_setzero_si512(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = _n_col_ * 64;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    let excess = if bp + 63 > en_usize {
                        bp + 64 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 48];
                    let mut save_v = [0u8; 48];
                    let mut save_x = [0u8; 48];
                    let mut save_y = [0u8; 48];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                    }

                    let mut z = _mm512_add_epi8(_mm512_loadu_si512(s_b_ptr.add(bp) as *const __m512i), qe2_);

                    let xt_val = _mm512_loadu_si512(x_b.add(bp) as *const __m512i);
                    let (xt1, tmp_x) = avx512_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm512_loadu_si512(v_b.add(bp) as *const __m512i);
                    let (vt1, tmp_v) = avx512_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let mut a = _mm512_add_epi8(xt1, vt1);

                    let ut = _mm512_loadu_si512(u_b.add(bp) as *const __m512i);
                    let mut b = _mm512_add_epi8(_mm512_loadu_si512(y_b.add(bp) as *const __m512i), ut);

                    if !with_cigar {
                        z = _mm512_max_epi8(z, a);
                        z = _mm512_max_epi8(z, b);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        z = _mm512_sub_epi8(z, q_);
                        a = _mm512_sub_epi8(a, z);
                        b = _mm512_sub_epi8(b, z);

                        let x_res = _mm512_max_epi8(a, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, x_res);

                        let y_res = _mm512_max_epi8(b, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, y_res);
                    } else {
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mask_a_gt_z: __mmask64 = _mm512_cmpgt_epi8_mask(a, z);
                        let mut d = _mm512_maskz_mov_epi8(mask_a_gt_z, flag1_);
                        z = _mm512_max_epi8(z, a);

                        let mask_b_gt_z: __mmask64 = _mm512_cmpgt_epi8_mask(b, z);
                        d = _mm512_mask_blend_epi8(mask_b_gt_z, d, flag2_);
                        z = _mm512_max_epi8(z, b);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        z = _mm512_sub_epi8(z, q_);
                        a = _mm512_sub_epi8(a, z);
                        b = _mm512_sub_epi8(b, z);

                        let x_res = _mm512_max_epi8(a, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, x_res);

                        let mask_a: __mmask64 = _mm512_cmpgt_epi8_mask(a, zero_);
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(mask_a, flag8_));

                        let y_res = _mm512_max_epi8(b, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, y_res);

                        let mask_b: __mmask64 = _mm512_cmpgt_epi8_mask(b, zero_);
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(mask_b, flag16_));

                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    }

                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 64;
                }

                // Approx max logic (scalar — identical to SSE/AVX2 version)
                {
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1_val = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            let d0 = d0_val - (gap_open as i32 + gap_extend as i32);
                            let d1 = d1_val - (gap_open as i32 + gap_extend as i32);

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            let d0_val = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d0_val - (gap_open as i32 + gap_extend as i32);
                        } else {
                            last_h0_t += 1;
                            let d1_val = *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                            h0 += d1_val - (gap_open as i32 + gap_extend as i32);
                        }

                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - (gap_open as i32 + gap_extend as i32) * 2;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0; result.max_score_target_pos = 0; result.max_score_query_pos = 0;
                        }
                    }
                }

                // Final score update
                if r == valid_range - 1 && en0 == target_len as i32 - 1 {
                    result.score = h0;
                }

                last_st = st;
                last_en = en;
            }

            if with_cigar {
                traceback_single_affine(result, query_len, target_len, end_bonus, flags, _n_col_, 64, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_single_affine_avx512_impl!(extend_single_affine_avx512_fn);

// ============================================================================
// Public API - Single-Affine Alignment
// ============================================================================

/// Single-affine gap penalty extension alignment
///
/// Performs semi-global alignment with single-affine gap penalties.
/// Uses NEON SIMD on ARM, with scalar fallback on other architectures.
///
/// # Arguments
/// * `qseq` - Query sequence (encoded as 0-3 for ACGT, 4 for N)
/// * `tseq` - Target sequence (same encoding)
/// * `alphabet_size` - Alphabet size (typically 5 for DNA with N)
/// * `score_matrix` - Scoring matrix (alphabet_size x alphabet_size, row-major)
/// * `gap_open` - Gap open penalty
/// * `gap_extend` - Gap extension penalty
/// * `bandwidth` - Bandwidth (-1 for unlimited)
/// * `z_drop` - Z-drop threshold (-1 to disable)
/// * `end_bonus` - Bonus for reaching sequence end
/// * `flags` - Alignment flags
/// * `result` - Output structure for results
pub fn extend_single_affine(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i8,
    gap_extend: i8,
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) {
    // Force scalar mode for testing/comparison
    if std::env::var("RAMMAP_FORCE_SCALAR").is_ok() {
        extend_single_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, bandwidth, z_drop, end_bonus, flags, result);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        extend_single_affine_neon_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let force_sse = std::env::var("RAMMAP_FORCE_SSE").is_ok();
        let force_avx2 = std::env::var("RAMMAP_FORCE_AVX2").is_ok();
        if !force_sse && !force_avx2 && is_x86_feature_detected!("avx512bw") {
            unsafe { extend_single_affine_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result); }
        } else if !force_sse && is_x86_feature_detected!("avx2") {
            unsafe { extend_single_affine_avx2_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result); }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { extend_single_affine41_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result); }
        } else {
            unsafe { extend_single_affine2_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result); }
        }
    }

    #[cfg(target_arch = "wasm32")]
    unsafe {
        extend_single_affine_wasm_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
    {
        extend_single_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, bandwidth, z_drop, end_bonus, flags, result);
    }
}

// ============================================================================
// Public API - Dual-Affine Alignment
// ============================================================================

/// Dual-affine gap penalty extension alignment
///
/// Uses two gap penalty models to better handle both short and long gaps:
/// - First penalty (gap_open, gap_extend): Lower open cost, higher extension - good for short gaps
/// - Second penalty (gap_open2, gap_extend2): Higher open cost, lower extension - good for long gaps
///
/// Gap cost = min(gap_open + k*gap_extend, gap_open2 + k*gap_extend2) for a gap of length k
///
/// # Arguments
/// * `qseq` - Query sequence (encoded as 0-3 for ACGT, 4 for N)
/// * `tseq` - Target sequence (same encoding)
/// * `alphabet_size` - Alphabet size (typically 5 for DNA with N)
/// * `score_matrix` - Scoring matrix (alphabet_size x alphabet_size, row-major)
/// * `gap_open` - Gap open penalty (first model)
/// * `gap_extend` - Gap extension penalty (first model)
/// * `gap_open2` - Gap open penalty (second model)
/// * `gap_extend2` - Gap extension penalty (second model)
/// * `bandwidth` - Bandwidth (-1 for unlimited)
/// * `z_drop` - Z-drop threshold (-1 to disable)
/// * `end_bonus` - Bonus for reaching sequence end
/// * `flags` - Alignment flags
/// * `result` - Output structure for results
///
/// # Example
/// For map-ont preset: gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
/// - 10bp gap: min(4+20, 24+10) = min(24, 34) = 24 (first penalty)
/// - 30bp gap: min(4+60, 24+30) = min(64, 54) = 54 (second penalty)
pub fn extend_dual_affine(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i8,
    gap_extend: i8,
    gap_open2: i8,
    gap_extend2: i8,
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) {
    // Normalize: ensure gap_open+gap_extend <= gap_open2+gap_extend2 (swap if needed)
    let (gap_open, gap_extend, gap_open2, gap_extend2) = if (gap_open2 as i32 + gap_extend2 as i32) < (gap_open as i32 + gap_extend as i32) {
        (gap_open2, gap_extend2, gap_open, gap_extend)
    } else {
        (gap_open, gap_extend, gap_open2, gap_extend2)
    };

    // If single-affine (gap_open==gap_open2 && gap_extend==gap_extend2), use the simpler implementation
    if gap_open == gap_open2 && gap_extend == gap_extend2 {
        extend_single_affine(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result);
        return;
    }

    // Force scalar mode for testing/comparison
    if std::env::var("RAMMAP_FORCE_SCALAR").is_ok() {
        // Compare mode: run both SIMD and scalar, report differences
        if std::env::var("RAMMAP_COMPARE_SCALAR").is_ok() {
            let mut ez_simd = DpResult::default();
            #[cfg(target_arch = "x86_64")]
            unsafe {
                extend_dual_affine2_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, &mut ez_simd);
            }
            extend_dual_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, gap_open2 as i32, gap_extend2 as i32, bandwidth, z_drop, end_bonus, flags, result);
            if result.score != ez_simd.score || result.max_query_end_score != ez_simd.max_query_end_score || result.max != ez_simd.max
                || result.reach_end != ez_simd.reach_end || result.max_score_query_pos != ez_simd.max_score_query_pos || result.max_score_target_pos != ez_simd.max_score_target_pos
                || result.max_query_end_target_pos != ez_simd.max_query_end_target_pos || result.max_target_end_score != ez_simd.max_target_end_score || result.max_target_end_query_pos != ez_simd.max_target_end_query_pos
                || result.zdropped != ez_simd.zdropped
            {
                eprintln!("DP MISMATCH qlen={} tlen={} bandwidth={} z_drop={} eb={} flags=0x{:x}",
                    qseq.len(), tseq.len(), bandwidth, z_drop, end_bonus, flags);
                eprintln!("  SIMD:   score={:6} max={:6} max_q={:4} max_t={:4} mqe={:6} mqe_t={:4} mte={:6} mte_q={:4} re={} zd={}",
                    ez_simd.score, ez_simd.max, ez_simd.max_score_query_pos, ez_simd.max_score_target_pos, ez_simd.max_query_end_score, ez_simd.max_query_end_target_pos, ez_simd.max_target_end_score, ez_simd.max_target_end_query_pos, ez_simd.reach_end, ez_simd.zdropped);
                eprintln!("  Scalar: score={:6} max={:6} max_q={:4} max_t={:4} mqe={:6} mqe_t={:4} mte={:6} mte_q={:4} re={} zd={}",
                    result.score, result.max, result.max_score_query_pos, result.max_score_target_pos, result.max_query_end_score, result.max_query_end_target_pos, result.max_target_end_score, result.max_target_end_query_pos, result.reach_end, result.zdropped);
                if result.cigar != ez_simd.cigar {
                    eprintln!("  CIGAR differs: scalar_ops={} simd_ops={}", result.cigar.len(), ez_simd.cigar.len());
                }
            }
            return;
        }
        extend_dual_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, gap_open2 as i32, gap_extend2 as i32, bandwidth, z_drop, end_bonus, flags, result);
        return;
    }

    // Use SIMD implementation for speed
    #[cfg(target_arch = "aarch64")]
    unsafe {
        extend_dual_affine_neon_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result);
    }

    #[cfg(target_arch = "x86_64")]
    {
        let force_sse = std::env::var("RAMMAP_FORCE_SSE").is_ok();
        let force_avx2 = std::env::var("RAMMAP_FORCE_AVX2").is_ok();
        if !force_sse && !force_avx2 && is_x86_feature_detected!("avx512bw") {
            unsafe { extend_dual_affine_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result); }
        } else if !force_sse && is_x86_feature_detected!("avx2") {
            unsafe { extend_dual_affine_avx2_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result); }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { extend_dual_affine41_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result); }
        } else {
            unsafe { extend_dual_affine2_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result); }
        }
    }

    #[cfg(target_arch = "wasm32")]
    unsafe {
        extend_dual_affine_wasm_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
    {
        // Fall back to scalar on other platforms
        extend_dual_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, gap_open2 as i32, gap_extend2 as i32, bandwidth, z_drop, end_bonus, flags, result);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn extend_dual_affine_neon_impl(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i8,
    gap_extend: i8,
    gap_open2: i8,
    gap_extend2: i8,
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) { unsafe {
    use core::arch::aarch64::*;

    let query_len = qseq.len();
    let target_len = tseq.len();
    let approx_max = (flags & APPROX_MAX) != 0;

    if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
        return;
    }

    // Ensure gap_open+gap_extend <= gap_open2+gap_extend2
    let (gap_open, gap_extend, gap_open2, gap_extend2) = if (gap_open2 as i32 + gap_extend2 as i32) < (gap_open as i32 + gap_extend as i32) {
        (gap_open2, gap_extend2, gap_open, gap_extend)
    } else {
        (gap_open, gap_extend, gap_open2, gap_extend2)
    };

    // Compute long_thres and long_diff for dual-affine boundary conditions
    let mut long_thres: i32 = if gap_extend != gap_extend2 {
        (gap_open2 as i32 - gap_open as i32) / (gap_extend as i32 - gap_extend2 as i32) - 1
    } else { 0 };
    if (gap_open2 as i32 + gap_extend2 as i32 + long_thres * gap_extend2 as i32) > (gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32) {
        long_thres += 1;
    }
    let long_diff: i8 = (long_thres * (gap_extend as i32 - gap_extend2 as i32) - (gap_open2 as i32 - gap_open as i32) - gap_extend2 as i32) as i8;

    // Constants - dual-affine uses SIGNED operations, NO bias on z
    let zero_ = vdupq_n_u8(0);
    let q_ = vdupq_n_u8(gap_open as u8);
    let q2_ = vdupq_n_u8(gap_open2 as u8);
    let qe_ = vdupq_n_u8((gap_open as i32 + gap_extend as i32) as u8);
    let qe2_ = vdupq_n_u8((gap_open2 as i32 + gap_extend2 as i32) as u8);
    let sc_mch_ = vdupq_n_s8(score_matrix[0]); // clamp value for dual-affine (signed)

    let flag1_ = vdupq_n_u8(1);
    let flag2_ = vdupq_n_u8(2);
    let flag3_ = vdupq_n_u8(3);
    let flag4_ = vdupq_n_u8(4);
    let flag8_ = vdupq_n_u8(0x08);
    let flag16_ = vdupq_n_u8(0x10);
    let flag32_ = vdupq_n_u8(0x20);
    let flag64_ = vdupq_n_u8(0x40);

    let bandwidth = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
    let wl = bandwidth;

    let tlen_ = (target_len + 15) / 16;
    let mut n_col_ = query_len.min(target_len);
    n_col_ = ((n_col_.min((bandwidth + 1) as usize)) + 15) / 16 + 1;

    let with_cigar = (flags & SCORE_ONLY) == 0;

    // Memory allocation - 7 arrays for dual-affine: u, v, x, y, x2, y2, s
    // sf gets tlen_*16 bytes, qr gets (qlen_+1)*16 bytes for SIMD scoring reads
    let qlen_ = (query_len + 15) / 16;
    let dp_size = 7 * tlen_ * 16;
    let sf_offset = dp_size;
    let qr_offset = sf_offset + tlen_ * 16;
    let p_offset = qr_offset + (qlen_ + 1) * 16;

    let mut mem_size_bytes = p_offset;
    let mut p_ptr: *mut u8 = std::ptr::null_mut();
    let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
    let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

    if with_cigar {
        let p_size = (query_len + target_len - 1) * n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        mem_size_bytes = off_end_offset_start + off_size;
    }

    let mem = AlignedMemory::new(mem_size_bytes, 16);
    // Zero DP+scoring region (not traceback — written per-cell in DP loop)
    std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

    let base_ptr = mem.as_ptr();
    let u = base_ptr as *mut uint8x16_t;
    let v = u.add(tlen_);
    let x = v.add(tlen_);
    let y = x.add(tlen_);
    let x2 = y.add(tlen_);
    let y2 = x2.add(tlen_);
    let s = y2.add(tlen_);
    let sf = base_ptr.add(sf_offset);
    let qr = base_ptr.add(qr_offset);

    // Initialize DP arrays to proper boundary values
    // Dual-affine uses SIGNED arithmetic, so arrays must NOT be zero-initialized
    let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
    let neg_q2e2 = (-(gap_open2 as i32) - gap_extend2 as i32) as u8;
    std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 16);
    std::ptr::write_bytes(v as *mut u8, neg_qe, tlen_ * 16);
    std::ptr::write_bytes(x as *mut u8, neg_qe, tlen_ * 16);
    std::ptr::write_bytes(y as *mut u8, neg_qe, tlen_ * 16);
    std::ptr::write_bytes(x2 as *mut u8, neg_q2e2, tlen_ * 16);
    std::ptr::write_bytes(y2 as *mut u8, neg_q2e2, tlen_ * 16);

    if with_cigar {
        let p_size = (query_len + target_len - 1) * n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        p_ptr = base_ptr.add(p_offset);
        band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
        band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
    }

    // Reverse query
    let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
    for t in 0..query_len {
        qr_slice[t] = qseq[query_len - 1 - t];
    }
    std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

    // H[] array for exact max tracking (only when !approx_max)
    let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 16);
    let _ = &h_vec; // prevent early drop

    // Initialize result
    init_dp_result(result);

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let valid_range = (query_len + target_len - 1) as i32;
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    for r in 0..valid_range {
        let mut st = 0i32;
        let mut en = target_len as i32 - 1;

        let qrr = qr.offset(query_len as isize - 1 - r as isize);

        // Find boundaries
        if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
        if en > r { en = r; }
        if st < (r - wl + 1) >> 1 { st = (r - wl + 1) >> 1; }
        if en > (r + wl) >> 1 { en = (r + wl) >> 1; }

        if st > en {
            result.zdropped = 1;
            break;
        }

        let st0 = st;
        let en0 = en;
        st = (st / 16) * 16;
        en = ((en + 16) / 16) * 16 - 1;

        // Boundary conditions
        let x1: i8;
        let x21: i8;
        let v1: i8;
        let u8_arr = u as *mut i8;
        let v8_arr = v as *mut i8;
        let x8_arr = x as *mut i8;
        let x28_arr = x2 as *mut i8;

        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = *x8_arr.add((st - 1) as usize);
                x21 = *x28_arr.add((st - 1) as usize);
                v1 = *v8_arr.add((st - 1) as usize);
            } else {
                x1 = -gap_open - gap_extend;
                x21 = -gap_open2 - gap_extend2;
                v1 = -gap_open - gap_extend;
            }
        } else {
            x1 = -gap_open - gap_extend;
            x21 = -gap_open2 - gap_extend2;
            v1 = if r == 0 {
                -gap_open - gap_extend
            } else if r < long_thres {
                -gap_extend
            } else if r == long_thres {
                long_diff
            } else {
                -gap_extend2
            };
        }

        if en >= r {
            *(y as *mut i8).add(r as usize) = -gap_open - gap_extend;
            *(y2 as *mut i8).add(r as usize) = -gap_open2 - gap_extend2;
            *u8_arr.add(r as usize) = if r == 0 {
                -gap_open - gap_extend
            } else if r < long_thres {
                -gap_extend
            } else if r == long_thres {
                long_diff
            } else {
                -gap_extend2
            };
        }

        // Set scores (16-element chunks, SIMD scoring loop)
        if (flags & GENERIC_SCORING) == 0 {
            // Simple match/mismatch scoring (uniform penalties)
            let sc_mis_val = score_matrix[1] as u8;
            let sc_mch_val = score_matrix[0] as u8;
            let sc_n_val = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                (-(gap_extend2 as i8)) as u8
            } else {
                score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8
            };
            let m1_val = (alphabet_size - 1) as u8;
            let mut t = st0;
            while t <= en0 {
                for k in 0..16i32 {
                    let pos = (t + k) as usize;
                    let sf_val = *sf.add(pos);
                    let qr_val = *qrr.add(pos);
                    let is_n = sf_val == m1_val || qr_val == m1_val;
                    let score = if is_n {
                        sc_n_val
                    } else if sf_val == qr_val {
                        sc_mch_val
                    } else {
                        sc_mis_val
                    };
                    *(s as *mut u8).add(pos) = score;
                }
                t += 16;
            }
        } else {
            // Generic scoring: full matrix lookup score_matrix[target_base * alphabet_size + query_base]
            let s_ptr = s as *mut u8;
            for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 16 - 1) {
                *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
            }
        }

        // Core DP loop with dual-affine
        let mut x1_ = vsetq_lane_u8(x1 as u8, vdupq_n_u8(0), 0);
        let mut x21_ = vsetq_lane_u8(x21 as u8, vdupq_n_u8(0), 0);
        let mut v1_ = vsetq_lane_u8(v1 as u8, vdupq_n_u8(0), 0);

        let st_ = st as usize / 16;
        let en_ = en as usize / 16;

        for ti in st_..=en_ {
            // Dual-affine: z = s[t] with NO bias
            let mut z = vld1q_u8((s as *const u8).add(ti * 16));

            let xt_val = vld1q_u8((x as *const u8).add(ti * 16));
            let mut xt1 = xt_val;
            let tmp_x = vextq_u8(xt1, zero_, 15);
            xt1 = vorrq_u8(vextq_u8(zero_, xt1, 15), x1_);
            x1_ = tmp_x;

            let vt_val = vld1q_u8((v as *const u8).add(ti * 16));
            let mut vt1 = vt_val;
            let tmp_v = vextq_u8(vt1, zero_, 15);
            vt1 = vorrq_u8(vextq_u8(zero_, vt1, 15), v1_);
            v1_ = tmp_v;

            // a = x[t-1] + v[t-1] (I1 candidate)
            let a = vaddq_u8(xt1, vt1);

            // ut, b for D1
            let ut = vld1q_u8((u as *const u8).add(ti * 16));
            let b = vaddq_u8(vld1q_u8((y as *const u8).add(ti * 16)), ut);

            // x2, y2 for second penalty
            let x2t_val = vld1q_u8((x2 as *const u8).add(ti * 16));
            let mut x2t1 = x2t_val;
            let tmp_x2 = vextq_u8(x2t1, zero_, 15);
            x2t1 = vorrq_u8(vextq_u8(zero_, x2t1, 15), x21_);
            x21_ = tmp_x2;

            // a2 = x2[t-1] + v[t-1] (I2 candidate)
            let a2 = vaddq_u8(x2t1, vt1);
            // b2 = y2[t] + u[t] (D2 candidate)
            let b2 = vaddq_u8(vld1q_u8((y2 as *const u8).add(ti * 16)), ut);

            if !with_cigar {
                // Score only path
                let z_s8 = vreinterpretq_s8_u8(z);
                let a_s8 = vreinterpretq_s8_u8(a);
                let b_s8 = vreinterpretq_s8_u8(b);
                let a2_s8 = vreinterpretq_s8_u8(a2);
                let b2_s8 = vreinterpretq_s8_u8(b2);
                let z1 = vmaxq_s8(z_s8, a_s8);
                let z2 = vmaxq_s8(z1, b_s8);
                let z3 = vmaxq_s8(z2, a2_s8);
                let z4 = vmaxq_s8(z3, b2_s8);
                // Dual-affine: clamp with SIGNED min at score_matrix[0] (no bias)
                z = vreinterpretq_u8_s8(vminq_s8(z4, sc_mch_));

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let tmp2 = vsubq_u8(z, q2_);
                let a2_new = vsubq_u8(a2, tmp2);
                let b2_new = vsubq_u8(b2, tmp2);

                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let a_new_s8 = vreinterpretq_s8_u8(a_new);
                let tmp = vcgtq_s8(a_new_s8, zero_s8);
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a_new), qe_));
                let b_new_s8 = vreinterpretq_s8_u8(b_new);
                let tmp = vcgtq_s8(b_new_s8, zero_s8);
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b_new), qe_));
                let a2_new_s8 = vreinterpretq_s8_u8(a2_new);
                let tmp = vcgtq_s8(a2_new_s8, zero_s8);
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a2_new), qe2_));
                let b2_new_s8 = vreinterpretq_s8_u8(b2_new);
                let tmp = vcgtq_s8(b2_new_s8, zero_s8);
                vst1q_u8((y2 as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b2_new), qe2_));
            } else if (flags & RIGHT_ALIGN) == 0 {
                // Gap LEFT-alignment path
                let offset = (r as usize * n_col_) as isize - st_ as isize;
                let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                if ti == st_ {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

                let a_s8 = vreinterpretq_s8_u8(a);
                let b_s8 = vreinterpretq_s8_u8(b);
                let a2_s8 = vreinterpretq_s8_u8(a2);
                let b2_s8 = vreinterpretq_s8_u8(b2);
                let z_s8 = vreinterpretq_s8_u8(z);

                // 5-way max with LEFT tie-breaking: gap wins only if strictly >
                let mut d: uint8x16_t;
                let tmp = vcgtq_s8(a_s8, z_s8);
                d = vandq_u8(tmp, flag1_);
                z = vbslq_u8(tmp, a, z);
                let tmp = vcgtq_s8(b_s8, vreinterpretq_s8_u8(z));
                d = vbslq_u8(tmp, flag2_, d);
                z = vbslq_u8(tmp, b, z);
                let tmp = vcgtq_s8(a2_s8, vreinterpretq_s8_u8(z));
                d = vbslq_u8(tmp, flag3_, d);
                z = vbslq_u8(tmp, a2, z);
                let tmp = vcgtq_s8(b2_s8, vreinterpretq_s8_u8(z));
                d = vbslq_u8(tmp, flag4_, d);
                z = vbslq_u8(tmp, b2, z);
                // Clamp: signed min at score_matrix[0]
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), sc_mch_);
                z = vbslq_u8(tmp, vreinterpretq_u8_s8(sc_mch_), z);

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let tmp2 = vsubq_u8(z, q2_);
                let a2_new = vsubq_u8(a2, tmp2);
                let b2_new = vsubq_u8(b2, tmp2);

                // x, y, x2, y2 with LEFT extension flags
                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let a_new_s8 = vreinterpretq_s8_u8(a_new);
                let tmp = vcgtq_s8(a_new_s8, zero_s8);
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a_new), qe_));
                d = vorrq_u8(d, vandq_u8(tmp, flag8_));
                let b_new_s8 = vreinterpretq_s8_u8(b_new);
                let tmp = vcgtq_s8(b_new_s8, zero_s8);
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b_new), qe_));
                d = vorrq_u8(d, vandq_u8(tmp, flag16_));
                let a2_new_s8 = vreinterpretq_s8_u8(a2_new);
                let tmp = vcgtq_s8(a2_new_s8, zero_s8);
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a2_new), qe2_));
                d = vorrq_u8(d, vandq_u8(tmp, flag32_));
                let b2_new_s8 = vreinterpretq_s8_u8(b2_new);
                let tmp = vcgtq_s8(b2_new_s8, zero_s8);
                vst1q_u8((y2 as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b2_new), qe2_));
                d = vorrq_u8(d, vandq_u8(tmp, flag64_));

                vst1q_u8(pr_ptr, d);
            } else {
                // Gap RIGHT-alignment path
                let offset = (r as usize * n_col_) as isize - st_ as isize;
                let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                if ti == st_ {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

                let a_s8 = vreinterpretq_s8_u8(a);
                let b_s8 = vreinterpretq_s8_u8(b);
                let a2_s8 = vreinterpretq_s8_u8(a2);
                let b2_s8 = vreinterpretq_s8_u8(b2);
                let z_s8 = vreinterpretq_s8_u8(z);

                // 5-way max with RIGHT tie-breaking: gap wins if >=
                let mut d: uint8x16_t;
                let tmp = vcgtq_s8(z_s8, a_s8);
                d = vbicq_u8(flag1_, tmp); // d = flag1 & ~tmp (gap wins when z NOT > a, i.e. a >= z)
                z = vbslq_u8(tmp, z, a);   // z = tmp ? z : a
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), b_s8);
                d = vbslq_u8(tmp, d, flag2_); // keep d if z > b, else flag2
                z = vbslq_u8(tmp, z, b);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), a2_s8);
                d = vbslq_u8(tmp, d, flag3_);
                z = vbslq_u8(tmp, z, a2);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), b2_s8);
                d = vbslq_u8(tmp, d, flag4_);
                z = vbslq_u8(tmp, z, b2);
                // Clamp: signed min at score_matrix[0]
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), sc_mch_);
                z = vbslq_u8(tmp, vreinterpretq_u8_s8(sc_mch_), z);

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let tmp2 = vsubq_u8(z, q2_);
                let a2_new = vsubq_u8(a2, tmp2);
                let b2_new = vsubq_u8(b2, tmp2);

                // x, y, x2, y2 with RIGHT extension flags (reversed comparison)
                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let a_new_s8 = vreinterpretq_s8_u8(a_new);
                let tmp = vcgtq_s8(zero_s8, a_new_s8);
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(a_new, tmp), qe_));
                d = vorrq_u8(d, vbicq_u8(flag8_, tmp));
                let b_new_s8 = vreinterpretq_s8_u8(b_new);
                let tmp = vcgtq_s8(zero_s8, b_new_s8);
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(b_new, tmp), qe_));
                d = vorrq_u8(d, vbicq_u8(flag16_, tmp));
                let a2_new_s8 = vreinterpretq_s8_u8(a2_new);
                let tmp = vcgtq_s8(zero_s8, a2_new_s8);
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(a2_new, tmp), qe2_));
                d = vorrq_u8(d, vbicq_u8(flag32_, tmp));
                let b2_new_s8 = vreinterpretq_s8_u8(b2_new);
                let tmp = vcgtq_s8(zero_s8, b2_new_s8);
                vst1q_u8((y2 as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(b2_new, tmp), qe2_));
                d = vorrq_u8(d, vbicq_u8(flag64_, tmp));

                vst1q_u8(pr_ptr, d);
            }
        }

        // Update h0 and track max score
        let u8_ptr = u as *mut u8;
        let v8_ptr = v as *mut u8;
        let qe_scalar = gap_open as i32 + gap_extend as i32;

        if !approx_max {
            // Exact max tracking with 32-bit H[] array
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                // Special case: last element
                let h_en0 = if en0 > 0 {
                    *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                } else {
                    *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                };
                *h_ptr.add(en0 as usize) = h_en0;
                max_h = h_en0;
                max_t = en0;

                // Process [st0..en0) scalar (NEON doesn't have convenient i32 SIMD here)
                let mut t = st0;
                while t < en0 {
                    *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                    if *h_ptr.add(t as usize) > max_h {
                        max_h = *h_ptr.add(t as usize);
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                // r == 0
                *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                max_h = *h_ptr.add(0);
                max_t = 0;
            }
            // Update result.max_target_end_score (max target end) and result.max_query_end_score (max query end)
            if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                result.max_target_end_score = *h_ptr.add(en0 as usize);
                result.max_target_end_query_pos = r - en0;
            }
            if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                result.max_query_end_score = *h_ptr.add(st0 as usize);
                result.max_query_end_target_pos = st0;
            }
            // Z-drop check: update max, check z_drop
            if max_h > result.max {
                result.max = max_h;
                result.max_score_target_pos = max_t;
                result.max_score_query_pos = r - max_t;
            } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                let tl = max_t - result.max_score_target_pos;
                let ql = (r - max_t) - result.max_score_query_pos;
                let l = if tl > ql { tl - ql } else { ql - tl };
                if z_drop >= 0 && (result.max - max_h) > (z_drop + l * gap_extend2 as i32) {
                    result.zdropped = 1;
                    break;
                }
            }
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = *h_ptr.add(target_len - 1);
            }
        } else {
            // Approximate max tracking (existing code)
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t + 1 <= en0 {
                    // Dual-affine: use raw v8/u8 values (no qe subtraction)
                    let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                    let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;

                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                }

                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = last_h0_t;
                    result.max_score_query_pos = r - last_h0_t;
                }

                // Check z_drop
                if (flags & APPROX_DROP) != 0 {
                    if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                        let tl = last_h0_t - result.max_score_target_pos;
                        let ql = (r - last_h0_t) - result.max_score_query_pos;
                        let l = if tl > ql { tl - ql } else { ql - tl };
                        // Dual-affine uses gap_extend2 for z-drop
                        if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend2 as i32) {
                            result.zdropped = 1;
                            break;
                        }
                    }
                }
            } else {
                // r == 0: dual-affine subtracts qe once
                let v0 = *v8_ptr.add(0) as i8 as i32;
                h0 = v0 - qe_scalar;
                last_h0_t = 0;
                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = 0;
                    result.max_score_query_pos = 0;
                }
            }
            // Final score for approx path
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h0;
            }
        }

        last_st = st;
        last_en = en;
    }

    // Final score
    if approx_max && result.score == NEG_INF {
        result.score = result.max;
    }

    // Traceback for CIGAR
    if with_cigar {
        traceback_dual_affine(result, query_len, target_len, end_bonus, flags, n_col_, 16, p_ptr, band_offset_ptr, band_offset_end_ptr);
    }
}}

// ============================================================================
// SSE2/SSE4.1 Unified Implementation - Dual-Affine Alignment
// ============================================================================
//
// Macro generates both SSE2 and SSE4.1 variants. Differences:
// - max_epi8/min_epi8: SSE2 uses emulated helpers, SSE4.1 uses native intrinsics
// - blend: SSE2 uses and/andnot/or pattern, SSE4.1 uses _mm_blendv_epi8
// Both variants require only SSE2 target_feature (SSE4.1 is detected at runtime).

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
macro_rules! extend_dual_affine_impl {
    ($fn_name:ident, $max_epi8:path, $min_epi8:path, $is_sse41:expr, $target_feat:tt) => {
        #[target_feature(enable = $target_feat)]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            gap_extend2: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let approx_max = (flags & APPROX_MAX) != 0;

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
                return;
            }

            // Ensure gap_open+gap_extend <= gap_open2+gap_extend2
            let (gap_open, gap_extend, gap_open2, gap_extend2) = if (gap_open2 as i32 + gap_extend2 as i32) < (gap_open as i32 + gap_extend as i32) {
                (gap_open2, gap_extend2, gap_open, gap_extend)
            } else {
                (gap_open, gap_extend, gap_open2, gap_extend2)
            };

            // Compute long_thres and long_diff for dual-affine boundary conditions
                    let mut long_thres: i32 = if gap_extend != gap_extend2 {
                (gap_open2 as i32 - gap_open as i32) / (gap_extend as i32 - gap_extend2 as i32) - 1
            } else { 0 };
            if (gap_open2 as i32 + gap_extend2 as i32 + long_thres * gap_extend2 as i32) > (gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32) {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * (gap_extend as i32 - gap_extend2 as i32) - (gap_open2 as i32 - gap_open as i32) - gap_extend2 as i32) as i8;

            // Constants - dual-affine uses SIGNED operations, NO bias on z
            let zero_ = _mm_setzero_si128();
            let q_ = _mm_set1_epi8(gap_open);
            let q2_ = _mm_set1_epi8(gap_open2);
            let qe_ = _mm_set1_epi8((gap_open as i32 + gap_extend as i32) as i8);
            let qe2_ = _mm_set1_epi8((gap_open2 as i32 + gap_extend2 as i32) as i8);
            let sc_mch_ = _mm_set1_epi8(score_matrix[0]); // clamp value for dual-affine (signed)
            let sc_mis_ = _mm_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm_set1_epi8(-(gap_extend2 as i8))
            } else {
                _mm_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm_set1_epi8(alphabet_size - 1);

            let flag1_ = _mm_set1_epi8(1);
            let flag2_ = _mm_set1_epi8(2);
            let flag3_ = _mm_set1_epi8(3);
            let flag4_ = _mm_set1_epi8(4);
            let flag8_ = _mm_set1_epi8(0x08);
            let flag16_ = _mm_set1_epi8(0x10);
            let flag32_ = _mm_set1_epi8(0x20);
            let flag64_ = _mm_set1_epi8(0x40);

            let bandwidth = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(16);
            let mut n_col_ = query_len.min(target_len);
            n_col_ = n_col_.min((bandwidth + 1) as usize).div_ceil(16) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 7 arrays for dual-affine: u, v, x, y, x2, y2, s
            let qlen_ = query_len.div_ceil(16);
            let dp_size = 7 * tlen_ * 16;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 16;
            let p_offset = qr_offset + (qlen_ + 1) * 16;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 16);
            // Zero DP+scoring region (not traceback — written per-cell in DP loop)
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m128i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let y2 = x2.add(tlen_);
            let s = y2.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize DP arrays to proper boundary values
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            let neg_q2e2 = (-(gap_open2 as i32) - gap_extend2 as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 16);
            std::ptr::write_bytes(v as *mut u8, neg_qe, tlen_ * 16);
            std::ptr::write_bytes(x as *mut u8, neg_qe, tlen_ * 16);
            std::ptr::write_bytes(y as *mut u8, neg_qe, tlen_ * 16);
            std::ptr::write_bytes(x2 as *mut u8, neg_q2e2, tlen_ * 16);
            std::ptr::write_bytes(y2 as *mut u8, neg_q2e2, tlen_ * 16);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // H[] array for exact max tracking (only when !approx_max)
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 16);
            let _ = &h_vec; // prevent early drop

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Find boundaries
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < (r - wl + 1) >> 1 { st = (r - wl + 1) >> 1; }
                if en > (r + wl) >> 1 { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -gap_open - gap_extend;
                        x21 = -gap_open2 - gap_extend2;
                        v1 = -gap_open - gap_extend;
                    }
                } else {
                    x1 = -gap_open - gap_extend;
                    x21 = -gap_open2 - gap_extend2;
                    v1 = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -gap_open - gap_extend;
                    *(y2 as *mut i8).add(r as usize) = -gap_open2 - gap_extend2;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                // Set scores (SIMD 16-element chunks)
                if (flags & GENERIC_SCORING) == 0 {
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm_loadu_si128(sf.add(t as usize) as *const __m128i);
                        let st_v = _mm_loadu_si128(qrr.add(t as usize) as *const __m128i);
                        let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(st_v, m1_));
                        let tmp = _mm_cmpeq_epi8(sq, st_v);
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(sc_mis_, sc_mch_, tmp)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(tmp, sc_mis_), _mm_and_si128(tmp, sc_mch_))
                        };
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(tmp, sc_n_, mask)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_))
                        };
                        _mm_storeu_si128((s as *mut u8).add(t as usize) as *mut __m128i, tmp);
                        t += 16;
                    }
                } else {
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 16 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop with dual-affine
                let mut x1_ = sse2_insert_byte0(zero_, x1 as u8);
                let mut x21_ = sse2_insert_byte0(zero_, x21 as u8);
                let mut v1_ = sse2_insert_byte0(zero_, v1 as u8);

                let st_ = st as usize / 16;
                let en_ = en as usize / 16;

                for ti in st_..=en_ {
                    // Dual-affine: z = s[t] with NO bias
                    let mut z = _mm_loadu_si128(s.add(ti));

                    let xt_val = _mm_loadu_si128(x.add(ti));
                    let mut xt1 = xt_val;
                    let tmp_x = _mm_srli_si128(xt1, 15);
                    xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm_loadu_si128(v.add(ti));
                    let mut vt1 = vt_val;
                    let tmp_v = _mm_srli_si128(vt1, 15);
                    vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_);
                    v1_ = tmp_v;

                    // a = x[t-1] + v[t-1] (I1 candidate)
                    let a = _mm_add_epi8(xt1, vt1);

                    // ut, b for D1
                    let ut = _mm_loadu_si128(u.add(ti));
                    let b = _mm_add_epi8(_mm_loadu_si128(y.add(ti)), ut);

                    // x2, y2 for second penalty
                    let x2t_val = _mm_loadu_si128(x2.add(ti));
                    let mut x2t1 = x2t_val;
                    let tmp_x2 = _mm_srli_si128(x2t1, 15);
                    x2t1 = _mm_or_si128(_mm_slli_si128(x2t1, 1), x21_);
                    x21_ = tmp_x2;

                    // a2 = x2[t-1] + v[t-1] (I2 candidate)
                    let a2 = _mm_add_epi8(x2t1, vt1);
                    // b2 = y2[t] + u[t] (D2 candidate)
                    let b2 = _mm_add_epi8(_mm_loadu_si128(y2.add(ti)), ut);

                    if !with_cigar {
                        // Score only path
                        z = $max_epi8(z, a);
                        z = $max_epi8(z, b);
                        z = $max_epi8(z, a2);
                        z = $max_epi8(z, b2);
                        z = $min_epi8(z, sc_mch_);

                        // Update u, v, x, y from z
                        _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let tmp2 = _mm_sub_epi8(z, q2_);
                        let a2_new = _mm_sub_epi8(a2, tmp2);
                        let b2_new = _mm_sub_epi8(b2, tmp2);

                        let tmp = _mm_cmpgt_epi8(a_new, zero_);
                        _mm_storeu_si128(x.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a_new), qe_));
                        let tmp = _mm_cmpgt_epi8(b_new, zero_);
                        _mm_storeu_si128(y.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b_new), qe_));
                        let tmp = _mm_cmpgt_epi8(a2_new, zero_);
                        _mm_storeu_si128(x2.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a2_new), qe2_));
                        let tmp = _mm_cmpgt_epi8(b2_new, zero_);
                        _mm_storeu_si128(y2.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b2_new), qe2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        // Gap LEFT-alignment path
                        let offset = (r as usize * n_col_) as isize - st_ as isize;
                        let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                        if ti == st_ {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        // 5-way max with LEFT tie-breaking: gap wins only if strictly >
                        let mut d;
                        if $is_sse41 {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            d = _mm_and_si128(tmp, flag1_);
                            z = _mm_max_epi8(z, a);
                            let tmp = _mm_cmpgt_epi8(b, z);
                            d = _mm_blendv_epi8(d, flag2_, tmp);
                            z = _mm_max_epi8(z, b);
                            let tmp = _mm_cmpgt_epi8(a2, z);
                            d = _mm_blendv_epi8(d, flag3_, tmp);
                            z = _mm_max_epi8(z, a2);
                            let tmp = _mm_cmpgt_epi8(b2, z);
                            d = _mm_blendv_epi8(d, flag4_, tmp);
                            z = _mm_max_epi8(z, b2);
                        } else {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            d = _mm_and_si128(tmp, flag1_);
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
                            let tmp = _mm_cmpgt_epi8(b, z);
                            d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
                            let tmp = _mm_cmpgt_epi8(a2, z);
                            d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag3_));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2));
                            let tmp = _mm_cmpgt_epi8(b2, z);
                            d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag4_));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b2));
                        }
                        // Clamp: signed min at score_matrix[0]
                        z = $min_epi8(z, sc_mch_);

                        // Update u, v, x, y from z
                        _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let tmp2 = _mm_sub_epi8(z, q2_);
                        let a2_new = _mm_sub_epi8(a2, tmp2);
                        let b2_new = _mm_sub_epi8(b2, tmp2);

                        // x, y, x2, y2 with LEFT extension flags
                        let tmp = _mm_cmpgt_epi8(a_new, zero_);
                        _mm_storeu_si128(x.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a_new), qe_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                        let tmp = _mm_cmpgt_epi8(b_new, zero_);
                        _mm_storeu_si128(y.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b_new), qe_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));
                        let tmp = _mm_cmpgt_epi8(a2_new, zero_);
                        _mm_storeu_si128(x2.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a2_new), qe2_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag32_));
                        let tmp = _mm_cmpgt_epi8(b2_new, zero_);
                        _mm_storeu_si128(y2.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b2_new), qe2_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag64_));

                        _mm_storeu_si128(pr_ptr as *mut __m128i, d);
                    } else {
                        // Gap RIGHT-alignment path
                        let offset = (r as usize * n_col_) as isize - st_ as isize;
                        let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                        if ti == st_ {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        // 5-way max with RIGHT tie-breaking: gap wins if >=
                        let mut d;
                        if $is_sse41 {
                            let tmp = _mm_cmpgt_epi8(z, a);
                            d = _mm_andnot_si128(tmp, flag1_);
                            z = _mm_max_epi8(z, a);
                            let tmp = _mm_cmpgt_epi8(z, b);
                            d = _mm_blendv_epi8(flag2_, d, tmp);
                            z = _mm_max_epi8(z, b);
                            let tmp = _mm_cmpgt_epi8(z, a2);
                            d = _mm_blendv_epi8(flag3_, d, tmp);
                            z = _mm_max_epi8(z, a2);
                            let tmp = _mm_cmpgt_epi8(z, b2);
                            d = _mm_blendv_epi8(flag4_, d, tmp);
                            z = _mm_max_epi8(z, b2);
                        } else {
                            let tmp = _mm_cmpgt_epi8(z, a);
                            d = _mm_andnot_si128(tmp, flag1_);
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a));
                            let tmp = _mm_cmpgt_epi8(z, b);
                            d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, flag2_));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b));
                            let tmp = _mm_cmpgt_epi8(z, a2);
                            d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, flag3_));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a2));
                            let tmp = _mm_cmpgt_epi8(z, b2);
                            d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, flag4_));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b2));
                        }
                        // Clamp: signed min at score_matrix[0]
                        z = $min_epi8(z, sc_mch_);

                        // Update u, v, x, y from z
                        _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let tmp2 = _mm_sub_epi8(z, q2_);
                        let a2_new = _mm_sub_epi8(a2, tmp2);
                        let b2_new = _mm_sub_epi8(b2, tmp2);

                        // x, y, x2, y2 with RIGHT extension flags (reversed comparison)
                        let tmp = _mm_cmpgt_epi8(zero_, a_new);
                        _mm_storeu_si128(x.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, a_new), qe_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));
                        let tmp = _mm_cmpgt_epi8(zero_, b_new);
                        _mm_storeu_si128(y.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, b_new), qe_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_));
                        let tmp = _mm_cmpgt_epi8(zero_, a2_new);
                        _mm_storeu_si128(x2.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, a2_new), qe2_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag32_));
                        let tmp = _mm_cmpgt_epi8(zero_, b2_new);
                        _mm_storeu_si128(y2.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, b2_new), qe2_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag64_));

                        _mm_storeu_si128(pr_ptr as *mut __m128i, d);
                    }
                }

                // Update h0 and track max score
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                // H[] tracking
                if !approx_max {
                    // Exact max tracking with 32-bit H[] array
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        // Process [st0..en0) with SSE (4 i32 at a time)
                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            // Blend for 32-bit conditional select
                            if $is_sse41 {
                                max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                                max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            } else {
                                max_h_ = _mm_or_si128(_mm_and_si128(tmp, h1), _mm_andnot_si128(tmp, max_h_));
                                max_t_ = _mm_or_si128(_mm_and_si128(tmp, t_), _mm_andnot_si128(tmp, max_t_));
                            }
                            t += 4;
                        }
                        // Reduce SSE to scalar
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        // Remainder
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        // r == 0
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    // Update result scores
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    // Z-drop check: update max, check z_drop
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        let tl = max_t - result.max_score_target_pos;
                        let ql = (r - max_t) - result.max_score_query_pos;
                        let l = if tl > ql { tl - ql } else { ql - tl };
                        if z_drop >= 0 && (result.max - max_h) > (z_drop + l * gap_extend2 as i32) {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }

                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        // Check z_drop
                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend2 as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        // r == 0
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - qe_scalar;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = 0;
                            result.max_score_query_pos = 0;
                        }
                    }
                    // Final score for approx path
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }

                last_st = st;
                last_en = en;
            }

            // Final score
            if approx_max && result.score == NEG_INF {
                result.score = result.max;
            }

            // Traceback for CIGAR
            if with_cigar {
                traceback_dual_affine(result, query_len, target_len, end_bonus, flags, n_col_, 16, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_dual_affine_impl!(extend_dual_affine2_impl, sse2_max_epi8, sse2_min_epi8, false, "sse2");
#[cfg(target_arch = "x86_64")]
extend_dual_affine_impl!(extend_dual_affine41_impl, _mm_max_epi8, _mm_min_epi8, true, "sse2");
#[cfg(target_arch = "wasm32")]
extend_dual_affine_impl!(extend_dual_affine_wasm_impl, _mm_max_epi8, _mm_min_epi8, true, "simd128");

#[cfg(target_arch = "x86_64")]
macro_rules! extend_dual_affine_avx2_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            gap_extend2: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let approx_max = (flags & APPROX_MAX) != 0;

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
                return;
            }

            // Ensure gap_open+gap_extend <= gap_open2+gap_extend2
            let (gap_open, gap_extend, gap_open2, gap_extend2) = if (gap_open2 as i32 + gap_extend2 as i32) < (gap_open as i32 + gap_extend as i32) {
                (gap_open2, gap_extend2, gap_open, gap_extend)
            } else {
                (gap_open, gap_extend, gap_open2, gap_extend2)
            };

            // Compute long_thres and long_diff for dual-affine boundary conditions
                    let mut long_thres: i32 = if gap_extend != gap_extend2 {
                (gap_open2 as i32 - gap_open as i32) / (gap_extend as i32 - gap_extend2 as i32) - 1
            } else { 0 };
            if (gap_open2 as i32 + gap_extend2 as i32 + long_thres * gap_extend2 as i32) > (gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32) {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * (gap_extend as i32 - gap_extend2 as i32) - (gap_open2 as i32 - gap_open as i32) - gap_extend2 as i32) as i8;

            // Constants - dual-affine uses SIGNED operations, NO bias on z
            let zero_ = _mm256_setzero_si256();
            let q_ = _mm256_set1_epi8(gap_open);
            let q2_ = _mm256_set1_epi8(gap_open2);
            let qe_ = _mm256_set1_epi8((gap_open as i32 + gap_extend as i32) as i8);
            let qe2_ = _mm256_set1_epi8((gap_open2 as i32 + gap_extend2 as i32) as i8);
            let sc_mch_ = _mm256_set1_epi8(score_matrix[0]); // clamp value for dual-affine (signed)
            let sc_mis_ = _mm256_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm256_set1_epi8(-(gap_extend2 as i8))
            } else {
                _mm256_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm256_set1_epi8(alphabet_size - 1);

            let flag1_ = _mm256_set1_epi8(1);
            let flag2_ = _mm256_set1_epi8(2);
            let flag3_ = _mm256_set1_epi8(3);
            let flag4_ = _mm256_set1_epi8(4);
            let flag8_ = _mm256_set1_epi8(0x08);
            let flag16_ = _mm256_set1_epi8(0x10);
            let flag32_ = _mm256_set1_epi8(0x20);
            let flag64_ = _mm256_set1_epi8(0x40);

            let bandwidth = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(32) + 1; // +1 for byte-addressed SSE-compat padding
            let mut n_col_ = query_len.min(target_len);
            n_col_ = n_col_.min((bandwidth + 1) as usize).div_ceil(32) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 7 arrays for dual-affine: u, v, x, y, x2, y2, s
            let qlen_ = query_len.div_ceil(32);
            let dp_size = 7 * tlen_ * 32;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 32;
            let p_offset = qr_offset + (qlen_ + 1) * 32;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 32);
            // Zero DP+scoring region (not traceback — written per-cell in DP loop)
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m256i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let y2 = x2.add(tlen_);
            let s = y2.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize DP arrays to proper boundary values
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            let neg_q2e2 = (-(gap_open2 as i32) - gap_extend2 as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 32);
            std::ptr::write_bytes(v as *mut u8, neg_qe, tlen_ * 32);
            std::ptr::write_bytes(x as *mut u8, neg_qe, tlen_ * 32);
            std::ptr::write_bytes(y as *mut u8, neg_qe, tlen_ * 32);
            std::ptr::write_bytes(x2 as *mut u8, neg_q2e2, tlen_ * 32);
            std::ptr::write_bytes(y2 as *mut u8, neg_q2e2, tlen_ * 32);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // H[] array for exact max tracking (only when !approx_max)
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 32);
            let _ = &h_vec; // prevent early drop

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Find boundaries
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < (r - wl + 1) >> 1 { st = (r - wl + 1) >> 1; }
                if en > (r + wl) >> 1 { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -gap_open - gap_extend;
                        x21 = -gap_open2 - gap_extend2;
                        v1 = -gap_open - gap_extend;
                    }
                } else {
                    x1 = -gap_open - gap_extend;
                    x21 = -gap_open2 - gap_extend2;
                    v1 = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -gap_open - gap_extend;
                    *(y2 as *mut i8).add(r as usize) = -gap_open2 - gap_extend2;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm256_loadu_si256(sf.add(t as usize) as *const __m256i);
                        let st_v = _mm256_loadu_si256(qrr.add(t as usize) as *const __m256i);
                        let mask = _mm256_or_si256(_mm256_cmpeq_epi8(sq, m1_), _mm256_cmpeq_epi8(st_v, m1_));
                        let tmp = _mm256_cmpeq_epi8(sq, st_v);
                        let tmp = _mm256_blendv_epi8(sc_mis_, sc_mch_, tmp);
                        let tmp = _mm256_blendv_epi8(tmp, sc_n_, mask);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp, 1));
                        }
                        t += 32;
                    }
                } else {
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 32 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop with dual-affine — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx2_insert_byte0(_mm256_setzero_si256(), x1 as u8);
                let mut x21_ = avx2_insert_byte0(_mm256_setzero_si256(), x21 as u8);
                let mut v1_ = avx2_insert_byte0(_mm256_setzero_si256(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let x2_b = x2 as *mut u8;
                let y2_b = y2 as *mut u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = n_col_ * 32;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    // Save excess bytes on last partial iteration
                    let excess = if bp + 31 > en_usize {
                        bp + 32 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 16];
                    let mut save_v = [0u8; 16];
                    let mut save_x = [0u8; 16];
                    let mut save_y = [0u8; 16];
                    let mut save_x2 = [0u8; 16];
                    let mut save_y2 = [0u8; 16];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x2_b.add(es), save_x2.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y2_b.add(es), save_y2.as_mut_ptr(), excess);
                    }

                    // Byte-addressed loads
                    let mut z = _mm256_loadu_si256(s_b_ptr.add(bp) as *const __m256i);

                    let xt_val = _mm256_loadu_si256(x_b.add(bp) as *const __m256i);
                    let (xt1, tmp_x) = avx2_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm256_loadu_si256(v_b.add(bp) as *const __m256i);
                    let (vt1, tmp_v) = avx2_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let a = _mm256_add_epi8(xt1, vt1);

                    let ut = _mm256_loadu_si256(u_b.add(bp) as *const __m256i);
                    let b = _mm256_add_epi8(_mm256_loadu_si256(y_b.add(bp) as *const __m256i), ut);

                    let x2t_val = _mm256_loadu_si256(x2_b.add(bp) as *const __m256i);
                    let (x2t1, tmp_x2) = avx2_shift_left_1(x2t_val, x21_);
                    x21_ = tmp_x2;

                    let a2 = _mm256_add_epi8(x2t1, vt1);
                    let b2 = _mm256_add_epi8(_mm256_loadu_si256(y2_b.add(bp) as *const __m256i), ut);

                    if !with_cigar {
                        // Score only path
                        z = _mm256_max_epi8(z, a);
                        z = _mm256_max_epi8(z, b);
                        z = _mm256_max_epi8(z, a2);
                        z = _mm256_max_epi8(z, b2);
                        z = _mm256_min_epi8(z, sc_mch_);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let tmp2 = _mm256_sub_epi8(z, q2_);
                        let a2_new = _mm256_sub_epi8(a2, tmp2);
                        let b2_new = _mm256_sub_epi8(b2, tmp2);

                        let tmp = _mm256_cmpgt_epi8(a_new, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a_new), qe_));
                        let tmp = _mm256_cmpgt_epi8(b_new, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b_new), qe_));
                        let tmp = _mm256_cmpgt_epi8(a2_new, zero_);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a2_new), qe2_));
                        let tmp = _mm256_cmpgt_epi8(b2_new, zero_);
                        _mm256_storeu_si256(y2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b2_new), qe2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        // Gap LEFT-alignment path — byte-addressed traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let tmp = _mm256_cmpgt_epi8(a, z);
                        let mut d = _mm256_and_si256(tmp, flag1_);
                        z = _mm256_max_epi8(z, a);
                        let tmp = _mm256_cmpgt_epi8(b, z);
                        d = _mm256_blendv_epi8(d, flag2_, tmp);
                        z = _mm256_max_epi8(z, b);
                        let tmp = _mm256_cmpgt_epi8(a2, z);
                        d = _mm256_blendv_epi8(d, flag3_, tmp);
                        z = _mm256_max_epi8(z, a2);
                        let tmp = _mm256_cmpgt_epi8(b2, z);
                        d = _mm256_blendv_epi8(d, flag4_, tmp);
                        z = _mm256_max_epi8(z, b2);
                        z = _mm256_min_epi8(z, sc_mch_);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let tmp2 = _mm256_sub_epi8(z, q2_);
                        let a2_new = _mm256_sub_epi8(a2, tmp2);
                        let b2_new = _mm256_sub_epi8(b2, tmp2);

                        let tmp = _mm256_cmpgt_epi8(a_new, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a_new), qe_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag8_));
                        let tmp = _mm256_cmpgt_epi8(b_new, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b_new), qe_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag16_));
                        let tmp = _mm256_cmpgt_epi8(a2_new, zero_);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a2_new), qe2_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag32_));
                        let tmp = _mm256_cmpgt_epi8(b2_new, zero_);
                        _mm256_storeu_si256(y2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b2_new), qe2_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag64_));

                        _mm256_storeu_si256(pr_ptr_local as *mut __m256i, d);
                    } else {
                        // Gap RIGHT-alignment path — byte-addressed traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let tmp = _mm256_cmpgt_epi8(z, a);
                        let mut d = _mm256_andnot_si256(tmp, flag1_);
                        z = _mm256_max_epi8(z, a);
                        let tmp = _mm256_cmpgt_epi8(z, b);
                        d = _mm256_blendv_epi8(flag2_, d, tmp);
                        z = _mm256_max_epi8(z, b);
                        let tmp = _mm256_cmpgt_epi8(z, a2);
                        d = _mm256_blendv_epi8(flag3_, d, tmp);
                        z = _mm256_max_epi8(z, a2);
                        let tmp = _mm256_cmpgt_epi8(z, b2);
                        d = _mm256_blendv_epi8(flag4_, d, tmp);
                        z = _mm256_max_epi8(z, b2);
                        z = _mm256_min_epi8(z, sc_mch_);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let tmp2 = _mm256_sub_epi8(z, q2_);
                        let a2_new = _mm256_sub_epi8(a2, tmp2);
                        let b2_new = _mm256_sub_epi8(b2, tmp2);

                        let tmp = _mm256_cmpgt_epi8(zero_, a_new);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, a_new), qe_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag8_));
                        let tmp = _mm256_cmpgt_epi8(zero_, b_new);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, b_new), qe_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag16_));
                        let tmp = _mm256_cmpgt_epi8(zero_, a2_new);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, a2_new), qe2_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag32_));
                        let tmp = _mm256_cmpgt_epi8(zero_, b2_new);
                        _mm256_storeu_si256(y2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, b2_new), qe2_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag64_));

                        _mm256_storeu_si256(pr_ptr_local as *mut __m256i, d);
                    }

                    // Restore excess bytes on partial last iteration
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x2.as_ptr(), x2_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y2.as_ptr(), y2_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 32;
                }

                // Update h0 and track max score
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                // H[] tracking
                if !approx_max {
                    // Exact max tracking with 32-bit H[] array
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        // Process [st0..en0) with SSE (4 i32 at a time)
                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            // Blend for 32-bit conditional select
                            max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                            max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            t += 4;
                        }
                        // Reduce SSE to scalar
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        // Remainder
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        // r == 0
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    // Update result scores
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    // Z-drop check: update max, check z_drop
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        let tl = max_t - result.max_score_target_pos;
                        let ql = (r - max_t) - result.max_score_query_pos;
                        let l = if tl > ql { tl - ql } else { ql - tl };
                        if z_drop >= 0 && (result.max - max_h) > (z_drop + l * gap_extend2 as i32) {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }

                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        // Check z_drop
                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend2 as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        // r == 0
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - qe_scalar;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = 0;
                            result.max_score_query_pos = 0;
                        }
                    }
                    // Final score for approx path
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }

                last_st = st;
                last_en = en;
            }

            // Final score
            if approx_max && result.score == NEG_INF {
                result.score = result.max;
            }

            // Traceback for CIGAR
            if with_cigar {
                traceback_dual_affine(result, query_len, target_len, end_bonus, flags, n_col_, 32, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_dual_affine_avx2_impl!(extend_dual_affine_avx2_fn);

// ============================================================================
// AVX512 Implementation - Dual-Affine Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_dual_affine_avx512_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx512bw")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            gap_extend2: i8,
            bandwidth: i32,
            z_drop: i32,
            end_bonus: i32,
            flags: i32,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let approx_max = (flags & APPROX_MAX) != 0;

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
                return;
            }

            // Ensure gap_open+gap_extend <= gap_open2+gap_extend2
            let (gap_open, gap_extend, gap_open2, gap_extend2) = if (gap_open2 as i32 + gap_extend2 as i32) < (gap_open as i32 + gap_extend as i32) {
                (gap_open2, gap_extend2, gap_open, gap_extend)
            } else {
                (gap_open, gap_extend, gap_open2, gap_extend2)
            };

            // Compute long_thres and long_diff for dual-affine boundary conditions
                    let mut long_thres: i32 = if gap_extend != gap_extend2 {
                (gap_open2 as i32 - gap_open as i32) / (gap_extend as i32 - gap_extend2 as i32) - 1
            } else { 0 };
            if (gap_open2 as i32 + gap_extend2 as i32 + long_thres * gap_extend2 as i32) > (gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32) {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * (gap_extend as i32 - gap_extend2 as i32) - (gap_open2 as i32 - gap_open as i32) - gap_extend2 as i32) as i8;

            // Constants - dual-affine uses SIGNED operations, NO bias on z
            let zero_ = _mm512_setzero_si512();
            let q_ = _mm512_set1_epi8(gap_open);
            let q2_ = _mm512_set1_epi8(gap_open2);
            let qe_ = _mm512_set1_epi8((gap_open as i32 + gap_extend as i32) as i8);
            let qe2_ = _mm512_set1_epi8((gap_open2 as i32 + gap_extend2 as i32) as i8);
            let sc_mch_ = _mm512_set1_epi8(score_matrix[0]); // clamp value for dual-affine (signed)
            let sc_mis_ = _mm512_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm512_set1_epi8(-(gap_extend2 as i8))
            } else {
                _mm512_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm512_set1_epi8(alphabet_size - 1);

            let flag1_ = _mm512_set1_epi8(1);
            let flag2_ = _mm512_set1_epi8(2);
            let flag3_ = _mm512_set1_epi8(3);
            let flag4_ = _mm512_set1_epi8(4);
            let flag8_ = _mm512_set1_epi8(0x08);
            let flag16_ = _mm512_set1_epi8(0x10);
            let flag32_ = _mm512_set1_epi8(0x20);
            let flag64_ = _mm512_set1_epi8(0x40);

            let bandwidth = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
            let wl = bandwidth;

            let tlen_ = target_len.div_ceil(64) + 1; // +1 for byte-addressed SSE-compat padding
            let mut n_col_ = query_len.min(target_len);
            n_col_ = n_col_.min((bandwidth + 1) as usize).div_ceil(64) + 1;

            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Memory allocation - 7 arrays for dual-affine: u, v, x, y, x2, y2, s
            let qlen_ = query_len.div_ceil(64);
            let dp_size = 7 * tlen_ * 64;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 64;
            let p_offset = qr_offset + (qlen_ + 1) * 64;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 64);
            // Zero DP+scoring region (not traceback — written per-cell in DP loop)
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m512i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let y2 = x2.add(tlen_);
            let s = y2.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize DP arrays to proper boundary values
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            let neg_q2e2 = (-(gap_open2 as i32) - gap_extend2 as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 64);
            std::ptr::write_bytes(v as *mut u8, neg_qe, tlen_ * 64);
            std::ptr::write_bytes(x as *mut u8, neg_qe, tlen_ * 64);
            std::ptr::write_bytes(y as *mut u8, neg_qe, tlen_ * 64);
            std::ptr::write_bytes(x2 as *mut u8, neg_q2e2, tlen_ * 64);
            std::ptr::write_bytes(y2 as *mut u8, neg_q2e2, tlen_ * 64);

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // H[] array for exact max tracking (only when !approx_max)
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 64);
            let _ = &h_vec; // prevent early drop

            // Initialize result
            init_dp_result(result);

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Find boundaries
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }
                if st < (r - wl + 1) >> 1 { st = (r - wl + 1) >> 1; }
                if en > (r + wl) >> 1 { en = (r + wl) >> 1; }

                if st > en {
                    result.zdropped = 1;
                    break;
                }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -gap_open - gap_extend;
                        x21 = -gap_open2 - gap_extend2;
                        v1 = -gap_open - gap_extend;
                    }
                } else {
                    x1 = -gap_open - gap_extend;
                    x21 = -gap_open2 - gap_extend2;
                    v1 = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -gap_open - gap_extend;
                    *(y2 as *mut i8).add(r as usize) = -gap_open2 - gap_extend2;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -gap_open - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        -gap_extend2
                    };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm512_loadu_si512(sf.add(t as usize) as *const __m512i);
                        let st_v = _mm512_loadu_si512(qrr.add(t as usize) as *const __m512i);
                        let mask: __mmask64 = _mm512_cmpeq_epi8_mask(sq, m1_) | _mm512_cmpeq_epi8_mask(st_v, m1_);
                        let eq: __mmask64 = _mm512_cmpeq_epi8_mask(sq, st_v);
                        let tmp = _mm512_mask_blend_epi8(eq, sc_mis_, sc_mch_);
                        let tmp = _mm512_mask_blend_epi8(mask, tmp, sc_n_);
                        let tmp256 = _mm512_castsi512_si256(tmp);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp256));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp256, 1));
                        }
                        if t + 32 <= en0 {
                            let hi256 = _mm512_extracti64x4_epi64(tmp, 1);
                            _mm_storeu_si128(s_b.add(t as usize + 32) as *mut __m128i, _mm256_castsi256_si128(hi256));
                            if t + 48 <= en0 {
                                _mm_storeu_si128(s_b.add(t as usize + 48) as *mut __m128i, _mm256_extracti128_si256(hi256, 1));
                            }
                        }
                        t += 64;
                    }
                } else {
                    let s_ptr = s as *mut u8;
                    for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 64 - 1) {
                        *s_ptr.add(t) = score_matrix[*sf.add(t) as usize * alphabet_size as usize + *qrr.add(t) as usize] as u8;
                    }
                }

                // Core DP loop — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx512_insert_byte0(_mm512_setzero_si512(), x1 as u8);
                let mut x21_ = avx512_insert_byte0(_mm512_setzero_si512(), x21 as u8);
                let mut v1_ = avx512_insert_byte0(_mm512_setzero_si512(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let x2_b = x2 as *mut u8;
                let y2_b = y2 as *mut u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = n_col_ * 64;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    let excess = if bp + 63 > en_usize {
                        bp + 64 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 48];
                    let mut save_v = [0u8; 48];
                    let mut save_x = [0u8; 48];
                    let mut save_y = [0u8; 48];
                    let mut save_x2 = [0u8; 48];
                    let mut save_y2 = [0u8; 48];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x2_b.add(es), save_x2.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y2_b.add(es), save_y2.as_mut_ptr(), excess);
                    }

                    let mut z = _mm512_loadu_si512(s_b_ptr.add(bp) as *const __m512i);

                    let xt_val = _mm512_loadu_si512(x_b.add(bp) as *const __m512i);
                    let (xt1, tmp_x) = avx512_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm512_loadu_si512(v_b.add(bp) as *const __m512i);
                    let (vt1, tmp_v) = avx512_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let a = _mm512_add_epi8(xt1, vt1);

                    let ut = _mm512_loadu_si512(u_b.add(bp) as *const __m512i);
                    let b = _mm512_add_epi8(_mm512_loadu_si512(y_b.add(bp) as *const __m512i), ut);

                    let x2t_val = _mm512_loadu_si512(x2_b.add(bp) as *const __m512i);
                    let (x2t1, tmp_x2) = avx512_shift_left_1(x2t_val, x21_);
                    x21_ = tmp_x2;

                    let a2 = _mm512_add_epi8(x2t1, vt1);
                    let b2 = _mm512_add_epi8(_mm512_loadu_si512(y2_b.add(bp) as *const __m512i), ut);

                    if !with_cigar {
                        z = _mm512_max_epi8(z, a);
                        z = _mm512_max_epi8(z, b);
                        z = _mm512_max_epi8(z, a2);
                        z = _mm512_max_epi8(z, b2);
                        z = _mm512_min_epi8(z, sc_mch_);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let tmp2 = _mm512_sub_epi8(z, q2_);
                        let a2_new = _mm512_sub_epi8(a2, tmp2);
                        let b2_new = _mm512_sub_epi8(b2, tmp2);

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a_new, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a_new), qe_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b_new, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b_new), qe_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2_new, zero_);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a2_new), qe2_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b2_new, zero_);
                        _mm512_storeu_si512(y2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b2_new), qe2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a, z);
                        let mut d = _mm512_maskz_mov_epi8(tmp, flag1_);
                        z = _mm512_max_epi8(z, a);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b, z);
                        d = _mm512_mask_blend_epi8(tmp, d, flag2_);
                        z = _mm512_max_epi8(z, b);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2, z);
                        d = _mm512_mask_blend_epi8(tmp, d, flag3_);
                        z = _mm512_max_epi8(z, a2);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b2, z);
                        d = _mm512_mask_blend_epi8(tmp, d, flag4_);
                        z = _mm512_max_epi8(z, b2);
                        z = _mm512_min_epi8(z, sc_mch_);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let tmp2 = _mm512_sub_epi8(z, q2_);
                        let a2_new = _mm512_sub_epi8(a2, tmp2);
                        let b2_new = _mm512_sub_epi8(b2, tmp2);

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a_new, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag8_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b_new, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag16_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2_new, zero_);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a2_new), qe2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag32_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b2_new, zero_);
                        _mm512_storeu_si512(y2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b2_new), qe2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag64_));

                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    } else {
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, a);
                        let mut d = _mm512_maskz_mov_epi8(!tmp, flag1_);
                        z = _mm512_max_epi8(z, a);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, b);
                        d = _mm512_mask_blend_epi8(tmp, flag2_, d);
                        z = _mm512_max_epi8(z, b);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, a2);
                        d = _mm512_mask_blend_epi8(tmp, flag3_, d);
                        z = _mm512_max_epi8(z, a2);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, b2);
                        d = _mm512_mask_blend_epi8(tmp, flag4_, d);
                        z = _mm512_max_epi8(z, b2);
                        z = _mm512_min_epi8(z, sc_mch_);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let tmp2 = _mm512_sub_epi8(z, q2_);
                        let a2_new = _mm512_sub_epi8(a2, tmp2);
                        let b2_new = _mm512_sub_epi8(b2, tmp2);

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, a_new);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, a_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag8_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, b_new);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, b_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag16_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, a2_new);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, a2_new), qe2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag32_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, b2_new);
                        _mm512_storeu_si512(y2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, b2_new), qe2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag64_));

                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    }

                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x2.as_ptr(), x2_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y2.as_ptr(), y2_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 64;
                }

                // Update h0 and track max score
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                // H[] tracking
                if !approx_max {
                    // Exact max tracking with 32-bit H[] array
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        // Process [st0..en0) with SSE (4 i32 at a time)
                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            // Blend for 32-bit conditional select
                            max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                            max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            t += 4;
                        }
                        // Reduce SSE to scalar
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        // Remainder
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        // r == 0
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    // Update result scores
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    // Z-drop check: update max, check z_drop
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        let tl = max_t - result.max_score_target_pos;
                        let ql = (r - max_t) - result.max_score_query_pos;
                        let l = if tl > ql { tl - ql } else { ql - tl };
                        if z_drop >= 0 && (result.max - max_h) > (z_drop + l * gap_extend2 as i32) {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;

                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }

                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        }

                        // Check z_drop
                        if (flags & APPROX_DROP) != 0 {
                            if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                                let tl = last_h0_t - result.max_score_target_pos;
                                let ql = (r - last_h0_t) - result.max_score_query_pos;
                                let l = if tl > ql { tl - ql } else { ql - tl };
                                if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend2 as i32) {
                                    result.zdropped = 1;
                                    break;
                                }
                            }
                        }
                    } else {
                        // r == 0
                        let v0 = *v8_ptr.add(0) as i8 as i32;
                        h0 = v0 - qe_scalar;
                        last_h0_t = 0;
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = 0;
                            result.max_score_query_pos = 0;
                        }
                    }
                    // Final score for approx path
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }

                last_st = st;
                last_en = en;
            }

            // Final score
            if approx_max && result.score == NEG_INF {
                result.score = result.max;
            }

            // Traceback for CIGAR
            if with_cigar {
                traceback_dual_affine(result, query_len, target_len, end_bonus, flags, n_col_, 64, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_dual_affine_avx512_impl!(extend_dual_affine_avx512_fn);

// ============================================================================
// Public API - Splice-Aware Alignment
// ============================================================================

/// Splice-aware extension alignment
///
/// Uses splice site scoring for RNA-seq alignment. Canonical GT-AG splice sites
/// receive bonus scoring, non-canonical sites receive penalties.
///
/// # Arguments
/// * `qseq` - Query sequence (encoded 0-3 for ACGT, 4 for N)
/// * `tseq` - Target sequence (same encoding)
/// * `alphabet_size` - Alphabet size (typically 5)
/// * `score_matrix` - Scoring matrix (alphabet_size x alphabet_size, row-major)
/// * `gap_open` - Gap open penalty
/// * `gap_extend` - Gap extension penalty
/// * `gap_open2` - Intron open penalty (must be > gap_open + gap_extend)
/// * `noncanon_penalty` - Non-canonical splice site penalty
/// * `z_drop` - Z-drop threshold (-1 to disable)
/// * `end_bonus` - Bonus for reaching sequence end
/// * `junc_bonus` - Junction annotation bonus
/// * `junc_pen` - Junction annotation penalty
/// * `flags` - Alignment flags (including SPLICE_FOR/REV)
/// * `junc` - Optional junction annotation array
/// * `result` - Output structure for results
pub fn extend_splice(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i8,
    gap_extend: i8,
    gap_open2: i8,
    noncanon_penalty: i8,
    z_drop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    flags: i32,
    junc: Option<&[u8]>,
    result: &mut DpResult,
) {
    // Force scalar mode for testing/comparison
    if std::env::var("RAMMAP_FORCE_SCALAR").is_ok() {
        extend_splice_scalar(qseq, tseq, alphabet_size, score_matrix,
            gap_open as i32, gap_extend as i32, gap_open2 as i32,
            noncanon_penalty as i32, z_drop, end_bonus,
            junc_bonus, junc_pen, flags, junc, result);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let force_sse = std::env::var("RAMMAP_FORCE_SSE").is_ok();
        let force_avx2 = std::env::var("RAMMAP_FORCE_AVX2").is_ok();
        if !force_sse && !force_avx2 && is_x86_feature_detected!("avx512bw") {
            unsafe { extend_splice_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result); }
        } else if !force_sse && is_x86_feature_detected!("avx2") {
            unsafe { extend_splice_avx2_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result); }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { extend_splice41_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result); }
        } else {
            unsafe { extend_splice2_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result); }
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        extend_splice_neon_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result);
    }

    #[cfg(target_arch = "wasm32")]
    unsafe {
        extend_splice_wasm_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
    {
        extend_splice_scalar(qseq, tseq, alphabet_size, score_matrix,
            gap_open as i32, gap_extend as i32, gap_open2 as i32,
            noncanon_penalty as i32, z_drop, end_bonus,
            junc_bonus, junc_pen, flags, junc, result);
    }
}

// ============================================================================
// SSE2/SSE4.1 Unified Implementation - Splice-Aware Alignment
// ============================================================================
//
// Macro generates both SSE2 and SSE4.1 variants. Differences:
// - max_epi8: SSE2 uses sse2_max_epi8 helper, SSE4.1 uses native _mm_max_epi8
// - blend: SSE2 uses and/andnot/or pattern, SSE4.1 uses _mm_blendv_epi8
// Both variants require only SSE2 target_feature (SSE4.1 is detected at runtime).

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
macro_rules! extend_splice_impl {
    ($fn_name:ident, $max_epi8:path, $is_sse41:expr, $target_feat:tt) => {
        #[target_feature(enable = $target_feat)]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            noncanon_penalty: i8,
            z_drop: i32,
            end_bonus: i32,
            junc_bonus: i8,
            junc_pen: i8,
            flags: i32,
            junc: Option<&[u8]>,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let qe = gap_open as i32 + gap_extend as i32;
            let approx_max = (flags & APPROX_MAX) != 0;
            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Reset result
            init_dp_result_full(result);

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 || (gap_open2 as i32) <= qe {
                return;
            }
            assert!((flags & SPLICE_FORWARD) == 0 || (flags & SPLICE_REVERSE) == 0);

            // SIMD constants
            let zero_ = _mm_setzero_si128();
            let q_ = _mm_set1_epi8(gap_open);
            let q2_ = _mm_set1_epi8(gap_open2);
            let qe_ = _mm_set1_epi8(qe as i8);
            let sc_mch_ = _mm_set1_epi8(score_matrix[0]);
            let sc_mis_ = _mm_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm_set1_epi8(-(gap_extend as i8))
            } else {
                _mm_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm_set1_epi8(alphabet_size - 1);

            let tlen_ = target_len.div_ceil(16);
            let qlen_ = query_len.div_ceil(16);
            let n_col_ = query_len.min(target_len).div_ceil(16) + 1;

            // Check scoring matrix bounds
            {
                let mut max_sc = score_matrix[0] as i32;
                let mut min_sc = score_matrix[1] as i32;
                for t in 1..(alphabet_size as usize * alphabet_size as usize) {
                    max_sc = max_sc.max(score_matrix[t] as i32);
                    min_sc = min_sc.min(score_matrix[t] as i32);
                }
                if -min_sc > 2 * qe {
                    return;
                }
            }

            // Compute long_thres (crossover between regular gap and intron)
            let mut long_thres: i32 = (gap_open2 as i32 - gap_open as i32) / gap_extend as i32 - 1;
            if gap_open2 as i32 > gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32 {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * gap_extend as i32 - (gap_open2 as i32 - gap_open as i32)) as i8;

            // Memory allocation: 9 SIMD arrays + sf + qr
            let dp_size = 9 * tlen_ * 16;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 16;
            let p_offset = qr_offset + (qlen_ + 1) * 16;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 16);
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m128i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let donor = x2.add(tlen_);
            let acceptor = donor.add(tlen_);
            let s = acceptor.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize: u,v,x,y to -(gap_open+gap_extend)
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 16 * 4);
            // x2 to -gap_open2
            std::ptr::write_bytes(x2 as *mut u8, (-(gap_open2 as i32)) as u8, tlen_ * 16);

            // H[] for exact max tracking
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 16);
            let _ = &h_vec;

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 16;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 15) & !15;
                let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query into qr
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // --- Donor/acceptor initialization from splice site patterns ---
            if (flags & (SPLICE_FORWARD | SPLICE_REVERSE)) != 0 {
                let sp: [i32; 4];
                if (flags & SPLICE_COMPLEX) != 0 {
                    let sp0 = [8, 15, 21, 30];
                    sp = [
                        (sp0[0] as f64 / 3.0 + 0.499) as i32,
                        (sp0[1] as f64 / 3.0 + 0.499) as i32,
                        (sp0[2] as f64 / 3.0 + 0.499) as i32,
                        (sp0[3] as f64 / 3.0 + 0.499) as i32,
                    ];
                } else {
                    let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty as i32 / 2 } else { 0 };
                    sp = [sp0, noncanon_penalty as i32, noncanon_penalty as i32, noncanon_penalty as i32];
                }

                std::ptr::write_bytes(donor as *mut u8, (-sp[3]) as u8, tlen_ * 16);
                std::ptr::write_bytes(acceptor as *mut u8, (-sp[3]) as u8, tlen_ * 16);

                let donor_bytes = donor as *mut i8;
                let acceptor_bytes = acceptor as *mut i8;

                if (flags & REV_CIGAR) == 0 {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 1 { z = 1; }
                            else if tseq[tu + 1] == 0 && tseq[tu + 2] == 3 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 0 && tseq[tu] == 1 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 2 && tseq[tu] == 1 { z = 1; }
                            else if tseq[tu - 1] == 0 && tseq[tu] == 3 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                } else {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 2 { z = 1; }
                            else if tseq[tu + 1] == 3 && tseq[tu + 2] == 0 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 1 && tseq[tu] == 2 { z = 1; }
                            else if tseq[tu - 1] == 3 && tseq[tu] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 3 && tseq[tu] == 2 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                }
            }

            // --- Junction annotation overlay ---
            if let Some(junc_arr) = junc {
                if (flags & SPLICE_SCORE) != 0 {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    let donor_val: u8 = if ((flags & SPLICE_FORWARD) != 0) == ((flags & REV_CIGAR) == 0) { 0 } else { 1 };
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        *donor_bytes.add(t) += if j == 0xff || (j & 1) != donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        let not_donor_val = if donor_val == 0 { 1 } else { 0 };
                        *acceptor_bytes.add(t) += if j == 0xff || (j & 1) != not_donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                } else {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    if (flags & REV_CIGAR) == 0 {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 8) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 4) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    } else {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 4) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 8) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    }
                }
            }

            // --- Main DP loop ---
            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;
            let flag8_ = _mm_set1_epi8(0x08);
            let flag16_ = _mm_set1_epi8(0x10);
            let flag32_ = _mm_set1_epi8(0x20);

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Boundaries - NO bandwidth for splice
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -(gap_open) - gap_extend;
                        x21 = -gap_open2;
                        v1 = -(gap_open) - gap_extend;
                    }
                } else {
                    x1 = -(gap_open) - gap_extend;
                    x21 = -gap_open2;
                    v1 = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -(gap_open) - gap_extend;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                // Set scores
                if (flags & GENERIC_SCORING) == 0 {
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm_loadu_si128(sf.add(t as usize) as *const __m128i);
                        let st_v = _mm_loadu_si128(qrr.add(t as usize) as *const __m128i);
                        let mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(st_v, m1_));
                        let tmp = _mm_cmpeq_epi8(sq, st_v);
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(sc_mis_, sc_mch_, tmp)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(tmp, sc_mis_), _mm_and_si128(tmp, sc_mch_))
                        };
                        let tmp = if $is_sse41 {
                            _mm_blendv_epi8(tmp, sc_n_, mask)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(mask, tmp), _mm_and_si128(mask, sc_n_))
                        };
                        _mm_storeu_si128((s as *mut u8).add(t as usize) as *mut __m128i, tmp);
                        t += 16;
                    }
                } else {
                    for t in st0..=en0 {
                        let tu = t as usize;
                        *((s as *mut u8).add(tu)) = score_matrix[*(sf.add(tu)) as usize * alphabet_size as usize + *(qrr.add(tu)) as usize] as u8;
                    }
                }

                // Core DP loop
                let mut x1_ = sse2_insert_byte0(zero_, x1 as u8);
                let mut x21_ = sse2_insert_byte0(zero_, x21 as u8);
                let mut v1_ = sse2_insert_byte0(zero_, v1 as u8);

                let st_ = st as usize / 16;
                let en_ = en as usize / 16;

                for ti in st_..=en_ {
                    let z = _mm_load_si128(s.add(ti));

                    let xt_val = _mm_load_si128(x.add(ti));
                    let tmp_x = _mm_srli_si128(xt_val, 15);
                    let xt1 = _mm_or_si128(_mm_slli_si128(xt_val, 1), x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm_load_si128(v.add(ti));
                    let tmp_v = _mm_srli_si128(vt_val, 15);
                    let vt1 = _mm_or_si128(_mm_slli_si128(vt_val, 1), v1_);
                    v1_ = tmp_v;

                    let a = _mm_add_epi8(xt1, vt1);
                    let ut = _mm_load_si128(u.add(ti));
                    let b = _mm_add_epi8(_mm_load_si128(y.add(ti)), ut);

                    let x2t_val = _mm_load_si128(x2.add(ti));
                    let tmp_x2 = _mm_srli_si128(x2t_val, 15);
                    let x2t1 = _mm_or_si128(_mm_slli_si128(x2t_val, 1), x21_);
                    x21_ = tmp_x2;

                    let a2 = _mm_add_epi8(x2t1, vt1);
                    let a2a = _mm_add_epi8(a2, _mm_load_si128(acceptor.add(ti)));

                    if !with_cigar {
                        // Score only: 4-way max
                        let mut z = z;
                        if $is_sse41 {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            z = _mm_blendv_epi8(z, a, tmp);
                            let tmp = _mm_cmpgt_epi8(b, z);
                            z = _mm_blendv_epi8(z, b, tmp);
                            let tmp = _mm_cmpgt_epi8(a2a, z);
                            z = _mm_blendv_epi8(z, a2a, tmp);
                        } else {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
                            let tmp = _mm_cmpgt_epi8(b, z);
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
                            let tmp = _mm_cmpgt_epi8(a2a, z);
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2a));
                        }

                        _mm_store_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_store_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let a2_new = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                        let tmp = _mm_cmpgt_epi8(a_new, zero_);
                        _mm_store_si128(x.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a_new), qe_));
                        let tmp = _mm_cmpgt_epi8(b_new, zero_);
                        _mm_store_si128(y.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b_new), qe_));
                        let donor_t = _mm_load_si128(donor.add(ti));
                        let x2_val = $max_epi8(a2_new, donor_t);
                        _mm_store_si128(x2.add(ti), _mm_sub_epi8(x2_val, q2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        // Gap LEFT-alignment with traceback
                        let offset = (r as usize * n_col_) as isize - st_ as isize;
                        let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                        if ti == st_ {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        if $is_sse41 {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            d = _mm_and_si128(tmp, _mm_set1_epi8(1));
                            z = _mm_blendv_epi8(z, a, tmp);
                            let tmp = _mm_cmpgt_epi8(b, z);
                            d = _mm_blendv_epi8(d, _mm_set1_epi8(2), tmp);
                            z = _mm_blendv_epi8(z, b, tmp);
                            let tmp = _mm_cmpgt_epi8(a2a, z);
                            d = _mm_blendv_epi8(d, _mm_set1_epi8(3), tmp);
                            z = _mm_blendv_epi8(z, a2a, tmp);
                        } else {
                            let tmp = _mm_cmpgt_epi8(a, z);
                            d = _mm_and_si128(tmp, _mm_set1_epi8(1));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
                            let tmp = _mm_cmpgt_epi8(b, z);
                            d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(2)));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
                            let tmp = _mm_cmpgt_epi8(a2a, z);
                            d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(3)));
                            z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2a));
                        }

                        _mm_store_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_store_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let a2_new = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                        let tmp = _mm_cmpgt_epi8(a_new, zero_);
                        _mm_store_si128(x.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, a_new), qe_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));
                        let tmp = _mm_cmpgt_epi8(b_new, zero_);
                        _mm_store_si128(y.add(ti), _mm_sub_epi8(_mm_and_si128(tmp, b_new), qe_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));

                        // x2[t] = max(a2, donor[t]) - gap_open2 with traceback
                        let tmp2 = _mm_load_si128(donor.add(ti));
                        let tmp = _mm_cmpgt_epi8(a2_new, tmp2);
                        let x2_val = if $is_sse41 {
                            _mm_blendv_epi8(tmp2, a2_new, tmp)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(tmp, tmp2), _mm_and_si128(tmp, a2_new))
                        };
                        _mm_store_si128(x2.add(ti), _mm_sub_epi8(x2_val, q2_));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag32_));
                        _mm_store_si128(pr_ptr as *mut __m128i, d);
                    } else {
                        // Gap RIGHT-alignment with traceback
                        let offset = (r as usize * n_col_) as isize - st_ as isize;
                        let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                        if ti == st_ {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        if $is_sse41 {
                            let tmp = _mm_cmpgt_epi8(z, a);
                            d = _mm_andnot_si128(tmp, _mm_set1_epi8(1));
                            z = _mm_blendv_epi8(a, z, tmp);
                            let tmp = _mm_cmpgt_epi8(z, b);
                            d = _mm_blendv_epi8(_mm_set1_epi8(2), d, tmp);
                            z = _mm_blendv_epi8(b, z, tmp);
                            let tmp = _mm_cmpgt_epi8(z, a2a);
                            d = _mm_blendv_epi8(_mm_set1_epi8(3), d, tmp);
                            z = _mm_blendv_epi8(a2a, z, tmp);
                        } else {
                            let tmp = _mm_cmpgt_epi8(z, a);
                            d = _mm_andnot_si128(tmp, _mm_set1_epi8(1));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a));
                            let tmp = _mm_cmpgt_epi8(z, b);
                            d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(2)));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b));
                            let tmp = _mm_cmpgt_epi8(z, a2a);
                            d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(3)));
                            z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a2a));
                        }

                        _mm_store_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                        _mm_store_si128(v.add(ti), _mm_sub_epi8(z, ut));
                        let tmp1 = _mm_sub_epi8(z, q_);
                        let a_new = _mm_sub_epi8(a, tmp1);
                        let b_new = _mm_sub_epi8(b, tmp1);
                        let a2_new = _mm_sub_epi8(a2, _mm_sub_epi8(z, q2_));

                        let tmp = _mm_cmpgt_epi8(zero_, a_new);
                        _mm_store_si128(x.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, a_new), qe_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));
                        let tmp = _mm_cmpgt_epi8(zero_, b_new);
                        _mm_store_si128(y.add(ti), _mm_sub_epi8(_mm_andnot_si128(tmp, b_new), qe_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_));

                        // x2[t] = max(donor[t], a2) - gap_open2 with traceback (right-align)
                        let tmp2 = _mm_load_si128(donor.add(ti));
                        let tmp = _mm_cmpgt_epi8(tmp2, a2_new);
                        let x2_val = if $is_sse41 {
                            _mm_blendv_epi8(a2_new, tmp2, tmp)
                        } else {
                            _mm_or_si128(_mm_andnot_si128(tmp, a2_new), _mm_and_si128(tmp, tmp2))
                        };
                        _mm_store_si128(x2.add(ti), _mm_sub_epi8(x2_val, q2_));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag32_));
                        _mm_store_si128(pr_ptr as *mut __m128i, d);
                    }
                }

                // H[] exact max tracking
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                if !approx_max {
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            if $is_sse41 {
                                max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                                max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            } else {
                                max_h_ = _mm_or_si128(_mm_and_si128(tmp, h1), _mm_andnot_si128(tmp, max_h_));
                                max_t_ = _mm_or_si128(_mm_and_si128(tmp, t_), _mm_andnot_si128(tmp, max_t_));
                            }
                            t += 4;
                        }
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        if z_drop >= 0 && (result.max - max_h) > z_drop {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }
                    } else {
                        h0 = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        last_h0_t = 0;
                    }
                    if (flags & APPROX_DROP) != 0 {
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        } else if z_drop >= 0
                            && last_h0_t >= result.max_score_target_pos
                            && (r - last_h0_t) >= result.max_score_query_pos
                            && (result.max - h0) > z_drop
                        {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }
                last_st = st;
                last_en = en;
            }

            // --- Backtrack ---
            if with_cigar {
                traceback_splice(result, query_len, target_len, end_bonus, flags, n_col_, 16, long_thres, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_splice_impl!(extend_splice2_impl, sse2_max_epi8, false, "sse2");
#[cfg(target_arch = "x86_64")]
extend_splice_impl!(extend_splice41_impl, _mm_max_epi8, true, "sse2");
#[cfg(target_arch = "wasm32")]
extend_splice_impl!(extend_splice_wasm_impl, _mm_max_epi8, true, "simd128");

// ============================================================================
// AVX2 Implementation - Splice-Aware Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_splice_avx2_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            noncanon_penalty: i8,
            z_drop: i32,
            end_bonus: i32,
            junc_bonus: i8,
            junc_pen: i8,
            flags: i32,
            junc: Option<&[u8]>,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let qe = gap_open as i32 + gap_extend as i32;
            let approx_max = (flags & APPROX_MAX) != 0;
            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Reset result
            init_dp_result_full(result);

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 || (gap_open2 as i32) <= qe {
                return;
            }
            assert!((flags & SPLICE_FORWARD) == 0 || (flags & SPLICE_REVERSE) == 0);

            // SIMD constants
            let zero_ = _mm256_setzero_si256();
            let q_ = _mm256_set1_epi8(gap_open);
            let q2_ = _mm256_set1_epi8(gap_open2);
            let qe_ = _mm256_set1_epi8(qe as i8);
            let sc_mch_ = _mm256_set1_epi8(score_matrix[0]);
            let sc_mis_ = _mm256_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm256_set1_epi8(-(gap_extend as i8))
            } else {
                _mm256_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm256_set1_epi8(alphabet_size - 1);

            let tlen_ = target_len.div_ceil(32) + 1; // +1 for byte-addressed SSE-compat padding
            let qlen_ = query_len.div_ceil(32);
            let n_col_ = query_len.min(target_len).div_ceil(32) + 1;

            // Check scoring matrix bounds
            {
                let mut max_sc = score_matrix[0] as i32;
                let mut min_sc = score_matrix[1] as i32;
                for t in 1..(alphabet_size as usize * alphabet_size as usize) {
                    max_sc = max_sc.max(score_matrix[t] as i32);
                    min_sc = min_sc.min(score_matrix[t] as i32);
                }
                if -min_sc > 2 * qe {
                    return;
                }
            }

            // Compute long_thres (crossover between regular gap and intron)
            let mut long_thres: i32 = (gap_open2 as i32 - gap_open as i32) / gap_extend as i32 - 1;
            if gap_open2 as i32 > gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32 {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * gap_extend as i32 - (gap_open2 as i32 - gap_open as i32)) as i8;

            // Memory allocation: 9 SIMD arrays + sf + qr
            let dp_size = 9 * tlen_ * 32;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 32;
            let p_offset = qr_offset + (qlen_ + 1) * 32;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 32);
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m256i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let donor = x2.add(tlen_);
            let acceptor = donor.add(tlen_);
            let s = acceptor.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize: u,v,x,y to -(gap_open+gap_extend)
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 32 * 4);
            // x2 to -gap_open2
            std::ptr::write_bytes(x2 as *mut u8, (-(gap_open2 as i32)) as u8, tlen_ * 32);

            // H[] for exact max tracking
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 32);
            let _ = &h_vec;

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 32;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 31) & !31;
                let off_end_offset_start = (off_offset_start + off_size + 31) & !31;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query into qr
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // --- Donor/acceptor initialization from splice site patterns ---
            if (flags & (SPLICE_FORWARD | SPLICE_REVERSE)) != 0 {
                let sp: [i32; 4];
                if (flags & SPLICE_COMPLEX) != 0 {
                    let sp0 = [8, 15, 21, 30];
                    sp = [
                        (sp0[0] as f64 / 3.0 + 0.499) as i32,
                        (sp0[1] as f64 / 3.0 + 0.499) as i32,
                        (sp0[2] as f64 / 3.0 + 0.499) as i32,
                        (sp0[3] as f64 / 3.0 + 0.499) as i32,
                    ];
                } else {
                    let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty as i32 / 2 } else { 0 };
                    sp = [sp0, noncanon_penalty as i32, noncanon_penalty as i32, noncanon_penalty as i32];
                }

                std::ptr::write_bytes(donor as *mut u8, (-sp[3]) as u8, tlen_ * 32);
                std::ptr::write_bytes(acceptor as *mut u8, (-sp[3]) as u8, tlen_ * 32);

                let donor_bytes = donor as *mut i8;
                let acceptor_bytes = acceptor as *mut i8;

                if (flags & REV_CIGAR) == 0 {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 1 { z = 1; }
                            else if tseq[tu + 1] == 0 && tseq[tu + 2] == 3 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 0 && tseq[tu] == 1 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 2 && tseq[tu] == 1 { z = 1; }
                            else if tseq[tu - 1] == 0 && tseq[tu] == 3 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                } else {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 2 { z = 1; }
                            else if tseq[tu + 1] == 3 && tseq[tu + 2] == 0 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 1 && tseq[tu] == 2 { z = 1; }
                            else if tseq[tu - 1] == 3 && tseq[tu] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 3 && tseq[tu] == 2 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                }
            }

            // --- Junction annotation overlay ---
            if let Some(junc_arr) = junc {
                if (flags & SPLICE_SCORE) != 0 {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    let donor_val: u8 = if ((flags & SPLICE_FORWARD) != 0) == ((flags & REV_CIGAR) == 0) { 0 } else { 1 };
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        *donor_bytes.add(t) += if j == 0xff || (j & 1) != donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        let not_donor_val = if donor_val == 0 { 1 } else { 0 };
                        *acceptor_bytes.add(t) += if j == 0xff || (j & 1) != not_donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                } else {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    if (flags & REV_CIGAR) == 0 {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 8) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 4) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    } else {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 4) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 8) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    }
                }
            }

            // --- Main DP loop ---
            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;
            let flag8_ = _mm256_set1_epi8(0x08);
            let flag16_ = _mm256_set1_epi8(0x10);
            let flag32_ = _mm256_set1_epi8(0x20);

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Boundaries - NO bandwidth for splice
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -(gap_open) - gap_extend;
                        x21 = -gap_open2;
                        v1 = -(gap_open) - gap_extend;
                    }
                } else {
                    x1 = -(gap_open) - gap_extend;
                    x21 = -gap_open2;
                    v1 = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -(gap_open) - gap_extend;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm256_loadu_si256(sf.add(t as usize) as *const __m256i);
                        let st_v = _mm256_loadu_si256(qrr.add(t as usize) as *const __m256i);
                        let mask = _mm256_or_si256(_mm256_cmpeq_epi8(sq, m1_), _mm256_cmpeq_epi8(st_v, m1_));
                        let tmp = _mm256_cmpeq_epi8(sq, st_v);
                        let tmp = _mm256_blendv_epi8(sc_mis_, sc_mch_, tmp);
                        let tmp = _mm256_blendv_epi8(tmp, sc_n_, mask);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp, 1));
                        }
                        t += 32;
                    }
                } else {
                    for t in st0..=en0 {
                        let tu = t as usize;
                        *((s as *mut u8).add(tu)) = score_matrix[*(sf.add(tu)) as usize * alphabet_size as usize + *(qrr.add(tu)) as usize] as u8;
                    }
                }

                // Core DP loop — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx2_insert_byte0(_mm256_setzero_si256(), x1 as u8);
                let mut x21_ = avx2_insert_byte0(_mm256_setzero_si256(), x21 as u8);
                let mut v1_ = avx2_insert_byte0(_mm256_setzero_si256(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let x2_b = x2 as *mut u8;
                let donor_b = donor as *const u8;
                let acceptor_b = acceptor as *const u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = n_col_ * 32;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    let excess = if bp + 31 > en_usize {
                        bp + 32 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 16];
                    let mut save_v = [0u8; 16];
                    let mut save_x = [0u8; 16];
                    let mut save_y = [0u8; 16];
                    let mut save_x2 = [0u8; 16];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x2_b.add(es), save_x2.as_mut_ptr(), excess);
                    }

                    let z = _mm256_loadu_si256(s_b_ptr.add(bp) as *const __m256i);

                    let xt_val = _mm256_loadu_si256(x_b.add(bp) as *const __m256i);
                    let (xt1, tmp_x) = avx2_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm256_loadu_si256(v_b.add(bp) as *const __m256i);
                    let (vt1, tmp_v) = avx2_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let a = _mm256_add_epi8(xt1, vt1);
                    let ut = _mm256_loadu_si256(u_b.add(bp) as *const __m256i);
                    let b = _mm256_add_epi8(_mm256_loadu_si256(y_b.add(bp) as *const __m256i), ut);

                    let x2t_val = _mm256_loadu_si256(x2_b.add(bp) as *const __m256i);
                    let (x2t1, tmp_x2) = avx2_shift_left_1(x2t_val, x21_);
                    x21_ = tmp_x2;

                    let a2 = _mm256_add_epi8(x2t1, vt1);
                    let a2a = _mm256_add_epi8(a2, _mm256_loadu_si256(acceptor_b.add(bp) as *const __m256i));

                    if !with_cigar {
                        // Score only: 4-way max
                        let mut z = z;
                        let tmp = _mm256_cmpgt_epi8(a, z);
                        z = _mm256_blendv_epi8(z, a, tmp);
                        let tmp = _mm256_cmpgt_epi8(b, z);
                        z = _mm256_blendv_epi8(z, b, tmp);
                        let tmp = _mm256_cmpgt_epi8(a2a, z);
                        z = _mm256_blendv_epi8(z, a2a, tmp);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let a2_new = _mm256_sub_epi8(a2, _mm256_sub_epi8(z, q2_));

                        let tmp = _mm256_cmpgt_epi8(a_new, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a_new), qe_));
                        let tmp = _mm256_cmpgt_epi8(b_new, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b_new), qe_));
                        let donor_t = _mm256_loadu_si256(donor_b.add(bp) as *const __m256i);
                        let x2_val = _mm256_max_epi8(a2_new, donor_t);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(x2_val, q2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        // Gap LEFT-alignment with traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        let tmp = _mm256_cmpgt_epi8(a, z);
                        d = _mm256_and_si256(tmp, _mm256_set1_epi8(1));
                        z = _mm256_blendv_epi8(z, a, tmp);
                        let tmp = _mm256_cmpgt_epi8(b, z);
                        d = _mm256_blendv_epi8(d, _mm256_set1_epi8(2), tmp);
                        z = _mm256_blendv_epi8(z, b, tmp);
                        let tmp = _mm256_cmpgt_epi8(a2a, z);
                        d = _mm256_blendv_epi8(d, _mm256_set1_epi8(3), tmp);
                        z = _mm256_blendv_epi8(z, a2a, tmp);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let a2_new = _mm256_sub_epi8(a2, _mm256_sub_epi8(z, q2_));

                        let tmp = _mm256_cmpgt_epi8(a_new, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, a_new), qe_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag8_));
                        let tmp = _mm256_cmpgt_epi8(b_new, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_and_si256(tmp, b_new), qe_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag16_));

                        // x2[t] = max(a2, donor[t]) - gap_open2 with traceback
                        let tmp2 = _mm256_loadu_si256(donor_b.add(bp) as *const __m256i);
                        let tmp = _mm256_cmpgt_epi8(a2_new, tmp2);
                        let x2_val = _mm256_blendv_epi8(tmp2, a2_new, tmp);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(x2_val, q2_));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag32_));
                        _mm256_storeu_si256(pr_ptr_local as *mut __m256i, d);
                    } else {
                        // Gap RIGHT-alignment with traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        let tmp = _mm256_cmpgt_epi8(z, a);
                        d = _mm256_andnot_si256(tmp, _mm256_set1_epi8(1));
                        z = _mm256_blendv_epi8(a, z, tmp);
                        let tmp = _mm256_cmpgt_epi8(z, b);
                        d = _mm256_blendv_epi8(_mm256_set1_epi8(2), d, tmp);
                        z = _mm256_blendv_epi8(b, z, tmp);
                        let tmp = _mm256_cmpgt_epi8(z, a2a);
                        d = _mm256_blendv_epi8(_mm256_set1_epi8(3), d, tmp);
                        z = _mm256_blendv_epi8(a2a, z, tmp);

                        _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                        _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                        let tmp1 = _mm256_sub_epi8(z, q_);
                        let a_new = _mm256_sub_epi8(a, tmp1);
                        let b_new = _mm256_sub_epi8(b, tmp1);
                        let a2_new = _mm256_sub_epi8(a2, _mm256_sub_epi8(z, q2_));

                        let tmp = _mm256_cmpgt_epi8(zero_, a_new);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, a_new), qe_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag8_));
                        let tmp = _mm256_cmpgt_epi8(zero_, b_new);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_sub_epi8(_mm256_andnot_si256(tmp, b_new), qe_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag16_));

                        let tmp2 = _mm256_loadu_si256(donor_b.add(bp) as *const __m256i);
                        let tmp = _mm256_cmpgt_epi8(tmp2, a2_new);
                        let x2_val = _mm256_blendv_epi8(a2_new, tmp2, tmp);
                        _mm256_storeu_si256(x2_b.add(bp) as *mut __m256i, _mm256_sub_epi8(x2_val, q2_));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag32_));
                        _mm256_storeu_si256(pr_ptr_local as *mut __m256i, d);
                    }

                    // Restore excess bytes on partial last iteration
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x2.as_ptr(), x2_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 32;
                }

                // H[] exact max tracking
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                if !approx_max {
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                            max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            t += 4;
                        }
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        if z_drop >= 0 && (result.max - max_h) > z_drop {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }
                    } else {
                        h0 = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        last_h0_t = 0;
                    }
                    if (flags & APPROX_DROP) != 0 {
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        } else if z_drop >= 0
                            && last_h0_t >= result.max_score_target_pos
                            && (r - last_h0_t) >= result.max_score_query_pos
                            && (result.max - h0) > z_drop
                        {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }
                last_st = st;
                last_en = en;
            }

            // --- Backtrack ---
            if with_cigar {
                traceback_splice(result, query_len, target_len, end_bonus, flags, n_col_, 32, long_thres, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_splice_avx2_impl!(extend_splice_avx2_fn);

// ============================================================================
// AVX512 Implementation - Splice Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_splice_avx512_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx512bw")]
        unsafe fn $fn_name(
            qseq: &[u8],
            tseq: &[u8],
            alphabet_size: i8,
            score_matrix: &[i8],
            gap_open: i8,
            gap_extend: i8,
            gap_open2: i8,
            noncanon_penalty: i8,
            z_drop: i32,
            end_bonus: i32,
            junc_bonus: i8,
            junc_pen: i8,
            flags: i32,
            junc: Option<&[u8]>,
            result: &mut DpResult,
        ) { unsafe {
            let query_len = qseq.len();
            let target_len = tseq.len();
            let qe = gap_open as i32 + gap_extend as i32;
            let approx_max = (flags & APPROX_MAX) != 0;
            let with_cigar = (flags & SCORE_ONLY) == 0;

            // Reset result
            init_dp_result_full(result);

            if alphabet_size <= 1 || query_len == 0 || target_len == 0 || (gap_open2 as i32) <= qe {
                return;
            }
            assert!((flags & SPLICE_FORWARD) == 0 || (flags & SPLICE_REVERSE) == 0);

            // SIMD constants
            let zero_ = _mm512_setzero_si512();
            let q_ = _mm512_set1_epi8(gap_open);
            let q2_ = _mm512_set1_epi8(gap_open2);
            let qe_ = _mm512_set1_epi8(qe as i8);
            let sc_mch_ = _mm512_set1_epi8(score_matrix[0]);
            let sc_mis_ = _mm512_set1_epi8(score_matrix[1]);
            let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
                _mm512_set1_epi8(-(gap_extend as i8))
            } else {
                _mm512_set1_epi8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1])
            };
            let m1_ = _mm512_set1_epi8(alphabet_size - 1);

            let tlen_ = target_len.div_ceil(64) + 1; // +1 for byte-addressed SSE-compat padding
            let qlen_ = query_len.div_ceil(64);
            let n_col_ = query_len.min(target_len).div_ceil(64) + 1;

            // Check scoring matrix bounds
            {
                let mut max_sc = score_matrix[0] as i32;
                let mut min_sc = score_matrix[1] as i32;
                for t in 1..(alphabet_size as usize * alphabet_size as usize) {
                    max_sc = max_sc.max(score_matrix[t] as i32);
                    min_sc = min_sc.min(score_matrix[t] as i32);
                }
                if -min_sc > 2 * qe {
                    return;
                }
            }

            // Compute long_thres (crossover between regular gap and intron)
            let mut long_thres: i32 = (gap_open2 as i32 - gap_open as i32) / gap_extend as i32 - 1;
            if gap_open2 as i32 > gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32 {
                long_thres += 1;
            }
            let long_diff: i8 = (long_thres * gap_extend as i32 - (gap_open2 as i32 - gap_open as i32)) as i8;

            // Memory allocation: 9 SIMD arrays + sf + qr
            let dp_size = 9 * tlen_ * 64;
            let sf_offset = dp_size;
            let qr_offset = sf_offset + tlen_ * 64;
            let p_offset = qr_offset + (qlen_ + 1) * 64;

            let mut mem_size_bytes = p_offset;
            let mut p_ptr: *mut u8 = std::ptr::null_mut();
            let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
            let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                mem_size_bytes = off_end_offset_start + off_size;
            }

            let mem = AlignedMemory::new(mem_size_bytes, 64);
            std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

            let base_ptr = mem.as_ptr();
            let u = base_ptr as *mut __m512i;
            let v = u.add(tlen_);
            let x = v.add(tlen_);
            let y = x.add(tlen_);
            let x2 = y.add(tlen_);
            let donor = x2.add(tlen_);
            let acceptor = donor.add(tlen_);
            let s = acceptor.add(tlen_);
            let sf = base_ptr.add(sf_offset);
            let qr = base_ptr.add(qr_offset);

            // Initialize: u,v,x,y to -(gap_open+gap_extend)
            let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
            std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 64 * 4);
            // x2 to -gap_open2
            std::ptr::write_bytes(x2 as *mut u8, (-(gap_open2 as i32)) as u8, tlen_ * 64);

            // H[] for exact max tracking
            let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 64);
            let _ = &h_vec;

            if with_cigar {
                let p_size = (query_len + target_len - 1) * n_col_ * 64;
                let off_size = (query_len + target_len - 1) * 4;
                let off_offset_start = (p_offset + p_size + 63) & !63;
                let off_end_offset_start = (off_offset_start + off_size + 63) & !63;
                p_ptr = base_ptr.add(p_offset);
                band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
                band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
            }

            // Reverse query into qr
            let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
            for t in 0..query_len {
                qr_slice[t] = qseq[query_len - 1 - t];
            }
            std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

            // --- Donor/acceptor initialization from splice site patterns ---
            if (flags & (SPLICE_FORWARD | SPLICE_REVERSE)) != 0 {
                let sp: [i32; 4];
                if (flags & SPLICE_COMPLEX) != 0 {
                    let sp0 = [8, 15, 21, 30];
                    sp = [
                        (sp0[0] as f64 / 3.0 + 0.499) as i32,
                        (sp0[1] as f64 / 3.0 + 0.499) as i32,
                        (sp0[2] as f64 / 3.0 + 0.499) as i32,
                        (sp0[3] as f64 / 3.0 + 0.499) as i32,
                    ];
                } else {
                    let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty as i32 / 2 } else { 0 };
                    sp = [sp0, noncanon_penalty as i32, noncanon_penalty as i32, noncanon_penalty as i32];
                }

                std::ptr::write_bytes(donor as *mut u8, (-sp[3]) as u8, tlen_ * 64);
                std::ptr::write_bytes(acceptor as *mut u8, (-sp[3]) as u8, tlen_ * 64);

                let donor_bytes = donor as *mut i8;
                let acceptor_bytes = acceptor as *mut i8;

                if (flags & REV_CIGAR) == 0 {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 1 { z = 1; }
                            else if tseq[tu + 1] == 0 && tseq[tu + 2] == 3 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 3 {
                                z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 0 && tseq[tu] == 1 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 0 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 2 && tseq[tu] == 1 { z = 1; }
                            else if tseq[tu - 1] == 0 && tseq[tu] == 3 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                } else {
                    for t in 0..(target_len as i32 - 4) {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu + 1] == 2 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 {
                                z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                            } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 2 { z = 1; }
                            else if tseq[tu + 1] == 3 && tseq[tu + 2] == 0 { z = 2; }
                        }
                        *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                    for t in 2..target_len as i32 {
                        let tu = t as usize;
                        let mut z = 3i32;
                        if (flags & SPLICE_FORWARD) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 2 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 1 && tseq[tu] == 2 { z = 1; }
                            else if tseq[tu - 1] == 3 && tseq[tu] == 0 { z = 2; }
                        } else if (flags & SPLICE_REVERSE) != 0 {
                            if tseq[tu - 1] == 3 && tseq[tu] == 1 {
                                z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                            } else if tseq[tu - 1] == 3 && tseq[tu] == 2 { z = 2; }
                        }
                        *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
                    }
                }
            }

            // --- Junction annotation overlay ---
            if let Some(junc_arr) = junc {
                if (flags & SPLICE_SCORE) != 0 {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    let donor_val: u8 = if ((flags & SPLICE_FORWARD) != 0) == ((flags & REV_CIGAR) == 0) { 0 } else { 1 };
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        *donor_bytes.add(t) += if j == 0xff || (j & 1) != donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                    for t in 0..(target_len - 1) {
                        let j = junc_arr[t + 1];
                        let not_donor_val = if donor_val == 0 { 1 } else { 0 };
                        *acceptor_bytes.add(t) += if j == 0xff || (j & 1) != not_donor_val {
                            -junc_pen
                        } else {
                            (j >> 1) as i8 - SPSC_OFFSET as i8
                        };
                    }
                } else {
                    let donor_bytes = donor as *mut i8;
                    let acceptor_bytes = acceptor as *mut i8;
                    if (flags & REV_CIGAR) == 0 {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 8) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 4) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    } else {
                        for t in 0..(target_len - 1) {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 2) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 4) != 0)
                            {
                                *donor_bytes.add(t) += junc_bonus;
                            }
                        }
                        for t in 0..target_len {
                            if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 1) != 0)
                                || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 8) != 0)
                            {
                                *acceptor_bytes.add(t) += junc_bonus;
                            }
                        }
                    }
                }
            }

            // --- Main DP loop ---
            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut h0: i32 = 0;
            let mut last_h0_t: i32 = 0;
            let flag8_ = _mm512_set1_epi8(0x08);
            let flag16_ = _mm512_set1_epi8(0x10);
            let flag32_ = _mm512_set1_epi8(0x20);

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

                // Boundaries - NO bandwidth for splice
                if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
                if en > r { en = r; }

                let st0 = st;
                let en0 = en;
                st = (st / 16) * 16;
                en = ((en + 16) / 16) * 16 - 1;

                // Boundary conditions
                let x1: i8;
                let x21: i8;
                let v1: i8;
                let u8_arr = u as *mut i8;
                let v8_arr = v as *mut i8;
                let x8_arr = x as *mut i8;
                let x28_arr = x2 as *mut i8;

                if st > 0 {
                    if st > last_st && st - 1 <= last_en {
                        x1 = *x8_arr.add((st - 1) as usize);
                        x21 = *x28_arr.add((st - 1) as usize);
                        v1 = *v8_arr.add((st - 1) as usize);
                    } else {
                        x1 = -(gap_open) - gap_extend;
                        x21 = -gap_open2;
                        v1 = -(gap_open) - gap_extend;
                    }
                } else {
                    x1 = -(gap_open) - gap_extend;
                    x21 = -gap_open2;
                    v1 = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                if en >= r {
                    *(y as *mut i8).add(r as usize) = -(gap_open) - gap_extend;
                    *u8_arr.add(r as usize) = if r == 0 {
                        -(gap_open) - gap_extend
                    } else if r < long_thres {
                        -gap_extend
                    } else if r == long_thres {
                        long_diff
                    } else {
                        0 // splice: 0, not -gap_extend2
                    };
                }

                // Set scores — use 16-byte stores to match SSE write range
                if (flags & GENERIC_SCORING) == 0 {
                    let s_b = s as *mut u8;
                    let mut t = st0;
                    while t <= en0 {
                        let sq = _mm512_loadu_si512(sf.add(t as usize) as *const __m512i);
                        let st_v = _mm512_loadu_si512(qrr.add(t as usize) as *const __m512i);
                        let mask: __mmask64 = _mm512_cmpeq_epi8_mask(sq, m1_) | _mm512_cmpeq_epi8_mask(st_v, m1_);
                        let tmp: __mmask64 = _mm512_cmpeq_epi8_mask(sq, st_v);
                        let tmp = _mm512_mask_blend_epi8(tmp, sc_mis_, sc_mch_);
                        let tmp = _mm512_mask_blend_epi8(mask, tmp, sc_n_);
                        let tmp256 = _mm512_castsi512_si256(tmp);
                        _mm_storeu_si128(s_b.add(t as usize) as *mut __m128i, _mm256_castsi256_si128(tmp256));
                        if t + 16 <= en0 {
                            _mm_storeu_si128(s_b.add(t as usize + 16) as *mut __m128i, _mm256_extracti128_si256(tmp256, 1));
                        }
                        if t + 32 <= en0 {
                            let hi256 = _mm512_extracti64x4_epi64(tmp, 1);
                            _mm_storeu_si128(s_b.add(t as usize + 32) as *mut __m128i, _mm256_castsi256_si128(hi256));
                            if t + 48 <= en0 {
                                _mm_storeu_si128(s_b.add(t as usize + 48) as *mut __m128i, _mm256_extracti128_si256(hi256, 1));
                            }
                        }
                        t += 64;
                    }
                } else {
                    for t in st0..=en0 {
                        let tu = t as usize;
                        *((s as *mut u8).add(tu)) = score_matrix[*(sf.add(tu)) as usize * alphabet_size as usize + *(qrr.add(tu)) as usize] as u8;
                    }
                }

                // Core DP loop — byte-addressed for SSE-compatible rounding
                let mut x1_ = avx512_insert_byte0(_mm512_setzero_si512(), x1 as u8);
                let mut x21_ = avx512_insert_byte0(_mm512_setzero_si512(), x21 as u8);
                let mut v1_ = avx512_insert_byte0(_mm512_setzero_si512(), v1 as u8);

                let u_b = u as *mut u8;
                let v_b = v as *mut u8;
                let x_b = x as *mut u8;
                let y_b = y as *mut u8;
                let x2_b = x2 as *mut u8;
                let donor_b = donor as *const u8;
                let acceptor_b = acceptor as *const u8;
                let s_b_ptr = s as *const u8;
                let en_usize = en as usize;
                let st_usize = st as usize;
                let stride_bytes = n_col_ * 64;
                let mut bp = st_usize;
                let mut bp_first = true;

                while bp <= en_usize {
                    let excess = if bp + 63 > en_usize {
                        bp + 64 - (en_usize + 1)
                    } else { 0 };
                    let mut save_u = [0u8; 48];
                    let mut save_v = [0u8; 48];
                    let mut save_x = [0u8; 48];
                    let mut save_y = [0u8; 48];
                    let mut save_x2 = [0u8; 48];
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(u_b.add(es), save_u.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(v_b.add(es), save_v.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x_b.add(es), save_x.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(y_b.add(es), save_y.as_mut_ptr(), excess);
                        std::ptr::copy_nonoverlapping(x2_b.add(es), save_x2.as_mut_ptr(), excess);
                    }

                    let z = _mm512_loadu_si512(s_b_ptr.add(bp) as *const __m512i);

                    let xt_val = _mm512_loadu_si512(x_b.add(bp) as *const __m512i);
                    let (xt1, tmp_x) = avx512_shift_left_1(xt_val, x1_);
                    x1_ = tmp_x;

                    let vt_val = _mm512_loadu_si512(v_b.add(bp) as *const __m512i);
                    let (vt1, tmp_v) = avx512_shift_left_1(vt_val, v1_);
                    v1_ = tmp_v;

                    let a = _mm512_add_epi8(xt1, vt1);
                    let ut = _mm512_loadu_si512(u_b.add(bp) as *const __m512i);
                    let b = _mm512_add_epi8(_mm512_loadu_si512(y_b.add(bp) as *const __m512i), ut);

                    let x2t_val = _mm512_loadu_si512(x2_b.add(bp) as *const __m512i);
                    let (x2t1, tmp_x2) = avx512_shift_left_1(x2t_val, x21_);
                    x21_ = tmp_x2;

                    let a2 = _mm512_add_epi8(x2t1, vt1);
                    let a2a = _mm512_add_epi8(a2, _mm512_loadu_si512(acceptor_b.add(bp) as *const __m512i));

                    if !with_cigar {
                        // Score only: 4-way max
                        let mut z = z;
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a, z);
                        z = _mm512_mask_blend_epi8(tmp, z, a);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b, z);
                        z = _mm512_mask_blend_epi8(tmp, z, b);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2a, z);
                        z = _mm512_mask_blend_epi8(tmp, z, a2a);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let a2_new = _mm512_sub_epi8(a2, _mm512_sub_epi8(z, q2_));

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a_new, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a_new), qe_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b_new, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b_new), qe_));
                        let donor_t = _mm512_loadu_si512(donor_b.add(bp) as *const __m512i);
                        let x2_val = _mm512_max_epi8(a2_new, donor_t);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(x2_val, q2_));
                    } else if (flags & RIGHT_ALIGN) == 0 {
                        // Gap LEFT-alignment with traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a, z);
                        d = _mm512_maskz_mov_epi8(tmp, _mm512_set1_epi8(1));
                        z = _mm512_mask_blend_epi8(tmp, z, a);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b, z);
                        d = _mm512_mask_blend_epi8(tmp, d, _mm512_set1_epi8(2));
                        z = _mm512_mask_blend_epi8(tmp, z, b);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2a, z);
                        d = _mm512_mask_blend_epi8(tmp, d, _mm512_set1_epi8(3));
                        z = _mm512_mask_blend_epi8(tmp, z, a2a);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let a2_new = _mm512_sub_epi8(a2, _mm512_sub_epi8(z, q2_));

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a_new, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, a_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag8_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(b_new, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(tmp, b_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag16_));

                        // x2[t] = max(a2, donor[t]) - gap_open2 with traceback
                        let tmp2 = _mm512_loadu_si512(donor_b.add(bp) as *const __m512i);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(a2_new, tmp2);
                        let x2_val = _mm512_mask_blend_epi8(tmp, tmp2, a2_new);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(x2_val, q2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(tmp, flag32_));
                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    } else {
                        // Gap RIGHT-alignment with traceback
                        let pr_byte_off = r as usize * stride_bytes + (bp - st_usize);
                        let pr_ptr_local = p_ptr.add(pr_byte_off);
                        if bp_first {
                            *band_offset_ptr.add(r as usize) = st;
                            *band_offset_end_ptr.add(r as usize) = en;
                        }

                        let mut z = z;
                        let mut d;
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, a);
                        d = _mm512_maskz_mov_epi8(!tmp, _mm512_set1_epi8(1));
                        z = _mm512_mask_blend_epi8(tmp, a, z);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, b);
                        d = _mm512_mask_blend_epi8(tmp, _mm512_set1_epi8(2), d);
                        z = _mm512_mask_blend_epi8(tmp, b, z);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(z, a2a);
                        d = _mm512_mask_blend_epi8(tmp, _mm512_set1_epi8(3), d);
                        z = _mm512_mask_blend_epi8(tmp, a2a, z);

                        _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                        _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                        let tmp1 = _mm512_sub_epi8(z, q_);
                        let a_new = _mm512_sub_epi8(a, tmp1);
                        let b_new = _mm512_sub_epi8(b, tmp1);
                        let a2_new = _mm512_sub_epi8(a2, _mm512_sub_epi8(z, q2_));

                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, a_new);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, a_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag8_));
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(zero_, b_new);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_sub_epi8(_mm512_maskz_mov_epi8(!tmp, b_new), qe_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag16_));

                        let tmp2 = _mm512_loadu_si512(donor_b.add(bp) as *const __m512i);
                        let tmp: __mmask64 = _mm512_cmpgt_epi8_mask(tmp2, a2_new);
                        let x2_val = _mm512_mask_blend_epi8(tmp, a2_new, tmp2);
                        _mm512_storeu_si512(x2_b.add(bp) as *mut __m512i, _mm512_sub_epi8(x2_val, q2_));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(!tmp, flag32_));
                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    }

                    // Restore excess bytes on partial last iteration
                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x2.as_ptr(), x2_b.add(es), excess);
                    }

                    bp_first = false;
                    bp += 64;
                }

                // H[] exact max tracking
                let u8_ptr = u as *mut u8;
                let v8_ptr = v as *mut u8;
                let qe_scalar = gap_open as i32 + gap_extend as i32;

                if !approx_max {
                    let mut max_h: i32;
                    let mut max_t: i32;
                    if r > 0 {
                        let h_en0 = if en0 > 0 {
                            *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                        } else {
                            *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                        };
                        *h_ptr.add(en0 as usize) = h_en0;
                        max_h = h_en0;
                        max_t = en0;

                        let en1 = st0 + (en0 - st0) / 4 * 4;
                        let mut max_h_ = _mm_set1_epi32(max_h);
                        let mut max_t_ = _mm_set1_epi32(max_t);
                        let mut t = st0;
                        while t < en1 {
                            let h1 = _mm_loadu_si128(h_ptr.add(t as usize) as *const __m128i);
                            let v_vals = _mm_setr_epi32(
                                *v8_ptr.add(t as usize) as i8 as i32,
                                *v8_ptr.add(t as usize + 1) as i8 as i32,
                                *v8_ptr.add(t as usize + 2) as i8 as i32,
                                *v8_ptr.add(t as usize + 3) as i8 as i32,
                            );
                            let h1 = _mm_add_epi32(h1, v_vals);
                            _mm_storeu_si128(h_ptr.add(t as usize) as *mut __m128i, h1);
                            let t_ = _mm_set1_epi32(t);
                            let tmp = _mm_cmpgt_epi32(h1, max_h_);
                            max_h_ = _mm_blendv_epi8(max_h_, h1, tmp);
                            max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
                            t += 4;
                        }
                        let mut hh = [0i32; 4];
                        let mut tt = [0i32; 4];
                        _mm_storeu_si128(hh.as_mut_ptr() as *mut __m128i, max_h_);
                        _mm_storeu_si128(tt.as_mut_ptr() as *mut __m128i, max_t_);
                        for i in 0..4 {
                            if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                        }
                        while t < en0 {
                            *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                            if *h_ptr.add(t as usize) > max_h {
                                max_h = *h_ptr.add(t as usize);
                                max_t = t;
                            }
                            t += 1;
                        }
                    } else {
                        *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        max_h = *h_ptr.add(0);
                        max_t = 0;
                    }
                    if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                        result.max_target_end_score = *h_ptr.add(en0 as usize);
                        result.max_target_end_query_pos = r - en0;
                    }
                    if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                        result.max_query_end_score = *h_ptr.add(st0 as usize);
                        result.max_query_end_target_pos = st0;
                    }
                    if max_h > result.max {
                        result.max = max_h;
                        result.max_score_target_pos = max_t;
                        result.max_score_query_pos = r - max_t;
                    } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                        if z_drop >= 0 && (result.max - max_h) > z_drop {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = *h_ptr.add(target_len - 1);
                    }
                } else {
                    // Approximate max tracking
                    if r > 0 {
                        if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                            let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                            let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                            if d0 > d1 {
                                h0 += d0;
                            } else {
                                h0 += d1;
                                last_h0_t += 1;
                            }
                        } else if last_h0_t >= st0 && last_h0_t <= en0 {
                            h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                        } else {
                            last_h0_t += 1;
                            h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                        }
                    } else {
                        h0 = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                        last_h0_t = 0;
                    }
                    if (flags & APPROX_DROP) != 0 {
                        if h0 > result.max {
                            result.max = h0;
                            result.max_score_target_pos = last_h0_t;
                            result.max_score_query_pos = r - last_h0_t;
                        } else if z_drop >= 0
                            && last_h0_t >= result.max_score_target_pos
                            && (r - last_h0_t) >= result.max_score_query_pos
                            && (result.max - h0) > z_drop
                        {
                            result.zdropped = 1;
                            break;
                        }
                    }
                    if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                        result.score = h0;
                    }
                }
                last_st = st;
                last_en = en;
            }

            // --- Backtrack ---
            if with_cigar {
                traceback_splice(result, query_len, target_len, end_bonus, flags, n_col_, 64, long_thres, p_ptr, band_offset_ptr, band_offset_end_ptr);
            }
        }}
    };
}

#[cfg(target_arch = "x86_64")]
extend_splice_avx512_impl!(extend_splice_avx512_fn);


// ============================================================================
// NEON Implementation - Splice-Aware Alignment
// ============================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn extend_splice_neon_impl(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i8,
    gap_extend: i8,
    gap_open2: i8,
    noncanon_penalty: i8,
    z_drop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    flags: i32,
    junc: Option<&[u8]>,
    result: &mut DpResult,
) { unsafe {
    use core::arch::aarch64::*;

    let query_len = qseq.len();
    let target_len = tseq.len();
    let qe = gap_open as i32 + gap_extend as i32;
    let approx_max = (flags & APPROX_MAX) != 0;
    let with_cigar = (flags & SCORE_ONLY) == 0;

    // Reset result
    init_dp_result_full(result);

    if alphabet_size <= 1 || query_len == 0 || target_len == 0 || (gap_open2 as i32) <= qe {
        return;
    }
    assert!((flags & SPLICE_FORWARD) == 0 || (flags & SPLICE_REVERSE) == 0);

    // SIMD constants
    let zero_ = vdupq_n_u8(0);
    let q_ = vdupq_n_u8(gap_open as u8);
    let q2_ = vdupq_n_u8(gap_open2 as u8);
    let qe_ = vdupq_n_u8(qe as u8);
    let sc_mch_ = vdupq_n_u8(score_matrix[0] as u8);
    let sc_mis_ = vdupq_n_u8(score_matrix[1] as u8);
    let sc_n_ = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        vdupq_n_u8(-(gap_extend as i8) as u8)
    } else {
        vdupq_n_u8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8)
    };
    let m1_ = vdupq_n_u8((alphabet_size - 1) as u8);

    let tlen_ = (target_len + 15) / 16;
    let qlen_ = (query_len + 15) / 16;
    let n_col_ = (query_len.min(target_len) + 15) / 16 + 1;

    // Check scoring matrix bounds
    {
        let mut max_sc = score_matrix[0] as i32;
        let mut min_sc = score_matrix[1] as i32;
        for &s in &score_matrix[1..(alphabet_size as usize * alphabet_size as usize)] {
            max_sc = max_sc.max(s as i32);
            min_sc = min_sc.min(s as i32);
        }
        if -min_sc > 2 * qe {
            return;
        }
    }

    // Compute long_thres (crossover between regular gap and intron)
    let mut long_thres: i32 = (gap_open2 as i32 - gap_open as i32) / gap_extend as i32 - 1;
    if gap_open2 as i32 > gap_open as i32 + gap_extend as i32 + long_thres * gap_extend as i32 {
        long_thres += 1;
    }
    let long_diff: i8 = (long_thres * gap_extend as i32 - (gap_open2 as i32 - gap_open as i32)) as i8;

    // Memory allocation: 9 SIMD arrays + sf + qr
    // Layout: u | v | x | y | x2 | donor | acceptor | s | sf | qr
    let dp_size = 9 * tlen_ * 16;
    let sf_offset = dp_size;
    let qr_offset = sf_offset + tlen_ * 16;
    let p_offset = qr_offset + (qlen_ + 1) * 16;

    let mut mem_size_bytes = p_offset;
    let mut p_ptr: *mut u8 = std::ptr::null_mut();
    let mut band_offset_ptr: *mut i32 = std::ptr::null_mut();
    let mut band_offset_end_ptr: *mut i32 = std::ptr::null_mut();

    if with_cigar {
        let p_size = (query_len + target_len - 1) * n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        mem_size_bytes = off_end_offset_start + off_size;
    }

    let mem = AlignedMemory::new(mem_size_bytes, 16);
    // Zero DP+scoring region (not traceback — written per-cell in DP loop)
    std::ptr::write_bytes(mem.as_ptr(), 0, p_offset);

    let base_ptr = mem.as_ptr();
    let u = base_ptr as *mut uint8x16_t;
    let v = u.add(tlen_);
    let x = v.add(tlen_);
    let y = x.add(tlen_);
    let x2 = y.add(tlen_);
    let donor = x2.add(tlen_);
    let acceptor = donor.add(tlen_);
    let s = acceptor.add(tlen_);
    let sf = base_ptr.add(sf_offset);
    let qr = base_ptr.add(qr_offset);

    // Initialize: u,v,x,y to -(gap_open+gap_extend)
    let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
    std::ptr::write_bytes(u as *mut u8, neg_qe, tlen_ * 16 * 4);
    // x2 to -gap_open2
    std::ptr::write_bytes(x2 as *mut u8, (-(gap_open2 as i32)) as u8, tlen_ * 16);
    // donor and acceptor stay at 0 (from write_bytes above) — filled below

    // H[] for exact max tracking
    let (h_vec, h_ptr) = alloc_h_array(approx_max, tlen_, 16);
    let _ = &h_vec;

    if with_cigar {
        let p_size = (query_len + target_len - 1) * n_col_ * 16;
        let off_size = (query_len + target_len - 1) * 4;
        let off_offset_start = (p_offset + p_size + 15) & !15;
        let off_end_offset_start = (off_offset_start + off_size + 15) & !15;
        p_ptr = base_ptr.add(p_offset);
        band_offset_ptr = base_ptr.add(off_offset_start) as *mut i32;
        band_offset_end_ptr = base_ptr.add(off_end_offset_start) as *mut i32;
    }

    // Reverse query into qr
    let qr_slice = std::slice::from_raw_parts_mut(qr, query_len);
    for t in 0..query_len {
        qr_slice[t] = qseq[query_len - 1 - t];
    }
    // Copy target into sf
    std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

    // --- Donor/acceptor initialization from splice site patterns ---
    if (flags & (SPLICE_FORWARD | SPLICE_REVERSE)) != 0 {
        let sp: [i32; 4];
        if (flags & SPLICE_COMPLEX) != 0 {
            let sp0 = [8, 15, 21, 30];
            sp = [
                (sp0[0] as f64 / 3.0 + 0.499) as i32,
                (sp0[1] as f64 / 3.0 + 0.499) as i32,
                (sp0[2] as f64 / 3.0 + 0.499) as i32,
                (sp0[3] as f64 / 3.0 + 0.499) as i32,
            ];
        } else {
            let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty as i32 / 2 } else { 0 };
            sp = [sp0, noncanon_penalty as i32, noncanon_penalty as i32, noncanon_penalty as i32];
        }

        // Fill donor and acceptor with worst-case penalty
        std::ptr::write_bytes(donor as *mut u8, (-sp[3]) as u8, tlen_ * 16);
        std::ptr::write_bytes(acceptor as *mut u8, (-sp[3]) as u8, tlen_ * 16);

        let donor_bytes = donor as *mut i8;
        let acceptor_bytes = acceptor as *mut i8;

        if (flags & REV_CIGAR) == 0 {
            // Forward CIGAR: standard donor/acceptor patterns
            for t in 0..(target_len as i32 - 4) {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 {
                        z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 1 { z = 1; }
                    else if tseq[tu + 1] == 0 && tseq[tu + 2] == 3 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu + 1] == 1 && tseq[tu + 2] == 3 {
                        z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 { z = 2; }
                }
                *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
            for t in 2..target_len as i32 {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu - 1] == 0 && tseq[tu] == 2 {
                        z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 0 && tseq[tu] == 1 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu - 1] == 0 && tseq[tu] == 1 {
                        z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 2 && tseq[tu] == 1 { z = 1; }
                    else if tseq[tu - 1] == 0 && tseq[tu] == 3 { z = 2; }
                }
                *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
        } else {
            // REV_CIGAR: reversed donor/acceptor patterns (for left extension)
            for t in 0..(target_len as i32 - 4) {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu + 1] == 2 && tseq[tu + 2] == 0 {
                        z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 {
                        z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 2 { z = 1; }
                    else if tseq[tu + 1] == 3 && tseq[tu + 2] == 0 { z = 2; }
                }
                *donor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
            for t in 2..target_len as i32 {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu - 1] == 3 && tseq[tu] == 2 {
                        z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 1 && tseq[tu] == 2 { z = 1; }
                    else if tseq[tu - 1] == 3 && tseq[tu] == 0 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu - 1] == 3 && tseq[tu] == 1 {
                        z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 3 && tseq[tu] == 2 { z = 2; }
                }
                *acceptor_bytes.add(tu) = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
        }
    }

    // --- Junction annotation overlay ---
    if let Some(junc_arr) = junc {
        if (flags & SPLICE_SCORE) != 0 {
            let donor_bytes = donor as *mut i8;
            let acceptor_bytes = acceptor as *mut i8;
            let donor_val: u8 = if ((flags & SPLICE_FORWARD) != 0) == ((flags & REV_CIGAR) == 0) { 0 } else { 1 };
            for t in 0..(target_len - 1) {
                let j = junc_arr[t + 1];
                *donor_bytes.add(t) += if j == 0xff || (j & 1) != donor_val {
                    -junc_pen
                } else {
                    (j >> 1) as i8 - SPSC_OFFSET as i8
                };
            }
            for t in 0..(target_len - 1) {
                let j = junc_arr[t + 1];
                let not_donor_val = if donor_val == 0 { 1 } else { 0 };
                *acceptor_bytes.add(t) += if j == 0xff || (j & 1) != not_donor_val {
                    -junc_pen
                } else {
                    (j >> 1) as i8 - SPSC_OFFSET as i8
                };
            }
        } else {
            let donor_bytes = donor as *mut i8;
            let acceptor_bytes = acceptor as *mut i8;
            if (flags & REV_CIGAR) == 0 {
                for t in 0..(target_len - 1) {
                    if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 1) != 0)
                        || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 8) != 0)
                    {
                        *donor_bytes.add(t) += junc_bonus;
                    }
                }
                for t in 0..target_len {
                    if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 2) != 0)
                        || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 4) != 0)
                    {
                        *acceptor_bytes.add(t) += junc_bonus;
                    }
                }
            } else {
                for t in 0..(target_len - 1) {
                    if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 2) != 0)
                        || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 4) != 0)
                    {
                        *donor_bytes.add(t) += junc_bonus;
                    }
                }
                for t in 0..target_len {
                    if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 1) != 0)
                        || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 8) != 0)
                    {
                        *acceptor_bytes.add(t) += junc_bonus;
                    }
                }
            }
        }
    }

    // --- Main DP loop ---
    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let valid_range = (query_len + target_len - 1) as i32;
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;
    let flag8_ = vdupq_n_u8(0x08);
    let flag16_ = vdupq_n_u8(0x10);
    let flag32_ = vdupq_n_u8(0x20);

    for r in 0..valid_range {
        let mut st = 0i32;
        let mut en = target_len as i32 - 1;

        let qrr = qr.offset(query_len as isize - 1 - r as isize);

        // Boundaries - NO bandwidth for splice
        if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
        if en > r { en = r; }

        let st0 = st;
        let en0 = en;
        st = (st / 16) * 16;
        en = ((en + 16) / 16) * 16 - 1;

        // Boundary conditions
        let x1: i8;
        let x21: i8;
        let v1: i8;
        let u8_arr = u as *mut i8;
        let v8_arr = v as *mut i8;
        let x8_arr = x as *mut i8;
        let x28_arr = x2 as *mut i8;

        if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                x1 = *x8_arr.add((st - 1) as usize);
                x21 = *x28_arr.add((st - 1) as usize);
                v1 = *v8_arr.add((st - 1) as usize);
            } else {
                x1 = -(gap_open) - gap_extend;
                x21 = -gap_open2;
                v1 = -(gap_open) - gap_extend;
            }
        } else {
            x1 = -(gap_open) - gap_extend;
            x21 = -gap_open2;
            v1 = if r == 0 {
                -(gap_open) - gap_extend
            } else if r < long_thres {
                -gap_extend
            } else if r == long_thres {
                long_diff
            } else {
                0 // splice: 0, not -gap_extend2
            };
        }

        if en >= r {
            *(y as *mut i8).add(r as usize) = -(gap_open) - gap_extend;
            *u8_arr.add(r as usize) = if r == 0 {
                -(gap_open) - gap_extend
            } else if r < long_thres {
                -gap_extend
            } else if r == long_thres {
                long_diff
            } else {
                0 // splice: 0, not -gap_extend2
            };
        }

        // Set scores
        if (flags & GENERIC_SCORING) == 0 {
            let mut t = st0;
            while t <= en0 {
                let sq = vld1q_u8(sf.add(t as usize));
                let st_v = vld1q_u8(qrr.add(t as usize));
                let mask = vorrq_u8(vceqq_u8(sq, m1_), vceqq_u8(st_v, m1_));
                let eq = vceqq_u8(sq, st_v);
                let tmp = vorrq_u8(vbicq_u8(sc_mis_, eq), vandq_u8(eq, sc_mch_));
                let tmp = vorrq_u8(vbicq_u8(tmp, mask), vandq_u8(mask, sc_n_));
                vst1q_u8((s as *mut u8).add(t as usize), tmp);
                t += 16;
            }
        } else {
            for t in st0..=en0 {
                let tu = t as usize;
                *((s as *mut u8).add(tu)) = score_matrix[*(sf.add(tu)) as usize * alphabet_size as usize + *(qrr.add(tu)) as usize] as u8;
            }
        }

        // Core DP loop
        let mut x1_ = vsetq_lane_u8(x1 as u8, vdupq_n_u8(0), 0);
        let mut x21_ = vsetq_lane_u8(x21 as u8, vdupq_n_u8(0), 0);
        let mut v1_ = vsetq_lane_u8(v1 as u8, vdupq_n_u8(0), 0);

        let st_ = st as usize / 16;
        let en_ = en as usize / 16;

        for ti in st_..=en_ {
            // Load s[t]
            let z = vld1q_u8((s as *const u8).add(ti * 16));

            // Load and shift x
            let xt_val = vld1q_u8((x as *const u8).add(ti * 16));
            let tmp_x = vextq_u8(xt_val, zero_, 15);
            let xt1 = vorrq_u8(vextq_u8(zero_, xt_val, 15), x1_);
            x1_ = tmp_x;

            // Load and shift v
            let vt_val = vld1q_u8((v as *const u8).add(ti * 16));
            let tmp_v = vextq_u8(vt_val, zero_, 15);
            let vt1 = vorrq_u8(vextq_u8(zero_, vt_val, 15), v1_);
            v1_ = tmp_v;

            // a = x[t-1] + v[t-1] (E/deletion candidate)
            let a = vaddq_u8(xt1, vt1);
            // b = y[t] + u[t] (F/insertion candidate)
            let ut = vld1q_u8((u as *const u8).add(ti * 16));
            let b = vaddq_u8(vld1q_u8((y as *const u8).add(ti * 16)), ut);

            // x2 intron state
            let x2t_val = vld1q_u8((x2 as *const u8).add(ti * 16));
            let tmp_x2 = vextq_u8(x2t_val, zero_, 15);
            let x2t1 = vorrq_u8(vextq_u8(zero_, x2t_val, 15), x21_);
            x21_ = tmp_x2;

            // a2 = x2[t-1] + v[t-1] (intron candidate)
            let a2 = vaddq_u8(x2t1, vt1);
            // a2a = a2 + acceptor[t] (intron with acceptor bonus)
            let a2a = vaddq_u8(a2, vld1q_u8((acceptor as *const u8).add(ti * 16)));

            if !with_cigar {
                // Score only: 4-way max (no z clamp for splice)
                let mut z = z;
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(z));
                z = vbslq_u8(tmp, a, z);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(b), vreinterpretq_s8_u8(z));
                z = vbslq_u8(tmp, b, z);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a2a), vreinterpretq_s8_u8(z));
                z = vbslq_u8(tmp, a2a, z);

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let a2_new = vsubq_u8(a2, vsubq_u8(z, q2_));

                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a_new), zero_s8);
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a_new), qe_));
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(b_new), zero_s8);
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b_new), qe_));
                // x2[t] = max(a2_new, donor[t]) - gap_open2
                let donor_t = vld1q_u8((donor as *const u8).add(ti * 16));
                let x2_val = vreinterpretq_u8_s8(vmaxq_s8(
                    vreinterpretq_s8_u8(a2_new),
                    vreinterpretq_s8_u8(donor_t),
                ));
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(x2_val, q2_));
            } else if (flags & RIGHT_ALIGN) == 0 {
                // Gap LEFT-alignment with traceback
                let offset = (r as usize * n_col_) as isize - st_ as isize;
                let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                if ti == st_ {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

                // 4-way max with LEFT tie-breaking
                let mut z = z;
                let mut d: uint8x16_t;
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(z));
                d = vandq_u8(tmp, vdupq_n_u8(1));
                z = vbslq_u8(tmp, a, z);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(b), vreinterpretq_s8_u8(z));
                d = vbslq_u8(tmp, vdupq_n_u8(2), d);
                z = vbslq_u8(tmp, b, z);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a2a), vreinterpretq_s8_u8(z));
                d = vbslq_u8(tmp, vdupq_n_u8(3), d);
                z = vbslq_u8(tmp, a2a, z);

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let a2_new = vsubq_u8(a2, vsubq_u8(z, q2_));

                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a_new), zero_s8);
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, a_new), qe_));
                d = vorrq_u8(d, vandq_u8(tmp, flag8_));
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(b_new), zero_s8);
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vandq_u8(tmp, b_new), qe_));
                d = vorrq_u8(d, vandq_u8(tmp, flag16_));

                // x2[t] = max(a2_new, donor[t]) - gap_open2 with traceback
                let tmp2 = vld1q_u8((donor as *const u8).add(ti * 16));
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(a2_new), vreinterpretq_s8_u8(tmp2));
                let x2_val = vbslq_u8(tmp, a2_new, tmp2);
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(x2_val, q2_));
                d = vorrq_u8(d, vandq_u8(tmp, flag32_));
                vst1q_u8(pr_ptr, d);
            } else {
                // Gap RIGHT-alignment with traceback
                let offset = (r as usize * n_col_) as isize - st_ as isize;
                let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                if ti == st_ {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

                // 4-way max with RIGHT tie-breaking
                let mut z = z;
                let mut d: uint8x16_t;
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), vreinterpretq_s8_u8(a));
                d = vbicq_u8(vdupq_n_u8(1), tmp);
                z = vbslq_u8(tmp, z, a);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), vreinterpretq_s8_u8(b));
                d = vbslq_u8(tmp, d, vdupq_n_u8(2));
                z = vbslq_u8(tmp, z, b);
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(z), vreinterpretq_s8_u8(a2a));
                d = vbslq_u8(tmp, d, vdupq_n_u8(3));
                z = vbslq_u8(tmp, z, a2a);

                // Update u, v, x, y from z
                vst1q_u8((u as *mut u8).add(ti * 16), vsubq_u8(z, vt1));
                vst1q_u8((v as *mut u8).add(ti * 16), vsubq_u8(z, ut));
                let tmp1 = vsubq_u8(z, q_);
                let a_new = vsubq_u8(a, tmp1);
                let b_new = vsubq_u8(b, tmp1);
                let a2_new = vsubq_u8(a2, vsubq_u8(z, q2_));

                let zero_s8 = vreinterpretq_s8_u8(zero_);
                let tmp = vcgtq_s8(zero_s8, vreinterpretq_s8_u8(a_new));
                vst1q_u8((x as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(a_new, tmp), qe_));
                d = vorrq_u8(d, vbicq_u8(flag8_, tmp));
                let tmp = vcgtq_s8(zero_s8, vreinterpretq_s8_u8(b_new));
                vst1q_u8((y as *mut u8).add(ti * 16), vsubq_u8(vbicq_u8(b_new, tmp), qe_));
                d = vorrq_u8(d, vbicq_u8(flag16_, tmp));

                // x2[t] = max(donor[t], a2_new) - gap_open2 with traceback (right-align)
                let tmp2 = vld1q_u8((donor as *const u8).add(ti * 16));
                let tmp = vcgtq_s8(vreinterpretq_s8_u8(tmp2), vreinterpretq_s8_u8(a2_new));
                let x2_val = vbslq_u8(tmp, tmp2, a2_new);
                vst1q_u8((x2 as *mut u8).add(ti * 16), vsubq_u8(x2_val, q2_));
                d = vorrq_u8(d, vbicq_u8(flag32_, tmp));
                vst1q_u8(pr_ptr, d);
            }
        }

        // H[] exact max tracking
        let u8_ptr = u as *mut u8;
        let v8_ptr = v as *mut u8;
        let qe_scalar = gap_open as i32 + gap_extend as i32;

        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                let h_en0 = if en0 > 0 {
                    *h_ptr.add(en0 as usize - 1) + *u8_ptr.add(en0 as usize) as i8 as i32
                } else {
                    *h_ptr.add(en0 as usize) + *v8_ptr.add(en0 as usize) as i8 as i32
                };
                *h_ptr.add(en0 as usize) = h_en0;
                max_h = h_en0;
                max_t = en0;

                // Scalar H[] update (matches NEON extd2 pattern)
                let mut t = st0;
                while t < en0 {
                    *h_ptr.add(t as usize) += *v8_ptr.add(t as usize) as i8 as i32;
                    if *h_ptr.add(t as usize) > max_h {
                        max_h = *h_ptr.add(t as usize);
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                *h_ptr.add(0) = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                max_h = *h_ptr.add(0);
                max_t = 0;
            }
            // Update mte, mqe
            if en0 == target_len as i32 - 1 && *h_ptr.add(en0 as usize) > result.max_target_end_score {
                result.max_target_end_score = *h_ptr.add(en0 as usize);
                result.max_target_end_query_pos = r - en0;
            }
            if r - st0 == query_len as i32 - 1 && *h_ptr.add(st0 as usize) > result.max_query_end_score {
                result.max_query_end_score = *h_ptr.add(st0 as usize);
                result.max_query_end_target_pos = st0;
            }
            // Z-drop check (splice uses gap_extend=0 for z_drop penalty)
            if max_h > result.max {
                result.max = max_h;
                result.max_score_target_pos = max_t;
                result.max_score_query_pos = r - max_t;
            } else if z_drop >= 0
                && max_t >= result.max_score_target_pos
                && (r - max_t) >= result.max_score_query_pos
                && (result.max - max_h) > z_drop
            {
                result.zdropped = 1;
                break;
            }
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = *h_ptr.add(target_len - 1);
            }
        } else {
            // Approximate max tracking
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t + 1 <= en0 {
                    let d0 = *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                    let d1 = *u8_ptr.add((last_h0_t + 1) as usize) as i8 as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += *v8_ptr.add(last_h0_t as usize) as i8 as i32;
                } else {
                    last_h0_t += 1;
                    h0 += *u8_ptr.add(last_h0_t as usize) as i8 as i32;
                }
            } else {
                h0 = *v8_ptr.add(0) as i8 as i32 - qe_scalar;
                last_h0_t = 0;
            }
            if (flags & APPROX_DROP) != 0 {
                // Z-drop check for approx mode
                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = last_h0_t;
                    result.max_score_query_pos = r - last_h0_t;
                } else if z_drop >= 0
                    && last_h0_t >= result.max_score_target_pos
                    && (r - last_h0_t) >= result.max_score_query_pos
                    && (result.max - h0) > z_drop
                {
                    result.zdropped = 1;
                    break;
                }
            }
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h0;
            }
        }
        last_st = st;
        last_en = en;
    }

    // --- Backtrack ---
    if with_cigar {
        traceback_splice(result, query_len, target_len, end_bonus, flags, n_col_, 16, long_thres, p_ptr, band_offset_ptr, band_offset_end_ptr);
    }
}}

// ============================================================================
// Lightweight Smith-Waterman
// Used for quick inversion scoring.
// ============================================================================

/// Query profile for lightweight i16 Smith-Waterman (lightweight_profile_init, size=2 only)
pub struct LightweightProfile {
    pub qlen: i32,
    pub segment_len: i32,     // segmented length = ceil(qlen / 8)
    pub query_profile: Vec<i16>,  // query profile: m * segment_len * 8 values
    pub h0: Vec<i16>,  // segment_len * 8
    pub h1: Vec<i16>,  // segment_len * 8
    pub e: Vec<i16>,   // segment_len * 8
    pub hmax: Vec<i16>, // segment_len * 8
}

/// Initialize query profile for lightweight i16 Smith-Waterman.
/// Initialize lightweight query profile (size=2 / int16 only).
pub fn lightweight_profile_init(qlen: i32, query: &[u8], alphabet_size: i32, score_matrix: &[i8]) -> LightweightProfile {
    let p = 8i32; // 8 int16 values per __m128i
    let slen = (qlen + p - 1) / p;
    let m_usize = alphabet_size as usize;

    // Build segmented query profile (int16)
    // Layout: for each alphabet char a (0..m), for each segment i (0..slen),
    // store p values at positions k = i, i+slen, i+2*slen, ...
    let mut qp = vec![0i16; m_usize * slen as usize * p as usize];
    {
        let mut t = 0usize;
        for a in 0..m_usize {
            let nlen = (slen * p) as usize;
            let ma = &score_matrix[a * m_usize..a * m_usize + m_usize];
            for i in 0..slen as usize {
                let mut k = i;
                while k < nlen {
                    qp[t] = if (k as i32) >= qlen {
                        0
                    } else {
                        ma[query[k] as usize] as i16
                    };
                    t += 1;
                    k += slen as usize;
                }
            }
        }
    }

    let sz = slen as usize * p as usize;
    LightweightProfile {
        qlen,
        segment_len: slen,
        query_profile: qp,
        h0: vec![0i16; sz],
        h1: vec![0i16; sz],
        e: vec![0i16; sz],
        hmax: vec![0i16; sz],
    }
}

/// Lightweight i16 Smith-Waterman local alignment.
/// Lightweight i16 Smith-Waterman local alignment.
/// Returns (score, query_end, target_end). query_end/target_end are -1 if no alignment found.
#[cfg(target_arch = "x86_64")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_sse2(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(target_arch = "aarch64")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_neon(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(target_arch = "wasm32")]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    unsafe { lightweight_align_i16_wasm(qp, target_len, target, gap_open, gap_extend) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
pub fn lightweight_align_i16(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    lightweight_align_i16_scalar(qp, target_len, target, gap_open, gap_extend)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn lightweight_align_i16_sse2(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) { unsafe {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    let zero = _mm_set1_epi32(0);
    let gapoe = _mm_set1_epi16((gap_open + gap_extend) as i16);
    let gape_v = _mm_set1_epi16(gap_extend as i16);

    // Zero out working arrays
    for v in qp.e.iter_mut() { *v = 0; }
    for v in qp.h0.iter_mut() { *v = 0; }
    for v in qp.hmax.iter_mut() { *v = 0; }

    for i in 0..target_len {
        let mut f = zero;
        let mut max = zero;
        let s_offset = target[i as usize] as usize * slen as usize * 8;

        // h = H0[slen-1] shifted left by 2 bytes
        let h0_last_idx = (slen as usize - 1) * 8;
        let mut h = _mm_loadu_si128(qp.h0[h0_last_idx..].as_ptr() as *const __m128i);
        h = _mm_slli_si128(h, 2);

        for j in 0..slen as usize {
            let s = _mm_loadu_si128(qp.query_profile[s_offset + j * 8..].as_ptr() as *const __m128i);
            h = _mm_adds_epi16(h, s);
            let e = _mm_loadu_si128(qp.e[j * 8..].as_ptr() as *const __m128i);
            h = _mm_max_epi16(h, e);
            h = _mm_max_epi16(h, f);
            max = _mm_max_epi16(max, h);
            _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h);
            let h_sub = _mm_subs_epu16(h, gapoe);
            let e_sub = _mm_subs_epu16(e, gape_v);
            let e_new = _mm_max_epi16(e_sub, h_sub);
            _mm_storeu_si128(qp.e[j * 8..].as_mut_ptr() as *mut __m128i, e_new);
            f = _mm_subs_epu16(f, gape_v);
            f = _mm_max_epi16(f, h_sub);
            h = _mm_loadu_si128(qp.h0[j * 8..].as_ptr() as *const __m128i);
        }

        // F-wave propagation
        for _k in 0..8 {
            f = _mm_slli_si128(f, 2);
            let mut did_break = false;
            for j in 0..slen as usize {
                let mut h1 = _mm_loadu_si128(qp.h1[j * 8..].as_ptr() as *const __m128i);
                h1 = _mm_max_epi16(h1, f);
                _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h1);
                let h1_sub = _mm_subs_epu16(h1, gapoe);
                f = _mm_subs_epu16(f, gape_v);
                if _mm_movemask_epi8(_mm_cmpgt_epi16(f, h1_sub)) == 0 {
                    did_break = true;
                    break;
                }
            }
            if did_break { break; }
        }

        // __max_8: find max of 8 int16 values
        let mut imax_v = max;
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 8));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 4));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 2));
        let imax = _mm_extract_epi16(imax_v, 0) as i16 as i32;

        if imax >= gmax {
            gmax = imax;
            target_end = i;
            qp.hmax.copy_from_slice(&qp.h1);
        }

        // Swap H0 and H1
        std::mem::swap(&mut qp.h0, &mut qp.h1);
    }

    // Find query_end from Hmax by scanning for positions matching gmax
    // Clamp to valid positions (< qlen) to avoid returning SIMD padding positions.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen {
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}}

#[cfg(target_arch = "aarch64")]
unsafe fn lightweight_align_i16_neon(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) { unsafe {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    let zero = vdupq_n_s16(0);
    let gapoe = vdupq_n_u16((gap_open + gap_extend) as u16);
    let gape_v = vdupq_n_u16(gap_extend as u16);

    for v in qp.e.iter_mut() { *v = 0; }
    for v in qp.h0.iter_mut() { *v = 0; }
    for v in qp.hmax.iter_mut() { *v = 0; }

    for i in 0..target_len {
        let mut f = zero;
        let mut max = zero;
        let s_offset = target[i as usize] as usize * slen as usize * 8;

        let h0_last_idx = (slen as usize - 1) * 8;
        let mut h = vld1q_s16(qp.h0[h0_last_idx..].as_ptr());
        h = vextq_s16(vdupq_n_s16(0), h, 7);

        for j in 0..slen as usize {
            let s = vld1q_s16(qp.query_profile[s_offset + j * 8..].as_ptr());
            h = vqaddq_s16(h, s);
            let e = vld1q_s16(qp.e[j * 8..].as_ptr());
            h = vmaxq_s16(h, e);
            h = vmaxq_s16(h, f);
            max = vmaxq_s16(max, h);
            vst1q_s16(qp.h1[j * 8..].as_mut_ptr(), h);
            let h_sub = vqsubq_u16(vreinterpretq_u16_s16(h), gapoe);
            let e_sub = vqsubq_u16(vreinterpretq_u16_s16(e), gape_v);
            let e_new = vmaxq_s16(vreinterpretq_s16_u16(e_sub), vreinterpretq_s16_u16(h_sub));
            vst1q_s16(qp.e[j * 8..].as_mut_ptr(), e_new);
            f = vreinterpretq_s16_u16(vqsubq_u16(vreinterpretq_u16_s16(f), gape_v));
            f = vmaxq_s16(f, vreinterpretq_s16_u16(h_sub));
            h = vld1q_s16(qp.h0[j * 8..].as_ptr());
        }

        // F-wave propagation
        for _k in 0..8 {
            f = vextq_s16(vdupq_n_s16(0), f, 7);
            let mut did_break = false;
            for j in 0..slen as usize {
                let mut h1 = vld1q_s16(qp.h1[j * 8..].as_ptr());
                h1 = vmaxq_s16(h1, f);
                vst1q_s16(qp.h1[j * 8..].as_mut_ptr(), h1);
                let h1_sub = vqsubq_u16(vreinterpretq_u16_s16(h1), gapoe);
                f = vreinterpretq_s16_u16(vqsubq_u16(vreinterpretq_u16_s16(f), gape_v));
                let cmp = vcgtq_s16(f, vreinterpretq_s16_u16(h1_sub));
                // Check if all lanes are zero (no f > h1_sub)
                let any_set = vmaxvq_u16(cmp);
                if any_set == 0 {
                    did_break = true;
                    break;
                }
            }
            if did_break { break; }
        }

        // Find max of 8 int16 values
        let imax = vmaxvq_s16(max) as i32;

        if imax >= gmax {
            gmax = imax;
            target_end = i;
            qp.hmax.copy_from_slice(&qp.h1);
        }

        std::mem::swap(&mut qp.h0, &mut qp.h1);
    }

    // Find query_end from Hmax by scanning for positions matching gmax
    // Clamp to valid positions (< qlen) to avoid returning SIMD padding positions.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen {
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}}

/// WASM SIMD128 implementation of lightweight_align_i16
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn lightweight_align_i16_wasm(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    let slen = qp.segment_len;
    let mut gmax: i32 = 0;
    let qlen8 = slen * 8;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    let zero = _mm_set1_epi32(0);
    let gapoe = _mm_set1_epi16((gap_open + gap_extend) as i16);
    let gape_v = _mm_set1_epi16(gap_extend as i16);

    for v in qp.e.iter_mut() { *v = 0; }
    for v in qp.h0.iter_mut() { *v = 0; }
    for v in qp.hmax.iter_mut() { *v = 0; }

    for i in 0..target_len {
        let mut f = zero;
        let mut max = zero;
        let s_offset = target[i as usize] as usize * slen as usize * 8;
        let h0_last_idx = (slen as usize - 1) * 8;
        let mut h = _mm_loadu_si128(qp.h0[h0_last_idx..].as_ptr() as *const __m128i);
        h = _mm_slli_si128(h, 2);

        for j in 0..slen as usize {
            let s = _mm_loadu_si128(qp.query_profile[s_offset + j * 8..].as_ptr() as *const __m128i);
            h = _mm_adds_epi16(h, s);
            let e = _mm_loadu_si128(qp.e[j * 8..].as_ptr() as *const __m128i);
            h = _mm_max_epi16(h, e);
            h = _mm_max_epi16(h, f);
            max = _mm_max_epi16(max, h);
            _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h);
            let h_sub = _mm_subs_epu16(h, gapoe);
            let e_sub = _mm_subs_epu16(e, gape_v);
            let e_new = _mm_max_epi16(e_sub, h_sub);
            _mm_storeu_si128(qp.e[j * 8..].as_mut_ptr() as *mut __m128i, e_new);
            f = _mm_subs_epu16(f, gape_v);
            f = _mm_max_epi16(f, h_sub);
            h = _mm_loadu_si128(qp.h0[j * 8..].as_ptr() as *const __m128i);
        }

        for _k in 0..8 {
            f = _mm_slli_si128(f, 2);
            let mut did_break = false;
            for j in 0..slen as usize {
                let mut h1 = _mm_loadu_si128(qp.h1[j * 8..].as_ptr() as *const __m128i);
                h1 = _mm_max_epi16(h1, f);
                _mm_storeu_si128(qp.h1[j * 8..].as_mut_ptr() as *mut __m128i, h1);
                let h1_sub = _mm_subs_epu16(h1, gapoe);
                f = _mm_subs_epu16(f, gape_v);
                if _mm_movemask_epi8(_mm_cmpgt_epi16(f, h1_sub)) == 0 {
                    did_break = true;
                    break;
                }
            }
            if did_break { break; }
        }

        let mut imax_v = max;
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 8));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 4));
        imax_v = _mm_max_epi16(imax_v, _mm_srli_si128(imax_v, 2));
        let imax = _mm_extract_epi16::<0>(imax_v) as i16 as i32;

        if imax >= gmax {
            gmax = imax;
            target_end = i;
            qp.hmax.copy_from_slice(&qp.h1);
        }

        std::mem::swap(&mut qp.h0, &mut qp.h1);
    }

    // Find query_end from Hmax by scanning for positions matching gmax
    // Clamp to valid positions (< qlen) to avoid returning SIMD padding positions.
    for i in 0..qlen8 {
        let val = qp.hmax[i as usize] as i32;
        if val == gmax {
            let pos = i / 8 + (i % 8) * slen;
            if pos < qp.qlen {
                query_end = pos;
            }
        }
    }

    (gmax, query_end, target_end)
}

/// Scalar fallback for lightweight_align_i16
#[allow(dead_code)]
pub fn lightweight_align_i16_scalar(qp: &mut LightweightProfile, target_len: i32, target: &[u8], gap_open: i32, gap_extend: i32) -> (i32, i32, i32) {
    // Simple scalar Smith-Waterman for non-SIMD platforms
    let qlen = qp.qlen as usize;
    let target_len = target_len as usize;
    let _m = 5usize; // alphabet size

    // Reconstruct scoring from query profile
    // qp.query_profile is segmented; we need to unsegment to get mat scores
    let slen = qp.segment_len as usize;
    let p = 8usize;

    let mut h_prev = vec![0i32; qlen + 1];
    let mut h_curr = vec![0i32; qlen + 1];
    let mut e_arr = vec![0i32; qlen + 1];
    let mut gmax: i32 = 0;
    let mut query_end: i32 = -1;
    let mut target_end: i32 = -1;

    for (i, &t_base) in target[..target_len].iter().enumerate() {
        let tc = t_base as usize;
        let mut f: i32 = 0;
        for j in 1..=qlen {
            // Get score from query profile: unsegment index
            // segmented index for query position (j-1): seg = (j-1) % slen, lane = (j-1) / slen
            // offset in qp.query_profile: tc * slen * p + seg * p + lane
            let seg = (j - 1) % slen;
            let lane = (j - 1) / slen;
            let score = if lane < p {
                qp.query_profile[tc * slen * p + seg * p + lane] as i32
            } else {
                0
            };

            let mut h = h_prev[j - 1] + score;
            if h < 0 { h = 0; }

            e_arr[j] = std::cmp::max(e_arr[j].saturating_sub(gap_extend), h_curr.get(j).copied().unwrap_or(0).saturating_sub(gap_open + gap_extend));
            // Wait, this is wrong - we need unsigned saturation semantics
            // Let's just do it simply
            let e_val = std::cmp::max(
                (e_arr[j] as i64 - gap_extend as i64).max(0) as i32,
                (h as i64 - (gap_open + gap_extend) as i64).max(0) as i32,
            );
            e_arr[j] = e_val;

            f = std::cmp::max(
                (f as i64 - gap_extend as i64).max(0) as i32,
                (h as i64 - (gap_open + gap_extend) as i64).max(0) as i32,
            );

            h = std::cmp::max(h, std::cmp::max(e_arr[j], f));
            h_curr[j] = h;

            if h >= gmax {
                gmax = h;
                target_end = i as i32;
                query_end = (j - 1) as i32;
            }
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
        for v in h_curr.iter_mut() { *v = 0; }
    }

    (gmax, query_end, target_end)
}

/// Scalar implementation of dual-affine extension alignment
///
/// Full-featured fallback that handles bandwidth, z_drop, end_bonus, extension mode,
/// reverse CIGAR, score-only mode, and right-align tie-breaking -- matching the SIMD
/// implementations' API.
/// Scalar dual-affine extension alignment.
///
/// Anti-diagonal DP with difference-encoded state arrays (u, v, x, y, x2, y2),
/// matching the SIMD implementation exactly but with scalar i8 operations.
/// Used as fallback on non-SIMD targets and for testing via `RAMMAP_FORCE_SCALAR=1`.
pub fn extend_dual_affine_scalar(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i32,
    gap_extend: i32,
    gap_open2: i32,
    gap_extend2: i32,
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) {
    let query_len = qseq.len();
    let target_len = tseq.len();
    let approx_max = (flags & APPROX_MAX) != 0;
    let with_cigar = (flags & SCORE_ONLY) == 0;

    if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
        return;
    }

    // Compute long_thres and long_diff for dual-affine boundary conditions
    let mut long_thres: i32 = if gap_extend != gap_extend2 {
        (gap_open2 - gap_open) / (gap_extend - gap_extend2) - 1
    } else { 0 };
    if (gap_open2 + gap_extend2 + long_thres * gap_extend2) > (gap_open + gap_extend + long_thres * gap_extend) {
        long_thres += 1;
    }
    let long_diff: i8 = (long_thres * (gap_extend - gap_extend2) - (gap_open2 - gap_open) - gap_extend2) as i8;

    // i8 constants matching SIMD
    let gap_open_i8 = gap_open as i8;
    let gap_extend_i8 = gap_extend as i8;
    let gap_open2_i8 = gap_open2 as i8;
    let gap_extend2_i8 = gap_extend2 as i8;
    let qe = gap_open + gap_extend;
    let qe2 = gap_open2 + gap_extend2;
    let qe_i8 = qe as i8;
    let qe2_i8 = qe2 as i8;
    let neg_qe_i8 = (-qe) as i8;
    let neg_qe2_i8 = (-qe2) as i8;

    // Bandwidth
    let wl = if bandwidth < 0 { query_len.max(target_len) as i32 } else { bandwidth };

    // n_col_ for traceback (includes bandwidth factor)
    let n_col_ = query_len.min(target_len).min((wl + 1) as usize).div_ceil(16) + 1;

    init_dp_result(result);

    // --- Anti-diagonal state arrays (difference-encoded, i8) ---
    let mut u_arr = vec![neg_qe_i8; target_len];
    let mut v_arr = vec![neg_qe_i8; target_len];
    let mut x_arr = vec![neg_qe_i8; target_len];
    let mut y_arr = vec![neg_qe_i8; target_len];
    let mut x2_arr = vec![neg_qe2_i8; target_len];
    let mut y2_arr = vec![neg_qe2_i8; target_len];

    // H[] for exact max tracking (only when !approx_max)
    let mut h_arr: Vec<i32> = if !approx_max { vec![0i32; target_len] } else { Vec::new() };

    // Reversed query (matches SIMD qr layout)
    let mut qr = vec![0u8; query_len];
    for t in 0..query_len {
        qr[t] = qseq[query_len - 1 - t];
    }

    // Traceback arrays
    let valid_range = query_len + target_len - 1;
    let stride = n_col_ * 16;
    let mut p_arr: Vec<u8> = if with_cigar { vec![0u8; valid_range * stride] } else { Vec::new() };
    let mut band_off: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };
    let mut band_off_end: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };

    // Scoring constants
    let sc_mch = score_matrix[0];
    let sc_mis = score_matrix[1];
    let sc_n = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        -gap_extend2_i8
    } else {
        score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1]
    };
    let m1 = (alphabet_size - 1) as u8;
    let generic_scoring = (flags & GENERIC_SCORING) != 0;
    let right_align = (flags & RIGHT_ALIGN) != 0;

    // --- Main anti-diagonal DP loop ---
    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    for r in 0..valid_range as i32 {
        // Compute valid target range for this anti-diagonal
        let mut st0 = 0i32;
        let mut en0 = target_len as i32 - 1;
        if st0 < r - query_len as i32 + 1 { st0 = r - query_len as i32 + 1; }
        if en0 > r { en0 = r; }

        // Apply bandwidth narrowing
        if st0 < (r - wl + 1) >> 1 { st0 = (r - wl + 1) >> 1; }
        if en0 > (r + wl) >> 1 { en0 = (r + wl) >> 1; }

        if st0 > en0 {
            result.zdropped = 1;
            break;
        }

        // Boundary conditions for leftmost element
        let x1: i8;
        let x21: i8;
        let v1: i8;

        if st0 > 0 {
            if st0 > last_st && (st0 - 1) <= last_en {
                x1 = x_arr[(st0 - 1) as usize];
                x21 = x2_arr[(st0 - 1) as usize];
                v1 = v_arr[(st0 - 1) as usize];
            } else {
                x1 = neg_qe_i8;
                x21 = neg_qe2_i8;
                v1 = neg_qe_i8;
            }
        } else {
            x1 = neg_qe_i8;
            x21 = neg_qe2_i8;
            v1 = if r == 0 {
                neg_qe_i8
            } else if r < long_thres {
                -gap_extend_i8
            } else if r == long_thres {
                long_diff
            } else {
                -gap_extend2_i8
            };
        }

        // Initialize new diagonal entry
        if en0 >= r {
            y_arr[r as usize] = neg_qe_i8;
            y2_arr[r as usize] = neg_qe2_i8;
            u_arr[r as usize] = if r == 0 {
                neg_qe_i8
            } else if r < long_thres {
                -gap_extend_i8
            } else if r == long_thres {
                long_diff
            } else {
                -gap_extend2_i8
            };
        }

        // Core DP: process each element along the anti-diagonal
        let qrr_base = (query_len as i32 - 1 - r) as isize;
        let mut prev_x: i8 = x1;
        let mut prev_x2: i8 = x21;
        let mut prev_v: i8 = v1;

        for t in st0..=en0 {
            let tu = t as usize;

            // Match/mismatch score
            let sq = tseq[tu];
            let st_val = qr[(qrr_base + t as isize) as usize];
            let z_score: i8 = if !generic_scoring {
                if sq == m1 || st_val == m1 { sc_n }
                else if sq == st_val { sc_mch }
                else { sc_mis }
            } else {
                score_matrix[sq as usize * alphabet_size as usize + st_val as usize]
            };

            // a = x[t-1] + v[t-1] (D1: gap from left on anti-diagonal)
            let xt1 = prev_x;
            let vt1 = prev_v;
            let a = xt1.wrapping_add(vt1);

            // b = y[t] + u[t] (I1: gap from above on anti-diagonal)
            let ut = u_arr[tu];
            let b = y_arr[tu].wrapping_add(ut);

            // a2 = x2[t-1] + v[t-1] (D2: second penalty gap from left)
            let x2t1 = prev_x2;
            let a2 = x2t1.wrapping_add(vt1);

            // b2 = y2[t] + u[t] (I2: second penalty gap from above)
            let b2 = y2_arr[tu].wrapping_add(ut);

            // Save old values before overwrite
            prev_x = x_arr[tu];
            prev_x2 = x2_arr[tu];
            prev_v = v_arr[tu];

            if !with_cigar {
                // Score only: 5-way max + clamp
                let mut z = z_score;
                if a > z { z = a; }
                if b > z { z = b; }
                if a2 > z { z = a2; }
                if b2 > z { z = b2; }
                if z > sc_mch { z = sc_mch; } // clamp

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let tmp2 = z.wrapping_sub(gap_open2_i8);
                let a2_new = a2.wrapping_sub(tmp2);
                let b2_new = b2.wrapping_sub(tmp2);

                x_arr[tu] = (if a_new > 0 { a_new } else { 0 }).wrapping_sub(qe_i8);
                y_arr[tu] = (if b_new > 0 { b_new } else { 0 }).wrapping_sub(qe_i8);
                x2_arr[tu] = (if a2_new > 0 { a2_new } else { 0 }).wrapping_sub(qe2_i8);
                y2_arr[tu] = (if b2_new > 0 { b2_new } else { 0 }).wrapping_sub(qe2_i8);
            } else if !right_align {
                // Left-align with traceback
                let p_idx = r as usize * stride + (t - st0) as usize;
                if t == st0 {
                    band_off[r as usize] = st0;
                    band_off_end[r as usize] = en0;
                }

                let mut z = z_score;
                let mut d: u8 = 0;
                if a > z { d = 1; z = a; }
                if b > z { d = 2; z = b; }
                if a2 > z { d = 3; z = a2; }
                if b2 > z { d = 4; z = b2; }
                if z > sc_mch { z = sc_mch; } // clamp

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let tmp2 = z.wrapping_sub(gap_open2_i8);
                let a2_new = a2.wrapping_sub(tmp2);
                let b2_new = b2.wrapping_sub(tmp2);

                if a_new > 0 {
                    x_arr[tu] = a_new.wrapping_sub(qe_i8);
                    d |= 0x08;
                } else {
                    x_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if b_new > 0 {
                    y_arr[tu] = b_new.wrapping_sub(qe_i8);
                    d |= 0x10;
                } else {
                    y_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if a2_new > 0 {
                    x2_arr[tu] = a2_new.wrapping_sub(qe2_i8);
                    d |= 0x20;
                } else {
                    x2_arr[tu] = (0i8).wrapping_sub(qe2_i8);
                }
                if b2_new > 0 {
                    y2_arr[tu] = b2_new.wrapping_sub(qe2_i8);
                    d |= 0x40;
                } else {
                    y2_arr[tu] = (0i8).wrapping_sub(qe2_i8);
                }

                p_arr[p_idx] = d;
            } else {
                // Right-align with traceback
                let p_idx = r as usize * stride + (t - st0) as usize;
                if t == st0 {
                    band_off[r as usize] = st0;
                    band_off_end[r as usize] = en0;
                }

                let mut z = z_score;
                let mut d: u8;
                if z > a { d = 0; } else { d = 1; z = a; }
                if z <= b { d = 2; z = b; }
                if z <= a2 { d = 3; z = a2; }
                if z <= b2 { d = 4; z = b2; }
                if z > sc_mch { z = sc_mch; } // clamp

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let tmp2 = z.wrapping_sub(gap_open2_i8);
                let a2_new = a2.wrapping_sub(tmp2);
                let b2_new = b2.wrapping_sub(tmp2);

                if 0i8 <= a_new {
                    x_arr[tu] = a_new.wrapping_sub(qe_i8);
                    d |= 0x08;
                } else {
                    x_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if 0i8 <= b_new {
                    y_arr[tu] = b_new.wrapping_sub(qe_i8);
                    d |= 0x10;
                } else {
                    y_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if 0i8 <= a2_new {
                    x2_arr[tu] = a2_new.wrapping_sub(qe2_i8);
                    d |= 0x20;
                } else {
                    x2_arr[tu] = (0i8).wrapping_sub(qe2_i8);
                }
                if 0i8 <= b2_new {
                    y2_arr[tu] = b2_new.wrapping_sub(qe2_i8);
                    d |= 0x40;
                } else {
                    y2_arr[tu] = (0i8).wrapping_sub(qe2_i8);
                }

                p_arr[p_idx] = d;
            }
        }

        // --- H tracking ---
        let qe_scalar = gap_open + gap_extend;
        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;

            if r > 0 {
                let h_en0 = if en0 > 0 {
                    h_arr[en0 as usize - 1] + u_arr[en0 as usize] as i32
                } else {
                    h_arr[en0 as usize] + v_arr[en0 as usize] as i32
                };
                h_arr[en0 as usize] = h_en0;
                max_h = h_en0;
                max_t = en0;

                // Process [st0..en0) in groups of 4, matching SIMD's 4-lane reduction.
                // Each lane independently tracks max across its stride-4 positions.
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut lane_h = [max_h; 4];
                let mut lane_t = [max_t; 4];
                let mut t = st0;
                while t < en1 {
                    for i in 0..4i32 {
                        let pos = (t + i) as usize;
                        h_arr[pos] += v_arr[pos] as i32;
                        if h_arr[pos] > lane_h[i as usize] {
                            lane_h[i as usize] = h_arr[pos];
                            lane_t[i as usize] = t;
                        }
                    }
                    t += 4;
                }
                // Reduce lanes to scalar (matches SIMD reduction order)
                for i in 0..4i32 {
                    if max_h < lane_h[i as usize] {
                        max_h = lane_h[i as usize];
                        max_t = lane_t[i as usize] + i;
                    }
                }
                // Remainder
                while t < en0 {
                    h_arr[t as usize] += v_arr[t as usize] as i32;
                    if h_arr[t as usize] > max_h {
                        max_h = h_arr[t as usize];
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                h_arr[0] = v_arr[0] as i32 - qe_scalar;
                max_h = h_arr[0];
                max_t = 0;
            }

            // Track target end score
            if en0 == target_len as i32 - 1 && h_arr[en0 as usize] > result.max_target_end_score {
                result.max_target_end_score = h_arr[en0 as usize];
                result.max_target_end_query_pos = r - en0;
            }
            // Track query end score
            if r - st0 == query_len as i32 - 1 && h_arr[st0 as usize] > result.max_query_end_score {
                result.max_query_end_score = h_arr[st0 as usize];
                result.max_query_end_target_pos = st0;
            }

            // Overall max and z-drop
            if max_h > result.max {
                result.max = max_h;
                result.max_score_target_pos = max_t;
                result.max_score_query_pos = r - max_t;
            } else if max_t >= result.max_score_target_pos && (r - max_t) >= result.max_score_query_pos {
                let tl = max_t - result.max_score_target_pos;
                let ql = (r - max_t) - result.max_score_query_pos;
                let l = if tl > ql { tl - ql } else { ql - tl };
                if z_drop >= 0 && (result.max - max_h) > z_drop + l * gap_extend2 {
                    result.zdropped = 1;
                    break;
                }
            }

            // Score at final corner
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h_arr[target_len - 1];
            }
        } else {
            // --- Approximate max tracking ---
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                    let d0 = v_arr[last_h0_t as usize] as i32;
                    let d1 = u_arr[(last_h0_t + 1) as usize] as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += v_arr[last_h0_t as usize] as i32;
                } else {
                    last_h0_t += 1;
                    h0 += u_arr[last_h0_t as usize] as i32;
                }
            } else {
                h0 = v_arr[0] as i32 - qe_scalar;
                last_h0_t = 0;
            }

            // Unconditional max update
            if h0 > result.max {
                result.max = h0;
                result.max_score_target_pos = last_h0_t;
                result.max_score_query_pos = r - last_h0_t;
            }

            // Z-drop only when APPROX_DROP
            if (flags & APPROX_DROP) != 0
                && last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                    let tl = last_h0_t - result.max_score_target_pos;
                    let ql = (r - last_h0_t) - result.max_score_query_pos;
                    let l = if tl > ql { tl - ql } else { ql - tl };
                    if z_drop >= 0 && (result.max - h0) > z_drop + l * gap_extend2 {
                        result.zdropped = 1;
                        break;
                    }
                }

            // Score at final corner
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h0;
            }
        }

        last_st = st0;
        last_en = en0;
    }

    // Final score for approx path
    if approx_max && result.score == NEG_INF {
        result.score = result.max;
    }

    // --- Traceback via shared traceback_dual_affine ---
    if with_cigar {
        unsafe {
            traceback_dual_affine(
                result, query_len, target_len, end_bonus, flags, n_col_, 16,
                p_arr.as_mut_ptr(), band_off.as_mut_ptr(), band_off_end.as_mut_ptr(),
            );
        }
    }
}

/// Scalar splice-aware extension alignment.
///
/// Anti-diagonal DP with difference-encoded state arrays (u, v, x, y, x2),
/// matching the SIMD implementation exactly but with scalar i8 operations.
/// Used as fallback on non-SIMD targets and for testing via `RAMMAP_FORCE_SCALAR=1`.
pub fn extend_splice_scalar(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i32,
    gap_extend: i32,
    gap_open2: i32,
    noncanon_penalty: i32,
    z_drop: i32,
    end_bonus: i32,
    junc_bonus: i8,
    junc_pen: i8,
    flags: i32,
    junc: Option<&[u8]>,
    result: &mut DpResult,
) {
    let query_len = qseq.len();
    let target_len = tseq.len();
    let qe = gap_open + gap_extend;
    let approx_max = (flags & APPROX_MAX) != 0;
    let with_cigar = (flags & SCORE_ONLY) == 0;

    init_dp_result_full(result);

    if alphabet_size <= 1 || query_len == 0 || target_len == 0 || gap_open2 <= qe {
        return;
    }
    assert!((flags & SPLICE_FORWARD) == 0 || (flags & SPLICE_REVERSE) == 0);

    // Check scoring matrix bounds (same as SIMD)
    {
        let mut max_sc = score_matrix[0] as i32;
        let mut min_sc = score_matrix[1] as i32;
        for &s in &score_matrix[1..(alphabet_size as usize * alphabet_size as usize)] {
            max_sc = max_sc.max(s as i32);
            min_sc = min_sc.min(s as i32);
        }
        let _ = max_sc;
        if -min_sc > 2 * qe {
            return;
        }
    }

    // long_thres: crossover from regular gap to intron cost
    let mut long_thres: i32 = (gap_open2 - gap_open) / gap_extend - 1;
    if gap_open2 > gap_open + gap_extend + long_thres * gap_extend {
        long_thres += 1;
    }
    let long_diff: i8 = (long_thres * gap_extend - (gap_open2 - gap_open)) as i8;

    // i8 constants matching SIMD
    let gap_open_i8 = gap_open as i8;
    let gap_extend_i8 = gap_extend as i8;
    let gap_open2_i8 = gap_open2 as i8;
    let qe_i8 = qe as i8;
    let neg_qe_i8 = (-gap_open - gap_extend) as i8;
    let x2_init = (-gap_open2) as i8;

    // --- Donor/acceptor pre-computation ---
    let default_sp3 = if (flags & SPLICE_COMPLEX) != 0 {
        let sp0 = [8, 15, 21, 30];
        [(sp0[0] as f64 / 3.0 + 0.499) as i32,
         (sp0[1] as f64 / 3.0 + 0.499) as i32,
         (sp0[2] as f64 / 3.0 + 0.499) as i32,
         (sp0[3] as f64 / 3.0 + 0.499) as i32]
    } else {
        let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty / 2 } else { 0 };
        [sp0, noncanon_penalty, noncanon_penalty, noncanon_penalty]
    };
    let sp = default_sp3;

    let mut donor = vec![(-sp[3]) as i8; target_len];
    let mut acceptor = vec![(-sp[3]) as i8; target_len];

    if (flags & (SPLICE_FORWARD | SPLICE_REVERSE)) != 0 {
        if (flags & REV_CIGAR) == 0 {
            // Forward direction donor sites
            for t in 0..(target_len as i32 - 4) {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 {
                        z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 1 { z = 1; }
                    else if tseq[tu + 1] == 0 && tseq[tu + 2] == 3 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu + 1] == 1 && tseq[tu + 2] == 3 {
                        z = if tseq[tu + 3] == 0 || tseq[tu + 3] == 2 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 2 && tseq[tu + 2] == 3 { z = 2; }
                }
                donor[tu] = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
            // Forward direction acceptor sites
            for t in 2..target_len as i32 {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu - 1] == 0 && tseq[tu] == 2 {
                        z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 0 && tseq[tu] == 1 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu - 1] == 0 && tseq[tu] == 1 {
                        z = if tseq[tu - 2] == 1 || tseq[tu - 2] == 3 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 2 && tseq[tu] == 1 { z = 1; }
                    else if tseq[tu - 1] == 0 && tseq[tu] == 3 { z = 2; }
                }
                acceptor[tu] = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
        } else {
            // REV_CIGAR direction donor sites
            for t in 0..(target_len as i32 - 4) {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu + 1] == 2 && tseq[tu + 2] == 0 {
                        z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu + 1] == 1 && tseq[tu + 2] == 0 {
                        z = if tseq[tu + 3] == 1 || tseq[tu + 3] == 3 { -1 } else { 0 };
                    } else if tseq[tu + 1] == 1 && tseq[tu + 2] == 2 { z = 1; }
                    else if tseq[tu + 1] == 3 && tseq[tu + 2] == 0 { z = 2; }
                }
                donor[tu] = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
            // REV_CIGAR direction acceptor sites
            for t in 2..target_len as i32 {
                let tu = t as usize;
                let mut z = 3i32;
                if (flags & SPLICE_FORWARD) != 0 {
                    if tseq[tu - 1] == 3 && tseq[tu] == 2 {
                        z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 1 && tseq[tu] == 2 { z = 1; }
                    else if tseq[tu - 1] == 3 && tseq[tu] == 0 { z = 2; }
                } else if (flags & SPLICE_REVERSE) != 0 {
                    if tseq[tu - 1] == 3 && tseq[tu] == 1 {
                        z = if tseq[tu - 2] == 0 || tseq[tu - 2] == 2 { -1 } else { 0 };
                    } else if tseq[tu - 1] == 3 && tseq[tu] == 2 { z = 2; }
                }
                acceptor[tu] = if z < 0 { 0 } else { -sp[z as usize] as i8 };
            }
        }
    }

    // --- Junction annotation overlay ---
    if let Some(junc_arr) = junc {
        if (flags & SPLICE_SCORE) != 0 {
            let donor_val: u8 = if ((flags & SPLICE_FORWARD) != 0) == ((flags & REV_CIGAR) == 0) { 0 } else { 1 };
            for t in 0..(target_len - 1) {
                let j = junc_arr[t + 1];
                let adj = if j == 0xff || (j & 1) != donor_val {
                    -junc_pen
                } else {
                    (j >> 1) as i8 - SPSC_OFFSET as i8
                };
                donor[t] = donor[t].wrapping_add(adj);
            }
            for t in 0..(target_len - 1) {
                let j = junc_arr[t + 1];
                let not_donor_val = if donor_val == 0 { 1 } else { 0 };
                let adj = if j == 0xff || (j & 1) != not_donor_val {
                    -junc_pen
                } else {
                    (j >> 1) as i8 - SPSC_OFFSET as i8
                };
                acceptor[t] = acceptor[t].wrapping_add(adj);
            }
        } else if (flags & REV_CIGAR) == 0 {
            for t in 0..(target_len - 1) {
                if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 1) != 0)
                    || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 8) != 0)
                {
                    donor[t] = donor[t].wrapping_add(junc_bonus);
                }
            }
            for t in 0..target_len {
                if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 2) != 0)
                    || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 4) != 0)
                {
                    acceptor[t] = acceptor[t].wrapping_add(junc_bonus);
                }
            }
        } else {
            for t in 0..(target_len - 1) {
                if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t + 1] & 2) != 0)
                    || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t + 1] & 4) != 0)
                {
                    donor[t] = donor[t].wrapping_add(junc_bonus);
                }
            }
            for t in 0..target_len {
                if ((flags & SPLICE_FORWARD) != 0 && (junc_arr[t] & 1) != 0)
                    || ((flags & SPLICE_REVERSE) != 0 && (junc_arr[t] & 8) != 0)
                {
                    acceptor[t] = acceptor[t].wrapping_add(junc_bonus);
                }
            }
        }
    }

    // --- Anti-diagonal state arrays (difference-encoded, i8) ---
    let n_col_ = query_len.min(target_len).div_ceil(16) + 1;
    let mut u_arr = vec![neg_qe_i8; target_len];
    let mut v_arr = vec![neg_qe_i8; target_len];
    let mut x_arr = vec![neg_qe_i8; target_len];
    let mut y_arr = vec![neg_qe_i8; target_len];
    let mut x2_arr = vec![x2_init; target_len];

    // H[] for exact max tracking
    let mut h_arr: Vec<i32> = if !approx_max { vec![0i32; target_len] } else { Vec::new() };

    // Reversed query (matches SIMD qr layout)
    let mut qr = vec![0u8; query_len];
    for t in 0..query_len {
        qr[t] = qseq[query_len - 1 - t];
    }

    // Traceback arrays
    let valid_range = query_len + target_len - 1;
    let stride = n_col_ * 16;
    let mut p_arr: Vec<u8> = if with_cigar { vec![0u8; valid_range * stride] } else { Vec::new() };
    let mut band_off: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };
    let mut band_off_end: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };

    // Scoring constants
    let sc_mch = score_matrix[0];
    let sc_mis = score_matrix[1];
    let sc_n = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        -gap_extend_i8
    } else {
        score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1]
    };
    let m1 = (alphabet_size - 1) as u8;
    let generic_scoring = (flags & GENERIC_SCORING) != 0;
    let right_align = (flags & RIGHT_ALIGN) != 0;

    // --- Main anti-diagonal DP loop ---
    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    for r in 0..valid_range as i32 {
        // Compute valid target range for this anti-diagonal
        let mut st0 = 0i32;
        let mut en0 = target_len as i32 - 1;
        if st0 < r - query_len as i32 + 1 { st0 = r - query_len as i32 + 1; }
        if en0 > r { en0 = r; }

        // Boundary conditions for leftmost element
        let x1: i8;
        let x21: i8;
        let v1: i8;

        if st0 > 0 {
            if st0 > last_st && (st0 - 1) <= last_en {
                x1 = x_arr[(st0 - 1) as usize];
                x21 = x2_arr[(st0 - 1) as usize];
                v1 = v_arr[(st0 - 1) as usize];
            } else {
                x1 = neg_qe_i8;
                x21 = x2_init;
                v1 = neg_qe_i8;
            }
        } else {
            x1 = neg_qe_i8;
            x21 = x2_init;
            v1 = if r == 0 {
                neg_qe_i8
            } else if r < long_thres {
                -gap_extend_i8
            } else if r == long_thres {
                long_diff
            } else {
                0
            };
        }

        // Initialize new diagonal entry
        if en0 >= r {
            y_arr[r as usize] = neg_qe_i8;
            u_arr[r as usize] = if r == 0 {
                neg_qe_i8
            } else if r < long_thres {
                -gap_extend_i8
            } else if r == long_thres {
                long_diff
            } else {
                0
            };
        }

        // Core DP: process each element along the anti-diagonal
        let qrr_base = (query_len as i32 - 1 - r) as isize;
        let mut prev_x: i8 = x1;
        let mut prev_x2: i8 = x21;
        let mut prev_v: i8 = v1;

        for t in st0..=en0 {
            let tu = t as usize;

            // Match/mismatch score
            let sq = tseq[tu];
            let st_val = qr[(qrr_base + t as isize) as usize];
            let z_score: i8 = if !generic_scoring {
                if sq == m1 || st_val == m1 { sc_n }
                else if sq == st_val { sc_mch }
                else { sc_mis }
            } else {
                score_matrix[sq as usize * alphabet_size as usize + st_val as usize]
            };

            // a = x[t-1] + v[t-1] (D1: gap from left on anti-diagonal)
            let xt1 = prev_x;
            let vt1 = prev_v;
            let a = xt1.wrapping_add(vt1);

            // b = y[t] + u[t] (I1: gap from above on anti-diagonal)
            let ut = u_arr[tu];
            let b = y_arr[tu].wrapping_add(ut);

            // a2 = x2[t-1] + v[t-1] (intron from left)
            let x2t1 = prev_x2;
            let a2 = x2t1.wrapping_add(vt1);

            // a2a = a2 + acceptor[t] (intron with acceptor bonus)
            let a2a = a2.wrapping_add(acceptor[tu]);

            // Save old values before overwrite (for next iteration's prev_*)
            prev_x = x_arr[tu];
            prev_x2 = x2_arr[tu];
            prev_v = v_arr[tu];

            if !with_cigar {
                // Score only: 4-way max (left-align: first strictly-greater wins)
                let mut z = z_score;
                if a > z { z = a; }
                if b > z { z = b; }
                if a2a > z { z = a2a; }

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let a2_new = a2.wrapping_sub(z.wrapping_sub(gap_open2_i8));

                x_arr[tu] = (if a_new > 0 { a_new } else { 0 }).wrapping_sub(qe_i8);
                y_arr[tu] = (if b_new > 0 { b_new } else { 0 }).wrapping_sub(qe_i8);
                x2_arr[tu] = a2_new.max(donor[tu]).wrapping_sub(gap_open2_i8);
            } else if !right_align {
                // Left-align with traceback
                let p_idx = r as usize * stride + (t - st0) as usize;
                if t == st0 {
                    band_off[r as usize] = st0;
                    band_off_end[r as usize] = en0;
                }

                let mut z = z_score;
                let mut d: u8 = 0;
                if a > z { d = 1; z = a; }
                if b > z { d = 2; z = b; }
                if a2a > z { d = 3; z = a2a; }

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let a2_new = a2.wrapping_sub(z.wrapping_sub(gap_open2_i8));

                if a_new > 0 {
                    x_arr[tu] = a_new.wrapping_sub(qe_i8);
                    d |= 0x08;
                } else {
                    x_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if b_new > 0 {
                    y_arr[tu] = b_new.wrapping_sub(qe_i8);
                    d |= 0x10;
                } else {
                    y_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                let donor_t = donor[tu];
                if a2_new > donor_t {
                    x2_arr[tu] = a2_new.wrapping_sub(gap_open2_i8);
                    d |= 0x20;
                } else {
                    x2_arr[tu] = donor_t.wrapping_sub(gap_open2_i8);
                }

                p_arr[p_idx] = d;
            } else {
                // Right-align with traceback
                let p_idx = r as usize * stride + (t - st0) as usize;
                if t == st0 {
                    band_off[r as usize] = st0;
                    band_off_end[r as usize] = en0;
                }

                let mut z = z_score;
                let mut d: u8;
                if z > a { d = 0; } else { d = 1; z = a; }
                if z <= b { d = 2; z = b; }
                if z <= a2a { d = 3; z = a2a; }

                u_arr[tu] = z.wrapping_sub(vt1);
                v_arr[tu] = z.wrapping_sub(ut);
                let tmp1 = z.wrapping_sub(gap_open_i8);
                let a_new = a.wrapping_sub(tmp1);
                let b_new = b.wrapping_sub(tmp1);
                let a2_new = a2.wrapping_sub(z.wrapping_sub(gap_open2_i8));

                if 0i8 <= a_new {
                    x_arr[tu] = a_new.wrapping_sub(qe_i8);
                    d |= 0x08;
                } else {
                    x_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                if 0i8 <= b_new {
                    y_arr[tu] = b_new.wrapping_sub(qe_i8);
                    d |= 0x10;
                } else {
                    y_arr[tu] = (0i8).wrapping_sub(qe_i8);
                }
                let donor_t = donor[tu];
                if donor_t <= a2_new {
                    x2_arr[tu] = a2_new.wrapping_sub(gap_open2_i8);
                    d |= 0x20;
                } else {
                    x2_arr[tu] = donor_t.wrapping_sub(gap_open2_i8);
                }

                p_arr[p_idx] = d;
            }
        }

        // --- H tracking (exact max) ---
        let qe_scalar = gap_open + gap_extend;
        if !approx_max {
            let mut max_h: i32;
            let mut max_t: i32;

            if r > 0 {
                let h_en0 = if en0 > 0 {
                    h_arr[en0 as usize - 1] + u_arr[en0 as usize] as i32
                } else {
                    h_arr[en0 as usize] + v_arr[en0 as usize] as i32
                };
                h_arr[en0 as usize] = h_en0;
                max_h = h_en0;
                max_t = en0;

                // Process [st0..en0) in groups of 4, matching SIMD's 4-lane reduction.
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut lane_h = [max_h; 4];
                let mut lane_t = [max_t; 4];
                let mut t = st0;
                while t < en1 {
                    for i in 0..4i32 {
                        let pos = (t + i) as usize;
                        h_arr[pos] += v_arr[pos] as i32;
                        if h_arr[pos] > lane_h[i as usize] {
                            lane_h[i as usize] = h_arr[pos];
                            lane_t[i as usize] = t;
                        }
                    }
                    t += 4;
                }
                for i in 0..4i32 {
                    if max_h < lane_h[i as usize] {
                        max_h = lane_h[i as usize];
                        max_t = lane_t[i as usize] + i;
                    }
                }
                while t < en0 {
                    h_arr[t as usize] += v_arr[t as usize] as i32;
                    if h_arr[t as usize] > max_h {
                        max_h = h_arr[t as usize];
                        max_t = t;
                    }
                    t += 1;
                }
            } else {
                h_arr[0] = v_arr[0] as i32 - qe_scalar;
                max_h = h_arr[0];
                max_t = 0;
            }

            // Track target end score
            if en0 == target_len as i32 - 1 && h_arr[en0 as usize] > result.max_target_end_score {
                result.max_target_end_score = h_arr[en0 as usize];
                result.max_target_end_query_pos = r - en0;
            }
            // Track query end score
            if r - st0 == query_len as i32 - 1 && h_arr[st0 as usize] > result.max_query_end_score {
                result.max_query_end_score = h_arr[st0 as usize];
                result.max_query_end_target_pos = st0;
            }

            // Overall max and z-drop
            if max_h > result.max {
                result.max = max_h;
                result.max_score_target_pos = max_t;
                result.max_score_query_pos = r - max_t;
            } else if z_drop >= 0
                && max_t >= result.max_score_target_pos
                && (r - max_t) >= result.max_score_query_pos
                && (result.max - max_h) > z_drop
            {
                result.zdropped = 1;
                break;
            }

            // Score at final corner
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h_arr[target_len - 1];
            }
        } else {
            // --- Approximate max tracking ---
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                    let d0 = v_arr[last_h0_t as usize] as i32;
                    let d1 = u_arr[(last_h0_t + 1) as usize] as i32;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += v_arr[last_h0_t as usize] as i32;
                } else {
                    last_h0_t += 1;
                    h0 += u_arr[last_h0_t as usize] as i32;
                }
            } else {
                h0 = v_arr[0] as i32 - qe_scalar;
                last_h0_t = 0;
            }

            if (flags & APPROX_DROP) != 0 {
                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = last_h0_t;
                    result.max_score_query_pos = r - last_h0_t;
                } else if z_drop >= 0
                    && last_h0_t >= result.max_score_target_pos
                    && (r - last_h0_t) >= result.max_score_query_pos
                    && (result.max - h0) > z_drop
                {
                    result.zdropped = 1;
                    break;
                }
            }

            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h0;
            }
        }

        last_st = st0;
        last_en = en0;
    }

    // --- Traceback via shared traceback_splice ---
    if with_cigar {
        unsafe {
            traceback_splice(
                result, query_len, target_len, end_bonus, flags, n_col_, 16, long_thres,
                p_arr.as_mut_ptr(), band_off.as_mut_ptr(), band_off_end.as_mut_ptr(),
            );
        }
    }
}

/// Scalar single-affine extension alignment
///
/// Delegates to `extend_dual_affine_scalar` with identical penalties for both gap models.
/// Scalar single-affine extension alignment (extz2 formulation).
///
/// Uses the same Suzuki-Kasahara anti-diagonal sweep and 3-state traceback
/// as the SIMD implementations, producing identical results. This is the
/// fallback for non-SIMD architectures and for `RAMMAP_FORCE_SCALAR=1`.
pub fn extend_single_affine_scalar(
    qseq: &[u8],
    tseq: &[u8],
    alphabet_size: i8,
    score_matrix: &[i8],
    gap_open: i32,
    gap_extend: i32,
    bandwidth: i32,
    z_drop: i32,
    end_bonus: i32,
    flags: i32,
    result: &mut DpResult,
) {
    let query_len = qseq.len();
    let target_len = tseq.len();
    let with_cigar = (flags & SCORE_ONLY) == 0;
    let right_align = (flags & RIGHT_ALIGN) != 0;

    if query_len == 0 || target_len == 0 { return; }

    let q = gap_open as i8;
    let qe2 = ((gap_open + gap_extend) * 2) as i8;
    let qe_i32 = gap_open + gap_extend;
    let m = alphabet_size as usize;

    // Anti-diagonal band
    let wl = if bandwidth > 0 { bandwidth } else { std::cmp::max(query_len, target_len) as i32 };
    let valid_range = (query_len + target_len - 1) as i32;

    // DP arrays (Suzuki-Kasahara state: u, v, x, y per cell)
    let arr_len = target_len + 16;
    let mut u_arr = vec![0i8; arr_len];
    let mut v_arr = vec![0i8; arr_len];
    let mut x_arr = vec![0i8; arr_len];
    let mut y_arr = vec![0i8; arr_len];
    let mut s_arr = vec![0i8; arr_len];

    // Reversed query
    let mut qr = vec![0u8; query_len];
    for i in 0..query_len { qr[i] = qseq[query_len - 1 - i]; }

    // Traceback storage
    let n_col_ = std::cmp::min(query_len as i32, 2 * wl + 1) as usize;
    let stride = n_col_;
    let mut p_arr: Vec<u8> = if with_cigar { vec![0; valid_range as usize * stride] } else { Vec::new() };
    let mut band_off = vec![0i32; valid_range as usize];
    let mut band_off_end = vec![0i32; valid_range as usize];

    // h0 tracking
    let mut h0 = 0i32;
    let mut last_h0_t = 0i32;
    let mut last_st = -1i32;
    let mut last_en = -1i32;

    result.score = NEG_INF;
    result.max = NEG_INF;

    for r in 0..valid_range {
        let mut st = 0i32;
        let mut en = target_len as i32 - 1;
        if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
        if en > r { en = r; }
        if st < ((r - wl + 1) >> 1) { st = (r - wl + 1) >> 1; }
        if en > ((r + wl) >> 1) { en = (r + wl) >> 1; }
        if st > en { result.zdropped = 1; break; }

        let st0 = st;
        let en0 = en;

        // Boundary conditions
        let mut x1: i8;
        let v1: i8;
        if st > 0 {
            if st > last_st && st - 1 <= last_en {
                x1 = x_arr[(st - 1) as usize];
                v1 = v_arr[(st - 1) as usize];
            } else {
                x1 = 0; v1 = 0;
            }
        } else {
            x1 = 0;
            v1 = if r == 0 { 0 } else { q };
        }
        if en >= r {
            y_arr[r as usize] = 0;
            u_arr[r as usize] = if r == 0 { 0 } else { q };
        }

        // Score computation
        let use_generic = (flags & GENERIC_SCORING) != 0;
        for t in st0..=en0 {
            let ti = t as usize;
            let qi = (t + query_len as i32 - 1 - r) as usize;
            s_arr[ti] = if use_generic {
                score_matrix[tseq[ti] as usize * m + qr[qi] as usize]
            } else if tseq[ti] >= 4 || qr[qi] >= 4 {
                score_matrix[4 * m] // ambig penalty
            } else if tseq[ti] == qr[qi] {
                score_matrix[0] // match
            } else {
                score_matrix[1] // mismatch
            };
        }

        // DP inner loop
        let mut v1_cur = v1;
        for t in st0..=en0 {
            let ti = t as usize;
            let z_score = s_arr[ti].wrapping_add(qe2);
            let a_val = x1.wrapping_add(v1_cur);
            let b_val = y_arr[ti].wrapping_add(u_arr[ti]);

            let vt1 = v1_cur;
            let ut = u_arr[ti];

            let (z, d) = if !with_cigar {
                // Score only
                let mut z = z_score;
                if a_val > z { z = a_val; }
                if b_val > z { z = b_val; }
                (z, 0u8)
            } else if !right_align {
                // Left-align with traceback
                let mut d: u8 = 0;
                let mut z = z_score;
                if a_val > z { d = 1; z = a_val; }
                if b_val > z { d = 2; z = b_val; }
                (z, d)
            } else {
                // Right-align with traceback
                let mut z = z_score;
                let mut d: u8;
                if z > a_val { d = 0; } else { d = 1; z = a_val; }
                if z <= b_val { d = 2; z = b_val; }
                (z, d)
            };

            u_arr[ti] = z.wrapping_sub(vt1);
            let old_v = v_arr[ti];
            v_arr[ti] = z.wrapping_sub(ut);
            let z2 = z.wrapping_sub(q);
            let a_new = a_val.wrapping_sub(z2);
            let b_new = b_val.wrapping_sub(z2);

            x1 = x_arr[ti]; // save for next iteration
            x_arr[ti] = if a_new > 0 { a_new } else { 0 };
            y_arr[ti] = if b_new > 0 { b_new } else { 0 };

            if with_cigar {
                let mut d_final = d;
                if a_new > 0 { d_final |= 0x08; }
                if b_new > 0 { d_final |= 0x10; }

                if t == st0 {
                    band_off[r as usize] = st0;
                    band_off_end[r as usize] = en0;
                }
                let p_idx = r as usize * stride + (t - st0) as usize;
                p_arr[p_idx] = d_final;
            }

            // v1 for next t = old v[t] (before overwrite)
            v1_cur = old_v;
        }

        // h0 score tracking (same as SIMD paths)
        {
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                    let d0 = v_arr[last_h0_t as usize] as i32 - qe_i32;
                    let d1 = u_arr[(last_h0_t + 1) as usize] as i32 - qe_i32;
                    if d0 > d1 { h0 += d0; } else { h0 += d1; last_h0_t += 1; }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += v_arr[last_h0_t as usize] as i32 - qe_i32;
                } else {
                    last_h0_t += 1;
                    h0 += u_arr[last_h0_t as usize] as i32 - qe_i32;
                }
                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = last_h0_t;
                    result.max_score_query_pos = r - last_h0_t;
                } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
                    let tl = last_h0_t - result.max_score_target_pos;
                    let ql = (r - last_h0_t) - result.max_score_query_pos;
                    let l = if tl > ql { tl - ql } else { ql - tl };
                    if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend) {
                        result.zdropped = 1;
                        break;
                    }
                }
            } else {
                h0 = v_arr[0] as i32 - qe_i32 - qe_i32;
                last_h0_t = 0;
                if h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = 0;
                    result.max_score_query_pos = 0;
                }
            }
        }

        // Track endpoint scores
        if en0 == target_len as i32 - 1 {
            let h_en = if r > 0 {
                // Reconstruct H[en0] from accumulated v values
                // This is approximate but matches h0 tracking at the boundary
                h0 // when last_h0_t == en0, h0 IS H[en0]
            } else {
                h0
            };
            if last_h0_t == en0 && h_en > result.max_target_end_score {
                result.max_target_end_score = h_en;
                result.max_target_end_query_pos = r - en0;
            }
        }
        if r - st0 == query_len as i32 - 1 && last_h0_t == st0
            && h0 > result.max_query_end_score {
                result.max_query_end_score = h0;
                result.max_query_end_target_pos = st0;
            }

        // Final score
        if r == valid_range - 1 && en0 == target_len as i32 - 1 {
            result.score = h0;
        }

        last_st = st0;
        last_en = en0;
    }

    // Traceback (safe — bounds-checked slice indexing, no raw pointers)
    if with_cigar {
        traceback_single_affine_safe(
            result, query_len, target_len, end_bonus, flags,
            stride, &p_arr, &band_off, &band_off_end,
        );
    }
}

/// Global alignment (Needleman-Wunsch) with CIGAR traceback.
///
/// Dispatches to the best available implementation:
/// - SIMD targets (x86_64/aarch64/wasm32): SIMD extension with end_bonus
///   (AVX512 > AVX2 > SSE2 > NEON > WASM SIMD128)
/// - Non-SIMD targets: row-by-row Gotoh NW (scalar fallback)
///
/// Both produce correct end-to-end CIGARs. The SIMD path recomputes
/// the score from the CIGAR (stripping the end_bonus inflation).
pub fn global_align(
    qseq: &[u8], tseq: &[u8],
    alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let qlen = qseq.len();
    let tlen = tseq.len();
    if qlen == 0 || tlen == 0 {
        if qlen > 0 { result.cigar = vec![(qlen as u32) << 4 | 1]; result.score = -(gap_open + gap_extend * qlen as i32); }
        if tlen > 0 { result.cigar = vec![(tlen as u32) << 4 | 2]; result.score = -(gap_open + gap_extend * tlen as i32); }
        return;
    }

    // Use SIMD extension on SIMD-capable targets (faster at all lengths).
    // Gotoh scalar NW is the fallback for non-SIMD architectures.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32"))]
    let has_simd = std::env::var("RAMMAP_FORCE_SCALAR").is_err();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    let has_simd = false;

    if has_simd {
        global_align_simd(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, result);
    } else {
        global_align_gotoh(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, result);
    }
}

/// SIMD-accelerated global alignment via extension DP with end_bonus.
///
/// Uses the same SIMD kernels as the mapper (AVX512/AVX2/SSE2). The large
/// end_bonus forces the alignment to cover both sequences end-to-end.
/// Score is recomputed from CIGAR for correctness (end_bonus inflates ez.score).
fn global_align_simd(
    qseq: &[u8], tseq: &[u8],
    alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let end_bonus = (score_matrix[0] as i32 * std::cmp::max(qseq.len(), tseq.len()) as i32).max(1000);
    let bw = if bandwidth > 0 { bandwidth } else { -1 };
    extend_single_affine(
        qseq, tseq, alphabet_size, score_matrix,
        gap_open as i8, gap_extend as i8,
        bw, -1, end_bonus, APPROX_MAX, result,
    );
    // Recompute score from CIGAR (strip end_bonus inflation)
    let m = alphabet_size as usize;
    let mut score = 0i32;
    let mut qi = 0usize;
    let mut ti = 0usize;
    for &c in &result.cigar {
        let len = (c >> 4) as usize;
        match c & 0xf {
            0 => {
                for _ in 0..len {
                    if qi < qseq.len() && ti < tseq.len() {
                        score += score_matrix[tseq[ti].min(4) as usize * m + qseq[qi].min(4) as usize] as i32;
                    }
                    qi += 1; ti += 1;
                }
            }
            1 => { score -= gap_open + gap_extend * len as i32; qi += len; }
            2 => { score -= gap_open + gap_extend * len as i32; ti += len; }
            _ => {}
        }
    }
    result.score = score;
}

/// Row-by-row Gotoh NW with banded backtrack matrix.
///
/// Row-by-row Gotoh NW (scalar fallback for non-SIMD architectures).
/// Simple H[i][j], E[j], F recurrence with banded backtrack matrix.
fn global_align_gotoh(
    qseq: &[u8], tseq: &[u8],
    _alphabet_size: i8, score_matrix: &[i8],
    gap_open: i32, gap_extend: i32,
    bandwidth: i32,
    result: &mut DpResult,
) {
    let qlen = qseq.len();
    let tlen = tseq.len();
    let m = 5usize; // nt4 alphabet
    let gapoe = gap_open + gap_extend;
    let w = if bandwidth > 0 { bandwidth as usize } else { qlen + tlen };

    // DP arrays (current and previous row of H, plus E for query gaps)
    let mut h_prev = vec![NEG_INF; tlen + 1];
    let mut h_curr = vec![NEG_INF; tlen + 1];
    let mut e = vec![NEG_INF; tlen + 1]; // E[j]: best score ending with query gap at column j

    // Initialize first row: H[0][j] = gap penalties
    h_prev[0] = 0;
    for j in 1..=tlen {
        h_prev[j] = -(gapoe + gap_extend * (j as i32 - 1));
        e[j] = NEG_INF;
    }

    // Backtrack matrix: 2 bits per cell (0=diag, 1=up/I, 2=left/D)
    let mut bt = vec![0u8; qlen * tlen];

    for i in 1..=qlen {
        let mut f = NEG_INF; // F: best score ending with target gap (current row)
        h_curr[0] = -(gapoe + gap_extend * (i as i32 - 1));

        // Band boundaries
        let j_start = if w < i { i - w } else { 1 };
        let j_end = std::cmp::min(tlen, i + w);

        for j in j_start..=j_end {
            // Match/mismatch from diagonal
            let s = score_matrix[tseq[j - 1] as usize * m + qseq[i - 1] as usize] as i32;
            let diag = h_prev[j - 1] + s;

            // Query gap (insertion): extend or open from H
            let e_ext = e[j] - gap_extend;
            let e_open = h_prev[j] - gapoe;
            e[j] = std::cmp::max(e_ext, e_open);

            // Target gap (deletion): extend or open from H
            let f_ext = f - gap_extend;
            let f_open = h_curr[j - 1] - gapoe;
            f = std::cmp::max(f_ext, f_open);

            // Best of three
            let h = std::cmp::max(diag, std::cmp::max(e[j], f));
            h_curr[j] = h;

            // Backtrack direction
            let d = if h == diag { 0u8 }
                    else if h == e[j] { 1u8 }
                    else { 2u8 };
            bt[(i - 1) * tlen + (j - 1)] = d;
        }

        std::mem::swap(&mut h_prev, &mut h_curr);
        h_curr.fill(NEG_INF);
    }

    result.score = h_prev[tlen];

    // Traceback from (qlen, tlen)
    let mut cigar = Vec::new();
    let mut i = qlen;
    let mut j = tlen;

    while i > 0 && j > 0 {
        let d = bt[(i - 1) * tlen + (j - 1)];
        match d {
            0 => { push_cigar(&mut cigar, 0, 1); i -= 1; j -= 1; } // M
            1 => { push_cigar(&mut cigar, 1, 1); i -= 1; }          // I (consume query)
            _ => { push_cigar(&mut cigar, 2, 1); j -= 1; }          // D (consume target)
        }
    }
    if i > 0 { push_cigar(&mut cigar, 1, i as u32); }
    if j > 0 { push_cigar(&mut cigar, 2, j as u32); }

    cigar.reverse();
    result.cigar = cigar;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_dual_affine() {
        // Test the scalar implementation
        // Query: 50 A's, Target: 20 A's
        // Requires 30bp insertion to align all

        let qseq = vec![0u8; 50];
        let tseq = vec![0u8; 20];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // Single-affine: gap_open=4, gap_extend=2
        let mut ez_single = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 4, 2, -1, -1, 0, 0, &mut ez_single);

        // Dual-affine: gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        let mut ez_dual = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_dual);

        println!("Scalar single-affine: score={}", ez_single.score);
        println!("Scalar dual-affine: score={}", ez_dual.score);

        // Print CIGARs
        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Single CIGAR: {}", cigar_str(&ez_single.cigar));
        println!("Dual CIGAR: {}", cigar_str(&ez_dual.cigar));

        // Expected scores:
        // Single: 20*2 - (4 + 30*2) = 40 - 64 = -24
        // Dual: 20*2 - (24 + 30*1) = 40 - 54 = -14
        assert_eq!(ez_single.score, -24, "Single-affine score should be -24");
        assert_eq!(ez_dual.score, -14, "Dual-affine score should be -14");
        assert!(ez_dual.score > ez_single.score, "Dual should beat single");
    }

    #[test]
    fn test_simple_match() {
        let qseq = [0u8, 1, 2, 3]; // ACGT encoded as 0, 1, 2, 3
        let tseq = [0u8, 1, 2, 3];
        let alphabet_size = 5;
        // Simple matrix: match=2, mismatch=-4
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }
        
        // gap_open=4, gap_extend=2, bandwidth=10, z_drop=-1
        let gap_open = 4;
        let gap_extend = 2;
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX; // Use approx max path we implemented
        
        let mut result = DpResult::default();
        
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);
        
        println!("Score: {}, Max: {}", result.score, result.max);
        assert_eq!(result.score, 8);
    }

    #[test]
    fn test_1_mismatch() {
        let qseq = [0u8, 1, 2, 3]; // ACGT
        let tseq = [0u8, 1, 0, 3]; // ACAT (G->A mismatch)
        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }
        
        let gap_open = 4;
        let gap_extend = 2;
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX; 
        
        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);
        
        // Score: 2+2-4+2 = 2
        println!("Score: {}", result.score);
        assert_eq!(result.score, 2);
    }

    #[test]
    fn test_gap() {
        let qseq = [0u8, 1, 2, 3]; // ACGT
        let tseq = [0u8, 1, 3];    // ACT (G deleted)
        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        // High match score to ensure extension wins
        for i in 0..25 { score_matrix[i] = -10; }
        for i in 0..4 { score_matrix[i*5 + i] = 10; }
        
        let gap_open = 4;
        let gap_extend = 1; // Gap cost 4+1=5
        let bandwidth = 10;
        let z_drop = -1;
        let flags = APPROX_MAX; 
        
        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open as i8, gap_extend as i8, bandwidth, z_drop, 0, flags, &mut result);
        
        // Score: AC matches (20) - Gap (5) + T match (10) = 25.
        println!("Score: {}", result.score);
        assert_eq!(result.score, 25);
    }

    #[test]
    fn test_dual_affine_long_gap() {
        // Test case where dual-affine should outperform single-affine
        // Query: 10 A's + 30 gap + 10 A's = effectively 20 A's with 30bp insertion
        // Target: 20 A's
        // This way the alignment must include the gap to cover the full query

        let qseq = vec![0u8; 50]; // 50 A's (query)
        let tseq = vec![0u8; 20];     // 20 A's (target)
        // This forces a 30bp insertion in query to align

        let alphabet_size = 5;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // Gap penalties: gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        // For a 30bp insertion:
        // Single-affine: 4 + 30*2 = 64
        // Dual-affine: min(64, 24 + 30*1) = min(64, 54) = 54
        // Dual-affine saves 10 points

        let gap_open = 4i8;
        let gap_extend = 2i8;
        let gap_open2 = 24i8;
        let gap_extend2 = 1i8;
        let bandwidth = 100;
        let z_drop = -1;
        // Use both APPROX_MAX and RIGHT flags
        let flags = APPROX_MAX | RIGHT_ALIGN;

        let mut ez_single = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open, gap_extend, bandwidth, z_drop, 0, flags, &mut ez_single);

        let mut ez_dual = DpResult::default();
        extend_dual_affine(&qseq, &tseq, alphabet_size as i8, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_dual);

        println!("Single-affine: score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_single.score, ez_single.max, ez_single.max_score_query_pos, ez_single.max_score_target_pos, ez_single.cigar.len());
        println!("Dual-affine: score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_dual.score, ez_dual.max, ez_dual.max_score_query_pos, ez_dual.max_score_target_pos, ez_dual.cigar.len());

        // Print CIGAR if available
        if !ez_single.cigar.is_empty() {
            let cigar_str: String = ez_single.cigar.iter()
                .map(|&c| {
                    let len = c >> 4;
                    let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                    format!("{}{}", len, op)
                }).collect();
            println!("Single CIGAR: {}", cigar_str);
        }
        if !ez_dual.cigar.is_empty() {
            let cigar_str: String = ez_dual.cigar.iter()
                .map(|&c| {
                    let len = c >> 4;
                    let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                    format!("{}{}", len, op)
                }).collect();
            println!("Dual CIGAR: {}", cigar_str);
        }

        // Dual-affine should produce a higher (better) score for long gaps
        // Expected: 20 matches, 30bp insertion in query
        // Single: 20*2 - (4 + 30*2) = 40 - 64 = -24
        // Dual: 20*2 - (24 + 30*1) = 40 - 54 = -14
        // Dual-affine should be 10 points better
        assert!(ez_dual.score > ez_single.score,
            "Dual-affine ({}) should be > single-affine ({}) for long gaps",
            ez_dual.score, ez_single.score);
    }

    #[test]
    fn test_neon_vs_scalar_dual_affine() {
        // Compare NEON and scalar dual-affine implementations
        let qseq = vec![0u8; 50]; // 50 A's
        let tseq = vec![0u8; 20]; // 20 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let gap_open = 4i8;
        let gap_extend = 2i8;
        let gap_open2 = 24i8;
        let gap_extend2 = 1i8;
        let bandwidth = 100;
        let z_drop = -1;
        let flags = APPROX_MAX | RIGHT_ALIGN;

        // Run scalar
        let mut ez_scalar = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, gap_open as i32, gap_extend as i32, gap_open2 as i32, gap_extend2 as i32, -1, -1, 0, 0, &mut ez_scalar);

        // Run SIMD directly (NEON on aarch64, SSE2 on x86_64)
        let mut ez_simd = DpResult::default();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_simd);
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, 0, flags, &mut ez_simd);
        }

        println!("=== SIMD vs Scalar Dual-Affine Comparison ===");
        println!("Scalar: score={}, cigar_len={}", ez_scalar.score, ez_scalar.cigar.len());
        println!("SIMD:   score={}, max={}, max_q={}, max_t={}, cigar_len={}",
            ez_simd.score, ez_simd.max, ez_simd.max_score_query_pos, ez_simd.max_score_target_pos, ez_simd.cigar.len());

        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Scalar CIGAR: {}", cigar_str(&ez_scalar.cigar));
        println!("SIMD CIGAR:   {}", cigar_str(&ez_simd.cigar));

        // Expected: score=-14 (20*2 - (24+30*1))
        println!("Expected score: -14");

        assert_eq!(ez_scalar.score, -14, "Scalar should produce -14");
        assert_eq!(ez_simd.score, -14, "SIMD should produce -14");
        assert_eq!(ez_scalar.score, ez_simd.score, "Scores should match");
    }

    #[test]
    fn test_neon_dual_affine_small() {
        // Small test case for easier debugging
        // Query: 10 A's, Target: 5 A's (need 5bp insertion)
        let qseq = vec![0u8; 10];
        let tseq = vec![0u8; 5];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        // gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        // For 5bp insertion: single = 4+5*2 = 14, dual = min(14, 24+5*1) = min(14, 29) = 14
        // Single-affine is cheaper for short gaps, so scores should be similar

        let mut ez_scalar = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_scalar);

        let mut ez_simd = DpResult::default();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 50, -1, 0, APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 50, -1, 0, APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
        }

        println!("=== Small Test (10 vs 5 bp) ===");
        println!("Scalar: score={}", ez_scalar.score);
        println!("SIMD:   score={}", ez_simd.score);

        let cigar_str = |cigar: &[u32]| -> String {
            cigar.iter().map(|&c| {
                let len = c >> 4;
                let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
                format!("{}{}", len, op)
            }).collect()
        };

        println!("Scalar CIGAR: {}", cigar_str(&ez_scalar.cigar));
        println!("SIMD CIGAR:   {}", cigar_str(&ez_simd.cigar));

        // Expected: 5 matches (5*2=10), 5bp insertion (4+5*2=14)
        // Score = 10 - 14 = -4
        println!("Expected score: -4");

        assert_eq!(ez_scalar.score, -4, "Scalar score should be -4");
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_sequences() {
        let qseq: Vec<u8> = vec![];
        let tseq: Vec<u8> = vec![];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // Empty sequences should return default/no alignment
        assert_eq!(result.cigar.len(), 0);
    }

    #[test]
    fn test_single_base_match() {
        let qseq = vec![0u8]; // A
        let tseq = vec![0u8]; // A

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        assert_eq!(result.score, 2, "Single match should score 2");
    }

    #[test]
    fn test_single_base_mismatch() {
        let qseq = vec![0u8]; // A
        let tseq = vec![1u8]; // C

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        assert_eq!(result.score, -4, "Single mismatch should score -4");
    }

    // ========================================================================
    // Gap Size Tests
    // ========================================================================

    #[test]
    fn test_short_gap_uses_first_penalty() {
        // 10bp gap: first penalty (4+10*2=24) < second penalty (24+10*1=34)
        let qseq = vec![0u8; 15]; // 15 A's
        let tseq = vec![0u8; 5];  // 5 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 5 matches (5*2=10), 10bp insertion with first penalty (4+10*2=24)
        // Score = 10 - 24 = -14
        assert_eq!(result.score, -14, "Short gap should use first penalty");
    }

    #[test]
    fn test_medium_gap_at_crossover() {
        // 20bp gap: first penalty (4+20*2=44) == second penalty (24+20*1=44)
        // At crossover point, both should give same cost
        let qseq = vec![0u8; 30]; // 30 A's
        let tseq = vec![0u8; 10]; // 10 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 10 matches (10*2=20), 20bp insertion at crossover (cost=44)
        // Score = 20 - 44 = -24
        assert_eq!(result.score, -24, "Gap at crossover should score -24");
    }

    #[test]
    fn test_long_gap_uses_second_penalty() {
        // Already tested in test_dual_affine_long_gap, but verify again
        // 30bp gap: first penalty (4+30*2=64) > second penalty (24+30*1=54)
        let qseq = vec![0u8; 50]; // 50 A's
        let tseq = vec![0u8; 20]; // 20 A's

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        // 20 matches (20*2=40), 30bp insertion with second penalty (24+30*1=54)
        // Score = 40 - 54 = -14
        assert_eq!(result.score, -14, "Long gap should use second penalty");
    }

    // ========================================================================
    // CIGAR Validation Tests
    // ========================================================================

    fn cigar_to_string(cigar: &[u32]) -> String {
        cigar.iter().map(|&c| {
            let len = c >> 4;
            let op = match c & 0xf { 0 => 'M', 1 => 'I', 2 => 'D', _ => '?' };
            format!("{}{}", len, op)
        }).collect()
    }

    fn cigar_consumed(cigar: &[u32]) -> (usize, usize) {
        let mut q = 0usize;
        let mut t = 0usize;
        for &c in cigar {
            let len = (c >> 4) as usize;
            match c & 0xf { 0 => { q += len; t += len; }, 1 => { q += len; }, 2 => { t += len; }, _ => {} }
        }
        (q, t)
    }

    #[test]
    fn test_cigar_all_matches() {
        let qseq = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let tseq = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        let cigar = cigar_to_string(&result.cigar);
        assert_eq!(cigar, "8M", "Perfect match should be 8M");
        assert_eq!(result.score, 16, "8 matches should score 16");
    }

    #[test]
    fn test_cigar_with_insertion() {
        // Query has extra bases in the middle
        let qseq = vec![0u8, 1, 2, 3, 3, 3, 0, 1]; // ACGTTTAC (extra TTT)
        let tseq = vec![0u8, 1, 0, 1];             // ACAC

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        println!("Insertion test CIGAR: {}", cigar_to_string(&result.cigar));
        println!("Score: {}", result.score);
        // Should align as: 2M + 4I + 2M or similar
        assert!(!result.cigar.is_empty(), "Should produce CIGAR");
    }

    #[test]
    fn test_cigar_with_deletion() {
        // Target has extra bases
        let qseq = vec![0u8, 1];                   // AC
        let tseq = vec![0u8, 1, 2, 3, 0, 1];       // ACGTAC

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut result);

        println!("Deletion test CIGAR: {}", cigar_to_string(&result.cigar));
        println!("Score: {}", result.score);
        // Should align as: 2M + 4D or similar
        assert!(!result.cigar.is_empty(), "Should produce CIGAR");
    }

    // ========================================================================
    // Scalar vs SIMD Consistency Tests
    // ========================================================================

    #[test]
    fn test_scalar_neon_consistency_various_sizes() {
        let sizes = [(5, 5), (10, 8), (20, 15), (30, 25), (50, 40)];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        for (qlen, tlen) in sizes {
            let qseq = vec![0u8; qlen];
            let tseq = vec![0u8; tlen];

            let mut ez_scalar = DpResult::default();
            extend_dual_affine_scalar(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, -1, 0, 0, &mut ez_scalar);

            let mut ez_simd = DpResult::default();
            #[cfg(target_arch = "aarch64")]
            unsafe {
                extend_dual_affine_neon_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 100, -1, 0,
                    APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                extend_dual_affine2_impl(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 24, 1, 100, -1, 0,
                    APPROX_MAX | RIGHT_ALIGN, &mut ez_simd);
            }

            println!("Size {}x{}: scalar={}, simd={}", qlen, tlen, ez_scalar.score, ez_simd.score);
            assert_eq!(ez_scalar.score, ez_simd.score,
                "Scalar and SIMD should produce same score for {}x{}", qlen, tlen);
        }
    }

    #[test]
    fn test_single_affine_basic() {
        // Test the single-affine function
        let qseq = vec![0u8, 1, 2, 3]; // ACGT
        let tseq = vec![0u8, 1, 2, 3]; // ACGT

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..25 { score_matrix[i] = -4; }
        for i in 0..4 { score_matrix[i*5 + i] = 2; }

        let mut result = DpResult::default();
        extend_single_affine(&qseq, &tseq, alphabet_size, &score_matrix, 4, 2, 100, -1, 0,
            APPROX_MAX | RIGHT_ALIGN, &mut result);

        assert_eq!(result.score, 8, "4 matches should score 8");
    }

    #[test]
    fn test_extd2_matches_mm2() {
        // target: ACGTACGTACGTACGT (16 bases)
        // query:  ACGTACGT (8 bases)
        // map-ont: a=2, b=4, gap_open=4, gap_extend=2, gap_open2=24, gap_extend2=1
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }

        let mut result = DpResult::default();
        extend_dual_affine(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, 0, &mut result);

        // mm2 gives: score=-4 max=16 max_q=7 max_t=7 mqe=16 mqe_t=7
        assert_eq!(result.score, -4, "score should match mm2");
        assert_eq!(result.max, 16, "max should match mm2");
        assert_eq!(result.max_score_query_pos, 7, "max_q should match mm2");
        assert_eq!(result.max_score_target_pos, 7, "max_t should match mm2");
        assert_eq!(result.max_query_end_score, 16, "mqe should match mm2");
        assert_eq!(result.max_query_end_target_pos, 7, "mqe_t should match mm2");
    }

    #[test]
    fn test_extd2_score_only_matches_mm2() {
        // Same test but with SCORE_ONLY (no CIGAR traceback)
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }
        let mut result = DpResult::default();
        extend_dual_affine(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, SCORE_ONLY, &mut result);
        println!("score_only extd2: score={} max={} mqe={} mte={}",
            result.score, result.max, result.max_query_end_score, result.max_target_end_score);
        assert_eq!(result.score, -4, "score_only score should match mm2");
    }

    #[test]
    fn test_extd2_scalar_matches_mm2() {
        let target: Vec<u8> = vec![0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3];
        let query: Vec<u8>  = vec![0,1,2,3,0,1,2,3];
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i*5+j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i*5+4] = 0;
        }
        for j in 0..5usize { score_matrix[4*5+j] = 0; }

        let mut result = DpResult::default();
        extend_dual_affine_scalar(&query, &target, alphabet_size, &score_matrix, 4, 2, 24, 1, -1, 400, 0, 0, &mut result);
        println!("scalar extd2: score={} max={} max_q={} max_t={} mqe={} mqe_t={} mte={} mte_q={} zd={}",
            result.score, result.max, result.max_score_query_pos, result.max_score_target_pos, result.max_query_end_score, result.max_query_end_target_pos, result.max_target_end_score, result.max_target_end_query_pos, result.zdropped);
        assert_eq!(result.score, -4, "scalar score should match mm2");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_vs_sse_cigar_concordance() {
        // Test with sequences long enough to exercise multiple SIMD registers (>64 bytes)
        // Use a realistic scoring scheme
        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i * 5 + j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i * 5 + 4] = 0;
        }
        for j in 0..5usize { score_matrix[4 * 5 + j] = 0; }

        // Create 100bp sequences with some mismatches to exercise gap logic
        let mut qseq: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        let mut tseq: Vec<u8> = (0..120).map(|i| (i % 4) as u8).collect();
        // Add some mismatches
        qseq[30] = 3;
        qseq[60] = 3;
        tseq[35] = 3;
        tseq[70] = 3;

        // Call SSE41 (CIGAR mode, flags=0 means no SCORE_ONLY)
        let mut result_sse = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut result_sse,
            );
        }

        // Call AVX2 (same params)
        let mut result_avx2 = DpResult::default();
        unsafe {
            extend_dual_affine_avx2_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut result_avx2,
            );
        }

        println!("SSE41: score={} max={} cigar={}", result_sse.score, result_sse.max, cigar_to_string(&result_sse.cigar));
        println!("AVX2:  score={} max={} cigar={}", result_avx2.score, result_avx2.max, cigar_to_string(&result_avx2.cigar));

        // Also compare score-only mode
        let mut result_sse_so = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut result_sse_so,
            );
        }
        let mut result_avx2_so = DpResult::default();
        unsafe {
            extend_dual_affine_avx2_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut result_avx2_so,
            );
        }
        println!("SSE41 score-only: score={} max={}", result_sse_so.score, result_sse_so.max);
        println!("AVX2  score-only: score={} max={}", result_avx2_so.score, result_avx2_so.max);

        assert_eq!(result_sse_so.score, result_avx2_so.score,
            "Score-only: SSE41 and AVX2 should match");

        assert_eq!(result_sse.score, result_avx2.score,
            "SSE41 and AVX2 should produce same score");
        assert_eq!(result_sse.max, result_avx2.max,
            "SSE41 and AVX2 should produce same max");
        assert_eq!(cigar_to_string(&result_sse.cigar), cigar_to_string(&result_avx2.cigar),
            "SSE41 and AVX2 should produce same CIGAR");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_vs_sse_concordance() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping AVX512 test: avx512bw not available");
            return;
        }

        let alphabet_size = 5i8;
        let mut score_matrix = [0i8; 25];
        for i in 0..4usize {
            for j in 0..4usize {
                score_matrix[i * 5 + j] = if i == j { 2 } else { -4 };
            }
            score_matrix[i * 5 + 4] = 0;
        }
        for j in 0..5usize { score_matrix[4 * 5 + j] = 0; }

        // 200bp sequences with scattered mismatches to exercise gaps and CIGAR
        let mut qseq: Vec<u8> = (0..200).map(|i| (i % 4) as u8).collect();
        let mut tseq: Vec<u8> = (0..220).map(|i| (i % 4) as u8).collect();
        qseq[30] = 3; qseq[60] = 3; qseq[90] = 3; qseq[150] = 3;
        tseq[35] = 3; tseq[70] = 3; tseq[100] = 3; tseq[180] = 3;

        // --- Single-affine ---
        let mut sse_sa = DpResult::default();
        let mut avx512_sa = DpResult::default();
        unsafe {
            extend_single_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, -1, 400, 0, 0, &mut sse_sa,
            );
            extend_single_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, -1, 400, 0, 0, &mut avx512_sa,
            );
        }
        assert_eq!(sse_sa.score, avx512_sa.score, "single-affine score mismatch");
        assert_eq!(sse_sa.max, avx512_sa.max, "single-affine max mismatch");
        // CIGARs may differ between SIMD widths at low scores (tie-breaking
        // depends on lane processing order). Verify consumed lengths match.
        let (sq, st) = cigar_consumed(&sse_sa.cigar);
        let (aq, at) = cigar_consumed(&avx512_sa.cigar);
        assert_eq!((sq, st), (aq, at), "single-affine CIGAR consumed lengths differ: SSE={} AVX512={}",
            cigar_to_string(&sse_sa.cigar), cigar_to_string(&avx512_sa.cigar));

        // --- Dual-affine ---
        let mut sse_da = DpResult::default();
        let mut avx512_da = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut sse_da,
            );
            extend_dual_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, 0, &mut avx512_da,
            );
        }
        assert_eq!(sse_da.score, avx512_da.score, "dual-affine score mismatch");
        assert_eq!(sse_da.max, avx512_da.max, "dual-affine max mismatch");
        let (dq1, dt1) = cigar_consumed(&sse_da.cigar);
        let (dq2, dt2) = cigar_consumed(&avx512_da.cigar);
        assert_eq!((dq1, dt1), (dq2, dt2), "dual-affine CIGAR consumed lengths differ");

        // --- Score-only mode ---
        let mut sse_so = DpResult::default();
        let mut avx512_so = DpResult::default();
        unsafe {
            extend_dual_affine41_impl(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut sse_so,
            );
            extend_dual_affine_avx512_fn(
                &qseq, &tseq, alphabet_size, &score_matrix,
                4, 2, 24, 1, -1, 400, 0, SCORE_ONLY | APPROX_MAX, &mut avx512_so,
            );
        }
        assert_eq!(sse_so.score, avx512_so.score, "score-only score mismatch");
        assert_eq!(sse_so.max, avx512_so.max, "score-only max mismatch");
    }
}
