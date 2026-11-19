// Dual-affine gap penalty DP kernels

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use super::common::simd_compat::*;

use super::common::*;
use super::single::extend_single_affine;

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
    if *crate::align::env_flags::FORCE_SCALAR {
        // Compare mode: run both SIMD and scalar, report differences
        if *crate::align::env_flags::COMPARE_SCALAR {
            // only mutated by the x86_64 SIMD call below; on other targets `mut` is unused
            #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
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
        if super::use_avx512() {
            unsafe { extend_dual_affine_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, gap_extend2, bandwidth, z_drop, end_bonus, flags, result); }
        } else if super::use_avx2() {
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
pub(super) unsafe fn extend_dual_affine_neon_impl(
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

    // Scoring constants for NEON SIMD scoring loop (hoisted from inner loop)
    let sc_mis_v = vdupq_n_u8(score_matrix[1] as u8);
    let sc_mch_v = vdupq_n_u8(score_matrix[0] as u8);
    let sc_n_v = vdupq_n_u8(if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        (-gap_extend2) as u8
    } else {
        score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8
    });
    let m1_v = vdupq_n_u8((alphabet_size - 1) as u8);

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

    let tlen_ = target_len.div_ceil(16);
    let mut n_col_ = query_len.min(target_len);
    n_col_ = n_col_.min((bandwidth + 1) as usize).div_ceil(16) + 1;

    let with_cigar = (flags & SCORE_ONLY) == 0;

    // Memory allocation - 7 arrays for dual-affine: u, v, x, y, x2, y2, s
    // sf gets tlen_*16 bytes, qr gets (qlen_+1)*16 bytes for SIMD scoring reads
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

        // Set scores (16-element NEON SIMD scoring loop)
        if (flags & GENERIC_SCORING) == 0 {
            let mut t = st0;
            while t <= en0 {
                let sq = vld1q_u8(sf.add(t as usize));
                let st_v = vld1q_u8(qrr.add(t as usize));
                let n_mask = vorrq_u8(vceqq_u8(sq, m1_v), vceqq_u8(st_v, m1_v));
                let eq_mask = vceqq_u8(sq, st_v);
                let tmp = vbslq_u8(eq_mask, sc_mch_v, sc_mis_v);
                let tmp = vbslq_u8(n_mask, sc_n_v, tmp);
                vst1q_u8((s as *mut u8).add(t as usize), tmp);
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

                // Process [st0..en0) in groups of 4, matching the SSE/AVX kernels'
                // 4-lane max reduction (and the scalar reference, which mimics it).
                // A plain linear scan here picks a different cell among equal-max
                // ties, which diverges from the other backends on tie-break.
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut lane_h = [max_h; 4];
                let mut lane_t = [max_t; 4];
                let mut t = st0;
                while t < en1 {
                    for i in 0..4i32 {
                        let pos = (t + i) as usize;
                        let hv = *h_ptr.add(pos) + *v8_ptr.add(pos) as i8 as i32;
                        *h_ptr.add(pos) = hv;
                        if hv > lane_h[i as usize] {
                            lane_h[i as usize] = hv;
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
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
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

                if (flags & APPROX_DROP) != 0 {
                    if h0 > result.max {
                        result.max = h0;
                        result.max_score_target_pos = last_h0_t;
                        result.max_score_query_pos = r - last_h0_t;
                    } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
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
                // r == 0: dual-affine subtracts qe once
                let v0 = *v8_ptr.add(0) as i8 as i32;
                h0 = v0 - qe_scalar;
                last_h0_t = 0;
                if (flags & APPROX_DROP) != 0 && h0 > result.max {
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
        pub(super) unsafe fn $fn_name(
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

                        if (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if (flags & APPROX_DROP) != 0 && h0 > result.max {
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
extend_dual_affine_impl!(extend_dual_affine41_impl, _mm_max_epi8, _mm_min_epi8, true, "sse4.1");
#[cfg(target_arch = "wasm32")]
extend_dual_affine_impl!(extend_dual_affine_wasm_impl, _mm_max_epi8, _mm_min_epi8, true, "simd128");

#[cfg(target_arch = "x86_64")]
macro_rules! extend_dual_affine_avx2_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn $fn_name(
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

                        if (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if (flags & APPROX_DROP) != 0 && h0 > result.max {
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
        pub(super) unsafe fn $fn_name(
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

                        if (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if (flags & APPROX_DROP) != 0 && h0 > result.max {
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


/// Scalar dual-affine extension alignment.
///
/// A safe, lane-by-lane transcription of the SSE2 kernel ([`extend_dual_affine_impl`]):
/// each 128-bit register is a `[i8; 16]` and every load/store is a bounds-checked copy
/// into one buffer laid out exactly like the SIMD allocation (u, v, x, y, x2, y2, s, sf,
/// qr). That reproduces the SIMD kernel's behavior bit for bit, including the parts that
/// depend on its block structure: anti-diagonals are computed over whole 16-cell blocks
/// (`st/16*16 ..= (en+16)/16*16-1`) using leftover scores outside `[st0, en0]`, the band
/// offsets use the rounded `st`, and the 16-byte score stores may run past `s` into `sf`.
///
/// Used as fallback on non-SIMD targets and for testing via `RAMMAP_FORCE_SCALAR=1`.
#[allow(clippy::too_many_arguments)]
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
    type V = [i8; 16];
    fn ld(buf: &[u8], at: usize) -> V { std::array::from_fn(|k| buf[at + k] as i8) }
    fn st16(buf: &mut [u8], at: usize, v: V) { for k in 0..16 { buf[at + k] = v[k] as u8; } }
    fn add(a: V, b: V) -> V { std::array::from_fn(|k| a[k].wrapping_add(b[k])) }
    fn sub(a: V, b: V) -> V { std::array::from_fn(|k| a[k].wrapping_sub(b[k])) }
    fn max(a: V, b: V) -> V { std::array::from_fn(|k| a[k].max(b[k])) }
    fn min(a: V, b: V) -> V { std::array::from_fn(|k| a[k].min(b[k])) }
    /// `_mm_cmpgt_epi8` as a lane mask
    fn gt(a: V, b: V) -> [bool; 16] { std::array::from_fn(|k| a[k] > b[k]) }
    /// `(v << 1 byte) | carry`, returning the shifted vector and the new carry (`v >> 15 bytes`)
    fn shift_in(v: V, carry: i8) -> (V, i8) { (std::array::from_fn(|k| if k == 0 { carry } else { v[k - 1] }), v[15]) }

    let query_len = qseq.len();
    let target_len = tseq.len();
    let approx_max = (flags & APPROX_MAX) != 0;

    if alphabet_size <= 1 || query_len == 0 || target_len == 0 {
        return;
    }

    let (gap_open, gap_extend, gap_open2, gap_extend2) = (gap_open as i8, gap_extend as i8, gap_open2 as i8, gap_extend2 as i8);
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

    let q_: V = [gap_open; 16];
    let q2_: V = [gap_open2; 16];
    let qe_: V = [(gap_open as i32 + gap_extend as i32) as i8; 16];
    let qe2_: V = [(gap_open2 as i32 + gap_extend2 as i32) as i8; 16];
    let sc_mch = score_matrix[0];
    let sc_mis = score_matrix[1];
    let sc_n = if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        -gap_extend2
    } else {
        score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1]
    };
    let sc_mch_: V = [sc_mch; 16];
    let m1 = alphabet_size - 1;

    let bandwidth = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
    let wl = bandwidth;

    let tlen_ = target_len.div_ceil(16);
    let n_col_ = query_len.min(target_len).min((bandwidth + 1) as usize).div_ceil(16) + 1;
    let with_cigar = (flags & SCORE_ONLY) == 0;

    // One buffer with the SIMD layout: u, v, x, y, x2, y2, s, then sf, then qr
    let qlen_ = query_len.div_ceil(16);
    let (u, v, x, y, x2, y2, s) = (0, tlen_ * 16, 2 * tlen_ * 16, 3 * tlen_ * 16, 4 * tlen_ * 16, 5 * tlen_ * 16, 6 * tlen_ * 16);
    let sf = 7 * tlen_ * 16;
    let qr = sf + tlen_ * 16;
    let mut mem = vec![0u8; qr + (qlen_ + 1) * 16];

    let neg_qe = (-(gap_open as i32) - gap_extend as i32) as u8;
    let neg_q2e2 = (-(gap_open2 as i32) - gap_extend2 as i32) as u8;
    mem[u..u + 4 * tlen_ * 16].fill(neg_qe);
    mem[x2..x2 + 2 * tlen_ * 16].fill(neg_q2e2);
    for t in 0..query_len {
        mem[qr + t] = qseq[query_len - 1 - t];
    }
    mem[sf..sf + target_len].copy_from_slice(tseq);

    let valid_range = query_len + target_len - 1;
    let mut p: Vec<u8> = if with_cigar { vec![0u8; valid_range * n_col_ * 16] } else { Vec::new() };
    let mut band_off: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };
    let mut band_off_end: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };

    // H[] array for exact max tracking (only when !approx_max)
    let mut h: Vec<i32> = if !approx_max { vec![NEG_INF; tlen_ * 16] } else { Vec::new() };

    init_dp_result(result);

    let right_align = (flags & RIGHT_ALIGN) != 0;
    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    let mut h0: i32 = 0;
    let mut last_h0_t: i32 = 0;

    for r in 0..valid_range as i32 {
        let mut st = 0i32;
        let mut en = target_len as i32 - 1;

        // qrr = qr + (qlen - 1 - r); qrr[t] is only read for t >= st0 >= r - qlen + 1
        let qrr = qr as isize + query_len as isize - 1 - r as isize;

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
        let (x1, x21, v1): (i8, i8, i8) = if st > 0 {
            if st > last_st && st - 1 <= last_en {
                let k = (st - 1) as usize;
                (mem[x + k] as i8, mem[x2 + k] as i8, mem[v + k] as i8)
            } else {
                (-gap_open - gap_extend, -gap_open2 - gap_extend2, -gap_open - gap_extend)
            }
        } else {
            let v1 = if r == 0 { -gap_open - gap_extend } else if r < long_thres { -gap_extend } else if r == long_thres { long_diff } else { -gap_extend2 };
            (-gap_open - gap_extend, -gap_open2 - gap_extend2, v1)
        };

        if en >= r {
            let k = r as usize;
            mem[y + k] = (-gap_open - gap_extend) as u8;
            mem[y2 + k] = (-gap_open2 - gap_extend2) as u8;
            mem[u + k] = (if r == 0 { -gap_open - gap_extend } else if r < long_thres { -gap_extend } else if r == long_thres { long_diff } else { -gap_extend2 }) as u8;
        }

        // Set scores in 16-cell chunks from st0 (the last chunk may run past en0)
        if (flags & GENERIC_SCORING) == 0 {
            let mut t = st0 as usize;
            while t as i32 <= en0 {
                let sq = ld(&mem, sf + t);
                let sv = ld(&mem, (qrr + t as isize) as usize);
                let sc: V = std::array::from_fn(|k| {
                    if sq[k] == m1 || sv[k] == m1 { sc_n } else if sq[k] == sv[k] { sc_mch } else { sc_mis }
                });
                st16(&mut mem, s + t, sc);
                t += 16;
            }
        } else {
            for t in st0 as usize..=(en0 as usize).min(st0 as usize + tlen_ * 16 - 1) {
                let i = mem[sf + t] as usize * alphabet_size as usize + mem[(qrr + t as isize) as usize] as usize;
                mem[s + t] = score_matrix[i] as u8;
            }
        }

        // Core DP over whole 16-cell blocks
        let (mut x1_, mut x21_, mut v1_) = (x1, x21, v1);
        let st_ = st as usize / 16;
        let en_ = en as usize / 16;
        if with_cigar {
            band_off[r as usize] = st;
            band_off_end[r as usize] = en;
        }

        for ti in st_..=en_ {
            let o = ti * 16;
            let mut z = ld(&mem, s + o);
            let xt1; let vt1; let x2t1;
            (xt1, x1_) = shift_in(ld(&mem, x + o), x1_);
            (vt1, v1_) = shift_in(ld(&mem, v + o), v1_);
            let a = add(xt1, vt1);                       // I1 candidate
            let ut = ld(&mem, u + o);
            let b = add(ld(&mem, y + o), ut);            // D1 candidate
            (x2t1, x21_) = shift_in(ld(&mem, x2 + o), x21_);
            let a2 = add(x2t1, vt1);                     // I2 candidate
            let b2 = add(ld(&mem, y2 + o), ut);          // D2 candidate

            let mut d: V = [0; 16];
            if !with_cigar {
                z = max(max(max(max(z, a), b), a2), b2);
            } else if !right_align {
                // LEFT tie-breaking: a gap wins only if strictly greater
                for (cand, flag) in [(a, 1i8), (b, 2), (a2, 3), (b2, 4)] {
                    let m = gt(cand, z);
                    for k in 0..16 { if m[k] { d[k] = flag; z[k] = cand[k]; } }
                }
            } else {
                // RIGHT tie-breaking: a gap wins if >=
                let m = gt(z, a);
                for k in 0..16 { d[k] = if m[k] { 0 } else { 1 }; if !m[k] { z[k] = a[k]; } }
                for (cand, flag) in [(b, 2i8), (a2, 3), (b2, 4)] {
                    let m = gt(z, cand);
                    for k in 0..16 { if !m[k] { d[k] = flag; z[k] = cand[k]; } }
                }
            }
            z = min(z, sc_mch_);

            st16(&mut mem, u + o, sub(z, vt1));
            st16(&mut mem, v + o, sub(z, ut));
            let tmp1 = sub(z, q_);
            let tmp2 = sub(z, q2_);
            let news = [(sub(a, tmp1), x, qe_, 0x08i8), (sub(b, tmp1), y, qe_, 0x10), (sub(a2, tmp2), x2, qe2_, 0x20), (sub(b2, tmp2), y2, qe2_, 0x40u8 as i8)];
            for (nv, arr, qe, flag) in news {
                // left: keep where nv > 0; right: keep where !(0 > nv)
                let keep: [bool; 16] = std::array::from_fn(|k| if right_align && with_cigar { nv[k] >= 0 } else { nv[k] > 0 });
                let kept: V = std::array::from_fn(|k| if keep[k] { nv[k] } else { 0 });
                st16(&mut mem, arr + o, sub(kept, qe));
                if with_cigar { for k in 0..16 { if keep[k] { d[k] |= flag; } } }
            }

            if with_cigar {
                let pr = (r as usize * n_col_ + ti - st_) * 16;
                for k in 0..16 { p[pr + k] = d[k] as u8; }
            }
        }

        let qe_scalar = gap_open as i32 + gap_extend as i32;
        let u8_at = |mem: &[u8], t: i32| mem[u + t as usize] as i8 as i32;
        let v8_at = |mem: &[u8], t: i32| mem[v + t as usize] as i8 as i32;

        if !approx_max {
            // Exact max tracking with 32-bit H[]
            let mut max_h: i32;
            let mut max_t: i32;
            if r > 0 {
                let h_en0 = if en0 > 0 { h[en0 as usize - 1] + u8_at(&mem, en0) } else { h[en0 as usize] + v8_at(&mem, en0) };
                h[en0 as usize] = h_en0;
                max_h = h_en0;
                max_t = en0;

                // [st0, en1) in 4-lane groups, then reduce lanes (SSE order)
                let en1 = st0 + (en0 - st0) / 4 * 4;
                let mut hh = [max_h; 4];
                let mut tt = [max_t; 4];
                let mut t = st0;
                while t < en1 {
                    for i in 0..4 {
                        let k = (t + i as i32) as usize;
                        h[k] += v8_at(&mem, t + i as i32);
                        if h[k] > hh[i] { hh[i] = h[k]; tt[i] = t; }
                    }
                    t += 4;
                }
                for i in 0..4 {
                    if max_h < hh[i] { max_h = hh[i]; max_t = tt[i] + i as i32; }
                }
                while t < en0 {
                    h[t as usize] += v8_at(&mem, t);
                    if h[t as usize] > max_h { max_h = h[t as usize]; max_t = t; }
                    t += 1;
                }
            } else {
                h[0] = v8_at(&mem, 0) - qe_scalar;
                max_h = h[0];
                max_t = 0;
            }
            if en0 == target_len as i32 - 1 && h[en0 as usize] > result.max_target_end_score {
                result.max_target_end_score = h[en0 as usize];
                result.max_target_end_query_pos = r - en0;
            }
            if r - st0 == query_len as i32 - 1 && h[st0 as usize] > result.max_query_end_score {
                result.max_query_end_score = h[st0 as usize];
                result.max_query_end_target_pos = st0;
            }
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
                result.score = h[target_len - 1];
            }
        } else {
            // Approximate max tracking
            if r > 0 {
                if last_h0_t >= st0 && last_h0_t <= en0 && last_h0_t + 1 >= st0 && last_h0_t < en0 {
                    let d0 = v8_at(&mem, last_h0_t);
                    let d1 = u8_at(&mem, last_h0_t + 1);
                    if d0 > d1 { h0 += d0; } else { h0 += d1; last_h0_t += 1; }
                } else if last_h0_t >= st0 && last_h0_t <= en0 {
                    h0 += v8_at(&mem, last_h0_t);
                } else {
                    last_h0_t += 1;
                    h0 += u8_at(&mem, last_h0_t);
                }

                if (flags & APPROX_DROP) != 0 {
                    if h0 > result.max {
                        result.max = h0;
                        result.max_score_target_pos = last_h0_t;
                        result.max_score_query_pos = r - last_h0_t;
                    } else if last_h0_t >= result.max_score_target_pos && (r - last_h0_t) >= result.max_score_query_pos {
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
                h0 = v8_at(&mem, 0) - qe_scalar;
                last_h0_t = 0;
                if (flags & APPROX_DROP) != 0 && h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = 0;
                    result.max_score_query_pos = 0;
                }
            }
            if r == query_len as i32 + target_len as i32 - 2 && en0 == target_len as i32 - 1 {
                result.score = h0;
            }
        }

        last_st = st;
        last_en = en;
    }

    if approx_max && result.score == NEG_INF {
        result.score = result.max;
    }

    if with_cigar {
        traceback_dual_affine_safe(
            result, query_len, target_len, end_bonus, flags, n_col_ * 16,
            &p, &band_off, &band_off_end,
        );
    }
}
