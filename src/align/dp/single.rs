// Single-affine gap penalty DP kernels

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use super::common::simd_compat::*;

use super::common::*;

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn extend_single_affine_neon_impl(
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
    let approx_max = (flags & APPROX_MAX) != 0;
    
    if alphabet_size <= 0 {
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
    
    let _m1_ = vdupq_n_u8((alphabet_size - 1) as u8);

    // Scoring constants for NEON SIMD scoring loop (unsigned u8 for vbslq_u8)
    let sc_mis_v = vdupq_n_u8(score_matrix[1] as u8);
    let sc_mch_v = vdupq_n_u8(score_matrix[0] as u8);
    let sc_n_v = vdupq_n_u8(if score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] == 0 {
        (-gap_extend) as u8
    } else {
        score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8
    });
    let m1_v = _m1_; // reuse existing constant

    // Dimension calculations
    let bandwidth = if bandwidth < 0 { if target_len > query_len { target_len as i32 } else { query_len as i32 } } else { bandwidth };
    let wl = bandwidth;

    let tlen_ = target_len.div_ceil(16); // Number of 16-byte blocks for target_len

    // _n_col_ is for traceback arrays p, off, off_end
    let mut _n_col_ = if query_len < target_len { query_len } else { target_len };
    _n_col_ = (if _n_col_ < (bandwidth + 1) as usize { _n_col_ } else { (bandwidth + 1) as usize }).div_ceil(16) + 1;
    
    let with_cigar = (flags & SCORE_ONLY) == 0;
    
    // Calculate total memory needed for a single allocation
    // Buffer sizing: sf gets tlen_*16 bytes, qr gets (qlen_+1)*16 bytes
    let qlen_ = query_len.div_ceil(16);
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
    std::ptr::copy_nonoverlapping(tseq.as_ptr(), sf, target_len);

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
        
        // Align st down to 16, en up to 16-1
        st = (st / 16) * 16;
        en = ((en + 16) / 16) * 16 - 1;
        
        // set boundary conditions
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
             
             let tmp = vextq_u8(xt1, zero_, 15);
             
             let shifted_xt1 = vextq_u8(zero_, xt1, 15); 
             xt1 = vorrq_u8(shifted_xt1, x1_);
             x1_ = tmp;
             
             let vt_val = vld1q_u8((v as *const u8).add(ti*16));
             let mut vt1 = vt_val;
             
             let tmp_v = vextq_u8(vt1, zero_, 15);
             
             let shifted_vt1 = vextq_u8(zero_, vt1, 15);
             vt1 = vorrq_u8(shifted_vt1, v1_);
             v1_ = tmp_v;
             
             let mut a = vaddq_u8(xt1, vt1);
             
             let ut = vld1q_u8((u as *const u8).add(ti*16));
             
             let yt = vld1q_u8((y as *const u8).add(ti*16));
             let mut b = vaddq_u8(yt, ut);
             
             let b_final_s8 = vreinterpretq_s8_u8(b);
             let b_s8 = b_final_s8;
             let z_s8 = vreinterpretq_s8_u8(z);
             let a_s8 = vreinterpretq_s8_u8(a);
             
             if with_cigar {
                 let offset = (r as usize * _n_col_) as isize - st_ as isize;
                 let pr_ptr = p_ptr.add((offset + ti as isize) as usize * 16);
                 
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
        
        // Score and max tracking
        if approx_max {
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

                // Track max when not in pure APPROX_MAX mode (i.e. either exact-equivalent approx_max=false,or APPROX_DROP also set).
                // z-drop check only under APPROX_DROP.
                if (flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0 {
                    if h0 > result.max {
                        result.max = h0;
                        result.max_score_target_pos = last_h0_t;
                        result.max_score_query_pos = r - last_h0_t;
                    } else if (flags & APPROX_DROP) != 0
                        && last_h0_t >= result.max_score_target_pos
                        && (r - last_h0_t) >= result.max_score_query_pos {
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
                if ((flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0) && h0 > result.max {
                    result.max = h0; result.max_score_target_pos = 0; result.max_score_query_pos = 0;
                }
            }
        }
        
        // Final score update
        if r == valid_range - 1 /* query_len+target_len-2 */ {
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
// The macro below expands into three variants that differ only in how they
// realize a handful of intrinsics (signed max_epi8 and blendv_epi8 are SSE4.1
// additions). Each variant is compiled with the narrowest target_feature it
// actually needs, so LLVM does not emit emulation when native instructions
// are available:
//   extend_single_affine2_impl    → target_feature("sse2"),    emulated helpers
//   extend_single_affine41_impl   → target_feature("sse4.1"),  native intrinsics
//   extend_single_affine_wasm_impl → target_feature("simd128"), native intrinsics
// Runtime dispatch selects among them in the parent function.

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
macro_rules! extend_single_affine_impl {
    ($fn_name:ident, $max_epi8:path, $is_sse41:expr, $target_feat:tt) => {
        #[target_feature(enable = $target_feat)]
        pub(super) unsafe fn $fn_name(
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

                        // x = max(a, 0)
                        let x_res = $max_epi8(a, zero_);
                        _mm_storeu_si128(x.add(ti), x_res);

                        // d |= (a > 0 ? 0x08 : 0)
                        let mask_a = _mm_cmpgt_epi8(a, zero_);
                        d = _mm_or_si128(d, _mm_and_si128(flag8_, mask_a));

                        // y = max(b, 0)
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

                        if (flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if (flags & APPROX_DROP) != 0
                                && last_h0_t >= result.max_score_target_pos
                                && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if ((flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0) && h0 > result.max {
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
extend_single_affine_impl!(extend_single_affine41_impl, _mm_max_epi8, true, "sse4.1");
#[cfg(target_arch = "wasm32")]
extend_single_affine_impl!(extend_single_affine_wasm_impl, _mm_max_epi8, true, "simd128");

// ============================================================================
// AVX2 Implementation - Single-Affine Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_single_affine_avx2_impl {
    ($fn_name:ident) => {
        #[target_feature(enable = "avx2")]
        pub(super) unsafe fn $fn_name(
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

                        // x = max(a, 0)
                        let x_res = _mm256_max_epi8(a, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, x_res);

                        // d |= (a > 0 ? 0x08 : 0)
                        let mask_a = _mm256_cmpgt_epi8(a, zero_);
                        d = _mm256_or_si256(d, _mm256_and_si256(flag8_, mask_a));

                        // y = max(b, 0)
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

                // Score and max tracking (scalar — identical to SSE version)
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

                        if (flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if (flags & APPROX_DROP) != 0
                                && last_h0_t >= result.max_score_target_pos
                                && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if ((flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0) && h0 > result.max {
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
        pub(super) unsafe fn $fn_name(
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

                // Score and max tracking (scalar — identical to SSE/AVX2 version)
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

                        if (flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0 {
                            if h0 > result.max {
                                result.max = h0;
                                result.max_score_target_pos = last_h0_t;
                                result.max_score_query_pos = r - last_h0_t;
                            } else if (flags & APPROX_DROP) != 0
                                && last_h0_t >= result.max_score_target_pos
                                && (r - last_h0_t) >= result.max_score_query_pos {
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
                        if ((flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0) && h0 > result.max {
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
    if *crate::align::env_flags::FORCE_SCALAR {
        extend_single_affine_scalar(qseq, tseq, alphabet_size, score_matrix, gap_open as i32, gap_extend as i32, bandwidth, z_drop, end_bonus, flags, result);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        extend_single_affine_neon_impl(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::use_avx512() {
            unsafe { extend_single_affine_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, bandwidth, z_drop, end_bonus, flags, result); }
        } else if super::use_avx2() {
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
                if (flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0 {
                    if h0 > result.max {
                        result.max = h0;
                        result.max_score_target_pos = last_h0_t;
                        result.max_score_query_pos = r - last_h0_t;
                    } else if (flags & APPROX_DROP) != 0
                        && last_h0_t >= result.max_score_target_pos
                        && (r - last_h0_t) >= result.max_score_query_pos {
                        let tl = last_h0_t - result.max_score_target_pos;
                        let ql = (r - last_h0_t) - result.max_score_query_pos;
                        let l = if tl > ql { tl - ql } else { ql - tl };
                        if z_drop >= 0 && (result.max - h0) > (z_drop + l * gap_extend) {
                            result.zdropped = 1;
                            break;
                        }
                    }
                }
            } else {
                h0 = v_arr[0] as i32 - qe_i32 - qe_i32;
                last_h0_t = 0;
                if ((flags & APPROX_MAX) == 0 || (flags & APPROX_DROP) != 0) && h0 > result.max {
                    result.max = h0;
                    result.max_score_target_pos = 0;
                    result.max_score_query_pos = 0;
                }
            }
        }

        // Track endpoint scores. h0 equals H[last_h0_t], so the right-boundary
        // score H[en0] is available exactly when the diagonal has reached en0.
        if en0 == target_len as i32 - 1
            && last_h0_t == en0
            && h0 > result.max_target_end_score
        {
            result.max_target_end_score = h0;
            result.max_target_end_query_pos = r - en0;
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
