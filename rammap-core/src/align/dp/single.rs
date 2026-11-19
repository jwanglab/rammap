// Single-affine gap penalty DP kernels

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use super::common::simd_compat::*;

use super::common::*;

// ============================================================================
// NEON Implementation - Single-Affine Alignment
// ============================================================================

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

    if alphabet_size <= 0 || query_len == 0 || target_len == 0 {
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
    let max_sc_ = vdupq_n_u8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as u8);
    let right_align = (flags & RIGHT_ALIGN) != 0;
    let zero_s = vdupq_n_s8(0);

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

    init_dp_result_full(result);
    // Skip alignment when a mismatch could never beat two gaps
    {
        let n = alphabet_size as usize * alphabet_size as usize;
        let min_sc = score_matrix[1..n].iter().fold(score_matrix[1], |a, &b| a.min(b)) as i32;
        if -min_sc > 2 * (gap_open as i32 + gap_extend as i32) { return; }
    }

    let mut last_st = -1;
    let mut last_en = -1;
    let valid_range = (query_len + target_len - 1) as i32;
    let mut track = SingleAffineTrack::new(flags, query_len, target_len, gap_open as i32, gap_extend as i32, z_drop);
    
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

        if with_cigar {
            *band_offset_ptr.add(r as usize) = st;
            *band_offset_end_ptr.add(r as usize) = en;
        }

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
             
             let a_s8 = vreinterpretq_s8_u8(a);
             let b_s8 = vreinterpretq_s8_u8(b);
             let mut zs = vreinterpretq_s8_u8(z);
             let mut d = zero_;
             if with_cigar && !right_align {
                 // gap left-alignment: a gap wins only if strictly greater
                 d = vandq_u8(vcgtq_s8(a_s8, zs), flag1_);        // d = a > z? 1 : 0
                 zs = vmaxq_s8(zs, a_s8);
                 d = vbslq_u8(vcgtq_s8(b_s8, zs), flag2_, d);     // d = b > z? 2 : d
             } else if with_cigar {
                 // gap right-alignment: a gap wins on ties
                 d = vbicq_u8(flag1_, vcgtq_s8(zs, a_s8));        // d = z > a? 0 : 1
                 zs = vmaxq_s8(zs, a_s8);
                 d = vbslq_u8(vcgtq_s8(zs, b_s8), d, flag2_);     // d = z > b? d : 2
             } else {
                 zs = vmaxq_s8(zs, a_s8);
             }

             z = vmaxq_u8(vreinterpretq_u8_s8(zs), b); // both non-negative
             z = vminq_u8(z, max_sc_);
             vst1q_u8((u as *mut u8).add(ti*16), vsubq_u8(z, vt1));
             vst1q_u8((v as *mut u8).add(ti*16), vsubq_u8(z, ut));
             z = vsubq_u8(z, q_);
             a = vsubq_u8(a, z);
             b = vsubq_u8(b, z);
             let a_s8 = vreinterpretq_s8_u8(a);
             let b_s8 = vreinterpretq_s8_u8(b);

             if with_cigar && right_align {
                 let tmp = vcgtq_s8(zero_s, a_s8);
                 vst1q_u8((x as *mut u8).add(ti*16), vbicq_u8(a, tmp));
                 d = vorrq_u8(d, vbicq_u8(flag8_, tmp));          // d = 0 > a? 0 : 0x08
                 let tmp = vcgtq_s8(zero_s, b_s8);
                 vst1q_u8((y as *mut u8).add(ti*16), vbicq_u8(b, tmp));
                 d = vorrq_u8(d, vbicq_u8(flag16_, tmp));         // d = 0 > b? 0 : 0x10
             } else {
                 let tmp = vcgtq_s8(a_s8, zero_s);
                 vst1q_u8((x as *mut u8).add(ti*16), vandq_u8(a, tmp));
                 d = vorrq_u8(d, vandq_u8(flag8_, tmp));          // d = a > 0? 0x08 : 0
                 let tmp = vcgtq_s8(b_s8, zero_s);
                 vst1q_u8((y as *mut u8).add(ti*16), vandq_u8(b, tmp));
                 d = vorrq_u8(d, vandq_u8(flag16_, tmp));         // d = b > 0? 0x10 : 0
             }
             if with_cigar {
                 vst1q_u8(p_ptr.add((r as usize * _n_col_ + ti - st_) * 16), d);
             }
        }

        // Score and max tracking (exact H[] unless APPROX_MAX)
        let u8s = std::slice::from_raw_parts(u8_ptr as *const u8, target_len);
        let v8s = std::slice::from_raw_parts(v8_ptr as *const u8, target_len);
        if track.update(result, u8s, v8s, r, st0, en0) { break; }

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
            let max_sc_ = _mm_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);
            let right_align = (flags & RIGHT_ALIGN) != 0;

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

            init_dp_result_full(result);
            // Skip alignment when a mismatch could never beat two gaps
            {
                let n = alphabet_size as usize * alphabet_size as usize;
                let min_sc = score_matrix[1..n].iter().fold(score_matrix[1], |a, &b| a.min(b)) as i32;
                if -min_sc > 2 * (gap_open as i32 + gap_extend as i32) { return; }
            }

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut track = SingleAffineTrack::new(flags, query_len, target_len, gap_open as i32, gap_extend as i32, z_drop);

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

                if with_cigar {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

                for ti in st_..=en_ {
                    // __dp_code_block1
                    let mut z = _mm_add_epi8(_mm_loadu_si128(s.add(ti)), qe2_);
                    let xt_val = _mm_loadu_si128(x.add(ti));
                    let xt1 = _mm_or_si128(_mm_slli_si128(xt_val, 1), x1_);
                    x1_ = _mm_srli_si128(xt_val, 15);
                    let vt_val = _mm_loadu_si128(v.add(ti));
                    let vt1 = _mm_or_si128(_mm_slli_si128(vt_val, 1), v1_);
                    v1_ = _mm_srli_si128(vt_val, 15);
                    let mut a = _mm_add_epi8(xt1, vt1);
                    let ut = _mm_loadu_si128(u.add(ti));
                    let mut b = _mm_add_epi8(_mm_loadu_si128(y.add(ti)), ut);

                    let mut d = zero_;
                    if with_cigar && !right_align {
                        // gap left-alignment: a gap wins only if strictly greater
                        d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1_);                       // d = a > z? 1 : 0
                        z = $max_epi8(z, a);
                        let tmp = _mm_cmpgt_epi8(b, z);
                        d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, flag2_)); // d = b > z? 2 : d
                    } else if with_cigar {
                        // gap right-alignment: a gap wins on ties
                        d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), flag1_);                    // d = z > a? 0 : 1
                        z = $max_epi8(z, a);
                        let tmp = _mm_cmpgt_epi8(z, b);
                        d = _mm_or_si128(_mm_andnot_si128(tmp, flag2_), _mm_and_si128(tmp, d)); // d = z > b? d : 2
                    } else {
                        z = $max_epi8(z, a);
                    }

                    // __dp_code_block2
                    z = _mm_max_epu8(z, b); // both non-negative
                    z = _mm_min_epu8(z, max_sc_);
                    _mm_storeu_si128(u.add(ti), _mm_sub_epi8(z, vt1));
                    _mm_storeu_si128(v.add(ti), _mm_sub_epi8(z, ut));
                    z = _mm_sub_epi8(z, q_);
                    a = _mm_sub_epi8(a, z);
                    b = _mm_sub_epi8(b, z);

                    if with_cigar && right_align {
                        let tmp = _mm_cmpgt_epi8(zero_, a);
                        _mm_storeu_si128(x.add(ti), _mm_andnot_si128(tmp, a));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag8_));  // d = 0 > a? 0 : 0x08
                        let tmp = _mm_cmpgt_epi8(zero_, b);
                        _mm_storeu_si128(y.add(ti), _mm_andnot_si128(tmp, b));
                        d = _mm_or_si128(d, _mm_andnot_si128(tmp, flag16_)); // d = 0 > b? 0 : 0x10
                    } else {
                        let tmp = _mm_cmpgt_epi8(a, zero_);
                        _mm_storeu_si128(x.add(ti), _mm_and_si128(tmp, a));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag8_));     // d = a > 0? 0x08 : 0
                        let tmp = _mm_cmpgt_epi8(b, zero_);
                        _mm_storeu_si128(y.add(ti), _mm_and_si128(tmp, b));
                        d = _mm_or_si128(d, _mm_and_si128(tmp, flag16_));    // d = b > 0? 0x10 : 0
                    }
                    if with_cigar {
                        let pr_ptr = p_ptr.add((r as usize * _n_col_ + ti - st_) * 16);
                        _mm_storeu_si128(pr_ptr as *mut __m128i, d);
                    }
                }

                // Score and max tracking (exact H[] unless APPROX_MAX)
                let u8s = std::slice::from_raw_parts(u8_ptr as *const u8, target_len);
                let v8s = std::slice::from_raw_parts(v8_ptr as *const u8, target_len);
                if track.update(result, u8s, v8s, r, st0, en0) { break; }

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
            let max_sc_ = _mm256_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);
            let right_align = (flags & RIGHT_ALIGN) != 0;

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

            init_dp_result_full(result);
            // Skip alignment when a mismatch could never beat two gaps
            {
                let n = alphabet_size as usize * alphabet_size as usize;
                let min_sc = score_matrix[1..n].iter().fold(score_matrix[1], |a, &b| a.min(b)) as i32;
                if -min_sc > 2 * (gap_open as i32 + gap_extend as i32) { return; }
            }

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut track = SingleAffineTrack::new(flags, query_len, target_len, gap_open as i32, gap_extend as i32, z_drop);

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;
                let x1: i8;
                let v1: i8;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

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

                if with_cigar {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

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

                    let mut d = zero_;
                    if with_cigar && !right_align {
                        // gap left-alignment: a gap wins only if strictly greater
                        d = _mm256_and_si256(_mm256_cmpgt_epi8(a, z), flag1_);          // d = a > z? 1 : 0
                        z = _mm256_max_epi8(z, a);
                        let tmp = _mm256_cmpgt_epi8(b, z);
                        d = _mm256_blendv_epi8(d, flag2_, tmp);                          // d = b > z? 2 : d
                    } else if with_cigar {
                        // gap right-alignment: a gap wins on ties
                        d = _mm256_andnot_si256(_mm256_cmpgt_epi8(z, a), flag1_);       // d = z > a? 0 : 1
                        z = _mm256_max_epi8(z, a);
                        let tmp = _mm256_cmpgt_epi8(z, b);
                        d = _mm256_blendv_epi8(flag2_, d, tmp);                          // d = z > b? d : 2
                    } else {
                        z = _mm256_max_epi8(z, a);
                    }

                    z = _mm256_max_epu8(z, b); // both non-negative
                    z = _mm256_min_epu8(z, max_sc_);
                    _mm256_storeu_si256(u_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, vt1));
                    _mm256_storeu_si256(v_b.add(bp) as *mut __m256i, _mm256_sub_epi8(z, ut));
                    z = _mm256_sub_epi8(z, q_);
                    a = _mm256_sub_epi8(a, z);
                    b = _mm256_sub_epi8(b, z);

                    if with_cigar && right_align {
                        let tmp = _mm256_cmpgt_epi8(zero_, a);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_andnot_si256(tmp, a));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag8_));    // d = 0 > a? 0 : 0x08
                        let tmp = _mm256_cmpgt_epi8(zero_, b);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_andnot_si256(tmp, b));
                        d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag16_));   // d = 0 > b? 0 : 0x10
                    } else {
                        let tmp = _mm256_cmpgt_epi8(a, zero_);
                        _mm256_storeu_si256(x_b.add(bp) as *mut __m256i, _mm256_and_si256(tmp, a));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag8_));       // d = a > 0? 0x08 : 0
                        let tmp = _mm256_cmpgt_epi8(b, zero_);
                        _mm256_storeu_si256(y_b.add(bp) as *mut __m256i, _mm256_and_si256(tmp, b));
                        d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag16_));      // d = b > 0? 0x10 : 0
                    }
                    if with_cigar {
                        let pr_ptr_local = p_ptr.add(r as usize * stride_bytes + (bp - st_usize));
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

                    bp += 32;
                }

                // Score and max tracking (exact H[] unless APPROX_MAX)
                let u8s = std::slice::from_raw_parts(u_b as *const u8, target_len);
                let v8s = std::slice::from_raw_parts(v_b as *const u8, target_len);
                if track.update(result, u8s, v8s, r, st0, en0) { break; }

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
            let max_sc_ = _mm512_set1_epi8((score_matrix[0] as i32 + (gap_open as i32 + gap_extend as i32) * 2) as i8);
            let right_align = (flags & RIGHT_ALIGN) != 0;

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

            init_dp_result_full(result);
            // Skip alignment when a mismatch could never beat two gaps
            {
                let n = alphabet_size as usize * alphabet_size as usize;
                let min_sc = score_matrix[1..n].iter().fold(score_matrix[1], |a, &b| a.min(b)) as i32;
                if -min_sc > 2 * (gap_open as i32 + gap_extend as i32) { return; }
            }

            let mut last_st: i32 = -1;
            let mut last_en: i32 = -1;
            let valid_range = (query_len + target_len - 1) as i32;
            let mut track = SingleAffineTrack::new(flags, query_len, target_len, gap_open as i32, gap_extend as i32, z_drop);

            for r in 0..valid_range {
                let mut st = 0i32;
                let mut en = target_len as i32 - 1;
                let x1: i8;
                let v1: i8;

                let qrr = qr.offset(query_len as isize - 1 - r as isize);

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

                if with_cigar {
                    *band_offset_ptr.add(r as usize) = st;
                    *band_offset_end_ptr.add(r as usize) = en;
                }

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

                    let mut d = zero_;
                    if with_cigar && !right_align {
                        // gap left-alignment: a gap wins only if strictly greater
                        d = _mm512_maskz_mov_epi8(_mm512_cmpgt_epi8_mask(a, z), flag1_);   // d = a > z? 1 : 0
                        z = _mm512_max_epi8(z, a);
                        d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(b, z), d, flag2_); // d = b > z? 2 : d
                    } else if with_cigar {
                        // gap right-alignment: a gap wins on ties
                        d = _mm512_maskz_mov_epi8(!_mm512_cmpgt_epi8_mask(z, a), flag1_);  // d = z > a? 0 : 1
                        z = _mm512_max_epi8(z, a);
                        d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(z, b), flag2_, d); // d = z > b? d : 2
                    } else {
                        z = _mm512_max_epi8(z, a);
                    }

                    z = _mm512_max_epu8(z, b); // both non-negative
                    z = _mm512_min_epu8(z, max_sc_);
                    _mm512_storeu_si512(u_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, vt1));
                    _mm512_storeu_si512(v_b.add(bp) as *mut __m512i, _mm512_sub_epi8(z, ut));
                    z = _mm512_sub_epi8(z, q_);
                    a = _mm512_sub_epi8(a, z);
                    b = _mm512_sub_epi8(b, z);

                    if with_cigar && right_align {
                        let keep = !_mm512_cmpgt_epi8_mask(zero_, a);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_maskz_mov_epi8(keep, a));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(keep, flag8_));   // d = 0 > a? 0 : 0x08
                        let keep = !_mm512_cmpgt_epi8_mask(zero_, b);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_maskz_mov_epi8(keep, b));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(keep, flag16_));  // d = 0 > b? 0 : 0x10
                    } else {
                        let keep = _mm512_cmpgt_epi8_mask(a, zero_);
                        _mm512_storeu_si512(x_b.add(bp) as *mut __m512i, _mm512_maskz_mov_epi8(keep, a));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(keep, flag8_));   // d = a > 0? 0x08 : 0
                        let keep = _mm512_cmpgt_epi8_mask(b, zero_);
                        _mm512_storeu_si512(y_b.add(bp) as *mut __m512i, _mm512_maskz_mov_epi8(keep, b));
                        d = _mm512_or_si512(d, _mm512_maskz_mov_epi8(keep, flag16_));  // d = b > 0? 0x10 : 0
                    }
                    if with_cigar {
                        let pr_ptr_local = p_ptr.add(r as usize * stride_bytes + (bp - st_usize));
                        _mm512_storeu_si512(pr_ptr_local as *mut __m512i, d);
                    }

                    if excess > 0 {
                        let es = en_usize + 1;
                        std::ptr::copy_nonoverlapping(save_u.as_ptr(), u_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_v.as_ptr(), v_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_x.as_ptr(), x_b.add(es), excess);
                        std::ptr::copy_nonoverlapping(save_y.as_ptr(), y_b.add(es), excess);
                    }

                    bp += 64;
                }

                // Score and max tracking (exact H[] unless APPROX_MAX)
                let u8s = std::slice::from_raw_parts(u_b as *const u8, target_len);
                let v8s = std::slice::from_raw_parts(v_b as *const u8, target_len);
                if track.update(result, u8s, v8s, r, st0, en0) { break; }

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
/// AVX-512/AVX2/SSE on x86_64, NEON on aarch64, SIMD128 on wasm32, the scalar kernel
/// elsewhere; all produce identical results.
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

/// Scalar single-affine extension alignment.
///
/// A safe, lane-by-lane version of the SSE kernel: each 128-bit register is a `[i8; 16]`
/// and every load/store is a bounds-checked copy into one buffer laid out like the SIMD
/// allocation (u, v, x, y, s, sf, qr), so block rounding, leftover scores outside the
/// band and the score-store overrun into `sf` all match. Score bookkeeping goes through
/// [`SingleAffineTrack`].
///
/// This is the reference the SIMD kernels are tested against, the fallback for targets
/// without a verified SIMD kernel, and the path for `RAMMAP_FORCE_SCALAR=1`.
#[allow(clippy::too_many_arguments)]
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
    type V = [i8; 16];
    fn ld(buf: &[u8], at: usize) -> V { std::array::from_fn(|k| buf[at + k] as i8) }
    fn st16(buf: &mut [u8], at: usize, v: V) { for k in 0..16 { buf[at + k] = v[k] as u8; } }
    fn add(a: V, b: V) -> V { std::array::from_fn(|k| a[k].wrapping_add(b[k])) }
    fn sub(a: V, b: V) -> V { std::array::from_fn(|k| a[k].wrapping_sub(b[k])) }
    fn max_s(a: V, b: V) -> V { std::array::from_fn(|k| a[k].max(b[k])) }
    fn max_u(a: V, b: V) -> V { std::array::from_fn(|k| (a[k] as u8).max(b[k] as u8) as i8) }
    fn min_u(a: V, b: V) -> V { std::array::from_fn(|k| (a[k] as u8).min(b[k] as u8) as i8) }
    /// `(v << 1 byte) | carry`, returning the shifted vector and the new carry (`v >> 15 bytes`)
    fn shift_in(v: V, carry: i8) -> (V, i8) { (std::array::from_fn(|k| if k == 0 { carry } else { v[k - 1] }), v[15]) }

    init_dp_result_full(result);
    let query_len = qseq.len();
    let target_len = tseq.len();
    if alphabet_size <= 0 || query_len == 0 || target_len == 0 {
        return;
    }
    let m = alphabet_size as usize;
    let (q, e) = (gap_open as i8, gap_extend as i8);
    let qe = gap_open + gap_extend;

    // Skip alignment when a mismatch could never beat two gaps
    let min_sc = score_matrix[1..m * m].iter().fold(score_matrix[1], |a, &b| a.min(b)) as i32;
    if -min_sc > 2 * qe { return; }

    let q_: V = [q; 16];
    let qe2_: V = [(qe * 2) as i8; 16];
    let max_sc_: V = [(score_matrix[0] as i32 + qe * 2) as i8; 16];
    let (sc_mch, sc_mis) = (score_matrix[0], score_matrix[1]);
    let sc_n = if score_matrix[m * m - 1] == 0 { -e } else { score_matrix[m * m - 1] };
    let m1 = alphabet_size - 1;

    let w = if bandwidth < 0 { target_len.max(query_len) as i32 } else { bandwidth };
    let tlen_ = target_len.div_ceil(16);
    let n_col_ = query_len.min(target_len).min((w + 1) as usize).div_ceil(16) + 1;
    let with_cigar = (flags & SCORE_ONLY) == 0;
    let right_align = (flags & RIGHT_ALIGN) != 0;

    // One zeroed buffer with the SIMD layout: u, v, x, y, s, then sf, then qr
    let qlen_ = query_len.div_ceil(16);
    let (u, v, x, y, s) = (0, tlen_ * 16, 2 * tlen_ * 16, 3 * tlen_ * 16, 4 * tlen_ * 16);
    let sf = 5 * tlen_ * 16;
    let qr = sf + tlen_ * 16;
    let mut mem = vec![0u8; qr + (qlen_ + 1) * 16];
    for t in 0..query_len {
        mem[qr + t] = qseq[query_len - 1 - t];
    }
    mem[sf..sf + target_len].copy_from_slice(tseq);

    let valid_range = query_len + target_len - 1;
    let mut p: Vec<u8> = if with_cigar { vec![0u8; valid_range * n_col_ * 16] } else { Vec::new() };
    let mut band_off: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };
    let mut band_off_end: Vec<i32> = if with_cigar { vec![0i32; valid_range] } else { Vec::new() };
    let mut track = SingleAffineTrack::new(flags, query_len, target_len, gap_open, gap_extend, z_drop);

    let mut last_st: i32 = -1;
    let mut last_en: i32 = -1;
    for r in 0..valid_range as i32 {
        let mut st = 0i32;
        let mut en = target_len as i32 - 1;
        // qrr = qr + (qlen - 1 - r); qrr[t] is only read for t >= st0 >= r - qlen + 1
        let qrr = qr as isize + query_len as isize - 1 - r as isize;

        if st < r - query_len as i32 + 1 { st = r - query_len as i32 + 1; }
        if en > r { en = r; }
        if st < (r - w + 1) >> 1 { st = (r - w + 1) >> 1; }
        if en > (r + w) >> 1 { en = (r + w) >> 1; }
        if st > en {
            result.zdropped = 1;
            break;
        }
        let (st0, en0) = (st, en);
        st = st / 16 * 16;
        en = (en + 16) / 16 * 16 - 1;

        // Boundary conditions
        let (x1, v1): (i8, i8) = if st > 0 {
            if st - 1 >= last_st && st - 1 <= last_en {
                (mem[x + (st - 1) as usize] as i8, mem[v + (st - 1) as usize] as i8)
            } else { (0, 0) }
        } else { (0, if r > 0 { q } else { 0 }) };
        if en >= r {
            mem[y + r as usize] = 0;
            mem[u + r as usize] = if r > 0 { q as u8 } else { 0 };
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
            for t in st0 as usize..=en0 as usize {
                mem[s + t] = score_matrix[mem[sf + t] as usize * m + mem[(qrr + t as isize) as usize] as usize] as u8;
            }
        }

        // Core DP over whole 16-cell blocks
        let (mut x1_, mut v1_) = (x1, v1);
        let (st_, en_) = (st as usize / 16, en as usize / 16);
        if with_cigar {
            band_off[r as usize] = st;
            band_off_end[r as usize] = en;
        }
        for ti in st_..=en_ {
            let o = ti * 16;
            // __dp_code_block1
            let mut z = add(ld(&mem, s + o), qe2_);
            let (xt1, vt1);
            (xt1, x1_) = shift_in(ld(&mem, x + o), x1_);
            (vt1, v1_) = shift_in(ld(&mem, v + o), v1_);
            let mut a = add(xt1, vt1);
            let ut = ld(&mem, u + o);
            let mut b = add(ld(&mem, y + o), ut);

            let mut d: V = [0; 16];
            if with_cigar && !right_align {
                for k in 0..16 { if a[k] > z[k] { d[k] = 1; } }     // d = a > z? 1 : 0
                z = max_s(z, a);
                for k in 0..16 { if b[k] > z[k] { d[k] = 2; } }     // d = b > z? 2 : d
            } else if with_cigar {
                for k in 0..16 { d[k] = if z[k] > a[k] { 0 } else { 1 }; } // d = z > a? 0 : 1
                z = max_s(z, a);
                for k in 0..16 { if !(z[k] > b[k]) { d[k] = 2; } }  // d = z > b? d : 2
            } else {
                z = max_s(z, a);
            }

            // __dp_code_block2
            z = max_u(z, b);
            z = min_u(z, max_sc_);
            st16(&mut mem, u + o, sub(z, vt1));
            st16(&mut mem, v + o, sub(z, ut));
            z = sub(z, q_);
            a = sub(a, z);
            b = sub(b, z);

            let (xa, yb): (V, V);
            if with_cigar && right_align {
                xa = std::array::from_fn(|k| if 0 > a[k] { 0 } else { a[k] });
                yb = std::array::from_fn(|k| if 0 > b[k] { 0 } else { b[k] });
                for k in 0..16 {
                    if !(0 > a[k]) { d[k] |= 0x08; }
                    if !(0 > b[k]) { d[k] |= 0x10; }
                }
            } else {
                xa = std::array::from_fn(|k| if a[k] > 0 { a[k] } else { 0 });
                yb = std::array::from_fn(|k| if b[k] > 0 { b[k] } else { 0 });
                if with_cigar {
                    for k in 0..16 {
                        if a[k] > 0 { d[k] |= 0x08; }
                        if b[k] > 0 { d[k] |= 0x10; }
                    }
                }
            }
            st16(&mut mem, x + o, xa);
            st16(&mut mem, y + o, yb);
            if with_cigar {
                let pr = (r as usize * n_col_ + ti - st_) * 16;
                for k in 0..16 { p[pr + k] = d[k] as u8; }
            }
        }

        if track.update(result, &mem[u..u + target_len], &mem[v..v + target_len], r, st0, en0) { break; }
        last_st = st;
        last_en = en;
    }

    if with_cigar {
        traceback_single_affine_safe(
            result, query_len, target_len, end_bonus, flags, n_col_ * 16,
            &p, &band_off, &band_off_end,
        );
    }
}

