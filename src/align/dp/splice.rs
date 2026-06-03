// Splice-aware DP kernels

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "wasm32")]
use super::common::simd_compat::*;

use super::common::*;

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
    if *crate::align::env_flags::FORCE_SCALAR {
        extend_splice_scalar(qseq, tseq, alphabet_size, score_matrix,
            gap_open as i32, gap_extend as i32, gap_open2 as i32,
            noncanon_penalty as i32, z_drop, end_bonus,
            junc_bonus, junc_pen, flags, junc, result);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::use_avx512() {
            unsafe { extend_splice_avx512_fn(qseq, tseq, alphabet_size, score_matrix, gap_open, gap_extend, gap_open2, noncanon_penalty, z_drop, end_bonus, junc_bonus, junc_pen, flags, junc, result); }
        } else if super::use_avx2() {
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
        pub(super) unsafe fn $fn_name(
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
extend_splice_impl!(extend_splice41_impl, _mm_max_epi8, true, "sse4.1");
#[cfg(target_arch = "wasm32")]
extend_splice_impl!(extend_splice_wasm_impl, _mm_max_epi8, true, "simd128");

// ============================================================================
// AVX2 Implementation - Splice-Aware Alignment
// ============================================================================

#[cfg(target_arch = "x86_64")]
macro_rules! extend_splice_avx2_impl {
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
        pub(super) unsafe fn $fn_name(
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
pub(super) unsafe fn extend_splice_neon_impl(
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
        vdupq_n_u8(-gap_extend as u8)
    } else {
        vdupq_n_u8(score_matrix[(alphabet_size as usize * alphabet_size as usize) - 1] as u8)
    };
    let m1_ = vdupq_n_u8((alphabet_size - 1) as u8);

    let tlen_ = target_len.div_ceil(16);
    let qlen_ = query_len.div_ceil(16);
    let n_col_ = query_len.min(target_len).div_ceil(16) + 1;

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
        let sp: [i32; 4] = if (flags & SPLICE_COMPLEX) != 0 {
            let sp0 = [8, 15, 21, 30];
            [
                (sp0[0] as f64 / 3.0 + 0.499) as i32,
                (sp0[1] as f64 / 3.0 + 0.499) as i32,
                (sp0[2] as f64 / 3.0 + 0.499) as i32,
                (sp0[3] as f64 / 3.0 + 0.499) as i32,
            ]
        } else {
            let sp0 = if (flags & SPLICE_FLANK) != 0 { noncanon_penalty as i32 / 2 } else { 0 };
            [sp0, noncanon_penalty as i32, noncanon_penalty as i32, noncanon_penalty as i32]
        };

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
