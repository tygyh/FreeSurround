/*
Copyright (c) 2003-2010, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use super::types::*;
use std::f64::consts::PI;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn pollards_rho(n: usize) -> usize {
    if n % 2 == 0 {
        return 2;
    }

    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    let mut x = (seed as i64 % (n as i64 - 1)) + 1;
    let mut y = x;
    let c = (seed as i64 / 1000 % (n as i64 - 1)) + 1;
    let mut d = 1i64;

    while d == 1 {
        x = (x * x + c) % n as i64;
        y = (y * y + c) % n as i64;
        y = (y * y + c) % n as i64;
        d = gcd((x - y).abs(), n as i64);
    }

    if d == n as i64 {
        0
    } else {
        d as usize
    }
}

fn kf_factor(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    
    while n > 1 {
        let factor = pollards_rho(n);
        if factor == 0 {
            break;
        }

        while n % factor == 0 {
            n /= factor;
            factors.push(factor);
            factors.push(n);
        }
    }
    
    factors
}

fn kf_bfly2(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: usize) {
    let mut tw1_idx = 0;
    let mut fout_idx = 0;
    let mut fout2_idx = m;

    for _ in 0..m {
        let t = c_mul(fout[fout2_idx], st.twiddles[tw1_idx]);
        tw1_idx += fstride;
        fout[fout2_idx] = c_sub(fout[fout_idx], t);
        fout[fout_idx] = c_add(fout[fout_idx], t);
        fout2_idx += 1;
        fout_idx += 1;
    }
}

fn kf_bfly3(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: usize) {
    let m2 = 2 * m;
    let mut tw1_idx = 0;
    let mut tw2_idx = 0;
    let epi3 = st.twiddles[fstride * m];

    for fout_idx in 0..m {
        let scratch1 = c_mul(fout[fout_idx + m], st.twiddles[tw1_idx]);
        let scratch2 = c_mul(fout[fout_idx + m2], st.twiddles[tw2_idx]);

        let scratch3 = c_add(scratch1, scratch2);
        let scratch0 = c_sub(scratch1, scratch2);
        tw1_idx += fstride;
        tw2_idx += fstride * 2;

        fout[fout_idx + m].r = fout[fout_idx].r - half_of(scratch3.r);
        fout[fout_idx + m].i = fout[fout_idx].i - half_of(scratch3.i);

        let scratch0_mul = c_mulbyscalar(scratch0, epi3.i);

        fout[fout_idx] = c_add(fout[fout_idx], scratch3);

        fout[fout_idx + m2].r = fout[fout_idx + m].r + scratch0_mul.i;
        fout[fout_idx + m2].i = fout[fout_idx + m].i - scratch0_mul.r;

        fout[fout_idx + m].r -= scratch0_mul.i;
        fout[fout_idx + m].i += scratch0_mul.r;
    }
}

fn kf_bfly4(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: usize) {
    let m2 = 2 * m;
    let m3 = 3 * m;
    let mut tw1_idx = 0;
    let mut tw2_idx = 0;
    let mut tw3_idx = 0;

    for fout_idx in 0..m {
        let scratch0 = c_mul(fout[fout_idx + m], st.twiddles[tw1_idx]);
        let scratch1 = c_mul(fout[fout_idx + m2], st.twiddles[tw2_idx]);
        let scratch2 = c_mul(fout[fout_idx + m3], st.twiddles[tw3_idx]);

        let scratch5 = c_sub(fout[fout_idx], scratch1);
        fout[fout_idx] = c_add(fout[fout_idx], scratch1);
        let scratch3 = c_add(scratch0, scratch2);
        let scratch4 = c_sub(scratch0, scratch2);
        fout[fout_idx + m2] = c_sub(fout[fout_idx], scratch3);
        tw1_idx += fstride;
        tw2_idx += fstride * 2;
        tw3_idx += fstride * 3;
        fout[fout_idx] = c_add(fout[fout_idx], scratch3);

        if st.inverse {
            fout[fout_idx + m].r = scratch5.r - scratch4.i;
            fout[fout_idx + m].i = scratch5.i + scratch4.r;
            fout[fout_idx + m3].r = scratch5.r + scratch4.i;
            fout[fout_idx + m3].i = scratch5.i - scratch4.r;
        } else {
            fout[fout_idx + m].r = scratch5.r + scratch4.i;
            fout[fout_idx + m].i = scratch5.i - scratch4.r;
            fout[fout_idx + m3].r = scratch5.r - scratch4.i;
            fout[fout_idx + m3].i = scratch5.i + scratch4.r;
        }
    }
}

fn kf_bfly5(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: usize) {
    let ya = st.twiddles[fstride * m];
    let yb = st.twiddles[fstride * 2 * m];

    for u in 0..m {
        let fout0_idx = u;
        let fout1_idx = u + m;
        let fout2_idx = u + 2 * m;
        let fout3_idx = u + 3 * m;
        let fout4_idx = u + 4 * m;

        let scratch0 = fout[fout0_idx];

        let scratch1 = c_mul(fout[fout1_idx], st.twiddles[u * fstride]);
        let scratch2 = c_mul(fout[fout2_idx], st.twiddles[2 * u * fstride]);
        let scratch3 = c_mul(fout[fout3_idx], st.twiddles[3 * u * fstride]);
        let scratch4 = c_mul(fout[fout4_idx], st.twiddles[4 * u * fstride]);

        let scratch7 = c_add(scratch1, scratch4);
        let scratch10 = c_sub(scratch1, scratch4);
        let scratch8 = c_add(scratch2, scratch3);
        let scratch9 = c_sub(scratch2, scratch3);

        fout[fout0_idx].r += scratch7.r + scratch8.r;
        fout[fout0_idx].i += scratch7.i + scratch8.i;

        let scratch5 = KissFftCpx {
            r: scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
            i: scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
        };

        let scratch6 = KissFftCpx {
            r: scratch10.i * ya.i + scratch9.i * yb.i,
            i: -scratch10.r * ya.i - scratch9.r * yb.i,
        };

        fout[fout1_idx] = c_sub(scratch5, scratch6);
        fout[fout4_idx] = c_add(scratch5, scratch6);

        let scratch11 = KissFftCpx {
            r: scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
            i: scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
        };
        let scratch12 = KissFftCpx {
            r: -scratch10.i * yb.i + scratch9.i * ya.i,
            i: scratch10.r * yb.i - scratch9.r * ya.i,
        };

        fout[fout2_idx] = c_add(scratch11, scratch12);
        fout[fout3_idx] = c_sub(scratch11, scratch12);
    }
}

fn kf_bfly_generic(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: usize, p: usize) {
    let norig = st.nfft;
    let mut scratch = vec![KissFftCpx::zero(); p];

    for u in 0..m {
        let mut i = u;
        for q1 in 0..p {
            scratch[q1] = fout[i];
            i += m;
        }

        let mut j = u;
        for _ in 0..p {
            let mut twidx = 0;
            fout[j] = scratch[0];
            for q in 1..p {
                twidx += fstride * j;
                if twidx >= norig {
                    twidx -= norig;
                }
                let t = c_mul(scratch[q], st.twiddles[twidx]);
                fout[j] = c_add(fout[j], t);
            }
            j += m;
        }
    }
}

fn kf_work(
    fout: &mut [KissFftCpx],
    f: &[KissFftCpx],
    f_offset: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[usize],
    factor_idx: usize,
    st: &KissFftState,
) {
    if factor_idx >= factors.len() {
        return;
    }

    let p = factors[factor_idx];
    let m = factors[factor_idx + 1];

    if m == 1 {
        for i in 0..p {
            let idx = f_offset + fstride * in_stride * i;
            if idx < f.len() {
                fout[i] = f[idx];
            }
        }
    } else {
        for i in 0..p {
            kf_work(
                &mut fout[i * m..(i + 1) * m],
                f,
                f_offset + fstride * in_stride * i,
                fstride * p,
                in_stride,
                factors,
                factor_idx + 2,
                st,
            );
        }
    }

    match p {
        2 => kf_bfly2(fout, fstride, st, m),
        3 => kf_bfly3(fout, fstride, st, m),
        4 => kf_bfly4(fout, fstride, st, m),
        5 => kf_bfly5(fout, fstride, st, m),
        _ => kf_bfly_generic(fout, fstride, st, m, p),
    }
}

pub fn kiss_fft_alloc(nfft: usize, inverse: bool) -> KissFftState {
    let mut twiddles = vec![KissFftCpx::zero(); nfft];

    for i in 0..nfft {
        let mut phase = -2.0 * PI * i as f64 / nfft as f64;
        if inverse {
            phase *= -1.0;
        }
        twiddles[i] = kf_cexp(phase);
    }

    let factors = kf_factor(nfft);

    KissFftState {
        nfft,
        inverse,
        factors,
        twiddles,
    }
}

pub fn kiss_fft(cfg: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    kiss_fft_stride(cfg, fin, fout, 1);
}

pub fn kiss_fft_stride(cfg: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx], fin_stride: usize) {
    if fin.as_ptr() != fout.as_ptr() {
        kf_work(fout, fin, 0, 1, fin_stride, &cfg.factors, 0, cfg);
    } else {
        let mut tmpbuf = vec![KissFftCpx::zero(); cfg.nfft];
        kf_work(&mut tmpbuf, fin, 0, 1, fin_stride, &cfg.factors, 0, cfg);
        fout.copy_from_slice(&tmpbuf);
    }
}

pub fn kiss_fft_next_fast_size(n: usize) -> usize {
    let mut hamming_numbers = vec![1];
    let mut i2 = 0;
    let mut i3 = 0;
    let mut i5 = 0;

    loop {
        let next2 = hamming_numbers[i2] * 2;
        let next3 = hamming_numbers[i3] * 3;
        let next5 = hamming_numbers[i5] * 5;

        let next_hamming = next2.min(next3).min(next5);

        if next_hamming >= n {
            return next_hamming;
        }

        hamming_numbers.push(next_hamming);

        if next_hamming == next2 {
            i2 += 1;
        }
        if next_hamming == next3 {
            i3 += 1;
        }
        if next_hamming == next5 {
            i5 += 1;
        }
    }
}
