/*
Copyright (c) 2003-2004, Mark Borgerding

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

use super::fft::*;
use super::types::*;
use std::f64::consts::PI;

pub struct KissFftrState {
    pub substate: KissFftState,
    pub tmpbuf: Vec<KissFftCpx>,
    pub super_twiddles: Vec<KissFftCpx>,
}

pub fn kiss_fftr_alloc(nfft: usize, inverse_fft: bool) -> Result<KissFftrState, &'static str> {
    if nfft & 1 != 0 {
        return Err("Real FFT optimization must be even.");
    }

    let nfft_half = nfft >> 1;
    let substate = kiss_fft_alloc(nfft_half, inverse_fft);
    let tmpbuf = vec![KissFftCpx::zero(); nfft_half];
    let mut super_twiddles = vec![KissFftCpx::zero(); nfft_half / 2];

    for i in 0..(nfft_half / 2) {
        let mut phase = -PI * (i as f64 + 1.0) / nfft_half as f64 + (-PI * 0.5);
        if inverse_fft {
            phase *= -1.0;
        }
        super_twiddles[i] = kf_cexp(phase);
    }

    Ok(KissFftrState {
        substate,
        tmpbuf,
        super_twiddles,
    })
}

pub fn kiss_fftr(cfg: &mut KissFftrState, timedata: &[f32], freqdata: &mut [KissFftCpx]) {
    if cfg.substate.inverse {
        eprintln!("kiss fft usage error: improper alloc");
        std::process::exit(1);
    }

    let ncfft = cfg.substate.nfft;

    // Pack the real time data as complex
    let timedata_cpx: Vec<KissFftCpx> = timedata
        .chunks(2)
        .map(|chunk| KissFftCpx {
            r: chunk[0],
            i: if chunk.len() > 1 { chunk[1] } else { 0.0 },
        })
        .collect();

    // Perform the parallel FFT of two real signals packed in real,imag
    kiss_fft(&cfg.substate, &timedata_cpx, &mut cfg.tmpbuf);

    let tdc_r = cfg.tmpbuf[0].r;
    let tdc_i = cfg.tmpbuf[0].i;
    
    freqdata[0].r = tdc_r + tdc_i;
    freqdata[ncfft].r = tdc_r - tdc_i;
    freqdata[ncfft].i = 0.0;
    freqdata[0].i = 0.0;

    for k in 1..=(ncfft / 2) {
        let fpk = cfg.tmpbuf[k];
        let fpnk = KissFftCpx {
            r: cfg.tmpbuf[ncfft - k].r,
            i: -cfg.tmpbuf[ncfft - k].i,
        };

        let f1k = c_add(fpk, fpnk);
        let f2k = c_sub(fpk, fpnk);
        let tw = c_mul(f2k, cfg.super_twiddles[k - 1]);

        freqdata[k].r = half_of(f1k.r + tw.r);
        freqdata[k].i = half_of(f1k.i + tw.i);
        freqdata[ncfft - k].r = half_of(f1k.r - tw.r);
        freqdata[ncfft - k].i = half_of(tw.i - f1k.i);
    }
}

pub fn kiss_fftri(cfg: &mut KissFftrState, freqdata: &[KissFftCpx], timedata: &mut [f32]) {
    if !cfg.substate.inverse {
        eprintln!("kiss fft usage error: improper alloc");
        std::process::exit(1);
    }

    let ncfft = cfg.substate.nfft;

    cfg.tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    cfg.tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;

    for k in 1..=(ncfft / 2) {
        let fk = freqdata[k];
        let fnkc = KissFftCpx {
            r: freqdata[ncfft - k].r,
            i: -freqdata[ncfft - k].i,
        };

        let fek = c_add(fk, fnkc);
        let tmp = c_sub(fk, fnkc);
        let fok = c_mul(tmp, cfg.super_twiddles[k - 1]);
        cfg.tmpbuf[k] = c_add(fek, fok);
        cfg.tmpbuf[ncfft - k] = c_sub(fek, fok);
        cfg.tmpbuf[ncfft - k].i *= -1.0;
    }

    let mut timedata_cpx = vec![KissFftCpx::zero(); ncfft];
    kiss_fft(&cfg.substate, &cfg.tmpbuf, &mut timedata_cpx);

    // Unpack the complex data back to real
    for (i, cpx) in timedata_cpx.iter().enumerate() {
        if 2 * i < timedata.len() {
            timedata[2 * i] = cpx.r;
        }
        if 2 * i + 1 < timedata.len() {
            timedata[2 * i + 1] = cpx.i;
        }
    }
}
