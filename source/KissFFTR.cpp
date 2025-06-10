/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted
provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
this list of conditions
and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
this list of
conditions and the following disclaimer in the documentation and/or other
materials provided with
the distribution.
    * Neither the author nor the names of any contributors may be used to
endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <iostream>
#include <ostream>

#include "../include/FreeSurround/KissFFTR.h"
#include "../include/FreeSurround/_KissFFTGuts.h"

struct kiss_fftr_state
{
    kiss_fft_cfg substate;
    kiss_fft_cpx *tmpbuf;
    kiss_fft_cpx *super_twiddles;
#ifdef USE_SIMD
    void *pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft, const int inverse_fft, void *mem, size_t *lenmem)
{
    kiss_fftr_cfg st = nullptr;
    size_t subsize = 65536 * 4;
    size_t memneeded = 0;

    if (nfft & 1)
    {
        std::println(std::cerr, "Real FFT optimization must be even.");
        return nullptr;
    }
    nfft >>= 1;

    kiss_fft_alloc(nfft, inverse_fft, nullptr, &subsize);
    memneeded = sizeof(kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * (nfft * 3 / 2);

    if (lenmem == nullptr)
    {
        st = static_cast<kiss_fftr_cfg>(operator new(memneeded));
    }
    else
    {
        if (*lenmem >= memneeded)
            st = static_cast<kiss_fftr_cfg>(mem);
        *lenmem = memneeded;
    }
    if (!st)
        return nullptr;

    st->substate = std::bit_cast<kiss_fft_cfg>(st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = std::bit_cast<kiss_fft_cpx *>(std::bit_cast<char *>(st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

    for (int i = 0; i < nfft / 2; ++i)
    {
        double phase = -pi * (static_cast<double>(i + 1) / nfft + .5);
        if (inverse_fft)
            phase *= -1;
        kf_cexp(st->super_twiddles + i, phase);
    }
    return st;
}

void kiss_fftr(kiss_fftr_cfg cfg, const kiss_fft_scalar *timedata, kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    kiss_fft_cpx tdc;

    if (cfg->substate->inverse)
    {
        std::println(std::cerr, "kiss fft usage error: improper alloc");
        exit(1);
    }

    int ncfft = cfg->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft(cfg->substate, std::bit_cast<const kiss_fft_cpx *>(timedata), cfg->tmpbuf);
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product)
     * [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = cfg->tmpbuf[0].r;
    tdc.i = cfg->tmpbuf[0].i;
    c_fixdiv(tdc, 2);
    check_overflow_add(tdc.r, tdc.i);
    check_overflow_sub(tdc.r, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for (int k = 1; k <= ncfft / 2; ++k)
    {
        kiss_fft_cpx fpnk;
        const kiss_fft_cpx fpk = cfg->tmpbuf[k];
        fpnk.r = cfg->tmpbuf[ncfft - k].r;
        fpnk.i = -cfg->tmpbuf[ncfft - k].i;
        c_fixdiv(fpk, 2);
        c_fixdiv(fpnk, 2);

        const auto [f1k_r, f1k_i] = c_add(fpk, fpnk);
        const kiss_fft_cpx f2k = c_sub(fpk, fpnk);
        const auto [tw_r, tw_i] = c_mul(f2k, cfg->super_twiddles[k - 1]);

        freqdata[k].r = half_of(f1k_r + tw_r);
        freqdata[k].i = half_of(f1k_i + tw_i);
        freqdata[ncfft - k].r = half_of(f1k_r - tw_r);
        freqdata[ncfft - k].i = half_of(tw_i - f1k_i);
    }
}

void kiss_fftri(kiss_fftr_cfg cfg, const kiss_fft_cpx *freqdata, kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */

    if (cfg->substate->inverse == 0)
    {
        std::println(std::cerr, "kiss fft usage error: improper alloc");
        exit(1);
    }

    int ncfft = cfg->substate->nfft;

    cfg->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    cfg->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    c_fixdiv(st->tmpbuf[0], 2);

    for (int k = 1; k <= ncfft / 2; ++k)
    {
        kiss_fft_cpx fnkc;
        kiss_fft_cpx fek;
        kiss_fft_cpx fok;
        kiss_fft_cpx tmp;
        const kiss_fft_cpx fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        c_fixdiv(fk, 2);
        c_fixdiv(fnkc, 2);

        fek = c_add(fk, fnkc);
        tmp=c_sub( fk, fnkc);
        fok=c_mul( tmp, cfg->super_twiddles[k - 1]);
        cfg->tmpbuf[k]=c_add( fek, fok);
        cfg->tmpbuf[ncfft - k]=c_sub( fek, fok);
#ifdef USE_SIMD
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
        cfg->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft(cfg->substate, cfg->tmpbuf, std::bit_cast<kiss_fft_cpx *>(timedata));
}
