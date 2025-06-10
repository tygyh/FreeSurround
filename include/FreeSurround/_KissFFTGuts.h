/*
Copyright (c) 2003-2010, Mark Borgerding

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

/* kiss_fft.h
   defines kiss_fft_scalar as either short or a float type
   and defines
   typedef struct { kiss_fft_scalar r; kiss_fft_scalar i; }kiss_fft_cpx; */
#pragma once
#include "KissFFT.h"

#include <array>
#include <source_location>
#if defined(USE_SIMD)
#include <xmmintrin.h>
#endif

constexpr int MAXFACTORS = 32;
/* e.g. an fft of length 128 has 4 factors
 as far as kissfft is concerned
 4*4*4*2
 */

struct kiss_fft_state
{
    int nfft;
    int inverse;
    std::array<int, 2 * MAXFACTORS> factors;
    std::array<kiss_fft_cpx, 1> twiddles;
};

#ifdef FIXED_POINT
#if (FIXED_POINT == 32)
constexpr int FRACBITS = 31;
constexpr auto SAMPPROD = int64_t;
constexpr int SAMP_MAX = 2147483647;
#else
constexpr int FRACBITS = 15;
constexpr auto SAMPPROD = int32_t;
constexpr int SAMP_MAX = 32767;
#endif

constexpr int SAMP_MIN = -SAMP_MAX;

#if defined(CHECK_OVERFLOW)
template <typename T>
void check_overflow_add(T a, T b, const std::source_location &loc = std::source_location::current())
{
    SAMPPROD result = static_cast<SAMPPROD>(a) + static_cast<SAMPPROD>(b);
    if (result > SAMP_MAX || result < SAMP_MIN)
    {
        fprintf(stderr, "WARNING:overflow @ %s(%u): (%d + %d) = %ld\n", loc.file_name(), loc.line(),
                static_cast<int>(a), static_cast<int>(b), result);
    }
}

template <typename T>
void check_overflow_sub(T a, T b, const std::source_location &loc = std::source_location::current())
{
    SAMPPROD result = static_cast<SAMPPROD>(a) - static_cast<SAMPPROD>(b);
    if (result > SAMP_MAX || result < SAMP_MIN)
    {
        fprintf(stderr, "WARNING:overflow @ %s(%u): (%d - %d) = %ld\n", loc.file_name(), loc.line(),
                static_cast<int>(a), static_cast<int>(b), result);
    }
}

template <typename T>
void check_overflow_mul(T a, T b, const std::source_location &loc = std::source_location::current())
{
    SAMPPROD result = static_cast<SAMPPROD>(a) * static_cast<SAMPPROD>(b);
    if (result > SAMP_MAX || result < SAMP_MIN)
    {
        fprintf(stderr, "WARNING:overflow @ %s(%u): (%d * %d) = %ld\n", loc.file_name(), loc.line(),
                static_cast<int>(a), static_cast<int>(b), result);
    }
}
#endif

template <typename T>
constexpr SAMPPROD smul(T a, T b)
{
    return static_cast<SAMPPROD>(a) * static_cast<SAMPPROD>(b);
}

template <typename T>
constexpr kiss_fft_scalar sround(T x)
{
    return static_cast<kiss_fft_scalar>((x + (1 << (FRACBITS - 1))) >> FRACBITS);
}

template <typename T>
T s_mul(T a, T b)
{
    return sround(smul(a, b));
}

template <typename ComplexType>
ComplexType c_mul(const ComplexType &a, const ComplexType &b)
{
    return {sround(smul(a.r, b.r) - smul(a.i, b.i)), sround(smul(a.r, b.i) + smul(a.i, b.r))};
}

template <typename T>
void divscalar(T &x, int k)
{
    x = sround(smul(x, SAMP_MAX / k));
}

template <typename ComplexType>
void c_fixdiv(ComplexType &c, int div)
{
    divscalar(c.r, div);
    divscalar(c.i, div);
}

template <typename ComplexType, typename ScalarType>
ComplexType c_mulbyscalar(const ComplexType &c, ScalarType s)
{
    return {sround(smul(c.r, s)), sround(smul(c.i, s))};
}

#else /* not FIXED_POINT*/

template <typename T>
constexpr T s_mul(T a, T b)
{
    return a * b;
}
template <typename ComplexType>
ComplexType c_mul(const ComplexType &a, const ComplexType &b,
                  [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
template <typename ComplexType>
void c_fixdiv([[maybe_unused]] ComplexType &c, [[maybe_unused]] int div,
              [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
    // No operation for floating point
}
template <typename ComplexType, typename ScalarType>
ComplexType c_mulbyscalar(const ComplexType &c, ScalarType s,
                          [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
    return {c.r * s, c.i * s};
}
#endif

#ifndef CHECK_OVERFLOW_OP
template <typename T>
void check_overflow_add([[maybe_unused]] T a, [[maybe_unused]] T b,
                        [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
}

template <typename T>
void check_overflow_sub([[maybe_unused]] T a, [[maybe_unused]] T b,
                        [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
}

template <typename T>
void check_overflow_mul([[maybe_unused]] T a, [[maybe_unused]] T b,
                        [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
    // No operation - gets optimized away in release builds
}

template <typename T, typename Op>
void check_overflow_op([[maybe_unused]] T a, [[maybe_unused]] T b, [[maybe_unused]] Op op,
                       [[maybe_unused]] const char *op_str,
                       [[maybe_unused]] const std::source_location &loc = std::source_location::current())
{
    // No operation - gets optimized away in release builds
}
#endif

template <typename ComplexType>
ComplexType c_add(const ComplexType &a, const ComplexType &b,
                  const std::source_location &loc = std::source_location::current())
{
    check_overflow_add(a.r, b.r, loc);
    check_overflow_add(a.i, b.i, loc);
    return {a.r + b.r, a.i + b.i};
}

template <typename ComplexType>
ComplexType c_sub(const ComplexType &a, const ComplexType &b,
                  const std::source_location &loc = std::source_location::current())
{
    check_overflow_sub(a.r, b.r, loc);
    check_overflow_sub(a.i, b.i, loc);
    return {a.r - b.r, a.i - b.i};
}

#ifdef FIXED_POINT
template <typename T>
constexpr SAMP kiss_fft_cos(T phase)
{
    return static_cast<SAMP>(std::floor(0.5 + SAMP_MAX * std::cos(phase)));
}

template <typename T>
constexpr SAMP kiss_fft_sin(T phase)
{
    return static_cast<SAMP>(std::floor(0.5 + SAMP_MAX * std::sin(phase)));
}

template <typename T>
constexpr T half_of(T x)
{
    return x >> 1;
}
#elif defined(USE_SIMD)
template <typename T>
__m128 kiss_fft_cos(T phase)
{
    return _mm_set1_ps(std::cos(phase));
}

template <typename T>
__m128 kiss_fft_sin(T phase)
{
    return _mm_set1_ps(std::sin(phase));
}

inline __m128 half_of(__m128 x) { return _mm_mul_ps(x, _mm_set1_ps(0.5f)); }
#else
template <typename T>
constexpr kiss_fft_scalar kiss_fft_cos(T phase)
{
    return static_cast<kiss_fft_scalar>(std::cos(phase));
}

template <typename T>
constexpr kiss_fft_scalar kiss_fft_sin(T phase)
{
    return static_cast<kiss_fft_scalar>(std::sin(phase));
}

template <typename T>
constexpr T half_of(T x)
{
    return x * static_cast<T>(0.5);
}
#endif

template <typename ComplexType, typename PhaseType>
ComplexType kf_cexp(PhaseType phase)
{
    return {kiss_fft_cos(phase), kiss_fft_sin(phase)};
}

/* a debugging function */
template <typename ComplexType>
void pcpx_debug(const ComplexType *c, const std::source_location &loc = std::source_location::current())
{
    c ? std::fprintf(stderr, "%g + %gi (at %s:%u)\n", static_cast<double>(c->r), static_cast<double>(c->i),
                     loc.file_name(), loc.line())
      : std::fprintf(stderr, "null complex pointer (at %s:%u)\n", loc.file_name(), loc.line());
}
#ifdef KISS_FFT_USE_ALLOCA
// define this to allow use of alloca instead of malloc for temporary buffers
// Temporary buffers are used in two case:
// 1. FFT sizes that have "bad" factors. i.e. not 2,3 and 5
// 2. "in-place" FFTs.  Notice the quotes, since kissfft does not really do an
// in-place transform.
#include <alloca.h>
inline void *kiss_fft_tmp_alloc(size_t nbytes) { return alloca(nbytes); }
inline void kiss_fft_tmp_free([[maybe_unused]] void *ptr) {}
#else
inline void *kiss_fft_tmp_alloc(const size_t nbytes) { return kiss_fft_malloc(nbytes); }
inline void kiss_fft_tmp_free(void *ptr) { kiss_fft_free(ptr); }
#endif
