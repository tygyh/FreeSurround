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

use std::f32;

pub const MAXFACTORS: usize = 32;

// Complex number type for FFT
#[derive(Debug, Clone, Copy)]
pub struct KissFftCpx {
    pub r: f32,
    pub i: f32,
}

impl KissFftCpx {
    pub fn new(r: f32, i: f32) -> Self {
        KissFftCpx { r, i }
    }

    pub fn zero() -> Self {
        KissFftCpx { r: 0.0, i: 0.0 }
    }
}

// FFT state configuration
pub struct KissFftState {
    pub nfft: usize,
    pub inverse: bool,
    pub factors: Vec<usize>,
    pub twiddles: Vec<KissFftCpx>,
}

// Helper functions for complex arithmetic
pub fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

pub fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

pub fn c_mul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

pub fn c_mulbyscalar(c: KissFftCpx, s: f32) -> KissFftCpx {
    KissFftCpx {
        r: c.r * s,
        i: c.i * s,
    }
}

pub fn half_of(x: f32) -> f32 {
    x * 0.5
}

pub fn kf_cexp(phase: f64) -> KissFftCpx {
    KissFftCpx {
        r: phase.cos() as f32,
        i: phase.sin() as f32,
    }
}
