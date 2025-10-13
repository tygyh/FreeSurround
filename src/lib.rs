/*
FreeSurround - Audio decoder that converts stereo to surround sound

Copyright (C) 2007-2010 Christian Kothe
Copyright (c) 2003-2010 Mark Borgerding (KissFFT)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
*/

pub mod kiss_fft;
pub mod channel_maps;
pub mod decoder;

pub use decoder::*;
pub use kiss_fft::{KissFftCpx, KissFftState, KissFftrState};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_allocation() {
        let cfg = kiss_fft::kiss_fft_alloc(256, false);
        assert_eq!(cfg.nfft, 256);
        assert!(!cfg.inverse);
    }
}
