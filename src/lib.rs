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
pub use channel_maps::ChannelSetup;
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

    #[test]
    fn test_decoder_initialization() {
        let mut decoder = DPL2FSDecoder::new();
        assert!(!decoder.buffered() > 0);
        
        // Initialize for 5.1 surround
        let result = decoder.init(ChannelSetup::FivePointOne, 4096, 48000);
        assert!(result.is_ok());
        
        // Verify parameters can be set
        decoder.set_circular_wrap(90.0);
        decoder.set_focus(0.5);
        decoder.set_bass_redirection(true);
    }

    #[test]
    fn test_decoder_flush() {
        let mut decoder = DPL2FSDecoder::new();
        let _ = decoder.init(ChannelSetup::FivePointOne, 1024, 44100);
        decoder.flush();
        assert_eq!(decoder.buffered(), 0);
    }

    #[test]
    fn test_channel_setup_5_1() {
        let mut decoder = DPL2FSDecoder::new();
        let result = decoder.init(ChannelSetup::FivePointOne, 2048, 48000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_channel_setup_7_1() {
        let mut decoder = DPL2FSDecoder::new();
        let result = decoder.init(ChannelSetup::SevenPointOne, 2048, 48000);
        assert!(result.is_ok());
    }
}
