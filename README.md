# FreeSurround

A Rust library for converting stereo audio to surround sound using the FreeSurround algorithm.

## Description

FreeSurround is an audio decoder that transforms stereo (2-channel) audio into multi-channel surround sound formats. It uses sophisticated DSP techniques including FFT analysis, phase/amplitude processing, and spatial positioning to extract and enhance spatial information from stereo recordings.

This is a Rust port of the original C++ implementation by Christian Kothe, incorporating the KissFFT library by Mark Borgerding.

## Features

- Convert stereo to 5.1 or 7.1 surround sound
- Configurable spatial parameters:
  - Circular wrap (soundstage angle)
  - Shift (forward/backward positioning)
  - Depth (soundstage extension)
  - Focus (localization precision)
  - Center image presence
  - Front/rear separation control
- Bass redirection (LFE channel support)
- Configurable LFE cutoff frequencies
- Real-time processing capability

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
freesurround = "0.1"
```

## Usage

```rust
use freesurround::{DPL2FSDecoder, ChannelSetup};

fn main() -> Result<(), String> {
    // Create decoder instance
    let mut decoder = DPL2FSDecoder::new();
    
    // Initialize for 5.1 output, 4096 sample blocksize, 48kHz sample rate
    decoder.init(
        ChannelSetup::FivePointOne,
        4096,
        48000
    )?;
    
    // Configure parameters (optional)
    decoder.set_circular_wrap(90.0);
    decoder.set_focus(0.0);
    decoder.set_bass_redirection(true);
    
    // Process stereo audio (2 * blocksize samples)
    let stereo_input: Vec<f32> = vec![0.0; 4096 * 2];
    let surround_output = decoder.decode(&stereo_input)?;
    
    // surround_output contains 6 * blocksize samples for 5.1
    // or 8 * blocksize samples for 7.1
    
    Ok(())
}
```

## Supported Channel Setups

- **5.1 Surround**: Front Left, Front Center, Front Right, Back Left, Back Right, LFE
- **7.1 Surround**: Front Left, Front Center, Front Right, Side Left, Side Right, Back Left, Back Right, LFE

## Algorithm Overview

The FreeSurround decoder works by:

1. Converting stereo input to frequency domain using FFT
2. Analyzing phase and amplitude differences between left/right channels
3. Mapping phase/amplitude space to spatial positions (x/y coordinates)
4. Distributing frequency components across surround channels based on spatial position
5. Converting back to time domain using inverse FFT
6. Applying overlap-add windowing for smooth output

## Parameters

### Spatial Parameters

- **circular_wrap** (0-360°, default 90°): Angle of front soundstage
- **shift** (-1 to 1, default 0): Forward/backward offset
- **depth** (0-5, default 1): Backward soundstage extension
- **focus** (0-1, default 0): Sound event localization precision

### Channel Parameters

- **center_image** (0-1, default 1): Center speaker presence
- **front_separation** (0-1, default 1): Front stereo width
- **rear_separation** (0-1, default 1): Rear stereo width

### LFE Parameters

- **low_cutoff** (20-150 Hz, default 40): LFE low frequency cutoff
- **high_cutoff** (40-300 Hz, default 90): LFE high frequency cutoff
- **bass_redirection** (bool, default true): Enable LFE channel

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## License

This project incorporates code from multiple sources:

- **FreeSurround Algorithm**: GPL-2.0-or-later
  - Copyright (C) 2007-2010 Christian Kothe

- **KissFFT Library**: BSD-3-Clause
  - Copyright (c) 2003-2010 Mark Borgerding

See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Original FreeSurround: https://github.com/AviSynth/FreeSurround
- KissFFT: https://github.com/mborgerding/kissfft

## Credits

- **Christian Kothe**: Original FreeSurround algorithm and implementation
- **Mark Borgerding**: KissFFT library
- Rust port contributors
