// src/filters.rs
//! Pre/post compression delta filters..
//!
//! Applied to the raw input BEFORE folding, reversed AFTER unfolding.
//! Converts smooth-varying binary data (PCM audio, BMP pixels) into
//! near-zero residuals that LZ can match at dramatically higher density.
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1  (generic 8-bit binary)
//!   2 = delta stride 2  (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3  (24-bit RGB pixels)
//!   4 = delta stride 4  (32-bit RGBA / stereo 16-bit PCM)

pub const FILTER_NONE:   u8 = 0;
pub const FILTER_DELTA1: u8 = 1;
pub const FILTER_DELTA2: u8 = 2;
pub const FILTER_DELTA3: u8 = 3;
pub const FILTER_DELTA4: u8 = 4;

/// Inspect magic bytes and file headers to determine the best delta stride.
/// Returns FILTER_NONE for unknown or uncompressible formats.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // WAV / RIFF audio
    if &input[0..4] == b"RIFF" && input.len() >= 12 && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }

    // BMP image
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    FILTER_NONE
}

fn detect_wav_stride(input: &[u8]) -> u8 {
    let mut pos = 12usize;
    while pos + 8 <= input.len() {
        let id        = &input[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([
            input[pos + 4], input[pos + 5], input[pos + 6], input[pos + 7],
        ]) as usize;

        if id == b"fmt " && chunk_len >= 16 && pos + 8 + 16 <= input.len() {
            let channels    = u16::from_le_bytes([input[pos + 10], input[pos + 11]]);
            let bits_sample = u16::from_le_bytes([input[pos + 22], input[pos + 23]]);
            let stride      = (channels as usize) * (bits_sample as usize / 8);
            println!("WAV fmt: {} ch, {} bps → delta stride {}", channels, bits_sample, stride);
            return match stride {
                1 => FILTER_DELTA1,
                2 => FILTER_DELTA2,
                3 => FILTER_DELTA3,
                4 => FILTER_DELTA4,
                _ => FILTER_DELTA2,
            };
        }

        pos += 8 + chunk_len;
        if chunk_len % 2 != 0 { pos += 1; }
    }

    println!("WAV: fmt chunk not found, falling back to delta2");
    FILTER_DELTA2
}

fn detect_bmp_stride(input: &[u8]) -> u8 {
    let bpp = u16::from_le_bytes([input[28], input[29]]);
    println!("BMP: {} bpp → delta stride {}", bpp, (bpp as usize / 8).max(1));
    match bpp {
        8  => FILTER_DELTA1,
        16 => FILTER_DELTA2,
        24 => FILTER_DELTA3,
        32 => FILTER_DELTA4,
        _  => FILTER_DELTA3,
    }
}

/// Transform input bytes with the chosen filter before compression.
pub fn apply_filter(input: &[u8], filter: u8) -> Vec<u8> {
    let stride = filter as usize;
    if stride == 0 || stride > 4 { return input.to_vec(); }
    delta_encode(input, stride)
}

/// Reverse the filter applied during compression.
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    let stride = filter as usize;
    if stride == 0 || stride > 4 { return input.to_vec(); }
    delta_decode(input, stride)
}

fn delta_encode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = input[i].wrapping_sub(input[i - stride]);
    }
    out
}

fn delta_decode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = out[i].wrapping_add(out[i - stride]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_delta2() {
        let orig: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        for stride in 1u8..=4 {
            let enc = apply_filter(&orig, stride);
            let dec = undo_filter(&enc, stride);
            assert_eq!(dec, orig, "delta{} roundtrip failed", stride);
        }
    }

    #[test]
    fn smooth_gradient_compresses_well() {
        let pixels: Vec<u8> = (0..300usize).map(|i| (i % 256) as u8).collect();
        let enc = delta_encode(&pixels, 3);
        let residuals = &enc[3..];
        let mut counts = [0u32; 256];
        for &b in residuals { counts[b as usize] += 1; }
        let max_count = *counts.iter().max().unwrap();
        let dominant_pct = max_count as f64 / residuals.len() as f64;
        assert!(
            dominant_pct > 0.8,
            "residuals not constant enough: dominant value covers only {:.1}%",
            dominant_pct * 100.0
        );
    }
}
