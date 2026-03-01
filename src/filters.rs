// src/filters.rs
//! Pre/post compression delta filters.
//!
//! Applied to the raw input BEFORE folding, reversed AFTER unfolding.
//! Converts smooth-varying binary data (PCM audio, BMP pixels, fax bitmaps)
//! into near-zero residuals that LZ can match at dramatically higher density.
//!
//! Detection strategy:
//!   1. Known magic bytes (WAV, BMP) → structured header parse for exact stride.
//!   2. Unknown files → empirical trial: apply each stride 1–4, estimate byte
//!      entropy of the result, pick the winner if it beats passthrough by at
//!      least EMPIRICAL_MIN_GAIN bits/byte. Text and already-random data never
//!      pass this threshold so the filter naturally stays off for them.
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1  (generic 8-bit binary / fax bitmap)
//!   2 = delta stride 2  (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3  (24-bit RGB pixels)
//!   4 = delta stride 4  (32-bit RGBA / stereo 16-bit PCM)

pub const FILTER_NONE:   u8 = 0;
pub const FILTER_DELTA1: u8 = 1;
pub const FILTER_DELTA2: u8 = 2;
pub const FILTER_DELTA3: u8 = 3;
pub const FILTER_DELTA4: u8 = 4;

/// Minimum file size for empirical detection — avoid overhead on tiny inputs.
const EMPIRICAL_MIN_BYTES: usize = 512;

/// Minimum entropy reduction (bits/byte) required for empirical filter to fire.
/// 0.30 is conservative — avoids false positives on text and structured data
/// while reliably catching fax bitmaps (expected reduction ~2–4 bits/byte).
const EMPIRICAL_MIN_GAIN: f64 = 0.30;

// ── Public API ────────────────────────────────────────────────────────────────

/// Inspect magic bytes and file headers to determine the best delta stride.
/// Falls back to empirical entropy trial for unknown binary formats.
/// Returns FILTER_NONE when no beneficial stride is found.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 4 { return FILTER_NONE; }

    // ── Known magic: structured header parse ─────────────────────────────────
    if &input[0..4] == b"RIFF" && input.len() >= 12 && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    // ── Unknown format: empirical entropy trial ───────────────────────────────
    if input.len() >= EMPIRICAL_MIN_BYTES {
        return empirical_best_stride(input);
    }

    FILTER_NONE
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

// ── Known-format detection ────────────────────────────────────────────────────

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

// ── Empirical detection ───────────────────────────────────────────────────────

/// Try delta strides 1–4 on the input, pick the one with lowest byte entropy.
/// Only returns a non-zero filter if the winner beats passthrough by at least
/// EMPIRICAL_MIN_GAIN bits/byte. Returns FILTER_NONE otherwise.
///
/// Uses Shannon entropy of the byte frequency distribution as a proxy for
/// compressibility — lower entropy means better Huffman and LZ density.
/// The full delta is applied to the entire input for accuracy; this is O(4n)
/// which is fast relative to the LZ scan that follows.
fn empirical_best_stride(input: &[u8]) -> u8 {
    let base_entropy = byte_entropy(input);

    let mut best_stride: u8  = FILTER_NONE;
    let mut best_entropy: f64 = base_entropy;

    for stride in 1u8..=4 {
        let filtered = delta_encode(input, stride as usize);
        let h        = byte_entropy(&filtered);
        if h < best_entropy {
            best_entropy  = h;
            best_stride   = stride;
        }
    }

    let gain = base_entropy - best_entropy;
    if gain >= EMPIRICAL_MIN_GAIN {
        println!(
            "Empirical filter: stride {} wins (entropy {:.4} → {:.4}, gain {:.4} bits/byte)",
            best_stride, base_entropy, best_entropy, gain
        );
        best_stride
    } else {
        if best_stride != FILTER_NONE {
            println!(
                "Empirical filter: best stride {} gain {:.4} bits/byte < threshold {:.2} — skipped",
                best_stride, gain, EMPIRICAL_MIN_GAIN
            );
        }
        FILTER_NONE
    }
}

/// Shannon entropy of the byte frequency distribution, in bits per byte.
/// H = -sum(p_i * log2(p_i)) over all byte values with non-zero frequency.
/// Range: 0.0 (single repeated byte) to 8.0 (uniform random).
fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }

    let mut freq = [0u64; 256];
    for &b in data { freq[b as usize] += 1; }

    let n = data.len() as f64;
    freq.iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / n;
            -p * p.log2()
        })
        .sum()
}

// ── Core delta routines ───────────────────────────────────────────────────────

/// Delta encode: out[i] = input[i].wrapping_sub(input[i - stride]).
/// First `stride` bytes are copied unchanged (no prior context).
fn delta_encode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = input[i].wrapping_sub(input[i - stride]);
    }
    out
}

/// Delta decode: out[i] = encoded[i].wrapping_add(out[i - stride]).
/// Inverse of delta_encode — iterates forwards, accumulating context.
fn delta_decode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = out[i].wrapping_add(out[i - stride]);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_strides() {
        let orig: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        for stride in 1u8..=4 {
            let enc = apply_filter(&orig, stride);
            let dec = undo_filter(&enc, stride);
            assert_eq!(dec, orig, "delta{} roundtrip failed", stride);
        }
    }

    #[test]
    fn empirical_fires_on_binary_runs() {
        // Simulate fax-like data: long runs of 0x00 and 0xFF.
        let mut data = Vec::with_capacity(1024);
        for _ in 0..16 {
            data.extend_from_slice(&[0x00u8; 32]);
            data.extend_from_slice(&[0xFFu8; 32]);
        }
        // Original entropy: 2 values at 50% each = 1.0 bits/byte.
        // Delta stride 1: runs become 0x00, transitions become 0xFF or 0x01.
        // 0x00 dominates heavily → entropy drops well below 1.0.
        let filter = empirical_best_stride(&data);
        assert_eq!(filter, FILTER_DELTA1,
            "expected stride 1 for fax-like data, got {}", filter);
    }

    #[test]
    fn empirical_skips_text() {
        // English-like ASCII text — delta residuals are scattered, entropy stays high.
        let text = b"the quick brown fox jumps over the lazy dog. \
                     pack my box with five dozen liquor jugs. \
                     how vexingly quick daft zebras jump. ";
        let data: Vec<u8> = text.iter().cycle().take(1024).copied().collect();
        let filter = empirical_best_stride(&data);
        assert_eq!(filter, FILTER_NONE,
            "expected no filter for text, got {}", filter);
    }

    #[test]
    fn empirical_skips_random() {
        // Random bytes — no delta stride helps, entropy stays near 8.0.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let data: Vec<u8> = (0u64..1024)
            .map(|i| { let mut h = DefaultHasher::new(); i.hash(&mut h); h.finish() as u8 })
            .collect();
        let filter = empirical_best_stride(&data);
        assert_eq!(filter, FILTER_NONE,
            "expected no filter for random data, got {}", filter);
    }

    #[test]
    fn smooth_gradient_compresses_well() {
        let pixels: Vec<u8> = (0..300usize).map(|i| (i % 256) as u8).collect();
        let enc = delta_encode(&pixels, 3);
        let residuals = &enc[3..];
        let mut counts = [0u32; 256];
        for &b in residuals { counts[b as usize] += 1; }
        let max_count    = *counts.iter().max().unwrap();
        let dominant_pct = max_count as f64 / residuals.len() as f64;
        assert!(
            dominant_pct > 0.8,
            "residuals not constant enough: dominant value covers only {:.1}%",
            dominant_pct * 100.0
        );
    }
            }
