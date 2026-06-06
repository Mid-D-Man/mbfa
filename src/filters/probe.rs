// src/filters/probe.rs
//! Multi-stride entropy probe + WAV / BMP stride-selection helpers.
//!
//! The entropy probe samples up to 8 KB of the file header, computes Shannon
//! entropy before and after a delta-encode at strides 1–4, and returns the
//! best-improving stride above the threshold.  Fires on terrain heightmaps,
//! raw sensor streams, multi-channel binary data, etc.
//!
//! WAV and BMP helpers read their format headers to derive the exact stride
//! (sample_bytes × channels for WAV; bits-per-pixel / 8 for BMP).

use crate::filters::{FILTER_DELTA1, FILTER_DELTA2, FILTER_DELTA3, FILTER_DELTA4, FILTER_NONE};

/// Minimum file size before the entropy stride probe is attempted.
pub const PROBE_MIN_BYTES: usize = 512;

/// Entropy improvement threshold in bits/byte for the stride probe.
/// Below this the filter is not applied even if it reduces entropy.
pub const PROBE_DELTA_THRESHOLD: f64 = 0.45;

// ── WAV / BMP stride selection ────────────────────────────────────────────────

/// Parse a WAV/RIFF "fmt " chunk to determine the optimal delta stride.
pub fn detect_wav_stride(input: &[u8]) -> u8 {
    let mut pos = 12usize; // skip "RIFF<size>WAVE"
    while pos + 8 <= input.len() {
        let id        = &input[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([
            input[pos + 4], input[pos + 5], input[pos + 6], input[pos + 7],
        ]) as usize;

        if id == b"fmt " && chunk_len >= 16 && pos + 8 + 16 <= input.len() {
            let channels    = u16::from_le_bytes([input[pos + 10], input[pos + 11]]);
            let bits_sample = u16::from_le_bytes([input[pos + 22], input[pos + 23]]);
            let stride      = (channels as usize) * (bits_sample as usize / 8);
            println!(
                "WAV fmt: {} ch, {} bps → delta stride {}",
                channels, bits_sample, stride
            );
            return match stride {
                1 => FILTER_DELTA1,
                2 => FILTER_DELTA2,
                3 => FILTER_DELTA3,
                4 => FILTER_DELTA4,
                _ => FILTER_DELTA2, // default for unusual configs
            };
        }

        pos += 8 + chunk_len;
        if chunk_len % 2 != 0 { pos += 1; } // RIFF chunk padding
    }
    println!("WAV: fmt chunk not found, falling back to delta2");
    FILTER_DELTA2
}

/// Read a BMP file header to determine the optimal delta stride.
pub fn detect_bmp_stride(input: &[u8]) -> u8 {
    let bpp = u16::from_le_bytes([input[28], input[29]]);
    println!("BMP: {} bpp → delta stride {}", bpp, (bpp as usize / 8).max(1));
    match bpp {
        8  => FILTER_DELTA1,
        16 => FILTER_DELTA2,
        24 => FILTER_DELTA3,
        32 => FILTER_DELTA4,
        _  => FILTER_DELTA3, // default for unusual depths
    }
}

// ── Entropy probe ─────────────────────────────────────────────────────────────

/// Shannon entropy in bits/byte of the given slice.
pub fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for &b in data { freq[b as usize] += 1; }
    let n = data.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
}

/// Entropy improvement (bits/byte) from delta-encoding `data` at `stride`.
/// Samples up to 8 KB.  Returns 0.0 if there is no improvement.
pub fn probe_delta_improvement(data: &[u8], stride: usize) -> f64 {
    const SAMPLE: usize = 8192;
    let sample = if data.len() > SAMPLE { &data[..SAMPLE] } else { data };
    if sample.len() < stride * 2 { return 0.0; }
    let raw_entropy = byte_entropy(sample);
    let mut delta   = sample.to_vec();
    for i in stride..delta.len() {
        delta[i] = sample[i].wrapping_sub(sample[i - stride]);
    }
    (raw_entropy - byte_entropy(&delta)).max(0.0)
}

/// Test strides 1–4 and return the (filter_flag, improvement) pair with the
/// largest improvement.  Returns (FILTER_NONE, 0.0) when data is empty.
pub fn probe_best_stride(data: &[u8]) -> (u8, f64) {
    let candidates = [
        (FILTER_DELTA1, probe_delta_improvement(data, 1)),
        (FILTER_DELTA2, probe_delta_improvement(data, 2)),
        (FILTER_DELTA3, probe_delta_improvement(data, 3)),
        (FILTER_DELTA4, probe_delta_improvement(data, 4)),
    ];
    candidates
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((FILTER_NONE, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_fires_on_smooth_int16() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let imp = probe_delta_improvement(&data, 2);
        assert!(
            imp >= PROBE_DELTA_THRESHOLD,
            "probe should fire (improvement={:.2}, threshold={:.2})",
            imp, PROBE_DELTA_THRESHOLD
        );
    }

    #[test]
    fn probe_does_not_fire_on_random() {
        let mut state: u32 = 0xdeadbeef;
        let data: Vec<u8> = (0..1024).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }).collect();
        let imp = probe_delta_improvement(&data, 2);
        assert!(
            imp < PROBE_DELTA_THRESHOLD,
            "probe should not fire on random (improvement={:.2})",
            imp
        );
    }

    #[test]
    fn probe_does_not_fire_on_text() {
        let data: Vec<u8> = b"the quick brown fox jumps over the lazy dog \
            hello world foo bar baz qux the end and the beginning"
            .iter().cycle().take(512).copied().collect();
        let imp = probe_delta_improvement(&data, 2);
        assert!(
            imp < PROBE_DELTA_THRESHOLD,
            "probe should not fire on text (improvement={:.2})",
            imp
        );
    }

    #[test]
    fn probe_best_stride_picks_delta2_for_int16() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let (filter, imp) = probe_best_stride(&data);
        assert_eq!(
            filter, FILTER_DELTA2,
            "expected FILTER_DELTA2 for int16, got FILTER_DELTA{}",
            filter
        );
        assert!(imp >= PROBE_DELTA_THRESHOLD);
    }

    #[test]
    fn probe_best_stride_picks_delta4_for_int32() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..2048usize {
            let h: u32 = ((i as u64 * 2048) % (u32::MAX as u64 + 1)) as u32;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let (filter, imp) = probe_best_stride(&data);
        assert!(
            imp >= PROBE_DELTA_THRESHOLD,
            "probe should fire on smooth int32 (improvement={:.2})",
            imp
        );
        assert_eq!(
            filter, FILTER_DELTA4,
            "expected FILTER_DELTA4 for int32, got FILTER_DELTA{}",
            filter
        );
    }

    #[test]
    fn probe_multi_freq_terrain_fires() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let x = i as f64;
            let height = (
                (x * 0.05).sin() * (x * 0.07).cos() * 0.4
                    + (x * 0.02).sin() * 0.3
                    + (x * 0.1).sin() * 0.15
                    + 0.5
            ).clamp(0.0, 1.0) * 65535.0;
            data.extend_from_slice(&(height as u16).to_le_bytes());
        }
        let (filter, imp) = probe_best_stride(&data);
        assert!(
            imp >= PROBE_DELTA_THRESHOLD,
            "multi-freq terrain should fire (improvement={:.2}, threshold={:.2})",
            imp, PROBE_DELTA_THRESHOLD
        );
        assert_eq!(filter, FILTER_DELTA2);
    }
}
