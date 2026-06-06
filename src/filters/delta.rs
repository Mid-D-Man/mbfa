// src/filters/delta.rs
//! Stride-delta pre-filter (flags 1–4).
//!
//! For each byte at position i ≥ stride:
//!   encode: out[i] = in[i] − in[i − stride]  (wrapping)
//!   decode: out[i] = out[i] + out[i − stride] (wrapping, left-to-right)
//!
//! Bytes 0..stride are left unchanged.  The transform is its own inverse up to
//! initial conditions — encoding then decoding recovers the original exactly.
//! Typical gain: 0.5–2.0 bits/byte on smoothly-varying numeric streams (audio
//! PCM, heightmaps, multi-channel sensor data).

/// Delta-encode `input` with the given `stride`.
pub fn delta_encode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = input[i].wrapping_sub(input[i - stride]);
    }
    out
}

/// Delta-decode `input` with the given `stride` (inverse of `delta_encode`).
pub fn delta_decode(input: &[u8], stride: usize) -> Vec<u8> {
    let mut out = input.to_vec();
    for i in stride..input.len() {
        out[i] = out[i].wrapping_add(out[i - stride]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{apply_filter, undo_filter};

    #[test]
    fn roundtrip_delta_all_strides() {
        let orig: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        for stride in 1u8..=4 {
            let enc = apply_filter(&orig, stride);
            let dec = undo_filter(&enc, stride);
            assert_eq!(dec, orig, "delta{} roundtrip failed", stride);
        }
    }

    #[test]
    fn inplace_delta_encode_ltr_is_correct() {
        let orig = vec![10u8, 12, 15, 11, 14];
        let enc  = delta_encode(&orig, 1);
        // Expected: [10, 12-10=2, 15-12=3, 11-15=252(wrapping), 14-11=3]
        assert_eq!(enc, vec![10, 2, 3, 252, 3]);
        let dec = delta_decode(&enc, 1);
        assert_eq!(dec, orig);
    }

    #[test]
    fn delta_stride2_roundtrip() {
        let orig: Vec<u8> = (0..256).map(|i| ((i as u32 * 137) % 256) as u8).collect();
        let enc = delta_encode(&orig, 2);
        let dec = delta_decode(&enc, 2);
        assert_eq!(dec, orig);
    }

    #[test]
    fn delta_preserves_first_stride_bytes() {
        let orig = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let enc = delta_encode(&orig, 3);
        // First `stride` bytes unchanged.
        assert_eq!(&enc[..3], &[0xAA, 0xBB, 0xCC]);
    }
}
