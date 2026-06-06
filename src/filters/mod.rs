// src/filters/mod.rs
//! Pre/post-compression delta filters + byte-plane shuffle + BCJ transform.
//!
//! Sub-modules:
//!   delta  — stride-delta filters (flags 1–4)
//!   stl    — STL byte-plane shuffle + per-plane delta (flag 7)
//!   ply    — PLY byte-plane shuffle + per-plane delta (flag 8)
//!   bcj    — x86 BCJ normalisation for PE/COFF executables (flag 9)
//!   probe  — multi-stride entropy probe + WAV/BMP stride detection helpers
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1    (generic 8-bit binary)
//!   2 = delta stride 2    (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3    (24-bit RGB)
//!   4 = delta stride 4    (32-bit RGBA / stereo 16-bit PCM)
//!   7 = STL: byte-plane shuffle × 4 + per-plane delta1
//!   8 = PLY: byte-plane shuffle × 4 + per-plane delta1
//!   9 = x86 BCJ (xz-style): E8/E9 rel→abs, MSByte gate, 25-bit normalisation
//!
//! Detection order:
//!   1. Binary STL  — exact size equation               → flag 7
//!   2. WAV/RIFF    — "RIFF....WAVE" magic              → flag 1–4
//!   3. BMP         — "BM" magic                        → flag 1–4
//!   4. Binary PLY  — "ply\n" + binary_little_endian    → flag 8
//!   5. PE/COFF     — "MZ" + PE offset + "PE\0\0"       → flag 9
//!   6. Stride probe — 8 KB entropy sample, threshold 0.45 bits/byte → best of 1–4

pub mod delta;
pub mod stl;
pub mod ply;
pub mod bcj;
pub mod probe;

pub use delta::{delta_encode, delta_decode};
pub use stl::{detect_stl, shuffle4_stl_delta_encode, shuffle4_stl_delta_decode};
pub use ply::{parse_ply_layout, shuffle4_ply_delta_encode, shuffle4_ply_delta_decode};
pub use bcj::{detect_pe_coff, bcj_x86_encode, bcj_x86_decode};
pub use probe::{
    byte_entropy, probe_best_stride,
    detect_wav_stride, detect_bmp_stride,
    PROBE_MIN_BYTES, PROBE_DELTA_THRESHOLD,
};

// ── Filter flag constants ─────────────────────────────────────────────────────

pub const FILTER_NONE:           u8 = 0;
pub const FILTER_DELTA1:         u8 = 1;
pub const FILTER_DELTA2:         u8 = 2;
pub const FILTER_DELTA3:         u8 = 3;
pub const FILTER_DELTA4:         u8 = 4;
// flags 5 and 6 were the old simple (non-compound) STL/PLY shuffles — removed.
/// STL: byte-plane shuffle + per-plane delta1.
pub const FILTER_SHUFFLE4_DELTA: u8 = 7;
/// PLY: byte-plane shuffle + per-plane delta1.
pub const FILTER_PLY_DELTA:      u8 = 8;
/// x86 BCJ normalisation for PE/COFF binaries (xz-style).
pub const FILTER_BCJ:            u8 = 9;

// ── Public API ────────────────────────────────────────────────────────────────

/// Inspect magic bytes and file structure to determine the best pre-filter.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL — exact size equation (no magic bytes).
    if input.len() >= 84 && detect_stl(input).is_some() {
        println!("Binary STL detected → FILTER_SHUFFLE4_DELTA (compound)");
        return FILTER_SHUFFLE4_DELTA;
    }

    // 2. WAV / RIFF
    if &input[0..4] == b"RIFF" && input.len() >= 12 && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }

    // 3. BMP
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    // 4. Binary PLY
    if input.len() >= 4 && &input[0..4] == b"ply\n" && parse_ply_layout(input).is_some() {
        println!("Binary PLY detected → FILTER_PLY_DELTA (compound)");
        return FILTER_PLY_DELTA;
    }

    // 5. PE/COFF (must precede stride probe — stride probe may fire on .text section).
    if detect_pe_coff(input) {
        println!("PE/COFF binary detected → FILTER_BCJ");
        return FILTER_BCJ;
    }

    // 6. Multi-stride entropy probe (headerless strided binary: terrain, raw PCM, etc.).
    if input.len() >= PROBE_MIN_BYTES {
        let (best_filter, improvement) = probe_best_stride(input);
        if improvement >= PROBE_DELTA_THRESHOLD {
            println!(
                "Stride probe: FILTER_DELTA{} entropy improvement {:.2} bits/byte \
                 (threshold {:.2}) → applying filter",
                best_filter, improvement, PROBE_DELTA_THRESHOLD
            );
            return best_filter;
        }
        if improvement > 0.1 {
            println!(
                "Stride probe: best FILTER_DELTA{} improvement {:.2} bits/byte \
                 — below threshold {:.2}, no filter applied",
                best_filter, improvement, PROBE_DELTA_THRESHOLD
            );
        }
    }

    FILTER_NONE
}

/// Transform input bytes with the chosen filter before compression.
pub fn apply_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_encode(input, filter as usize)
        }
        FILTER_SHUFFLE4_DELTA => shuffle4_stl_delta_encode(input),
        FILTER_PLY_DELTA      => shuffle4_ply_delta_encode(input),
        FILTER_BCJ            => bcj_x86_encode(input),
        _                     => input.to_vec(),
    }
}

/// Reverse the filter applied during compression.
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_decode(input, filter as usize)
        }
        FILTER_SHUFFLE4_DELTA => shuffle4_stl_delta_decode(input),
        FILTER_PLY_DELTA      => shuffle4_ply_delta_decode(input),
        FILTER_BCJ            => bcj_x86_decode(input),
        _                     => input.to_vec(),
    }
}

// ── Tests for detect_filter dispatch ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_filter_returns_none_for_short_input() {
        assert_eq!(detect_filter(&[0u8; 4]), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_none_for_random_data() {
        // High-entropy data with no recognisable header.
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        // Stride probe should not fire on uniform distribution.
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_wav_stride_for_riff_wave() {
        // Minimal RIFF/WAVE header: mono 16-bit PCM → stride 2.
        let mut data = vec![0u8; 44];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WAVE");
        data[12..16].copy_from_slice(b"fmt ");
        data[16..20].copy_from_slice(&16u32.to_le_bytes()); // chunk len
        // channels=1, bits_per_sample=16 → stride = 1*(16/8) = 2
        data[22..24].copy_from_slice(&1u16.to_le_bytes());  // channels
        data[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits
        assert_eq!(detect_filter(&data), FILTER_DELTA2);
    }

    #[test]
    fn detect_filter_returns_bmp_delta3_for_24bpp() {
        let mut data = vec![0u8; 32];
        data[0..2].copy_from_slice(b"BM");
        data[28..30].copy_from_slice(&24u16.to_le_bytes()); // bpp
        assert_eq!(detect_filter(&data), FILTER_DELTA3);
    }
}
