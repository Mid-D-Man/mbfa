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
//! Filter flags (header byte 3):
//!   0 = none
//!   1 = delta stride 1
//!   2 = delta stride 2
//!   3 = delta stride 3
//!   4 = delta stride 4
//!   7 = STL: byte-plane shuffle × 4 + per-plane delta1
//!   8 = PLY: byte-plane shuffle × 4 + per-plane delta1
//!   9 = x86 BCJ for PE/COFF executables
//!
//! Detection order:
//!   1. Binary STL  — exact size equation               → flag 7
//!   2. WAV/RIFF    — "RIFF....WAVE" magic              → flag 1–4
//!   3. BMP         — "BM" magic                        → flag 1–4
//!   4. Binary PLY  — "ply\n" + binary_little_endian    → flag 8
//!   5. DixScript binary — magic 0x4D444958 LE          → flag 0 (FILTER_NONE, skip probe)
//!   6. PE/COFF     — "MZ" + PE offset + "PE\0\0"       → flag 9
//!   7. Stride probe — 8 KB entropy, threshold 0.45     → best of 1–4

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
pub const FILTER_SHUFFLE4_DELTA: u8 = 7;
pub const FILTER_PLY_DELTA:      u8 = 8;
pub const FILTER_BCJ:            u8 = 9;

/// DixScript binary format magic number: `0x4D444958` ("MDIX" big-endian)
/// stored on disk as little-endian u32 → bytes [0x58, 0x49, 0x44, 0x4D].
///
/// This is a typed key-value data format (CONFIG / ENUMS / DATA / SECURITY sections
/// with value tags 0x01–0x0F).  No transform is applied — FILTER_NONE is returned
/// so the stride probe is skipped and MBFA's LZ+entropy path handles it directly.
///
/// Why no transform:
///   • The format contains typed values (not numeric streams) → delta unhelpful.
///   • No x86 CALL/JMP opcodes → BCJ inappropriate.
///   • Type tags (0x01–0x0F) are low-range → entropy coding extremely efficient.
///   • Repeated string keys ("id","name","value"…) → LZ excellent without pre-filter.
const DIXSCRIPT_MAGIC_BYTES: [u8; 4] = [0x58, 0x49, 0x44, 0x4D]; // 0x4D444958 as LE u32

// ── Public API ────────────────────────────────────────────────────────────────

pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL
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

    // 5. DixScript binary (.mdix compiled, magic 0x4D444958 written as LE u32).
    //    On-disk bytes: [0x58, 0x49, 0x44, 0x4D].
    //    No transform — LZ+entropy handles the typed key-value structure directly.
    //    Returning FILTER_NONE here (before the stride probe) prevents the probe
    //    from incorrectly firing on the value-tag byte patterns.
    if input.len() >= 16 && input[0..4] == DIXSCRIPT_MAGIC_BYTES {
        println!("DixScript binary (.mdix compiled) detected — MDIX magic 0x4D444958 → FILTER_NONE");
        return FILTER_NONE;
    }

    // 6. PE/COFF
    if detect_pe_coff(input) {
        println!("PE/COFF binary detected → FILTER_BCJ");
        return FILTER_BCJ;
    }

    // 7. Multi-stride entropy probe
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_filter_returns_none_for_short_input() {
        assert_eq!(detect_filter(&[0u8; 4]), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_none_for_random_data() {
        let mut state: u32 = 0xdeadbeef;
        let data: Vec<u8> = (0..1024).map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        }).collect();
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_wav_stride_for_riff_wave() {
        let mut data = vec![0u8; 44];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WAVE");
        data[12..16].copy_from_slice(b"fmt ");
        data[16..20].copy_from_slice(&16u32.to_le_bytes());
        data[22..24].copy_from_slice(&1u16.to_le_bytes());
        data[34..36].copy_from_slice(&16u16.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_DELTA2);
    }

    #[test]
    fn detect_filter_returns_bmp_delta3_for_24bpp() {
        let mut data = vec![0u8; 32];
        data[0..2].copy_from_slice(b"BM");
        data[28..30].copy_from_slice(&24u16.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_DELTA3);
    }

    #[test]
    fn detect_dixscript_binary_returns_filter_none() {
        // Minimal MDIX header: magic 0x4D444958 stored as LE u32 = [0x58,0x49,0x44,0x4D]
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&0x4D444958u32.to_le_bytes()); // = [0x58,0x49,0x44,0x4D]
        data[4] = 1; // version_major
        data[5] = 0; // version_minor
        data[6] = 0; // version_patch
        data[7] = 0x05; // flags: CONFIG | DATA
        assert_eq!(
            detect_filter(&data), FILTER_NONE,
            "MDIX magic should return FILTER_NONE (data format, no transform)"
        );
    }

    #[test]
    fn mdix_magic_bytes_are_correct_le_encoding() {
        // Verify our constant matches what binary_format.rs writes
        let expected = 0x4D444958u32.to_le_bytes();
        assert_eq!(DIXSCRIPT_MAGIC_BYTES, expected,
            "DIXSCRIPT_MAGIC_BYTES must equal 0x4D444958.to_le_bytes()");
        // Sanity: bytes spell "XIDM" in ASCII (LE of "MDIX")
        assert_eq!(&DIXSCRIPT_MAGIC_BYTES, b"XIDM");
    }

    #[test]
    fn detect_pe_coff_still_returns_bcj() {
        let mut data = vec![0u8; 256];
        data[0] = b'M'; data[1] = b'Z';
        data[0x3C] = 0x40;
        data[0x40] = b'P'; data[0x41] = b'E';
        data[0x42] = 0x00; data[0x43] = 0x00;
        assert_eq!(detect_filter(&data), FILTER_BCJ);
    }

    #[test]
    fn dixscript_magic_does_not_match_pe_coff() {
        // PE/COFF starts with 'M','Z' = [0x4D, 0x5A]; MDIX starts with [0x58, 0x49, 0x44, 0x4D]
        // These are different and must not collide.
        assert_ne!(&DIXSCRIPT_MAGIC_BYTES[0..2], b"MZ");
    }
        }
