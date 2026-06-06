// src/filters/bcj.rs
//! x86 BCJ (Branch-Call-Jump) filter for PE/COFF executables (flag 9).
//!
//! Converts CALL (E8) and JMP (E9) relative offsets to absolute addresses
//! using the xz-style BCJ algorithm, then reverses the transform on
//! decompression.  The absolute addresses cluster near 0x00xxxxxx (near-forward)
//! or 0xFFxxxxxx (near-backward) regardless of load address, maximising LZ
//! dictionary hit frequency across object files at different base addresses.
//!
//! Algorithm features:
//!   MSByte gate    — only process E8/E9 whose 32-bit LE operand has high byte
//!                    0x00 (near forward) or 0xFF (near backward).  Skips data
//!                    bytes that happen to be E8/E9 with far-range operands.
//!   prev_mask      — 3-bit rolling state that detects false positives arising
//!                    from E8/E9 bytes embedded inside a prior instruction's
//!                    4-byte operand.
//!   25-bit normalisation (encode only):
//!                    norm  = abs & 0x01FF_FFFF
//!                    norm |= 0u32.wrapping_sub(norm & 0x0100_0000)
//!                    Sign-extends bit 24 into bits 25–31 so all near-forward
//!                    targets map to 0x00xxxxxx and near-backward to 0xFFxxxxxx.
//!                    Wrapping subtraction in decode exactly reverses encode.
//!
//! JCC (0F 8x) is not handled — xz BCJ does not handle it either, and the
//! MSByte gate provides sufficient gain without added complexity.

// ── PE/COFF detection ─────────────────────────────────────────────────────────

/// Returns true when `data` starts with a valid MZ / PE header.
pub fn detect_pe_coff(data: &[u8]) -> bool {
    if data.len() < 0x40 { return false; }
    if data[0] != b'M' || data[1] != b'Z' { return false; }
    let pe_offset =
        u32::from_le_bytes([data[0x3C], data[0x3D], data[0x3E], data[0x3F]]) as usize;
    if pe_offset.saturating_add(4) > data.len() { return false; }
    data[pe_offset..pe_offset + 4] == *b"PE\x00\x00"
}

// ── BCJ tables ────────────────────────────────────────────────────────────────

/// Whether a prev_mask state may be processing a legitimate CALL/JMP.
static MASK_TO_ALLOWED: [bool; 8] = [true, true, true, false, true, false, false, false];

/// How many bytes back the operand byte being tested is.
static MASK_TO_BIT: [usize; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

#[inline(always)]
fn is_near_operand_high_byte(b: u8) -> bool {
    b == 0x00 || b == 0xFF
}

// ── BCJ encode ────────────────────────────────────────────────────────────────

/// Apply BCJ transform: convert near CALL/JMP relative → normalised absolute.
pub fn bcj_x86_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        // Only CALL (E8) and JMP near (E9) are candidates.
        if data[i] & 0xFE != 0xE8 {
            i += 1;
            continue;
        }

        // ── prev_mask false-positive gate ──────────────────────────────────────
        let dist = i.wrapping_sub(prev_pos);
        if dist <= 3 {
            prev_mask = (prev_mask << dist.wrapping_sub(1)) & 7;
            if prev_mask != 0 {
                let b = data[i + 4 - MASK_TO_BIT[prev_mask]];
                if !MASK_TO_ALLOWED[prev_mask] || is_near_operand_high_byte(b) {
                    prev_pos  = i;
                    prev_mask = (prev_mask << 1) | 1;
                    i        += 1;
                    continue;
                }
            }
        } else {
            prev_mask = 0;
        }
        prev_pos = i;

        // ── MSByte gate ────────────────────────────────────────────────────────
        if is_near_operand_high_byte(data[i + 4]) {
            let rel = i32::from_le_bytes([data[i+1], data[i+2], data[i+3], data[i+4]]);
            let abs = rel.wrapping_add(i as i32 + 5) as u32;

            // 25-bit normalisation: clusters near-forward at 0x00xxxxxx,
            // near-backward at 0xFFxxxxxx regardless of call-site address.
            let mut norm  = abs & 0x01FF_FFFF;
            norm         |= 0u32.wrapping_sub(norm & 0x0100_0000);

            out[i+1..i+5].copy_from_slice(&norm.to_le_bytes());
            i += 4; // +1 below → total advance = 5 (full instruction)
        } else {
            prev_mask = (prev_mask << 1) | 1;
        }

        i += 1;
    }

    out
}

// ── BCJ decode ────────────────────────────────────────────────────────────────

/// Reverse BCJ transform: convert normalised absolute → relative offset.
pub fn bcj_x86_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        if data[i] & 0xFE != 0xE8 {
            i += 1;
            continue;
        }

        let dist = i.wrapping_sub(prev_pos);
        if dist <= 3 {
            prev_mask = (prev_mask << dist.wrapping_sub(1)) & 7;
            if prev_mask != 0 {
                let b = data[i + 4 - MASK_TO_BIT[prev_mask]];
                if !MASK_TO_ALLOWED[prev_mask] || is_near_operand_high_byte(b) {
                    prev_pos  = i;
                    prev_mask = (prev_mask << 1) | 1;
                    i        += 1;
                    continue;
                }
            }
        } else {
            prev_mask = 0;
        }
        prev_pos = i;

        // Stored value is the 25-bit-normalised absolute address.
        // Wrapping subtraction exactly reverses the wrapping addition in encode.
        if is_near_operand_high_byte(data[i + 4]) {
            let abs = u32::from_le_bytes([data[i+1], data[i+2], data[i+3], data[i+4]]);
            let rel = abs.wrapping_sub(i as u32 + 5) as i32;
            out[i+1..i+5].copy_from_slice(&rel.to_le_bytes());
            i += 4;
        } else {
            prev_mask = (prev_mask << 1) | 1;
        }

        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{detect_filter, FILTER_BCJ};

    fn make_minimal_pe() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0] = b'M';
        data[1] = b'Z';
        data[0x3C] = 0x40;           // PE header at offset 0x40
        data[0x40] = b'P';
        data[0x41] = b'E';
        data[0x42] = 0x00;
        data[0x43] = 0x00;
        data
    }

    #[test]
    fn detect_pe_coff_basic() {
        assert!(detect_pe_coff(&make_minimal_pe()), "should detect PE");
    }

    #[test]
    fn detect_pe_coff_rejects_non_pe() {
        assert!(!detect_pe_coff(b"Not a PE file at all"));
    }

    #[test]
    fn detect_pe_coff_rejects_missing_signature() {
        let mut data = make_minimal_pe();
        data[0x40] = 0x00; // corrupt PE signature
        assert!(!detect_pe_coff(&data));
    }

    #[test]
    fn detect_filter_returns_bcj_for_pe() {
        assert_eq!(detect_filter(&make_minimal_pe()), FILTER_BCJ);
    }

    #[test]
    fn bcj_encode_call_near_correct() {
        // E8 at position 0x60, rel32 = 0x00000010 (MSByte 0x00 → near forward).
        // abs = 0x10 + (0x60 + 5) = 0x75.  Normalisation: identity (bit24=0).
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x10; data[0x62] = 0x00; data[0x63] = 0x00; data[0x64] = 0x00;

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x61], enc[0x62], enc[0x63], enc[0x64]]);
        assert_eq!(abs, 0x75, "BCJ encode: abs should be 0x75, got 0x{:X}", abs);
    }

    #[test]
    fn bcj_far_call_not_transformed() {
        // E8 at 0x60 with MSByte = 0x30 → far call, skip.
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x00; data[0x62] = 0x00; data[0x63] = 0x00; data[0x64] = 0x30;

        let enc = bcj_x86_encode(&data);
        assert_eq!(&enc[0x60..0x65], &data[0x60..0x65], "far call should not be transformed");
    }

    #[test]
    fn bcj_encode_jmp_near_backward_correct() {
        // E9 at 0x65 (101), rel32 = -128 (MSByte 0xFF → near backward).
        let mut data = vec![0u8; 128];
        data[0x65] = 0xE9;
        let rel: i32 = -128;
        data[0x66..0x6A].copy_from_slice(&rel.to_le_bytes()); // data[0x69] = 0xFF

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x66], enc[0x67], enc[0x68], enc[0x69]]);
        let expected = (-128i32).wrapping_add(0x65i32 + 5); // = -22
        assert_eq!(abs, expected,
            "BCJ encode JMP: abs should be {}, got {}", expected, abs);
    }

    #[test]
    fn bcj_roundtrip_near_calls() {
        let mut data = vec![0u8; 256];
        // CALL near-forward
        data[0x10] = 0xE8;
        data[0x11] = 0x20; data[0x12] = 0x00; data[0x13] = 0x00; data[0x14] = 0x00;
        // JMP near-backward
        data[0x20] = 0xE9;
        let rel: i32 = -0x10;
        data[0x21..0x25].copy_from_slice(&rel.to_le_bytes());
        // Non-near E8 (MSByte 0x30 → far, should be skipped)
        data[0x40] = 0xE8;
        data[0x44] = 0x30;

        let enc = bcj_x86_encode(&data);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "BCJ roundtrip failed");
    }

    #[test]
    fn bcj_roundtrip_false_positive_data_bytes() {
        // E8/E9 with non-near MSBytes → skipped by MSByte gate.
        let mut data = vec![0u8; 64];
        data[0]  = 0xE8;
        data[1]  = 0xAB; data[2] = 0xCD; data[3] = 0xEF; data[4] = 0x01; // MSByte 0x01
        data[10] = 0xE9;
        data[11] = 0x42; data[12] = 0x00; data[13] = 0xFF; data[14] = 0x7F; // MSByte 0x7F

        let enc = bcj_x86_encode(&data);
        assert_eq!(enc, data, "non-near bytes should be unchanged by MSByte gate");
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_identity_on_no_e8_e9() {
        let data: Vec<u8> = (0u8..=127).collect();
        assert!(!data.contains(&0xE8));
        assert!(!data.contains(&0xE9));
        let enc = bcj_x86_encode(&data);
        assert_eq!(enc, data, "BCJ should be identity when no CALL/JMP opcodes");
    }

    #[test]
    fn bcj_25bit_norm_clusters_near_forward_to_zero_high_byte() {
        let mut data = vec![0u8; 16];
        // E8 at 0, rel=0x100. abs = 0x100 + (0 + 5) = 0x105. bit24=0 → norm=0x105.
        data[0] = 0xE8;
        data[1] = 0x00; data[2] = 0x01; data[3] = 0x00; data[4] = 0x00; // MSByte 0x00

        let enc = bcj_x86_encode(&data);
        assert_eq!(enc[4], 0x00, "near-forward high byte should be 0x00 after normalisation");

        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec[1..5], data[1..5], "normalisation should be exactly reversed by decode");
    }
                                           }
