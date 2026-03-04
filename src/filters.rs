// src/filters.rs
//! Pre/post compression delta filters + BCJ x86 filter.
//!
//! Applied to the raw input BEFORE folding, reversed AFTER unfolding.
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1  (generic 8-bit binary)
//!   2 = delta stride 2  (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3  (24-bit RGB pixels)
//!   4 = delta stride 4  (32-bit RGBA / stereo 16-bit PCM)
//!   5 = BCJ x86         (ELF / PE executables — CALL/JMP relative→absolute)

pub const FILTER_NONE:    u8 = 0;
pub const FILTER_DELTA1:  u8 = 1;
pub const FILTER_DELTA2:  u8 = 2;
pub const FILTER_DELTA3:  u8 = 3;
pub const FILTER_DELTA4:  u8 = 4;
pub const FILTER_BCJ_X86: u8 = 5;

/// Inspect magic bytes to determine the best filter.
/// Returns FILTER_NONE for unknown or uncompressible formats.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // ELF executable
    if input[0..4] == [0x7F, b'E', b'L', b'F'] {
        println!("ELF magic detected → BCJ x86 filter");
        return FILTER_BCJ_X86;
    }

    // PE / COFF (Windows executable / DLL)
    if input[0..2] == [b'M', b'Z'] {
        println!("PE/MZ magic detected → BCJ x86 filter");
        return FILTER_BCJ_X86;
    }

    // WAV / RIFF audio
    if input[0..4] == *b"RIFF" && input[8..12] == *b"WAVE" {
        return detect_wav_stride(input);
    }

    // BMP image
    if input[0..2] == *b"BM" && input.len() >= 30 {
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
    match filter {
        FILTER_DELTA1..=FILTER_DELTA4 => delta_encode(input, filter as usize),
        FILTER_BCJ_X86                => bcj_x86_encode(input),
        _                             => input.to_vec(),
    }
}

/// Reverse the filter applied during compression.
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1..=FILTER_DELTA4 => delta_decode(input, filter as usize),
        FILTER_BCJ_X86                => bcj_x86_decode(input),
        _                             => input.to_vec(),
    }
}

// ── Delta routines ────────────────────────────────────────────────────────────

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

// ── BCJ x86 routines ──────────────────────────────────────────────────────────
//
// x86 CALL near (E8) and JMP near (E9) both encode a signed 32-bit
// relative displacement to the TARGET from the NEXT instruction:
//
//   target = current_pos + 5 + displacement   (instruction is 5 bytes)
//   displacement = target - (current_pos + 5)
//
// Converting relative→absolute makes duplicate call targets — common in
// executables — appear as identical 4-byte sequences that LZ can match.
//
// The transform is applied to ALL E8/E9 bytes regardless of whether they
// are actually instructions (false positives happen but are harmless since
// the transform is its own inverse up to the sign flip).

/// Encode: replace relative CALL/JMP offsets with absolute addresses.
fn bcj_x86_encode(input: &[u8]) -> Vec<u8> {
    let mut out = input.to_vec();
    let n = input.len();
    let mut i = 0;

    while i + 5 <= n {
        if input[i] == 0xE8 || input[i] == 0xE9 {
            // Read signed little-endian 32-bit relative offset
            let rel = i32::from_le_bytes([
                input[i + 1], input[i + 2], input[i + 3], input[i + 4],
            ]);
            // Convert to absolute: abs = rel + (i + 5)
            // Use wrapping to handle boundary values safely
            let abs = rel.wrapping_add((i as i32).wrapping_add(5));
            let abs_bytes = abs.to_le_bytes();
            out[i + 1] = abs_bytes[0];
            out[i + 2] = abs_bytes[1];
            out[i + 3] = abs_bytes[2];
            out[i + 4] = abs_bytes[3];
            // Skip the 4 operand bytes — they are now transformed and
            // should not be re-scanned as potential opcodes.
            i += 5;
        } else {
            i += 1;
        }
    }

    out
}

/// Decode: restore relative CALL/JMP offsets from absolute addresses.
/// Exact inverse of bcj_x86_encode.
fn bcj_x86_decode(input: &[u8]) -> Vec<u8> {
    let mut out = input.to_vec();
    let n = input.len();
    let mut i = 0;

    while i + 5 <= n {
        // We look at the ORIGINAL (pre-transform) byte at position i.
        // Since encode only modifies bytes i+1..i+4 and not byte i itself,
        // out[i] still equals input[i] here — safe to check.
        if input[i] == 0xE8 || input[i] == 0xE9 {
            let abs = i32::from_le_bytes([
                input[i + 1], input[i + 2], input[i + 3], input[i + 4],
            ]);
            // Convert back to relative: rel = abs - (i + 5)
            let rel = abs.wrapping_sub((i as i32).wrapping_add(5));
            let rel_bytes = rel.to_le_bytes();
            out[i + 1] = rel_bytes[0];
            out[i + 2] = rel_bytes[1];
            out[i + 3] = rel_bytes[2];
            out[i + 4] = rel_bytes[3];
            i += 5;
        } else {
            i += 1;
        }
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn roundtrip_bcj_x86() {
        // Synthesize a buffer with a few CALL and JMP instructions at known offsets.
        let mut input = vec![0u8; 64];
        // CALL at offset 0 with relative displacement 0x00001234
        input[0]  = 0xE8;
        input[1]  = 0x34; input[2] = 0x12; input[3] = 0x00; input[4] = 0x00;
        // JMP at offset 10 with relative displacement -0x0000_0010 (backwards)
        input[10] = 0xE9;
        let rel: i32 = -0x10;
        let rb = rel.to_le_bytes();
        input[11] = rb[0]; input[12] = rb[1]; input[13] = rb[2]; input[14] = rb[3];
        // Some filler to make sure non-E8/E9 bytes are untouched
        input[20] = 0x90; // NOP
        input[21] = 0x55; // PUSH rbp

        let encoded = apply_filter(&input, FILTER_BCJ_X86);
        let decoded = undo_filter(&encoded, FILTER_BCJ_X86);
        assert_eq!(decoded, input, "BCJ x86 roundtrip failed");
    }

    #[test]
    fn bcj_x86_absolute_conversion() {
        // CALL at offset 0, displacement = 0x00000005
        // Expected absolute = 0 + 5 + 5 = 10 = 0x0000000A
        let mut input = vec![0u8; 10];
        input[0] = 0xE8;
        input[1] = 0x05; input[2] = 0x00; input[3] = 0x00; input[4] = 0x00;

        let enc = bcj_x86_encode(&input);
        let abs = i32::from_le_bytes([enc[1], enc[2], enc[3], enc[4]]);
        assert_eq!(abs, 10, "BCJ absolute conversion wrong: got {}", abs);
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
