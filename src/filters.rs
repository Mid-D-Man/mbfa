// src/filters.rs
//! Pre/post compression delta filters + byte-plane shuffle for float data.
//!
//! Applied to the raw input BEFORE folding, reversed AFTER unfolding.
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1  (generic 8-bit binary)
//!   2 = delta stride 2  (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3  (24-bit RGB pixels)
//!   4 = delta stride 4  (32-bit RGBA / stereo 16-bit PCM)
//!   5 = float byte-plane shuffle (binary STL mesh)
//!       Transposes N×12 float32 values into 4 byte-planes
//!       (byte[0] of all floats, byte[1], byte[2], byte[3]).
//!       Size-preserving. Header and attribute bytes passed verbatim.

pub const FILTER_NONE:     u8 = 0;
pub const FILTER_DELTA1:   u8 = 1;
pub const FILTER_DELTA2:   u8 = 2;
pub const FILTER_DELTA3:   u8 = 3;
pub const FILTER_DELTA4:   u8 = 4;
pub const FILTER_SHUFFLE4: u8 = 5;

/// Inspect magic bytes and file structure to determine the best filter.
/// Returns FILTER_NONE for unknown or already-compressed formats.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // Binary STL — no universal magic bytes; use the exact size equation:
    //   file_size == 84 + num_triangles * 50
    // num_triangles is u32 LE at bytes [80..84].
    // False-positive risk is negligible: stride-50 with exact byte-count match
    // is an extremely specific predicate on arbitrary binary data.
    if input.len() >= 84 {
        if let Some(f) = detect_stl(input) {
            return f;
        }
    }

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

// ── STL detection ─────────────────────────────────────────────────────────────

fn detect_stl(data: &[u8]) -> Option<u8> {
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;

    // Reject degenerate/empty mesh
    if n_tris == 0 { return None; }

    // Overflow-safe size check
    let expected = 84usize.checked_add(n_tris.checked_mul(50)?)?;
    if data.len() != expected { return None; }

    println!("Binary STL: {} triangle(s) → FILTER_SHUFFLE4", n_tris);
    Some(FILTER_SHUFFLE4)
}

// ── WAV/BMP helpers ───────────────────────────────────────────────────────────

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

// ── Public filter API ─────────────────────────────────────────────────────────

/// Transform input bytes with the chosen filter before compression.
pub fn apply_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_encode(input, filter as usize)
        }
        FILTER_SHUFFLE4 => shuffle4_encode(input),
        _ => input.to_vec(),
    }
}

/// Reverse the filter applied during compression.
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_decode(input, filter as usize)
        }
        FILTER_SHUFFLE4 => shuffle4_decode(input),
        _ => input.to_vec(),
    }
}

// ── Delta encode/decode ───────────────────────────────────────────────────────

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

// ── Float byte-plane shuffle for binary STL ───────────────────────────────────
//
// Binary STL layout:
//   [80 bytes] freeform ASCII header
//   [ 4 bytes] num_triangles  u32 LE
//   N × 50 bytes per triangle:
//     [12 bytes] normal vector  (3 × float32 LE)
//     [36 bytes] vertices       (9 × float32 LE)
//     [ 2 bytes] attribute      (usually 0x0000)
//
// After shuffle:
//   [84 bytes]     header verbatim  (includes triangle count — needed by decode)
//   [N×12 bytes]   plane 0: byte[0] of every float across all triangles
//   [N×12 bytes]   plane 1: byte[1] of every float
//   [N×12 bytes]   plane 2: byte[2] of every float
//   [N×12 bytes]   plane 3: byte[3] of every float  (exponent + sign — most
//                           compressible: floats of similar magnitude share this)
//   [N×2 bytes]    attribute bytes verbatim
//
// Total: 84 + N*48 + N*2 = 84 + N*50 = input.len()  — size-preserving.
//
// Why this helps LZ dramatically: float32 values for normals and vertices on a
// smooth mesh have very similar exponents. After separation, plane 3 becomes
// long runs of nearly-identical bytes. Planes 0-2 (mantissa) also compress far
// better when freed from exponent interleaving. xz achieves 68.7% on raw STL
// by brute-force; the shuffle makes the LZ job straightforward.

fn shuffle4_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84, "shuffle4_encode: data shorter than STL header");

    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let float_count = n_tris * 12; // 12 floats per triangle (3 normal + 9 vertex)
    let mut plane0 = Vec::with_capacity(float_count);
    let mut plane1 = Vec::with_capacity(float_count);
    let mut plane2 = Vec::with_capacity(float_count);
    let mut plane3 = Vec::with_capacity(float_count);
    let mut attrs  = Vec::with_capacity(n_tris * 2);

    for tri in 0..n_tris {
        let base = 84 + tri * 50;
        // 12 floats × 4 bytes = 48 bytes
        for f in 0..12usize {
            let fb = base + f * 4;
            plane0.push(data[fb]);
            plane1.push(data[fb + 1]);
            plane2.push(data[fb + 2]);
            plane3.push(data[fb + 3]);
        }
        // 2 attribute bytes
        attrs.push(data[base + 48]);
        attrs.push(data[base + 49]);
    }

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..84]);
    out.extend_from_slice(&plane0);
    out.extend_from_slice(&plane1);
    out.extend_from_slice(&plane2);
    out.extend_from_slice(&plane3);
    out.extend_from_slice(&attrs);
    out
}

fn shuffle4_decode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }

    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let plane_size   = n_tris * 12;
    let planes_start = 84usize;
    let attrs_start  = planes_start + 4 * plane_size;
    let expected_len = attrs_start + n_tris * 2;

    if data.len() < expected_len {
        eprintln!(
            "shuffle4_decode: data too short — have {} bytes, need {} (n_tris={})",
            data.len(), expected_len, n_tris
        );
        return data.to_vec();
    }

    let plane0 = &data[planes_start               ..planes_start +     plane_size];
    let plane1 = &data[planes_start +   plane_size..planes_start + 2 * plane_size];
    let plane2 = &data[planes_start + 2*plane_size..planes_start + 3 * plane_size];
    let plane3 = &data[planes_start + 3*plane_size..planes_start + 4 * plane_size];
    let attrs  = &data[attrs_start                ..attrs_start + n_tris * 2];

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..84]);

    for tri in 0..n_tris {
        for f in 0..12usize {
            let idx = tri * 12 + f;
            out.push(plane0[idx]);
            out.push(plane1[idx]);
            out.push(plane2[idx]);
            out.push(plane3[idx]);
        }
        out.push(attrs[tri * 2]);
        out.push(attrs[tri * 2 + 1]);
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

    #[test]
    fn roundtrip_shuffle4_minimal_stl() {
        // Minimal valid binary STL: 84-byte header + 2 triangles
        let n_tris: u32 = 2;
        let mut data = vec![0u8; 80];
        data.extend_from_slice(&n_tris.to_le_bytes());
        // Fill 2×50 triangle bytes with recognisable pattern
        for i in 0u8..100 {
            data.push(i);
        }
        assert_eq!(data.len(), 84 + 2 * 50);

        let enc = apply_filter(&data, FILTER_SHUFFLE4);
        assert_eq!(enc.len(), data.len(), "shuffle4 must be size-preserving");

        let dec = undo_filter(&enc, FILTER_SHUFFLE4);
        assert_eq!(dec, data, "shuffle4 roundtrip failed");
    }

    #[test]
    fn roundtrip_shuffle4_larger_stl() {
        use std::f32::consts::PI;
        let n_tris: u32 = 500;
        let mut data = vec![0u8; 84 + 500 * 50];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        // Fill with float-like data (varied but correlated, like real geometry)
        for tri in 0..500usize {
            let base = 84 + tri * 50;
            let angle = (tri as f32) * PI / 250.0;
            let floats: [f32; 12] = [
                angle.sin(), angle.cos(), 0.0,
                angle.sin() * 5.0, angle.cos() * 5.0, 0.0,
                angle.sin() * 5.0, angle.cos() * 5.0, 1.0,
                angle.sin() * 5.0, angle.cos() * 5.0, -1.0,
            ];
            for (i, &f) in floats.iter().enumerate() {
                let bytes = f.to_le_bytes();
                data[base + i*4..base + i*4 + 4].copy_from_slice(&bytes);
            }
        }

        let enc = apply_filter(&data, FILTER_SHUFFLE4);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4);
        assert_eq!(dec, data, "shuffle4 roundtrip failed on geometric data");
    }

    #[test]
    fn detect_binary_stl() {
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_SHUFFLE4);
    }

    #[test]
    fn detect_stl_rejects_wrong_size() {
        // Correct header but one extra byte — must not detect as STL
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50 + 1];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4);
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let mut data = vec![0u8; 84];
        let n_tris: u32 = 0;
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    #[test]
    fn detect_stl_rejects_short_data() {
        let data = vec![0u8; 50];
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }
            }
