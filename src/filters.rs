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
//!   5 = float byte-plane shuffle (binary STL)
//!       Transposes N×12 float32 values into 4 byte-planes.
//!       Size-preserving. Header and attribute bytes passed verbatim.
//!   6 = float byte-plane shuffle (binary PLY)
//!       Parses ASCII header to locate vertex float32 section.
//!       Transposes all vertex floats into 4 byte-planes.
//!       Only fires when all vertex properties are float32.
//!       Header and face data passed verbatim.
//!
//! Detection order:
//!   1. Binary STL  — exact size equation (no magic bytes)
//!   2. WAV / RIFF  — magic `RIFF....WAVE`
//!   3. BMP         — magic `BM`
//!   4. Binary PLY  — magic `ply\n` + `format binary_little_endian`
//!   5. Stride entropy probe — fires on headerless strided int16 binary

pub const FILTER_NONE:     u8 = 0;
pub const FILTER_DELTA1:   u8 = 1;
pub const FILTER_DELTA2:   u8 = 2;
pub const FILTER_DELTA3:   u8 = 3;
pub const FILTER_DELTA4:   u8 = 4;
pub const FILTER_SHUFFLE4: u8 = 5;
pub const FILTER_PLY:      u8 = 6;

/// Minimum file size before the entropy probe runs.
const PROBE_MIN_BYTES: usize = 512;

/// Entropy improvement threshold (bits/byte) for the stride-2 probe.
/// Smooth int16 terrain: 2–5 bits/byte. Random/mixed binary: ~0–0.5.
const PROBE_DELTA2_THRESHOLD: f64 = 1.0;

// ── Public API ────────────────────────────────────────────────────────────────

/// Inspect magic bytes and file structure to determine the best filter.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL — exact size equation, no magic
    if input.len() >= 84 {
        if let Some(f) = detect_stl(input) { return f; }
    }

    // 2. WAV / RIFF
    if &input[0..4] == b"RIFF" && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }

    // 3. BMP
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    // 4. Binary PLY — before stride probe to avoid false interaction
    if input.len() >= 4 && &input[0..4] == b"ply\n" {
        if let Some(f) = detect_ply(input) { return f; }
    }

    // 5. Stride entropy probe (headerless strided binary e.g. terrain .raw)
    if input.len() >= PROBE_MIN_BYTES {
        let improvement = probe_delta2_improvement(input);
        if improvement >= PROBE_DELTA2_THRESHOLD {
            println!(
                "Stride probe: delta2 entropy improvement {:.2} bits/byte → FILTER_DELTA2",
                improvement
            );
            return FILTER_DELTA2;
        }
        if improvement > 0.3 {
            println!(
                "Stride probe: delta2 improvement {:.2} bits/byte — below threshold {:.1}, no filter",
                improvement, PROBE_DELTA2_THRESHOLD
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
        FILTER_SHUFFLE4 => shuffle4_stl_encode(input),
        FILTER_PLY      => shuffle4_ply_encode(input),
        _               => input.to_vec(),
    }
}

/// Reverse the filter applied during compression.
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_decode(input, filter as usize)
        }
        FILTER_SHUFFLE4 => shuffle4_stl_decode(input),
        FILTER_PLY      => shuffle4_ply_decode(input),
        _               => input.to_vec(),
    }
}

// ── STL detection ─────────────────────────────────────────────────────────────

fn detect_stl(data: &[u8]) -> Option<u8> {
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return None; }
    let expected = 84usize.checked_add(n_tris.checked_mul(50)?)?;
    if data.len() != expected { return None; }
    println!("Binary STL: {} triangle(s) → FILTER_SHUFFLE4", n_tris);
    Some(FILTER_SHUFFLE4)
}

// ── PLY detection and layout parsing ─────────────────────────────────────────

struct PlyLayout {
    /// Byte offset of first byte after `end_header\n` (or `end_header\r\n`).
    header_end:        usize,
    vertex_count:      usize,
    floats_per_vertex: usize,
}

/// Parse binary PLY header. Returns None if:
/// - No `end_header` found
/// - Format is not `binary_little_endian`
/// - Vertex element has no float properties
/// - Vertex element has any non-float properties (mixed layout — skip to be safe)
fn parse_ply_layout(data: &[u8]) -> Option<PlyLayout> {
    // Find end_header — check both LF and CRLF line endings
    const END_LF:   &[u8] = b"end_header\n";
    const END_CRLF: &[u8] = b"end_header\r\n";

    let header_end = find_subsequence(data, END_CRLF)
        .map(|p| p + END_CRLF.len())
        .or_else(|| find_subsequence(data, END_LF).map(|p| p + END_LF.len()))?;

    let header = std::str::from_utf8(&data[..header_end]).ok()?;

    // Must be binary little-endian
    if !header.contains("format binary_little_endian") { return None; }

    let mut vertex_count      = 0usize;
    let mut floats_per_vertex = 0usize;
    let mut in_vertex         = false;
    let mut vertex_all_float  = true;

    for line in header.lines() {
        let line = line.trim();
        if line.starts_with("element vertex ") {
            let n = line["element vertex ".len()..].trim().parse::<usize>().ok()?;
            vertex_count  = n;
            in_vertex     = true;
        } else if line.starts_with("element ") {
            // Any other element ends the vertex property block
            in_vertex = false;
        } else if in_vertex && line.starts_with("property float ") {
            floats_per_vertex += 1;
        } else if in_vertex && line.starts_with("property ") {
            // Non-float property in vertex element — skip filter to avoid misalignment
            vertex_all_float = false;
        }
    }

    if vertex_count == 0 || floats_per_vertex == 0 || !vertex_all_float {
        return None;
    }

    Some(PlyLayout { header_end, vertex_count, floats_per_vertex })
}

fn detect_ply(data: &[u8]) -> Option<u8> {
    let layout = parse_ply_layout(data)?;

    // Sanity: vertex section must fit within the file
    let vertex_bytes = layout.vertex_count
        .checked_mul(layout.floats_per_vertex)?
        .checked_mul(4)?;
    if layout.header_end.checked_add(vertex_bytes)? > data.len() { return None; }

    println!(
        "Binary PLY: {} vertices × {} float32 properties ({} bytes) → FILTER_PLY",
        layout.vertex_count,
        layout.floats_per_vertex,
        vertex_bytes,
    );
    Some(FILTER_PLY)
}

// ── PLY shuffle encode / decode ───────────────────────────────────────────────
//
// Layout after encode:
//   [header verbatim, header_end bytes]
//   [plane0: byte[0] of every vertex float, float_count bytes]
//   [plane1: byte[1] of every vertex float, float_count bytes]
//   [plane2: byte[2] of every vertex float, float_count bytes]
//   [plane3: byte[3] of every vertex float, float_count bytes]
//   [face data verbatim, remainder of file]
//
// Size-preserving: header + 4*float_count + face_bytes = original size.
// Decode re-parses the header (stored verbatim) to recover the same layout.

fn shuffle4_ply_encode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };

    let float_count  = layout.vertex_count * layout.floats_per_vertex;
    let vertex_bytes = float_count * 4;
    let vertex_start = layout.header_end;
    let vertex_end   = vertex_start + vertex_bytes;

    if vertex_end > data.len() { return data.to_vec(); }

    let mut plane0 = Vec::with_capacity(float_count);
    let mut plane1 = Vec::with_capacity(float_count);
    let mut plane2 = Vec::with_capacity(float_count);
    let mut plane3 = Vec::with_capacity(float_count);

    for chunk in data[vertex_start..vertex_end].chunks_exact(4) {
        plane0.push(chunk[0]);
        plane1.push(chunk[1]);
        plane2.push(chunk[2]);
        plane3.push(chunk[3]);
    }

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..vertex_start]);
    out.extend_from_slice(&plane0);
    out.extend_from_slice(&plane1);
    out.extend_from_slice(&plane2);
    out.extend_from_slice(&plane3);
    out.extend_from_slice(&data[vertex_end..]);
    out
}

fn shuffle4_ply_decode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };

    let float_count  = layout.vertex_count * layout.floats_per_vertex;
    let plane_size   = float_count;
    let vertex_start = layout.header_end;
    let planes_end   = vertex_start + 4 * plane_size;

    if planes_end > data.len() {
        eprintln!(
            "shuffle4_ply_decode: data too short — have {} bytes, need {}",
            data.len(), planes_end
        );
        return data.to_vec();
    }

    let plane0 = &data[vertex_start              ..vertex_start +     plane_size];
    let plane1 = &data[vertex_start +   plane_size..vertex_start + 2 * plane_size];
    let plane2 = &data[vertex_start + 2*plane_size..vertex_start + 3 * plane_size];
    let plane3 = &data[vertex_start + 3*plane_size..vertex_start + 4 * plane_size];

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..vertex_start]);
    for i in 0..float_count {
        out.push(plane0[i]);
        out.push(plane1[i]);
        out.push(plane2[i]);
        out.push(plane3[i]);
    }
    out.extend_from_slice(&data[planes_end..]);
    out
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

// ── Stride entropy probe ──────────────────────────────────────────────────────

fn probe_delta2_improvement(data: &[u8]) -> f64 {
    const SAMPLE: usize = 8192;
    let sample = if data.len() > SAMPLE { &data[..SAMPLE] } else { data };
    if sample.len() < 4 { return 0.0; }

    let raw_entropy = byte_entropy(sample);
    let mut delta = sample.to_vec();
    for i in 2..delta.len() {
        delta[i] = sample[i].wrapping_sub(sample[i - 2]);
    }
    raw_entropy - byte_entropy(&delta)
}

fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for &b in data { freq[b as usize] += 1; }
    let n = data.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
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

// ── STL shuffle encode / decode ───────────────────────────────────────────────
//
// Binary STL layout:
//   [80 bytes] freeform header
//   [ 4 bytes] num_triangles u32 LE
//   N × 50 bytes: [12 normal+vertex floats][2 attr bytes]
//
// After encode:
//   [84 bytes]   header verbatim
//   [N×12 bytes] plane 0..3: byte[i] of every float
//   [N×2 bytes]  attribute bytes verbatim

fn shuffle4_stl_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let float_count = n_tris * 12;
    let mut plane0 = Vec::with_capacity(float_count);
    let mut plane1 = Vec::with_capacity(float_count);
    let mut plane2 = Vec::with_capacity(float_count);
    let mut plane3 = Vec::with_capacity(float_count);
    let mut attrs  = Vec::with_capacity(n_tris * 2);

    for tri in 0..n_tris {
        let base = 84 + tri * 50;
        for f in 0..12usize {
            let fb = base + f * 4;
            plane0.push(data[fb]);
            plane1.push(data[fb + 1]);
            plane2.push(data[fb + 2]);
            plane3.push(data[fb + 3]);
        }
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

fn shuffle4_stl_decode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let plane_size   = n_tris * 12;
    let planes_start = 84usize;
    let attrs_start  = planes_start + 4 * plane_size;
    let expected_len = attrs_start + n_tris * 2;

    if data.len() < expected_len {
        eprintln!(
            "shuffle4_stl_decode: data too short — have {} need {} (n_tris={})",
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

// ── Utility ───────────────────────────────────────────────────────────────────

/// Find first occurrence of `needle` in `haystack`. Returns start index.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() { return None; }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Delta ─────────────────────────────────────────────────────────────────

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
        assert!(dominant_pct > 0.8,
            "residuals not constant enough: {:.1}%", dominant_pct * 100.0);
    }

    // ── STL shuffle ───────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_shuffle4_stl_minimal() {
        let n_tris: u32 = 2;
        let mut data = vec![0u8; 80];
        data.extend_from_slice(&n_tris.to_le_bytes());
        for i in 0u8..100 { data.push(i); }
        assert_eq!(data.len(), 84 + 2 * 50);
        let enc = apply_filter(&data, FILTER_SHUFFLE4);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4);
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_shuffle4_stl_larger() {
        let n_tris: u32 = 500;
        let mut data = vec![0u8; 84 + 500 * 50];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        for tri in 0..500usize {
            let base = 84 + tri * 50;
            let v = (tri as u32 * 1000) % 6284;
            let floats: [u32; 12] = [v, 6284-v, tri as u32,
                v*5, (6284-v)*5, 0, v*5, (6284-v)*5, 1000,
                v*5, (6284-v)*5, 0xFFFF];
            for (i, &f) in floats.iter().enumerate() {
                data[base + i*4..base + i*4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        let enc = apply_filter(&data, FILTER_SHUFFLE4);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4);
        assert_eq!(dec, data);
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
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50 + 1];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4);
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let mut data = vec![0u8; 84];
        data[80..84].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    // ── PLY shuffle ───────────────────────────────────────────────────────────

    fn make_ply(n_verts: usize, floats_per_vert: usize) -> Vec<u8> {
        let mut hdr = String::new();
        hdr.push_str("ply\n");
        hdr.push_str("format binary_little_endian 1.0\n");
        hdr.push_str(&format!("element vertex {}\n", n_verts));
        for i in 0..floats_per_vert {
            hdr.push_str(&format!("property float coord{}\n", i));
        }
        hdr.push_str("element face 2\n");
        hdr.push_str("property list uchar int vertex_indices\n");
        hdr.push_str("end_header\n");

        let mut data = hdr.into_bytes();
        // vertex floats: recognisable pattern
        for v in 0..n_verts {
            for f in 0..floats_per_vert {
                let val = (v * floats_per_vert + f) as u32;
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        // 2 dummy face records: uchar(3) + 3×i32
        for _ in 0..2 {
            data.push(3u8);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&2u32.to_le_bytes());
        }
        data
    }

    #[test]
    fn detect_ply_fires_on_all_float_vertex() {
        let data = make_ply(100, 8);
        assert_eq!(detect_filter(&data), FILTER_PLY);
    }

    #[test]
    fn detect_ply_rejects_ascii_format() {
        let hdr = "ply\nformat ascii 1.0\nelement vertex 10\n\
                   property float x\nend_header\n";
        let mut data = hdr.as_bytes().to_vec();
        data.extend_from_slice(&vec![0u8; 40]);
        assert_ne!(detect_filter(&data), FILTER_PLY);
    }

    #[test]
    fn detect_ply_rejects_mixed_vertex_props() {
        let hdr = "ply\nformat binary_little_endian 1.0\n\
                   element vertex 10\nproperty float x\nproperty uchar r\n\
                   end_header\n";
        let mut data = hdr.as_bytes().to_vec();
        // 10 × (4 + 1) = 50 bytes vertex data
        data.extend_from_slice(&vec![0u8; 50]);
        assert_ne!(detect_filter(&data), FILTER_PLY);
    }

    #[test]
    fn roundtrip_ply_shuffle_small() {
        let data = make_ply(50, 3);
        assert_eq!(detect_filter(&data), FILTER_PLY);
        let enc = apply_filter(&data, FILTER_PLY);
        assert_eq!(enc.len(), data.len(), "PLY shuffle must be size-preserving");
        let dec = undo_filter(&enc, FILTER_PLY);
        assert_eq!(dec, data, "PLY shuffle roundtrip failed");
    }

    #[test]
    fn roundtrip_ply_shuffle_8props() {
        // Matches gen_ply_binary layout: 8 float props per vertex
        let data = make_ply(2000, 8);
        assert_eq!(detect_filter(&data), FILTER_PLY);
        let enc = apply_filter(&data, FILTER_PLY);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_PLY);
        assert_eq!(dec, data, "PLY 8-float roundtrip failed");
    }

    #[test]
    fn ply_shuffle_is_not_identity() {
        // Verify the shuffle actually reorders bytes (not a no-op)
        let data = make_ply(10, 4);
        let enc = apply_filter(&data, FILTER_PLY);
        // Find header end
        let hdr_end = find_subsequence(&data, b"end_header\n").unwrap()
            + b"end_header\n".len();
        assert_ne!(
            &enc[hdr_end..hdr_end + 4],
            &data[hdr_end..hdr_end + 4],
            "shuffle should reorder bytes"
        );
    }

    // ── Stride probe ──────────────────────────────────────────────────────────

    #[test]
    fn probe_fires_on_smooth_int16() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let imp = probe_delta2_improvement(&data);
        assert!(imp >= PROBE_DELTA2_THRESHOLD,
            "probe should fire on smooth terrain (improvement={:.2})", imp);
    }

    #[test]
    fn probe_does_not_fire_on_random() {
        let mut state: u32 = 0xdeadbeef;
        let data: Vec<u8> = (0..1024).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }).collect();
        let imp = probe_delta2_improvement(&data);
        assert!(imp < PROBE_DELTA2_THRESHOLD,
            "probe should not fire on random data (improvement={:.2})", imp);
    }

    #[test]
    fn probe_does_not_fire_on_text() {
        let data: Vec<u8> = b"the quick brown fox jumps over the lazy dog \
            hello world foo bar baz qux the end and the beginning \
            the quick brown fox jumps over the lazy dog hello world"
            .iter().cycle().take(512).copied().collect();
        let imp = probe_delta2_improvement(&data);
        assert!(imp < PROBE_DELTA2_THRESHOLD,
            "probe should not fire on text (improvement={:.2})", imp);
    }

    #[test]
    fn roundtrip_delta2_int16_terrain() {
        let mut data = Vec::with_capacity(2048);
        for i in 0..1024usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let enc = apply_filter(&data, FILTER_DELTA2);
        let dec = undo_filter(&enc, FILTER_DELTA2);
        assert_eq!(dec, data);
    }
                         }
