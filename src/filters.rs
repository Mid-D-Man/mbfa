// src/filters.rs
//! Pre/post compression delta filters + byte-plane shuffle + BCJ transform.
//!
//! Applied to the raw input BEFORE folding, reversed AFTER unfolding.
//!
//! Filter flags stored in header byte 3:
//!   0 = none
//!   1 = delta stride 1  (generic 8-bit binary)
//!   2 = delta stride 2  (16-bit mono PCM / 16-bit pixels)
//!   3 = delta stride 3  (24-bit RGB pixels)
//!   4 = delta stride 4  (32-bit RGBA / stereo 16-bit PCM)
//!   5 = float byte-plane shuffle (binary STL) — BACKWARD COMPAT ONLY
//!       New compressions of STL use flag 7 instead. Flag 5 remains
//!       supported in undo_filter so old files decompress correctly.
//!   6 = float byte-plane shuffle (binary PLY) — BACKWARD COMPAT ONLY
//!       New compressions of PLY use flag 8 instead.
//!   7 = STL byte-plane shuffle + per-plane delta1  (NEW)
//!       Byte-plane shuffle as in flag 5, then delta1-encode within each
//!       of the 4 planes independently. The exponent plane (byte 3 of each
//!       float) becomes near-constant for smooth geometry; the mantissa
//!       planes compress better with delta applied before LZ.
//!   8 = PLY byte-plane shuffle + per-plane delta1  (NEW)
//!       Same compound transform for binary PLY vertex floats.
//!   9 = x86 BCJ (Branch-Call-Jump) for PE/COFF executables  (NEW)
//!       Converts x86 CALL (E8) and JMP (E9) relative offsets to absolute
//!       addresses. The same function target then produces identical 4-byte
//!       sequences regardless of call site position, greatly improving LZ
//!       match frequency in native code sections. Fully reversible bijection.
//!       Also normalises 0F 8x conditional jumps (6-byte form).
//!
//! Detection order:
//!   1. Binary STL  — exact size equation → FILTER_SHUFFLE4_DELTA (7)
//!   2. WAV / RIFF  — magic `RIFF....WAVE`
//!   3. BMP         — magic `BM`
//!   4. Binary PLY  — magic `ply\n` + `format binary_little_endian` → FILTER_PLY_DELTA (8)
//!   5. PE/COFF     — magic `MZ` + PE offset + `PE\0\0` → FILTER_BCJ (9)
//!   6. Multi-stride entropy probe — fires on headerless strided binary
//!      Tests strides 1-4, picks the best, threshold 0.45 bits/byte.

pub const FILTER_NONE:           u8 = 0;
pub const FILTER_DELTA1:         u8 = 1;
pub const FILTER_DELTA2:         u8 = 2;
pub const FILTER_DELTA3:         u8 = 3;
pub const FILTER_DELTA4:         u8 = 4;
/// Byte-plane shuffle for STL — kept for backward compatibility only.
/// detect_filter now returns FILTER_SHUFFLE4_DELTA (7) for new compressions.
pub const FILTER_SHUFFLE4:       u8 = 5;
/// Byte-plane shuffle for PLY — kept for backward compatibility only.
/// detect_filter now returns FILTER_PLY_DELTA (8) for new compressions.
pub const FILTER_PLY:            u8 = 6;
/// STL: byte-plane shuffle + per-plane delta1 (compound, NEW).
pub const FILTER_SHUFFLE4_DELTA: u8 = 7;
/// PLY: byte-plane shuffle + per-plane delta1 (compound, NEW).
pub const FILTER_PLY_DELTA:      u8 = 8;
/// x86 BCJ normalization for PE/COFF binaries (NEW).
pub const FILTER_BCJ:            u8 = 9;

/// Minimum file size before the entropy stride probe runs.
const PROBE_MIN_BYTES: usize = 512;

/// Entropy improvement threshold (bits/byte) for the stride probe.
/// 0.45 catches smooth multi-frequency heightmaps while avoiding text/random.
const PROBE_DELTA_THRESHOLD: f64 = 0.45;

// ── Public API ────────────────────────────────────────────────────────────────

/// Inspect magic bytes and file structure to determine the best filter.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL — exact size equation (no magic bytes).
    //    Returns compound flag 7, not the old flag 5.
    if input.len() >= 84 {
        if detect_stl(input).is_some() {
            println!("Binary STL detected → FILTER_SHUFFLE4_DELTA (compound)");
            return FILTER_SHUFFLE4_DELTA;
        }
    }

    // 2. WAV / RIFF
    if &input[0..4] == b"RIFF" && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }

    // 3. BMP
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    // 4. Binary PLY — returns compound flag 8, not the old flag 6.
    if input.len() >= 4 && &input[0..4] == b"ply\n" {
        if parse_ply_layout(input).is_some() {
            println!("Binary PLY detected → FILTER_PLY_DELTA (compound)");
            return FILTER_PLY_DELTA;
        }
    }

    // 5. PE/COFF binary (must come before stride probe).
    if detect_pe_coff(input) {
        println!("PE/COFF binary detected → FILTER_BCJ");
        return FILTER_BCJ;
    }

    // 6. Multi-stride entropy probe (headerless strided binary: terrain, etc.)
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
        FILTER_SHUFFLE4       => shuffle4_stl_encode(input),
        FILTER_PLY            => shuffle4_ply_encode(input),
        FILTER_SHUFFLE4_DELTA => shuffle4_stl_delta_encode(input),
        FILTER_PLY_DELTA      => shuffle4_ply_delta_encode(input),
        FILTER_BCJ            => bcj_x86_encode(input),
        _                     => input.to_vec(),
    }
}

/// Reverse the filter applied during compression.
/// Handles both old flags (5, 6) for backward compatibility and new flags (7, 8, 9).
pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_decode(input, filter as usize)
        }
        FILTER_SHUFFLE4       => shuffle4_stl_decode(input),
        FILTER_PLY            => shuffle4_ply_decode(input),
        FILTER_SHUFFLE4_DELTA => shuffle4_stl_delta_decode(input),
        FILTER_PLY_DELTA      => shuffle4_ply_delta_decode(input),
        FILTER_BCJ            => bcj_x86_decode(input),
        _                     => input.to_vec(),
    }
}

// ── STL detection ─────────────────────────────────────────────────────────────

/// Returns Some(()) if the data matches the binary STL exact-size equation.
fn detect_stl(data: &[u8]) -> Option<()> {
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return None; }
    let expected = 84usize.checked_add(n_tris.checked_mul(50)?)?;
    if data.len() != expected { return None; }
    println!("Binary STL: {} triangle(s)", n_tris);
    Some(())
}

// ── PLY detection and layout parsing ─────────────────────────────────────────

struct PlyLayout {
    /// Byte offset of first byte after `end_header\n` (or CRLF variant).
    header_end:        usize,
    vertex_count:      usize,
    floats_per_vertex: usize,
}

/// Parse binary PLY header. Returns None if format is not binary_little_endian,
/// vertex has non-float properties, or vertex section exceeds the file.
fn parse_ply_layout(data: &[u8]) -> Option<PlyLayout> {
    const END_LF:   &[u8] = b"end_header\n";
    const END_CRLF: &[u8] = b"end_header\r\n";

    let header_end = find_subsequence(data, END_CRLF)
        .map(|p| p + END_CRLF.len())
        .or_else(|| find_subsequence(data, END_LF).map(|p| p + END_LF.len()))?;

    let header = std::str::from_utf8(&data[..header_end]).ok()?;
    if !header.contains("format binary_little_endian") { return None; }

    let mut vertex_count      = 0usize;
    let mut floats_per_vertex = 0usize;
    let mut in_vertex         = false;
    let mut vertex_all_float  = true;

    for line in header.lines() {
        let line = line.trim();
        if line.starts_with("element vertex ") {
            vertex_count  = line["element vertex ".len()..].trim().parse().ok()?;
            in_vertex     = true;
        } else if line.starts_with("element ") {
            in_vertex = false;
        } else if in_vertex && line.starts_with("property float ") {
            floats_per_vertex += 1;
        } else if in_vertex && line.starts_with("property ") {
            vertex_all_float = false;
        }
    }

    if vertex_count == 0 || floats_per_vertex == 0 || !vertex_all_float {
        return None;
    }

    let vertex_bytes = vertex_count.checked_mul(floats_per_vertex)?.checked_mul(4)?;
    if header_end.checked_add(vertex_bytes)? > data.len() { return None; }

    println!(
        "Binary PLY: {} vertices × {} float32 properties ({} bytes)",
        vertex_count, floats_per_vertex, vertex_bytes,
    );
    Some(PlyLayout { header_end, vertex_count, floats_per_vertex })
}

// ── PLY shuffle encode / decode (simple, backward compat flag 6) ─────────────

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

// ── PLY compound: shuffle + per-plane delta (flag 8) ─────────────────────────

/// Encode: byte-plane shuffle then delta1-encode within each of the 4 planes.
///
/// Right-to-left in-place delta: `out[i] = out[i] - out[i-1]` for i from end
/// down to start+1. This means out[i-1] is always the original (unmodified)
/// value when out[i] is processed, giving a correct reversible delta.
fn shuffle4_ply_delta_encode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };

    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let vertex_end  = layout.header_end + 4 * plane_size;
    if vertex_end > data.len() { return data.to_vec(); }

    // Step 1: byte-plane shuffle
    let mut out = shuffle4_ply_encode(data);

    // Step 2: delta1-encode within each plane independently (right-to-left)
    for plane_idx in 0..4usize {
        let ps = layout.header_end + plane_idx * plane_size;
        let pe = ps + plane_size;
        if pe > out.len() { break; }
        for i in (ps + 1..pe).rev() {
            let prev = out[i - 1];
            out[i] = out[i].wrapping_sub(prev);
        }
    }
    out
}

/// Decode: undo per-plane delta1 then undo byte-plane shuffle.
fn shuffle4_ply_delta_decode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };

    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let planes_end  = layout.header_end + 4 * plane_size;

    if planes_end > data.len() {
        eprintln!(
            "shuffle4_ply_delta_decode: data too short — have {} bytes, need {}",
            data.len(), planes_end
        );
        return data.to_vec();
    }

    // Step 1: undo delta1 within each plane (left-to-right cumsum)
    let mut undelta = data.to_vec();
    for plane_idx in 0..4usize {
        let ps = layout.header_end + plane_idx * plane_size;
        let pe = ps + plane_size;
        for i in ps + 1..pe {
            let prev = undelta[i - 1];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }

    // Step 2: undo byte-plane shuffle
    shuffle4_ply_decode(&undelta)
}

// ── STL shuffle encode / decode (simple, backward compat flag 5) ─────────────

fn shuffle4_stl_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let float_count = n_tris * 12;
    let mut plane0  = Vec::with_capacity(float_count);
    let mut plane1  = Vec::with_capacity(float_count);
    let mut plane2  = Vec::with_capacity(float_count);
    let mut plane3  = Vec::with_capacity(float_count);
    let mut attrs   = Vec::with_capacity(n_tris * 2);

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

// ── STL compound: shuffle + per-plane delta (flag 7) ─────────────────────────

/// Encode: byte-plane shuffle then delta1-encode within each of the 4 planes.
///
/// Post-shuffle layout:
///   [0..84]              : header verbatim
///   [84..84+n*12]        : plane0 (float byte 0 — LSB)
///   [84+n*12..84+n*24]   : plane1 (float byte 1)
///   [84+n*24..84+n*36]   : plane2 (float byte 2)
///   [84+n*36..84+n*48]   : plane3 (float byte 3 — MSB, sign+exponent)
///   [84+n*48..84+n*50]   : attribute bytes verbatim
/// where n = n_tris.
fn shuffle4_stl_delta_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    // Step 1: byte-plane shuffle
    let mut out = shuffle4_stl_encode(data);

    // Step 2: delta1-encode within each of the 4 planes (right-to-left in-place)
    let plane_size   = n_tris * 12;
    let planes_start = 84usize;

    for plane_idx in 0..4usize {
        let ps = planes_start + plane_idx * plane_size;
        let pe = ps + plane_size;
        if pe > out.len() { break; }
        for i in (ps + 1..pe).rev() {
            let prev = out[i - 1];
            out[i] = out[i].wrapping_sub(prev);
        }
    }
    out
}

/// Decode: undo per-plane delta1 then undo byte-plane shuffle.
fn shuffle4_stl_delta_decode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let plane_size   = n_tris * 12;
    let planes_start = 84usize;
    let expected_len = planes_start + 4 * plane_size + n_tris * 2;

    if data.len() < expected_len {
        eprintln!(
            "shuffle4_stl_delta_decode: data too short — have {} need {} (n_tris={})",
            data.len(), expected_len, n_tris
        );
        return data.to_vec();
    }

    // Step 1: undo delta1 within each plane (left-to-right cumsum)
    let mut undelta = data.to_vec();
    for plane_idx in 0..4usize {
        let ps = planes_start + plane_idx * plane_size;
        let pe = ps + plane_size;
        for i in ps + 1..pe {
            let prev = undelta[i - 1];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }

    // Step 2: undo byte-plane shuffle
    shuffle4_stl_decode(&undelta)
}

// ── PE/COFF detection ─────────────────────────────────────────────────────────

/// Returns true if data starts with a valid DOS/PE header:
/// MZ magic at 0, valid e_lfanew at 0x3C, and PE\0\0 at that offset.
fn detect_pe_coff(data: &[u8]) -> bool {
    if data.len() < 0x40 { return false; }
    if data[0] != b'M' || data[1] != b'Z' { return false; }
    let pe_offset = u32::from_le_bytes([data[0x3C], data[0x3D], data[0x3E], data[0x3F]]) as usize;
    if pe_offset.saturating_add(4) > data.len() { return false; }
    data[pe_offset..pe_offset + 4] == *b"PE\x00\x00"
}

// ── BCJ x86 encode / decode ───────────────────────────────────────────────────
//
// Forward transform (encode): relative offset → absolute address.
// For CALL E8 at position i: abs = rel + (i + 5).
// For JMP  E9 at position i: same formula.
// For JCC  0F 8x at position i: abs = rel + (i + 6).
//
// Reverse transform (decode): absolute address → relative offset.
// Inverts encode exactly: rel = abs - (i + 5) for E8/E9.
//
// False positives: data bytes that happen to be E8/E9 are transformed
// "incorrectly" but decode undoes exactly the same transformation,
// restoring the original bytes. Same approach as XZ's x86 BCJ filter.

fn bcj_x86_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n       = data.len();
    let mut i   = 0;

    while i < n {
        if (data[i] == 0xE8 || data[i] == 0xE9) && i + 5 <= n {
            let rel = i32::from_le_bytes([
                data[i + 1], data[i + 2], data[i + 3], data[i + 4],
            ]);
            let abs = rel.wrapping_add(i as i32 + 5);
            out[i + 1..i + 5].copy_from_slice(&abs.to_le_bytes());
            i += 5;
        } else if i + 6 <= n && data[i] == 0x0F && (data[i + 1] & 0xF0) == 0x80 {
            let rel = i32::from_le_bytes([
                data[i + 2], data[i + 3], data[i + 4], data[i + 5],
            ]);
            let abs = rel.wrapping_add(i as i32 + 6);
            out[i + 2..i + 6].copy_from_slice(&abs.to_le_bytes());
            i += 6;
        } else {
            i += 1;
        }
    }
    out
}

fn bcj_x86_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n       = data.len();
    let mut i   = 0;

    while i < n {
        if (data[i] == 0xE8 || data[i] == 0xE9) && i + 5 <= n {
            let abs = i32::from_le_bytes([
                data[i + 1], data[i + 2], data[i + 3], data[i + 4],
            ]);
            let rel = abs.wrapping_sub(i as i32 + 5);
            out[i + 1..i + 5].copy_from_slice(&rel.to_le_bytes());
            i += 5;
        } else if i + 6 <= n && data[i] == 0x0F && (data[i + 1] & 0xF0) == 0x80 {
            let abs = i32::from_le_bytes([
                data[i + 2], data[i + 3], data[i + 4], data[i + 5],
            ]);
            let rel = abs.wrapping_sub(i as i32 + 6);
            out[i + 2..i + 6].copy_from_slice(&rel.to_le_bytes());
            i += 6;
        } else {
            i += 1;
        }
    }
    out
}

// ── WAV/BMP stride detection helpers ─────────────────────────────────────────

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

// ── Multi-stride entropy probe ────────────────────────────────────────────────

fn probe_delta_improvement(data: &[u8], stride: usize) -> f64 {
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

fn probe_best_stride(data: &[u8]) -> (u8, f64) {
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

// ── Utility ───────────────────────────────────────────────────────────────────

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() { return None; }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Delta roundtrip ───────────────────────────────────────────────────────

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
    fn inplace_delta_encode_rtorl_is_correct() {
        // Verified by hand: [10, 12, 15, 11, 14] → [10, 2, 3, 252, 3]
        let orig = vec![10u8, 12, 15, 11, 14];
        let enc  = delta_encode(&orig, 1);
        assert_eq!(enc, vec![10, 2, 3, 252, 3]);
        let dec = delta_decode(&enc, 1);
        assert_eq!(dec, orig);
    }

    // ── Stride probe ─────────────────────────────────────────────────────────

    #[test]
    fn probe_fires_on_smooth_int16() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let imp = probe_delta_improvement(&data, 2);
        assert!(imp >= PROBE_DELTA_THRESHOLD,
            "probe should fire (improvement={:.2}, threshold={:.2})", imp, PROBE_DELTA_THRESHOLD);
    }

    #[test]
    fn probe_does_not_fire_on_random() {
        let mut state: u32 = 0xdeadbeef;
        let data: Vec<u8> = (0..1024).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }).collect();
        let imp = probe_delta_improvement(&data, 2);
        assert!(imp < PROBE_DELTA_THRESHOLD,
            "probe should not fire on random (improvement={:.2})", imp);
    }

    #[test]
    fn probe_does_not_fire_on_text() {
        let data: Vec<u8> = b"the quick brown fox jumps over the lazy dog \
            hello world foo bar baz qux the end and the beginning"
            .iter().cycle().take(512).copied().collect();
        let imp = probe_delta_improvement(&data, 2);
        assert!(imp < PROBE_DELTA_THRESHOLD,
            "probe should not fire on text (improvement={:.2})", imp);
    }

    #[test]
    fn probe_best_stride_picks_delta2_for_int16() {
        let mut data = Vec::with_capacity(8192);
        for i in 0..4096usize {
            let h: u16 = ((i * 16) % 65536) as u16;
            data.extend_from_slice(&h.to_le_bytes());
        }
        let (filter, imp) = probe_best_stride(&data);
        assert_eq!(filter, FILTER_DELTA2,
            "expected FILTER_DELTA2 for int16, got FILTER_DELTA{}", filter);
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
        assert!(imp >= PROBE_DELTA_THRESHOLD,
            "probe should fire on smooth int32 (improvement={:.2})", imp);
        assert_eq!(filter, FILTER_DELTA4,
            "expected FILTER_DELTA4 for int32, got FILTER_DELTA{}", filter);
    }

    #[test]
    fn probe_multi_freq_terrain_fires_at_lower_threshold() {
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
        assert!(imp >= PROBE_DELTA_THRESHOLD,
            "multi-freq terrain should fire (improvement={:.2}, threshold={:.2})",
            imp, PROBE_DELTA_THRESHOLD);
        assert_eq!(filter, FILTER_DELTA2);
    }

    // ── STL helpers ───────────────────────────────────────────────────────────

    fn make_stl(n_tris: u32) -> Vec<u8> {
        let mut data = vec![0u8; 84 + n_tris as usize * 50];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        for tri in 0..n_tris as usize {
            let base = 84 + tri * 50;
            for byte_pos in 0..48usize {
                data[base + byte_pos] = ((tri * 48 + byte_pos) & 0xFF) as u8;
            }
        }
        data
    }

    // ── STL: simple shuffle (flag 5, backward compat) ─────────────────────────

    #[test]
    fn roundtrip_shuffle4_stl_simple_flag5() {
        let data = make_stl(4);
        let enc = apply_filter(&data, FILTER_SHUFFLE4);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4);
        assert_eq!(dec, data, "backward compat flag 5 roundtrip failed");
    }

    #[test]
    fn detect_binary_stl_returns_compound_flag() {
        let data = make_stl(10);
        assert_eq!(detect_filter(&data), FILTER_SHUFFLE4_DELTA,
            "detect_filter should return FILTER_SHUFFLE4_DELTA (7) for STL");
    }

    #[test]
    fn detect_stl_rejects_wrong_size() {
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50 + 1];
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4_DELTA);
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4);
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let data = vec![0u8; 84];
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    // ── STL: compound shuffle+delta (flag 7) ──────────────────────────────────

    #[test]
    fn roundtrip_shuffle4_stl_delta_minimal() {
        let data = make_stl(2);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len(), "compound STL filter must be size-preserving");
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "STL compound roundtrip failed");
    }

    #[test]
    fn roundtrip_shuffle4_stl_delta_larger() {
        let data = make_stl(500);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "STL compound roundtrip failed for 500 tris");
    }

    #[test]
    fn stl_compound_differs_from_simple_shuffle() {
        let data = make_stl(10);
        let simple   = apply_filter(&data, FILTER_SHUFFLE4);
        let compound = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_ne!(&compound[84..], &simple[84..],
            "compound filter should produce different plane bytes than simple shuffle");
    }

    #[test]
    fn stl_delta_reduces_entropy_in_planes() {
        // Geometry designed so per-plane delta provably helps:
        // All 12 floats per triangle ramp linearly from 1.0 → 3.0 across all
        // triangles. Each value is in [1.0, 2.0) or [2.0, 3.0), so byte3
        // (the IEEE 754 sign+exponent MSB) is 0x3F for the first half and 0x40
        // for the second half — two distinct values with roughly equal frequency.
        //
        //   Simple plane3:   [...0x3F, 0x3F, ..., 0x40, 0x40...] → entropy ≈ 1 bit
        //   Compound plane3: [0x3F, 0x00, 0x00, ..., 0x01, 0x00...] → entropy ≈ 0.01 bit
        //
        // This is the geometry the compound filter is designed for: a terrain-like
        // height field where floats change slowly through a small number of
        // exponent-byte buckets. Contrast with full-sphere geometry ([-1, 1] in
        // all dims) where byte3 oscillates between 0x3F and 0xBF, making delta
        // potentially worse than the simple shuffle.
        let n_tris       = 200usize;
        let total_floats = n_tris * 12;

        let mut data = vec![0u8; 84 + n_tris * 50];
        data[80..84].copy_from_slice(&(n_tris as u32).to_le_bytes());

        for tri in 0..n_tris {
            let base = 84 + tri * 50;
            for f in 0..12usize {
                let idx = tri * 12 + f;
                // Linear ramp 1.0 → 3.0: first half in [1.0, 2.0) → byte3=0x3F,
                // second half in [2.0, 3.0) → byte3=0x40.
                let v: f32 = 1.0 + (idx as f32 / total_floats as f32) * 2.0;
                let bytes = v.to_le_bytes();
                data[base + f * 4..base + f * 4 + 4].copy_from_slice(&bytes);
            }
        }

        let simple   = apply_filter(&data, FILTER_SHUFFLE4);
        let compound = apply_filter(&data, FILTER_SHUFFLE4_DELTA);

        // Roundtrip verification
        let dec = undo_filter(&compound, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "compound STL roundtrip failed for ramp geometry");

        // Entropy comparison: compound plane3 must be <= simple plane3.
        let plane_size   = n_tris * 12;
        let plane3_start = 84 + 3 * plane_size;

        let simple_ent   = byte_entropy(&simple[plane3_start..plane3_start + plane_size]);
        let compound_ent = byte_entropy(&compound[plane3_start..plane3_start + plane_size]);

        assert!(
            compound_ent <= simple_ent,
            "compound plane3 entropy ({:.4}) should be <= simple ({:.4}) \
             for ramp geometry (linear 1.0→3.0)",
            compound_ent, simple_ent
        );
    }

    // ── PLY helpers ───────────────────────────────────────────────────────────

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
        for v in 0..n_verts {
            for f in 0..floats_per_vert {
                let val = (v * floats_per_vert + f) as u32;
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        for _ in 0..2 {
            data.push(3u8);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&2u32.to_le_bytes());
        }
        data
    }

    // ── PLY: simple shuffle (flag 6, backward compat) ─────────────────────────

    #[test]
    fn roundtrip_ply_simple_flag6() {
        let data = make_ply(50, 3);
        let enc = apply_filter(&data, FILTER_PLY);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_PLY);
        assert_eq!(dec, data, "backward compat flag 6 roundtrip failed");
    }

    #[test]
    fn detect_ply_returns_compound_flag() {
        let data = make_ply(100, 8);
        assert_eq!(detect_filter(&data), FILTER_PLY_DELTA,
            "detect_filter should return FILTER_PLY_DELTA (8) for binary PLY");
    }

    #[test]
    fn detect_ply_rejects_ascii_format() {
        let hdr = "ply\nformat ascii 1.0\nelement vertex 10\n\
                   property float x\nend_header\n";
        let mut data = hdr.as_bytes().to_vec();
        data.extend_from_slice(&vec![0u8; 40]);
        assert_ne!(detect_filter(&data), FILTER_PLY_DELTA);
        assert_ne!(detect_filter(&data), FILTER_PLY);
    }

    #[test]
    fn detect_ply_rejects_mixed_vertex_props() {
        let hdr = "ply\nformat binary_little_endian 1.0\n\
                   element vertex 10\nproperty float x\nproperty uchar r\n\
                   end_header\n";
        let mut data = hdr.as_bytes().to_vec();
        data.extend_from_slice(&vec![0u8; 50]);
        assert_ne!(detect_filter(&data), FILTER_PLY_DELTA);
    }

    // ── PLY: compound shuffle+delta (flag 8) ──────────────────────────────────

    #[test]
    fn roundtrip_ply_compound_small() {
        let data = make_ply(50, 3);
        let enc = apply_filter(&data, FILTER_PLY_DELTA);
        assert_eq!(enc.len(), data.len(), "PLY compound must be size-preserving");
        let dec = undo_filter(&enc, FILTER_PLY_DELTA);
        assert_eq!(dec, data, "PLY compound roundtrip failed");
    }

    #[test]
    fn roundtrip_ply_compound_8props() {
        let data = make_ply(2000, 8);
        let enc = apply_filter(&data, FILTER_PLY_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_PLY_DELTA);
        assert_eq!(dec, data, "PLY compound 8-float roundtrip failed");
    }

    #[test]
    fn ply_compound_differs_from_simple_shuffle() {
        let data = make_ply(10, 4);
        let simple   = apply_filter(&data, FILTER_PLY);
        let compound = apply_filter(&data, FILTER_PLY_DELTA);
        let layout   = parse_ply_layout(&data).unwrap();
        let plane_start = layout.header_end;
        assert_ne!(
            &compound[plane_start..],
            &simple[plane_start..],
            "compound PLY filter should produce different plane bytes"
        );
    }

    // ── BCJ: PE detection ─────────────────────────────────────────────────────

    fn make_minimal_pe() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0] = b'M';
        data[1] = b'Z';
        data[0x3C] = 0x40; // e_lfanew: PE header at 0x40
        data[0x40] = b'P';
        data[0x41] = b'E';
        data[0x42] = 0x00;
        data[0x43] = 0x00;
        data
    }

    #[test]
    fn detect_pe_coff_basic() {
        let data = make_minimal_pe();
        assert!(detect_pe_coff(&data), "should detect PE");
    }

    #[test]
    fn detect_pe_coff_rejects_non_pe() {
        let data = b"Not a PE file at all".to_vec();
        assert!(!detect_pe_coff(&data));
    }

    #[test]
    fn detect_pe_coff_rejects_missing_signature() {
        let mut data = make_minimal_pe();
        data[0x40] = 0x00;
        assert!(!detect_pe_coff(&data));
    }

    #[test]
    fn detect_filter_returns_bcj_for_pe() {
        let data = make_minimal_pe();
        assert_eq!(detect_filter(&data), FILTER_BCJ);
    }

    // ── BCJ: encode/decode correctness ────────────────────────────────────────

    #[test]
    fn bcj_encode_call_correct() {
        // E8 at position 0x60, rel32 = 0x00000010
        // abs = 0x10 + (0x60 + 5) = 0x10 + 0x65 = 0x75
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x10;
        data[0x62] = 0x00;
        data[0x63] = 0x00;
        data[0x64] = 0x00;

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x61], enc[0x62], enc[0x63], enc[0x64]]);
        assert_eq!(abs, 0x75, "BCJ encode: abs should be 0x75, got 0x{:X}", abs);
    }

    #[test]
    fn bcj_encode_jmp_negative_rel_correct() {
        // E9 at position 0x65 (decimal 101), rel32 = -128.
        // abs = rel + (position + 5) = -128 + (101 + 5) = -128 + 106 = -22.
        //
        // IMPORTANT: use (-128i32).wrapping_add(...) NOT -128i32.wrapping_add(...)
        // In Rust, method calls bind tighter than unary minus, so without parens:
        //   -128i32.wrapping_add(x) = -(128i32.wrapping_add(x)) = -(128+106) = -234 ← WRONG
        //   (-128i32).wrapping_add(x) = (-128) + 106 = -22 ← CORRECT
        let mut data = vec![0u8; 128];
        data[0x65] = 0xE9;
        let rel: i32 = -128;
        data[0x66..0x6A].copy_from_slice(&rel.to_le_bytes());

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x66], enc[0x67], enc[0x68], enc[0x69]]);
        let expected = (-128i32).wrapping_add(0x65i32 + 5); // = -22
        assert_eq!(abs, expected,
            "BCJ encode JMP: abs should be {}, got {}", expected, abs);
    }

    #[test]
    fn bcj_roundtrip_call_and_jmp() {
        let mut data = vec![0u8; 256];
        data[0x10] = 0xE8;
        data[0x11] = 0x20; data[0x12] = 0x00; data[0x13] = 0x00; data[0x14] = 0x00;
        data[0x20] = 0xE9;
        let rel: i32 = -0x10;
        data[0x21..0x25].copy_from_slice(&rel.to_le_bytes());
        data[0x30] = 0x0F; data[0x31] = 0x84;
        data[0x32] = 0x50; data[0x33] = 0x00; data[0x34] = 0x00; data[0x35] = 0x00;

        let enc = bcj_x86_encode(&data);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "BCJ roundtrip failed");
    }

    #[test]
    fn bcj_roundtrip_false_positive_data_bytes() {
        let mut data = vec![0u8; 64];
        data[0]  = 0xE8;
        data[1]  = 0xAB; data[2]  = 0xCD; data[3]  = 0xEF; data[4]  = 0x01;
        data[10] = 0xE9;
        data[11] = 0x42; data[12] = 0x00; data[13] = 0xFF; data[14] = 0x7F;

        let enc = bcj_x86_encode(&data);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "BCJ false-positive roundtrip failed");
    }

    #[test]
    fn bcj_identity_on_no_calls() {
        // Data with no E8/E9/0F8x bytes should pass through unchanged
        let data: Vec<u8> = (0u8..=127).collect();
        assert!(!data.contains(&0xE8));
        assert!(!data.contains(&0xE9));
        let enc = bcj_x86_encode(&data);
        assert_eq!(enc, data, "BCJ should be identity when no branch opcodes present");
    }

    #[test]
    fn bcj_roundtrip_jcc_conditional_jump() {
        let mut data = vec![0u8; 64];
        data[5] = 0x0F;
        data[6] = 0x84;
        let rel: i32 = 100;
        data[7..11].copy_from_slice(&rel.to_le_bytes());

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[7], enc[8], enc[9], enc[10]]);
        let expected_abs = rel.wrapping_add(5 + 6);
        assert_eq!(abs, expected_abs);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "JCC roundtrip failed");
    }

    #[test]
    fn undo_filter_handles_old_flags_5_and_6() {
        let stl_data = make_stl(4);
        let ply_data = make_ply(20, 3);

        let enc_stl = apply_filter(&stl_data, FILTER_SHUFFLE4);
        let dec_stl = undo_filter(&enc_stl, FILTER_SHUFFLE4);
        assert_eq!(dec_stl, stl_data, "undo flag 5 (backward compat) failed");

        let enc_ply = apply_filter(&ply_data, FILTER_PLY);
        let dec_ply = undo_filter(&enc_ply, FILTER_PLY);
        assert_eq!(dec_ply, ply_data, "undo flag 6 (backward compat) failed");
    }
}
