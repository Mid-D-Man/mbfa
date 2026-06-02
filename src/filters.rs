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
//!   7 = STL byte-plane shuffle + per-plane delta1
//!       Extracts float bytes into 4 planes (LSB→MSB), then delta1-encodes
//!       each plane independently. Exponent plane near-constant for smooth
//!       geometry; mantissa planes compress better with delta before LZ.
//!   8 = PLY byte-plane shuffle + per-plane delta1
//!       Same compound transform for binary PLY vertex floats.
//!   9 = x86 BCJ (Branch-Call-Jump) xz-style for PE/COFF executables
//!       Converts CALL (E8) and JMP (E9) relative offsets to absolute
//!       addresses using the xz BCJ algorithm:
//!         - MSByte gate: only process when the operand high byte is 0x00 or
//!           0xFF (near ±16 MB call target). Skips data bytes that happen to
//!           be E8/E9 and have non-near operands.
//!         - prev_mask: 3-bit rolling mask detects false positives arising
//!           from E8/E9 bytes embedded in a previous instruction's operand.
//!         - 25-bit normalisation: sign-extends bit 24 into bits 25-31 so all
//!           near-forward targets cluster at 0x00xxxxxx and near-backward
//!           targets at 0xFFxxxxxx regardless of call-site address. This
//!           maximises LZ match frequency across object files loaded at
//!           different base addresses.
//!       Fully reversible bijection.
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
// flags 5 and 6 were the old simple (non-compound) STL/PLY shuffles — removed.
/// STL: byte-plane shuffle + per-plane delta1.
pub const FILTER_SHUFFLE4_DELTA: u8 = 7;
/// PLY: byte-plane shuffle + per-plane delta1.
pub const FILTER_PLY_DELTA:      u8 = 8;
/// x86 BCJ normalization for PE/COFF binaries (xz-style).
pub const FILTER_BCJ:            u8 = 9;

/// Minimum file size before the entropy stride probe runs.
const PROBE_MIN_BYTES: usize = 512;

/// Entropy improvement threshold (bits/byte) for the stride probe.
const PROBE_DELTA_THRESHOLD: f64 = 0.45;

// ── Public API ────────────────────────────────────────────────────────────────

/// Inspect magic bytes and file structure to determine the best filter.
pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL — exact size equation (no magic bytes).
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

    // 4. Binary PLY
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

// ── STL detection ─────────────────────────────────────────────────────────────

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
    header_end:        usize,
    vertex_count:      usize,
    floats_per_vertex: usize,
}

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

// ── PLY simple shuffle (internal helper, used by delta compound) ─────────────

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

fn shuffle4_ply_delta_encode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };
    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let vertex_end  = layout.header_end + 4 * plane_size;
    if vertex_end > data.len() { return data.to_vec(); }

    let mut out = shuffle4_ply_encode(data);
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

    let mut undelta = data.to_vec();
    for plane_idx in 0..4usize {
        let ps = layout.header_end + plane_idx * plane_size;
        let pe = ps + plane_size;
        for i in ps + 1..pe {
            let prev = undelta[i - 1];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }
    shuffle4_ply_decode(&undelta)
}

// ── STL simple shuffle (internal helper, used by delta compound) ──────────────

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

fn shuffle4_stl_delta_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let mut out = shuffle4_stl_encode(data);
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

    let mut undelta = data.to_vec();
    for plane_idx in 0..4usize {
        let ps = planes_start + plane_idx * plane_size;
        let pe = ps + plane_size;
        for i in ps + 1..pe {
            let prev = undelta[i - 1];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }
    shuffle4_stl_decode(&undelta)
}

// ── PE/COFF detection ─────────────────────────────────────────────────────────

fn detect_pe_coff(data: &[u8]) -> bool {
    if data.len() < 0x40 { return false; }
    if data[0] != b'M' || data[1] != b'Z' { return false; }
    let pe_offset = u32::from_le_bytes([data[0x3C], data[0x3D], data[0x3E], data[0x3F]]) as usize;
    if pe_offset.saturating_add(4) > data.len() { return false; }
    data[pe_offset..pe_offset + 4] == *b"PE\x00\x00"
}

// ── BCJ x86 encode / decode (xz-style) ───────────────────────────────────────
//
// Algorithm derived from the XZ embedded BCJ filter.
//
// MSByte gate: only process E8/E9 instructions whose 32-bit LE operand
// has a high byte of 0x00 (near forward call) or 0xFF (near backward call).
// This limits transformation to ±16 MB call range and avoids transforming
// data bytes that happen to be E8/E9.
//
// prev_mask: 3-bit rolling state that detects false positives arising from
// an E8/E9 byte embedded within the operand of a previous call instruction.
//
// 25-bit normalisation (encode only, self-cancelling in decode):
//   norm = abs & 0x01ff_ffff;
//   norm |= 0u32.wrapping_sub(norm & 0x0100_0000);
// Sign-extends bit 24 into bits 25-31 so all near-forward targets map to
// 0x00xxxxxx and all near-backward targets map to 0xFFxxxxxx regardless
// of the call-site virtual address. Maximises LZ match frequency.
// The decode wrapping subtraction exactly reverses this (all modular arithmetic).
//
// JCC (0F 8x) is not handled — xz BCJ does not handle it, and the MSByte
// gate provides sufficient benefit without the added complexity.

#[inline(always)]
fn bcj_x86_test_msbyte(b: u8) -> bool {
    b == 0 || b == 0xff
}

fn bcj_x86_encode(data: &[u8]) -> Vec<u8> {
    static MASK_TO_ALLOWED: [bool; 8]  = [true, true, true, false, true, false, false, false];
    static MASK_TO_BIT:     [usize; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        // Only E8 (CALL) and E9 (JMP near) are candidates.
        if data[i] & 0xFE != 0xE8 {
            i += 1;
            continue;
        }

        // ── prev_mask false-positive gate ──────────────────────────────────
        // Detects when the current E8/E9 byte is actually part of the operand
        // of a previous call instruction, not a real opcode.
        let dist = i.wrapping_sub(prev_pos);
        if dist <= 3 {
            prev_mask = (prev_mask << dist.wrapping_sub(1)) & 7;
            if prev_mask != 0 {
                let b = data[i + 4 - MASK_TO_BIT[prev_mask]];
                if !MASK_TO_ALLOWED[prev_mask] || bcj_x86_test_msbyte(b) {
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

        // ── MSByte gate ────────────────────────────────────────────────────
        // data[i+4] is the high byte of the 32-bit LE relative operand.
        // 0x00 = small positive (near forward call).
        // 0xFF = large negative (near backward call, i.e. near -1...-16M).
        if bcj_x86_test_msbyte(data[i + 4]) {
            let rel = i32::from_le_bytes([data[i+1], data[i+2], data[i+3], data[i+4]]);
            let abs = rel.wrapping_add(i as i32 + 5) as u32;

            // 25-bit normalisation: clusters all near-forward targets at
            // 0x00xxxxxx and near-backward targets at 0xFFxxxxxx.
            let mut norm  = abs & 0x01ff_ffff;
            norm         |= 0u32.wrapping_sub(norm & 0x0100_0000);

            out[i+1..i+5].copy_from_slice(&norm.to_le_bytes());
            i += 4; // +1 below → total +5
        } else {
            prev_mask = (prev_mask << 1) | 1;
        }

        i += 1;
    }

    out
}

fn bcj_x86_decode(data: &[u8]) -> Vec<u8> {
    static MASK_TO_ALLOWED: [bool; 8]  = [true, true, true, false, true, false, false, false];
    static MASK_TO_BIT:     [usize; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size = n - 4;
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
                if !MASK_TO_ALLOWED[prev_mask] || bcj_x86_test_msbyte(b) {
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

        // The stored value is the 25-bit-normalised absolute address.
        // By construction its high byte is 0x00 or 0xFF.
        // Wrapping subtraction exactly reverses the wrapping addition in encode.
        if bcj_x86_test_msbyte(data[i + 4]) {
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
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let data = vec![0u8; 84];
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    // ── STL compound (flag 7) ─────────────────────────────────────────────────

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
    fn stl_delta_reduces_entropy_in_planes() {
        let n_tris       = 200usize;
        let total_floats = n_tris * 12;

        let mut data = vec![0u8; 84 + n_tris * 50];
        data[80..84].copy_from_slice(&(n_tris as u32).to_le_bytes());

        for tri in 0..n_tris {
            let base = 84 + tri * 50;
            for f in 0..12usize {
                let idx = tri * 12 + f;
                let v: f32 = 1.0 + (idx as f32 / total_floats as f32) * 2.0;
                let bytes = v.to_le_bytes();
                data[base + f * 4..base + f * 4 + 4].copy_from_slice(&bytes);
            }
        }

        let compound = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        let dec = undo_filter(&compound, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "compound STL roundtrip failed for ramp geometry");

        // Simple shuffle (internal) for entropy comparison
        let simple = shuffle4_stl_encode(&data);
        let plane_size   = n_tris * 12;
        let plane3_start = 84 + 3 * plane_size;

        let simple_ent   = byte_entropy(&simple[plane3_start..plane3_start + plane_size]);
        let compound_ent = byte_entropy(&compound[plane3_start..plane3_start + plane_size]);

        assert!(
            compound_ent <= simple_ent,
            "compound plane3 entropy ({:.4}) should be <= simple ({:.4}) \
             for ramp geometry",
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

    // ── PLY compound (flag 8) ─────────────────────────────────────────────────

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

    // ── BCJ: PE detection ─────────────────────────────────────────────────────

    fn make_minimal_pe() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0] = b'M';
        data[1] = b'Z';
        data[0x3C] = 0x40;
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
    fn bcj_encode_call_near_correct() {
        // E8 at position 0x60, rel32 = 0x00000010 (MSByte 0x00 → near forward)
        // abs = 0x10 + (0x60 + 5) = 0x75. Normalisation: identity (bit24=0).
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x10;
        data[0x62] = 0x00;
        data[0x63] = 0x00;
        data[0x64] = 0x00; // MSByte = 0x00 → processed

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x61], enc[0x62], enc[0x63], enc[0x64]]);
        assert_eq!(abs, 0x75, "BCJ encode: abs should be 0x75, got 0x{:X}", abs);
    }

    #[test]
    fn bcj_far_call_not_transformed() {
        // E8 at position 0x60, rel32 = 0x30000000 (MSByte 0x30 → far, skip)
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x00;
        data[0x62] = 0x00;
        data[0x63] = 0x00;
        data[0x64] = 0x30; // MSByte = 0x30 → NOT near, must be skipped

        let enc = bcj_x86_encode(&data);
        // Should be unchanged
        assert_eq!(&enc[0x60..0x65], &data[0x60..0x65],
            "far call should not be transformed");
    }

    #[test]
    fn bcj_encode_jmp_near_backward_correct() {
        // E9 at position 0x65 (101), rel32 = -128 (MSByte 0xFF → near backward)
        let mut data = vec![0u8; 128];
        data[0x65] = 0xE9;
        let rel: i32 = -128;
        data[0x66..0x6A].copy_from_slice(&rel.to_le_bytes());
        // data[0x69] = 0xFF (MSByte of -128 in LE)

        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x66], enc[0x67], enc[0x68], enc[0x69]]);
        let expected = (-128i32).wrapping_add(0x65i32 + 5); // = -22
        assert_eq!(abs, expected,
            "BCJ encode JMP: abs should be {}, got {}", expected, abs);
    }

    #[test]
    fn bcj_roundtrip_near_calls() {
        let mut data = vec![0u8; 256];
        // CALL at 0x10 with near forward rel
        data[0x10] = 0xE8;
        data[0x11] = 0x20; data[0x12] = 0x00; data[0x13] = 0x00; data[0x14] = 0x00;
        // JMP at 0x20 with near backward rel
        data[0x20] = 0xE9;
        let rel: i32 = -0x10;
        data[0x21..0x25].copy_from_slice(&rel.to_le_bytes());
        // Non-near E8 (should be skipped by MSByte gate)
        data[0x40] = 0xE8;
        data[0x41] = 0x00; data[0x42] = 0x00; data[0x43] = 0x00; data[0x44] = 0x30;

        let enc = bcj_x86_encode(&data);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "BCJ roundtrip failed");
    }

    #[test]
    fn bcj_roundtrip_false_positive_data_bytes() {
        // These bytes have E8/E9 opcodes but non-near operands → MSByte gate skips them
        let mut data = vec![0u8; 64];
        data[0]  = 0xE8;
        data[1]  = 0xAB; data[2]  = 0xCD; data[3]  = 0xEF; data[4]  = 0x01; // MSByte 0x01 → skip
        data[10] = 0xE9;
        data[11] = 0x42; data[12] = 0x00; data[13] = 0xFF; data[14] = 0x7F; // MSByte 0x7F → skip

        let enc = bcj_x86_encode(&data);
        // Both should be skipped — enc equals data
        assert_eq!(enc, data, "non-near bytes should be unchanged by MSByte gate");
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data, "BCJ false-positive roundtrip failed");
    }

    #[test]
    fn bcj_identity_on_no_e8_e9() {
        let data: Vec<u8> = (0u8..=127).collect();
        assert!(!data.contains(&0xE8));
        assert!(!data.contains(&0xE9));
        let enc = bcj_x86_encode(&data);
        assert_eq!(enc, data, "BCJ should be identity when no CALL/JMP opcodes present");
    }

    #[test]
    fn bcj_25bit_norm_clusters_values() {
        // After 25-bit normalisation, all near-forward targets should have
        // high byte 0x00 and near-backward targets should have high byte 0xFF.
        let mut data = vec![0u8; 16];
        // Near forward: rel=0x100, abs = 0x100 + 0 + 5 = 0x105. Bit24=0. norm=0x105. High byte 0x00.
        data[0] = 0xE8;
        data[1] = 0x00; data[2] = 0x01; data[3] = 0x00; data[4] = 0x00; // MSByte 0x00

        let enc = bcj_x86_encode(&data);
        assert_eq!(enc[4], 0x00, "near-forward should have high byte 0x00 after normalisation");

        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec[1..5], data[1..5], "normalisation should be reversed by decode");
    }
        }
