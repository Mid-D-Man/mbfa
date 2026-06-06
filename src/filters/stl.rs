// src/filters/stl.rs
//! STL binary compound filter: byte-plane shuffle + per-plane delta1 (flag 7).
//!
//! Binary STL triangle data (12 floats per triangle: 3 normal + 9 vertex coords)
//! is reorganised into four byte-planes (one per IEEE 754 byte position) and
//! then delta1-encoded within each plane.  The exponent plane is near-constant
//! for typical geometry with bounded coordinate ranges, and mantissa planes
//! compress significantly better after delta1 removes inter-coordinate correlation.
//!
//! Attribute bytes (2 per triangle) are segregated after the float planes and
//! are not delta-encoded (they are typically zero or file-specific data).
//!
//! Detection uses the exact-size equation: file_len == 84 + n_tris * 50.
//! This is bijective for valid non-empty STL files.

// ── Detection ─────────────────────────────────────────────────────────────────

/// Returns `Some(())` if `data` matches the binary STL size equation for a
/// non-empty mesh.
pub fn detect_stl(data: &[u8]) -> Option<()> {
    if data.len() < 84 { return None; }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return None; }
    let expected = 84usize.checked_add(n_tris.checked_mul(50)?)?;
    if data.len() != expected { return None; }
    println!("Binary STL: {} triangle(s)", n_tris);
    Some(())
}

// ── Simple shuffle (internal) ─────────────────────────────────────────────────

fn shuffle4_stl_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let float_count = n_tris * 12; // 12 floats per triangle (normal + 3 vertices)
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

// ── Compound filter: shuffle + per-plane delta1 (flag 7) ──────────────────────

/// Apply STL compound filter: byte-plane shuffle then per-plane delta1 encoding.
pub fn shuffle4_stl_delta_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let mut out      = shuffle4_stl_encode(data);
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

/// Reverse the STL compound filter.
pub fn shuffle4_stl_delta_decode(data: &[u8]) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{apply_filter, detect_filter, undo_filter, FILTER_SHUFFLE4_DELTA};
    use crate::filters::probe::byte_entropy;

    fn make_stl(n_tris
