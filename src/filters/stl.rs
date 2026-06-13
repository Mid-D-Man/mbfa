// src/filters/stl.rs
//! STL binary compound filter: byte-plane shuffle + per-triangle-stride delta (flag 7).
//!
//! Binary STL triangle data (12 floats per triangle: 3 normal + 9 vertex coords)
//! is reorganised into four byte-planes (one per IEEE 754 byte position) and
//! then delta-encoded within each plane using stride = FLOATS_PER_TRIANGLE (12).
//!
//! ## Why stride = 12 rather than stride-1?
//!
//! After the byte-plane shuffle, plane b contains:
//!   plane_b[k * 12 + j] = byte_b(float j of triangle k)
//!
//! **stride-1 (old):**
//!   plane_b[k*12 + j] -= plane_b[k*12 + j - 1]
//!   = byte_b(float j of tri k) - byte_b(float j-1 of tri k)
//! This crosses float-field boundaries (e.g., subtracts a vertex-y byte
//! from a normal-z byte).  For unrelated IEEE-754 fields the difference
//! is essentially random → poor LZ compressibility.
//!
//! **stride-12 (new):**
//!   plane_b[k*12 + j] -= plane_b[(k-1)*12 + j]
//!   = byte_b(float j of tri k) - byte_b(float j of tri k-1)
//! Only the same semantic field is differenced across consecutive triangles.
//! Adjacent triangles in a mesh share edges and have similar normals and
//! vertex positions, so these deltas cluster near zero.  For a lat/lon
//! sphere mesh the same-field deltas repeat with period ≈ 2*n_lon triangles,
//! giving LZ strong periodic structure inside each plane. ✓
//!
//! Attribute bytes (2 per triangle) are segregated after the float planes and
//! are not delta-encoded (they are typically zero or file-specific data).
//!
//! Detection uses the exact-size equation: file_len == 84 + n_tris * 50.
//! This is bijective for valid non-empty STL files.

/// Number of floats per STL triangle (3 normal + 3 × 3 vertex coords).
const FLOATS_PER_TRIANGLE: usize = 12;

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

    let float_count = n_tris * FLOATS_PER_TRIANGLE;
    let mut plane0  = Vec::with_capacity(float_count);
    let mut plane1  = Vec::with_capacity(float_count);
    let mut plane2  = Vec::with_capacity(float_count);
    let mut plane3  = Vec::with_capacity(float_count);
    let mut attrs   = Vec::with_capacity(n_tris * 2);

    for tri in 0..n_tris {
        let base = 84 + tri * 50;
        for f in 0..FLOATS_PER_TRIANGLE {
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

    let plane_size   = n_tris * FLOATS_PER_TRIANGLE;
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
        for f in 0..FLOATS_PER_TRIANGLE {
            let idx = tri * FLOATS_PER_TRIANGLE + f;
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

// ── Compound filter: shuffle + per-triangle-stride delta (flag 7) ─────────────

/// Apply STL compound filter: byte-plane shuffle then per-triangle-stride
/// delta encoding (stride = FLOATS_PER_TRIANGLE = 12).
///
/// The first 12 bytes in each plane section are unchanged (seed values for
/// the first triangle).  All subsequent bytes are the difference between
/// the same float field at consecutive triangles.
///
/// Encoding traverses HIGH → LOW so we always read the original
/// (not-yet-modified) value at `[i - stride]`.
pub fn shuffle4_stl_delta_encode(data: &[u8]) -> Vec<u8> {
    debug_assert!(data.len() >= 84);
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let mut out      = shuffle4_stl_encode(data);
    let plane_size   = n_tris * FLOATS_PER_TRIANGLE;
    let planes_start = 84usize;

    for plane_idx in 0..4usize {
        let ps = planes_start + plane_idx * plane_size;
        let pe = ps + plane_size;
        if pe > out.len() { break; }
        // Traverse HIGH → LOW: at each position i, out[i - FLOATS_PER_TRIANGLE]
        // has not been touched yet, so we read the original shuffled byte.
        for i in (ps + FLOATS_PER_TRIANGLE..pe).rev() {
            let prev = out[i - FLOATS_PER_TRIANGLE];
            out[i] = out[i].wrapping_sub(prev);
        }
    }
    out
}

/// Reverse the STL compound filter.
///
/// Decoding traverses LOW → HIGH: at each position i, the value at
/// `[i - stride]` has already been restored to its original, so adding
/// it back correctly inverts the subtraction.
pub fn shuffle4_stl_delta_decode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let plane_size   = n_tris * FLOATS_PER_TRIANGLE;
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
        // Traverse LOW → HIGH: at position i, undelta[i - FLOATS_PER_TRIANGLE]
        // has already been decoded back to its original value.
        for i in ps + FLOATS_PER_TRIANGLE..pe {
            let prev = undelta[i - FLOATS_PER_TRIANGLE];
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
        assert_eq!(
            detect_filter(&data), FILTER_SHUFFLE4_DELTA,
            "detect_filter should return FILTER_SHUFFLE4_DELTA (7) for STL"
        );
    }

    #[test]
    fn detect_stl_rejects_wrong_size() {
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50 + 1]; // one byte too many
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4_DELTA);
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let data = vec![0u8; 84];
        assert_eq!(detect_filter(&data), crate::filters::FILTER_NONE);
    }

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

    /// Single-triangle edge case: no delta is applied (plane sections have
    /// size = FLOATS_PER_TRIANGLE, so the range `ps+12..pe` is empty).
    #[test]
    fn roundtrip_shuffle4_stl_delta_single_tri() {
        let data = make_stl(1);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "STL compound roundtrip failed for 1 triangle");
    }

    #[test]
    fn stl_delta_reduces_entropy_in_planes() {
        let n_tris       = 200usize;
        let total_floats = n_tris * FLOATS_PER_TRIANGLE;

        let mut data = vec![0u8; 84 + n_tris * 50];
        data[80..84].copy_from_slice(&(n_tris as u32).to_le_bytes());

        // Fill with a smoothly-increasing float ramp so stride-12 delta
        // produces mostly-zero exponent differences.
        for tri in 0..n_tris {
            let base = 84 + tri * 50;
            for f in 0..FLOATS_PER_TRIANGLE {
                let idx = tri * FLOATS_PER_TRIANGLE + f;
                let v: f32 = 1.0 + (idx as f32 / total_floats as f32) * 2.0;
                let bytes = v.to_le_bytes();
                data[base + f * 4..base + f * 4 + 4].copy_from_slice(&bytes);
            }
        }

        let compound = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        let dec      = undo_filter(&compound, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "compound STL roundtrip failed for ramp geometry");

        // Compare entropy in plane3 (exponent byte) vs simple shuffle.
        // With stride-12 delta the exponent byte only changes when the float
        // crosses a power-of-two boundary (~12 positions out of 2400),
        // so entropy should be far lower than the undelted simple shuffle.
        let simple     = shuffle4_stl_encode(&data);
        let plane_size = n_tris * FLOATS_PER_TRIANGLE;
        let p3s        = 84 + 3 * plane_size;

        let simple_ent   = byte_entropy(&simple[p3s..p3s + plane_size]);
        let compound_ent = byte_entropy(&compound[p3s..p3s + plane_size]);

        assert!(
            compound_ent <= simple_ent,
            "compound plane3 entropy ({:.4}) should be ≤ simple ({:.4}) for ramp geometry",
            compound_ent, simple_ent
        );
    }

    /// Verify the stride-12 delta is a strict improvement over stride-1
    /// for a ramp — i.e., the new approach doesn't regress on the entropy test.
    #[test]
    fn stride12_delta_at_least_as_good_as_stride1_for_ramp() {
        let n_tris       = 200usize;
        let total_floats = n_tris * FLOATS_PER_TRIANGLE;

        let mut data = vec![0u8; 84 + n_tris * 50];
        data[80..84].copy_from_slice(&(n_tris as u32).to_le_bytes());

        for tri in 0..n_tris {
            let base = 84 + tri * 50;
            for f in 0..FLOATS_PER_TRIANGLE {
                let idx = tri * FLOATS_PER_TRIANGLE + f;
                let v: f32 = 1.0 + (idx as f32 / total_floats as f32) * 2.0;
                data[base + f * 4..base + f * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }

        // Build stride-1 version manually for comparison.
        let shuffled = shuffle4_stl_encode(&data);
        let plane_size   = n_tris * FLOATS_PER_TRIANGLE;
        let planes_start = 84usize;

        let mut stride1 = shuffled.clone();
        for plane_idx in 0..4usize {
            let ps = planes_start + plane_idx * plane_size;
            let pe = ps + plane_size;
            for i in (ps + 1..pe).rev() {
                let prev = stride1[i - 1];
                stride1[i] = stride1[i].wrapping_sub(prev);
            }
        }

        // stride-12 compound (our new filter).
        let stride12 = apply_filter(&data, FILTER_SHUFFLE4_DELTA);

        // For each plane, stride-12 entropy ≤ stride-1 entropy on a ramp.
        for plane_idx in 0..4usize {
            let ps = planes_start + plane_idx * plane_size;
            let pe = ps + plane_size;
            let e1  = byte_entropy(&stride1[ps..pe]);
            let e12 = byte_entropy(&stride12[ps..pe]);
            assert!(
                e12 <= e1 + 0.01, // tiny float tolerance
                "plane {}: stride-12 entropy ({:.4}) should be ≤ stride-1 ({:.4})",
                plane_idx, e12, e1,
            );
        }
    }
        }
