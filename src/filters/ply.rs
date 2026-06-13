// src/filters/ply.rs
//! PLY binary compound filter: byte-plane shuffle + per-vertex-stride delta (flag 8).
//!
//! Binary PLY vertex floats are reorganised into four byte-planes (one per
//! IEEE 754 byte position: LSB → MSB) and then delta-encoded within each
//! plane using **stride = floats_per_vertex**.
//!
//! ## Why stride = floats_per_vertex rather than stride-1?
//!
//! After the byte-plane shuffle, plane b contains all byte_b values in
//! vertex-major, float-minor order:
//!   plane_b[k * fpv + j] = byte_b(float j of vertex k)
//!
//! **stride-1 (old behaviour):**
//!   plane_b[i] -= plane_b[i - 1]
//!   = byte_b(float j of vertex k) - byte_b(float j-1 of vertex k)
//! This crosses float-field boundaries — subtracting, say, byte-0 of the
//! u-texture-coordinate from byte-0 of the nz-normal component.  For
//! unrelated IEEE-754 fields the difference is essentially random. ✗
//!
//! **stride-fpv (new behaviour):**
//!   plane_b[k*fpv + j] -= plane_b[(k-1)*fpv + j]
//!   = byte_b(float j of vertex k) - byte_b(float j of vertex k-1)
//! Only the SAME semantic field is differenced across consecutive vertices.
//! For smooth geometry, neighbouring vertices have very similar values in
//! each float field → small deltas → much higher LZ match rate. ✓
//!
//! ## Grid PLY periodicity
//!
//! For a heightmap grid (grid_w × grid_h vertices in row-major order), float
//! fields that depend only on the column index (x-coordinate, u-texture) have
//! *identical* delta sequences in every row.  This creates a strong LZ period
//! of exactly `grid_w × fpv` bytes inside each plane — well within MBFA's
//! Phase C window selection.  Fields that vary with the row (y, z, v, normals)
//! still produce bounded, slowly-varying deltas that compress significantly
//! better than the cross-boundary values produced by stride-1.
//!
//! Detection requires:
//!   - "ply\n" magic
//!   - "format binary_little_endian" in header
//!   - "element vertex N" with N > 0
//!   - All vertex properties are "property float" (no uchar/int mixing)
//!   - Sufficient data for the vertex block

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() { return None; }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ── Layout parsing ────────────────────────────────────────────────────────────

pub struct PlyLayout {
    pub header_end:        usize,
    pub vertex_count:      usize,
    pub floats_per_vertex: usize,
}

/// Parse the PLY header and return layout info, or None if not a supported
/// binary-float-only PLY.
pub fn parse_ply_layout(data: &[u8]) -> Option<PlyLayout> {
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
            vertex_count = line["element vertex ".len()..].trim().parse().ok()?;
            in_vertex    = true;
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
        vertex_count, floats_per_vertex, vertex_bytes
    );
    Some(PlyLayout { header_end, vertex_count, floats_per_vertex })
}

// ── Simple shuffle (internal) ─────────────────────────────────────────────────

fn shuffle4_ply_encode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };
    let float_count  = layout.vertex_count * layout.floats_per_vertex;
    let vertex_bytes = float_count * 4;
    let vs           = layout.header_end;
    let ve           = vs + vertex_bytes;
    if ve > data.len() { return data.to_vec(); }

    let mut plane0 = Vec::with_capacity(float_count);
    let mut plane1 = Vec::with_capacity(float_count);
    let mut plane2 = Vec::with_capacity(float_count);
    let mut plane3 = Vec::with_capacity(float_count);

    for chunk in data[vs..ve].chunks_exact(4) {
        plane0.push(chunk[0]);
        plane1.push(chunk[1]);
        plane2.push(chunk[2]);
        plane3.push(chunk[3]);
    }

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..vs]);
    out.extend_from_slice(&plane0);
    out.extend_from_slice(&plane1);
    out.extend_from_slice(&plane2);
    out.extend_from_slice(&plane3);
    out.extend_from_slice(&data[ve..]);
    out
}

fn shuffle4_ply_decode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };
    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let vs          = layout.header_end;
    let planes_end  = vs + 4 * plane_size;

    if planes_end > data.len() {
        eprintln!(
            "shuffle4_ply_decode: data too short — have {} bytes, need {}",
            data.len(), planes_end
        );
        return data.to_vec();
    }

    let plane0 = &data[vs              ..vs +     plane_size];
    let plane1 = &data[vs +   plane_size..vs + 2 * plane_size];
    let plane2 = &data[vs + 2*plane_size..vs + 3 * plane_size];
    let plane3 = &data[vs + 3*plane_size..vs + 4 * plane_size];

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..vs]);
    for i in 0..float_count {
        out.push(plane0[i]);
        out.push(plane1[i]);
        out.push(plane2[i]);
        out.push(plane3[i]);
    }
    out.extend_from_slice(&data[planes_end..]);
    out
}

// ── Compound filter: shuffle + per-vertex-stride delta (flag 8) ───────────────

/// Apply PLY compound filter: byte-plane shuffle then per-vertex-stride
/// delta encoding.
///
/// The stride equals `floats_per_vertex` (fpv) so that position
/// `k*fpv + j` in each plane is differenced only against position
/// `(k-1)*fpv + j` — the same float field in the preceding vertex.
///
/// The first `fpv` bytes in each plane section are unchanged (seed values
/// for the first vertex).
///
/// Encoding traverses HIGH → LOW so we always subtract the original
/// (not-yet-modified) value at `[i - stride]`.
pub fn shuffle4_ply_delta_encode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };
    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let ve          = layout.header_end + 4 * plane_size;
    if ve > data.len() { return data.to_vec(); }

    let mut out    = shuffle4_ply_encode(data);
    let stride     = layout.floats_per_vertex; // same float field across vertices
    let vs         = layout.header_end;

    for plane_idx in 0..4usize {
        let ps = vs + plane_idx * plane_size;
        let pe = ps + plane_size;
        if pe > out.len() { break; }
        // HIGH → LOW: out[i - stride] is still the original shuffled byte.
        for i in (ps + stride..pe).rev() {
            let prev = out[i - stride];
            out[i] = out[i].wrapping_sub(prev);
        }
    }
    out
}

/// Reverse the PLY compound filter.
///
/// Decoding traverses LOW → HIGH: at position i, `undelta[i - stride]`
/// has already been restored to its original value and can be added back.
pub fn shuffle4_ply_delta_decode(data: &[u8]) -> Vec<u8> {
    let layout = match parse_ply_layout(data) {
        Some(l) => l,
        None    => return data.to_vec(),
    };
    let float_count = layout.vertex_count * layout.floats_per_vertex;
    let plane_size  = float_count;
    let vs          = layout.header_end;
    let planes_end  = vs + 4 * plane_size;

    if planes_end > data.len() {
        eprintln!(
            "shuffle4_ply_delta_decode: data too short — have {} bytes, need {}",
            data.len(), planes_end
        );
        return data.to_vec();
    }

    let stride       = layout.floats_per_vertex;
    let mut undelta  = data.to_vec();

    for plane_idx in 0..4usize {
        let ps = vs + plane_idx * plane_size;
        let pe = ps + plane_size;
        // LOW → HIGH: undelta[i - stride] is already the decoded original.
        for i in ps + stride..pe {
            let prev = undelta[i - stride];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }
    shuffle4_ply_decode(&undelta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{apply_filter, detect_filter, undo_filter, FILTER_PLY_DELTA};

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

    /// Single-vertex edge case: no delta applied (plane size = fpv, range is empty).
    #[test]
    fn roundtrip_ply_compound_single_vertex() {
        let data = make_ply(1, 4);
        let enc = apply_filter(&data, FILTER_PLY_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_PLY_DELTA);
        assert_eq!(dec, data, "PLY compound roundtrip failed for 1 vertex");
    }

    /// Two-vertex case: exactly one delta applied per float position per plane.
    #[test]
    fn roundtrip_ply_compound_two_vertices() {
        let data = make_ply(2, 6);
        let enc = apply_filter(&data, FILTER_PLY_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_PLY_DELTA);
        assert_eq!(dec, data, "PLY compound roundtrip failed for 2 vertices");
    }

    #[test]
    fn detect_ply_returns_compound_flag() {
        let data = make_ply(100, 8);
        assert_eq!(
            detect_filter(&data), FILTER_PLY_DELTA,
            "detect_filter should return FILTER_PLY_DELTA (8) for binary PLY"
        );
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

    /// Verify that stride-fpv delta reduces entropy in the exponent plane
    /// compared to the raw shuffled plane, for a ramp of float values.
    #[test]
    fn stride_fpv_delta_reduces_exponent_plane_entropy() {
        use crate::filters::probe::byte_entropy;

        let n_verts = 200usize;
        let fpv     = 8usize;
        let data    = make_ply(n_verts, fpv);

        let compound = apply_filter(&data, FILTER_PLY_DELTA);
        let dec      = undo_filter(&compound, FILTER_PLY_DELTA);
        assert_eq!(dec, data, "PLY compound roundtrip failed for entropy test");

        // Locate plane 3 (MSByte / exponent) in the shuffled output.
        let layout = parse_ply_layout(&data).expect("make_ply produces valid PLY");
        let plane_size = layout.vertex_count * layout.floats_per_vertex;
        let p3s        = layout.header_end + 3 * plane_size;
        let p3e        = p3s + plane_size;

        // Build the simple-shuffle (no delta) for comparison.
        let simple     = shuffle4_ply_encode(&data);
        let simple_ent   = byte_entropy(&simple[p3s..p3e]);
        let compound_ent = byte_entropy(&compound[p3s..p3e]);

        assert!(
            compound_ent <= simple_ent,
            "plane3 entropy after stride-fpv delta ({:.4}) should be ≤ \
             simple-shuffle entropy ({:.4})",
            compound_ent, simple_ent,
        );
    }
            }
