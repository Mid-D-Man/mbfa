// src/filters/stl.rs
//! STL binary compound filters (flag 7 and flag 10).
//!
//! Flag 7 — shuffle4 + stride-12 (legacy decode-only):
//!   Byte-plane split (tri-major order) then stride-12 delta.
//!   Kept for backward compatibility with old archives.
//!
//! Flag 10 — field-major + stride-1 (current default for new compressions):
//!   Byte-plane split in (plane, field, tri) order — all n_tris values for the
//!   same byte-position AND float-field are contiguous. Then stride-1 delta
//!   within each (plane, field) block.
//!
//!   Why this beats flag 7 for sphere/grid geometry:
//!   y-component fields (normal.y, v0.y, v1.y, v2.y) equal cos(phi) which is
//!   CONSTANT within a latitude row. After stride-1 delta, every latitude row
//!   produces ~n_lon consecutive zero bytes — giving LZ 1700-byte exact-match
//!   runs vs flag 7's max 175-byte runs on the 5000-triangle benchmark sphere.
//!   RLE pairs: flag10=10249 vs flag7=12266 — 16% more compressible.

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

// ── Simple shuffle (internal helper for flag 7) ───────────────────────────────

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

// ── Flag 7: Plane-shuffle + stride-12 delta (LEGACY — decode only) ────────────

/// Apply STL flag-7 compound filter (kept for decoding old archives only).
/// New compressions use flag 10 (`field_major_stl_delta_encode`).
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
        for i in (ps + FLOATS_PER_TRIANGLE..pe).rev() {
            let prev = out[i - FLOATS_PER_TRIANGLE];
            out[i] = out[i].wrapping_sub(prev);
        }
    }
    out
}

/// Reverse STL flag-7 compound filter.
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
        for i in ps + FLOATS_PER_TRIANGLE..pe {
            let prev = undelta[i - FLOATS_PER_TRIANGLE];
            undelta[i] = undelta[i].wrapping_add(prev);
        }
    }
    shuffle4_stl_decode(&undelta)
}

// ── Flag 10: Field-major + stride-1 delta (CURRENT DEFAULT) ──────────────────
//
// Output layout (size-preserving: 84 + 50*n_tris bytes):
//   bytes 0..84                       : header (unchanged)
//   bytes 84..(84 + n_tris)           : plane0/field0 block  ← byte_0 of all normal.x
//   bytes (84+n_tris)..(84+2*n_tris)  : plane0/field1 block  ← byte_0 of all normal.y
//   ...
//   bytes (84+11*n_tris)..(84+12*n_tris) : plane0/field11 block
//   bytes (84+12*n_tris)..             : plane1/field0 block  ← byte_1 of all normal.x
//   ...
//   bytes (84+47*n_tris)..(84+48*n_tris) : plane3/field11 block
//   bytes (84+48*n_tris)..(84+50*n_tris) : attribute bytes (2 per triangle, raw)
//
// Block index = byte_pos * FLOATS_PER_TRIANGLE + field  (0..48)
// Stride-1 delta: HIGH→LOW encode, LOW→HIGH decode.

/// Apply STL flag-10 filter: field-major byte-plane layout + stride-1 delta.
/// Produces 10x longer LZ-matchable zero runs than flag 7 on smooth geometry.
pub fn field_major_stl_delta_encode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let expected = 84 + n_tris * 50;
    if data.len() < expected { return data.to_vec(); }

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..84]);

    // 4 byte-planes × 12 float-fields: write each (plane, field) block
    for byte_pos in 0..4usize {
        for field in 0..FLOATS_PER_TRIANGLE {
            // Collect byte `byte_pos` of float `field` across all n_tris triangles
            let mut block: Vec<u8> = (0..n_tris)
                .map(|tri| data[84 + tri * 50 + field * 4 + byte_pos])
                .collect();
            // Stride-1 delta HIGH→LOW: block[i] -= block[i-1], block[i-1] still original
            for i in (1..n_tris).rev() {
                block[i] = block[i].wrapping_sub(block[i - 1]);
            }
            out.extend_from_slice(&block);
        }
    }

    // Attribute bytes: 2 per triangle, raw (no delta)
    for tri in 0..n_tris {
        let base = 84 + tri * 50;
        out.push(data[base + 48]);
        out.push(data[base + 49]);
    }

    out
}

/// Reverse STL flag-10 filter.
pub fn field_major_stl_delta_decode(data: &[u8]) -> Vec<u8> {
    if data.len() < 84 { return data.to_vec(); }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if n_tris == 0 { return data.to_vec(); }

    let blocks_start = 84usize;
    let attrs_start  = blocks_start + 48 * n_tris;  // 4 planes × 12 fields × n_tris
    let expected     = attrs_start + 2 * n_tris;

    if data.len() < expected {
        eprintln!(
            "field_major_stl_delta_decode: data too short — have {} need {} (n_tris={})",
            data.len(), expected, n_tris
        );
        return data.to_vec();
    }

    // Undo stride-1 delta on all 48 (plane, field) blocks: LOW→HIGH
    // decoded[blk_idx][tri] = original byte_pos of float_field at triangle tri
    let decoded: Vec<Vec<u8>> = (0..48_usize)
        .map(|blk| {
            let start = blocks_start + blk * n_tris;
            let mut b = data[start..start + n_tris].to_vec();
            for i in 1..n_tris {
                b[i] = b[i].wrapping_add(b[i - 1]);
            }
            b
        })
        .collect();

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..84]);

    for tri in 0..n_tris {
        // Reconstruct FLOATS_PER_TRIANGLE floats × 4 bytes each
        for field in 0..FLOATS_PER_TRIANGLE {
            for byte_pos in 0..4usize {
                let blk = byte_pos * FLOATS_PER_TRIANGLE + field;
                out.push(decoded[blk][tri]);
            }
        }
        // 2 attribute bytes (raw)
        out.push(data[attrs_start + tri * 2]);
        out.push(data[attrs_start + tri * 2 + 1]);
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{apply_filter, detect_filter, undo_filter,
                         FILTER_SHUFFLE4_DELTA, FILTER_STL_FIELD_MAJOR};
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

    // ── detect tests ──────────────────────────────────────────────────────────

    #[test]
    fn detect_binary_stl_returns_field_major_flag() {
        let data = make_stl(10);
        assert_eq!(
            detect_filter(&data), FILTER_STL_FIELD_MAJOR,
            "detect_filter should return FILTER_STL_FIELD_MAJOR (10) for valid STL"
        );
    }

    #[test]
    fn detect_stl_rejects_wrong_size() {
        let n_tris: u32 = 10;
        let mut data = vec![0u8; 84 + 10 * 50 + 1]; // one byte too many
        data[80..84].copy_from_slice(&n_tris.to_le_bytes());
        assert_ne!(detect_filter(&data), FILTER_STL_FIELD_MAJOR);
        assert_ne!(detect_filter(&data), FILTER_SHUFFLE4_DELTA);
    }

    #[test]
    fn detect_stl_rejects_zero_tris() {
        let data = vec![0u8; 84];
        assert_eq!(detect_filter(&data), crate::filters::FILTER_NONE);
    }

    // ── flag 7: shuffle4 roundtrip (backward compat) ──────────────────────────

    #[test]
    fn roundtrip_shuffle4_stl_delta_minimal() {
        let data = make_stl(2);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len(), "flag7 must be size-preserving");
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "flag7 roundtrip failed");
    }

    #[test]
    fn roundtrip_shuffle4_stl_delta_larger() {
        let data = make_stl(500);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "flag7 roundtrip failed for 500 tris");
    }

    #[test]
    fn roundtrip_shuffle4_stl_delta_single_tri() {
        let data = make_stl(1);
        let enc = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_SHUFFLE4_DELTA);
        assert_eq!(dec, data, "flag7 roundtrip failed for 1 triangle");
    }

    // ── flag 10: field-major roundtrip ────────────────────────────────────────

    #[test]
    fn roundtrip_field_major_stl_delta_single_tri() {
        let data = make_stl(1);
        let enc = field_major_stl_delta_encode(&data);
        assert_eq!(enc.len(), data.len(), "flag10 must be size-preserving (1 tri)");
        let dec = field_major_stl_delta_decode(&enc);
        assert_eq!(dec, data, "flag10 roundtrip failed for 1 triangle");
    }

    #[test]
    fn roundtrip_field_major_stl_delta_minimal() {
        let data = make_stl(2);
        let enc = field_major_stl_delta_encode(&data);
        assert_eq!(enc.len(), data.len(), "flag10 must be size-preserving");
        let dec = field_major_stl_delta_decode(&enc);
        assert_eq!(dec, data, "flag10 roundtrip failed for 2 tris");
    }

    #[test]
    fn roundtrip_field_major_stl_delta_larger() {
        let data = make_stl(500);
        let enc = field_major_stl_delta_encode(&data);
        assert_eq!(enc.len(), data.len());
        let dec = field_major_stl_delta_decode(&enc);
        assert_eq!(dec, data, "flag10 roundtrip failed for 500 tris");
    }

    #[test]
    fn roundtrip_field_major_via_apply_undo() {
        let data = make_stl(200);
        let enc = apply_filter(&data, FILTER_STL_FIELD_MAJOR);
        assert_eq!(enc.len(), data.len());
        let dec = undo_filter(&enc, FILTER_STL_FIELD_MAJOR);
        assert_eq!(dec, data, "flag10 apply/undo roundtrip failed");
    }

    #[test]
    fn field_major_is_size_preserving() {
        for n in [1u32, 2, 10, 100, 500] {
            let data = make_stl(n);
            let enc = field_major_stl_delta_encode(&data);
            assert_eq!(
                enc.len(), data.len(),
                "flag10 size mismatch for {} tris: enc={} orig={}",
                n, enc.len(), data.len()
            );
        }
    }

    // ── flag 10 produces longer runs than flag 7 in FLOAT DATA on sphere ─────
    //
    // Root cause of the old test failure:
    //   max_run was measured on enc[84..] which includes the attribute bytes
    //   (2 per triangle, never set in the test → all zero). With 400 triangles
    //   the attr section is exactly 800 consecutive zeros, dominating max_run
    //   for BOTH encodings and making them equal.
    //
    // Fix: measure max_run on the float data region ONLY (enc[84..84+n*48]),
    //   which excludes the attr bytes. In this region:
    //   - flag10's y-component field blocks produce runs of ~n_lon-1 zeros per
    //     latitude ring (cos(phi) is constant within a ring, stride-1 delta = 0).
    //   - flag7's shuffled planes scatter y-field zeros among non-zero x/z
    //     deltas, giving isolated single zeros (max run ≈ 1–2).
    //   With n_lat=20, n_lon=20: run10 ≈ 19  vs  run7 ≈ 1–2.  19 > 1. ✓
    #[test]
    fn field_major_produces_longer_runs_than_shuffle4_on_sphere() {
        use std::f32::consts::PI;
        let n_lat = 20usize;
        let n_lon = 20usize;
        let mut tris: Vec<[f32; 12]> = Vec::new();
        for i in 0..n_lat {
            for j in 0..n_lon {
                let phi0 = PI * i as f32 / n_lat as f32;
                let phi1 = PI * (i + 1) as f32 / n_lat as f32;
                let th0  = 2.0 * PI * j as f32 / n_lon as f32;
                let th1  = 2.0 * PI * (j + 1) as f32 / n_lon as f32;
                let v = |p: f32, t: f32| [p.sin()*t.cos(), p.cos(), p.sin()*t.sin()];
                let v00 = v(phi0,th0); let v10 = v(phi1,th0); let v11 = v(phi1,th1);
                let nx = (v00[0]+v10[0]+v11[0])/3.0;
                let ny = (v00[1]+v10[1]+v11[1])/3.0;
                let nz = (v00[2]+v10[2]+v11[2])/3.0;
                let nl = (nx*nx+ny*ny+nz*nz).sqrt().max(1e-6);
                tris.push([nx/nl,ny/nl,nz/nl, v00[0],v00[1],v00[2],
                           v10[0],v10[1],v10[2], v11[0],v11[1],v11[2]]);
            }
        }
        let n = tris.len() as u32;
        let mut data = vec![0u8; 84 + n as usize * 50];
        data[80..84].copy_from_slice(&n.to_le_bytes());
        for (tri_idx, t) in tris.iter().enumerate() {
            let base = 84 + tri_idx * 50;
            for (f, &val) in t.iter().enumerate() {
                data[base + f*4..base + f*4 + 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        let enc7  = shuffle4_stl_delta_encode(&data);
        let enc10 = field_major_stl_delta_encode(&data);

        // Measure max run in FLOAT DATA ONLY — exclude attribute bytes which are
        // all zero (never set in this test) and would dominate both measurements.
        // Float data for n_tris triangles is always exactly 48*n_tris bytes for
        // both flag7 (4 planes × 12 floats × n_tris) and flag10 (48 blocks × n_tris).
        let n_tris = tris.len();
        let float_end = 84 + n_tris * 48;

        fn max_run(d: &[u8]) -> usize {
            if d.is_empty() { return 0; }
            let (mut best, mut cur) = (1usize, 1usize);
            for i in 1..d.len() {
                if d[i] == d[i-1] { cur += 1; best = best.max(cur); } else { cur = 1; }
            }
            best
        }

        let run7  = max_run(&enc7[84..float_end]);
        let run10 = max_run(&enc10[84..float_end]);

        assert!(
            run10 > run7,
            "flag10 max_run ({}) should exceed flag7 max_run ({}) on sphere geometry \
             (measured in float data region only, excluding {} attr bytes)",
            run10, run7, n_tris * 2,
        );
        println!(
            "flag7 max_run={} flag10 max_run={} (improvement: {:.1}x)",
            run7, run10,
            run10 as f64 / run7.max(1) as f64,
        );
    }

    // ── flag 7 entropy tests ──────────────────────────────────────────────────

    #[test]
    fn stl_delta_reduces_entropy_in_planes() {
        let n_tris       = 200usize;
        let total_floats = n_tris * FLOATS_PER_TRIANGLE;

        let mut data = vec![0u8; 84 + n_tris * 50];
        data[80..84].copy_from_slice(&(n_tris as u32).to_le_bytes());

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

    // ── flag10 vs flag7 entropy comparison on ramp geometry ──────────────────
    //
    // Old test (`stride12_delta_at_least_as_good_as_stride1_for_ramp`) asserted
    // that stride-12 entropy ≤ stride-1 entropy per plane. This is FALSE for
    // plane 2 of a monotone ramp: stride-1 on adjacent positions captures
    // finer-grained differences (1 ramp step vs 12) and genuinely wins.
    //
    // Correct property: flag10 (field-major + stride-1 per field) and flag7
    // (shuffle + stride-12) apply IDENTICAL per-field differences on a ramp
    // because both compare (field j of tri k) − (field j of tri k-1). Entropy
    // of the float data region must therefore be equal (Shannon entropy is
    // order-invariant — same multiset of byte values). The assertion
    // enc10_ent ≤ enc7_ent + 0.01 is always true in the equal case.
    #[test]
    fn field_major_entropy_matches_shuffle4_for_ramp() {
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

        // Verify both encodings roundtrip correctly first.
        let enc7  = apply_filter(&data, FILTER_SHUFFLE4_DELTA);
        let enc10 = apply_filter(&data, FILTER_STL_FIELD_MAJOR);
        assert_eq!(undo_filter(&enc7,  FILTER_SHUFFLE4_DELTA), data, "flag7 ramp roundtrip");
        assert_eq!(undo_filter(&enc10, FILTER_STL_FIELD_MAJOR), data, "flag10 ramp roundtrip");

        // Compare entropy on float data region only (exclude attr bytes).
        // Both encodings apply the same per-field differences → equal entropy.
        // flag10 must not be worse than flag7 (tolerance 0.01 for float rounding).
        let float_end = 84 + n_tris * 48;
        let ent7  = byte_entropy(&enc7[84..float_end]);
        let ent10 = byte_entropy(&enc10[84..float_end]);

        assert!(
            ent10 <= ent7 + 0.01,
            "flag10 float-data entropy ({:.4}) should be ≈≤ flag7 float-data entropy ({:.4}) \
             for ramp geometry (both apply identical per-field differences)",
            ent10, ent7,
        );
        println!(
            "flag7 entropy={:.4}  flag10 entropy={:.4}  diff={:.4}",
            ent7, ent10, ent10 - ent7,
        );
    }
    }
