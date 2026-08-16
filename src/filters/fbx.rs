// src/filters/fbx.rs
//! Binary FBX (Autodesk's 3D interchange format) array-delta filter.
//!
//! Real format, from the Blender Foundation's primary-source spec (Alexander
//! Gessler, 2013, code.blender.org -- the standard reference, since FBX
//! itself is undocumented/proprietary): the file is a sequence of "node
//! records", each with an EndOffset (so parsers can skip unknown nodes),
//! a tuple of typed properties, and an optional nested list of child node
//! records terminated by an all-zero "NULL record".
//!
//! What matters for compression: several property types (`f`/`d`/`l`/`i`)
//! are typed NUMERIC ARRAYS -- vertex positions, normals, UVs, indices --
//! and each array independently records whether its content is stored raw
//! or zlib-deflated (`Encoding` field). Deflated arrays are already
//! compressed (nothing to gain, like PNG/JPEG). RAW arrays are exactly the
//! same situation PLY's filter already exploits: consecutive same-typed
//! elements (e.g. adjacent vertices' X/Y/Z) are usually closer in value
//! than in raw byte pattern, so a per-element-width byte delta exposes
//! redundancy plain LZ misses.
//!
//! This filter walks the real node-tree structure (handling both the
//! pre-7500 32-bit and the 7500+ 64-bit header field width -- the format
//! changed field width for large-file support, per the FBX 7.5 release
//! notes) to find every raw array's exact Contents byte range, and
//! delta-encodes each range in place at its own element width. Nothing
//! else in the file is touched.
//!
//! Safety: byte-level delta is its own exact inverse on ANY byte range
//! regardless of what it represents, so a wrong array-boundary guess can
//! never corrupt data BY ITSELF -- but it matters that encode and decode
//! agree on the SAME boundaries, and they only do that if neither one is
//! ever computed from a byte the other side's transform changed. This
//! filter only ever writes inside an array's own Contents bytes -- never
//! a TypeCode, an EndOffset/NumProperties/PropertyListLen/NameLen field,
//! a Name, or another property's bytes -- so decode's re-walk over the
//! delta-transformed file finds the exact same tree structure and thus
//! the exact same regions to invert. `detect_filter` still runs a full
//! encode-then-decode round-trip against the real input before ever
//! activating this filter, the same discipline as the CFBF filter, given
//! how easy hand-rolled binary tree walkers are to get subtly wrong.

const FBX_MAGIC: &[u8] = b"Kaydara FBX Binary  \x00";
const HEADER_LEN: usize = 27; // 21 (magic) + 2 (unknown) + 4 (version)
const WIDE_VERSION_CUTOFF: u32 = 7500; // FBX 7.5+: 64-bit node header fields

struct ArrayRegion {
    start: usize,
    len: usize,
    elem_size: usize,
}

fn ru32(d: &[u8], o: usize) -> Option<u32> {
    d.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn ru64(d: &[u8], o: usize) -> Option<u64> {
    d.get(o..o + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap()))
}

/// Reads a node-record header field (EndOffset/NumProperties/
/// PropertyListLen), which is u32 pre-7500 and u64 from 7500 on.
fn read_header_field(d: &[u8], o: usize, wide: bool) -> Option<u64> {
    if wide { ru64(d, o) } else { ru32(d, o).map(|v| v as u64) }
}

fn is_all_zero(d: &[u8], o: usize, len: usize) -> bool {
    d.get(o..o + len).map(|s| s.iter().all(|&b| b == 0)).unwrap_or(false)
}

/// Recursively walks node records starting at `pos`, collecting every raw
/// (Encoding=0) array property's Contents byte range into `out`. Returns
/// the record's EndOffset on success. Returns None on ANY structural
/// inconsistency -- callers must not use partial results.
fn walk_node(data: &[u8], pos: usize, wide: bool, depth: u32, out: &mut Vec<ArrayRegion>) -> Option<usize> {
    if depth > 256 { return None; } // pathological-nesting guard
    let field_w = if wide { 8 } else { 4 };
    let end_offset  = read_header_field(data, pos, wide)? as usize;
    let num_props   = read_header_field(data, pos + field_w, wide)?;
    let _prop_list_len = read_header_field(data, pos + field_w * 2, wide)?;
    let name_len_off = pos + field_w * 3;
    let name_len = *data.get(name_len_off)? as usize;
    let mut cur = name_len_off + 1 + name_len;
    if end_offset > data.len() || cur > end_offset { return None; }

    for _ in 0..num_props {
        let type_code = *data.get(cur)?;
        cur += 1;
        match type_code {
            b'Y' => cur += 2,
            b'C' => cur += 1,
            b'I' => cur += 4,
            b'F' => cur += 4,
            b'D' => cur += 8,
            b'L' => cur += 8,
            b'f' | b'd' | b'l' | b'i' | b'b' => {
                let array_len   = ru32(data, cur)? as usize;
                let encoding    = ru32(data, cur + 4)?;
                let comp_len    = ru32(data, cur + 8)? as usize;
                let contents_at = cur + 12;
                if contents_at + comp_len > data.len() || contents_at + comp_len > end_offset {
                    return None;
                }
                let elem_size = match type_code {
                    b'f' | b'i' => 4,
                    b'd' | b'l' => 8,
                    b'b' => 1,
                    _ => unreachable!(),
                };
                if encoding == 0 {
                    // Raw contents length should be array_len * elem_size;
                    // trust comp_len (what's really on disk) for the byte
                    // range, but require it to agree as a sanity check --
                    // any file where it doesn't just isn't touched here.
                    if comp_len != array_len * elem_size { return None; }
                    if comp_len > 0 {
                        out.push(ArrayRegion { start: contents_at, len: comp_len, elem_size });
                    }
                }
                // encoding==1 (zlib) or anything else: already compressed
                // or unrecognized -- skip over it untouched either way.
                cur = contents_at + comp_len;
            }
            b'S' | b'R' => {
                let len = ru32(data, cur)? as usize;
                cur += 4;
                if cur + len > data.len() { return None; }
                cur += len;
            }
            _ => return None, // unknown property type code
        }
    }

    if cur > end_offset { return None; }
    if cur < end_offset {
        // Nested list follows, terminated by an all-zero NULL record.
        let null_len = field_w * 3 + 1;
        loop {
            if cur > end_offset { return None; }
            if is_all_zero(data, cur, null_len) {
                cur += null_len;
                break;
            }
            cur = walk_node(data, cur, wide, depth + 1, out)?;
        }
    }
    if cur != end_offset { return None; }
    Some(end_offset)
}

/// Walks every top-level node record until the structure stops looking
/// like a valid node (end of the recognized region -- e.g. the format's
/// documented-as-unknown footer), collecting raw array regions as it goes.
fn scan_fbx(data: &[u8]) -> Option<Vec<ArrayRegion>> {
    if data.len() < HEADER_LEN || &data[0..21] != FBX_MAGIC { return None; }
    let version = ru32(data, 23)?;
    let wide = version >= WIDE_VERSION_CUTOFF;

    let mut regions = Vec::new();
    let mut pos = HEADER_LEN;
    let field_w = if wide { 8 } else { 4 };
    let null_len = field_w * 3 + 1;
    while pos + null_len <= data.len() {
        if is_all_zero(data, pos, null_len) { break; } // no more top-level nodes
        match walk_node(data, pos, wide, 0, &mut regions) {
            Some(end) if end > pos => pos = end,
            _ => break, // unrecognized from here on (e.g. the undocumented footer) -- stop, don't guess
        }
    }
    if regions.is_empty() { return None; }
    Some(regions)
}

fn delta_encode_region(buf: &mut [u8], r: &ArrayRegion) {
    let region = &mut buf[r.start..r.start + r.len];
    let n = r.elem_size;
    let mut i = region.len();
    while i >= 2 * n {
        i -= n;
        for k in 0..n {
            region[i + k] = region[i + k].wrapping_sub(region[i - n + k]);
        }
    }
}

fn delta_decode_region(buf: &mut [u8], r: &ArrayRegion) {
    let region = &mut buf[r.start..r.start + r.len];
    let n = r.elem_size;
    let mut i = n;
    while i + n <= region.len() {
        for k in 0..n {
            region[i + k] = region[i + k].wrapping_add(region[i - n + k]);
        }
        i += n;
    }
}

pub fn fbx_delta_encode(input: &[u8]) -> Vec<u8> {
    match scan_fbx(input) {
        Some(regions) => {
            let mut out = input.to_vec();
            for r in &regions { delta_encode_region(&mut out, r); }
            out
        }
        None => input.to_vec(),
    }
}

pub fn fbx_delta_decode(input: &[u8]) -> Vec<u8> {
    match scan_fbx(input) {
        Some(regions) => {
            let mut out = input.to_vec();
            for r in &regions { delta_decode_region(&mut out, r); }
            out
        }
        None => input.to_vec(),
    }
}

/// Returns `Some(FILTER_FBX_ARRAY_DELTA)` only if `input` parses as a
/// structurally-consistent binary FBX file with at least one raw array AND
/// a real encode/decode round-trip against these exact bytes reproduces
/// them exactly. See the module doc for why the round-trip isn't optional.
pub fn detect_fbx(input: &[u8]) -> Option<u8> {
    if input.len() < HEADER_LEN || &input[0..21] != FBX_MAGIC { return None; }
    scan_fbx(input)?;
    let encoded = fbx_delta_encode(input);
    let decoded = fbx_delta_decode(&encoded);
    if decoded != input { return None; }
    Some(crate::filters::FILTER_FBX_ARRAY_DELTA)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal, real, valid binary FBX (pre-7500, 32-bit header
    /// fields) with a top-level node containing one raw f32 array
    /// property (smooth, correlated values -- exactly what delta should
    /// help) and one already-"compressed" (encoding=1) array that must be
    /// left untouched.
    fn build_test_fbx() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(FBX_MAGIC);
        d.extend_from_slice(&[0x1A, 0x00]);
        d.extend_from_slice(&7300u32.to_le_bytes()); // version (narrow header)

        // -- top-level node "Geometry" with 2 properties: raw f32[] array,
        //    then a "compressed" i32[] array (encoding=1, content is just
        //    arbitrary bytes here since we never touch or interpret it).
        let name = b"Geometry";
        let raw_vals: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.01).collect();
        let mut raw_bytes = Vec::new();
        for v in &raw_vals { raw_bytes.extend_from_slice(&v.to_le_bytes()); }

        let fake_compressed = vec![0xEEu8; 40]; // stand-in "zlib" bytes, never touched

        let mut props = Vec::new();
        // property 0: f, raw array
        props.push(b'f');
        props.extend_from_slice(&(raw_vals.len() as u32).to_le_bytes()); // ArrayLength
        props.extend_from_slice(&0u32.to_le_bytes()); // Encoding = 0 (raw)
        props.extend_from_slice(&(raw_bytes.len() as u32).to_le_bytes()); // CompressedLength
        props.extend_from_slice(&raw_bytes);
        // property 1: i, "compressed" array -- must survive untouched
        props.push(b'i');
        props.extend_from_slice(&10u32.to_le_bytes()); // ArrayLength (irrelevant when encoding=1)
        props.extend_from_slice(&1u32.to_le_bytes()); // Encoding = 1
        props.extend_from_slice(&(fake_compressed.len() as u32).to_le_bytes());
        props.extend_from_slice(&fake_compressed);

        let num_properties = 2u32;
        let property_list_len = props.len() as u32;
        let header_len = 4 + 4 + 4 + 1 + name.len();
        let end_offset = (d.len() + header_len + props.len()) as u32;

        d.extend_from_slice(&end_offset.to_le_bytes());
        d.extend_from_slice(&num_properties.to_le_bytes());
        d.extend_from_slice(&property_list_len.to_le_bytes());
        d.push(name.len() as u8);
        d.extend_from_slice(name);
        d.extend_from_slice(&props);

        d
    }

    #[test]
    fn detect_fbx_fires_on_valid_file() {
        let data = build_test_fbx();
        assert!(detect_fbx(&data).is_some());
    }

    #[test]
    fn fbx_roundtrips_exactly() {
        let data = build_test_fbx();
        let encoded = fbx_delta_encode(&data);
        let decoded = fbx_delta_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn fbx_actually_delta_encodes_raw_array() {
        let data = build_test_fbx();
        let encoded = fbx_delta_encode(&data);
        assert_ne!(encoded, data, "encoding should change the raw f32 array bytes");
    }

    #[test]
    fn fbx_leaves_compressed_array_untouched() {
        let data = build_test_fbx();
        let encoded = fbx_delta_encode(&data);
        // The 40 x 0xEE "compressed" bytes must appear byte-identical
        // somewhere in the encoded output (their position doesn't move).
        let needle = vec![0xEEu8; 40];
        assert!(encoded.windows(40).any(|w| w == needle.as_slice()),
            "encoding=1 array content must never be touched");
    }

    #[test]
    fn detect_fbx_rejects_wrong_magic() {
        let mut data = build_test_fbx();
        data[0] = 0;
        assert!(detect_fbx(&data).is_none());
    }

    #[test]
    fn detect_fbx_rejects_truncated_file() {
        let data = vec![0u8; 10];
        assert!(detect_fbx(&data).is_none());
    }

    #[test]
    fn detect_fbx_rejects_no_arrays() {
        // Valid header, valid empty top-level node, but zero properties ->
        // nothing to defragment/delta, so this filter should decline
        // (something else, or none, should handle the file).
        let mut d = Vec::new();
        d.extend_from_slice(FBX_MAGIC);
        d.extend_from_slice(&[0x1A, 0x00]);
        d.extend_from_slice(&7300u32.to_le_bytes());
        let end_offset = (d.len() + 4 + 4 + 4 + 1) as u32;
        d.extend_from_slice(&end_offset.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes());
        d.push(0);
        assert!(detect_fbx(&d).is_none());
    }

    #[test]
    fn fbx_wide_header_7500_roundtrips() {
        // Same shape as build_test_fbx but with 64-bit node header fields,
        // as used from FBX 7.5 onward.
        let mut d = Vec::new();
        d.extend_from_slice(FBX_MAGIC);
        d.extend_from_slice(&[0x1A, 0x00]);
        d.extend_from_slice(&7500u32.to_le_bytes()); // wide version

        let name = b"Geometry";
        let raw_vals: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.5).collect();
        let mut raw_bytes = Vec::new();
        for v in &raw_vals { raw_bytes.extend_from_slice(&v.to_le_bytes()); }

        let mut props = Vec::new();
        props.push(b'd');
        props.extend_from_slice(&(raw_vals.len() as u32).to_le_bytes());
        props.extend_from_slice(&0u32.to_le_bytes());
        props.extend_from_slice(&(raw_bytes.len() as u32).to_le_bytes());
        props.extend_from_slice(&raw_bytes);

        let header_len = 8 + 8 + 8 + 1 + name.len();
        let end_offset = (d.len() + header_len + props.len()) as u64;

        d.extend_from_slice(&end_offset.to_le_bytes());
        d.extend_from_slice(&1u64.to_le_bytes()); // NumProperties
        d.extend_from_slice(&(props.len() as u64).to_le_bytes());
        d.push(name.len() as u8);
        d.extend_from_slice(name);
        d.extend_from_slice(&props);

        assert!(detect_fbx(&d).is_some());
        let encoded = fbx_delta_encode(&d);
        assert_ne!(encoded, d);
        let decoded = fbx_delta_decode(&encoded);
        assert_eq!(decoded, d);
    }

    #[test]
    fn fbx_nested_children_roundtrip() {
        // Parent node with zero properties of its own, one child node that
        // has a raw array, terminated by the 13-byte NULL record.
        let mut d = Vec::new();
        d.extend_from_slice(FBX_MAGIC);
        d.extend_from_slice(&[0x1A, 0x00]);
        d.extend_from_slice(&7300u32.to_le_bytes());

        // Build the child node bytes first (same shape as a single-prop
        // node in build_test_fbx, reused conceptually).
        let child_name = b"Vertices";
        let raw_vals: Vec<f32> = (0..20).map(|i| 1.0 + (i as f32) * 0.25).collect();
        let mut raw_bytes = Vec::new();
        for v in &raw_vals { raw_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut child_props = Vec::new();
        child_props.push(b'f');
        child_props.extend_from_slice(&(raw_vals.len() as u32).to_le_bytes());
        child_props.extend_from_slice(&0u32.to_le_bytes());
        child_props.extend_from_slice(&(raw_bytes.len() as u32).to_le_bytes());
        child_props.extend_from_slice(&raw_bytes);

        let parent_name = b"Geometry";
        let parent_header_len = 4 + 4 + 4 + 1 + parent_name.len();
        let parent_start = d.len();
        let child_start = parent_start + parent_header_len; // parent has 0 properties
        let child_header_len = 4 + 4 + 4 + 1 + child_name.len();
        let child_end = child_start + child_header_len + child_props.len();
        let null_record_end = child_end + 13;

        // parent header
        d.extend_from_slice(&(null_record_end as u32).to_le_bytes()); // EndOffset
        d.extend_from_slice(&0u32.to_le_bytes()); // NumProperties = 0
        d.extend_from_slice(&0u32.to_le_bytes()); // PropertyListLen = 0
        d.push(parent_name.len() as u8);
        d.extend_from_slice(parent_name);
        // child node
        d.extend_from_slice(&(child_end as u32).to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&(child_props.len() as u32).to_le_bytes());
        d.push(child_name.len() as u8);
        d.extend_from_slice(child_name);
        d.extend_from_slice(&child_props);
        // NULL record terminating the parent's nested list
        d.extend_from_slice(&[0u8; 13]);

        assert!(detect_fbx(&d).is_some());
        let encoded = fbx_delta_encode(&d);
        assert_ne!(encoded, d);
        let decoded = fbx_delta_decode(&encoded);
        assert_eq!(decoded, d);
    }
}
