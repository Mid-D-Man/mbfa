// src/filters/gltf.rs
//! Binary glTF (GLB) buffer array-delta filter.
//!
//! GLB structure (Khronos glTF 2.0 spec, public/open, unlike FBX): a 12-byte
//! header (magic "glTF", version, total length) followed by "chunks", each
//! `chunk_length:u32 + chunk_type:u32 + chunk_data(chunk_length bytes,
//! padded to 4)`. chunk_type 0x4E4F534A ("JSON") is the scene descriptor;
//! chunk_type 0x004E4942 ("BIN\0") holds the actual buffer bytes the JSON's
//! `bufferViews` point into.
//!
//! Same situation as FBX: a plain (no Draco/Meshopt extension) GLB stores
//! vertex positions/normals/UVs/indices as RAW typed arrays in the BIN
//! chunk -- exactly the kind of data PLY's filter already exploits, and
//! exactly why the existing GLB_Binary benchmark fixture shows MBFA
//! treating it as incompressible noise (100.01%, i.e. slightly EXPANDING)
//! today. Draco/Meshopt-compressed buffers use a totally different
//! mechanism (`extensions.KHR_draco_mesh_compression` on the primitive,
//! not a plain accessor->bufferView reference), so this filter naturally
//! never touches them -- it only ever processes bufferViews reached via a
//! standard, direct `accessor.bufferView` reference.
//!
//! No JSON library is used (the crate has zero required dependencies by
//! design) -- a small purpose-built scanner extracts exactly the fields
//! needed (`bufferViews[].byteOffset/byteLength/byteStride`,
//! `accessors[].bufferView/componentType/type`) via brace/bracket-depth
//! matching, not a general JSON parser. `bufferViews` with a `byteStride`
//! (meaning multiple attributes are interleaved in one region) are left
//! untouched entirely -- correctly handling interleaved layouts needs
//! per-attribute offsets within the stride, which is real added
//! complexity for a case tool-generated exports don't always use; only the
//! common non-interleaved case is delta-filtered.
//!
//! Safety: exactly the same reasoning as fbx.rs -- byte-level delta is its
//! own exact inverse on any byte range, and this filter only ever writes
//! inside a bufferView's own bytes in the BIN chunk, never inside the JSON
//! chunk itself (which is what both encode and decode re-derive the same
//! regions from, unchanged). `detect_filter` still requires a full real
//! encode/decode round-trip against the actual input before ever
//! activating this filter.

const GLB_MAGIC: [u8; 4] = *b"glTF";
const CHUNK_TYPE_JSON: u32 = 0x4E4F_534A;
const CHUNK_TYPE_BIN:  u32 = 0x004E_4942;

struct BufferRegion {
    start: usize,
    len: usize,
    elem_size: usize,
}

fn ru32(d: &[u8], o: usize) -> Option<u32> {
    d.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn component_byte_size(component_type: i64) -> Option<usize> {
    match component_type {
        5120 | 5121 => Some(1), // BYTE / UNSIGNED_BYTE
        5122 | 5123 => Some(2), // SHORT / UNSIGNED_SHORT
        5125 | 5126 => Some(4), // UNSIGNED_INT / FLOAT
        _ => None,
    }
}

fn type_component_count(ty: &str) -> Option<usize> {
    match ty {
        "SCALAR" => Some(1),
        "VEC2" => Some(2),
        "VEC3" => Some(3),
        "VEC4" => Some(4),
        "MAT2" => Some(4),
        "MAT3" => Some(9),
        "MAT4" => Some(16),
        _ => None,
    }
}

/// Finds the byte range `[open, close]` (inclusive of both bracket chars)
/// of the JSON array or object value that immediately follows `"key":` in
/// `json`, searched from `from`. Only tracks `{}`/`[]` depth and double
/// quotes (skipping escaped quotes) -- not a general JSON parser, just
/// enough to find matching boundaries in well-formed exporter output.
fn find_value_span(json: &[u8], key: &str, from: usize) -> Option<(usize, usize)> {
    let needle = format!("\"{}\"", key);
    let key_pos = find_bytes(json, needle.as_bytes(), from)?;
    let mut i = key_pos + needle.len();
    while i < json.len() && (json[i] as char).is_whitespace() { i += 1; }
    if json.get(i) != Some(&b':') { return None; }
    i += 1;
    while i < json.len() && (json[i] as char).is_whitespace() { i += 1; }
    let open = i;
    let open_ch = *json.get(open)?;
    let (open_b, close_b) = match open_ch {
        b'[' => (b'[', b']'),
        b'{' => (b'{', b'}'),
        _ => return None,
    };
    let mut depth = 0i32;
    let mut in_str = false;
    let mut j = open;
    while j < json.len() {
        let c = json[j];
        if in_str {
            if c == b'\\' { j += 1; }
            else if c == b'"' { in_str = false; }
        } else if c == b'"' {
            in_str = true;
        } else if c == open_b {
            depth += 1;
        } else if c == close_b {
            depth -= 1;
            if depth == 0 { return Some((open, j)); }
        }
        j += 1;
    }
    None
}

fn find_bytes(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= hay.len() { return None; }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

/// Splits a bracket-matched JSON array span `(open, close)` (as returned by
/// `find_value_span`) into the byte ranges of its immediate top-level
/// elements (typically objects).
fn split_array_elements(json: &[u8], open: usize, close: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut elem_start: Option<usize> = None;
    let mut i = open + 1;
    while i < close {
        let c = json[i];
        if in_str {
            if c == b'\\' { i += 1; }
            else if c == b'"' { in_str = false; }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' | b'[' => {
                    if depth == 0 { elem_start = Some(i); }
                    depth += 1;
                }
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = elem_start.take() { out.push((s, i)); }
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    out
}

/// Reads a `"key": <integer>` numeric field from within `obj` (a
/// bracket-matched `{...}` span). Returns None if absent.
fn find_int_field(json: &[u8], obj: (usize, usize), key: &str) -> Option<i64> {
    let needle = format!("\"{}\"", key);
    let pos = find_bytes(&json[obj.0..=obj.1], needle.as_bytes(), 0)? + obj.0;
    let mut i = pos + needle.len();
    while i <= obj.1 && (json[i] as char).is_whitespace() { i += 1; }
    if json.get(i) != Some(&b':') { return None; }
    i += 1;
    while i <= obj.1 && (json[i] as char).is_whitespace() { i += 1; }
    let start = i;
    let mut end = i;
    while end <= obj.1 && (json[end] == b'-' || json[end].is_ascii_digit()) { end += 1; }
    if end == start { return None; }
    std::str::from_utf8(&json[start..end]).ok()?.parse().ok()
}

fn find_str_field<'a>(json: &'a [u8], obj: (usize, usize), key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\"", key);
    let pos = find_bytes(&json[obj.0..=obj.1], needle.as_bytes(), 0)? + obj.0;
    let mut i = pos + needle.len();
    while i <= obj.1 && (json[i] as char).is_whitespace() { i += 1; }
    if json.get(i) != Some(&b':') { return None; }
    i += 1;
    while i <= obj.1 && (json[i] as char).is_whitespace() { i += 1; }
    if json.get(i) != Some(&b'"') { return None; }
    let start = i + 1;
    let mut end = start;
    while end <= obj.1 && json[end] != b'"' { end += 1; }
    std::str::from_utf8(&json[start..end]).ok()
}

struct Chunks {
    json_start: usize,
    json_len: usize,
    bin_start: usize,
    bin_len: usize,
}

fn parse_glb_chunks(data: &[u8]) -> Option<Chunks> {
    if data.len() < 12 || data[0..4] != GLB_MAGIC { return None; }
    let total_len = ru32(data, 8)? as usize;
    if total_len > data.len() { return None; }

    let mut pos = 12;
    let mut json_range = None;
    let mut bin_range = None;
    while pos + 8 <= total_len {
        let chunk_len  = ru32(data, pos)? as usize;
        let chunk_type = ru32(data, pos + 4)?;
        let data_start = pos + 8;
        if data_start + chunk_len > total_len { return None; }
        match chunk_type {
            CHUNK_TYPE_JSON => json_range = Some((data_start, chunk_len)),
            CHUNK_TYPE_BIN  => bin_range  = Some((data_start, chunk_len)),
            _ => {} // unknown chunk type (extension) — ignore, don't touch
        }
        pos = data_start + chunk_len;
    }
    let (json_start, json_len) = json_range?;
    let (bin_start, bin_len) = bin_range.unwrap_or((0, 0));
    Some(Chunks { json_start, json_len, bin_start, bin_len })
}

fn scan_gltf(data: &[u8]) -> Option<Vec<BufferRegion>> {
    let chunks = parse_glb_chunks(data)?;
    if chunks.bin_len == 0 { return None; } // nothing to filter without a BIN chunk
    let json = &data[chunks.json_start..chunks.json_start + chunks.json_len];

    let (bv_open, bv_close) = find_value_span(json, "bufferViews", 0)?;
    let buffer_views = split_array_elements(json, bv_open, bv_close);
    if buffer_views.is_empty() { return None; }

    // elem_size per bufferView index, from whichever accessor(s) reference
    // it directly (non-strided only).
    let mut elem_size_for_bv: Vec<Option<usize>> = vec![None; buffer_views.len()];
    if let Some((ac_open, ac_close)) = find_value_span(json, "accessors", 0) {
        for acc in split_array_elements(json, ac_open, ac_close) {
            let bv_idx = match find_int_field(json, acc, "bufferView") {
                Some(v) if v >= 0 => v as usize,
                _ => continue, // sparse-only accessor, no direct bufferView — skip
            };
            if bv_idx >= buffer_views.len() { continue; }
            let component_type = match find_int_field(json, acc, "componentType") {
                Some(v) => v,
                None => continue,
            };
            let ty = match find_str_field(json, acc, "type") {
                Some(v) => v,
                None => continue,
            };
            let (Some(csz), Some(ncomp)) = (component_byte_size(component_type), type_component_count(ty)) else { continue };
            elem_size_for_bv[bv_idx] = Some(csz * ncomp);
        }
    }

    let mut regions = Vec::new();
    for (i, &bv) in buffer_views.iter().enumerate() {
        if find_int_field(json, bv, "byteStride").is_some() { continue; } // interleaved — skip
        let byte_offset = find_int_field(json, bv, "byteOffset").unwrap_or(0);
        let byte_length = match find_int_field(json, bv, "byteLength") {
            Some(v) if v > 0 => v as usize,
            _ => continue,
        };
        let elem_size = match elem_size_for_bv[i] {
            Some(sz) if sz > 0 && byte_length % sz == 0 => sz,
            _ => continue, // no clean single-accessor mapping — skip, don't guess
        };
        if byte_offset < 0 { continue; }
        let start = chunks.bin_start + byte_offset as usize;
        if start + byte_length > chunks.bin_start + chunks.bin_len { continue; }
        regions.push(BufferRegion { start, len: byte_length, elem_size });
    }
    if regions.is_empty() { return None; }
    Some(regions)
}

fn delta_encode_region(buf: &mut [u8], r: &BufferRegion) {
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

fn delta_decode_region(buf: &mut [u8], r: &BufferRegion) {
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

pub fn gltf_delta_encode(input: &[u8]) -> Vec<u8> {
    match scan_gltf(input) {
        Some(regions) => {
            let mut out = input.to_vec();
            for r in &regions { delta_encode_region(&mut out, r); }
            out
        }
        None => input.to_vec(),
    }
}

pub fn gltf_delta_decode(input: &[u8]) -> Vec<u8> {
    match scan_gltf(input) {
        Some(regions) => {
            let mut out = input.to_vec();
            for r in &regions { delta_decode_region(&mut out, r); }
            out
        }
        None => input.to_vec(),
    }
}

/// Returns `Some(FILTER_GLTF_BUFFER_DELTA)` only if `input` parses as a
/// structurally-consistent GLB with at least one eligible raw bufferView
/// AND a real encode/decode round-trip against these exact bytes
/// reproduces them exactly.
pub fn detect_gltf(input: &[u8]) -> Option<u8> {
    if input.len() < 12 || input[0..4] != GLB_MAGIC { return None; }
    scan_gltf(input)?;
    let encoded = gltf_delta_encode(input);
    let decoded = gltf_delta_decode(&encoded);
    if decoded != input { return None; }
    Some(crate::filters::FILTER_GLTF_BUFFER_DELTA)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a real, valid, minimal GLB: JSON chunk describing one
    /// non-interleaved VEC3 FLOAT accessor + one SCALAR UNSIGNED_SHORT
    /// index accessor, backed by a BIN chunk with real (smooth,
    /// correlated) raw data for both.
    fn build_test_glb(index_count: usize, vert_count: usize) -> Vec<u8> {
        let idx_vals: Vec<u16> = (0..index_count as u16).collect();
        let mut idx_bytes = Vec::new();
        for v in &idx_vals { idx_bytes.extend_from_slice(&v.to_le_bytes()); }
        while idx_bytes.len() % 4 != 0 { idx_bytes.push(0); }

        let vert_vals: Vec<f32> = (0..vert_count * 3).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let mut vert_bytes = Vec::new();
        for v in &vert_vals { vert_bytes.extend_from_slice(&v.to_le_bytes()); }

        let mut bin = idx_bytes.clone();
        bin.extend_from_slice(&vert_bytes);
        while bin.len() % 4 != 0 { bin.push(0); }

        let json_text = format!(
            "{{\"asset\":{{\"version\":\"2.0\"}},\"accessors\":[\
             {{\"bufferView\":0,\"componentType\":5123,\"count\":{},\"type\":\"SCALAR\"}},\
             {{\"bufferView\":1,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\"}}\
             ],\"bufferViews\":[\
             {{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{}}},\
             {{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}\
             ],\"buffers\":[{{\"byteLength\":{}}}]}}",
            index_count, vert_count,
            idx_bytes.len(),
            idx_bytes.len(), vert_bytes.len(),
            bin.len(),
        );
        let mut json = json_text.into_bytes();
        while json.len() % 4 != 0 { json.push(b' '); }

        let mut d = Vec::new();
        d.extend_from_slice(&GLB_MAGIC);
        d.extend_from_slice(&2u32.to_le_bytes()); // version
        let total_len_pos = d.len();
        d.extend_from_slice(&0u32.to_le_bytes()); // total_length placeholder

        d.extend_from_slice(&(json.len() as u32).to_le_bytes());
        d.extend_from_slice(&CHUNK_TYPE_JSON.to_le_bytes());
        d.extend_from_slice(&json);

        d.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        d.extend_from_slice(&CHUNK_TYPE_BIN.to_le_bytes());
        d.extend_from_slice(&bin);

        let total_len = d.len() as u32;
        d[total_len_pos..total_len_pos + 4].copy_from_slice(&total_len.to_le_bytes());
        d
    }

    #[test]
    fn detect_gltf_fires_on_valid_file() {
        let data = build_test_glb(6, 40);
        assert!(detect_gltf(&data).is_some());
    }

    #[test]
    fn gltf_roundtrips_exactly() {
        let data = build_test_glb(6, 40);
        let encoded = gltf_delta_encode(&data);
        let decoded = gltf_delta_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn gltf_actually_delta_encodes() {
        let data = build_test_glb(6, 40);
        let encoded = gltf_delta_encode(&data);
        assert_ne!(encoded, data);
    }

    #[test]
    fn gltf_json_chunk_never_touched() {
        let data = build_test_glb(6, 40);
        let encoded = gltf_delta_encode(&data);
        let chunks = parse_glb_chunks(&data).unwrap();
        let json_before = &data[chunks.json_start..chunks.json_start + chunks.json_len];
        let json_after  = &encoded[chunks.json_start..chunks.json_start + chunks.json_len];
        assert_eq!(json_before, json_after, "JSON chunk bytes must never change");
    }

    #[test]
    fn detect_gltf_rejects_wrong_magic() {
        let mut data = build_test_glb(6, 40);
        data[0] = 0;
        assert!(detect_gltf(&data).is_none());
    }

    #[test]
    fn detect_gltf_rejects_truncated() {
        let data = vec![0u8; 8];
        assert!(detect_gltf(&data).is_none());
    }

    #[test]
    fn detect_gltf_rejects_no_bin_chunk() {
        // JSON-only GLB (valid per spec, e.g. an all-external-buffer file)
        // has nothing in-file to delta.
        let json_text = "{\"asset\":{\"version\":\"2.0\"}}";
        let mut json = json_text.as_bytes().to_vec();
        while json.len() % 4 != 0 { json.push(b' '); }
        let mut d = Vec::new();
        d.extend_from_slice(&GLB_MAGIC);
        d.extend_from_slice(&2u32.to_le_bytes());
        let total_pos = d.len();
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&(json.len() as u32).to_le_bytes());
        d.extend_from_slice(&CHUNK_TYPE_JSON.to_le_bytes());
        d.extend_from_slice(&json);
        let total_len = d.len() as u32;
        d[total_pos..total_pos + 4].copy_from_slice(&total_len.to_le_bytes());
        assert!(detect_gltf(&d).is_none());
    }

    #[test]
    fn gltf_skips_interleaved_bufferview_with_stride() {
        // A bufferView with byteStride must be left completely untouched.
        let idx_vals: Vec<u16> = (0..6u16).collect();
        let mut idx_bytes = Vec::new();
        for v in &idx_vals { idx_bytes.extend_from_slice(&v.to_le_bytes()); }
        while idx_bytes.len() % 4 != 0 { idx_bytes.push(0); }

        // interleaved pos+normal, stride 24 (VEC3 f32 + VEC3 f32), 10 verts
        let interleaved: Vec<u8> = (0..10 * 24).map(|i| (i % 251) as u8).collect();

        let mut bin = idx_bytes.clone();
        bin.extend_from_slice(&interleaved);
        while bin.len() % 4 != 0 { bin.push(0); }

        let json_text = format!(
            "{{\"accessors\":[\
             {{\"bufferView\":0,\"componentType\":5123,\"count\":6,\"type\":\"SCALAR\"}},\
             {{\"bufferView\":1,\"componentType\":5126,\"count\":10,\"type\":\"VEC3\"}}\
             ],\"bufferViews\":[\
             {{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{}}},\
             {{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"byteStride\":24}}\
             ]}}",
            idx_bytes.len(), idx_bytes.len(), interleaved.len(),
        );
        let mut json = json_text.into_bytes();
        while json.len() % 4 != 0 { json.push(b' '); }

        let mut d = Vec::new();
        d.extend_from_slice(&GLB_MAGIC);
        d.extend_from_slice(&2u32.to_le_bytes());
        let total_pos = d.len();
        d.extend_from_slice(&0u32.to_le_bytes());
        d.extend_from_slice(&(json.len() as u32).to_le_bytes());
        d.extend_from_slice(&CHUNK_TYPE_JSON.to_le_bytes());
        d.extend_from_slice(&json);
        d.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        d.extend_from_slice(&CHUNK_TYPE_BIN.to_le_bytes());
        d.extend_from_slice(&bin);
        let total_len = d.len() as u32;
        d[total_pos..total_pos + 4].copy_from_slice(&total_len.to_le_bytes());

        // Only the (non-strided) index bufferView is eligible -- still
        // enough to activate the filter, but the interleaved region must
        // be byte-identical before/after.
        if let Some(flag) = detect_gltf(&d) {
            assert_eq!(flag, crate::filters::FILTER_GLTF_BUFFER_DELTA);
            let encoded = gltf_delta_encode(&d);
            let chunks = parse_glb_chunks(&d).unwrap();
            let interleaved_start = chunks.bin_start + idx_bytes.len();
            let before = &d[interleaved_start..interleaved_start + interleaved.len()];
            let after  = &encoded[interleaved_start..interleaved_start + interleaved.len()];
            assert_eq!(before, after, "strided bufferView must never be touched");
            let decoded = gltf_delta_decode(&encoded);
            assert_eq!(decoded, d);
        }
    }
}
