// src/png_xform.rs
use std::io::{Read, Write};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

pub fn is_png(data: &[u8]) -> bool {
    data.len() >= 8 && data[0..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
}

#[derive(Debug, Clone)]
pub struct PngMeta {
    pub width:       u32,
    pub height:      u32,
    pub bit_depth:   u8,
    pub color_type:  u8,
    pub stride:      usize,
    pub header_blob: Vec<u8>,
}

pub fn extract_raw_pixels(data: &[u8]) -> std::io::Result<(Vec<u8>, PngMeta)> {
    if !is_png(data) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "not a PNG"));
    }

    let mut pos = 8usize;
    let mut idat_compressed: Vec<u8> = Vec::new();
    let mut header_blob: Vec<u8> = data[0..8].to_vec();
    let mut width      = 0u32;
    let mut height     = 0u32;
    let mut bit_depth  = 0u8;
    let mut color_type = 0u8;

    while pos + 12 <= data.len() {
        let length     = u32::from_be_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &data[pos+4..pos+8];

        if pos + 8 + length > data.len() { break; }

        if chunk_type == b"IHDR" {
            let chunk_data = &data[pos+8..pos+8+length];
            width      = u32::from_be_bytes(chunk_data[0..4].try_into().unwrap());
            height     = u32::from_be_bytes(chunk_data[4..8].try_into().unwrap());
            bit_depth  = chunk_data[8];
            color_type = chunk_data[9];
            header_blob.extend_from_slice(&data[pos..pos+12+length]);
        } else if chunk_type == b"IDAT" {
            let chunk_data = &data[pos+8..pos+8+length];
            idat_compressed.extend_from_slice(chunk_data);
            // IDAT is NOT stored in header_blob — we rebuild it on repack
        } else if chunk_type == b"IEND" {
            header_blob.extend_from_slice(&data[pos..pos+12+length]);
        } else {
            // Ancillary chunks: gAMA, tEXt, bKGD, etc. — preserve verbatim
            header_blob.extend_from_slice(&data[pos..pos+12+length]);
        }

        pos += 12 + length;
    }

    if idat_compressed.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "PNG: no IDAT chunks found"));
    }

    // Inflate IDAT (zlib stream)
    let mut decoder = ZlibDecoder::new(&idat_compressed[..]);
    let mut filtered_pixels: Vec<u8> = Vec::new();
    decoder.read_to_end(&mut filtered_pixels)?;

    let stride = stride_from_color(color_type, bit_depth);
    let row_bytes = width as usize * stride;

    // Undo PNG per-scanline filters → truly raw pixels
    let mut raw = vec![0u8; height as usize * row_bytes];
    let mut prev_row = vec![0u8; row_bytes];

    for row in 0..height as usize {
        let src_off = row * (row_bytes + 1); // +1 for the filter byte
        let dst_off = row * row_bytes;

        if src_off >= filtered_pixels.len() { break; }
        let filter_byte = filtered_pixels[src_off];

        let row_end = (src_off + 1 + row_bytes).min(filtered_pixels.len());
        let src = &filtered_pixels[src_off+1..row_end];
        let actual_cols = src.len().min(row_bytes);

        for i in 0..actual_cols {
            let a = if i >= stride { raw[dst_off + i - stride] } else { 0 };
            let b = prev_row[i];
            let c = if i >= stride { prev_row[i - stride] } else { 0 };

            raw[dst_off + i] = match filter_byte {
                0 => src[i],
                1 => src[i].wrapping_add(a),
                2 => src[i].wrapping_add(b),
                3 => src[i].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => src[i].wrapping_add(paeth(a, b, c)),
                _ => src[i],
            };
        }
        prev_row.copy_from_slice(&raw[dst_off..dst_off+row_bytes]);
    }

    let meta = PngMeta { width, height, bit_depth, color_type, stride, header_blob };
    Ok((raw, meta))
}

/// Reconstruct PngMeta from a stored header_blob (used during decompression).
pub fn meta_from_blob(header_blob: &[u8]) -> std::io::Result<PngMeta> {
    if header_blob.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "PNG meta: blob too short"));
    }
    let mut pos = 8usize;
    while pos + 12 <= header_blob.len() {
        let length     = u32::from_be_bytes(header_blob[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &header_blob[pos+4..pos+8];
        if pos + 8 + length > header_blob.len() { break; }

        if chunk_type == b"IHDR" && length >= 10 {
            let chunk_data = &header_blob[pos+8..pos+8+length];
            let width      = u32::from_be_bytes(chunk_data[0..4].try_into().unwrap());
            let height     = u32::from_be_bytes(chunk_data[4..8].try_into().unwrap());
            let bit_depth  = chunk_data[8];
            let color_type = chunk_data[9];
            let stride     = stride_from_color(color_type, bit_depth);
            return Ok(PngMeta {
                width, height, bit_depth, color_type, stride,
                header_blob: header_blob.to_vec(),
            });
        }
        pos += 12 + length;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData, "PNG meta: IHDR not found in blob"))
}

pub fn repack_png(raw_pixels: &[u8], meta: &PngMeta) -> std::io::Result<Vec<u8>> {
    let row_bytes = meta.width as usize * meta.stride;

    // Apply Sub filter (type 1) per scanline — simple, effective for
    // horizontally correlated pixel data
    let mut filtered: Vec<u8> = Vec::with_capacity(raw_pixels.len() + meta.height as usize);
    for row in 0..meta.height as usize {
        filtered.push(1u8); // Sub filter
        let src_start = row * row_bytes;
        let src_end   = (src_start + row_bytes).min(raw_pixels.len());
        let src       = &raw_pixels[src_start..src_end];
        for i in 0..src.len() {
            let a = if i >= meta.stride { src[i - meta.stride] } else { 0 };
            filtered.push(src[i].wrapping_sub(a));
        }
    }

    // Re-compress with zlib
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&filtered)?;
    let compressed = encoder.finish()?;

    // Find IEND position in header_blob so we can insert IDAT before it
    let iend_pos = header_blob_find_iend(&meta.header_blob);

    let mut out = Vec::new();
    out.extend_from_slice(&meta.header_blob[..iend_pos]);

    // Write IDAT chunk
    let idat_len = compressed.len() as u32;
    out.extend_from_slice(&idat_len.to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(&compressed);
    out.extend_from_slice(&crc32_ieee(b"IDAT", &compressed).to_be_bytes());

    // Write remaining header_blob (IEND and anything after)
    out.extend_from_slice(&meta.header_blob[iend_pos..]);
    Ok(out)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn stride_from_color(color_type: u8, bit_depth: u8) -> usize {
    let samples: usize = match color_type {
        0 => 1, // grayscale
        2 => 3, // RGB
        3 => 1, // indexed
        4 => 2, // grayscale + alpha
        6 => 4, // RGBA
        _ => 1,
    };
    samples * (bit_depth as usize / 8).max(1)
}

fn header_blob_find_iend(header_blob: &[u8]) -> usize {
    // Walk chunks to find the byte offset of the IEND length field
    if header_blob.len() < 8 { return header_blob.len(); }
    let mut pos = 8usize;
    while pos + 12 <= header_blob.len() {
        let length     = u32::from_be_bytes(header_blob[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &header_blob[pos+4..pos+8];
        if chunk_type == b"IEND" { return pos; }
        pos += 12 + length;
    }
    header_blob.len()
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (a as i16, b as i16, c as i16);
    let p  = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc { a as u8 }
    else if pb <= pc { b as u8 }
    else { c as u8 }
}

fn crc32_ieee(chunk_type: &[u8], data: &[u8]) -> u32 {
    let table: [u32; 256] = {
        let mut t = [0u32; 256];
        for n in 0..256usize {
            let mut c = n as u32;
            for _ in 0..8 { c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 }; }
            t[n] = c;
        }
        t
    };
    let mut crc = 0xFFFFFFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
  }
