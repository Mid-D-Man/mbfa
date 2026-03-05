// src/png_xform.rs
// PNG transform for MBFA: inflate IDAT → fold filtered bytes → re-deflate on decode.
// We deliberately do NOT undo PNG's per-scanline filters. They already produce
// near-zero residuals that MBFA's LZ engine handles well. Avoiding the
// unfilter/re-filter round-trip eliminates a class of bugs and simplifies repack.
//
// Roundtrip note: output PNG is pixel-identical to input but not byte-identical
// because the re-deflated IDAT uses different zlib settings than the original encoder.

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
    pub header_blob: Vec<u8>, // all chunks except IDAT, in original order
}

/// Inflate all IDAT chunks into a single filtered-scanline byte buffer.
/// The filter byte at the start of each scanline is preserved — we do NOT
/// undo PNG's per-scanline filters. MBFA folds this buffer directly.
pub fn extract_filtered_bytes(data: &[u8]) -> std::io::Result<(Vec<u8>, PngMeta)> {
    if !is_png(data) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "not a PNG"));
    }

    let mut pos              = 8usize;
    let mut idat_compressed  = Vec::new();
    let mut header_blob: Vec<u8> = data[0..8].to_vec(); // PNG magic
    let mut width      = 0u32;
    let mut height     = 0u32;
    let mut bit_depth  = 0u8;
    let mut color_type = 0u8;

    while pos + 12 <= data.len() {
        let chunk_len  = u32::from_be_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &data[pos+4..pos+8];
        let chunk_end  = pos + 12 + chunk_len;

        if chunk_end > data.len() { break; }

        if chunk_type == b"IHDR" {
            let d  = &data[pos+8..pos+8+chunk_len];
            width      = u32::from_be_bytes(d[0..4].try_into().unwrap());
            height     = u32::from_be_bytes(d[4..8].try_into().unwrap());
            bit_depth  = d[8];
            color_type = d[9];
            header_blob.extend_from_slice(&data[pos..chunk_end]);
        } else if chunk_type == b"IDAT" {
            idat_compressed.extend_from_slice(&data[pos+8..pos+8+chunk_len]);
            // IDAT intentionally excluded from header_blob — rebuilt on repack
        } else {
            // IEND + all ancillary chunks (PLTE, gAMA, tEXt, etc.) preserved verbatim
            header_blob.extend_from_slice(&data[pos..chunk_end]);
        }

        pos = chunk_end;
    }

    if idat_compressed.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "PNG: no IDAT chunks found"));
    }

    // Inflate the zlib IDAT stream.
    // Result is raw PNG-filtered scanline data: [filter_byte, pixels...] per row.
    let mut decoder      = ZlibDecoder::new(&idat_compressed[..]);
    let mut filtered_buf = Vec::new();
    decoder.read_to_end(&mut filtered_buf)?;

    let stride = stride_from_color(color_type, bit_depth);
    let meta   = PngMeta { width, height, bit_depth, color_type, stride, header_blob };

    println!(
        "PNG extract: {}x{} color_type={} stride={} → {} filtered bytes (inflated from {} compressed)",
        width, height, color_type, stride, filtered_buf.len(), idat_compressed.len()
    );

    Ok((filtered_buf, meta))
}

/// Reconstruct PngMeta from a stored header_blob.
/// Used during decompression — we need width/height/color_type to know stride.
pub fn meta_from_blob(header_blob: &[u8]) -> std::io::Result<PngMeta> {
    if header_blob.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "PNG meta: blob too short"));
    }
    let mut pos = 8usize;
    while pos + 12 <= header_blob.len() {
        let chunk_len  = u32::from_be_bytes(header_blob[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &header_blob[pos+4..pos+8];
        let chunk_end  = pos + 12 + chunk_len;
        if chunk_end > header_blob.len() { break; }

        if chunk_type == b"IHDR" && chunk_len >= 10 {
            let d          = &header_blob[pos+8..pos+8+chunk_len];
            let width      = u32::from_be_bytes(d[0..4].try_into().unwrap());
            let height     = u32::from_be_bytes(d[4..8].try_into().unwrap());
            let bit_depth  = d[8];
            let color_type = d[9];
            let stride     = stride_from_color(color_type, bit_depth);
            return Ok(PngMeta {
                width, height, bit_depth, color_type, stride,
                header_blob: header_blob.to_vec(),
            });
        }
        pos = chunk_end;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData, "PNG meta: IHDR not found in blob"))
}

/// Re-deflate the filtered bytes and rebuild the PNG container.
/// Output is pixel-identical to the original but not byte-identical
/// (different zlib compression parameters than original encoder).
pub fn repack_png(filtered_bytes: &[u8], meta: &PngMeta) -> std::io::Result<Vec<u8>> {
    // Re-deflate the filtered scanline bytes
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(filtered_bytes)?;
    let compressed = encoder.finish()?;

    // Find where IEND starts in header_blob so we insert IDAT before it
    let iend_pos = find_iend_pos(&meta.header_blob);

    let mut out = Vec::new();
    out.extend_from_slice(&meta.header_blob[..iend_pos]);

    // Write single IDAT chunk
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(&compressed);
    out.extend_from_slice(&crc32_png(b"IDAT", &compressed).to_be_bytes());

    // IEND (and anything after it, though there shouldn't be anything valid)
    out.extend_from_slice(&meta.header_blob[iend_pos..]);

    println!(
        "PNG repack: {} filtered bytes → {} compressed → {} PNG bytes",
        filtered_bytes.len(), compressed.len(), out.len()
    );

    Ok(out)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn stride_from_color(color_type: u8, bit_depth: u8) -> usize {
    let samples: usize = match color_type {
        0 => 1, // grayscale
        2 => 3, // RGB
        3 => 1, // indexed — one palette index per pixel
        4 => 2, // grayscale + alpha
        6 => 4, // RGBA
        _ => 1,
    };
    samples * (bit_depth as usize / 8).max(1)
}

fn find_iend_pos(header_blob: &[u8]) -> usize {
    if header_blob.len() < 8 { return header_blob.len(); }
    let mut pos = 8usize;
    while pos + 12 <= header_blob.len() {
        let chunk_len  = u32::from_be_bytes(header_blob[pos..pos+4].try_into().unwrap()) as usize;
        let chunk_type = &header_blob[pos+4..pos+8];
        if chunk_type == b"IEND" { return pos; }
        pos += 12 + chunk_len;
    }
    header_blob.len()
}

/// PNG CRC covers the chunk type bytes concatenated with the chunk data bytes.
fn crc32_png(chunk_type: &[u8], data: &[u8]) -> u32 {
    // Build table once inline — avoids adding a crc32 dependency
    let table: [u32; 256] = {
        let mut t = [0u32; 256];
        for n in 0..256usize {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
            }
            t[n] = c;
        }
        t
    };
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
        }
