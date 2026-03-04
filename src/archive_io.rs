// src/archive_io.rs
//! Index serialization and deserialization for MBFA archives.
//! The index is written at the end of the archive file.
//! Format per block:
//!   block_offset:     u64 LE
//!   compressed_size:  u32 LE
//!   original_size:    u64 LE
//!   file_count:       u32 LE
//!   For each file:
//!     path_len:           u16 LE
//!     path:               UTF-8 bytes
//!     original_size:      u64 LE
//!     offset_in_block:    u64 LE
//!     is_split:           u8  (0 or 1)
//!     chunk_index:        u32 LE
//!     total_chunks:       u32 LE

use std::io::{self, Write};
use crate::archive::{BlockEntry, FileEntry};

pub fn write_index<W: Write>(out: &mut W, blocks: &[BlockEntry]) -> io::Result<()> {
    for block in blocks {
        out.write_all(&block.block_offset.to_le_bytes())?;
        out.write_all(&block.compressed_size.to_le_bytes())?;
        out.write_all(&block.original_size.to_le_bytes())?;
        out.write_all(&(block.files.len() as u32).to_le_bytes())?;

        for file in &block.files {
            let path_bytes = file.path.as_bytes();
            out.write_all(&(path_bytes.len() as u16).to_le_bytes())?;
            out.write_all(path_bytes)?;
            out.write_all(&file.original_size.to_le_bytes())?;
            out.write_all(&file.offset_in_block.to_le_bytes())?;
            out.write_all(&[file.is_split as u8])?;
            out.write_all(&file.chunk_index.to_le_bytes())?;
            out.write_all(&file.total_chunks.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn read_index(data: &[u8], block_count: usize) -> io::Result<Vec<BlockEntry>> {
    let mut cur = 0usize;
    let mut blocks = Vec::with_capacity(block_count);

    for b in 0..block_count {
        if cur + 24 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("index: block {} header truncated", b),
            ));
        }

        let block_offset    = u64::from_le_bytes(data[cur..cur+8].try_into().unwrap()); cur += 8;
        let compressed_size = u32::from_le_bytes(data[cur..cur+4].try_into().unwrap()); cur += 4;
        let original_size   = u64::from_le_bytes(data[cur..cur+8].try_into().unwrap()); cur += 8;
        let file_count      = u32::from_le_bytes(data[cur..cur+4].try_into().unwrap()) as usize; cur += 4;

        let mut files = Vec::with_capacity(file_count);

        for fi in 0..file_count {
            if cur + 2 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("index: block {} file {} path_len truncated", b, fi),
                ));
            }
            let path_len = u16::from_le_bytes(data[cur..cur+2].try_into().unwrap()) as usize; cur += 2;

            if cur + path_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("index: block {} file {} path truncated", b, fi),
                ));
            }
            let path = String::from_utf8(data[cur..cur+path_len].to_vec())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            cur += path_len;

            // 8 + 8 + 1 + 4 + 4 = 25 bytes of fixed metadata
            if cur + 25 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("index: block {} file {} metadata truncated", b, fi),
                ));
            }
            let file_orig_size   = u64::from_le_bytes(data[cur..cur+8].try_into().unwrap()); cur += 8;
            let offset_in_block  = u64::from_le_bytes(data[cur..cur+8].try_into().unwrap()); cur += 8;
            let is_split         = data[cur] != 0; cur += 1;
            let chunk_index      = u32::from_le_bytes(data[cur..cur+4].try_into().unwrap()); cur += 4;
            let total_chunks     = u32::from_le_bytes(data[cur..cur+4].try_into().unwrap()); cur += 4;

            files.push(FileEntry {
                path,
                original_size: file_orig_size,
                offset_in_block,
                is_split,
                chunk_index,
                total_chunks,
            });
        }

        blocks.push(BlockEntry { block_offset, compressed_size, original_size, files });
    }

    Ok(blocks)
}
