// src/archive.rs
//! MBFA archive — create, extract, list.
//!
//! Archive header layout (18 bytes):
//!   0..5:   magic [0x4D,0x42,0x46,0x41,0xAA]
//!   5:      version u8 = 1
//!   6..10:  block_count u32 LE
//!   10..18: index_offset u64 LE  (written last via seek)
//!
//! Block data follows immediately after the header.
//! Index table is appended after all block data.
//! index_offset is patched into the header once all blocks are written.
//!
//! Incompressible files (per-file entropy > FILE_ENTROPY_THRESHOLD) are
//! isolated into their own single-file blocks before planning. This prevents
//! pre-compressed content (PNG, JPEG, zip, etc.) from polluting the LZ
//! dictionary of neighbouring compressible files, and avoids running the
//! full adaptive scan on data that can never shrink. compress() will detect
//! the incompressible block via its own entropy gate and passthrough cleanly.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use rayon::prelude::*;
use crate::platform::auto_chunk_size;

pub const ARCHIVE_MAGIC: [u8; 5] = [0x4D, 0x42, 0x46, 0x41, 0xAA];
const ARCHIVE_VERSION:    u8     = 1;
const HEADER_SIZE:        usize  = 18;
const INDEX_OFFSET_FIELD: u64    = 10;

/// Per-file Shannon entropy threshold above which a file is treated as
/// incompressible and isolated into its own block. Set slightly below the
/// block-level threshold in lib.rs (7.8) to catch borderline cases such as
/// MP3, compressed PDF, and partially-random binary before they dilute
/// adjacent compressible content.
const FILE_ENTROPY_THRESHOLD: f64 = 7.5;

// ── Public data types ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct FileEntry {
    pub path:            String,
    pub original_size:   u64,
    pub offset_in_block: u64,
    pub is_split:        bool,
    pub chunk_index:     u32,
    pub total_chunks:    u32,
}

#[derive(Debug)]
pub struct BlockEntry {
    pub block_offset:     u64,
    pub compressed_size:  u32,
    pub original_size:    u64,
    pub files:            Vec<FileEntry>,
}

// ── Similarity grouping ───────────────────────────────────────────────────────

pub fn similarity_group(ext: &str) -> u8 {
    match ext.to_lowercase().as_str() {
        "rs" | "c" | "h" | "cpp" | "cc" | "cs" | "go" | "py" | "js" | "ts"
        | "java" | "kt" | "swift" | "rb" | "php" => 0,

        "txt" | "md" | "html" | "htm" | "xml" | "json" | "yaml" | "yml"
        | "toml" | "csv" | "log" | "ini" | "cfg" => 1,

        "bin" | "dat" | "obj" | "o" | "a" | "lib" | "wasm" | "exe"
        | "dll" | "so" | "dylib" => 2,

        "gz" | "zip" | "zst" | "xz" | "br" | "png" | "jpg" | "jpeg"
        | "gif" | "mp3" | "mp4" | "mkv" | "pdf" => 3,

        _ => 4,
    }
}

pub const GROUP_NAMES: [&str; 5] = [
    "Source", "Markup/Data", "Binary", "Compressed/Media", "Other",
];

// ── Per-file entropy sampling ─────────────────────────────────────────────────

/// Reads up to 8 KB from the start of `path` and returns the Shannon entropy
/// in bits per byte. Returns 0.0 if the file is empty or unreadable (callers
/// treat unreadable files as compressible to avoid silently dropping them).
fn sample_file_entropy(path: &Path) -> io::Result<f64> {
    const SAMPLE: usize = 8192;
    let mut buf = vec![0u8; SAMPLE];
    let mut f   = File::open(path)?;
    let n       = f.read(&mut buf)?;
    if n == 0 { return Ok(0.0); }
    let mut freq = [0u32; 256];
    for &b in &buf[..n] { freq[b as usize] += 1; }
    let total   = n as f64;
    let entropy = freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / total; -p * p.log2() })
        .sum();
    Ok(entropy)
}

// ── File collection ───────────────────────────────────────────────────────────

struct RawFile {
    rel_path:          String,
    abs_path:          PathBuf,
    size:              u64,
    group:             u8,
    /// True when the file's entropy sample exceeded FILE_ENTROPY_THRESHOLD.
    /// Such files are routed to isolated single-file blocks by plan_blocks().
    is_incompressible: bool,
}

fn collect_files(input_dir: &Path) -> io::Result<Vec<RawFile>> {
    let mut files          = Vec::new();
    let mut n_incompressible = 0usize;

    for entry in WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let abs_path = entry.path().to_path_buf();
        let rel_path = abs_path
            .strip_prefix(input_dir)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            .to_string_lossy()
            .replace('\\', "/");

        let size  = entry.metadata()?.len();
        let ext   = abs_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let group = similarity_group(ext);

        // Sample the file's entropy. If reading fails for any reason (e.g.
        // a symlink or permission issue) default to compressible so the file
        // is still included in the archive rather than silently dropped.
        let is_incompressible = sample_file_entropy(&abs_path)
            .map(|e| e > FILE_ENTROPY_THRESHOLD)
            .unwrap_or(false);

        if is_incompressible { n_incompressible += 1; }

        files.push(RawFile { rel_path, abs_path, size, group, is_incompressible });
    }

    files.sort_by(|a, b| a.group.cmp(&b.group).then(b.size.cmp(&a.size)));

    println!(
        "Collected {} file(s) — {} compressible, {} detected as incompressible (will be stored)",
        files.len(),
        files.len() - n_incompressible,
        n_incompressible,
    );

    Ok(files)
}

// ── Block planning ────────────────────────────────────────────────────────────

struct PlannedChunk {
    rel_path:     String,
    abs_path:     PathBuf,
    file_size:    u64,
    is_split:     bool,
    chunk_index:  u32,
    total_chunks: u32,
}

fn plan_blocks(files: Vec<RawFile>, chunk_size: usize) -> Vec<Vec<PlannedChunk>> {
    let mut blocks:        Vec<Vec<PlannedChunk>> = Vec::new();
    let mut current_block: Vec<PlannedChunk>      = Vec::new();
    let mut current_size:  usize                  = 0;

    for file in files {
        // ── Incompressible files: always isolate into their own block(s). ──────
        // This prevents pre-compressed content from polluting the LZ dictionary
        // of adjacent compressible files. We still respect chunk_size for
        // memory safety — each chunk passhthroughs individually in compress().
        if file.is_incompressible {
            if !current_block.is_empty() {
                blocks.push(std::mem::take(&mut current_block));
                current_size = 0;
            }

            if file.size as usize > chunk_size {
                let total_chunks =
                    ((file.size as usize + chunk_size - 1) / chunk_size) as u32;
                for chunk_idx in 0..total_chunks {
                    blocks.push(vec![PlannedChunk {
                        rel_path:     file.rel_path.clone(),
                        abs_path:     file.abs_path.clone(),
                        file_size:    file.size,
                        is_split:     true,
                        chunk_index:  chunk_idx,
                        total_chunks,
                    }]);
                }
            } else {
                blocks.push(vec![PlannedChunk {
                    rel_path:     file.rel_path.clone(),
                    abs_path:     file.abs_path.clone(),
                    file_size:    file.size,
                    is_split:     false,
                    chunk_index:  0,
                    total_chunks: 1,
                }]);
            }
            continue;
        }

        // ── Compressible oversized files: split into per-chunk blocks. ─────────
        if file.size as usize > chunk_size {
            if !current_block.is_empty() {
                blocks.push(std::mem::take(&mut current_block));
                current_size = 0;
            }
            let total_chunks =
                ((file.size as usize + chunk_size - 1) / chunk_size) as u32;
            for chunk_idx in 0..total_chunks {
                blocks.push(vec![PlannedChunk {
                    rel_path:     file.rel_path.clone(),
                    abs_path:     file.abs_path.clone(),
                    file_size:    file.size,
                    is_split:     true,
                    chunk_index:  chunk_idx,
                    total_chunks,
                }]);
            }
            continue;
        }

        // ── Compressible small file: group with neighbours. ────────────────────
        if current_size + file.size as usize > chunk_size && !current_block.is_empty() {
            blocks.push(std::mem::take(&mut current_block));
            current_size = 0;
        }
        current_size += file.size as usize;
        current_block.push(PlannedChunk {
            rel_path:     file.rel_path,
            abs_path:     file.abs_path,
            file_size:    file.size,
            is_split:     false,
            chunk_index:  0,
            total_chunks: 1,
        });
    }

    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    blocks
}

// ── Internal block data ───────────────────────────────────────────────────────

/// Raw (uncompressed) block data, ready to be passed to crate::compress.
struct RawBlock {
    stream:       Vec<u8>,
    file_entries: Vec<FileEntry>,
    orig_size:    u64,
}

// ── Archive creation ──────────────────────────────────────────────────────────

pub fn create_archive(input_dir: &Path, output_path: &Path) -> io::Result<()> {
    let chunk_size = auto_chunk_size();
    let files      = collect_files(input_dir)?;

    println!("Archiving {} file(s) from {:?}", files.len(), input_dir);

    let blocks = plan_blocks(files, chunk_size);
    println!("Planned {} block(s)", blocks.len());

    // ── Phase 1: read all blocks sequentially (IO-bound) ─────────────────────
    let mut raw_blocks: Vec<RawBlock> = Vec::with_capacity(blocks.len());

    for block_chunks in &blocks {
        let mut stream:       Vec<u8>        = Vec::new();
        let mut file_entries: Vec<FileEntry> = Vec::new();
        let mut orig_size:    u64            = 0;

        for chunk in block_chunks {
            let offset_in_block = stream.len() as u64;
            let chunk_data = if chunk.is_split {
                read_file_chunk(
                    &chunk.abs_path,
                    chunk.chunk_index,
                    chunk_size,
                    chunk.file_size,
                )?
            } else {
                fs::read(&chunk.abs_path)?
            };
            orig_size += chunk_data.len() as u64;
            stream.extend_from_slice(&chunk_data);
            file_entries.push(FileEntry {
                path:            chunk.rel_path.clone(),
                original_size:   chunk.file_size,
                offset_in_block,
                is_split:        chunk.is_split,
                chunk_index:     chunk.chunk_index,
                total_chunks:    chunk.total_chunks,
            });
        }

        raw_blocks.push(RawBlock { stream, file_entries, orig_size });
    }

    // ── Phase 2: compress all blocks in parallel (CPU-bound) ─────────────────
    // Each block is independent — crate::compress takes &[u8] and returns
    // an owned Vec<u8>. Safe for rayon with no shared mutable state.
    //
    // NOTE: compress() internally calls scan_adaptive which emits println!.
    // With multiple blocks in flight those lines will interleave in stdout.
    // This is a cosmetic issue only — correctness is unaffected.
    let compressed_results: Vec<io::Result<Vec<u8>>> = raw_blocks
        .par_iter()
        .map(|rb| crate::compress(&rb.stream, 8))
        .collect();

    // ── Phase 3: write sequentially (IO-bound, needs sequential offsets) ──────
    let mut out = File::create(output_path)?;

    out.write_all(&ARCHIVE_MAGIC)?;
    out.write_all(&[ARCHIVE_VERSION])?;
    out.write_all(&(raw_blocks.len() as u32).to_le_bytes())?;
    out.write_all(&0u64.to_le_bytes())?; // index_offset placeholder

    let mut block_entries: Vec<BlockEntry> = Vec::with_capacity(raw_blocks.len());

    for (idx, (rb, comp_result)) in raw_blocks
        .into_iter()
        .zip(compressed_results.into_iter())
        .enumerate()
    {
        print!("  Block {:>4}/{} ... ", idx + 1, blocks.len());
        let _ = std::io::stdout().flush();

        let compressed   = comp_result?;
        let block_offset = out.seek(SeekFrom::Current(0))?;
        let comp_len     = compressed.len();

        out.write_all(&compressed)?;

        println!(
            "{} bytes → {} bytes ({:.1}%)",
            rb.orig_size,
            comp_len,
            comp_len as f64 / rb.orig_size.max(1) as f64 * 100.0
        );

        block_entries.push(BlockEntry {
            block_offset,
            compressed_size: comp_len as u32,
            original_size:   rb.orig_size,
            files:           rb.file_entries,
        });
    }

    // Write index and patch header
    let index_offset = out.seek(SeekFrom::Current(0))?;
    crate::archive_io::write_index(&mut out, &block_entries)?;
    out.seek(SeekFrom::Start(INDEX_OFFSET_FIELD))?;
    out.write_all(&index_offset.to_le_bytes())?;

    println!(
        "Done. {} block(s), index @ offset {} — archive: {:?}",
        block_entries.len(),
        index_offset,
        output_path
    );

    Ok(())
}

// ── Archive extraction ────────────────────────────────────────────────────────

pub fn extract_archive(
    input_path:    &Path,
    output_dir:    &Path,
    specific_file: Option<&str>,
) -> io::Result<()> {
    let (block_entries, mut f) = load_index(input_path)?;
    fs::create_dir_all(output_dir)?;

    if let Some(target) = specific_file {
        extract_one_file(&block_entries, &mut f, target, output_dir)?;
    } else {
        extract_all(&block_entries, &mut f, output_dir)?;
    }

    Ok(())
}

fn extract_all(
    block_entries: &[BlockEntry],
    f:             &mut File,
    output_dir:    &Path,
) -> io::Result<()> {
    for (idx, block) in block_entries.iter().enumerate() {
        print!("  Block {:>4}/{} ... ", idx + 1, block_entries.len());
        let _ = std::io::stdout().flush();

        let decompressed = read_and_decompress_block(f, block)?;

        for file in &block.files {
            let out_path = output_dir.join(&file.path);
            if let Some(p) = out_path.parent() { fs::create_dir_all(p)?; }

            let chunk_data = slice_file_from_block(&decompressed, file, block);
            write_file_chunk(&out_path, chunk_data, file.is_split, file.chunk_index)?;
        }

        println!("ok ({} file(s))", block.files.len());
    }

    Ok(())
}

fn extract_one_file(
    block_entries: &[BlockEntry],
    f:             &mut File,
    target:        &str,
    output_dir:    &Path,
) -> io::Result<()> {
    let mut hits: Vec<(&BlockEntry, &FileEntry)> = block_entries.iter()
        .flat_map(|b| b.files.iter().map(move |fe| (b, fe)))
        .filter(|(_, fe)| fe.path == target)
        .collect();

    if hits.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("'{}' not found in archive", target),
        ));
    }

    hits.sort_by_key(|(_, fe)| fe.chunk_index);

    let out_path = output_dir.join(target);
    if let Some(p) = out_path.parent() { fs::create_dir_all(p)?; }

    for (block, file) in hits {
        let decompressed = read_and_decompress_block(f, block)?;
        let chunk_data   = slice_file_from_block(&decompressed, file, block);
        write_file_chunk(&out_path, chunk_data, file.is_split, file.chunk_index)?;
    }

    println!("Extracted: {}", target);
    Ok(())
}

// ── Archive listing ───────────────────────────────────────────────────────────

pub fn list_archive(input_path: &Path) -> io::Result<()> {
    let (block_entries, _) = load_index(input_path)?;

    for (i, block) in block_entries.iter().enumerate() {
        let group_name = block.files.first()
            .and_then(|fe| Path::new(&fe.path).extension())
            .and_then(|e| e.to_str())
            .map(|ext| GROUP_NAMES[similarity_group(ext) as usize])
            .unwrap_or("Other");

        println!(
            "Block {:>4}  [{:<16}]  {:>4} file(s)  orig: {:>10}  compressed: {:>10}",
            i,
            group_name,
            block.files.len(),
            fmt_bytes(block.original_size),
            fmt_bytes(block.compressed_size as u64),
        );

        for fe in &block.files {
            if fe.is_split {
                println!(
                    "    {} [chunk {}/{}]  total {}",
                    fe.path,
                    fe.chunk_index + 1,
                    fe.total_chunks,
                    fmt_bytes(fe.original_size),
                );
            } else {
                println!("    {}  {}", fe.path, fmt_bytes(fe.original_size));
            }
        }
    }

    Ok(())
}

// ── Detection ─────────────────────────────────────────────────────────────────

pub fn is_archive(data: &[u8]) -> bool {
    data.len() >= 5 && data[0..5] == ARCHIVE_MAGIC
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn load_index(input_path: &Path) -> io::Result<(Vec<BlockEntry>, File)> {
    let mut f = File::open(input_path)?;

    let mut header = [0u8; HEADER_SIZE];
    f.read_exact(&mut header)?;

    if header[0..5] != ARCHIVE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a valid MBFA archive",
        ));
    }

    let block_count  = u32::from_le_bytes(header[6..10].try_into().unwrap()) as usize;
    let index_offset = u64::from_le_bytes(header[10..18].try_into().unwrap());

    f.seek(SeekFrom::Start(index_offset))?;
    let mut index_data = Vec::new();
    f.read_to_end(&mut index_data)?;

    let block_entries = crate::archive_io::read_index(&index_data, block_count)?;
    Ok((block_entries, f))
}

fn read_and_decompress_block(f: &mut File, block: &BlockEntry) -> io::Result<Vec<u8>> {
    f.seek(SeekFrom::Start(block.block_offset))?;
    let mut compressed = vec![0u8; block.compressed_size as usize];
    f.read_exact(&mut compressed)?;
    crate::decompress(&compressed)
}

fn slice_file_from_block<'a>(
    decompressed: &'a [u8],
    file:         &FileEntry,
    block:        &BlockEntry,
) -> &'a [u8] {
    let start = file.offset_in_block as usize;
    let end   = block.files.iter()
        .filter(|f| f.offset_in_block > file.offset_in_block)
        .map(|f| f.offset_in_block as usize)
        .min()
        .unwrap_or(decompressed.len());
    &decompressed[start..end.min(decompressed.len())]
}

fn write_file_chunk(
    path:        &Path,
    data:        &[u8],
    is_split:    bool,
    chunk_index: u32,
) -> io::Result<()> {
    if is_split && chunk_index > 0 {
        let mut f = OpenOptions::new().append(true).open(path)?;
        f.write_all(data)
    } else {
        fs::write(path, data)
    }
}

fn read_file_chunk(
    path:        &Path,
    chunk_index: u32,
    chunk_size:  usize,
    file_size:   u64,
) -> io::Result<Vec<u8>> {
    let mut f     = File::open(path)?;
    let start     = chunk_index as u64 * chunk_size as u64;
    let remaining = file_size.saturating_sub(start) as usize;
    let to_read   = remaining.min(chunk_size);

    f.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; to_read];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn fmt_bytes(n: u64) -> String {
    if      n >= 1_073_741_824 { format!("{:.1}GB", n as f64 / 1_073_741_824.0) }
    else if n >= 1_048_576     { format!("{:.1}MB", n as f64 / 1_048_576.0) }
    else if n >= 1_024         { format!("{:.1}KB", n as f64 / 1_024.0) }
    else                       { format!("{}B",     n) }
}
