# archive

The multi-file archive container, the CLI that drives it, RAM-aware
block sizing, and the standalone offset-profiling binary. Gated behind
the `"archive"` feature (except `main.rs`, which needs it to build at
all).

## Modules

### archive.rs

**What it does:** Creates, extracts, and lists MBFA archives — an
18-byte header (magic, version, block count, index offset) followed by
block data and an appended index table. Files are planned into blocks up
to `platform::auto_chunk_size()` bytes, each block compressed
independently via `lib::compress`.

**Decisions:**
- Files whose per-file Shannon entropy exceeds `FILE_ENTROPY_THRESHOLD`
  (7.5 — deliberately a little below `lib.rs`'s block-level threshold of
  7.8, to also catch borderline cases like MP3, compressed PDF, and
  partially-random binary) are isolated into their own single-file
  blocks before planning. This keeps pre-compressed content (PNG, JPEG,
  zip, etc.) from diluting the LZ dictionary of neighboring compressible
  files, and avoids running the full adaptive scan on data that can
  never shrink — `compress()`'s own entropy gate still passes it through
  cleanly regardless.

### archive_io.rs

**What it does:** Serializes and deserializes the archive's index table
(the per-file metadata written after all block data).

### platform.rs

**What it does:** Picks the target block size for archive operations,
scaling with available system RAM (`available / 256`, clamped to
[1 MB, 8 MB]) so low-memory systems don't stage unnecessarily large
blocks, while capping at 8 MB so individual `compress()` calls stay
within reasonable wall-clock time.

**Benchmarks:** At the 8 MB ceiling, prose-like content compresses in
roughly 2-8 seconds per block (0.1-0.4 MB/s). Highly repetitive data is
much faster; incompressible data passes through in milliseconds
regardless of block size (see `compress()`'s entropy gate in
[core.md](core.md)).

### main.rs

**What it does:** The MBFA CLI — compress/decompress single files and
create/extract/list archives, dispatching to `lib::compress`/`decompress`
or `archive::*` depending on the subcommand.

### profile_offsets.rs

**What it does:** A standalone diagnostic binary that measures how often
backref offsets land in one of the encoder's ring-buffer slots, as a way
to sanity-check the ring buffer's real hit rate against real compressed
output.

## Fixes and Problems

### profile_offsets.rs

Updated to profile the top 4 ring slots (previously top 3) to match
`MAX_RING_SLOTS = 4` once the encoder's ring buffer grew to four slots.
