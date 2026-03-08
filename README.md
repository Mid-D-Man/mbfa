# MBFA — MidMans Bit Folding Algorithm

**MidManStudio** | Research Compression Algorithm | Active Development

---

> ⚠️ **Development Status:** This algorithm is under active research and development. Benchmark results, file formats, and internal behaviour are subject to change between versions. The only component guaranteed stable is the core instruction-set folding logic — the fixed opcode vocabulary (BACKREF / LIT / END) and the multi-fold instruction-chain architecture that defines MBFA as a distinct algorithm.

---

## What is MBFA?

MBFA is a novel multi-fold iterative compression algorithm. The core idea is architecturally distinct from existing compression algorithms — instead of performing a single compression pass, MBFA folds data through multiple passes where each pass produces a bitstream of instructions that reconstruct the previous pass.

This is not just "compress it twice." Each fold operates on a fundamentally different type of data than the last — fold 1 sees raw bytes, fold 2 sees an instruction stream, fold 3 sees an instruction stream about an instruction stream. The reduced alphabet and structural regularity of each successive layer is what the algorithm exploits.

---

## How It Works
```
Original bytes
    ↓  Pre-filter (optional): delta transform for structured binary (WAV, BMP)
    ↓  Fold 1: adaptive LZ scan → token stream → fixed-opcode bitstream
Fold 1 output
    ↓  Fold 2: token pair encoding with Cantor-paired operands (if large enough)
               OR entropy coding (Huffman, 5 variants) — whichever is smaller
Fold 2 output
    ↓  Fold 3+: LZ on whatever bytes came out of the previous fold
    ↓  ... until stopping condition fires
Final compressed seed + header
```

**Decompression** reads the header, runs the exact inverse number of passes, and reconstructs the original bytes exactly.

---

## Fixed Opcode Vocabulary

The opcode vocabulary is fixed and shared between encoder and decoder. **Never transmitted.** This is a core design decision — no per-fold table overhead. This vocabulary is the stable foundation of MBFA and will not change.

| Opcode | Bit Pattern | Total Bits | Meaning | Operands |
|--------|-------------|------------|---------|----------|
| BACKREF | `0` | 24 bits | Copy from output history | 15-bit offset + 8-bit length |
| LIT | `10` | 10 bits | Emit one literal byte | 8-bit byte value |
| END | `11` | 2 bits | End of stream | none |

BACKREF gets the 1-bit code because it becomes the dominant token on any repetitive data after fold 1. Offset and length field widths are **adaptive at runtime** — the values in the table above are defaults, not hard limits.

---

## Adaptive Window Sizing

The encoder scans in three phases:

**Phase A — Fingerprint:** Samples the first 8 KB of input to classify data:
- Highly repetitive (entropy < 2.0 bits/byte) → skip to Phase C
- Small file (< 32 KB) → predict minimum covering window
- Default → predict baseline parameters (ob=17, lb=8)

**Phase B — Single scan:** Runs one full-quality LZ scan at the predicted parameters. A ceiling saturation check (90% of field maximum) determines whether the window was sufficient. If yes — done in one scan. If the window was nearly exhausted — fall through to Phase C.

**Phase C — Discovery path:** Fast approximate scan to find actual maximum offset and length used, followed by a constrained full-quality re-scan at the discovered parameters. Baseline and discovery run in parallel (via Rayon) on inputs under 1 MB.

---

## Pair Encoding (Fold 2)

When fold 1 output exceeds 512 bytes and pairing is beneficial, fold 2 uses token pair encoding instead of raw LZ. Adjacent tokens are combined into typed pairs:

| Pair | Prefix | Meaning |
|------|--------|---------|
| LL | `000` | LIT + LIT |
| LB | `001` | LIT + BACKREF |
| BL | `010` | BACKREF + LIT |
| BB | `011` | BACKREF + BACKREF |
| SL | `100` | single LIT (odd token out) |
| SB | `101` | single BACKREF |
| END | `110` | stream terminator |

**BACKREF operands** are compressed using Cantor pairing — `(offset, length)` encoded as a single number `cantor(x,y) = (x+y)(x+y+1)/2 + y`. If the result fits in 16 bits it wins over raw encoding. Otherwise falls back to raw. No table, no transmission, fully reversible.

---

## Entropy Coding

When fold 1 output exceeds 400 bytes and pair encoding is not used, MBFA tries five Huffman entropy coding variants in parallel and picks the smallest result:

| Flag | Variant | Description |
|------|---------|-------------|
| v1 | Joint | Single lit/length Huffman + offset bucket Huffman |
| v2 | 2-context | Separate lit/length tables for after-literal vs after-backref positions |
| v3 | 8-context | Eight lit/length tables split by character category and position context |
| v4 | Slotted | v1 + recent-offset slot reuse (LRU cache of last 8 offsets) |
| v5 | Slotted 2-ctx | v2 + recent-offset slot reuse |

Offsets are bucket-coded (similar to DEFLATE distance codes) with variable extra bits. All tables are serialised into the output header — no shared state required between encoder and decoder.

If no entropy variant beats the raw token stream, entropy coding is skipped entirely.

---

## Delta Filters

Before folding, MBFA detects structured binary formats and applies a delta transform to convert smoothly-varying values into near-zero residuals that LZ can match at dramatically higher density:

| Filter | Stride | Applied to |
|--------|--------|------------|
| delta1 | 1 byte | Generic 8-bit binary |
| delta2 | 2 bytes | 16-bit mono PCM, 16-bit pixels |
| delta3 | 3 bytes | 24-bit RGB |
| delta4 | 4 bytes | 32-bit RGBA, stereo 16-bit PCM |

Format detection reads magic bytes and file headers (WAV `fmt ` chunk, BMP `bpp` field). The filter flag is stored in the header and reversed exactly on decompression.

---

## Archive System

MBFA includes a multi-file archive format with similarity-based block grouping:

- Files are sorted by extension group (source code, markup, binary, compressed/media) and size, then packed into blocks up to the platform chunk size
- Files detected as incompressible (per-file entropy > 7.5 bits/byte) are isolated into their own blocks before planning — this prevents pre-compressed content (PNG, JPEG, ZIP, MP3) from polluting the LZ dictionary of neighbouring compressible files
- Blocks are compressed in parallel via Rayon, written sequentially
- The index table is appended at the end of the archive and its offset is patched into the header after all blocks are written
- Chunk size scales automatically with available RAM (available / 256, clamped to 1 MB – 8 MB)

---

## Stopping Conditions

The encoder stops folding when any of these are true:

- Next fold output is not at least **1.5% smaller** than current (ratio ≥ 0.985)
- Output is at or below **64 bits** — too small for meaningful matching
- **Maximum 8 folds** reached

---

## File Format
```
Byte 0:          fold_count
Byte 1:          pair_flag        (1 = fold 2 used pair encoding)
Byte 2:          entropy_flag     (0 = none, 1–5 = entropy variant)
Byte 3:          filter_flag      (0 = none, 1–4 = delta stride)
Bytes 4..4+N:    offset_bits[0..N]   N = fold_count
Bytes 4+N..4+2N: length_bits[0..N]
Remaining:       compressed payload
```

`fold_count = 0` means passthrough — data was incompressible and is stored raw.

---

## Current Benchmark Results

> ⚠️ These results reflect the current development build and are subject to change as the algorithm evolves.

### Canterbury Corpus

| File | Size | MBFA | gzip | zstd |
|------|------|------|------|------|
| alice29.txt | 148 KB | **34.6%** | 35.8% | 37.5% |
| asyoulik.txt | 122 KB | **38.5%** | 39.1% | 40.2% |
| cp.html | 24 KB | 33.6% | **32.5%** | 34.4% |
| fields.c | 10 KB | 30.4% | **28.2%** | 30.3% |
| grammar.lsp | 4 KB | 36.4% | **33.5%** | 34.8% |
| kennedy.xls | 1 MB | 12.7% | 20.1% | **10.9%** |
| lcet10.txt | 416 KB | **30.9%** | 34.0% | 33.0% |
| plrabn12.txt | 470 KB | **37.7%** | 40.5% | 39.8% |
| ptt5 | 501 KB | **10.4%** | 11.0% | 10.6% |
| sum | 37 KB | 34.8% | **33.8%** | 35.0% |
| xargs.1 | 4 KB | 36.4% | **33.5%** | 34.8% |

### Custom Suite

| Dataset | MBFA | gzip | zstd | Notes |
|---------|------|------|------|-------|
| Repetitive 12 KB | **0.2%** | 0.8% | 0.3% | ✅ MBFA wins |
| Repetitive 2 MB | **0.0%** | 0.5% | 0.5% | ✅ MBFA wins |
| Prose 20 KB | **42.7%** | 41.6% | 42.9% | ✅ MBFA wins |
| Prose 100 KB | 35.8% | **36.4%** | 37.6% | ✅ competitive |
| War and Peace 3 MB | **31.0%** | 36.5% | 35.1% | ✅ MBFA wins |
| Source code 19 KB | 28.8% | **26.8%** | 28.5% | close |
| Source code 1 MB | **2.8%** | 21.0% | 3.1% | ✅ MBFA wins |
| JSON 100 KB | **6.2%** | 7.3% | 8.3% | ✅ MBFA wins |
| JSON 2 MB | **5.8%** | 7.4% | 8.3% | ✅ MBFA wins |
| Audio WAV 44 KB | **10.2%** | 10.9% | 10.2% | ✅ tie |
| Image BMP 29 KB | **2.2%** | 91.0% | 91.1% | ✅ MBFA dominant |
| PNG 117 KB | 100.0% | 100.1% | 100.0% | ✅ correct (already compressed) |
| Random 10 KB | 100.0% | 100.3% | 100.1% | ✅ correct (incompressible) |
| Mesh FBX 99 KB | 40.8% | **43.8%** | 44.8% | close |

### Archive Results

| Test Suite | MBFA | Notes |
|------------|------|-------|
| Source tree | 4.02% | Similarity grouping + cross-file LZ |
| Text files | 31.03% | |
| JSON / CSV | 10.40% | |
| Binary / media | 30.74% | Incompressibility isolation active |
| Mixed directory | 31.48% | |
| Full corpus | 26.40% | |

Lower % = better. MBFA's strongest advantages are on highly repetitive data, structured binary with delta filters, large JSON/structured text, and large source trees where cross-file LZ similarity grouping pays off.

---

## Project Structure
```
mbfa/
├── src/
│   ├── main.rs          CLI — compress / decompress / archive / extract / list
│   ├── lib.rs           Public API, incompressibility gate, entropy variant selection
│   ├── opcode.rs        Token enum, fixed opcode constants, adaptive field helpers
│   ├── encoder.rs       Adaptive LZ scanner — Phase A/B/C fingerprint pipeline
│   ├── bitwriter.rs     Token stream → packed bitstream
│   ├── bitreader.rs     Packed bitstream → token stream
│   ├── decoder.rs       Token stream → reconstructed bytes
│   ├── pairing.rs       Token pair encoding + Cantor operand compression
│   ├── fold.rs          Orchestrates fold passes + stopping logic
│   ├── unfold.rs        Reverses N fold passes using header
│   ├── entropy.rs       Huffman entropy coding — 5 variants (v1–v5)
│   ├── filters.rs       Delta pre/post filters for structured binary
│   ├── archive.rs       Multi-file archive — create, extract, list
│   ├── archive_io.rs    Archive index serialisation / deserialisation
│   └── platform.rs      RAM-aware chunk size selection
├── benches/
│   └── compare.rs       Criterion benchmarks vs gzip and zstd
└── .github/
    └── workflows/
        └── mbfa-ci.yml  CI — build, test, benchmark, deploy
```

---

## Usage
```bash
# Compress a single file
cargo run --release -- compress input.txt output.mbfa

# Decompress
cargo run --release -- decompress output.mbfa recovered.txt

# Verify roundtrip
diff input.txt recovered.txt

# Create an archive from a directory
cargo run --release -- archive ./my_project output.mbfa

# Extract an archive
cargo run --release -- extract output.mbfa ./recovered/

# Extract a single file from an archive
cargo run --release -- extract output.mbfa ./recovered/ --file src/main.rs

# List archive contents
cargo run --release -- list output.mbfa
```

---

## CI / Benchmarks

Normal push runs build + tests on Ubuntu, macOS, Windows.

Add `--publish` or `--deploy` to your commit message to run full benchmarks and deploy the HTML report to GitHub Pages:
```bash
git commit -m "your message --publish"
git push
```

Results published at: **https://mid-d-man.github.io/mbfa/**

---

## Related Work

MBFA is architecturally distinct from all of the following but informed by them:

| Algorithm | Relationship |
|-----------|-------------|
| Fractal compression | Iterative self-referential encoding — but image-specific only |
| Re-Pair (Larsson & Moffat 1999) | Recursive symbol pairing — MBFA's pair encoding is independently derived, operates at token level not byte level |
| Iterated Function Systems | Mathematical ancestor of fractal compression — geometric not instruction-chain |
| Kolmogorov complexity | Theoretical "shortest program" framing — MBFA is the practical counterpart |
| Deflate / Zstd | LZ + entropy coding single-pass — MBFA's multi-fold instruction chain is the key distinction |

---

## Research Potential

The algorithm has been assessed as having genuine Masters/PhD research potential subject to:

- Formal convergence proof — under what conditions is each fold guaranteed to shrink?
- Information-theoretic analysis of instruction stream entropy vs raw data entropy
- Full benchmark suite on Canterbury and Silesia corpora vs Deflate, Zstd, LZMA
- Academic writeup positioning as "multi-fold program-style transform + fixed coding"

---

## Next Steps (Active)

- [ ] Formal benchmark on Silesia corpus
- [ ] Convergence analysis writeup
- [ ] Replace Cantor pairing with Exp-Golomb for offset encoding — no quadratic blowup
- [ ] Move-To-Front transform between fold 1 and fold 2
- [ ] Entropy coding on large streams only — size-gated to avoid header overhead on small files

---

*MidManStudio — MidMans Bit Folding Algorithm — active research, results subject to change*
