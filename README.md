# MBFA — MidMans Bit Folding Algorithm

**MidManStudio** | Research Compression Algorithm | Active Development

---

> ⚠️ **Development Status:** This algorithm is under active research and development. Benchmark results, file formats, and internal behaviour are subject to change between versions. The only component guaranteed stable is the core instruction-set folding logic — the fixed opcode vocabulary (BACKREF / LIT / END) and the multi-fold instruction-chain architecture that defines MBFA as a distinct algorithm.

Full per-module design documentation lives in [docs/mbfa.md](docs/mbfa.md) (source-level "what/why", not duplicated here). See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

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
    ↓  Fold 2: token pair encoding with Exp-Golomb-coded operands (if large enough)
               OR entropy coding (10 variants — Huffman and adaptive range coder) — whichever is smaller
Fold 2 output
    ↓  Fold 3+: LZ on whatever bytes came out of the previous fold
    ↓  ... until stopping condition fires
Final compressed seed + header
```

**Decompression** reads the header, runs the exact inverse number of passes, and reconstructs the original bytes exactly.

---

## Fixed Opcode Vocabulary

The opcode vocabulary is fixed and shared between encoder and decoder. **Never transmitted.** This is a core design decision — no per-fold table overhead. This vocabulary is the stable foundation of MBFA and will not change.

| Opcode | Bit Pattern | Meaning | Operands |
|--------|-------------|---------|----------|
| BACKREF | `0` | Copy from output history | adaptive-width offset + length |
| LIT | `10` | Emit one literal byte | 8-bit byte value |
| REPREF (ring-active mode only) | `110` | Copy from one of the 4 most-recently-used offsets | 2-bit slot index + length |
| END | `11` (legacy) / `111` (ring-active) | End of stream | none |

BACKREF gets the 1-bit code because it becomes the dominant token on any repetitive data after fold 1. Offset and length field widths are **adaptive at runtime** — the values in the table above are defaults, not hard limits.

**Ring-active mode** (a per-header flag) lets the final output scan reuse one of the 4 most-recently-used backref offsets via REPREF instead of re-encoding a fresh offset, saving `offset_bits − 4` bits per hit. It's only used for the final compression pass — all internal comparison scans that pick offset/length field widths always use plain BACKREF, so those decisions aren't skewed by ring hits.

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

**BACKREF operands** are compressed using order-0 Exp-Golomb coding — `offset − 1` and `length − 1` are each encoded independently as self-delimiting Exp-Golomb codes. No flag bit, no 16-bit ceiling, no table, no transmission, fully reversible. (An earlier version used Cantor pairing — `(offset, length)` encoded as a single number, falling back to raw encoding above a 16-bit ceiling; replaced with Exp-Golomb to remove that ceiling.)

---

## Entropy Coding

When fold 1 output exceeds 400 bytes and pair encoding is not used, MBFA tries ten entropy coding variants in parallel (a mix of Huffman-family and adaptive range-coder designs) and picks the smallest result:

| Flag | Variant | Description |
|------|---------|-------------|
| v1 | Joint | Single lit/length Huffman + offset bucket Huffman |
| v2 | 2-context | Separate lit/length tables for after-literal vs after-backref positions |
| v3 | 8-context | Eight lit/length tables split by character category and position context |
| v4 | Slotted | v1 + recent-offset slot reuse (LRU cache of recent offsets) |
| v5 | Slotted 2-ctx | v2 + recent-offset slot reuse |
| v6 | Split-stream | Literal bytes and the token-type/length sequence coded as two separate Huffman streams |
| v7 | Range coder | Adaptive binary range coder (LZMA-style) — no tables transmitted at all |
| v8 | Block-split | Token stream split into segments, each with its own literal/length table, boundaries chosen by direct measurement |
| v9 | Range coder + rep-aware | v7's range coder plus cheap encoding for genuinely-reused ring-buffer offsets |
| v10 | DP-optimal | Bounded-horizon, price-aware optimal parse (bypasses the normal LZ scan), encoded with v9's format |

Huffman variants (v1-v6) bucket-code offsets (similar to DEFLATE distance codes) with variable extra bits and serialise all tables into the output header. Range-coder variants (v7/v9/v10) transmit no tables at all — both sides rebuild the same adaptive probability model from a fixed initial state. See [docs/mbfa/entropy.md](docs/mbfa/entropy.md) for the full design rationale of each.

If no entropy variant beats the raw token stream, entropy coding is skipped entirely.

---

## Filters

Before folding, MBFA detects known binary layouts and applies a reversible pre-filter that reorganizes bytes into a shape the LZ scanner and entropy coders can exploit better than the original interleaved layout:

| Filter | Applies to |
|--------|------------|
| Stride-delta (flags 1–4) | Generic fixed-stride binary — 8/16/24/32-bit PCM audio, pixel data |
| STL (flags 7 legacy, 10 current) | Binary STL 3D meshes — byte-plane split + delta |
| PLY (flag 8) | Binary PLY 3D meshes — byte-plane shuffle + per-vertex-stride delta |
| BCJ (flags 9, 11–15) | x86 / ARM / ARM64 / PowerPC / SPARC / RISC-V executables — branch-target normalisation |
| CFBF (flag 16) | Legacy `.xls` / `.doc` / `.ppt` (OLE2 container) — sector defragmentation |
| FBX (flag 17) | Binary FBX 3D interchange files — numeric array delta |
| glTF/GLB (flag 18) | Binary glTF buffers — numeric array delta |

Format detection reads magic bytes and structural headers (WAV `fmt ` chunk, BMP `bpp` field, PLY/FBX/glTF/CFBF/executable magic numbers), falling back to a generic multi-stride entropy probe when nothing more specific matches. The filter flag is stored in the header and reversed exactly on decompression. See [docs/mbfa/filters.md](docs/mbfa/filters.md) for the full flag table and per-filter design notes.

MBFA also carries small per-format static dictionaries (DixScript source and compiled binary, Unity, Unreal, generic config/YAML/TOML) that seed the LZ scanner with known format boilerplate as addressable history, for files too small to contain much repetition of their own — see [docs/mbfa/dictionary.md](docs/mbfa/dictionary.md).

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
Byte 1:          pair_flag        bit 0: fold 2 used pair encoding
                                  bit 1: fold 1 LZ bitstream uses ring-active opcodes
Byte 2:          entropy_flag     (0 = none, 1–10 = entropy variant)
Byte 3:          filter_flag      (0 = none; see Filters above)
Byte 4:          dict_flag        (0 = none, 1–5 = per-format static dictionary)
Bytes 5..5+N:    offset_bits[0..N]   N = fold_count
Bytes 5+N..5+2N: length_bits[0..N]
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
│   ├── main.rs               CLI — compress / decompress / archive / extract / list
│   ├── lib.rs                Public API, header format, entropy variant tournament
│   ├── opcode.rs              Token enum, fixed opcode constants, adaptive field helpers
│   ├── encoder.rs             Adaptive LZ scanner — Phase A/B/C fingerprint pipeline
│   ├── bitwriter.rs           Token stream → packed bitstream
│   ├── bitreader.rs           Packed bitstream → token stream
│   ├── decoder.rs             Token stream → reconstructed bytes
│   ├── pairing.rs             Token pair encoding + Exp-Golomb operand compression
│   ├── fold.rs                Orchestrates fold passes + stopping logic
│   ├── unfold.rs              Reverses N fold passes using header
│   ├── par.rs                 Parallel/serial iterator shim
│   ├── entropy.rs             Entropy variants v1–v7 + table serialisation
│   ├── entropy_v9.rs          v9 range coder (rep-aware) + v10 entry point
│   ├── price_table.rs         Fractional-bit price tables for the DP parser
│   ├── optimal_parse.rs       Bounded-horizon DP optimal parser (v10)
│   ├── dictionary/            Per-format static dictionaries
│   │   ├── mod.rs                DictId, format detection (candidates_for)
│   │   ├── dixscript.rs          DixScript source-text dictionary
│   │   ├── dixscript_binary.rs   Compiled DixScript binary dictionary
│   │   ├── unity.rs              Unity serialized-YAML dictionary
│   │   ├── unreal.rs             Unreal .uplugin dictionary
│   │   └── config.rs             Generic YAML/TOML config dictionary
│   ├── filters/               Format-aware pre/post-compression filters
│   │   ├── mod.rs                Filter flag vocabulary + detection/dispatch
│   │   ├── delta.rs              Generic stride-delta (flags 1–4)
│   │   ├── probe.rs              Multi-stride entropy probe + WAV/BMP helpers
│   │   ├── ply.rs                Binary PLY compound filter (flag 8)
│   │   ├── stl.rs                Binary STL compound filters (flags 7, 10)
│   │   ├── bcj.rs                x86/ARM/ARM64/PPC/SPARC/RISC-V BCJ (flags 9, 11–15)
│   │   ├── cfbf.rs               CFBF/OLE2 sector defrag (flag 16)
│   │   ├── fbx.rs                Binary FBX array delta (flag 17)
│   │   └── gltf.rs               Binary glTF/GLB buffer delta (flag 18)
│   ├── archive.rs             Multi-file archive — create, extract, list
│   ├── archive_io.rs          Archive index serialisation / deserialisation
│   ├── platform.rs            RAM-aware chunk size selection
│   └── profile_offsets.rs     Ring-slot hit-rate profiling binary
├── benches/
│   └── compare.rs         Criterion benchmarks vs gzip and zstd
├── docs/
│   ├── mbfa.md             Documentation index — overview, parts, CI/workflows
│   └── mbfa/               Per-part docs (core, entropy, archive, dictionary, filters)
├── scripts/
│   └── gen_special_files.py    Generates the special benchmark suite's input files
├── CONTRIBUTING.md         Contribution guide, incl. documentation/commenting conventions
└── .github/
    └── workflows/          CI — build/test, per-suite benchmarks, structure export
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

Add one or more flags to your commit message to trigger additional CI behaviour:
```bash
git commit -m "your message --publish"
git push
```

| Flag | Effect |
|------|--------|
| `--publish` / `--deploy` | Run the main benchmark suite and deploy the HTML report to GitHub Pages |
| `--special` | Run the special benchmark suite (`mbfa-special.yml`, files from `scripts/gen_special_files.py`) |
| `--corp` | Run the Canterbury Corpus benchmark (`mbfa-corpus.yml`) |
| `--archive` | Run the archive-specific benchmark (`mbfa-archive.yml`) |
| `--all` | Run every benchmark suite above |

`mbfa-expose.yml` additionally verifies the crate builds standalone on every push to `main`, and `mbfa-ci.yml` regenerates `others/ProjectStructure.txt` automatically when the file tree changes.

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
- [x] Replace Cantor pairing with Exp-Golomb for offset encoding — no quadratic blowup
- [ ] Move-To-Front transform between fold 1 and fold 2
- [x] Entropy coding on large streams only — size-gated to avoid header overhead on small files

---

*MidManStudio — MidMans Bit Folding Algorithm — active research, results subject to change*
