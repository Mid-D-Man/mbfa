# filters

Pre/post-compression byte-level transforms that reorganize known binary
layouts (3D meshes, executables, legacy container formats) into a shape
the LZ scanner and entropy coders can exploit better than the original
interleaved bytes would allow.

## Modules

### mod.rs

**What it does:** Owns the filter flag vocabulary (header byte 3),
format detection (`detect_filter`), and dispatch to each sub-module's
`apply_filter`/`undo_filter`.

Filter flags:

| Flag | Filter |
|---|---|
| 0 | none |
| 1-4 | stride-delta (delta.rs), stride 1-4 |
| 7 | STL: plane-shuffle + stride-12 delta (legacy decode only) |
| 8 | PLY: plane-shuffle x4 + per-vertex-stride delta (ply.rs) |
| 9 | x86 BCJ, also covers x86 ELF/Mach-O/a.out (bcj.rs) |
| 10 | STL: field-major plane-split + stride-1 delta, current (stl.rs) |
| 11-15 | ARM / ARM64 / PowerPC / SPARC / RISC-V BCJ (bcj.rs) |
| 16 | CFBF sector defragmentation, legacy .xls/.doc/.ppt (cfbf.rs) |
| 17 | FBX raw numeric array delta, binary FBX (fbx.rs) |
| 18 | glTF/GLB buffer array delta, non-interleaved bufferViews only (gltf.rs) |

**Decisions:**
- Detection order is fixed and significant (checked top-to-bottom in
  `detect_filter`): exact-size structural formats (binary STL) and
  strong magic-number formats (WAV/RIFF, BMP, PLY, CFBF, FBX, glTF,
  DixScript binary, PE/COFF, ELF, Mach-O, a.out) are all checked before
  falling back to the generic multi-stride entropy probe, so a
  format-specific filter always wins over the generic one when both
  would apply.
- DixScript's own compiled-binary magic is checked explicitly so it
  skips the filter probe entirely (flag 0) — its dictionary-based
  handling lives in `dictionary/` (see [dictionary.md](dictionary.md))
  instead of a byte-shuffling filter.

### delta.rs

**What it does:** Generic stride-N delta pre-filter (flags 1-4): each
byte is replaced with its difference from the byte N positions earlier,
which turns slowly-varying fixed-stride binary data (audio samples,
bitmap rows) into smaller, more repetitive values.

### probe.rs

**What it does:** Multi-stride entropy probe used as the final fallback
in `detect_filter`'s order — samples 8 KB, tries each delta stride 1-4,
and picks the best if any clears an entropy-reduction threshold (0.45).
Also hosts the WAV/BMP-specific stride-selection helpers used earlier
in the detection order (sample width/channel count for WAV, bits-per-
pixel/row-stride for BMP), since both formats have a directly-readable
header field that determines the correct stride instead of needing the
generic probe.

### ply.rs

**What it does:** Binary PLY (flag 8): reorganizes vertex float data
into four byte-planes (one per IEEE 754 byte position, LSB to MSB), then
delta-encodes within each plane using stride = floats-per-vertex, so
each delta differences the same semantic float field across consecutive
vertices rather than crossing field boundaries.

**Decisions:**
- Stride = floats-per-vertex rather than a naive stride-1 delta,
  because for smooth geometry, neighboring vertices have similar values
  within the *same* field (x, y, z, normal, uv, ...), but adjacent bytes
  across different fields are essentially uncorrelated. Per-field
  striding produces much smaller deltas and a higher LZ match rate.

### fbx.rs

**What it does:** Binary FBX (Autodesk's 3D interchange format, flag 17):
raw numeric array delta, applied per-array at its native element width.

### cfbf.rs

**What it does:** CFBF (Compound File Binary Format, a.k.a. OLE2/
"Structured Storage" — the container format under legacy .xls/.doc/.ppt,
flag 16): sector defragmentation, reorganizing the format's sector chain
into contiguous order before compression.

### gltf.rs

**What it does:** Binary glTF/GLB (flag 18): buffer array delta for
non-interleaved `bufferView`s.

### stl.rs

**What it does:** Binary STL (flags 7 and 10): flag 10 (current) does a
field-major plane-split plus stride-1 delta; flag 7 (plane-shuffle plus
stride-12 delta) is kept for decoding older archives but is no longer
produced by the encoder.

### bcj.rs

**What it does:** Branch-Call-Jump normalization filters (flag 9 for
x86 PE/COFF/ELF/Mach-O/a.out, flags 11-15 for ARM/ARM64/PowerPC/SPARC/
RISC-V): rewrites relative branch-target addresses in executable code
into absolute form (and back on decode), which turns many
near-identical-but-shifted instruction encodings into exactly-repeated
byte sequences the LZ scanner can match.
