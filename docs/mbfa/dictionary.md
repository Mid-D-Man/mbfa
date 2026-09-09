# dictionary

Per-format static dictionaries used as addressable pre-history for small
structured files, so a file's compressor can find backref matches
against known format boilerplate even when the file itself is too small
to contain much repetition of its own.

## Modules

### mod.rs

**What it does:** Defines `DictId` (which, if any, per-format dictionary
a file's fold-1 pass used — stored verbatim as the header's `dict_flag`
byte) and `candidates_for`, a cheap structural sniff that shortlists
which dictionaries are worth trying for a given input.

**Decisions:**
- `candidates_for` is a shortlist, not a verdict — `fold.rs` still
  measures actual bit cost for each candidate and only keeps whichever
  one genuinely wins. Each detection signal is tied to something
  structurally guaranteed for that format (e.g. DixScript source is
  detected via one of the seven real top-level section markers the
  grammar allows; DixScript compiled binary via its own magic number;
  these two are mutually exclusive by construction since source text
  can't also start with the binary's 4-byte magic).
- When nothing matches, `candidates_for` still falls back to the generic
  config dictionary rather than skipping outright — a false negative
  only costs a missed ratio win, never correctness, and the config
  dictionary is the smallest non-DixScript one, so it's the cheapest
  blind guess.
- `DictId::None.bytes()` returns `&[]`, which makes every dict-aware call
  site (`scan_with_dict`, `min_offset_bits_for_dict`,
  `decoder::reconstruct`) behave exactly like the no-dictionary case for
  free, with no separate "dictionary present?" branch needed anywhere
  downstream.

### dixscript.rs

**What it does:** Dictionary for DixScript *source* text (`.mdix` files).

**Decisions:**
- Content sourcing, all verified against real files rather than guessed:
  section-marker keywords and grammar vocabulary are copied directly
  from `others/midx.ebnf`'s terminals, cross-checked for real occurrence
  in the corpus rather than only asserted from the grammar; everything
  else comes from longest-common-block mining across all 99 real,
  non-synthetic `.mdix` files in a commit-pinned clone of
  Mid-D-Man/DixScript-Rust, with each mined entry verified to recur
  byte-for-byte in at least 4 independent files. Aggregate coverage
  across that same corpus: 45.9%.
- Compiled-binary content deliberately lives in `dixscript_binary.rs`
  instead of here (see below) — source dictionary content measured only
  24.0% coverage against synthetic compiled binaries, versus 59.5% from
  a dedicated dictionary built from the binary format's own wire
  encoding.

### dixscript_binary.rs

**What it does:** Dictionary for *compiled* DixScript binaries (the
`.mdix.enc`-shaped output of DixScript-Rust's own compiler), split out
from `dixscript.rs` because compiled binaries and source text share
almost no bytes.

**Decisions:**
- Content is derived directly from the real writer source
  (`BinarySerialization/SectionWriters/{config,security}_section_writer.rs`,
  `value_encoder.rs`), not mined: `@CONFIG`/`@SECURITY` sections are
  dominated by strings drawn from the grammar's own fixed vocabulary,
  each written as a length-prefixed field in the writer's real wire
  format. Because this is derived directly from writer source rather
  than mined and held out, there's no cross-validation uncertainty about
  whether it's representative the way source-text mining has.
- `@DATA` is deliberately not covered: `DataEntry` names are user-chosen
  identifiers, not grammar vocabulary, so there's nothing safe to
  hardcode there. A real cross-file `@DATA` dictionary would need mining
  against a corpus of real compiled binaries the way `dixscript.rs`'s
  source content was mined against real `.mdix` files.
- A 32-byte SHA-256 checksum ends every compiled binary with no
  length-prefix or marker before it — cryptographically indistinguishable
  from random and unreachable by any dictionary or compressor. This is a
  real, unavoidable floor on the achievable ratio for small files in this
  category (any other compressor hits the identical floor on the
  identical bytes).

**Benchmarks:** Against 15 varied, format-faithful synthetic compiled
binaries: this file's dedicated ~338B dictionary reached 59.5% coverage,
versus 24.0% using `dixscript.rs`'s source-oriented content, versus 64.0%
combining both (confirming the source dictionary contributes very
little to compiled-binary coverage, justifying the split).

### unity.rs

**What it does:** Dictionary of fixed engine/schema serialization fields
Unity's YAML serializer unconditionally emits (`GameObject`/`Transform`/
`MeshRenderer` blocks).

**Benchmarks:** Held-out same-format testing against independently-varied
Unity YAML samples measured roughly 50-79% byte coverage and roughly
13-50% real projected compressed-size reduction.

### unreal.rs

**What it does:** Dictionary of fixed descriptor fields Unreal's own
plugin-descriptor writer unconditionally emits (`FileVersion`/`Version`/
`VersionName`/`Modules` block, etc.) for `.uplugin` and adjacent JSON.

**Benchmarks:** Verified via held-out same-format testing on the same
basis as `unity.rs`.

### config.rs

**What it does:** Generic catch-all dictionary for structured config
files that aren't specifically Unity, Unreal, or DixScript — Kubernetes
ConfigMap boilerplate plus Cargo-style TOML section/key layout.
Deliberately kept small and generic, since it's the fallback used when
`candidates_for` finds no more specific format match.

## Fixes and Problems

### mod.rs / dixscript.rs / dixscript_binary.rs / unity.rs / unreal.rs / config.rs

This module used to be a single flat `dictionary.rs` file with one
combined dictionary for all formats. Splitting it into per-format files
(this directory) means each format's dictionary only contains bytes that
format can actually match — a DixScript file no longer pays
addressability cost for Unity/Unreal/Config content it can never match,
and vice versa — and lets `candidates_for` skip the scan entirely for
files that plainly aren't a given format. This is also why the header
gained a `dict_flag` byte: with exactly one possible dictionary, the
decoder never needed to be told anything; with five differently-sized
ones, it does (see [core.md](core.md)'s `lib.rs`/`decoder.rs` notes).

A regression investigation (dictionary vs. benchmark fixture, mid-2026)
found that a CI benchmark rank regression for `DixScript_Source`/
`DixScript_Compiled` was *not* caused by updating `dixscript.rs`'s
content from placeholder pseudo-syntax (`module Platform { ... }`,
never valid DixScript) to real section-marker syntax mined from the
actual corpus — that update was correct, since real DixScript-Rust
source universally uses the new syntax and never used the old one. The
regression was actually caused by `scripts/gen_special_files.py`'s
`PLATFORM_MDIX` benchmark fixture not being updated to match: it still
generated its "DixScript_Source" benchmark file using the old,
never-valid syntax, so the (now-correct) dictionary shared almost no
bytes with the (still-wrong) fixture. Measured: the corrected dictionary
covered only 48.7% of the old fixture (down from the old dictionary's
91.6%, which was really overfitting to that one fixture's exact
wording) but covers 45.9% of real DixScript-Rust source repo-wide (up
from 42.9% for the old combined dictionary). The fixture has since been
corrected to real syntax; the dictionary now covers 50.7% of it.
`gen_special_files.py`'s `gen_binary_dixscript()` (source of the
"DixScript_Compiled" row) had the analogous problem — a generic Object
blob with made-up keys instead of the real `ConfigSection` wire format —
and was fixed in the same pass.

Regression-guard tests in `dixscript.rs` check that the old
`module Platform { ... }` pseudo-syntax doesn't creep back into the
dictionary content.
