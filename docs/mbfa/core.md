# core

The compression pipeline itself: the public API, the LZ scanner and its
inverse, fold/unfold orchestration, pair encoding, the token vocabulary,
bitstream I/O, and the parallelism shim.

## Modules

### lib.rs

**What it does:** Owns the public `compress`/`decompress` API and the
file header format (fold count, pair/ring flags, entropy variant,
filter, dictionary — see the header layout doc comment on `compress()`
for the exact byte layout). `compress()` runs an optional pre-filter,
an incompressible-data early exit, one or more fold passes (via
`fold::fold`), then a brute-force entropy tournament across all ten
variants (`v1`-`v10`) in parallel, keeping whichever produces the
smallest payload — or PAIR encoding if it beats entropy outright. v8
("block-split") lives here as `build_v8_candidate`/`try_entropy_v8`
rather than in `entropy.rs`, since it's a MBFA-specific structural idea
(split token stream into segments, each with its own table) rather than
an alternative low-level coder.

**Decisions:**
- Ten entropy variants compete by direct measurement (encode, compare
  resulting sizes) rather than by heuristic, matching the rest of the
  codebase's "measure, don't guess" philosophy.
- v9's gate/encode use the *unresolved* ring-active token stream
  (`tokens_raw`), not the `resolve_ring()`-flattened one v1-v8/v7 use,
  since v9 is only meaningfully different from v7 when `Token::RepRef`
  is still present.
- v7 and v9 each verify their own encode/decode roundtrip before being
  trusted as tournament candidates — cheap, and catches any encode/decode
  desync before it could ship.
- v10 ("v9-optimal") bypasses `fold.rs` entirely and always decodes with
  zero fold-unwind passes and an empty dict, since it builds its token
  stream directly from `to_fold` via a bounded-horizon price-DP
  (`optimal_parse.rs`) rather than fold's own greedy/lazy tokenization.

### opcode.rs

**What it does:** Defines the fixed `Token` vocabulary (`Lit`, `Backref`,
`RepRef`, `End`) and the two opcode encodings: the plain mode (Backref
only) and the ring-active mode (`Backref` + `RepRef`, used when the
encoder's 4-slot recent-offset ring is active). `resolve_ring()` converts
a ring-active token stream back to plain `Backref`-only form for
consumers (entropy coders, `pair_encode`) that don't understand `RepRef`.

### bitwriter.rs / bitreader.rs

**What they do:** Write and read a `Token` stream to/from a compact
bitstream — the on-disk representation of a single fold pass before any
entropy coding is applied.

### encoder.rs

**What it does:** The core LZ scanner: an O(n)-time, O(window)-memory
hash-chain matcher with one step of lazy lookahead. `scan_adaptive` is
the real entry point used by `fold.rs`: it picks offset/length bit
widths (`ob`/`lb`) via a multi-phase process (fingerprint-based
prediction, a cheap "Phase B" comparison scan, then escalating to a
wider "Phase C" scan only when saturation checks on Phase B's output
indicate the current bit widths are too narrow), then re-scans once more
at the winning widths to produce final output tokens.

**Decisions:**
- All internal comparison/decision scans (Phase B, Phase C, discovery)
  run with `emit_repref=false`, so they always see pure `Backref` tokens.
  This keeps the saturation checks (`length_peak_saturated`,
  `offset_or_upper_half_saturated`) accurate — a mix of `Backref` and
  ring-slot `RepRef` tokens would hide large offsets behind ring hits
  and under-trigger the escalation to wider bit widths.
  Only the *final* output scan (after `ob`/`lb` is already decided) uses
  `emit_repref=true`, and only for unfiltered data — filtered binary data
  (terrain, STL, PLY, DLL) always uses plain `Backref` output too, since
  the ring buffer doesn't meaningfully help filtered binary data.
- `scan_with_dict`/`scan_from` generalize the scanner to seed hash chains
  over a prepended static dictionary without ever emitting tokens *for*
  the dictionary region itself — only real input positions can be match
  targets; the dictionary is only ever a source of matches.
- The optional expansion-bail check (`SCAN_EXPANSION_INTERVAL_MASK`)
  periodically checks whether the token stream so far is expanding
  (mostly literals, more bits than input) and aborts early on
  incompressible input rather than scanning the whole file for nothing.

### decoder.rs

**What it does:** The exact inverse of the ring-aware parts of
`encoder.rs`: reconstructs a byte slice from a `Token` stream, mirroring
the encoder's 4-slot ring-buffer state so `RepRef` tokens resolve
identically on both sides. Deterministic and stateless per call (the
ring is rebuilt fresh each `reconstruct()`), so it's safe to call
concurrently on different streams.

**Decisions:**
- Static-dictionary support treats the addressable "history" for any
  offset as conceptually `dict ++ output`, not just `output`. A normal
  backref within the real window resolves exactly as it would with no
  dictionary; only offsets that deliberately reach further back (only
  ever produced by `encoder::scan_with_dict`) fall into the dictionary
  region. `dict` is passed in by the caller (`unfold.rs`), resolved from
  the header's `dict_flag` byte via `dictionary::DictId::bytes()` —
  passing `&[]` reproduces plain no-dictionary decoding exactly.

### fold.rs / unfold.rs

**What they do:** Orchestrate one or more compression/decompression
passes. `fold()` runs fold 1 (the real LZ pass, optionally seeded with a
per-format static dictionary chosen by `dictionary::candidates_for`),
then zero or more additional folds over the *compressed bytes themselves*
(treating the fold-1 output as raw bytes to re-scan), stopping when a
fold no longer shrinks the data or `max_folds` is reached. `unfold()`
reverses the chain, dispatching on the header's `entropy_flag` to the
matching decoder for the final fold, then unwinding any additional folds
underneath it.

**Decisions:**
- The fold-2-and-later bitstream is never ring-encoded (it's scanning
  already-compressed/packed bytes, not the original ring-aware token
  stream), so `fold()` always calls `resolve_ring()` to convert any
  `RepRef` back to `Backref` before `pair_encode`/`write_tokens` ever see
  a fold-2+ stream. The decoder assumes fold 2+ is never ring-active
  (`unfold.rs` only reads ring-active opcodes when `folds_done==1`), so
  this conversion is required for correctness, not just an optimization.
- A static dictionary trial (fold 1 only) is measured against the
  non-dictionary baseline by raw bit cost and only kept if it actually
  wins — the caller never assumes the dictionary helps.

### pairing.rs

**What it does:** Token-level pair encoding: bundles two adjacent tokens
(literal or backref) into one of seven prefix codes (`LL`/`LB`/`BL`/`BB`/
`SL`/`SB`/`END`), with offset-1 and length-1 each encoded independently
via order-0 Exp-Golomb (self-delimiting, no flag bit, no length limit).
`pair_encode` never sees `Token::RepRef` — `fold.rs` always calls
`resolve_ring()` first, and the `RepRef` match arm is marked unreachable
to satisfy Rust's exhaustiveness check.

### par.rs

**What it does:** A tiny compatibility shim so call sites in `lib.rs` and
`encoder.rs` can use `.into_par_iter()`/`join(..)` identically whether or
not the `"parallel"` feature (rayon) is enabled — with it off, both fall
back to a plain serial iterator and sequential calls, same output, no
threads.

## Fixes and Problems

### opcode.rs

An earlier version of the ring-update match bound `length` from
`Token::Backref` unnecessarily; only `offset` is needed for the ring
update (`length` is preserved separately via `token.clone()`), so the
match now uses `..` to skip binding it.

### decoder.rs

Before the `dictionary/` subdirectory split (see
[dictionary.md](dictionary.md)), there was exactly one possible
dictionary, so `reconstruct()` could import it as a fixed constant. With
five per-format dictionaries now available, the dictionary bytes are
passed in as a parameter instead, resolved by the caller from the
header's `dict_flag`.

### fold.rs / unfold.rs

The ring-buffer feature (`Token::RepRef`, 4-slot recent-offset reuse) was
added after the original plain-`Backref`-only design:
- `write_tokens`/`read_tokens` (bitwriter.rs/bitreader.rs) changed shape
  to return/accept a `ring_active` flag alongside the token bytes.
- `ring_was_used` is captured from fold 1's `write_tokens` result and
  threaded through as an extra element of `fold()`'s return tuple, and
  ultimately into the header's `pair_flag` byte (bit 1).
- Because `pair_encode` doesn't understand `RepRef`, `fold()` resolves
  any ring references back to plain `Backref` before pairing.
- The diagnostic helpers (`diag_stream_bit_cost`, `log_window_diagnostics`)
  were updated to handle `Token::RepRef` as a ring hit for saturation
  accounting purposes.

A related correctness bug: fold 2+ bitstreams must never be ring-encoded,
but for a period the scan feeding fold 2+ could still legitimately emit
`RepRef` tokens (since the scanner's ring-probing behavior didn't
distinguish which fold it was being used for). Because the decoder
unconditionally assumes fold 2+ is never ring-active, any `RepRef` that
slipped through desynced the opcode set on decode. Fixed by always
calling `resolve_ring()` before `write_tokens` for fold 2+, guaranteeing
`ring_active` is false in that case regardless of what the scan produced.

### pairing.rs

Pair encoding originally used a Cantor-pairing scheme to jointly encode
`(offset, length)` as a single number. Replaced with independent
order-0 Exp-Golomb coding of `offset-1` and `length-1`: self-delimiting
(no flag bit needed) and with no 16-bit limit on either value, unlike
the Cantor scheme.

### encoder.rs

The ring-buffer feature (`emit_repref`) went through several iterations
before reaching its current form. Earlier attempts modified only
emission thresholds and comparison costs, but didn't prevent `RepRef`
tokens from appearing in the *comparison* (Phase B) scan itself. With
`RepRef` present in a comparison scan, large-offset backrefs could
silently become ring hits (which carry no explicit offset field), so
`offset_or_upper_half_saturated` under-counted upper-half matches, the
saturation fraction stayed under threshold, Phase C never triggered, and
some inputs (e.g. terrain data needing `ob=19`) stayed at a too-narrow
`ob=17` and produced LZ output larger than the input.

The fix that stuck: give `scan()`/`scan_from` an explicit `emit_repref`
parameter. `emit_repref=false` (used for *all* Phase B/C comparison and
decision scans) skips ring probing entirely and only ever emits
`Backref`, exactly matching pre-ring-buffer behavior — so the saturation
checks always see the true picture. `emit_repref=true` is used only for
the final output scan once `ob`/`lb` are already decided. Two smaller
fixes accompanied this: `ref_worthwhile` uses the `backref_bits`
threshold uniformly for both `Backref` and `RepRef` candidates (keeping
the minimum effective match length consistent regardless of mode), and a
`RepRef` is only preferred over a hash-chain `Backref` when its match
length is equal-or-longer (never shorter, even though a ring hit is
cheaper per bit).

### lib.rs

Three entropy variants were added incrementally to the tournament:
- **v7** (adaptive binary range coder): closes a gap where Huffman's
  integer-bit floor is visible on near-degenerate distributions. No
  pre-built tables are needed since the range coder adapts its own
  probability model during encoding.
- **v8** (block-split): splits the token stream into segments, each with
  its own literal+length table, picking boundaries by direct measurement.
- **v9**: v7's range coder plus repeat-offset (ring) awareness, mirroring
  LZMA's `is_rep`/`is_rep0`/`is_rep1`/`is_rep2` shape (see
  [entropy.md](entropy.md)).

The header's `dict_flag` byte was added when the single-file
`dictionary.rs` was split into the five per-format files under
`dictionary/` (see [dictionary.md](dictionary.md)) — with only one
possible dictionary, the decoder could assume it implicitly; with five
differently-sized ones, it needs to be told explicitly which was used.

The defense-in-depth roundtrip check on v7/v9 candidates (re-decoding a
candidate payload and comparing it against the original tokens before
trusting it as a tournament entry) was added after a real fold-2+
ring-encoding mismatch shipped undetected; the root cause is fixed at
its source in `fold.rs` (see above), but the roundtrip check remains as
a cheap general safety net against any future encode/decode desync.
