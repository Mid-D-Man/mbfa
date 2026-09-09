# entropy

The ten entropy back ends MBFA's tournament (`lib.rs::compress`) chooses
between, plus the DP-driven optimal parser that feeds v10.

## Modules

### entropy.rs

**What it does:** Implements entropy variants v1-v7 (v8 lives in
`lib.rs` since it's a structural idea rather than a coder; v9/v10 live
in `entropy_v9.rs`/`optimal_parse.rs`). v1-v6 are Huffman-family variants
differing in how literal/length/offset tables are split and contextu-
alized (single table, context-split by previous byte, eight-context,
slotted offsets, etc.); v7 is an adaptive binary range coder in the LZMA
style, needing no transmitted tables at all since both sides start from
the same initial probability state. Also owns the four table
serialization formats (fmt0-fmt3 fixed/RLE encodings, fmt4 adaptive
range-coded) that `serialize_table` brute-force-compares to pick the
smallest for any given table.

**Decisions:**
- v7's range coder architecture mirrors the reference xz/LZMA
  `RcDecoder` design directly (carry propagation via a 64-bit `low` with
  the carry byte accessible at bits 32-39, `cache`/`cache_ff` for
  deferred-byte carry handling).
- fmt4 reuses v7's own range-coder primitives (`Rc7Enc`/`Rc7Dec`) to code
  a table's length sequence, rather than introducing a second coder —
  a "same as previous" bit captures runs of any length (including runs
  of 1-2 that fmt3's run-of-3+ RLE sentinel can't touch), and nonzero
  lengths go through an adaptive 8-bit bittree. It costs more than fmt3
  on tiny tables (the range coder's fixed drain plus a 2-byte length
  prefix outweighs the savings) but wins once there's real structure to
  exploit — `serialize_table` always compares and picks the smaller.

### entropy_v9.rs

**What it does:** v9 = v7's adaptive range coder plus repeat-offset
modeling: adds cheap encoding for genuinely-reused ring offsets
(`Token::RepRef`) on top of v7's plain range coder, mirroring LZMA's
`is_rep`/`is_rep0`/`is_rep1`/`is_rep2` shape. Unlike v7, callers must
pass the *unresolved* ring-active token stream (not `resolve_ring()`'d),
since resolving away the ring information first would defeat the
purpose. Also hosts `write_tokens_v9_optimal`, the entry point v10 uses
in `lib.rs`.

**Decisions:**
- Simplified relative to LZMA's version: `Token::RepRef { slot, length }`
  carries no offset at all, only a ring-slot index — `decoder::reconstruct`
  (not this file) resolves slot to actual offset and performs the copy,
  so v9 itself never needs to track ring state, just cheaply distinguish
  "this token is RepRef with slot N" from "this token is Backref with a
  fresh offset."
- Probability-model constants (`RC_PROB_BITS`, `RC_PROB_SCALE`, `RC_SHIFT`,
  `RC_PROB_INIT`) are redefined locally rather than importing them from
  `entropy.rs`, to keep this file's only dependency on `entropy.rs`
  limited to the two range-coder primitive structs.
- The length-coding tiers are laid out so the HI tier caps at value 254
  (not 255), applied uniformly across LO/MID/HI — kept consistent with
  v7's reserved-255-for-End-sentinel convention rather than reusing the
  full 8-bit range.

### price_table.rs

**What it does:** Fractional-bit price lookup (units of 1/16 bit) used
by the DP optimal parser (`optimal_parse.rs`) to ask "how many bits would
this candidate cost under the current adaptive model" via a single array
lookup instead of computing `-log2(prob/2048)` directly.

**Decisions:**
- The 128-entry price table is ported verbatim from lzma-rust2's
  `range_enc.rs` price tables. This is safe to port byte-for-byte rather
  than regenerate, because MBFA's range coder (entropy.rs's v7/v9) uses
  the identical probability scale (11-bit probs, 0..2048 range) that the
  table was built for.
- `get_bittree_price`/`get_direct_bit_price` intentionally mirror
  entropy.rs's *real* indexing conventions for `encode_bittree` (first
  slot accessed is `probs[base]`, not `probs[base+1]`) and `encode_direct`
  (`probs[base + i]` indexed by bit position, not tree depth) — getting
  either convention wrong would silently mis-price candidates without
  any type error to catch it.
- `get_raw_bits_price` exists for LZMA-comparison purposes and any future
  fully-raw coding tier; MBFA's own `rc_encode_distance` doesn't
  currently have a tier that would use it.

### optimal_parse.rs

**What it does:** A bounded-horizon, price-aware alternative to
`encoder.rs::scan_from` for fold 1, used when the caller intends to
encode the result with entropy_v9 (or v7). Prices every candidate —
literal, each of the 4 rep slots, and every point on the fresh-match
length/distance frontier — under v9's actual current adaptive model via
`price_table.rs`, and picks whichever sequence of tokens is cheapest
over a lookahead window (`HORIZON`, default 32) via a forward price-DP
with backward traceback. This is what lets a nearby short match beat a
distant long one once distance-coding cost is priced in — something
`encoder.rs`'s greedy/lazy scanner can't see.

**Decisions:**
- Modeled directly on lzma-rust2's real optimal parser
  (`encoder_normal.rs`'s `get_optimum`/`Optimum`/`opts[]` design), with
  two deliberate simplifications:
  1. **Bounded horizon, not dynamic.** LZMA extends its lookahead
     position by position with `nice_len`-based early termination; this
     always fills a fixed-size window before tracing back. Simpler to
     reason about and verify; the cost is not being able to extend the
     horizon further when a very long match is found nearby.
  2. **No LZMA `state` tracking.** LZMA's `is_match`-equivalent is
     context-split by a 12-state machine tracking what kind of token
     preceded it. MBFA's v9 `PROB_MATCH` is a single unconditional
     probability with no such split in the format at all (see
     entropy_v9.rs's prob layout), so there's nothing state-dependent to
     track here — this isn't a simplification relative to what v9 can
     represent, just a note that if v9 ever grows state-splitting, this
     parser's price computation would need a `state` field to match.
- `find_matches_tiered`'s hash-chain walk is ported directly from
  `encoder.rs`'s matcher (same `hash3`, same chain-limit logic) rather
  than a separate structure like LZMA's BT4 — cross-checked against a
  faithful reimplementation during development. It records a new
  frontier point every time the walk reaches a new length record;
  because `prev[]` links strictly older (larger-offset) positions,
  offsets are non-decreasing as the chain is walked, so "first time we
  see length >= L" is automatically "cheapest distance achieving length
  >= L" — mirroring lzma-rust2's `Matches { len, dist }` semantics
  without porting BT4 itself.

## Fixes and Problems

### entropy.rs

Three entropy-coding capabilities were added incrementally: the v7
adaptive range coder itself; fmt3 RLE table serialization (to reduce
per-table overhead on small files like `.uplugin`/YAML/TOML/INI); and
later fmt4, an adaptive range-coded table serialization format built on
v7's own bit primitives. `serialize_table` brute-force-compares all
formats and keeps whichever is smallest, so none of these additions
changed which format wins for any table that already had a clear winner
before they existed.
