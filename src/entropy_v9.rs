// src/entropy_v9.rs
//! v9: v7 (adaptive binary range coder) + repeat-offset modeling.
//!
//! ## The gap this closes
//!
//! v7's `write_tokens_v7` requires `Token::RepRef` to already be flattened
//! to `Token::Backref` via `resolve_ring()` before it's called (see that
//! function's doc comment and its `Token::RepRef { .. } => unreachable!()`
//! arm). That means every backref -- whether it reuses the exact offset
//! from two tokens ago or has never been seen before -- pays the full
//! `rc_encode_distance` cost: a 6-bit adaptive slot bittree plus up to 24
//! adaptively-modeled extra bits. A genuinely-reused offset gets no
//! discount at all at the entropy layer, even though fold.rs's P6 ring
//! buffer (opcode.rs's `Token::RepRef`, `MAX_RING_SLOTS = 4`) already
//! *identifies* reuse during matching -- v7 just throws that information
//! away before coding.
//!
//! LZMA closes exactly this gap with `is_rep`/`is_rep0`/`is_rep1`/`is_rep2`:
//! a genuinely-reused rep0 offset costs as little as 2-3 adaptively-modeled
//! bits total, vs. the 15-46+ bits of a full fresh-distance re-encode.
//! v9 mirrors that shape, adapted to what MBFA's `Token` type actually
//! carries.
//!
//! ## Why this is simpler than LZMA's version, not harder
//!
//! LZMA's rep-index bits exist because its encoder/decoder need to know
//! the ACTUAL byte distance to perform the copy -- so it tracks rep0-3 as
//! real distances internally. MBFA's architecture splits that concern out
//! already: `Token::RepRef { slot, length }` carries no offset at all: it's
//! addressed purely by ring-slot index, and `decoder::reconstruct` (not
//! this file) is what already resolves slot -> actual offset and performs
//! the byte copy, using its own ring array. v9 therefore never needs to
//! track ring state itself -- it just needs to cheaply encode "this token
//! is `Token::RepRef` with slot N" vs. "this token is `Token::Backref`
//! with a fresh offset", and hand the resulting `Token` stream to the
//! SAME `decoder::reconstruct` that already handles both cases correctly
//! (verified in decoder.rs's `dict_reconstruct_tests`). LZMA also
//! distinguishes a "short rep" (single-byte rep0 match, no length coded)
//! from "long rep" -- MBFA has no equivalent, since `ref_worthwhile =
//! best_len >= 2` in encoder.rs means no reference of any kind is ever
//! emitted for a 1-byte match, so v9 has one fewer case to handle.
//!
//! ## Prob layout (v9 vs. v7, entropy.rs)
//!
//! Everything after the new is_rep/is_rep0/is_rep1/is_rep2 bits and the
//! new dedicated rep-length tier is byte-for-byte the same shape as v7,
//! just shifted by the 275 new probs' width (4 + 2+7+7+255 = 275):
//!
//! ```text
//!   PROB_MATCH            0   ( 1)   0=literal, 1=match (rep or fresh)
//!   PROB_IS_REP           1   ( 1)   0=fresh backref, 1=rep
//!   PROB_IS_REP0          2   ( 1)   0=slot 0, 1=not slot 0
//!   PROB_IS_REP1          3   ( 1)   0=slot 1, 1=not slot 1
//!   PROB_IS_REP2          4   ( 1)   0=slot 2, 1=slot 3
//!   PROB_REP_LEN_CHOICE   5   ( 2)   mirrors v7's PROB_LEN_CHOICE
//!   PROB_REP_LEN_LO       7   ( 7)
//!   PROB_REP_LEN_MID     14   ( 7)
//!   PROB_REP_LEN_HI      21   (255)
//!   PROB_LEN_CHOICE     276   ( 2)   fresh-match length (was 1 in v7)
//!   PROB_LEN_LO         278   ( 7)   (was 3)
//!   PROB_LEN_MID        285   ( 7)   (was 10)
//!   PROB_LEN_HI         292   (255)  (was 17)
//!   PROB_DIST_SLOT      547   (252)  (was 272)
//!   PROB_DIST_EXTRA     799   ( 24)  (was 524)
//!   PROB_LIT            823   (2040) (was 548)
//!   PROB_TOTAL         2863
//! ```
//!
//! End-of-stream reuses v7's exact sentinel trick (fresh-length-tier HI
//! value 255 = length 273), routed through the is_rep=0 (fresh) branch --
//! no new signal needed, matching v7's approach of not needing a dedicated
//! end opcode.
//!
//! ## Format byte
//!
//! This is entropy_flag = 9 in the header (see lib.rs's format doc comment
//! -- add "9=v9 (rep-aware range coder)" alongside the existing 1-8 list
//! when wiring this in).
//!
//! ## Caller contract
//!
//! Unlike v7, `write_tokens_v9` takes the token stream AS PRODUCED BY
//! fold.rs's scan (with `Token::RepRef` intact, NOT resolve_ring()'d) --
//! resolving away the ring information before calling this would defeat
//! the entire point. `read_tokens_v9`'s output is directly consumable by
//! `decoder::reconstruct` exactly like v7's `read_tokens_v7` output is.

use crate::opcode::{Token, MAX_RING_SLOTS};
use crate::entropy::{Rc7Enc, Rc7Dec, offset_to_bucket, bucket_to_offset, bucket_extra_bits};

// ── Probability model constants (must match entropy.rs's RC_* exactly --
//    verified identical: RC_PROB_BITS=11, RC_PROB_SCALE=2048, RC_SHIFT=5,
//    RC_PROB_INIT=1024. Redefined locally rather than importing entropy.rs's
//    private consts, to keep this file's only entropy.rs dependency limited
//    to the two range-coder primitive structs.) ──────────────────────────────
const RC_PROB_INIT: u16 = 1024;

const RC_LEN_LO_BASE:  u32 = 2;
const RC_LEN_MID_BASE: u32 = 10;
const RC_LEN_HI_BASE:  u32 = 18;
const RC_LEN_SENTINEL: u32 = RC_LEN_HI_BASE + 255; // 273, same as v7

pub const RC_MAX_BACKREF_LEN: u32 = 272; // same ceiling as v7 (HI value 255 reserved)

/// Gate mirroring lib.rs's real `tokens_safe_for_v7`: v9 (like v7) can only
/// encode lengths up to RC_MAX_BACKREF_LEN (272) -- callers must check this
/// before trying v9 in an entropy-variant tournament, exactly as lib.rs
/// already does for v7 via `tokens_safe_for_v7`/`v7_ok`. Unlike v7's
/// version, this also checks `Token::RepRef` lengths (v7 never sees
/// RepRef at all, so its gate never needed to).
pub fn tokens_safe_for_v9(tokens: &[Token]) -> bool {
    tokens.iter().all(|t| match t {
        Token::Backref { length, .. } => *length <= RC_MAX_BACKREF_LEN,
        Token::RepRef  { length, .. } => *length <= RC_MAX_BACKREF_LEN,
        _ => true,
    })
}

// ── v9 prob layout ─────────────────────────────────────────────────────────
const PROB_MATCH:          usize = 0;
const PROB_IS_REP:         usize = 1;
const PROB_IS_REP0:        usize = 2;
const PROB_IS_REP1:        usize = 3;
const PROB_IS_REP2:        usize = 4;

const PROB_REP_LEN_CHOICE: usize = 5;   // 2 probs
const PROB_REP_LEN_LO:     usize = 7;   // 7 probs
const PROB_REP_LEN_MID:    usize = 14;  // 7 probs
const PROB_REP_LEN_HI:     usize = 21;  // 255 probs

const PROB_LEN_CHOICE:     usize = 276; // 2 probs
const PROB_LEN_LO:         usize = 278; // 7 probs
const PROB_LEN_MID:        usize = 285; // 7 probs
const PROB_LEN_HI:         usize = 292; // 255 probs

const PROB_DIST_SLOT:      usize = 547; // 4*63 = 252 probs
const PROB_DIST_EXTRA:     usize = 799; // 24 probs

const PROB_LIT:            usize = 823; // 8*255 = 2040 probs

pub const PROB_TOTAL_V9:   usize = 2863;

fn rc_init_v9() -> Vec<u16> { vec![RC_PROB_INIT; PROB_TOTAL_V9] }

#[inline]
fn length_class(length: u32) -> usize {
    match length { 0..=1 => 0, 2 => 1, 3 => 2, _ => 3 }
}

// ── Length coding: one generic implementation shared by the fresh-match
//    and rep-match tiers (identical shape, different prob banks -- avoids
//    duplicating entropy.rs's 3-tier LO/MID/HI logic twice).
//
//    HI tier caps at 254, NOT 255: value 255 (length RC_LEN_HI_BASE+255=273
//    = RC_LEN_SENTINEL) is reserved for the End marker on the FRESH branch
//    (mirrors v7's real rc_encode_length, which does the identical
//    `.min(254)` for the identical reason -- caught by cross-checking
//    against entropy.rs's actual code rather than assuming). Applied
//    uniformly to the REP branch too, even though End never rides the rep
//    branch and 255 would be safe to allow there: keeping ONE effective
//    ceiling (RC_MAX_BACKREF_LEN = 272) across every token kind is a
//    simpler contract for callers than tracking two different max lengths
//    depending on which opcode a given token happens to be. ─────────────────
fn rc_encode_len_generic(
    enc: &mut Rc7Enc, probs: &mut [u16],
    choice_base: usize, lo_base: usize, mid_base: usize, hi_base: usize,
    length: u32,
) {
    let v = length.saturating_sub(RC_LEN_LO_BASE);
    if v < 8 {
        enc.encode_bit(&mut probs[choice_base], 0);
        enc.encode_bittree(probs, lo_base, 3, v);
    } else {
        enc.encode_bit(&mut probs[choice_base], 1);
        let v2 = v - 8;
        if v2 < 8 {
            enc.encode_bit(&mut probs[choice_base + 1], 0);
            enc.encode_bittree(probs, mid_base, 3, v2);
        } else {
            enc.encode_bit(&mut probs[choice_base + 1], 1);
            let v3 = (v2 - 8).min(254);
            enc.encode_bittree(probs, hi_base, 8, v3);
        }
    }
}

fn rc_decode_len_generic(
    dec: &mut Rc7Dec, probs: &mut [u16],
    choice_base: usize, lo_base: usize, mid_base: usize, hi_base: usize,
) -> std::io::Result<u32> {
    if dec.decode_bit(&mut probs[choice_base])? == 0 {
        Ok(dec.decode_bittree(probs, lo_base, 3)? + RC_LEN_LO_BASE)
    } else if dec.decode_bit(&mut probs[choice_base + 1])? == 0 {
        Ok(dec.decode_bittree(probs, mid_base, 3)? + RC_LEN_MID_BASE)
    } else {
        Ok(dec.decode_bittree(probs, hi_base, 8)? + RC_LEN_HI_BASE)
    }
}

#[inline]
fn rc_encode_fresh_length(enc: &mut Rc7Enc, probs: &mut [u16], length: u32) {
    rc_encode_len_generic(enc, probs, PROB_LEN_CHOICE, PROB_LEN_LO, PROB_LEN_MID, PROB_LEN_HI, length);
}
#[inline]
fn rc_decode_fresh_length(dec: &mut Rc7Dec, probs: &mut [u16]) -> std::io::Result<u32> {
    rc_decode_len_generic(dec, probs, PROB_LEN_CHOICE, PROB_LEN_LO, PROB_LEN_MID, PROB_LEN_HI)
}
#[inline]
fn rc_encode_rep_length(enc: &mut Rc7Enc, probs: &mut [u16], length: u32) {
    rc_encode_len_generic(enc, probs, PROB_REP_LEN_CHOICE, PROB_REP_LEN_LO, PROB_REP_LEN_MID, PROB_REP_LEN_HI, length);
}
#[inline]
fn rc_decode_rep_length(dec: &mut Rc7Dec, probs: &mut [u16]) -> std::io::Result<u32> {
    rc_decode_len_generic(dec, probs, PROB_REP_LEN_CHOICE, PROB_REP_LEN_LO, PROB_REP_LEN_MID, PROB_REP_LEN_HI)
}

// ── Distance coding: identical to entropy.rs's rc_encode_distance /
//    rc_decode_distance, just against v9's shifted PROB_DIST_SLOT /
//    PROB_DIST_EXTRA bases. Only ever called for Token::Backref (fresh) --
//    Token::RepRef never codes a distance at all, see module doc comment. ──
fn rc_encode_distance(enc: &mut Rc7Enc, probs: &mut [u16], offset: u32, length: u32) {
    let (slot, extra_bits, extra_val) = offset_to_bucket(offset);
    let slot_c = slot.min(63);
    let lc = length_class(length);
    enc.encode_bittree(probs, PROB_DIST_SLOT + lc * 63, 6, slot_c);
    if extra_bits > 0 {
        let eb = extra_bits.min(24);
        enc.encode_direct(probs, PROB_DIST_EXTRA, eb, extra_val);
    }
}

fn rc_decode_distance(dec: &mut Rc7Dec, probs: &mut [u16], length: u32) -> std::io::Result<u32> {
    let lc = length_class(length);
    let slot = dec.decode_bittree(probs, PROB_DIST_SLOT + lc * 63, 6)?;
    let extra_bits = bucket_extra_bits(slot).min(24);
    let extra_val = if extra_bits > 0 {
        dec.decode_direct(probs, PROB_DIST_EXTRA, extra_bits)?
    } else { 0 };
    Ok(bucket_to_offset(slot, extra_val))
}

// ── Rep-slot cascade: is_rep0 / is_rep1 / is_rep2, a 3-bit-max binary
//    cascade selecting among MAX_RING_SLOTS=4 slots (matches LZMA's
//    is_rep0/is_rep1/is_rep2 shape adapted to REPS=4, same slot count). ──────
fn encode_rep_slot(enc: &mut Rc7Enc, probs: &mut [u16], slot: u8) {
    match slot {
        0 => enc.encode_bit(&mut probs[PROB_IS_REP0], 0),
        1 => {
            enc.encode_bit(&mut probs[PROB_IS_REP0], 1);
            enc.encode_bit(&mut probs[PROB_IS_REP1], 0);
        }
        2 => {
            enc.encode_bit(&mut probs[PROB_IS_REP0], 1);
            enc.encode_bit(&mut probs[PROB_IS_REP1], 1);
            enc.encode_bit(&mut probs[PROB_IS_REP2], 0);
        }
        3 => {
            enc.encode_bit(&mut probs[PROB_IS_REP0], 1);
            enc.encode_bit(&mut probs[PROB_IS_REP1], 1);
            enc.encode_bit(&mut probs[PROB_IS_REP2], 1);
        }
        _ => unreachable!("Token::RepRef.slot must be < MAX_RING_SLOTS (4), got {}", slot),
    }
}

fn decode_rep_slot(dec: &mut Rc7Dec, probs: &mut [u16]) -> std::io::Result<u8> {
    if dec.decode_bit(&mut probs[PROB_IS_REP0])? == 0 { return Ok(0); }
    if dec.decode_bit(&mut probs[PROB_IS_REP1])? == 0 { return Ok(1); }
    if dec.decode_bit(&mut probs[PROB_IS_REP2])? == 0 { return Ok(2); } else { return Ok(3); }
}

// ── Public encode / decode ────────────────────────────────────────────────────

/// Encode a token stream with v9's rep-aware adaptive binary range coder.
///
/// Unlike write_tokens_v7, `Token::RepRef` is a first-class input here --
/// do NOT resolve_ring() the stream before calling this.
pub fn write_tokens_v9(tokens: &[Token]) -> std::io::Result<Vec<u8>> {
    let mut probs = rc_init_v9();
    let mut enc = Rc7Enc::new();
    let mut prev: u8 = 0;

    for token in tokens {
        match token {
            Token::Lit { byte } => {
                enc.encode_bit(&mut probs[PROB_MATCH], 0);
                let ctx = PROB_LIT + (((prev as usize) >> 5) & 7) * 255;
                enc.encode_bittree(&mut probs, ctx, 8, *byte as u32);
                prev = *byte;
            }
            Token::RepRef { slot, length } => {
                if *slot as usize >= MAX_RING_SLOTS {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("v9: RepRef.slot {} >= MAX_RING_SLOTS {}", slot, MAX_RING_SLOTS)));
                }
                enc.encode_bit(&mut probs[PROB_MATCH], 1);
                enc.encode_bit(&mut probs[PROB_IS_REP], 1);
                encode_rep_slot(&mut enc, &mut probs, *slot);
                rc_encode_rep_length(&mut enc, &mut probs, *length);
            }
            Token::Backref { offset, length } => {
                enc.encode_bit(&mut probs[PROB_MATCH], 1);
                enc.encode_bit(&mut probs[PROB_IS_REP], 0);
                rc_encode_fresh_length(&mut enc, &mut probs, *length);
                rc_encode_distance(&mut enc, &mut probs, *offset, *length);
            }
            Token::End => {
                // End rides the fresh-match branch's length sentinel, exactly
                // like v7: is_match=1, is_rep=0, then fresh-length HI=255.
                enc.encode_bit(&mut probs[PROB_MATCH], 1);
                enc.encode_bit(&mut probs[PROB_IS_REP], 0);
                enc.encode_bit(&mut probs[PROB_LEN_CHOICE],     1);
                enc.encode_bit(&mut probs[PROB_LEN_CHOICE + 1], 1);
                enc.encode_bittree(&mut probs, PROB_LEN_HI, 8, 255);
            }
        }
    }

    Ok(enc.finish())
}

/// Decode a v9 range-coded token stream. Output is directly usable by
/// `decoder::reconstruct` -- same contract as `read_tokens_v7`'s output,
/// just with `Token::RepRef` preserved instead of pre-flattened.
pub fn read_tokens_v9(input: &[u8]) -> std::io::Result<Vec<Token>> {
    let mut probs = rc_init_v9();
    let mut dec = Rc7Dec::new(input)?;
    let mut tokens = Vec::new();
    let mut prev: u8 = 0;

    loop {
        let is_match = dec.decode_bit(&mut probs[PROB_MATCH])?;
        if is_match == 0 {
            let ctx = PROB_LIT + (((prev as usize) >> 5) & 7) * 255;
            let byte = dec.decode_bittree(&mut probs, ctx, 8)? as u8;
            tokens.push(Token::Lit { byte });
            prev = byte;
            continue;
        }

        let is_rep = dec.decode_bit(&mut probs[PROB_IS_REP])?;
        if is_rep == 1 {
            let slot = decode_rep_slot(&mut dec, &mut probs)?;
            let length = rc_decode_rep_length(&mut dec, &mut probs)?;
            tokens.push(Token::RepRef { slot, length });
        } else {
            let length = rc_decode_fresh_length(&mut dec, &mut probs)?;
            if length >= RC_LEN_SENTINEL {
                tokens.push(Token::End);
                break;
            }
            let offset = rc_decode_distance(&mut dec, &mut probs, length)?;
            if offset == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    "v9: decoded offset=0 (corrupt stream)"));
            }
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod v9_tests {
    use super::*;

    fn rt(tokens: &[Token]) -> Vec<Token> {
        let enc = write_tokens_v9(tokens).expect("v9 encode failed");
        read_tokens_v9(&enc).expect("v9 decode failed")
    }

    #[test]
    fn roundtrip_end_only() {
        let tokens = vec![Token::End];
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn roundtrip_literals_only() {
        let tokens: Vec<Token> = b"hello world".iter().map(|&b| Token::Lit { byte: b })
            .chain(std::iter::once(Token::End)).collect();
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn roundtrip_fresh_backref() {
        let tokens = vec![
            Token::Lit { byte: b'A' }, Token::Lit { byte: b'B' }, Token::Lit { byte: b'C' },
            Token::Backref { offset: 3, length: 6 },
            Token::End,
        ];
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn roundtrip_all_four_rep_slots() {
        for slot in 0u8..4 {
            let tokens = vec![
                Token::Lit { byte: b'X' },
                Token::RepRef { slot, length: 10 },
                Token::End,
            ];
            assert_eq!(rt(&tokens), tokens, "failed for slot {}", slot);
        }
    }

    #[test]
    fn roundtrip_mixed_stream() {
        let tokens = vec![
            Token::Lit { byte: b'a' },
            Token::Backref { offset: 100, length: 8 },
            Token::RepRef { slot: 0, length: 20 },
            Token::Lit { byte: b'z' },
            Token::RepRef { slot: 2, length: 4 },
            Token::Backref { offset: 50000, length: 15 },
            Token::RepRef { slot: 3, length: 272 }, // max length
            Token::End,
        ];
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn roundtrip_min_and_max_lengths() {
        let tokens = vec![
            Token::Lit { byte: b'A' }, Token::Lit { byte: b'B' },
            Token::Backref { offset: 2, length: 2 },   // RC_LEN_LO_BASE, minimum
            Token::RepRef { slot: 0, length: 2 },
            Token::Backref { offset: 2, length: RC_MAX_BACKREF_LEN }, // maximum
            Token::RepRef { slot: 0, length: RC_MAX_BACKREF_LEN },
            Token::End,
        ];
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn roundtrip_offset_at_various_bucket_boundaries() {
        // Exercise offset_to_bucket's small-offset direct path (<=4) and
        // its geometric-octave path (>4) with several boundary values.
        let offsets = [1u32, 2, 3, 4, 5, 6, 9, 100, 1000, 65535, 1_000_000];
        let mut tokens = vec![Token::Lit { byte: 0 }];
        for &off in &offsets {
            tokens.push(Token::Backref { offset: off, length: 5 });
            tokens.push(Token::Lit { byte: 1 }); // separator so matches don't chain oddly
        }
        tokens.push(Token::End);
        assert_eq!(rt(&tokens), tokens);
    }

    #[test]
    fn rejects_out_of_range_rep_slot() {
        let tokens = vec![Token::RepRef { slot: 4, length: 5 }, Token::End];
        assert!(write_tokens_v9(&tokens).is_err());
    }

    #[test]
    fn prob_layout_offsets_are_internally_consistent() {
        // Cross-check the doc comment's claimed layout against the actual
        // consts, so the two can't silently drift apart.
        assert_eq!(PROB_MATCH, 0);
        assert_eq!(PROB_IS_REP, 1);
        assert_eq!(PROB_IS_REP0, 2);
        assert_eq!(PROB_IS_REP1, 3);
        assert_eq!(PROB_IS_REP2, 4);
        assert_eq!(PROB_REP_LEN_CHOICE, 5);
        assert_eq!(PROB_REP_LEN_LO, 7);
        assert_eq!(PROB_REP_LEN_MID, 14);
        assert_eq!(PROB_REP_LEN_HI, 21);
        assert_eq!(PROB_LEN_CHOICE, 21 + 255);
        assert_eq!(PROB_LEN_LO, PROB_LEN_CHOICE + 2);
        assert_eq!(PROB_LEN_MID, PROB_LEN_LO + 7);
        assert_eq!(PROB_LEN_HI, PROB_LEN_MID + 7);
        assert_eq!(PROB_DIST_SLOT, PROB_LEN_HI + 255);
        assert_eq!(PROB_DIST_EXTRA, PROB_DIST_SLOT + 4 * 63);
        assert_eq!(PROB_LIT, PROB_DIST_EXTRA + 24);
        assert_eq!(PROB_TOTAL_V9, PROB_LIT + 8 * 255);
        assert_eq!(PROB_TOTAL_V9, 2863);
    }
}
#[cfg(test)]
mod v9_fuzz_tests {
    use super::*;
    use crate::opcode::Token;

    // Simple deterministic PRNG (no external crate deps needed for this
    // standalone harness) -- xorshift64.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, lo: u32, hi: u32) -> u32 {
            lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
        }
    }

    #[test]
    fn fuzz_random_token_streams_roundtrip() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for trial in 0..500 {
            let n_tokens = rng.range(1, 40);
            let mut tokens = Vec::new();
            for _ in 0..n_tokens {
                match rng.range(0, 2) {
                    0 => tokens.push(Token::Lit { byte: rng.range(0, 255) as u8 }),
                    1 => tokens.push(Token::Backref {
                        offset: rng.range(1, 2_000_000),
                        length: rng.range(2, RC_MAX_BACKREF_LEN),
                    }),
                    _ => tokens.push(Token::RepRef {
                        slot: rng.range(0, 3) as u8,
                        length: rng.range(2, RC_MAX_BACKREF_LEN),
                    }),
                }
            }
            tokens.push(Token::End);

            let enc = write_tokens_v9(&tokens)
                .unwrap_or_else(|e| panic!("trial {trial} encode failed: {e}"));
            let dec = read_tokens_v9(&enc)
                .unwrap_or_else(|e| panic!("trial {trial} decode failed: {e}"));
            assert_eq!(dec, tokens, "trial {trial} roundtrip mismatch");
        }
    }
  }
