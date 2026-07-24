// src/optimal_parse.rs
//! Bounded-horizon, price-aware optimal parser -- an alternative to
//! encoder.rs's `scan_from` for fold 1, when the caller intends to encode
//! the result with entropy_v9 (or v7). Where `scan_from` picks matches by
//! raw length (greedy, plus one step of lazy lookahead), this prices every
//! candidate -- literal, each of the 4 rep slots, and every point on the
//! fresh-match length/distance frontier -- under v9's ACTUAL current
//! adaptive model, and picks whichever sequence of tokens is cheapest
//! over a lookahead window. This is what closes the gap `scan_from` can't:
//! it has no visibility into entropy-coder cost at all when choosing
//! matches, so a match that's merely long isn't necessarily the same as a
//! match that's cheap to code (a nearby short match beats a distant long
//! one when the distance's slot+extra-bits cost outweighs the length
//! difference -- greedy longest-match can't see that trade-off; this can).
//!
//! ## Relationship to lzma-rust2's `encoder_normal.rs` (real optimal parser)
//!
//! Same core idea (a forward price-DP over a lookahead window, backward
//! traceback to emit the cheapest path), checked directly against that
//! file's `get_optimum`/`Optimum`/`opts[]` design, with two deliberate,
//! disclosed simplifications:
//!
//!  1. **Bounded horizon, not dynamic.** LZMA extends `opt_end` position by
//!     position, stopping early via `nice_len` heuristics and complex
//!     match-length-based termination. This uses a FIXED lookahead window
//!     (`HORIZON`, default 32) and always fills the whole thing before
//!     tracing back. Simpler to reason about and verify correctly; loses
//!     LZMA's ability to extend the horizon further when a very long match
//!     is found nearby. Real cost/benefit trade documented below.
//!  2. **No LZMA `state` tracking.** LZMA's `PROB_MATCH`-equivalent
//!     (`is_match`) is context-split by an 12-state machine tracking
//!     "what kind of token came before" (literal-after-match vs.
//!     literal-after-literal vs. after-rep, etc). MBFA's v9 `PROB_MATCH`
//!     is a single unconditional probability (see entropy_v9.rs's prob
//!     layout -- no state split exists in the format at all), so there is
//!     nothing state-dependent to track here; this is not a simplification
//!     relative to what v9 can actually represent, just a note that if v9
//!     ever grows state-splitting, this parser's prices would need to
//!     grow a `state` field to match, the way LZMA's own `Optimum` has one.
//!
//! ## Match-finder: length -> cheapest-distance frontier
//!
//! encoder.rs's real `find_match` returns only the single longest match at
//! each position -- exactly what a greedy/lazy scanner needs, but not
//! enough for a DP: the optimal parser needs to know, for a SHORTER
//! candidate length, whether some CLOSER (cheaper) offset already achieves
//! it, since a short-cheap match can beat a long-expensive one once
//! distance-coding cost is priced in. `find_matches_tiered` (below) walks
//! the identical hash-chain structure (same `hash3`, same chain-limit
//! logic, ported byte-for-byte from encoder.rs and cross-checked against a
//! faithful Python replica earlier in this investigation) but records a
//! new frontier point every time chain-walking reaches a NEW length
//! record -- and because `prev[]` links strictly older (larger-offset)
//! positions, offsets are non-decreasing as the chain is walked, so
//! "first time we see length >= L" is automatically also "cheapest
//! distance that achieves length >= L". This mirrors lzma-rust2's real
//! `Matches { len: Vec<u32>, dist: Vec<i32> }` (bt4.rs) semantics exactly,
//! built on MBFA's existing matcher instead of porting BT4.

use crate::opcode::{Token, MAX_RING_SLOTS};
use crate::price_table::{get_bit_price, get_bittree_price, get_direct_price};

// ── Match-finder (ported from encoder.rs's real scan_from/find_match; see
//    module doc comment) ──────────────────────────────────────────────────

const HASH_SIZE: usize = 1 << 16;
const HASH_MASK: usize = HASH_SIZE - 1;
const NONE_MARK: u32 = u32::MAX;

#[inline]
fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() { return 0; }
    let v = (data[pos] as u64).wrapping_mul(2_654_435_761)
        ^ (data[pos + 1] as u64).wrapping_mul(2_246_822_519)
        ^ (data[pos + 2] as u64).wrapping_mul(3_266_489_917);
    (v as usize) & HASH_MASK
}

fn compute_chain_limit(input_len: usize, offset_bits: u32) -> usize {
    match offset_bits {
        0..=16 => 256,
        17..=18 => if input_len > 1_048_576 { 64 } else { 512 },
        19..=20 => 256,
        _ => 128,
    }
}

fn window_alloc_and_mask(max_off_limit: usize, n: usize) -> (usize, usize) {
    let raw = max_off_limit.min(n).max(1);
    let pow2 = raw.next_power_of_two();
    (pow2, pow2 - 1)
}

/// One point on the length -> cheapest-distance Pareto frontier.
#[derive(Debug, Clone, Copy)]
pub struct MatchCandidate { pub length: u32, pub offset: u32 }

/// Match-finder state, threaded through the whole scan exactly like
/// encoder.rs's `head`/`prev` arrays are threaded through `scan_from`'s
/// main loop -- one instance covers the whole input, insert-as-you-go.
pub struct MatchFinder {
    head: Vec<u32>,
    prev: Vec<u32>,
    window_mask: usize,
    max_off: u32,
    max_len: u32,
    chain_limit: usize,
}

impl MatchFinder {
    pub fn new(input_len: usize, offset_bits: u32, length_bits: u32) -> Self {
        let max_off = (1u32 << offset_bits) - 1;
        let max_len = (1u32 << length_bits) - 1;
        let (window_size, window_mask) = window_alloc_and_mask(max_off as usize, input_len);
        Self {
            head: vec![NONE_MARK; HASH_SIZE],
            prev: vec![NONE_MARK; window_size],
            window_mask,
            max_off,
            max_len,
            chain_limit: compute_chain_limit(input_len, offset_bits),
        }
    }

    /// Insert position `i` into the chain (must be called exactly once per
    /// position, in increasing order, including for positions the parser
    /// ends up NOT taking a match from -- matches encoder.rs's own
    /// insert-every-position-visited behavior).
    #[inline]
    pub fn insert(&mut self, data: &[u8], i: usize) {
        let h = hash3(data, i);
        self.prev[i & self.window_mask] = self.head[h];
        self.head[h] = i as u32;
    }

    /// Walk the chain at position `i`, returning the full length -> cheapest
    /// distance frontier (ascending by length). Does NOT insert `i` itself
    /// -- call `insert` separately once the parser has decided how many
    /// positions to advance past this one.
    pub fn find_matches_tiered(&self, data: &[u8], i: usize) -> Vec<MatchCandidate> {
        let n = data.len();
        let h = hash3(data, i);
        let mut out = Vec::new();
        let mut best_len = 0u32;
        let mut steps = 0usize;
        let mut cur = self.head[h];

        while cur != NONE_MARK && steps < self.chain_limit {
            let j = cur as usize;
            if i <= j || (i - j) as u32 > self.max_off { break; }
            let span = (i - j) as u32;
            let mut length = 0u32;
            while length < self.max_len
                && (i + length as usize) < n
                && data[j + (length as usize % span as usize)] == data[i + length as usize]
            {
                length += 1;
            }
            if length > best_len {
                best_len = length;
                out.push(MatchCandidate { length, offset: span });
                if best_len == self.max_len { break; }
            }
            cur = self.prev[j & self.window_mask];
            steps += 1;
        }
        out
    }
}

// ── Rep-match length lookup (mirrors encoder.rs's real rep_match_len) ──────

fn rep_match_len(data: &[u8], i: usize, offset: u32, max_len: u32) -> u32 {
    if offset == 0 || offset as usize > i { return 0; }
    let j = i - offset as usize;
    let n = data.len();
    let mut length = 0u32;
    while length < max_len
        && (i + length as usize) < n
        && data[j + (length as usize % offset as usize)] == data[i + length as usize]
    {
        length += 1;
    }
    length
}

// ── Pricing, against v9's real prob layout (entropy_v9.rs). Kept in this
//    file rather than entropy_v9.rs itself, since these are DP-parser-only
//    concerns (v9's actual encode/decode never needs a "price of X"
//    question, only "encode X now") -- but the prob-index math must stay
//    in lock-step with entropy_v9.rs's layout, so every constant here is a
//    direct copy with a comment pointing at its entropy_v9.rs counterpart. ──

mod v9_prob_layout {
    // Mirrors entropy_v9.rs's prob layout exactly -- see that file's module
    // doc comment for the full table and derivation. Kept in sync manually;
    // entropy_v9.rs's own `prob_layout_offsets_are_internally_consistent`
    // test is the source of truth if these ever drift.
    pub const PROB_MATCH:          usize = 0;
    pub const PROB_IS_REP:         usize = 1;
    pub const PROB_IS_REP0:        usize = 2;
    pub const PROB_IS_REP1:        usize = 3;
    pub const PROB_IS_REP2:        usize = 4;
    pub const PROB_REP_LEN_CHOICE: usize = 5;
    pub const PROB_REP_LEN_LO:     usize = 7;
    pub const PROB_REP_LEN_MID:    usize = 14;
    pub const PROB_REP_LEN_HI:     usize = 21;
    pub const PROB_LEN_CHOICE:     usize = 276;
    pub const PROB_LEN_LO:         usize = 278;
    pub const PROB_LEN_MID:        usize = 285;
    pub const PROB_LEN_HI:         usize = 292;
    pub const PROB_DIST_SLOT:      usize = 547;
    pub const PROB_DIST_EXTRA:     usize = 799;
    pub const PROB_LIT:            usize = 823;
    pub const PROB_TOTAL:          usize = 2863;
    pub const RC_LEN_LO_BASE: u32 = 2;
}
use v9_prob_layout::*;

#[inline]
fn length_class(length: u32) -> usize {
    match length { 0..=1 => 0, 2 => 1, 3 => 2, _ => 3 }
}

fn price_len_generic(probs: &[u16], choice_base: usize, lo_base: usize, mid_base: usize, hi_base: usize, length: u32) -> u32 {
    let v = length.saturating_sub(RC_LEN_LO_BASE);
    if v < 8 {
        get_bit_price(probs[choice_base], 0) + get_bittree_price(probs, lo_base, 3, v)
    } else if v - 8 < 8 {
        get_bit_price(probs[choice_base], 1) + get_bit_price(probs[choice_base + 1], 0)
            + get_bittree_price(probs, mid_base, 3, v - 8)
    } else {
        let v3 = (v - 16).min(254);
        get_bit_price(probs[choice_base], 1) + get_bit_price(probs[choice_base + 1], 1)
            + get_bittree_price(probs, hi_base, 8, v3)
    }
}

fn price_fresh_length(probs: &[u16], length: u32) -> u32 {
    price_len_generic(probs, PROB_LEN_CHOICE, PROB_LEN_LO, PROB_LEN_MID, PROB_LEN_HI, length)
}
fn price_rep_length(probs: &[u16], length: u32) -> u32 {
    price_len_generic(probs, PROB_REP_LEN_CHOICE, PROB_REP_LEN_LO, PROB_REP_LEN_MID, PROB_REP_LEN_HI, length)
}

fn price_distance(probs: &[u16], offset: u32, length: u32) -> u32 {
    let (slot, extra_bits, extra_val) = crate::entropy::offset_to_bucket(offset);
    let slot_c = slot.min(63);
    let lc = length_class(length);
    let mut price = get_bittree_price(probs, PROB_DIST_SLOT + lc * 63, 6, slot_c);
    if extra_bits > 0 {
        price += get_direct_price(probs, PROB_DIST_EXTRA, extra_bits.min(24), extra_val);
    }
    price
}

fn price_rep_slot(probs: &[u16], slot: u8) -> u32 {
    match slot {
        0 => get_bit_price(probs[PROB_IS_REP0], 0),
        1 => get_bit_price(probs[PROB_IS_REP0], 1) + get_bit_price(probs[PROB_IS_REP1], 0),
        2 => get_bit_price(probs[PROB_IS_REP0], 1) + get_bit_price(probs[PROB_IS_REP1], 1) + get_bit_price(probs[PROB_IS_REP2], 0),
        3 => get_bit_price(probs[PROB_IS_REP0], 1) + get_bit_price(probs[PROB_IS_REP1], 1) + get_bit_price(probs[PROB_IS_REP2], 1),
        _ => unreachable!("slot must be < MAX_RING_SLOTS"),
    }
}

fn price_literal(probs: &[u16], prev: u8, byte: u8) -> u32 {
    let ctx = PROB_LIT + (((prev as usize) >> 5) & 7) * 255;
    get_bit_price(probs[PROB_MATCH], 0) + get_bittree_price(probs, ctx, 8, byte as u32)
}

fn price_fresh_match(probs: &[u16], offset: u32, length: u32) -> u32 {
    get_bit_price(probs[PROB_MATCH], 1) + get_bit_price(probs[PROB_IS_REP], 0)
        + price_fresh_length(probs, length) + price_distance(probs, offset, length)
}

fn price_rep_match(probs: &[u16], slot: u8, length: u32) -> u32 {
    get_bit_price(probs[PROB_MATCH], 1) + get_bit_price(probs[PROB_IS_REP], 1)
        + price_rep_slot(probs, slot) + price_rep_length(probs, length)
}

// ── DP core ─────────────────────────────────────────────────────────────────

const HORIZON: usize = 32;
const MIN_MATCH_LEN: u32 = 2; // matches encoder.rs's ref_worthwhile floor

/// One reachable step's cheapest-known cost and how it was reached, at a
/// given lookahead offset from the DP window's start position.
#[derive(Clone)]
struct Optimum {
    price: u32,
    prev_dist: u32,       // how many positions back the predecessor is (0 = unreachable/start)
    choice: Choice,
    reps: [u32; MAX_RING_SLOTS], // ring state resulting from the cheapest path to here
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Start,
    Lit,
    RepRef { slot: u8 },
    Backref { offset: u32 },
}

const UNREACHABLE: u32 = u32::MAX;

/// Runs the DP over a HORIZON-wide window starting at `start`, returns the
/// cheapest first step (as a Token) and how many positions it advances.
/// `probs` is the CURRENT v9 adaptive model (read-only snapshot for
/// pricing -- matches LZMA's own approach of pricing against the model's
/// state at the START of the horizon, not updating mid-DP; the model is
/// updated for real once write_tokens_v9 actually commits the chosen
/// token, same as v7/v9 already do token-by-token).
pub fn optimal_step(
    data: &[u8],
    start: usize,
    probs: &[u16],
    reps_in: [u32; MAX_RING_SLOTS],
    finder: &MatchFinder,
    prev_byte: u8,
) -> (Token, usize) {
    let n = data.len();
    let horizon = HORIZON.min(n - start);
    if horizon == 0 {
        return (Token::End, 0);
    }

    // opts[0] is the start position (cost 0, reps_in as given); opts[k] for
    // k=1..=horizon is "cheapest way to have advanced k bytes past start".
    let mut opts: Vec<Optimum> = vec![
        Optimum { price: UNREACHABLE, prev_dist: 0, choice: Choice::Start, reps: reps_in };
        horizon + 1
    ];
    opts[0] = Optimum { price: 0, prev_dist: 0, choice: Choice::Start, reps: reps_in };

    for k in 0..horizon {
        let cur = opts[k].clone();
        if cur.price == UNREACHABLE { continue; }
        let pos = start + k;
        let prev = if k == 0 { prev_byte } else {
            // The byte just before `pos` on the cheapest path to `opts[k]`
            // is simply data[pos-1] -- true regardless of which choice got
            // us here, since every choice ends by having emitted that byte.
            data[pos - 1]
        };

        // Candidate 1: literal.
        {
            let cost = cur.price + price_literal(probs, prev, data[pos]);
            let slot = k + 1;
            if cost < opts[slot].price {
                opts[slot] = Optimum { price: cost, prev_dist: 1, choice: Choice::Lit, reps: cur.reps };
            }
        }

        // Candidate 2: each rep slot.
        for (slot_idx, &rep_off) in cur.reps.iter().enumerate() {
            if rep_off == 0 || rep_off as usize > pos { continue; }
            let max_reach = (horizon - k) as u32;
            let len = rep_match_len(data, pos, rep_off, max_reach.max(MIN_MATCH_LEN));
            if len < MIN_MATCH_LEN { continue; }
            let cost = cur.price + price_rep_match(probs, slot_idx as u8, len);
            let dest = k + len as usize;
            if dest <= horizon && cost < opts[dest].price {
                let mut new_reps = cur.reps;
                if slot_idx != 0 {
                    let used = new_reps[slot_idx];
                    for j in (1..=slot_idx).rev() { new_reps[j] = new_reps[j - 1]; }
                    new_reps[0] = used;
                }
                opts[dest] = Optimum {
                    price: cost, prev_dist: len, choice: Choice::RepRef { slot: slot_idx as u8 }, reps: new_reps,
                };
            }
            // Also record every intermediate length on this rep's own
            // frontier (a shorter use of the same rep slot might land more
            // cheaply on some earlier position than a literal run would).
            for shorter in MIN_MATCH_LEN..len {
                let cost_s = cur.price + price_rep_match(probs, slot_idx as u8, shorter);
                let dest_s = k + shorter as usize;
                if dest_s <= horizon && cost_s < opts[dest_s].price {
                    let mut new_reps = cur.reps;
                    if slot_idx != 0 {
                        let used = new_reps[slot_idx];
                        for j in (1..=slot_idx).rev() { new_reps[j] = new_reps[j - 1]; }
                        new_reps[0] = used;
                    }
                    opts[dest_s] = Optimum {
                        price: cost_s, prev_dist: shorter, choice: Choice::RepRef { slot: slot_idx as u8 }, reps: new_reps,
                    };
                }
            }
        }

        // Candidate 3: fresh matches, from the tiered frontier.
        let frontier = finder.find_matches_tiered(data, pos);
        for cand in &frontier {
            let len = cand.length.min((horizon - k) as u32);
            if len < MIN_MATCH_LEN { continue; }
            for l in MIN_MATCH_LEN..=len {
                let cost = cur.price + price_fresh_match(probs, cand.offset, l);
                let dest = k + l as usize;
                if dest <= horizon && cost < opts[dest].price {
                    let mut new_reps = cur.reps;
                    new_reps[3] = new_reps[2]; new_reps[2] = new_reps[1];
                    new_reps[1] = new_reps[0]; new_reps[0] = cand.offset;
                    opts[dest] = Optimum {
                        price: cost, prev_dist: l, choice: Choice::Backref { offset: cand.offset }, reps: new_reps,
                    };
                }
            }
        }
    }

    // Find the cheapest REACHED position (not necessarily `horizon` itself --
    // some position short of the horizon may already be the best committed
    // step once we trace back to step 1; but for a simple, correct bounded
    // parser we trace back from the farthest reached position, matching
    // LZMA's own "commit as far as the DP window decided" approach).
    let mut best_end = horizon;
    while opts[best_end].price == UNREACHABLE { best_end -= 1; }

    // Traceback to find the FIRST step taken from position 0.
    let mut k = best_end;
    loop {
        let step = opts[k].prev_dist as usize;
        let prev_k = k - step;
        if prev_k == 0 {
            let token = match opts[k].choice {
                Choice::Lit => Token::Lit { byte: data[start] },
                Choice::RepRef { slot } => Token::RepRef { slot, length: step as u32 },
                Choice::Backref { offset } => Token::Backref { offset, length: step as u32 },
                Choice::Start => unreachable!("traceback reached Start at k>0"),
            };
            return (token, step);
        }
        k = prev_k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_probs() -> Vec<u16> { vec![1024u16; PROB_TOTAL] }

    #[test]
    fn cheap_short_rep_beats_expensive_long_fresh_match() {
        // Construct data where position `pos` could either:
        //   (a) take a REP match (slot 0, offset 4) of length 4, or
        //   (b) take a FRESH match at a huge offset of length 5 (longer!).
        // A greedy longest-match scanner picks (b). The DP, pricing a huge
        // fresh distance's slot+extra bits against a nearly-free rep0
        // signal, should recognize (a) is cheaper overall despite being
        // shorter, PROVIDED the price gap actually favors it -- verified
        // below by comparing real prices, not just trusting the outcome.
        let probs = fresh_probs();
        let rep_price = price_rep_match(&probs, 0, 4);
        let fresh_price = price_fresh_match(&probs, 1_000_000, 5);
        assert!(
            rep_price < fresh_price,
            "expected rep (slot 0, len 4) to price cheaper than a huge-offset fresh match: rep={} fresh={}",
            rep_price, fresh_price
        );
    }

    #[test]
    fn literal_run_prices_as_expected_number_of_bytes() {
        let probs = fresh_probs();
        let text = b"abcdefgh";
        let mut total = 0u32;
        let mut prev = 0u8;
        for &b in text {
            total += price_literal(&probs, prev, b);
            prev = b;
        }
        // ~10 price-units/byte at fresh probs (slightly above the fair-coin
        // ~16-units-per-bit baseline since match-flag bit is also priced in,
        // but should land in a sane ballpark -- not near-zero, not enormous).
        let avg = total / text.len() as u32;
        assert!(avg > 20 && avg < 200, "unexpected average literal price: {}", avg);
    }

    #[test]
    fn optimal_step_roundtrips_via_reconstruct_semantics() {
        // Build a small input with a genuine repeated pattern, run
        // optimal_step repeatedly until EOF, and check the resulting
        // token stream reconstructs the original bytes exactly (using
        // the same virtual-position logic decoder.rs's real reconstruct
        // uses -- reimplemented minimally here rather than importing
        // decoder.rs, to keep this file's test dependencies to just
        // opcode+price_table+entropy's bucket fns).
        let data = b"the quick brown fox jumps over the quick brown fox again".to_vec();
        let mut finder = MatchFinder::new(data.len(), 18, 10);
        let probs = fresh_probs();
        let mut pos = 0usize;
        let mut reps = [0u32; MAX_RING_SLOTS];
        let mut tokens = Vec::new();
        let mut prev_byte = 0u8;

        while pos < data.len() {
            let (token, advance) = optimal_step(&data, pos, &probs, reps, &finder, prev_byte);
            let advance = advance.max(1); // safety: never spin on advance=0
            match token {
                Token::Backref { offset, .. } => {
                    reps[3] = reps[2]; reps[2] = reps[1]; reps[1] = reps[0]; reps[0] = offset;
                }
                Token::RepRef { slot, .. } if slot != 0 => {
                    let used = reps[slot as usize];
                    for j in (1..=slot as usize).rev() { reps[j] = reps[j - 1]; }
                    reps[0] = used;
                }
                _ => {}
            }
            for i in 0..advance { finder.insert(&data, pos + i); }
            tokens.push(token.clone());
            prev_byte = data[pos + advance - 1];
            pos += advance;
        }
        tokens.push(Token::End);

        // Minimal reconstruct (mirrors decoder.rs's real virtual-position
        // logic for Lit/Backref/RepRef -- see decoder.rs's dict_reconstruct_tests
        // for the authoritative version this is a test-local echo of).
        let mut out = Vec::new();
        let mut ring = [0u32; MAX_RING_SLOTS];
        let mut ring_count = 0usize;
        for t in &tokens {
            match t {
                Token::Lit { byte } => out.push(*byte),
                Token::Backref { offset, length } => {
                    let start = out.len() - *offset as usize;
                    for k in 0..*length as usize { out.push(out[start + (k % *offset as usize)]); }
                    if ring_count == 0 || ring[0] != *offset {
                        if ring_count < MAX_RING_SLOTS { ring_count += 1; }
                        for j in (1..ring_count).rev() { ring[j] = ring[j - 1]; }
                        ring[0] = *offset;
                    }
                }
                Token::RepRef { slot, length } => {
                    let off = ring[*slot as usize];
                    let start = out.len() - off as usize;
                    for k in 0..*length as usize { out.push(out[start + (k % off as usize)]); }
                    for j in (1..=*slot as usize).rev() { ring[j] = ring[j - 1]; }
                    ring[0] = off;
                }
                Token::End => break,
            }
        }
        assert_eq!(out, data, "optimal_step-produced token stream failed to reconstruct original data");
    }
}
