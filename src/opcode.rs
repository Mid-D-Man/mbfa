// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/core.md, section "opcode.rs"
// ============================================================================
//! Fixed opcode vocabulary — known to both encoder and decoder.
//!
//! ## Ring-encoding mode (ring_flag = bit 1 of pair_flag header byte)
//!
//! When ring_flag = 0 (legacy):       When ring_flag = 1 (ring-active):
//!   "0"   → Backref                    "0"   → Backref     (unchanged)
//!   "10"  → Lit                        "10"  → Lit         (unchanged)
//!   "11"  → End (2-bit)                "110" → RepRef + 2-bit slot (0–3)
//!                                      "111" → End (3-bit)
//!
//! RepRef encodes a match at one of the 4 most-recently-used back-reference
//! distances (LRU ring buffer, mirrored identically by encoder and decoder).
//!
//! Cost of RepRef : 3 + 2 + lb = 5 + lb bits.
//! Cost of Backref: 1 + ob + lb bits.
//! Saving per ring hit: ob − 4 bits (e.g. ob=17 → saves 13 bits).

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Lit { byte: u8 },
    Backref { offset: u32, length: u32 },
    /// Ring back-reference. `slot` ∈ 0..MAX_RING_SLOTS−1, LRU order (0 = most recent).
    RepRef { slot: u8, length: u32 },
    End,
}

// ── Legacy opcode bit patterns (ring_flag = 0) ────────────────────────────────

pub const OPCODE_BACKREF_BITS: u32 = 1;
pub const OPCODE_BACKREF_VAL:  u32 = 0b0;

pub const OPCODE_LIT_BITS:     u32 = 2;
pub const OPCODE_LIT_VAL:      u32 = 0b10;

/// Legacy End — 2 bits "11". Used when ring_flag = 0.
pub const OPCODE_END_BITS:     u32 = 2;
pub const OPCODE_END_VAL:      u32 = 0b11;

pub const BYTE_BITS: u32 = 8;

// ── Ring-active additions (ring_flag = 1) ─────────────────────────────────────

pub const OPCODE_REPREF_BITS: u32 = 3;
pub const OPCODE_REPREF_VAL:  u32 = 0b110;

pub const REPREF_SLOT_BITS: u32 = 2;

pub const MAX_RING_SLOTS: usize = 4;

pub const OPCODE_END_RING_BITS: u32 = 3;
pub const OPCODE_END_RING_VAL:  u32 = 0b111;

// ── Adaptive offset configuration ─────────────────────────────────────────────

pub const OFFSET_BITS_MIN:     u32 = 7;
pub const OFFSET_BITS_MAX:     u32 = 24;
pub const OFFSET_BITS_DEFAULT: u32 = 15;

// ── Adaptive length configuration ─────────────────────────────────────────────

pub const LENGTH_BITS_MIN:     u32 = 8;
pub const LENGTH_BITS_MAX:     u32 = 24;
pub const LENGTH_BITS_DEFAULT: u32 = 8;

#[inline] pub fn max_offset(offset_bits: u32) -> usize { (1usize << offset_bits) - 1 }
#[inline] pub fn max_length(length_bits: u32) -> usize { (1usize << length_bits) - 1 }

// ── Token bit cost helpers ────────────────────────────────────────────────────

pub const LIT_TOTAL_BITS: u32 = OPCODE_LIT_BITS + BYTE_BITS;
pub const END_TOTAL_BITS: u32 = OPCODE_END_BITS;

pub fn backref_total_bits(offset_bits: u32, length_bits: u32) -> u32 {
    OPCODE_BACKREF_BITS + offset_bits + length_bits
}

/// RepRef cost: opcode (3) + slot (2) + length_bits = 5 + length_bits.
pub fn repref_total_bits(length_bits: u32) -> u32 {
    OPCODE_REPREF_BITS + REPREF_SLOT_BITS + length_bits
}

pub fn token_bit_cost(token: &Token, offset_bits: u32, length_bits: u32) -> u32 {
    match token {
        Token::Lit { .. }     => LIT_TOTAL_BITS,
        Token::Backref { .. } => backref_total_bits(offset_bits, length_bits),
        Token::RepRef { .. }  => repref_total_bits(length_bits),
        Token::End            => END_TOTAL_BITS,
    }
}

// ── Optimal parameter computation ─────────────────────────────────────────────

pub fn compute_optimal_offset_bits(tokens: &[Token]) -> u32 {
    let max_used = tokens.iter().filter_map(|t| {
        if let Token::Backref { offset, .. } = t { Some(*offset) } else { None }
    }).max().unwrap_or(0);

    if max_used == 0 { return OFFSET_BITS_MIN; }
    let bits_needed = 32 - max_used.leading_zeros();
    bits_needed.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
}

pub fn compute_optimal_length_bits(tokens: &[Token]) -> u32 {
    let max_used = tokens.iter().filter_map(|t| {
        match t {
            Token::Backref { length, .. } | Token::RepRef { length, .. } => Some(*length),
            _ => None,
        }
    }).max().unwrap_or(0);

    if max_used == 0 { return LENGTH_BITS_MIN; }
    let bits_needed = 32 - max_used.leading_zeros();
    bits_needed.clamp(LENGTH_BITS_MIN, LENGTH_BITS_MAX)
}

pub const ENTROPY_SAFE_MAX_LENGTH: u32 = 65280;

// ── Ring resolver ─────────────────────────────────────────────────────────────

/// Split matches longer than `max_len` into several equal-sized pieces, so a
/// stream produced with wide length fields can go through coders that cap
/// match length (the v9 range coder: 272). The expansion is unchanged.
///
/// The extra pieces repeat the offset of the first, so in a stream that uses
/// ring tokens they are `RepRef { slot: 0 }` (cheap to code); in a stream with
/// no ring tokens they are plain `Backref`s. Pieces are never shorter than
/// `max_len / 2`, so no piece falls below a coder's minimum length.
pub fn split_long_matches(tokens: &[Token], max_len: u32) -> Vec<Token> {
    let has_rep = tokens.iter().any(|t| matches!(t, Token::RepRef { .. }));
    let mut out = Vec::with_capacity(tokens.len());
    for t in tokens {
        match *t {
            Token::Backref { offset, length } if length > max_len => {
                let k = (length + max_len - 1) / max_len;
                let (base, rem) = (length / k, length % k);
                for i in 0..k {
                    let l = base + if i < rem { 1 } else { 0 };
                    out.push(if i == 0 || !has_rep {
                        Token::Backref { offset, length: l }
                    } else {
                        Token::RepRef { slot: 0, length: l }
                    });
                }
            }
            Token::RepRef { slot, length } if length > max_len => {
                let k = (length + max_len - 1) / max_len;
                let (base, rem) = (length / k, length % k);
                for i in 0..k {
                    let l = base + if i < rem { 1 } else { 0 };
                    out.push(Token::RepRef { slot: if i == 0 { slot } else { 0 }, length: l });
                }
            }
            ref other => out.push(other.clone()),
        }
    }
    out
}

/// Convert all `Token::RepRef` to `Token::Backref` by maintaining an LRU ring
/// state identical to the decoder's ring.
pub fn resolve_ring(tokens: &[Token]) -> Vec<Token> {
    let mut result     = Vec::with_capacity(tokens.len());
    let mut ring       = [0u32; MAX_RING_SLOTS];
    let mut ring_count = 0usize;

    for token in tokens {
        match token {
            Token::Backref { offset, .. } => {
                if ring_count == 0 || ring[0] != *offset {
                    if ring_count < MAX_RING_SLOTS { ring_count += 1; }
                    for j in (1..ring_count).rev() { ring[j] = ring[j - 1]; }
                    ring[0] = *offset;
                }
                result.push(token.clone());
            }
            Token::RepRef { slot, length } => {
                let s = *slot as usize;
                if s < ring_count && ring[s] != 0 {
                    let offset = ring[s];
                    for j in (1..=s).rev() { ring[j] = ring[j - 1]; }
                    ring[0] = offset;
                    result.push(Token::Backref { offset, length: *length });
                } else {
                    eprintln!(
                        "resolve_ring: invalid RepRef slot {} (ring_count={})",
                        s, ring_count
                    );
                }
            }
            other => result.push(other.clone()),
        }
    }

    result
                                                    }


#[cfg(test)]
mod split_tests {
    use super::*;

    fn expand(tokens: &[Token]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for t in resolve_ring(tokens) {
            match t {
                Token::Lit { byte } => out.push(byte),
                Token::Backref { offset, length } => {
                    for _ in 0..length { let b = out[out.len() - offset as usize]; out.push(b); }
                }
                _ => {}
            }
        }
        out
    }

    fn stream(with_rep: bool) -> Vec<Token> {
        let mut t: Vec<Token> = (0..40u8).map(|b| Token::Lit { byte: b.wrapping_mul(7) }).collect();
        t.push(Token::Backref { offset: 30, length: 900 });       // overlapping, long
        t.push(Token::Lit { byte: 9 });
        t.push(Token::Backref { offset: 17, length: 273 });       // just over the limit
        t.push(Token::Backref { offset: 5, length: 272 });        // exactly the limit: untouched
        if with_rep {
            t.push(Token::RepRef { slot: 0, length: 700 });
            t.push(Token::Backref { offset: 120, length: 3 });
            t.push(Token::RepRef { slot: 1, length: 1000 });
        }
        t.push(Token::End);
        t
    }

    #[test]
    fn splitting_keeps_the_expansion_and_the_length_limit() {
        for &with_rep in &[false, true] {
            let t = stream(with_rep);
            let s = split_long_matches(&t, 272);
            assert_eq!(expand(&s), expand(&t), "with_rep {with_rep}");
            for tok in &s {
                match tok {
                    Token::Backref { length, .. } | Token::RepRef { length, .. } => {
                        assert!(*length <= 272, "piece length {length}");
                    }
                    _ => {}
                }
            }
            // every piece cut from a long match is at least half the limit
            let longest_in = |ts: &[Token]| ts.iter().filter_map(|t| match t {
                Token::Backref { length, .. } | Token::RepRef { length, .. } => Some(*length), _ => None }).max();
            assert!(longest_in(&t).unwrap() > 272);
            assert!(longest_in(&s).unwrap() <= 272);
        }
    }

    #[test]
    fn nothing_to_split_is_unchanged() {
        let t = vec![
            Token::Lit { byte: 1 }, Token::Lit { byte: 2 },
            Token::Backref { offset: 2, length: 272 }, Token::End,
        ];
        assert_eq!(split_long_matches(&t, 272), t);
    }
}
