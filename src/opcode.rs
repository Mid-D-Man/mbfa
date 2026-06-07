// src/opcode.rs
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

/// Convert all `Token::RepRef` to `Token::Backref` by maintaining an LRU ring
/// state identical to the decoder's ring.
pub fn resolve_ring(tokens: &[Token]) -> Vec<Token> {
    let mut result     = Vec::with_capacity(tokens.len());
    let mut ring       = [0u32; MAX_RING_SLOTS];
    let mut ring_count = 0usize;

    for token in tokens {
        match token {
            // P6 fix: `..` instead of binding `length` — only `offset` is used
            // for the ring update; `length` is implicitly preserved via token.clone().
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
