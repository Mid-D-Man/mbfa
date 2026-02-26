// src/opcode.rs
//! Fixed opcode vocabulary — known to both encoder and decoder.
//! offset_bits and length_bits are both adaptive at runtime.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Lit { byte: u8 },
    Backref { offset: u32, length: u32 },
    End,
}

// ── Fixed opcode bit patterns ─────────────────────────────────────────────────
pub const OPCODE_BACKREF_BITS: u32 = 1;
pub const OPCODE_BACKREF_VAL:  u32 = 0b0;

pub const OPCODE_LIT_BITS:     u32 = 2;
pub const OPCODE_LIT_VAL:      u32 = 0b10;

pub const OPCODE_END_BITS:     u32 = 2;
pub const OPCODE_END_VAL:      u32 = 0b11;

pub const BYTE_BITS: u32 = 8;

// ── Adaptive offset configuration ─────────────────────────────────────────────
/// Minimum offset bits — 127 byte lookback.
pub const OFFSET_BITS_MIN:     u32 = 7;
/// Maximum offset bits — ~16MB lookback for large files.
pub const OFFSET_BITS_MAX:     u32 = 24;
/// Default fallback when header is missing or malformed.
pub const OFFSET_BITS_DEFAULT: u32 = 15;

// ── Adaptive length configuration ─────────────────────────────────────────────
/// Minimum length bits — 255 byte max copy (matches old fixed behaviour).
pub const LENGTH_BITS_MIN:     u32 = 8;
/// Maximum length bits — 32767 byte max copy.
/// Capped at 15 so length symbols (255+len) stay within u16 for table serialisation.
pub const LENGTH_BITS_MAX:     u32 = 15;
/// Default fallback.
pub const LENGTH_BITS_DEFAULT: u32 = 8;

/// Maximum lookback window for a given offset_bits value.
#[inline] pub fn max_offset(offset_bits: u32) -> usize { (1usize << offset_bits) - 1 }

/// Maximum copy length for a given length_bits value.
#[inline] pub fn max_length(length_bits: u32) -> usize { (1usize << length_bits) - 1 }

// ── Token bit cost helpers ────────────────────────────────────────────────────
pub const LIT_TOTAL_BITS: u32 = OPCODE_LIT_BITS + BYTE_BITS; // 10 — fixed
pub const END_TOTAL_BITS: u32 = OPCODE_END_BITS;               // 2  — fixed

pub fn backref_total_bits(offset_bits: u32, length_bits: u32) -> u32 {
    OPCODE_BACKREF_BITS + offset_bits + length_bits
}

pub fn token_bit_cost(token: &Token, offset_bits: u32, length_bits: u32) -> u32 {
    match token {
        Token::Lit { .. }     => LIT_TOTAL_BITS,
        Token::Backref { .. } => backref_total_bits(offset_bits, length_bits),
        Token::End            => END_TOTAL_BITS,
    }
}

/// Minimum offset_bits needed to represent all offsets in a token stream.
pub fn compute_optimal_offset_bits(tokens: &[Token]) -> u32 {
    let max_used = tokens.iter().filter_map(|t| {
        if let Token::Backref { offset, .. } = t { Some(*offset) } else { None }
    }).max().unwrap_or(0);

    if max_used == 0 { return OFFSET_BITS_MIN; }
    let bits_needed = 32 - max_used.leading_zeros();
    bits_needed.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
}

/// Minimum length_bits needed to represent all copy lengths in a token stream.
pub fn compute_optimal_length_bits(tokens: &[Token]) -> u32 {
    let max_used = tokens.iter().filter_map(|t| {
        if let Token::Backref { length, .. } = t { Some(*length) } else { None }
    }).max().unwrap_or(0);

    if max_used == 0 { return LENGTH_BITS_MIN; }
    let bits_needed = 32 - max_used.leading_zeros();
    bits_needed.clamp(LENGTH_BITS_MIN, LENGTH_BITS_MAX)
}
