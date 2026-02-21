//! Fixed opcode vocabulary — known to both encoder and decoder.
//! Never transmitted (except offset_bits, which goes in the file header).
//! Single source of truth for all bit widths.

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

pub const LENGTH_BITS: u32 = 8;   // 255 byte max copy length — fixed
pub const BYTE_BITS:   u32 = 8;

// ── Adaptive offset configuration ────────────────────────────────────────────
/// Minimum offset bits — covers 127 byte lookback, smallest sensible window.
pub const OFFSET_BITS_MIN: u32 = 7;
/// Maximum offset bits — 131071 byte (~128KB) lookback for large files.
pub const OFFSET_BITS_MAX: u32 = 17;
/// Default (used when no token scan is available, e.g. in tests).
pub const OFFSET_BITS_DEFAULT: u32 = 15;

/// Maximum lookback window for a given offset_bits value.
pub fn max_offset(offset_bits: u32) -> usize {
    (1usize << offset_bits) - 1
}

// ── Token bit cost helpers (offset_bits is runtime) ──────────────────────────
pub const LIT_TOTAL_BITS: u32 = OPCODE_LIT_BITS + BYTE_BITS;   // 10 — fixed
pub const END_TOTAL_BITS: u32 = OPCODE_END_BITS;                 // 2  — fixed

pub fn backref_total_bits(offset_bits: u32) -> u32 {
    OPCODE_BACKREF_BITS + offset_bits + LENGTH_BITS
}

pub fn token_bit_cost(token: &Token, offset_bits: u32) -> u32 {
    match token {
        Token::Lit { .. }     => LIT_TOTAL_BITS,
        Token::Backref { .. } => backref_total_bits(offset_bits),
        Token::End            => END_TOTAL_BITS,
    }
}

/// Compute the minimum offset_bits needed to represent all offsets
/// in a token stream, clamped to [OFFSET_BITS_MIN, OFFSET_BITS_MAX].
/// Returns OFFSET_BITS_MIN if there are no BACKREFs.
pub fn compute_optimal_offset_bits(tokens: &[Token]) -> u32 {
    let max_used = tokens.iter().filter_map(|t| {
        if let Token::Backref { offset, .. } = t { Some(*offset) } else { None }
    }).max().unwrap_or(0);

    if max_used == 0 {
        return OFFSET_BITS_MIN;
    }

    // Bits needed = floor(log2(max_used)) + 1, minimum OFFSET_BITS_MIN
    let bits_needed = 32 - max_used.leading_zeros();
    bits_needed.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
}
