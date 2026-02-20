//! Fixed opcode vocabulary — known to both encoder and decoder.
//! Never transmitted. Single source of truth for all bit widths.

/// A single instruction token produced by the encoder
/// and consumed by the decoder.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Emit one literal byte
    Lit { byte: u8 },
    /// Copy `length` bytes starting `offset` bytes back in the output buffer
    Backref { offset: u32, length: u32 },
    /// End of instruction stream for this fold
    End,
}

// ── Opcode bit patterns (value, bit_width) ────────────────────────────────────
pub const OPCODE_BACKREF_BITS: u32 = 1;
pub const OPCODE_BACKREF_VAL:  u32 = 0b0;

pub const OPCODE_LIT_BITS:     u32 = 2;
pub const OPCODE_LIT_VAL:      u32 = 0b10;

pub const OPCODE_END_BITS:     u32 = 2;
pub const OPCODE_END_VAL:      u32 = 0b11;

// ── Operand bit widths ────────────────────────────────────────────────────────
// OFFSET: 15 bits = 32767 byte lookback window (matches gzip territory)
// LENGTH:  8 bits = 255  byte max copy length
pub const OFFSET_BITS: u32 = 15;
pub const LENGTH_BITS: u32 = 8;
pub const BYTE_BITS:   u32 = 8;

// ── Token bit cost helpers ────────────────────────────────────────────────────
pub const LIT_TOTAL_BITS:     u32 = OPCODE_LIT_BITS + BYTE_BITS;
pub const BACKREF_TOTAL_BITS: u32 = OPCODE_BACKREF_BITS + OFFSET_BITS + LENGTH_BITS;
pub const END_TOTAL_BITS:     u32 = OPCODE_END_BITS;

/// Returns how many bits a token costs when encoded
pub fn token_bit_cost(token: &Token) -> u32 {
    match token {
        Token::Lit { .. }     => LIT_TOTAL_BITS,
        Token::Backref { .. } => BACKREF_TOTAL_BITS,
        Token::End            => END_TOTAL_BITS,
    }
}