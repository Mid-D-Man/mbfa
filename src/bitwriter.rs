// src/bitwriter.rs
//! Writes a Token stream to a compact bitstream.
//!
//! Both `offset_bits` and `length_bits` are passed at runtime.
//!
//! Ring-encoding mode is auto-detected: if the token stream contains any
//! `Token::RepRef`, the writer uses the ring-active opcode set (3-bit End,
//! 3+2-bit RepRef) and the returned bool indicates ring_was_used = true.
//! Without RepRef tokens, the legacy opcode set is used (ring_was_used = false).

use bitstream_io::{BitWriter, BigEndian, BitWrite};
use crate::opcode::*;

/// Write `tokens` to a compact bitstream.
///
/// Returns `(bytes, ring_was_used)` where `ring_was_used` indicates whether
/// ring-active opcodes were emitted (i.e., at least one RepRef token present).
/// The caller must store this flag in the header (pair_flag bit 1) so the
/// decompressor knows which opcode set to use when reading the stream back.
pub fn write_tokens(tokens: &[Token], offset_bits: u32, length_bits: u32) -> std::io::Result<(Vec<u8>, bool)> {
    let ring_active = tokens.iter().any(|t| matches!(t, Token::RepRef { .. }));

    // Capacity estimate (conservative upper bound):
    //   Worst case per token (no RepRef): Backref = 1+ob+lb bits.
    //   With RepRef: all become 3+2+lb bits ≤ 1+ob+lb for ob≥5.
    //   +2 bytes: End token + byte-align padding.
    let est_bytes = (tokens.len() as u64
        * (1 + offset_bits + length_bits) as u64
        / 8 + 4) as usize;
    let mut output = Vec::with_capacity(est_bytes);

    {
        let mut writer = BitWriter::endian(&mut output, BigEndian);

        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    writer.write(OPCODE_LIT_BITS, OPCODE_LIT_VAL)?;
                    writer.write(BYTE_BITS, *byte as u32)?;
                }
                Token::Backref { offset, length } => {
                    writer.write(OPCODE_BACKREF_BITS, OPCODE_BACKREF_VAL)?;
                    writer.write(offset_bits, *offset)?;
                    writer.write(length_bits, *length)?;
                }
                Token::RepRef { slot, length } => {
                    // ring_active is always true here (checked above).
                    writer.write(OPCODE_REPREF_BITS, OPCODE_REPREF_VAL)?;
                    writer.write(REPREF_SLOT_BITS, *slot as u32)?;
                    writer.write(length_bits, *length)?;
                }
                Token::End => {
                    if ring_active {
                        writer.write(OPCODE_END_RING_BITS, OPCODE_END_RING_VAL)?;
                    } else {
                        writer.write(OPCODE_END_BITS, OPCODE_END_VAL)?;
                    }
                }
            }
        }

        writer.byte_align()?;
    }

    Ok((output, ring_active))
}
