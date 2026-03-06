// src/bitwriter.rs
//! Writes a Token stream to a compact bitstream using the fixed opcode vocabulary.
//! Both offset_bits and length_bits are passed at runtime.

use bitstream_io::{BitWriter, BigEndian, BitWrite};
use crate::opcode::*;

pub fn write_tokens(tokens: &[Token], offset_bits: u32, length_bits: u32) -> std::io::Result<Vec<u8>> {
    // Task 4: pre-allocate output buffer.
    // Worst case: every token is a Backref → (1 + offset_bits + length_bits) bits each.
    // +2 bytes: 1 for the END token, 1 for byte_align padding.
    // This is a safe overestimate — Lit tokens are only 10 bits vs up to 49 bits
    // for a max Backref (ob=24, lb=24), so real output is always smaller.
    let est_bytes = (tokens.len() as u64
        * (1 + offset_bits + length_bits) as u64
        / 8 + 2) as usize;
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
                Token::End => {
                    writer.write(OPCODE_END_BITS, OPCODE_END_VAL)?;
                }
            }
        }

        writer.byte_align()?;
    }
    Ok(output)
}
