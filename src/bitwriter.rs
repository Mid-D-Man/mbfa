//! Writes a Token stream to a compact bitstream using the fixed opcode vocabulary.

use bitstream_io::{BitWriter, BigEndian, BitWrite};
use crate::opcode::*;

pub fn write_tokens(tokens: &[Token]) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
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
                    writer.write(OFFSET_BITS, *offset)?;
                    writer.write(LENGTH_BITS, *length)?;
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