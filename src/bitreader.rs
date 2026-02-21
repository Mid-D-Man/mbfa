// src/bitreader.rs
//! Reads a compact bitstream back into a Token stream.
//! offset_bits must match the value used during encoding (read from file header).

use bitstream_io::{BitReader, BigEndian, BitRead};
use crate::opcode::*;

pub fn read_tokens(input: &[u8], offset_bits: u32) -> std::io::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut reader = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let first_bit = match reader.read::<u32>(1) {
            Ok(b) => b,
            Err(_) => break,
        };

        if first_bit == OPCODE_BACKREF_VAL {
            let offset = reader.read::<u32>(offset_bits)?;
            let length = reader.read::<u32>(LENGTH_BITS)?;
            tokens.push(Token::Backref { offset, length });
        } else {
            let second_bit = reader.read::<u32>(1)?;
            if second_bit == 0 {
                let byte = reader.read::<u32>(BYTE_BITS)? as u8;
                tokens.push(Token::Lit { byte });
            } else {
                tokens.push(Token::End);
                break;
            }
        }
    }

    Ok(tokens)
}
