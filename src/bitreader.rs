// src/bitreader.rs
//! Reads a compact bitstream back into a Token stream.
//!
//! Both `offset_bits` and `length_bits` must match the values used during
//! encoding.  The `ring_active` flag must match the `ring_was_used` bool
//! returned by `bitwriter::write_tokens` (stored in pair_flag bit 1 in the
//! file header).
//!
//! When `ring_active = false` (legacy mode):
//!   "0"  → Backref + offset_bits + length_bits
//!   "10" → Lit + 8-bit byte
//!   "11" → End
//!
//! When `ring_active = true` (ring mode):
//!   "0"   → Backref + offset_bits + length_bits
//!   "10"  → Lit + 8-bit byte
//!   "110" → RepRef + 2-bit slot + length_bits
//!   "111" → End

use bitstream_io::{BitReader, BigEndian, BitRead};
use crate::opcode::*;

pub fn read_tokens(
    input:       &[u8],
    offset_bits: u32,
    length_bits: u32,
    ring_active: bool,
) -> std::io::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut reader = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        // Read the first bit: 0 = Backref, 1 = Lit / RepRef / End.
        let first_bit = match reader.read::<u32>(1) {
            Ok(b)  => b,
            Err(_) => break,
        };

        if first_bit == OPCODE_BACKREF_VAL {
            // Backref: same in both ring and non-ring mode.
            let offset = reader.read::<u32>(offset_bits)?;
            let length = reader.read::<u32>(length_bits)?;
            tokens.push(Token::Backref { offset, length });
        } else {
            // second bit distinguishes Lit from End/RepRef.
            let second_bit = reader.read::<u32>(1)?;
            if second_bit == 0 {
                // "10" → Lit.
                let byte = reader.read::<u32>(BYTE_BITS)? as u8;
                tokens.push(Token::Lit { byte });
            } else if !ring_active {
                // Legacy: "11" → End.
                tokens.push(Token::End);
                break;
            } else {
                // Ring-active: third bit disambiguates RepRef from End.
                let third_bit = reader.read::<u32>(1)?;
                if third_bit == 0 {
                    // "110" → RepRef + 2-bit slot + length.
                    let slot   = reader.read::<u32>(REPREF_SLOT_BITS)? as u8;
                    let length = reader.read::<u32>(length_bits)?;
                    tokens.push(Token::RepRef { slot, length });
                } else {
                    // "111" → End.
                    tokens.push(Token::End);
                    break;
                }
            }
        }
    }

    Ok(tokens)
                    }
