// src/pairing.rs
//! Token-level pair encoding with Exp-Golomb operand compression.
//! Fold 2 uses this instead of raw LZ on bytes.
//!
//! Operand encoding strategy (replaces Cantor pairing):
//! Each operand (offset, length) encoded independently.
//! 1 flag bit: 0 = exp-golomb follows, 1 = raw bits follow.
//! Offset raw = 15 bits, Length raw = 8 bits.
//! Exp-Golomb used only when cheaper than raw — never worse than raw.

use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

// 3-bit pair/single/end prefix — fixed, never transmitted
const PREFIX_LL:   u32 = 0b000;
const PREFIX_LB:   u32 = 0b001;
const PREFIX_BL:   u32 = 0b010;
const PREFIX_BB:   u32 = 0b011;
const PREFIX_SL:   u32 = 0b100;
const PREFIX_SB:   u32 = 0b101;
const PREFIX_END:  u32 = 0b110;
const PREFIX_BITS: u32 = 3;

const OFFSET_RAW_BITS: u32 = 15;
const LENGTH_RAW_BITS: u32 = 8;

/// How many bits exp-golomb needs for value n (k=0)
/// Total = 2 * floor(log2(n+1)) + 1
fn exp_golomb_bit_cost(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let k = 31 - (n + 1).leading_zeros(); // floor(log2(n+1))
    2 * k + 1
}

/// Write value n using k=0 Exp-Golomb coding
fn write_exp_golomb<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    n: u32,
) -> std::io::Result<()> {
    let m = n + 1;
    let k = 31 - m.leading_zeros(); // floor(log2(m))
    // Write k zero bits
    for _ in 0..k {
        w.write(1, 0u32)?;
    }
    // Write m in (k+1) bits
    w.write(k + 1, m)?;
    Ok(())
}

/// Read k=0 Exp-Golomb coded value
fn read_exp_golomb<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<u32> {
    let mut k = 0u32;
    // Count leading zeros
    loop {
        let bit = r.read::<u32>(1)?;
        if bit == 1 {
            break;
        }
        k += 1;
    }
    // We already consumed the leading 1 of m, read remaining k bits
    let rest = if k > 0 { r.read::<u32>(k)? } else { 0 };
    let m = (1 << k) | rest;
    Ok(m - 1)
}

/// Write a BACKREF offset operand.
/// Flag=0: exp-golomb (when cheaper). Flag=1: raw 15-bit.
fn write_offset<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    offset: u32,
) -> std::io::Result<()> {
    let eg_bits = exp_golomb_bit_cost(offset);
    if eg_bits < OFFSET_RAW_BITS {
        w.write(1, 0u32)?; // flag: exp-golomb
        write_exp_golomb(w, offset)?;
    } else {
        w.write(1, 1u32)?; // flag: raw
        w.write(OFFSET_RAW_BITS, offset)?;
    }
    Ok(())
}

/// Write a BACKREF length operand.
/// Flag=0: exp-golomb (when cheaper). Flag=1: raw 8-bit.
fn write_length<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    length: u32,
) -> std::io::Result<()> {
    let eg_bits = exp_golomb_bit_cost(length);
    if eg_bits < LENGTH_RAW_BITS {
        w.write(1, 0u32)?; // flag: exp-golomb
        write_exp_golomb(w, length)?;
    } else {
        w.write(1, 1u32)?; // flag: raw
        w.write(LENGTH_RAW_BITS, length)?;
    }
    Ok(())
}

fn read_offset<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<u32> {
    let flag = r.read::<u32>(1)?;
    if flag == 0 {
        read_exp_golomb(r)
    } else {
        Ok(r.read::<u32>(OFFSET_RAW_BITS)?)
    }
}

fn read_length<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<u32> {
    let flag = r.read::<u32>(1)?;
    if flag == 0 {
        read_exp_golomb(r)
    } else {
        Ok(r.read::<u32>(LENGTH_RAW_BITS)?)
    }
}

/// Encode a token stream into a pair-encoded bitstream
pub fn pair_encode(tokens: &[Token]) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);

        let data: Vec<&Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .collect();

        let mut i = 0;
        while i < data.len() {
            if i + 1 < data.len() {
                match (data[i], data[i + 1]) {
                    (Token::Lit { byte: b1 }, Token::Lit { byte: b2 }) => {
                        w.write(PREFIX_BITS, PREFIX_LL)?;
                        w.write(8, *b1 as u32)?;
                        w.write(8, *b2 as u32)?;
                    }
                    (Token::Lit { byte: b }, Token::Backref { offset, length }) => {
                        w.write(PREFIX_BITS, PREFIX_LB)?;
                        w.write(8, *b as u32)?;
                        write_offset(&mut w, *offset)?;
                        write_length(&mut w, *length)?;
                    }
                    (Token::Backref { offset, length }, Token::Lit { byte: b }) => {
                        w.write(PREFIX_BITS, PREFIX_BL)?;
                        write_offset(&mut w, *offset)?;
                        write_length(&mut w, *length)?;
                        w.write(8, *b as u32)?;
                    }
                    (Token::Backref { offset: o1, length: l1 },
                     Token::Backref { offset: o2, length: l2 }) => {
                        w.write(PREFIX_BITS, PREFIX_BB)?;
                        write_offset(&mut w, *o1)?;
                        write_length(&mut w, *l1)?;
                        write_offset(&mut w, *o2)?;
                        write_length(&mut w, *l2)?;
                    }
                    _ => unreachable!("END should be filtered"),
                }
                i += 2;
            } else {
                match data[i] {
                    Token::Lit { byte: b } => {
                        w.write(PREFIX_BITS, PREFIX_SL)?;
                        w.write(8, *b as u32)?;
                    }
                    Token::Backref { offset, length } => {
                        w.write(PREFIX_BITS, PREFIX_SB)?;
                        write_offset(&mut w, *offset)?;
                        write_length(&mut w, *length)?;
                    }
                    _ => unreachable!(),
                }
                i += 1;
            }
        }

        w.write(PREFIX_BITS, PREFIX_END)?;
        w.byte_align()?;
    }
    Ok(output)
}

/// Decode a pair-encoded bitstream back to a token stream
pub fn pair_decode(input: &[u8]) -> std::io::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let prefix = match r.read::<u32>(PREFIX_BITS) {
            Ok(p) => p,
            Err(_) => break,
        };

        match prefix {
            p if p == PREFIX_LL => {
                let b1 = r.read::<u32>(8)? as u8;
                let b2 = r.read::<u32>(8)? as u8;
                tokens.push(Token::Lit { byte: b1 });
                tokens.push(Token::Lit { byte: b2 });
            }
            p if p == PREFIX_LB => {
                let b = r.read::<u32>(8)? as u8;
                let offset = read_offset(&mut r)?;
                let length = read_length(&mut r)?;
                tokens.push(Token::Lit { byte: b });
                tokens.push(Token::Backref { offset, length });
            }
            p if p == PREFIX_BL => {
                let offset = read_offset(&mut r)?;
                let length = read_length(&mut r)?;
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Backref { offset, length });
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_BB => {
                let o1 = read_offset(&mut r)?;
                let l1 = read_length(&mut r)?;
                let o2 = read_offset(&mut r)?;
                let l2 = read_length(&mut r)?;
                tokens.push(Token::Backref { offset: o1, length: l1 });
                tokens.push(Token::Backref { offset: o2, length: l2 });
            }
            p if p == PREFIX_SL => {
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_SB => {
                let offset = read_offset(&mut r)?;
                let length = read_length(&mut r)?;
                tokens.push(Token::Backref { offset, length });
            }
            p if p == PREFIX_END => {
                tokens.push(Token::End);
                break;
            }
            _ => return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid pair prefix",
            )),
        }
    }

    Ok(tokens)
}
