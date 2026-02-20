//! Token-level pair encoding with Cantor operand compression.
//! Fold 2 uses this instead of raw LZ on bytes.

use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

// 3-bit pair/single/end prefix — fixed, never transmitted
const PREFIX_LL:  u32 = 0b000; // LIT + LIT
const PREFIX_LB:  u32 = 0b001; // LIT + BACKREF
const PREFIX_BL:  u32 = 0b010; // BACKREF + LIT
const PREFIX_BB:  u32 = 0b011; // BACKREF + BACKREF
const PREFIX_SL:  u32 = 0b100; // single LIT
const PREFIX_SB:  u32 = 0b101; // single BACKREF
const PREFIX_END: u32 = 0b110; // end of stream
const PREFIX_BITS: u32 = 3;

// Bit cost summary vs current encoding:
// LL:  3+8+8   = 19 bits  (was 10+10  = 20)  saves 1 bit
// LB:  3+8+17  = 28 bits  (was 10+24  = 34)  saves 6 bits  (if cantor fits 16 bits)
// BL:  3+17+8  = 28 bits  (was 24+10  = 34)  saves 6 bits
// BB:  3+17+17 = 37 bits  (was 24+24  = 48)  saves 11 bits ← biggest win
// SL:  3+8     = 11 bits  (was        10)    costs 1 bit
// SB:  3+17    = 20 bits  (was        24)    saves 4 bits
// END: 3 bits             (was         2)    costs 1 bit

/// Cantor pairing — encodes two natural numbers into one
/// cantor(x,y) = (x+y)(x+y+1)/2 + y
fn cantor_pair(x: u32, y: u32) -> u64 {
    let s = (x + y) as u64;
    s * (s + 1) / 2 + y as u64
}

/// Inverse Cantor — recovers x and y from cantor(x,y)
fn cantor_unpair(z: u64) -> (u32, u32) {
    let w = ((((8 * z + 1) as f64).sqrt() - 1.0) / 2.0).floor() as u64;
    let t = (w * w + w) / 2;
    let y = z - t;
    let x = w - y;
    (x as u32, y as u32)
}

/// Write a backref operand.
/// If cantor(offset,length) < 65536: write flag=0 + 16-bit cantor value (17 bits total)
/// Otherwise: write flag=1 + raw 15-bit offset + 8-bit length (24 bits total)
fn write_backref_operand<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    offset: u32,
    length: u32,
) -> std::io::Result<()> {
    let c = cantor_pair(offset, length);
    if c < 65536 {
        w.write(1, 0u32)?;
        w.write(16, c as u32)?;
    } else {
        w.write(1, 1u32)?;
        w.write(15, offset)?;
        w.write(8, length)?;
    }
    Ok(())
}

fn read_backref_operand<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<(u32, u32)> {
    let flag = r.read::<u32>(1)?;
    if flag == 0 {
        let c = r.read::<u32>(16)? as u64;
        Ok(cantor_unpair(c))
    } else {
        let offset = r.read::<u32>(15)?;
        let length = r.read::<u32>(8)?;
        Ok((offset, length))
    }
}

/// Encode a token stream into a pair-encoded bitstream
pub fn pair_encode(tokens: &[Token]) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);

        // Collect all non-END tokens — END is written separately at the end
        let data: Vec<&Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .collect();

        let mut i = 0;
        while i < data.len() {
            if i + 1 < data.len() {
                // Write a pair
                match (data[i], data[i + 1]) {
                    (Token::Lit { byte: b1 }, Token::Lit { byte: b2 }) => {
                        w.write(PREFIX_BITS, PREFIX_LL)?;
                        w.write(8, *b1 as u32)?;
                        w.write(8, *b2 as u32)?;
                    }
                    (Token::Lit { byte: b }, Token::Backref { offset, length }) => {
                        w.write(PREFIX_BITS, PREFIX_LB)?;
                        w.write(8, *b as u32)?;
                        write_backref_operand(&mut w, *offset, *length)?;
                    }
                    (Token::Backref { offset, length }, Token::Lit { byte: b }) => {
                        w.write(PREFIX_BITS, PREFIX_BL)?;
                        write_backref_operand(&mut w, *offset, *length)?;
                        w.write(8, *b as u32)?;
                    }
                    (Token::Backref { offset: o1, length: l1 },
                        Token::Backref { offset: o2, length: l2 }) => {
                        w.write(PREFIX_BITS, PREFIX_BB)?;
                        write_backref_operand(&mut w, *o1, *l1)?;
                        write_backref_operand(&mut w, *o2, *l2)?;
                    }
                    _ => unreachable!("END should be filtered"),
                }
                i += 2;
            } else {
                // Odd token out — write as single
                match data[i] {
                    Token::Lit { byte: b } => {
                        w.write(PREFIX_BITS, PREFIX_SL)?;
                        w.write(8, *b as u32)?;
                    }
                    Token::Backref { offset, length } => {
                        w.write(PREFIX_BITS, PREFIX_SB)?;
                        write_backref_operand(&mut w, *offset, *length)?;
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
                let (offset, length) = read_backref_operand(&mut r)?;
                tokens.push(Token::Lit { byte: b });
                tokens.push(Token::Backref { offset, length });
            }
            p if p == PREFIX_BL => {
                let (offset, length) = read_backref_operand(&mut r)?;
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Backref { offset, length });
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_BB => {
                let (o1, l1) = read_backref_operand(&mut r)?;
                let (o2, l2) = read_backref_operand(&mut r)?;
                tokens.push(Token::Backref { offset: o1, length: l1 });
                tokens.push(Token::Backref { offset: o2, length: l2 });
            }
            p if p == PREFIX_SL => {
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_SB => {
                let (offset, length) = read_backref_operand(&mut r)?;
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