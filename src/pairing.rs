// src/pairing.rs
//! Token-level pair encoding with Cantor operand compression.
//! Both offset_bits and length_bits are runtime parameters.

use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

const PREFIX_LL:   u32 = 0b000;
const PREFIX_LB:   u32 = 0b001;
const PREFIX_BL:   u32 = 0b010;
const PREFIX_BB:   u32 = 0b011;
const PREFIX_SL:   u32 = 0b100;
const PREFIX_SB:   u32 = 0b101;
const PREFIX_END:  u32 = 0b110;
const PREFIX_BITS: u32 = 3;

fn cantor_pair(x: u32, y: u32) -> u64 {
    let s = (x + y) as u64;
    s * (s + 1) / 2 + y as u64
}

fn cantor_unpair(z: u64) -> (u32, u32) {
    let w = ((((8u64.saturating_mul(z).saturating_add(1)) as f64).sqrt() - 1.0) / 2.0).floor() as u64;
    let w = if w > 0 && (w * (w + 1) / 2) > z { w.saturating_sub(1) } else { w };
    let t = w * (w + 1) / 2;
    if t > z {
        eprintln!("Warning: cantor_unpair({}) invalid t={}", z, t);
        return (1, 2);
    }
    let y = z - t;
    if y > w {
        eprintln!("Warning: cantor_unpair({}) y={} > w={}", z, y, w);
        return (1, 2);
    }
    let x = w - y;
    (x as u32, y as u32)
}

fn write_backref_operand<W: std::io::Write>(
    w:           &mut BitWriter<W, BigEndian>,
    offset:      u32,
    length:      u32,
    offset_bits: u32,
    length_bits: u32,
) -> std::io::Result<()> {
    let c = cantor_pair(offset, length);
    if c < 65536 {
        w.write(1, 0u32)?;
        w.write(16, c as u32)?;
    } else {
        // Raw fallback — use the actual field widths from the enclosing fold.
        w.write(1, 1u32)?;
        w.write(offset_bits, offset)?;
        w.write(length_bits, length)?;
    }
    Ok(())
}

fn read_backref_operand<R: std::io::Read>(
    r:           &mut BitReader<R, BigEndian>,
    offset_bits: u32,
    length_bits: u32,
) -> std::io::Result<(u32, u32)> {
    let flag = r.read::<u32>(1)?;
    if flag == 0 {
        let c = r.read::<u32>(16)? as u64;
        let (offset, length) = cantor_unpair(c);
        if offset == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("cantor_unpair produced offset=0 for z={}", c),
            ));
        }
        Ok((offset, length))
    } else {
        let offset = r.read::<u32>(offset_bits)?;
        let length = r.read::<u32>(length_bits)?;
        Ok((offset, length))
    }
}

pub fn pair_encode(tokens: &[Token], offset_bits: u32, length_bits: u32) -> std::io::Result<Vec<u8>> {
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
                        write_backref_operand(&mut w, *offset, *length, offset_bits, length_bits)?;
                    }
                    (Token::Backref { offset, length }, Token::Lit { byte: b }) => {
                        w.write(PREFIX_BITS, PREFIX_BL)?;
                        write_backref_operand(&mut w, *offset, *length, offset_bits, length_bits)?;
                        w.write(8, *b as u32)?;
                    }
                    (Token::Backref { offset: o1, length: l1 },
                     Token::Backref { offset: o2, length: l2 }) => {
                        w.write(PREFIX_BITS, PREFIX_BB)?;
                        write_backref_operand(&mut w, *o1, *l1, offset_bits, length_bits)?;
                        write_backref_operand(&mut w, *o2, *l2, offset_bits, length_bits)?;
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
                        write_backref_operand(&mut w, *offset, *length, offset_bits, length_bits)?;
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

pub fn pair_decode(input: &[u8], offset_bits: u32, length_bits: u32) -> std::io::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let prefix = match r.read::<u32>(PREFIX_BITS) {
            Ok(p)  => p,
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
                let (off, len) = read_backref_operand(&mut r, offset_bits, length_bits)?;
                tokens.push(Token::Lit { byte: b });
                tokens.push(Token::Backref { offset: off, length: len });
            }
            p if p == PREFIX_BL => {
                let (off, len) = read_backref_operand(&mut r, offset_bits, length_bits)?;
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Backref { offset: off, length: len });
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_BB => {
                let (o1, l1) = read_backref_operand(&mut r, offset_bits, length_bits)?;
                let (o2, l2) = read_backref_operand(&mut r, offset_bits, length_bits)?;
                tokens.push(Token::Backref { offset: o1, length: l1 });
                tokens.push(Token::Backref { offset: o2, length: l2 });
            }
            p if p == PREFIX_SL => {
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_SB => {
                let (off, len) = read_backref_operand(&mut r, offset_bits, length_bits)?;
                tokens.push(Token::Backref { offset: off, length: len });
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
