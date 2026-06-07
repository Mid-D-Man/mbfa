// src/pairing.rs
//! Token-level pair encoding with Exp-Golomb operand compression.
//!
//! Replaces the previous Cantor-pairing scheme. Key differences:
//!
//!   Old (Cantor): encode (offset, length) jointly as a single number.
//!   New (Exp-Golomb): encode offset-1 and length-1 independently using
//!     order-0 Exp-Golomb. Self-delimiting, no flag bit, no 16-bit limit.
//!
//!   offset_bits and length_bits parameters are retained in the public API
//!   for forward compatibility but are not used internally.
//!
//! P6: pair_encode never sees Token::RepRef — resolve_ring() is called by
//! fold.rs before pair_encode. The RepRef arm is marked unreachable to
//! satisfy Rust's exhaustiveness check.
//!
//! Pair prefix vocabulary:
//!   LL  000 — LIT + LIT
//!   LB  001 — LIT + BACKREF
//!   BL  010 — BACKREF + LIT
//!   BB  011 — BACKREF + BACKREF
//!   SL  100 — single LIT (odd token out)
//!   SB  101 — single BACKREF
//!   END 110 — stream terminator

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

// ── Exp-Golomb coding ─────────────────────────────────────────────────────────

#[inline]
fn eg_write<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    n: u32,
) -> std::io::Result<()> {
    let k: u32 = if n == 0 {
        0
    } else {
        u32::BITS - (n + 1).leading_zeros() - 1
    };

    if k > 0 {
        w.write(k, 0u32)?;
    }
    w.write(1, 1u32)?;
    if k > 0 {
        let suffix = (n + 1) - (1u32 << k);
        w.write(k, suffix)?;
    }
    Ok(())
}

#[inline]
fn eg_read<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<u32> {
    let mut k: u32 = 0;
    loop {
        let bit = r.read::<u32>(1)?;
        if bit == 1 { break; }
        k += 1;
        if k > 31 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "eg_read: prefix exceeds 31 bits — malformed pair stream",
            ));
        }
    }
    if k == 0 {
        return Ok(0);
    }
    let suffix = r.read::<u32>(k)?;
    Ok((1u32 << k) - 1 + suffix)
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn pair_encode(
    tokens:       &[Token],
    _offset_bits: u32,
    _length_bits: u32,
) -> std::io::Result<Vec<u8>> {
    let est_bytes = (tokens.len() / 2 + 2) * 4;
    let mut output = Vec::with_capacity(est_bytes);
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);

        let mut data: Vec<&Token> = Vec::with_capacity(tokens.len());
        data.extend(tokens.iter().filter(|t| !matches!(t, Token::End)));

        let mut i = 0;
        while i < data.len() {
            if i + 1 < data.len() {
                // Pair match — `_` catches RepRef (unreachable: resolve_ring called first)
                match (data[i], data[i + 1]) {
                    (Token::Lit { byte: b1 }, Token::Lit { byte: b2 }) => {
                        w.write(PREFIX_BITS, PREFIX_LL)?;
                        w.write(8, *b1 as u32)?;
                        w.write(8, *b2 as u32)?;
                    }
                    (Token::Lit { byte: b }, Token::Backref { offset, length }) => {
                        w.write(PREFIX_BITS, PREFIX_LB)?;
                        w.write(8, *b as u32)?;
                        eg_write(&mut w, offset - 1)?;
                        eg_write(&mut w, length - 1)?;
                    }
                    (Token::Backref { offset, length }, Token::Lit { byte: b }) => {
                        w.write(PREFIX_BITS, PREFIX_BL)?;
                        eg_write(&mut w, offset - 1)?;
                        eg_write(&mut w, length - 1)?;
                        w.write(8, *b as u32)?;
                    }
                    (
                        Token::Backref { offset: o1, length: l1 },
                        Token::Backref { offset: o2, length: l2 },
                    ) => {
                        w.write(PREFIX_BITS, PREFIX_BB)?;
                        eg_write(&mut w, o1 - 1)?;
                        eg_write(&mut w, l1 - 1)?;
                        eg_write(&mut w, o2 - 1)?;
                        eg_write(&mut w, l2 - 1)?;
                    }
                    _ => unreachable!("END and RepRef tokens must not reach pair_encode"),
                }
                i += 2;
            } else {
                // Single (odd token out)
                match data[i] {
                    Token::Lit { byte: b } => {
                        w.write(PREFIX_BITS, PREFIX_SL)?;
                        w.write(8, *b as u32)?;
                    }
                    Token::Backref { offset, length } => {
                        w.write(PREFIX_BITS, PREFIX_SB)?;
                        eg_write(&mut w, offset - 1)?;
                        eg_write(&mut w, length - 1)?;
                    }
                    Token::End => unreachable!("END tokens filtered before this loop"),
                    // P6: resolve_ring() is called before pair_encode — RepRef is unreachable.
                    Token::RepRef { .. } => unreachable!(
                        "RepRef must be resolved via resolve_ring() before pair_encode"
                    ),
                }
                i += 1;
            }
        }

        w.write(PREFIX_BITS, PREFIX_END)?;
        w.byte_align()?;
    }
    Ok(output)
}

pub fn pair_decode(
    input:        &[u8],
    _offset_bits: u32,
    _length_bits: u32,
) -> std::io::Result<Vec<Token>> {
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
                let b      = r.read::<u32>(8)? as u8;
                let offset = eg_read(&mut r)? + 1;
                let length = eg_read(&mut r)? + 1;
                tokens.push(Token::Lit { byte: b });
                tokens.push(Token::Backref { offset, length });
            }
            p if p == PREFIX_BL => {
                let offset = eg_read(&mut r)? + 1;
                let length = eg_read(&mut r)? + 1;
                let b      = r.read::<u32>(8)? as u8;
                tokens.push(Token::Backref { offset, length });
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_BB => {
                let o1 = eg_read(&mut r)? + 1;
                let l1 = eg_read(&mut r)? + 1;
                let o2 = eg_read(&mut r)? + 1;
                let l2 = eg_read(&mut r)? + 1;
                tokens.push(Token::Backref { offset: o1, length: l1 });
                tokens.push(Token::Backref { offset: o2, length: l2 });
            }
            p if p == PREFIX_SL => {
                let b = r.read::<u32>(8)? as u8;
                tokens.push(Token::Lit { byte: b });
            }
            p if p == PREFIX_SB => {
                let offset = eg_read(&mut r)? + 1;
                let length = eg_read(&mut r)? + 1;
                tokens.push(Token::Backref { offset, length });
            }
            p if p == PREFIX_END => {
                tokens.push(Token::End);
                break;
            }
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("pair_decode: invalid prefix 0b{:03b}", other),
                ));
            }
        }
    }

    Ok(tokens)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::Token;

    fn round_trip(tokens: &[Token]) -> Vec<Token> {
        let encoded = pair_encode(tokens, 17, 8).expect("encode failed");
        pair_decode(&encoded, 17, 8).expect("decode failed")
    }

    fn eg_round_trip(n: u32) -> u32 {
        let mut output = Vec::new();
        {
            let mut w = BitWriter::endian(&mut output, BigEndian);
            eg_write(&mut w, n).unwrap();
            w.byte_align().unwrap();
        }
        let mut r = BitReader::endian(std::io::Cursor::new(&output), BigEndian);
        eg_read(&mut r).unwrap()
    }

    #[test]
    fn eg_roundtrip_small_values() {
        for n in 0..=255u32 {
            assert_eq!(eg_round_trip(n), n, "EG roundtrip failed for n={}", n);
        }
    }

    #[test]
    fn eg_roundtrip_powers_of_two() {
        for k in 0..24u32 {
            let n = (1u32 << k) - 1;
            assert_eq!(eg_round_trip(n), n, "EG roundtrip failed for n={}", n);
            let n2 = 1u32 << k;
            assert_eq!(eg_round_trip(n2), n2, "EG roundtrip failed for n={}", n2);
        }
    }

    #[test]
    fn eg_roundtrip_large_values() {
        for &n in &[1000u32, 10000, 100000, 1_000_000, 16_777_214] {
            assert_eq!(eg_round_trip(n), n, "EG roundtrip failed for n={}", n);
        }
    }

    #[test]
    fn eg_bit_lengths_are_correct() {
        let expected = [(0, 1), (1, 3), (2, 3), (3, 5), (6, 5), (7, 7), (14, 7), (15, 9)];
        for (n, expected_bits) in expected {
            let k: u32 = if n == 0 { 0 } else { u32::BITS - (n + 1).leading_zeros() - 1 };
            assert_eq!(2 * k + 1, expected_bits as u32,
                "EG bit length wrong for n={}: got {}, expected {}",
                n, 2*k+1, expected_bits);
        }
    }

    #[test]
    fn roundtrip_all_lits() {
        let tokens = vec![
            Token::Lit { byte: b'h' },
            Token::Lit { byte: b'e' },
            Token::Lit { byte: b'l' },
            Token::Lit { byte: b'l' },
            Token::Lit { byte: b'o' },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        let expected: Vec<Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .cloned()
            .chain(std::iter::once(Token::End))
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn roundtrip_backrefs_small_offset() {
        let tokens = vec![
            Token::Backref { offset: 1,  length: 4 },
            Token::Backref { offset: 3,  length: 2 },
            Token::Backref { offset: 10, length: 8 },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        let lz_tokens: Vec<Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .cloned()
            .chain(std::iter::once(Token::End))
            .collect();
        assert_eq!(decoded, lz_tokens);
    }

    #[test]
    fn roundtrip_backrefs_large_offset() {
        let tokens = vec![
            Token::Backref { offset: 1000,   length: 10  },
            Token::Backref { offset: 32767,  length: 255 },
            Token::Backref { offset: 131071, length: 1   },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        let expected: Vec<Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .cloned()
            .chain(std::iter::once(Token::End))
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn roundtrip_mixed_tokens() {
        let tokens = vec![
            Token::Lit    { byte: b'a' },
            Token::Backref { offset: 5,   length: 3 },
            Token::Lit    { byte: b'b' },
            Token::Lit    { byte: b'c' },
            Token::Backref { offset: 100, length: 10 },
            Token::Backref { offset: 200, length: 50 },
            Token::Lit    { byte: b'd' },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        let expected: Vec<Token> = tokens.iter()
            .filter(|t| !matches!(t, Token::End))
            .cloned()
            .chain(std::iter::once(Token::End))
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn roundtrip_odd_token_count() {
        let tokens = vec![
            Token::Lit { byte: 1 },
            Token::Lit { byte: 2 },
            Token::Lit { byte: 3 },
            Token::Lit { byte: 4 },
            Token::Lit { byte: 5 },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        assert_eq!(decoded.len(), 6);
        for (i, tok) in decoded[..5].iter().enumerate() {
            assert_eq!(*tok, Token::Lit { byte: (i + 1) as u8 });
        }
        assert_eq!(decoded[5], Token::End);
    }

    #[test]
    fn roundtrip_single_backref() {
        let tokens = vec![
            Token::Backref { offset: 42, length: 7 },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        assert_eq!(decoded[0], Token::Backref { offset: 42, length: 7 });
        assert_eq!(decoded[1], Token::End);
    }

    #[test]
    fn roundtrip_empty_except_end() {
        let tokens = vec![Token::End];
        let decoded = round_trip(&tokens);
        assert_eq!(decoded, vec![Token::End]);
    }

    #[test]
    fn eg_pair_beats_raw_for_small_offsets() {
        let tokens: Vec<Token> = (0..100)
            .map(|_| Token::Backref { offset: 5, length: 3 })
            .chain(std::iter::once(Token::End))
            .collect();
        let encoded = pair_encode(&tokens, 17, 8).unwrap();
        let decoded = pair_decode(&encoded, 17, 8).unwrap();
        let expected: Vec<Token> = (0..100)
            .map(|_| Token::Backref { offset: 5, length: 3 })
            .chain(std::iter::once(Token::End))
            .collect();
        assert_eq!(decoded, expected);
        assert!(encoded.len() < 400, "EG pair should be compact for small offsets");
    }
}
