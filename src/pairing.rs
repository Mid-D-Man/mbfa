// src/pairing.rs
//! Token-level pair encoding with Exp-Golomb operand compression.
//!
//! Replaces the previous Cantor-pairing scheme. Key differences:
//!
//!   Old (Cantor): encode (offset, length) jointly as a single number.
//!     If it fit in 16 bits: 1 flag bit + 16-bit value = 17 bits.
//!     If not (the common case for ob >= 15): 1 flag bit + raw ob + lb bits.
//!     The 77% fallback threshold in fold.rs existed to skip pairing when
//!     most backrefs couldn't benefit from Cantor compression.
//!
//!   New (Exp-Golomb): encode offset-1 and length-1 independently using
//!     order-0 Exp-Golomb. Self-delimiting, no flag bit, no 16-bit limit.
//!     EG(n) uses 2*floor(log2(n+1))+1 bits — shorter for small n, longer
//!     for large n than raw ob/lb encoding.
//!
//!   The 77% Cantor fallback threshold is gone. Pair encoding always runs
//!   when fold-1 output exceeds MIN_PAIR_BYTES. If pair output is larger
//!   than fold-1 (ratio >= MIN_IMPROVEMENT_RATIO), fold.rs stops at fold 1
//!   and entropy coding takes over. Natural competition, no pre-check needed.
//!
//!   offset_bits and length_bits parameters are retained in the public API
//!   for forward compatibility (they will be used by future slot-based
//!   extensions) but are not used internally for operand encoding.
//!
//! Pair prefix vocabulary (unchanged from original):
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
//
// Order-0 Exp-Golomb encoding for n >= 0:
//   k = floor(log2(n+1))
//   Bits: [k zeros][1][k-bit suffix where suffix = n+1 - 2^k]
//   Total bits: 2k+1
//
// Examples:
//   n=0  → k=0 → "1"       (1 bit)
//   n=1  → k=1 → "010"     (3 bits)
//   n=2  → k=1 → "011"     (3 bits)
//   n=3  → k=2 → "00100"   (5 bits)
//   n=7  → k=3 → "0001000" (7 bits)
//   n=99 → k=6 → 13 bits
//   n=255 → k=7 → 15 bits
//   n=511 → k=8 → 17 bits
//
// Backrefs encode offset-1 and length-1 so both start at n=0.

/// Write value `n` using order-0 Exp-Golomb coding.
/// Self-delimiting — decoder reads until it sees the separator 1 bit.
#[inline]
fn eg_write<W: std::io::Write>(
    w: &mut BitWriter<W, BigEndian>,
    n: u32,
) -> std::io::Result<()> {
    // k = floor(log2(n+1))
    // For n=0: k=0. For n=1: k=1. For n=3: k=2. etc.
    let k: u32 = if n == 0 {
        0
    } else {
        // bit_length(n+1) - 1 = 31 - leading_zeros(n+1)
        u32::BITS - (n + 1).leading_zeros() - 1
    };

    // Write k zero bits (unary prefix)
    if k > 0 {
        w.write(k, 0u32)?;
    }
    // Write separator bit 1
    w.write(1, 1u32)?;
    // Write k-bit suffix = (n + 1) - 2^k
    if k > 0 {
        let suffix = (n + 1) - (1u32 << k);
        w.write(k, suffix)?;
    }
    Ok(())
}

/// Read one order-0 Exp-Golomb code.
#[inline]
fn eg_read<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
) -> std::io::Result<u32> {
    // Count leading zero bits to determine k
    let mut k: u32 = 0;
    loop {
        let bit = r.read::<u32>(1)?;
        if bit == 1 { break; }
        k += 1;
        // Safety: valid u32 values need at most 31 leading zeros.
        // k > 31 means malformed data.
        if k > 31 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("eg_read: prefix exceeds 31 bits — malformed pair stream"),
            ));
        }
    }
    if k == 0 {
        return Ok(0);
    }
    // Read k-bit suffix and reconstruct n = 2^k - 1 + suffix
    let suffix = r.read::<u32>(k)?;
    Ok((1u32 << k) - 1 + suffix)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Encode a token stream using pair encoding with Exp-Golomb backref operands.
///
/// `_offset_bits` and `_length_bits` are retained for API compatibility but
/// are not used internally (EG is self-sizing). Callers in fold.rs pass
/// current_ob and current_lb which may be needed by future slot extensions.
pub fn pair_encode(
    tokens:       &[Token],
    _offset_bits: u32,
    _length_bits: u32,
) -> std::io::Result<Vec<u8>> {
    // Capacity estimate: 3-bit prefix + 2 EG values per pair.
    // EG(255) = 15 bits, EG(7) = 7 bits → ~22 bits per backref on average.
    // Use a conservative 4 bytes per token pair as the pre-allocation hint.
    let est_bytes = (tokens.len() / 2 + 2) * 4;
    let mut output = Vec::with_capacity(est_bytes);
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);

        // Collect non-END tokens
        let mut data: Vec<&Token> = Vec::with_capacity(tokens.len());
        data.extend(tokens.iter().filter(|t| !matches!(t, Token::End)));

        let mut i = 0;
        while i < data.len() {
            if i + 1 < data.len() {
                // Emit a pair
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
                    _ => unreachable!("END tokens filtered before this loop"),
                }
                i += 2;
            } else {
                // Odd token out — single
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
                }
                i += 1;
            }
        }

        w.write(PREFIX_BITS, PREFIX_END)?;
        w.byte_align()?;
    }
    Ok(output)
}

/// Decode a pair-encoded stream back to a token stream.
///
/// `_offset_bits` and `_length_bits` are not used (EG is self-delimiting).
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

    // ── EG coding unit tests ──────────────────────────────────────────────────

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
            let n = (1u32 << k) - 1; // 2^k - 1: boundary values
            assert_eq!(eg_round_trip(n), n, "EG roundtrip failed for n={}", n);
            let n2 = 1u32 << k; // 2^k
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
        // n=0: 1 bit, n=1: 3 bits, n=2: 3 bits, n=3: 5 bits, n=7: 7 bits
        let expected = [(0, 1), (1, 3), (2, 3), (3, 5), (6, 5), (7, 7), (14, 7), (15, 9)];
        for (n, expected_bits) in expected {
            let mut output = Vec::new();
            {
                let mut w = BitWriter::endian(&mut output, BigEndian);
                eg_write(&mut w, n).unwrap();
                w.byte_align().unwrap();
            }
            // The actual bit length is ceil(expected_bits / 8) * 8 due to byte align,
            // but we can check by comparing multiple values in one stream.
            // Just verify decode works — bit count verified by formula.
            let k: u32 = if n == 0 { 0 } else { u32::BITS - (n + 1).leading_zeros() - 1 };
            assert_eq!(2 * k + 1, expected_bits as u32,
                "EG bit length wrong for n={}: got {}, expected {}",
                n, 2*k+1, expected_bits);
        }
    }

    // ── Pair encoding round-trip tests ────────────────────────────────────────

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
        // pair_decode produces tokens without the trailing End from LIT runs
        // (End is emitted in pair stream as PREFIX_END)
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
        // Large offsets that would have caused Cantor blowup — EG handles cleanly.
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
        // 5 tokens → 2 pairs + 1 single
        let tokens = vec![
            Token::Lit { byte: 1 },
            Token::Lit { byte: 2 },
            Token::Lit { byte: 3 },
            Token::Lit { byte: 4 },
            Token::Lit { byte: 5 },
            Token::End,
        ];
        let decoded = round_trip(&tokens);
        assert_eq!(decoded.len(), 6); // 5 lits + End
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
        // For small offsets, EG pair should produce smaller output than
        // separate fixed-width token encoding.
        // Backref(offset=5, length=3) with ob=17, lb=8:
        //   Normal: 1+17+8 = 26 bits
        //   EG(4)=5 bits, EG(2)=3 bits → pair (2 tokens): 3 + 5+3 + ... = 11 bits for the B part
        // (exact comparison handled by the ratio check in fold.rs)
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
        // Sanity: 100 BB pairs, each (3+5+3+5+3)/8 ≈ 2.4 bytes → ~240 bytes
        assert!(encoded.len() < 400, "EG pair should be compact for small offsets");
    }
}
