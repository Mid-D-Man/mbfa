// src/rans_entropy.rs
//! rANS entropy coding — entropy_flag = 7
//!
//! Uses the `rans` crate (wrapper over Fabian Giesen's ryg-rans reference impl).
//!
//! Structure mirrors v2 (joint lit/length table + separate offset bucket table)
//! but replaces Huffman integer-bit codes with rANS fractional-bit coding.
//!
//! LIFO property: rANS is a stack — symbols are decoded in reverse encode order.
//! This module handles the reversal transparently:
//!   - encode: iterate tokens BACKWARD, push bucket then lit/length per token
//!   - decode: iterate forward, get lit/length then bucket per token
//!
//! Extra bits (bucket residuals) are collected forward into a separate byte
//! stream appended after the rANS bytes. The decoder reads them sequentially.
//!
//! Output layout per compressed block:
//!   [rans_byte_len: u32 LE][rans_bytes: N bytes][extra_bit_bytes: M bytes]

use std::collections::HashMap;

use rans::byte_decoder::{ByteRansDecSymbol, ByteRansDecoder};
use rans::byte_encoder::{ByteRansEncSymbol, ByteRansEncoder};
use rans::{RansDecSymbol, RansDecoder, RansEncSymbol, RansEncoder, RansEncoderMulti};

use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};

use crate::entropy::{bucket_extra_bits, bucket_to_offset, count_offset_bucket_freq,
                     offset_to_bucket};
use crate::opcode::Token;

// ── Scale ─────────────────────────────────────────────────────────────────────

/// SCALE_BITS = 12 → SCALE = 4096.
/// Handles up to 4096 distinct symbols with freq ≥ 1 each.
/// The joint lit/length alphabet has at most ~320 symbols (256 bytes + lengths + END).
/// The offset bucket alphabet has at most ~56 symbols (48 buckets + 8 slots).
/// Both comfortably fit under 4096.
pub const SCALE_BITS: u32 = 12;
pub const SCALE: u32 = 1 << SCALE_BITS; // 4096

// ── Symbol helpers (mirror entropy.rs conventions) ────────────────────────────

const SYM_END: u32 = 256;

#[inline]
fn sym_from_length(len: u32) -> u32 { 255 + len }

#[inline]
fn length_from_sym(sym: u32) -> u32 { sym - 255 }

// ── Frequency counting ────────────────────────────────────────────────────────

fn count_joint_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for t in tokens {
        match t {
            Token::Lit { byte } => {
                *freq.entry(*byte as u32).or_insert(0) += 1;
            }
            Token::Backref { length, .. } => {
                *freq.entry(sym_from_length(*length)).or_insert(0) += 1;
            }
            Token::End => {
                *freq.entry(SYM_END).or_insert(0) += 1;
            }
        }
    }
    // Ensure END always present so decode always terminates
    freq.entry(SYM_END).or_insert(1);
    freq
}

// ── Normalization ─────────────────────────────────────────────────────────────

/// Normalize raw frequency counts to sum exactly to SCALE.
/// Every present symbol gets freq ≥ 1.
/// Returns (sym, normalized_freq) pairs sorted by sym ascending.
pub fn normalize_freqs(raw: &HashMap<u32, u64>) -> Vec<(u32, u32)> {
    if raw.is_empty() {
        return Vec::new();
    }

    let total: u64 = raw.values().sum();
    let mut entries: Vec<(u32, u64)> = raw.iter().map(|(&s, &f)| (s, f)).collect();
    entries.sort_by_key(|&(s, _)| s);

    // Proportional: floor(f * SCALE / total), minimum 1
    let mut normalized: Vec<(u32, u32)> = entries
        .iter()
        .map(|&(s, f)| {
            let scaled = ((f as u128 * SCALE as u128) / total as u128) as u32;
            (s, scaled.max(1))
        })
        .collect();

    let sum: u32 = normalized.iter().map(|&(_, f)| f).sum();

    if sum < SCALE {
        // Add remainder to the highest raw-frequency symbol
        let best = entries
            .iter()
            .enumerate()
            .max_by_key(|(_, &(_, f))| f)
            .map(|(i, _)| i)
            .unwrap_or(0);
        normalized[best].1 += SCALE - sum;
    } else if sum > SCALE {
        // Remove excess from highest-normalized symbols, keeping each ≥ 1
        let mut excess = sum - SCALE;
        let mut order: Vec<usize> = (0..normalized.len()).collect();
        order.sort_by(|&a, &b| normalized[b].1.cmp(&normalized[a].1));
        for idx in order {
            if excess == 0 {
                break;
            }
            let removable = normalized[idx].1.saturating_sub(1);
            let remove = removable.min(excess);
            normalized[idx].1 -= remove;
            excess -= remove;
        }
    }

    debug_assert_eq!(
        normalized.iter().map(|&(_, f)| f).sum::<u32>(),
        SCALE,
        "normalize_freqs: sum != SCALE"
    );

    normalized
}

// ── Table types ───────────────────────────────────────────────────────────────

/// Encode table: symbol → (cumulative_start, freq) normalized to SCALE
pub type RansEncTable = HashMap<u32, (u32, u32)>;

/// Decode table: array of SCALE entries; dec[cf] = symbol for cumulative index cf
pub type RansDecTable = Vec<u32>;

fn build_tables(normalized: &[(u32, u32)]) -> (RansEncTable, RansDecTable) {
    let mut enc = RansEncTable::new();
    let mut dec = vec![SYM_END; SCALE as usize]; // safe default = END
    let mut cum = 0u32;
    for &(sym, freq) in normalized {
        enc.insert(sym, (cum, freq));
        for i in cum..cum + freq {
            dec[i as usize] = sym;
        }
        cum += freq;
    }
    (enc, dec)
}

// ── Public table builders ─────────────────────────────────────────────────────

/// Build rANS tables for the joint lit/length symbol alphabet.
/// Returns (enc_table, dec_table, normalized_freqs) or None if tokens is empty.
pub fn build_lit_tables(
    tokens: &[Token],
) -> Option<(RansEncTable, RansDecTable, Vec<(u32, u32)>)> {
    let raw = count_joint_freq(tokens);
    if raw.is_empty() {
        return None;
    }
    let norm = normalize_freqs(&raw);
    let (enc, dec) = build_tables(&norm);
    Some((enc, dec, norm))
}

/// Build rANS tables for the offset bucket symbol alphabet.
/// Returns None if there are no BACKREF tokens.
pub fn build_offset_tables(
    tokens: &[Token],
) -> Option<(RansEncTable, RansDecTable, Vec<(u32, u32)>)> {
    let raw = count_offset_bucket_freq(tokens);
    if raw.is_empty() {
        return None;
    }
    let norm = normalize_freqs(&raw);
    let (enc, dec) = build_tables(&norm);
    Some((enc, dec, norm))
}

// ── Table serialization ───────────────────────────────────────────────────────

/// Serialize normalized freq table.
/// Format: [N: u16 LE][sym: u16 LE, freq: u16 LE] × N
pub fn serialize_rans_table(normalized: &[(u32, u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 4 * normalized.len());
    out.extend_from_slice(&(normalized.len() as u16).to_le_bytes());
    for &(sym, freq) in normalized {
        out.extend_from_slice(&(sym as u16).to_le_bytes());
        out.extend_from_slice(&(freq as u16).to_le_bytes());
    }
    out
}

/// Deserialize rANS freq table from `data`.
/// Returns (enc_table, dec_table, bytes_consumed).
pub fn deserialize_rans_table(
    data: &[u8],
) -> std::io::Result<(RansEncTable, RansDecTable, usize)> {
    if data.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "rans table: too short",
        ));
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let needed = 2 + 4 * n;
    if data.len() < needed {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "rans table: truncated",
        ));
    }

    let mut normalized: Vec<(u32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let b = 2 + i * 4;
        let sym = u16::from_le_bytes([data[b], data[b + 1]]) as u32;
        let freq = u16::from_le_bytes([data[b + 2], data[b + 3]]) as u32;
        normalized.push((sym, freq));
    }

    // Validate
    let sum: u32 = normalized.iter().map(|&(_, f)| f).sum();
    if sum != SCALE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("rans table: freq sum {} != SCALE {}", sum, SCALE),
        ));
    }

    let (enc, dec) = build_tables(&normalized);
    Ok((enc, dec, needed))
}

// ── Encoding ──────────────────────────────────────────────────────────────────

/// Encode token stream with rANS.
///
/// The rANS stream is LIFO: tokens are pushed in reverse so they decode forward.
/// For each token in reverse:
///   - BACKREF: push bucket symbol first (decoded second), then lit/length (decoded first)
///   - LIT / END: push lit symbol only
///
/// Extra bits (bucket residuals) are written in forward order into a separate
/// byte stream appended after the rANS bytes.
///
/// Output: [rans_byte_len: u32 LE][rans_bytes][extra_bit_bytes]
///
/// Returns None if any token's symbol is absent from the tables (safety guard).
pub fn write_tokens_rans(
    tokens: &[Token],
    lit_enc: &RansEncTable,
    offset_enc: &RansEncTable,
) -> Option<Vec<u8>> {
    // Pass 1: collect extra bits forward
    let mut extra_buf: Vec<u8> = Vec::new();
    {
        let mut w = BitWriter::endian(&mut extra_buf, BigEndian);
        for t in tokens {
            if let Token::Backref { offset, .. } = t {
                let (_, extra_cnt, extra_val) = offset_to_bucket(*offset);
                if extra_cnt > 0 {
                    w.write(extra_cnt, extra_val).ok()?;
                }
            }
        }
        w.byte_align().ok()?;
    }

    // Pass 2: encode rANS symbols in REVERSE token order
    let capacity = tokens.len() * 3 + 128;
    let mut encoder = ByteRansEncoder::new(capacity);

    for t in tokens.iter().rev() {
        match t {
            Token::Lit { byte } => {
                let &(cum, freq) = lit_enc.get(&(*byte as u32))?;
                encoder.put(&ByteRansEncSymbol::new(cum, freq, SCALE_BITS));
            }
            Token::Backref { offset, length } => {
                // Push bucket FIRST (decoded SECOND — LIFO inverts within token)
                let (bucket, _, _) = offset_to_bucket(*offset);
                let &(bcum, bfreq) = offset_enc.get(&bucket)?;
                encoder.put(&ByteRansEncSymbol::new(bcum, bfreq, SCALE_BITS));

                // Push lit/length SECOND (decoded FIRST)
                let sym = sym_from_length(*length);
                let &(cum, freq) = lit_enc.get(&sym)?;
                encoder.put(&ByteRansEncSymbol::new(cum, freq, SCALE_BITS));
            }
            Token::End => {
                let &(cum, freq) = lit_enc.get(&SYM_END)?;
                encoder.put(&ByteRansEncSymbol::new(cum, freq, SCALE_BITS));
            }
        }
    }

    encoder.flush();
    // RansEncoderMulti trait must be in scope for .data()
    let rans_bytes = encoder.data().to_owned();

    let mut out = Vec::with_capacity(4 + rans_bytes.len() + extra_buf.len());
    out.extend_from_slice(&(rans_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&rans_bytes);
    out.extend_from_slice(&extra_buf);

    Some(out)
}

// ── Decoding ──────────────────────────────────────────────────────────────────

/// Decode a rANS-encoded token stream.
///
/// Reads rANS symbols forward (which corresponds to reverse push order = forward
/// encode order). Reads extra bits (bucket residuals) from the appended byte stream.
pub fn read_tokens_rans(
    input: &[u8],
    lit_enc: &RansEncTable,
    lit_dec: &RansDecTable,
    offset_enc: &RansEncTable,
    offset_dec: &RansDecTable,
) -> std::io::Result<Vec<Token>> {
    if input.len() < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "rans stream: header too short",
        ));
    }

    let rans_len = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
    if input.len() < 4 + rans_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "rans stream: rans section truncated",
        ));
    }

    let rans_bytes = input[4..4 + rans_len].to_vec();
    let extra_bytes = &input[4 + rans_len..];

    let mut decoder = ByteRansDecoder::new(rans_bytes);
    let mut extra_r = BitReader::endian(std::io::Cursor::new(extra_bytes), BigEndian);

    let mut tokens = Vec::new();

    loop {
        // Decode lit/length symbol
        let cf = decoder.get(SCALE_BITS);
        if cf as usize >= lit_dec.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("rans: lit cf {} out of range [0, {})", cf, lit_dec.len()),
            ));
        }
        let sym = lit_dec[cf as usize];

        let &(cum, freq) = lit_enc.get(&sym).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("rans: decoded sym {} missing from lit_enc", sym),
            )
        })?;
        decoder.advance(&ByteRansDecSymbol::new(cum, freq), SCALE_BITS);

        if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else {
            let length = length_from_sym(sym);

            let bcf = decoder.get(SCALE_BITS);
            if bcf as usize >= offset_dec.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("rans: offset cf {} out of range", bcf),
                ));
            }
            let bucket = offset_dec[bcf as usize];

            let &(bcum, bfreq) = offset_enc.get(&bucket).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("rans: decoded bucket {} missing from offset_enc", bucket),
                )
            })?;
            decoder.advance(&ByteRansDecSymbol::new(bcum, bfreq), SCALE_BITS);

            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 {
                extra_r.read::<u32>(extra_cnt).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!("rans: extra bits read failed: {}", e),
                    )
                })?
            } else {
                0
            };

            let offset = bucket_to_offset(bucket, extra_val);
            tokens.push(Token::Backref { offset, length });
        }
    }

    Ok(tokens)
      }
