// src/entropy.rs
//! Joint Huffman entropy coding — fold output post-processing.
//!
//! Joint alphabet (symbols 0–510):
//!   0–255  = LIT byte values
//!   256    = END of stream
//!   257–510 = BACKREF length codes (symbol = 255 + length, length = 2..255)
//!
//! Offsets are encoded raw at OFFSET_BITS (15) bits immediately after each
//! BACKREF length symbol. Offsets have too flat a distribution for a Huffman
//! table to pay off vs the header cost.
//!
//! This eliminates the fixed LIT opcode (2 bits per LIT token) and the fixed
//! BACKREF opcode (1 bit) + length field (8 bits), replacing them with a
//! single variable-length code per token. Savings are ~2 bits per LIT token
//! and ~6 bits per BACKREF token on typical prose.
//!
//! Only applied when fold output exceeds ENTROPY_MIN_BYTES and only on
//! non-pair-encoded streams.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::{Token, OFFSET_BITS};

/// Minimum compressed stream size in bytes before joint entropy fires.
pub const ENTROPY_MIN_BYTES: usize = 5000;

// ── Symbol encoding ───────────────────────────────────────────────────────────
const SYM_END: u32 = 256;
#[inline] fn sym_from_length(len: u32)  -> u32 { 255 + len }       // len 2..255 → 257..510
#[inline] fn length_from_sym(sym: u32)  -> u32 { sym - 255 }       // 257..510 → 2..255

pub type EncodeTable = HashMap<u32, (u32, u32)>; // symbol → (code, bit_len)
pub type DecodeTable = HashMap<(u32, u32), u32>; // (code, bit_len) → symbol

// ── Frequency counting ────────────────────────────────────────────────────────

fn count_joint_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for t in tokens {
        match t {
            Token::Lit { byte }          => { *freq.entry(*byte as u32).or_insert(0) += 1; }
            Token::Backref { length, .. } => { *freq.entry(sym_from_length(*length)).or_insert(0) += 1; }
            Token::End                    => { *freq.entry(SYM_END).or_insert(0) += 1; }
        }
    }
    // Ensure END is always present even if tokens had no End marker
    freq.entry(SYM_END).or_insert(1);
    freq
}

// ── Huffman tree → code lengths ───────────────────────────────────────────────

fn assign_code_lengths(freq: &HashMap<u32, u64>) -> HashMap<u32, u32> {
    let n = freq.len();
    if n == 0 { return HashMap::new(); }
    if n == 1 {
        let sym = *freq.keys().next().unwrap();
        return [(sym, 1)].into_iter().collect();
    }

    // Flat node store
    let mut node_freq:  Vec<u64>         = Vec::with_capacity(2 * n);
    let mut left_child: Vec<Option<usize>> = Vec::with_capacity(2 * n);
    let mut right_child:Vec<Option<usize>> = Vec::with_capacity(2 * n);

    let mut sym_to_node: HashMap<u32, usize> = HashMap::new();
    // Sort for determinism before inserting into heap
    let mut sym_list: Vec<(u32, u64)> = freq.iter().map(|(&s,&f)|(s,f)).collect();
    sym_list.sort_by_key(|&(s,_)| s);

    for (sym, f) in &sym_list {
        let id = node_freq.len();
        sym_to_node.insert(*sym, id);
        node_freq.push(*f);
        left_child.push(None);
        right_child.push(None);
    }

    // (Reverse<freq>, tie-break counter, node_id)
    let mut heap: BinaryHeap<(Reverse<u64>, Reverse<usize>, usize)> = sym_to_node
        .values()
        .map(|&id| (Reverse(node_freq[id]), Reverse(id), id))
        .collect();

    let mut counter = node_freq.len();
    while heap.len() > 1 {
        let (Reverse(f1), _, id1) = heap.pop().unwrap();
        let (Reverse(f2), _, id2) = heap.pop().unwrap();
        let pid = node_freq.len();
        node_freq.push(f1 + f2);
        left_child.push(Some(id1));
        right_child.push(Some(id2));
        heap.push((Reverse(f1 + f2), Reverse(counter), pid));
        counter += 1;
    }

    let root = heap.pop().unwrap().2;
    let node_to_sym: HashMap<usize, u32> =
        sym_to_node.iter().map(|(&s,&id)| (id,s)).collect();

    // Iterative DFS for depths
    let mut depths: HashMap<u32, u32> = HashMap::new();
    let mut stack: Vec<(usize, u32)> = vec![(root, 0)];
    while let Some((node, depth)) = stack.pop() {
        if left_child[node].is_none() {
            if let Some(&sym) = node_to_sym.get(&node) {
                depths.insert(sym, depth.max(1));
            }
        } else {
            if let Some(l) = left_child[node]  { stack.push((l, depth + 1)); }
            if let Some(r) = right_child[node] { stack.push((r, depth + 1)); }
        }
    }
    depths
}

// ── Canonical code assignment ─────────────────────────────────────────────────

fn canonical_codes_from_lengths(lengths: &HashMap<u32, u32>) -> EncodeTable {
    let mut sorted: Vec<(u32, u32)> = lengths.iter().map(|(&s,&l)|(s,l)).collect();
    sorted.sort_by_key(|&(s,l)| (l, s));

    let mut table = EncodeTable::new();
    let mut code: u32 = 0;
    let mut prev_len: u32 = 0;

    for (sym, len) in sorted {
        if len == 0 { continue; }
        if prev_len > 0 {
            code = (code + 1) << (len - prev_len);
        }
        table.insert(sym, (code, len));
        prev_len = len;
    }
    table
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Build a joint encode table from a token stream.
pub fn build_encode_table(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_joint_freq(tokens);
    if freq.is_empty() { return None; }
    let lengths = assign_code_lengths(&freq);
    Some(canonical_codes_from_lengths(&lengths))
}

/// Build a decode table from an encode table.
pub fn decode_table_from_encode(enc: &EncodeTable) -> DecodeTable {
    enc.iter().map(|(&sym, &(code, len))| ((code, len), sym)).collect()
}

/// Serialize: [n_symbols: u16 LE] [n × (symbol: u16 LE, bit_len: u8)]
pub fn serialize_table(table: &EncodeTable) -> Vec<u8> {
    let mut entries: Vec<(u32, u8)> = table
        .iter()
        .map(|(&s, &(_, l))| (s, l as u8))
        .collect();
    entries.sort_by_key(|&(s, _)| s);

    let n = entries.len() as u16;
    let mut out = Vec::with_capacity(2 + entries.len() * 3);
    out.extend_from_slice(&n.to_le_bytes());
    for (sym, len) in entries {
        out.extend_from_slice(&(sym as u16).to_le_bytes());
        out.push(len);
    }
    out
}

/// Deserialize table. Returns (encode_table, bytes_consumed).
pub fn deserialize_table(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof, "joint table: too short"));
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let needed = 2 + n * 3;
    if data.len() < needed {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof, "joint table: truncated"));
    }
    let mut lengths: HashMap<u32, u32> = HashMap::new();
    for i in 0..n {
        let base = 2 + i * 3;
        let sym = u16::from_le_bytes([data[base], data[base + 1]]) as u32;
        let len = data[base + 2] as u32;
        lengths.insert(sym, len);
    }
    let table = canonical_codes_from_lengths(&lengths);
    Ok((table, needed))
}

// ── Bit-level encode ──────────────────────────────────────────────────────────

/// Encode a token stream using the joint Huffman table.
/// LIT  → joint_code(byte_value)
/// END  → joint_code(256)
/// BACKREF → joint_code(255+length) + raw OFFSET_BITS offset
pub fn write_tokens_joint(tokens: &[Token], table: &EncodeTable) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    let sym = *byte as u32;
                    let &(code, len) = table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("lit sym {} not in table", sym))
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("backref len sym {} not in table", sym))
                    })?;
                    w.write(len, code)?;
                    w.write(OFFSET_BITS, *offset)?;
                }
                Token::End => {
                    let &(code, len) = table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            "END sym not in table")
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

// ── Bit-level decode ──────────────────────────────────────────────────────────

/// Decode a joint-Huffman bitstream back into a token stream.
pub fn read_tokens_joint(input: &[u8], dtable: &DecodeTable) -> std::io::Result<Vec<Token>> {
    let max_code_len = dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        // Read the joint Huffman code for the next symbol
        let sym = match read_huffman_sym(&mut r, dtable, max_code_len) {
            Ok(s)  => s,
            Err(_) => break,  // EOF — byte-aligned padding may cause this
        };

        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            // BACKREF length symbol — read raw offset immediately after
            let length = length_from_sym(sym);
            let offset = match r.read::<u32>(OFFSET_BITS) {
                Ok(o)  => o,
                Err(e) => return Err(e),
            };
            tokens.push(Token::Backref { offset, length });
        }
    }

    Ok(tokens)
}

fn read_huffman_sym<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
    dtable: &DecodeTable,
    max_len: u32,
) -> std::io::Result<u32> {
    let mut code: u32 = 0;
    for len in 1..=max_len {
        let bit = r.read::<u32>(1)?;
        code = (code << 1) | bit;
        if let Some(&sym) = dtable.get(&(code, len)) {
            return Ok(sym);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid huffman symbol after {} bits", max_len),
    ))
}
