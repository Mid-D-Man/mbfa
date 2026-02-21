// src/entropy.rs
//! Canonical Huffman entropy coding applied to LIT byte values only.
//!
//! The opcode bits (BACKREF/LIT/END) stay fixed-width as normal.
//! Only the 8-bit byte payload inside LIT tokens is Huffman-coded.
//! This is applied as post-processing on the final fold output when
//! the stream is large enough for the table overhead to be worth it.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::{
    Token,
    OPCODE_BACKREF_BITS, OPCODE_BACKREF_VAL,
    OPCODE_LIT_BITS, OPCODE_LIT_VAL,
    OPCODE_END_BITS, OPCODE_END_VAL,
    OFFSET_BITS, LENGTH_BITS,
};

/// Minimum compressed stream size (bytes) before entropy coding fires.
/// Below this the table overhead dominates any savings.
pub const ENTROPY_MIN_BYTES: usize = 5000;

/// symbol -> (canonical_code: u32, bit_length: u32)
pub type EncodeTable = HashMap<u8, (u32, u32)>;
/// (canonical_code: u32, bit_length: u32) -> symbol
pub type DecodeTable = HashMap<(u32, u32), u8>;

// ── Frequency counting ───────────────────────────────────────────────────────

fn count_lit_freq(tokens: &[Token]) -> HashMap<u8, u64> {
    let mut freq: HashMap<u8, u64> = HashMap::new();
    for t in tokens {
        if let Token::Lit { byte } = t {
            *freq.entry(*byte).or_insert(0) += 1;
        }
    }
    freq
}

// ── Huffman tree construction → code lengths ─────────────────────────────────

fn assign_code_lengths(freq: &HashMap<u8, u64>) -> HashMap<u8, u32> {
    let n = freq.len();
    if n == 0 { return HashMap::new(); }
    if n == 1 {
        let sym = *freq.keys().next().unwrap();
        return [(sym, 1)].into_iter().collect();
    }

    // Flat node storage: index = node id
    let mut node_freq: Vec<u64> = Vec::with_capacity(2 * n);
    let mut left_child:  Vec<Option<usize>> = Vec::with_capacity(2 * n);
    let mut right_child: Vec<Option<usize>> = Vec::with_capacity(2 * n);

    // symbol -> leaf node id
    let mut sym_to_node: HashMap<u8, usize> = HashMap::new();

    for (&sym, &f) in freq {
        let id = node_freq.len();
        sym_to_node.insert(sym, id);
        node_freq.push(f);
        left_child.push(None);
        right_child.push(None);
    }

    // Min-heap by frequency
    let mut heap: BinaryHeap<(Reverse<u64>, usize)> = sym_to_node
        .values()
        .map(|&id| (Reverse(node_freq[id]), id))
        .collect();

    // Merge two lowest-frequency nodes until one root remains
    while heap.len() > 1 {
        let (Reverse(f1), id1) = heap.pop().unwrap();
        let (Reverse(f2), id2) = heap.pop().unwrap();
        let parent_id = node_freq.len();
        node_freq.push(f1 + f2);
        left_child.push(Some(id1));
        right_child.push(Some(id2));
        heap.push((Reverse(f1 + f2), parent_id));
    }

    let root = heap.pop().unwrap().1;

    // Iterative DFS to assign depths
    let node_to_sym: HashMap<usize, u8> = sym_to_node.iter().map(|(&s, &id)| (id, s)).collect();
    let mut depths: HashMap<u8, u32> = HashMap::new();
    let mut stack: Vec<(usize, u32)> = vec![(root, 0)];

    while let Some((node, depth)) = stack.pop() {
        if left_child[node].is_none() {
            // Leaf node
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

// ── Canonical code assignment ────────────────────────────────────────────────

fn canonical_codes_from_lengths(lengths: &HashMap<u8, u32>) -> EncodeTable {
    // Sort by (length, symbol) — this is the canonical ordering
    let mut sorted: Vec<(u8, u32)> = lengths.iter().map(|(&s, &l)| (s, l)).collect();
    sorted.sort_by_key(|&(s, l)| (l, s));

    let mut table = EncodeTable::new();
    let mut code: u32 = 0;
    let mut prev_len: u32 = 0;

    for (sym, len) in sorted {
        if len == 0 { continue; }
        if prev_len > 0 {
            // Shift code left by the difference in lengths, then increment
            code = (code + 1) << (len - prev_len);
        }
        // First symbol: code stays 0
        table.insert(sym, (code, len));
        prev_len = len;
    }

    table
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Build an encode table from the LIT byte distribution in a token stream.
/// Returns None if there are no LIT tokens.
pub fn build_encode_table(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_lit_freq(tokens);
    if freq.is_empty() { return None; }
    let lengths = assign_code_lengths(&freq);
    Some(canonical_codes_from_lengths(&lengths))
}

/// Build a decode table from an encode table.
pub fn decode_table_from_encode(enc: &EncodeTable) -> DecodeTable {
    enc.iter().map(|(&sym, &(code, len))| ((code, len), sym)).collect()
}

/// Serialize table: [n_symbols: u8] then n × [symbol: u8, bit_length: u8]
/// Sorted by symbol for determinism. Total: 1 + 2n bytes.
pub fn serialize_table(table: &EncodeTable) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(table.len() as u8);
    let mut entries: Vec<(u8, u8)> = table
        .iter()
        .map(|(&s, &(_, l))| (s, l as u8))
        .collect();
    entries.sort_by_key(|&(s, _)| s);
    for (sym, len) in entries {
        out.push(sym);
        out.push(len);
    }
    out
}

/// Deserialize table. Returns (encode_table, bytes_consumed).
pub fn deserialize_table(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entropy table: empty",
        ));
    }
    let n = data[0] as usize;
    let needed = 1 + n * 2;
    if data.len() < needed {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "entropy table: truncated",
        ));
    }
    let mut lengths: HashMap<u8, u32> = HashMap::new();
    for i in 0..n {
        let sym = data[1 + i * 2];
        let len = data[2 + i * 2] as u32;
        lengths.insert(sym, len);
    }
    // Rebuild canonical codes from the same lengths — encoder and decoder
    // will produce identical codes because canonical assignment is deterministic.
    let table = canonical_codes_from_lengths(&lengths);
    Ok((table, needed))
}

// ── Bit-level encode / decode ────────────────────────────────────────────────

/// Write a token stream to bytes.
/// BACKREF and END use fixed opcodes as normal.
/// LIT opcode bits are fixed (2 bits "10"), but the byte payload uses Huffman codes.
pub fn write_tokens_entropy(tokens: &[Token], table: &EncodeTable) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    w.write(OPCODE_LIT_BITS, OPCODE_LIT_VAL)?;
                    match table.get(byte) {
                        Some(&(code, len)) => w.write(len, code)?,
                        None => {
                            // Symbol not seen during table build — shouldn't happen
                            // since we build from the same token stream we encode.
                            // Fallback: raw 8 bits (marks this token as uncompressed).
                            w.write(8, *byte as u32)?;
                        }
                    }
                }
                Token::Backref { offset, length } => {
                    w.write(OPCODE_BACKREF_BITS, OPCODE_BACKREF_VAL)?;
                    w.write(OFFSET_BITS, *offset)?;
                    w.write(LENGTH_BITS, *length)?;
                }
                Token::End => {
                    w.write(OPCODE_END_BITS, OPCODE_END_VAL)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

/// Read a token stream from a Huffman-coded bitstream.
pub fn read_tokens_entropy(input: &[u8], dtable: &DecodeTable) -> std::io::Result<Vec<Token>> {
    let max_code_len = dtable.keys().map(|&(_, l)| l).max().unwrap_or(16);
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let first_bit = match r.read::<u32>(1) {
            Ok(b) => b,
            Err(_) => break,
        };

        if first_bit == OPCODE_BACKREF_VAL {
            // BACKREF: fixed-width operands, unchanged
            let offset = r.read::<u32>(OFFSET_BITS)?;
            let length = r.read::<u32>(LENGTH_BITS)?;
            tokens.push(Token::Backref { offset, length });
        } else {
            let second_bit = r.read::<u32>(1)?;
            if second_bit == 1 {
                // END
                tokens.push(Token::End);
                break;
            }
            // LIT: read Huffman-coded byte value
            // Huffman payload only starts here, after both opcode bits are consumed.
            // No ambiguity with BACKREF opcode (which is caught by first_bit == 0 above).
            let byte = read_huffman_byte(&mut r, dtable, max_code_len)?;
            tokens.push(Token::Lit { byte });
        }
    }

    Ok(tokens)
}

fn read_huffman_byte<R: std::io::Read>(
    r: &mut BitReader<R, BigEndian>,
    dtable: &DecodeTable,
    max_len: u32,
) -> std::io::Result<u8> {
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
        format!("invalid huffman code after {} bits: {:#b}", max_len, code),
    ))
                                }
