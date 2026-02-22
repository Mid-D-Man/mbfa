// src/entropy.rs
//! Joint Huffman entropy coding with deflate-style offset bucket coding.
//!
//! Symbol table layout:
//!   0–255:   LIT byte values
//!   256:     END
//!   257–510: BACKREF length symbols  (sym = 255 + length)
//!   512–545: Offset bucket symbols   (sym = SYM_OFFSET_BASE + bucket_index)
//!
//! For each BACKREF, two symbols are written: the length symbol then the
//! offset bucket symbol. The bucket symbol is followed by a fixed number
//! of raw extra bits (the within-bucket remainder). The bucket table is
//! fixed and shared between encoder and decoder — never transmitted.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

/// Minimum compressed stream size (bytes) before joint entropy fires.
pub const ENTROPY_MIN_BYTES: usize = 800;

/// Offset bucket table: (base_offset, extra_bits).
/// Fixed vocabulary — modelled on DEFLATE distance codes, extended to cover
/// the full offset_bits=17 window (max offset 131071).
/// 34 buckets, contiguous, covering offsets 1..=131072.
const OFFSET_BUCKETS: &[(u32, u32)] = &[
    (1,      0),  //  0: offset 1
    (2,      0),  //  1: offset 2
    (3,      0),  //  2: offset 3
    (4,      0),  //  3: offset 4
    (5,      1),  //  4: 5–6
    (7,      1),  //  5: 7–8
    (9,      2),  //  6: 9–12
    (13,     2),  //  7: 13–16
    (17,     3),  //  8: 17–24
    (25,     3),  //  9: 25–32
    (33,     4),  // 10: 33–48
    (49,     4),  // 11: 49–64
    (65,     5),  // 12: 65–96
    (97,     5),  // 13: 97–128
    (129,    6),  // 14: 129–192
    (193,    6),  // 15: 193–256
    (257,    7),  // 16: 257–384
    (385,    7),  // 17: 385–512
    (513,    8),  // 18: 513–768
    (769,    8),  // 19: 769–1024
    (1025,   9),  // 20: 1025–1536
    (1537,   9),  // 21: 1537–2048
    (2049,  10),  // 22: 2049–3072
    (3073,  10),  // 23: 3073–4096
    (4097,  11),  // 24: 4097–6144
    (6145,  11),  // 25: 6145–8192
    (8193,  12),  // 26: 8193–12288
    (12289, 12),  // 27: 12289–16384
    (16385, 13),  // 28: 16385–24576
    (24577, 13),  // 29: 24577–32768
    (32769, 14),  // 30: 32769–49152
    (49153, 14),  // 31: 49153–65536
    (65537, 15),  // 32: 65537–98304
    (98305, 15),  // 33: 98305–131072
];

const N_OFFSET_BUCKETS: usize = 34;
const SYM_END: u32 = 256;
const SYM_OFFSET_BASE: u32 = 512;

#[inline] fn sym_from_length(len: u32) -> u32 { 255 + len }
#[inline] fn length_from_sym(sym: u32) -> u32 { sym - 255 }

/// Map offset → (bucket_index, extra_value).
/// Uses binary search on the fixed base table.
fn offset_to_bucket(offset: u32) -> (u32, u32) {
    let mut lo = 0usize;
    let mut hi = N_OFFSET_BUCKETS - 1;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if OFFSET_BUCKETS[mid].0 <= offset {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (base, _) = OFFSET_BUCKETS[lo];
    (lo as u32, offset - base)
}

/// Reconstruct offset from bucket index and extra value.
#[inline]
fn bucket_to_offset(bucket: u32, extra: u32) -> u32 {
    OFFSET_BUCKETS[bucket as usize].0 + extra
}

pub type EncodeTable = HashMap<u32, (u32, u32)>;
pub type DecodeTable = HashMap<(u32, u32), u32>;

// ── Frequency counting ────────────────────────────────────────────────────────
// Each BACKREF contributes TWO symbols: the length symbol and the offset
// bucket symbol. Both need Huffman codes in the joint table.

fn count_joint_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for t in tokens {
        match t {
            Token::Lit { byte } => {
                *freq.entry(*byte as u32).or_insert(0) += 1;
            }
            Token::Backref { offset, length } => {
                *freq.entry(sym_from_length(*length)).or_insert(0) += 1;
                let (bucket, _) = offset_to_bucket(*offset);
                *freq.entry(SYM_OFFSET_BASE + bucket).or_insert(0) += 1;
            }
            Token::End => {
                *freq.entry(SYM_END).or_insert(0) += 1;
            }
        }
    }
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

    let mut node_freq:   Vec<u64>           = Vec::with_capacity(2 * n);
    let mut left_child:  Vec<Option<usize>> = Vec::with_capacity(2 * n);
    let mut right_child: Vec<Option<usize>> = Vec::with_capacity(2 * n);

    let mut sym_to_node: HashMap<u32, usize> = HashMap::new();
    let mut sym_list: Vec<(u32, u64)> = freq.iter().map(|(&s, &f)| (s, f)).collect();
    sym_list.sort_by_key(|&(s, _)| s);

    for (sym, f) in &sym_list {
        let id = node_freq.len();
        sym_to_node.insert(*sym, id);
        node_freq.push(*f);
        left_child.push(None);
        right_child.push(None);
    }

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
        sym_to_node.iter().map(|(&s, &id)| (id, s)).collect();

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
    let mut sorted: Vec<(u32, u32)> = lengths.iter().map(|(&s, &l)| (s, l)).collect();
    sorted.sort_by_key(|&(s, l)| (l, s));

    let mut table    = EncodeTable::new();
    let mut code     = 0u32;
    let mut prev_len = 0u32;

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

pub fn build_encode_table(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_joint_freq(tokens);
    if freq.is_empty() { return None; }
    let lengths = assign_code_lengths(&freq);
    Some(canonical_codes_from_lengths(&lengths))
}

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
// offset_bits parameter removed — offsets are now bucket-coded via the fixed
// OFFSET_BUCKETS table. No raw offset field is written.

pub fn write_tokens_joint(
    tokens: &[Token],
    table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
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
                    // 1. Write Huffman code for the length symbol.
                    let lsym = sym_from_length(*length);
                    let &(lcode, llen) = table.get(&lsym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("backref len sym {} not in table", lsym))
                    })?;
                    w.write(llen, lcode)?;

                    // 2. Write Huffman code for the offset bucket symbol.
                    let (bucket, extra_val) = offset_to_bucket(*offset);
                    let osym = SYM_OFFSET_BASE + bucket;
                    let &(ocode, olen) = table.get(&osym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("offset bucket sym {} not in table", osym))
                    })?;
                    w.write(olen, ocode)?;

                    // 3. Write raw extra bits for within-bucket precision.
                    let extra_bits = OFFSET_BUCKETS[bucket as usize].1;
                    if extra_bits > 0 {
                        w.write(extra_bits, extra_val)?;
                    }
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

pub fn read_tokens_joint(
    input: &[u8],
    dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    let max_code_len = dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let sym = match read_huffman_sym(&mut r, dtable, max_code_len) {
            Ok(s)  => s,
            Err(_) => break,
        };

        if sym < 256 {
            // LIT byte
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else if sym >= 257 && sym < SYM_OFFSET_BASE {
            // BACKREF: length symbol followed immediately by offset bucket symbol.
            let length = length_from_sym(sym);

            let osym = match read_huffman_sym(&mut r, dtable, max_code_len) {
                Ok(s)  => s,
                Err(e) => return Err(e),
            };
            if osym < SYM_OFFSET_BASE || osym >= SYM_OFFSET_BASE + N_OFFSET_BUCKETS as u32 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected offset bucket sym (>={}), got {}", SYM_OFFSET_BASE, osym),
                ));
            }

            let bucket = osym - SYM_OFFSET_BASE;
            let extra_bits = OFFSET_BUCKETS[bucket as usize].1;
            let extra_val = if extra_bits > 0 {
                r.read::<u32>(extra_bits)?
            } else {
                0
            };

            let offset = bucket_to_offset(bucket, extra_val);
            tokens.push(Token::Backref { offset, length });
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected top-level sym {} in entropy stream", sym),
            ));
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
