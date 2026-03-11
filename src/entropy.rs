// src/entropy.rs
//
// Entropy coding variants — all operate on the fold-1 token stream.
// Variant numbering:
//
//   v1  entropy_flag=1  joint lit/length Huffman + offset bucket Huffman
//   v2  entropy_flag=2  2-context lit/length Huffman + offset bucket Huffman
//   v3  entropy_flag=3  8-context lit/length Huffman + offset bucket Huffman
//   v4  entropy_flag=4  v1 + recent-offset slot reuse
//   v5  entropy_flag=5  v2 + recent-offset slot reuse
//   v6  entropy_flag=6  separate literal stream + sequence stream Huffman
//
// v6 architecture (zstd-style literal separation):
//   Literal bytes are extracted from the token stream, Huffman-coded as a
//   dedicated byte-only stream, and stored separately from the sequence stream.
//   The sequence stream encodes SYM_V6_LIT (a single marker symbol) in place
//   of each literal's inline byte value. The sequence Huffman table covers only
//   {SYM_V6_LIT, length_syms, SYM_END} — no byte values — so length symbols
//   get full frequency mass and shorter codes. Literal bytes get a clean
//   byte-only table with no dilution from length symbols.
//
//   v6 payload layout:
//     [lit_huffman_table]
//     [seq_huffman_table]
//     [offset_huffman_table]
//     [lit_count:           u32 LE]  — number of literal bytes encoded
//     [lit_bitstream_len:   u32 LE]  — byte length of lit_bitstream
//     [lit_bitstream]
//     [seq_bitstream]
//
// NOTE: entropy_flag=7 (prose-tuned 8-context split) was trialled in session 6
// and removed — it lost to v3 on WarAndPeace by 433B and won nowhere.
// The match arm in unfold.rs is kept to return a clean error for any files
// compressed with that development build. No v7 code exists in this file.
//
// Core algorithm (fold/unfold/LZ/pairing) is untouched by this module.
//
// Frequency counting uses fixed arrays instead of HashMap:
//   - Lit/length channel: vec![0u64; 65536]
//   - Offset bucket channel: [u64; 64]
//   - Slot channel: [u64; NUM_SLOTS]

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

pub const ENTROPY_MIN_BYTES:    usize = 400;
pub const ENTROPY_V2_MIN_BYTES: usize = 1000;

const SYM_END: u32 = 256;
#[inline] fn sym_from_length(len: u32) -> u32 { 255 + len }
#[inline] fn length_from_sym(sym: u32)  -> u32 { sym - 255 }

const SYM_V6_LIT: u32 = 0;

const LIT_LEN_FREQ_SIZE: usize = 65536;
const BUCKET_FREQ_SIZE:  usize = 64;

pub type EncodeTable = HashMap<u32, (u32, u32)>;
pub type DecodeTable = HashMap<(u32, u32), u32>;

// ── Recent-offset slot reuse ──────────────────────────────────────────────────

pub const NUM_SLOTS:        usize = 8;
pub const SLOT_SYMBOL_BASE: u32   = 1000;

pub struct OffsetSlots {
    slots: [u32; NUM_SLOTS],
    count: usize,
}

impl OffsetSlots {
    pub fn new() -> Self { Self { slots: [0; NUM_SLOTS], count: 0 } }

    pub fn access(&mut self, offset: u32) -> Option<usize> {
        let found = (0..self.count).find(|&i| self.slots[i] == offset);
        if let Some(idx) = found {
            for j in (1..=idx).rev() { self.slots[j] = self.slots[j - 1]; }
            self.slots[0] = offset;
            Some(idx)
        } else {
            let new_count = (self.count + 1).min(NUM_SLOTS);
            for j in (1..new_count).rev() { self.slots[j] = self.slots[j - 1]; }
            self.slots[0] = offset;
            self.count     = new_count;
            None
        }
    }

    pub fn access_by_slot(&mut self, slot_idx: usize) -> Option<u32> {
        if slot_idx >= self.count { return None; }
        let offset = self.slots[slot_idx];
        for j in (1..=slot_idx).rev() { self.slots[j] = self.slots[j - 1]; }
        self.slots[0] = offset;
        Some(offset)
    }
}

// ── Offset bucket scheme ──────────────────────────────────────────────────────

pub fn offset_to_bucket(offset: u32) -> (u32, u32, u32) {
    debug_assert!(offset >= 1);
    if offset <= 4 { return (offset - 1, 0, 0); }
    let extra_bits = (offset - 1).ilog2().saturating_sub(1);
    let size  = 1u32 << extra_bits;
    let base  = 1 + 2 * size;
    let half  = (offset - base) / size;
    let bucket = 2 + 2 * extra_bits + half;
    let extra_val = offset - base - half * size;
    (bucket, extra_bits, extra_val)
}

pub fn bucket_to_offset(bucket: u32, extra_val: u32) -> u32 {
    if bucket < 4 { return bucket + 1; }
    let extra_bits = (bucket - 2) >> 1;
    let size = 1u32 << extra_bits;
    let base = 1 + 2 * size;
    let half = bucket & 1;
    base + half * size + extra_val
}

#[inline]
pub fn bucket_extra_bits(bucket: u32) -> u32 {
    if bucket < 4 { 0 } else { (bucket - 2) >> 1 }
}

// ── Frequency counting — fixed array implementations ─────────────────────────

#[inline]
fn lit_len_array_to_map(freq: &[u64]) -> HashMap<u32, u64> {
    let mut map: HashMap<u32, u64> = freq.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u32, c))
        .collect();
    map.entry(SYM_END).or_insert(1);
    map
}

#[inline]
fn bucket_array_to_map(freq: &[u64; BUCKET_FREQ_SIZE]) -> HashMap<u32, u64> {
    freq.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u32, c))
        .collect()
}

fn count_joint_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq = vec![0u64; LIT_LEN_FREQ_SIZE];
    for t in tokens {
        match t {
            Token::Lit { byte }           => freq[*byte as usize] += 1,
            Token::Backref { length, .. } => freq[(255 + length) as usize] += 1,
            Token::End                    => freq[SYM_END as usize] += 1,
        }
    }
    lit_len_array_to_map(&freq)
}

pub fn count_offset_bucket_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq = [0u64; BUCKET_FREQ_SIZE];
    for t in tokens {
        if let Token::Backref { offset, .. } = t {
            let (bucket, _, _) = offset_to_bucket(*offset);
            freq[bucket as usize] += 1;
        }
    }
    bucket_array_to_map(&freq)
}

pub fn count_offset_bucket_freq_slotted(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut bucket_freq = [0u64; BUCKET_FREQ_SIZE];
    let mut slot_freq   = [0u64; NUM_SLOTS];
    let mut slots       = OffsetSlots::new();

    for t in tokens {
        if let Token::Backref { offset, .. } = t {
            if let Some(slot_idx) = slots.access(*offset) {
                slot_freq[slot_idx] += 1;
            } else {
                let (bucket, _, _) = offset_to_bucket(*offset);
                bucket_freq[bucket as usize] += 1;
            }
        }
    }

    let mut map = bucket_array_to_map(&bucket_freq);
    for (i, &c) in slot_freq.iter().enumerate() {
        if c > 0 {
            map.insert(SLOT_SYMBOL_BASE + i as u32, c);
        }
    }
    map
}

fn count_joint_freq_by_context(tokens: &[Token]) -> (HashMap<u32, u64>, HashMap<u32, u64>) {
    let mut freq0 = vec![0u64; LIT_LEN_FREQ_SIZE];
    let mut freq1 = vec![0u64; LIT_LEN_FREQ_SIZE];
    let mut after_br = false;

    for t in tokens {
        let freq = if after_br { &mut freq1 } else { &mut freq0 };
        match t {
            Token::Lit { byte } => {
                freq[*byte as usize] += 1;
                after_br = false;
            }
            Token::Backref { length, .. } => {
                freq[(255 + length) as usize] += 1;
                after_br = true;
            }
            Token::End => {
                freq[SYM_END as usize] += 1;
            }
        }
    }

    (lit_len_array_to_map(&freq0), lit_len_array_to_map(&freq1))
}

// ── Huffman tree ──────────────────────────────────────────────────────────────

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
    let node_to_sym: HashMap<usize, u32> = sym_to_node.iter().map(|(&s, &id)| (id, s)).collect();

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

fn canonical_codes_from_lengths(lengths: &HashMap<u32, u32>) -> EncodeTable {
    let mut sorted: Vec<(u32, u32)> = lengths.iter().map(|(&s, &l)| (s, l)).collect();
    sorted.sort_by_key(|&(s, l)| (l, s));
    let mut table = EncodeTable::new();
    let mut code  = 0u32;
    let mut prev_len = 0u32;
    for (sym, len) in sorted {
        if len == 0 { continue; }
        if prev_len > 0 { code = (code + 1) << (len - prev_len); }
        table.insert(sym, (code, len));
        prev_len = len;
    }
    table
}

// ── Public table builders ─────────────────────────────────────────────────────

pub fn build_encode_table(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_joint_freq(tokens);
    if freq.is_empty() { return None; }
    Some(canonical_codes_from_lengths(&assign_code_lengths(&freq)))
}

pub fn build_offset_encode_table(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_offset_bucket_freq(tokens);
    if freq.is_empty() { return None; }
    Some(canonical_codes_from_lengths(&assign_code_lengths(&freq)))
}

pub fn build_offset_encode_table_slotted(tokens: &[Token]) -> Option<EncodeTable> {
    let freq = count_offset_bucket_freq_slotted(tokens);
    if freq.is_empty() { return None; }
    Some(canonical_codes_from_lengths(&assign_code_lengths(&freq)))
}

pub fn build_encode_tables_by_context(tokens: &[Token]) -> Option<(EncodeTable, EncodeTable)> {
    let (freq0, freq1) = count_joint_freq_by_context(tokens);
    if freq1.is_empty() { return None; }
    let l0 = assign_code_lengths(&freq0);
    let l1 = assign_code_lengths(&freq1);
    Some((canonical_codes_from_lengths(&l0), canonical_codes_from_lengths(&l1)))
}

pub fn decode_table_from_encode(enc: &EncodeTable) -> DecodeTable {
    enc.iter().map(|(&sym, &(code, len))| ((code, len), sym)).collect()
}

// ── v3 eight-context builder ──────────────────────────────────────────────────

#[inline]
pub fn byte_category(b: u8) -> usize {
    match b {
        b'a' | b'e' | b'i' | b'o' | b'u' => 0,
        b'A'..=b'Z' => 1,
        b'b'..=b'z' => 1,
        9 | 10 | 13 | 32        => 2,
        33 | 44 | 46 | 58 | 59  => 2,
        39 | 40 | 41 | 63 | 45  => 2,
        34 | 91 | 93 | 123 | 125 => 2,
        _ => 3,
    }
}

#[inline]
pub fn context_idx(after_br: bool, prev_byte: u8) -> usize {
    (after_br as usize) * 4 + byte_category(prev_byte)
}

fn count_joint_freq_v3(tokens: &[Token]) -> ([HashMap<u32, u64>; 8], HashMap<u32, u64>) {
    let mut lit_freqs: [Vec<u64>; 8] = std::array::from_fn(|_| vec![0u64; LIT_LEN_FREQ_SIZE]);
    let mut bucket_freq = [0u64; BUCKET_FREQ_SIZE];

    let mut prev_byte: u8 = b' ';
    let mut after_br      = false;

    for t in tokens {
        let ctx = context_idx(after_br, prev_byte);
        match t {
            Token::Lit { byte } => {
                lit_freqs[ctx][*byte as usize] += 1;
                prev_byte = *byte;
                after_br  = false;
            }
            Token::Backref { offset, length } => {
                lit_freqs[ctx][(255 + length) as usize] += 1;
                let (bucket, _, _) = offset_to_bucket(*offset);
                bucket_freq[bucket as usize] += 1;
                after_br = true;
            }
            Token::End => {
                lit_freqs[ctx][SYM_END as usize] += 1;
            }
        }
    }

    let lit_maps: [HashMap<u32, u64>; 8] = std::array::from_fn(|i| {
        lit_len_array_to_map(&lit_freqs[i])
    });

    (lit_maps, bucket_array_to_map(&bucket_freq))
}

pub fn build_encode_tables_v3(tokens: &[Token]) -> Option<([EncodeTable; 8], EncodeTable)> {
    let (lit_freqs, offset_freq) = count_joint_freq_v3(tokens);
    let has_lits = lit_freqs.iter().any(|f| !f.is_empty());
    if !has_lits { return None; }

    let lit_tables: [EncodeTable; 8] = std::array::from_fn(|i| {
        if lit_freqs[i].is_empty() {
            EncodeTable::new()
        } else {
            canonical_codes_from_lengths(&assign_code_lengths(&lit_freqs[i]))
        }
    });

    let offset_table = if offset_freq.is_empty() {
        EncodeTable::new()
    } else {
        canonical_codes_from_lengths(&assign_code_lengths(&offset_freq))
    };

    Some((lit_tables, offset_table))
}

// ── Table serialisation ───────────────────────────────────────────────────────

pub fn serialize_table(table: &EncodeTable) -> Vec<u8> {
    if table.is_empty() { return vec![0x00u8, 0x00, 0x00]; }

    let f0 = fmt0_explicit(table);
    let f1 = fmt1_range(table);

    let has_bytes   = table.keys().any(|&s| s <= 255);
    let has_lengths = table.keys().any(|&s| s >= 257);

    let mut best = if f1.len() < f0.len() { f1 } else { f0 };

    if has_bytes && has_lengths {
        let max_len_val = table.keys()
            .filter(|&&s| s >= 257)
            .map(|&s| s - 255)
            .max()
            .unwrap_or(0);
        if max_len_val <= 255 {
            let f2 = fmt2_two_range(table);
            if f2.len() < best.len() { best = f2; }
        }
    }

    best
}

fn fmt0_explicit(table: &EncodeTable) -> Vec<u8> {
    let mut entries: Vec<(u32, u8)> = table.iter()
        .map(|(&s, &(_, l))| (s, l as u8))
        .collect();
    entries.sort_by_key(|&(s, _)| s);
    let mut out = vec![0x00u8];
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (sym, len) in &entries {
        out.extend_from_slice(&(*sym as u16).to_le_bytes());
        out.push(*len);
    }
    out
}

fn fmt1_range(table: &EncodeTable) -> Vec<u8> {
    let min_sym = *table.keys().min().unwrap();
    let max_sym = *table.keys().max().unwrap();
    let mut out = vec![0x01u8];
    out.extend_from_slice(&(min_sym as u16).to_le_bytes());
    out.extend_from_slice(&(max_sym as u16).to_le_bytes());
    for sym in min_sym..=max_sym {
        out.push(table.get(&sym).map(|&(_, l)| l as u8).unwrap_or(0));
    }
    out
}

fn fmt2_two_range(table: &EncodeTable) -> Vec<u8> {
    let min_byte = table.keys().filter(|&&s| s <= 255).min().copied().unwrap();
    let max_byte = table.keys().filter(|&&s| s <= 255).max().copied().unwrap();
    let end_len  = table.get(&256).map(|&(_, l)| l as u8).unwrap_or(0);
    let min_len  = table.keys().filter(|&&s| s >= 257).map(|&s| s - 255).min().unwrap();
    let max_len  = table.keys().filter(|&&s| s >= 257).map(|&s| s - 255).max().unwrap();

    let mut out = vec![0x02u8];
    out.push(min_byte as u8);
    out.push(max_byte as u8);
    out.push(end_len);
    for b in min_byte..=max_byte {
        out.push(table.get(&b).map(|&(_, l)| l as u8).unwrap_or(0));
    }
    out.push(min_len as u8);
    out.push(max_len as u8);
    for l in min_len..=max_len {
        let sym = 255 + l;
        out.push(table.get(&sym).map(|&(_, l2)| l2 as u8).unwrap_or(0));
    }
    out
}

pub fn deserialize_table(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "table: empty"));
    }
    match data[0] {
        0x00 => deserialize_fmt0(data),
        0x01 => deserialize_fmt1(data),
        0x02 => deserialize_fmt2(data),
        b    => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown table format: 0x{:02x}", b))),
    }
}

fn deserialize_fmt0(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.len() < 3 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt0: too short"));
    }
    let n = u16::from_le_bytes([data[1], data[2]]) as usize;
    let needed = 3 + n * 3;
    if data.len() < needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt0: truncated"));
    }
    let mut lengths: HashMap<u32, u32> = HashMap::new();
    for i in 0..n {
        let base = 3 + i * 3;
        let sym  = u16::from_le_bytes([data[base], data[base + 1]]) as u32;
        let len  = data[base + 2] as u32;
        lengths.insert(sym, len);
    }
    Ok((canonical_codes_from_lengths(&lengths), needed))
}

fn deserialize_fmt1(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.len() < 5 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt1: too short"));
    }
    let min_sym = u16::from_le_bytes([data[1], data[2]]) as u32;
    let max_sym = u16::from_le_bytes([data[3], data[4]]) as u32;
    if max_sym < min_sym {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt1: max < min"));
    }
    let range  = (max_sym - min_sym + 1) as usize;
    let needed = 5 + range;
    if data.len() < needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt1: truncated"));
    }
    let mut lengths: HashMap<u32, u32> = HashMap::new();
    for i in 0..range {
        let l = data[5 + i] as u32;
        if l > 0 { lengths.insert(min_sym + i as u32, l); }
    }
    Ok((canonical_codes_from_lengths(&lengths), needed))
}

fn deserialize_fmt2(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt2: too short"));
    }
    let min_byte   = data[1] as u32;
    let max_byte   = data[2] as u32;
    let end_len    = data[3] as u32;
    if max_byte < min_byte {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt2: max_byte < min_byte"));
    }
    let byte_range = (max_byte - min_byte + 1) as usize;
    let len_hdr    = 4 + byte_range;
    if data.len() < len_hdr + 2 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt2: byte section truncated"));
    }
    let mut lengths: HashMap<u32, u32> = HashMap::new();
    for i in 0..byte_range {
        let l = data[4 + i] as u32;
        if l > 0 { lengths.insert(min_byte + i as u32, l); }
    }
    if end_len > 0 { lengths.insert(256, end_len); }
    let min_len   = data[len_hdr]     as u32;
    let max_len   = data[len_hdr + 1] as u32;
    if max_len < min_len {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt2: max_len < min_len"));
    }
    let len_range = (max_len - min_len + 1) as usize;
    let needed    = len_hdr + 2 + len_range;
    if data.len() < needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt2: len section truncated"));
    }
    for i in 0..len_range {
        let l = data[len_hdr + 2 + i] as u32;
        if l > 0 {
            let sym = 255 + (min_len + i as u32);
            lengths.insert(sym, l);
        }
    }
    Ok((canonical_codes_from_lengths(&lengths), needed))
}

// ── v1: joint lit/length Huffman + offset bucket Huffman ─────────────────────

pub fn write_tokens_v1(
    tokens:       &[Token],
    lit_table:    &EncodeTable,
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    let &(code, len) = lit_table.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v1 lit sym missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lit_table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v1 length sym {} missing", sym))
                    })?;
                    w.write(len, code)?;
                    let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                    let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v1 offset bucket {} missing", bucket))
                    })?;
                    w.write(blen, bcode)?;
                    if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                }
                Token::End => {
                    let &(code, len) = lit_table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v1 END sym missing")
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_v1(
    input:         &[u8],
    lit_dtable:    &DecodeTable,
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    let lit_max    = lit_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens = Vec::new();
    let mut r = BitReader::endian(std::io::Cursor::new(input), BigEndian);

    loop {
        let sym = match read_huffman_sym(&mut r, lit_dtable, lit_max) {
            Ok(s)  => s,
            Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            let offset = bucket_to_offset(bucket, extra_val);
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

// ── v2: 2-context lit/length Huffman + offset bucket Huffman ─────────────────

pub fn write_tokens_v2(
    tokens:       &[Token],
    lit_table0:   &EncodeTable,
    lit_table1:   &EncodeTable,
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
        let mut after_br = false;

        for token in tokens {
            let lt = if after_br { lit_table1 } else { lit_table0 };
            match token {
                Token::Lit { byte } => {
                    let &(code, len) = lt.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v2 lit sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                    after_br = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lt.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v2 length sym {} missing ctx={}", sym, after_br as u8))
                    })?;
                    w.write(len, code)?;
                    let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                    let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v2 offset bucket {} missing", bucket))
                    })?;
                    w.write(blen, bcode)?;
                    if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    after_br = true;
                }
                Token::End => {
                    let &(code, len) = lt.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v2 END sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_v2(
    input:         &[u8],
    lit_dtable0:   &DecodeTable,
    lit_dtable1:   &DecodeTable,
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    let lit_max0   = lit_dtable0.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let lit_max1   = lit_dtable1.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens = Vec::new();
    let mut r      = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut after_br = false;

    loop {
        let (dt, ml) = if after_br { (lit_dtable1, lit_max1) } else { (lit_dtable0, lit_max0) };
        let sym = match read_huffman_sym(&mut r, dt, ml) {
            Ok(s)  => s,
            Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
            after_br = false;
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            let offset = bucket_to_offset(bucket, extra_val);
            tokens.push(Token::Backref { offset, length });
            after_br = true;
        }
    }
    Ok(tokens)
}

// ── v3: 8-context lit/length Huffman + offset bucket Huffman ─────────────────

pub fn write_tokens_v3(
    tokens:       &[Token],
    tables:       &[EncodeTable],
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    assert!(tables.len() == 8, "v3 requires exactly 8 literal tables");
    let mut output = Vec::new();
    {
        let mut w         = BitWriter::endian(&mut output, BigEndian);
        let mut prev_byte: u8 = b' ';
        let mut after_br      = false;

        for token in tokens {
            let ctx   = context_idx(after_br, prev_byte);
            let table = &tables[ctx];

            match token {
                Token::Lit { byte } => {
                    let &(code, len) = table.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 lit byte {} missing in ctx {}", byte, ctx))
                    })?;
                    w.write(len, code)?;
                    prev_byte = *byte;
                    after_br  = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 length sym {} missing in ctx {}", sym, ctx))
                    })?;
                    w.write(len, code)?;

                    if !offset_table.is_empty() {
                        let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                        let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v3 offset bucket {} missing", bucket))
                        })?;
                        w.write(blen, bcode)?;
                        if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    }

                    after_br = true;
                }
                Token::End => {
                    let &(code, len) = table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 END sym missing in ctx {}", ctx))
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_v3(
    input:         &[u8],
    dtables:       &[DecodeTable],
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    assert!(dtables.len() == 8, "v3 requires exactly 8 decode tables");

    let max_lens: [u32; 8] = std::array::from_fn(|i| {
        dtables[i].keys().map(|&(_, l)| l).max().unwrap_or(1)
    });
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);

    let mut tokens    = Vec::new();
    let mut r         = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut prev_byte: u8 = b' ';
    let mut after_br      = false;

    loop {
        let ctx = context_idx(after_br, prev_byte);
        let sym = match read_huffman_sym(&mut r, &dtables[ctx], max_lens[ctx]) {
            Ok(s)  => s,
            Err(_) => break,
        };

        if sym < 256 {
            let byte = sym as u8;
            tokens.push(Token::Lit { byte });
            prev_byte = byte;
            after_br  = false;
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length = length_from_sym(sym);
            let offset = if offset_dtable.is_empty() {
                1
            } else {
                let bucket    = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
                let extra_cnt = bucket_extra_bits(bucket);
                let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
                bucket_to_offset(bucket, extra_val)
            };
            tokens.push(Token::Backref { offset, length });
            after_br = true;
        }
    }
    Ok(tokens)
}

// ── v4: joint lit/length Huffman + slotted offset ────────────────────────────

pub fn write_tokens_v4(
    tokens:       &[Token],
    lit_table:    &EncodeTable,
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w     = BitWriter::endian(&mut output, BigEndian);
        let mut slots = OffsetSlots::new();

        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    let &(code, len) = lit_table.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v4 lit sym missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lit_table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v4 length sym {} missing", sym))
                    })?;
                    w.write(len, code)?;

                    if let Some(slot_idx) = slots.access(*offset) {
                        let slot_sym = SLOT_SYMBOL_BASE + slot_idx as u32;
                        let &(scode, slen) = offset_table.get(&slot_sym).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v4 slot sym {} missing", slot_sym))
                        })?;
                        w.write(slen, scode)?;
                    } else {
                        let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                        let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v4 offset bucket {} missing", bucket))
                        })?;
                        w.write(blen, bcode)?;
                        if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    }
                }
                Token::End => {
                    let &(code, len) = lit_table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v4 END sym missing")
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_v4(
    input:         &[u8],
    lit_dtable:    &DecodeTable,
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    let lit_max    = lit_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens = Vec::new();
    let mut r      = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut slots  = OffsetSlots::new();

    loop {
        let sym = match read_huffman_sym(&mut r, lit_dtable, lit_max) {
            Ok(s)  => s,
            Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length  = length_from_sym(sym);
            let off_sym = read_huffman_sym(&mut r, offset_dtable, offset_max)?;

            let offset = if off_sym >= SLOT_SYMBOL_BASE {
                let slot_idx = (off_sym - SLOT_SYMBOL_BASE) as usize;
                slots.access_by_slot(slot_idx).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("v4 invalid slot index {}", slot_idx))
                })?
            } else {
                let extra_cnt = bucket_extra_bits(off_sym);
                let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
                let off = bucket_to_offset(off_sym, extra_val);
                slots.access(off);
                off
            };
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

// ── v5: 2-context lit/length Huffman + slotted offset ────────────────────────

pub fn write_tokens_v5(
    tokens:       &[Token],
    lit_table0:   &EncodeTable,
    lit_table1:   &EncodeTable,
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w        = BitWriter::endian(&mut output, BigEndian);
        let mut after_br = false;
        let mut slots    = OffsetSlots::new();

        for token in tokens {
            let lt = if after_br { lit_table1 } else { lit_table0 };
            match token {
                Token::Lit { byte } => {
                    let &(code, len) = lt.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v5 lit sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                    after_br = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lt.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v5 length sym {} missing ctx={}", sym, after_br as u8))
                    })?;
                    w.write(len, code)?;

                    if let Some(slot_idx) = slots.access(*offset) {
                        let slot_sym = SLOT_SYMBOL_BASE + slot_idx as u32;
                        let &(scode, slen) = offset_table.get(&slot_sym).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v5 slot sym {} missing", slot_sym))
                        })?;
                        w.write(slen, scode)?;
                    } else {
                        let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                        let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v5 offset bucket {} missing", bucket))
                        })?;
                        w.write(blen, bcode)?;
                        if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    }
                    after_br = true;
                }
                Token::End => {
                    let &(code, len) = lt.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v5 END sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_v5(
    input:         &[u8],
    lit_dtable0:   &DecodeTable,
    lit_dtable1:   &DecodeTable,
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    let lit_max0   = lit_dtable0.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let lit_max1   = lit_dtable1.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens   = Vec::new();
    let mut r        = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut after_br = false;
    let mut slots    = OffsetSlots::new();

    loop {
        let (dt, ml) = if after_br { (lit_dtable1, lit_max1) } else { (lit_dtable0, lit_max0) };
        let sym = match read_huffman_sym(&mut r, dt, ml) {
            Ok(s)  => s,
            Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
            after_br = false;
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length   = length_from_sym(sym);
            let off_sym  = read_huffman_sym(&mut r, offset_dtable, offset_max)?;

            let offset = if off_sym >= SLOT_SYMBOL_BASE {
                let slot_idx = (off_sym - SLOT_SYMBOL_BASE) as usize;
                slots.access_by_slot(slot_idx).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("v5 invalid slot index {}", slot_idx))
                })?
            } else {
                let extra_cnt = bucket_extra_bits(off_sym);
                let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
                let off = bucket_to_offset(off_sym, extra_val);
                slots.access(off);
                off
            };
            tokens.push(Token::Backref { offset, length });
            after_br = true;
        }
    }
    Ok(tokens)
}

// ── v6: separate literal stream + sequence stream Huffman ─────────────────────

pub fn build_v6_tables(tokens: &[Token]) -> Option<(EncodeTable, EncodeTable, EncodeTable)> {
    let mut lit_freq    = vec![0u64; 256];
    let mut seq_freq:   HashMap<u32, u64> = HashMap::new();
    let mut bucket_freq = [0u64; BUCKET_FREQ_SIZE];
    let mut has_lits    = false;

    for t in tokens {
        match t {
            Token::Lit { byte } => {
                lit_freq[*byte as usize] += 1;
                *seq_freq.entry(SYM_V6_LIT).or_insert(0) += 1;
                has_lits = true;
            }
            Token::Backref { offset, length } => {
                let sym = sym_from_length(*length);
                *seq_freq.entry(sym).or_insert(0) += 1;
                let (bucket, _, _) = offset_to_bucket(*offset);
                bucket_freq[bucket as usize] += 1;
            }
            Token::End => {
                *seq_freq.entry(SYM_END).or_insert(0) += 1;
            }
        }
    }

    if !has_lits { return None; }

    seq_freq.entry(SYM_END).or_insert(1);

    let lit_freq_map: HashMap<u32, u64> = lit_freq.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u32, c))
        .collect();

    let lit_table    = canonical_codes_from_lengths(&assign_code_lengths(&lit_freq_map));
    let seq_table    = canonical_codes_from_lengths(&assign_code_lengths(&seq_freq));
    let bucket_map   = bucket_array_to_map(&bucket_freq);
    let offset_table = if bucket_map.is_empty() {
        EncodeTable::new()
    } else {
        canonical_codes_from_lengths(&assign_code_lengths(&bucket_map))
    };

    Some((lit_table, seq_table, offset_table))
}

pub fn write_tokens_v6(
    tokens:       &[Token],
    lit_table:    &EncodeTable,
    seq_table:    &EncodeTable,
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    let mut lit_output: Vec<u8> = Vec::new();
    let mut lit_count: u32      = 0;
    {
        let mut w = BitWriter::endian(&mut lit_output, BigEndian);
        for t in tokens {
            if let Token::Lit { byte } = t {
                let &(code, len) = lit_table.get(&(*byte as u32)).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("v6: literal byte {} missing from lit_table", byte),
                    )
                })?;
                w.write(len, code)?;
                lit_count += 1;
            }
        }
        w.byte_align()?;
    }

    let mut seq_output: Vec<u8> = Vec::new();
    {
        let mut w = BitWriter::endian(&mut seq_output, BigEndian);
        for t in tokens {
            match t {
                Token::Lit { .. } => {
                    let &(code, len) = seq_table.get(&SYM_V6_LIT).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "v6: SYM_V6_LIT missing from seq_table",
                        )
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = seq_table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("v6: length sym {} missing from seq_table", sym),
                        )
                    })?;
                    w.write(len, code)?;
                    let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                    let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("v6: offset bucket {} missing", bucket),
                        )
                    })?;
                    w.write(blen, bcode)?;
                    if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                }
                Token::End => {
                    let &(code, len) = seq_table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "v6: SYM_END missing from seq_table",
                        )
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }

    let mut payload = serialize_table(lit_table);
    payload.extend_from_slice(&serialize_table(seq_table));
    payload.extend_from_slice(&serialize_table(offset_table));
    payload.extend_from_slice(&lit_count.to_le_bytes());
    payload.extend_from_slice(&(lit_output.len() as u32).to_le_bytes());
    payload.extend_from_slice(&lit_output);
    payload.extend_from_slice(&seq_output);
    Ok(payload)
}

pub fn read_tokens_v6(
    input:         &[u8],
    lit_dtable:    &DecodeTable,
    seq_dtable:    &DecodeTable,
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    if input.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "v6: payload too short for lit_count + lit_bitstream_len header",
        ));
    }

    let lit_count         = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let lit_bitstream_len = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;

    if input.len() < 8 + lit_bitstream_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "v6: lit_bitstream truncated (need {} bytes, have {})",
                lit_bitstream_len,
                input.len().saturating_sub(8)
            ),
        ));
    }

    let lit_bitstream = &input[8..8 + lit_bitstream_len];
    let seq_bitstream = &input[8 + lit_bitstream_len..];

    let lit_max    = lit_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let seq_max    = seq_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);

    let mut lit_bytes: Vec<u8> = Vec::with_capacity(lit_count);
    {
        let mut r = BitReader::endian(std::io::Cursor::new(lit_bitstream), BigEndian);
        for idx in 0..lit_count {
            let sym = read_huffman_sym(&mut r, lit_dtable, lit_max).map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!("v6: lit_bitstream error at literal {}/{}: {}", idx, lit_count, e),
                )
            })?;
            if sym > 255 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("v6: non-byte symbol {} in lit_bitstream at index {}", sym, idx),
                ));
            }
            lit_bytes.push(sym as u8);
        }
    }

    let mut tokens  = Vec::new();
    let mut lit_idx = 0usize;
    let mut r = BitReader::endian(std::io::Cursor::new(seq_bitstream), BigEndian);

    loop {
        let sym = match read_huffman_sym(&mut r, seq_dtable, seq_max) {
            Ok(s)  => s,
            Err(_) => break,
        };

        if sym == SYM_V6_LIT {
            if lit_idx >= lit_bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "v6: lit buffer exhausted at seq position {} (lit_count={})",
                        lit_idx, lit_count
                    ),
                ));
            }
            tokens.push(Token::Lit { byte: lit_bytes[lit_idx] });
            lit_idx += 1;
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            let offset = bucket_to_offset(bucket, extra_val);
            tokens.push(Token::Backref { offset, length });
        }
    }

    Ok(tokens)
}

// ── Shared Huffman reader ─────────────────────────────────────────────────────

fn read_huffman_sym<R: std::io::Read>(
    r:       &mut BitReader<R, BigEndian>,
    dtable:  &DecodeTable,
    max_len: u32,
) -> std::io::Result<u32> {
    let mut code: u32 = 0;
    for len in 1..=max_len {
        let bit = r.read::<u32>(1)?;
        code = (code << 1) | bit;
        if let Some(&sym) = dtable.get(&(code, len)) { return Ok(sym); }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("invalid huffman symbol after {} bits", max_len),
    ))
}
