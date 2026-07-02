// src/entropy.rs
//
// Entropy coding variants -- all operate on the fold-1 token stream.
//
// P10 addition: v7 -- adaptive binary range coder.
//   Architecture mirrors xz RcDecoder (P10 analysis, src/decoder.rs).
//   Closes the Huffman integer-bit-floor gap on near-degenerate distributions.
//   No tables transmitted -- decoder rebuilds probability model from same
//   initial state (all probs = RC_PROB_INIT = 1024).
//
// P10 addition: fmt3 RLE serialization for Huffman tables.
//   Reduces table overhead for small files (Unreal_uplugin, YAML/TOML/INI).

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

pub const NUM_SLOTS:        usize = 4;
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

// ── Frequency counting ────────────────────────────────────────────────────────

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
            Token::RepRef { .. }          => unreachable!(
                "RepRef must be resolved via resolve_ring() before entropy encoding"
            ),
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
        if c > 0 { map.insert(SLOT_SYMBOL_BASE + i as u32, c); }
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
            Token::Lit { byte } => { freq[*byte as usize] += 1; after_br = false; }
            Token::Backref { length, .. } => {
                freq[(255 + length) as usize] += 1; after_br = true;
            }
            Token::End => { freq[SYM_END as usize] += 1; }
            Token::RepRef { .. } => unreachable!(
                "RepRef must be resolved via resolve_ring() before entropy encoding"
            ),
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

    let mut node_freq:   Vec<u64>            = Vec::with_capacity(2 * n);
    let mut left_child:  Vec<Option<usize>>  = Vec::with_capacity(2 * n);
    let mut right_child: Vec<Option<usize>>  = Vec::with_capacity(2 * n);
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

fn canonical_codes_from_lengths(lengths: &HashMap<u32, u32>) -> EncodeTable {
    let mut sorted: Vec<(u32, u32)> = lengths.iter().map(|(&s, &l)| (s, l)).collect();
    sorted.sort_by_key(|&(s, l)| (l, s));
    let mut table    = EncodeTable::new();
    let mut code     = 0u32;
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
        9 | 10 | 13 | 32         => 2,
        33 | 44 | 46 | 58 | 59   => 2,
        39 | 40 | 41 | 63 | 45   => 2,
        34 | 91 | 93 | 123 | 125 => 2,
        _ => 3,
    }
}

#[inline]
pub fn context_idx(after_br: bool, prev_byte: u8) -> usize {
    (after_br as usize) * 4 + byte_category(prev_byte)
}

fn count_joint_freq_v3(tokens: &[Token]) -> ([HashMap<u32, u64>; 8], HashMap<u32, u64>) {
    let mut lit_freqs: [Vec<u64>; 8] =
        std::array::from_fn(|_| vec![0u64; LIT_LEN_FREQ_SIZE]);
    let mut bucket_freq = [0u64; BUCKET_FREQ_SIZE];
    let mut prev_byte: u8 = b' ';
    let mut after_br      = false;

    for t in tokens {
        let ctx = context_idx(after_br, prev_byte);
        match t {
            Token::Lit { byte } => {
                lit_freqs[ctx][*byte as usize] += 1;
                prev_byte = *byte; after_br = false;
            }
            Token::Backref { offset, length } => {
                lit_freqs[ctx][(255 + length) as usize] += 1;
                let (bucket, _, _) = offset_to_bucket(*offset);
                bucket_freq[bucket as usize] += 1;
                after_br = true;
            }
            Token::End => { lit_freqs[ctx][SYM_END as usize] += 1; }
            Token::RepRef { .. } => unreachable!(
                "RepRef must be resolved via resolve_ring() before entropy encoding"
            ),
        }
    }

    let lit_maps: [HashMap<u32, u64>; 8] =
        std::array::from_fn(|i| lit_len_array_to_map(&lit_freqs[i]));
    (lit_maps, bucket_array_to_map(&bucket_freq))
}

pub fn build_encode_tables_v3(tokens: &[Token]) -> Option<([EncodeTable; 8], EncodeTable)> {
    let (lit_freqs, offset_freq) = count_joint_freq_v3(tokens);
    let has_lits = lit_freqs.iter().any(|f| !f.is_empty());
    if !has_lits { return None; }

    let lit_tables: [EncodeTable; 8] = std::array::from_fn(|i| {
        if lit_freqs[i].is_empty() { EncodeTable::new() }
        else { canonical_codes_from_lengths(&assign_code_lengths(&lit_freqs[i])) }
    });
    let offset_table = if offset_freq.is_empty() { EncodeTable::new() }
    else { canonical_codes_from_lengths(&assign_code_lengths(&offset_freq)) };

    Some((lit_tables, offset_table))
}

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
            Token::End => { *seq_freq.entry(SYM_END).or_insert(0) += 1; }
            Token::RepRef { .. } => unreachable!(
                "RepRef must be resolved via resolve_ring() before entropy encoding"
            ),
        }
    }

    if !has_lits { return None; }
    seq_freq.entry(SYM_END).or_insert(1);

    let lit_freq_map: HashMap<u32, u64> = lit_freq.iter().enumerate()
        .filter(|(_, &c)| c > 0).map(|(i, &c)| (i as u32, c)).collect();
    let lit_table    = canonical_codes_from_lengths(&assign_code_lengths(&lit_freq_map));
    let seq_table    = canonical_codes_from_lengths(&assign_code_lengths(&seq_freq));
    let bucket_map   = bucket_array_to_map(&bucket_freq);
    let offset_table = if bucket_map.is_empty() { EncodeTable::new() }
    else { canonical_codes_from_lengths(&assign_code_lengths(&bucket_map)) };

    Some((lit_table, seq_table, offset_table))
}

// ── Table serialisation ───────────────────────────────────────────────────────
//
// Format index:
//   0x00  fmt0  explicit (sym u16, len u8) list
//   0x01  fmt1  contiguous range, one len byte per symbol
//   0x02  fmt2  two-range split (bytes + lengths)
//   0x03  fmt3  RLE over contiguous range (P10)
//
// fmt3 RLE tokens within [min_sym..=max_sym]:
//   0x00-0xFD : literal code length (0 = symbol absent)
//   0xFE       : zero-run; next byte = count - 3  (encodes 3..258 zeros)
//   0xFF       : nonzero-repeat; next byte = value; byte after = count - 3
//
// Sentinels 0xFE/0xFF never conflict with real code lengths because
// MBFA's Huffman tree produces lengths <= 30 on any practical alphabet.

pub fn serialize_table(table: &EncodeTable) -> Vec<u8> {
    if table.is_empty() { return vec![0x00u8, 0x00, 0x00]; }

    let f0 = fmt0_explicit(table);
    let f1 = fmt1_range(table);
    let f3 = fmt3_rle(table);

    let has_bytes   = table.keys().any(|&s| s <= 255);
    let has_lengths = table.keys().any(|&s| s >= 257);

    let mut best = if f1.len() < f0.len() { f1 } else { f0 };
    if !f3.is_empty() && f3.len() < best.len() { best = f3; }

    if has_bytes && has_lengths {
        let max_len_val = table.keys()
            .filter(|&&s| s >= 257).map(|&s| s - 255).max().unwrap_or(0);
        if max_len_val <= 255 {
            let f2 = fmt2_two_range(table);
            if f2.len() < best.len() { best = f2; }
        }
    }
    best
}

fn fmt0_explicit(table: &EncodeTable) -> Vec<u8> {
    let mut entries: Vec<(u32, u8)> = table.iter()
        .map(|(&s, &(_, l))| (s, l as u8)).collect();
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
    out.push(min_byte as u8); out.push(max_byte as u8); out.push(end_len);
    for b in min_byte..=max_byte {
        out.push(table.get(&b).map(|&(_, l)| l as u8).unwrap_or(0));
    }
    out.push(min_len as u8); out.push(max_len as u8);
    for l in min_len..=max_len {
        out.push(table.get(&(255 + l)).map(|&(_, l2)| l2 as u8).unwrap_or(0));
    }
    out
}

fn fmt3_rle(table: &EncodeTable) -> Vec<u8> {
    // Guard: sentinels 0xFE/0xFF must not appear as valid code lengths
    if table.values().any(|&(_, l)| l >= 0xFE) { return Vec::new(); }

    let min_sym = *table.keys().min().unwrap();
    let max_sym = *table.keys().max().unwrap();
    let lengths: Vec<u8> = (min_sym..=max_sym)
        .map(|s| table.get(&s).map(|&(_, l)| l as u8).unwrap_or(0))
        .collect();

    let mut out = vec![0x03u8];
    out.extend_from_slice(&(min_sym as u16).to_le_bytes());
    out.extend_from_slice(&(max_sym as u16).to_le_bytes());

    let n = lengths.len();
    let mut i = 0;
    while i < n {
        let v = lengths[i];
        let mut run = 1;
        while i + run < n && lengths[i + run] == v && run < 258 { run += 1; }
        if run >= 3 {
            if v == 0 { out.push(0xFE); out.push((run - 3) as u8); }
            else      { out.push(0xFF); out.push(v); out.push((run - 3) as u8); }
            i += run;
        } else {
            for j in 0..run { out.push(lengths[i + j]); }
            i += run;
        }
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
        0x03 => deserialize_fmt3(data),
        b    => Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
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
    let min_byte = data[1] as u32; let max_byte = data[2] as u32;
    let end_len  = data[3] as u32;
    if max_byte < min_byte {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt2: max_byte < min_byte"));
    }
    let byte_range = (max_byte - min_byte + 1) as usize;
    let len_hdr    = 4 + byte_range;
    if data.len() < len_hdr + 2 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt2: truncated"));
    }
    let mut lengths: HashMap<u32, u32> = HashMap::new();
    for i in 0..byte_range {
        let l = data[4 + i] as u32;
        if l > 0 { lengths.insert(min_byte + i as u32, l); }
    }
    if end_len > 0 { lengths.insert(256, end_len); }
    let min_len = data[len_hdr] as u32; let max_len = data[len_hdr + 1] as u32;
    if max_len < min_len {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt2: max_len < min_len"));
    }
    let len_range = (max_len - min_len + 1) as usize;
    let needed    = len_hdr + 2 + len_range;
    if data.len() < needed {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt2: truncated"));
    }
    for i in 0..len_range {
        let l = data[len_hdr + 2 + i] as u32;
        if l > 0 { lengths.insert(255 + (min_len + i as u32), l); }
    }
    Ok((canonical_codes_from_lengths(&lengths), needed))
}

fn deserialize_fmt3(data: &[u8]) -> std::io::Result<(EncodeTable, usize)> {
    if data.len() < 5 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "fmt3: too short"));
    }
    let min_sym = u16::from_le_bytes([data[1], data[2]]) as u32;
    let max_sym = u16::from_le_bytes([data[3], data[4]]) as u32;
    if max_sym < min_sym {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fmt3: max < min"));
    }

    let range = (max_sym - min_sym + 1) as usize;
    let mut lengths = vec![0u8; range];
    let mut pos = 0usize;
    let mut cur = 5usize;

    while pos < range {
        if cur >= data.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                format!("fmt3: payload truncated at pos {}/{}", pos, range)));
        }
        let b = data[cur]; cur += 1;
        match b {
            0xFE => {
                if cur >= data.len() {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                        "fmt3: zero-run count missing"));
                }
                let count = data[cur] as usize + 3; cur += 1;
                if pos + count > range {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("fmt3: zero-run overflow (pos={} count={} range={})",
                            pos, count, range)));
                }
                pos += count;
            }
            0xFF => {
                if cur + 1 >= data.len() {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                        "fmt3: repeat header truncated"));
                }
                let v = data[cur]; let count = data[cur + 1] as usize + 3; cur += 2;
                if pos + count > range {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("fmt3: repeat overflow (pos={} count={} range={})",
                            pos, count, range)));
                }
                for _ in 0..count { lengths[pos] = v; pos += 1; }
            }
            v => { lengths[pos] = v; pos += 1; }
        }
    }

    let mut length_map: HashMap<u32, u32> = HashMap::new();
    for (i, &l) in lengths.iter().enumerate() {
        if l > 0 { length_map.insert(min_sym + i as u32, l as u32); }
    }
    Ok((canonical_codes_from_lengths(&length_map), cur))
}

// ── v1 ────────────────────────────────────────────────────────────────────────

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
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
            Ok(s) => s, Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            tokens.push(Token::Backref { offset: bucket_to_offset(bucket, extra_val), length });
        }
    }
    Ok(tokens)
}

// ── v2 ────────────────────────────────────────────────────────────────────────

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
                    w.write(len, code)?; after_br = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lt.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v2 length sym {} missing", sym))
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
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
    let mut tokens   = Vec::new();
    let mut r        = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut after_br = false;
    loop {
        let (dt, ml) = if after_br { (lit_dtable1, lit_max1) } else { (lit_dtable0, lit_max0) };
        let sym = match read_huffman_sym(&mut r, dt, ml) { Ok(s) => s, Err(_) => break };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 }); after_br = false;
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            tokens.push(Token::Backref { offset: bucket_to_offset(bucket, extra_val), length });
            after_br = true;
        }
    }
    Ok(tokens)
}

// ── v3 ────────────────────────────────────────────────────────────────────────

pub fn write_tokens_v3(
    tokens:       &[Token],
    tables:       &[EncodeTable],
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    assert!(tables.len() == 8);
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
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
                    w.write(len, code)?; prev_byte = *byte; after_br = false;
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
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
    assert!(dtables.len() == 8);
    let max_lens: [u32; 8] =
        std::array::from_fn(|i| dtables[i].keys().map(|&(_, l)| l).max().unwrap_or(1));
    let offset_max = offset_dtable.keys().map(|&(_, l)| l).max().unwrap_or(32);
    let mut tokens    = Vec::new();
    let mut r         = BitReader::endian(std::io::Cursor::new(input), BigEndian);
    let mut prev_byte: u8 = b' ';
    let mut after_br      = false;
    loop {
        let ctx = context_idx(after_br, prev_byte);
        let sym = match read_huffman_sym(&mut r, &dtables[ctx], max_lens[ctx]) {
            Ok(s) => s, Err(_) => break,
        };
        if sym < 256 {
            let byte = sym as u8;
            tokens.push(Token::Lit { byte }); prev_byte = byte; after_br = false;
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
        } else {
            let length = length_from_sym(sym);
            let offset = if offset_dtable.is_empty() { 1 } else {
                let bucket    = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
                let extra_cnt = bucket_extra_bits(bucket);
                let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
                bucket_to_offset(bucket, extra_val)
            };
            tokens.push(Token::Backref { offset, length }); after_br = true;
        }
    }
    Ok(tokens)
}

// ── v4 ────────────────────────────────────────────────────────────────────────

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
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
            Ok(s) => s, Err(_) => break,
        };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
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
                slots.access(off); off
            };
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

// ── v5 ────────────────────────────────────────────────────────────────────────

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
                    w.write(len, code)?; after_br = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lt.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v5 length sym {} missing", sym))
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
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
        let sym = match read_huffman_sym(&mut r, dt, ml) { Ok(s) => s, Err(_) => break };
        if sym < 256 {
            tokens.push(Token::Lit { byte: sym as u8 }); after_br = false;
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
        } else {
            let length  = length_from_sym(sym);
            let off_sym = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
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
                slots.access(off); off
            };
            tokens.push(Token::Backref { offset, length }); after_br = true;
        }
    }
    Ok(tokens)
}

// ── v6 ────────────────────────────────────────────────────────────────────────

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
                    std::io::Error::new(std::io::ErrorKind::InvalidData,
                        format!("v6: literal byte {} missing", byte))
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
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v6: SYM_V6_LIT missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = seq_table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v6: length sym {} missing", sym))
                    })?;
                    w.write(len, code)?;
                    let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                    let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v6: offset bucket {} missing", bucket))
                    })?;
                    w.write(blen, bcode)?;
                    if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                }
                Token::End => {
                    let &(code, len) = seq_table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v6: SYM_END missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::RepRef { .. } => unreachable!(
                    "RepRef must be resolved via resolve_ring() before entropy encoding"
                ),
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
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "v6: payload too short"));
    }
    let lit_count         = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let lit_bitstream_len = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    if input.len() < 8 + lit_bitstream_len {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "v6: lit_bitstream truncated"));
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
                std::io::Error::new(e.kind(),
                    format!("v6: lit error at {}/{}: {}", idx, lit_count, e))
            })?;
            if sym > 255 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("v6: non-byte symbol {} in lit_bitstream", sym)));
            }
            lit_bytes.push(sym as u8);
        }
    }

    let mut tokens  = Vec::new();
    let mut lit_idx = 0usize;
    let mut r = BitReader::endian(std::io::Cursor::new(seq_bitstream), BigEndian);
    loop {
        let sym = match read_huffman_sym(&mut r, seq_dtable, seq_max) {
            Ok(s) => s, Err(_) => break,
        };
        if sym == SYM_V6_LIT {
            if lit_idx >= lit_bytes.len() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    "v6: lit buffer exhausted"));
            }
            tokens.push(Token::Lit { byte: lit_bytes[lit_idx] });
            lit_idx += 1;
        } else if sym == SYM_END {
            tokens.push(Token::End); break;
        } else {
            let length = length_from_sym(sym);
            let bucket = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
            let extra_cnt = bucket_extra_bits(bucket);
            let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
            tokens.push(Token::Backref { offset: bucket_to_offset(bucket, extra_val), length });
        }
    }
    Ok(tokens)
}

// ── v7: Adaptive binary range coder ──────────────────────────────────────────
//
// LZMA-style range coder. Architecture:
//   range: u32  -- interval width, starts at 0xFFFF_FFFF
//   low:   u64  -- lower bound (64-bit so carry byte is accessible at bits 32-39)
//   cache: u8   -- last output byte buffered for carry propagation
//   cache_ff: u32 -- count of 0xFF bytes pending (may become 0x00 on carry)
//
// Carry analysis (LZMA invariant):
//   Before each flush_byte call, low < 2^33. After low += bound_u32:
//   low in [0, 2^33), so (low >> 32) in {0, 1}.
//   Output byte = bits 24-31 of low (cast to u32 first, then >> 24).
//   Carry = bits 32-39 of low (>> 32, & 0xFF), always 0 or 1.
//
// Encoder prefixes output with 0x00 (initial cache=0 flushes as first byte),
// so decoder primes by reading 5 bytes into a u32 code register.
//
// Context layout (2588 total u16 probability values):
//   [0]           is_match flag (1)
//   [1..2]        length choice (2) -- tier selection: lo / mid / hi
//   [3..9]        length LO bittree 3-bit (7)
//   [10..16]      length MID bittree 3-bit (7)
//   [17..271]     length HI bittree 8-bit (255)
//   [272..523]    dist_slot 4x63 bittrees 6-bit (252)
//   [524..547]    dist_extra direct bits (24, shared across all slots)
//   [548..2587]   literal 8x255 bittrees, keyed by (prev_byte >> 5) & 7 (2040)
//
// Length encoding:
//   lengths 2-9   -> LO  tier (val = length-2,  3-bit bittree)
//   lengths 10-17 -> MID tier (val = length-10, 3-bit bittree)
//   lengths 18-273-> HI  tier (val = length-18, 8-bit bittree)
//   End token     -> HI  tier value 255 (= length 273 = RC_LEN_SENTINEL)
//
// Distance encoding:
//   Reuses MBFA's offset_to_bucket/bucket_to_offset scheme directly.
//   Slot = bucket index (6-bit bittree, 4 independent contexts by length class).
//   Extra bits sent via shared direct-bit contexts (PROB_DIST_EXTRA).

pub const RC_MAX_BACKREF_LEN: u32 = 272; // HI tier value 255 reserved for End

const RC_PROB_INIT: u16 = 1024;
const RC_PROB_MIN:  u16 = 1;
const RC_PROB_MAX:  u16 = 2047;
const RC_PROB_BITS: u32 = 11;
const RC_PROB_SCALE: u32 = 1 << RC_PROB_BITS; // 2048
const RC_SHIFT:     u32 = 5;                   // 1/32 adaptation rate
const RC_NORM:      u32 = 1 << 24;             // normalise threshold

const RC_LEN_LO_BASE:    u32 = 2;
const RC_LEN_MID_BASE:   u32 = 10;
const RC_LEN_HI_BASE:    u32 = 18;
const RC_LEN_SENTINEL:   u32 = 273; // End marker: HI bittree value 255

const PROB_MATCH:      usize = 0;
const PROB_LEN_CHOICE: usize = 1;   // 2 probs
const PROB_LEN_LO:     usize = 3;   // 7 probs  (2^3 - 1)
const PROB_LEN_MID:    usize = 10;  // 7 probs
const PROB_LEN_HI:     usize = 17;  // 255 probs (2^8 - 1)
const PROB_DIST_SLOT:  usize = 272; // 4*63 = 252 probs
const PROB_DIST_EXTRA: usize = 524; // 24 probs (max bucket_extra_bits = 22, round up)
const PROB_LIT:        usize = 548; // 8*255 = 2040 probs
// Total = 548 + 2040 = 2588
const PROB_TOTAL:      usize = 2588;

fn rc_init() -> Vec<u16> { vec![RC_PROB_INIT; PROB_TOTAL] }

// ── Encoder ───────────────────────────────────────────────────────────────────

struct Rc7Enc {
    out:      Vec<u8>,
    low:      u64,
    range:    u32,
    cache:    u8,
    cache_ff: u32,
}

impl Rc7Enc {
    fn new() -> Self {
        Self { out: Vec::new(), low: 0, range: 0xFFFF_FFFF, cache: 0, cache_ff: 0 }
    }

    #[inline]
    fn encode_bit(&mut self, prob: &mut u16, sym: u32) {
        let p     = *prob as u64;
        let bound = ((self.range as u64) >> RC_PROB_BITS) * p;
        let bound = bound as u32;
        if sym == 0 {
            // LPS branch: range narrows to bound, prob adapts toward 0
            self.range = bound;
            *prob = (p as u32 + ((RC_PROB_SCALE - p as u32) >> RC_SHIFT)) as u16;
        } else {
            // MPS branch: low shifts up by bound, range shrinks
            self.low   += bound as u64;
            self.range -= bound;
            *prob = (p as u32 - (p as u32 >> RC_SHIFT)) as u16;
        }
        *prob = (*prob).clamp(RC_PROB_MIN, RC_PROB_MAX);
        if self.range < RC_NORM {
            self.flush_byte();
        }
    }

    fn flush_byte(&mut self) {
        // Extract output byte and carry from low.
        // Cast low to u32 first (discards carry bits 32+), then shift right 24.
        let out_byte = ((self.low as u32) >> 24) as u8;
        // Carry is bit 32 of low (0 or 1 per LZMA invariant)
        let carry    = ((self.low >> 32) & 1) as u8;

        if carry != 0 || out_byte != 0xFF {
            // Flush cache + carry propagation
            self.out.push(self.cache.wrapping_add(carry));
            // All pending 0xFF bytes either become 0x00 (carry=1) or stay 0xFF (carry=0)
            let fill = if carry != 0 { 0x00u8 } else { 0xFFu8 };
            for _ in 0..self.cache_ff { self.out.push(fill); }
            self.cache_ff = 0;
            self.cache    = out_byte;
        } else {
            // out_byte == 0xFF and no carry: defer output (carry might come later)
            self.cache_ff += 1;
        }

        // Advance: keep only the low 24 bits of low (the byte just flushed),
        // shift up by 8 to make room for the next byte
        self.low   = ((self.low & 0x00FF_FFFF) as u64) << 8;
        self.range <<= 8;
    }

    fn finish(mut self) -> Vec<u8> {
        // Flush 5 bytes to drain the low register
        for _ in 0..5 { self.flush_byte(); }
        self.out
    }

    fn encode_bittree(&mut self, probs: &mut [u16], base: usize, bits: u32, sym: u32) {
        let mut ctx = 1u32;
        for i in (0..bits).rev() {
            let bit = (sym >> i) & 1;
            self.encode_bit(&mut probs[base + ctx as usize - 1], bit);
            ctx = (ctx << 1) | bit;
        }
    }

    fn encode_direct(&mut self, probs: &mut [u16], base: usize, bits: u32, val: u32) {
        // Direct bits: sent MSB-first with per-position (not per-value) context
        for i in (0..bits).rev() {
            self.encode_bit(&mut probs[base + i as usize], (val >> i) & 1);
        }
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

struct Rc7Dec<'a> {
    input: &'a [u8],
    pos:   usize,
    code:  u32,
    range: u32,
}

impl<'a> Rc7Dec<'a> {
    fn new(input: &'a [u8]) -> std::io::Result<Self> {
        if input.len() < 5 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                "v7: stream too short to initialise range coder (need >= 5 bytes)"));
        }
        let mut dec = Rc7Dec { input, pos: 0, code: 0, range: 0xFFFF_FFFF };
        // Prime by reading 5 bytes. The first byte is the encoder's initial
        // 0x00 guard (from cache=0); it shifts out of the u32 naturally.
        for _ in 0..5 {
            dec.code = (dec.code << 8) | dec.next_byte()? as u32;
        }
        Ok(dec)
    }

    #[inline]
    fn next_byte(&mut self) -> std::io::Result<u8> {
        let b = if self.pos < self.input.len() { self.input[self.pos] } else { 0 };
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn decode_bit(&mut self, prob: &mut u16) -> std::io::Result<u32> {
        let p     = *prob as u32;
        let bound = (self.range >> RC_PROB_BITS) * p;
        let sym;
        if self.code < bound {
            self.range = bound;
            sym = 0u32;
            *prob = (p + ((RC_PROB_SCALE - p) >> RC_SHIFT)) as u16;
        } else {
            self.code  -= bound;
            self.range -= bound;
            sym = 1u32;
            *prob = (p - (p >> RC_SHIFT)) as u16;
        }
        *prob = (*prob).clamp(RC_PROB_MIN, RC_PROB_MAX);
        if self.range < RC_NORM {
            self.code  = (self.code << 8) | self.next_byte()? as u32;
            self.range <<= 8;
        }
        Ok(sym)
    }

    fn decode_bittree(&mut self, probs: &mut [u16], base: usize, bits: u32) -> std::io::Result<u32> {
        let mut ctx = 1u32;
        for _ in 0..bits {
            let bit = self.decode_bit(&mut probs[base + ctx as usize - 1])?;
            ctx = (ctx << 1) | bit;
        }
        Ok(ctx - (1 << bits))
    }

    fn decode_direct(&mut self, probs: &mut [u16], base: usize, bits: u32) -> std::io::Result<u32> {
        let mut val = 0u32;
        for i in (0..bits).rev() {
            let bit = self.decode_bit(&mut probs[base + i as usize])?;
            val |= bit << i;
        }
        Ok(val)
    }
}

// ── Length helpers ────────────────────────────────────────────────────────────

fn rc_encode_length(enc: &mut Rc7Enc, probs: &mut [u16], length: u32) {
    let v = length.saturating_sub(RC_LEN_LO_BASE); // 0-based
    if v < 8 {
        enc.encode_bit(&mut probs[PROB_LEN_CHOICE],     0);
        enc.encode_bittree(probs, PROB_LEN_LO, 3, v);
    } else {
        enc.encode_bit(&mut probs[PROB_LEN_CHOICE],     1);
        let v2 = v - 8;
        if v2 < 8 {
            enc.encode_bit(&mut probs[PROB_LEN_CHOICE + 1], 0);
            enc.encode_bittree(probs, PROB_LEN_MID, 3, v2);
        } else {
            enc.encode_bit(&mut probs[PROB_LEN_CHOICE + 1], 1);
            // HI tier: value 0-254 = lengths 18-272; value 255 reserved for End
            let v3 = (v2 - 8).min(254);
            enc.encode_bittree(probs, PROB_LEN_HI, 8, v3);
        }
    }
}

fn rc_decode_length(dec: &mut Rc7Dec, probs: &mut [u16]) -> std::io::Result<u32> {
    if dec.decode_bit(&mut probs[PROB_LEN_CHOICE])? == 0 {
        Ok(dec.decode_bittree(probs, PROB_LEN_LO, 3)? + RC_LEN_LO_BASE)
    } else if dec.decode_bit(&mut probs[PROB_LEN_CHOICE + 1])? == 0 {
        Ok(dec.decode_bittree(probs, PROB_LEN_MID, 3)? + RC_LEN_MID_BASE)
    } else {
        // HI tier: 0-254 -> lengths 18-272; 255 -> End sentinel (273)
        Ok(dec.decode_bittree(probs, PROB_LEN_HI, 8)? + RC_LEN_HI_BASE)
    }
}

#[inline]
fn length_class(length: u32) -> usize {
    match length { 0..=1 => 0, 2 => 1, 3 => 2, _ => 3 }
}

// ── Distance helpers ──────────────────────────────────────────────────────────

fn rc_encode_distance(enc: &mut Rc7Enc, probs: &mut [u16], offset: u32, length: u32) {
    let (slot, extra_bits, extra_val) = offset_to_bucket(offset);
    let slot_c = slot.min(63); // 6-bit bittree: 64 slots max
    let lc     = length_class(length);
    enc.encode_bittree(probs, PROB_DIST_SLOT + lc * 63, 6, slot_c);
    if extra_bits > 0 {
        // Cap at 24 (PROB_DIST_EXTRA has 24 slots)
        let eb = extra_bits.min(24);
        enc.encode_direct(probs, PROB_DIST_EXTRA, eb, extra_val);
    }
}

fn rc_decode_distance(dec: &mut Rc7Dec, probs: &mut [u16], length: u32)
    -> std::io::Result<u32>
{
    let lc     = length_class(length);
    let slot   = dec.decode_bittree(probs, PROB_DIST_SLOT + lc * 63, 6)?;
    let extra_bits = bucket_extra_bits(slot).min(24);
    let extra_val = if extra_bits > 0 {
        dec.decode_direct(probs, PROB_DIST_EXTRA, extra_bits)?
    } else { 0 };
    Ok(bucket_to_offset(slot, extra_val))
}

// ── Public encode / decode ────────────────────────────────────────────────────

/// Encode token stream with adaptive binary range coding.
///
/// Caller must ensure all Token::Backref lengths <= RC_MAX_BACKREF_LEN (272).
/// Token::RepRef must have been resolved via resolve_ring() before calling.
pub fn write_tokens_v7(tokens: &[Token]) -> std::io::Result<Vec<u8>> {
    let mut probs = rc_init();
    let mut enc   = Rc7Enc::new();
    let mut prev:  u8 = 0;

    for token in tokens {
        match token {
            Token::Lit { byte } => {
                enc.encode_bit(&mut probs[PROB_MATCH], 0);
                let ctx = PROB_LIT + (((prev as usize) >> 5) & 7) * 255;
                enc.encode_bittree(&mut probs, ctx, 8, *byte as u32);
                prev = *byte;
            }
            Token::Backref { offset, length } => {
                enc.encode_bit(&mut probs[PROB_MATCH], 1);
                rc_encode_length(&mut enc, &mut probs, *length);
                rc_encode_distance(&mut enc, &mut probs, *offset, *length);
            }
            Token::End => {
                // End: is_match=1, then HI tier value 255 (sentinel)
                enc.encode_bit(&mut probs[PROB_MATCH], 1);
                enc.encode_bit(&mut probs[PROB_LEN_CHOICE],     1);
                enc.encode_bit(&mut probs[PROB_LEN_CHOICE + 1], 1);
                enc.encode_bittree(&mut probs, PROB_LEN_HI, 8, 255);
            }
            Token::RepRef { .. } => unreachable!(
                "RepRef must be resolved via resolve_ring() before entropy encoding"
            ),
        }
    }

    Ok(enc.finish())
}

/// Decode a v7 range-coded token stream.
/// Reconstructs the same probability model as write_tokens_v7 from scratch.
pub fn read_tokens_v7(input: &[u8]) -> std::io::Result<Vec<Token>> {
    let mut probs  = rc_init();
    let mut dec    = Rc7Dec::new(input)?;
    let mut tokens = Vec::new();
    let mut prev:  u8 = 0;

    loop {
        let is_match = dec.decode_bit(&mut probs[PROB_MATCH])?;
        if is_match == 0 {
            let ctx  = PROB_LIT + (((prev as usize) >> 5) & 7) * 255;
            let byte = dec.decode_bittree(&mut probs, ctx, 8)? as u8;
            tokens.push(Token::Lit { byte });
            prev = byte;
        } else {
            let length = rc_decode_length(&mut dec, &mut probs)?;
            // RC_LEN_SENTINEL = RC_LEN_HI_BASE + 255 = 18 + 255 = 273
            if length >= RC_LEN_SENTINEL {
                tokens.push(Token::End);
                break;
            }
            let offset = rc_decode_distance(&mut dec, &mut probs, length)?;
            if offset == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    "v7: decoded offset=0 (corrupt stream)"));
            }
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

// ── v7 tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod v7_tests {
    use super::*;

    fn rt(tokens: &[Token]) -> Vec<Token> {
        let enc = write_tokens_v7(tokens).expect("v7 encode failed");
        read_tokens_v7(&enc).expect("v7 decode failed")
    }

    #[test]
    fn v7_roundtrip_end_only() {
        assert_eq!(rt(&[Token::End]), vec![Token::End]);
    }

    #[test]
    fn v7_roundtrip_lits_only() {
        let mut t: Vec<Token> = b"hello world"
            .iter().map(|&b| Token::Lit { byte: b }).collect();
        t.push(Token::End);
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_roundtrip_all_byte_values() {
        let mut t: Vec<Token> = (0u8..=255).map(|b| Token::Lit { byte: b }).collect();
        t.push(Token::End);
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_roundtrip_single_backref() {
        let t = vec![
            Token::Lit { byte: b'a' }, Token::Lit { byte: b'b' },
            Token::Backref { offset: 2, length: 4 }, Token::End,
        ];
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_roundtrip_large_offsets() {
        let t = vec![
            Token::Backref { offset: 1,       length: 2 },
            Token::Backref { offset: 4,       length: 2 },
            Token::Backref { offset: 5,       length: 2 },
            Token::Backref { offset: 100,     length: 2 },
            Token::Backref { offset: 10000,   length: 2 },
            Token::Backref { offset: 1000000, length: 2 },
            Token::End,
        ];
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_roundtrip_length_tiers() {
        let t = vec![
            Token::Backref { offset: 1, length: 2   }, // LO start
            Token::Backref { offset: 1, length: 9   }, // LO end
            Token::Backref { offset: 1, length: 10  }, // MID start
            Token::Backref { offset: 1, length: 17  }, // MID end
            Token::Backref { offset: 1, length: 18  }, // HI start
            Token::Backref { offset: 1, length: 200 }, // HI mid
            Token::Backref { offset: 1, length: 272 }, // HI max = RC_MAX_BACKREF_LEN
            Token::End,
        ];
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_roundtrip_mixed() {
        let t = vec![
            Token::Lit    { byte: b'T' },
            Token::Lit    { byte: b'e' },
            Token::Backref { offset: 2,     length: 3   },
            Token::Lit    { byte: b' '  },
            Token::Backref { offset: 100,   length: 10  },
            Token::Backref { offset: 65535, length: 255 },
            Token::End,
        ];
        assert_eq!(rt(&t), t);
    }

    #[test]
    fn v7_beats_raw_bits_on_repetitive_stream() {
        // 500 identical Backrefs -- RC adapts after a few and spends ~0 bits each
        let t: Vec<Token> = std::iter::repeat(Token::Backref { offset: 4096, length: 200 })
            .take(500)
            .chain(std::iter::once(Token::End))
            .collect();
        let v7_size = write_tokens_v7(&t).unwrap().len();
        // Raw fixed-width: 500 * (1+17+8)/8 = 500*26/8 = 1625 bytes (ob=17, lb=8)
        assert!(v7_size < 1625,
            "v7 ({} B) should beat raw bitstream (1625 B) on repetitive stream", v7_size);
        println!("v7 repetitive 500x: {} bytes  (raw: 1625 bytes, {:.1}% of raw)",
            v7_size, v7_size as f64 / 1625.0 * 100.0);
    }
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
