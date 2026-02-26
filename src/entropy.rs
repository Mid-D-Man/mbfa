// src/entropy.rs — pre-slot baseline + v4 byte-category literal contexts

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use bitstream_io::{BitWriter, BitReader, BigEndian, BitWrite, BitRead};
use crate::opcode::Token;

pub const ENTROPY_MIN_BYTES:    usize = 400;
pub const ENTROPY_V3_MIN_BYTES: usize = 1000;

const SYM_END: u32 = 256;
#[inline] fn sym_from_length(len: u32) -> u32 { 255 + len }
#[inline] fn length_from_sym(sym: u32)  -> u32 { sym - 255 }

pub type EncodeTable = HashMap<u32, (u32, u32)>;
pub type DecodeTable = HashMap<(u32, u32), u32>;

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

fn count_joint_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for t in tokens {
        match t {
            Token::Lit { byte }           => { *freq.entry(*byte as u32).or_insert(0) += 1; }
            Token::Backref { length, .. } => { *freq.entry(sym_from_length(*length)).or_insert(0) += 1; }
            Token::End                    => { *freq.entry(SYM_END).or_insert(0) += 1; }
        }
    }
    freq.entry(SYM_END).or_insert(1);
    freq
}

pub fn count_offset_bucket_freq(tokens: &[Token]) -> HashMap<u32, u64> {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for t in tokens {
        if let Token::Backref { offset, .. } = t {
            let (bucket, _, _) = offset_to_bucket(*offset);
            *freq.entry(bucket).or_insert(0) += 1;
        }
    }
    freq
}

fn count_joint_freq_by_context(tokens: &[Token]) -> (HashMap<u32, u64>, HashMap<u32, u64>) {
    let mut freq0: HashMap<u32, u64> = HashMap::new();
    let mut freq1: HashMap<u32, u64> = HashMap::new();
    let mut after_br = false;

    for t in tokens {
        let freq = if after_br { &mut freq1 } else { &mut freq0 };
        match t {
            Token::Lit { byte } => {
                *freq.entry(*byte as u32).or_insert(0) += 1;
                after_br = false;
            }
            Token::Backref { length, .. } => {
                *freq.entry(sym_from_length(*length)).or_insert(0) += 1;
                after_br = true;
            }
            Token::End => { *freq.entry(SYM_END).or_insert(0) += 1; }
        }
    }
    freq0.entry(SYM_END).or_insert(1);
    (freq0, freq1)
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

// ── v1 encode/decode ──────────────────────────────────────────────────────────

pub fn write_tokens_joint(
    tokens:      &[Token],
    table:       &EncodeTable,
    offset_bits: u32,
    length_bits: u32,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut w = BitWriter::endian(&mut output, BigEndian);
        for token in tokens {
            match token {
                Token::Lit { byte } => {
                    let &(code, len) = table.get(&(*byte as u32)).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v1 lit sym missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v1 length sym {} missing", sym))
                    })?;
                    w.write(len, code)?;
                    w.write(offset_bits, *offset)?;
                    let _ = length_bits;
                }
                Token::End => {
                    let &(code, len) = table.get(&SYM_END).ok_or_else(|| {
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

pub fn read_tokens_joint(
    input:        &[u8],
    dtable:       &DecodeTable,
    offset_bits:  u32,
    _length_bits: u32,
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
            tokens.push(Token::Lit { byte: sym as u8 });
        } else if sym == SYM_END {
            tokens.push(Token::End);
            break;
        } else {
            let length = length_from_sym(sym);
            let offset = r.read::<u32>(offset_bits)?;
            tokens.push(Token::Backref { offset, length });
        }
    }
    Ok(tokens)
}

// ── v2 encode/decode ──────────────────────────────────────────────────────────

pub fn write_tokens_joint_v2(
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
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v2 lit sym missing")
                    })?;
                    w.write(len, code)?;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lit_table.get(&sym).ok_or_else(|| {
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
                }
                Token::End => {
                    let &(code, len) = lit_table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "v2 END sym missing")
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_joint_v2(
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

// ── v3 encode/decode ──────────────────────────────────────────────────────────

pub fn write_tokens_joint_v3(
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
                            format!("v3 lit sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                    after_br = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = lt.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 length sym {} missing ctx={}", sym, after_br as u8))
                    })?;
                    w.write(len, code)?;
                    let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                    let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 offset bucket {} missing", bucket))
                    })?;
                    w.write(blen, bcode)?;
                    if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    after_br = true;
                }
                Token::End => {
                    let &(code, len) = lt.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v3 END sym missing ctx={}", after_br as u8))
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

pub fn read_tokens_joint_v3(
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

// ── v4 byte-category literal contexts ────────────────────────────────────────
//
// 8 literal/length Huffman tables indexed by:
//   ctx = (after_br as usize) * 4 + byte_category(prev_byte)
//
// byte_category:
//   0 = lowercase vowel  (a e i o u)
//   1 = alphabetic other (A-Z uppercase, b-z consonants)
//   2 = whitespace / common punctuation
//   3 = everything else
//
// prev_byte sentinel at stream start: b' ' (space → category 2)
// after_br initial: false
//
// Shared offset bucket table identical to v2/v3 — no raw bits.

/// Maps a byte to one of 4 categories used for v4 context selection.
#[inline]
pub fn byte_category(b: u8) -> usize {
    match b {
        // lowercase vowels — category 0
        b'a' | b'e' | b'i' | b'o' | b'u' => 0,
        // uppercase A-Z — category 1
        b'A'..=b'Z' => 1,
        // lowercase consonants (b-z minus vowels) — category 1
        // note: vowels already matched above, so this range only hits consonants
        b'b'..=b'z' => 1,
        // whitespace and common punctuation — category 2
        9 | 10 | 13 | 32        => 2,   // tab, LF, CR, space
        33 | 44 | 46 | 58 | 59  => 2,   // ! , . : ;
        39 | 40 | 41 | 63 | 45  => 2,   // ' ( ) ? -
        34 | 91 | 93 | 123 | 125 => 2,  // " [ ] { }
        // everything else — category 3
        _ => 3,
    }
}

/// Computes the 3-bit context index (0–7) for the v4 literal Huffman table.
///
/// `after_br` — true if the preceding token was a BACKREF.
/// `prev_byte` — the most recent LIT byte seen; b' ' at stream start.
#[inline]
pub fn context_idx(after_br: bool, prev_byte: u8) -> usize {
    (after_br as usize) * 4 + byte_category(prev_byte)
}

/// Count per-context literal/length frequencies and offset bucket frequencies
/// for v4. Returns ([lit_freq; 8], offset_freq).
fn count_joint_freq_v4(tokens: &[Token]) -> ([HashMap<u32, u64>; 8], HashMap<u32, u64>) {
    let mut lit_freqs: [HashMap<u32, u64>; 8] = std::array::from_fn(|_| HashMap::new());
    let mut offset_freq: HashMap<u32, u64> = HashMap::new();

    let mut prev_byte: u8 = b' ';
    let mut after_br      = false;

    for t in tokens {
        let ctx = context_idx(after_br, prev_byte);
        match t {
            Token::Lit { byte } => {
                *lit_freqs[ctx].entry(*byte as u32).or_insert(0) += 1;
                prev_byte = *byte;
                after_br  = false;
            }
            Token::Backref { offset, length } => {
                *lit_freqs[ctx].entry(sym_from_length(*length)).or_insert(0) += 1;
                let (bucket, _, _) = offset_to_bucket(*offset);
                *offset_freq.entry(bucket).or_insert(0) += 1;
                // prev_byte unchanged — stays as last literal seen
                after_br = true;
            }
            Token::End => {
                *lit_freqs[ctx].entry(SYM_END).or_insert(0) += 1;
            }
        }
    }
    (lit_freqs, offset_freq)
}

/// Build 8 literal/length encode tables + 1 shared offset bucket table for v4.
/// Returns None if there are no BACKREF tokens (offset table would be empty
/// and we can't guarantee all 8 lit tables are populated).
/// An empty lit table for an unused context is valid — that context never fires.
pub fn build_encode_tables_v4(tokens: &[Token]) -> Option<([EncodeTable; 8], EncodeTable)> {
    let (lit_freqs, offset_freq) = count_joint_freq_v4(tokens);

    // Need at least one lit token in some context
    let has_lits = lit_freqs.iter().any(|f| !f.is_empty());
    if !has_lits { return None; }

    let lit_tables: [EncodeTable; 8] = std::array::from_fn(|i| {
        if lit_freqs[i].is_empty() {
            EncodeTable::new()
        } else {
            canonical_codes_from_lengths(&assign_code_lengths(&lit_freqs[i]))
        }
    });

    // Offset table: if no backrefs, build a trivial empty table (v4 still works
    // as pure-literal — offset table serialises as 3 bytes of fmt0 with 0 entries).
    let offset_table = if offset_freq.is_empty() {
        EncodeTable::new()
    } else {
        canonical_codes_from_lengths(&assign_code_lengths(&offset_freq))
    };

    Some((lit_tables, offset_table))
}

/// Encode a token stream using 8 context-selected literal/length tables
/// and a shared offset bucket table. No raw offset bits — all offsets go
/// through bucket coding identical to v2/v3.
pub fn write_tokens_joint_v4(
    tokens:       &[Token],
    tables:       &[EncodeTable],   // length 8, indexed by context_idx(after_br, prev_byte)
    offset_table: &EncodeTable,
) -> std::io::Result<Vec<u8>> {
    assert!(tables.len() == 8, "v4 requires exactly 8 literal tables");
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
                            format!("v4 lit byte {} missing in ctx {}", byte, ctx))
                    })?;
                    w.write(len, code)?;
                    prev_byte = *byte;
                    after_br  = false;
                }
                Token::Backref { offset, length } => {
                    let sym = sym_from_length(*length);
                    let &(code, len) = table.get(&sym).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v4 length sym {} missing in ctx {}", sym, ctx))
                    })?;
                    w.write(len, code)?;

                    if !offset_table.is_empty() {
                        let (bucket, extra_cnt, extra_val) = offset_to_bucket(*offset);
                        let &(bcode, blen) = offset_table.get(&bucket).ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData,
                                format!("v4 offset bucket {} missing", bucket))
                        })?;
                        w.write(blen, bcode)?;
                        if extra_cnt > 0 { w.write(extra_cnt, extra_val)?; }
                    }

                    // prev_byte unchanged — stays as last LIT byte seen
                    after_br = true;
                }
                Token::End => {
                    let &(code, len) = table.get(&SYM_END).ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData,
                            format!("v4 END sym missing in ctx {}", ctx))
                    })?;
                    w.write(len, code)?;
                }
            }
        }
        w.byte_align()?;
    }
    Ok(output)
}

/// Decode a v4-encoded bitstream. The 8 decode tables and offset decode table
/// must come from deserialising the tables written by write_tokens_joint_v4.
/// State (prev_byte, after_br) is maintained identically to the encoder.
pub fn read_tokens_joint_v4(
    input:         &[u8],
    dtables:       &[DecodeTable],   // length 8
    offset_dtable: &DecodeTable,
) -> std::io::Result<Vec<Token>> {
    assert!(dtables.len() == 8, "v4 requires exactly 8 decode tables");

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
                1  // degenerate: no backrefs were encoded
            } else {
                let bucket    = read_huffman_sym(&mut r, offset_dtable, offset_max)?;
                let extra_cnt = bucket_extra_bits(bucket);
                let extra_val = if extra_cnt > 0 { r.read::<u32>(extra_cnt)? } else { 0 };
                bucket_to_offset(bucket, extra_val)
            };
            tokens.push(Token::Backref { offset, length });
            after_br = true;
            // prev_byte unchanged
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
