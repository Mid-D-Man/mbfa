// src/filters/cfbf.rs
//! CFBF (Compound File Binary Format, a.k.a. OLE2/"Structured Storage" --
//! the container format under legacy .xls/.doc/.ppt) sector defragmentation
//! filter.
//!
//! A CFBF file is a small filesystem-in-a-file: fixed-size "sectors"
//! (usually 512 bytes) chained together via a FAT, holding named "storage"
//! (directories) and "stream" (file) entries. A single logical stream's
//! bytes are NOT guaranteed to be contiguous on disk -- if the file was
//! built up incrementally (which spreadsheets/documents typically are, one
//! save-as-you-go operation at a time), a stream's sectors can end up
//! scattered and interleaved with other streams' sectors. That defeats
//! LZ's local-window matching: two adjacent, highly-repetitive bytes 3
//! sectors apart inside the SAME logical stream might sit 50+ sectors
//! apart on disk once other streams' fragments are interleaved between
//! them.
//!
//! This filter parses the FAT + directory structure (real, documented
//! format -- Microsoft's own [MS-CFB] spec) well enough to know which
//! sectors belong to which stream and in what order, then rewrites the
//! sector *contents* so each stream's sectors become contiguous -- without
//! moving or rewriting the FAT/DIFAT/directory sectors themselves at all.
//! That's what makes this safely reversible with zero extra stored
//! metadata: those structural sectors stay byte-identical and
//! position-identical in the filtered output, so undo_filter can re-parse
//! them directly and rediscover the exact same stream-to-sector mapping
//! the encoder used, then invert it.
//!
//! Scope: only the "big" streams stored as regular FAT-chained sectors are
//! defragmented. Streams smaller than the mini-stream cutoff (typically
//! 4096 bytes) live packed into 64-byte "mini-sectors" inside a separate
//! mini-stream/mini-FAT structure; those are left untouched (parsed only
//! far enough to confirm the file is well-formed) -- the big streams (e.g.
//! a spreadsheet's actual "Workbook" data) are where the real payload and
//! thus the real compression opportunity is.
//!
//! Safety: this is a hand-written parser for a real, non-trivial binary
//! format, so `detect_filter` never trusts it blindly -- it runs a full
//! encode-then-decode round-trip against the actual input and only returns
//! this filter if the result is byte-for-byte identical to the original.
//! Any parse inconsistency, or any file whose streams are already
//! contiguous, degrades to a safe no-op rather than a wrong transform.

const CFBF_SIG: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

const FREESECT:   u32 = 0xFFFF_FFFF;
const ENDOFCHAIN: u32 = 0xFFFF_FFFE;
const FATSECT:    u32 = 0xFFFF_FFFD;
const DIFSECT:    u32 = 0xFFFF_FFFC;

fn ru16(d: &[u8], o: usize) -> u16 { u16::from_le_bytes([d[o], d[o + 1]]) }
fn ru32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

/// The reversible part of the transform: which original sector positions
/// (`slots`, ascending) get which original sectors' content (`values`, in
/// defragmented stream order) written into them. `slots` and `values` are
/// the same set, just differently ordered -- so applying the same pairing
/// in reverse (value's position <- slot's content) exactly undoes it.
struct CfbfLayout {
    sector_size: usize,
    num_sectors: usize,
    slots:  Vec<u32>,
    values: Vec<u32>,
}

/// Parse a CFBF file well enough to build the stream-defragmentation
/// permutation. Returns None on ANY structural inconsistency, truncation,
/// or unexpected value -- callers must treat that as "don't touch this
/// file", never as a best-effort partial result.
fn parse_cfbf(data: &[u8]) -> Option<CfbfLayout> {
    if data.len() < 512 || data[0..8] != CFBF_SIG { return None; }

    let version_major = ru16(data, 26);
    let sector_shift   = ru16(data, 30);
    if sector_shift != 9 && sector_shift != 12 { return None; } // 512 or 4096
    let sector_size = 1usize << sector_shift;
    if version_major == 4 && data.len() < 4096 { return None; }

    let size_fat          = ru32(data, 44) as usize;
    let ofs_dir            = ru32(data, 48);
    let mini_cutoff        = ru32(data, 56) as usize;
    let ofs_difat          = ru32(data, 68);
    let size_difat         = ru32(data, 72) as usize;

    if data.len() <= sector_size { return None; }
    let payload = data.len() - sector_size;
    if payload % sector_size != 0 { return None; }
    let num_sectors = payload / sector_size;
    if num_sectors == 0 || size_fat == 0 { return None; }

    let sector_at = |n: u32| -> Option<&[u8]> {
        let n = n as usize;
        if n >= num_sectors { return None; }
        let start = sector_size + n * sector_size;
        Some(&data[start..start + sector_size])
    };

    // ── DIFAT: 109 entries embedded in the header, then chained sectors ──
    let mut fat_locs: Vec<u32> = Vec::with_capacity(size_fat);
    for i in 0..109 {
        let v = ru32(data, 76 + i * 4);
        if v != FREESECT { fat_locs.push(v); }
    }
    if fat_locs.len() < size_fat {
        let entries_per = sector_size / 4 - 1;
        let mut cur = ofs_difat;
        let mut guard = 0usize;
        while fat_locs.len() < size_fat {
            if cur == ENDOFCHAIN || cur == FREESECT { break; }
            guard += 1;
            if guard > num_sectors + 1 { return None; } // cycle guard
            let s = sector_at(cur)?;
            for i in 0..entries_per {
                if fat_locs.len() >= size_fat { break; }
                let v = ru32(s, i * 4);
                if v != FREESECT { fat_locs.push(v); }
            }
            cur = ru32(s, entries_per * 4);
        }
    }
    if fat_locs.len() != size_fat { return None; }

    // ── Full FAT array ──
    let per_fat = sector_size / 4;
    let mut fat: Vec<u32> = vec![FREESECT; size_fat * per_fat];
    for (fi, &loc) in fat_locs.iter().enumerate() {
        let s = sector_at(loc)?;
        for i in 0..per_fat {
            fat[fi * per_fat + i] = ru32(s, i * 4);
        }
    }

    let chain_of = |start: u32, fat: &[u32]| -> Option<Vec<u32>> {
        if start == ENDOFCHAIN || start == FREESECT { return Some(Vec::new()); }
        let mut out = Vec::new();
        let mut cur = start;
        let mut guard = 0usize;
        while cur != ENDOFCHAIN {
            if cur == FREESECT || cur == FATSECT || cur == DIFSECT { return None; }
            let idx = cur as usize;
            if idx >= num_sectors || idx >= fat.len() { return None; }
            out.push(cur);
            guard += 1;
            if guard > num_sectors + 1 { return None; } // cycle guard
            cur = fat[idx];
        }
        Some(out)
    };

    // ── Directory sectors + entries (flat scan, tree structure not needed) ──
    let dir_chain = chain_of(ofs_dir, &fat)?;
    if dir_chain.is_empty() { return None; }
    let per_dir = sector_size / 128;
    let mut saw_root = false;
    let mut streams: Vec<(u32, u64)> = Vec::new(); // (start_sector, size)
    for &sec in &dir_chain {
        let s = sector_at(sec)?;
        for i in 0..per_dir {
            let base = i * 128;
            let obj_type   = s[base + 66];
            let start_sect = ru32(s, base + 116);
            let size_lo    = ru32(s, base + 120) as u64;
            let size_hi    = ru32(s, base + 124) as u64;
            let size       = size_lo | (size_hi << 32);
            match obj_type {
                5 => { saw_root = true; } // root storage — mini-stream container, out of scope
                2 if (size as usize) >= mini_cutoff => streams.push((start_sect, size)),
                _ => {} // unused slot, other storage, or a mini-stream-resident small stream
            }
        }
    }
    if !saw_root { return None; }

    // ── Collect each eligible stream's chain, in declaration order ──
    let mut values: Vec<u32> = Vec::new();
    let mut seen = vec![false; num_sectors];
    for (start, _size) in &streams {
        let chain = chain_of(*start, &fat)?;
        for &sec in &chain {
            let idx = sec as usize;
            if idx >= num_sectors || seen[idx] { return None; } // bounds / double-claim guard
            seen[idx] = true;
            values.push(sec);
        }
    }
    if values.is_empty() { return None; } // nothing to defragment

    let mut slots = values.clone();
    slots.sort_unstable();

    Some(CfbfLayout { sector_size, num_sectors, slots, values })
}

fn sector_bytes(data: &[u8], sector_size: usize, n: u32) -> &[u8] {
    let start = sector_size + (n as usize) * sector_size;
    &data[start..start + sector_size]
}

/// Reorder each stream's fragmented sectors to be contiguous. `layout` was
/// parsed from `input` itself.
fn apply_with_layout(input: &[u8], layout: &CfbfLayout) -> Vec<u8> {
    let mut out = input.to_vec();
    for (&slot, &value) in layout.slots.iter().zip(layout.values.iter()) {
        let src = sector_bytes(input, layout.sector_size, value).to_vec();
        let dst = layout.sector_size + (slot as usize) * layout.sector_size;
        out[dst..dst + layout.sector_size].copy_from_slice(&src);
    }
    out
}

/// Inverse of `apply_with_layout`, given a layout parsed from the FILTERED
/// bytes (valid since structural sectors never move, so re-parsing finds
/// the identical slots/values pairing the encoder used).
fn undo_with_layout(input: &[u8], layout: &CfbfLayout) -> Vec<u8> {
    let mut out = input.to_vec();
    for (&slot, &value) in layout.slots.iter().zip(layout.values.iter()) {
        let src = sector_bytes(input, layout.sector_size, slot).to_vec();
        let dst = layout.sector_size + (value as usize) * layout.sector_size;
        out[dst..dst + layout.sector_size].copy_from_slice(&src);
    }
    out
}

pub fn cfbf_defrag_encode(input: &[u8]) -> Vec<u8> {
    match parse_cfbf(input) {
        Some(layout) => apply_with_layout(input, &layout),
        None => input.to_vec(), // should not happen if detect_filter gated this; stay safe anyway
    }
}

pub fn cfbf_defrag_decode(input: &[u8]) -> Vec<u8> {
    match parse_cfbf(input) {
        Some(layout) => undo_with_layout(input, &layout),
        None => input.to_vec(),
    }
}

/// Returns `Some(FILTER_CFBF_DEFRAG)` only if `input` parses as a
/// structurally-consistent CFBF file AND a real encode/decode round-trip
/// against these exact bytes reproduces them exactly. See the module doc
/// for why the round-trip check isn't optional here.
pub fn detect_cfbf(input: &[u8]) -> Option<u8> {
    if input.len() < 512 || input[0..8] != CFBF_SIG { return None; }
    let layout = parse_cfbf(input)?;
    let encoded = apply_with_layout(input, &layout);
    // Decode must re-derive its OWN layout from the encoded bytes, exactly
    // as the real pipeline will at decompress time -- not reuse `layout`,
    // or this check would validate the wrong thing.
    let decoded = cfbf_defrag_decode(&encoded);
    if decoded != input { return None; }
    Some(crate::filters::FILTER_CFBF_DEFRAG)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal, real, valid CFBF v3 file (512-byte sectors) with
    /// one big stream deliberately fragmented into non-contiguous chunks
    /// interleaved with filler sectors from a second stream, so the
    /// defragmentation this filter exists for actually has something to do.
    fn build_test_cfbf() -> Vec<u8> {
        const SS: usize = 512;
        // Sector plan (indices are REGULAR sector numbers, i.e. file offset
        // = (n+1)*SS):
        //   0: FAT sector
        //   1: directory sector (root + two stream entries)
        //   2,4,6: stream A's data (fragmented, interleaved with B)
        //   3,5:   stream B's data (small fragments, interleaved with A)
        //   6..9:  stream A continues, contiguous tail
        // Content: stream A = repeating 'A' pattern varied per sector so a
        // roundtrip mismatch would be obvious; stream B = repeating 'B'.
        let a_chain = [2u32, 4, 6, 7, 8]; // fragmented then a contiguous tail
        let b_chain = [3u32, 5];
        let num_sectors = 9usize;

        let mut fat = vec![FREESECT; num_sectors];
        fat[0] = FATSECT;                 // sector 0 is itself the FAT
        for w in a_chain.windows(2) { fat[w[0] as usize] = w[1]; }
        fat[*a_chain.last().unwrap() as usize] = ENDOFCHAIN;
        for w in b_chain.windows(2) { fat[w[0] as usize] = w[1]; }
        fat[*b_chain.last().unwrap() as usize] = ENDOFCHAIN;
        fat[1] = ENDOFCHAIN; // directory is a single sector

        let mut data = vec![0u8; SS * (1 + num_sectors)]; // header + sectors
        data[0..8].copy_from_slice(&CFBF_SIG);
        data[26..28].copy_from_slice(&3u16.to_le_bytes());   // version_major
        data[30..32].copy_from_slice(&9u16.to_le_bytes());   // sector_shift (512)
        data[32..34].copy_from_slice(&6u16.to_le_bytes());   // mini_sector_shift
        data[44..48].copy_from_slice(&1u32.to_le_bytes());   // size_fat = 1
        data[48..52].copy_from_slice(&1u32.to_le_bytes());   // ofs_dir = sector 1
        data[56..60].copy_from_slice(&4096u32.to_le_bytes()); // mini cutoff
        data[60..64].copy_from_slice(&ENDOFCHAIN.to_le_bytes()); // no mini-FAT
        data[68..72].copy_from_slice(&ENDOFCHAIN.to_le_bytes()); // no chained DIFAT
        // DIFAT[0] = sector 0 (the FAT sector); rest FREESECT (already 0xFF-filled below)
        for i in 0..109 {
            let off = 76 + i * 4;
            data[off..off + 4].copy_from_slice(&FREESECT.to_le_bytes());
        }
        data[76..80].copy_from_slice(&0u32.to_le_bytes());

        let sec_off = |n: u32| SS + (n as usize) * SS;

        // FAT sector content
        {
            let off = sec_off(0);
            for i in 0..(SS / 4) {
                let v = fat.get(i).copied().unwrap_or(FREESECT);
                data[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }

        // Directory sector: root (type 5) + stream A (type 2, size = 5*SS)
        // + stream B (type 2, but size < cutoff — must stay < mini_cutoff
        // to be correctly EXCLUDED... except here we WANT it included to
        // prove multi-stream defrag, so size = 2*SS >= cutoff too? No —
        // mini_cutoff=4096=8 sectors, so a 2-sector (1024B) stream would
        // normally be mini-resident. For this test we lower expectations:
        // make B's size >= cutoff too by reporting it larger than it
        // physically needs (parser only trusts start_sect + chain
        // traversal here, not size, for chain-walking) so it's still
        // treated as a regular eligible stream.
        {
            let off = sec_off(1);
            // entry 0: root storage
            data[off + 66] = 5; // object_type = root storage
            data[off + 116..off + 120].copy_from_slice(&ENDOFCHAIN.to_le_bytes());
            data[off + 120..off + 128].copy_from_slice(&0u64.to_le_bytes());
            // entry 1: stream A. Declared size must be >= mini_cutoff or a
            // real CFBF reader would treat it as mini-resident regardless
            // of chain length -- parse_cfbf only uses `size` for that
            // classification, then trusts the FAT chain (terminated by
            // ENDOFCHAIN) for the actual sector list, so an inflated
            // declared size here is fine and matches how a real >=4096B
            // stream's last sector is typically only partially used too.
            let e1 = off + 128;
            data[e1 + 66] = 2;
            data[e1 + 116..e1 + 120].copy_from_slice(&a_chain[0].to_le_bytes());
            data[e1 + 120..e1 + 128].copy_from_slice(&(mini_cutoff_for_test() as u64).to_le_bytes());
            // entry 2: stream B
            let e2 = off + 256;
            data[e2 + 66] = 2;
            data[e2 + 116..e2 + 120].copy_from_slice(&b_chain[0].to_le_bytes());
            data[e2 + 120..e2 + 128].copy_from_slice(&(mini_cutoff_for_test() as u64).to_le_bytes());
        }

        // Data sectors: A = 0xAA + sector-position marker, B = 0xBB + marker
        for (pos, &sec) in a_chain.iter().enumerate() {
            let off = sec_off(sec);
            for b in data[off..off + SS].iter_mut() { *b = 0xAA; }
            data[off] = pos as u8; // makes each fragment distinguishable
        }
        for (pos, &sec) in b_chain.iter().enumerate() {
            let off = sec_off(sec);
            for b in data[off..off + SS].iter_mut() { *b = 0xBB; }
            data[off] = pos as u8;
        }

        data
    }

    fn mini_cutoff_for_test() -> usize { 4096 }

    #[test]
    fn detect_cfbf_fires_on_fragmented_streams() {
        let data = build_test_cfbf();
        assert!(detect_cfbf(&data).is_some(),
            "well-formed fragmented CFBF should be detected");
    }

    #[test]
    fn cfbf_roundtrips_exactly() {
        let data = build_test_cfbf();
        let encoded = cfbf_defrag_encode(&data);
        let decoded = cfbf_defrag_decode(&encoded);
        assert_eq!(decoded, data, "defrag encode/decode must be an exact inverse");
    }

    #[test]
    fn cfbf_actually_defragments() {
        let data = build_test_cfbf();
        let encoded = cfbf_defrag_encode(&data);
        // Stream A's 5 fragments (originally at sectors 2,4,6,7,8) should
        // now occupy 5 CONSECUTIVE sector slots among the data-sector
        // positions {2,3,4,5,6,7,8} (structural sectors 0,1 untouched).
        // Concretely: after defrag, walking sector positions 2..=8 in
        // order should show all-0xAA (marker 0,1,2,3,4) contiguously
        // before or after all-0xBB (marker 0,1) — i.e. no interleaving —
        // whereas the ORIGINAL interleaves A/B/A/B/A/A/A.
        let sec = |d: &[u8], n: usize| -> u8 { d[512 + n * 512] }; // first byte = marker
        let tag = |d: &[u8], n: usize| -> u8 { d[512 + n * 512 + 1] }; // 0xAA or 0xBB region marker (offset 1, still within fill)
        let _ = tag;
        let markers: Vec<u8> = (2..9).map(|n| sec(&encoded, n)).collect();
        // Original interleave order was A0,B0,A1,B1,A2,A3,A4 — assert the
        // encoded layout is no longer interleaved (a strictly-non-strict
        // check: all same-stream fragments now form one contiguous run).
        // Since both streams start their marker sequence at 0, detect
        // contiguity by checking the *content byte* (0xAA/0xBB) run
        // structure instead of marker values directly.
        let kinds: Vec<u8> = (2..9).map(|n| encoded[512 + n * 512 + 2]).collect(); // byte[2] is always 0xAA/0xBB (untouched by the marker byte at [0])
        let runs = kinds.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(runs <= 1,
            "expected at most one A/B boundary after defrag, got kinds={:?} markers={:?}",
            kinds, markers);
    }

    #[test]
    fn detect_cfbf_rejects_wrong_signature() {
        let mut data = build_test_cfbf();
        data[0] = 0x00;
        assert!(detect_cfbf(&data).is_none());
    }

    #[test]
    fn detect_cfbf_rejects_truncated_file() {
        let data = vec![0u8; 511];
        assert!(detect_cfbf(&data).is_none());
    }

    #[test]
    fn detect_cfbf_rejects_short_but_signed_input() {
        let mut data = vec![0u8; 512];
        data[0..8].copy_from_slice(&CFBF_SIG);
        // No valid FAT/dir behind the signature — must not crash, must return None.
        assert!(detect_cfbf(&data).is_none());
    }

    #[test]
    fn cfbf_noop_when_already_contiguous() {
        // Build the same file but with stream A's chain already contiguous
        // (2,3,4,5,6) and B's separately contiguous (7,8) — defrag should
        // be a pure identity transform (slots == values already sorted the
        // same way), still safe, just no-op.
        const SS: usize = 512;
        let a_chain = [2u32, 3, 4, 5, 6];
        let b_chain = [7u32, 8];
        let num_sectors = 9usize;
        let mut fat = vec![FREESECT; num_sectors];
        fat[0] = FATSECT;
        for w in a_chain.windows(2) { fat[w[0] as usize] = w[1]; }
        fat[*a_chain.last().unwrap() as usize] = ENDOFCHAIN;
        for w in b_chain.windows(2) { fat[w[0] as usize] = w[1]; }
        fat[*b_chain.last().unwrap() as usize] = ENDOFCHAIN;
        fat[1] = ENDOFCHAIN;

        let mut data = vec![0u8; SS * (1 + num_sectors)];
        data[0..8].copy_from_slice(&CFBF_SIG);
        data[26..28].copy_from_slice(&3u16.to_le_bytes());
        data[30..32].copy_from_slice(&9u16.to_le_bytes());
        data[32..34].copy_from_slice(&6u16.to_le_bytes());
        data[44..48].copy_from_slice(&1u32.to_le_bytes());
        data[48..52].copy_from_slice(&1u32.to_le_bytes());
        data[56..60].copy_from_slice(&4096u32.to_le_bytes());
        data[60..64].copy_from_slice(&ENDOFCHAIN.to_le_bytes());
        data[68..72].copy_from_slice(&ENDOFCHAIN.to_le_bytes());
        for i in 0..109 {
            let off = 76 + i * 4;
            data[off..off + 4].copy_from_slice(&FREESECT.to_le_bytes());
        }
        data[76..80].copy_from_slice(&0u32.to_le_bytes());
        let sec_off = |n: u32| SS + (n as usize) * SS;
        {
            let off = sec_off(0);
            for i in 0..(SS / 4) {
                let v = fat.get(i).copied().unwrap_or(FREESECT);
                data[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        {
            let off = sec_off(1);
            data[off + 66] = 5;
            data[off + 116..off + 120].copy_from_slice(&ENDOFCHAIN.to_le_bytes());
            let e1 = off + 128;
            data[e1 + 66] = 2;
            data[e1 + 116..e1 + 120].copy_from_slice(&a_chain[0].to_le_bytes());
            data[e1 + 120..e1 + 128].copy_from_slice(&4096u64.to_le_bytes()); // >= mini_cutoff, see build_test_cfbf's comment
            let e2 = off + 256;
            data[e2 + 66] = 2;
            data[e2 + 116..e2 + 120].copy_from_slice(&b_chain[0].to_le_bytes());
            data[e2 + 120..e2 + 128].copy_from_slice(&4096u64.to_le_bytes());
        }
        for &s in &a_chain { let o = sec_off(s); for b in data[o..o+SS].iter_mut() { *b = 0xAA; } }
        for &s in &b_chain { let o = sec_off(s); for b in data[o..o+SS].iter_mut() { *b = 0xBB; } }

        let encoded = cfbf_defrag_encode(&data);
        assert_eq!(encoded, data, "already-contiguous streams should encode to an identical file");
        let decoded = cfbf_defrag_decode(&encoded);
        assert_eq!(decoded, data);
    }
  }
