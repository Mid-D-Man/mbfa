//! Precomputed per-position match table ("match oracle") for the hash-chain
//! scan, and the policy that decides when to use one.
//!
//! # Why this exists
//!
//! `encoder::scan_from` spends most of its time in `find_match`, a hash-chain
//! walk. That walk does not depend on the parse: a position is inserted into
//! the chains exactly once, in order, whatever tokens are emitted, so the
//! candidates for a query at `q` are always "the earlier positions with the
//! same hash". Only *which* positions get queried, the lazy look-ahead, and
//! the rep-slot ring depend on the parse. That makes the expensive part
//! data-parallel: compute `(offset, length)` for every position at once
//! (on a GPU, see `gpu.rs`), then let the sequential scan read the table.
//!
//! This module is the exact, CPU-only definition of that table and of how the
//! scan consumes it. It is always compiled; the GPU kernel only has to
//! reproduce `compute_table_cpu` bit for bit, and the tests prove that
//! scanning through a table gives exactly the tokens the plain scan gives.
//!
//! # Exactness details (all covered by tests)
//!
//! * `PrevIndex::prev[p]` is the newest earlier position with the same hash,
//!   i.e. what `head[hash]` held when `p` was inserted. Positions with
//!   `p + 2 >= n` are never inserted by the match path and are not indexed.
//! * Look-ahead queries (`i + 1`, `i + 2`) happen before `i` is inserted, so
//!   they must not see the 1 or 2 newest positions. `Oracle::find` skips
//!   those chain nodes without counting them as steps, exactly as the real
//!   chain does. When nothing is hidden the table entry is used as is.
//! * The last two positions hash to bucket 0 (`hash3` returns 0 there) and
//!   share that chain with real bucket-0 positions; whether they were
//!   inserted depends on the parse, so those queries are answered by a small
//!   exact walk (`find_tail`) using the insertion flags the scan reports.
//! * A table entry compares at most `COMPARE_CAP` bytes so that a GPU thread
//!   does bounded work. If a candidate reaches the cap with room to go, the
//!   entry is marked `OVERFLOW` and the query is answered by an uncapped CPU
//!   walk instead.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::encoder::{hash3, match_len, HASH_SIZE};

pub(crate) const NONE: u32 = u32::MAX;

/// Longest comparison a single table entry attempts (bounded GPU work).
pub const COMPARE_CAP: usize = 2048;

/// Table entries in a table: `[offset, length]` per query position.
/// `length == OVERFLOW` means "recompute this one on the CPU".
pub(crate) const OVERFLOW: u32 = u32::MAX;

/// Inputs smaller than this are never worth the transfer (policy, not exactness).
pub const ACCEL_MIN_BYTES: usize = 1 << 20;
/// Inputs larger than this keep the CPU path (host memory: index 4 B/byte, table 8 B/byte).
pub const ACCEL_MAX_BYTES: usize = 48 << 20;

// ── Policy ──────────────────────────────────────────────────────────────────

/// How `compress` may use accelerated match finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accel {
    /// Never: plain CPU hash chains. Always available, and the default when
    /// the crate is built without the `gpu` feature.
    Off,
    /// Use the GPU when the `gpu` feature is built in, a suitable adapter is
    /// found, and the input is large enough to be worth it. Otherwise CPU.
    Auto,
    /// Like `Auto` but without the size minimum (testing and benchmarking).
    Force,
    /// Build the table on the CPU. Exists so tests can prove the table path
    /// equals the plain scan without a GPU. Not useful for speed.
    #[doc(hidden)]
    CpuTable,
}

static ACCEL: AtomicU8 = AtomicU8::new(1); // Auto

/// Set the process-wide acceleration policy.
pub fn set_accel(a: Accel) {
    ACCEL.store(a as u8, Ordering::Relaxed);
}

/// The current acceleration policy.
pub fn accel() -> Accel {
    match ACCEL.load(Ordering::Relaxed) {
        0 => Accel::Off,
        1 => Accel::Auto,
        2 => Accel::Force,
        _ => Accel::CpuTable,
    }
}

// ── Index: newest earlier position with the same hash, for every position ────

/// Parse-independent chain links for one input buffer.
pub struct PrevIndex {
    pub(crate) prev: Vec<u32>,
    /// Indexed positions whose hash is 0, ascending (the bucket the last two
    /// positions of every input also land in).
    pub(crate) zero: Vec<u32>,
    /// The bytes this index was built from, so a cache hit is verified by
    /// content and never trusted by address.
    pub(crate) data: Vec<u8>,
}

impl PrevIndex {
    pub fn build(input: &[u8]) -> PrevIndex {
        let n = input.len();
        let mut head = vec![NONE; HASH_SIZE];
        let mut prev = vec![NONE; n];
        let mut zero = Vec::new();
        for p in 0..n.saturating_sub(2) {
            let h = hash3(input, p);
            prev[p] = head[h];
            head[h] = p as u32;
            if h == 0 { zero.push(p as u32); }
        }
        PrevIndex { prev, zero, data: input.to_vec() }
    }
}

static INDEX_CACHE: Mutex<Option<Arc<PrevIndex>>> = Mutex::new(None);

/// The index for `input`, reusing the last one built if the bytes are identical
/// (one compress call scans the same buffer several times with different
/// window and length settings).
pub fn index_for(input: &[u8]) -> Arc<PrevIndex> {
    let mut slot = INDEX_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ix) = slot.as_ref() {
        if ix.data.len() == input.len() && ix.data[..] == *input {
            return Arc::clone(ix);
        }
    }
    let ix = Arc::new(PrevIndex::build(input));
    *slot = Some(Arc::clone(&ix));
    ix
}

// ── Table ────────────────────────────────────────────────────────────────────

/// Scan parameters a table is valid for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub start_at: usize,
    pub max_off: usize,
    pub max_len: usize,
    pub chain_limit: usize,
}

/// `[offset, length]` for each query position `start_at .. n-2`.
pub struct Table {
    pub start: usize,
    pub entries: Vec<u32>,
}

impl Table {
    #[inline]
    fn get(&self, q: usize) -> (u32, u32) {
        let k = (q - self.start) * 2;
        (self.entries[k], self.entries[k + 1])
    }
}

/// The chain walk of `encoder::find_match`, over an explicit chain.
///
/// `cap`: compare at most this many bytes per candidate; a candidate that
/// reaches the cap with room to go makes the result `None` (overflow).
#[inline]
fn walk(
    input: &[u8], q: usize, mut cur: u32, prev: &[u32], p: &Params, cap: Option<usize>,
) -> Option<(usize, usize)> {
    let limit = p.max_len.min(input.len().saturating_sub(q));
    let (mut best_off, mut best_len, mut steps) = (0usize, 0usize, 0usize);
    while cur != NONE && steps < p.chain_limit {
        let j = cur as usize;
        if q <= j || q - j > p.max_off { break; }
        if best_len >= limit { break; }
        let span = q - j;
        let probe = if best_len < span { best_len } else { best_len % span };
        if input[j + probe] == input[q + best_len] {
            let cmp_limit = match cap { Some(c) => limit.min(c), None => limit };
            let len = match_len(input, q, span, cmp_limit);
            if let Some(c) = cap {
                if len == c && limit > c { return None; }
            }
            if len > best_len {
                best_len = len;
                best_off = span;
                if best_len == p.max_len { break; }
            }
        }
        cur = prev[j];
        steps += 1;
    }
    Some((best_off, best_len))
}

/// Same walk over an explicit newest-first candidate list (used for the tail).
fn walk_list(input: &[u8], q: usize, cands: &[u32], p: &Params) -> (usize, usize) {
    let limit = p.max_len.min(input.len().saturating_sub(q));
    let (mut best_off, mut best_len, mut steps) = (0usize, 0usize, 0usize);
    for &c in cands {
        if steps >= p.chain_limit { break; }
        let j = c as usize;
        if q <= j || q - j > p.max_off { break; }
        if best_len >= limit { break; }
        let span = q - j;
        let probe = if best_len < span { best_len } else { best_len % span };
        if input[j + probe] == input[q + best_len] {
            let len = match_len(input, q, span, limit);
            if len > best_len {
                best_len = len;
                best_off = span;
                if best_len == p.max_len { break; }
            }
        }
        steps += 1;
    }
    (best_off, best_len)
}

/// Reference implementation of the table: what any accelerated backend must
/// reproduce exactly. One entry per query position `start_at .. n-2`.
pub fn compute_table_cpu(input: &[u8], idx: &PrevIndex, p: &Params) -> Table {
    let n = input.len();
    let end = n.saturating_sub(2);
    let start = p.start_at.min(end);
    let mut entries = Vec::with_capacity((end - start) * 2);
    for q in start..end {
        match walk(input, q, idx.prev[q], &idx.prev, p, Some(COMPARE_CAP)) {
            Some((off, len)) => { entries.push(off as u32); entries.push(len as u32); }
            None             => { entries.push(0);          entries.push(OVERFLOW); }
        }
    }
    Table { start, entries }
}

// ── Oracle: what scan_from reads ─────────────────────────────────────────────

pub struct Oracle<'a> {
    input: &'a [u8],
    idx: Arc<PrevIndex>,
    table: Table,
    p: Params,
    n: usize,
    /// Was position `n-2` / `n-1` inserted into the bucket-0 chain? (The scan
    /// inserts the tail only on its literal paths, so this depends on the parse.)
    tail_inserted: [bool; 2],
}

impl<'a> Oracle<'a> {
    /// The scan inserted position `pos` into the chains on a literal path.
    #[inline]
    pub(crate) fn note_insert(&mut self, pos: usize) {
        if pos + 2 >= self.n {
            let k = pos + 2 - self.n;          // n-2 -> 0, n-1 -> 1
            if k < 2 { self.tail_inserted[k] = true; }
        }
    }

    /// `find_match(input, q, ..)` as the scan would see it when `hidden`
    /// newest positions (0 for the main query, 1 and 2 for the look-aheads)
    /// have not been inserted yet.
    #[inline]
    pub(crate) fn find(&self, q: usize, hidden: usize) -> (usize, usize) {
        if q + 2 >= self.n {
            return self.find_tail(q, hidden);
        }
        let first = self.idx.prev[q];
        let mut cur = first;
        if hidden > 0 {
            // Chain nodes newer than q-1-hidden are not visible yet. They are
            // skipped without being counted as steps.
            while cur != NONE && (cur as usize) + hidden >= q {
                cur = self.idx.prev[cur as usize];
            }
        }
        if cur == first {
            let (off, len) = self.table.get(q);
            if len != OVERFLOW { return (off as usize, len as usize); }
        }
        walk(self.input, q, cur, &self.idx.prev, &self.p, None).unwrap_or((0, 0))
    }

    /// Queries at the last two positions: bucket 0 holds real bucket-0
    /// positions plus whichever tail positions the scan has inserted.
    fn find_tail(&self, q: usize, hidden: usize) -> (usize, usize) {
        let newest_visible = q.saturating_sub(1 + hidden);
        if q < 1 + hidden { return (0, 0); }
        let mut cands: Vec<u32> = Vec::new();
        for k in (0..2usize).rev() {
            let pos = self.n - 2 + k;
            if self.tail_inserted[k] && pos <= newest_visible { cands.push(pos as u32); }
        }
        let z = &self.idx.zero;
        let mut hi = z.partition_point(|&v| (v as usize) <= newest_visible);
        while hi > 0 && cands.len() < self.p.chain_limit + 2 {
            hi -= 1;
            cands.push(z[hi]);
        }
        walk_list(self.input, q, &cands, &self.p)
    }
}

/// Build the oracle for one scan, or `None` to use the plain hash chains.
pub(crate) fn oracle_for<'a>(input: &'a [u8], p: Params, mode: Accel) -> Option<Oracle<'a>> {
    if mode == Accel::Off { return None; }
    let n = input.len();
    if n < 3 || n > u32::MAX as usize - 1 || p.start_at + 3 > n { return None; }

    let table_and_index = match mode {
        Accel::CpuTable => {
            let idx = index_for(input);
            let table = compute_table_cpu(input, &idx, &p);
            Some((idx, table))
        }
        _ => {
            let span = n - p.start_at;
            if mode == Accel::Auto && span < ACCEL_MIN_BYTES { return None; }
            if span > ACCEL_MAX_BYTES { return None; }
            gpu_table(input, &p)
        }
    };
    let (idx, table) = table_and_index?;
    Some(Oracle { input, idx, table, p, n, tail_inserted: [false; 2] })
}

#[cfg(feature = "gpu")]
fn gpu_table(input: &[u8], p: &Params) -> Option<(Arc<PrevIndex>, Table)> {
    let ctx = crate::gpu::context()?;
    let idx = index_for(input);
    let table = ctx.compute_table(input, &idx, p)?;
    Some((idx, table))
}

#[cfg(not(feature = "gpu"))]
fn gpu_table(_input: &[u8], _p: &Params) -> Option<(Arc<PrevIndex>, Table)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::scan_from_mode;
    use crate::opcode::Token;

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    }

    /// Runs, short periods, repeated blocks, text-like words and noise. Lots of
    /// repeats right behind the query position (hidden look-ahead nodes), long
    /// runs (compare-cap overflow) and repeated endings (tail handling).
    fn nasty(seed: u64, len: usize) -> Vec<u8> {
        let mut r = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let words: [&[u8]; 8] = [b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy ", b"dog\n"];
        let mut v: Vec<u8> = Vec::new();
        while v.len() < len {
            match r.below(7) {
                0 => { let b = r.next() as u8; let n = 1 + r.below(60); v.extend(std::iter::repeat(b).take(n)); }
                1 => { let p = 1 + r.below(12); let pat: Vec<u8> = (0..p).map(|_| r.next() as u8).collect();
                       for k in 0..(p * (1 + r.below(9)) + r.below(p)) { v.push(pat[k % p]); } }
                2 if v.len() > 40 => { let s = r.below(v.len() - 20); let n = 1 + r.below(50).min(v.len() - s - 1); let blk = v[s..s + n].to_vec(); v.extend(blk); }
                3 => { for _ in 0..(1 + r.below(12)) { v.extend_from_slice(words[r.below(8)]); } }
                4 if r.below(40) == 0 => { let b = r.next() as u8; let n = 2500 + r.below(1500); v.extend(std::iter::repeat(b).take(n)); }
                _ => { for _ in 0..(1 + r.below(20)) { v.push((r.next() % 5) as u8); } }
            }
        }
        v.truncate(len);
        v
    }

    fn assert_same(label: &str, input: &[u8], start_at: usize, ob: u32, lb: u32, repref: bool) {
        let want = scan_from_mode(input, start_at, ob, lb, true, repref, Accel::Off);
        let got  = scan_from_mode(input, start_at, ob, lb, true, repref, Accel::CpuTable);
        assert!(want.0 == got.0,
            "{label}: tokens differ (n={} start_at={start_at} ob={ob} lb={lb} repref={repref}); \
             plain {} tokens vs table {} tokens, first difference at token {:?}",
            input.len(), want.0.len(), got.0.len(),
            want.0.iter().zip(got.0.iter()).position(|(a, b)| a != b));
        assert_eq!(want.1, got.1);
    }

    #[test]
    fn table_scan_equals_plain_scan_on_nasty_data() {
        let configs: [(u32, u32); 5] = [(7, 8), (10, 8), (15, 8), (16, 12), (12, 13)];
        for seed in 1..=8u64 {
            let data = nasty(seed, 3000 + (seed as usize) * 517);
            for &(ob, lb) in &configs {
                for &repref in &[false, true] {
                    assert_same(&format!("seed {seed}"), &data, 0, ob, lb, repref);
                }
            }
        }
    }

    #[test]
    fn table_scan_equals_plain_scan_at_every_tail_length() {
        // The last two positions share bucket 0 and are inserted only on literal
        // paths: vary the length so the end of the input lands in every state.
        let base = nasty(77, 700);
        for n in 3..=160usize {
            assert_same("tail", &base[..n], 0, 15, 8, true);
            // 7 offset bits + 8 length bits make even a 2-byte match "worthwhile"
            // (16 bits < 2 literals), so the tail matches actually show up in the tokens.
            assert_same("tail", &base[..n], 0, 7, 8, false);
            assert_same("tail", &base[..n], 0, 7, 8, true);
        }
        for n in 560..=700usize {
            assert_same("tail-long", &base[..n], 0, 12, 9, true);
        }
        // repeated endings: the end of the input matches earlier text
        let mut rep = nasty(5, 400);
        for extra in 0..=12usize {
            let mut d = rep.clone();
            d.extend_from_slice(&rep[100..100 + extra]);
            assert_same("repeated ending", &d, 0, 12, 8, true);
        }
        rep.extend_from_slice(&[0u8; 5]);
        assert_same("zero ending", &rep, 0, 12, 8, true);
    }

    #[test]
    fn table_scan_equals_plain_scan_with_a_dictionary_prefix() {
        for seed in 1..=5u64 {
            let dict = nasty(seed + 100, 200 + (seed as usize) * 37);
            let real = nasty(seed + 200, 3500);
            let mut combined = dict.clone();
            combined.extend_from_slice(&real);
            for &(ob, lb) in &[(12u32, 8u32), (16, 9)] {
                assert_same("dict", &combined, dict.len(), ob, lb, true);
            }
        }
    }

    #[test]
    fn table_scan_equals_plain_scan_on_long_runs_past_the_compare_cap() {
        // Runs longer than COMPARE_CAP with a length field wide enough to use them.
        let mut d = Vec::new();
        d.extend_from_slice(b"header-bytes");
        d.extend(std::iter::repeat(0u8).take(COMPARE_CAP * 3 + 17));
        d.extend_from_slice(b"middle section with some text in it");
        d.extend(std::iter::repeat(7u8).take(COMPARE_CAP + 5));
        d.extend(std::iter::repeat(0u8).take(COMPARE_CAP * 2));
        d.extend_from_slice(b"end");
        for &(ob, lb) in &[(14u32, 13u32), (14, 16), (16, 12)] {
            assert_same("long runs", &d, 0, ob, lb, false);
            assert_same("long runs", &d, 0, ob, lb, true);
        }
        // and the table really did overflow somewhere, so that path ran
        let idx = PrevIndex::build(&d);
        let p = Params { start_at: 0, max_off: (1 << 14) - 1, max_len: (1 << 16) - 1, chain_limit: 64 };
        let t = compute_table_cpu(&d, &idx, &p);
        assert!(t.entries.chunks(2).any(|e| e[1] == OVERFLOW), "expected overflow entries");
    }

    #[test]
    fn table_scan_equals_plain_scan_on_larger_mixed_data() {
        let data = nasty(9001, 40_000);
        assert_same("mixed 40k", &data, 0, 15, 8, true);
        assert_same("mixed 40k", &data, 0, 17, 9, false);
        let mut r = Xs(1234);
        let noise: Vec<u8> = (0..20_000).map(|_| r.next() as u8).collect();
        assert_same("noise", &noise, 0, 15, 8, true);
    }

    #[test]
    fn find_equals_chain_walk_for_every_hidden_count() {
        // Direct check of Oracle::find against the real chain, for hidden = 0, 1, 2.
        let data = nasty(31337, 4000);
        let n = data.len();
        let (ob, lb) = (14u32, 8u32);
        let max_off = crate::opcode::max_offset(ob);
        let max_len = crate::opcode::max_length(lb);
        let chain_limit = crate::encoder::compute_chain_limit(n, ob);
        let (wsz, wmask) = crate::encoder::window_alloc_and_mask(max_off, n);
        let p = Params { start_at: 0, max_off, max_len, chain_limit };
        let idx = index_for(&data);
        let table = compute_table_cpu(&data, &idx, &p);
        let mut o = Oracle { input: &data, idx: Arc::clone(&idx), table, p, n, tail_inserted: [false; 2] };
        let mut head = vec![u32::MAX; HASH_SIZE];
        let mut prev = vec![u32::MAX; wsz];
        for i in 0..n {
            for hidden in 0..=2usize {
                let q = i + hidden;
                if q >= n { continue; }
                let h = hash3(&data, q);
                let want = crate::encoder::find_match(&data, q, h, &head, &prev, max_off, max_len, wmask, chain_limit);
                assert_eq!(o.find(q, hidden), want, "i {i} hidden {hidden} q {q}");
            }
            // the scan's literal path: insert i, and tell the oracle about tail inserts
            let h = hash3(&data, i);
            prev[i & wmask] = head[h];
            head[h] = i as u32;
            o.note_insert(i);
        }
    }

    #[test]
    fn accel_policy_roundtrip() {
        // Only checks the setter/getter; other tests pass modes explicitly.
        let before = accel();
        set_accel(Accel::Off);
        assert_eq!(accel(), Accel::Off);
        set_accel(Accel::Force);
        assert_eq!(accel(), Accel::Force);
        set_accel(before);
    }

    #[test]
    fn tokens_are_comparable() {
        assert_eq!(Token::End, Token::End);
    }
}
