// src/encoder.rs
// LZ-style scanner with rolling-window hash chain.
// O(n) time, O(window) memory -- safe for large files.
// Both offset_bits and length_bits are adaptive at runtime.
//
// scan_adaptive: Phase A/B/C fingerprint pipeline.
//
//   Phase A/B -- fingerprint + single scan (LARGE FILES ONLY, > 1 MB):
//     Phase A fingerprint_predict classifies the input and predicts (ob, lb).
//     Phase B runs one full-quality scan at the predicted parameters, then
//     ceiling_saturated() checks whether the window was sufficient.
//       Not saturated -> return immediately (1 scan total).
//       Saturated, pred matches baseline -> reuse tokens as baseline in Phase C,
//         skip the baseline re-scan (saves 1 scan).
//       Saturated, pred differs from baseline -> fall through to Phase C fresh.
//
//     Phase A/B is skipped entirely for files <= PARALLEL_SCAN_THRESHOLD (1 MB).
//     On small/medium files rayon::join(baseline, discovery) is already optimal.
//     Phase B runs sequentially before discovery, killing the parallel speedup
//     whenever ceiling fires (which it does on most medium prose/source/binary
//     files with diverse offsets).
//
//   Phase A fingerprint rules (applied only when input > 1 MB):
//     Rule 1: entropy < 2.0 (highly repetitive) -> defer to Phase C.
//             lb will saturate at lb=8; ceiling check fires immediately anyway.
//             Skipping Phase B saves a wasted full-quality scan.
//     Rule 2: size < 32 KB -> predict ob = ceil(log2(size)), lb = 8.
//             Not reached in practice (gated behind > 1 MB check), kept for
//             correctness if threshold changes.
//     Rule 3: default -> predict baseline (ob=17, lb=8).
//
//   Phase C -- discovery path (original logic, unchanged):
//     Run scan_discover() -> compute wide_ob/wide_lb -> constrained re-scan if
//     different -> entropy safety cap check -> pick best result.
//     Baseline and discovery run IN PARALLEL via rayon::join when
//     input.len() <= PARALLEL_SCAN_THRESHOLD (1 MB).
//
//   Dynamic chain limit (full scan only):
//   Cache-aware tiered cap (W7: flattened ob 17-18 tier from 512 to 256):
//     ob <= 20 (window <=  1MB)  -> 256
//     ob >= 21 (window >=  2MB)  -> 128
//   The previous 512-depth tier at ob=17-18 caused greedy tokenisation
//   divergence on large prose (WarAndPeace), where deeper chains found
//   different match boundaries that propagated worse results downstream.
//   LZ tokenisation quality is NOT monotone with chain depth.
//
//   Power-of-two prev array (W8):
//   prev is sized to next_power_of_two(max_off.min(n)) rather than max_off.min(n).
//   This ensures j & window_mask is a valid index, replacing j % window_size
//   (a real integer division) in the chain traversal hot loop.
//   The extra entry is always u32::MAX (uninitialized); find_match's
//   i - j > max_off guard rejects any position at distance >= 1<<ob immediately.
//
//   Lazy hash updates in scan_discover (W8):
//   When a match of length L is found during discovery, only position i is
//   added to the hash table (not all L intermediate positions). Discovery tokens
//   are discarded — we only need the maximum ob/lb values, not quality chains.
//   Skipping the inner O(L) loop avoids catastrophic slowdown on files with
//   very long matches (e.g. Source_1MB with lb=20, where a single match can
//   span hundreds of KB and trigger hundreds of thousands of hash updates).
//
//   Ceiling saturation check (two conditions, either fires Phase C):
//     Peak check:       any Backref offset or length >= 90% of field max.
//     Upper-half check: >= 20% of Backrefs have offset in upper half of window.
//                       Catches files like WarAndPeace where the window is
//                       heavily used but no single backref hits the peak.
//
//   Rep-match tiebreaking in scan():
//     Tracks the last REP_SLOTS (3) distinct offsets emitted as Backrefs.
//     After find_match returns a candidate, scan() checks each slot: if the
//     slot offset matches at the current position with length >= best_len, it
//     substitutes the rep-match offset. The token type is unchanged
//     (Token::Backref) -- this is purely a scanner-quality improvement.
//     No opcode vocabulary change. No format change. Fully backward compatible.
//     v4/v5 entropy slot LRU applies on top and captures additional coding gain.
//
//   Discovery scan (scan_discover):
//   Uses DISCOVER_CHAIN_LIMIT (fixed 256) and caps prev array at
//   DISCOVER_MAX_PREV_SIZE (2MB). No lazy matching. Fast and approximate --
//   only used to determine ob/lb range, never for final output.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MIN, OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};
use crate::entropy::{offset_to_bucket, bucket_extra_bits};
use rayon;

const BASELINE_OFFSET_BITS:       u32   = 17;
const BASELINE_LENGTH_BITS:       u32   = LENGTH_BITS_MIN; // 8
const ENTROPY_SAFE_LENGTH_BITS:   u32   = 15;
const ENTROPY_MIN_BYTES_FOR_SCAN: usize = 400;
const ENTROPY_TIE_THRESHOLD:      f64   = 0.05;

// -- Fingerprint constants ----------------------------------------------------

const FINGERPRINT_ENTROPY_REPETITIVE: f64   = 2.0;
const FINGERPRINT_SMALL_FILE_BYTES:   usize = 32768;

// -- Ceiling saturation constants ---------------------------------------------

const CEILING_SATURATION_THRESHOLD: f64 = 0.9;
const CEILING_UPPER_HALF_THRESHOLD: f64 = 0.20;

// -- Rep-match tiebreaking ----------------------------------------------------

const REP_SLOTS: usize = 3;

const HASH_SIZE:              usize = 1 << 16;
const HASH_MASK:              usize = HASH_SIZE - 1;
const LAZY_SHORT_LEN:         usize = 6;

const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

// -- Discovery scan constants -------------------------------------------------

const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;  // 2^21 bytes — also a power of two

// -- Tiered chain limit -------------------------------------------------------

/// Cache-aware tiered chain depth limit for full-quality scans.
///
/// W7: flattened from the previous four-tier layout to two tiers.
/// The ob=17-18 tier previously used 512 which caused tokenisation divergence
/// on large prose. 256 recovers quality while retaining the speed gains from W3.
///
///   ob <= 20   window <=   1MB   -> 256
///   ob >= 21   window >=   2MB   -> 128
pub fn compute_chain_limit(_input_len: usize, offset_bits: u32) -> usize {
    match offset_bits {
        0..=20 => 256,
        _      => 128,
    }
}

// -- Window helpers -----------------------------------------------------------

/// Returns (alloc_size, window_mask) for the prev circular buffer.
///
/// alloc_size = next_power_of_two(max_off.min(n).max(1))
/// window_mask = alloc_size - 1
///
/// Because alloc_size is always a power of two, `j & window_mask` replaces
/// `j % window_size` in the find_match hot loop — AND instead of division.
///
/// Correctness: the one extra entry (vs old max_off.min(n)) is initialised to
/// u32::MAX. find_match guards `i - j > max_off` before using any chain entry,
/// so the extra slot can never yield a false match.
#[inline]
fn window_alloc_and_mask(max_off_limit: usize, n: usize) -> (usize, usize) {
    let raw  = max_off_limit.min(n).max(1);
    let pow2 = raw.next_power_of_two();
    (pow2, pow2 - 1)
}

// -- Hash ---------------------------------------------------------------------

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos]     as usize).wrapping_mul(2_654_435_761)
        ^   (input[pos + 1] as usize).wrapping_mul(2_246_822_519)
        ^   (input[pos + 2] as usize).wrapping_mul(3_266_489_917);
    v & HASH_MASK
}

// -- Rep-match slot tracker ---------------------------------------------------

struct RepSlots {
    slots: [u32; REP_SLOTS],
    len:   usize,
}

impl RepSlots {
    #[inline]
    fn new() -> Self {
        Self { slots: [0u32; REP_SLOTS], len: 0 }
    }

    #[inline]
    fn push(&mut self, offset: u32) {
        if self.len > 0 && self.slots[0] == offset { return; }
        if REP_SLOTS > 2 { self.slots[2] = self.slots[1]; }
        if REP_SLOTS > 1 { self.slots[1] = self.slots[0]; }
        self.slots[0] = offset;
        if self.len < REP_SLOTS { self.len += 1; }
    }

    #[inline]
    fn valid(&self) -> &[u32] {
        &self.slots[..self.len]
    }
}

// -- Rep-match length probe ---------------------------------------------------

#[inline]
fn rep_match_len(input: &[u8], i: usize, offset: u32, max_len: usize) -> usize {
    let off = offset as usize;
    if off == 0 || off > i { return 0; }
    let j   = i - off;
    let n   = input.len();
    let mut len = 0;
    while len < max_len && (i + len) < n && input[j + (len % off)] == input[i + len] {
        len += 1;
    }
    len
}

// -- Core scanner -------------------------------------------------------------

/// Full-quality production scan with rep-match tiebreaking.
pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off      = max_offset(offset_bits);
    let max_len      = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);
    let chain_limit  = compute_chain_limit(input.len(), offset_bits);

    let n = input.len();
    let (window_size, window_mask) = window_alloc_and_mask(max_off, n);

    let mut head      = vec![u32::MAX; HASH_SIZE];
    let mut prev      = vec![u32::MAX; window_size];
    let mut tokens    = Vec::with_capacity(n / 2 + 1);
    let mut rep_slots = RepSlots::new();
    let mut i         = 0;

    while i < n {
        let h = hash3(input, i);
        let (mut best_offset, mut best_len) =
            find_match(input, i, h, &head, &prev, max_off, max_len, window_mask, chain_limit);

        // -- Rep-match tiebreaking --------------------------------------------
        let ob_beats_lit = offset_bits > LIT_TOTAL_BITS;

        for &slot_off in rep_slots.valid() {
            if slot_off as usize > max_off { continue; }
            let rlen = rep_match_len(input, i, slot_off, max_len);
            if rlen == 0 { continue; }

            let prefer = if rlen >= best_len {
                true
            } else if ob_beats_lit && best_len > 0 && rlen == best_len.saturating_sub(1) {
                true
            } else {
                false
            };

            if prefer {
                best_offset = slot_off as usize;
                best_len    = rlen;
                break;
            }
        }
        // -- End rep-match tiebreaking ----------------------------------------

        let backref_worthwhile = best_len >= 2
            && backref_bits < (best_len as u32 * LIT_TOTAL_BITS);

        if backref_worthwhile {
            let lazy = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, len1) = find_match(
                    input, i + 1, h1, &head, &prev, max_off, max_len, window_mask, chain_limit,
                );
                if len1 > best_len {
                    true
                } else if best_len <= LAZY_SHORT_LEN && i + 2 < n {
                    let h2 = hash3(input, i + 2);
                    let (_, len2) = find_match(
                        input, i + 2, h2, &head, &prev, max_off, max_len, window_mask, chain_limit,
                    );
                    len2 > best_len + 2
                } else {
                    false
                }
            } else {
                false
            };

            if lazy {
                prev[i & window_mask] = head[h];
                head[h] = i as u32;
                tokens.push(Token::Lit { byte: input[i] });
                i += 1;
            } else {
                for k in 0..best_len {
                    if i + k + 2 < n {
                        let hk = hash3(input, i + k);
                        prev[(i + k) & window_mask] = head[hk];
                        head[hk] = (i + k) as u32;
                    }
                }
                rep_slots.push(best_offset as u32);
                tokens.push(Token::Backref {
                    offset: best_offset as u32,
                    length: best_len as u32,
                });
                i += best_len;
            }
        } else {
            prev[i & window_mask] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Lit { byte: input[i] });
            i += 1;
        }
    }

    tokens.push(Token::End);
    tokens
}

/// Fast discovery scan — used only to determine ob/lb range. Token stream
/// is discarded by the caller.
///
/// Differences from scan():
/// - Single hash update per matched span (not the full O(L) inner loop).
///   Since tokens are discarded we only need max ob/lb, not quality chains.
///   The inner loop is the dominant cost on highly-compressible files where
///   single matches can span hundreds of KB (e.g. Source_1MB with lb=20).
/// - No rep-match tracking needed — tokens discarded.
/// - No lazy matching.
pub fn scan_discover(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off = max_offset(offset_bits).min(DISCOVER_MAX_PREV_SIZE);
    let max_len = max_length(length_bits);
    let n       = input.len();

    let (window_size, window_mask) = window_alloc_and_mask(max_off, n);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::with_capacity(n / 2 + 1);
    let mut i      = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) = find_match(
            input, i, h, &head, &prev,
            max_off, max_len, window_mask,
            DISCOVER_CHAIN_LIMIT,
        );

        if best_len >= 2 {
            // Update current position only — skip intermediate positions i+1..i+best_len.
            // We need max ob/lb values, not dense chains. Skipping the O(best_len) inner
            // loop avoids catastrophic slowdown when best_len is very large (e.g. lb=20
            // on Source_1MB where a single match can be hundreds of KB long).
            prev[i & window_mask] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Backref {
                offset: best_offset as u32,
                length: best_len as u32,
            });
            i += best_len;
        } else {
            prev[i & window_mask] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Lit { byte: input[i] });
            i += 1;
        }
    }

    tokens.push(Token::End);
    tokens
}

fn find_match(
    input:       &[u8],
    i:           usize,
    h:           usize,
    head:        &[u32],
    prev:        &[u32],
    max_off:     usize,
    max_len:     usize,
    window_mask: usize,   // AND mask — prev has next_power_of_two(max_off) entries
    chain_limit: usize,
) -> (usize, usize) {
    let n = input.len();
    let mut best_offset = 0;
    let mut best_len    = 0;
    let mut steps       = 0;

    let mut cur = head[h];
    while cur != u32::MAX && steps < chain_limit {
        let j = cur as usize;
        if i <= j || i - j > max_off { break; }

        let span = i - j;
        let mut len = 0;
        while len < max_len
            && (i + len) < n
            && input[j + (len % span)] == input[i + len]
        {
            len += 1;
        }

        if len > best_len {
            best_len    = len;
            best_offset = span;
            if best_len == max_len { break; }
        }

        cur   = prev[j & window_mask];
        steps += 1;
    }

    (best_offset, best_len)
}

// -- Cost helpers -------------------------------------------------------------

fn stream_bit_cost(tokens: &[Token], ob: u32, lb: u32) -> u64 {
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => backref_total_bits(ob, lb) as u64,
        Token::End            => END_TOTAL_BITS as u64,
    }).sum()
}

fn stream_entropy_cost(tokens: &[Token]) -> f64 {
    use std::collections::HashMap;

    let mut lit_len_freq: HashMap<u32, u64> = HashMap::new();
    let mut bucket_freq:  HashMap<u32, u64> = HashMap::new();
    let mut raw_extra:    u64               = 0;

    for t in tokens {
        match t {
            Token::Lit { byte } => {
                *lit_len_freq.entry(*byte as u32).or_insert(0) += 1;
            }
            Token::Backref { offset, length } => {
                let length_sym = 255u32 + length;
                *lit_len_freq.entry(length_sym).or_insert(0) += 1;
                let (bucket, extra_bits, _) = offset_to_bucket(*offset);
                *bucket_freq.entry(bucket).or_insert(0) += 1;
                raw_extra += extra_bits as u64;
            }
            Token::End => {
                *lit_len_freq.entry(256u32).or_insert(0) += 1;
            }
        }
    }

    shannon_bits(&lit_len_freq) + shannon_bits(&bucket_freq) + raw_extra as f64
}

#[inline]
fn shannon_bits(freq: &std::collections::HashMap<u32, u64>) -> f64 {
    let total: u64 = freq.values().sum();
    if total == 0 { return 0.0; }
    let total_f = total as f64;
    freq.values()
        .filter(|&&c| c > 0)
        .map(|&c| -(c as f64) * (c as f64 / total_f).log2())
        .sum()
}

// -- Scan result wrapper ------------------------------------------------------

struct ScanResult {
    tokens:   Vec<Token>,
    ob:       u32,
    lb:       u32,
    raw_cost: u64,
    label:    &'static str,
}

impl ScanResult {
    fn new(tokens: Vec<Token>, ob: u32, lb: u32, label: &'static str) -> Self {
        let raw_cost = stream_bit_cost(&tokens, ob, lb);
        ScanResult { tokens, ob, lb, raw_cost, label }
    }

    fn beats(&self, other: &ScanResult) -> bool {
        let min_cost = self.raw_cost.min(other.raw_cost) as f64;
        let diff     = (self.raw_cost as f64 - other.raw_cost as f64).abs();

        if min_cost > 0.0 && diff / min_cost < ENTROPY_TIE_THRESHOLD {
            let self_e  = stream_entropy_cost(&self.tokens);
            let other_e = stream_entropy_cost(&other.tokens);
            println!(
                "  entropy tiebreaker: {} est={:.0}b vs {} est={:.0}b -> {} wins",
                self.label, self_e,
                other.label, other_e,
                if self_e < other_e { self.label } else { other.label },
            );
            self_e < other_e
        } else {
            self.raw_cost < other.raw_cost
        }
    }
}

// -- Fingerprint helpers ------------------------------------------------------

fn sample_entropy_fingerprint(data: &[u8]) -> f64 {
    const SAMPLE: usize = 8192;
    let sample = if data.len() > SAMPLE { &data[..SAMPLE] } else { data };
    if sample.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for &b in sample { freq[b as usize] += 1; }
    let n = sample.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
}

fn fingerprint_predict(data: &[u8]) -> Option<(u32, u32)> {
    let ent = sample_entropy_fingerprint(data);
    println!("  fingerprint: entropy={:.2} size={}", ent, data.len());

    if ent < FINGERPRINT_ENTROPY_REPETITIVE {
        println!("  fingerprint Rule 1: repetitive -> defer to Phase C");
        return None;
    }

    if data.len() < FINGERPRINT_SMALL_FILE_BYTES {
        let ob = if data.is_empty() {
            OFFSET_BITS_MIN
        } else {
            let bits = (usize::BITS - data.len().leading_zeros()) as u32;
            bits.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
        };
        println!("  fingerprint Rule 2: small file -> ob={} lb={}", ob, LENGTH_BITS_MIN);
        return Some((ob, LENGTH_BITS_MIN));
    }

    println!("  fingerprint Rule 3: default -> ob={} lb={}", BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    Some((BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS))
}

fn ceiling_saturated(tokens: &[Token], ob: u32, lb: u32) -> bool {
    let max_off  = max_offset(ob);
    let max_len  = max_length(lb);
    let ob_ceil  = ((max_off as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    let lb_ceil  = ((max_len as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    let half_off = (max_off / 2) as u32;

    let mut total_br:   u64 = 0;
    let mut upper_half: u64 = 0;

    for t in tokens {
        if let Token::Backref { offset, length } = t {
            if *offset >= ob_ceil || *length >= lb_ceil {
                println!("  ceiling check: peak hit offset={} (ceil={}) or length={} (ceil={})",
                    offset, ob_ceil, length, lb_ceil);
                return true;
            }
            total_br += 1;
            if *offset > half_off { upper_half += 1; }
        }
    }

    if total_br > 0 {
        let upper_frac = upper_half as f64 / total_br as f64;
        if upper_frac >= CEILING_UPPER_HALF_THRESHOLD {
            println!(
                "  ceiling check: upper-half saturated {:.1}% >= {:.0}% threshold -- Phase C needed",
                upper_frac * 100.0,
                CEILING_UPPER_HALF_THRESHOLD * 100.0,
            );
            return true;
        }
        println!(
            "  ceiling check: ok -- upper-half {:.1}% < {:.0}% threshold",
            upper_frac * 100.0,
            CEILING_UPPER_HALF_THRESHOLD * 100.0,
        );
    }

    false
}

// -- Phase C inner logic (shared by both paths) -------------------------------

fn phase_c_from_baseline(
    baseline:       ScanResult,
    wide_discovery: Vec<Token>,
    input:          &[u8],
) -> (Vec<Token>, u32, u32) {

    println!(
        "  chain_limit (baseline ob={}): {}  raw_cost={}",
        BASELINE_OFFSET_BITS,
        compute_chain_limit(input.len(), BASELINE_OFFSET_BITS),
        baseline.raw_cost,
    );

    let wide_ob = compute_optimal_offset_bits(&wide_discovery);
    let wide_lb = compute_optimal_length_bits(&wide_discovery);
    drop(wide_discovery);

    println!(
        "  discovery (capped 2MB/chain={}): discovered ob={} lb={}",
        DISCOVER_CHAIN_LIMIT, wide_ob, wide_lb,
    );

    if wide_ob == BASELINE_OFFSET_BITS && wide_lb == BASELINE_LENGTH_BITS {
        println!("  wide agrees with baseline -- no re-scan needed");
        return (baseline.tokens, baseline.ob, baseline.lb);
    }

    let constrained = ScanResult::new(
        scan(input, wide_ob, wide_lb),
        wide_ob,
        wide_lb,
        "constrained",
    );
    println!(
        "  chain_limit (constrained ob={}): {}  raw_cost={}",
        wide_ob,
        compute_chain_limit(input.len(), wide_ob),
        constrained.raw_cost,
    );

    if constrained.lb > ENTROPY_SAFE_LENGTH_BITS {
        let constrained_bytes = (constrained.raw_cost as usize + 7) / 8;
        if constrained_bytes >= ENTROPY_MIN_BYTES_FOR_SCAN {
            let capped = ScanResult::new(
                scan(input, constrained.ob, ENTROPY_SAFE_LENGTH_BITS),
                constrained.ob,
                ENTROPY_SAFE_LENGTH_BITS,
                "capped",
            );
            println!(
                "  lb cap: constrained lb={} output={}B >= threshold {}B \
                 -- capped raw_cost={} chain_limit={}",
                constrained.lb, constrained_bytes, ENTROPY_MIN_BYTES_FOR_SCAN,
                capped.raw_cost,
                compute_chain_limit(input.len(), capped.ob),
            );

            return if capped.beats(&baseline) {
                println!("  capped wins over baseline");
                (capped.tokens, capped.ob, capped.lb)
            } else {
                println!("  baseline wins over capped");
                (baseline.tokens, baseline.ob, baseline.lb)
            };
        }
    }

    if constrained.beats(&baseline) {
        println!(
            "  constrained wins: ob={} lb={} raw_cost={} vs baseline raw_cost={}",
            constrained.ob, constrained.lb, constrained.raw_cost, baseline.raw_cost,
        );
        (constrained.tokens, constrained.ob, constrained.lb)
    } else {
        println!(
            "  baseline wins: ob={} lb={} raw_cost={} vs constrained raw_cost={}",
            baseline.ob, baseline.lb, baseline.raw_cost, constrained.raw_cost,
        );
        (baseline.tokens, baseline.ob, baseline.lb)
    }
}

// -- Public API ---------------------------------------------------------------

pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {

    if input.len() > PARALLEL_SCAN_THRESHOLD {
        match fingerprint_predict(input) {
            None => {
                // Rule 1: highly repetitive -- fall through to Phase C.
            }
            Some((pred_ob, pred_lb)) => {
                let tokens = scan(input, pred_ob, pred_lb);
                let result = ScanResult::new(tokens, pred_ob, pred_lb, "phase_b");

                println!(
                    "  Phase B scan: ob={} lb={} chain_limit={} raw_cost={}",
                    pred_ob, pred_lb,
                    compute_chain_limit(input.len(), pred_ob),
                    result.raw_cost,
                );

                if !ceiling_saturated(&result.tokens, pred_ob, pred_lb) {
                    println!("  ceiling check: ok -- Phase B wins (1 scan total)");
                    return (result.tokens, result.ob, result.lb);
                }

                println!("  ceiling check: saturated -> falling through to Phase C");

                if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                    let baseline  = ScanResult { label: "baseline", ..result };
                    let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                    return phase_c_from_baseline(baseline, discovery, input);
                }
            }
        }
    }

    let (baseline, wide_discovery) = if input.len() <= PARALLEL_SCAN_THRESHOLD {
        let (bt, disc) = rayon::join(
            || scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS),
            || scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX),
        );
        (
            ScanResult::new(bt, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, "baseline"),
            disc,
        )
    } else {
        let bt   = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
        let disc = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
        (
            ScanResult::new(bt, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, "baseline"),
            disc,
        )
    };

    phase_c_from_baseline(baseline, wide_discovery, input)
}
