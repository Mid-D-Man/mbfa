// src/encoder.rs
// LZ-style scanner with rolling-window hash chain.
// O(n) time, O(window) memory -- safe for large files.
// Both offset_bits and length_bits are adaptive at runtime.
//
// scan_adaptive: Phase A/B/C fingerprint pipeline.
//
//   Phase A/B -- fingerprint + single scan (LARGE FILES ONLY, > 1 MB):
//     Phase A fingerprint_predict classifies the input and predicts (ob, lb).
//     Phase B runs one full-quality LZ scan at the predicted parameters, then
//     applies three checks IN ORDER before deciding whether to run Phase C:
//
//       1. Peak saturation (hard):  any Backref offset or length >= 90% of
//          field max. If true, the field is genuinely too small and Phase C
//          is REQUIRED. Must run BEFORE early exit — Repetitive_2MB has
//          phase_b_ratio << 15% but lb=8 saturates (99.9% of lengths = 255).
//          If early exit ran first it would bypass Phase C and return wrong ob/lb.
//
//       2. Early exit (only safe after peak passes): if Phase B ratio 
//          PHASE_B_EARLY_EXIT_RATIO (15%), Phase C gain is negligible.
//          Calibrated from JSON_2MB: ratio=8.41%, Phase C gain=0.33pp at cost
//          of a full extra scan on 3.1MB. Peak does NOT fire for JSON (lengths
//          typically 10-50 bytes, never near lb_ceil=229 at lb=8).
//
//       3. Upper-half saturation (softer): >= 20% of Backrefs have offset in
//          upper half of window. Only reached when peak is clear and ratio is
//          above the early-exit threshold. Catches WarAndPeace (large file,
//          many refs > 64KB in a 128KB window). JSON_2MB's upper-half check
//          would also fire (>20% of refs go > 64KB in 128KB window on 3MB data)
//          but early exit fires first at step 2, so JSON_2MB never reaches here.
//
//     Phase A/B is skipped entirely for files <= PARALLEL_SCAN_THRESHOLD (1 MB).
//     On small/medium files rayon::join(baseline, discovery) is already optimal.
//
//   Phase A fingerprint rules (applied only when input > 1 MB):
//     Rule 1: entropy < 2.0 (highly repetitive) -> defer to Phase C.
//     Rule 2: size < 32 KB -> predict ob = ceil(log2(size)), lb = 8.
//     Rule 3: default -> predict baseline (ob=17, lb=8).
//
//   Phase C -- discovery path:
//     Run scan_discover() -> compute wide_ob/wide_lb -> constrained re-scan if
//     different -> entropy safety cap check -> pick best result.
//     Baseline and discovery run IN PARALLEL via rayon::join when
//     input.len() <= PARALLEL_SCAN_THRESHOLD (1 MB).
//
//   Dynamic chain limit (full scan only):
//   Cache-aware tiered cap with size-aware branch at ob=17-18:
//
//     ob <= 16   window <=  64KB                        -> 256
//     ob 17-18   window <= 256KB
//       input <= PARALLEL_SCAN_THRESHOLD (1 MB)         -> 512  (small/medium: quality)
//       input >  PARALLEL_SCAN_THRESHOLD                -> 64   (large: avoid tokenisation
//                                                                 divergence on prose)
//     ob 19-20   window <=   1MB                        -> 256
//     ob >= 21   window >=   2MB                        -> 128
//
//   Power-of-two prev array:
//     prev is sized to next_power_of_two(max_off.min(n)) rather than
//     max_off.min(n). This makes j & window_mask a valid index, replacing
//     j % window_size (integer division) in the find_match hot loop.
//
//   Lazy hash updates in scan_discover:
//     When a match of length L is found during discovery, only position i is
//     added to the hash table. Discovery tokens are discarded. Measured 26-41%
//     speedup on large files.
//
//   Rep-match tiebreaking in scan():
//     Tracks the last REP_SLOTS (3) distinct offsets emitted as Backrefs.
//     After find_match returns a candidate, scan() checks each slot: if the
//     slot offset matches at the current position with length >= best_len, it
//     substitutes the rep-match offset. Fully backward compatible.

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

// -- Phase B early-exit threshold ---------------------------------------------
//
// Applied ONLY after peak saturation check passes (step 2 of 3).
// If Phase B ratio < 15%, Phase C gain is negligible (JSON_2MB: 8.41%, +0.33pp).
// Safe to apply here because peak saturation already confirmed the window is not
// too small — Repetitive_2MB never reaches this check (peak fires at step 1).
const PHASE_B_EARLY_EXIT_RATIO: f64 = 0.15;

// -- Fingerprint constants ----------------------------------------------------

const FINGERPRINT_ENTROPY_REPETITIVE: f64   = 2.0;
const FINGERPRINT_SMALL_FILE_BYTES:   usize = 32768;

// -- Ceiling saturation constants ---------------------------------------------

const CEILING_SATURATION_THRESHOLD: f64 = 0.9;
const CEILING_UPPER_HALF_THRESHOLD: f64 = 0.20;

// -- Rep-match tiebreaking ----------------------------------------------------

const REP_SLOTS: usize = 3;

const HASH_SIZE:  usize = 1 << 16;
const HASH_MASK:  usize = HASH_SIZE - 1;
const LAZY_SHORT_LEN: usize = 6;

const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

// -- Discovery scan constants -------------------------------------------------

const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;

// -- Tiered chain limit -------------------------------------------------------

pub fn compute_chain_limit(input_len: usize, offset_bits: u32) -> usize {
    match offset_bits {
        0..=16 => 256,
        17..=18 => {
            if input_len > PARALLEL_SCAN_THRESHOLD { 64 } else { 512 }
        }
        19..=20 => 256,
        _       => 128,
    }
}

// -- Window helpers -----------------------------------------------------------

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
    fn new() -> Self { Self { slots: [0u32; REP_SLOTS], len: 0 } }

    #[inline]
    fn push(&mut self, offset: u32) {
        if self.len > 0 && self.slots[0] == offset { return; }
        if REP_SLOTS > 2 { self.slots[2] = self.slots[1]; }
        if REP_SLOTS > 1 { self.slots[1] = self.slots[0]; }
        self.slots[0] = offset;
        if self.len < REP_SLOTS { self.len += 1; }
    }

    #[inline]
    fn valid(&self) -> &[u32] { &self.slots[..self.len] }
}

// -- Rep-match length probe ---------------------------------------------------

#[inline]
fn rep_match_len(input: &[u8], i: usize, offset: u32, max_len: usize) -> usize {
    let off = offset as usize;
    if off == 0 || off > i { return 0; }
    let j = i - off;
    let n = input.len();
    let mut len = 0;
    while len < max_len && (i + len) < n && input[j + (len % off)] == input[i + len] {
        len += 1;
    }
    len
}

// -- Core scanner -------------------------------------------------------------

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

        prev[i & window_mask] = head[h];
        head[h] = i as u32;

        if best_len >= 2 {
            tokens.push(Token::Backref {
                offset: best_offset as u32,
                length: best_len as u32,
            });
            i += best_len;
        } else {
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
    window_mask: usize,
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

// -- Ceiling saturation: split into two independent checks --------------------
//
// peak_saturated:       HARD check — any single Backref hits >= 90% of field max.
//                       Means the window or length field is genuinely too small.
//                       Must run BEFORE early exit.
//
// upper_half_saturated: SOFT check — >= 20% of Backrefs reference the upper half
//                       of the offset window. Means longer lookback is likely useful.
//                       Only runs AFTER peak and early exit both pass.

fn peak_saturated(tokens: &[Token], ob: u32, lb: u32) -> bool {
    let max_off = max_offset(ob);
    let max_len = max_length(lb);
    let ob_ceil = ((max_off as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    let lb_ceil = ((max_len as f64) * CEILING_SATURATION_THRESHOLD) as u32;

    for t in tokens {
        if let Token::Backref { offset, length } = t {
            if *offset >= ob_ceil || *length >= lb_ceil {
                println!(
                    "  ceiling check: peak hit offset={} (ceil={}) or length={} (ceil={})",
                    offset, ob_ceil, length, lb_ceil
                );
                return true;
            }
        }
    }
    false
}

fn upper_half_saturated(tokens: &[Token], ob: u32) -> bool {
    let max_off  = max_offset(ob);
    let half_off = (max_off / 2) as u32;

    let mut total_br:   u64 = 0;
    let mut upper_half: u64 = 0;

    for t in tokens {
        if let Token::Backref { offset, .. } = t {
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

fn log_window_diagnostics(tokens: &[Token], offset_bits: u32, length_bits: u32) {
    let max_off = (1u32 << offset_bits) - 1;
    let max_len = (1u32 << length_bits) - 1;
    let mut total_br:   u64 = 0;
    let mut at_max_off: u64 = 0;
    let mut above_half: u64 = 0;
    let mut at_max_len: u64 = 0;
    let mut total_lit:  u64 = 0;

    for t in tokens {
        match t {
            Token::Backref { offset, length } => {
                total_br += 1;
                if *offset == max_off { at_max_off += 1; }
                if *offset > max_off / 2 { above_half += 1; }
                if *length == max_len { at_max_len += 1; }
            }
            Token::Lit { .. } => total_lit += 1,
            Token::End => {}
        }
    }

    if total_br + total_lit == 0 { return; }
    let total    = total_br + total_lit;
    let br_pct   = total_br    as f64 / total    as f64 * 100.0;
    let sat_pct  = if total_br > 0 { at_max_off as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let deep_pct = if total_br > 0 { above_half as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let len_pct  = if total_br > 0 { at_max_len as f64 / total_br as f64 * 100.0 } else { 0.0 };

    println!(
        "Window diagnostics: {}/{} tokens BACKREF ({:.1}%) | \
         offset: {:.1}% at max ({}) | {:.1}% upper half | \
         length: {:.1}% at max ({}) | offset_bits={} length_bits={} — {}",
        total_br, total, br_pct,
        sat_pct, max_off, deep_pct,
        len_pct, max_len,
        offset_bits, length_bits,
        if sat_pct > 5.0        { "⚠ OFFSET SATURATED — window too small" }
        else if len_pct > 5.0   { "⚠ LENGTH SATURATED — length field too small" }
        else if deep_pct > 30.0 { "HEAVY — large offsets dominant" }
        else                    { "OK" }
    );
}

/// Baseline parameters used by Phase B in encoder.rs — mirrored here
/// solely for the Phase C gain diagnostic.
const DIAG_BASELINE_OB: u32 = 17;
const DIAG_BASELINE_LB: u32 = 8;

fn diag_stream_bit_cost(tokens: &[Token], ob: u32, lb: u32) -> u64 {
    let br_bits = backref_total_bits(ob, lb) as u64;
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => br_bits,
        Token::End            => END_TOTAL_BITS as u64,
    }).sum()
}

fn log_phase_c_gain(tokens: &[Token], ob: u32, lb: u32, input_len: usize) {
    if ob <= DIAG_BASELINE_OB && lb <= DIAG_BASELINE_LB {
        return;
    }
    let input_bits    = input_len as f64 * 8.0;
    let actual_cost   = diag_stream_bit_cost(tokens, ob, lb);
    let baseline_cost = diag_stream_bit_cost(tokens, DIAG_BASELINE_OB, DIAG_BASELINE_LB);
    let actual_pct    = actual_cost   as f64 / input_bits * 100.0;
    let baseline_pct  = baseline_cost as f64 / input_bits * 100.0;
    let gain_pp       = baseline_pct - actual_pct;
    println!(
        "  DIAG phase_c_gain: ob={} lb={} \
         actual_ratio={:.4}% baseline_hyp_ratio={:.4}% \
         gain_pp={:.4}pp input_bytes={}",
        ob, lb, actual_pct, baseline_pct, gain_pp, input_len,
    );
}

// -- Phase C inner logic ------------------------------------------------------

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

                // ── Step 1: Peak saturation (HARD — must run before early exit) ──
                //
                // Catches Repetitive_2MB: lb=8 max=255, practically every match hits
                // length=255 → lb_ceil=229 fires immediately. Phase B ratio is also
                // low (~<1%) but we CANNOT early exit here — the field is genuinely
                // too small. Returning ob=17 lb=8 would produce a 183-byte output
                // instead of the correct 131-byte output at ob=10 lb=21.
                if peak_saturated(&result.tokens, pred_ob, pred_lb) {
                    println!("  ceiling check: peak saturated -> Phase C required");
                    if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                        let baseline  = ScanResult { label: "baseline", ..result };
                        let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                        return phase_c_from_baseline(baseline, discovery, input);
                    }
                    // pred_ob != baseline (Rule 2, not reached for >1MB): fall through
                    // to parallel Phase C below.
                } else {
                    // ── Step 2: Early exit (safe — peak saturation already cleared) ──
                    //
                    // Catches JSON_2MB: lengths 10-50 bytes, never near lb_ceil=229.
                    // phase_b_ratio=8.41% < 15%. Phase C would find ob=21 but only
                    // gains 0.09pp at cost of a full extra scan on 3.1MB (~1200ms).
                    //
                    // JSON_2MB upper-half% is >20% at ob=17 (many refs span >64KB in
                    // a 3MB file with 128KB window) so step 3 WOULD fire without this
                    // early exit. We never reach step 3 for JSON_2MB.
                    let phase_b_ratio = result.raw_cost as f64
                        / (input.len() as f64 * 8.0);

                    if phase_b_ratio < PHASE_B_EARLY_EXIT_RATIO {
                        println!(
                            "  Phase B early exit: ratio={:.4}% < {:.0}% threshold \
                             -- peak clear, skipping Phase C",
                            phase_b_ratio * 100.0,
                            PHASE_B_EARLY_EXIT_RATIO * 100.0,
                        );
                        return (result.tokens, result.ob, result.lb);
                    }

                    // ── Step 3: Upper-half saturation (SOFT) ─────────────────────
                    //
                    // Catches WarAndPeace and other large prose/source/binary files
                    // where ob=17 is genuinely too small. These have high ratio (>15%)
                    // so early exit does not fire.
                    if upper_half_saturated(&result.tokens, pred_ob) {
                        println!("  ceiling check: upper-half saturated -> Phase C");
                        if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                            let baseline  = ScanResult { label: "baseline", ..result };
                            let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                            return phase_c_from_baseline(baseline, discovery, input);
                        }
                        // pred_ob != baseline: fall through to parallel Phase C below.
                    } else {
                        // Not peak-saturated, not upper-half saturated, ratio above
                        // threshold: Phase B window is sufficient, return as-is.
                        println!("  ceiling check: ok -- Phase B wins (1 scan total)");
                        return (result.tokens, result.ob, result.lb);
                    }
                }
            }
        }
    }

    // ── Phase C parallel path ─────────────────────────────────────────────────
    // Reached by:
    //   (a) files <= PARALLEL_SCAN_THRESHOLD (rayon::join baseline + discovery)
    //   (b) files > threshold where fingerprint returned None (Rule 1: repetitive)
    //   (c) files > threshold where pred_ob != BASELINE (Rule 2: not reached in
    //       practice for >1MB, kept for correctness)
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

    let (tokens, ob, lb) = phase_c_from_baseline(baseline, wide_discovery, input);

    log_window_diagnostics(&tokens, ob, lb);
    log_phase_c_gain(&tokens, ob, lb, input.len());

    (tokens, ob, lb)
}
