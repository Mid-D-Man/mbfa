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
//   Cache-aware tiered cap:
//     ob <= 16 (window <=  64KB, L2-warm)  -> 256
//     ob 17-18 (window <= 256KB, L3-warm)  -> 512
//     ob 19-20 (window <=   1MB, L3-tight) -> 256
//     ob >= 21 (window >=   2MB, DRAM)     -> 128
//
//   Ceiling saturation check (two conditions, either fires Phase C):
//     Peak check:      any Backref offset or length >= 90% of field max.
//     Upper-half check: >= 20% of Backrefs have offset in upper half of window.
//                       Catches files like WarAndPeace where the window is
//                       heavily used but no single backref hits the peak.
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

/// Peak check: any single Backref offset or length at or above this fraction
/// of field maximum is treated as saturated.
const CEILING_SATURATION_THRESHOLD: f64 = 0.9;

/// Upper-half check: if this fraction or more of all Backrefs have an offset
/// in the upper half of the current window, treat as saturated and run Phase C.
///
/// Catches files where the window is heavily used across its full depth but no
/// single backref actually peaks at 90% -- the classic case being large prose
/// files (WarAndPeace: 25% upper-half, JSON_2MB: 24% upper-half) where a wider
/// ob discovers substantially better ratio.
///
/// Calibration against known results:
///   WarAndPeace 25.0% -- FIRES (needs ob=19+)
///   JSON_2MB    24.3% -- FIRES (marginal, wider ob doesn't hurt)
///   JSON_100KB   4.4% -- silent
///   lcet10       2.8% -- silent
///   alice29      0.7% -- silent
///   Source_1MB   0.7% -- silent
///   ptt5         0.0% -- silent
const CEILING_UPPER_HALF_THRESHOLD: f64 = 0.20;

const HASH_SIZE:              usize = 1 << 16;
const HASH_MASK:              usize = HASH_SIZE - 1;
const LAZY_SHORT_LEN:         usize = 6;

const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

// -- Discovery scan constants -------------------------------------------------

const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;

// -- Tiered chain limit -------------------------------------------------------

/// Cache-aware tiered chain depth limit for full-quality scans.
///
///   ob <= 16   window <=  64KB   prev <=   64KB   L2-resident  -> 256
///   ob 17-18   window <= 256KB   prev <=  256KB   L3-resident  -> 512
///   ob 19-20   window <=   1MB   prev <=    1MB   L3-tight     -> 256
///   ob >= 21   window >=   2MB   prev >=    2MB   DRAM         -> 128
pub fn compute_chain_limit(_input_len: usize, offset_bits: u32) -> usize {
    match offset_bits {
        0..=16 => 256,
        17..=18 => 512,
        19..=20 => 256,
        _ => 128,
    }
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

// -- Core scanner -------------------------------------------------------------

pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off      = max_offset(offset_bits);
    let max_len      = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);
    let chain_limit  = compute_chain_limit(input.len(), offset_bits);

    let n           = input.len();
    let window_size = max_off.min(n).max(1);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::with_capacity(n / 2 + 1);
    let mut i      = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) =
            find_match(input, i, h, &head, &prev, max_off, max_len, window_size, chain_limit);

        let backref_worthwhile = best_len >= 2
            && backref_bits < (best_len as u32 * LIT_TOTAL_BITS);

        if backref_worthwhile {
            let lazy = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, len1) = find_match(
                    input, i + 1, h1, &head, &prev, max_off, max_len, window_size, chain_limit,
                );
                if len1 > best_len {
                    true
                } else if best_len <= LAZY_SHORT_LEN && i + 2 < n {
                    let h2 = hash3(input, i + 2);
                    let (_, len2) = find_match(
                        input, i + 2, h2, &head, &prev, max_off, max_len, window_size, chain_limit,
                    );
                    len2 > best_len + 2
                } else {
                    false
                }
            } else {
                false
            };

            if lazy {
                prev[i % window_size] = head[h];
                head[h] = i as u32;
                tokens.push(Token::Lit { byte: input[i] });
                i += 1;
            } else {
                for k in 0..best_len {
                    if i + k + 2 < n {
                        let hk = hash3(input, i + k);
                        prev[(i + k) % window_size] = head[hk];
                        head[hk] = (i + k) as u32;
                    }
                }
                tokens.push(Token::Backref {
                    offset: best_offset as u32,
                    length: best_len as u32,
                });
                i += best_len;
            }
        } else {
            prev[i % window_size] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Lit { byte: input[i] });
            i += 1;
        }
    }

    tokens.push(Token::End);
    tokens
}

pub fn scan_discover(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off     = max_offset(offset_bits).min(DISCOVER_MAX_PREV_SIZE);
    let max_len     = max_length(length_bits);
    let n           = input.len();
    let window_size = max_off.min(n).max(1);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::with_capacity(n / 2 + 1);
    let mut i      = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) = find_match(
            input, i, h, &head, &prev,
            max_off, max_len, window_size,
            DISCOVER_CHAIN_LIMIT,
        );

        if best_len >= 2 {
            for k in 0..best_len {
                if i + k + 2 < n {
                    let hk = hash3(input, i + k);
                    prev[(i + k) % window_size] = head[hk];
                    head[hk] = (i + k) as u32;
                }
            }
            tokens.push(Token::Backref {
                offset: best_offset as u32,
                length: best_len as u32,
            });
            i += best_len;
        } else {
            prev[i % window_size] = head[h];
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
    window_size: usize,
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

        cur   = prev[j % window_size];
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

/// Returns true when the Phase B scan window is insufficient and Phase C
/// should run to find a better ob/lb.
///
/// Two independent conditions — either fires Phase C:
///
/// Peak check: any single Backref has offset or length >= 90% of field max.
///   Catches files where the window is genuinely too small and individual
///   backrefs are bumping against the ceiling.
///
/// Upper-half check: >= 20% of all Backrefs have offset in the upper half
///   of the current window. Catches files like WarAndPeace (25% upper-half)
///   where the window is heavily used across its full depth but no single
///   backref hits 90% -- a wider window would still improve ratio.
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
            // Peak check fires immediately
            if *offset >= ob_ceil || *length >= lb_ceil {
                println!("  ceiling check: peak hit offset={} (ceil={}) or length={} (ceil={})",
                    offset, ob_ceil, length, lb_ceil);
                return true;
            }
            total_br += 1;
            if *offset > half_off { upper_half += 1; }
        }
    }

    // Upper-half distribution check
    if total_br > 0 {
        let upper_frac = upper_half as f64 / total_br as f64;
        if upper_frac >= CEILING_UPPER_HALF_THRESHOLD {
            println!(
                "  ceiling check: upper-half saturated {:.1}% >= {:.0}% threshold — Phase C needed",
                upper_frac * 100.0,
                CEILING_UPPER_HALF_THRESHOLD * 100.0,
            );
            return true;
        }
        println!(
            "  ceiling check: ok — upper-half {:.1}% < {:.0}% threshold",
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
