// src/encoder.rs
//! LZ-style scanner with rolling-window hash chain.
//! O(n) time, O(window) memory — safe for large files.
//! Both offset_bits and length_bits are adaptive at runtime.
//!
//! scan_adaptive: Phase A/B/C fingerprint pipeline.
//!
//!   Phase A/B — fingerprint + single scan (LARGE FILES ONLY, > 1 MB):
//!     Phase A fingerprint_predict classifies the input and predicts (ob, lb).
//!     Phase B runs one full-quality scan at the predicted parameters, then
//!     ceiling_saturated() checks whether the window was sufficient.
//!       Not saturated → return immediately (1 scan total).
//!       Saturated, pred matches baseline → reuse tokens as baseline in Phase C,
//!         skip the baseline re-scan (saves 1 scan).
//!       Saturated, pred differs from baseline → fall through to Phase C fresh.
//!
//!     Phase A/B is skipped entirely for files <= PARALLEL_SCAN_THRESHOLD (1 MB).
//!     On small/medium files rayon::join(baseline, discovery) is already optimal —
//!     Phase B runs sequentially before discovery, killing the parallel speedup
//!     whenever ceiling fires (which it does on most medium prose/source/binary
//!     files with diverse offsets).
//!
//!   Phase A fingerprint rules (applied only when input > 1 MB):
//!     Rule 1: entropy < 2.0 (highly repetitive) → defer to Phase C.
//!             lb will saturate at lb=8; ceiling check fires immediately anyway.
//!             Skipping Phase B saves a wasted full-quality scan.
//!     Rule 2: size < 32 KB → predict ob = ceil(log2(size)), lb = 8.
//!             Not reached in practice (gated behind > 1 MB check), kept for
//!             correctness if threshold changes.
//!     Rule 3: default → predict baseline (ob=17, lb=8).
//!
//!   Phase C — discovery path (original logic, unchanged):
//!     Run scan_discover() → compute wide_ob/wide_lb → constrained re-scan if
//!     different → entropy safety cap check → pick best result.
//!     Baseline and discovery run IN PARALLEL via rayon::join when
//!     input.len() <= PARALLEL_SCAN_THRESHOLD (1 MB).
//!
//!   Dynamic chain limit (full scan only):
//!   CHAIN_LIMIT = clamp(max(√n, window×32/HASH_SIZE), 64, 4096)
//!
//!   Discovery scan (scan_discover):
//!   Uses DISCOVER_CHAIN_LIMIT (fixed 256) and caps prev array at
//!   DISCOVER_MAX_PREV_SIZE (2MB). No lazy matching. Fast and approximate —
//!   only used to determine ob/lb range, never for final output.

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

// ── Fingerprint constants ─────────────────────────────────────────────────────

/// Below this entropy, data is highly repetitive — lb will saturate in Phase B
/// ceiling check anyway. Skip Phase B to avoid a wasted full-quality scan.
const FINGERPRINT_ENTROPY_REPETITIVE: f64   = 2.0;

/// Below this size, predict ob = minimum bits to span the whole file.
const FINGERPRINT_SMALL_FILE_BYTES:   usize = 32768;

/// Fraction of a field's maximum value at which a backref is considered
/// saturated. Conservative (90%): false trigger costs 1 extra scan (cheap);
/// missed trigger costs ratio quality (expensive).
const CEILING_SATURATION_THRESHOLD:   f64   = 0.9;

const HASH_SIZE:       usize = 1 << 16;
const HASH_MASK:       usize = HASH_SIZE - 1;
const CHAIN_LIMIT_MIN: usize = 64;
const CHAIN_LIMIT_MAX: usize = 4096;
const LAZY_SHORT_LEN:  usize = 6;

/// Input size above which baseline and discovery scans run sequentially,
/// AND above which Phase A/B fingerprint optimisation is enabled.
///
/// Below this threshold rayon::join(baseline, discovery) is already optimal.
/// Phase B would run sequentially before discovery, killing the parallelism
/// whenever ceiling fires — which it does on most medium prose/source/binary
/// files with diverse offsets. Measured regression: +18-29% on alice29,
/// asyoulik, kennedy when Phase B was applied unconditionally.
///
/// Above this threshold scans were already sequential. Phase B reuses the
/// baseline scan for Phase C when ceiling fires, saving one full scan.
const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

// ── Discovery scan constants ──────────────────────────────────────────────────

const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;

// ── Dynamic chain limit ───────────────────────────────────────────────────────

pub fn compute_chain_limit(input_len: usize, offset_bits: u32) -> usize {
    let window       = max_offset(offset_bits);
    let effective    = input_len.min(window);
    let window_based = effective.saturating_mul(32) / HASH_SIZE;
    let sqrt_based   = isqrt(input_len);
    let raw          = window_based.max(sqrt_based);
    raw.clamp(CHAIN_LIMIT_MIN, CHAIN_LIMIT_MAX)
}

#[inline]
fn isqrt(n: usize) -> usize {
    if n == 0 { return 0; }
    let mut s = (n as f64).sqrt() as usize;
    while s > 0 && s.saturating_mul(s) > n { s -= 1; }
    while (s + 1).saturating_mul(s + 1) <= n { s += 1; }
    s
}

// ── Hash ──────────────────────────────────────────────────────────────────────

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos]     as usize).wrapping_mul(2_654_435_761)
        ^   (input[pos + 1] as usize).wrapping_mul(2_246_822_519)
        ^   (input[pos + 2] as usize).wrapping_mul(3_266_489_917);
    v & HASH_MASK
}

// ── Core scanner ──────────────────────────────────────────────────────────────

/// Full-quality production scan. Chain limit computed dynamically.
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

/// Fast discovery scan — used only to determine ob/lb range. Token stream
/// is discarded by the caller.
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

// ── Cost helpers ──────────────────────────────────────────────────────────────

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

// ── Scan result wrapper ───────────────────────────────────────────────────────

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
                "  entropy tiebreaker: {} est={:.0}b vs {} est={:.0}b → {} wins",
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

// ── Fingerprint helpers ───────────────────────────────────────────────────────

/// Sample Shannon entropy from the first 8 KB of data.
/// Private to this module — intentionally not shared with lib.rs::sample_entropy.
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

/// Predict (offset_bits, length_bits) for a Phase B single scan, or return
/// None to defer directly to Phase C.
///
/// Only called for inputs > PARALLEL_SCAN_THRESHOLD.
///
/// Rule 1: entropy < 2.0 → defer to Phase C (lb saturates immediately anyway).
/// Rule 2: size < 32 KB  → predict ob = ceil(log2(size)), lb = 8.
/// Rule 3: default       → predict baseline (ob=17, lb=8).
fn fingerprint_predict(data: &[u8]) -> Option<(u32, u32)> {
    let ent = sample_entropy_fingerprint(data);
    println!("  fingerprint: entropy={:.2} size={}", ent, data.len());

    // Rule 1: highly repetitive
    if ent < FINGERPRINT_ENTROPY_REPETITIVE {
        println!("  fingerprint Rule 1: repetitive → defer to Phase C");
        return None;
    }

    // Rule 2: small file (not reached in practice above 1MB gate, kept for safety)
    if data.len() < FINGERPRINT_SMALL_FILE_BYTES {
        let ob = if data.is_empty() {
            OFFSET_BITS_MIN
        } else {
            let bits = (usize::BITS - data.len().leading_zeros()) as u32;
            bits.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
        };
        println!("  fingerprint Rule 2: small file → ob={} lb={}", ob, LENGTH_BITS_MIN);
        return Some((ob, LENGTH_BITS_MIN));
    }

    // Rule 3: default
    println!("  fingerprint Rule 3: default → ob={} lb={}", BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    Some((BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS))
}

/// Returns true when any Backref has an offset or length at or above
/// CEILING_SATURATION_THRESHOLD (90%) of the maximum encodable value.
fn ceiling_saturated(tokens: &[Token], ob: u32, lb: u32) -> bool {
    let ob_ceil = ((max_offset(ob) as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    let lb_ceil = ((max_length(lb) as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    tokens.iter().any(|t| {
        if let Token::Backref { offset, length } = t {
            *offset >= ob_ceil || *length >= lb_ceil
        } else {
            false
        }
    })
}

// ── Phase C inner logic (shared by both paths) ────────────────────────────────

/// Given a precomputed baseline ScanResult and a discovery token stream,
/// run the constrained re-scan + entropy safety cap + final pick.
/// Called both from the large-file Phase B fallthrough and the small-file
/// parallel path.
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
        println!("  wide agrees with baseline — no re-scan needed");
        return (baseline.tokens, baseline.ob, baseline.lb);
    }

    // Constrained re-scan
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

    // Entropy-safety cap
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
                 — capped raw_cost={} chain_limit={}",
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

    // Final pick
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

// ── Public API ────────────────────────────────────────────────────────────────

/// Adaptive scan — Phase A/B/C pipeline.
///
/// Phase A/B (large files only, > PARALLEL_SCAN_THRESHOLD):
///   Fingerprint predicts (ob, lb). Single full-quality scan. Ceiling check
///   gates whether Phase B result is returned or Phase C runs.
///   When pred matches baseline and ceiling fires, Phase B tokens are reused
///   as the baseline in Phase C — saving one full baseline scan.
///
/// Phase C (all files, or large files where Phase B fell through):
///   discovery + constrained re-scan. Parallel for small/medium files.
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {

    // ── Phase A/B: large files only ───────────────────────────────────────────
    if input.len() > PARALLEL_SCAN_THRESHOLD {
        match fingerprint_predict(input) {
            None => {
                // Rule 1: highly repetitive — fall through to Phase C below.
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
                    println!("  ceiling check: ok — Phase B wins (1 scan total)");
                    return (result.tokens, result.ob, result.lb);
                }

                println!("  ceiling check: saturated → falling through to Phase C");

                // If Phase B used baseline params, reuse its tokens as baseline
                // in Phase C — saves the baseline re-scan entirely.
                if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                    let baseline  = ScanResult { label: "baseline", ..result };
                    let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                    return phase_c_from_baseline(baseline, discovery, input);
                }

                // Phase B used non-baseline params (Rule 2 on a large file,
                // edge case). Fall through to fresh Phase C below.
            }
        }
    }

    // ── Phase C: parallel for small/medium, sequential for large fallthrough ──
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
        // Large file where Phase B deferred (Rule 1) or used non-baseline params.
        let bt   = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
        let disc = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
        (
            ScanResult::new(bt, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, "baseline"),
            disc,
        )
    };

    phase_c_from_baseline(baseline, wide_discovery, input)
}
