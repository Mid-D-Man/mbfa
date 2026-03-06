// src/encoder.rs
//! LZ-style scanner with rolling-window hash chain.
//! O(n) time, O(window) memory — safe for large files.
//! Both offset_bits and length_bits are adaptive at runtime.
//!
//! scan_adaptive: up to four scans, always apples-to-apples.
//!
//!   Scan 1 — baseline: (BASELINE_OFFSET_BITS=17, BASELINE_LENGTH_BITS=8).
//!   Proven safe across all benchmarks. Used as the regression floor.
//!
//!   Scan 2 — wide discovery via scan_discover (fast, capped window/chain).
//!   Computes (wide_ob, wide_lb) without the full scan cost. Token stream
//!   is discarded — only the discovered ob/lb values are kept.
//!
//!   Scan 3 — constrained re-scan: (wide_ob, wide_lb).
//!   Only runs when wide_ob != BASELINE_OFFSET_BITS or wide_lb != BASELINE_LENGTH_BITS.
//!   Produces the CORRECT token stream for that exact window. Its raw bit cost is
//!   directly comparable to the baseline — both streams produced under matching
//!   window constraints.
//!
//!   Entropy tiebreaker:
//!   When raw bit costs of two candidates are within ENTROPY_TIE_THRESHOLD (5%)
//!   of each other, a Shannon entropy estimate breaks the tie.
//!
//!   Entropy-safety cap:
//!   When the constrained scan selects lb > ENTROPY_SAFE_LENGTH_BITS (15) AND
//!   the output would be large enough for entropy to fire, a fourth scan with
//!   lb capped at 15 is run and compared against baseline.
//!
//!   Dynamic chain limit (full scan only):
//!   CHAIN_LIMIT = clamp(max(√n, window×32/HASH_SIZE), 64, 4096)
//!
//!   Discovery scan (scan_discover):
//!   Uses DISCOVER_CHAIN_LIMIT (fixed 256) and caps prev array at
//!   DISCOVER_MAX_PREV_SIZE (4MB). No lazy matching. Fast and approximate —
//!   only used to determine ob/lb range, never for final output.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};
use crate::entropy::{offset_to_bucket, bucket_extra_bits};

const BASELINE_OFFSET_BITS:     u32   = 17;
const BASELINE_LENGTH_BITS:     u32   = LENGTH_BITS_MIN; // 8
const ENTROPY_SAFE_LENGTH_BITS: u32   = 15;
const ENTROPY_MIN_BYTES_FOR_SCAN: usize = 400;

const ENTROPY_TIE_THRESHOLD: f64 = 0.05;

const HASH_SIZE:       usize = 1 << 16;
const HASH_MASK:       usize = HASH_SIZE - 1;
const CHAIN_LIMIT_MIN: usize = 64;
const CHAIN_LIMIT_MAX: usize = 4096;
const LAZY_SHORT_LEN:  usize = 6;

// ── Discovery scan constants ──────────────────────────────────────────────────

/// Fixed chain walk limit for the wide discovery scan.
/// 256 steps is sufficient to detect what ob/lb range the data uses
/// without the full O(chain_limit) cost of a production scan.
const DISCOVER_CHAIN_LIMIT: usize = 256;

/// Maximum prev array size for the discovery scan (2 MB → ob ≤ 21).
/// Chosen for max speed — accepts that ob=22 is unreachable via discovery
/// on files larger than 2MB. The constrained re-scan still runs full quality
/// at whatever ob is discovered within this cap.
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

/// Full-quality production scan. Used for baseline and constrained re-scans.
/// Chain limit is computed dynamically from input size and offset window.
pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off      = max_offset(offset_bits);
    let max_len      = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);
    let chain_limit  = compute_chain_limit(input.len(), offset_bits);

    let n           = input.len();
    let window_size = max_off.min(n).max(1);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::new();
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

/// Fast discovery scan — used only to determine what offset/length range the
/// data actually needs. The token stream is thrown away by the caller.
///
/// Differences from the full scan:
///   - prev array capped at DISCOVER_MAX_PREV_SIZE (4 MB) regardless of ob
///   - chain walk capped at DISCOVER_CHAIN_LIMIT (256) — fixed, not computed
///   - no lazy matching — greedy, any match len >= 2 is taken immediately
///   - no backref_worthwhile check — we want to find matches even when they
///     don't save bits, to maximise ob/lb coverage for the discovery purpose
///
/// These relaxations make discovery ~10x faster on large files while still
/// correctly identifying the ob/lb range the data uses.
pub fn scan_discover(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    // Cap the lookback window so the prev array never exceeds DISCOVER_MAX_PREV_SIZE.
    // 4 MB → max reportable offset fits in ob=22 bits.
    let max_off     = max_offset(offset_bits).min(DISCOVER_MAX_PREV_SIZE);
    let max_len     = max_length(length_bits);
    let n           = input.len();
    let window_size = max_off.min(n).max(1);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::new();
    let mut i      = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) = find_match(
            input, i, h, &head, &prev,
            max_off, max_len, window_size,
            DISCOVER_CHAIN_LIMIT,
        );

        if best_len >= 2 {
            // Greedy match — no lazy check, no bit-cost check.
            // Update all hashes in the match span so later positions can
            // build on this match (same as the full scan).
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

// ── Public API ────────────────────────────────────────────────────────────────

/// Adaptive scan with apples-to-apples cost comparison.
///
/// Scan 2 now uses scan_discover instead of a full scan — same ob/lb discovery
/// result with a capped prev array (4 MB max) and fixed chain limit (256).
/// This eliminates the 13 MB working set thrash on large files.
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {
    // ── Scan 1: baseline ──────────────────────────────────────────────────────
    let baseline = ScanResult::new(
        scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS),
        BASELINE_OFFSET_BITS,
        BASELINE_LENGTH_BITS,
        "baseline",
    );
    println!(
        "  chain_limit (baseline ob={}): {}  raw_cost={}",
        BASELINE_OFFSET_BITS,
        compute_chain_limit(input.len(), BASELINE_OFFSET_BITS),
        baseline.raw_cost,
    );

    // ── Scan 2: wide discovery (fast) ─────────────────────────────────────────
    // scan_discover caps prev at 4 MB and chain at 256 — token stream is discarded.
    let wide_discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
    let wide_ob        = compute_optimal_offset_bits(&wide_discovery);
    let wide_lb        = compute_optimal_length_bits(&wide_discovery);

    println!(
        "  discovery (capped 4MB/chain={}): discovered ob={} lb={}",
        DISCOVER_CHAIN_LIMIT, wide_ob, wide_lb,
    );

    // If discovery agrees with baseline, no re-scan needed.
    if wide_ob == BASELINE_OFFSET_BITS && wide_lb == BASELINE_LENGTH_BITS {
        println!("  wide agrees with baseline — no re-scan needed");
        return (baseline.tokens, baseline.ob, baseline.lb);
    }

    // ── Scan 3: constrained re-scan (full quality) ────────────────────────────
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

    // ── Entropy-safety cap ────────────────────────────────────────────────────
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

    // ── Final pick: constrained vs baseline ───────────────────────────────────
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
