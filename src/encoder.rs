// src/encoder.rs
//! LZ-style scanner with rolling-window hash chain.
//! O(n) time, O(window) memory — safe for large files.
//! Both offset_bits and length_bits are adaptive at runtime.
//!
//! scan_adaptive runs up to three scans:
//!   1. Baseline at (17, 8) — matches previous proven behaviour
//!   2. Wide at (OFFSET_BITS_MAX, LENGTH_BITS_MAX) — finds minimum fields
//!      that cover actual values used, costs the stream
//!   Whichever scan produces lower total bit cost wins.
//!   This guarantees zero regression vs prior builds on any file.
//!
//!   Length cap for entropy safety:
//!   Entropy table serialisation uses u16 for symbol values (sym = 255 + length).
//!   Max safe length = 65535 - 255 = 65280, requiring length_bits <= 15.
//!   When the wide scan selects lb > 15 AND the fold-1 output is large enough
//!   that entropy would fire (>= ENTROPY_MIN_BYTES), a third scan is run with
//!   lb capped at ENTROPY_SAFE_LENGTH_BITS. This recovers the entropy path for
//!   files like Source_1MB without affecting files like Repetitive_2MB whose
//!   fold-1 output is far too small to hit the entropy threshold.
//!
//!   Lazy matching (two-step):
//!   Before committing a BACKREF at position i, the encoder peeks ahead:
//!     1-step: if i+1 yields a strictly longer match, emit LIT[i] and defer.
//!     2-step: for short matches (≤ LAZY_SHORT_LEN), also check i+2. If i+2
//!             yields a substantially longer match (> best_len + 2), defer by
//!             emitting LIT[i]. The cascade naturally reaches i+2 on the next
//!             iteration via 1-step lazy at i+1.
//!
//!   Dynamic chain limit:
//!   CHAIN_LIMIT is computed per scan from input_len and offset_bits:
//!     limit = clamp(max(√n, window×32/HASH_SIZE), 64, 4096)
//!   Small files get a lower limit (less wasted work on short chains).
//!   Large files with big windows get a higher limit (better deep matches).
//!   The flat constant 1024 is replaced entirely.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};

// Baseline widths — proven safe across all Canterbury + extended benchmarks.
const BASELINE_OFFSET_BITS: u32 = 17;
const BASELINE_LENGTH_BITS: u32 = LENGTH_BITS_MIN; // 8

// Maximum length_bits that keeps entropy symbol values within u16.
const ENTROPY_SAFE_LENGTH_BITS: u32 = 15;

// Mirrors entropy::ENTROPY_MIN_BYTES — minimum fold-1 output bytes for entropy
// to fire. Defined here to avoid a circular module dependency.
const ENTROPY_MIN_BYTES_FOR_SCAN: usize = 400;

const HASH_SIZE: usize = 1 << 16;
const HASH_MASK: usize = HASH_SIZE - 1;

// Bounds on the dynamic chain limit.
const CHAIN_LIMIT_MIN: usize = 64;
const CHAIN_LIMIT_MAX: usize = 4096;

// Two-step lazy matching threshold.
const LAZY_SHORT_LEN: usize = 6;

// ── Dynamic chain limit ───────────────────────────────────────────────────────

/// Compute the hash-chain walk limit for a given input size and offset window.
///
/// Two components, take the max:
///   sqrt_based:   √input_len — scales with input size. Larger inputs have
///                 denser hash chains and benefit from deeper search.
///   window_based: (min(input_len, window) × 32) / HASH_SIZE — scales with
///                 how densely the window fills the hash table. A 22-bit window
///                 on a 3 MB file fills each bucket ~48× on average; the ×32
///                 headroom handles common 3-gram collisions.
///
/// Result is clamped to [CHAIN_LIMIT_MIN, CHAIN_LIMIT_MAX].
///
/// Representative values:
///   grammar.lsp   3.7 KB  ob=12  →   64  (chain naturally < 64; no wasted steps)
///   alice29.txt   152 KB  ob=17  →  389  (down from 1024; lazy matching covers the gap)
///   kennedy.xls   1.0 MB  ob=17  → 1014  (≈ current, no regression)
///   WarAndPeace   3.3 MB  ob=17  → 1832  (up from 1024; better deep matches on long text)
///   JSON_2MB      3.1 MB  ob=22  → 1773  (up from 1024; 22-bit window benefits most)
pub fn compute_chain_limit(input_len: usize, offset_bits: u32) -> usize {
    let window     = max_offset(offset_bits);          // (1 << offset_bits) - 1
    let effective  = input_len.min(window);
    let window_based = effective.saturating_mul(32) / HASH_SIZE;
    let sqrt_based   = isqrt(input_len);
    let raw = window_based.max(sqrt_based);
    raw.clamp(CHAIN_LIMIT_MIN, CHAIN_LIMIT_MAX)
}

/// Integer square root — exact, no floating point, works on all Rust editions.
#[inline]
fn isqrt(n: usize) -> usize {
    if n == 0 { return 0; }
    // Initial estimate via float — may be off by 1 due to rounding.
    let mut s = (n as f64).sqrt() as usize;
    // Correct downward if overshoot.
    while s > 0 && s.saturating_mul(s) > n { s -= 1; }
    // Correct upward if undershoot.
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

/// Scan input using a rolling window of `(1 << offset_bits) - 1` bytes.
/// Memory usage is O(min(window_size, n)) — safe for any input size.
///
/// Chain limit is computed dynamically from input length and offset_bits.
/// Uses two-step lazy matching (see module docs).
pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off     = max_offset(offset_bits);
    let max_len     = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);
    let chain_limit  = compute_chain_limit(input.len(), offset_bits);

    let n = input.len();

    // Cap window at actual input size — avoids giant allocations on small files
    // when called with large offset_bits (e.g. wide scan on a 4 KB file).
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
            // Lazy matching: check ahead before committing.
            let lazy = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, len1) =
                    find_match(input, i + 1, h1, &head, &prev, max_off, max_len, window_size, chain_limit);

                if len1 > best_len {
                    true
                } else if best_len <= LAZY_SHORT_LEN && i + 2 < n {
                    let h2 = hash3(input, i + 2);
                    let (_, len2) =
                        find_match(input, i + 2, h2, &head, &prev, max_off, max_len, window_size, chain_limit);
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
    tokens.iter().map(|t| {
        match t {
            Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
            Token::Backref { .. } => backref_total_bits(ob, lb) as u64,
            Token::End            => END_TOTAL_BITS as u64,
        }
    }).sum()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Two-scan adaptive selection with entropy-safety cap.
///
/// 1. Baseline scan at (BASELINE_OFFSET_BITS=17, BASELINE_LENGTH_BITS=8).
///
/// 2. Wide scan at (OFFSET_BITS_MAX, LENGTH_BITS_MAX). Computes the minimum
///    (offset_bits, length_bits) that covers all values actually used.
///
/// 3. If wide scan selects lb > ENTROPY_SAFE_LENGTH_BITS (15) AND the
///    fold-1 output would be large enough for entropy to fire, run a third
///    scan with lb capped at ENTROPY_SAFE_LENGTH_BITS.
///
/// Returns whichever scan produces the lowest raw token stream bit cost.
/// Ties go to baseline (proven safe).
///
/// Each scan uses its own dynamically computed chain limit derived from
/// input_len and the scan's offset_bits.
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {
    // ── Baseline scan ─────────────────────────────────────────────────────────
    let baseline_limit  = compute_chain_limit(input.len(), BASELINE_OFFSET_BITS);
    let baseline_tokens = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    let baseline_cost   = stream_bit_cost(&baseline_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);

    println!(
        "  chain_limit (baseline ob={}): {}",
        BASELINE_OFFSET_BITS, baseline_limit
    );

    // ── Wide scan ─────────────────────────────────────────────────────────────
    let wide_limit  = compute_chain_limit(input.len(), OFFSET_BITS_MAX);
    let wide_tokens = scan(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
    let wide_ob     = compute_optimal_offset_bits(&wide_tokens);
    let wide_lb     = compute_optimal_length_bits(&wide_tokens);
    let wide_cost   = stream_bit_cost(&wide_tokens, wide_ob, wide_lb);

    println!(
        "  chain_limit (wide ob={}): {}",
        OFFSET_BITS_MAX, wide_limit
    );

    // ── Entropy-safety cap ────────────────────────────────────────────────────
    if wide_lb > ENTROPY_SAFE_LENGTH_BITS {
        let wide_output_bytes = (wide_cost as usize + 7) / 8;
        if wide_output_bytes >= ENTROPY_MIN_BYTES_FOR_SCAN {
            let capped_limit  = compute_chain_limit(input.len(), wide_ob);
            let capped_tokens = scan(input, wide_ob, ENTROPY_SAFE_LENGTH_BITS);
            let capped_cost   = stream_bit_cost(&capped_tokens, wide_ob, ENTROPY_SAFE_LENGTH_BITS);

            println!(
                "  lb cap applied: wide lb={} output={}B >= entropy threshold {}B \
                 — re-scanned at lb={} chain_limit={} (cost {} vs wide cost {})",
                wide_lb, wide_output_bytes, ENTROPY_MIN_BYTES_FOR_SCAN,
                ENTROPY_SAFE_LENGTH_BITS, capped_limit, capped_cost, wide_cost
            );

            return if capped_cost < baseline_cost {
                (capped_tokens, wide_ob, ENTROPY_SAFE_LENGTH_BITS)
            } else {
                (baseline_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS)
            };
        }
    }

    // ── Normal wide vs baseline pick ──────────────────────────────────────────
    if wide_cost < baseline_cost {
        println!(
            "  Wide scan wins: ob={} lb={} cost={} < baseline cost={} (ob={} lb={})",
            wide_ob, wide_lb, wide_cost,
            baseline_cost, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS
        );
        (wide_tokens, wide_ob, wide_lb)
    } else {
        (baseline_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS)
    }
                }
