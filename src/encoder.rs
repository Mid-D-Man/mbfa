// src/encoder.rs
//! LZ-style scanner with rolling-window hash chain.
//! O(n) time, O(window) memory — safe for large files.
//! Both offset_bits and length_bits are adaptive at runtime.
//!
//! scan_adaptive runs two scans:
//!   1. Baseline at (17, 8) — matches previous proven behaviour
//!   2. Wide at (OFFSET_BITS_MAX, LENGTH_BITS_MAX) — finds minimum fields
//!      that cover actual values used, costs the stream
//!  Whichever scan produces lower total bit cost wins.
//!  This guarantees zero regression vs prior builds on any file.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};

// Baseline widths — proven safe across all Canterbury + extended benchmarks.
// The wide scan only wins when it demonstrably beats these.
const BASELINE_OFFSET_BITS: u32 = 17;
const BASELINE_LENGTH_BITS: u32 = LENGTH_BITS_MIN; // 8

const HASH_SIZE:   usize = 1 << 16;
const HASH_MASK:   usize = HASH_SIZE - 1;
const CHAIN_LIMIT: usize = 1024;

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos]     as usize).wrapping_mul(2_654_435_761)
        ^   (input[pos + 1] as usize).wrapping_mul(2_246_822_519)
        ^   (input[pos + 2] as usize).wrapping_mul(3_266_489_917);
    v & HASH_MASK
}

/// Scan input using a rolling window of `(1 << offset_bits) - 1` bytes.
/// Memory usage is O(min(window_size, n)) — safe for any input size.
///
/// Uses one-step lazy matching: before committing to a BACKREF at i,
/// checks if i+1 yields a strictly longer match. Recovers ~2-5pp on text.
pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off = max_offset(offset_bits);
    let max_len = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);

    let n = input.len();

    // Cap window at actual input size — avoids giant allocations on small files
    // when called with large offset_bits (e.g. wide scan on a 4KB file).
    let window_size = max_off.min(n).max(1);

    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; window_size];
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) =
            find_match(input, i, h, &head, &prev, max_off, max_len, window_size);

        let backref_worthwhile = best_len >= 2
            && backref_bits < (best_len as u32 * LIT_TOTAL_BITS);

        if backref_worthwhile {
            // Lazy matching: peek one position ahead before committing.
            let lazy_better = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, lazy_len) =
                    find_match(input, i + 1, h1, &head, &prev, max_off, max_len, window_size);
                lazy_len > best_len
            } else {
                false
            };

            if lazy_better {
                // Emit literal, insert i into chain, let next iter find longer match.
                prev[i % window_size] = head[h];
                head[h] = i as u32;
                tokens.push(Token::Lit { byte: input[i] });
                i += 1;
            } else {
                // Commit to match — update chain for all covered positions.
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
) -> (usize, usize) {
    let n = input.len();
    let mut best_offset = 0;
    let mut best_len    = 0;
    let mut steps       = 0;

    let mut cur = head[h];
    while cur != u32::MAX && steps < CHAIN_LIMIT {
        let j = cur as usize;
        // Must be strictly before i and within the lookback window.
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

/// Two-scan adaptive selection:
///
/// 1. Baseline scan at (BASELINE_OFFSET_BITS=17, BASELINE_LENGTH_BITS=8).
///    This matches the previously proven behaviour — guaranteed no regression.
///
/// 2. Wide scan at (OFFSET_BITS_MAX, LENGTH_BITS_MAX). Computes the minimum
///    (offset_bits, length_bits) that covers all values actually used, then
///    costs the stream at those minimum widths.
///
/// Returns whichever scan produces lower total token stream bit cost.
/// If costs are equal, prefer baseline (cheaper to encode header, proven safe).
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {
    // ── Baseline scan ─────────────────────────────────────────────────────────
    let baseline_tokens = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    let baseline_cost   = stream_bit_cost(&baseline_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);

    // ── Wide scan ─────────────────────────────────────────────────────────────
    let wide_tokens = scan(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
    let wide_ob     = compute_optimal_offset_bits(&wide_tokens);
    let wide_lb     = compute_optimal_length_bits(&wide_tokens);
    let wide_cost   = stream_bit_cost(&wide_tokens, wide_ob, wide_lb);

    // ── Pick winner ───────────────────────────────────────────────────────────
    // Strict less-than: baseline wins ties to avoid unnecessary field widening.
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
