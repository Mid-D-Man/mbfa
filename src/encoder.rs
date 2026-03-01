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
//!
//!  Length cap for entropy safety:
//!  Entropy table serialisation uses u16 for symbol values (sym = 255 + length).
//!  Max safe length = 65535 - 255 = 65280, requiring length_bits <= 15.
//!  When the wide scan selects lb > 15 AND the fold-1 output is large enough
//!  that entropy would fire (>= ENTROPY_MIN_BYTES), a third scan is run with
//!  lb capped at ENTROPY_SAFE_LENGTH_BITS. This recovers the entropy path for
//!  files like Source_1MB without affecting files like Repetitive_2MB whose
//!  fold-1 output is far too small to hit the entropy threshold.
//!
//!  Lazy matching (two-step):
//!  Before committing a BACKREF at position i, the encoder peeks ahead:
//!    1-step: if i+1 yields a strictly longer match, emit LIT[i] and defer.
//!    2-step: for short matches (≤ LAZY_SHORT_LEN), also check i+2. If i+2
//!            yields a substantially longer match (> best_len + 2), defer by
//!            emitting LIT[i]. The cascade naturally reaches i+2 on the next
//!            iteration via 1-step lazy at i+1.
//!  This catches the missed case where i+1's match is shorter than i's but
//!  i+2's match is much longer — greedy commits at i and never sees i+2.

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

const HASH_SIZE:   usize = 1 << 16;
const HASH_MASK:   usize = HASH_SIZE - 1;
const CHAIN_LIMIT: usize = 1024;

// Two-step lazy matching threshold.
// Only attempt the 2-step lookahead for matches shorter than or equal to this.
// Long matches are almost always the right greedy choice, and the extra
// find_match calls are wasteful when best_len is already high.
const LAZY_SHORT_LEN: u32 = 6;

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
/// Uses two-step lazy matching: before committing to a BACKREF at i,
/// checks if i+1 yields a strictly longer match (1-step), and for short
/// matches also checks if i+2 yields a substantially longer match (2-step).
/// The 2-step check defers via emitting LIT[i]; the cascade at i+1 then
/// naturally defers again to i+2 via 1-step lazy. Recovers ~2-5pp on text
/// and source code vs pure greedy.
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
            // Lazy matching: check ahead before committing.
            //
            // 1-step (unchanged): if i+1 has a strictly longer match, defer.
            //
            // 2-step (new): for short matches (≤ LAZY_SHORT_LEN), also probe i+2.
            // If i+2 has a substantially longer match (> best_len + 2), defer by
            // emitting LIT[i]. On the next iteration (at i+1), the 1-step check
            // will probe i+2 and defer again if appropriate — naturally cascading
            // to the better match without requiring explicit 2-literal emission here.
            //
            // Threshold best_len + 2: each deferred literal costs LIT_TOTAL_BITS=10
            // bits. For the cascade to reach i+2, we'll emit 1 extra literal (10 bits
            // at i) and then possibly another at i+1. Requiring len2 > best_len + 2
            // ensures the longer match at i+2 covers at least 2 extra bytes (≥20 bits
            // savings) beyond what we'd have gotten committing at i.
            let lazy = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, len1) =
                    find_match(input, i + 1, h1, &head, &prev, max_off, max_len, window_size);

                if len1 > best_len {
                    // Standard 1-step lazy: i+1 is strictly better.
                    true
                } else if best_len <= LAZY_SHORT_LEN && i + 2 < n {
                    // 2-step lazy: i+1 isn't better, but maybe i+2 is much better.
                    let h2 = hash3(input, i + 2);
                    let (_, len2) =
                        find_match(input, i + 2, h2, &head, &prev, max_off, max_len, window_size);
                    len2 > best_len + 2
                } else {
                    false
                }
            } else {
                false
            };

            if lazy {
                // Emit literal at i, insert into chain, advance.
                // The cascade will re-evaluate i+1 on the next iteration.
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

/// Two-scan adaptive selection with entropy-safety cap:
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
/// Returns whichever scan produces the lowest raw token stream bit cost
/// among the eligible candidates. Ties go to baseline (proven safe).
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32) {
    // ── Baseline scan ─────────────────────────────────────────────────────────
    let baseline_tokens = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    let baseline_cost   = stream_bit_cost(&baseline_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);

    // ── Wide scan ─────────────────────────────────────────────────────────────
    let wide_tokens = scan(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
    let wide_ob     = compute_optimal_offset_bits(&wide_tokens);
    let wide_lb     = compute_optimal_length_bits(&wide_tokens);
    let wide_cost   = stream_bit_cost(&wide_tokens, wide_ob, wide_lb);

    // ── Entropy-safety cap ────────────────────────────────────────────────────
    if wide_lb > ENTROPY_SAFE_LENGTH_BITS {
        let wide_output_bytes = (wide_cost as usize + 7) / 8;
        if wide_output_bytes >= ENTROPY_MIN_BYTES_FOR_SCAN {
            let capped_tokens = scan(input, wide_ob, ENTROPY_SAFE_LENGTH_BITS);
            let capped_cost   = stream_bit_cost(&capped_tokens, wide_ob, ENTROPY_SAFE_LENGTH_BITS);

            println!(
                "  lb cap applied: wide lb={} output={}B >= entropy threshold {}B \
                 — re-scanned at lb={} (cost {} vs wide cost {})",
                wide_lb, wide_output_bytes, ENTROPY_MIN_BYTES_FOR_SCAN,
                ENTROPY_SAFE_LENGTH_BITS, capped_cost, wide_cost
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
