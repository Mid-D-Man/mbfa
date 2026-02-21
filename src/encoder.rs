// src/encoder.rs
//! LZ-style scanner with hash chain match finding.
//! O(n) average case. Accepts offset_bits at runtime for adaptive window sizing.

use crate::opcode::{Token, LENGTH_BITS, LIT_TOTAL_BITS, backref_total_bits, max_offset, OFFSET_BITS_MAX};

const MAX_LENGTH: usize = (1 << LENGTH_BITS) as usize - 1;   // 255

const HASH_SIZE:  usize = 1 << 16;
const HASH_MASK:  usize = HASH_SIZE - 1;
const CHAIN_LIMIT: usize = 256;

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos] as usize).wrapping_mul(2654435761)
        ^ (input[pos + 1] as usize).wrapping_mul(2246822519)
        ^ (input[pos + 2] as usize).wrapping_mul(3266489917);
    v & HASH_MASK
}

/// Scan input using a lookback window of `(1 << offset_bits) - 1` bytes.
/// Pass `crate::opcode::OFFSET_BITS_MAX` for the widest possible window
/// when you plan to call `compute_optimal_offset_bits` afterwards and
/// re-encode with the tight value.
pub fn scan(input: &[u8], offset_bits: u32) -> Vec<Token> {
    let max_off = max_offset(offset_bits);
    let backref_bits = backref_total_bits(offset_bits);

    let mut tokens = Vec::new();
    let n = input.len();

    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; n];

    let mut i = 0;
    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) = find_match(input, i, h, &head, &prev, max_off);

        if best_len >= 2 && backref_bits < (best_len as u32 * LIT_TOTAL_BITS) {
            for k in 0..best_len {
                if i + k + 2 < n {
                    let hk = hash3(input, i + k);
                    prev[i + k] = head[hk];
                    head[hk] = (i + k) as u32;
                }
            }
            tokens.push(Token::Backref {
                offset: best_offset as u32,
                length: best_len as u32,
            });
            i += best_len;
        } else {
            prev[i] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Lit { byte: input[i] });
            i += 1;
        }
    }

    tokens.push(Token::End);
    tokens
}

fn find_match(
    input: &[u8],
    i: usize,
    h: usize,
    head: &[u32],
    prev: &[u32],
    max_off: usize,
) -> (usize, usize) {
    let n = input.len();
    let mut best_offset = 0;
    let mut best_len = 0;
    let mut steps = 0;

    let mut cur = head[h];
    while cur != u32::MAX && steps < CHAIN_LIMIT {
        let j = cur as usize;
        if i - j > max_off { break; }

        let span = i - j;
        let mut len = 0;
        while len < MAX_LENGTH
            && (i + len) < n
            && input[j + (len % span)] == input[i + len]
        {
            len += 1;
        }

        if len > best_len {
            best_len = len;
            best_offset = span;
            if best_len == MAX_LENGTH { break; }
        }

        cur = prev[j];
        steps += 1;
    }

    (best_offset, best_len)
}

/// Convenience: scan with the maximum window, then re-encode with the
/// minimum offset_bits that covers all actual offsets used.
/// Returns (tokens, optimal_offset_bits).
pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32) {
    // Scan with maximum window to find all possible matches.
    let tokens = scan(input, OFFSET_BITS_MAX);
    // Compute minimum bits needed for the offsets actually used.
    let optimal = crate::opcode::compute_optimal_offset_bits(&tokens);
    (tokens, optimal)
        }
