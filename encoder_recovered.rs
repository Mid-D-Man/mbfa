//! LZ-style scanner.
//! Scans a byte slice and produces a Vec<Token> instruction stream.

use crate::opcode::{Token, OFFSET_BITS, LENGTH_BITS, BACKREF_TOTAL_BITS, LIT_TOTAL_BITS};

const MAX_OFFSET: usize = (1 << OFFSET_BITS) - 1;   // 63 — max value in 6 bits
const MAX_LENGTH: usize = (1 << LENGTH_BITS) - 1;    // 31 — max value in 5 bits

/// Scan `input` and return the most compact token stream we can find.
pub fn scan(input: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < input.len() {
        let (best_offset, best_len) = find_best_match(input, i);

        if best_len >= 2 && BACKREF_TOTAL_BITS < (best_len as u32 * LIT_TOTAL_BITS) {
            tokens.push(Token::Backref {
                offset: best_offset as u8,
                length: best_len as u8,
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

/// Find the longest match for input[i..] within the lookback window
fn find_best_match(input: &[u8], i: usize) -> (usize, usize) {
    let search_start = i.saturating_sub(MAX_OFFSET);
    let mut best_offset = 0;
    let mut best_len = 0;

    for j in search_start..i {
        let span = i - j;
        let mut len = 0;
        while len < MAX_LENGTH
            && (i + len) < input.len()
            && input[j + (len % span)] == input[i + len]
        {
            len += 1;
        }

        if len > best_len {
            best_len = len;
            best_offset = span;
        }
    }

    (best_offset, best_len)
}