// src/decoder.rs
//! Reconstructs a byte slice from a Token stream.

use crate::opcode::Token;

pub fn reconstruct(tokens: &[Token]) -> Vec<u8> {
    // Task 4: pre-pass to compute exact output byte count so we allocate once
    // and never reallocate. Cost: one O(n) pass over tokens — cheaper than
    // the O(n log n) reallocation chain the old Vec::new() + push loop caused
    // on large decompression (e.g. WarAndPeace unfold produces ~1.4MB tokens).
    //
    // Lit  → 1 byte
    // Backref { offset > 0, length } → length bytes
    // Backref { offset == 0 } → 0 bytes (corrupt, skipped in loop below)
    // End  → 0 bytes
    let capacity: usize = tokens.iter().map(|t| match t {
        Token::Lit { .. }                                    => 1,
        Token::Backref { offset, length } if *offset > 0    => *length as usize,
        _ => 0,
    }).sum();

    let mut output: Vec<u8> = Vec::with_capacity(capacity);

    for token in tokens {
        match token {
            Token::Lit { byte } => {
                output.push(*byte);
            }
            Token::Backref { offset, length } => {
                if *offset == 0 {
                    eprintln!("Warning: Backref offset=0 — skipping corrupt token");
                    continue;
                }
                let start = output.len().saturating_sub(*offset as usize);
                for k in 0..*length as usize {
                    let byte = output[start + (k % *offset as usize)];
                    output.push(byte);
                }
            }
            Token::End => break,
        }
    }

    output
}
