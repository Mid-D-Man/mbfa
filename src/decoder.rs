// src/decoder.rs
//! Reconstructs a byte slice from a Token stream.
//!
//! Maintains a 4-slot LRU ring buffer (`MAX_RING_SLOTS`) mirroring the
//! encoder's `RepSlots` state.  `Token::RepRef { slot, length }` looks up
//! the ring buffer to resolve the back-reference offset, then applies the
//! copy identically to a `Token::Backref`.
//!
//! Ring update rules (must exactly mirror the encoder):
//!   Backref { offset }  → push offset to front iff offset ≠ ring[0].
//!   RepRef  { slot   }  → move ring[slot] to front.
//!
//! The decoder is deterministic and stateless (ring rebuilt per reconstruct()
//! call), so it can be called concurrently on different token streams.
//!
//! Static dictionary support: the addressable "history" for any offset is
//! conceptually `DICTIONARY ++ output`, not just `output`. A normal backref
//! (offset <= output.len()) resolves identically to before -- the virtual
//! position is always >= DICT_LEN in that case, so this is a pure
//! generalization, not a behavior change, for every existing token stream.
//! Only offsets that deliberately reach further back (only ever produced by
//! encoder::scan_with_dict) actually fall into the dictionary.

use crate::opcode::{Token, MAX_RING_SLOTS};
use crate::dictionary::{DICTIONARY, DICT_LEN};

#[inline]
fn virtual_byte(output: &[u8], vpos: usize) -> u8 {
    if vpos < DICT_LEN { DICTIONARY[vpos] } else { output[vpos - DICT_LEN] }
}

pub fn reconstruct(tokens: &[Token]) -> Vec<u8> {
    // Pre-pass: compute exact output byte count to allocate once.
    let capacity: usize = tokens.iter().map(|t| match t {
        Token::Lit { .. }                                         => 1,
        Token::Backref { offset, length } if *offset > 0         => *length as usize,
        Token::RepRef  { length, .. }                             => *length as usize,
        _ => 0,
    }).sum();

    let mut output:     Vec<u8>         = Vec::with_capacity(capacity);
    let mut ring:       [u32; MAX_RING_SLOTS] = [0u32; MAX_RING_SLOTS];
    let mut ring_count: usize           = 0;

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
                // Apply copy. virtual_start is relative to DICT_LEN + output.len(),
                // so this is exactly the old `output.len() - offset` when offset
                // stays within the real window (virtual_start ends up >= DICT_LEN),
                // and falls into the dictionary when it doesn't.
                let virtual_start = (DICT_LEN + output.len()).saturating_sub(*offset as usize);
                for k in 0..*length as usize {
                    let vpos = virtual_start + (k % *offset as usize);
                    output.push(virtual_byte(&output, vpos));
                }
                // Update ring: push to front iff different from ring[0].
                if ring_count == 0 || ring[0] != *offset {
                    if ring_count < MAX_RING_SLOTS { ring_count += 1; }
                    for j in (1..ring_count).rev() { ring[j] = ring[j - 1]; }
                    ring[0] = *offset;
                }
            }

            Token::RepRef { slot, length } => {
                let s = *slot as usize;
                if s >= ring_count {
                    eprintln!(
                        "Warning: RepRef slot {} >= ring_count {} — skipping",
                        s, ring_count
                    );
                    continue;
                }
                let offset = ring[s];
                if offset == 0 {
                    eprintln!("Warning: RepRef resolved to offset=0 — skipping");
                    continue;
                }
                // Apply copy (same logic as Backref, dictionary-aware too).
                let virtual_start = (DICT_LEN + output.len()).saturating_sub(offset as usize);
                for k in 0..*length as usize {
                    let vpos = virtual_start + (k % offset as usize);
                    output.push(virtual_byte(&output, vpos));
                }
                // Update ring: move ring[s] to front (LRU move-to-front).
                for j in (1..=s).rev() { ring[j] = ring[j - 1]; }
                ring[0] = offset;
            }

            Token::End => break,
        }
    }

    output
        }

#[cfg(test)]
mod dict_reconstruct_tests {
    use super::*;

    #[test]
    fn normal_backref_unaffected_by_dictionary_generalization() {
        // offset well within output.len() -- must behave exactly as before.
        let tokens = vec![
            Token::Lit { byte: b'A' }, Token::Lit { byte: b'B' }, Token::Lit { byte: b'C' },
            Token::Backref { offset: 3, length: 3 }, // repeats "ABC"
            Token::End,
        ];
        assert_eq!(reconstruct(&tokens), b"ABCABC".to_vec());
    }

    #[test]
    fn backref_reaching_past_output_resolves_into_dictionary() {
        // First byte(s) of output, then a backref whose offset exceeds
        // output.len() entirely -- must resolve into the tail of DICTIONARY.
        // offset=6 with output.len()=1 at that point => virtual_start =
        // (DICT_LEN+1)-6 = DICT_LEN-5, i.e. the dictionary's last 5 bytes.
        let tail = &DICTIONARY[DICT_LEN - 5..];
        let tokens = vec![
            Token::Lit { byte: b'X' },
            Token::Backref { offset: 6, length: 5 },
            Token::End,
        ];
        let out = reconstruct(&tokens);
        assert_eq!(&out[1..], tail);
    }

    #[test]
    fn backref_straddling_dictionary_and_output_boundary() {
        // offset chosen so the copy starts inside the dictionary's last 2
        // bytes and continues into freshly-written output.
        // offset=4 with output.len()=2 at that point => virtual_start =
        // (DICT_LEN+2)-4 = DICT_LEN-2.
        let tokens = vec![
            Token::Lit { byte: b'Q' }, Token::Lit { byte: b'R' },
            Token::Backref { offset: 4, length: 4 },
            Token::End,
        ];
        let out = reconstruct(&tokens);
        let expected_start = &DICTIONARY[DICT_LEN - 2..];
        assert_eq!(&out[2..4], expected_start);
    }
                                                  }
