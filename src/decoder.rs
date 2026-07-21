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
//! conceptually `dict ++ output`, not just `output`. A normal backref
//! (offset <= output.len()) resolves identically to before -- the virtual
//! position is always >= dict.len() in that case, so this is a pure
//! generalization, not a behavior change, for every existing token stream.
//! Only offsets that deliberately reach further back (only ever produced by
//! encoder::scan_with_dict) actually fall into the dictionary.
//!
//! `dict` is passed in by the caller (unfold.rs), resolved from the
//! header's `dict_flag` byte via dictionary::DictId::bytes(). Before the
//! dictionary/ subdirectory split there was exactly one possible
//! dictionary, so this module could import it as a fixed const; now there
//! are four (plus "none"), so reconstruct() takes it as a parameter and
//! the caller is responsible for picking the right one. Passing `&[]`
//! (dict.len()==0) reproduces the original no-dictionary behavior exactly.

use crate::opcode::{Token, MAX_RING_SLOTS};

#[inline]
fn virtual_byte(output: &[u8], dict: &[u8], vpos: usize) -> u8 {
    if vpos < dict.len() { dict[vpos] } else { output[vpos - dict.len()] }
}

pub fn reconstruct(tokens: &[Token], dict: &[u8]) -> Vec<u8> {
    let dict_len = dict.len();
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
                // Apply copy. virtual_start is relative to dict_len + output.len(),
                // so this is exactly the old `output.len() - offset` when offset
                // stays within the real window (virtual_start ends up >= dict_len),
                // and falls into the dictionary when it doesn't.
                let virtual_start = (dict_len + output.len()).saturating_sub(*offset as usize);
                for k in 0..*length as usize {
                    let vpos = virtual_start + (k % *offset as usize);
                    output.push(virtual_byte(&output, dict, vpos));
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
                let virtual_start = (dict_len + output.len()).saturating_sub(offset as usize);
                for k in 0..*length as usize {
                    let vpos = virtual_start + (k % offset as usize);
                    output.push(virtual_byte(&output, dict, vpos));
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
    use crate::dictionary::dixscript;

    #[test]
    fn normal_backref_unaffected_by_dictionary_generalization() {
        // offset well within output.len() -- must behave exactly as before.
        let tokens = vec![
            Token::Lit { byte: b'A' }, Token::Lit { byte: b'B' }, Token::Lit { byte: b'C' },
            Token::Backref { offset: 3, length: 3 }, // repeats "ABC"
            Token::End,
        ];
        assert_eq!(reconstruct(&tokens, &[]), b"ABCABC".to_vec());
    }

    #[test]
    fn backref_reaching_past_output_resolves_into_dictionary() {
        // First byte(s) of output, then a backref whose offset exceeds
        // output.len() entirely -- must resolve into the tail of the dictionary.
        // offset=6 with output.len()=1 at that point => virtual_start =
        // (dict_len+1)-6 = dict_len-5, i.e. the dictionary's last 5 bytes.
        let dict = dixscript::DICTIONARY;
        let tail = &dict[dict.len() - 5..];
        let tokens = vec![
            Token::Lit { byte: b'X' },
            Token::Backref { offset: 6, length: 5 },
            Token::End,
        ];
        let out = reconstruct(&tokens, dict);
        assert_eq!(&out[1..], tail);
    }

    #[test]
    fn backref_straddling_dictionary_and_output_boundary() {
        // offset chosen so the copy starts inside the dictionary's last 2
        // bytes and continues into freshly-written output.
        // offset=4 with output.len()=2 at that point => virtual_start =
        // (dict_len+2)-4 = dict_len-2.
        let dict = dixscript::DICTIONARY;
        let tokens = vec![
            Token::Lit { byte: b'Q' }, Token::Lit { byte: b'R' },
            Token::Backref { offset: 4, length: 4 },
            Token::End,
        ];
        let out = reconstruct(&tokens, dict);
        let expected_start = &dict[dict.len() - 2..];
        assert_eq!(&out[2..4], expected_start);
    }

    #[test]
    fn empty_dict_matches_pre_split_no_dictionary_behavior() {
        // dict=&[] must be indistinguishable from "no dictionary" -- any
        // offset that would have reached the dictionary now has nowhere to
        // resolve, so this only matters for tokens whose offsets stay
        // within output.len(), which is exactly what fold 2+ and
        // no-dictionary-winner fold-1 streams guarantee by construction.
        let tokens = vec![
            Token::Lit { byte: b'H' }, Token::Lit { byte: b'I' },
            Token::Backref { offset: 2, length: 4 },
            Token::End,
        ];
        assert_eq!(reconstruct(&tokens, &[]), b"HIHIHI".to_vec());
    }
                }
