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

use crate::opcode::{Token, MAX_RING_SLOTS};

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
                // Apply copy.
                let start = output.len().saturating_sub(*offset as usize);
                for k in 0..*length as usize {
                    let byte = output[start + (k % *offset as usize)];
                    output.push(byte);
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
                // Apply copy (same logic as Backref).
                let start = output.len().saturating_sub(offset as usize);
                for k in 0..*length as usize {
                    let byte = output[start + (k % offset as usize)];
                    output.push(byte);
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
