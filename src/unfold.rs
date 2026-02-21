// src/unfold.rs
//! Reverses N fold passes.
//!
//! Header: [fold_count: u8][pair_flag: u8][entropy_flag: u8]
//!         [offset_bits[0]: u8] ... [offset_bits[fold_count-1]: u8]
//!         [payload...]
//!
//! offset_bits[i] is the LZ window size used for fold (i+1).
//! PAIR folds store 0 as a sentinel — they do not use an LZ bitstream.

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;
use crate::entropy;
use crate::opcode::{OFFSET_BITS_DEFAULT, OFFSET_BITS_MIN, OFFSET_BITS_MAX};

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() { return Ok(Vec::new()); }
    if input.len() < 3  { return Ok(input.to_vec()); }

    let fold_count   = input[0] as usize;
    let pair_flag    = input[1];
    let entropy_flag = input[2];

    // Read per-fold offset_bits array (fold_count bytes starting at byte 3).
    // Fall back gracefully for old 3-byte or 4-byte format files.
    let (offset_bits_per_fold, payload_start) = parse_offset_bits(
        input, fold_count
    );

    println!(
        "Unfolding {} pass(es) | pair_flag={} | entropy_flag={} | offset_bits={:?}",
        fold_count, pair_flag, entropy_flag, offset_bits_per_fold
    );

    // Helper: get offset_bits for fold N (1-indexed).
    // Returns OFFSET_BITS_DEFAULT if out of range (shouldn't happen on valid files).
    let ob_for_fold = |fold_n: usize| -> u32 {
        offset_bits_per_fold.get(fold_n.saturating_sub(1))
            .copied()
            .unwrap_or(OFFSET_BITS_DEFAULT)
    };

    // final_ob is the offset_bits of the outermost fold's LZ bitstream
    // (what entropy wraps, if entropy_flag=1).
    let final_ob = ob_for_fold(fold_count);

    // Entropy wraps the outermost fold — decode it first.
    // After decoding we have the raw bitstream of fold (fold_count - 1).
    let (mut current, folds_to_undo) = if entropy_flag == 1 {
        let payload = &input[payload_start..];
        let (enc_table, bytes_consumed) = entropy::deserialize_table(payload)?;
        let dtable   = entropy::decode_table_from_encode(&enc_table);
        let stream   = &payload[bytes_consumed..];
        let tokens   = entropy::read_tokens_joint(stream, &dtable, final_ob)?;
        let recovered = reconstruct(&tokens);
        println!("Joint entropy unfold: {} bytes recovered", recovered.len());
        // Entropy consumed the outermost fold; remaining folds = fold_count - 1.
        (recovered, fold_count.saturating_sub(1))
    } else {
        (input[payload_start..].to_vec(), fold_count)
    };

    if folds_to_undo == 0 {
        return Ok(current);
    }

    // Undo folds in reverse order.
    // pass iterates fold_count down to 1 (or fold_count-1 down to 1 if entropy ran).
    // Each pass uses the offset_bits that were used to ENCODE that fold.
    for pass in (1..=folds_to_undo).rev() {
        let ob = ob_for_fold(pass);

        if pass == 2 && pair_flag == 1 {
            // Fold 2 was PAIR — undo with pair_decode (no offset_bits needed).
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());

            // After unpairing, current = fold 1 bitstream.
            // If there's a fold 1 below this, undo it now.
            if pass > 1 {
                let ob1 = ob_for_fold(1);
                let tokens = read_tokens(&current, ob1)?;
                current = reconstruct(&tokens);
                println!("Unfold pass 1 (LZ): {} bytes", current.len());
            }
            return Ok(current);
        } else {
            // LZ fold — ob must be the value used when this fold was encoded.
            let tokens = read_tokens(&current, ob)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    Ok(current)
}

/// Parse the per-fold offset_bits array from the header.
///
/// New format (fold_count > 0): bytes 3..3+fold_count are offset_bits values.
/// Old 4-byte format: byte 3 is a single offset_bits (read as fold 1's value).
/// Old 3-byte format: use OFFSET_BITS_DEFAULT for all folds.
fn parse_offset_bits(input: &[u8], fold_count: usize) -> (Vec<u32>, usize) {
    // If file is long enough to hold fold_count bytes after byte 3, and all
    // values are in valid range, treat as new format.
    let new_format_end = 3 + fold_count;
    if fold_count > 0 && input.len() >= new_format_end {
        let candidate: Vec<u32> = input[3..new_format_end]
            .iter()
            .map(|&b| b as u32)
            .collect();
        // Validate: each non-zero value must be in [OFFSET_BITS_MIN, OFFSET_BITS_MAX].
        // Zero is the PAIR sentinel and is always valid.
        let all_valid = candidate.iter().all(|&ob| {
            ob == 0 || (ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX)
        });
        if all_valid {
            return (candidate, new_format_end);
        }
    }

    // Old 4-byte format: single offset_bits at byte 3.
    if input.len() >= 4 {
        let ob = input[3] as u32;
        if ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX {
            let mut v = vec![ob; fold_count.max(1)];
            // Ensure we have at least fold_count entries.
            v.resize(fold_count, ob);
            return (v, 4);
        }
    }

    // Fallback: old 3-byte format, use default for all folds.
    let v = vec![OFFSET_BITS_DEFAULT; fold_count];
    (v, 3)
        }
