// src/unfold.rs
//! Reverses N fold passes.
//!
//! Header: [fold_count: u8][pair_flag: u8][entropy_flag: u8]
//!         [offset_bits[0]: u8] ... [offset_bits[fold_count-1]: u8]
//!         [payload...]
//!
//! entropy_flag=1 means joint Huffman + offset bucket coding was applied
//! to the outermost fold. offset_bits are still stored for the LZ passes.

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

    let (offset_bits_per_fold, payload_start) = parse_offset_bits(input, fold_count);

    println!(
        "Unfolding {} pass(es) | pair_flag={} | entropy_flag={} | offset_bits={:?}",
        fold_count, pair_flag, entropy_flag, offset_bits_per_fold
    );

    let ob_for_fold = |fold_n: usize| -> u32 {
        offset_bits_per_fold.get(fold_n.saturating_sub(1))
            .copied()
            .unwrap_or(OFFSET_BITS_DEFAULT)
    };

    // Entropy wraps the outermost fold — decode it first if present.
    // offset_bits not passed — entropy now uses bucket coding internally.
    let (mut current, folds_to_undo) = if entropy_flag == 1 {
        let payload = &input[payload_start..];
        let (enc_table, bytes_consumed) = entropy::deserialize_table(payload)?;
        let dtable   = entropy::decode_table_from_encode(&enc_table);
        let stream   = &payload[bytes_consumed..];
        let tokens   = entropy::read_tokens_joint(stream, &dtable)?;
        let recovered = reconstruct(&tokens);
        println!("Joint entropy unfold: {} bytes recovered", recovered.len());
        (recovered, fold_count.saturating_sub(1))
    } else {
        (input[payload_start..].to_vec(), fold_count)
    };

    if folds_to_undo == 0 {
        return Ok(current);
    }

    // Undo folds in reverse order.
    for pass in (1..=folds_to_undo).rev() {
        let ob = ob_for_fold(pass);

        if pass == 2 && pair_flag == 1 {
            let ob1 = ob_for_fold(1);
            let tokens = pair_decode(&current, ob1)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR) + pass 1 (LZ): {} bytes", current.len());
            return Ok(current);
        } else {
            let tokens = read_tokens(&current, ob)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    Ok(current)
}

fn parse_offset_bits(input: &[u8], fold_count: usize) -> (Vec<u32>, usize) {
    let new_format_end = 3 + fold_count;
    if fold_count > 0 && input.len() >= new_format_end {
        let candidate: Vec<u32> = input[3..new_format_end]
            .iter()
            .map(|&b| b as u32)
            .collect();
        let all_valid = candidate.iter().all(|&ob| {
            ob == 0 || (ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX)
        });
        if all_valid {
            return (candidate, new_format_end);
        }
    }

    if input.len() >= 4 {
        let ob = input[3] as u32;
        if ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX {
            let mut v = vec![ob; fold_count.max(1)];
            v.resize(fold_count, ob);
            return (v, 4);
        }
    }

    let v = vec![OFFSET_BITS_DEFAULT; fold_count];
    (v, 3)
}
