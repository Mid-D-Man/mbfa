// src/unfold.rs
//! Reverses N fold passes.
//! Header: [fold_count: u8][pair_flag: u8][entropy_flag: u8][offset_bits: u8]

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;
use crate::entropy;
use crate::opcode::{OFFSET_BITS_DEFAULT, OFFSET_BITS_MIN, OFFSET_BITS_MAX};

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() { return Ok(Vec::new()); }
    if input.len() < 3  { return Ok(input.to_vec()); }

    let fold_count   = input[0];
    let pair_flag    = input[1];
    let entropy_flag = input[2];

    // Byte 3 = offset_bits (new header field).
    // Old 3-byte-header files fall back to OFFSET_BITS_DEFAULT (15).
    let (offset_bits, payload_start) = if input.len() >= 4 {
        let ob = input[3] as u32;
        if ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX {
            (ob, 4usize)
        } else {
            (OFFSET_BITS_DEFAULT, 3usize)
        }
    } else {
        (OFFSET_BITS_DEFAULT, 3usize)
    };

    println!(
        "Unfolding {} pass(es) | pair_flag={} | entropy_flag={} | offset_bits={}",
        fold_count, pair_flag, entropy_flag, offset_bits
    );

    // Entropy wraps the outermost fold — decode it first.
    let (mut current, folds_to_undo) = if entropy_flag == 1 {
        let payload = &input[payload_start..];
        let (enc_table, bytes_consumed) = entropy::deserialize_table(payload)?;
        let dtable   = entropy::decode_table_from_encode(&enc_table);
        let stream   = &payload[bytes_consumed..];
        let tokens   = entropy::read_tokens_joint(stream, &dtable, offset_bits)?;
        let recovered = reconstruct(&tokens);
        println!("Joint entropy unfold: {} bytes recovered", recovered.len());
        (recovered, fold_count.saturating_sub(1))
    } else {
        (input[payload_start..].to_vec(), fold_count)
    };

    if folds_to_undo == 0 {
        return Ok(current);
    }

    for pass in (1..=folds_to_undo).rev() {
        if pass == 2 && pair_flag == 1 {
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());
            if folds_to_undo == 2 {
                let tokens = read_tokens(&current, offset_bits)?;
                current = reconstruct(&tokens);
                println!("Unfold pass 1 (LZ): {} bytes", current.len());
            }
            return Ok(current);
        } else {
            let tokens = read_tokens(&current, offset_bits)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    Ok(current)
}
