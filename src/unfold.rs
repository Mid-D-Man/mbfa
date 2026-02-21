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

    // Support old 3-byte headers (offset_bits absent → use default 15)
    // and new 4-byte headers.
    if input.len() < 3 { return Ok(input.to_vec()); }

    let fold_count   = input[0];
    let pair_flag    = input[1];
    let entropy_flag = input[2];

    // Byte 3: offset_bits (new header field).
    // Files compressed before this change won't have it — fall back to 15.
    let (offset_bits, payload_start) = if input.len() >= 4 {
        let ob = input[3] as u32;
        // Sanity-check the value is in the valid range before trusting it.
        // Old files may have had an entropy_flag=0 followed by data that
        // happens to look like a valid offset_bits value or not.
        if ob >= OFFSET_BITS_MIN && ob <= OFFSET_BITS_MAX {
            (ob, 4usize)
        } else {
            // Looks like old-format file — treat byte 3 as start of payload.
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

    for pass in (1..=folds_to_undo).rev() {
        if pass == 2 && pair_flag == 1 {
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());
            // After unpairing we have the fold 1 bitstream — use stored offset_bits.
            if folds_to_undo == 2 {
                // fold 2 was PAIR, fold 1 was LZ — one more LZ pass to undo.
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
