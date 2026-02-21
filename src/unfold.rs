// src/unfold.rs
//! Reverses N fold passes.
//! Header: [fold_count: u8][pair_flag: u8][entropy_flag: u8]
//! entropy_flag 1 = joint Huffman wraps the outermost fold — decode first.

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;
use crate::entropy;

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() { return Ok(Vec::new()); }
    if input.len() < 3  { return Ok(input.to_vec()); }

    let fold_count   = input[0];
    let pair_flag    = input[1];
    let entropy_flag = input[2];

    // Entropy wraps the outermost fold.
    // Decode + reconstruct gives the output of fold (fold_count - 1).
    let (mut current, folds_to_undo) = if entropy_flag == 1 {
        let (enc_table, bytes_consumed) = entropy::deserialize_table(&input[3..])?;
        let dtable   = entropy::decode_table_from_encode(&enc_table);
        let payload  = &input[3 + bytes_consumed..];
        let tokens   = entropy::read_tokens_joint(payload, &dtable)?;
        let recovered = reconstruct(&tokens);
        println!("Joint entropy unfold: {} bytes recovered", recovered.len());
        (recovered, fold_count.saturating_sub(1))
    } else {
        (input[3..].to_vec(), fold_count)
    };

    println!("Unfolding {} pass(es) (pair_flag={})...", folds_to_undo, pair_flag);

    if folds_to_undo == 0 {
        return Ok(current);
    }

    for pass in (1..=folds_to_undo).rev() {
        if pass == 2 && pair_flag == 1 {
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());
            return Ok(current);
        } else {
            let tokens = read_tokens(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    Ok(current)
}
