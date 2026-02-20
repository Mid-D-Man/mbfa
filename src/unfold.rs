//! Reverses N fold passes.
//! Header format: [fold_count][pair_flag][data...]
//! pair_flag = 1 if fold 2 used pair encoding, 0 if it used LZ

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Header: byte 0 = fold count, byte 1 = pair flag
    if input.len() < 2 {
        return Ok(input.to_vec());
    }

    let fold_count = input[0];
    let pair_flag  = input[1];
    let mut current = input[2..].to_vec();

    println!("Unfolding {} passes (pair_flag={})...", fold_count, pair_flag);

    if fold_count == 0 {
        return Ok(current);
    }

    // Undo LZ folds above fold 2
    for pass in (3..=fold_count).rev() {
        let tokens = read_tokens(&current)?;
        current = reconstruct(&tokens);
        println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
    }

    // Undo fold 2
    if fold_count >= 2 {
        if pair_flag == 1 {
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());
        } else {
            let tokens = read_tokens(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (LZ): {} bytes", current.len());
        }
    }

    // Undo fold 1 — only needed if fold_count == 1
    // (fold 2 unfold already reconstructs original bytes via reconstruct)
    if fold_count == 1 {
        let tokens = read_tokens(&current)?;
        current = reconstruct(&tokens);
        println!("Unfold pass 1 (LZ): {} bytes", current.len());
    }

    Ok(current)
}