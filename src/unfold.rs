//! Reverses N fold passes.
//! Header: byte 0 = fold count, byte 1 = pair flag (1 = fold 2 used pairing)

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

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

    // Undo folds from highest down to 1
    // fold 2 is special — may be PAIR or LZ depending on pair_flag
    // all others are always LZ

    for pass in (1..=fold_count).rev() {
        if pass == 2 && pair_flag == 1 {
            // Fold 2 was PAIR — pair_decode gives fold 1 tokens directly
            // reconstruct on those gives original bytes — no fold 1 pass needed
            let tokens = pair_decode(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR): {} bytes", current.len());
            // PAIR already reconstructed through fold 1 so we are done
            return Ok(current);
        } else {
            // LZ fold — decode tokens and reconstruct
            let tokens = read_tokens(&current)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    Ok(current)
}
