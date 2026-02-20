use crate::encoder::scan;
use crate::bitwriter::write_tokens;
use crate::bitreader::read_tokens;
use crate::pairing::pair_encode;

const MIN_IMPROVEMENT_RATIO: f64 = 0.97;
const MIN_FOLD_BITS: usize = 64;
const MIN_PAIR_BYTES: usize = 512;

/// Returns (compressed_bytes, folds_done, used_pairing)
pub fn fold(input: &[u8], max_folds: u8) -> std::io::Result<(Vec<u8>, u8, bool)> {
    let mut current = input.to_vec();
    let mut folds_done: u8 = 0;
    let mut prev_size = input.len() * 8;
    let mut final_used_pairing = false;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        let use_pairing = fold_num == 2 && folds_done == 1 && current.len() >= MIN_PAIR_BYTES;

        let folded = if use_pairing {
            let tokens = read_tokens(&current)?;
            pair_encode(&tokens)?
        } else {
            let tokens = scan(&current);
            write_tokens(&tokens)?
        };

        let folded_bits = folded.len() * 8;
        let fold_type = if use_pairing { "PAIR" } else { "LZ" };
        println!("Fold {} ({}): {} bits ({} bytes)", fold_num, fold_type, folded_bits, folded.len());

        let ratio = folded_bits as f64 / prev_size as f64;
        if folded_bits >= (prev_size as f64 * MIN_IMPROVEMENT_RATIO) as usize {
            println!("Fold {} not worth it (ratio {:.3}), stopping at fold {}",
                     fold_num, ratio, folds_done);
            break;
        }

        if folded_bits <= MIN_FOLD_BITS {
            println!("Hit minimum size floor at fold {}", fold_num);
            current = folded;
            folds_done = fold_num;
            if use_pairing { final_used_pairing = true; }
            break;
        }

        current = folded;
        folds_done = fold_num;
        if use_pairing { final_used_pairing = true; }
        prev_size = folded_bits;
    }

    Ok((current, folds_done, final_used_pairing))
}