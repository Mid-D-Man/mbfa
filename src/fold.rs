// src/fold.rs
use crate::encoder::scan;
use crate::bitwriter::write_tokens;
use crate::bitreader::read_tokens;
use crate::pairing::pair_encode;
use crate::opcode::Token;

const MIN_IMPROVEMENT_RATIO: f64 = 0.97;
const MIN_FOLD_BITS: usize = 64;
const MIN_PAIR_BYTES: usize = 512;

/// Fraction of BACKREFs whose Cantor value exceeds 16 bits.
/// Above this threshold pairing is net-negative — the 3-bit prefix
/// overhead outweighs any savings from the pair structure.
const MAX_CANTOR_FALLBACK_RATE: f64 = 0.80;

fn cantor(x: u32, y: u32) -> u64 {
    let s = (x + y) as u64;
    s * (s + 1) / 2 + y as u64
}

/// Scan token stream and return fraction of BACKREFs where
/// cantor(offset, length) >= 65536 (would fall back to raw encoding).
/// Returns 0.0 if there are no BACKREFs at all.
fn cantor_fallback_rate(tokens: &[Token]) -> f64 {
    let mut total_br: u64 = 0;
    let mut fallbacks: u64 = 0;
    for t in tokens {
        if let Token::Backref { offset, length } = t {
            total_br += 1;
            if cantor(*offset, *length) >= 65536 {
                fallbacks += 1;
            }
        }
    }
    if total_br == 0 { return 0.0; }
    fallbacks as f64 / total_br as f64
}

/// Returns (compressed_bytes, folds_done, used_pairing)
pub fn fold(input: &[u8], max_folds: u8) -> std::io::Result<(Vec<u8>, u8, bool)> {
    let mut current = input.to_vec();
    let mut folds_done: u8 = 0;
    let mut prev_size = input.len() * 8;
    let mut final_used_pairing = false;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        // Pairing conditions:
        // 1. Must be fold 2 on a fold 1 output
        // 2. Fold 1 output must exceed MIN_PAIR_BYTES
        // 3. Cantor fallback rate must be below threshold — if most BACKREFs
        //    have large offsets, the pair prefix overhead makes output bigger
        let consider_pairing = fold_num == 2
            && folds_done == 1
            && current.len() >= MIN_PAIR_BYTES;

        let use_pairing = if consider_pairing {
            let tokens = read_tokens(&current)?;
            let fallback_rate = cantor_fallback_rate(&tokens);
            println!(
                "Fold 2 pairing pre-scan: Cantor fallback rate = {:.1}% (threshold {}%)",
                fallback_rate * 100.0,
                (MAX_CANTOR_FALLBACK_RATE * 100.0) as u32
            );
            if fallback_rate > MAX_CANTOR_FALLBACK_RATE {
                println!("Fold 2 pairing skipped — high fallback rate, LZ will do better");
                false
            } else {
                true
            }
        } else {
            false
        };

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
        if ratio >= MIN_IMPROVEMENT_RATIO {
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
