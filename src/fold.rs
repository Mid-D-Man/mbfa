// src/fold.rs
use crate::encoder::scan_adaptive;
use crate::bitwriter::write_tokens;
use crate::bitreader::read_tokens;
use crate::pairing::pair_encode;
use crate::opcode::{Token, OFFSET_BITS_MIN};

const MIN_IMPROVEMENT_RATIO: f64 = 0.97;
const MIN_FOLD_BITS: usize = 64;
const MIN_PAIR_BYTES: usize = 512;
const MAX_CANTOR_FALLBACK_RATE: f64 = 0.80;
const FOLD2_LZ_MAX_RATIO: f64 = 0.10;

fn cantor(x: u32, y: u32) -> u64 {
    let s = (x + y) as u64;
    s * (s + 1) / 2 + y as u64
}

fn cantor_fallback_rate(tokens: &[Token]) -> f64 {
    let mut total_br: u64 = 0;
    let mut fallbacks: u64 = 0;
    for t in tokens {
        if let Token::Backref { offset, length } = t {
            total_br += 1;
            if cantor(*offset, *length) >= 65536 { fallbacks += 1; }
        }
    }
    if total_br == 0 { return 0.0; }
    fallbacks as f64 / total_br as f64
}

fn log_window_diagnostics(tokens: &[Token], offset_bits: u32) {
    let max_off = (1u32 << offset_bits) - 1;
    let mut total_br: u64 = 0;
    let mut at_max: u64 = 0;
    let mut above_half: u64 = 0;
    let mut total_lit: u64 = 0;

    for t in tokens {
        match t {
            Token::Backref { offset, .. } => {
                total_br += 1;
                if *offset == max_off { at_max += 1; }
                if *offset > max_off / 2 { above_half += 1; }
            }
            Token::Lit { .. } => total_lit += 1,
            Token::End => {}
        }
    }

    if total_br + total_lit == 0 { return; }
    let total    = total_br + total_lit;
    let br_pct   = total_br as f64 / total as f64 * 100.0;
    let sat_pct  = if total_br > 0 { at_max     as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let deep_pct = if total_br > 0 { above_half as f64 / total_br as f64 * 100.0 } else { 0.0 };

    println!(
        "Window diagnostics: {}/{} tokens are BACKREF ({:.1}%) | \
         {:.1}% at max window ({} bytes) | {:.1}% in upper half | \
         offset_bits={} — {}",
        total_br, total, br_pct,
        sat_pct, max_off,
        deep_pct,
        offset_bits,
        if sat_pct > 5.0      { "⚠ SATURATED — window too small" }
        else if deep_pct > 30.0 { "HEAVY — large offsets dominant" }
        else                    { "OK" }
    );
}

/// Returns (compressed_bytes, folds_done, used_pairing, offset_bits_used)
pub fn fold(input: &[u8], max_folds: u8) -> std::io::Result<(Vec<u8>, u8, bool, u32)> {
    let mut current = input.to_vec();
    let mut folds_done: u8 = 0;
    let mut prev_size = input.len() * 8;
    let original_size = input.len() * 8;
    let mut final_used_pairing = false;
    let mut stored_offset_bits: u32 = OFFSET_BITS_MIN;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        let current_ratio = current.len() as f64 * 8.0 / original_size as f64;

        // ── Fold 1: adaptive LZ scan on raw bytes ────────────────────────────
        if fold_num == 1 {
            let (tokens, optimal_bits) = scan_adaptive(&current);
            stored_offset_bits = optimal_bits;

            log_window_diagnostics(&tokens, optimal_bits);
            println!(
                "Fold 1 adaptive: offset_bits={} (window={} bytes)",
                optimal_bits,
                (1u32 << optimal_bits) - 1
            );

            let folded = write_tokens(&tokens, optimal_bits)?;
            let folded_bits = folded.len() * 8;
            println!("Fold 1 (LZ): {} bits ({} bytes)", folded_bits, folded.len());

            let ratio = folded_bits as f64 / prev_size as f64;
            if ratio >= MIN_IMPROVEMENT_RATIO {
                println!("Fold 1 not worth it (ratio {:.3}), stopping at fold 0", ratio);
                break;
            }
            if folded_bits <= MIN_FOLD_BITS {
                println!("Hit minimum size floor at fold 1");
                current = folded;
                folds_done = 1;
                break;
            }
            current = folded;
            folds_done = 1;
            prev_size = folded_bits;
            continue;
        }

        // ── Fold 2+: decide strategy ─────────────────────────────────────────
        let allow_lz_on_packed = current_ratio < FOLD2_LZ_MAX_RATIO;

        let consider_pairing = fold_num == 2
            && folds_done == 1
            && current.len() >= MIN_PAIR_BYTES;

        let use_pairing = if consider_pairing {
            let tokens = read_tokens(&current, stored_offset_bits)?;
            let fallback_rate = cantor_fallback_rate(&tokens);
            println!(
                "Fold {} pairing pre-scan: Cantor fallback rate = {:.1}% (threshold {}%)",
                fold_num, fallback_rate * 100.0,
                (MAX_CANTOR_FALLBACK_RATE * 100.0) as u32
            );
            if fallback_rate > MAX_CANTOR_FALLBACK_RATE {
                println!("Fold {} pairing skipped — high fallback rate", fold_num);
                false
            } else {
                true
            }
        } else {
            false
        };

        if fold_num >= 2 && !use_pairing && !allow_lz_on_packed {
            println!(
                "Fold {} skipped — LZ on packed bytes not beneficial \
                 (current ratio {:.3}, threshold {:.2})",
                fold_num, current_ratio, FOLD2_LZ_MAX_RATIO
            );
            break;
        }

        let folded = if use_pairing {
            let tokens = read_tokens(&current, stored_offset_bits)?;
            pair_encode(&tokens)?
        } else {
            // LZ on packed bytes — only reached when current_ratio < 0.10.
            let (tokens, new_bits) = scan_adaptive(&current);
            stored_offset_bits = new_bits;
            write_tokens(&tokens, new_bits)?
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

    Ok((current, folds_done, final_used_pairing, stored_offset_bits))
        }
