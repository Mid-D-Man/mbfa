// src/fold.rs
use crate::encoder::scan;
use crate::bitwriter::write_tokens;
use crate::bitreader::read_tokens;
use crate::pairing::pair_encode;
use crate::opcode::Token;

const MIN_IMPROVEMENT_RATIO: f64 = 0.97;
const MIN_FOLD_BITS: usize = 64;
const MIN_PAIR_BYTES: usize = 512;
const MAX_CANTOR_FALLBACK_RATE: f64 = 0.80;

// Only attempt fold 2+ LZ on the packed bitstream if fold 1 compressed
// to below this fraction of the original. At higher ratios the bitstream
// bytes have no LZ-exploitable structure (they look quasi-random due to
// bit-packing boundary misalignment) and fold 2 LZ will expand the data.
// 0.10 = only try if fold 1 already achieved 90%+ reduction (e.g. highly
// repetitive data). For everything else, fold 2 is pairing-only or skip.
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
            if cantor(*offset, *length) >= 65536 {
                fallbacks += 1;
            }
        }
    }
    if total_br == 0 { return 0.0; }
    fallbacks as f64 / total_br as f64
}

/// Emit diagnostics about window utilization for a token stream.
/// Helps identify when the lookback window is the compression bottleneck.
fn log_window_diagnostics(tokens: &[Token]) {
    let max_offset_bits = crate::opcode::OFFSET_BITS;
    let max_offset = (1u32 << max_offset_bits) - 1;

    let mut total_br: u64 = 0;
    let mut at_max_window: u64 = 0;
    let mut above_half_window: u64 = 0;
    let mut total_lit: u64 = 0;

    for t in tokens {
        match t {
            Token::Backref { offset, .. } => {
                total_br += 1;
                if *offset == max_offset { at_max_window += 1; }
                if *offset > max_offset / 2 { above_half_window += 1; }
            }
            Token::Lit { .. } => total_lit += 1,
            Token::End => {}
        }
    }

    if total_br + total_lit == 0 { return; }
    let total = total_br + total_lit;
    let br_pct = total_br as f64 / total as f64 * 100.0;
    let saturated_pct = if total_br > 0 {
        at_max_window as f64 / total_br as f64 * 100.0
    } else { 0.0 };
    let deep_pct = if total_br > 0 {
        above_half_window as f64 / total_br as f64 * 100.0
    } else { 0.0 };

    println!(
        "Window diagnostics: {}/{} tokens are BACKREF ({:.1}%) | \
         {:.1}% at max window ({} bytes) | {:.1}% in upper half of window | \
         Window utilization: {}",
        total_br, total, br_pct,
        saturated_pct, max_offset,
        deep_pct,
        if saturated_pct > 5.0 {
            "⚠ SATURATED — increasing OFFSET_BITS would help"
        } else if deep_pct > 30.0 {
            "HEAVY — large offsets dominant"
        } else {
            "OK"
        }
    );
}

/// Returns (compressed_bytes, folds_done, used_pairing)
pub fn fold(input: &[u8], max_folds: u8) -> std::io::Result<(Vec<u8>, u8, bool)> {
    let mut current = input.to_vec();
    let mut folds_done: u8 = 0;
    let mut prev_size = input.len() * 8;
    let original_size = input.len() * 8;
    let mut final_used_pairing = false;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        // ── Decide fold strategy ─────────────────────────────────────────────
        //
        // Fold 1: always LZ on raw input bytes.
        //
        // Fold 2+: the current `bytes` are a packed BigEndian bitstream of
        // LIT/BACKREF/END tokens. Running LZ directly on these bytes is
        // almost always counterproductive — bit-packing misaligns token
        // boundaries with byte boundaries, so consecutive tokens produce
        // byte sequences with no local repetition even if the token STREAM
        // has high structural regularity. The LZ scanner finds nothing and
        // emits mostly LIT tokens, adding ~25% overhead.
        //
        // Strategy: for fold 2+ we only attempt pairing (token-level) when
        // the Cantor fallback rate is acceptable. We only fall back to LZ on
        // packed bytes when fold 1 achieved extreme compression (< 10% of
        // original) — which means the packed bytes ARE highly repetitive
        // (e.g. the repetitive_12KB case, fold 1 → 0.13%).

        let current_ratio = current.len() as f64 * 8.0 / original_size as f64;

        let consider_pairing = fold_num == 2
            && folds_done == 1
            && current.len() >= MIN_PAIR_BYTES;

        // Only try fold 2+ LZ if the stream is extremely compressed already.
        // For everything else, LZ on packed bytes will expand the data.
        let allow_lz_on_packed = current_ratio < FOLD2_LZ_MAX_RATIO;

        let use_pairing = if consider_pairing {
            let tokens = read_tokens(&current)?;

            // Log window diagnostics on fold 1 output — this is where we can
            // see if the lookback window is the bottleneck.
            if fold_num == 2 {
                log_window_diagnostics(&tokens);
            }

            let fallback_rate = cantor_fallback_rate(&tokens);
            println!(
                "Fold {} pairing pre-scan: Cantor fallback rate = {:.1}% (threshold {}%)",
                fold_num,
                fallback_rate * 100.0,
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

        // Skip this fold entirely if: not pairing AND LZ on packed bytes is
        // not allowed (stream is not extremely compressed). This avoids the
        // expand-then-stop cycle that wastes time and clutters fold logs.
        if fold_num >= 2 && !use_pairing && !allow_lz_on_packed {
            println!(
                "Fold {} skipped — LZ on packed bytes not beneficial \
                 (current ratio {:.3}, threshold {:.2}), stream not extreme enough",
                fold_num, current_ratio, FOLD2_LZ_MAX_RATIO
            );
            break;
        }

        let folded = if use_pairing {
            let tokens = read_tokens(&current)?;
            pair_encode(&tokens)?
        } else {
            // LZ on packed bytes — only reached when current_ratio < 0.10
            let tokens = scan(&current);
            write_tokens(&tokens)?
        };

        let folded_bits = folded.len() * 8;
        let fold_type = if use_pairing { "PAIR" } else { "LZ" };
        println!("Fold {} ({}): {} bits ({} bytes)", fold_num, fold_type, folded_bits, folded.len());

        let ratio = folded_bits as f64 / prev_size as f64;
        if ratio >= MIN_IMPROVEMENT_RATIO {
            println!(
                "Fold {} not worth it (ratio {:.3}), stopping at fold {}",
                fold_num, ratio, folds_done
            );
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
