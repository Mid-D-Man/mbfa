// src/fold.rs
//
// Changes from previous version:
//   - Cantor-related code removed (cantor(), cantor_fallback_rate(),
//     MAX_CANTOR_FALLBACK_RATE). Pair encoding now uses EG internally
//     (see pairing.rs) and always produces valid output, so no pre-check needed.
//
//   - Fold 2+ strategy is now: try pair encoding first (if applicable),
//     fall back to LZ if pair doesn't improve. Previously, pairing was
//     either tried OR LZ was tried (never both). The new structure lets
//     LZ still run for files like TOML_Config where pair fails but
//     allow_lz_on_packed is true. This prevents the regression where removing
//     the Cantor gate would have lost the LZ fold 2 improvement for such files.
//
//   - scan_adaptive now returns (Vec<Token>, u32, u32, bool) — the bool is
//     the incompressible signal. When true, we stop folding and let lib.rs
//     emit a fold_count=0 passthrough. See encoder.rs for details.
//
//   - fold() now takes filter_flag: u8. When filter_flag != 0 (a structural
//     pre-filter was applied), scan_adaptive is called with
//     skip_incompressible_bail=true so the scanner never bails on filtered
//     data. The filter has already validated exploitable structure exists.
//     Also, the fold-1 worthiness threshold is relaxed: for filtered data any
//     LZ improvement (ratio < 1.0) is accepted, whereas for unfiltered data
//     the existing MIN_IMPROVEMENT_RATIO (0.985) applies.

use crate::encoder::scan_adaptive;
use crate::bitwriter::write_tokens;
use crate::pairing::pair_encode;
use crate::opcode::{
    Token, OFFSET_BITS_MIN, LENGTH_BITS_MIN,
    LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
};

const MIN_IMPROVEMENT_RATIO: f64   = 0.985;
const MIN_FOLD_BITS:         usize = 64;
const MIN_PAIR_BYTES:        usize = 512;
/// LZ on fold 2+ bytes only when fold-1 compressed very aggressively.
/// When the ratio is already below 10%, fold 2 LZ can extract more.
const FOLD2_LZ_MAX_RATIO:    f64   = 0.10;

/// Baseline parameters mirrored from encoder.rs for the Phase C diagnostic.
const DIAG_BASELINE_OB: u32 = 17;
const DIAG_BASELINE_LB: u32 = 8;

fn log_window_diagnostics(tokens: &[Token], offset_bits: u32, length_bits: u32) {
    let max_off = (1u32 << offset_bits) - 1;
    let max_len = (1u32 << length_bits) - 1;
    let mut total_br: u64   = 0;
    let mut at_max_off: u64 = 0;
    let mut above_half: u64 = 0;
    let mut at_max_len: u64 = 0;
    let mut total_lit: u64  = 0;

    for t in tokens {
        match t {
            Token::Backref { offset, length } => {
                total_br += 1;
                if *offset == max_off { at_max_off += 1; }
                if *offset > max_off / 2 { above_half += 1; }
                if *length == max_len { at_max_len += 1; }
            }
            Token::Lit { .. } => total_lit += 1,
            Token::End => {}
        }
    }

    if total_br + total_lit == 0 { return; }
    let total    = total_br + total_lit;
    let br_pct   = total_br    as f64 / total    as f64 * 100.0;
    let sat_pct  = if total_br > 0 { at_max_off as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let deep_pct = if total_br > 0 { above_half as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let len_pct  = if total_br > 0 { at_max_len as f64 / total_br as f64 * 100.0 } else { 0.0 };

    println!(
        "Window diagnostics: {}/{} tokens BACKREF ({:.1}%) | \
         offset: {:.1}% at max ({}) | {:.1}% upper half | \
         length: {:.1}% at max ({}) | offset_bits={} length_bits={} — {}",
        total_br, total, br_pct,
        sat_pct, max_off, deep_pct,
        len_pct, max_len,
        offset_bits, length_bits,
        if sat_pct > 5.0        { "⚠ OFFSET SATURATED — window too small" }
        else if len_pct > 5.0   { "⚠ LENGTH SATURATED — length field too small" }
        else if deep_pct > 30.0 { "HEAVY — large offsets dominant" }
        else                    { "OK" }
    );
}

fn diag_stream_bit_cost(tokens: &[Token], ob: u32, lb: u32) -> u64 {
    let br_bits = backref_total_bits(ob, lb) as u64;
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => br_bits,
        Token::End            => END_TOTAL_BITS as u64,
    }).sum()
}

fn log_phase_c_gain(tokens: &[Token], ob: u32, lb: u32, input_len: usize) {
    if ob <= DIAG_BASELINE_OB && lb <= DIAG_BASELINE_LB { return; }
    let input_bits    = input_len as f64 * 8.0;
    let actual_cost   = diag_stream_bit_cost(tokens, ob, lb);
    let baseline_cost = diag_stream_bit_cost(tokens, DIAG_BASELINE_OB, DIAG_BASELINE_LB);
    let actual_pct    = actual_cost   as f64 / input_bits * 100.0;
    let baseline_pct  = baseline_cost as f64 / input_bits * 100.0;
    let gain_pp       = baseline_pct - actual_pct;
    println!(
        "  DIAG phase_c_gain: ob={} lb={} \
         actual_ratio={:.4}% baseline_hyp_ratio={:.4}% \
         gain_pp={:.4}pp input_bytes={}",
        ob, lb, actual_pct, baseline_pct, gain_pp, input_len,
    );
}

/// Orchestrate fold passes. Returns:
///   (compressed_bytes, folds_done, used_pairing,
///    offset_bits_per_fold, length_bits_per_fold, fold1_tokens)
///
/// When folds_done == 0, compressed_bytes is the unmodified input (passthrough).
/// This happens either because fold 1 didn't improve, or because scan_adaptive
/// signalled the data is incompressible.
///
/// filter_flag: the filter applied before folding (0 = FILTER_NONE).
/// When filter_flag != 0, incompressible bail is suppressed in scan_adaptive,
/// and the fold-1 worthiness threshold is relaxed to accept any LZ improvement.
pub fn fold(input: &[u8], max_folds: u8, filter_flag: u8)
    -> std::io::Result<(Vec<u8>, u8, bool, Vec<u32>, Vec<u32>, Option<Vec<Token>>)>
{
    // When a structural filter was applied, the scanner must not bail on
    // apparently-incompressible data. The filter reorganised bytes for LZ;
    // the scanner needs to run to completion to exploit that structure.
    // Also relax the fold-1 worthiness threshold: any LZ improvement is
    // acceptable when a filter has already transformed the data.
    let skip_incompressible_bail = filter_flag != 0u8;
    // Threshold: for filtered data accept ratio < 1.0 (any reduction).
    //            for unfiltered data require meaningful gain (< 0.985).
    let fold1_min_improvement = if skip_incompressible_bail { 1.0_f64 } else { MIN_IMPROVEMENT_RATIO };

    let mut current            = input.to_vec();
    let mut folds_done: u8     = 0;
    let mut prev_size          = input.len() * 8;
    let original_size          = input.len() * 8;
    let mut final_used_pairing = false;
    let mut fold1_tokens: Option<Vec<Token>> = None;

    let mut offset_bits_per_fold: Vec<u32> = Vec::with_capacity(max_folds as usize);
    let mut length_bits_per_fold: Vec<u32> = Vec::with_capacity(max_folds as usize);

    let mut current_ob: u32 = OFFSET_BITS_MIN;
    let mut current_lb: u32 = LENGTH_BITS_MIN;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        let current_ratio = current.len() as f64 * 8.0 / original_size as f64;

        // ── Fold 1: adaptive scan on raw input ───────────────────────────────
        if fold_num == 1 {
            let (tokens, ob, lb, definitely_incompressible) =
                scan_adaptive(&current, skip_incompressible_bail);

            if definitely_incompressible {
                // Scanner detected expansion. For unfiltered data this means
                // passthrough. For filtered data this path is unreachable since
                // skip_incompressible_bail=true prevents the bail — but keep
                // the check for completeness.
                println!(
                    "Fold 1: incompressible signal — passthrough ({} bytes stored as-is)",
                    current.len()
                );
                folds_done = 0;
                break;
            }

            log_window_diagnostics(&tokens, ob, lb);
            log_phase_c_gain(&tokens, ob, lb, current.len());

            println!(
                "Fold 1 adaptive: offset_bits={} (window={} bytes) length_bits={} (max_len={})",
                ob, (1u32 << ob) - 1, lb, (1u32 << lb) - 1
            );

            let folded      = write_tokens(&tokens, ob, lb)?;
            let folded_bits = folded.len() * 8;
            println!("Fold 1 (LZ): {} bits ({} bytes)", folded_bits, folded.len());

            let ratio = folded_bits as f64 / prev_size as f64;
            if ratio >= fold1_min_improvement {
                // For unfiltered data: not worth it (< 1.5% improvement).
                // For filtered data: only skip if LZ actually expanded (>= 1.0).
                println!("Fold 1 not worth it (ratio {:.3}), stopping at fold 0", ratio);
                break;
            }
            if folded_bits <= MIN_FOLD_BITS {
                println!("Hit minimum size floor at fold 1");
                current_ob = ob;
                current_lb = lb;
                current    = folded;
                folds_done = 1;
                offset_bits_per_fold.push(ob);
                length_bits_per_fold.push(lb);
                fold1_tokens = Some(tokens);
                break;
            }

            current_ob   = ob;
            current_lb   = lb;
            current      = folded;
            folds_done   = 1;
            prev_size    = folded_bits;
            fold1_tokens = Some(tokens);
            offset_bits_per_fold.push(ob);
            length_bits_per_fold.push(lb);
            continue;
        }

        // ── Fold 2+: strategy selection ───────────────────────────────────────
        //
        // allow_lz_on_packed: fold-1 produced very small output (< 10% of
        // original). Running LZ again on the packed token stream can extract
        // additional structure. Only worth it at extreme compression.
        let allow_lz_on_packed = current_ratio < FOLD2_LZ_MAX_RATIO;

        // consider_pair: fold-2 on fold-1 tokens using EG pair encoding.
        // No Cantor fallback rate check — EG always produces valid output.
        // If pair expands, the ratio check below stops it and we fall through to LZ.
        let consider_pair = fold_num == 2
            && folds_done == 1
            && current.len() >= MIN_PAIR_BYTES;

        // ── Try pair encoding first (fold 2 only) ────────────────────────────
        if consider_pair {
            let tokens = fold1_tokens.as_ref()
                .expect("fold1_tokens must be Some when folds_done == 1");
            let pair_result = pair_encode(tokens, current_ob, current_lb)?;
            let pair_bits   = pair_result.len() * 8;
            let pair_ratio  = pair_bits as f64 / prev_size as f64;

            println!(
                "Fold {} (PAIR/EG): {} bits ({} bytes)",
                fold_num, pair_bits, pair_result.len()
            );

            if pair_ratio < MIN_IMPROVEMENT_RATIO {
                // Pair encoding improved things — accept.
                if pair_bits <= MIN_FOLD_BITS {
                    println!("Hit minimum size floor at fold {} (PAIR)", fold_num);
                    current_ob = 0;
                    current_lb = 0;
                    current    = pair_result;
                    folds_done = fold_num;
                    offset_bits_per_fold.push(0);
                    length_bits_per_fold.push(0);
                    final_used_pairing = true;
                    break;
                }
                current_ob = 0;
                current_lb = 0;
                current    = pair_result;
                folds_done = fold_num;
                prev_size  = pair_bits;
                offset_bits_per_fold.push(0);
                length_bits_per_fold.push(0);
                final_used_pairing = true;
                continue;
            }

            // Pair didn't help (expanded or negligible improvement).
            // Fall through to LZ check — don't break yet.
            println!(
                "Fold {} PAIR not worth it (ratio {:.3}) — checking LZ fallback",
                fold_num, pair_ratio
            );
        }

        // ── Try LZ (fold 3+, or fold 2 LZ fallback after pair fails) ─────────
        if !allow_lz_on_packed {
            if !consider_pair {
                println!(
                    "Fold {} skipped — LZ on packed bytes not beneficial \
                     (current ratio {:.3}, threshold {:.2})",
                    fold_num, current_ratio, FOLD2_LZ_MAX_RATIO
                );
            } else {
                println!(
                    "Fold {} PAIR failed and LZ not applicable \
                     (ratio {:.3} >= threshold {:.2}) — stopping at fold {}",
                    fold_num, current_ratio, FOLD2_LZ_MAX_RATIO, folds_done
                );
            }
            break;
        }

        // LZ scan on the current (fold-1 packed) bytes.
        // Fold 2+ always runs with skip_incompressible_bail=false — the packed
        // token stream is not filtered data and normal bail logic applies.
        let (lz_tokens, lz_ob, lz_lb, definitely_incompressible) = scan_adaptive(&current, false);
        if definitely_incompressible {
            println!(
                "Fold {}: LZ scan incompressible signal — stopping at fold {}",
                fold_num, folds_done
            );
            break;
        }

        let encoded     = write_tokens(&lz_tokens, lz_ob, lz_lb)?;
        let folded_bits = encoded.len() * 8;
        println!(
            "Fold {} (LZ): {} bits ({} bytes)",
            fold_num, folded_bits, encoded.len()
        );

        let ratio = folded_bits as f64 / prev_size as f64;
        if ratio >= MIN_IMPROVEMENT_RATIO {
            println!(
                "Fold {} LZ not worth it (ratio {:.3}), stopping at fold {}",
                fold_num, ratio, folds_done
            );
            break;
        }

        if folded_bits <= MIN_FOLD_BITS {
            println!("Hit minimum size floor at fold {} (LZ)", fold_num);
            current_ob = lz_ob;
            current_lb = lz_lb;
            current    = encoded;
            folds_done = fold_num;
            offset_bits_per_fold.push(lz_ob);
            length_bits_per_fold.push(lz_lb);
            break;
        }

        current_ob = lz_ob;
        current_lb = lz_lb;
        current    = encoded;
        folds_done = fold_num;
        prev_size  = folded_bits;
        offset_bits_per_fold.push(lz_ob);
        length_bits_per_fold.push(lz_lb);
    }

    Ok((current, folds_done, final_used_pairing, offset_bits_per_fold, length_bits_per_fold, fold1_tokens))
}
