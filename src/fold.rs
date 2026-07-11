// src/fold.rs
//
// P6 changes vs previous version:
//   - write_tokens now returns (Vec<u8>, bool); destructure everywhere.
//   - ring_was_used captured from fold 1's write_tokens result and propagated
//     as the 7th element of fold()'s return tuple.
//   - Before pair_encode: resolve RepRef → Backref (pair_encode doesn't handle RepRef).
//   - diag_stream_bit_cost / log_window_diagnostics handle Token::RepRef.

use crate::encoder::scan_adaptive;
use crate::bitwriter::write_tokens;
use crate::pairing::pair_encode;
use crate::opcode::{
    self,
    Token, OFFSET_BITS_MIN, LENGTH_BITS_MIN,
    LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits, repref_total_bits,
};

const MIN_IMPROVEMENT_RATIO: f64   = 0.985;
const MIN_FOLD_BITS:         usize = 64;
const MIN_PAIR_BYTES:        usize = 512;
const FOLD2_LZ_MAX_RATIO:    f64   = 0.10;

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
            Token::RepRef { length, .. } => {
                // Ring reference: counts as a back-reference for saturation
                // diagnostics, but has no explicit offset to check.
                total_br += 1;
                if *length == max_len { at_max_len += 1; }
            }
            Token::Lit { .. } => total_lit += 1,
            Token::End => {}
        }
    }

    if total_br + total_lit == 0 { return; }
    let total    = total_br + total_lit;
    let br_pct   = total_br as f64 / total as f64 * 100.0;
    let sat_pct  = if total_br > 0 { at_max_off as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let deep_pct = if total_br > 0 { above_half as f64 / total_br as f64 * 100.0 } else { 0.0 };
    let len_pct  = if total_br > 0 { at_max_len as f64 / total_br as f64 * 100.0 } else { 0.0 };

    println!(
        "Window diagnostics: {}/{} tokens BACKREF/REPREF ({:.1}%) | \
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
    let rr_bits = repref_total_bits(lb) as u64;
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => br_bits,
        Token::RepRef { .. }  => rr_bits,
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

/// Orchestrate fold passes.
///
/// Returns:
///   (compressed_bytes, folds_done, used_pairing,
///    offset_bits_per_fold, length_bits_per_fold, fold1_tokens, ring_was_used)
///
/// `ring_was_used` is true when fold 1's LZ bitstream contains RepRef tokens
/// (ring-active encoding). The caller stores this in the header (pair_flag bit 1)
/// so the decompressor knows to parse fold 1's bitstream with ring opcodes.
///
/// `filter_flag`: when non-zero a structural filter was applied; suppresses
/// incompressible-bail in scan_adaptive and relaxes fold-1 worthiness threshold.
pub fn fold(input: &[u8], max_folds: u8, filter_flag: u8)
    -> std::io::Result<(Vec<u8>, u8, bool, Vec<u32>, Vec<u32>, Option<Vec<Token>>, bool)>
{
    let skip_incompressible_bail = filter_flag != 0u8;
    let fold1_min_improvement = if skip_incompressible_bail { 1.0_f64 } else { MIN_IMPROVEMENT_RATIO };

    let mut current            = input.to_vec();
    let mut folds_done: u8     = 0;
    let mut prev_size          = input.len() * 8;
    let original_size          = input.len() * 8;
    let mut final_used_pairing = false;
    let mut fold1_tokens: Option<Vec<Token>> = None;
    let mut ring_was_used      = false; // P6: set when fold 1 emits RepRef tokens

    let mut offset_bits_per_fold: Vec<u32> = Vec::with_capacity(max_folds as usize);
    let mut length_bits_per_fold: Vec<u32> = Vec::with_capacity(max_folds as usize);

    let mut current_ob: u32 = OFFSET_BITS_MIN;
    let mut current_lb: u32 = LENGTH_BITS_MIN;

    println!("Original size: {} bits ({} bytes)", prev_size, input.len());

    for fold_num in 1..=max_folds {
        let current_ratio = current.len() as f64 * 8.0 / original_size as f64;

        // ── Fold 1: adaptive scan on raw input ───────────────────────────────
        if fold_num == 1 {
            let (mut tokens, mut ob, lb, definitely_incompressible) =
                scan_adaptive(&current, skip_incompressible_bail);

            if definitely_incompressible {
                println!(
                    "Fold 1: incompressible signal — passthrough ({} bytes stored as-is)",
                    current.len()
                );
                folds_done = 0;
                break;
            }

            // Try the static-dictionary-seeded candidate too (dictionary.rs) --
            // same "measure both, keep whichever is actually smaller" pattern as
            // everything else in this pipeline. Only ever replaces the normal
            // scan if it genuinely wins; guarded off if any resulting match
            // length wouldn't fit the already-chosen length_bits (conservative:
            // skip the candidate rather than widen length_bits for it).
            {
                let dict_ob = crate::encoder::min_offset_bits_for_dict(current.len()).max(ob);
                let (dict_tokens, dict_bailed) =
                    crate::encoder::scan_with_dict(&current, dict_ob, lb, true, true);
                let lengths_fit = dict_tokens.iter().all(|t| match t {
                    Token::Backref { length, .. } | Token::RepRef { length, .. } =>
                        (*length as usize) <= crate::opcode::max_length(lb),
                    _ => true,
                });
                if !dict_bailed && !dict_tokens.is_empty() && lengths_fit {
                    if let (Ok((dict_folded, _)), Ok((normal_folded, _))) = (
                        write_tokens(&dict_tokens, dict_ob, lb),
                        write_tokens(&tokens, ob, lb),
                    ) {
                        if dict_folded.len() < normal_folded.len() {
                            println!(
                                "Dictionary candidate wins: {} B < {} B (raw, offset_bits {} -> {})",
                                dict_folded.len(), normal_folded.len(), ob, dict_ob
                            );
                            tokens = dict_tokens;
                            ob     = dict_ob;
                        }
                    }
                }
            }

            log_window_diagnostics(&tokens, ob, lb);
            log_phase_c_gain(&tokens, ob, lb, current.len());

            println!(
                "Fold 1 adaptive: offset_bits={} (window={} bytes) length_bits={} (max_len={})",
                ob, (1u32 << ob) - 1, lb, (1u32 << lb) - 1
            );

            // P6: write_tokens returns (bytes, ring_active).
            let (folded, fold1_ring_active) = write_tokens(&tokens, ob, lb)?;
            let folded_bits = folded.len() * 8;
            println!("Fold 1 (LZ): {} bits ({} bytes)", folded_bits, folded.len());

            let ratio = folded_bits as f64 / prev_size as f64;
            if ratio >= fold1_min_improvement {
                println!("Fold 1 not worth it (ratio {:.3}), stopping at fold 0", ratio);
                break;
            }
            if folded_bits <= MIN_FOLD_BITS {
                println!("Hit minimum size floor at fold 1");
                current_ob    = ob;
                current_lb    = lb;
                current       = folded;
                folds_done    = 1;
                ring_was_used = fold1_ring_active; // P6
                offset_bits_per_fold.push(ob);
                length_bits_per_fold.push(lb);
                fold1_tokens = Some(tokens);
                break;
            }

            current_ob    = ob;
            current_lb    = lb;
            current       = folded;
            folds_done    = 1;
            prev_size     = folded_bits;
            ring_was_used = fold1_ring_active; // P6
            fold1_tokens  = Some(tokens);
            offset_bits_per_fold.push(ob);
            length_bits_per_fold.push(lb);
            continue;
        }

        // ── Fold 2+: strategy selection ───────────────────────────────────────
        let allow_lz_on_packed = current_ratio < FOLD2_LZ_MAX_RATIO;

        let consider_pair = fold_num == 2
            && folds_done == 1
            && current.len() >= MIN_PAIR_BYTES;

        // ── Try pair encoding (fold 2 only) ───────────────────────────────────
        if consider_pair {
            let raw_tokens = fold1_tokens.as_ref()
                .expect("fold1_tokens must be Some when folds_done == 1");

            // P6: resolve RepRef → Backref before pair_encode, which doesn't
            // handle RepRef tokens.
            let resolved;
            let tokens_for_pair: &[Token] = if ring_was_used {
                resolved = opcode::resolve_ring(raw_tokens);
                &resolved
            } else {
                raw_tokens.as_slice()
            };

            let pair_result = pair_encode(tokens_for_pair, current_ob, current_lb)?;
            let pair_bits   = pair_result.len() * 8;
            let pair_ratio  = pair_bits as f64 / prev_size as f64;

            println!(
                "Fold {} (PAIR/EG): {} bits ({} bytes)",
                fold_num, pair_bits, pair_result.len()
            );

            if pair_ratio < MIN_IMPROVEMENT_RATIO {
                if pair_bits <= MIN_FOLD_BITS {
                    println!("Hit minimum size floor at fold {} (PAIR)", fold_num);
                    current_ob = 0; current_lb = 0;
                    current    = pair_result;
                    folds_done = fold_num;
                    offset_bits_per_fold.push(0);
                    length_bits_per_fold.push(0);
                    final_used_pairing = true;
                    break;
                }
                current_ob = 0; current_lb = 0;
                current    = pair_result;
                folds_done = fold_num;
                prev_size  = pair_bits;
                offset_bits_per_fold.push(0);
                length_bits_per_fold.push(0);
                final_used_pairing = true;
                continue;
            }

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

        // Fold 2+ always runs with skip_incompressible_bail=false — the packed
        // token stream is not filtered data and normal bail logic applies.
        let (lz_tokens, lz_ob, lz_lb, definitely_incompressible) =
            scan_adaptive(&current, false);
        if definitely_incompressible {
            println!(
                "Fold {}: LZ scan incompressible signal — stopping at fold {}",
                fold_num, folds_done
            );
            break;
        }

        // P6: fold 2+ bitstream must never be ring-encoded (scanning packed
        // bytes) -- but scan_adaptive's skip_incompressible_bail=false above
        // also sets final_emit_repref=true internally, so it CAN legitimately
        // emit RepRef tokens here. The decoder unconditionally assumes fold 2+
        // is never ring-active (read_ring_active is gated to folds_done==1
        // only), so any RepRef that slips through desyncs the opcode set on
        // decode. resolve_ring() enforces the documented invariant directly:
        // convert any RepRef back to its equivalent Backref before write_tokens
        // ever sees it, so ring_active is guaranteed false for fold 2+, always.
        let lz_tokens = opcode::resolve_ring(&lz_tokens);
        let (encoded, ring_used_fold2) = write_tokens(&lz_tokens, lz_ob, lz_lb)?;
        debug_assert!(!ring_used_fold2, "fold 2+ must never be ring-encoded");
        let folded_bits  = encoded.len() * 8;
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
            current_ob = lz_ob; current_lb = lz_lb;
            current    = encoded;
            folds_done = fold_num;
            offset_bits_per_fold.push(lz_ob);
            length_bits_per_fold.push(lz_lb);
            break;
        }

        current_ob = lz_ob; current_lb = lz_lb;
        current    = encoded;
        folds_done = fold_num;
        prev_size  = folded_bits;
        offset_bits_per_fold.push(lz_ob);
        length_bits_per_fold.push(lz_lb);
    }

    Ok((current, folds_done, final_used_pairing,
        offset_bits_per_fold, length_bits_per_fold, fold1_tokens, ring_was_used))
        }
