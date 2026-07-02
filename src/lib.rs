// src/lib.rs
//
// P10 addition: v7 (adaptive binary range coder) added to entropy tournament.
//   tokens_safe_for_v7() checks all Backref lengths <= RC_MAX_BACKREF_LEN (272).
//   try_entropy_v7() requires no pre-built tables -- the range coder adapts
//   its own probability model during encoding.
//   v7 competes in the same rayon tournament as v1-v6; it wins on
//   near-degenerate distributions where Huffman's integer-bit floor is visible.

pub mod opcode;
pub mod encoder;
pub mod bitwriter;
pub mod bitreader;
pub mod decoder;
pub mod pairing;
pub mod fold;
pub mod unfold;
pub mod entropy;
pub mod filters;
pub mod archive;
pub mod archive_io;
pub mod platform;

use std::io;
use std::sync::Arc;
use rayon::prelude::*;

const INCOMPRESSIBLE_ENTROPY_THRESHOLD: f64 = 7.8;
const INCOMPRESSIBLE_MIN_BYTES:         usize = 50_000;

fn sample_entropy(data: &[u8]) -> f64 {
    const SAMPLE_SIZE: usize = 8192;
    let sample = if data.len() > SAMPLE_SIZE { &data[..SAMPLE_SIZE] } else { data };
    let mut freq = [0u32; 256];
    for &b in sample { freq[b as usize] += 1; }
    let n = sample.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
}

/// File header layout (bytes 0-3 + N ob + N lb):
///   Byte 0: fold_count
///   Byte 1: pair_flag byte
///              bit 0 -- fold 2 used pair encoding (bool)
///              bit 1 -- fold 1 LZ bitstream uses ring-active opcodes (P6)
///   Byte 2: entropy_flag (0=none, 1-7=variant)
///   Byte 3: filter_flag  (0=none, 1-4=delta, 7=STL, 8=PLY, 9=BCJ, 10-15=BCJ variants)
///   Bytes 4..4+N:    offset_bits[0..N]  N = fold_count
///   Bytes 4+N..4+2N: length_bits[0..N]
///   Remaining: compressed payload
///
/// fold_count=0: passthrough -- payload is original uncompressed bytes.
///
/// entropy_flag=7: v7 range coder.
///   Payload is the raw range-coder byte stream.
///   No Huffman tables are stored -- the decoder reconstructs the
///   probability model from the same initial state (all probs = 1024).
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    // ── Pre-filter ────────────────────────────────────────────────────────────
    let filter_flag = filters::detect_filter(input);

    let filter_buf: Option<Vec<u8>> = if filter_flag != filters::FILTER_NONE {
        let f = filters::apply_filter(input, filter_flag);
        println!(
            "Filter flag={}: {} bytes in, {} bytes filtered",
            filter_flag, input.len(), f.len()
        );
        Some(f)
    } else {
        None
    };

    let to_fold: &[u8] = filter_buf.as_deref().unwrap_or(input);

    // ── Incompressible early exit ─────────────────────────────────────────────
    let skip_entropy_gate = filter_flag != filters::FILTER_NONE;

    if !skip_entropy_gate && input.len() > INCOMPRESSIBLE_MIN_BYTES {
        let ent = sample_entropy(to_fold);
        if ent > INCOMPRESSIBLE_ENTROPY_THRESHOLD {
            println!(
                "Incompressible early exit: entropy={:.3} bits/byte, {} bytes -- passthrough",
                ent, input.len()
            );
            let mut out = Vec::with_capacity(4 + input.len());
            out.push(0u8); out.push(0u8); out.push(0u8);
            out.push(filters::FILTER_NONE);
            out.extend_from_slice(input);
            return Ok(out);
        }
    }

    // ── Fold passes ───────────────────────────────────────────────────────────
    let (compressed, folds_done, used_pairing,
         offset_bits_per_fold, length_bits_per_fold,
         fold1_tokens_opt, ring_was_used) =
        fold::fold(to_fold, max_folds, filter_flag)?;

    let ob1 = offset_bits_per_fold.first().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);
    let lb1 = length_bits_per_fold.first().copied()
        .unwrap_or(opcode::LENGTH_BITS_MIN);
    let final_ob = offset_bits_per_fold.last().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_lb = length_bits_per_fold.last().copied()
        .unwrap_or(opcode::LENGTH_BITS_MIN);

    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag, out_pair_flag, out_folds, out_ob, out_lb) =
        if try_entropy_standard {
            // P6: the stored bitstream may use ring-active encoding if fold 1
            // is the final fold. Read with ring_active so RepRef tokens decode
            // correctly, then resolve -> Backref before entropy functions.
            let read_ring_active = ring_was_used && folds_done == 1;
            let tokens_raw =
                bitreader::read_tokens(&compressed, final_ob, final_lb, read_ring_active)?;
            let tokens = if read_ring_active {
                opcode::resolve_ring(&tokens_raw)
            } else {
                tokens_raw
            };

            let raw_size   = compressed.len();
            let entropy_ok = tokens_safe_for_entropy(&tokens);
            let v2_ok      = entropy_ok && raw_size >= entropy::ENTROPY_V2_MIN_BYTES;
            // v7 gate: all Backref lengths must be <= RC_MAX_BACKREF_LEN (272)
            let v7_ok      = tokens_safe_for_v7(&tokens);

            let lit_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
                entropy::build_encode_table(&tokens).map(Arc::new)
            } else { None };

            let ctx_arc: Option<(Arc<entropy::EncodeTable>, Arc<entropy::EncodeTable>)> =
                if v2_ok {
                    entropy::build_encode_tables_by_context(&tokens)
                        .map(|(t0, t1)| (Arc::new(t0), Arc::new(t1)))
                } else { None };

            let off_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
                entropy::build_offset_encode_table(&tokens).map(Arc::new)
            } else { None };

            let off_slot_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
                entropy::build_offset_encode_table_slotted(&tokens).map(Arc::new)
            } else { None };

            let v6_arc: Option<(
                Arc<entropy::EncodeTable>,
                Arc<entropy::EncodeTable>,
                Arc<entropy::EncodeTable>,
            )> = if entropy_ok {
                entropy::build_v6_tables(&tokens)
                    .map(|(lt, st, ot)| (Arc::new(lt), Arc::new(st), Arc::new(ot)))
            } else { None };

            // v7 shares the token vec via Arc -- no table building needed
            let tokens_arc = Arc::new(tokens);
            let ta = Arc::clone(&tokens_arc);

            let results: Vec<(u8, Option<Vec<u8>>)> = vec![1u8, 2, 3, 4, 5, 6, 7]
                .into_par_iter()
                .map(|flag| {
                    let tokens = &*ta;
                    let payload = match flag {
                        1 => match (&lit_arc, &off_arc) {
                            (Some(lt), Some(ot)) if entropy_ok =>
                                encode_v1_shared(tokens, lt, ot),
                            _ => None,
                        },
                        2 => match (&ctx_arc, &off_arc) {
                            (Some((t0, t1)), Some(ot)) if v2_ok =>
                                encode_v2_shared(tokens, t0, t1, ot),
                            _ => None,
                        },
                        3 => if v2_ok { try_entropy_v3(tokens) } else { None },
                        4 => match (&lit_arc, &off_slot_arc) {
                            (Some(lt), Some(ost)) if entropy_ok =>
                                encode_v4_shared(tokens, lt, ost),
                            _ => None,
                        },
                        5 => match (&ctx_arc, &off_slot_arc) {
                            (Some((t0, t1)), Some(ost)) if v2_ok =>
                                encode_v5_shared(tokens, t0, t1, ost),
                            _ => None,
                        },
                        6 => match &v6_arc {
                            Some((lt, st, ot)) => encode_v6_shared(tokens, lt, st, ot),
                            None => None,
                        },
                        7 => if v7_ok { try_entropy_v7(tokens) } else { None },
                        _ => None,
                    };
                    (flag, payload)
                })
                .collect();

            let sz = |o: &Option<Vec<u8>>| o.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);

            {
                let detail: Vec<String> = results.iter()
                    .map(|(f, o)| {
                        let s = sz(o);
                        if s == usize::MAX { format!("v{}=skip", f) }
                        else               { format!("v{}={}B", f, s) }
                    })
                    .collect();
                println!("Entropy variant sizes (raw={}B): {}", raw_size, detail.join(" "));
            }

            let best_size = results.iter().map(|(_, o)| sz(o)).min().unwrap_or(usize::MAX);

            if best_size >= raw_size {
                println!("Joint entropy skipped (no gain vs raw={}B)", raw_size);
                // Drop Arc before returning compressed
                drop(tokens_arc);
                (compressed, 0u8, false, folds_done,
                 offset_bits_per_fold, length_bits_per_fold)
            } else {
                drop(tokens_arc);
                let (flag, payload) = results.into_iter()
                    .filter_map(|(f, opt)| opt.map(|p| (f, p)))
                    .min_by_key(|(_, p)| p.len())
                    .unwrap();
                println!("Joint entropy flag={}: {} -> {} B", flag, raw_size, payload.len());
                (payload, flag, false, folds_done,
                 offset_bits_per_fold, length_bits_per_fold)
            }
        } else if used_pairing {
            let f1t = fold1_tokens_opt
                .expect("fold1_tokens must be Some when used_pairing is true");

            let f1t_resolved = if ring_was_used {
                opcode::resolve_ring(&f1t)
            } else {
                f1t
            };

            pair_vs_entropy(
                f1t_resolved, ob1, lb1, &compressed,
                folds_done, &offset_bits_per_fold, &length_bits_per_fold,
            )?
        } else {
            (compressed, 0u8, false, folds_done,
             offset_bits_per_fold, length_bits_per_fold)
        };

    // ── Serialise header ──────────────────────────────────────────────────────
    let final_ring_flag = ring_was_used;
    let pair_flag_byte  = (out_pair_flag as u8) | ((final_ring_flag as u8) << 1);

    let mut output = Vec::with_capacity(
        4 + 2 * out_folds as usize + final_payload.len()
    );
    output.push(out_folds);
    output.push(pair_flag_byte);
    output.push(entropy_flag);
    output.push(filter_flag);
    for &ob in &out_ob { output.push(ob as u8); }
    for &lb in &out_lb { output.push(lb as u8); }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

// ── Safety checks ─────────────────────────────────────────────────────────────

#[inline]
fn tokens_safe_for_entropy(tokens: &[opcode::Token]) -> bool {
    tokens.iter().all(|t| match t {
        opcode::Token::Backref { length, .. } |
        opcode::Token::RepRef  { length, .. } => *length <= opcode::ENTROPY_SAFE_MAX_LENGTH,
        _ => true,
    })
}

/// v7 gate: RC_MAX_BACKREF_LEN = 272 is the max encodable length.
/// Lengths 273+ use HI-tier value 255 which is reserved for the End sentinel.
#[inline]
fn tokens_safe_for_v7(tokens: &[opcode::Token]) -> bool {
    tokens.iter().all(|t| match t {
        opcode::Token::Backref { length, .. } => *length <= entropy::RC_MAX_BACKREF_LEN,
        _ => true,
    })
}

// ── Entropy variant encoders ──────────────────────────────────────────────────

fn encode_v1_shared(
    tokens:       &[opcode::Token],
    lit_table:    &entropy::EncodeTable,
    offset_table: &entropy::EncodeTable,
) -> Option<Vec<u8>> {
    let coded = entropy::write_tokens_v1(tokens, lit_table, offset_table).ok()?;
    let mut payload = entropy::serialize_table(lit_table);
    payload.extend_from_slice(&entropy::serialize_table(offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn encode_v2_shared(
    tokens:       &[opcode::Token],
    t0:           &entropy::EncodeTable,
    t1:           &entropy::EncodeTable,
    offset_table: &entropy::EncodeTable,
) -> Option<Vec<u8>> {
    let coded = entropy::write_tokens_v2(tokens, t0, t1, offset_table).ok()?;
    let mut payload = entropy::serialize_table(t0);
    payload.extend_from_slice(&entropy::serialize_table(t1));
    payload.extend_from_slice(&entropy::serialize_table(offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v3(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (lit_tables, offset_table) = entropy::build_encode_tables_v3(tokens)?;
    let coded = entropy::write_tokens_v3(tokens, &lit_tables, &offset_table).ok()?;
    let mut payload = Vec::new();
    for t in &lit_tables { payload.extend_from_slice(&entropy::serialize_table(t)); }
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn encode_v4_shared(
    tokens:               &[opcode::Token],
    lit_table:            &entropy::EncodeTable,
    offset_table_slotted: &entropy::EncodeTable,
) -> Option<Vec<u8>> {
    let coded = entropy::write_tokens_v4(tokens, lit_table, offset_table_slotted).ok()?;
    let mut payload = entropy::serialize_table(lit_table);
    payload.extend_from_slice(&entropy::serialize_table(offset_table_slotted));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn encode_v5_shared(
    tokens:               &[opcode::Token],
    t0:                   &entropy::EncodeTable,
    t1:                   &entropy::EncodeTable,
    offset_table_slotted: &entropy::EncodeTable,
) -> Option<Vec<u8>> {
    let coded = entropy::write_tokens_v5(tokens, t0, t1, offset_table_slotted).ok()?;
    let mut payload = entropy::serialize_table(t0);
    payload.extend_from_slice(&entropy::serialize_table(t1));
    payload.extend_from_slice(&entropy::serialize_table(offset_table_slotted));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn encode_v6_shared(
    tokens:       &[opcode::Token],
    lit_table:    &entropy::EncodeTable,
    seq_table:    &entropy::EncodeTable,
    offset_table: &entropy::EncodeTable,
) -> Option<Vec<u8>> {
    entropy::write_tokens_v6(tokens, lit_table, seq_table, offset_table).ok()
}

/// v7: no pre-built tables -- encode directly, payload is the raw RC byte stream.
fn try_entropy_v7(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    entropy::write_tokens_v7(tokens).ok()
}

// ── pair_vs_entropy ───────────────────────────────────────────────────────────

fn pair_vs_entropy(
    fold1_tokens:     Vec<opcode::Token>,
    ob1:              u32,
    lb1:              u32,
    pair_output:      &[u8],
    pair_folds_done:  u8,
    pair_ob_per_fold: &[u32],
    pair_lb_per_fold: &[u32],
) -> io::Result<(Vec<u8>, u8, bool, u8, Vec<u32>, Vec<u32>)> {
    let fold1_bytes_est = {
        let bits: u32 = fold1_tokens.iter()
            .map(|t| opcode::token_bit_cost(t, ob1, lb1))
            .sum();
        ((bits + 7) / 8) as usize
    };

    if fold1_bytes_est < entropy::ENTROPY_MIN_BYTES {
        println!(
            "PAIR kept (fold 1 est. {} B < entropy threshold {} B)",
            fold1_bytes_est, entropy::ENTROPY_MIN_BYTES
        );
        return Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done,
            pair_ob_per_fold.to_vec(),
            pair_lb_per_fold.to_vec(),
        ));
    }

    let entropy_ok = tokens_safe_for_entropy(&fold1_tokens);
    let v2_ok      = entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V2_MIN_BYTES;
    let v7_ok      = tokens_safe_for_v7(&fold1_tokens);

    let lit_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
        entropy::build_encode_table(&fold1_tokens).map(Arc::new)
    } else { None };

    let ctx_arc: Option<(Arc<entropy::EncodeTable>, Arc<entropy::EncodeTable>)> = if v2_ok {
        entropy::build_encode_tables_by_context(&fold1_tokens)
            .map(|(t0, t1)| (Arc::new(t0), Arc::new(t1)))
    } else { None };

    let off_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
        entropy::build_offset_encode_table(&fold1_tokens).map(Arc::new)
    } else { None };

    let off_slot_arc: Option<Arc<entropy::EncodeTable>> = if entropy_ok {
        entropy::build_offset_encode_table_slotted(&fold1_tokens).map(Arc::new)
    } else { None };

    let v6_arc: Option<(
        Arc<entropy::EncodeTable>,
        Arc<entropy::EncodeTable>,
        Arc<entropy::EncodeTable>,
    )> = if entropy_ok {
        entropy::build_v6_tables(&fold1_tokens)
            .map(|(lt, st, ot)| (Arc::new(lt), Arc::new(st), Arc::new(ot)))
    } else { None };

    let tokens_arc = Arc::new(fold1_tokens);
    let ta = Arc::clone(&tokens_arc);

    let results: Vec<(u8, Option<Vec<u8>>)> = vec![1u8, 2, 3, 4, 5, 6, 7]
        .into_par_iter()
        .map(|flag| {
            let tokens = &*ta;
            let payload = match flag {
                1 => match (&lit_arc, &off_arc) {
                    (Some(lt), Some(ot)) if entropy_ok =>
                        encode_v1_shared(tokens, lt, ot),
                    _ => None,
                },
                2 => match (&ctx_arc, &off_arc) {
                    (Some((t0, t1)), Some(ot)) if v2_ok =>
                        encode_v2_shared(tokens, t0, t1, ot),
                    _ => None,
                },
                3 => if v2_ok { try_entropy_v3(tokens) } else { None },
                4 => match (&lit_arc, &off_slot_arc) {
                    (Some(lt), Some(ost)) if entropy_ok =>
                        encode_v4_shared(tokens, lt, ost),
                    _ => None,
                },
                5 => match (&ctx_arc, &off_slot_arc) {
                    (Some((t0, t1)), Some(ost)) if v2_ok =>
                        encode_v5_shared(tokens, t0, t1, ost),
                    _ => None,
                },
                6 => match &v6_arc {
                    Some((lt, st, ot)) => encode_v6_shared(tokens, lt, st, ot),
                    None => None,
                },
                7 => if v7_ok { try_entropy_v7(tokens) } else { None },
                _ => None,
            };
            (flag, payload)
        })
        .collect();

    {
        let detail: Vec<String> = results.iter()
            .map(|(f, o)| match o {
                Some(p) => format!("v{}={}B", f, p.len()),
                None    => format!("v{}=skip", f),
            })
            .collect();
        println!(
            "Entropy variant sizes (fold1 est={}B): {}",
            fold1_bytes_est, detail.join(" ")
        );
    }

    drop(tokens_arc);

    let best_entropy = results.into_iter()
        .filter_map(|(f, opt)| opt.map(|p| (f, p)))
        .min_by_key(|(_, p)| p.len());

    match best_entropy {
        Some((flag, payload)) if payload.len() < pair_output.len() => {
            println!(
                "Entropy flag={} beats PAIR: {} B < {} B",
                flag, payload.len(), pair_output.len()
            );
            Ok((payload, flag, false, 1u8, vec![ob1], vec![lb1]))
        }
        _ => {
            println!("PAIR wins over entropy: {} B", pair_output.len());
            Ok((
                pair_output.to_vec(), 0u8, true,
                pair_folds_done,
                pair_ob_per_fold.to_vec(),
                pair_lb_per_fold.to_vec(),
            ))
        }
    }
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
        }
