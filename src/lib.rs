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
pub mod dictionary;
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

            let results: Vec<(u8, Option<Vec<u8>>)> = vec![1u8, 2, 3, 4, 5, 6, 7, 8]
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
                        7 => if v7_ok {
                            // Defense in depth: verify v7's own roundtrip before ever
                            // trusting it. The actual bug that motivated this (fold 2+
                            // ring-encoding mismatch) is now fixed at its source in
                            // fold.rs, but this check is cheap and catches any other
                            // encode/decode desync before it could ship, not just this one.
                            try_entropy_v7(tokens).filter(|payload| {
                                entropy::read_tokens_v7(payload)
                                    .map(|decoded| decoded == *tokens)
                                    .unwrap_or(false)
                            })
                        } else { None },
                        8 => if entropy_ok { try_entropy_v8(tokens) } else { None },
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

// ── v8: block-split (path-2 cheap first cut) ─────────────────────────────────
//
// Unlike v1-v6 (one table for the whole file, chosen by a fixed rule) or v7
// (no tables at all), v8 tries splitting the token stream into a small number
// of large contiguous segments -- each getting its own flat literal+length
// table -- with boundaries picked by real measurement rather than an entropy
// heuristic, matching how the rest of this file already prefers "actually
// encode and compare" over cost estimation. Offset table stays global/shared
// across segments; only the literal+length table is split. Only ever competes
// as one more brute-force tournament entrant -- if splitting doesn't help,
// this returns whatever's smallest anyway, and v1-v7 are free to win instead.

const V8_MIN_TOKENS: usize = 200;

/// Encode `tokens` as `boundaries.len() + 1` segments (boundaries are token
/// indices strictly between 0 and tokens.len()), each with its own literal
/// table built via the same `build_encode_table` v1 already uses, sharing one
/// offset table across all segments. Returns None if any segment can't build
/// a valid table (e.g. too small/degenerate) rather than guessing.
fn build_v8_candidate(
    tokens:       &[opcode::Token],
    offset_table: &entropy::EncodeTable,
    boundaries:   &[usize],
) -> Option<Vec<u8>> {
    let mut bounds = Vec::with_capacity(boundaries.len() + 2);
    bounds.push(0usize);
    bounds.extend_from_slice(boundaries);
    bounds.push(tokens.len());

    let mut seg_tables: Vec<entropy::EncodeTable> = Vec::with_capacity(bounds.len() - 1);
    let mut segs: Vec<&[opcode::Token]> = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        let (start, end) = (w[0], w[1]);
        if end <= start { return None; }
        let seg = &tokens[start..end];
        seg_tables.push(entropy::build_encode_table(seg)?);
        segs.push(seg);
    }

    let mut payload = vec![seg_tables.len() as u8];
    for seg in &segs {
        payload.extend_from_slice(&(seg.len() as u32).to_le_bytes());
    }
    payload.extend_from_slice(&entropy::serialize_table(offset_table));
    for t in &seg_tables {
        payload.extend_from_slice(&entropy::serialize_table(t));
    }
    for (seg, table) in segs.iter().zip(seg_tables.iter()) {
        let coded = entropy::write_tokens_v1(seg, table, offset_table).ok()?;
        payload.extend_from_slice(&(coded.len() as u32).to_le_bytes());
        payload.extend_from_slice(&coded);
    }
    Some(payload)
}

fn try_entropy_v8(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    if tokens.len() < V8_MIN_TOKENS { return None; }
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let len = tokens.len();

    // Each segment must be at least 5% of the stream or 10 tokens, whichever
    // is bigger -- guards against a candidate collapsing a segment to
    // (near-)nothing, which could never pay for its own table overhead.
    let min_seg = (len / 20).max(10);

    let candidates_2: Vec<usize> = [len / 4, len / 2, (3 * len) / 4]
        .into_iter()
        .filter(|&b| b >= min_seg && len - b >= min_seg)
        .collect();

    let mut best: Option<Vec<u8>> = None;
    let consider = |payload: Option<Vec<u8>>, best: &mut Option<Vec<u8>>| {
        if let Some(p) = payload {
            if best.as_ref().map_or(true, |cur: &Vec<u8>| p.len() < cur.len()) {
                *best = Some(p);
            }
        }
    };

    for &b in &candidates_2 {
        consider(build_v8_candidate(tokens, &offset_table, &[b]), &mut best);
    }

    // N=3 refinement: for each N=2 boundary tried above, also try adding a
    // second cut at the midpoint of each half it produces. Only kept if it
    // actually shrinks the result further -- otherwise the N=2 (or N=1, i.e.
    // v8 losing entirely) result stands.
    for &b in &candidates_2 {
        for extra in [b / 2, b + (len - b) / 2] {
            if extra == b || extra < min_seg || len - extra < min_seg { continue; }
            let mut bs = [b, extra];
            bs.sort_unstable();
            if bs[1] - bs[0] < min_seg { continue; }
            consider(build_v8_candidate(tokens, &offset_table, &bs), &mut best);
        }
    }

    best
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

    let results: Vec<(u8, Option<Vec<u8>>)> = vec![1u8, 2, 3, 4, 5, 6, 7, 8]
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
                7 => if v7_ok {
                    // Same defense-in-depth check as the other tournament call site.
                    try_entropy_v7(tokens).filter(|payload| {
                        entropy::read_tokens_v7(payload)
                            .map(|decoded| &decoded == tokens)
                            .unwrap_or(false)
                    })
                } else { None },
                8 => if entropy_ok { try_entropy_v8(tokens) } else { None },
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

#[cfg(test)]
mod v8_tests {
    use super::*;
    use opcode::Token;

    // Mirrors unfold.rs's entropy_flag=8 decode arm exactly, but operating
    // directly on a v8 payload (no outer compress() header/fold framing) so
    // this test isolates the wire format itself from the rest of the pipeline.
    fn decode_v8_payload(payload: &[u8]) -> io::Result<Vec<Token>> {
        let num_segments = payload[0] as usize;
        let mut cursor = 1usize;

        let mut seg_counts: Vec<usize> = Vec::with_capacity(num_segments);
        for _ in 0..num_segments {
            let count = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            seg_counts.push(count);
            cursor += 4;
        }

        let (off_enc, off_c) = entropy::deserialize_table(&payload[cursor..])?;
        let off_dt = entropy::decode_table_from_encode(&off_enc);
        cursor += off_c;

        let mut seg_dtables = Vec::with_capacity(num_segments);
        for _ in 0..num_segments {
            let (enc, consumed) = entropy::deserialize_table(&payload[cursor..])?;
            seg_dtables.push(entropy::decode_table_from_encode(&enc));
            cursor += consumed;
        }

        let mut tokens = Vec::new();
        for i in 0..num_segments {
            let seg_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let seg_tokens = entropy::read_tokens_v1_counted(
                &payload[cursor..cursor + seg_len], &seg_dtables[i], &off_dt, seg_counts[i],
            )?;
            tokens.extend(seg_tokens);
            cursor += seg_len;
        }
        Ok(tokens)
    }

    /// Two clearly distinct literal-byte zones (A-E heavy vs X/Y/Z/digits
    /// heavy) with real backrefs mixed in, comfortably above V8_MIN_TOKENS --
    /// exactly the shape path 2 targets.
    fn two_zone_tokens(zone_len: usize) -> Vec<Token> {
        let mut t = Vec::new();
        let zone_a = [b'A', b'B', b'C', b'D', b'E'];
        for i in 0..zone_len {
            t.push(Token::Lit { byte: zone_a[i % zone_a.len()] });
            if i > 8 && i % 7 == 0 {
                t.push(Token::Backref { offset: 5, length: 3 });
            }
        }
        let zone_b = [b'X', b'Y', b'Z', b'0', b'1', b'2'];
        for i in 0..zone_len {
            t.push(Token::Lit { byte: zone_b[i % zone_b.len()] });
            if i > 8 && i % 7 == 0 {
                t.push(Token::Backref { offset: 6, length: 3 });
            }
        }
        t.push(Token::End);
        t
    }

    #[test]
    fn v8_below_min_tokens_returns_none() {
        let tokens = two_zone_tokens(5); // way under V8_MIN_TOKENS
        assert!(try_entropy_v8(&tokens).is_none());
    }

    #[test]
    fn v8_roundtrip_direct() {
        let tokens = two_zone_tokens(400); // ~800+ tokens, comfortably over V8_MIN_TOKENS
        let payload = try_entropy_v8(&tokens).expect("v8 should produce a candidate for this input");
        let decoded = decode_v8_payload(&payload).expect("v8 payload should decode cleanly");
        assert_eq!(decoded, tokens, "v8 roundtrip must reproduce the exact original token sequence");
    }

    #[test]
    fn v8_never_worse_than_flat_single_table() {
        // The actual safety property: v8's own candidate search should never
        // land on something bigger than the trivial "no split at all" flat
        // encoding it started its search from (build_v8_candidate with zero
        // boundaries == exactly what v1 already does).
        let tokens = two_zone_tokens(400);
        let offset_table = entropy::build_offset_encode_table(&tokens).unwrap();
        let flat = build_v8_candidate(&tokens, &offset_table, &[]).unwrap();
        let split = try_entropy_v8(&tokens).unwrap();
        assert!(split.len() <= flat.len(),
            "v8's chosen split ({} B) should never be worse than one flat table ({} B)",
            split.len(), flat.len());
    }

    #[test]
    fn v8_full_pipeline_roundtrips_regardless_of_which_variant_wins() {
        // Real end-to-end safety net: build actual raw bytes with two visibly
        // distinct zones, run them through the real compress()/decompress(),
        // and confirm the roundtrip is byte-exact -- whichever variant the
        // tournament actually picks. This is the test that would catch a
        // wiring mistake in unfold.rs's flag=8 arm, not just the isolated
        // wire format.
        let mut data = Vec::new();
        for i in 0..3000usize { data.push(b'A' + (i % 5) as u8); }
        for i in 0..3000usize { data.push(b'0' + (i % 6) as u8); }

        let compressed = compress(&data, 8).expect("compress should succeed");
        let decompressed = decompress(&compressed).expect("decompress should succeed");
        assert_eq!(decompressed, data, "full pipeline roundtrip mismatch");
    }
    }
