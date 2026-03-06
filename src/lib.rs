// src/lib.rs
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
use rayon::prelude::*;

/// Entropy threshold above which data is treated as incompressible.
/// 7.8 bits/byte leaves a safe margin — truly random data hits ~8.0,
/// compressed formats (PNG, JPEG, zst) typically land 7.8-8.0.
const INCOMPRESSIBLE_ENTROPY_THRESHOLD: f64 = 7.8;

/// Minimum input size before the incompressible check fires.
/// Small files are skipped — small gradient BMPs CAN compress well in MBFA
/// (verified: MBFA beats gzip on small gradient BMPs at ~2.15%).
/// 50 KB keeps those paths alive while catching large incompressible blobs.
const INCOMPRESSIBLE_MIN_BYTES: usize = 50_000;

/// Sample Shannon entropy from the first 8 KB of data.
/// Returns bits-per-byte in [0.0, 8.0]. Cost: O(8192), essentially free.
fn sample_entropy(data: &[u8]) -> f64 {
    const SAMPLE_SIZE: usize = 8192;
    let sample = if data.len() > SAMPLE_SIZE { &data[..SAMPLE_SIZE] } else { data };
    let mut freq = [0u32; 256];
    for &b in sample {
        freq[b as usize] += 1;
    }
    let n = sample.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// File header layout:
///   Byte 0:          fold_count
///   Byte 1:          pair_flag    (1 = fold 2 used pair encoding)
///   Byte 2:          entropy_flag (0 = none, 1-5 = entropy variant)
///   Byte 3:          filter_flag  (0 = none, 1-4 = delta stride 1-4)
///   Bytes 4..4+N:    offset_bits[0..N]  N = fold_count
///   Bytes 4+N..4+2N: length_bits[0..N]
///   Remaining:       compressed payload
///
/// Special case — fold_count=0:
///   Passthrough header. Payload is the original uncompressed bytes verbatim.
///   Emitted when incompressible early exit fires. Decompressor sees fold_count=0,
///   runs zero unfold passes, returns payload unchanged.
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    // ── Pre-filter ────────────────────────────────────────────────────────────
    // Filter detection runs before the incompressible check so that WAV/BMP files
    // are evaluated on their filtered (low-entropy residual) form, not raw bytes.
    let filter_flag = filters::detect_filter(input);

    let filter_buf: Option<Vec<u8>> = if filter_flag != filters::FILTER_NONE {
        let f = filters::apply_filter(input, filter_flag);
        println!(
            "Filter delta{}: {} bytes in, {} bytes filtered",
            filter_flag, input.len(), f.len()
        );
        Some(f)
    } else {
        None
    };

    let to_fold: &[u8] = filter_buf.as_deref().unwrap_or(input);

    // ── Incompressible early exit ─────────────────────────────────────────────
    // Evaluated on `to_fold` (post-filter) so WAV/BMP delta-filtered residuals
    // get their correct (low) entropy. If a WAV file somehow produced high-entropy
    // residuals, we still bail correctly and return the ORIGINAL unfiltered bytes.
    if input.len() > INCOMPRESSIBLE_MIN_BYTES {
        let ent = sample_entropy(to_fold);
        if ent > INCOMPRESSIBLE_ENTROPY_THRESHOLD {
            println!(
                "Incompressible early exit: entropy={:.3} bits/byte, {} bytes — passthrough",
                ent, input.len()
            );
            // Header: fold_count=0, pair=0, entropy=0, filter=NONE
            // Payload: original input (unfiltered) — decompressor returns it unchanged.
            let mut out = Vec::with_capacity(4 + input.len());
            out.push(0u8);                  // fold_count = 0
            out.push(0u8);                  // pair_flag  = 0
            out.push(0u8);                  // entropy_flag = 0
            out.push(filters::FILTER_NONE); // filter_flag  = 0
            out.extend_from_slice(input);
            return Ok(out);
        }
    }

    // ── Fold passes ───────────────────────────────────────────────────────────
    // fold1_tokens_opt: cached fold 1 token stream to avoid re-scan in pair_vs_entropy.
    let (compressed, folds_done, used_pairing, offset_bits_per_fold, length_bits_per_fold, fold1_tokens_opt) =
        fold::fold(to_fold, max_folds)?;

    let ob1 = offset_bits_per_fold
        .first().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let lb1 = length_bits_per_fold
        .first().copied().unwrap_or(opcode::LENGTH_BITS_MIN);
    let final_ob = offset_bits_per_fold
        .last().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_lb = length_bits_per_fold
        .last().copied().unwrap_or(opcode::LENGTH_BITS_MIN);

    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag, out_pair_flag, out_folds, out_ob, out_lb) =
        if try_entropy_standard {
            let tokens   = bitreader::read_tokens(&compressed, final_ob, final_lb)?;
            let raw_size = compressed.len();
            let entropy_ok = tokens_safe_for_entropy(&tokens);

            // Run entropy variants v1–v5 in parallel — each takes &[Token]
            // and produces an independent Vec<u8>. No shared mutable state.
            let results: Vec<(u8, Option<Vec<u8>>)> = vec![
                (1u8, entropy_ok),
                (2u8, entropy_ok && raw_size >= entropy::ENTROPY_V2_MIN_BYTES),
                (3u8, entropy_ok && raw_size >= entropy::ENTROPY_V2_MIN_BYTES),
                (4u8, entropy_ok),
                (5u8, entropy_ok && raw_size >= entropy::ENTROPY_V2_MIN_BYTES),
            ]
            .into_par_iter()
            .map(|(flag, enabled)| {
                let payload = if enabled {
                    match flag {
                        1 => try_entropy_v1(&tokens),
                        2 => try_entropy_v2(&tokens),
                        3 => try_entropy_v3(&tokens),
                        4 => try_entropy_v4(&tokens),
                        5 => try_entropy_v5(&tokens),
                        _ => None,
                    }
                } else {
                    None
                };
                (flag, payload)
            })
            .collect();

            let sz = |o: &Option<Vec<u8>>| o.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let best_size = results.iter().map(|(_, o)| sz(o)).min().unwrap_or(usize::MAX);

            if best_size >= raw_size {
                let detail: Vec<String> = results.iter()
                    .map(|(f, o)| format!("v{}={}", f, sz(o)))
                    .collect();
                println!(
                    "Joint entropy skipped (no gain: {} vs raw={})",
                    detail.join(" "), raw_size
                );
                (compressed, 0u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            } else {
                let (flag, payload) = results.into_iter()
                    .filter_map(|(f, opt)| opt.map(|p| (f, p)))
                    .min_by_key(|(_, p)| p.len())
                    .unwrap();
                println!("Joint entropy flag={}: {} → {} B", flag, raw_size, payload.len());
                (payload, flag, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            }
        } else if used_pairing {
            // fold1_tokens_opt is guaranteed Some here:
            // pair path only fires when fold 1 succeeded, which always sets fold1_tokens.
            let f1t = fold1_tokens_opt
                .expect("fold1_tokens must be Some when used_pairing is true");
            pair_vs_entropy(
                f1t, ob1, lb1, &compressed,
                folds_done, &offset_bits_per_fold, &length_bits_per_fold,
            )?
        } else {
            (compressed, 0u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
        };

    // ── Serialise header ──────────────────────────────────────────────────────
    let mut output = Vec::with_capacity(4 + 2 * out_folds as usize + final_payload.len());
    output.push(out_folds);
    output.push(out_pair_flag as u8);
    output.push(entropy_flag);
    output.push(filter_flag);
    for &ob in &out_ob { output.push(ob as u8); }
    for &lb in &out_lb { output.push(lb as u8); }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

#[inline]
fn tokens_safe_for_entropy(tokens: &[opcode::Token]) -> bool {
    tokens.iter().all(|t| match t {
        opcode::Token::Backref { length, .. } => *length <= opcode::ENTROPY_SAFE_MAX_LENGTH,
        _ => true,
    })
}

fn try_entropy_v1(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let lit_table    = entropy::build_encode_table(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded        = entropy::write_tokens_v1(tokens, &lit_table, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&lit_table);
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v2(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (t0, t1)     = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded        = entropy::write_tokens_v2(tokens, &t0, &t1, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&t0);
    payload.extend_from_slice(&entropy::serialize_table(&t1));
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v3(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (lit_tables, offset_table) = entropy::build_encode_tables_v3(tokens)?;
    let coded = entropy::write_tokens_v3(tokens, &lit_tables, &offset_table).ok()?;
    let mut payload = Vec::new();
    for t in &lit_tables {
        payload.extend_from_slice(&entropy::serialize_table(t));
    }
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v4(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let lit_table    = entropy::build_encode_table(tokens)?;
    let offset_table = entropy::build_offset_encode_table_slotted(tokens)?;
    let coded        = entropy::write_tokens_v4(tokens, &lit_table, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&lit_table);
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v5(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (t0, t1)     = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table_slotted(tokens)?;
    let coded        = entropy::write_tokens_v5(tokens, &t0, &t1, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&t0);
    payload.extend_from_slice(&entropy::serialize_table(&t1));
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

/// Compare pair-encoded output against all entropy variants in parallel.
/// Receives the cached fold 1 token stream — no re-scan needed.
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

    // Entropy variants in parallel — same pattern as the standard path.
    let results: Vec<(u8, Option<Vec<u8>>)> = vec![
        (1u8, entropy_ok),
        (2u8, entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V2_MIN_BYTES),
        (3u8, entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V2_MIN_BYTES),
        (4u8, entropy_ok),
        (5u8, entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V2_MIN_BYTES),
    ]
    .into_par_iter()
    .map(|(flag, enabled)| {
        let payload = if enabled {
            match flag {
                1 => try_entropy_v1(&fold1_tokens),
                2 => try_entropy_v2(&fold1_tokens),
                3 => try_entropy_v3(&fold1_tokens),
                4 => try_entropy_v4(&fold1_tokens),
                5 => try_entropy_v5(&fold1_tokens),
                _ => None,
            }
        } else {
            None
        };
        (flag, payload)
    })
    .collect();

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
