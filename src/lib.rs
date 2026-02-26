// src/lib.rs
//! MBFA — Mid Bit Folding Algorithm — MidManStudio
pub mod opcode;
pub mod encoder;
pub mod bitwriter;
pub mod bitreader;
pub mod decoder;
pub mod pairing;
pub mod fold;
pub mod unfold;
pub mod entropy;

use std::io;

/// File header layout (v2):
///   Byte 0:              fold_count
///   Byte 1:              pair_flag    (1 = fold 2 used pair encoding)
///   Byte 2:              entropy_flag (0=raw, 1=v1, 2=v2, 3=v3)
///   Bytes 3..3+N:        offset_bits[0..N]  where N = fold_count
///   Bytes 3+N..3+2N:     length_bits[0..N]
///   Byte 3+2N onward:    compressed payload
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing, offset_bits_per_fold, length_bits_per_fold) =
        fold::fold(input, max_folds)?;

    let ob1      = offset_bits_per_fold.first().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let lb1      = length_bits_per_fold.first().copied().unwrap_or(opcode::LENGTH_BITS_MIN);
    let final_ob = offset_bits_per_fold.last().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_lb = length_bits_per_fold.last().copied().unwrap_or(opcode::LENGTH_BITS_MIN);

    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag, out_pair_flag, out_folds, out_ob, out_lb) =
        if try_entropy_standard {
            let tokens   = bitreader::read_tokens(&compressed, final_ob, final_lb)?;
            let raw_size = compressed.len();

            let v1 = try_entropy_v1(&tokens, final_ob, final_lb);
            let v2 = try_entropy_v2(&tokens);
            let v3 = if raw_size >= entropy::ENTROPY_V3_MIN_BYTES {
                try_entropy_v3(&tokens)
            } else {
                None
            };

            let v1_size   = v1.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let v2_size   = v2.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let v3_size   = v3.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let best_size = v1_size.min(v2_size).min(v3_size);

            if best_size >= raw_size {
                println!(
                    "Joint entropy skipped (no gain: v1={} v2={} v3={} vs raw={})",
                    v1_size, v2_size, v3_size, raw_size
                );
                (compressed, 0u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            } else if v3_size == best_size {
                println!("Joint entropy v3: {} → {} B", raw_size, v3_size);
                (v3.unwrap(), 3u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            } else if v2_size == best_size {
                println!("Joint entropy v2: {} → {} B", raw_size, v2_size);
                (v2.unwrap(), 2u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            } else {
                println!("Joint entropy v1: {} → {} B", raw_size, v1_size);
                (v1.unwrap(), 1u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            }

        } else if used_pairing {
            pair_vs_entropy(
                input, ob1, lb1, &compressed,
                folds_done, &offset_bits_per_fold, &length_bits_per_fold,
            )?
        } else {
            (compressed, 0u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
        };

    // ── Serialise header ──────────────────────────────────────────────────────
    let mut output = Vec::new();
    output.push(out_folds);
    output.push(out_pair_flag as u8);
    output.push(entropy_flag);
    for &ob in &out_ob { output.push(ob as u8); }
    for &lb in &out_lb { output.push(lb as u8); }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

fn try_entropy_v1(tokens: &[opcode::Token], ob: u32, lb: u32) -> Option<Vec<u8>> {
    let enc_table = entropy::build_encode_table(tokens)?;
    let coded     = entropy::write_tokens_joint(tokens, &enc_table, ob, lb).ok()?;
    let mut payload = entropy::serialize_table(&enc_table);
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v2(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let lit_table    = entropy::build_encode_table(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded        = entropy::write_tokens_joint_v2(tokens, &lit_table, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&lit_table);
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v3(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (t0, t1)    = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded        = entropy::write_tokens_joint_v3(tokens, &t0, &t1, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&t0);
    payload.extend_from_slice(&entropy::serialize_table(&t1));
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn pair_vs_entropy(
    input:            &[u8],
    ob1:              u32,
    lb1:              u32,
    pair_output:      &[u8],
    pair_folds_done:  u8,
    pair_ob_per_fold: &[u32],
    pair_lb_per_fold: &[u32],
) -> io::Result<(Vec<u8>, u8, bool, u8, Vec<u32>, Vec<u32>)> {
    let (fold1_tokens, _, _) = encoder::scan_adaptive(input);

    let fold1_bits: u32 = fold1_tokens
        .iter()
        .map(|t| opcode::token_bit_cost(t, ob1, lb1))
        .sum();
    let fold1_bytes_est = ((fold1_bits + 7) / 8) as usize;

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

    let v1 = try_entropy_v1(&fold1_tokens, ob1, lb1);
    let v2 = try_entropy_v2(&fold1_tokens);
    let v3 = if fold1_bytes_est >= entropy::ENTROPY_V3_MIN_BYTES {
        try_entropy_v3(&fold1_tokens)
    } else {
        None
    };

    let v1_size           = v1.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let v2_size           = v2.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let v3_size           = v3.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let best_entropy_size = v1_size.min(v2_size).min(v3_size);

    if best_entropy_size < pair_output.len() {
        let (best_payload, best_flag) = if v3_size == best_entropy_size {
            println!("Entropy v3 beats PAIR: {} B < {} B", v3_size, pair_output.len());
            (v3.unwrap(), 3u8)
        } else if v2_size == best_entropy_size {
            println!("Entropy v2 beats PAIR: {} B < {} B", v2_size, pair_output.len());
            (v2.unwrap(), 2u8)
        } else {
            println!("Entropy v1 beats PAIR: {} B < {} B", v1_size, pair_output.len());
            (v1.unwrap(), 1u8)
        };
        Ok((best_payload, best_flag, false, 1u8, vec![ob1], vec![lb1]))
    } else {
        println!("PAIR wins over entropy: {} B <= {} B", pair_output.len(), best_entropy_size);
        Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done,
            pair_ob_per_fold.to_vec(),
            pair_lb_per_fold.to_vec(),
        ))
    }
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
            }
