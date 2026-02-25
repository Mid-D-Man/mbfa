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

/// File header layout:
///   Byte 0:              fold_count
///   Byte 1:              pair_flag    (1 = fold 2 used pair encoding)
///   Byte 2:              entropy_flag
///                          0 = raw bitstream
///                          1 = joint Huffman v1 (lit+length coded, offsets raw)
///                          2 = joint Huffman v2 (lit+length coded, offsets bucketed)
///                          3 = joint Huffman v3 (two-context lit+length, offsets bucketed)
///   Bytes 3..3+N:        offset_bits[0..N] where N = fold_count
///   Byte 3+N onward:     compressed payload
///
/// entropy_flag=1 payload: [lit_table][bitstream]
/// entropy_flag=2 payload: [lit_table][offset_table][bitstream]
/// entropy_flag=3 payload: [lit_table_ctx0][lit_table_ctx1][offset_table][bitstream]
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing, offset_bits_per_fold) =
        fold::fold(input, max_folds)?;

    let ob1 = offset_bits_per_fold.first().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_ob = offset_bits_per_fold.last().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);

    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag, out_pair_flag, out_folds_done, out_ob_per_fold) =
        if try_entropy_standard {
            let tokens = bitreader::read_tokens(&compressed, final_ob)?;
            let raw_size = compressed.len();

            let v1 = try_entropy_v1(&tokens, final_ob);
            let v2 = try_entropy_v2(&tokens);
            let v3 = if raw_size >= entropy::ENTROPY_V3_MIN_BYTES {
                try_entropy_v3(&tokens)
            } else {
                None
            };

            let v1_size = v1.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let v2_size = v2.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let v3_size = v3.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let best_size = v1_size.min(v2_size).min(v3_size);

            if best_size >= raw_size {
                println!(
                    "Joint entropy skipped (no gain: v1={} v2={} v3={} vs raw={})",
                    v1_size, v2_size, v3_size, raw_size
                );
                (compressed, 0u8, false, folds_done, offset_bits_per_fold)
            } else if v3_size == best_size {
                println!(
                    "Joint entropy v3 applied: {} → {} bytes (v2={} v1={} B)",
                    raw_size, v3_size, v2_size, v1_size
                );
                (v3.unwrap(), 3u8, false, folds_done, offset_bits_per_fold)
            } else if v2_size == best_size {
                println!(
                    "Joint entropy v2 applied: {} → {} bytes (v3={} v1={} B)",
                    raw_size, v2_size, v3_size, v1_size
                );
                (v2.unwrap(), 2u8, false, folds_done, offset_bits_per_fold)
            } else {
                println!(
                    "Joint entropy v1 applied: {} → {} bytes (v2={} v3={} B)",
                    raw_size, v1_size, v2_size, v3_size
                );
                (v1.unwrap(), 1u8, false, folds_done, offset_bits_per_fold)
            }

        } else if used_pairing {
            pair_vs_entropy(input, ob1, &compressed, folds_done, &offset_bits_per_fold)?

        } else {
            (compressed, 0u8, false, folds_done, offset_bits_per_fold)
        };

    let mut output = Vec::new();
    output.push(out_folds_done);
    output.push(out_pair_flag as u8);
    output.push(entropy_flag);
    for &ob in &out_ob_per_fold {
        output.push(ob as u8);
    }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

/// Try entropy v1: [lit_table][bitstream].
fn try_entropy_v1(tokens: &[opcode::Token], ob: u32) -> Option<Vec<u8>> {
    let enc_table = entropy::build_encode_table(tokens)?;
    let coded = entropy::write_tokens_joint(tokens, &enc_table, ob).ok()?;
    let table_bytes = entropy::serialize_table(&enc_table);
    let mut payload = table_bytes;
    payload.extend_from_slice(&coded);
    Some(payload)
}

/// Try entropy v2: [lit_table][offset_table][bitstream].
fn try_entropy_v2(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let lit_table    = entropy::build_encode_table(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded = entropy::write_tokens_joint_v2(tokens, &lit_table, &offset_table).ok()?;
    let lit_table_bytes    = entropy::serialize_table(&lit_table);
    let offset_table_bytes = entropy::serialize_table(&offset_table);
    let mut payload = lit_table_bytes;
    payload.extend_from_slice(&offset_table_bytes);
    payload.extend_from_slice(&coded);
    Some(payload)
}

/// Try entropy v3: [lit_table_ctx0][lit_table_ctx1][offset_table][bitstream].
/// Returns None if no BACKREFs present (ctx1 empty) or offset table empty.
fn try_entropy_v3(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (lit_table0, lit_table1) = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded = entropy::write_tokens_joint_v3(
        tokens, &lit_table0, &lit_table1, &offset_table
    ).ok()?;
    let t0_bytes  = entropy::serialize_table(&lit_table0);
    let t1_bytes  = entropy::serialize_table(&lit_table1);
    let off_bytes = entropy::serialize_table(&offset_table);
    let mut payload = t0_bytes;
    payload.extend_from_slice(&t1_bytes);
    payload.extend_from_slice(&off_bytes);
    payload.extend_from_slice(&coded);
    Some(payload)
}

/// Compete PAIR output against all entropy variants on fold 1.
fn pair_vs_entropy(
    input: &[u8],
    ob1: u32,
    pair_output: &[u8],
    pair_folds_done: u8,
    pair_ob_per_fold: &[u32],
) -> io::Result<(Vec<u8>, u8, bool, u8, Vec<u32>)> {
    let (fold1_tokens, _) = encoder::scan_adaptive(input);

    let fold1_bits: u32 = fold1_tokens
        .iter()
        .map(|t| opcode::token_bit_cost(t, ob1))
        .sum();
    let fold1_bytes_est = ((fold1_bits + 7) / 8) as usize;

    if fold1_bytes_est < entropy::ENTROPY_MIN_BYTES {
        println!(
            "PAIR kept (fold 1 est. {} B < entropy threshold {} B)",
            fold1_bytes_est, entropy::ENTROPY_MIN_BYTES
        );
        return Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done, pair_ob_per_fold.to_vec(),
        ));
    }

    let v1 = try_entropy_v1(&fold1_tokens, ob1);
    let v2 = try_entropy_v2(&fold1_tokens);
    let v3 = if fold1_bytes_est >= entropy::ENTROPY_V3_MIN_BYTES {
        try_entropy_v3(&fold1_tokens)
    } else {
        None
    };

    let v1_size = v1.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let v2_size = v2.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let v3_size = v3.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let best_entropy_size = v1_size.min(v2_size).min(v3_size);

    if best_entropy_size < pair_output.len() {
        let (best_payload, best_flag) = if v3_size == best_entropy_size {
            println!(
                "Entropy v3 on fold 1 beats PAIR: {} B < {} B — switching path",
                v3_size, pair_output.len()
            );
            (v3.unwrap(), 3u8)
        } else if v2_size == best_entropy_size {
            println!(
                "Entropy v2 on fold 1 beats PAIR: {} B < {} B — switching path",
                v2_size, pair_output.len()
            );
            (v2.unwrap(), 2u8)
        } else {
            println!(
                "Entropy v1 on fold 1 beats PAIR: {} B < {} B — switching path",
                v1_size, pair_output.len()
            );
            (v1.unwrap(), 1u8)
        };
        Ok((best_payload, best_flag, false, 1u8, vec![ob1]))
    } else {
        println!(
            "PAIR wins over entropy: {} B <= best entropy {} B",
            pair_output.len(), best_entropy_size
        );
        Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done, pair_ob_per_fold.to_vec(),
        ))
    }
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
}
