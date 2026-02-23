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
///   Byte 2:              entropy_flag (0 = raw bitstream, 1 = joint Huffman)
///   Bytes 3..3+N:        offset_bits[0..N] where N = fold_count
///                        One byte per accepted fold, in fold order (fold 1 first).
///                        PAIR folds store 0 as a sentinel.
///   Byte 3+N onward:     compressed payload
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing, offset_bits_per_fold) =
        fold::fold(input, max_folds)?;

    let ob1 = offset_bits_per_fold.first().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_ob = offset_bits_per_fold.last().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);

    // ── Standard entropy path: no pairing, fold N output → Huffman ───────────
    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    // Returns (payload, entropy_flag, pair_flag, folds_done, ob_per_fold)
    let (final_payload, entropy_flag, out_pair_flag, out_folds_done, out_ob_per_fold) =
        if try_entropy_standard {
            let tokens = bitreader::read_tokens(&compressed, final_ob)?;
            match entropy::build_encode_table(&tokens) {
                Some(enc_table) => {
                    let coded       = entropy::write_tokens_joint(&tokens, &enc_table, final_ob)?;
                    let table_bytes = entropy::serialize_table(&enc_table);
                    let total       = coded.len() + table_bytes.len();
                    if total < compressed.len() {
                        println!(
                            "Joint entropy applied: {} → {} bytes (table {} B, stream {} B)",
                            compressed.len(), total, table_bytes.len(), coded.len()
                        );
                        let mut payload = table_bytes;
                        payload.extend_from_slice(&coded);
                        (payload, 1u8, false, folds_done, offset_bits_per_fold)
                    } else {
                        println!(
                            "Joint entropy skipped (no gain: {} vs {} raw)",
                            total, compressed.len()
                        );
                        (compressed, 0u8, false, folds_done, offset_bits_per_fold)
                    }
                }
                None => (compressed, 0u8, false, folds_done, offset_bits_per_fold),
            }

        } else if used_pairing {
            // ── PAIR vs entropy competition ───────────────────────────────────
            // PAIR was used for fold 2. Try entropy on fold 1 tokens as an
            // alternative. Re-deriving fold 1 tokens is safe: scan_adaptive is
            // deterministic and produces the same result as fold.rs did.
            // Decoder already handles fold_count=1,entropy_flag=1 correctly.
            pair_vs_entropy(input, ob1, &compressed, folds_done, &offset_bits_per_fold)?

        } else {
            // ── No entropy, no pairing (size too small or folds=0) ───────────
            (compressed, 0u8, false, folds_done, offset_bits_per_fold)
        };

    // ── Write header + payload ────────────────────────────────────────────────
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

/// Compete PAIR output against entropy on fold 1.
///
/// Returns (payload, entropy_flag, pair_flag, folds_done, ob_per_fold).
///
/// If entropy on fold 1 produces fewer bytes than the PAIR output, we switch to
/// that path. The decoder path for entropy-wins is:
///   fold_count=1, pair_flag=0, entropy_flag=1, offset_bits=[ob1]
/// which `unfold.rs` already handles correctly — entropy decodes → reconstruct
/// → original input, folds_to_undo=0, return immediately.
fn pair_vs_entropy(
    input: &[u8],
    ob1: u32,
    pair_output: &[u8],
    pair_folds_done: u8,
    pair_ob_per_fold: &[u32],
) -> io::Result<(Vec<u8>, u8, bool, u8, Vec<u32>)> {
    // Re-derive fold 1 tokens. scan_adaptive(input) is deterministic and
    // produces exactly the same tokens fold.rs used for fold 1.
    let (fold1_tokens, _recomputed_ob) = encoder::scan_adaptive(input);

    // Estimate fold 1 raw byte size from per-token bit costs.
    // This avoids materialising the full bitstream just for the size check.
    let fold1_bits: u32 = fold1_tokens
        .iter()
        .map(|t| opcode::token_bit_cost(t, ob1))
        .sum();
    let fold1_bytes_est = ((fold1_bits + 7) / 8) as usize;

    if fold1_bytes_est < entropy::ENTROPY_MIN_BYTES {
        // Fold 1 output is too small for entropy to be worthwhile — keep PAIR.
        println!(
            "PAIR kept (fold 1 est. {} B < entropy threshold {} B)",
            fold1_bytes_est, entropy::ENTROPY_MIN_BYTES
        );
        return Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done, pair_ob_per_fold.to_vec(),
        ));
    }

    match entropy::build_encode_table(&fold1_tokens) {
        Some(enc_table) => {
            let coded       = entropy::write_tokens_joint(&fold1_tokens, &enc_table, ob1)?;
            let table_bytes = entropy::serialize_table(&enc_table);
            let entropy_total = coded.len() + table_bytes.len();

            if entropy_total < pair_output.len() {
                println!(
                    "Entropy on fold 1 beats PAIR: {} B < {} B — switching path",
                    entropy_total, pair_output.len()
                );
                let mut payload = table_bytes;
                payload.extend_from_slice(&coded);
                // Entropy wins: emit as fold_count=1, pair_flag=0, entropy_flag=1
                Ok((payload, 1u8, false, 1u8, vec![ob1]))
            } else {
                println!(
                    "PAIR wins over entropy on fold 1: {} B <= {} B",
                    pair_output.len(), entropy_total
                );
                Ok((
                    pair_output.to_vec(), 0u8, true,
                    pair_folds_done, pair_ob_per_fold.to_vec(),
                ))
            }
        }
        None => Ok((
            pair_output.to_vec(), 0u8, true,
            pair_folds_done, pair_ob_per_fold.to_vec(),
        )),
    }
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
                }
