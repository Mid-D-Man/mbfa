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
///   Byte 2:              entropy_flag (0 = raw bitstream, 1 = joint Huffman + offset buckets)
///   Bytes 3..3+N:        offset_bits[0..N] where N = fold_count
///                        One byte per accepted fold, in fold order (fold 1 first).
///                        PAIR folds store 0 as a sentinel.
///                        Still written for the non-entropy LZ unfold passes.
///   Byte 3+N onward:     compressed payload
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing, offset_bits_per_fold) =
        fold::fold(input, max_folds)?;

    // final_ob is still needed to read the LZ bitstream back into tokens
    // for entropy coding — the raw bitstream uses offset_bits normally.
    let final_ob = offset_bits_per_fold.last().copied()
        .unwrap_or(opcode::OFFSET_BITS_MIN);

    let try_entropy = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag) = if try_entropy {
        let tokens = bitreader::read_tokens(&compressed, final_ob)?;
        match entropy::build_encode_table(&tokens) {
            Some(enc_table) => {
                // offset_bits no longer passed — entropy uses bucket coding
                let coded       = entropy::write_tokens_joint(&tokens, &enc_table)?;
                let table_bytes = entropy::serialize_table(&enc_table);
                let total       = coded.len() + table_bytes.len();
                if total < compressed.len() {
                    println!(
                        "Joint entropy applied: {} → {} bytes (table {} B, stream {} B)",
                        compressed.len(), total, table_bytes.len(), coded.len()
                    );
                    let mut payload = table_bytes;
                    payload.extend_from_slice(&coded);
                    (payload, 1u8)
                } else {
                    println!("Joint entropy skipped (no gain: {} vs {} raw)",
                             total, compressed.len());
                    (compressed, 0u8)
                }
            }
            None => (compressed, 0u8),
        }
    } else {
        (compressed, 0u8)
    };

    let mut output = Vec::new();
    output.push(folds_done);
    output.push(used_pairing as u8);
    output.push(entropy_flag);
    for &ob in &offset_bits_per_fold {
        output.push(ob as u8);
    }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
}
