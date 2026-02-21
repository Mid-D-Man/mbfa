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
///   Byte 0: fold_count
///   Byte 1: pair_flag   (1 = fold 2 used pair encoding)
///   Byte 2: entropy_flag (0 = raw, 1 = joint Huffman wraps outermost fold)
///   Byte 3: offset_bits  (actual offset field width used in LZ bitstream,
///                          range [7, 17]. Allows adaptive window per file.)
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing, offset_bits) =
        fold::fold(input, max_folds)?;

    let try_entropy = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag) = if try_entropy {
        let tokens = bitreader::read_tokens(&compressed, offset_bits)?;
        match entropy::build_encode_table(&tokens) {
            Some(enc_table) => {
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
                    println!("Joint entropy skipped (no gain: {} vs {} raw)", total, compressed.len());
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
    output.push(offset_bits as u8);   // NEW: byte 3
    output.extend_from_slice(&final_payload);
    Ok(output)
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
        }
