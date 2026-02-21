// src/lib.rs
//! MBFA — Mid Bit Folding Algorithm
//! MidManStudio
//!
//! Public API surface
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

/// Header layout: [fold_count: u8][pair_flag: u8][entropy_flag: u8]
/// If entropy_flag == 1: [huffman_table...][entropy_coded_bitstream]
/// If entropy_flag == 0: [standard_bitstream]
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing) = fold::fold(input, max_folds)?;

    // Entropy coding conditions:
    // - Not pair-encoded (pair streams are already small, table overhead would dominate)
    // - At least one fold ran (fold 0 = raw passthrough, nothing to entropy-code)
    // - Stream is above the size gate
    let try_entropy = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_table_bytes) = if try_entropy {
        let tokens = bitreader::read_tokens(&compressed)?;
        match entropy::build_encode_table(&tokens) {
            Some(enc_table) => {
                let coded        = entropy::write_tokens_entropy(&tokens, &enc_table)?;
                let table_bytes  = entropy::serialize_table(&enc_table);
                let total        = coded.len() + table_bytes.len();
                if total < compressed.len() {
                    println!(
                        "Entropy coding applied: {} → {} bytes (table {} B, stream {} B)",
                        compressed.len(), total, table_bytes.len(), coded.len()
                    );
                    (coded, Some(table_bytes))
                } else {
                    println!(
                        "Entropy coding skipped (no gain: {} vs {} raw)",
                        total, compressed.len()
                    );
                    (compressed, None)
                }
            }
            None => (compressed, None),
        }
    } else {
        (compressed, None)
    };

    let mut output = Vec::new();
    output.push(folds_done);
    output.push(used_pairing as u8);
    output.push(entropy_table_bytes.is_some() as u8); // entropy_flag
    if let Some(table_bytes) = entropy_table_bytes {
        output.extend_from_slice(&table_bytes);
    }
    output.extend_from_slice(&final_payload);
    Ok(output)
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
    }
