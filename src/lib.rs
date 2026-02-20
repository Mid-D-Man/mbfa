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

use std::io;

pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    let (compressed, folds_done, used_pairing) = fold::fold(input, max_folds)?;
    let mut output = Vec::new();
    output.push(folds_done);
    output.push(used_pairing as u8);  // 0 or 1
    output.extend_from_slice(&compressed);
    Ok(output)
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
}