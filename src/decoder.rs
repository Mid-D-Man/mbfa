//! Reconstructs a byte slice from a Token stream.

use crate::opcode::Token;

pub fn reconstruct(tokens: &[Token]) -> Vec<u8> {
    let mut output: Vec<u8> = Vec::new();

    for token in tokens {
        match token {
            Token::Lit { byte } => {
                output.push(*byte);
            }
            Token::Backref { offset, length } => {
                let start = output.len().saturating_sub(*offset as usize);
                for k in 0..*length as usize {
                    let byte = output[start + (k % *offset as usize)];
                    output.push(byte);
                }
            }
            Token::End => break,
        }
    }

    output
}