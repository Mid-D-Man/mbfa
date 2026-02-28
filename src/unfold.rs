// src/unfold.rs
//! Reverses N fold passes then undoes any pre-compression filter.
//!
//! Header v3 layout:
//!   [fold_count: u8][pair_flag: u8][entropy_flag: u8][filter_flag: u8]
//!   [offset_bits[0..N]: u8*N]
//!   [length_bits[0..N]: u8*N]
//!   [payload...]
//!
//! entropy_flag values:
//!   0 = raw (no entropy)
//!   1 = v1 joint Huffman, raw offsets
//!   2 = v2 joint Huffman + offset bucket Huffman
//!   3 = v3 two-context Huffman + shared offset buckets
//!   4 = v4 eight-context (byte-category) Huffman + shared offset buckets
//!
//! filter_flag values:
//!   0 = none
//!   1-4 = delta stride 1-4 (applied before fold, reversed after unfold)

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::pairing::pair_decode;
use crate::entropy;
use crate::filters;
use crate::opcode::{OFFSET_BITS_DEFAULT, OFFSET_BITS_MIN, OFFSET_BITS_MAX,
                    LENGTH_BITS_DEFAULT, LENGTH_BITS_MIN, LENGTH_BITS_MAX};

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() { return Ok(Vec::new()); }
    if input.len() < 4  { return Ok(input.to_vec()); }

    let fold_count   = input[0] as usize;
    let pair_flag    = input[1];
    let entropy_flag = input[2];
    let filter_flag  = input[3];

    let (offset_bits_per_fold, length_bits_per_fold, payload_start) =
        parse_header(input, fold_count);

    println!(
        "Unfolding {} pass(es) | pair_flag={} | entropy_flag={} | filter_flag={} | \
         offset_bits={:?} | length_bits={:?}",
        fold_count, pair_flag, entropy_flag, filter_flag,
        offset_bits_per_fold, length_bits_per_fold
    );

    let ob_for_fold = |n: usize| -> u32 {
        offset_bits_per_fold.get(n.saturating_sub(1))
            .copied().unwrap_or(OFFSET_BITS_DEFAULT)
    };
    let lb_for_fold = |n: usize| -> u32 {
        length_bits_per_fold.get(n.saturating_sub(1))
            .copied().unwrap_or(LENGTH_BITS_DEFAULT)
    };

    let final_ob = ob_for_fold(fold_count);
    let final_lb = lb_for_fold(fold_count);

    // ── Undo outermost entropy layer ─────────────────────────────────────────
    let (mut current, folds_to_undo) = if entropy_flag == 1 {
        let payload = &input[payload_start..];
        let (enc_table, consumed) = entropy::deserialize_table(payload)?;
        let dtable  = entropy::decode_table_from_encode(&enc_table);
        let tokens  = entropy::read_tokens_joint(&payload[consumed..], &dtable, final_ob, final_lb)?;
        let rec     = reconstruct(&tokens);
        println!("Entropy v1 unfold: {} bytes", rec.len());
        (rec, fold_count.saturating_sub(1))

    } else if entropy_flag == 2 {
        let payload = &input[payload_start..];
        let (lit_enc, lit_c) = entropy::deserialize_table(payload)?;
        let (off_enc, off_c) = entropy::deserialize_table(&payload[lit_c..])?;
        let lit_dt  = entropy::decode_table_from_encode(&lit_enc);
        let off_dt  = entropy::decode_table_from_encode(&off_enc);
        let tokens  = entropy::read_tokens_joint_v2(&payload[lit_c + off_c..], &lit_dt, &off_dt)?;
        let rec     = reconstruct(&tokens);
        println!("Entropy v2 unfold: {} bytes", rec.len());
        (rec, fold_count.saturating_sub(1))

    } else if entropy_flag == 3 {
        let payload = &input[payload_start..];
        let (enc0, c0) = entropy::deserialize_table(payload)?;
        let (enc1, c1) = entropy::deserialize_table(&payload[c0..])?;
        let (off_enc, c2) = entropy::deserialize_table(&payload[c0 + c1..])?;
        let dt0    = entropy::decode_table_from_encode(&enc0);
        let dt1    = entropy::decode_table_from_encode(&enc1);
        let off_dt = entropy::decode_table_from_encode(&off_enc);
        let tokens = entropy::read_tokens_joint_v3(
            &payload[c0 + c1 + c2..], &dt0, &dt1, &off_dt
        )?;
        let rec = reconstruct(&tokens);
        println!("Entropy v3 unfold: {} bytes", rec.len());
        (rec, fold_count.saturating_sub(1))

    } else if entropy_flag == 4 {
        let payload = &input[payload_start..];
        let mut cursor = 0usize;

        let mut lit_dtables: Vec<entropy::DecodeTable> = Vec::with_capacity(8);
        for i in 0..8usize {
            let (enc, consumed) = entropy::deserialize_table(&payload[cursor..])
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v4 unfold: lit table {} deserialise failed: {}", i, e)))?;
            lit_dtables.push(entropy::decode_table_from_encode(&enc));
            cursor += consumed;
        }

        let (off_enc, off_c) = entropy::deserialize_table(&payload[cursor..])
            .map_err(|e| std::io::Error::new(e.kind(),
                format!("v4 unfold: offset table deserialise failed: {}", e)))?;
        let off_dt = entropy::decode_table_from_encode(&off_enc);
        cursor += off_c;

        let arr: [entropy::DecodeTable; 8] = lit_dtables.try_into()
            .map_err(|_| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "v4 unfold: expected exactly 8 literal tables",
            ))?;

        let tokens = entropy::read_tokens_joint_v4(&payload[cursor..], &arr, &off_dt)?;
        let rec    = reconstruct(&tokens);
        println!("Entropy v4 unfold: {} bytes", rec.len());
        (rec, fold_count.saturating_sub(1))

    } else {
        // No entropy — payload is raw fold output (or raw filtered bytes if fold_count=0)
        (input[payload_start..].to_vec(), fold_count)
    };

    // ── Undo remaining LZ / PAIR folds in reverse ────────────────────────────
    for pass in (1..=folds_to_undo).rev() {
        let ob = ob_for_fold(pass);
        let lb = lb_for_fold(pass);

        if pass == 2 && pair_flag == 1 {
            let ob1 = ob_for_fold(1);
            let lb1 = lb_for_fold(1);
            let tokens = pair_decode(&current, ob1, lb1)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR) + pass 1 (LZ): {} bytes", current.len());
            // pair handling already consumed both passes — break out of loop
            break;
        } else {
            let tokens = read_tokens(&current, ob, lb)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ): {} bytes", pass, current.len());
        }
    }

    // ── Reverse pre-compression filter ───────────────────────────────────────
    if filter_flag != filters::FILTER_NONE {
        let before = current.len();
        current = filters::undo_filter(&current, filter_flag);
        println!(
            "Filter delta{} reversed: {} bytes → {} bytes",
            filter_flag, before, current.len()
        );
    }

    Ok(current)
}

/// Parse ob/lb arrays from the v3 header (4 fixed bytes + N ob + N lb).
/// Returns (offset_bits_per_fold, length_bits_per_fold, payload_start).
fn parse_header(input: &[u8], fold_count: usize) -> (Vec<u32>, Vec<u32>, usize) {
    // v3 header: 4 fixed bytes (fold_count, pair_flag, entropy_flag, filter_flag)
    //            + fold_count ob bytes + fold_count lb bytes
    let payload_start = 4 + 2 * fold_count;

    if fold_count == 0 {
        // No fold fields — payload starts immediately after 4 fixed bytes
        return (vec![], vec![], 4);
    }

    if input.len() >= payload_start {
        let ob_slice = &input[4..4 + fold_count];
        let lb_slice = &input[4 + fold_count..payload_start];

        let ob_valid = ob_slice.iter().all(|&b| {
            let v = b as u32;
            v == 0 || (v >= OFFSET_BITS_MIN && v <= OFFSET_BITS_MAX)
        });
        let lb_valid = lb_slice.iter().all(|&b| {
            let v = b as u32;
            v == 0 || (v >= LENGTH_BITS_MIN && v <= LENGTH_BITS_MAX)
        });

        if ob_valid && lb_valid {
            let ob: Vec<u32> = ob_slice.iter().map(|&b| b as u32).collect();
            let lb: Vec<u32> = lb_slice.iter().map(|&b| b as u32).collect();
            return (ob, lb, payload_start);
        }
    }

    // Fallback defaults — header too short or values out of range
    println!(
        "parse_header: fallback to defaults (input.len()={}, fold_count={})",
        input.len(), fold_count
    );
    (
        vec![OFFSET_BITS_DEFAULT; fold_count],
        vec![LENGTH_BITS_DEFAULT; fold_count],
        payload_start,
    )
          }
