// src/unfold.rs
//
// P6 changes: ring_flag parsed from bit 1 of pair_flag byte.
// P10 changes: entropy_flag=7 decoded via read_tokens_v7 (no tables needed).

use crate::bitreader::read_tokens;
use crate::decoder::reconstruct;
use crate::entropy;
use crate::filters;
use crate::opcode::{
    LENGTH_BITS_DEFAULT, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    OFFSET_BITS_DEFAULT, OFFSET_BITS_MAX, OFFSET_BITS_MIN,
};
use crate::pairing::pair_decode;

pub fn unfold(input: &[u8]) -> std::io::Result<Vec<u8>> {
    if input.is_empty() { return Ok(Vec::new()); }
    if input.len() < 4  { return Ok(input.to_vec()); }

    let fold_count    = input[0] as usize;
    let pair_flag_raw = input[1];
    let pair_flag     = pair_flag_raw & 0x01;
    let ring_flag     = (pair_flag_raw >> 1) & 0x01 != 0;
    let entropy_flag  = input[2];
    let filter_flag   = input[3];

    let (offset_bits_per_fold, length_bits_per_fold, payload_start) =
        parse_header(input, fold_count);

    println!(
        "Unfolding {} pass(es) | pair_flag={} | ring_flag={} | \
         entropy_flag={} | filter_flag={} | \
         offset_bits={:?} | length_bits={:?}",
        fold_count, pair_flag, ring_flag as u8, entropy_flag, filter_flag,
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

    // ── Entropy decode ────────────────────────────────────────────────────────
    let (mut current, folds_to_undo) = match entropy_flag {

        1 => {
            let payload = &input[payload_start..];
            let (lit_enc, lit_c) = entropy::deserialize_table(payload)?;
            let (off_enc, off_c) = entropy::deserialize_table(&payload[lit_c..])?;
            let lit_dt  = entropy::decode_table_from_encode(&lit_enc);
            let off_dt  = entropy::decode_table_from_encode(&off_enc);
            let tokens  = entropy::read_tokens_v1(
                &payload[lit_c + off_c..], &lit_dt, &off_dt)?;
            let rec = reconstruct(&tokens);
            println!("Entropy v1 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        2 => {
            let payload = &input[payload_start..];
            let (enc0, c0) = entropy::deserialize_table(payload)?;
            let (enc1, c1) = entropy::deserialize_table(&payload[c0..])?;
            let (off_enc, c2) = entropy::deserialize_table(&payload[c0 + c1..])?;
            let dt0    = entropy::decode_table_from_encode(&enc0);
            let dt1    = entropy::decode_table_from_encode(&enc1);
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            let tokens = entropy::read_tokens_v2(
                &payload[c0 + c1 + c2..], &dt0, &dt1, &off_dt)?;
            let rec = reconstruct(&tokens);
            println!("Entropy v2 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        3 => {
            let payload = &input[payload_start..];
            let mut cursor = 0usize;
            let mut lit_dtables: Vec<entropy::DecodeTable> = Vec::with_capacity(8);
            for i in 0..8usize {
                let (enc, consumed) = entropy::deserialize_table(&payload[cursor..])
                    .map_err(|e| std::io::Error::new(e.kind(),
                        format!("v3 unfold: lit table {} failed: {}", i, e)))?;
                lit_dtables.push(entropy::decode_table_from_encode(&enc));
                cursor += consumed;
            }
            let (off_enc, off_c) = entropy::deserialize_table(&payload[cursor..])
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v3 unfold: offset table failed: {}", e)))?;
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            cursor += off_c;
            let arr: [entropy::DecodeTable; 8] = lit_dtables.try_into()
                .map_err(|_| std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "v3 unfold: expected exactly 8 literal tables"))?;
            let tokens = entropy::read_tokens_v3(&payload[cursor..], &arr, &off_dt)?;
            let rec    = reconstruct(&tokens);
            println!("Entropy v3 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        4 => {
            let payload = &input[payload_start..];
            let (lit_enc, lit_c) = entropy::deserialize_table(payload)?;
            let (off_enc, off_c) = entropy::deserialize_table(&payload[lit_c..])?;
            let lit_dt = entropy::decode_table_from_encode(&lit_enc);
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            let tokens = entropy::read_tokens_v4(
                &payload[lit_c + off_c..], &lit_dt, &off_dt)?;
            let rec = reconstruct(&tokens);
            println!("Entropy v4 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        5 => {
            let payload = &input[payload_start..];
            let (enc0, c0) = entropy::deserialize_table(payload)?;
            let (enc1, c1) = entropy::deserialize_table(&payload[c0..])?;
            let (off_enc, c2) = entropy::deserialize_table(&payload[c0 + c1..])?;
            let dt0    = entropy::decode_table_from_encode(&enc0);
            let dt1    = entropy::decode_table_from_encode(&enc1);
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            let tokens = entropy::read_tokens_v5(
                &payload[c0 + c1 + c2..], &dt0, &dt1, &off_dt)?;
            let rec = reconstruct(&tokens);
            println!("Entropy v5 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        6 => {
            let payload = &input[payload_start..];
            let (lit_enc, lit_c) = entropy::deserialize_table(payload)
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v6 unfold: lit_table failed: {}", e)))?;
            let (seq_enc, seq_c) = entropy::deserialize_table(&payload[lit_c..])
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v6 unfold: seq_table failed: {}", e)))?;
            let (off_enc, off_c) = entropy::deserialize_table(&payload[lit_c + seq_c..])
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v6 unfold: offset_table failed: {}", e)))?;
            let lit_dt = entropy::decode_table_from_encode(&lit_enc);
            let seq_dt = entropy::decode_table_from_encode(&seq_enc);
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            let tokens = entropy::read_tokens_v6(
                &payload[lit_c + seq_c + off_c..],
                &lit_dt, &seq_dt, &off_dt,
            )?;
            let rec = reconstruct(&tokens);
            println!("Entropy v6 unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        // ── v7: adaptive binary range coder ──────────────────────────────────
        // No tables stored -- decoder reconstructs probability model from
        // the same initial state (all probs = RC_PROB_INIT = 1024).
        // Payload is the raw range-coder byte stream.
        7 => {
            let payload = &input[payload_start..];
            let tokens  = entropy::read_tokens_v7(payload)
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v7 unfold: range coder decode failed: {}", e)))?;
            let rec = reconstruct(&tokens);
            println!("Entropy v7 (range coder) unfold: {} bytes", rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        // ── v8: block-split (path-2 first cut) ────────────────────────────────
        // Header is num_segments, then each segment's token count, then the
        // shared offset table, then each segment's own literal table, then
        // each segment's length-prefixed token payload (decoded with a
        // *counted* read -- only the last segment's slice actually contains
        // Token::End, so earlier segments have nothing to break on otherwise).
        8 => {
            let payload = &input[payload_start..];
            let num_segments = payload[0] as usize;
            let mut cursor = 1usize;

            let mut seg_counts: Vec<usize> = Vec::with_capacity(num_segments);
            for _ in 0..num_segments {
                let count = u32::from_le_bytes(
                    payload[cursor..cursor + 4].try_into().unwrap()) as usize;
                seg_counts.push(count);
                cursor += 4;
            }

            let (off_enc, off_c) = entropy::deserialize_table(&payload[cursor..])
                .map_err(|e| std::io::Error::new(e.kind(),
                    format!("v8 unfold: offset table failed: {}", e)))?;
            let off_dt = entropy::decode_table_from_encode(&off_enc);
            cursor += off_c;

            let mut seg_dtables: Vec<entropy::DecodeTable> = Vec::with_capacity(num_segments);
            for i in 0..num_segments {
                let (enc, consumed) = entropy::deserialize_table(&payload[cursor..])
                    .map_err(|e| std::io::Error::new(e.kind(),
                        format!("v8 unfold: segment {} table failed: {}", i, e)))?;
                seg_dtables.push(entropy::decode_table_from_encode(&enc));
                cursor += consumed;
            }

            let mut tokens = Vec::new();
            for i in 0..num_segments {
                let seg_len = u32::from_le_bytes(
                    payload[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                let seg_tokens = entropy::read_tokens_v1_counted(
                    &payload[cursor..cursor + seg_len],
                    &seg_dtables[i], &off_dt, seg_counts[i],
                ).map_err(|e| std::io::Error::new(e.kind(),
                    format!("v8 unfold: segment {} tokens failed: {}", i, e)))?;
                tokens.extend(seg_tokens);
                cursor += seg_len;
            }
            let rec = reconstruct(&tokens);
            println!("Entropy v8 (block-split, {} segments) unfold: {} bytes", num_segments, rec.len());
            (rec, fold_count.saturating_sub(1))
        }

        _ => {
            // entropy_flag=0: raw LZ bitstream, no entropy coding
            // entropy_flag=9+: unknown/future, treat as raw
            (input[payload_start..].to_vec(), fold_count)
        }
    };

    // ── LZ / PAIR unfold passes ───────────────────────────────────────────────
    for pass in (1..=folds_to_undo).rev() {
        let ob = ob_for_fold(pass);
        let lb = lb_for_fold(pass);

        if pass == 2 && pair_flag == 1 {
            let ob1 = ob_for_fold(1);
            let lb1 = lb_for_fold(1);
            let tokens = pair_decode(&current, ob1, lb1)?;
            current = reconstruct(&tokens);
            println!("Unfold pass 2 (PAIR/EG) + pass 1 (LZ): {} bytes", current.len());
            break;
        } else {
            // P6: fold 1's bitstream uses ring-active opcodes when ring_flag=1.
            let ring_active = pass == 1 && ring_flag;
            let tokens = read_tokens(&current, ob, lb, ring_active)?;
            current = reconstruct(&tokens);
            println!("Unfold pass {} (LZ{}): {} bytes",
                pass,
                if ring_active { "+ring" } else { "" },
                current.len()
            );
        }
    }

    // ── Post-filter ───────────────────────────────────────────────────────────
    if filter_flag != filters::FILTER_NONE {
        let before = current.len();
        current = filters::undo_filter(&current, filter_flag);
        println!(
            "Filter flag={} reversed: {} bytes -> {} bytes",
            filter_flag, before, current.len()
        );
    }

    Ok(current)
}

fn parse_header(input: &[u8], fold_count: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let payload_start = 4 + 2 * fold_count;

    if fold_count == 0 {
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
