// src/lib.rs — pre-slot baseline + v4 byte-category entropy + delta filter support
// + v5/v6 recent-offset slot reuse + v7 block-split per-block Huffman
// + PNG filtered-bytes extraction via png_xform
pub mod opcode;
pub mod encoder;
pub mod bitwriter;
pub mod bitreader;
pub mod decoder;
pub mod pairing;
pub mod fold;
pub mod unfold;
pub mod entropy;
pub mod filters;
pub mod png_xform;
pub mod archive;
pub mod archive_io;
pub mod platform;
use std::io;

/// File header layout:
///   Byte 0:              fold_count
///   Byte 1:              pair_flag    (1 = fold 2 used pair encoding)
///   Byte 2:              entropy_flag (0-7)
///   Byte 3:              filter_flag
///                          0 = none  1-4 = delta stride 1-4
///                          5 = PNG (inflate IDAT path)
///   Bytes 4..4+N:        offset_bits[0..N]  N = fold_count
///   Bytes 4+N..4+2N:     length_bits[0..N]
///   --- if filter_flag == 5 (PNG) ---
///   Bytes 4+2N..4+2N+4:  header_blob_len u32 LE
///   Bytes 4+2N+4..:      header_blob (PNG chunks without IDAT)
///   --- end PNG section ---
///   Remaining:           compressed payload
pub fn compress(input: &[u8], max_folds: u8) -> io::Result<Vec<u8>> {
    // ── Pre-filter ────────────────────────────────────────────────────────────
    let mut filter_flag = filters::detect_filter(input);
    let mut png_meta_opt: Option<png_xform::PngMeta> = None;

    let filter_buf: Option<Vec<u8>> = if filter_flag == filters::FILTER_PNG {
        match png_xform::extract_filtered_bytes(input) {
            Ok((filtered, meta)) => {
                println!(
                    "PNG: {}x{} → {} filtered bytes ready for folding",
                    meta.width, meta.height, filtered.len()
                );
                png_meta_opt = Some(meta);
                Some(filtered)
            }
            Err(e) => {
                eprintln!("PNG extraction failed ({}), falling back to raw fold", e);
                filter_flag = filters::FILTER_NONE;
                None
            }
        }
    } else if filter_flag != filters::FILTER_NONE {
        let f = filters::apply_filter(input, filter_flag);
        println!(
            "Filter delta{}: {} bytes in, {} bytes filtered",
            filter_flag, input.len(), f.len()
        );
        Some(f)
    } else {
        None
    };

    let to_fold: &[u8] = filter_buf.as_deref().unwrap_or(input);

    // ── Fold passes ───────────────────────────────────────────────────────────
    let (compressed, folds_done, used_pairing, offset_bits_per_fold, length_bits_per_fold) =
        fold::fold(to_fold, max_folds)?;

    let ob1 = offset_bits_per_fold
        .first().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let lb1 = length_bits_per_fold
        .first().copied().unwrap_or(opcode::LENGTH_BITS_MIN);
    let final_ob = offset_bits_per_fold
        .last().copied().unwrap_or(opcode::OFFSET_BITS_MIN);
    let final_lb = length_bits_per_fold
        .last().copied().unwrap_or(opcode::LENGTH_BITS_MIN);

    let try_entropy_standard = !used_pairing
        && folds_done >= 1
        && compressed.len() >= entropy::ENTROPY_MIN_BYTES;

    let (final_payload, entropy_flag, out_pair_flag, out_folds, out_ob, out_lb) =
        if try_entropy_standard {
            let tokens   = bitreader::read_tokens(&compressed, final_ob, final_lb)?;
            let raw_size = compressed.len();
            let entropy_ok = tokens_safe_for_entropy(&tokens);

            let v1  = if entropy_ok { try_entropy_v1(&tokens, final_ob, final_lb) } else { None };
            let v2  = if entropy_ok { try_entropy_v2(&tokens) } else { None };
            let v3  = if entropy_ok && raw_size >= entropy::ENTROPY_V3_MIN_BYTES {
                try_entropy_v3(&tokens) } else { None };
            let v4  = if entropy_ok && raw_size >= entropy::ENTROPY_V3_MIN_BYTES {
                try_entropy_v4(&tokens) } else { None };
            let v2s = if entropy_ok { try_entropy_v2_slotted(&tokens) } else { None };
            let v3s = if entropy_ok && raw_size >= entropy::ENTROPY_V3_MIN_BYTES {
                try_entropy_v3_slotted(&tokens) } else { None };
            let v7  = if entropy_ok { try_entropy_block_split(&tokens) } else { None };

            let sz = |o: &Option<Vec<u8>>| o.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
            let best_size = sz(&v1).min(sz(&v2)).min(sz(&v3)).min(sz(&v4))
                                   .min(sz(&v2s)).min(sz(&v3s)).min(sz(&v7));

            if best_size >= raw_size {
                println!(
                    "Joint entropy skipped (no gain: v1={} v2={} v3={} v4={} \
                     v5={} v6={} v7={} vs raw={})",
                    sz(&v1), sz(&v2), sz(&v3), sz(&v4),
                    sz(&v2s), sz(&v3s), sz(&v7), raw_size
                );
                (compressed, 0u8, false, folds_done, offset_bits_per_fold, length_bits_per_fold)
            } else {
                let (flag, payload) = [
                    (1u8, &v1), (2u8, &v2), (3u8, &v3), (4u8, &v4),
                    (5u8, &v2s), (6u8, &v3s), (7u8, &v7),
                ]
                .iter()
                .filter_map(|(f, opt)| opt.as_ref().map(|p| (*f, p)))
                .min_by_key(|(_, p)| p.len())
                .unwrap();
                println!("Joint entropy flag={}: {} → {} B", flag, raw_size, payload.len());
                (payload.clone(), flag, false, folds_done,
                 offset_bits_per_fold, length_bits_per_fold)
            }
        } else if used_pairing {
            pair_vs_entropy(
                to_fold, ob1, lb1, &compressed,
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
    output.push(filter_flag);
    for &ob in &out_ob { output.push(ob as u8); }
    for &lb in &out_lb { output.push(lb as u8); }

    // PNG metadata block — header_blob stored between standard header and payload
    if filter_flag == filters::FILTER_PNG {
        if let Some(ref meta) = png_meta_opt {
            output.extend_from_slice(&(meta.header_blob.len() as u32).to_le_bytes());
            output.extend_from_slice(&meta.header_blob);
            println!("PNG header_blob stored: {} B", meta.header_blob.len());
        }
    }

    output.extend_from_slice(&final_payload);
    Ok(output)
}

#[inline]
fn tokens_safe_for_entropy(tokens: &[opcode::Token]) -> bool {
    tokens.iter().all(|t| match t {
        opcode::Token::Backref { length, .. } => *length <= opcode::ENTROPY_SAFE_MAX_LENGTH,
        _ => true,
    })
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
    let (t0, t1)     = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table(tokens)?;
    let coded        = entropy::write_tokens_joint_v3(tokens, &t0, &t1, &offset_table).ok()?;
    let mut payload  = entropy::serialize_table(&t0);
    payload.extend_from_slice(&entropy::serialize_table(&t1));
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v4(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (lit_tables, offset_table) = entropy::build_encode_tables_v4(tokens)?;
    let coded = entropy::write_tokens_joint_v4(tokens, &lit_tables, &offset_table).ok()?;
    let mut payload = Vec::new();
    for t in &lit_tables {
        payload.extend_from_slice(&entropy::serialize_table(t));
    }
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v2_slotted(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let lit_table    = entropy::build_encode_table(tokens)?;
    let offset_table = entropy::build_offset_encode_table_slotted(tokens)?;
    let coded        = entropy::write_tokens_joint_v2_slotted(
        tokens, &lit_table, &offset_table).ok()?;
    let mut payload = entropy::serialize_table(&lit_table);
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_v3_slotted(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let (t0, t1)     = entropy::build_encode_tables_by_context(tokens)?;
    let offset_table = entropy::build_offset_encode_table_slotted(tokens)?;
    let coded        = entropy::write_tokens_joint_v3_slotted(
        tokens, &t0, &t1, &offset_table).ok()?;
    let mut payload = entropy::serialize_table(&t0);
    payload.extend_from_slice(&entropy::serialize_table(&t1));
    payload.extend_from_slice(&entropy::serialize_table(&offset_table));
    payload.extend_from_slice(&coded);
    Some(payload)
}

fn try_entropy_block_split(tokens: &[opcode::Token]) -> Option<Vec<u8>> {
    let token_count = tokens.iter()
        .filter(|t| !matches!(t, opcode::Token::End))
        .count();

    let mut best: Option<Vec<u8>> = None;

    for &block_size in &[1024usize, 2048, 4096] {
        if token_count < block_size * 3 { continue; }
        if let Ok(payload) = entropy::write_tokens_block_split(tokens, block_size) {
            let is_better = best.as_ref().map_or(true, |b| payload.len() < b.len());
            if is_better {
                println!("  v7 block-split bs={}: {} B", block_size, payload.len());
                best = Some(payload);
            }
        }
    }

    best
}

fn pair_vs_entropy(
    filtered_input:   &[u8],
    ob1:              u32,
    lb1:              u32,
    pair_output:      &[u8],
    pair_folds_done:  u8,
    pair_ob_per_fold: &[u32],
    pair_lb_per_fold: &[u32],
) -> io::Result<(Vec<u8>, u8, bool, u8, Vec<u32>, Vec<u32>)> {
    let (fold1_tokens, _, _) = encoder::scan_adaptive(filtered_input);

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

    let entropy_ok = tokens_safe_for_entropy(&fold1_tokens);

    let v1  = if entropy_ok { try_entropy_v1(&fold1_tokens, ob1, lb1) } else { None };
    let v2  = if entropy_ok { try_entropy_v2(&fold1_tokens) } else { None };
    let v3  = if entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V3_MIN_BYTES {
        try_entropy_v3(&fold1_tokens) } else { None };
    let v4  = if entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V3_MIN_BYTES {
        try_entropy_v4(&fold1_tokens) } else { None };
    let v2s = if entropy_ok { try_entropy_v2_slotted(&fold1_tokens) } else { None };
    let v3s = if entropy_ok && fold1_bytes_est >= entropy::ENTROPY_V3_MIN_BYTES {
        try_entropy_v3_slotted(&fold1_tokens) } else { None };
    let v7  = if entropy_ok { try_entropy_block_split(&fold1_tokens) } else { None };

    let sz = |o: &Option<Vec<u8>>| o.as_ref().map(|p| p.len()).unwrap_or(usize::MAX);
    let best_entropy = [
        (1u8, &v1), (2u8, &v2), (3u8, &v3), (4u8, &v4),
        (5u8, &v2s), (6u8, &v3s), (7u8, &v7),
    ]
    .iter()
    .filter_map(|(f, opt)| opt.as_ref().map(|p| (*f, p)))
    .min_by_key(|(_, p)| p.len());

    match best_entropy {
        Some((flag, payload)) if payload.len() < pair_output.len() => {
            println!(
                "Entropy flag={} beats PAIR: {} B < {} B",
                flag, payload.len(), pair_output.len()
            );
            Ok((payload.clone(), flag, false, 1u8, vec![ob1], vec![lb1]))
        }
        _ => {
            println!("PAIR wins over entropy: {} B", pair_output.len());
            Ok((
                pair_output.to_vec(), 0u8, true,
                pair_folds_done,
                pair_ob_per_fold.to_vec(),
                pair_lb_per_fold.to_vec(),
            ))
        }
    }
}

pub fn decompress(input: &[u8]) -> io::Result<Vec<u8>> {
    unfold::unfold(input)
}
