// src/bin/profile_offsets.rs
//! Profiles recent-distance reuse potential across a set of files.
//!
//! Run with:
//!   cargo run --release --bin profile_offsets -- path/to/file1 path/to/file2 ...
//!
//! For each file it prints:
//!   - Total BACKREFs
//!   - How many had an offset matching the last 1, 2, or 3 seen offsets
//!   - Reuse % for each cache size
//!   - Bit savings estimate at offset_bits=17 (2 bits per hit vs 17 raw)

use std::collections::VecDeque;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: profile_offsets <file1> [file2 ...]");
        std::process::exit(1);
    }

    println!(
        "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "File", "BACKREFs", "Hit@1", "Hit@2", "Hit@3", "Pct@1", "Pct@2", "Pct@3"
    );
    println!("{}", "-".repeat(105));

    let mut grand_total  = 0u64;
    let mut grand_hit1   = 0u64;
    let mut grand_hit2   = 0u64;
    let mut grand_hit3   = 0u64;

    for path in &args {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("Cannot read {}: {}", path, e); continue; }
        };

        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path.as_str());

        // Run fold 1 only to get the LZ token stream
        let (compressed, _folds, _paired, ob_per_fold) =
            match mbfa::fold::fold(&data, 1) {
                Ok(r) => r,
                Err(e) => { eprintln!("{}: fold error: {}", name, e); continue; }
            };

        let ob = ob_per_fold.first().copied()
            .unwrap_or(mbfa::opcode::OFFSET_BITS_DEFAULT);

        let tokens = match mbfa::bitreader::read_tokens(&compressed, ob) {
            Ok(t) => t,
            Err(e) => { eprintln!("{}: read_tokens error: {}", name, e); continue; }
        };

        let (total, hit1, hit2, hit3) = profile_tokens(&tokens);

        let pct = |h: u64| if total > 0 { h as f64 / total as f64 * 100.0 } else { 0.0 };

        // Bit savings estimate: each cache hit saves (ob - 2) bits
        // (2 bits to encode which slot vs ob bits raw)
        let savings_bits = hit3 as f64 * (ob as f64 - 2.0);
        let total_backref_bits = total as f64 * ob as f64;
        let savings_pct = if total_backref_bits > 0.0 {
            savings_bits / total_backref_bits * 100.0
        } else {
            0.0
        };

        println!(
            "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9.2}%  {:>9.2}%  {:>9.2}%   est saving ~{:.1}% of offset bits",
            name, total, hit1, hit2, hit3,
            pct(hit1), pct(hit2), pct(hit3),
            savings_pct
        );

        grand_total += total;
        grand_hit1  += hit1;
        grand_hit2  += hit2;
        grand_hit3  += hit3;
    }

    if args.len() > 1 {
        let pct = |h: u64| if grand_total > 0 { h as f64 / grand_total as f64 * 100.0 } else { 0.0 };
        println!("{}", "-".repeat(105));
        println!(
            "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9.2}%  {:>9.2}%  {:>9.2}%",
            "TOTAL", grand_total, grand_hit1, grand_hit2, grand_hit3,
            pct(grand_hit1), pct(grand_hit2), pct(grand_hit3)
        );
    }
}

/// Returns (total_backrefs, hits_with_cache_size_1, size_2, size_3)
fn profile_tokens(tokens: &[mbfa::opcode::Token]) -> (u64, u64, u64, u64) {
    let mut recent: VecDeque<u32> = VecDeque::with_capacity(4);
    let mut total  = 0u64;
    let mut hit1   = 0u64;
    let mut hit2   = 0u64;
    let mut hit3   = 0u64;

    for token in tokens {
        if let mbfa::opcode::Token::Backref { offset, .. } = token {
            total += 1;

            let slots: Vec<u32> = recent.iter().copied().collect();

            if slots.first() == Some(offset) {
                hit1 += 1;
                hit2 += 1;
                hit3 += 1;
            } else if slots.get(1) == Some(offset) {
                hit2 += 1;
                hit3 += 1;
            } else if slots.get(2) == Some(offset) {
                hit3 += 1;
            }

            // Update recent list — push front, cap at 3
            if recent.front() != Some(offset) {
                recent.push_front(*offset);
                if recent.len() > 3 {
                    recent.pop_back();
                }
            }
        }
    }

    (total, hit1, hit2, hit3)
                                               }
