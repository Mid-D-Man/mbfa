// src/profile_offsets.rs
//
// P6 change: profile top-4 ring slots (was top-3) to match MAX_RING_SLOTS = 4.

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
        "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}",
        "File", "BACKREFs", "Hit@1", "Hit@2", "Hit@3", "Hit@4",
        "Pct@1", "Pct@2", "Pct@3", "Pct@4"
    );
    println!("{}", "-".repeat(118));

    let mut grand_total = 0u64;
    let mut grand_hit1  = 0u64;
    let mut grand_hit2  = 0u64;
    let mut grand_hit3  = 0u64;
    let mut grand_hit4  = 0u64;

    for path in &args {
        let data = match fs::read(path) {
            Ok(d)  => d,
            Err(e) => { eprintln!("Cannot read {}: {}", path, e); continue; }
        };

        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path.as_str());

        let (compressed, _folds, _paired, ob_per_fold, lb_per_fold, _fold1_tokens, _ring) =
            match mbfa::fold::fold(&data, 1, 0) {
                Ok(r)  => r,
                Err(e) => { eprintln!("{}: fold error: {}", name, e); continue; }
            };

        let ob = ob_per_fold.first().copied()
            .unwrap_or(mbfa::opcode::OFFSET_BITS_DEFAULT);
        let lb = lb_per_fold.first().copied()
            .unwrap_or(mbfa::opcode::LENGTH_BITS_DEFAULT);

        // P6: read with ring_active=false for profiling (fold doesn't emit RepRef
        // yet until encoder.rs is updated; safe to keep false here).
        let tokens = match mbfa::bitreader::read_tokens(&compressed, ob, lb, false) {
            Ok(t)  => t,
            Err(e) => { eprintln!("{}: read_tokens error: {}", name, e); continue; }
        };

        let (total, hit1, hit2, hit3, hit4) = profile_tokens(&tokens);
        let pct = |h: u64| if total > 0 { h as f64 / total as f64 * 100.0 } else { 0.0 };
        let savings_bits = hit4 as f64 * (ob as f64 - 4.0); // RepRef saves ob-4 bits per hit
        let total_br_bits = total as f64 * ob as f64;
        let savings_pct = if total_br_bits > 0.0 {
            savings_bits / total_br_bits * 100.0
        } else {
            0.0
        };

        println!(
            "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  \
             {:>8.2}%  {:>8.2}%  {:>8.2}%  {:>8.2}%   \
             est P6 saving ~{:.1}% of offset bits",
            name, total, hit1, hit2, hit3, hit4,
            pct(hit1), pct(hit2), pct(hit3), pct(hit4), savings_pct
        );

        grand_total += total;
        grand_hit1  += hit1;
        grand_hit2  += hit2;
        grand_hit3  += hit3;
        grand_hit4  += hit4;
    }

    if args.len() > 1 {
        let pct = |h: u64| if grand_total > 0 {
            h as f64 / grand_total as f64 * 100.0
        } else { 0.0 };
        println!("{}", "-".repeat(118));
        println!(
            "{:<25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  \
             {:>8.2}%  {:>8.2}%  {:>8.2}%  {:>8.2}%",
            "TOTAL", grand_total, grand_hit1, grand_hit2, grand_hit3, grand_hit4,
            pct(grand_hit1), pct(grand_hit2), pct(grand_hit3), pct(grand_hit4)
        );
    }
}

/// Profile top-4 ring slot hit rates in a token stream.
/// Returns (total_backrefs, hit1, hit2, hit3, hit4) where hitN means
/// the offset matched one of the N most-recently-seen offsets.
fn profile_tokens(tokens: &[mbfa::opcode::Token])
    -> (u64, u64, u64, u64, u64)
{
    let mut recent: VecDeque<u32> = VecDeque::with_capacity(5);
    let mut total = 0u64;
    let mut hit1  = 0u64;
    let mut hit2  = 0u64;
    let mut hit3  = 0u64;
    let mut hit4  = 0u64;

    for token in tokens {
        let offset = match token {
            mbfa::opcode::Token::Backref { offset, .. } => *offset,
            mbfa::opcode::Token::RepRef { slot, .. } => {
                // RepRef already IS a ring hit; count it at the appropriate slot.
                let s = *slot as usize;
                let slots: Vec<u32> = recent.iter().copied().collect();
                total += 1;
                if s == 0 { hit1 += 1; hit2 += 1; hit3 += 1; hit4 += 1; }
                else if s == 1 { hit2 += 1; hit3 += 1; hit4 += 1; }
                else if s == 2 { hit3 += 1; hit4 += 1; }
                else if s == 3 { hit4 += 1; }
                // Move matched slot to front (LRU).
                if s < slots.len() {
                    if recent.front() != Some(&slots[s]) {
                        recent.retain(|&x| x != slots[s]);
                        recent.push_front(slots[s]);
                        if recent.len() > 4 { recent.pop_back(); }
                    }
                }
                continue;
            }
            _ => continue,
        };

        // Backref: check how many recent offsets it matches.
        total += 1;
        let slots: Vec<u32> = recent.iter().copied().collect();
        if slots.first() == Some(&offset) {
            hit1 += 1; hit2 += 1; hit3 += 1; hit4 += 1;
        } else if slots.get(1) == Some(&offset) {
            hit2 += 1; hit3 += 1; hit4 += 1;
        } else if slots.get(2) == Some(&offset) {
            hit3 += 1; hit4 += 1;
        } else if slots.get(3) == Some(&offset) {
            hit4 += 1;
        }

        // LRU update.
        if recent.front() != Some(&offset) {
            recent.retain(|&x| x != offset);
            recent.push_front(offset);
            if recent.len() > 4 { recent.pop_back(); }
        }
    }

    (total, hit1, hit2, hit3, hit4)
            }
