// src/encoder.rs
// LZ-style scanner with rolling-window hash chain.
// O(n) time, O(window) memory -- safe for large files.
// Both offset_bits and length_bits are adaptive at runtime.
//
// scan_adaptive: Phase A/B/C fingerprint pipeline.
//
//   P6: scan() emits Token::RepRef for ring-buffer hits, saving (ob-4) bits
//   per hit vs Token::Backref.  REP_SLOTS=4. RepSlots mirrors decoder LRU.
//
//   P6 FIX 1 (ref_worthwhile): use backref_bits for BOTH Backref and RepRef.
//   RepRef must REPLACE a Backref, never CREATE new short matches.
//
//   P6 FIX 2 (stream_bit_cost): RepRef counted at backref_total_bits (not
//   repref_total_bits) so Phase C ob/lb selection is identical to pre-P6.
//   The actual write_tokens call still encodes RepRef at 13 bits — all P6
//   savings are preserved in the real output.
//
//   P6 FIX 3 (prefer logic): RepRef only fires when rlen >= best_len.
//   The old ob_beats_lit preference (RepRef wins with rlen == best_len-1)
//   caused the Phase C constrained scan to produce more tokens than pre-P6
//   (same cost per token via fix 2 but 1 fewer byte covered per RepRef).
//   This inflated constrained raw_cost, making ob=19 appear worse than ob=17
//   for terrain, and choosing wrong ob/lb for 3D formats. Removing ob_beats_lit
//   makes Phase C comparison identical to pre-P6 (token structure preserved,
//   only some Backref→RepRef substitutions at equal length which fix 2 treats
//   identically). Same-length ring hits still save ob-4 bits in write_tokens.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits, repref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MIN, OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};
use crate::entropy::offset_to_bucket;
use rayon;

const BASELINE_OFFSET_BITS:       u32   = 17;
const BASELINE_LENGTH_BITS:       u32   = LENGTH_BITS_MIN; // 8
const ENTROPY_SAFE_LENGTH_BITS:   u32   = 15;
const ENTROPY_MIN_BYTES_FOR_SCAN: usize = 400;
const ENTROPY_TIE_THRESHOLD:      f64   = 0.05;

const PHASE_B_EARLY_EXIT_RATIO: f64 = 0.15;

const FINGERPRINT_ENTROPY_REPETITIVE: f64   = 2.0;
const FINGERPRINT_SMALL_FILE_BYTES:   usize = 32768;

const CEILING_SATURATION_THRESHOLD: f64 = 0.9;
const CEILING_UPPER_HALF_THRESHOLD: f64 = 0.20;

const REP_SLOTS: usize = 4;

const HASH_SIZE:  usize = 1 << 16;
const HASH_MASK:  usize = HASH_SIZE - 1;
const LAZY_SHORT_LEN: usize = 6;

const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;

const EXPANSION_BAIL_PCT: u64  = 102;
const SCAN_EXPANSION_CHECK_INTERVAL: usize = 65_536;
const SCAN_EXPANSION_INTERVAL_MASK: usize  = SCAN_EXPANSION_CHECK_INTERVAL - 1;
const EXPANSION_LIT_PCT: usize = 95;
const EXPANSION_MIN_TOKENS: usize = 128;

pub fn compute_chain_limit(input_len: usize, offset_bits: u32) -> usize {
    match offset_bits {
        0..=16 => 256,
        17..=18 => {
            if input_len > PARALLEL_SCAN_THRESHOLD { 64 } else { 512 }
        }
        19..=20 => 256,
        _       => 128,
    }
}

#[inline]
fn window_alloc_and_mask(max_off_limit: usize, n: usize) -> (usize, usize) {
    let raw  = max_off_limit.min(n).max(1);
    let pow2 = raw.next_power_of_two();
    (pow2, pow2 - 1)
}

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos]     as usize).wrapping_mul(2_654_435_761)
        ^   (input[pos + 1] as usize).wrapping_mul(2_246_822_519)
        ^   (input[pos + 2] as usize).wrapping_mul(3_266_489_917);
    v & HASH_MASK
}

struct RepSlots {
    slots: [u32; REP_SLOTS],
    len:   usize,
}

impl RepSlots {
    #[inline]
    fn new() -> Self { Self { slots: [0u32; REP_SLOTS], len: 0 } }

    #[inline]
    fn push(&mut self, offset: u32) {
        if self.len > 0 && self.slots[0] == offset { return; }
        if REP_SLOTS > 3 { self.slots[3] = self.slots[2]; }
        if REP_SLOTS > 2 { self.slots[2] = self.slots[1]; }
        if REP_SLOTS > 1 { self.slots[1] = self.slots[0]; }
        self.slots[0] = offset;
        if self.len < REP_SLOTS { self.len += 1; }
    }

    #[inline]
    fn move_to_front(&mut self, k: usize) {
        if k == 0 { return; }
        let offset = self.slots[k];
        for j in (1..=k).rev() { self.slots[j] = self.slots[j - 1]; }
        self.slots[0] = offset;
    }

    #[inline]
    fn valid(&self) -> &[u32] { &self.slots[..self.len] }
}

#[inline]
fn rep_match_len(input: &[u8], i: usize, offset: u32, max_len: usize) -> usize {
    let off = offset as usize;
    if off == 0 || off > i { return 0; }
    let j = i - off;
    let n = input.len();
    let mut len = 0;
    while len < max_len && (i + len) < n && input[j + (len % off)] == input[i + len] {
        len += 1;
    }
    len
}

// Core scanner.
//
// P6 FIX 1: ref_worthwhile uses backref_bits for both Backref and RepRef.
// P6 FIX 3: prefer = rlen >= best_len  (no ob_beats_lit -1 advantage).
//   RepRef only fires when ring finds equal-or-longer match than hash-chain.
//   This keeps Phase C token structure identical to pre-P6 (fix 3 + fix 2
//   together make stream_bit_cost comparison equivalent to pre-P6).

pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32, skip_bail: bool) -> (Vec<Token>, bool) {
    let max_off      = max_offset(offset_bits);
    let max_len      = max_length(length_bits);
    let backref_bits = backref_total_bits(offset_bits, length_bits);
    let chain_limit  = compute_chain_limit(input.len(), offset_bits);

    let n = input.len();
    let (window_size, window_mask) = window_alloc_and_mask(max_off, n);

    let mut head      = vec![u32::MAX; HASH_SIZE];
    let mut prev      = vec![u32::MAX; window_size];
    let mut tokens    = Vec::with_capacity(n / 2 + 1);
    let mut rep_slots = RepSlots::new();
    let mut i         = 0;
    let mut lit_count: usize = 0;

    while i < n {
        if i > 0 && (i & SCAN_EXPANSION_INTERVAL_MASK) == 0 && !skip_bail {
            let total_tokens = tokens.len();
            if total_tokens >= EXPANSION_MIN_TOKENS {
                let backref_count = total_tokens - lit_count;
                let token_bits: u64 = (lit_count * (LIT_TOTAL_BITS as usize)
                    + backref_count * (backref_bits as usize)) as u64;
                let input_bits: u64 = (i as u64) * 8;
                let expanding   = token_bits * 100 > input_bits * EXPANSION_BAIL_PCT;
                let mostly_lits = lit_count * 100 > total_tokens * EXPANSION_LIT_PCT;
                if expanding && mostly_lits {
                    return (Vec::new(), true);
                }
            }
        }

        let h = hash3(input, i);
        let (mut best_offset, mut best_len) =
            find_match(input, i, h, &head, &prev, max_off, max_len, window_mask, chain_limit);

        let mut best_slot: Option<usize> = None;

        for (k, &slot_off) in rep_slots.valid().iter().enumerate() {
            if slot_off as usize > max_off { continue; }
            let rlen = rep_match_len(input, i, slot_off, max_len);
            if rlen == 0 { continue; }

            // P6 FIX 3: only prefer RepRef when it finds an equal-or-longer match.
            // Removing the ob_beats_lit (-1 length) preference prevents Phase C
            // ob/lb selection from being biased by artificially shorter RepRef
            // matches that inflate the constrained scan's token count.
            if rlen >= best_len {
                best_offset = slot_off as usize;
                best_len    = rlen;
                best_slot   = Some(k);
                break;
            }
        }

        // P6 FIX 1: use backref_bits for BOTH Backref and RepRef.
        let ref_worthwhile = best_len >= 2
            && backref_bits < (best_len as u32 * LIT_TOTAL_BITS);

        if ref_worthwhile {
            let lazy = if i + 1 < n {
                let h1 = hash3(input, i + 1);
                let (_, len1) = find_match(
                    input, i + 1, h1, &head, &prev, max_off, max_len, window_mask, chain_limit,
                );
                if len1 > best_len {
                    true
                } else if best_len <= LAZY_SHORT_LEN && i + 2 < n {
                    let h2 = hash3(input, i + 2);
                    let (_, len2) = find_match(
                        input, i + 2, h2, &head, &prev, max_off, max_len, window_mask, chain_limit,
                    );
                    len2 > best_len + 2
                } else {
                    false
                }
            } else {
                false
            };

            if lazy {
                prev[i & window_mask] = head[h];
                head[h] = i as u32;
                tokens.push(Token::Lit { byte: input[i] });
                lit_count += 1;
                i += 1;
            } else {
                for k in 0..best_len {
                    if i + k + 2 < n {
                        let hk = hash3(input, i + k);
                        prev[(i + k) & window_mask] = head[hk];
                        head[hk] = (i + k) as u32;
                    }
                }
                if let Some(k) = best_slot {
                    rep_slots.move_to_front(k);
                    tokens.push(Token::RepRef {
                        slot:   k as u8,
                        length: best_len as u32,
                    });
                } else {
                    rep_slots.push(best_offset as u32);
                    tokens.push(Token::Backref {
                        offset: best_offset as u32,
                        length: best_len as u32,
                    });
                }
                i += best_len;
            }
        } else {
            prev[i & window_mask] = head[h];
            head[h] = i as u32;
            tokens.push(Token::Lit { byte: input[i] });
            lit_count += 1;
            i += 1;
        }
    }

    tokens.push(Token::End);
    (tokens, false)
}

pub fn scan_discover(input: &[u8], offset_bits: u32, length_bits: u32) -> Vec<Token> {
    let max_off = max_offset(offset_bits).min(DISCOVER_MAX_PREV_SIZE);
    let max_len = max_length(length_bits);
    let n       = input.len();

    let (window_size, window_mask) = window_alloc_and_mask(max_off, n);

    let mut head   = vec![u32::MAX; HASH_SIZE];
    let mut prev   = vec![u32::MAX; window_size];
    let mut tokens = Vec::with_capacity(n / 2 + 1);
    let mut i      = 0;

    while i < n {
        let h = hash3(input, i);
        let (best_offset, best_len) = find_match(
            input, i, h, &head, &prev,
            max_off, max_len, window_mask,
            DISCOVER_CHAIN_LIMIT,
        );

        prev[i & window_mask] = head[h];
        head[h] = i as u32;

        if best_len >= 2 {
            tokens.push(Token::Backref {
                offset: best_offset as u32,
                length: best_len as u32,
            });
            i += best_len;
        } else {
            tokens.push(Token::Lit { byte: input[i] });
            i += 1;
        }
    }

    tokens.push(Token::End);
    tokens
}

fn find_match(
    input:       &[u8],
    i:           usize,
    h:           usize,
    head:        &[u32],
    prev:        &[u32],
    max_off:     usize,
    max_len:     usize,
    window_mask: usize,
    chain_limit: usize,
) -> (usize, usize) {
    let n = input.len();
    let mut best_offset = 0;
    let mut best_len    = 0;
    let mut steps       = 0;

    let mut cur = head[h];
    while cur != u32::MAX && steps < chain_limit {
        let j = cur as usize;
        if i <= j || i - j > max_off { break; }

        let span = i - j;
        let mut len = 0;
        while len < max_len
            && (i + len) < n
            && input[j + (len % span)] == input[i + len]
        {
            len += 1;
        }

        if len > best_len {
            best_len    = len;
            best_offset = span;
            if best_len == max_len { break; }
        }

        cur   = prev[j & window_mask];
        steps += 1;
    }

    (best_offset, best_len)
}

// -- Cost helpers -------------------------------------------------------------

/// Raw bit cost for Phase C ob/lb selection.
///
/// P6 FIX 2: RepRef counted at backref_total_bits(ob, lb) so Phase C sees the
/// same cost structure as pre-P6. Real write_tokens still encodes RepRef at
/// repref_total_bits (5+lb) — all P6 savings are preserved in actual output.
fn stream_bit_cost(tokens: &[Token], ob: u32, lb: u32) -> u64 {
    let br_bits = backref_total_bits(ob, lb) as u64;
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => br_bits,
        Token::RepRef { .. }  => br_bits,   // P6 FIX 2
        Token::End            => END_TOTAL_BITS as u64,
    }).sum()
}

/// Estimated entropy cost (tiebreaker only — uses accurate RepRef cost here).
fn stream_entropy_cost(tokens: &[Token]) -> f64 {
    use std::collections::HashMap;

    let mut lit_len_freq: HashMap<u32, u64> = HashMap::new();
    let mut bucket_freq:  HashMap<u32, u64> = HashMap::new();
    let mut raw_extra:    u64               = 0;

    for t in tokens {
        match t {
            Token::Lit { byte } => {
                *lit_len_freq.entry(*byte as u32).or_insert(0) += 1;
            }
            Token::Backref { offset, length } => {
                let length_sym = 255u32 + length;
                *lit_len_freq.entry(length_sym).or_insert(0) += 1;
                let (bucket, extra_bits, _) = offset_to_bucket(*offset);
                *bucket_freq.entry(bucket).or_insert(0) += 1;
                raw_extra += extra_bits as u64;
            }
            Token::RepRef { length, .. } => {
                let length_sym = 255u32 + length;
                *lit_len_freq.entry(length_sym).or_insert(0) += 1;
                raw_extra += 2u64;
            }
            Token::End => {
                *lit_len_freq.entry(256u32).or_insert(0) += 1;
            }
        }
    }

    shannon_bits(&lit_len_freq) + shannon_bits(&bucket_freq) + raw_extra as f64
}

#[inline]
fn shannon_bits(freq: &std::collections::HashMap<u32, u64>) -> f64 {
    let total: u64 = freq.values().sum();
    if total == 0 { return 0.0; }
    let total_f = total as f64;
    freq.values()
        .filter(|&&c| c > 0)
        .map(|&c| -(c as f64) * (c as f64 / total_f).log2())
        .sum()
}

struct ScanResult {
    tokens:   Vec<Token>,
    ob:       u32,
    lb:       u32,
    raw_cost: u64,
    label:    &'static str,
}

impl ScanResult {
    fn new(tokens: Vec<Token>, ob: u32, lb: u32, label: &'static str) -> Self {
        let raw_cost = stream_bit_cost(&tokens, ob, lb);
        ScanResult { tokens, ob, lb, raw_cost, label }
    }

    fn beats(&self, other: &ScanResult) -> bool {
        let min_cost = self.raw_cost.min(other.raw_cost) as f64;
        let diff     = (self.raw_cost as f64 - other.raw_cost as f64).abs();

        if min_cost > 0.0 && diff / min_cost < ENTROPY_TIE_THRESHOLD {
            let self_e  = stream_entropy_cost(&self.tokens);
            let other_e = stream_entropy_cost(&other.tokens);
            println!(
                "  entropy tiebreaker: {} est={:.0}b vs {} est={:.0}b -> {} wins",
                self.label, self_e, other.label, other_e,
                if self_e < other_e { self.label } else { other.label },
            );
            self_e < other_e
        } else {
            self.raw_cost < other.raw_cost
        }
    }
}

fn sample_entropy_fingerprint(data: &[u8]) -> f64 {
    const SAMPLE: usize = 8192;
    let sample = if data.len() > SAMPLE { &data[..SAMPLE] } else { data };
    if sample.is_empty() { return 0.0; }
    let mut freq = [0u32; 256];
    for &b in sample { freq[b as usize] += 1; }
    let n = sample.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
}

fn fingerprint_predict(data: &[u8]) -> Option<(u32, u32)> {
    let ent = sample_entropy_fingerprint(data);
    println!("  fingerprint: entropy={:.2} size={}", ent, data.len());

    if ent < FINGERPRINT_ENTROPY_REPETITIVE {
        println!("  fingerprint Rule 1: repetitive -> defer to Phase C");
        return None;
    }

    if data.len() < FINGERPRINT_SMALL_FILE_BYTES {
        let ob = if data.is_empty() {
            OFFSET_BITS_MIN
        } else {
            let bits = (usize::BITS - data.len().leading_zeros()) as u32;
            bits.clamp(OFFSET_BITS_MIN, OFFSET_BITS_MAX)
        };
        println!("  fingerprint Rule 2: small file -> ob={} lb={}", ob, LENGTH_BITS_MIN);
        return Some((ob, LENGTH_BITS_MIN));
    }

    println!("  fingerprint Rule 3: default -> ob={} lb={}", BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
    Some((BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS))
}

fn length_peak_saturated(tokens: &[Token], lb: u32) -> bool {
    let max_len = max_length(lb);
    let lb_ceil = ((max_len as f64) * CEILING_SATURATION_THRESHOLD) as u32;

    for t in tokens {
        let length_opt = match t {
            Token::Backref { length, .. } => Some(*length),
            Token::RepRef  { length, .. } => Some(*length),
            _ => None,
        };
        if let Some(length) = length_opt {
            if length >= lb_ceil {
                println!(
                    "  ceiling check: length peak hit {} >= lb_ceil={} (max_len={} lb={})",
                    length, lb_ceil, max_len, lb
                );
                return true;
            }
        }
    }
    println!("  ceiling check: length peak clear (lb={} lb_ceil={})", lb, lb_ceil);
    false
}

fn offset_or_upper_half_saturated(tokens: &[Token], ob: u32) -> bool {
    let max_off  = max_offset(ob);
    let ob_ceil  = ((max_off as f64) * CEILING_SATURATION_THRESHOLD) as u32;
    let half_off = (max_off / 2) as u32;

    let mut total_br:   u64 = 0;
    let mut upper_half: u64 = 0;

    for t in tokens {
        if let Token::Backref { offset, .. } = t {
            if *offset >= ob_ceil {
                println!(
                    "  ceiling check: offset peak hit {} >= ob_ceil={} (max_off={} ob={})",
                    offset, ob_ceil, max_off, ob
                );
                return true;
            }
            total_br += 1;
            if *offset > half_off { upper_half += 1; }
        }
    }

    if total_br > 0 {
        let upper_frac = upper_half as f64 / total_br as f64;
        if upper_frac >= CEILING_UPPER_HALF_THRESHOLD {
            println!(
                "  ceiling check: upper-half {:.1}% >= {:.0}% threshold -- Phase C needed",
                upper_frac * 100.0, CEILING_UPPER_HALF_THRESHOLD * 100.0,
            );
            return true;
        }
        println!(
            "  ceiling check: ok -- upper-half {:.1}% < {:.0}% threshold",
            upper_frac * 100.0, CEILING_UPPER_HALF_THRESHOLD * 100.0,
        );
    }
    false
}

fn phase_c_from_baseline(
    baseline:                 ScanResult,
    wide_discovery:           Vec<Token>,
    input:                    &[u8],
    skip_incompressible_bail: bool,
) -> (Vec<Token>, u32, u32) {

    println!(
        "  chain_limit (baseline ob={}): {}  raw_cost={}",
        BASELINE_OFFSET_BITS,
        compute_chain_limit(input.len(), BASELINE_OFFSET_BITS),
        baseline.raw_cost,
    );

    let wide_ob = compute_optimal_offset_bits(&wide_discovery);
    let wide_lb = compute_optimal_length_bits(&wide_discovery);
    drop(wide_discovery);

    println!(
        "  discovery (capped 2MB/chain={}): discovered ob={} lb={}",
        DISCOVER_CHAIN_LIMIT, wide_ob, wide_lb,
    );

    if wide_ob == BASELINE_OFFSET_BITS && wide_lb == BASELINE_LENGTH_BITS {
        println!("  wide agrees with baseline -- no re-scan needed");
        return (baseline.tokens, baseline.ob, baseline.lb);
    }

    let (constrained_tokens, constrained_bailed) =
        scan(input, wide_ob, wide_lb, skip_incompressible_bail);
    if constrained_bailed {
        println!("  Phase C constrained scan bailed early — baseline wins (still expanding)");
        return (baseline.tokens, baseline.ob, baseline.lb);
    }

    let constrained = ScanResult::new(constrained_tokens, wide_ob, wide_lb, "constrained");
    println!(
        "  chain_limit (constrained ob={}): {}  raw_cost={}",
        wide_ob,
        compute_chain_limit(input.len(), wide_ob),
        constrained.raw_cost,
    );

    if constrained.lb > ENTROPY_SAFE_LENGTH_BITS {
        let constrained_bytes = (constrained.raw_cost as usize + 7) / 8;
        if constrained_bytes >= ENTROPY_MIN_BYTES_FOR_SCAN {
            let (capped_tokens, capped_bailed) =
                scan(input, constrained.ob, ENTROPY_SAFE_LENGTH_BITS, skip_incompressible_bail);

            if capped_bailed {
                println!("  Phase C capped scan bailed — baseline wins");
                return (baseline.tokens, baseline.ob, baseline.lb);
            }

            let capped = ScanResult::new(
                capped_tokens,
                constrained.ob,
                ENTROPY_SAFE_LENGTH_BITS,
                "capped",
            );
            println!(
                "  lb cap: constrained lb={} output={}B >= threshold {}B \
                 -- capped raw_cost={} chain_limit={}",
                constrained.lb, constrained_bytes, ENTROPY_MIN_BYTES_FOR_SCAN,
                capped.raw_cost,
                compute_chain_limit(input.len(), capped.ob),
            );

            return if capped.beats(&baseline) {
                println!("  capped wins over baseline");
                (capped.tokens, capped.ob, capped.lb)
            } else {
                println!("  baseline wins over capped");
                (baseline.tokens, baseline.ob, baseline.lb)
            };
        }
    }

    if constrained.beats(&baseline) {
        println!(
            "  constrained wins: ob={} lb={} raw_cost={} vs baseline raw_cost={}",
            constrained.ob, constrained.lb, constrained.raw_cost, baseline.raw_cost,
        );
        (constrained.tokens, constrained.ob, constrained.lb)
    } else {
        println!(
            "  baseline wins: ob={} lb={} raw_cost={} vs constrained raw_cost={}",
            baseline.ob, baseline.lb, baseline.raw_cost, constrained.raw_cost,
        );
        (baseline.tokens, baseline.ob, baseline.lb)
    }
}

pub fn scan_adaptive(input: &[u8], skip_incompressible_bail: bool) -> (Vec<Token>, u32, u32, bool) {

    if input.len() > PARALLEL_SCAN_THRESHOLD {
        match fingerprint_predict(input) {
            None => {
                // Rule 1: highly repetitive -- fall through to Phase C.
            }
            Some((pred_ob, pred_lb)) => {
                let (phase_b_tokens, phase_b_bailed) =
                    scan(input, pred_ob, pred_lb, skip_incompressible_bail);
                if phase_b_bailed {
                    println!("  Phase B: in-scan expansion bail — incompressible, skipping Phase C");
                    return (Vec::new(), pred_ob, pred_lb, true);
                }

                let result = ScanResult::new(phase_b_tokens, pred_ob, pred_lb, "phase_b");

                println!(
                    "  Phase B scan: ob={} lb={} chain_limit={} raw_cost={}",
                    pred_ob, pred_lb,
                    compute_chain_limit(input.len(), pred_ob),
                    result.raw_cost,
                );

                if length_peak_saturated(&result.tokens, pred_lb) {
                    println!("  Step 1: length peak -- Phase C required");
                    if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                        let baseline  = ScanResult { label: "baseline", ..result };
                        let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                        let (tokens, ob, lb) = phase_c_from_baseline(
                            baseline, discovery, input, skip_incompressible_bail);
                        return (tokens, ob, lb, false);
                    }
                } else {
                    if !skip_incompressible_bail {
                        let input_bits = input.len() as u64 * 8;
                        if result.raw_cost * 100 >= input_bits * EXPANSION_BAIL_PCT {
                            println!(
                                "  Step 1.5: Phase B expanding ({:.3}) — incompressible, \
                                 no Phase C",
                                result.raw_cost as f64 / input_bits as f64
                            );
                            return (Vec::new(), pred_ob, pred_lb, true);
                        }
                    }

                    let phase_b_ratio = result.raw_cost as f64 / (input.len() as f64 * 8.0);

                    if phase_b_ratio < PHASE_B_EARLY_EXIT_RATIO {
                        println!(
                            "  Step 2: early exit ratio={:.4}% < {:.0}% threshold \
                             -- lb confirmed ok at step 1, skipping Phase C",
                            phase_b_ratio * 100.0,
                            PHASE_B_EARLY_EXIT_RATIO * 100.0,
                        );
                        return (result.tokens, result.ob, result.lb, false);
                    }

                    if offset_or_upper_half_saturated(&result.tokens, pred_ob) {
                        println!("  Step 3: offset/upper-half -- Phase C needed");
                        if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                            let baseline  = ScanResult { label: "baseline", ..result };
                            let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                            let (tokens, ob, lb) = phase_c_from_baseline(
                                baseline, discovery, input, skip_incompressible_bail);
                            return (tokens, ob, lb, false);
                        }
                    } else {
                        println!("  Step 3: ceiling clear -- Phase B wins (1 scan total)");
                        return (result.tokens, result.ob, result.lb, false);
                    }
                }
            }
        }
    }

    // Parallel Phase C path (files <= 1MB)
    let (baseline, wide_discovery) = if input.len() <= PARALLEL_SCAN_THRESHOLD {
        let ((bt_tokens, bt_bailed), disc) = rayon::join(
            || scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, skip_incompressible_bail),
            || scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX),
        );
        if bt_bailed {
            println!("  Parallel baseline: in-scan expansion bail — incompressible");
            return (Vec::new(), BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, true);
        }
        (
            ScanResult::new(bt_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, "baseline"),
            disc,
        )
    } else {
        let (bt_tokens, bt_bailed) =
            scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, skip_incompressible_bail);
        if bt_bailed {
            println!("  Sequential baseline (parallel path): expansion bail — incompressible");
            return (Vec::new(), BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, true);
        }
        let disc = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
        (
            ScanResult::new(bt_tokens, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, "baseline"),
            disc,
        )
    };

    if !skip_incompressible_bail {
        let input_bits = input.len() as u64 * 8;
        if baseline.raw_cost * 100 >= input_bits * EXPANSION_BAIL_PCT {
            println!(
                "  Parallel path post-scan: baseline expanding ({:.3}) — \
                 incompressible, skipping Phase C re-scan",
                baseline.raw_cost as f64 / input_bits as f64
            );
            return (Vec::new(), BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS, true);
        }
    }

    println!(
        "  chain_limit (baseline ob={}): {}  raw_cost={}",
        BASELINE_OFFSET_BITS,
        compute_chain_limit(input.len(), BASELINE_OFFSET_BITS),
        baseline.raw_cost,
    );

    let (tokens, ob, lb) =
        phase_c_from_baseline(baseline, wide_discovery, input, skip_incompressible_bail);
    (tokens, ob, lb, false)
}
