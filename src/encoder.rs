// src/encoder.rs
// LZ-style scanner with rolling-window hash chain.
// O(n) time, O(window) memory -- safe for large files.
// Both offset_bits and length_bits are adaptive at runtime.
//
// scan_adaptive: Phase A/B/C fingerprint pipeline.
//
//   Phase A/B -- fingerprint + single scan (LARGE FILES ONLY, > 1 MB):
//     Phase A fingerprint_predict classifies the input and predicts (ob, lb).
//     Phase B runs one full-quality LZ scan at the predicted parameters, then
//     applies three checks IN STRICT ORDER before deciding whether to run Phase C:
//
//       1. Length peak saturation (HARD, runs before early exit):
//          Any Backref length >= 90% of max_len. Means lb is too small.
//          MUST run before early exit — Repetitive_2MB has a very low Phase B
//          ratio but lb=8 is genuinely wrong (every match hits max_len=255).
//          Early exit would bypass Phase C and return ob=17 lb=8 giving 183B
//          instead of the correct 131B at ob=10 lb=21.
//          Does NOT fire for JSON_2MB (matches 5-30 bytes, lb_ceil=229 never hit).
//
//       1.5. Expansion bail (NEW — runs in the else branch after step 1 clears):
//          If Phase B raw_cost > input_bits * 1.02 (2% expansion), the data is
//          incompressible at these parameters. Since step 1 already confirmed lb
//          is adequate, no Phase C can help. Returns incompressible signal.
//          Unity_Terrain_2K: was 1003ms, now ~40ms (in-scan bail at first 64KB).
//          Unity_Terrain_1K: was 159ms, now ~20ms (parallel path bail).
//          Does NOT fire when step 1 fires (wrong lb might cause apparent expansion
//          that wider lb would fix — Phase C is still warranted in that case).
//
//       2. Early exit (safe ONLY after length peak passes AND expansion bail clears):
//          If Phase B ratio < PHASE_B_EARLY_EXIT_RATIO (15%), Phase C gain is
//          negligible. Calibrated from JSON_2MB: ratio=8.41%.
//
//       3. Offset peak + upper-half (softer, runs only when early exit doesn't fire):
//          Any Backref offset >= 90% of max_off, OR >= 20% of Backrefs in
//          upper half of window. Catches WarAndPeace at ob=17.
//
//   In-scan expansion bail (inside scan() itself):
//     Checked every SCAN_EXPANSION_CHECK_INTERVAL (64KB) of input consumed.
//     Fires when BOTH: token_bits > input_bits * 1.02 AND lit_fraction > 95%.
//     The dual condition prevents false positives on data like STL (7.8% backrefs,
//     92% literals — below the 95% threshold, bail does not fire even though
//     token_bits is near input_bits). Terrain (4.8% backrefs = 95.2% literals —
//     above threshold, bail fires at first check after 64KB). WarAndPeace (70.9%
//     backrefs = 29.1% literals — far below threshold, never fires).
//     Returns (Vec::new(), true) meaning "incompressible, discard".
//
//   Return type of scan_adaptive changed to (Vec<Token>, u32, u32, bool) where
//   the bool = is_incompressible. Callers check this before using tokens.
//
//   Phase A fingerprint rules (applied only when input > 1 MB):
//     Rule 1: entropy < 2.0 (highly repetitive) -> defer to Phase C.
//     Rule 2: size < 32 KB -> predict ob = ceil(log2(size)), lb = 8.
//     Rule 3: default -> predict baseline (ob=17, lb=8).
//
//   Phase C -- discovery path: unchanged from previous version.
//
//   Dynamic chain limit (full scan only): unchanged.
//
//   log_window_diagnostics and log_phase_c_gain are defined in fold.rs and
//   called there after scan_adaptive returns. Do NOT add them here.

use crate::opcode::{
    Token, LIT_TOTAL_BITS, END_TOTAL_BITS, backref_total_bits,
    max_offset, max_length,
    OFFSET_BITS_MIN, OFFSET_BITS_MAX, LENGTH_BITS_MAX, LENGTH_BITS_MIN,
    compute_optimal_offset_bits, compute_optimal_length_bits,
};
use crate::entropy::{offset_to_bucket, bucket_extra_bits};
use rayon;

const BASELINE_OFFSET_BITS:       u32   = 17;
const BASELINE_LENGTH_BITS:       u32   = LENGTH_BITS_MIN; // 8
const ENTROPY_SAFE_LENGTH_BITS:   u32   = 15;
const ENTROPY_MIN_BYTES_FOR_SCAN: usize = 400;
const ENTROPY_TIE_THRESHOLD:      f64   = 0.05;

// -- Phase B early-exit threshold ---------------------------------------------
const PHASE_B_EARLY_EXIT_RATIO: f64 = 0.15;

// -- Fingerprint constants ----------------------------------------------------
const FINGERPRINT_ENTROPY_REPETITIVE: f64   = 2.0;
const FINGERPRINT_SMALL_FILE_BYTES:   usize = 32768;

// -- Ceiling saturation constants ---------------------------------------------
const CEILING_SATURATION_THRESHOLD: f64 = 0.9;
const CEILING_UPPER_HALF_THRESHOLD: f64 = 0.20;

// -- Rep-match tiebreaking ----------------------------------------------------
const REP_SLOTS: usize = 3;

const HASH_SIZE:  usize = 1 << 16;
const HASH_MASK:  usize = HASH_SIZE - 1;
const LAZY_SHORT_LEN: usize = 6;

const PARALLEL_SCAN_THRESHOLD: usize = 1_048_576;

// -- Discovery scan constants -------------------------------------------------
const DISCOVER_CHAIN_LIMIT:   usize = 256;
const DISCOVER_MAX_PREV_SIZE: usize = 2 * 1024 * 1024;

// -- In-scan expansion bail ---------------------------------------------------
/// If token_bits * 100 > input_bits * this value, data is expanding.
/// 102 = 2% expansion threshold.
const EXPANSION_BAIL_PCT: u64  = 102;

/// Check for expansion every N input bytes (must be a power of 2 for bitmask check).
const SCAN_EXPANSION_CHECK_INTERVAL: usize = 65_536; // 64 KB
const SCAN_EXPANSION_INTERVAL_MASK: usize  = SCAN_EXPANSION_CHECK_INTERVAL - 1;

/// If literal token fraction > this percent, data is likely incompressible.
/// Terrain (4.8% backrefs → 95.2% literals): fires. STL (7.8% backrefs → 92.2%
/// literals): does NOT fire. Both conditions must be true to bail.
const EXPANSION_LIT_PCT: usize = 95;

/// Minimum token count before the in-scan expansion check is meaningful.
const EXPANSION_MIN_TOKENS: usize = 128;

// -- Tiered chain limit -------------------------------------------------------

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

// -- Window helpers -----------------------------------------------------------

#[inline]
fn window_alloc_and_mask(max_off_limit: usize, n: usize) -> (usize, usize) {
    let raw  = max_off_limit.min(n).max(1);
    let pow2 = raw.next_power_of_two();
    (pow2, pow2 - 1)
}

// -- Hash ---------------------------------------------------------------------

#[inline]
fn hash3(input: &[u8], pos: usize) -> usize {
    if pos + 2 >= input.len() { return 0; }
    let v = (input[pos]     as usize).wrapping_mul(2_654_435_761)
        ^   (input[pos + 1] as usize).wrapping_mul(2_246_822_519)
        ^   (input[pos + 2] as usize).wrapping_mul(3_266_489_917);
    v & HASH_MASK
}

// -- Rep-match slot tracker ---------------------------------------------------

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
        if REP_SLOTS > 2 { self.slots[2] = self.slots[1]; }
        if REP_SLOTS > 1 { self.slots[1] = self.slots[0]; }
        self.slots[0] = offset;
        if self.len < REP_SLOTS { self.len += 1; }
    }

    #[inline]
    fn valid(&self) -> &[u32] { &self.slots[..self.len] }
}

// -- Rep-match length probe ---------------------------------------------------

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

// -- Core scanner -------------------------------------------------------------
//
// Returns (tokens, bailed_early) where bailed_early=true means the scan
// detected clear incompressibility and returned early. In that case, the
// returned token Vec is empty and must be discarded by the caller.
//
// The in-scan bail fires when BOTH:
//   (a) running token bit cost exceeds input bit cost by >2%, AND
//   (b) more than 95% of tokens so far are literals.
//
// This dual condition prevents false positives on data like STL meshes that
// have ~8% backrefs (slightly below the 95% literal threshold) while reliably
// catching terrain data with ~5% backrefs (slightly above 95%).

pub fn scan(input: &[u8], offset_bits: u32, length_bits: u32) -> (Vec<Token>, bool) {
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
    let mut lit_count: usize = 0; // tracks literal token count for expansion check

    while i < n {
        // -- In-scan expansion bail (every 64 KB) ---------------------------------
        if i > 0 && (i & SCAN_EXPANSION_INTERVAL_MASK) == 0 {
            let total_tokens = tokens.len();
            if total_tokens >= EXPANSION_MIN_TOKENS {
                let backref_count = total_tokens - lit_count;
                // Integer arithmetic only — no floats in hot path.
                // token_bits = lit_count*10 + backref_count*(1+ob+lb)
                let token_bits: u64 = (lit_count * (LIT_TOTAL_BITS as usize)
                    + backref_count * (backref_bits as usize)) as u64;
                let input_bits: u64 = (i as u64) * 8;
                // Condition (a): expanding by more than 2%
                let expanding = token_bits * 100 > input_bits * EXPANSION_BAIL_PCT;
                // Condition (b): more than 95% of tokens are literals
                // (lit_count / total_tokens > 0.95 → lit_count * 100 > total_tokens * 95)
                let mostly_lits = lit_count * 100 > total_tokens * EXPANSION_LIT_PCT;
                if expanding && mostly_lits {
                    // Incompressible: return empty vec with bail signal.
                    // Caller (scan_adaptive) will detect this and return its own
                    // incompressible signal up to fold(), which passhthroughs.
                    return (Vec::new(), true);
                }
            }
        }

        let h = hash3(input, i);
        let (mut best_offset, mut best_len) =
            find_match(input, i, h, &head, &prev, max_off, max_len, window_mask, chain_limit);

        let ob_beats_lit = offset_bits > LIT_TOTAL_BITS;

        for &slot_off in rep_slots.valid() {
            if slot_off as usize > max_off { continue; }
            let rlen = rep_match_len(input, i, slot_off, max_len);
            if rlen == 0 { continue; }

            let prefer = if rlen >= best_len {
                true
            } else if ob_beats_lit && best_len > 0 && rlen == best_len.saturating_sub(1) {
                true
            } else {
                false
            };

            if prefer {
                best_offset = slot_off as usize;
                best_len    = rlen;
                break;
            }
        }

        let backref_worthwhile = best_len >= 2
            && backref_bits < (best_len as u32 * LIT_TOTAL_BITS);

        if backref_worthwhile {
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
                rep_slots.push(best_offset as u32);
                tokens.push(Token::Backref {
                    offset: best_offset as u32,
                    length: best_len as u32,
                });
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
    (tokens, false) // normal completion
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

fn stream_bit_cost(tokens: &[Token], ob: u32, lb: u32) -> u64 {
    tokens.iter().map(|t| match t {
        Token::Lit { .. }     => LIT_TOTAL_BITS as u64,
        Token::Backref { .. } => backref_total_bits(ob, lb) as u64,
        Token::End            => END_TOTAL_BITS as u64,
    }).sum()
}

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

// -- Scan result wrapper ------------------------------------------------------

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
                self.label, self_e,
                other.label, other_e,
                if self_e < other_e { self.label } else { other.label },
            );
            self_e < other_e
        } else {
            self.raw_cost < other.raw_cost
        }
    }
}

// -- Fingerprint helpers ------------------------------------------------------

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

// -- Ceiling checks -----------------------------------------------------------

fn length_peak_saturated(tokens: &[Token], lb: u32) -> bool {
    let max_len = max_length(lb);
    let lb_ceil = ((max_len as f64) * CEILING_SATURATION_THRESHOLD) as u32;

    for t in tokens {
        if let Token::Backref { length, .. } = t {
            if *length >= lb_ceil {
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
                upper_frac * 100.0,
                CEILING_UPPER_HALF_THRESHOLD * 100.0,
            );
            return true;
        }
        println!(
            "  ceiling check: ok -- upper-half {:.1}% < {:.0}% threshold",
            upper_frac * 100.0,
            CEILING_UPPER_HALF_THRESHOLD * 100.0,
        );
    }
    false
}

// -- Phase C inner logic ------------------------------------------------------

fn phase_c_from_baseline(
    baseline:       ScanResult,
    wide_discovery: Vec<Token>,
    input:          &[u8],
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

    // Constrained re-scan at discovered parameters. If the scan bails
    // (incompressible even with wider parameters), fall back to baseline.
    let (constrained_tokens, constrained_bailed) = scan(input, wide_ob, wide_lb);
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
                scan(input, constrained.ob, ENTROPY_SAFE_LENGTH_BITS);

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

// -- Public API ---------------------------------------------------------------
//
// Returns (tokens, offset_bits, length_bits, is_incompressible).
// When is_incompressible=true: tokens is empty, caller should passthrough
// original input without folding (fold_count=0 in the file header).

pub fn scan_adaptive(input: &[u8]) -> (Vec<Token>, u32, u32, bool) {

    if input.len() > PARALLEL_SCAN_THRESHOLD {
        match fingerprint_predict(input) {
            None => {
                // Rule 1: highly repetitive -- fall through to Phase C.
            }
            Some((pred_ob, pred_lb)) => {
                // Phase B: full-quality scan at fingerprint-predicted parameters.
                let (phase_b_tokens, phase_b_bailed) = scan(input, pred_ob, pred_lb);
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

                // ── Step 1: Length peak saturation (HARD, before early exit) ──────
                if length_peak_saturated(&result.tokens, pred_lb) {
                    println!("  Step 1: length peak -- Phase C required");
                    if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                        let baseline  = ScanResult { label: "baseline", ..result };
                        let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                        let (tokens, ob, lb) = phase_c_from_baseline(baseline, discovery, input);
                        return (tokens, ob, lb, false);
                    }
                    // pred_ob != baseline (Rule 2, not reached for >1MB in practice):
                    // fall through to parallel Phase C below.
                } else {
                    // ── Step 1.5: Expansion bail (NEW — lb confirmed OK at step 1) ─
                    //
                    // If Phase B is already expanding (ratio > 1.02), Phase C cannot
                    // help — the data is incompressible with any parameters. This fires
                    // for smooth terrain (4.8% backrefs, ratio ~1.10) but NOT for
                    // WarAndPeace (70.9% backrefs, ratio ~0.47) or JSON (ratio ~0.08).
                    //
                    // NOT reached when step 1 fires: wrong lb might cause apparent
                    // expansion that wider lb would fix. Phase C is still warranted.
                    {
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

                    // ── Step 2: Early exit (safe — lb OK at 1, not expanding at 1.5) ─
                    //
                    // JSON_2MB: ratio=8.41% < 15% → exits fast at ob=17.
                    // WarAndPeace: ratio ~60% >> 15% → does NOT early-exit.
                    // Terrain: never reaches here (step 1.5 fired).
                    let phase_b_ratio = result.raw_cost as f64
                        / (input.len() as f64 * 8.0);

                    if phase_b_ratio < PHASE_B_EARLY_EXIT_RATIO {
                        println!(
                            "  Step 2: early exit ratio={:.4}% < {:.0}% threshold \
                             -- lb confirmed ok at step 1, skipping Phase C",
                            phase_b_ratio * 100.0,
                            PHASE_B_EARLY_EXIT_RATIO * 100.0,
                        );
                        return (result.tokens, result.ob, result.lb, false);
                    }

                    // ── Step 3: Offset peak + upper-half ───────────────────────────
                    //
                    // WarAndPeace: some backrefs near 128KB edge → fires → Phase C.
                    // JSON_2MB: never reaches (early exit at step 2).
                    if offset_or_upper_half_saturated(&result.tokens, pred_ob) {
                        println!("  Step 3: offset/upper-half -- Phase C needed");
                        if pred_ob == BASELINE_OFFSET_BITS && pred_lb == BASELINE_LENGTH_BITS {
                            let baseline  = ScanResult { label: "baseline", ..result };
                            let discovery = scan_discover(input, OFFSET_BITS_MAX, LENGTH_BITS_MAX);
                            let (tokens, ob, lb) = phase_c_from_baseline(baseline, discovery, input);
                            return (tokens, ob, lb, false);
                        }
                        // pred_ob != baseline: fall through to parallel Phase C below.
                    } else {
                        println!("  Step 3: ceiling clear -- Phase B wins (1 scan total)");
                        return (result.tokens, result.ob, result.lb, false);
                    }
                }
            }
        }
    }

    // ── Parallel Phase C path ─────────────────────────────────────────────────
    // Reached by:
    //   (a) files <= PARALLEL_SCAN_THRESHOLD — rayon::join baseline + discovery
    //   (b) files > threshold, fingerprint returned None (Rule 1: repetitive)
    //   (c) files > threshold, pred_ob != BASELINE (Rule 2, not reached in practice)
    //
    // Expansion bail fires after the parallel scans complete if baseline is
    // still expanding. This saves the Phase C re-scan for clearly incompressible
    // data (e.g. Unity_Terrain_1K at 526KB hits this path and bails here).

    let (baseline, wide_discovery) = if input.len() <= PARALLEL_SCAN_THRESHOLD {
        let ((bt_tokens, bt_bailed), disc) = rayon::join(
            || scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS),
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
        // Sequential fallback for large files in the parallel path (Rule 1 / Rule 2 cases)
        let (bt_tokens, bt_bailed) = scan(input, BASELINE_OFFSET_BITS, BASELINE_LENGTH_BITS);
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

    // Post-scan expansion check: complete scan may still be expanding.
    // This catches cases where the in-scan bail didn't fire (e.g. expansion
    // ratio was just below 102% per interval but accumulates to > 102% overall).
    {
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

    let (tokens, ob, lb) = phase_c_from_baseline(baseline, wide_discovery, input);
    (tokens, ob, lb, false)
}
