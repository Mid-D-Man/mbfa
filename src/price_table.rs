// src/price_table.rs
//! Fractional-bit price lookup for the DP optimal parser (optimal_parse.rs).
//!
//! ## Why this exists
//!
//! The DP parser (see optimal_parse.rs) needs to ask "how many bits would
//! encoding choice X cost under the CURRENT adaptive model" tens of
//! thousands of times per file, at every lookahead position, for every
//! candidate (literal / fresh backref / each of 4 rep slots). Computing
//! `-log2(prob/2048)` per query is a real cost at that call volume. This
//! file replaces it with a single array lookup, exactly like every real
//! LZMA implementation does.
//!
//! ## Where this table came from
//!
//! Ported byte-for-byte from lzma-rust2's `src/enc/range_enc.rs` (the
//! `PRICES: &[u8; 128]` static table + `get_bit_price`/`get_bit_tree_price`/
//! `get_reverse_bit_tree_price`/`get_direct_bits_price`). This is safe to
//! port verbatim rather than regenerate, because MBFA's v7 (and the v9
//! extension this file supports -- see entropy.rs's v9 additions)
//! provably uses the identical probability scale, checked directly against
//! entropy.rs's real constants:
//!
//!   MBFA (entropy.rs)              LZMA (lzma-rust2, crate root lib.rs)
//!   RC_PROB_BITS  = 11        ==   BIT_MODEL_TOTAL_BITS = 11
//!   RC_PROB_SCALE = 2048       ==   BIT_MODEL_TOTAL      = 2048
//!   RC_SHIFT      = 5          ==   MOVE_BITS             = 5
//!   RC_PROB_INIT  = 1024       ==   PROB_INIT             = 1024
//!
//! Same scale, same adaptation rate, same init -- the price table's math
//! (`prob >> MOVE_REDUCING_BITS` indexing a `2^(11-4)=128`-entry table of
//! `-log2(p/2048)*16`-ish fixed-point values) transfers over exactly.
//! Prices are in units of 1/16 bit (`BIT_PRICE_SHIFT_BITS = 4`), matching
//! LZMA's convention -- so a "free" bit (prob near certainty) costs near 0,
//! and a genuinely 50/50 bit costs close to 16 (one full bit).

const MOVE_REDUCING_BITS: u32 = 4;
const BIT_PRICE_SHIFT_BITS: u32 = 4;

/// 128-entry price table, ported verbatim from lzma-rust2's `range_enc.rs`.
/// Index = `prob >> MOVE_REDUCING_BITS` (prob is an 11-bit value, 0..2048,
/// so this is a 128-bucket coarsening of it). Value = price in 1/16-bit
/// units of coding a bit whose probability-of-zero is `prob` with the
/// GIVEN actual bit -- see `get_bit_price` for how bit=1 is handled via
/// the XOR-complement trick (avoids needing a second table).
#[rustfmt::skip]
static PRICES: [u8; 128] = [
    0x80, 0x67, 0x5B, 0x54, 0x4E, 0x49, 0x45, 0x42, 0x3F, 0x3D, 0x3A, 0x38, 0x36, 0x34, 0x33, 0x31,
    0x30, 0x2E, 0x2D, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x22, 0x21,
    0x20, 0x1F, 0x1F, 0x1E, 0x1D, 0x1D, 0x1C, 0x1C, 0x1B, 0x1A, 0x1A, 0x19, 0x19, 0x18, 0x18, 0x17,
    0x17, 0x16, 0x16, 0x16, 0x15, 0x15, 0x14, 0x14, 0x13, 0x13, 0x13, 0x12, 0x12, 0x11, 0x11, 0x11,
    0x10, 0x10, 0x10, 0x0F, 0x0F, 0x0F, 0x0E, 0x0E, 0x0E, 0x0D, 0x0D, 0x0D, 0x0C, 0x0C, 0x0C, 0x0B, 0x0B, 0x0B,
    0x0B, 0x0A, 0x0A, 0x0A, 0x0A, 0x09, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x08, 0x07, 0x07, 0x07, 0x07, 0x06, 0x06,
    0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04, 0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01,
];

const RC_PROB_SCALE: u32 = 2048; // matches entropy.rs's RC_PROB_SCALE

/// Price (in 1/16-bit units) of coding one bit with probability-of-zero
/// `prob` (an 11-bit value, 1..=2047) when the actual bit is `bit` (0 or 1).
///
/// The XOR trick: for bit=0, `PRICES[prob >> 4]` directly. For bit=1, we
/// need the price under the complement `2047 - prob` (this table's index
/// space runs 0..2047, so the exact complement of an 11-bit value within
/// that space is `value XOR 2047`, i.e. flip all 11 bits) -- computed
/// without a branch via `prob ^ ((0 - bit) & 2047)`: when bit=0 the mask
/// is all-zero (no-op), when bit=1 the mask is all-ones over the low 11
/// bits, so the XOR flips every bit of `prob`, landing exactly on
/// `2047 - prob`. This is the literal transform every reference LZMA/7-Zip
/// implementation uses (`GetPrice0`/`GetPrice1`), not an approximation.
#[inline(always)]
pub fn get_bit_price(prob: u16, bit: u32) -> u32 {
    debug_assert!(bit == 0 || bit == 1);
    let prob = prob as u32;
    let i = (prob ^ ((0u32.wrapping_sub(bit)) & (RC_PROB_SCALE - 1))) >> MOVE_REDUCING_BITS;
    PRICES[i as usize] as u32
}

/// Price of encoding `symbol` through an adaptive bit-tree of `num_bits`
/// levels, using the EXACT indexing convention entropy.rs's real
/// `encode_bittree` uses: `probs[base + ctx - 1]` where `ctx` starts at 1
/// and doubles (plus the new bit) each level -- i.e. the first-accessed
/// slot is `probs[base]`, not `probs[base+1]` (LZMA's own C-array
/// convention reserves index 0 and starts at 1; MBFA's does not -- ported
/// against MBFA's actual code, not assumed from LZMA's).
pub fn get_bittree_price(probs: &[u16], base: usize, num_bits: u32, symbol: u32) -> u32 {
    let mut price = 0u32;
    let mut ctx = 1u32;
    for i in (0..num_bits).rev() {
        let bit = (symbol >> i) & 1;
        price += get_bit_price(probs[base + ctx as usize - 1], bit);
        ctx = (ctx << 1) | bit;
    }
    price
}

/// Price of one bit of entropy.rs's real `encode_direct` -- per-POSITION
/// (not per-value, not reverse-tree) adaptive coding: bit `i` of `bits`
/// total always uses `probs[base + i]` regardless of the bits coded
/// before it (MSB-first, matching encode_direct's real loop). This is
/// genuinely different from LZMA's own "direct bits" (LZMA's are fully
/// non-adaptive raw bits beyond the 4-bit aligned tail) -- MBFA models
/// every position adaptively instead. See price_extra_bits below for the
/// composite helper that sums this across all `bits` positions for a
/// given `val`.
#[inline(always)]
pub fn get_direct_bit_price(probs: &[u16], base: usize, i: usize, bit: u32) -> u32 {
    get_bit_price(probs[base + i], bit)
}

/// Price of encode_direct's full `bits`-wide adaptive MSB-first field for
/// value `val`, summing get_direct_bit_price across every position.
pub fn get_direct_price(probs: &[u16], base: usize, bits: u32, val: u32) -> u32 {
    let mut price = 0u32;
    for i in (0..bits).rev() {
        price += get_direct_bit_price(probs, base, i as usize, (val >> i) & 1);
    }
    price
}

/// Price of `count` bits written with a truly NON-adaptive raw/50-50
/// scheme (no probability model at all -- every such bit costs exactly
/// one full bit, 16 price-units, by construction). MBFA's rc_encode_distance
/// doesn't currently have a tier like this (see get_direct_price above,
/// which IS what it actually uses), but this is kept for LZMA-comparison
/// purposes and any future fully-raw tier.
#[inline(always)]
pub fn get_raw_bits_price(count: u32) -> u32 {
    count << BIT_PRICE_SHIFT_BITS
}

/// One bit's worth of price, in whole-bit terms, for sanity-checking /
/// tests: exactly 16 price-units by construction (`BIT_PRICE_SHIFT_BITS`).
#[cfg(test)]
pub const ONE_BIT: u32 = 1 << BIT_PRICE_SHIFT_BITS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certain_bit_is_nearly_free() {
        // prob near 2047 (bit=0 almost certain): coding bit=0 should be cheap.
        let cheap = get_bit_price(2047, 0);
        assert!(cheap < ONE_BIT / 4, "expected < 4 price-units, got {}", cheap);
    }

    #[test]
    fn certain_bit_wrong_guess_is_expensive() {
        // Same skewed prob, but the actual bit is the UNLIKELY one (1):
        // should cost close to (or more than) several whole bits.
        let expensive = get_bit_price(2047, 1);
        assert!(expensive > ONE_BIT * 4, "expected > 4 whole bits, got {}", expensive);
    }

    #[test]
    fn fair_coin_costs_about_one_bit() {
        // prob near the 1024 midpoint (RC_PROB_INIT): both outcomes should
        // cost close to one full bit (16 price-units), like an unmodeled
        // fair coin flip would.
        let p0 = get_bit_price(1024, 0);
        let p1 = get_bit_price(1024, 1);
        for p in [p0, p1] {
            assert!(
                (ONE_BIT as i32 - p as i32).abs() <= 3,
                "expected close to {} price-units for a fair-coin bit, got {}",
                ONE_BIT, p
            );
        }
    }

    #[test]
    fn bittree_price_is_sum_of_bit_prices() {
        // 3-bit tree, all probs at RC_PROB_INIT (1024): every path should
        // cost close to 3 full bits (48 price-units), same reasoning as
        // fair_coin_costs_about_one_bit but composed across 3 levels.
        // base=0, needs probs[0..7] per entropy.rs's real base+ctx-1 indexing
        // (ctx ranges 1..8 for a 3-bit tree, so base+ctx-1 ranges 0..7).
        let probs = [1024u16; 7];
        let price = get_bittree_price(&probs, 0, 3, 5);
        assert!(
            (price as i32 - 3 * ONE_BIT as i32).abs() <= 10,
            "expected close to {} for a 3-bit fair tree, got {}",
            3 * ONE_BIT, price
        );
    }

    #[test]
    fn direct_price_matches_manual_bit_sum() {
        let probs = [1200u16, 800, 1024, 1024];
        // val=0b1010=10, MSB-first over positions i=3..0: probs[i] pairs
        // with bit (val>>i)&1 -- position i, not tree depth, indexes probs
        // directly per encode_direct's real `probs[base + i]` convention.
        // i=3: bit=(10>>3)&1=1, probs[3]=1024
        // i=2: bit=(10>>2)&1=0, probs[2]=1024
        // i=1: bit=(10>>1)&1=1, probs[1]=800
        // i=0: bit=(10>>0)&1=0, probs[0]=1200
        let expected = get_bit_price(1024, 1) + get_bit_price(1024, 0)
            + get_bit_price(800, 1) + get_bit_price(1200, 0);
        let got = get_direct_price(&probs, 0, 4, 0b1010);
        assert_eq!(got, expected);
    }

    #[test]
    fn raw_bits_price_is_exactly_whole_bits() {
        assert_eq!(get_raw_bits_price(5), 5 * ONE_BIT);
        assert_eq!(get_raw_bits_price(0), 0);
    }
}
