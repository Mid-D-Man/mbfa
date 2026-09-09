// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/entropy.md, section "price_table.rs"
// ============================================================================
//! Fractional-bit price lookup for the DP optimal parser (optimal_parse.rs).
//!
//! The DP parser needs to ask "how many bits would encoding choice X cost
//! under the current adaptive model" at every lookahead position, for every
//! candidate. This file replaces the `-log2(prob/2048)` computation with a
//! single array lookup, ported from lzma-rust2's range coder price tables --
//! MBFA's range coder (see entropy.rs's v7/v9) uses the identical
//! probability scale (11-bit probs, 0..2048 range), so the table transfers
//! directly. Prices are in units of 1/16 bit: a near-certain bit costs close
//! to 0, a genuinely 50/50 bit costs close to 16.

const MOVE_REDUCING_BITS: u32 = 4;
const BIT_PRICE_SHIFT_BITS: u32 = 4;

/// 128-entry price table, ported verbatim from lzma-rust2's `range_enc.rs`.
/// Index = `prob >> MOVE_REDUCING_BITS` (prob is an 11-bit value, 0..2048,
/// coarsened to 128 buckets). Value = price in 1/16-bit units of coding a
/// bit whose probability-of-zero is `prob`, for the bit actually seen --
/// see `get_bit_price` for how bit=1 is handled via the XOR-complement trick.
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
/// bit=0 looks up `PRICES[prob >> 4]` directly. bit=1 needs the price under
/// the complement `2047 - prob`; `prob ^ ((0 - bit) & 2047)` computes that
/// branchlessly (the mask is all-zero when bit=0, all-ones over the low 11
/// bits when bit=1, so the XOR flips every bit of `prob` exactly onto its
/// complement). Same technique as the reference LZMA/7-Zip `GetPrice0`/`GetPrice1`.
#[inline(always)]
pub fn get_bit_price(prob: u16, bit: u32) -> u32 {
    debug_assert!(bit == 0 || bit == 1);
    let prob = prob as u32;
    let i = (prob ^ ((0u32.wrapping_sub(bit)) & (RC_PROB_SCALE - 1))) >> MOVE_REDUCING_BITS;
    PRICES[i as usize] as u32
}

/// Price of encoding `symbol` through an adaptive bit-tree of `num_bits`
/// levels. Uses entropy.rs's real `encode_bittree` indexing convention:
/// `probs[base + ctx - 1]` where `ctx` starts at 1 and doubles (plus the new
/// bit) each level, so the first-accessed slot is `probs[base]`.
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

/// Price of one bit of entropy.rs's real `encode_direct` -- per-position
/// (not per-value) adaptive coding: bit `i` of `bits` total always uses
/// `probs[base + i]`, regardless of the bits coded before it (MSB-first).
/// See `price_extra_bits` below for the composite helper across all
/// positions for a given `val`.
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

/// Price of `count` bits written with a non-adaptive raw/50-50 scheme --
/// every such bit costs exactly one full bit (16 price-units) by construction.
#[inline(always)]
pub fn get_raw_bits_price(count: u32) -> u32 {
    count << BIT_PRICE_SHIFT_BITS
}

/// One bit's worth of price, in whole-bit terms, for sanity-checking / tests:
/// exactly 16 price-units by construction (`BIT_PRICE_SHIFT_BITS`).
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
        // Same skewed prob, but the actual bit is the unlikely one (1):
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
        // cost close to 3 full bits (48 price-units). base=0 needs
        // probs[0..7] per entropy.rs's real base+ctx-1 indexing (ctx ranges
        // 1..8 for a 3-bit tree, so base+ctx-1 ranges 0..7).
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
