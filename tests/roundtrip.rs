// tests/roundtrip.rs
//! End-to-end safety net for the public API: `compress()` -> `decompress()`
//! must reproduce the input byte-for-byte for every shape of data below, and
//! `compress()` must never expand its input by more than the 5-byte
//! passthrough header.
//!
//! Why this file exists
//! --------------------
//! The unit tests live next to each module and check pieces in isolation. The
//! full pipeline (filter detection -> folds -> pair/entropy tournament ->
//! unfold) is otherwise only exercised by the benchmark workflows
//! (`mbfa-corpus.yml`, `mbfa-special.yml`, ...), which only run when a commit
//! message carries a flag. This file runs on every `cargo test`.
//!
//! No extra dependencies: a tiny deterministic PRNG generates the data, so a
//! failure always reproduces from the label printed in the panic message.
//!
//! Layout
//! ------
//! * Default tests are small enough for an unoptimised `cargo test`.
//! * Tests named `heavy_*` are `#[ignore]`d (tens of KB up to a few MB). Run
//!   just those, in release mode, with:
//!       cargo test --release --test roundtrip -- --ignored heavy_
//! * `corpus_dir_roundtrip` is an `#[ignore]` test driven by an environment
//!   variable, so any real-world corpus (Canterbury, Silesia, your own files)
//!   can be checked without editing code:
//!       MBFA_CORPUS_DIR=/path/to/files \
//!         cargo test --release --test roundtrip corpus_dir -- --ignored --nocapture
//!
//! The library prints a lot of diagnostics to stdout. libtest captures that
//! for passing tests, so the noise only shows up when something fails.

use mbfa::filters::{
    apply_filter, undo_filter, FILTER_BCJ, FILTER_BCJ_ARM, FILTER_BCJ_ARM64, FILTER_BCJ_PPC,
    FILTER_BCJ_RISCV, FILTER_BCJ_SPARC,
};
use mbfa::{compress, decompress};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

/// Same fold limit the CLI uses (`src/main.rs`).
const FOLDS: u8 = 8;

/// A passthrough output is `[0,0,0,filter,0] ++ input`, i.e. input + 5.
const MAX_EXPANSION: usize = 5;

/// Extra bytes tolerated on very short inputs. `compress()` has no final
/// "store raw if that is smaller" guard, and `fold.rs` accepts a fold whose
/// payload is under 98.5% of the input (MIN_IMPROVEMENT_RATIO) even though a
/// fold adds two header bytes. For inputs under ~130 bytes that can land the
/// output a byte or two above the passthrough size. The worst case observed
/// over ~307,000 exhaustive tiny inputs (binary up to length 16, ternary up to
/// 10, quaternary up to 8) was 1 byte. From `SHORT_INPUT` bytes upward the
/// bound is exact. Remove this allowance if a final guard is ever added.
const SHORT_INPUT: usize = 128;
const SHORT_INPUT_SLACK: usize = 2;

// ── Deterministic data generation ────────────────────────────────────────────

/// xorshift64* -- small, fast, and good enough for test data.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Never let the state be zero, and spread small seeds.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `block` repeated until the result is exactly `len` bytes long.
fn repeat_to(block: &[u8], len: usize) -> Vec<u8> {
    block.iter().copied().cycle().take(len).collect()
}

/// Pseudo-English text: words drawn with a skewed distribution so some words
/// repeat often and others rarely, like real prose.
fn text_like(rng: &mut Rng, len: usize) -> Vec<u8> {
    const WORDS: &[&str] = &[
        "the", "of", "and", "a", "to", "in", "is", "was", "he", "for", "it", "with", "as", "his",
        "on", "be", "at", "by", "had", "not", "are", "but", "from", "or", "have", "an", "they",
        "which", "one", "you", "were", "her", "all", "she", "there", "would", "their", "we",
        "him", "been", "has", "when", "who", "will", "more", "no", "if", "out", "so", "said",
        "what", "up", "its", "about", "into", "than", "them", "can", "only", "other", "new",
        "some", "could", "time", "these", "two", "may", "then", "do", "first", "any", "my",
        "now", "such", "like", "our", "over", "man", "me", "even", "most", "made", "after",
        "also", "did", "many", "before", "must", "through", "back", "years", "where", "much",
        "your", "way", "well", "down", "should", "because", "each", "just", "those", "people",
        "compression", "algorithm", "instruction", "stream", "token", "offset", "length",
        "quantum", "marmalade", "xylophone", "zeppelin", "obfuscate", "juxtapose",
    ];
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        // Squaring a uniform value skews picks toward the front of the list.
        let r = rng.below(WORDS.len());
        let idx = (r * r) / WORDS.len();
        out.extend_from_slice(WORDS[idx].as_bytes());
        out.push(if rng.below(12) == 0 { b'\n' } else { b' ' });
    }
    out.truncate(len);
    out
}

// ── The check every test funnels through ─────────────────────────────────────

/// Index and neighbourhood of the first differing byte, for readable failures.
fn first_mismatch(a: &[u8], b: &[u8]) -> String {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            let lo = i.saturating_sub(8);
            let hi = (i + 8).min(n);
            return format!(
                "first difference at byte {} (expected {:02x?}, got {:02x?})",
                i,
                &a[lo..hi],
                &b[lo..hi]
            );
        }
    }
    format!("common prefix matches; lengths differ ({} vs {})", a.len(), b.len())
}

/// Compress, decompress, compare, bound the expansion. Returns the compressed
/// size so callers can additionally assert on ratio when that is the point.
fn roundtrip(label: &str, data: &[u8]) -> usize {
    let packed = compress(data, FOLDS)
        .unwrap_or_else(|e| panic!("[{label}] compress failed on {} bytes: {e}", data.len()));
    let unpacked = decompress(&packed)
        .unwrap_or_else(|e| panic!("[{label}] decompress failed on {} bytes: {e}", data.len()));
    if unpacked != data {
        panic!(
            "[{label}] ROUNDTRIP MISMATCH on {} bytes (packed to {}): {}",
            data.len(),
            packed.len(),
            first_mismatch(data, &unpacked)
        );
    }
    let slack = if data.len() < SHORT_INPUT { SHORT_INPUT_SLACK } else { 0 };
    assert!(
        packed.len() <= data.len() + MAX_EXPANSION + slack,
        "[{label}] expanded {} -> {} bytes (limit is input + {})",
        data.len(),
        packed.len(),
        MAX_EXPANSION + slack
    );
    packed.len()
}

// ── Default (fast) tests ─────────────────────────────────────────────────────

#[test]
fn tiny_inputs_every_length() {
    // Lengths 0..=64 for several byte patterns. Covers the empty input, the
    // "shorter than the 5-byte header" region, and the first few LZ matches.
    for len in 0..=64usize {
        roundtrip(&format!("zeros len={len}"), &vec![0u8; len]);
        roundtrip(&format!("ff len={len}"), &vec![0xFFu8; len]);
        let ascending: Vec<u8> = (0..len).map(|i| i as u8).collect();
        roundtrip(&format!("ascending len={len}"), &ascending);
        let mut rng = Rng::new(len as u64 + 1);
        roundtrip(&format!("random len={len}"), &rng.bytes(len));
    }
}

#[test]
fn single_byte_runs_at_boundaries() {
    // Run lengths chosen around the field-width edges: 7-bit offsets (127/128),
    // the 8-bit default length field (255/256), and a few powers of two.
    let lens = [
        1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 126, 127, 128, 129, 254, 255,
        256, 257, 258, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
    ];
    for &len in &lens {
        for &b in &[0x00u8, 0x41, 0xFF] {
            roundtrip(&format!("run byte={b:#04x} len={len}"), &vec![b; len]);
        }
    }
}

#[test]
fn periodic_patterns_overlapping_matches() {
    // A random block of `period` bytes repeated to `len`. With period < match
    // length these are self-overlapping matches (offset smaller than length),
    // the case the decoder handles with its periodic-copy formula.
    let periods = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 63, 64, 65];
    let lens = [
        2usize, 255, 256, 257, 300, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
    ];
    for &period in &periods {
        let mut rng = Rng::new(1_000 + period as u64);
        let block = rng.bytes(period);
        for &len in &lens {
            if len < period {
                continue;
            }
            roundtrip(
                &format!("period={period} len={len}"),
                &repeat_to(&block, len),
            );
        }
    }
}

#[test]
fn long_range_match_across_small_windows() {
    // Block, then `gap` bytes of unrelated data, then the same block again.
    // The second copy must be found (or not) without corrupting anything,
    // whichever side of an offset-width boundary the distance lands on.
    let gaps = [
        0usize, 1, 2, 125, 126, 127, 128, 129, 253, 254, 255, 256, 257, 1023, 1024, 1025, 4093,
        4094, 4095, 4096, 4097,
    ];
    let mut rng = Rng::new(77);
    let block = rng.bytes(300);
    for &gap in &gaps {
        let mut data = block.clone();
        data.extend(rng.bytes(gap));
        data.extend_from_slice(&block);
        roundtrip(&format!("block300 gap={gap}"), &data);
    }
}

#[test]
fn text_like_small_and_medium() {
    for (i, &len) in [1usize, 17, 100, 511, 512, 513, 2_000, 20_000].iter().enumerate() {
        let mut rng = Rng::new(500 + i as u64);
        roundtrip(&format!("text len={len}"), &text_like(&mut rng, len));
    }
}

#[test]
fn random_data_never_expands_beyond_header() {
    // Truly incompressible input must come back as a passthrough: input + 5.
    for &len in &[1usize, 2, 4, 5, 6, 100, 1_000, 10_000] {
        let mut rng = Rng::new(9_000 + len as u64);
        roundtrip(&format!("random len={len}"), &rng.bytes(len));
    }
}

#[test]
fn mixed_regions_text_zeros_random() {
    // Regions with different statistics glued together: exercises the
    // fingerprint/window logic and the entropy tournament on non-stationary
    // input.
    let mut rng = Rng::new(31337);
    let mut data = text_like(&mut rng, 3_000);
    data.extend(vec![0u8; 2_000]);
    data.extend(rng.bytes(1_500));
    data.extend(text_like(&mut rng, 3_000));
    data.extend(repeat_to(b"abcabcabd", 2_500));
    roundtrip("mixed regions", &data);
}

#[test]
fn compressible_input_actually_shrinks() {
    // Guards against a regression that silently turns everything into a
    // passthrough: the roundtrip would still pass, the ratio would not.
    let mut rng = Rng::new(4);
    let text = text_like(&mut rng, 20_000);
    let packed = roundtrip("shrink text", &text);
    assert!(
        packed < text.len() * 3 / 4,
        "20 KB of word-soup should compress below 75%, got {} -> {}",
        text.len(),
        packed
    );

    let zeros = vec![0u8; 100_000];
    let packed = roundtrip("shrink zeros", &zeros);
    assert!(packed < 200, "100 KB of zeros should pack below 200 B, got {packed}");
}

#[test]
fn committed_bench_files_small() {
    // The small committed samples under bench_files/. The large ones are in
    // `heavy_committed_bench_files_large` below.
    for name in ["alice_small.txt", "source_code.rs", "repetitive.txt", "random.bin"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_files").join(name);
        let Ok(data) = std::fs::read(&path) else {
            // Not fatal: the file may have been removed from a stripped checkout.
            eprintln!("skipping {name}: not found at {}", path.display());
            continue;
        };
        roundtrip(name, &data);
    }
}

// ── Executable filters (BCJ) ──────────────────────────────────────────────────
//
// A branch-converter filter must be an exact bijection on ANY byte sequence,
// because the decoder cannot know what the encoder saw. Uniform random bytes
// almost never trigger the hard cases (an E8/E9 opcode whose operand ends in
// 0x00/0xFF, with another opcode within the last three bytes), which is why
// the x86 filter shipped with two bugs that these tests would have caught:
//
//   * the encoder omitted the refinement loop of the reference algorithm, and
//   * the decoder omitted the sign-normalisation of the restored operand.
//
// Found by the Silesia corpus: `ooffice` (a 6 MB DLL) came back with 307 wrong
// bytes, the first at offset 146,293 (`00` restored as `fe`). Against liblzma's
// x86 filter, the old encoder differed in 175 bytes of that file.

/// `(input, output of liblzma's x86 BCJ encoder)`. Every pair was produced with
/// Python's `lzma` module (`FILTER_X86`, then decoded with LZMA2 only to expose
/// the filter's output) and is an input on which the ORIGINAL encoder disagreed
/// with liblzma, so each one fails against the unfixed code.
const X86_BCJ_REFERENCE_VECTORS: &[(&[u8], &[u8])] = &[
    (&[0x8B, 0x80, 0x00, 0x80, 0x7F, 0x80, 0x01, 0xFF, 0xE9, 0x8B, 0xE8, 0xFE, 0xFE, 0x7F, 0xFF, 0x8B, 0xFF, 0x85, 0xFE, 0x7F, 0x0F, 0x7F, 0xE9, 0xE9, 0x85],
     &[0x8B, 0x80, 0x00, 0x80, 0x7F, 0x80, 0x01, 0xFF, 0xE9, 0x8B, 0xE8, 0x01, 0x01, 0x7F, 0xFF, 0x8B, 0xFF, 0x85, 0xFE, 0x7F, 0x0F, 0x7F, 0xE9, 0xE9, 0x85]),
    (&[0x01, 0xE9, 0x85, 0xE8, 0x01, 0xE9, 0xFE, 0xFE, 0x8B, 0x00, 0x7F],
     &[0x01, 0xE9, 0x85, 0xE8, 0x01, 0xE9, 0x01, 0x01, 0x8B, 0x00, 0x7F]),
    (&[0xE8, 0xE9, 0xFE, 0xFF, 0xFE, 0x00, 0x8B, 0xFF, 0x0F, 0xFE, 0x25, 0x80, 0x25, 0xE9, 0x0F, 0xFF],
     &[0xE8, 0xE9, 0x01, 0x00, 0x01, 0x00, 0x8B, 0xFF, 0x0F, 0xFE, 0x25, 0x80, 0x25, 0xE9, 0x0F, 0xFF]),
    (&[0xFF, 0x85, 0x8B, 0xFE, 0x0F, 0xFE, 0xFF, 0xE8, 0x25, 0x00, 0x00, 0x0F, 0x00, 0xFE, 0x85, 0xE9, 0x00, 0x01, 0xE8, 0xE8, 0x01, 0xFE, 0xFF, 0x0F, 0xFF, 0xE8, 0x0F, 0x01, 0xFE, 0x00],
     &[0xFF, 0x85, 0x8B, 0xFE, 0x0F, 0xFE, 0xFF, 0xE8, 0x25, 0x00, 0x00, 0x0F, 0x00, 0xFE, 0x85, 0xE9, 0x00, 0x01, 0xE8, 0x17, 0x01, 0xFE, 0xFF, 0x0F, 0xFF, 0xE8, 0x2D, 0x01, 0xFE, 0x00]),
    (&[0x7F, 0x85, 0x80, 0xE8, 0xFF, 0xFE, 0xE9, 0x80, 0xE9, 0x00, 0xE8, 0xFE, 0xFE, 0x01, 0xFF, 0x7F, 0xFF, 0x0F, 0x00, 0x00, 0x0F, 0xFE, 0x00, 0x0F, 0xFF, 0x25, 0xFE, 0x85, 0xFF, 0xFE],
     &[0x7F, 0x85, 0x80, 0xE8, 0xFF, 0xFE, 0xE9, 0x80, 0xE9, 0x00, 0xE8, 0x01, 0x01, 0x01, 0xFF, 0x7F, 0xFF, 0x0F, 0x00, 0x00, 0x0F, 0xFE, 0x00, 0x0F, 0xFF, 0x25, 0xFE, 0x85, 0xFF, 0xFE]),
    (&[0xFF, 0x85, 0x00, 0x85, 0x85, 0xFF, 0x7F, 0x00, 0xFF, 0x7F, 0x8B, 0x7F, 0x8B, 0x00, 0xE9, 0xE9, 0x01, 0xFF, 0xE9, 0xE8, 0xFF, 0x7F, 0x00, 0x80, 0x00, 0x80, 0x01, 0xE9],
     &[0xFF, 0x85, 0x00, 0x85, 0x85, 0xFF, 0x7F, 0x00, 0xFF, 0x7F, 0x8B, 0x7F, 0x8B, 0x00, 0xE9, 0xE9, 0x01, 0xFF, 0xE9, 0x17, 0xFF, 0x7F, 0x00, 0x80, 0x00, 0x80, 0x01, 0xE9]),
    (&[0x25, 0xFF, 0x7F, 0xFF, 0xE9, 0x80, 0x80, 0x80, 0x80, 0x85, 0xFF, 0x7F, 0x01, 0xFF, 0x25, 0x7F, 0xE9, 0xFF, 0x0F, 0xE9, 0x25, 0xE8, 0xFF, 0xFE, 0x25, 0x00, 0xFE, 0x80],
     &[0x25, 0xFF, 0x7F, 0xFF, 0xE9, 0x80, 0x80, 0x80, 0x80, 0x85, 0xFF, 0x7F, 0x01, 0xFF, 0x25, 0x7F, 0xE9, 0xFF, 0x0F, 0xE9, 0x25, 0xE8, 0x00, 0x01, 0x25, 0x00, 0xFE, 0x80]),
    (&[0xE8, 0x01, 0xE9, 0xFE, 0xFE, 0x01, 0xFF, 0x80, 0x00, 0xE9, 0xFF, 0x8B, 0xFE, 0xFE, 0x0F, 0xFF, 0xE9, 0xE9, 0xFF, 0x25, 0x7F, 0x0F, 0x8B, 0x01, 0x0F, 0x7F, 0x00, 0x0F],
     &[0xE8, 0x01, 0xE9, 0x01, 0x01, 0x01, 0xFF, 0x80, 0x00, 0xE9, 0xFF, 0x8B, 0xFE, 0xFE, 0x0F, 0xFF, 0xE9, 0xE9, 0xFF, 0x25, 0x7F, 0x0F, 0x8B, 0x01, 0x0F, 0x7F, 0x00, 0x0F]),
];

#[test]
fn x86_bcj_matches_liblzma_reference_vectors() {
    for (n, (input, expected)) in X86_BCJ_REFERENCE_VECTORS.iter().enumerate() {
        let encoded = apply_filter(input, FILTER_BCJ);
        assert_eq!(
            &encoded[..], *expected,
            "vector {n}: encoder output differs from liblzma's x86 filter"
        );
        let decoded = undo_filter(expected, FILTER_BCJ);
        assert_eq!(
            &decoded[..], *input,
            "vector {n}: decoder does not invert liblzma's encoding"
        );
    }
}

/// Random bytes drawn from a small pool, so the interesting values (opcodes,
/// 0x00/0xFF operand ends, filler) are dense instead of 1-in-128 rare.
fn skewed(rng: &mut Rng, pool: &[u8], len: usize) -> Vec<u8> {
    (0..len).map(|_| pool[rng.below(pool.len())]).collect()
}

#[test]
fn x86_bcj_is_exactly_invertible_on_adversarial_input() {
    let pool = [0xE8u8, 0xE9, 0x00, 0xFF, 0x00, 0xFF, 0x01, 0x80, 0x25, 0x8B, 0x0F, 0x85, 0xFE, 0x7F];
    let mut rng = Rng::new(86);
    for trial in 0..4000 {
        let len = 5 + rng.below(400);
        let data = if trial % 2 == 0 { skewed(&mut rng, &pool, len) } else { rng.bytes(len) };
        let back = undo_filter(&apply_filter(&data, FILTER_BCJ), FILTER_BCJ);
        assert_eq!(
            back, data,
            "x86 BCJ not invertible (trial {trial}, len {len}): {}",
            first_mismatch(&data, &back)
        );
    }
}

#[test]
fn architecture_bcj_filters_are_exactly_invertible() {
    // Opcode-ish bytes for ARM (BL = 0xEB), ARM64 (BL = 0x94/0x97), PPC (bl),
    // SPARC (call = 0x40..0x7F) and RISC-V (jal 0x6F, auipc 0x17).
    let pool = [
        0xEBu8, 0x94, 0x97, 0x48, 0x49, 0x4B, 0x40, 0x7F, 0x6F, 0x17, 0x97, 0x00, 0xFF, 0x01, 0x80,
    ];
    for (name, flag) in [
        ("arm", FILTER_BCJ_ARM),
        ("arm64", FILTER_BCJ_ARM64),
        ("ppc", FILTER_BCJ_PPC),
        ("sparc", FILTER_BCJ_SPARC),
        ("riscv", FILTER_BCJ_RISCV),
    ] {
        let mut rng = Rng::new(1_000 + flag as u64);
        for trial in 0..2000 {
            let len = 5 + rng.below(300);
            let data = if trial % 2 == 0 { skewed(&mut rng, &pool, len) } else { rng.bytes(len) };
            let back = undo_filter(&apply_filter(&data, flag), flag);
            assert_eq!(
                back, data,
                "{name} BCJ not invertible (trial {trial}, len {len}): {}",
                first_mismatch(&data, &back)
            );
        }
    }
}

#[test]
fn executable_roundtrips_through_the_full_pipeline() {
    // A synthetic x86 image: MZ/PE-looking head plus dense CALL/JMP traffic with
    // operand high bytes of 0x00/0xFF. The whole pipeline (detect -> BCJ -> LZ ->
    // entropy -> unfold -> undo BCJ) must hand back every byte.
    let mut rng = Rng::new(0x0F_F1CE);
    let mut image = vec![0u8; 0x400];
    image[0] = b'M';
    image[1] = b'Z';
    image[0x3C] = 0x80;
    image[0x80..0x84].copy_from_slice(b"PE\0\0");
    let ops = [0xE8u8, 0xE9, 0x8B, 0x55, 0xC3, 0x90, 0x0F, 0x85, 0x33, 0xC0];
    while image.len() < 120_000 {
        match rng.below(6) {
            0 | 1 => {
                // CALL/JMP rel32 with a plausible displacement (negative or positive)
                let rel: i32 = (rng.next() as i32) >> (8 + rng.below(12) as u32);
                image.push(ops[rng.below(2)]);
                image.extend_from_slice(&rel.to_le_bytes());
            }
            _ => image.push(ops[rng.below(ops.len())]),
        }
    }
    roundtrip("synthetic x86 image", &image);
}

// ── Heavy tests (run with --ignored, ideally --release) ──────────────────────

#[test]
#[ignore = "heavy: cargo test --release --test roundtrip -- --ignored heavy_"]
fn heavy_long_range_match_across_large_windows() {
    // Same idea as the small-window test, at the 15/17/18-bit window edges.
    let gaps = [
        32_766usize, 32_767, 32_768, 32_769, 65_279, 65_280, 65_281, 65_535, 65_536, 65_537,
        131_070, 131_071, 131_072, 131_073,
    ];
    let mut rng = Rng::new(78);
    let block = rng.bytes(400);
    for &gap in &gaps {
        let mut data = block.clone();
        data.extend(rng.bytes(gap));
        data.extend_from_slice(&block);
        roundtrip(&format!("block400 gap={gap}"), &data);
    }
}

#[test]
#[ignore = "heavy: cargo test --release --test roundtrip -- --ignored heavy_"]
fn heavy_runs_and_periods_at_large_boundaries() {
    // 255/256 length-field edge, ENTROPY_SAFE_MAX_LENGTH = 65_280, and the
    // 16-bit edge, for both a plain run and short-period repeats.
    let lens = [65_279usize, 65_280, 65_281, 65_535, 65_536, 65_537, 131_072, 262_144];
    for &len in &lens {
        roundtrip(&format!("run len={len}"), &vec![0x5Au8; len]);
        for &period in &[2usize, 3, 5, 64] {
            let mut rng = Rng::new(period as u64);
            let block = rng.bytes(period);
            roundtrip(&format!("period={period} len={len}"), &repeat_to(&block, len));
        }
    }
}

#[test]
#[ignore = "heavy: cargo test --release --test roundtrip -- --ignored heavy_"]
fn heavy_size_thresholds() {
    // Sizes straddling the internal switch points:
    //   50_000  -- incompressible early-exit gate (lib.rs)
    //   262_144 -- largest input the DP-optimal parser (v10) will attempt
    let mut rng = Rng::new(262_144);
    let unit = text_like(&mut rng, 997);
    for &len in &[49_999usize, 50_000, 50_001, 262_143, 262_144, 262_145] {
        roundtrip(&format!("repeating text len={len}"), &repeat_to(&unit, len));
        let mut rng = Rng::new(len as u64);
        roundtrip(&format!("random len={len}"), &rng.bytes(len));
    }
}

#[test]
#[ignore = "heavy: cargo test --release --test roundtrip -- --ignored heavy_"]
fn heavy_text_like_large() {
    for (i, &len) in [100_000usize, 262_144, 600_000].iter().enumerate() {
        let mut rng = Rng::new(700 + i as u64);
        roundtrip(&format!("text len={len}"), &text_like(&mut rng, len));
    }
}

#[test]
#[ignore = "heavy: cargo test --release --test roundtrip -- --ignored heavy_"]
fn heavy_committed_bench_files_large() {
    for name in ["alice.txt", "holmes.txt", "bible.txt"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_files").join(name);
        let Ok(data) = std::fs::read(&path) else {
            eprintln!("skipping {name}: not found at {}", path.display());
            continue;
        };
        roundtrip(name, &data);
    }
}

#[test]
#[ignore = "needs MBFA_CORPUS_DIR=<directory>; run with --ignored --nocapture"]
fn corpus_dir_roundtrip() {
    // Roundtrips every regular file directly inside $MBFA_CORPUS_DIR and
    // prints one result line per file. Point it at Canterbury, Silesia, or a
    // folder of your own samples (video, game assets, ...).
    let Ok(dir) = std::env::var("MBFA_CORPUS_DIR") else {
        panic!("set MBFA_CORPUS_DIR to a directory of files to test");
    };
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    assert!(!entries.is_empty(), "{dir} contains no files");

    let (mut total_in, mut total_out) = (0usize, 0usize);
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let started = std::time::Instant::now();
        let packed = roundtrip(&name, &data);
        eprintln!(
            "corpus {:<24} {:>10} -> {:>10} ({:>6.2}%) {:>7} ms",
            name,
            data.len(),
            packed,
            packed as f64 * 100.0 / data.len().max(1) as f64,
            started.elapsed().as_millis()
        );
        total_in += data.len();
        total_out += packed;
    }
    eprintln!(
        "corpus TOTAL {:>10} -> {:>10} ({:.2}%)",
        total_in,
        total_out,
        total_out as f64 * 100.0 / total_in.max(1) as f64
    );
}

// ── Malformed input ──────────────────────────────────────────────────────────
//
// `decompress` returns `io::Result`, so corrupt or truncated input should come
// back as `Err` (or at worst as wrong bytes) -- never as a panic or an abort.
// These two tests pin that contract. See the audit notes for the known gaps
// they currently expose.

/// Run `decompress` on `input`, converting a panic into `Err(message)`.
fn decompress_caught(input: &[u8]) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(|| decompress(input))) {
        Ok(_) => Ok(()),
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Err(msg)
        }
    }
}

#[test]
#[ignore = "known gap: decompress() can panic on malformed input; un-ignore once hardened"]
fn truncated_streams_do_not_panic() {
    let mut rng = Rng::new(2024);
    let samples: Vec<Vec<u8>> = vec![
        text_like(&mut rng, 4_000),
        repeat_to(b"0123456789", 3_000),
        rng.bytes(2_000),
    ];
    let mut failures = Vec::new();
    for (si, sample) in samples.iter().enumerate() {
        let packed = compress(sample, FOLDS).expect("compress");
        for cut in 0..packed.len().min(80) {
            if let Err(msg) = decompress_caught(&packed[..cut]) {
                failures.push(format!("sample {si}, truncated to {cut} bytes: {msg}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} truncated input(s) panicked:\n{}",
        failures.len(),
        failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
#[ignore = "known gap: decompress() can panic on malformed input; un-ignore once hardened"]
fn corrupted_streams_do_not_panic() {
    // Flip single bytes inside valid streams and feed pure noise.
    let mut rng = Rng::new(99);
    let sample = text_like(&mut rng, 6_000);
    let packed = compress(&sample, FOLDS).expect("compress");
    let mut failures = Vec::new();

    for pos in 0..packed.len().min(64) {
        for &delta in &[1u8, 0x80, 0xFF] {
            let mut bad = packed.clone();
            bad[pos] ^= delta;
            if let Err(msg) = decompress_caught(&bad) {
                failures.push(format!("byte {pos} ^ {delta:#04x}: {msg}"));
            }
        }
    }
    for i in 0..200 {
        let len = 1 + rng.below(300);
        let noise = rng.bytes(len);
        if let Err(msg) = decompress_caught(&noise) {
            failures.push(format!("noise #{i} ({len} bytes): {msg}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} malformed input(s) panicked:\n{}",
        failures.len(),
        failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}
