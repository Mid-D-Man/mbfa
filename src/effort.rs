//! Search effort for large inputs.
//!
//! `Fast` is the behaviour mbfa had before this setting existed: byte for byte
//! the same output. `Normal` (the default) and `Max` spend more time on inputs
//! larger than the optimal-parse limit (`entropy_v9::V9_OPTIMAL_MAX_BYTES`,
//! 256 KB), where the plain hash-chain scan is what decides the ratio:
//!
//! * Candidate parses are compared by their **entropy-coded size** (the v9
//!   range coder's output), not by the size of the raw fixed-width bitstream.
//!   The raw size charges every offset the full window width, so it prefers
//!   narrow windows that the entropy coder would have been happy to leave.
//! * One extra scan is tried with a window wide enough to reach the whole
//!   input, 8-bit lengths (what the v9 coder accepts) and a much deeper hash
//!   chain. It is kept only if its entropy-coded size is smaller.
//! * Matches longer than the v9 coder's limit (272) are split before v9 sees
//!   them, so v9 is available for every fold-1 stream.
//!
//! Nothing here changes the compressed format: the decoder is the same and
//! files made at any effort decompress with any build.
//!
//! Measured on 0.6-2 MB slices of Silesia text/binary files (total size / time
//! relative to `Fast`): Normal about -2.7% / +65%, Max about -3.8% / +300%.
//! Single large files gain more (Silesia `dickens`, 10 MB: -8%).

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    /// The original search. Fastest.
    Fast,
    /// Entropy-scored candidates plus one deep, wide scan (chain depth 512).
    Normal,
    /// Like `Normal` with chain depth 4096. Slow; for archival use.
    Max,
}

static EFFORT: AtomicU8 = AtomicU8::new(1); // Normal

/// Set the process-wide search effort.
pub fn set_effort(e: Effort) {
    EFFORT.store(e as u8, Ordering::Relaxed);
}

/// The current search effort.
pub fn effort() -> Effort {
    match EFFORT.load(Ordering::Relaxed) {
        0 => Effort::Fast,
        1 => Effort::Normal,
        _ => Effort::Max,
    }
}

impl Effort {
    /// Hash-chain depth of the extra deep scan, `None` for `Fast`.
    pub(crate) fn deep_chain(self) -> Option<usize> {
        match self {
            Effort::Fast => None,
            Effort::Normal => Some(512),
            Effort::Max => Some(4096),
        }
    }
}

/// Smallest input the extra search is applied to: above the optimal-parse
/// limit, where fold-1 tokens decide the result.
pub(crate) const TUNED_MIN_BYTES: usize = crate::entropy_v9::V9_OPTIMAL_MAX_BYTES + 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_roundtrips_and_fast_has_no_deep_scan() {
        let before = effort();
        for e in [Effort::Fast, Effort::Normal, Effort::Max] {
            set_effort(e);
            assert_eq!(effort(), e);
        }
        assert_eq!(Effort::Fast.deep_chain(), None);
        assert!(Effort::Normal.deep_chain().unwrap() < Effort::Max.deep_chain().unwrap());
        set_effort(before);
    }
}
