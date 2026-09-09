// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/core.md, section "par.rs"
// ============================================================================

// Compatibility shim so the entropy tournament (lib.rs) and the dual scan
// (encoder.rs) can call `.into_par_iter()` / `join(..)` the same way whether
// or not the "parallel" feature (rayon) is enabled. With it off, these fall
// back to a plain serial iterator and sequential calls -- same output, no
// threads, no call-site changes needed at either usage site.

#[cfg(feature = "parallel")]
pub(crate) use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(not(feature = "parallel"))]
pub(crate) trait IntoParallelIterator: IntoIterator + Sized {
    fn into_par_iter(self) -> <Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}
#[cfg(not(feature = "parallel"))]
impl<T: IntoIterator> IntoParallelIterator for T {}

#[cfg(feature = "parallel")]
pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (a(), b())
}
