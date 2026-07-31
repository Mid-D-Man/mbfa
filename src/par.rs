// src/par.rs
//
// Thin compatibility shim so the entropy tournament in lib.rs and the dual
// scan in encoder.rs can call `.into_par_iter()` / `par::join(..)` whether
// or not the "parallel" feature (rayon) is enabled, without maintaining two
// copies of either call site.
//
// With "parallel" on: these are rayon's real thread-pool-backed versions.
// With "parallel" off (e.g. wasm32-unknown-unknown, which has no threads
// without the separate Web Worker + SharedArrayBuffer plumbing
// wasm-bindgen-rayon needs): `.into_par_iter()` falls back to a plain
// std::vec::IntoIter, and `join(a, b)` just calls `a()` then `b()`. Same
// output either way, just serial instead of threaded -- the tournament's
// closures were already `Fn`, which both `Iterator::map` and rayon's
// `ParallelIterator::map` accept, so no call-site changes were needed
// beyond swapping which trait is in scope.

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
