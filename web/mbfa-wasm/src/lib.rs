// web/mbfa-wasm/src/lib.rs
//! WASM bindings for MBFA.
//!
//! Two functions, both thin wrappers over the real `mbfa` crate -- the same
//! code path the CLI uses, not a reimplementation:
//!
//!   compress(bytes: Uint8Array, maxFolds?: number) -> Uint8Array
//!   decompress(bytes: Uint8Array)                  -> Uint8Array
//!
//! This is the contract `.github/special-template.html` relies on: it imports
//! `../mbfa.js`, calls `mod.default()` to load `mbfa_bg.wasm`, then uses the
//! `compress` and `decompress` exports. Keep those names and shapes stable.
//!
//! ## Build
//!   wasm-pack build web/mbfa-wasm --target web --release \
//!       --out-dir pkg --out-name mbfa
//! The `--out-name mbfa` is what produces `mbfa.js` / `mbfa_bg.wasm` (the
//! page's expected file names) instead of `mbfa_wasm.js` / `mbfa_wasm_bg.wasm`.
//!
//! ## Why this should work on wasm32-unknown-unknown
//! With default features off, mbfa's core has no threads, no filesystem, no
//! clock and no environment access (checked by grep for `Instant`,
//! `SystemTime`, `std::fs`, `std::thread`, `std::env` and `rayon` in the core
//! modules; rayon exists only behind the `parallel` feature). Diagnostic
//! `println!`s in the library are no-ops on this target. Panics become traps,
//! so `console_error_panic_hook` is installed to log the real message first.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Message of the most recent Rust panic. A wasm trap reaches JavaScript as a bare
    /// "RuntimeError: unreachable"; the page asks for this string (`last_panic`) and shows it,
    /// because a phone browser has no console to read the real message from.
    static LAST_PANIC: RefCell<String> = RefCell::new(String::new());
}

/// Same fold limit the CLI passes (`src/main.rs`: `mbfa::compress(&input, 8)`).
const DEFAULT_MAX_FOLDS: u8 = 8;

/// Runs once when the module is instantiated.
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        LAST_PANIC.with(|p| *p.borrow_mut() = info.to_string());
        console_error_panic_hook::hook(info);
    }));
}

/// The message of the most recent Rust panic in this module instance, or "" if none.
/// (A trap that is not a panic, such as an allocation failure, leaves it empty.)
#[wasm_bindgen]
pub fn last_panic() -> String {
    LAST_PANIC.with(|p| p.borrow().clone())
}

/// Compress `data`. `max_folds` is optional (JS may omit it); the default
/// matches the CLI. Returns the MBFA stream, which `decompress` reverses.
#[wasm_bindgen]
pub fn compress(data: &[u8], max_folds: Option<u8>) -> Result<Vec<u8>, JsError> {
    mbfa::compress(data, max_folds.unwrap_or(DEFAULT_MAX_FOLDS))
        .map_err(|e| JsError::new(&format!("mbfa compress failed: {e}")))
}

/// Reverse `compress`. Malformed input is reported as an error where mbfa
/// reports one; see the audit notes for the cases where it currently panics
/// instead (the panic message reaches `console.error` via the hook).
#[wasm_bindgen]
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, JsError> {
    mbfa::decompress(data)
        .map_err(|e| JsError::new(&format!("mbfa decompress failed: {e}")))
}

#[cfg(test)]
mod tests {
    // Native tests: they exercise the wrapper's happy path through the rlib.
    // (Error paths build a `JsError`, which only works inside a browser.)
    use super::*;

    fn sample() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..4000u32 {
            v.extend_from_slice(format!("record {} of the sample stream\n", i % 37).as_bytes());
        }
        v
    }

    #[test]
    fn roundtrip_default_folds() {
        let data = sample();
        let packed = compress(&data, None).ok().expect("compress");
        assert!(packed.len() < data.len() / 4, "sample should compress well");
        let back = decompress(&packed).ok().expect("decompress");
        assert_eq!(back, data);
    }

    #[test]
    fn roundtrip_explicit_folds_and_tiny_inputs() {
        for folds in [1u8, 2, 8] {
            for len in [0usize, 1, 5, 64] {
                let data = vec![7u8; len];
                let packed = compress(&data, Some(folds)).ok().expect("compress");
                assert_eq!(decompress(&packed).ok().expect("decompress"), data);
            }
        }
    }
}
