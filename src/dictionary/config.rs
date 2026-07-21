// src/dictionary/config.rs
//! Generic YAML/TOML config-file scaffolding dictionary (Kubernetes
//! ConfigMap boilerplate + Cargo-style TOML section/key layout).
//!
//! This is the catch-all for structured config files that aren't Unity,
//! Unreal, or DixScript specifically -- kept intentionally small and
//! generic. Unchanged from the original single-file dictionary.rs; only
//! relocated here. See ../dictionary/mod.rs for the per-format selection
//! this enables.

pub const DICTIONARY: &[u8] = b"apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: \n  namespace: default\n  labels:\n    app: \nspec:\n[package]\nname    = \"\"\nversion = \"\"\nedition = \"2021\"\n\n[services.\nhost            = \"0.0.0.0\"\nport            = \nenabled = true\nmax_connections = \ntimeout_sec     = \n";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 2048, "config dictionary should stay compact, got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_contains_expected_boilerplate() {
        let s = DICTIONARY;
        assert!(s.windows(10).any(|w| w == b"apiVersion"));
        assert!(s.windows(9).any(|w| w == b"[package]"));
        assert!(s.windows(15).any(|w| w == b"max_connections"));
    }
}
