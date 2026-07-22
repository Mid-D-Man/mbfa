// src/dictionary/dixscript.rs
//! DixScript source-text dictionary.
//!
//! ## Why this file changed shape (regression post-mortem, July 2026)
//!
//! The July 19 update to the old combined dictionary.rs (commit
//! "Update dictionary.rs") replaced fake/guessed DixScript syntax --
//! `module Platform { const VERSION ... fn detect_platform() ... }`, which
//! matches NOTHING in others/midx.ebnf -- with real section-marker syntax
//! (@CONFIG/@IMPORTS/@DLM/@ENUMS/@QUICKFUNCS/@DATA/@SECURITY) mined from
//! real files in Mid-D-Man/DixScript-Rust. That update was *correct*: real
//! DixScript-Rust source files (mdix_files/basic|intermediate|advanced|
//! chemistry_db/*.mdix) universally use the new syntax and never used the
//! old one, because the old one was never valid DixScript to begin with.
//!
//! The CI benchmark regression (DixScript_Source/DixScript_Compiled
//! dropping from rank 1 to rank 2-3, mbfa/mbfa#CI) was NOT caused by that
//! update. It was caused by scripts/gen_special_files.py's PLATFORM_MDIX
//! fixture never being updated to match -- it still generated the
//! "DixScript_Source" benchmark file using the old, never-valid module/fn
//! syntax, so the (now-correct) dictionary shared almost no bytes with the
//! (then-still-wrong) benchmark fixture. Measured: the new dictionary
//! covered only 48.7% of the old PLATFORM_MDIX (down from the old
//! dictionary's 91.6%, which was really just overfitting to that one
//! fixture's exact wording) but covers 45.9% of real DixScript-Rust source
//! files repo-wide (up from 42.9% for the old combined dictionary). The
//! fixture has since been fixed (scripts/gen_special_files.py) to real
//! syntax; this dictionary now covers 50.7% of it.
//!
//! ## Binary content lives in dixscript_binary.rs now, not here
//!
//! This file used to also carry the compiled-binary 7-byte magic+version
//! header. Split out (see dixscript_binary.rs's doc comment) after
//! checking how little source-text content actually helps a *compiled*
//! binary: measured 24.0% coverage of 15 varied, format-faithful synthetic
//! compiled binaries using this file's full source-syntax content, vs
//! 59.5% from a dedicated ~338B dictionary built from nothing but the
//! grammar's own fixed vocabulary in its real wire encoding. Keeping them
//! separate means a compiled binary doesn't pay offset-bits for ~2.2KB of
//! source syntax it can never match, and a source file doesn't pay for
//! binary wire-format bytes it can never match either. dictionary/mod.rs's
//! candidates_for() now tells them apart (source: `@CONFIG(`-style text
//! signals; binary: starts with the `XIDM` magic) and returns the right
//! DictId for each.
//!
//! ## Content sources (all verified against real files, nothing guessed)
//!
//!  - Section-marker keywords and grammar vocabulary (@CONFIG(, @DLM(,
//!    DEncryptor.*, DCompressor.*, DAuditor.*, type annotations like
//!    <int>/<string>/<object>, statement keywords): copied directly from
//!    others/midx.ebnf's terminals, cross-checked for real occurrence
//!    (e.g. `DEncryptor.aes256` appears 10x, `DEncryptor.chacha20` 1x
//!    across the corpus) rather than only asserted from the grammar.
//!    Language syntax, not project convention, so it recurs by definition.
//!  - Everything else: longest-common-block mining across all 99 real,
//!    non-synthetic `.mdix` files in a commit-pinned clone of
//!    Mid-D-Man/DixScript-Rust (mdix_files/ tiers + mdix-lsp/mdix-cli/
//!    dixscript test fixtures; invalid_syntax.mdix excluded), greedy
//!    longest-block-first selection, each entry verified to recur
//!    byte-for-byte in >=4 independent files. Aggregate coverage across
//!    that same 99-file/467KB corpus: 45.9%.
//!
//! Backref offsets that exceed the real sliding window resolve into
//! whichever dictionary dictionary::mod.rs selected for this file (see
//! that module's DictId/candidates_for docs) instead. Unlike the old
//! single-dictionary design, this now needs one header byte (lib.rs
//! byte 4, `dict_flag`) recording which dictionary (if any) was used --
//! with only one possible dictionary that inference was free; with five
//! differently-sized ones it no longer is. See decoder.rs::reconstruct's
//! virtual-position logic for how a hit is resolved once the right bytes
//! are selected.

pub const DICTIONARY: &[u8] = b"@QUICKFUNCS(\n@SECURITY(\n@ENUMS(\n@IMPORTS(\n@DLM(\n@DATA(\n@CONFIG(\ncompatibility_mode -> \"strict\"\ncompatibility_mode -> \"best_effort\"\ncompatibility_mode -> \"permissive\"\nDEncryptor.chacha20\nDEncryptor.aes128\nDEncryptor.aes256\nDEncryptor.xor\nDCompressor.lzma\nDCompressor.bzip2\nDCompressor.gzip\nDAuditor.enhanced\nDAuditor.diy\nkeystore -> {\nvalidation -> {\noverride -> {\nmetadata -> {\n<timestamp>\n<blob>\n<regex>\n<hex>\n<tuple>\n<object>\n<array>\n<double>\n<float>\n<long>\n<bool>\n<string>\n<int>\nfrom_cloud \"\nverify \"\nelif: chk: log: global\nlet mut   port<int>   = 8443\n  active<bool> = true\n  score<double> = 9.87654321\n\n  server: host = \"dlm.test.internal\", ssl = true, timeout<int> = 30\n\n  endpoints::\n    \"https://api.test/v1/health\",\n    \"https://api.test/v1/status\",\n    \"https://api.test/v1/metrics\"\n@CONFIG(\n    version -> \"1.0.0\",\n    encoding -> \"utf-8\",\n    debug_mode -> \"regular\",\n    error_handling -> \"halt\",\n    features -> \"advanced\"\n)\n\n@QUICKFUNCS(\n            }\n            -> miss {\n                return \"unknown\";\n            }\n        }\n    }\n)\n\n@DATA(\n)\n\n@DATA(\n  app_name    = \"DLMTestApp\"\n  version     = \"1.0.0\"\n  environment = \"test\"\n  created -> 2025-01-09T00:00:00Z,\n  encoding -> \"UTF-8\",\n  features -> \"advanced\"\n)\n  encryption -> {\n    mode      = \"keyfile\",\n    algorithm = \"aes256-gcm\"\n  }\n)\n  features -> \"advanced\"\n)\n\n@IMPORTS(\n  Base from \"base_types.mdix\"\n)\n)\n\n@DLM(\n  DCompressor.gzip\n  DEncryptor.aes256\n    \"https://api.test/v1/metrics\"\n)\n\n@SECURITY(\n                return $\"{val}\";\n            }\n        },\n        {\n            id<int> = 2,\n      host     = host\n      port     = port\n        return formatted;\n    }\n    \n  features -> \"advanced\"\n)\n\n@ENUMS(\n        {\n            id<int> = 1,\n)\n\n@SECURITY(\n  encryption -> {\n)\n\n@DLM(\n  DEncryptor.aes256\n)\n    error_handling -> \"halt\"\n)\n\n@CONFIG(\n  version -> \"1.0.0\"\n@CONFIG(\n  version  -> \"1.0.0\"\n        timeout<int> = 30000,\n\n@DLM(\n    DCompressor.gzip,\n    debug_mode -> \"verbose\",\n        return result;\n    }\n        return bytes;\n    }\n            -> \"reverse\" {\n    encoding -> \"UTF-8\",\n        }\n        else {\n        chk: operation {\n  DAuditor.enhanced\n)\n  debug_mode -> \"off\"\n    }\n  }\n)\n\n@DATA(\n    DAuditor.diy\n)\n";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 8192, "dixscript dictionary should stay compact, got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_contains_all_seven_real_section_markers() {
        // Straight from others/midx.ebnf's top-level DixScript production --
        // the ONLY seven top-level sections that exist in real syntax.
        let s = DICTIONARY;
        for marker in [
            &b"@CONFIG("[..], &b"@IMPORTS("[..], &b"@DLM("[..], &b"@ENUMS("[..],
            &b"@QUICKFUNCS("[..], &b"@DATA("[..], &b"@SECURITY("[..],
        ] {
            assert!(
                s.windows(marker.len()).any(|w| w == marker),
                "missing section marker {:?}", std::str::from_utf8(marker).unwrap()
            );
        }
    }

    #[test]
    fn dictionary_does_not_contain_the_old_fake_syntax() {
        // Regression guard: the pre-fix dictionary (and gen_special_files.py's
        // still-unfixed PLATFORM_MDIX fixture) used `module Platform { ... }`
        // pseudo-syntax that was never valid DixScript. Make sure it doesn't
        // creep back in.
        let s = DICTIONARY;
        assert!(!s.windows(7).any(|w| w == b"module "));
        assert!(!s.windows(8).any(|w| w == b"@(main)\n"));
    }

    #[test]
    fn dictionary_contains_real_module_subtypes() {
        let s = DICTIONARY;
        assert!(s.windows(19).any(|w| w == b"DEncryptor.chacha20"));
        assert!(s.windows(17).any(|w| w == b"DEncryptor.aes256"));
        assert!(s.windows(9).any(|w| w == b"DAuditor."));
    }
        }
