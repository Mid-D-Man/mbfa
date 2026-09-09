// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/dictionary.md, section "dixscript.rs"
// ============================================================================
//! DixScript source-text dictionary.
//!
//! Content: section-marker keywords and grammar vocabulary copied from
//! others/midx.ebnf's terminals (cross-checked against real occurrence),
//! plus longest-common-block mining across real .mdix files from
//! Mid-D-Man/DixScript-Rust, each entry verified to recur in multiple
//! independent files.
//!
//! Compiled-binary content lives in dixscript_binary.rs, not here --
//! dictionary/mod.rs's candidates_for() tells the two apart by format
//! (source: `@CONFIG(`-style text signals; binary: `XIDM` magic).
//!
//! Backref offsets beyond the real sliding window resolve into whichever
//! dictionary dictionary/mod.rs selected (see DictId/candidates_for);
//! see decoder.rs::reconstruct for how a hit is resolved.

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
        // Regression guard: earlier dictionary content used
        // `module Platform { ... }` pseudo-syntax that was never valid
        // DixScript. Make sure it doesn't creep back in.
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
