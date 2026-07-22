// src/dictionary/dixscript_binary.rs
//! Compiled DixScript binary (.mdix.enc-shaped) dictionary -- split out from
//! dixscript.rs (source text) because the two share almost no bytes: a
//! measured check against `dixscript.rs` (source-syntax-oriented, 2218B)
//! against 15 varied, format-faithful synthetic compiled binaries got only
//! 24.0% coverage, while a dedicated ~338B dictionary built from nothing
//! but the real wire encoding of DixScript's own fixed vocabulary got
//! 59.5% -- combining them only reached 64.0%, i.e. dixscript.rs was doing
//! almost none of the work. Splitting means compiled binaries stop paying
//! offset-bits for 2218 bytes of source syntax they can't ever match, and
//! source files stop paying for this file's binary structure.
//!
//! ## Why this is worth this much, byte for byte
//!
//! Checked the real writers directly (BinarySerialization/SectionWriters/
//! {config,security}_section_writer.rs, value_encoder.rs): a compiled
//! binary's @CONFIG and @SECURITY sections are dominated by strings drawn
//! from the grammar's own FIXED vocabulary (others/midx.ebnf's ConfigKey /
//! DebugValue / ErrorHandlingValue / CompatibilityValue / FeatureValue /
//! SecurityBlockKey productions), each written as a plain length-prefixed
//! field: `[len: i32 LE][UTF-8 bytes]` for keys, `[0x05 String tag][len: i32
//! LE][UTF-8 bytes]` for enum-like string values (`encode_string`,
//! value_encoder.rs). That exact wire form is included below for every
//! vocabulary word the grammar allows -- not mined/guessed, derived
//! directly from the writer source, so there's no held-out-validation
//! uncertainty about whether it's "real" the way source-text mining has.
//!
//! `@DATA` is NOT covered here on purpose: DataEntry names (data_section_
//! writer.rs) are user-chosen identifiers, not grammar vocabulary, so
//! there's nothing safe to hardcode there -- a real cross-file @DATA
//! dictionary would need mining against many real compiled binaries the
//! way dixscript.rs's source content was mined against real .mdix files,
//! which needs a corpus of real .mdix.enc output this pass didn't have
//! (couldn't get DixScript-Rust's own crate building in this sandbox
//! either -- old toolchain, same issue as mbfa's rustc≥1.79 deps).
//!
//! ## The hard floor this can't touch
//!
//! Every compiled binary ends in a 32-byte SHA-256 checksum
//! (checksum_validator.rs::append_checksum) with no length-prefix or
//! marker before it -- just the raw hash. That's cryptographically
//! indistinguishable from random and is NOT reachable by any dictionary,
//! or any compressor: on a small file, 32 fully-incompressible bytes is a
//! real, unavoidable floor on the achievable ratio. Worth knowing before
//! judging "legitimately good" against gzip/xz/brotli on this category --
//! they hit the identical floor on the identical bytes.
//!
//! ## Known gap in the CI benchmark this pairs with
//!
//! scripts/gen_special_files.py's gen_binary_dixscript() (the source of
//! the "DixScript_Compiled" benchmark row) has the same disease the source
//! fixture had before this session's fix: its "@CONFIG" is a generic
//! Object blob (tag 0x09) with made-up keys (`max_folds`, `offset_bits_
//! min`, ...) instead of the real ConfigSection wire format (SectionId=1,
//! real ConfigKey vocabulary). Fixed alongside this file -- see that
//! script's updated gen_binary_dixscript().

pub const DICTIONARY: &[u8] = b"XIDM\x01\x00\x00\x07\x00\x00\x00version\x08\x00\x00\x00encoding\x06\x00\x00\x00author\x07\x00\x00\x00created\x08\x00\x00\x00features\n\x00\x00\x00debug_mode\x0e\x00\x00\x00error_handling\x12\x00\x00\x00compatibility_mode\x05\x05\x00\x00\x00basic\x05\x08\x00\x00\x00advanced\x05\x03\x00\x00\x00off\x05\x07\x00\x00\x00regular\x05\x07\x00\x00\x00verbose\x05\x04\x00\x00\x00halt\x05\x08\x00\x00\x00continue\x05\x07\x00\x00\x00recover\x05\x06\x00\x00\x00strict\x05\x0b\x00\x00\x00best_effort\x05\n\x00\x00\x00permissive\x05\x05\x00\x00\x00UTF-8\n\x00\x00\x00encryption\n\x00\x00\x00validation\x08\x00\x00\x00keystore\x08\x00\x00\x00override\x08\x00\x00\x00metadata\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 1024, "dixscript_binary dictionary should stay compact, got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_starts_with_real_mdix_binary_header() {
        // MAGIC_NUMBER=0x4D444958 LE + version 1.0.0, from the real
        // BinarySerialization/binary_header.rs source constants. Moved here
        // from dixscript.rs -- this is binary-only content.
        assert_eq!(&DICTIONARY[0..7], &[0x58, 0x49, 0x44, 0x4D, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn dictionary_contains_real_config_key_wire_encoding() {
        // "version" (7 bytes) as ConfigSectionWriter::write_config_entry
        // writes it: [len:4 LE][UTF-8], no type tag (keys aren't values).
        let s = DICTIONARY;
        let expected: &[u8] = b"\x07\x00\x00\x00version";
        assert!(s.windows(expected.len()).any(|w| w == expected));
    }

    #[test]
    fn dictionary_contains_real_config_value_wire_encoding() {
        // "strict" (CompatibilityValue) as value_encoder.rs::encode_string
        // writes it: [0x05 tag][len:4 LE][UTF-8].
        let s = DICTIONARY;
        let expected: &[u8] = b"\x05\x06\x00\x00\x00strict";
        assert!(s.windows(expected.len()).any(|w| w == expected));
    }

    #[test]
    fn dictionary_contains_real_security_block_key_wire_encoding() {
        // "keystore" (SecurityBlockKey) via SecuritySectionWriter's
        // write_string_field -- same [len:4 LE][UTF-8] convention as keys.
        let s = DICTIONARY;
        let expected: &[u8] = b"\x08\x00\x00\x00keystore";
        assert!(s.windows(expected.len()).any(|w| w == expected));
    }
}
