// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/dictionary.md, section "dixscript_binary.rs"
// ============================================================================
//! Compiled DixScript binary (.mdix.enc-shaped) dictionary. Split out from
//! dixscript.rs (source text) because compiled binaries and source text
//! share almost no bytes.
//!
//! Content is derived directly from the real writer source
//! (BinarySerialization/SectionWriters/{config,security}_section_writer.rs,
//! value_encoder.rs), not mined: @CONFIG/@SECURITY sections are dominated
//! by strings drawn from the grammar's fixed vocabulary, each written as
//! `[len: i32 LE][UTF-8 bytes]` for keys or `[0x05 tag][len: i32 LE][UTF-8
//! bytes]` for enum-like string values. `@DATA` is not covered -- its
//! entries are user-chosen identifiers, not grammar vocabulary.
//!
//! Every compiled binary ends in a 32-byte SHA-256 checksum with no
//! length-prefix or marker before it. Those bytes are cryptographically
//! indistinguishable from random and unreachable by any dictionary or
//! compressor -- a real, unavoidable floor on the achievable ratio for
//! small files in this category.

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
        // BinarySerialization/binary_header.rs source constants.
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
