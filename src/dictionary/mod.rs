// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mbfa/dictionary.md, section "mod.rs"
// ============================================================================
//! Per-format static dictionaries for cross-file backref addressing.
//!
//! Each format owns its own dictionary file and its own provenance --
//! see dixscript.rs, dixscript_binary.rs, unity.rs, unreal.rs, config.rs.
//! This keeps per-file match density high (a DixScript source file's
//! dictionary is 100% DixScript syntax, not diluted by other formats'
//! bytes) and lets fold.rs skip the scan entirely for files that plainly
//! aren't any of these formats.
//!
//! The header carries a `dict_flag` byte (lib.rs byte 4) recording which
//! `DictId` a file's fold-1 pass used, written by fold.rs and read back by
//! unfold.rs before any reconstruct() call. This is required because the
//! dictionaries differ in length -- decoding with the wrong one produces
//! wrong bytes rather than failing loudly, so the decoder must be told
//! exactly which one to use rather than inferring it.

pub mod config;
pub mod dixscript;
pub mod dixscript_binary;
pub mod unity;
pub mod unreal;

/// Which (if any) per-format dictionary was used for a file's fold-1 pass.
/// Stored verbatim as the header's `dict_flag` byte (see lib.rs / unfold.rs).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictId {
    None            = 0,
    DixScript       = 1,
    Unity           = 2,
    Unreal          = 3,
    Config          = 4,
    DixScriptBinary = 5,
}

impl DictId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(DictId::None),
            1 => Some(DictId::DixScript),
            2 => Some(DictId::Unity),
            3 => Some(DictId::Unreal),
            4 => Some(DictId::Config),
            5 => Some(DictId::DixScriptBinary),
            _ => None,
        }
    }

    /// The dictionary bytes this id refers to. `DictId::None` is `&[]`,
    /// which makes every dict-aware call site (scan_with_dict,
    /// min_offset_bits_for_dict, decoder::reconstruct) behave exactly like
    /// the no-dictionary case for free -- no separate "dictionary present?"
    /// branch needed anywhere downstream.
    pub fn bytes(self) -> &'static [u8] {
        match self {
            DictId::None            => &[],
            DictId::DixScript       => dixscript::DICTIONARY,
            DictId::Unity           => unity::DICTIONARY,
            DictId::Unreal          => unreal::DICTIONARY,
            DictId::Config          => config::DICTIONARY,
            DictId::DixScriptBinary => dixscript_binary::DICTIONARY,
        }
    }
}

/// All real (non-`None`) dictionaries, in the order `candidates_for`
/// prioritizes when nothing more specific matches.
pub const ALL: [DictId; 5] = [
    DictId::DixScript, DictId::DixScriptBinary, DictId::Unity, DictId::Unreal, DictId::Config,
];

/// Cheap format sniff so fold.rs doesn't have to scan every dictionary
/// against every file. This is a *shortlist*, not a verdict -- fold.rs
/// still measures actual bit cost for each candidate returned here and
/// only keeps whichever one genuinely wins (same "measure, don't guess"
/// pattern the rest of the pipeline uses). Getting this wrong costs a
/// wasted scan or a missed win, never correctness.
///
/// Detection signals are each tied to something structurally guaranteed
/// for that format rather than guessed:
///   - DixScript *source*: the first 256 bytes contain one of the seven
///     real section markers (@CONFIG(, @IMPORTS(, @DLM(, @ENUMS(,
///     @QUICKFUNCS(, @DATA(, @SECURITY() -- these are the only seven
///     top-level productions others/midx.ebnf allows, so this can't
///     false-positive on anything that isn't DixScript.
///   - DixScript *compiled binary*: starts with the `XIDM` MDIX magic.
///     Deliberately a separate DictId from source (see dixscript_binary.rs)
///     -- a file can only be one or the other, never both, so these two
///     conditions are mutually exclusive by construction (source text
///     can't also start with the 4-byte binary magic).
///   - Unity: `%TAG !u!`, on every Unity serialized-YAML file's header line.
///   - Unreal: `"FriendlyName"` or `"EnabledByDefault"`, both
///     unconditionally emitted by Unreal's plugin-descriptor writer.
///   - Config: `apiVersion:` (Kubernetes), `[package]` (Cargo-style TOML),
///     or bare `%YAML` (anything else YAML-ish) -- the generic fallback.
pub fn candidates_for(input: &[u8]) -> Vec<DictId> {
    let mut out = Vec::with_capacity(2);
    let head = &input[..input.len().min(256)];

    if input.starts_with(b"XIDM") {
        out.push(DictId::DixScriptBinary);
    } else if contains(head, b"@CONFIG(")
        || contains(head, b"@IMPORTS(")
        || contains(head, b"@DLM(")
        || contains(head, b"@ENUMS(")
        || contains(head, b"@QUICKFUNCS(")
        || contains(head, b"@DATA(")
        || contains(head, b"@SECURITY(")
    {
        out.push(DictId::DixScript);
    }

    let looks_unity = contains(head, b"%TAG !u!");
    if looks_unity {
        out.push(DictId::Unity);
    }

    if contains(input, b"\"FriendlyName\"") || contains(input, b"\"EnabledByDefault\"") {
        out.push(DictId::Unreal);
    }

    // Bare `%YAML` (no `%TAG !u!` alongside it) is generic YAML, not Unity --
    // every Unity file has BOTH markers, so checking this only when Unity
    // didn't already match keeps the two mutually exclusive.
    if contains(head, b"apiVersion:") || contains(head, b"[package]")
        || (contains(head, b"%YAML") && !looks_unity)
    {
        out.push(DictId::Config);
    }

    // No structural signal at all: still worth trying the generic config
    // dictionary rather than skipping outright -- a false negative here
    // only costs a missed ratio win, never correctness, and config.rs is
    // the smallest non-DixScript dictionary so it's the cheapest blind guess.
    if out.is_empty() {
        out.push(DictId::Config);
    }
    out
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.len() > hay.len() { return false; }
    hay.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_id_roundtrips_through_u8() {
        for id in [DictId::None, DictId::DixScript, DictId::Unity, DictId::Unreal,
                   DictId::Config, DictId::DixScriptBinary] {
            assert_eq!(DictId::from_u8(id as u8), Some(id));
        }
        assert_eq!(DictId::from_u8(6), None);
    }

    #[test]
    fn none_has_empty_bytes() {
        assert_eq!(DictId::None.bytes(), &[] as &[u8]);
    }

    #[test]
    fn candidates_detect_dixscript_source() {
        let input = b"@CONFIG(\n  version -> \"1.0.0\"\n)\n";
        assert_eq!(candidates_for(input), vec![DictId::DixScript]);
    }

    #[test]
    fn candidates_detect_dixscript_binary_by_magic() {
        let input = [b"XIDM\x01\x00\x00".as_ref(), &[0u8; 20]].concat();
        assert_eq!(candidates_for(&input), vec![DictId::DixScriptBinary]);
    }

    #[test]
    fn candidates_detect_unity_yaml() {
        let input = b"%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &1\n";
        assert_eq!(candidates_for(input), vec![DictId::Unity]);
    }

    #[test]
    fn candidates_detect_unreal_uplugin() {
        let input = br#"{"FileVersion": 3, "FriendlyName": "MyPlugin"}"#;
        assert_eq!(candidates_for(input), vec![DictId::Unreal]);
    }

    #[test]
    fn candidates_fall_back_to_config_when_nothing_matches() {
        let input = b"some arbitrary structured text with no known markers";
        assert_eq!(candidates_for(input), vec![DictId::Config]);
    }
}
