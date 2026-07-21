// src/dictionary/unreal.rs
//! Unreal .uplugin (and by extension .uproject/.ini-adjacent JSON)
//! boilerplate dictionary.
//!
//! Content is the fixed descriptor fields Unreal's own plugin-descriptor
//! writer unconditionally emits (FileVersion/Version/VersionName/Modules
//! block/etc) -- verified via held-out same-format testing, same basis as
//! unity.rs. Unchanged from the original single-file dictionary.rs; only
//! relocated here. See ../dictionary/mod.rs for the per-format selection
//! this enables.

pub const DICTIONARY: &[u8] = b"  \"FileVersion\": 3,\n  \"Version\": 1,\n  \"VersionName\": \"1.0\",\n  \"FriendlyName\": \"\",\n  \"Description\": \"\",\n  \"Category\": \"\",\n  \"CreatedBy\": \"\",\n  \"CreatedByURL\": \"https://\",\n  \"DocsURL\": \"\",\n  \"MarketplaceURL\": \"\",\n  \"SupportURL\": \"\",\n  \"CanContainContent\": true,\n  \"IsBetaVersion\": false,\n  \"IsExperimentalVersion\": false,\n  \"Installed\": false,\n  \"EnabledByDefault\": true,\n  \"Modules\": [\n    {\n      \"Name\": \"\",\n      \"Type\": \"Runtime\",\n      \"LoadingPhase\": \"Default\",\n      \"WhitelistPlatforms\": [\n        \"Win64\",\n        \"Mac\",\n        \"Linux\"\n      ],\n      \"AdditionalDependencies\": [\n        \"Engine\",\n        \"CoreUObject\"\n      ]\n    }\n  ],\n  \"Plugins\": []\n}";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 4096, "unreal dictionary should stay compact, got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_contains_expected_boilerplate() {
        let s = DICTIONARY;
        assert!(s.windows(18).any(|w| w == b"CanContainContent\""));
        assert!(s.windows(18).any(|w| w == b"\"EnabledByDefault\""));
        assert!(s.windows(15).any(|w| w == b"\"FriendlyName\":"));
    }
}
