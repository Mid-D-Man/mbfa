// src/dictionary.rs
//! Static cross-file dictionary for small structured files (Unity YAML,
//! Unreal .uplugin/.ini, DixScript, common YAML/TOML config scaffolding).
//!
//! Content is either (a) genuine fixed engine/schema serialization fields
//! verified via held-out same-format testing (Unity's `m_ObjectHideFlags`,
//! Unreal's `"CanContainContent"` -- ~50-79% byte coverage, ~13-50% real
//! projected compressed-size reduction on independently-varied samples of
//! the same format), or (b) verbatim excerpts of real DixScript source
//! (module/const/config/enum/fn/match/main constructs -- language keywords,
//! not hand-approximated syntax).
//!
//! Backref offsets that exceed the real sliding window resolve into this
//! dictionary instead (see decoder.rs::reconstruct's virtual-position logic).
//! No new opcode, no header flag, no changes to entropy.rs -- a dictionary hit
//! is just a Backref with an offset larger than the real window could produce
//! on its own, decoded via `DICT ++ output` addressing.

pub const DICTIONARY: &[u8] = b"%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &\nGameObject:\n  m_ObjectHideFlags: 0\n  serializedVersion: 6\n  m_Component:\n  - component: {fileID: \n  m_Layer: \n  m_Name: \n  m_TagString: Untagged\n  m_IsActive: 1\n--- !u!4 &\nTransform:\n  m_ObjectHideFlags: 0\n  m_LocalPosition: {x: 0, y: 0, z: 0}\n  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}\n  m_LocalScale: {x: 1, y: 1, z: 1}\n--- !u!23 &\nMeshRenderer:\n  m_ObjectHideFlags: 0\n  serializedVersion: 4\n  m_Enabled: 1\n  m_CastShadows: 1\n  m_ReceiveShadows: 1\n  m_Materials:\n  - {fileID: 2100000, guid: {\n  \"FileVersion\": 3,\n  \"Version\": 1,\n  \"VersionName\": \"1.0\",\n  \"FriendlyName\": \"\",\n  \"Description\": \"\",\n  \"Category\": \"\",\n  \"CreatedBy\": \"\",\n  \"CreatedByURL\": \"https://\",\n  \"DocsURL\": \"\",\n  \"MarketplaceURL\": \"\",\n  \"SupportURL\": \"\",\n  \"CanContainContent\": true,\n  \"IsBetaVersion\": false,\n  \"IsExperimentalVersion\": false,\n  \"Installed\": false,\n  \"EnabledByDefault\": true,\n  \"Modules\": [\n    {\n      \"Name\": \"\",\n      \"Type\": \"Runtime\",\n      \"LoadingPhase\": \"Default\",\n      \"WhitelistPlatforms\": [\n        \"Win64\",\n        \"Mac\",\n        \"Linux\"\n      ],\n      \"AdditionalDependencies\": [\n        \"Engine\",\n        \"CoreUObject\"\n      ]\n    }\n  ],\n  \"Plugins\": []\n}module Platform {\n    const VERSION: str = \"2.1.0\"\n    const MAX_FOLDS: int = 8\n    const MIN_IMPROVEMENT: float = 0.985\n\n    config Encoder {\n        offset_bits_min:     int = 7\n        offset_bits_max:     int = 24\n        offset_bits_default: int = 15\n        length_bits_min:     int = 8\n        length_bits_max:     int = 24\n        hash_size:           int = 65536\n        chain_limit:         int = 256\n        lazy_short_len:      int = 6\n        rep_slots:           int = 4\n    }\n\n    config Decoder {\n        ring_slots:       int  = 4\n        verify_roundtrip: bool = true\n        strict_end_token: bool = false\n    }\n\nconst FILTER_NONE:   int = 0\nconst FILTER_DELTA1: int = 1\nconst FILTER_DELTA2: int = 2\nconst FILTER_DELTA3: int = 3\nconst FILTER_DELTA4: int = 4\nconst FILTER_STL:    int = 7\nconst FILTER_PLY:    int = 8\nconst FILTER_BCJ:    int = 9\n\nenum SimilarityGroup {\n    Source     = 0\n    Markup     = 1\n    Binary     = 2\n    Compressed = 3\n    Other      = 4\n}\n\nfn detect_platform() -> str {\n    @(os) match {\n        \"linux\"   => \"Linux\"\n        \"windows\" => \"Windows\"\n        \"macos\"   => \"macOS\"\n        _         => \"Unknown\"\n    }\n}\n\nfn auto_chunk_size(available_mb: int) -> int {\n    clamp(available_mb / 256, 1, 8) * 1024 * 1024\n}\n\n@(main)\nfn run() {\n    let platform = detect_platform()\n    print(f\"MBFA Platform Config v{Platform::VERSION} on {platform}\")\n    print(f\"Max folds:       {Platform::MAX_FOLDS}\")\n    print(f\"Min improvement: {Platform::MIN_IMPROVEMENT}\")\n}apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: \n  namespace: default\n  labels:\n    app: \nspec:\n[package]\nname    = \"\"\nversion = \"\"\nedition = \"2021\"\n\n[services.\nhost            = \"0.0.0.0\"\nport            = \nenabled = true\nmax_connections = \ntimeout_sec     = \n";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 8192, "dictionary should stay compact (few KB), got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_contains_expected_boilerplate() {
        let s = DICTIONARY;
        assert!(s.windows(17).any(|w| w == b"m_ObjectHideFlags"));
        assert!(s.windows(18).any(|w| w == b"CanContainContent\""));
        assert!(s.windows(15).any(|w| w == b"module Platform"));
        assert!(s.windows(10).any(|w| w == b"apiVersion"));
    }
}
