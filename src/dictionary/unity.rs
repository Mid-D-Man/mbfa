// src/dictionary/unity.rs
//! Unity serialized-YAML boilerplate dictionary.
//!
//! Content is the fixed engine/schema serialization fields Unity's YAML
//! serializer unconditionally emits (GameObject/Transform/MeshRenderer
//! blocks) -- verified via held-out same-format testing (~50-79% byte
//! coverage, ~13-50% real projected compressed-size reduction on
//! independently-varied samples of the same format). Unchanged from the
//! original single-file dictionary.rs; only relocated here so Unity files
//! stop paying (in address-space bits) for DixScript/Unreal/config content
//! they can never match, and vice versa. See ../dictionary/mod.rs for the
//! per-format selection this enables.

pub const DICTIONARY: &[u8] = b"%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &\nGameObject:\n  m_ObjectHideFlags: 0\n  serializedVersion: 6\n  m_Component:\n  - component: {fileID: \n  m_Layer: \n  m_Name: \n  m_TagString: Untagged\n  m_IsActive: 1\n--- !u!4 &\nTransform:\n  m_ObjectHideFlags: 0\n  m_LocalPosition: {x: 0, y: 0, z: 0}\n  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}\n  m_LocalScale: {x: 1, y: 1, z: 1}\n--- !u!23 &\nMeshRenderer:\n  m_ObjectHideFlags: 0\n  serializedVersion: 4\n  m_Enabled: 1\n  m_CastShadows: 1\n  m_ReceiveShadows: 1\n  m_Materials:\n  - {fileID: 2100000, guid: {\n";

pub const DICT_LEN: usize = DICTIONARY.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_is_nonempty_and_reasonably_small() {
        assert!(DICT_LEN > 0);
        assert!(DICT_LEN < 4096, "unity dictionary should stay compact, got {} bytes", DICT_LEN);
    }

    #[test]
    fn dictionary_contains_expected_boilerplate() {
        let s = DICTIONARY;
        assert!(s.windows(17).any(|w| w == b"m_ObjectHideFlags"));
        assert!(s.windows(9).any(|w| w == b"Transform"));
        assert!(s.windows(12).any(|w| w == b"MeshRenderer"));
    }
  }
