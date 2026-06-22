// src/filters/bcj.rs
//! x86 BCJ (Branch-Call-Jump) filter for PE/COFF executables (flag 9).
//! ARM 32-bit BCJ filter (flag 11).
//! ARM64 BCJ filter (flag 12).
//! PowerPC BCJ filter (flag 13) — big-endian.
//! SPARC BCJ filter (flag 14) — big-endian.
//! RISC-V BCJ filter (flag 15) — variable-width instructions.
//!
//! All filters convert PC-relative branch offsets to absolute addresses during
//! compression, and reverse the transform during decompression. Absolute
//! addresses from nearby branches cluster near common values regardless of
//! load address, maximising LZ dictionary hit frequency.
//!
//! ELF and Mach-O binary detection helpers are also exported for use by
//! filters/mod.rs::detect_filter.

// ── PE/COFF detection ─────────────────────────────────────────────────────────

/// Returns true when `data` starts with a valid MZ / PE header.
pub fn detect_pe_coff(data: &[u8]) -> bool {
    if data.len() < 0x40 { return false; }
    if data[0] != b'M' || data[1] != b'Z' { return false; }
    let pe_offset =
        u32::from_le_bytes([data[0x3C], data[0x3D], data[0x3E], data[0x3F]]) as usize;
    if pe_offset.saturating_add(4) > data.len() { return false; }
    data[pe_offset..pe_offset + 4] == *b"PE\x00\x00"
}

// ── ELF detection ─────────────────────────────────────────────────────────────

/// Returns `Some(filter_flag)` when `data` is an ELF binary whose architecture
/// has a supported BCJ filter, else `None`.
///
/// Caller passes the filter flag constants to avoid a circular dependency
/// (those constants live in filters/mod.rs).
pub fn detect_elf(data: &[u8],
    flag_x86:   u8,
    flag_arm:   u8,
    flag_arm64: u8,
    flag_ppc:   u8,
    flag_sparc: u8,
    flag_riscv: u8,
) -> Option<u8> {
    if data.len() < 20 { return None; }
    if data[0..4] != *b"\x7fELF" { return None; }
    // EI_DATA at offset 5: 1 = little-endian, 2 = big-endian
    let is_be = data[5] == 2;
    // e_machine at offset 18, 16-bit field encoded per EI_DATA
    let e_machine = if is_be {
        u16::from_be_bytes([data[18], data[19]])
    } else {
        u16::from_le_bytes([data[18], data[19]])
    };
    match e_machine {
        0x03 | 0x3E => Some(flag_x86),    // EM_386, EM_X86_64
        0x28        => Some(flag_arm),     // EM_ARM (32-bit)
        0xB7        => Some(flag_arm64),   // EM_AARCH64
        0x14 | 0x15 => Some(flag_ppc),    // EM_PPC, EM_PPC64
        0x02 | 0x12 | 0x2B => Some(flag_sparc), // EM_SPARC, EM_SPARC32PLUS, EM_SPARCV9
        0xF3        => Some(flag_riscv),   // EM_RISCV
        _           => None,
    }
}

/// Returns `Some(filter_flag)` when `data` is a thin (non-fat) Mach-O binary
/// whose architecture has a supported BCJ filter, else `None`.
pub fn detect_macho(data: &[u8],
    flag_x86:   u8,
    flag_arm:   u8,
    flag_arm64: u8,
    flag_ppc:   u8,
) -> Option<u8> {
    if data.len() < 8 { return None; }
    // Read magic as LE u32 — lets us distinguish all four Mach-O variants by
    // their on-disk byte patterns.
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let is_be = match magic {
        0xFEEDFACE | 0xFEEDFACF => false, // LE 32/64-bit
        0xCEFAEDFE | 0xCFFAEDFE => true,  // BE 32/64-bit
        _ => return None,
    };
    // cpu_type at offset 4
    let cpu_type = if is_be {
        u32::from_be_bytes([data[4], data[5], data[6], data[7]])
    } else {
        u32::from_le_bytes([data[4], data[5], data[6], data[7]])
    };
    match cpu_type {
        0x0000000C => Some(flag_arm),           // CPU_TYPE_ARM (32-bit)
        0x0100000C => Some(flag_arm64),          // CPU_TYPE_ARM64
        0x00000007 | 0x01000007 => Some(flag_x86), // CPU_TYPE_X86, CPU_TYPE_X86_64
        0x00000012 | 0x01000012 => Some(flag_ppc), // CPU_TYPE_POWERPC, CPU_TYPE_POWERPC64
        _ => None,
    }
}

// ── BCJ x86 tables ────────────────────────────────────────────────────────────

/// Whether a prev_mask state may be processing a legitimate CALL/JMP.
static MASK_TO_ALLOWED: [bool; 8] = [true, true, true, false, true, false, false, false];

/// How many bytes back the operand byte being tested is.
static MASK_TO_BIT: [usize; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

#[inline(always)]
fn is_near_operand_high_byte(b: u8) -> bool {
    b == 0x00 || b == 0xFF
}

// ── BCJ x86 encode ────────────────────────────────────────────────────────────

/// Apply BCJ transform: convert near CALL/JMP relative → normalised absolute.
pub fn bcj_x86_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        if data[i] & 0xFE != 0xE8 {
            i += 1;
            continue;
        }

        let dist = i.wrapping_sub(prev_pos);
        if dist <= 3 {
            prev_mask = (prev_mask << dist.wrapping_sub(1)) & 7;
            if prev_mask != 0 {
                let b = data[i + 4 - MASK_TO_BIT[prev_mask]];
                if !MASK_TO_ALLOWED[prev_mask] || is_near_operand_high_byte(b) {
                    prev_pos  = i;
                    prev_mask = (prev_mask << 1) | 1;
                    i        += 1;
                    continue;
                }
            }
        } else {
            prev_mask = 0;
        }
        prev_pos = i;

        if is_near_operand_high_byte(data[i + 4]) {
            let rel = i32::from_le_bytes([data[i+1], data[i+2], data[i+3], data[i+4]]);
            let abs = rel.wrapping_add(i as i32 + 5) as u32;
            let mut norm  = abs & 0x01FF_FFFF;
            norm         |= 0u32.wrapping_sub(norm & 0x0100_0000);
            out[i+1..i+5].copy_from_slice(&norm.to_le_bytes());
            i += 4;
        } else {
            prev_mask = (prev_mask << 1) | 1;
        }

        i += 1;
    }

    out
}

// ── BCJ x86 decode ────────────────────────────────────────────────────────────

/// Reverse BCJ transform: convert normalised absolute → relative offset.
pub fn bcj_x86_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        if data[i] & 0xFE != 0xE8 {
            i += 1;
            continue;
        }

        let dist = i.wrapping_sub(prev_pos);
        if dist <= 3 {
            prev_mask = (prev_mask << dist.wrapping_sub(1)) & 7;
            if prev_mask != 0 {
                let b = data[i + 4 - MASK_TO_BIT[prev_mask]];
                if !MASK_TO_ALLOWED[prev_mask] || is_near_operand_high_byte(b) {
                    prev_pos  = i;
                    prev_mask = (prev_mask << 1) | 1;
                    i        += 1;
                    continue;
                }
            }
        } else {
            prev_mask = 0;
        }
        prev_pos = i;

        if is_near_operand_high_byte(data[i + 4]) {
            let abs = u32::from_le_bytes([data[i+1], data[i+2], data[i+3], data[i+4]]);
            let rel = abs.wrapping_sub(i as u32 + 5) as i32;
            out[i+1..i+5].copy_from_slice(&rel.to_le_bytes());
            i += 4;
        } else {
            prev_mask = (prev_mask << 1) | 1;
        }

        i += 1;
    }

    out
}

// ── BCJ ARM 32-bit (flag 11) ──────────────────────────────────────────────────
//
// ARM 32-bit instructions are always 4 bytes at 4-byte-aligned addresses.
// In little-endian ARM memory layout, byte[3] == 0xEB identifies a BL
// instruction (condition=0xE=always, opcode=0xB=BL). The 24-bit signed
// word offset is stored LE in bytes [0..2].
//
// ARM pipeline adds 8 bytes (2 instructions) to PC at execute time, so:
//   target = instruction_addr + 8 + offset_words * 4
//
// Encode: store (instruction_addr/4 + 2 + offset_words) in the 24-bit field.
// Decode: recover offset_words = stored_abs - instruction_addr/4 - 2.

/// Apply ARM 32-bit BCJ transform.
pub fn bcj_arm_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if out[i + 3] == 0xEB {
            // Read 24-bit LE offset from bytes [0..2]
            let raw = (out[i] as u32)
                | ((out[i + 1] as u32) << 8)
                | ((out[i + 2] as u32) << 16);
            // Sign-extend 24-bit value to i32
            let offset_words = ((raw << 8) as i32) >> 8;
            let current_word = (i as i32) / 4;
            // abs_word = current_word + 2 (pipeline) + offset_words
            let abs = ((current_word + 2 + offset_words) as u32) & 0x00FF_FFFF;
            out[i]     = abs as u8;
            out[i + 1] = (abs >> 8) as u8;
            out[i + 2] = (abs >> 16) as u8;
            // out[i+3] stays 0xEB
        }
        i += 4;
    }
    out
}

/// Reverse ARM 32-bit BCJ transform.
pub fn bcj_arm_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if out[i + 3] == 0xEB {
            let abs_raw = (out[i] as u32)
                | ((out[i + 1] as u32) << 8)
                | ((out[i + 2] as u32) << 16);
            // Sign-extend stored absolute word index
            let abs_words = ((abs_raw << 8) as i32) >> 8;
            let current_word = (i as i32) / 4;
            let offset_words = (abs_words - current_word - 2) as u32;
            out[i]     = offset_words as u8;
            out[i + 1] = (offset_words >> 8) as u8;
            out[i + 2] = (offset_words >> 16) as u8;
        }
        i += 4;
    }
    out
}

// ── BCJ ARM64 (flag 12) ───────────────────────────────────────────────────────
//
// AArch64 unconditional branch instructions (B and BL) have a 26-bit signed
// PC-relative offset in instruction units (bits 25:0). The top 6 bits select
// the opcode: B = 0x05 (0b000101), BL = 0x25 (0b100101).
//
// Encode: replace imm26 with (current_instruction_index + imm26).
// Decode: replace stored absolute index with (stored - current_instruction_index).

/// Apply ARM64 BCJ transform.
pub fn bcj_arm64_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        let top6 = instr >> 26;
        if top6 == 0x05 || top6 == 0x25 {
            // Sign-extend 26-bit immediate
            let imm26 = ((instr & 0x3FF_FFFF) as i32) << 6 >> 6;
            let current_instr = (i / 4) as i32;
            let abs_instr = current_instr + imm26;
            let new_instr = (instr & 0xFC00_0000) | (abs_instr as u32 & 0x3FF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

/// Reverse ARM64 BCJ transform.
pub fn bcj_arm64_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        let top6 = instr >> 26;
        if top6 == 0x05 || top6 == 0x25 {
            let abs_raw = ((instr & 0x3FF_FFFF) as i32) << 6 >> 6;
            let current_instr = (i / 4) as i32;
            let rel = abs_raw - current_instr;
            let new_instr = (instr & 0xFC00_0000) | (rel as u32 & 0x3FF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ PowerPC (flag 13) ─────────────────────────────────────────────────────
//
// PowerPC is big-endian. B/BL instructions have opcode 18 (bits 31:26).
// Bit 1 (AA) = 0 selects relative addressing; bit 0 (LK) = 0/1 for B/BL.
// 24-bit signed byte offset in bits 25:2 (value already in bytes; low 2 bits
// of the instruction are AA=0 and LK).
//
// Encode: replace the 24-bit field with (current_byte_pos + original_offset).
// Decode: recover original_offset = stored_abs - current_byte_pos.

/// Apply PowerPC BCJ transform.
pub fn bcj_ppc_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        // opcode=18, AA=0 (relative), LK=0 or 1
        if (instr >> 26) == 18 && (instr & 0x2) == 0 {
            // 24-bit signed byte offset in bits [25:2]; bits [1:0] are AA,LK
            let raw = (instr & 0x03FF_FFFC) as i32;
            let offset = (raw << 6) >> 6; // sign-extend from bit 25
            let abs_addr = (i as i32) + offset;
            let new_instr = (instr & 0xFC00_0003) | (abs_addr as u32 & 0x03FF_FFFC);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

/// Reverse PowerPC BCJ transform.
pub fn bcj_ppc_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 26) == 18 && (instr & 0x2) == 0 {
            let raw = (instr & 0x03FF_FFFC) as i32;
            let abs_addr = (raw << 6) >> 6;
            let rel = abs_addr - (i as i32);
            let new_instr = (instr & 0xFC00_0003) | (rel as u32 & 0x03FF_FFFC);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ SPARC (flag 14) ───────────────────────────────────────────────────────
//
// SPARC is big-endian. CALL instruction: bits 31:30 = 0b01.
// 30-bit signed word displacement in bits 29:0 (words = 4 bytes each).
// target = PC + disp30 * 4.
//
// Encode: store (current_word_index + disp30) as the 30-bit field.
// Decode: recover disp30 = stored_abs - current_word_index.

/// Apply SPARC BCJ transform.
pub fn bcj_sparc_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 30) == 1 {
            // 30-bit signed displacement in bits 29:0
            let disp30 = ((instr & 0x3FFF_FFFF) as i32) << 2 >> 2; // sign-extend
            let current_word = (i / 4) as i32;
            let abs_word = current_word + disp30;
            let new_instr = (instr & 0xC000_0000) | (abs_word as u32 & 0x3FFF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

/// Reverse SPARC BCJ transform.
pub fn bcj_sparc_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 30) == 1 {
            let abs_raw = ((instr & 0x3FFF_FFFF) as i32) << 2 >> 2;
            let current_word = (i / 4) as i32;
            let rel = abs_raw - current_word;
            let new_instr = (instr & 0xC000_0000) | (rel as u32 & 0x3FFF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ RISC-V (flag 15) ──────────────────────────────────────────────────────
//
// RISC-V has variable-width instructions: 16-bit (bits 1:0 != 11) or 32-bit
// (bits 1:0 == 11). We only transform 32-bit JAL instructions (opcode 0x6F).
//
// JAL immediate encoding (scrambled in instruction bits):
//   instr[31]    = imm[20]    (sign bit)
//   instr[30:21] = imm[10:1]
//   instr[20]    = imm[11]
//   instr[19:12] = imm[19:12]
//   imm[0] is always 0 (2-byte aligned target)
//
// Total 21-bit signed offset (±1 MB range).
//
// Encode: replace the 20-bit immediate payload with abs_addr = current_pos + offset.
// Decode: recover offset = stored_abs - current_pos.

/// Decode the scrambled JAL 21-bit immediate to a signed byte offset.
#[inline]
fn jal_decode_imm(instr: u32) -> i32 {
    let imm20    = (instr >> 31) & 1;
    let imm10_1  = (instr >> 21) & 0x3FF;
    let imm11    = (instr >> 20) & 1;
    let imm19_12 = (instr >> 12) & 0xFF;
    let raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    // Sign-extend from bit 20 (21-bit signed value)
    ((raw as i32) << 11) >> 11
}

/// Re-encode a signed byte offset into the scrambled JAL immediate fields.
#[inline]
fn jal_encode_imm(instr: u32, value: i32) -> u32 {
    let v = value as u32;
    let new_imm20    = (v >> 20) & 1;
    let new_imm10_1  = (v >> 1) & 0x3FF;
    let new_imm11    = (v >> 11) & 1;
    let new_imm19_12 = (v >> 12) & 0xFF;
    let new_bits =
        (new_imm20    << 31)
        | (new_imm10_1  << 21)
        | (new_imm11    << 20)
        | (new_imm19_12 << 12);
    // Preserve rd (bits 11:7) and opcode (bits 6:0); replace immediate bits.
    (instr & 0x0000_0FFF) | new_bits
}

/// Apply RISC-V BCJ transform (JAL instructions only).
pub fn bcj_riscv_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        // 16-bit compressed instruction — skip 2 bytes
        if (out[i] & 0x3) != 0x3 {
            i += 2;
            continue;
        }
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr & 0x7F) == 0x6F {
            let offset = jal_decode_imm(instr);
            let abs_addr = (i as i32) + offset;
            let new_instr = jal_encode_imm(instr, abs_addr);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

/// Reverse RISC-V BCJ transform.
pub fn bcj_riscv_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if (out[i] & 0x3) != 0x3 {
            i += 2;
            continue;
        }
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr & 0x7F) == 0x6F {
            let abs_addr = jal_decode_imm(instr);
            let rel = abs_addr - (i as i32);
            let new_instr = jal_encode_imm(instr, rel);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{detect_filter, FILTER_BCJ};

    // ── x86 tests (existing) ──────────────────────────────────────────────────

    fn make_minimal_pe() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0] = b'M'; data[1] = b'Z';
        data[0x3C] = 0x40;
        data[0x40] = b'P'; data[0x41] = b'E';
        data[0x42] = 0x00; data[0x43] = 0x00;
        data
    }

    #[test]
    fn detect_pe_coff_basic() {
        assert!(detect_pe_coff(&make_minimal_pe()));
    }

    #[test]
    fn detect_pe_coff_rejects_non_pe() {
        assert!(!detect_pe_coff(b"Not a PE file at all"));
    }

    #[test]
    fn detect_pe_coff_rejects_missing_signature() {
        let mut data = make_minimal_pe();
        data[0x40] = 0x00;
        assert!(!detect_pe_coff(&data));
    }

    #[test]
    fn detect_filter_returns_bcj_for_pe() {
        assert_eq!(detect_filter(&make_minimal_pe()), FILTER_BCJ);
    }

    #[test]
    fn bcj_encode_call_near_correct() {
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x10; data[0x62] = 0x00; data[0x63] = 0x00; data[0x64] = 0x00;
        let enc = bcj_x86_encode(&data);
        let abs = i32::from_le_bytes([enc[0x61], enc[0x62], enc[0x63], enc[0x64]]);
        assert_eq!(abs, 0x75);
    }

    #[test]
    fn bcj_far_call_not_transformed() {
        let mut data = vec![0u8; 128];
        data[0x60] = 0xE8;
        data[0x61] = 0x00; data[0x62] = 0x00; data[0x63] = 0x00; data[0x64] = 0x30;
        let enc = bcj_x86_encode(&data);
        assert_eq!(&enc[0x60..0x65], &data[0x60..0x65]);
    }

    #[test]
    fn bcj_roundtrip_near_calls() {
        let mut data = vec![0u8; 256];
        data[0x10] = 0xE8;
        data[0x11] = 0x20; data[0x12] = 0x00; data[0x13] = 0x00; data[0x14] = 0x00;
        data[0x20] = 0xE9;
        let rel: i32 = -0x10;
        data[0x21..0x25].copy_from_slice(&rel.to_le_bytes());
        data[0x40] = 0xE8; data[0x44] = 0x30;
        let enc = bcj_x86_encode(&data);
        let dec = bcj_x86_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_identity_on_no_e8_e9() {
        let data: Vec<u8> = (0u8..=127).collect();
        let enc = bcj_x86_encode(&data);
        assert_eq!(enc, data);
    }

    // ── ARM 32-bit tests ──────────────────────────────────────────────────────

    #[test]
    fn bcj_arm_roundtrip_forward_branch() {
        // BL at position 0, offset_words = 100 (forward branch)
        let mut data = vec![0u8; 512];
        data[3] = 0xEB; // BL opcode byte
        data[0] = 100; data[1] = 0; data[2] = 0; // offset_words = 100
        let enc = bcj_arm_encode(&data);
        assert_ne!(enc[0], data[0], "encode should modify the offset bytes");
        let dec = bcj_arm_decode(&enc);
        assert_eq!(dec, data, "ARM roundtrip failed (forward)");
    }

    #[test]
    fn bcj_arm_roundtrip_backward_branch() {
        // BL at position 32, offset_words = -5 (backward branch)
        // -5 in 24-bit two's complement = 0xFFFFFB → bytes [0xFB, 0xFF, 0xFF]
        let mut data = vec![0u8; 64];
        data[32 + 3] = 0xEB;
        let offset: i32 = -5;
        let raw = (offset as u32) & 0xFFFFFF;
        data[32] = raw as u8;
        data[33] = (raw >> 8) as u8;
        data[34] = (raw >> 16) as u8;
        let enc = bcj_arm_encode(&data);
        let dec = bcj_arm_decode(&enc);
        assert_eq!(dec, data, "ARM roundtrip failed (backward)");
    }

    #[test]
    fn bcj_arm_ignores_non_bl_bytes() {
        // Only byte[3] == 0xEB triggers the filter
        let mut data = vec![0u8; 16];
        data[3] = 0xEA; // B (branch without link) — not filtered
        data[0] = 42;
        let enc = bcj_arm_encode(&data);
        assert_eq!(enc, data, "ARM should ignore non-BL instructions");
    }

    #[test]
    fn bcj_arm_multiple_roundtrip() {
        let mut data = vec![0u8; 64];
        // Two BL instructions at positions 0 and 16
        data[3] = 0xEB; data[0] = 10;
        data[19] = 0xEB; data[16] = 20; data[17] = 0; data[18] = 0;
        let enc = bcj_arm_encode(&data);
        let dec = bcj_arm_decode(&enc);
        assert_eq!(dec, data, "ARM multi-instruction roundtrip failed");
    }

    // ── ARM64 tests ───────────────────────────────────────────────────────────

    #[test]
    fn bcj_arm64_roundtrip_b_forward() {
        // B at position 0, imm26 = 10 (10 instructions = 40 bytes ahead)
        let mut data = vec![0u8; 64];
        let instr: u32 = 0x14000000 | 10; // B with imm26=10
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        let dec = bcj_arm64_decode(&enc);
        assert_eq!(dec, data, "ARM64 B roundtrip failed");
    }

    #[test]
    fn bcj_arm64_roundtrip_bl_backward() {
        // BL at position 16, imm26 = -4 (branch backward 4 instructions)
        let mut data = vec![0u8; 64];
        let imm26: u32 = ((-4i32) as u32) & 0x3FF_FFFF;
        let instr: u32 = 0x94000000 | imm26; // BL with negative offset
        data[16..20].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        let dec = bcj_arm64_decode(&enc);
        assert_eq!(dec, data, "ARM64 BL backward roundtrip failed");
    }

    #[test]
    fn bcj_arm64_ignores_other_instructions() {
        // top6 != 5 and != 37 → unchanged
        let mut data = vec![0u8; 16];
        let instr: u32 = 0xD2800000; // MOV x0, #0 — top6 = 0x34 != 5/37
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        assert_eq!(enc, data, "ARM64 should ignore non-branch instructions");
    }

    // ── PowerPC tests ─────────────────────────────────────────────────────────

    #[test]
    fn bcj_ppc_roundtrip_bl_forward() {
        // BL at position 0: opcode=18, LK=1, AA=0, offset=+40 bytes
        // Instruction: (18 << 26) | (40 & 0x03FFFFFC) | 1 = 0x48000029
        let mut data = vec![0u8; 64];
        let instr: u32 = (18 << 26) | (40u32 & 0x03FF_FFFC) | 1;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_ppc_encode(&data);
        let dec = bcj_ppc_decode(&enc);
        assert_eq!(dec, data, "PPC BL roundtrip failed");
    }

    #[test]
    fn bcj_ppc_ignores_absolute_branches() {
        // AA=1 (absolute) → not filtered
        let mut data = vec![0u8; 16];
        let instr: u32 = (18 << 26) | 40 | 2; // AA=1
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_ppc_encode(&data);
        assert_eq!(enc, data, "PPC should not filter AA=1 branches");
    }

    // ── SPARC tests ───────────────────────────────────────────────────────────

    #[test]
    fn bcj_sparc_roundtrip_call() {
        // CALL at position 0 with disp30 = 10 (10 words = 40 bytes)
        // Instruction: (1 << 30) | 10 = 0x4000000A
        let mut data = vec![0u8; 64];
        let instr: u32 = (1 << 30) | 10;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_sparc_encode(&data);
        let dec = bcj_sparc_decode(&enc);
        assert_eq!(dec, data, "SPARC CALL roundtrip failed");
    }

    #[test]
    fn bcj_sparc_roundtrip_negative_call() {
        // CALL at position 16 (word 4) with disp30 = -2
        let mut data = vec![0u8; 32];
        let disp: u32 = ((-2i32) as u32) & 0x3FFF_FFFF;
        let instr: u32 = (1 << 30) | disp;
        data[16..20].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_sparc_encode(&data);
        let dec = bcj_sparc_decode(&enc);
        assert_eq!(dec, data, "SPARC negative CALL roundtrip failed");
    }

    // ── RISC-V tests ──────────────────────────────────────────────────────────

    #[test]
    fn bcj_riscv_roundtrip_jal_forward() {
        // JAL x1 (ra), +16 bytes — at position 0
        // rd=1 (x1=ra), offset=16 bytes
        // Encoding: imm[20]=0, imm[10:1]=8, imm[11]=0, imm[19:12]=0
        // (offset=16, so imm=0x10: bit10:1=8, rest 0)
        let mut data = vec![0u8; 32];
        let offset = 16i32; // 16 bytes forward
        let instr: u32 = jal_encode_imm(0x0000_00EF, offset); // rd=1 (bits11:7=1), opcode=0x6F
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_riscv_encode(&data);
        let dec = bcj_riscv_decode(&enc);
        assert_eq!(dec, data, "RISC-V JAL roundtrip failed");
    }

    #[test]
    fn bcj_riscv_roundtrip_jal_backward() {
        // JAL x0, -8 bytes at position 8
        let mut data = vec![0u8; 32];
        let offset = -8i32;
        let instr: u32 = jal_encode_imm(0x0000_006F, offset); // rd=0, opcode=0x6F
        data[8..12].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_riscv_encode(&data);
        let dec = bcj_riscv_decode(&enc);
        assert_eq!(dec, data, "RISC-V JAL backward roundtrip failed");
    }

    #[test]
    fn bcj_riscv_skips_compressed_instructions() {
        // First two bytes: 0x01 0x00 = C.NOP (16-bit compressed, bits 1:0 = 01 != 11)
        // Should be skipped, not modified
        let mut data = vec![0u8; 16];
        data[0] = 0x01; data[1] = 0x00; // C.NOP
        let enc = bcj_riscv_encode(&data);
        assert_eq!(enc[0..2], data[0..2], "RISC-V should skip 16-bit instructions");
    }

    #[test]
    fn bcj_riscv_ignores_non_jal_instructions() {
        // ADDI x0, x0, 0 (NOP) — opcode 0x13, not 0x6F
        let mut data = vec![0u8; 16];
        let instr: u32 = 0x0000_0013;
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_riscv_encode(&data);
        assert_eq!(enc, data, "RISC-V should ignore non-JAL instructions");
    }

    // ── ELF detection tests ───────────────────────────────────────────────────

    #[test]
    fn detect_elf_arm_returns_arm_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1; // LE
        data[18] = 0x28; data[19] = 0x00; // EM_ARM = 0x28
        let result = detect_elf(&data, 9, 11, 12, 13, 14, 15);
        assert_eq!(result, Some(11), "ELF ARM should return flag 11");
    }

    #[test]
    fn detect_elf_arm64_returns_arm64_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1; // LE
        data[18] = 0xB7; data[19] = 0x00; // EM_AARCH64 = 0xB7
        let result = detect_elf(&data, 9, 11, 12, 13, 14, 15);
        assert_eq!(result, Some(12), "ELF ARM64 should return flag 12");
    }

    #[test]
    fn detect_elf_x86_returns_bcj_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1; // LE
        data[18] = 0x03; data[19] = 0x00; // EM_386
        let result = detect_elf(&data, 9, 11, 12, 13, 14, 15);
        assert_eq!(result, Some(9), "ELF x86 should return existing BCJ flag 9");
    }

    #[test]
    fn detect_elf_riscv_returns_riscv_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1; // LE
        data[18] = 0xF3; data[19] = 0x00; // EM_RISCV
        let result = detect_elf(&data, 9, 11, 12, 13, 14, 15);
        assert_eq!(result, Some(15), "ELF RISC-V should return flag 15");
    }

    #[test]
    fn detect_elf_unknown_arch_returns_none() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0x08; data[19] = 0x00; // EM_MIPS — no filter
        let result = detect_elf(&data, 9, 11, 12, 13, 14, 15);
        assert_eq!(result, None, "MIPS ELF should return None");
    }

    #[test]
    fn detect_non_elf_returns_none() {
        assert_eq!(
            detect_elf(b"MZ\x00\x00", 9, 11, 12, 13, 14, 15),
            None,
            "PE header should not be detected as ELF"
        );
    }
                            }
