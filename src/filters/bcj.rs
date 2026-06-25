// src/filters/bcj.rs
//! x86 BCJ (Branch-Call-Jump) filter for PE/COFF executables (flag 9).
//! ARM 32-bit BCJ filter (flag 11).
//! ARM64 BCJ filter (flag 12).
//! PowerPC BCJ filter (flag 13) — big-endian.
//! SPARC BCJ filter (flag 14) — big-endian.
//! RISC-V BCJ filter (flag 15) — variable-width instructions.
//!
//! ELF, Mach-O, and Unix a.out detection helpers are exported for
//! use by filters/mod.rs::detect_filter.

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
pub fn detect_elf(
    data:       &[u8],
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
        0x03 | 0x3E        => Some(flag_x86),
        0x28               => Some(flag_arm),
        0xB7               => Some(flag_arm64),
        0x14 | 0x15        => Some(flag_ppc),
        0x02 | 0x12 | 0x2B => Some(flag_sparc),
        0xF3               => Some(flag_riscv),
        _                  => None,
    }
}

// ── Mach-O detection ──────────────────────────────────────────────────────────

/// Returns `Some(filter_flag)` when `data` is a thin (non-fat) Mach-O binary
/// whose architecture has a supported BCJ filter, else `None`.
pub fn detect_macho(
    data:       &[u8],
    flag_x86:   u8,
    flag_arm:   u8,
    flag_arm64: u8,
    flag_ppc:   u8,
) -> Option<u8> {
    if data.len() < 8 { return None; }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let is_be = match magic {
        0xFEEDFACE | 0xFEEDFACF => false,
        0xCEFAEDFE | 0xCFFAEDFE => true,
        _ => return None,
    };
    let cpu_type = if is_be {
        u32::from_be_bytes([data[4], data[5], data[6], data[7]])
    } else {
        u32::from_le_bytes([data[4], data[5], data[6], data[7]])
    };
    match cpu_type {
        0x0000_000C => Some(flag_arm),
        0x0100_000C => Some(flag_arm64),
        0x0000_0007 | 0x0100_0007 => Some(flag_x86),
        0x0000_0012 | 0x0100_0012 => Some(flag_ppc),
        _ => None,
    }
}

// ── Unix a.out detection ──────────────────────────────────────────────────────

/// Returns `Some(filter_flag)` when `data` looks like a Unix a.out executable
/// for an architecture with a supported BCJ filter.
///
/// Targets the SunOS 4.x exec header layout used by SPARC/68k workstations
/// of the early 1990s (the era of the Canterbury benchmark corpus):
///
///   byte 0:    flags  (bit 7 = dynamic; bits 6:0 = toolversion)
///   byte 1:    machtype
///                1 = Sun 68000    2 = Sun 68020    3 = SPARC
///                4 = Sun 386i     5 = 68030        6 = 68040
///                7 = SPARC V8+    8 = SPARC 64-bit
///   bytes 2-3: magic (big-endian)
///                0x0107 = OMAGIC (writable text)
///                0x0108 = NMAGIC (read-only text)
///                0x010B = ZMAGIC (demand-paged)
///
/// Guards:
///   • byte 0 must be < 0x80 — the flags field is 7-bit toolversion + 1-bit
///     dynamic; real executables have this well under 0x80.
///   • machtype must be 1–9 — covers all known SunOS machine types.
///   • Both guards together make accidental matches on binary/text data
///     negligibly unlikely.
///
/// The Canterbury corpus `sum` file is a SunOS 4.1.3 SPARC a.out (ZMAGIC)
/// binary.  It is NOT ELF (Solaris 2.x introduced ELF; SunOS 4.x uses a.out),
/// which is why detect_elf correctly returns None for it.
pub fn detect_aout(data: &[u8], flag_x86: u8, flag_sparc: u8) -> Option<u8> {
    if data.len() < 8 { return None; }

    let flags    = data[0];
    let machtype = data[1];
    let magic    = u16::from_be_bytes([data[2], data[3]]);

    if flags < 0x80
        && machtype >= 1 && machtype <= 9
        && matches!(magic, 0x0107 | 0x0108 | 0x010B)
    {
        return match machtype {
            4 => Some(flag_x86),   // Sun 386i — x86 BCJ
            _ => Some(flag_sparc), // SPARC (3), 68k (1/2/5/6), SPARC64 (7/8)
        };
    }

    None
}

// ── BCJ x86 tables ────────────────────────────────────────────────────────────

static MASK_TO_ALLOWED: [bool; 8] = [true, true, true, false, true, false, false, false];
static MASK_TO_BIT:     [usize; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

#[inline(always)]
fn is_near_operand_high_byte(b: u8) -> bool { b == 0x00 || b == 0xFF }

// ── BCJ x86 encode ────────────────────────────────────────────────────────────

pub fn bcj_x86_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        if data[i] & 0xFE != 0xE8 { i += 1; continue; }

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

pub fn bcj_x86_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = data.len();
    if n < 5 { return out; }

    let size      = n - 4;
    let mut i:         usize = 0;
    let mut prev_pos:  usize = usize::MAX;
    let mut prev_mask: usize = 0;

    while i < size {
        if data[i] & 0xFE != 0xE8 { i += 1; continue; }

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

pub fn bcj_arm_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if out[i + 3] == 0xEB {
            let raw = (out[i] as u32)
                | ((out[i + 1] as u32) << 8)
                | ((out[i + 2] as u32) << 16);
            let offset_words  = ((raw << 8) as i32) >> 8;
            let current_word  = (i as i32) / 4;
            let abs = ((current_word + 2 + offset_words) as u32) & 0x00FF_FFFF;
            out[i]     = abs as u8;
            out[i + 1] = (abs >> 8) as u8;
            out[i + 2] = (abs >> 16) as u8;
        }
        i += 4;
    }
    out
}

pub fn bcj_arm_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if out[i + 3] == 0xEB {
            let abs_raw      = (out[i] as u32)
                | ((out[i + 1] as u32) << 8)
                | ((out[i + 2] as u32) << 16);
            let abs_words    = ((abs_raw << 8) as i32) >> 8;
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

pub fn bcj_arm64_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        let top6 = instr >> 26;
        if top6 == 0x05 || top6 == 0x25 {
            let imm26        = ((instr & 0x3FF_FFFF) as i32) << 6 >> 6;
            let current      = (i / 4) as i32;
            let new_instr    = (instr & 0xFC00_0000) | ((current + imm26) as u32 & 0x3FF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

pub fn bcj_arm64_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        let top6 = instr >> 26;
        if top6 == 0x05 || top6 == 0x25 {
            let abs_raw   = ((instr & 0x3FF_FFFF) as i32) << 6 >> 6;
            let current   = (i / 4) as i32;
            let new_instr = (instr & 0xFC00_0000) | ((abs_raw - current) as u32 & 0x3FF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ PowerPC (flag 13) ─────────────────────────────────────────────────────

pub fn bcj_ppc_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 26) == 18 && (instr & 0x2) == 0 {
            let raw       = (instr & 0x03FF_FFFC) as i32;
            let offset    = (raw << 6) >> 6;
            let abs_addr  = (i as i32) + offset;
            let new_instr = (instr & 0xFC00_0003) | (abs_addr as u32 & 0x03FF_FFFC);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

pub fn bcj_ppc_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 26) == 18 && (instr & 0x2) == 0 {
            let raw       = (instr & 0x03FF_FFFC) as i32;
            let abs_addr  = (raw << 6) >> 6;
            let rel       = abs_addr - (i as i32);
            let new_instr = (instr & 0xFC00_0003) | (rel as u32 & 0x03FF_FFFC);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ SPARC (flag 14) ───────────────────────────────────────────────────────

pub fn bcj_sparc_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 30) == 1 {
            let disp30    = ((instr & 0x3FFF_FFFF) as i32) << 2 >> 2;
            let cur_word  = (i / 4) as i32;
            let new_instr = (instr & 0xC000_0000) | ((cur_word + disp30) as u32 & 0x3FFF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

pub fn bcj_sparc_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let instr = u32::from_be_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr >> 30) == 1 {
            let abs_raw   = ((instr & 0x3FFF_FFFF) as i32) << 2 >> 2;
            let cur_word  = (i / 4) as i32;
            let new_instr = (instr & 0xC000_0000) | ((abs_raw - cur_word) as u32 & 0x3FFF_FFFF);
            out[i..i+4].copy_from_slice(&new_instr.to_be_bytes());
        }
        i += 4;
    }
    out
}

// ── BCJ RISC-V (flag 15) ──────────────────────────────────────────────────────

#[inline]
fn jal_decode_imm(instr: u32) -> i32 {
    let imm20    = (instr >> 31) & 1;
    let imm10_1  = (instr >> 21) & 0x3FF;
    let imm11    = (instr >> 20) & 1;
    let imm19_12 = (instr >> 12) & 0xFF;
    let raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    ((raw as i32) << 11) >> 11
}

#[inline]
pub fn jal_encode_imm(instr: u32, value: i32) -> u32 {
    let v = value as u32;
    let new_bits =
          ((v >> 20) & 1) << 31
        | ((v >>  1) & 0x3FF) << 21
        | ((v >> 11) & 1) << 20
        | ((v >> 12) & 0xFF) << 12;
    (instr & 0x0000_0FFF) | new_bits
}

pub fn bcj_riscv_encode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if (out[i] & 0x3) != 0x3 { i += 2; continue; }
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr & 0x7F) == 0x6F {
            let offset    = jal_decode_imm(instr);
            let new_instr = jal_encode_imm(instr, (i as i32) + offset);
            out[i..i+4].copy_from_slice(&new_instr.to_le_bytes());
        }
        i += 4;
    }
    out
}

pub fn bcj_riscv_decode(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let n = out.len();
    let mut i = 0usize;
    while i + 4 <= n {
        if (out[i] & 0x3) != 0x3 { i += 2; continue; }
        let instr = u32::from_le_bytes([out[i], out[i+1], out[i+2], out[i+3]]);
        if (instr & 0x7F) == 0x6F {
            let abs_addr  = jal_decode_imm(instr);
            let new_instr = jal_encode_imm(instr, abs_addr - (i as i32));
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

    // ── x86 tests ─────────────────────────────────────────────────────────────

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
        let mut data = vec![0u8; 512];
        data[3] = 0xEB;
        data[0] = 100;
        let enc = bcj_arm_encode(&data);
        assert_ne!(enc[0], data[0]);
        let dec = bcj_arm_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_arm_roundtrip_backward_branch() {
        let mut data = vec![0u8; 64];
        data[32 + 3] = 0xEB;
        let offset: i32 = -5;
        let raw = (offset as u32) & 0xFFFFFF;
        data[32] = raw as u8;
        data[33] = (raw >> 8) as u8;
        data[34] = (raw >> 16) as u8;
        let enc = bcj_arm_encode(&data);
        let dec = bcj_arm_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_arm_ignores_non_bl_bytes() {
        let mut data = vec![0u8; 16];
        data[3] = 0xEA; data[0] = 42;
        let enc = bcj_arm_encode(&data);
        assert_eq!(enc, data);
    }

    // ── ARM64 tests ───────────────────────────────────────────────────────────

    #[test]
    fn bcj_arm64_roundtrip_b_forward() {
        let mut data = vec![0u8; 64];
        let instr: u32 = 0x14000000 | 10;
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        let dec = bcj_arm64_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_arm64_roundtrip_bl_backward() {
        let mut data = vec![0u8; 64];
        let imm26: u32 = ((-4i32) as u32) & 0x3FF_FFFF;
        let instr: u32 = 0x94000000 | imm26;
        data[16..20].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        let dec = bcj_arm64_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_arm64_ignores_other_instructions() {
        let mut data = vec![0u8; 16];
        let instr: u32 = 0xD2800000;
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_arm64_encode(&data);
        assert_eq!(enc, data);
    }

    // ── PowerPC tests ─────────────────────────────────────────────────────────

    #[test]
    fn bcj_ppc_roundtrip_bl_forward() {
        let mut data = vec![0u8; 64];
        let instr: u32 = (18 << 26) | (40u32 & 0x03FF_FFFC) | 1;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_ppc_encode(&data);
        let dec = bcj_ppc_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_ppc_ignores_absolute_branches() {
        let mut data = vec![0u8; 16];
        let instr: u32 = (18 << 26) | 40 | 2;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_ppc_encode(&data);
        assert_eq!(enc, data);
    }

    // ── SPARC tests ───────────────────────────────────────────────────────────

    #[test]
    fn bcj_sparc_roundtrip_call() {
        let mut data = vec![0u8; 64];
        let instr: u32 = (1 << 30) | 10;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_sparc_encode(&data);
        let dec = bcj_sparc_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_sparc_roundtrip_negative_call() {
        let mut data = vec![0u8; 32];
        let disp: u32 = ((-2i32) as u32) & 0x3FFF_FFFF;
        let instr: u32 = (1 << 30) | disp;
        data[16..20].copy_from_slice(&instr.to_be_bytes());
        let enc = bcj_sparc_encode(&data);
        let dec = bcj_sparc_decode(&enc);
        assert_eq!(dec, data);
    }

    // ── RISC-V tests ──────────────────────────────────────────────────────────

    #[test]
    fn bcj_riscv_roundtrip_jal_forward() {
        let mut data = vec![0u8; 32];
        let instr: u32 = jal_encode_imm(0x0000_00EF, 16);
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_riscv_encode(&data);
        let dec = bcj_riscv_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_riscv_roundtrip_jal_backward() {
        let mut data = vec![0u8; 32];
        let instr: u32 = jal_encode_imm(0x0000_006F, -8);
        data[8..12].copy_from_slice(&instr.to_le_bytes());
        let enc = bcj_riscv_encode(&data);
        let dec = bcj_riscv_decode(&enc);
        assert_eq!(dec, data);
    }

    #[test]
    fn bcj_riscv_skips_compressed_instructions() {
        let mut data = vec![0u8; 16];
        data[0] = 0x01; data[1] = 0x00;
        let enc = bcj_riscv_encode(&data);
        assert_eq!(enc[0..2], data[0..2]);
    }

    // ── ELF detection tests ───────────────────────────────────────────────────

    #[test]
    fn detect_elf_arm_returns_arm_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0x28; data[19] = 0x00;
        assert_eq!(detect_elf(&data, 9, 11, 12, 13, 14, 15), Some(11));
    }

    #[test]
    fn detect_elf_arm64_returns_arm64_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0xB7; data[19] = 0x00;
        assert_eq!(detect_elf(&data, 9, 11, 12, 13, 14, 15), Some(12));
    }

    #[test]
    fn detect_elf_x86_returns_bcj_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0x03; data[19] = 0x00;
        assert_eq!(detect_elf(&data, 9, 11, 12, 13, 14, 15), Some(9));
    }

    #[test]
    fn detect_elf_riscv_returns_riscv_flag() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0xF3; data[19] = 0x00;
        assert_eq!(detect_elf(&data, 9, 11, 12, 13, 14, 15), Some(15));
    }

    #[test]
    fn detect_elf_unknown_arch_returns_none() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1;
        data[18] = 0x08; data[19] = 0x00; // EM_MIPS
        assert_eq!(detect_elf(&data, 9, 11, 12, 13, 14, 15), None);
    }

    // ── a.out detection tests ─────────────────────────────────────────────────

    fn make_sunos_aout(machtype: u8, magic: u16) -> Vec<u8> {
        let mut data = vec![0u8; 32];
        data[0] = 0x01; // flags: toolversion=1, not dynamic
        data[1] = machtype;
        data[2] = (magic >> 8) as u8;
        data[3] = magic as u8;
        // a_text size (some non-zero value)
        data[4..8].copy_from_slice(&0x0000_4000u32.to_be_bytes());
        data
    }

    #[test]
    fn detect_aout_sparc_zmagic() {
        // SunOS SPARC ZMAGIC: machtype=3, magic=0x010B
        let data = make_sunos_aout(3, 0x010B);
        assert_eq!(detect_aout(&data, 9, 14), Some(14),
            "SunOS SPARC ZMAGIC should return FILTER_BCJ_SPARC (14)");
    }

    #[test]
    fn detect_aout_sparc_nmagic() {
        let data = make_sunos_aout(3, 0x0108);
        assert_eq!(detect_aout(&data, 9, 14), Some(14));
    }

    #[test]
    fn detect_aout_sparc_omagic() {
        let data = make_sunos_aout(3, 0x0107);
        assert_eq!(detect_aout(&data, 9, 14), Some(14));
    }

    #[test]
    fn detect_aout_68k_returns_sparc_flag() {
        // SunOS 68020 binary: machtype=2, magic=0x010B
        // 68k doesn't have a BCJ filter; we fall back to SPARC flag which
        // applies SPARC BCJ (best available for big-endian RISC-like code).
        let data = make_sunos_aout(2, 0x010B);
        assert_eq!(detect_aout(&data, 9, 14), Some(14));
    }

    #[test]
    fn detect_aout_sun386i_returns_x86_flag() {
        // Sun 386i (x86): machtype=4
        let data = make_sunos_aout(4, 0x010B);
        assert_eq!(detect_aout(&data, 9, 14), Some(9),
            "Sun386i a.out should return x86 BCJ flag");
    }

    #[test]
    fn detect_aout_rejects_high_flags_byte() {
        // byte 0 >= 0x80 → not a valid a.out flags byte
        let mut data = make_sunos_aout(3, 0x010B);
        data[0] = 0xFF;
        assert_eq!(detect_aout(&data, 9, 14), None,
            "High flags byte should not match a.out");
    }

    #[test]
    fn detect_aout_rejects_bad_machtype() {
        // machtype=0 and machtype=10 are outside the valid range
        let mut data = make_sunos_aout(0, 0x010B);
        assert_eq!(detect_aout(&data, 9, 14), None);
        data[1] = 10;
        assert_eq!(detect_aout(&data, 9, 14), None);
    }

    #[test]
    fn detect_aout_rejects_wrong_magic() {
        let mut data = make_sunos_aout(3, 0x0200);
        assert_eq!(detect_aout(&data, 9, 14), None,
            "Non-a.out magic should not match");
    }

    #[test]
    fn detect_aout_too_short_returns_none() {
        let data = vec![0x01u8, 0x03, 0x01, 0x0B]; // only 4 bytes
        assert_eq!(detect_aout(&data, 9, 14), None);
    }

    #[test]
    fn detect_aout_does_not_match_elf() {
        // ELF header: \x7fELF should not match a.out
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[5] = 1; data[18] = 0x03;
        // \x7f >= 0x80? No, 0x7F < 0x80 so the flags guard passes.
        // But byte 2 of ELF = 0x01 (EI_CLASS) and byte 3 = EI_DATA = 1 or 2.
        // 0x0101 and 0x0102 are not valid a.out magic values (0x0107/0x0108/0x010B).
        assert_eq!(detect_aout(&data, 9, 14), None,
            "ELF magic should not match a.out detection");
    }

    #[test]
    fn detect_aout_does_not_match_pe() {
        let mut data = make_minimal_pe();
        // PE starts MZ: byte 0=0x4D, byte 1=0x5A — 0x4D < 0x80 and
        // machtype=0x5A=90 is outside 1-9, so detection fails correctly.
        assert_eq!(detect_aout(&data, 9, 14), None,
            "PE/COFF header should not match a.out detection");
    }

    // ── Mach-O detection tests ────────────────────────────────────────────────

    fn make_macho_le(cpu_type: u32) -> Vec<u8> {
        let mut data = vec![0u8; 64];
        data[0] = 0xCF; data[1] = 0xFA; data[2] = 0xED; data[3] = 0xFE;
        data[4..8].copy_from_slice(&cpu_type.to_le_bytes());
        data
    }

    #[test]
    fn detect_macho_arm64_returns_arm64_flag() {
        assert_eq!(detect_macho(&make_macho_le(0x0100_000C), 9, 11, 12, 13), Some(12));
    }

    #[test]
    fn detect_macho_arm_returns_arm_flag() {
        assert_eq!(detect_macho(&make_macho_le(0x0000_000C), 9, 11, 12, 13), Some(11));
    }

    #[test]
    fn detect_macho_x86_64_returns_bcj() {
        assert_eq!(detect_macho(&make_macho_le(0x0100_0007), 9, 11, 12, 13), Some(9));
    }
        }
