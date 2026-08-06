// src/filters/mod.rs
//! Pre/post-compression delta filters + byte-plane shuffle + BCJ transforms.
//!
//! Sub-modules:
//!   delta  — stride-delta filters (flags 1–4)
//!   stl    — STL byte-plane filters (flag 7 legacy, flag 10 current)
//!   ply    — PLY byte-plane shuffle + per-vertex-stride delta (flag 8)
//!   bcj    — x86/ARM/ARM64/PPC/SPARC/RISC-V BCJ normalisation (flags 9, 11-15)
//!   probe  — multi-stride entropy probe + WAV/BMP stride detection helpers
//!   cfbf   — CFBF (OLE2, legacy .xls/.doc/.ppt) sector defragmentation (flag 16)
//!
//! Filter flags (header byte 3):
//!    0 = none
//!    1 = delta stride 1
//!    2 = delta stride 2
//!    3 = delta stride 3
//!    4 = delta stride 4
//!    7 = STL: plane-shuffle + stride-12 delta (LEGACY DECODE ONLY)
//!    8 = PLY: plane-shuffle × 4 + per-vertex-stride delta1
//!    9 = x86 BCJ for PE/COFF executables (and x86 ELF/Mach-O/a.out)
//!   10 = STL: field-major plane-split + stride-1 delta (CURRENT)
//!   11 = ARM 32-bit BCJ
//!   12 = ARM64 BCJ
//!   13 = PowerPC BCJ (big-endian)
//!   14 = SPARC BCJ (big-endian)
//!   15 = RISC-V BCJ
//!   16 = CFBF sector defragmentation (legacy .xls/.doc/.ppt)
//!
//! Detection order:
//!    1. Binary STL  — exact size equation               → flag 10
//!    2. WAV/RIFF    — "RIFF....WAVE" magic              → flag 1–4
//!    3. BMP         — "BM" magic                        → flag 1–4
//!    4. Binary PLY  — "ply\n" + binary_little_endian    → flag 8
//!    5. CFBF        — OLE2 signature + FAT/dir round-trip → flag 16
//!    6. DixScript   — magic 0x4D444958 LE               → flag 0 (skip probe)
//!    7. PE/COFF     — "MZ" + PE offset + "PE\0\0"       → flag 9
//!    8. ELF         — "\x7fELF" magic + e_machine       → flag 9/11–15
//!    9. Mach-O      — Mach-O magic + cpu_type           → flag 9/11–13
//!   10. Unix a.out  — SunOS 4.x exec header             → flag 9/14
//!   11. Stride probe — 8 KB entropy, threshold 0.45     → best of 1–4

pub mod delta;
pub mod stl;
pub mod ply;
pub mod bcj;
pub mod probe;
pub mod cfbf;

pub use delta::{delta_encode, delta_decode};
pub use stl::{
    detect_stl,
    shuffle4_stl_delta_encode, shuffle4_stl_delta_decode,
    field_major_stl_delta_encode, field_major_stl_delta_decode,
};
pub use ply::{parse_ply_layout, shuffle4_ply_delta_encode, shuffle4_ply_delta_decode};
pub use bcj::{
    detect_pe_coff, bcj_x86_encode, bcj_x86_decode,
    detect_elf, detect_macho, detect_aout,
    bcj_arm_encode,   bcj_arm_decode,
    bcj_arm64_encode, bcj_arm64_decode,
    bcj_ppc_encode,   bcj_ppc_decode,
    bcj_sparc_encode, bcj_sparc_decode,
    bcj_riscv_encode, bcj_riscv_decode,
};
pub use probe::{
    byte_entropy, probe_best_stride,
    detect_wav_stride, detect_bmp_stride,
    PROBE_MIN_BYTES, PROBE_DELTA_THRESHOLD,
};
pub use cfbf::{detect_cfbf, cfbf_defrag_encode, cfbf_defrag_decode};

// ── Filter flag constants ─────────────────────────────────────────────────────

pub const FILTER_NONE:            u8 = 0;
pub const FILTER_DELTA1:          u8 = 1;
pub const FILTER_DELTA2:          u8 = 2;
pub const FILTER_DELTA3:          u8 = 3;
pub const FILTER_DELTA4:          u8 = 4;
/// Legacy STL filter (plane-shuffle + stride-12). Decode only.
pub const FILTER_SHUFFLE4_DELTA:  u8 = 7;
pub const FILTER_PLY_DELTA:       u8 = 8;
pub const FILTER_BCJ:             u8 = 9;   // x86
pub const FILTER_STL_FIELD_MAJOR: u8 = 10;
pub const FILTER_BCJ_ARM:         u8 = 11;
pub const FILTER_BCJ_ARM64:       u8 = 12;
pub const FILTER_BCJ_PPC:         u8 = 13;
pub const FILTER_BCJ_SPARC:       u8 = 14;
pub const FILTER_BCJ_RISCV:       u8 = 15;
pub const FILTER_CFBF_DEFRAG:     u8 = 16;

/// DixScript binary magic: 0x4D444958 LE = bytes [0x58, 0x49, 0x44, 0x4D].
const DIXSCRIPT_MAGIC_BYTES: [u8; 4] = [0x58, 0x49, 0x44, 0x4D];

// ── Public API ────────────────────────────────────────────────────────────────

pub fn detect_filter(input: &[u8]) -> u8 {
    if input.len() < 12 { return FILTER_NONE; }

    // 1. Binary STL → flag 10
    if input.len() >= 84 && detect_stl(input).is_some() {
        println!("Binary STL detected → FILTER_STL_FIELD_MAJOR (flag 10)");
        return FILTER_STL_FIELD_MAJOR;
    }

    // 2. WAV / RIFF
    if &input[0..4] == b"RIFF" && input.len() >= 12 && &input[8..12] == b"WAVE" {
        return detect_wav_stride(input);
    }

    // 3. BMP
    if &input[0..2] == b"BM" && input.len() >= 30 {
        return detect_bmp_stride(input);
    }

    // 4. Binary PLY
    if input.len() >= 4 && &input[0..4] == b"ply\n" {
        if parse_ply_layout(input).is_some() {
            println!("Binary PLY detected → FILTER_PLY_DELTA (flag 8)");
            return FILTER_PLY_DELTA;
        }
    }

    // 5. CFBF (OLE2) — legacy .xls/.doc/.ppt. Round-trip self-verified inside
    //    detect_cfbf itself (see cfbf.rs's module doc for why); only ever
    //    returns Some if re-encoding+decoding these exact bytes reproduces
    //    them exactly, so a parse edge case degrades to "skip", never wrong.
    if let Some(flag) = detect_cfbf(input) {
        println!("CFBF (OLE2) compound file detected → flag {}", flag);
        return flag;
    }

    // 6. DixScript — LZ+entropy handles it directly, no BCJ
    if input.len() >= 16 && input[0..4] == DIXSCRIPT_MAGIC_BYTES {
        println!("DixScript binary (.mdix compiled) detected — MDIX magic → FILTER_NONE");
        return FILTER_NONE;
    }

    // 7. PE/COFF → flag 9
    if detect_pe_coff(input) {
        println!("PE/COFF binary detected → FILTER_BCJ (flag 9)");
        return FILTER_BCJ;
    }

    // 8. ELF — arch-specific BCJ
    if let Some(flag) = detect_elf(
        input,
        FILTER_BCJ,
        FILTER_BCJ_ARM,
        FILTER_BCJ_ARM64,
        FILTER_BCJ_PPC,
        FILTER_BCJ_SPARC,
        FILTER_BCJ_RISCV,
    ) {
        println!("ELF binary detected (e_machine) → flag {}", flag);
        return flag;
    }

    // 9. Mach-O — arch-specific BCJ
    if let Some(flag) = detect_macho(
        input,
        FILTER_BCJ,
        FILTER_BCJ_ARM,
        FILTER_BCJ_ARM64,
        FILTER_BCJ_PPC,
    ) {
        println!("Mach-O binary detected (cpu_type) → flag {}", flag);
        return flag;
    }

    // 10. Unix a.out — covers SunOS 4.x SPARC/68k/386i executables.
    //    The Canterbury corpus `sum` file is a SunOS 4.1.3 SPARC a.out (ZMAGIC).
    //    It is NOT ELF (SunOS 4.x pre-dates ELF; Solaris 2.x introduced ELF),
    //    so it falls through steps 7-9 and is caught here.
    if let Some(flag) = detect_aout(input, FILTER_BCJ, FILTER_BCJ_SPARC) {
        println!(
            "Unix a.out binary detected (SunOS exec header) → flag {}",
            flag
        );
        return flag;
    }

    // 11. Multi-stride entropy probe (generic numeric streams)
    if input.len() >= PROBE_MIN_BYTES {
        let (best_filter, improvement) = probe_best_stride(input);
        if improvement >= PROBE_DELTA_THRESHOLD {
            println!(
                "Stride probe: FILTER_DELTA{} entropy improvement {:.2} bits/byte \
                 (threshold {:.2}) → applying filter",
                best_filter, improvement, PROBE_DELTA_THRESHOLD
            );
            return best_filter;
        }
        if improvement > 0.1 {
            println!(
                "Stride probe: best FILTER_DELTA{} improvement {:.2} bits/byte \
                 — below threshold {:.2}, no filter applied",
                best_filter, improvement, PROBE_DELTA_THRESHOLD
            );
        }
    }

    FILTER_NONE
}

pub fn apply_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_encode(input, filter as usize)
        }
        FILTER_SHUFFLE4_DELTA  => shuffle4_stl_delta_encode(input),
        FILTER_PLY_DELTA       => shuffle4_ply_delta_encode(input),
        FILTER_BCJ             => bcj_x86_encode(input),
        FILTER_STL_FIELD_MAJOR => field_major_stl_delta_encode(input),
        FILTER_BCJ_ARM         => bcj_arm_encode(input),
        FILTER_BCJ_ARM64       => bcj_arm64_encode(input),
        FILTER_BCJ_PPC         => bcj_ppc_encode(input),
        FILTER_BCJ_SPARC       => bcj_sparc_encode(input),
        FILTER_BCJ_RISCV       => bcj_riscv_encode(input),
        FILTER_CFBF_DEFRAG     => cfbf_defrag_encode(input),
        _                      => input.to_vec(),
    }
}

pub fn undo_filter(input: &[u8], filter: u8) -> Vec<u8> {
    match filter {
        FILTER_DELTA1 | FILTER_DELTA2 | FILTER_DELTA3 | FILTER_DELTA4 => {
            delta_decode(input, filter as usize)
        }
        FILTER_SHUFFLE4_DELTA  => shuffle4_stl_delta_decode(input),
        FILTER_PLY_DELTA       => shuffle4_ply_delta_decode(input),
        FILTER_BCJ             => bcj_x86_decode(input),
        FILTER_STL_FIELD_MAJOR => field_major_stl_delta_decode(input),
        FILTER_BCJ_ARM         => bcj_arm_decode(input),
        FILTER_BCJ_ARM64       => bcj_arm64_decode(input),
        FILTER_BCJ_PPC         => bcj_ppc_decode(input),
        FILTER_BCJ_SPARC       => bcj_sparc_decode(input),
        FILTER_BCJ_RISCV       => bcj_riscv_decode(input),
        FILTER_CFBF_DEFRAG     => cfbf_defrag_decode(input),
        _                      => input.to_vec(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_filter_returns_none_for_short_input() {
        assert_eq!(detect_filter(&[0u8; 4]), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_none_for_random_data() {
        let mut state: u32 = 0xdeadbeef;
        let data: Vec<u8> = (0..1024).map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        }).collect();
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    #[test]
    fn detect_filter_returns_wav_stride_for_riff_wave() {
        let mut data = vec![0u8; 44];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WAVE");
        data[12..16].copy_from_slice(b"fmt ");
        data[16..20].copy_from_slice(&16u32.to_le_bytes());
        data[22..24].copy_from_slice(&1u16.to_le_bytes());
        data[34..36].copy_from_slice(&16u16.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_DELTA2);
    }

    #[test]
    fn detect_filter_returns_bmp_delta3_for_24bpp() {
        let mut data = vec![0u8; 32];
        data[0..2].copy_from_slice(b"BM");
        data[28..30].copy_from_slice(&24u16.to_le_bytes());
        assert_eq!(detect_filter(&data), FILTER_DELTA3);
    }

    #[test]
    fn detect_dixscript_binary_returns_filter_none() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&0x4D444958u32.to_le_bytes());
        data[4] = 1;
        assert_eq!(detect_filter(&data), FILTER_NONE);
    }

    #[test]
    fn mdix_magic_bytes_are_correct_le_encoding() {
        assert_eq!(DIXSCRIPT_MAGIC_BYTES, 0x4D444958u32.to_le_bytes());
        assert_eq!(&DIXSCRIPT_MAGIC_BYTES, b"XIDM");
    }

    #[test]
    fn detect_pe_coff_still_returns_bcj() {
        let mut data = vec![0u8; 256];
        data[0] = b'M'; data[1] = b'Z';
        data[0x3C] = 0x40;
        data[0x40] = b'P'; data[0x41] = b'E';
        data[0x42] = 0x00; data[0x43] = 0x00;
        assert_eq!(detect_filter(&data), FILTER_BCJ);
    }

    #[test]
    fn stl_field_major_flag_is_10() {
        assert_eq!(FILTER_STL_FIELD_MAJOR, 10u8);
    }

    #[test]
    fn apply_undo_flag10_roundtrip_sanity() {
        let mut data = vec![0u8; 84 + 50];
        data[80..84].copy_from_slice(&1u32.to_le_bytes());
        for i in 0..48usize { data[84 + i] = (i * 7 + 3) as u8; }
        let enc = apply_filter(&data, FILTER_STL_FIELD_MAJOR);
        let dec = undo_filter(&enc, FILTER_STL_FIELD_MAJOR);
        assert_eq!(dec, data);
    }

    // ── ELF detection via detect_filter ──────────────────────────────────────

    fn make_elf(e_machine_le: u16) -> Vec<u8> {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fELF");
        data[4] = 1; data[5] = 1;
        data[18..20].copy_from_slice(&e_machine_le.to_le_bytes());
        data
    }

    #[test]
    fn detect_filter_elf_arm_returns_bcj_arm() {
        assert_eq!(detect_filter(&make_elf(0x28)), FILTER_BCJ_ARM);
    }

    #[test]
    fn detect_filter_elf_arm64_returns_bcj_arm64() {
        assert_eq!(detect_filter(&make_elf(0xB7)), FILTER_BCJ_ARM64);
    }

    #[test]
    fn detect_filter_elf_ppc_returns_bcj_ppc() {
        assert_eq!(detect_filter(&make_elf(0x14)), FILTER_BCJ_PPC);
    }

    #[test]
    fn detect_filter_elf_sparc_returns_bcj_sparc() {
        assert_eq!(detect_filter(&make_elf(0x02)), FILTER_BCJ_SPARC);
    }

    #[test]
    fn detect_filter_elf_riscv_returns_bcj_riscv() {
        assert_eq!(detect_filter(&make_elf(0xF3)), FILTER_BCJ_RISCV);
    }

    #[test]
    fn detect_filter_elf_x86_returns_bcj() {
        assert_eq!(detect_filter(&make_elf(0x03)), FILTER_BCJ);
    }

    #[test]
    fn detect_filter_elf_x86_64_returns_bcj() {
        assert_eq!(detect_filter(&make_elf(0x3E)), FILTER_BCJ);
    }

    // ── Mach-O detection ─────────────────────────────────────────────────────

    fn make_macho_le(cpu_type: u32) -> Vec<u8> {
        let mut data = vec![0u8; 64];
        data[0] = 0xCF; data[1] = 0xFA; data[2] = 0xED; data[3] = 0xFE;
        data[4..8].copy_from_slice(&cpu_type.to_le_bytes());
        data
    }

    #[test]
    fn detect_filter_macho_arm64_returns_bcj_arm64() {
        assert_eq!(detect_filter(&make_macho_le(0x0100_000C)), FILTER_BCJ_ARM64);
    }

    #[test]
    fn detect_filter_macho_arm_returns_bcj_arm() {
        assert_eq!(detect_filter(&make_macho_le(0x0000_000C)), FILTER_BCJ_ARM);
    }

    #[test]
    fn detect_filter_macho_x86_64_returns_bcj() {
        assert_eq!(detect_filter(&make_macho_le(0x0100_0007)), FILTER_BCJ);
    }

    // ── a.out detection via detect_filter ─────────────────────────────────────

    fn make_sunos_aout_input(mid: u16, magic: u16) -> Vec<u8> {
        let mut data = vec![0u8; 32];
        data[0..2].copy_from_slice(&mid.to_be_bytes());
        data[2..4].copy_from_slice(&magic.to_be_bytes());
        data[4..8].copy_from_slice(&0x0000_4000u32.to_be_bytes()); // a_text
        data
    }

    #[test]
    fn detect_filter_aout_sparc_returns_bcj_sparc() {
        let data = make_sunos_aout_input(3, 0x010B);
        assert_eq!(detect_filter(&data), FILTER_BCJ_SPARC,
            "SunOS SPARC ZMAGIC a.out should → FILTER_BCJ_SPARC");
    }

    #[test]
    fn detect_filter_aout_68k_returns_bcj_sparc() {
        // 68020 binary uses SPARC BCJ as best available
        let data = make_sunos_aout_input(2, 0x0108);
        assert_eq!(detect_filter(&data), FILTER_BCJ_SPARC);
    }

    #[test]
    fn detect_filter_aout_sun386i_returns_bcj() {
        let data = make_sunos_aout_input(4, 0x010B);
        assert_eq!(detect_filter(&data), FILTER_BCJ,
            "Sun 386i a.out (x86) should → FILTER_BCJ");
    }

    #[test]
    fn detect_filter_aout_does_not_fire_for_text() {
        // English text: bytes 0-1 as a big-endian mid will almost certainly
        // exceed 9 (e.g. "th" = 0x7468), so the mid guard alone rejects it
        // regardless of what bytes 2-3 hold.
        let data = b"the quick brown fox jumps over the lazy dog ".to_vec();
        // If it somehow returns a BCJ flag, that's a false positive.
        // With real text this should be FILTER_NONE (no probe at < 512 bytes).
        let result = detect_filter(&data);
        assert!(
            result == FILTER_NONE || result == FILTER_DELTA1 || result == FILTER_DELTA2,
            "English text should not trigger BCJ a.out detection, got flag {}",
            result
        );
    }

    // ── New filter flag constants ─────────────────────────────────────────────

    #[test]
    fn new_bcj_flag_constants_are_correct() {
        assert_eq!(FILTER_BCJ_ARM,   11u8);
        assert_eq!(FILTER_BCJ_ARM64, 12u8);
        assert_eq!(FILTER_BCJ_PPC,   13u8);
        assert_eq!(FILTER_BCJ_SPARC, 14u8);
        assert_eq!(FILTER_BCJ_RISCV, 15u8);
    }

    // ── apply/undo roundtrips for new filters ─────────────────────────────────

    #[test]
    fn apply_undo_arm_bcj_roundtrip() {
        let mut data = vec![0u8; 64];
        data[3] = 0xEB; data[0] = 50;
        let enc = apply_filter(&data, FILTER_BCJ_ARM);
        let dec = undo_filter(&enc, FILTER_BCJ_ARM);
        assert_eq!(dec, data);
    }

    #[test]
    fn apply_undo_arm64_bcj_roundtrip() {
        let mut data = vec![0u8; 32];
        let instr: u32 = 0x14000014;
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = apply_filter(&data, FILTER_BCJ_ARM64);
        let dec = undo_filter(&enc, FILTER_BCJ_ARM64);
        assert_eq!(dec, data);
    }

    #[test]
    fn apply_undo_ppc_bcj_roundtrip() {
        let mut data = vec![0u8; 32];
        let instr: u32 = (18u32 << 26) | (20u32 & 0x03FF_FFFC) | 1;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = apply_filter(&data, FILTER_BCJ_PPC);
        let dec = undo_filter(&enc, FILTER_BCJ_PPC);
        assert_eq!(dec, data);
    }

    #[test]
    fn apply_undo_sparc_bcj_roundtrip() {
        let mut data = vec![0u8; 32];
        let instr: u32 = (1u32 << 30) | 15;
        data[0..4].copy_from_slice(&instr.to_be_bytes());
        let enc = apply_filter(&data, FILTER_BCJ_SPARC);
        let dec = undo_filter(&enc, FILTER_BCJ_SPARC);
        assert_eq!(dec, data);
    }

    #[test]
    fn apply_undo_riscv_bcj_roundtrip() {
        use crate::filters::bcj::jal_encode_imm;
        let mut data = vec![0u8; 32];
        let instr: u32 = jal_encode_imm(0x0000_006F, 12);
        data[0..4].copy_from_slice(&instr.to_le_bytes());
        let enc = apply_filter(&data, FILTER_BCJ_RISCV);
        let dec = undo_filter(&enc, FILTER_BCJ_RISCV);
        assert_eq!(dec, data);
    }
    }
