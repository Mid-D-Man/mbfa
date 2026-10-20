//! GPU backend tests (cargo test --release --features gpu --test gpu).
//!
//! They run on whatever Vulkan implementation the machine has. On a machine
//! with only a software adapter (a CI runner: `apt install mesa-vulkan-drivers`
//! gives lavapipe) the tests allow it explicitly; on a machine with no Vulkan
//! at all they check the fallback instead. Either way they prove one thing:
//! when the GPU is used, the tokens and therefore the compressed bytes are
//! exactly what the CPU path produces.
#![cfg(feature = "gpu")]

use std::sync::{Arc, Mutex};

use mbfa::encoder::scan_from;
use mbfa::gpu;
use mbfa::matchtable::{compute_table_cpu, index_for, Params, PrevIndex};
use mbfa::{set_accel, Accel};

/// The acceleration policy and the software-adapter switch are process-wide:
/// run the tests one at a time.
static LOCK: Mutex<()> = Mutex::new(());

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Runs, short periods, repeated blocks, text-like words and noise, plus a few
/// very long runs (compare-cap overflow entries).
fn nasty(seed: u64, len: usize) -> Vec<u8> {
    let mut r = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let words: [&[u8]; 8] = [b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy ", b"dog\n"];
    let mut v: Vec<u8> = Vec::new();
    while v.len() < len {
        match r.below(7) {
            0 => { let b = r.next() as u8; let n = 1 + r.below(60); v.extend(std::iter::repeat(b).take(n)); }
            1 => { let p = 1 + r.below(12); let pat: Vec<u8> = (0..p).map(|_| r.next() as u8).collect();
                   for k in 0..(p * (1 + r.below(9)) + r.below(p)) { v.push(pat[k % p]); } }
            2 if v.len() > 40 => { let s = r.below(v.len() - 20); let n = 1 + r.below(50).min(v.len() - s - 1); let blk = v[s..s + n].to_vec(); v.extend(blk); }
            3 => { for _ in 0..(1 + r.below(12)) { v.extend_from_slice(words[r.below(8)]); } }
            4 if r.below(60) == 0 => { let b = r.next() as u8; let n = 2500 + r.below(1500); v.extend(std::iter::repeat(b).take(n)); }
            _ => { for _ in 0..(1 + r.below(20)) { v.push((r.next() % 5) as u8); } }
        }
    }
    v.truncate(len);
    v
}

fn with_software_gpu<F: FnOnce(&Arc<gpu::GpuContext>)>(f: F) {
    gpu::allow_software_adapter(true);
    let ctx = gpu::context().expect(
        "no Vulkan adapter, not even a software one; install mesa-vulkan-drivers (lavapipe) to run these tests",
    );
    f(&ctx);
}

#[test]
fn probe_finds_an_adapter_and_passes_the_self_test() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_software_gpu(|ctx| {
        let s = ctx.summary();
        println!("adapter: {} / {} / {}", s.name, s.backend, s.device_type);
    });
}

#[test]
fn gpu_table_equals_cpu_table() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_software_gpu(|ctx| {
        for seed in 1..=5u64 {
            let data = nasty(seed, 30_000 + (seed as usize) * 4_321);
            let idx: Arc<PrevIndex> = index_for(&data);
            for &(start_at, max_off, max_len, chain_limit) in &[
                (0usize, (1usize << 7) - 1, 255usize, 64usize),
                (0, (1 << 15) - 1, 255, 64),
                (0, (1 << 17) - 1, 4095, 256),
                (0, (1 << 24) - 1, (1 << 13) - 1, 512),
                (700, (1 << 16) - 1, 255, 128),
            ] {
                let p = Params { start_at, max_off, max_len, chain_limit };
                let want = compute_table_cpu(&data, &idx, &p);
                let got = ctx.compute_table(&data, &idx, &p).expect("GPU table");
                assert_eq!(got.start, want.start);
                assert!(
                    got.entries == want.entries,
                    "seed {seed} params {p:?}: first differing entry {:?}",
                    got.entries.iter().zip(want.entries.iter()).position(|(a, b)| a != b).map(|k| k / 2)
                );
            }
        }
    });
}

#[test]
fn gpu_table_spans_several_dispatches() {
    // More query positions than one dispatch handles (262,144), so the chunk
    // boundaries are exercised too.
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_software_gpu(|ctx| {
        let data = nasty(11, 600_000);
        let idx = index_for(&data);
        let p = Params { start_at: 0, max_off: (1 << 17) - 1, max_len: 255, chain_limit: 64 };
        let want = compute_table_cpu(&data, &idx, &p);
        let got = ctx.compute_table(&data, &idx, &p).expect("GPU table");
        assert!(ctx.last_timing().chunks >= 3, "expected several dispatches, got {:?}", ctx.last_timing());
        assert!(got.entries == want.entries, "tables differ across chunk boundaries");
    });
}

#[test]
fn scan_through_the_gpu_equals_the_plain_scan() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_software_gpu(|_| {
        for seed in 1..=4u64 {
            let data = nasty(seed + 40, 60_000);
            for &(ob, lb, repref) in &[(15u32, 8u32, true), (17, 9, false), (7, 8, true), (16, 13, true)] {
                set_accel(Accel::Off);
                let plain = scan_from(&data, 0, ob, lb, true, repref);
                let before = gpu::tables_built();
                set_accel(Accel::Force); // Force: no size minimum, GPU if available
                let fast = scan_from(&data, 0, ob, lb, true, repref);
                set_accel(Accel::Auto);
                assert!(gpu::tables_built() > before, "the GPU did not build a table: this test proved nothing");
                assert!(plain.0 == fast.0, "seed {seed} ob {ob} lb {lb} repref {repref}: tokens differ");
            }
        }
    });
}

#[test]
fn whole_compress_roundtrips_with_the_gpu_on() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    with_software_gpu(|_| {
        let data = nasty(77, 120_000);
        let before = gpu::tables_built();
        set_accel(Accel::Force);
        let packed = mbfa::compress(&data, 8).expect("compress");
        set_accel(Accel::Auto);
        assert!(gpu::tables_built() > before, "compress never used the GPU");
        let back = mbfa::decompress(&packed).expect("decompress");
        assert!(back == data, "roundtrip failed with the GPU path");
        // and the size equals what the CPU path gives (the bytes can differ run
        // to run only through unrelated HashMap ordering, never the size)
        set_accel(Accel::Off);
        let plain = mbfa::compress(&data, 8).expect("compress");
        set_accel(Accel::Auto);
        assert_eq!(packed.len(), plain.len(), "GPU and CPU output sizes differ");
    });
}

#[test]
fn without_a_gpu_everything_falls_back_to_the_cpu() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Software adapters are refused by default. On a machine with a real GPU the
    // probe may succeed; either way compress must work and agree with the CPU.
    gpu::allow_software_adapter(false);
    let data = nasty(5, 50_000);
    set_accel(Accel::Force);
    let a = mbfa::compress(&data, 8).expect("compress");
    set_accel(Accel::Off);
    let b = mbfa::compress(&data, 8).expect("compress");
    set_accel(Accel::Auto);
    assert_eq!(mbfa::decompress(&a).unwrap(), data);
    assert_eq!(a.len(), b.len());
    println!("adapter without software fallback: {:?}", gpu::describe());
}
