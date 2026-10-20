//! GPU match-table benchmark and equality check.
//!
//!   cargo run --release --features gpu --example gpubench -- [--software] FILE...
//!
//! For each file it times the plain CPU scan, the CPU parts the GPU path still
//! needs (building the chain index), the GPU table itself (upload, kernel,
//! readback), and the whole scan driven by the GPU table, then checks that both
//! scans produce identical tokens. `--software` lets it run on a software
//! Vulkan adapter such as lavapipe, which proves correctness but says nothing
//! about speed: only a real GPU's numbers mean anything.
use std::time::Instant;

use mbfa::encoder::{compute_chain_limit, scan_from};
use mbfa::gpu;
use mbfa::matchtable::{index_for, Params};
use mbfa::opcode::{max_length, max_offset};
use mbfa::{set_accel, Accel};

fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn main() {
    let mut files: Vec<String> = Vec::new();
    for a in std::env::args().skip(1) {
        if a == "--software" { gpu::allow_software_adapter(true); } else { files.push(a); }
    }
    if files.is_empty() {
        eprintln!("usage: gpubench [--software] FILE...");
        return;
    }
    match gpu::describe() {
        Some(s) => println!("adapter: {} ({}, {})", s.name, s.backend, s.device_type),
        None => {
            println!("no usable GPU adapter found (software adapters need --software); nothing to measure");
            for line in gpu::probe_report() { println!("  {line}"); }
            return;
        }
    }
    let ctx = gpu::context().expect("adapter vanished");

    println!("{:<22} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}  {}",
        "file", "bytes", "cpu scan", "index", "gpu upload", "gpu table", "gpu scan", "tokens");
    for path in files {
        let data = std::fs::read(&path).expect("read file");
        let (ob, lb) = (17u32, 8u32);

        set_accel(Accel::Off);
        let t = Instant::now();
        let plain = scan_from(&data, 0, ob, lb, true, true);
        let cpu_ms = ms(t);

        let t = Instant::now();
        let idx = index_for(&data);
        let index_ms = ms(t);

        let p = Params {
            start_at: 0,
            max_off: max_offset(ob),
            max_len: max_length(lb),
            chain_limit: compute_chain_limit(data.len(), ob),
        };
        let table = ctx.compute_table(&data, &idx, &p);
        let tm = ctx.last_timing();

        set_accel(Accel::Force);
        let t = Instant::now();
        let fast = scan_from(&data, 0, ob, lb, true, true);
        let gpu_scan_ms = ms(t);
        set_accel(Accel::Auto);

        let verdict = if table.is_none() { "GPU table failed".to_string() }
            else if plain.0 == fast.0 { "identical".to_string() } else { "DIFFERENT".to_string() };
        println!("{:<22} {:>9} {:>8.0}ms {:>8.0}ms {:>8.0}ms {:>8.0}ms {:>8.0}ms  {}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            data.len(), cpu_ms, index_ms, tm.upload_ms, tm.kernel_and_readback_ms, gpu_scan_ms, verdict);
    }
    println!("(gpu scan = index + upload + kernel + readback + the sequential parse; cpu scan = the plain hash-chain scan)");
}
