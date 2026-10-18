// scratch: compress each file, print sha-ish checksum + size + ms (stderr), so before/after builds can be diffed
use std::time::Instant;
fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let t0 = Instant::now();
        let c = mbfa::compress(&data, 8).expect("compress");
        let ms = t0.elapsed().as_millis();
        let mut h: u64 = 1469598103934665603;
        for &b in &c { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
        let d = mbfa::decompress(&c).expect("decompress");
        eprintln!("CZ\t{}\t{}\t{}\t{:016x}\t{}\t{}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            data.len(), c.len(), h, ms, if d == data { "PASS" } else { "FAIL" });
    }
}
