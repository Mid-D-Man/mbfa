// Scratch harness (NOT a deliverable): roundtrip files via the public API,
// report to stderr because the library prints diagnostics to stdout.
use std::time::Instant;
fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let t0 = Instant::now();
        let c = mbfa::compress(&data, 8).expect("compress");
        let ms_c = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let d = mbfa::decompress(&c).expect("decompress");
        let ms_d = t1.elapsed().as_millis();
        let ok = d == data;
        eprintln!("RESULT\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            data.len(), c.len(), c.len() as f64 * 100.0 / data.len().max(1) as f64,
            ms_c, ms_d, if ok { "PASS" } else { "FAIL" });
    }
}
