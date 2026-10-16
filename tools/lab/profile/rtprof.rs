// Scratch: compress only, print the stage breakdown (serial build, so stages are exclusive wall time).
fn main() {
    let names = mbfa::prof::NAMES;
    for path in std::env::args().skip(1) {
        let d = std::fs::read(&path).unwrap();
        let _ = mbfa::prof::take();
        let t = std::time::Instant::now();
        let c = mbfa::compress(&d, 8).unwrap();
        let wall = t.elapsed().as_secs_f64();
        let p = mbfa::prof::take();
        let total = p[7] as f64;
        let ms = |i: usize| p[i] as f64 / 1e6;
        let entropy_and_rest = total - p[0] as f64 - p[1] as f64;
        eprintln!("PROF\t{}\t{}\t{}\t{:.2}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(), d.len(), c.len(), wall,
            ms(0), ms(1), ms(2), ms(3), ms(4), ms(5), entropy_and_rest / 1e6);
        let _ = names;
    }
}
