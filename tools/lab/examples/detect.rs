fn main() { for p in std::env::args().skip(1) { let d = std::fs::read(&p).unwrap();
  eprintln!("DETECT\t{}\t{}", std::path::Path::new(&p).file_name().unwrap().to_string_lossy(), mbfa::filters::detect_filter(&d)); } }
