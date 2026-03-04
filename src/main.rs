//! MBFA CLI
//! Usage:
//!   mbfa compress   <input_file> <output_file>
//!   mbfa decompress <input_file> <output_file>
//!   mbfa archive    <input_dir>  <output_file>
//!   mbfa extract    <input_file> <output_dir>  [--file <relative/path>]
//!   mbfa list       <input_file>

use std::{env, fs, path::Path, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "compress" => {
            expect_args(&args, 4);
            let input  = fs::read(&args[2]).unwrap_or_else(|e| die(&format!("read {}: {}", &args[2], e)));
            let result = mbfa::compress(&input, 8);
            write_result(result, &args[3], input.len());
        }

        "decompress" => {
            expect_args(&args, 4);
            let input = fs::read(&args[2]).unwrap_or_else(|e| die(&format!("read {}: {}", &args[2], e)));

            // Detect archive vs single-file
            if mbfa::archive::is_archive(&input) {
                die("This is an MBFA archive — use 'mbfa extract' instead");
            }

            let result = mbfa::decompress(&input);
            write_result(result, &args[3], input.len());
        }

        "archive" => {
            expect_args(&args, 4);
            let input_dir   = Path::new(&args[2]);
            let output_path = Path::new(&args[3]);

            if !input_dir.is_dir() {
                die(&format!("'{}' is not a directory", &args[2]));
            }

            mbfa::archive::create_archive(input_dir, output_path)
                .unwrap_or_else(|e| die(&format!("archive failed: {}", e)));
        }

        "extract" => {
            if args.len() < 4 {
                print_usage();
                process::exit(1);
            }

            let input_path = Path::new(&args[2]);
            let output_dir = Path::new(&args[3]);

            // Parse optional --file <path>
            let specific_file: Option<&str> = if args.len() >= 6 && args[4] == "--file" {
                Some(&args[5])
            } else {
                None
            };

            mbfa::archive::extract_archive(input_path, output_dir, specific_file)
                .unwrap_or_else(|e| die(&format!("extract failed: {}", e)));
        }

        "list" => {
            expect_args(&args, 3);
            let input_path = Path::new(&args[2]);

            mbfa::archive::list_archive(input_path)
                .unwrap_or_else(|e| die(&format!("list failed: {}", e)));
        }

        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            process::exit(1);
        }
    }
}

fn write_result(result: std::io::Result<Vec<u8>>, output_path: &str, input_len: usize) {
    match result {
        Ok(output) => {
            std::fs::write(output_path, &output)
                .unwrap_or_else(|e| die(&format!("write {}: {}", output_path, e)));
            println!("Done. {} bytes → {} bytes", input_len, output.len());
        }
        Err(e) => die(&format!("error: {}", e)),
    }
}

fn expect_args(args: &[String], count: usize) {
    if args.len() < count {
        print_usage();
        process::exit(1);
    }
}

fn die(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    process::exit(1);
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  mbfa compress   <input_file> <output_file>");
    eprintln!("  mbfa decompress <input_file> <output_file>");
    eprintln!("  mbfa archive    <input_dir>  <output_file>");
    eprintln!("  mbfa extract    <input_file> <output_dir>");
    eprintln!("  mbfa extract    <input_file> <output_dir> --file <relative/path>");
    eprintln!("  mbfa list       <input_file>");
}
