//! MBFA CLI
//! Usage:
//!   mbfa compress <input_file> <output_file>
//!   mbfa decompress <input_file> <output_file>

use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage:");
        eprintln!("  mbfa compress   <input> <output>");
        eprintln!("  mbfa decompress <input> <output>");
        process::exit(1);
    }

    let command     = &args[1];
    let input_path  = &args[2];
    let output_path = &args[3];

    let input = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", input_path, e);
        process::exit(1);
    });

    let result = match command.as_str() {
        "compress"   => mbfa::compress(&input, 8),
        "decompress" => mbfa::decompress(&input),
        _ => {
            eprintln!("Unknown command: {}", command);
            process::exit(1);
        }
    };

    match result {
        Ok(output) => {
            fs::write(output_path, &output).unwrap_or_else(|e| {
                eprintln!("Failed to write {}: {}", output_path, e);
                process::exit(1);
            });
            println!("Done. {} bytes → {} bytes", input.len(), output.len());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    }
}
