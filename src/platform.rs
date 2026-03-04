// src/platform.rs
//! RAM detection for auto chunk size calculation.
use sysinfo::{MemoryRefreshKind, System};

/// Returns a safe chunk size based on available system RAM.
/// Uses 25% of available RAM, clamped between 64MB and 512MB.
pub fn auto_chunk_size() -> usize {
    let mut sys = System::new();
    sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
    let available = sys.available_memory() as usize;
    let chunk = available / 4;
    let clamped = chunk.clamp(64 * 1024 * 1024, 512 * 1024 * 1024);
    println!(
        "Available RAM: {} MB — chunk size: {} MB",
        available / (1024 * 1024),
        clamped / (1024 * 1024)
    );
    clamped
                              }
