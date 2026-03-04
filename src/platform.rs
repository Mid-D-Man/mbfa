// src/platform.rs
use sysinfo::{MemoryRefreshKind, System};

pub fn auto_chunk_size() -> usize {
    let mut sys = System::new();
    sys.refresh_memory_specifics(MemoryRefreshKind::new().with_ram());
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
