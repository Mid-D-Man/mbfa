// src/platform.rs
use sysinfo::{MemoryRefreshKind, System};

/// Returns the target block size for archive operations.
///
/// Scales proportionally with available RAM so low-memory systems don't
/// stage unnecessarily large blocks, but caps at 8 MB so individual
/// compress() calls stay within reasonable wall-clock time.
///
/// Scaling:
///   available / 256, clamped to [1 MB, 8 MB]
///
///   256 MB RAM → 1 MB   (floor)
///   512 MB RAM → 2 MB
///   1 GB  RAM → 4 MB
///   ≥2 GB RAM → 8 MB   (ceiling)
///
/// At 8 MB, prose compresses in ~2–8 s per block (0.1–0.4 MB/s).
/// Highly repetitive data is much faster; incompressible data passhthroughs
/// in milliseconds regardless of block size.
pub fn auto_chunk_size() -> usize {
    let mut sys = System::new();
    sys.refresh_memory_specifics(MemoryRefreshKind::new().with_ram());
    let available = sys.available_memory() as usize;
    let chunk   = available / 256;
    let clamped = chunk.clamp(1 * 1024 * 1024, 8 * 1024 * 1024);
    println!(
        "Available RAM: {} MB — chunk size: {} MB",
        available / (1024 * 1024),
        clamped   / (1024 * 1024)
    );
    clamped
}
