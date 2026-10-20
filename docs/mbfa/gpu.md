# GPU match finding (optional, off by default)

`mbfa` can build the per-position match table for its hash-chain scan on a GPU
(`wgpu`: Vulkan, Metal, DirectX 12). It changes **speed only**: the table, the
tokens and the compressed bytes are identical to the CPU path.

## What is accelerated

`encoder::scan_from` spends most of its time in `find_match`, a hash-chain
walk. That walk does not depend on the parse (every position is inserted
exactly once, in order), so it can be computed for all positions at once.
`src/matchtable.rs` defines the table exactly and is always compiled;
`src/gpu.rs` + `src/gpu/match.wgsl` compute the same table on the GPU. The
sequential parts (lazy matching, the rep-slot ring, token choice) stay on the
CPU and read the table.

Not accelerated: `scan_discover`, the optimal parse (inputs up to 256 KB), the
entropy stage and decompression. Inputs up to 256 KB are dominated by the
sequential optimal parse, so the GPU does not help there.

## Turning it on

```
cargo build --release --features gpu        # needs a current Rust (see below)
```

Run time policy (`mbfa::Accel`, `mbfa::set_accel`, or `MBFA_GPU=off|auto|force`
for the CLI):

* `Off`: never use the GPU.
* `Auto` (default): GPU only if the feature is built in, an adapter passes the
  probe, and the scanned region is at least 1 MiB (`ACCEL_MIN_BYTES`) and at
  most 48 MiB (`ACCEL_MAX_BYTES`, host memory: index 4 B/byte + table 8 B/byte).
* `Force`: like `Auto` without the size minimum (tests, benchmarks).

## Safety gates (all fall back to the CPU path silently)

1. Cargo feature `gpu` is off by default; the default build never compiles wgpu.
2. The adapter must be a real GPU (software adapters are refused unless
   `gpu::allow_software_adapter(true)`), with enough storage-buffer and compute
   limits.
3. **Self-test at probe time**: a synthetic buffer is run through the GPU and
   the table must equal the CPU reference exactly, for three parameter sets. A
   driver or kernel that disagrees is never used.
4. Any wgpu error or lost device at run time marks the context broken and the
   caller continues on the CPU.
5. Oversized inputs, inputs larger than the adapter's buffer limit, and
   positions whose comparison would exceed the per-thread cap (`COMPARE_CAP`)
   are handled on the CPU (the table marks such entries and the scan
   recomputes them exactly).

`gpu::probe_report()` lists why each adapter was accepted or rejected.

## Rust version

`wgpu` 30 declares Rust 1.87, and at the time of writing a transitive
dependency needs 1.90. Only builds with `--features gpu` are affected; the
default build is unchanged. Use `cargo update <crate> --precise <version>` if
you need to hold back a dependency.

## Verifying and measuring

* `cargo test --release --features gpu --test gpu -- --test-threads=1` proves
  table, scan and whole-compress equality. On a machine with only a software
  Vulkan adapter (`apt install mesa-vulkan-drivers`) it runs on that; CI does
  this in `.github/workflows/mbfa-gpu.yml`.
* `cargo run --release --features gpu --example gpubench -- FILE...` prints CPU
  scan time against index, upload, kernel and the full GPU-driven scan, and
  checks the tokens are identical. Add `--software` to run on a software
  adapter (correctness only: its timings say nothing about real GPUs).

## Known limits

* Standard CI runners have no GPU, so speed has to be measured on a real
  machine. Nothing here claims a particular speedup.
* The kernel compares byte by byte from packed words and each scan uploads
  nothing new but the parameters; transfer and a CPU-built index are still on
  the critical path. These are the obvious next optimisations.
