# tools/lab: mbfa measurement lab

Measurement tooling used to audit mbfa. Everything runs on a SCRATCH copy; the repo is never modified.

## Setup (about 1 minute)
    git clone https://github.com/Mid-D-Man/mbfa.git && cd mbfa
    bash tools/lab/setup_lab.sh . ~/mbfa-lab
Installs rustc/cargo 1.75 from apt if missing, copies the repo, pins `bitstream-io =2.3.0`
(newer needs rustc 1.79), strips dev-deps/benches, builds the examples serially.

## Tools
| what | how |
|---|---|
| roundtrip + size + time | `~/mbfa-lab/target/release/examples/rt FILE...`  (stderr: `RESULT name orig packed pct compress_ms decompress_ms PASS/FAIL`; stdout is noise, send it to /dev/null) |
| stage profile | `python3 tools/lab/profile/apply_prof_patch.py ~/mbfa-lab/mbfa`, then `cd ~/mbfa-lab/mbfa && CARGO_TARGET_DIR=~/mbfa-lab/target-prof cargo build --release --no-default-features --ignore-rust-version --example rtprof`, then run `rtprof FILE...` (columns documented in the patch script) |
| DP-limit experiment | copy the scratch tree, `sed -i 's/V9_OPTIMAL_MAX_BYTES: usize = 262_144/V9_OPTIMAL_MAX_BYTES: usize = 16_777_216/' src/entropy_v9.rs`, build `rt` into another target dir, compare with the default build |
| filters are bijections? | `filter_bij` (E8/E9-dense and uniform random bytes through every BCJ/delta filter; do not feed format-specific filters 7/8/10/16/17/18 arbitrary bytes) |
| which filter is chosen | `detect FILE...` |
| Silesia baselines (gzip-6, zstd-3, xz-6) | `python3 tools/lab/silesia_baselines.py DIR_WITH_SILESIA_FILES` |
| CI results by size | `git clone --branch gh-pages https://github.com/Mid-D-Man/mbfa.git ghp && python3 tools/lab/ci_tables.py ghp` |

Silesia without the official host: `git clone --depth 1 https://github.com/MiloszKrajewski/SilesiaCorpus` then unzip each file.
(Sizes were verified against the official table: 211,938,580 bytes in total.)

## Sandbox gotchas learned the hard way
* `/bin/sh` is dash: no `<<<`, `${x:0:2}`, `declare -A`, `PIPESTATUS`, `<(...)`. Use `bash script.sh` or Python.
* `cut -c` can split a multibyte character and make the whole tool output unreadable; truncate in Python.
* The container is recycled every ~10-15 minutes and detached jobs die with it; keep jobs under ~280 s per call, or relaunch.
* `pip install` needs `--break-system-packages`. api.github.com and web_fetch on github.com are blocked; `git clone` works.
* The apt toolchain has no wasm32 std at any version, so a real wasm32 build is impossible here.
