#!/bin/bash
set -e

mkdir -p bench_files
cd bench_files

echo "Downloading benchmark files..."
curl -s -o alice.txt "https://www.gutenberg.org/files/11/11-0.txt"

echo "Generating synthetic test files..."
python3 -c "print('the the the and the and ' * 500)" > repetitive.txt
python3 -c "import os; open('random.bin','wb').write(os.urandom(10000))"
# Small slice of alice for quick testing
head -c 20000 alice.txt > alice_small.txt
cp ../src/encoder.rs source_code.rs

cd ..
cargo build -q --release 2>/dev/null

echo ""
echo "Running benchmarks (using release build)..."
echo "================================================================"

run_bench() {
    local file=$1
    local name=$2
    local orig_size=$(wc -c < "bench_files/$file")

    # MBFA — use release build for fair timing
    local mbfa_start=$SECONDS
    ./target/release/mbfa compress "bench_files/$file" "bench_files/$file.mbfa" 2>/dev/null
    local mbfa_time=$((SECONDS - mbfa_start))
    local mbfa_size=$(wc -c < "bench_files/$file.mbfa" 2>/dev/null || echo 0)

    # gzip
    gzip -k -f "bench_files/$file" 2>/dev/null
    local gzip_size=$(wc -c < "bench_files/$file.gz" 2>/dev/null || echo 0)

    local mbfa_pct=$(echo "scale=1; $mbfa_size * 100 / $orig_size" | bc)
    local gzip_pct=$(echo "scale=1; $gzip_size * 100 / $orig_size" | bc)

    printf "%-20s %8s bytes | MBFA: %5s%% | gzip: %5s%% | MBFA time: %ss\n" \
        "$name" "$orig_size" "$mbfa_pct" "$gzip_pct" "$mbfa_time"
}

run_bench "repetitive.txt"  "Repetitive"
run_bench "alice_small.txt" "Prose (20KB)"
run_bench "random.bin"      "Random"
run_bench "source_code.rs"  "Source code"

echo "================================================================"
echo "Lower % = better compression"