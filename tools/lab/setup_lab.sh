#!/usr/bin/env bash
# Build the mbfa measurement lab in a SCRATCH copy (the repo itself is never touched).
# Usage: ./setup_lab.sh /path/to/mbfa-clone [workdir]
set -euo pipefail
REPO=${1:?path to a clone of Mid-D-Man/mbfa}
WORK=${2:-$HOME/mbfa-lab}
KIT=$(cd "$(dirname "$0")" && pwd)

# 1. rustc/cargo 1.75 from apt (the ceiling the user set; apt also offers 1.85/1.89/1.91 if it is ever lifted)
if ! command -v rustc >/dev/null; then
  apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends rustc cargo
fi
rustc --version

# 2. scratch copy
rm -rf "$WORK/mbfa"; mkdir -p "$WORK/mbfa"
(cd "$REPO" && tar --exclude=.git --exclude=target --exclude=Cargo.lock --exclude=web -cf - .) | tar -xf - -C "$WORK/mbfa"
cd "$WORK/mbfa"

# 3. bitstream-io >= 2.4.1 declares rustc 1.79 (inline const) and cannot build on 1.75.
#    2.3.0 builds, and gave byte-identical output to the CI build on every file compared.
sed -i 's/^bitstream-io = "2.4"/bitstream-io = "=2.3.0"/' Cargo.toml
grep -n '^bitstream-io' Cargo.toml

# 4. drop dev-dependencies and benches (criterion pulls edition-2024 crates that cargo 1.75 cannot parse)
python3 - <<'PY'
import re
s = open('Cargo.toml').read()
s = re.sub(r'\[dev-dependencies\].*?(?=\n\[|\Z)', '', s, flags=re.S)
s = re.sub(r'\[\[bench\]\].*?(?=\n\[|\Z)', '', s, flags=re.S)
open('Cargo.toml', 'w').write(s)
PY
rm -rf benches
mkdir -p examples && cp "$KIT"/examples/*.rs examples/

# 5. serial build (no rayon): matches the parallel CI build byte for byte, and stage timings are exclusive
export CARGO_TARGET_DIR="$WORK/target"
cargo build --release --no-default-features --ignore-rust-version --examples
echo
echo "lab ready:  $WORK/mbfa   (examples in $WORK/target/release/examples)"
echo "  roundtrip + size + time :  $WORK/target/release/examples/rt FILE...   (results on stderr, lines start with RESULT)"
