# MBFA — MidMans Bit Folding Algorithm

**MidManStudio** | Research Compression Algorithm

---

## What is MBFA?

MBFA is a novel multi-fold iterative compression algorithm under active research and development. The core idea is architecturally distinct from existing compression algorithms — instead of performing a single compression pass, MBFA folds data through multiple passes where each pass produces a bitstream of instructions that reconstruct the previous pass.

This is not just "compress it twice." Each fold operates on a fundamentally different type of data than the last — fold 1 sees raw bytes, fold 2 sees an instruction stream, fold 3 sees an instruction stream about an instruction stream. The reduced alphabet and structural regularity of each successive layer is what the algorithm exploits.

> **Status:** Research implementation. Functionally correct. Actively being improved.

---

## How It Works

```
Original bytes
    ↓  Fold 1: LZ scan → token stream → fixed-opcode bitstream
Fold 1 output
    ↓  Fold 2: token pair encoding with Cantor-paired operands (if large enough)
Fold 2 output
    ↓  Fold 3+: LZ on whatever bytes came out of the previous fold
    ↓  ... until stopping condition fires
Final compressed seed + 2-byte header
```

**Decompression** reads the header, runs the exact inverse number of passes, and reconstructs the original bytes exactly.

---

## Fixed Opcode Vocabulary

The opcode vocabulary is fixed and shared between encoder and decoder. **Never transmitted.** This is a core design decision — no per-fold table overhead.

| Opcode | Bit Pattern | Total Bits | Meaning | Operands |
|--------|-------------|------------|---------|----------|
| BACKREF | `0` | 24 bits | Copy from output history | 15-bit offset + 8-bit length |
| LIT | `10` | 10 bits | Emit one literal byte | 8-bit byte value |
| END | `11` | 2 bits | End of stream | none |

BACKREF gets the 1-bit code because it becomes the dominant token on any repetitive data after fold 1.

---

## Pair Encoding (Fold 2)

When fold 1 output exceeds 512 bytes, fold 2 uses token pair encoding instead of raw LZ. Adjacent tokens are combined into typed pairs:

| Pair | Prefix | Meaning |
|------|--------|---------|
| LL | `000` | LIT + LIT |
| LB | `001` | LIT + BACKREF |
| BL | `010` | BACKREF + LIT |
| BB | `011` | BACKREF + BACKREF |
| SL | `100` | single LIT (odd token out) |
| SB | `101` | single BACKREF |
| END | `110` | stream terminator |

**BACKREF operands** are compressed using Cantor pairing — `(offset, length)` encoded as a single number `cantor(x,y) = (x+y)(x+y+1)/2 + y`. If the result fits in 16 bits (1 flag bit + 16 value = 17 bits total), it wins over raw 24-bit encoding. Otherwise falls back to raw. No table, no transmission, fully reversible.

---

## Stopping Conditions

The encoder stops folding when any of these are true:

- Next fold output is not at least **3% smaller** than current (ratio ≥ 0.97)
- Output is at or below **64 bits** — too small for meaningful matching
- **Maximum 8 folds** reached

---

## File Format

```
Byte 0:    fold count
Byte 1:    pair flag (1 = fold 2 used pair encoding, 0 = used LZ)
Byte 2..N: compressed bitstream
```

---

## Current Benchmark Results

Tested against gzip and zstd on standard data types:

| Dataset | MBFA | gzip | zstd | Verdict |
|---------|------|------|------|---------|
| Repetitive 12KB | **0.2%** | 0.8% | ~0.3% | ✅ MBFA wins |
| Prose 20KB | 61.3% | 41.6% | 42.8% | ❌ gap remains |
| Prose 100KB | 50.5% | 36.4% | 37.6% | ❌ gap remains |
| Random 10KB | 100% | 100.3% | 100.1% | ✅ tie (correct) |
| Source code 3KB | 52.5% | 39.9% | 41.4% | ❌ gap remains |

Lower % = better. MBFA's advantage is on highly repetitive data where multi-fold convergence finds structure that single-pass algorithms physically cannot see.

**Known gap on prose/source:** The flat-cost LIT encoding (10 bits per byte regardless of frequency) is the primary bottleneck. gzip's adaptive Huffman gives common letters like `e` and `t` 3-4 bits. Closing this gap is the active focus.

---

## Project Structure

```
mbfa/
├── src/
│   ├── main.rs        CLI — compress / decompress
│   ├── lib.rs         public API
│   ├── opcode.rs      Token enum, fixed opcode constants
│   ├── encoder.rs     LZ scanner with hash chain — O(n) average
│   ├── bitwriter.rs   Token stream → packed bitstream
│   ├── bitreader.rs   Packed bitstream → Token stream
│   ├── decoder.rs     Token stream → reconstructed bytes
│   ├── pairing.rs     Token pair encoding + Cantor operand compression
│   ├── fold.rs        Orchestrates fold passes + stopping logic
│   └── unfold.rs      Reverses N fold passes using header
├── benches/
│   └── compare.rs     Criterion benchmarks
└── .github/
    └── workflows/
        └── mbfa-ci.yml  CI — build, test, benchmark, deploy
```

---

## Usage

```bash
# Compress
cargo run --release -- compress input.txt output.mbfa

# Decompress
cargo run --release -- decompress output.mbfa recovered.txt

# Verify roundtrip
diff input.txt recovered.txt
```

---

## CI / Benchmarks

Normal push runs build + tests on Ubuntu, macOS, Windows.

Add `--publish` or `--deploy` to your commit message to run full benchmarks and deploy the HTML report to GitHub Pages:

```bash
git commit -m "your message --publish"
git push
```

Results published at: **https://mid-d-man.github.io/mbfa/**

---

## Related Work

MBFA is architecturally distinct from all of the following but informed by them:

| Algorithm | Relationship |
|-----------|-------------|
| Fractal compression | Iterative self-referential encoding — but image-specific only |
| Re-Pair (Larsson & Moffat 1999) | Recursive symbol pairing — MBFA's pair encoding is independently derived, operates at token level not byte level |
| Iterated Function Systems | Mathematical ancestor of fractal compression — geometric not instruction-chain |
| Kolmogorov complexity | Theoretical "shortest program" framing — MBFA is the practical counterpart |
| Deflate / Zstd | LZ + entropy coding single-pass — MBFA's multi-fold chain is the key distinction |

---

## Research Potential

The algorithm has been assessed as having genuine Masters/PhD research potential subject to:

- Benchmarks on Canterbury and Silesia corpora vs Deflate, Zstd, LZMA
- Formal convergence proof — under what conditions is each fold guaranteed to shrink?
- Information-theoretic analysis of instruction stream entropy vs raw data entropy
- Full academic writeup positioning as "multi-fold program-style transform + fixed coding"

---

## Next Steps (Active)

- [ ] Replace Cantor pairing with Exp-Golomb for offset encoding — no quadratic blowup
- [ ] Move-To-Front transform between fold 1 and fold 2 — clusters opcodes before pairing
- [ ] Entropy coding on large streams only — size-gated to avoid header overhead on small files
- [ ] Formal benchmark on Canterbury corpus
- [ ] Convergence analysis writeup

---

*MidManStudio — MidMans Bit Folding Algorithm*
