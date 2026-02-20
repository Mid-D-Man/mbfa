# MBFA — Mid Bit Folding Algorithm

**MidManStudio**

A novel iterative instruction-chain compression algorithm.
Rather than performing a single compression pass, MBFA folds data
through multiple passes where each pass produces an instruction stream
that constructs the previous pass — converging toward a minimal seed.

## Fixed Opcode Vocabulary

| Opcode | Bits | Meaning   | Operands                        |
|--------|------|-----------|---------------------------------|
| `0`    | 1    | BACKREF   | [6-bit offset][5-bit length]    |
| `10`   | 2    | LIT       | [8-bit byte]                    |
| `11`   | 2    | END       | none                            |

## Usage

```bash
cargo run --release -- compress   input.txt output.mbfa
cargo run --release -- decompress output.mbfa recovered.txt
```

## Status
🚧 Early research implementation
