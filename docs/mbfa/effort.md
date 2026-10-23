# Search effort (Fast / Normal / Max)

`mbfa::Effort` controls how hard the encoder searches on inputs **larger than
256 KB** (the optimal-parse limit). It never changes the format: files made at
any effort decompress with any build. `Fast` reproduces the output of builds
before this setting existed byte for byte.

| Effort | What it does |
|---|---|
| `Fast` | The original search. Also what the browser (wasm) build uses. |
| `Normal` (default) | Compares candidate parses by entropy-coded size (v9 range coder) instead of raw bitstream size, tries one extra scan with a window that reaches the whole input, 8-bit lengths and a hash chain 512 deep, and lets v9 code matches longer than 272 by splitting them. |
| `Max` | Same, chain depth 4096. Slow; for archival use. |

Set it with `mbfa::set_effort(mbfa::Effort::Fast)` or, for the CLI,
`MBFA_EFFORT=fast|normal|max`.

## Why

The raw bitstream charges every offset the full window width and every length
the full length width, so the old comparison preferred narrow windows and short
length fields even though the entropy coder (which decides the final size) does
not care. The old chain depth was also cut to 64 for inputs over 1 MB.
Separately, streams with long matches (length field above 8 bits) could not use
v9 at all.

## Measured (Silesia files and slices, same machine)

| Input | Fast | Normal |
|---|---|---|
| `xml` (5.3 MB) | 543,458 | 499,234 (-8.1%) |
| `mr` (10 MB) | 3,387,525 | 3,211,758 (-5.2%) |
| `dickens` (10 MB) | 3,650,454 | 3,286,366 (-10.0%) |
| `ooffice` (6.1 MB) | 2,434,228 | 2,405,619 (-1.2%) |
| 7 slices of 0.6-3 MB (total) | 3,347,967 | 3,257,300 (-2.7%); `Max` 3,219,544 (-3.8%) |

Time: Normal about 1.7x Fast on the slices and about 2.1x on the big files;
Max about 3.2x on the slices. Timings on a shared machine are noisy; compare
runs made back to back.

## Notes

* The extra deep scan is the biggest single scan, which is exactly what the GPU
  path (`docs/mbfa/gpu.md`) can build a table for.
* When an entropy-scored choice (the deep scan or a dictionary candidate) wins,
  folding stops after fold 1: the entropy stage codes the last fold's stream, so
  a second fold chosen by raw size would discard the estimate that picked the
  winner. (Without this rule `Terrain_large.raw` got 22% worse.)
* Inputs up to 256 KB are unaffected: the optimal parse decides those.
* The extra scan is accepted only if its v9-coded size is smaller than the
  current best. A better v9 estimate does not guarantee a smaller file in every
  case (another entropy variant can win on the baseline), so check your own data;
  `MBFA_EFFORT=fast` is the escape hatch.
