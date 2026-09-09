# Contributing to MBFA

Thanks for taking a look at MBFA. This is an active research project, so
the bar for contributions is mostly about not losing information: keep
correctness verifiable, keep the documentation honest, and don't let
design history get lost.

## Before you start

- Check [docs/mbfa.md](docs/mbfa.md) for the documentation index and a
  map of the crate's five parts (core, entropy, archive, dictionary,
  filters). Each part's doc file explains what its modules do, the
  design decisions behind them, and their fix/problem history — read the
  relevant part before changing a file in it.
- MBFA has no formal issue tracker workflow beyond GitHub issues/PRs.
  For anything non-trivial (a new entropy variant, a new filter, a
  change to the file format), opening an issue first to discuss the
  approach is appreciated but not required.

## Building and testing

```bash
# Build (default features: parallel + archive)
cargo build --release

# Run the test suite
cargo test

# Build without the archive/parallel features (e.g. for a
# single-threaded or filesystem-less target)
cargo build --release --no-default-features
```

The `"archive"` feature implies `"parallel"` (archive block processing
is unconditionally parallel). Disabling `"parallel"` alone still needs
`compress()`/`decompress()` to produce byte-identical output to the
parallel path — verify this if you touch anything under
`#[cfg(feature = "parallel")]`.

## Benchmarks

```bash
cargo bench
```

Criterion benchmarks live in `benches/compare.rs` and measure MBFA
against gzip/zstd. CI also runs three additional benchmark suites
(archive, Canterbury Corpus, special) — see the README's
[CI / Benchmarks](README.md#ci--benchmarks) section for how to trigger
them from a commit message. `criterion`'s `html_reports` feature (which
pulls in `plotters`) adds 30-60s to a clean build; it's fine to drop it
from `Cargo.toml`'s `[dev-dependencies]` during active development and
re-add it only for benchmark/publish runs.

## Documentation and commenting conventions

MBFA follows [docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md](docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md)
for all source comments and crate documentation. The short version:

- Every source file has a NOTICE header at the top pointing to its doc
  file and section (e.g. `docs/mbfa/core.md`, section "encoder.rs").
- Inline comments explain what isn't obvious from the code itself —
  they never carry a fix history, a bug report, or a decision log. That
  content belongs in the relevant part's doc file instead.
- When you touch a file, check whether it already has a section in its
  part's doc file. If not, add one. If your change fixes something or
  makes a non-obvious decision, add it to that doc file's own
  "Fixes and Problems" section (or the module's own bullets) rather
  than leaving it as a comment in the source.

Read the guidelines file itself before writing anything nontrivial —
it also covers writing style (plain, direct, no marketing language) and
how to handle large multi-file changes.

## Code changes

- **Don't change the wire format casually.** The header layout, opcode
  vocabulary, and entropy variant flags are all documented in the
  README and in `docs/mbfa/core.md`/`docs/mbfa/entropy.md`. A file
  format change needs a version bump story (how does an old archive or
  compressed file still decode?) before it's mergeable.
- **New entropy variants or filters** should compete honestly: if it's
  an entropy variant, it should participate in the same brute-force
  tournament as the existing ones (measure actual output size, don't
  guess); if it's a filter, it needs its own `detect`/`apply`/`undo`
  triple and a slot in `filters/mod.rs`'s detection order.
- **Roundtrip everything.** Any new codec path (entropy variant, filter,
  dictionary) needs a test that compresses and decompresses real data
  and asserts byte-exact equality, not just that it doesn't panic.
- Keep pull requests scoped to one concern where practical — a new
  filter, a bug fix, a documentation pass — rather than mixing several
  unrelated changes.

## Reporting issues

When filing a bug, include the smallest input file you can that
reproduces it, and (if compression-side) which entropy variant/filter/
dictionary flag the compressed output's header claims to have used —
`main.rs`'s CLI or a short script against `lib::compress`/`decompress`
can print the header bytes directly.
