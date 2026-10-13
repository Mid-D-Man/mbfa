// web/mbfa-wasm/smoke.mjs
//
// Runs the REAL built wasm module under Node and roundtrips test data through
// it. The CI workflow (.github/workflows/mbfa-wasm.yml) runs this between the
// build and the deploy, so a wasm that cannot actually compress never reaches
// gh-pages. It is the check that catches wasm32-specific breakage (a stray
// std::time::Instant, a thread spawn, an unexpected panic) which a native
// `cargo test` can never see.
//
// Usage, from the repo root, after
//   wasm-pack build web/mbfa-wasm --target web --release --out-dir pkg --out-name mbfa
//   node web/mbfa-wasm/smoke.mjs            # looks in web/mbfa-wasm/pkg
//   node web/mbfa-wasm/smoke.mjs <pkg-dir>  # or point at another directory
//
// Exit code 0 = every case roundtripped; 1 = something failed.

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgDir = path.resolve(process.argv[2] ?? path.join(here, 'pkg'));

// ── Load the module the same way the page does (glue + .wasm bytes) ───────────
const glue = await import(pathToFileURL(path.join(pkgDir, 'mbfa.js')).href);
const wasmBytes = readFileSync(path.join(pkgDir, 'mbfa_bg.wasm'));
try {
  await glue.default({ module_or_path: wasmBytes }); // wasm-bindgen >= 0.2.93
} catch (_) {
  await glue.default(wasmBytes);                      // older positional form
}
if (typeof glue.compress !== 'function' || typeof glue.decompress !== 'function') {
  console.error('FAIL: the module must export compress() and decompress(); got',
    Object.keys(glue).join(', '));
  process.exit(1);
}
console.log(`loaded ${path.join(pkgDir, 'mbfa_bg.wasm')} (${wasmBytes.length} bytes)`);

// ── Deterministic test data ───────────────────────────────────────────────────
function prng(seed) {                 // xorshift32
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;  s >>>= 0;
    return s;
  };
}
function randomBytes(n, seed) {
  const r = prng(seed), a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = r() & 0xff;
  return a;
}
function repeatBlock(block, n) {
  const a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = block[i % block.length];
  return a;
}
function textLike(n, seed) {
  const words = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'was', 'compression',
    'algorithm', 'token', 'offset', 'length', 'stream', 'quantum', 'marmalade'];
  const r = prng(seed), out = [];
  let size = 0;
  while (size < n) {
    const w = words[(r() % words.length) * (r() % words.length) % words.length] +
      (r() % 12 === 0 ? '\n' : ' ');
    out.push(w);
    size += w.length;
  }
  return new TextEncoder().encode(out.join('')).slice(0, n);
}

// Bound mirrors tests/roundtrip.rs: passthrough is input + 5, and inputs under
// 128 bytes may exceed that by up to 2 bytes (no final "store raw" guard).
const maxLen = (n) => n + 5 + (n < 128 ? 2 : 0);

const cases = [
  { name: 'empty',            data: new Uint8Array(0) },
  { name: '1 byte',           data: new Uint8Array([42]) },
  { name: '5 zero bytes',     data: new Uint8Array(5) },
  { name: '255 x 0x41',       data: new Uint8Array(255).fill(0x41) },
  { name: '256 x 0x41',       data: new Uint8Array(256).fill(0x41) },
  { name: 'period 3, 3000 B', data: repeatBlock(new Uint8Array([1, 2, 3]), 3000) },
  { name: 'text, 50 KB',      data: textLike(50_000, 7),  expectBelow: 0.75 },
  { name: 'zeros, 100 KB',    data: new Uint8Array(100_000), expectBelowBytes: 200 },
  { name: 'random, 20 KB',    data: randomBytes(20_000, 99) },
  { name: 'random, 60 KB',    data: randomBytes(60_000, 5) },   // above the 50 KB early-exit gate
];

// ── Run ───────────────────────────────────────────────────────────────────────
let failed = 0;
for (const c of cases) {
  const t0 = performance.now();
  try {
    const packed = glue.compress(c.data);              // maxFolds omitted: CLI default
    const t1 = performance.now();
    const back = glue.decompress(packed);
    const t2 = performance.now();

    const same = back.length === c.data.length && back.every((b, i) => b === c.data[i]);
    const problems = [];
    if (!same) problems.push('roundtrip mismatch');
    if (packed.length > maxLen(c.data.length)) {
      problems.push(`expanded ${c.data.length} -> ${packed.length}`);
    }
    if (c.expectBelow !== undefined && packed.length >= c.data.length * c.expectBelow) {
      problems.push(`expected below ${(c.expectBelow * 100) | 0}% but got ${packed.length}/${c.data.length}`);
    }
    if (c.expectBelowBytes !== undefined && packed.length >= c.expectBelowBytes) {
      problems.push(`expected below ${c.expectBelowBytes} B but got ${packed.length}`);
    }

    console.log(`${problems.length ? 'FAIL' : 'ok  '} ${c.name.padEnd(18)} ` +
      `${String(c.data.length).padStart(7)} -> ${String(packed.length).padStart(7)} B  ` +
      `compress ${(t1 - t0).toFixed(0)} ms, decompress ${(t2 - t1).toFixed(0)} ms` +
      (problems.length ? `  <-- ${problems.join('; ')}` : ''));
    if (problems.length) failed++;
  } catch (e) {
    console.log(`FAIL ${c.name.padEnd(18)} threw: ${e && e.message ? e.message : e}`);
    failed++;
  }
}

// An explicit maxFolds argument must be accepted too (the page omits it).
try {
  const d = textLike(5_000, 3);
  const back = glue.decompress(glue.compress(d, 1));
  if (back.length !== d.length || !back.every((b, i) => b === d[i])) throw new Error('mismatch');
  console.log('ok   explicit maxFolds=1 roundtrip');
} catch (e) {
  console.log(`FAIL explicit maxFolds=1: ${e && e.message ? e.message : e}`);
  failed++;
}

if (failed) {
  console.error(`\n${failed} case(s) failed`);
  process.exit(1);
}
console.log('\nall wasm smoke cases passed');
