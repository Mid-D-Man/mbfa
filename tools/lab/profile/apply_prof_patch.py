#!/usr/bin/env python3
"""Instrument the SCRATCH copy of mbfa with stage timers (never run this on the real repo).

Usage:  python3 tools/lab/profile/apply_prof_patch.py $WORK/mbfa
Then:   cargo build --release --no-default-features --ignore-rust-version --example rtprof
        target/release/examples/rtprof FILE...      (stderr lines starting PROF)

It wraps, without changing behaviour:  compress, fold::fold, encoder::scan_adaptive,
encoder::scan_with_dict, fold's pair_encode call, lib::try_entropy_v10, and the filter
detect+apply. Use the SERIAL build (--no-default-features) so the stages are exclusive wall time.
PROF columns: name, bytes_in, bytes_out, wall_s, filter_ms, fold_total_ms, scan_adaptive_ms,
scan_with_dict_ms, pair_encode_ms, v10_dp_encode_ms, entropy_and_rest_ms
(fold_total includes scan_adaptive, scan_with_dict and pair_encode; entropy_and_rest = compress - filter - fold).
"""
import re, sys, os, shutil
root = sys.argv[1]
rd = lambda p: open(os.path.join(root, p), encoding='utf-8').read()
def wr(p, s): open(os.path.join(root, p), 'w', encoding='utf-8').write(s)

def once(s, old, new, label):
    assert s.count(old) == 1, f"{label}: expected exactly one match, found {s.count(old)}"
    return s.replace(old, new, 1)

wr('src/prof.rs', '''use std::sync::atomic::{AtomicU64, Ordering};
pub static NS: [AtomicU64; 8] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                                 AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub const NAMES: [&str; 8] = ["filter", "fold_total", "scan_adaptive", "scan_with_dict", "pair_encode", "v10_dp_encode", "-", "compress_total"];
pub fn add(i: usize, d: std::time::Duration) { NS[i].fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
pub fn take() -> [u64; 8] { let mut o = [0u64; 8]; for i in 0..8 { o[i] = NS[i].swap(0, Ordering::Relaxed); } o }
''')

# lib.rs ---------------------------------------------------------------------------------
s = rd('src/lib.rs')
s = once(s, 'pub mod optimal_parse;', 'pub mod optimal_parse;\npub mod prof;', 'lib: mod prof')
m = re.search(r'pub fn compress\(([^)]*)\)\s*->\s*([^{]+)\{', s); assert m, "compress signature not found"
args, ret = m.group(1), m.group(2).strip()
s = s.replace(m.group(0), f'pub fn compress({args}) -> {ret} {{\n    let __t = std::time::Instant::now();\n    let __r = compress_inner(input, max_folds);\n    prof::add(7, __t.elapsed());\n    __r\n}}\n\nfn compress_inner({args}) -> {ret} {{', 1)
s = once(s, 'fn try_entropy_v10(data: &[u8], ob: u32, lb: u32) -> Option<Vec<u8>> {\n    entropy_v9::write_tokens_v9_optimal(data, ob, lb).ok()',
         'fn try_entropy_v10(data: &[u8], ob: u32, lb: u32) -> Option<Vec<u8>> {\n    let __t = std::time::Instant::now();\n    let __r = entropy_v9::write_tokens_v9_optimal(data, ob, lb).ok();\n    prof::add(5, __t.elapsed());\n    __r', 'lib: v10')
s = once(s, 'let filter_flag = filters::detect_filter(input);', 'let __tf = std::time::Instant::now();\n    let filter_flag = filters::detect_filter(input);', 'lib: filter start')
s = once(s, 'let f = filters::apply_filter(input, filter_flag);', 'let f = filters::apply_filter(input, filter_flag);\n        prof::add(0, __tf.elapsed());', 'lib: filter end')
wr('src/lib.rs', s)

# fold.rs --------------------------------------------------------------------------------
f = rd('src/fold.rs')
mm = re.search(r'pub fn fold\(input: &\[u8\], max_folds: u8, filter_flag: u8\)\s*\n\s*->\s*([^{]+)\{', f); assert mm, "fold signature not found"
ret = mm.group(1).strip()
f = f.replace(mm.group(0), f'pub fn fold(input: &[u8], max_folds: u8, filter_flag: u8)\n    -> {ret} {{\n    let __t = std::time::Instant::now();\n    let __r = fold_inner(input, max_folds, filter_flag);\n    crate::prof::add(1, __t.elapsed());\n    __r\n}}\n\nfn fold_inner(input: &[u8], max_folds: u8, filter_flag: u8)\n    -> {ret} {{', 1)
f = once(f, 'let pair_result = pair_encode(tokens_for_pair, current_ob, current_lb)?;',
         'let __tp = std::time::Instant::now();\n            let pair_result = pair_encode(tokens_for_pair, current_ob, current_lb)?;\n            crate::prof::add(4, __tp.elapsed());', 'fold: pair')
wr('src/fold.rs', f)

# encoder.rs -----------------------------------------------------------------------------
e = rd('src/encoder.rs')
ma = re.search(r'pub fn scan_adaptive\(input: &\[u8\], skip_incompressible_bail: bool\) -> ([^{]+)\{', e); assert ma, "scan_adaptive signature not found"
ra = ma.group(1).strip()
e = e.replace(ma.group(0), f'pub fn scan_adaptive(input: &[u8], skip_incompressible_bail: bool) -> {ra} {{\n    let __t = std::time::Instant::now();\n    let __r = scan_adaptive_inner(input, skip_incompressible_bail);\n    crate::prof::add(2, __t.elapsed());\n    __r\n}}\n\nfn scan_adaptive_inner(input: &[u8], skip_incompressible_bail: bool) -> {ra} {{', 1)
md = re.search(r'pub fn scan_with_dict\((?P<params>[^)]*)\)\s*->\s*\(Vec<Token>, bool\)\s*\{', e); assert md, "scan_with_dict signature not found"
params = md.group('params'); names = ', '.join(p.split(':')[0].strip() for p in params.split(',') if p.strip())
e = e.replace(md.group(0), f'pub fn scan_with_dict({params}) -> (Vec<Token>, bool) {{\n    let __t = std::time::Instant::now();\n    let __r = scan_with_dict_inner({names});\n    crate::prof::add(3, __t.elapsed());\n    __r\n}}\n\nfn scan_with_dict_inner({params}) -> (Vec<Token>, bool) {{', 1)
wr('src/encoder.rs', e)

os.makedirs(os.path.join(root, 'examples'), exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rtprof.rs'), os.path.join(root, 'examples', 'rtprof.rs'))
print("instrumented", root, "- now build the rtprof example")
