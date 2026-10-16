#!/usr/bin/env python3
"""Per-file MBFA vs xz/brotli from the CI results on the gh-pages branch, split by the DP size limit.

Usage:  git clone --branch gh-pages https://github.com/Mid-D-Man/mbfa.git ghp
        python3 tools/lab/ci_tables.py ghp
History of any suite:  cd ghp && git log --format='%h %ad %s' --date=short -- corpus/results.json
"""
import json, sys, os
root = sys.argv[1] if len(sys.argv) > 1 else 'ghp'
DP_LIMIT = 262144          # entropy_v9::V9_OPTIMAL_MAX_BYTES: the v10 optimal parse only runs at or below this size
rows = []
for path, suite in (('corpus/results.json', 'Canterbury'), ('silesia/results.json', 'Silesia')):
    p = os.path.join(root, path)
    if not os.path.exists(p): continue
    for r in json.load(open(p)):
        rows.append((suite, r.get('file') or r.get('dataset'), r['original_bytes'], r['mbfa_pct'], r['xz_pct'],
                     r['brotli_pct'], r['mbfa_bytes'], r['xz_bytes']))
print(f"{'suite':<11}{'file':<13}{'size':>10} {'DP path':>8} {'MBFA%':>7} {'xz%':>7} {'brotli%':>8} {'vs xz':>8}")
for suite, name, o, m, x, b, mb, xb in sorted(rows, key=lambda t: (t[0], t[2])):
    print(f"{suite:<11}{name:<13}{o:>10} {('yes' if o <= DP_LIMIT else 'no'):>8} {m:>7} {x:>7} {b:>8} {(mb/xb-1)*100:>+7.1f}%")
for label, sel in (("<= 262,144 B (DP path)", lambda o: o <= DP_LIMIT), ("> 262,144 B (no DP)", lambda o: o > DP_LIMIT)):
    g = [(mb/xb-1)*100 for _, _, o, _, _, _, mb, xb in rows if sel(o)]
    if g: print(f"mean gap to xz, files {label}: {sum(g)/len(g):+.1f}%  (n={len(g)})")
