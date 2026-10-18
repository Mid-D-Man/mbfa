#!/usr/bin/env python3
"""Run extra baseline compressors over a benchmark's files and merge the
numbers into that benchmark's results.json.

The three benchmark workflows (Canterbury, Silesia, special) already compare
MBFA against gzip -6, zstd -3, xz -6, brotli -11 and zpaq. This script adds
more reference points without touching their shell loops: after the main
step has written results.json it runs every extra tool that is installed on
the runner over the same input files and stores the outcome in each row under
`extra`, e.g.

    "extra": {
      "xz9e": {"label": "xz-9e", "cmd": "xz -9e", "bytes": 912345, "pct": 38.12,
               "ms": 18234, "dms": 410, "rt": true},
      "zopfli": {"label": "zopfli", "cmd": "zopfli --i5", "bytes": null,
                 "error": "timeout after 900s"}
    }

Rows written before this script existed have no `extra`; the HTML templates
treat that as "no extra columns", so old results.json files still render.

Every tool is verified: its output is decompressed and compared (size and
CRC-32) with the original, and `rt` says whether that matched. A tool that is
not installed is skipped silently (it simply gets no column); one that fails
or exceeds the time limit gets an entry with `bytes: null` and an `error`, so a
timeout can never be mistaken for a 0% result.

All tools run single-threaded, one at a time, like the existing baselines, so
the timings are comparable with each other and with MBFA's serial numbers.

Usage:
    extra_baselines.py RESULTS_JSON FILES_DIR [--key file|dataset]
                       [--file-map MAP.tsv] [--tools bzip2,xz9e,...]
                       [--timeout SECONDS]

    --key       which row field names the file (default: file; the special
                benchmark uses dataset)
    --file-map  TSV of `row-name<TAB>file-name` for benchmarks whose row name is
                not the file name (the special benchmark)
    --tools     comma-separated subset of the tool keys below (default: all)
    --timeout   per compress or decompress run (default: 900)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

# key, short label (table header), command shown in tooltips,
# compress argv, decompress argv, mode.
#   mode "stdio": compress reads the original on stdin and writes stdout;
#                 decompress reads the compressed file on stdin, writes stdout.
#   mode "file" : argv contains {src} and {out}; decompress argv contains {out}
#                 and writes the original to stdout.
#   mode "src_stdout": compress argv contains {src} and writes stdout (the tool
#                 cannot read stdin, e.g. zopfli); decompress as for "stdio".
TOOLS = [
    dict(key="lz4", label="lz4-9", cmd="lz4 -9", mode="stdio",
         comp=["lz4", "-9", "-q", "-c"], decomp=["lz4", "-d", "-q", "-c"]),
    dict(key="bzip2", label="bzip2-9", cmd="bzip2 -9", mode="stdio",
         comp=["bzip2", "-9", "-c"], decomp=["bzip2", "-d", "-c"]),
    dict(key="bzip3", label="bzip3", cmd="bzip3 -e -b 64 -j 1", mode="stdio",
         comp=["bzip3", "-e", "-b", "64", "-j", "1"], decomp=["bzip3", "-d", "-j", "1"]),
    dict(key="zstd19", label="zstd-19", cmd="zstd -19 --long=27", mode="stdio",
         comp=["zstd", "-19", "--long=27", "-T1", "-q", "-c"],
         decomp=["zstd", "-d", "--long=27", "-q", "-c"]),
    dict(key="xz9e", label="xz-9e", cmd="xz -9e", mode="stdio",
         comp=["xz", "-9e", "-T1", "-c"], decomp=["xz", "-d", "-c"]),
    dict(key="ppmd", label="7z-ppmd", cmd="7z a -m0=PPMd:o=32:mem=512m -mx=9", mode="file",
         comp=["7z", "a", "-t7z", "-m0=PPMd:o=32:mem=512m", "-mx=9", "-mmt=1",
               "-bd", "-bso0", "-bsp0", "-y", "{out}", "{src}"],
         decomp=["7z", "e", "-so", "-bd", "-bsp0", "-y", "{out}"]),
    dict(key="zopfli", label="zopfli", cmd="zopfli --i5", mode="src_stdout",
         comp=["zopfli", "--i5", "-c", "{src}"], decomp=["gzip", "-d", "-c"]),
]

CHUNK = 4 * 1024 * 1024


def available(tool):
    return all(shutil.which(argv0) for argv0 in (tool["comp"][0], tool["decomp"][0]))


def file_crc(path):
    crc, size = 0, 0
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            crc = zlib.crc32(b, crc)
            size += len(b)
    return crc, size


def run_compress(tool, src, out, timeout):
    """Compress src to out. Returns (ok, seconds, error)."""
    t0 = time.perf_counter()
    try:
        if tool["mode"] == "stdio":
            with open(src, "rb") as fin, open(out, "wb") as fout:
                p = subprocess.run(tool["comp"], stdin=fin, stdout=fout,
                                   stderr=subprocess.DEVNULL, timeout=timeout)
        elif tool["mode"] == "src_stdout":
            argv = [a.replace("{src}", src) for a in tool["comp"]]
            with open(out, "wb") as fout:
                p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=fout,
                                   stderr=subprocess.DEVNULL, timeout=timeout)
        else:
            if os.path.exists(out):
                os.unlink(out)
            argv = [a.replace("{src}", src).replace("{out}", out) for a in tool["comp"]]
            p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, f"timeout after {timeout}s"
    except OSError as e:
        return False, time.perf_counter() - t0, f"could not run: {e}"
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return False, dt, f"exit status {p.returncode}"
    return True, dt, None


def run_decompress(tool, packed, timeout):
    """Decompress and stream the output through a CRC. Returns (crc, size, seconds, error)."""
    if tool["mode"] in ("stdio", "src_stdout"):
        argv, stdin = tool["decomp"], open(packed, "rb")
    else:
        argv, stdin = [a.replace("{out}", packed) for a in tool["decomp"]], subprocess.DEVNULL
    t0 = time.perf_counter()
    crc, size = 0, 0
    try:
        p = subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while True:
            b = p.stdout.read(CHUNK)
            if not b:
                break
            crc = zlib.crc32(b, crc)
            size += len(b)
            if time.perf_counter() - t0 > timeout:
                p.kill()
                p.wait()
                return crc, size, time.perf_counter() - t0, f"decompress timeout after {timeout}s"
        p.wait()
    except OSError as e:
        return crc, size, time.perf_counter() - t0, f"could not run: {e}"
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        return crc, size, dt, f"decompress exit status {p.returncode}"
    return crc, size, dt, None


def measure(tool, src, orig_size, orig_crc, workdir, timeout):
    entry = {"label": tool["label"], "cmd": tool["cmd"]}
    out = os.path.join(workdir, "packed." + tool["key"])
    ok, secs, err = run_compress(tool, src, out, timeout)
    entry["ms"] = int(secs * 1000)
    if not ok:
        entry.update(bytes=None, error=err)
        return entry
    packed = os.path.getsize(out) if os.path.exists(out) else 0
    if packed <= 0 and orig_size > 0:
        entry.update(bytes=None, error="produced no output")
        return entry
    entry["bytes"] = packed
    entry["pct"] = round(packed * 100.0 / orig_size, 2) if orig_size else 100.0
    crc, size, dsecs, derr = run_decompress(tool, out, timeout)
    entry["dms"] = int(dsecs * 1000)
    entry["rt"] = derr is None and size == orig_size and crc == orig_crc
    if derr:
        entry["error"] = derr
    elif not entry["rt"]:
        entry["error"] = "decompressed output differs from the original"
    try:
        os.unlink(out)
    except OSError:
        pass
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("files_dir")
    ap.add_argument("--key", default="file")
    ap.add_argument("--file-map")
    ap.add_argument("--tools")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    wanted = None
    if args.tools:
        wanted = [t.strip() for t in args.tools.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in {x["key"] for x in TOOLS}]
        if unknown:
            sys.exit(f"unknown tool(s): {', '.join(unknown)}")
    tools = [t for t in TOOLS if wanted is None or t["key"] in wanted]
    missing = [t["cmd"].split()[0] for t in tools if not available(t)]
    tools = [t for t in tools if available(t)]
    if missing:
        print("not installed, skipped: " + ", ".join(sorted(set(missing))))
    if not tools:
        print("no extra tools available; results.json left unchanged")
        return 0

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)

    name_to_file = {}
    if args.file_map:
        with open(args.file_map, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and "\t" in line:
                    a, b = line.split("\t", 1)
                    name_to_file[a] = b

    print("extra baselines: " + ", ".join(t["label"] for t in tools))
    problems = 0
    with tempfile.TemporaryDirectory(prefix="extra_baselines_") as workdir:
        for row in rows:
            name = row.get(args.key) or row.get("file") or row.get("dataset")
            fname = name_to_file.get(name, name)
            src = os.path.join(args.files_dir, fname) if fname else None
            if not src or not os.path.isfile(src):
                print(f"  {name}: file not found, skipped")
                continue
            orig_size = os.path.getsize(src)
            orig_crc, _ = file_crc(src)
            extra = {}
            parts = []
            for tool in tools:
                e = measure(tool, src, orig_size, orig_crc, workdir, args.timeout)
                extra[tool["key"]] = e
                if e.get("bytes") is None or not e.get("rt", False):
                    problems += 1
                    parts.append(f"{tool['label']}: {e.get('error', 'failed')}")
                else:
                    parts.append(f"{tool['label']} {e['pct']}% C:{e['ms']}ms")
            row["extra"] = extra
            print(f"  {name:<28} {orig_size:>10} B | " + " | ".join(parts), flush=True)
            # Keep partial progress on disk so a cancelled run still has results.
            tmp = args.results + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)
            os.replace(tmp, args.results)

    print(f"done; {problems} tool run(s) failed, timed out or did not round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
