// Match table kernel: for every query position q, the same chain walk as
// encoder::find_match (see matchtable.rs for the exact definition).
//
// One thread per position. `prev` holds, for each position, the newest earlier
// position with the same hash (built once on the CPU), so the walk is just
// following links. The result is [offset, length]; length 0xFFFFFFFF marks an
// entry that hit the compare cap and must be recomputed by the CPU.

struct Params {
    n: u32,
    base: u32,        // first query position of this dispatch
    count: u32,       // number of query positions in this dispatch
    max_off: u32,
    max_len: u32,
    chain_limit: u32,
    cap: u32,         // longest comparison one entry attempts
    pad: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> data: array<u32>;        // input bytes, little-endian words
@group(0) @binding(2) var<storage, read> prev: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;   // 2 words per query position

const NONE: u32 = 0xFFFFFFFFu;

fn byte_at(i: u32) -> u32 {
    return (data[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

// Number of leading bytes where data[q + t] == data[j + t], at most `limit`.
// For an overlapping match (q - j < length) this reads bytes that are part of
// the repeating pattern itself, which is exactly the periodic comparison the
// CPU does: all earlier bytes matched, so data[j + t] == data[q + t - span].
fn match_len(q: u32, j: u32, limit: u32) -> u32 {
    var t: u32 = 0u;
    loop {
        if (t >= limit) { break; }
        if (byte_at(q + t) != byte_at(j + t)) { break; }
        t = t + 1u;
    }
    return t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= P.count) { return; }
    let q = P.base + k;

    var cur = prev[q];
    let limit = min(P.max_len, P.n - q);
    var best_len: u32 = 0u;
    var best_off: u32 = 0u;
    var steps: u32 = 0u;
    var overflow = false;

    loop {
        if (cur == NONE || steps >= P.chain_limit) { break; }
        let j = cur;
        if (q <= j || q - j > P.max_off) { break; }
        if (best_len >= limit) { break; }
        let span = q - j;
        var probe = best_len;
        if (best_len >= span) { probe = best_len % span; }
        if (byte_at(j + probe) == byte_at(q + best_len)) {
            let cmp_limit = min(limit, P.cap);
            let len = match_len(q, j, cmp_limit);
            if (len == P.cap && limit > P.cap) { overflow = true; break; }
            if (len > best_len) {
                best_len = len;
                best_off = span;
                if (best_len == P.max_len) { break; }
            }
        }
        cur = prev[j];
        steps = steps + 1u;
    }

    if (overflow) {
        out[2u * k] = 0u;
        out[2u * k + 1u] = 0xFFFFFFFFu;
    } else {
        out[2u * k] = best_off;
        out[2u * k + 1u] = best_len;
    }
}
