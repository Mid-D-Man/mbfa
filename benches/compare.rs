//! Criterion benchmarks
//! Compares MBFA compressed sizes against baseline

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_compress(c: &mut Criterion) {
    let repetitive = b"the the the and the and the and the cat sat on the mat".repeat(100);
    let random_ish: Vec<u8> = (0u8..=255).cycle().take(5000).collect();

    c.bench_function("mbfa_compress_repetitive", |b| {
        b.iter(|| mbfa::compress(&repetitive, 8).unwrap())
    });

    c.bench_function("mbfa_compress_random", |b| {
        b.iter(|| mbfa::compress(&random_ish, 8).unwrap())
    });
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
