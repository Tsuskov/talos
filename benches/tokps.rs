//! Tokens/sec benchmark (M5).
//!
//! Measures decode throughput of `Model::forward` for the F32 and Q4_0 bench
//! models, plus a scalar-vs-SIMD micro-benchmark of the matvec hot path. Skips
//! any model that's absent (the `.gguf` files are git-ignored). Criterion's
//! reported throughput (`thrpt: [.. Kelem/s]`) is tokens/sec.
//!
//! Regenerate the models from Hephaistos (`gen_bench_model`) then:
//!   cargo bench

use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use talos::model::Model;
use talos::sample::argmax;

const DECODE_LEN: u64 = 64;

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(DECODE_LEN));
    group.sample_size(20);

    for (name, path) in [("f32", "models/bench_f32.gguf"), ("q4", "models/bench_q4.gguf")] {
        if !Path::new(path).exists() {
            eprintln!("skipping {name}: {path} not found");
            continue;
        }
        let mut model = Model::load(Path::new(path)).expect("load model");
        let vocab = model.cfg.vocab_size as u32;
        group.bench_function(name, |b| {
            b.iter(|| {
                model.reset();
                let mut tok = 1u32;
                for pos in 0..DECODE_LEN as usize {
                    let logits = model.forward(tok, pos);
                    tok = argmax(black_box(&logits)) % vocab;
                }
                black_box(tok)
            });
        });
    }
    group.finish();
}

/// Scalar baseline dot, to size the SIMD win on the matvec hot path.
fn scalar_matvec(w: &[f32], x: &[f32], out: &mut [f32], cols: usize) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

fn bench_matvec(c: &mut Criterion) {
    // Single-threaded comparison (one row chunk) so it measures the kernel, not
    // rayon. 4096×4096 ~ a real model's projection.
    let (rows, cols) = (4096usize, 4096usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.02 - 0.12).collect();
    let mut out = vec![0.0f32; rows];

    let mut group = c.benchmark_group("matvec_4096");
    group.bench_function("scalar", |b| {
        b.iter(|| scalar_matvec(black_box(&w), black_box(&x), &mut out, cols));
    });
    group.bench_function("simd", |b| {
        b.iter(|| {
            for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
                *o = talos::math::matmul::dot(black_box(row), black_box(&x));
            }
        });
    });
    // The actual production CPU path (rayon-parallel over rows) — the honest
    // baseline for the GPU numbers below. `scalar`/`simd` above isolate the
    // single-row kernel.
    group.bench_function("cpu_parallel", |b| {
        b.iter(|| talos::math::matmul::matvec(black_box(&w), black_box(&x), &mut out, rows, cols));
    });
    // M7 vs M8.0: uploading the weight every call (the per-token cost) vs keeping
    // it resident on the GPU.
    #[cfg(feature = "metal")]
    {
        use talos::math::metal;
        group.bench_function("gpu_upload", |b| {
            b.iter(|| metal::matvec_f32(black_box(&w), black_box(&x), &mut out, rows, cols));
        });
        group.bench_function("gpu_resident", |b| {
            b.iter(|| {
                metal::matvec_f32_resident("bench.w", black_box(&w), black_box(&x), &mut out, rows, cols)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decode, bench_matvec);
criterion_main!(benches);
