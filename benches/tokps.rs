//! Tokens/sec benchmark (M5 scaffold). Owner: "bench" agent.
//!
//! Measures decode throughput of `Model::forward`. Requires `models/tiny.gguf`;
//! if absent, the bench skips cleanly so CI without a model stays green.
//! Compare the reported tok/s against llama.cpp on the same GGUF.

use std::path::Path;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};

const MODEL: &str = "models/tiny.gguf";

fn bench_decode(c: &mut Criterion) {
    if !Path::new(MODEL).exists() {
        eprintln!("skipping tokps bench: {MODEL} not found");
        return;
    }

    // Implementer: load once, then time N forward() steps per iteration and
    // report tokens/sec via a throughput element count.
    let _ = &c;
    let _ = Instant::now;
    todo!("load Model, warm up, time forward() decode loop; report tok/s")
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
