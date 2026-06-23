# Build plan & module ownership

Parallel build. Interfaces are fixed in the scaffold; each module is filled in
against its signatures and self-contained tests. The crate compiles green at all
times (stubs are `todo!()`), so a half-done module never blocks the others.

## Wave 1 — independent leaf modules (parallel)

No cross-dependencies; each ships with its own tests that build a tiny synthetic
fixture in-test (no shared model file needed).

| Module | Files | Done when |
|--------|-------|-----------|
| **gguf-reader** | `src/gguf/dtype.rs`, `src/gguf/reader.rs` | round-trips a synthetic GGUF written in-test (mirror Hephaistos's writer byte layout); `tensor_f32` returns correct values; metadata accessors work |
| **tokenizer** | `src/tokenizer.rs` | `decode(encode(s)) == s` for arbitrary UTF-8, using a small synthetic vocab/merges built in-test |
| **math-ops** | `src/math/ops.rs`, `src/math/matmul.rs` | unit tests: rmsnorm/softmax/silu/swiglu known values; `matvec` vs naive reference; rope rotates adjacent pairs correctly |
| **bench** | `benches/tokps.rs` | criterion harness compiles and skips cleanly when `models/tiny.gguf` is absent |

## Wave 2 — integration & depth (lead, sequential)

- **M2** `model/{config,weights,llama}.rs`, `kv_cache.rs` — wire wave 1 together into the forward pass. Gated by `tests/parity.rs`. **Do not parallelize.**
- **M3** `sample.rs`, `main.rs run` — sampling + CLI.
- **M4** quantization (extends `gguf/dtype.rs`, `matmul.rs`).
- **M5** SIMD + fused dequant.

## Conventions every module must hold

- **RoPE.** Hephaistos's `export_gguf` permutes q/k to GGUF interleaved layout, so
  RoPE is applied to **adjacent pairs** `(x[2i], x[2i+1])` per head, with
  `theta_i = pos * freq_base^(-2i/head_dim)`. Weights and rope must agree or M2
  parity fails.
- **Weight layout.** GGUF `ne` dims are reversed vs row-major shape. Data is
  row-major `[rows, cols] = [out, in]`; `matvec` takes `rows=out, cols=in`.
- **Single-token kernels.** Inference runs one position at a time, not `[B,T]`.
- **Zero-copy.** Weights are `&[f32]` views into the mmap; don't clone tensors.

## Hard rules

- **No `git commit` and no `git push`.** Leave all work uncommitted in the
  working tree. (Applies to every contributor and agent.)
- Keep the crate compiling: `cargo check` and `cargo test` must pass (stubbed
  tests are `#[ignore]`d or self-contained).
