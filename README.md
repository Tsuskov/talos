# Talos

A minimal LLM inference engine in Rust. Talos loads a GGUF model, runs a
Llama-style forward pass with a KV cache, and samples tokens — a tiny
`llama.cpp` written from scratch to understand how inference actually works.

It's the runtime half of a from-scratch loop: **train → forge GGUF → run**.
The companion project trains the model and exports the GGUF; Talos runs it.

![Talos heaving a boulder at the Argo, as block-ASCII art](assets/talos.png)

> In myth, Talos was the bronze automaton forged by **Hephaistos** to guard
> Crete — the machine that runs what the smith made. Above: Asmus Jacob
> Carstens' engraving of Talos (public domain), rendered to colored block-ASCII
> with [ren-ascii-sance](https://github.com/Tsuskov/ren-ascii-sance).

## Status

M0–M5 implemented, plus grouped-query attention: GGUF reader, byte-BPE
tokenizer, math kernels, and the
Llama forward pass with a KV cache, verified against the trainer's logits.

| Milestone | What | Verify |
|-----------|------|--------|
| M0 | GGUF reader (mmap, metadata, tensor index) | `talos inspect model.gguf` lists every tensor + hyperparam |
| M1 | Byte-BPE tokenizer | `decode(encode(s)) == s` |
| M2 | F32 forward + KV cache, greedy decode | first-token logits match the trainer within 1e-4 (`tests/parity.rs`) |
| M3 | Sampling (temp / top-k / top-p) + `run` CLI | coherent generations; temp=0 == greedy |
| M4 | Quantization (Q8_0, Q4_0) | perplexity within a few % of F32; ~4× smaller |
| M5 | SIMD matmul + fused dequant | tok/s vs llama.cpp (`benches/tokps.rs`) |

## Usage (target)

```sh
talos inspect models/tiny.gguf
talos run models/tiny.gguf --prompt "Once upon a time" -n 128 --temp 0.8
```

## Layout

```
src/gguf/      GGUF v3 reader (header, metadata, tensor index)
src/tokenizer  byte-level BPE
src/math/      rmsnorm, rope, softmax, swiglu, matvec
src/model/     config, weight handles, Llama forward pass
src/kv_cache   per-layer key/value cache
src/sample     logit sampling
tests/parity   the numerical contract
benches/tokps  throughput
```

See `BUILD.md` for module ownership and the parallel build plan.
