# Talos: writing llama.cpp — and a GPU backend — from scratch

Talos is a minimal LLM inference engine in Rust. It loads a GGUF model, runs a
Llama-style forward pass with a KV cache, samples tokens, and — with
`--features metal` — runs the whole forward pass on the Apple GPU. No ML
frameworks, no `ggml`, no `candle`: every kernel, from the byte-BPE tokenizer to
the attention softmax, is hand-written and checked against a reference.

It's the runtime half of a from-scratch loop with its sibling project
**Hephaistos** (a Llama trainer, also written from scratch): *train → forge a
GGUF → run*. Hephaistos is the smith; Talos is the bronze automaton that runs
what the smith made.

## Why build it

I wanted to actually understand how inference works — not call a library that
hides it. So the rule was: if I can't write it and test it, I don't get to use
it. That turned into a chain of milestones, each one a small thing I could verify
against a ground truth before moving on.

## The CPU engine (M0–M6)

- **GGUF reader** — mmap the file, parse the header / metadata / tensor index,
  hand out zero-copy views into the data section.
- **Byte-level BPE tokenizer** — `decode(encode(s)) == s` for arbitrary UTF-8.
- **Math kernels** — rmsnorm, RoPE (interleaved-pair convention, matched to the
  trainer's weight layout), softmax, SwiGLU, and a SIMD `matvec`.
- **The forward pass** — embed → per layer { rmsnorm, q/k/v, RoPE, KV-cache
  append, causal attention with grouped-query attention, output proj, residual,
  SwiGLU MLP, residual } → final norm → logits.
- **Quantization** — Q8_0 and Q4_0, dequantized on the fly during the matvec.
- **Perplexity harness** — the honest quality number.

The contract that kept it rigorous: Talos's logits have to match Hephaistos's
within `1e-4` for the same model and prompt (`tests/parity.rs`). Vibes don't
pass that test.

## The GPU backend (M7–M8.2)

This is the part I'm proudest of, because each step is independently measured and
honest about what it did *and didn't* buy.

**M7 — matvec on the GPU.** Decode is matrix-×-vector (one token at a time), so
the kernels launch one thread per output row. F32 plus *fused dequant* for
Q8_0/Q4_0 — the quantized weights stay quantized in GPU memory and are decoded
inside the shader, byte-for-byte matching the CPU path (f16 scale, signed int8,
Q4_0 nibble layout). Every kernel is checked against its CPU twin on random
inputs.

**M8.0 — resident weights.** The naive version re-uploaded each weight matrix
*every token* — effectively shipping the whole model to the GPU per token, which
is a bandwidth disaster. Keeping each tensor resident (uploaded once, keyed by
name) was a **6.3×** speedup on a 4096×4096 matvec (6.98 ms → 1.10 ms).

**M8.1 — a kernel that actually uses the GPU.** One simdgroup per output row: the
lanes stride across the row so adjacent threads read adjacent addresses
(coalesced), then `simd_sum` reduces the partials. **0.79 ms** (1.4× over M8.0).
Honest caveat at this point: still ~1.6× *slower* than the multithreaded CPU,
because every matvec was its own command buffer — commit/wait overhead on each
of the ~7 matvecs per layer.

**M8.2 — the whole forward pass on the GPU.** The residual stream and KV cache
stay resident across all layers; every op (rmsnorm, q/k/v, RoPE, attention,
SwiGLU, output projection) is encoded into a **single command buffer per token**,
reading back only the logits. The key trick: all dispatches go into one *serial*
compute encoder, so Metal orders them with memory coherence between kernels — no
inter-kernel races, no manual barriers. That removed the per-matvec overhead and
finally beat the CPU end-to-end:

| Decode (64 tokens) | CPU (rayon) | GPU | Speedup |
|---|---|---|---|
| F32 | 664 tok/s | **1843 tok/s** | **2.8×** |
| Q4_0 | 617 tok/s | **2302 tok/s** | **3.7×** |

Q4_0 wins most because decode is bandwidth-bound and there are fewer bytes to
move. The GPU path is wired into the actual `run` CLI, so real generation runs at
these speeds.

## How I kept it correct

Writing GPU kernels by hand is a great way to get silently-wrong numbers (no
bounds checks, accumulation-order differences, races). So correctness was never
assumed:

- Every kernel is tested against the CPU implementation on random data, with a
  *relative* tolerance (GPU and CPU sum in different orders, so bit-equality is
  the wrong bar).
- The full GPU forward is checked against the CPU forward on a synthetic model
  over a multi-token GQA sequence — it matches to **~1e-7** — and two GPU runs
  are required to be **bit-identical** (a race would show up as nondeterminism).
- The single-serial-encoder design means race-freedom is structural, not
  something I have to hope for.

## What I'd do next

- A coalesced / simdgroup-reduction attention kernel and fp16 compute, to push
  the speedup further on larger models.
- The kernel index math is `uint`; large models (7B+) would need `ulong` byte
  offsets.

## Stack

Rust (`rayon`, `memmap2`, `bytemuck`, `wide` for SIMD), Apple Metal via the
`metal` crate, hand-written `.metal` compute shaders. No inference or ML
framework. ~2.7k lines.
