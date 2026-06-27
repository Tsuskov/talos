//! M8.2 safety net: the GPU forward must match the CPU forward.
//!
//! No real `.gguf` is present (parity.rs skips), so this builds a tiny synthetic
//! llama model with random F32 weights, loads it, and runs both
//! `Model::forward` (CPU reference) and `Model::forward_gpu` over a multi-token
//! sequence — exercising the growing KV cache and GQA. The GPU result must match
//! the CPU within tolerance, and be identical across two runs (a race in the
//! single command buffer would show up as nondeterminism).
//!
//! Only built with `--features metal`.
#![cfg(feature = "metal")]

use std::io::Write;

use rand::Rng;
use talos::model::Model;

const T_UINT32: u32 = 4;
const T_FLOAT32: u32 = 6;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const ALIGN: usize = 32;

fn put_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}
fn kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
    put_string(buf, key);
    buf.extend_from_slice(&T_UINT32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}
fn kv_f32(buf: &mut Vec<u8>, key: &str, v: f32) {
    put_string(buf, key);
    buf.extend_from_slice(&T_FLOAT32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}
fn kv_str(buf: &mut Vec<u8>, key: &str, v: &str) {
    put_string(buf, key);
    buf.extend_from_slice(&T_STRING.to_le_bytes());
    put_string(buf, v);
}
fn kv_arr_str(buf: &mut Vec<u8>, key: &str, vals: &[String]) {
    put_string(buf, key);
    buf.extend_from_slice(&T_ARRAY.to_le_bytes());
    buf.extend_from_slice(&T_STRING.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for s in vals {
        put_string(buf, s);
    }
}

/// (name, ne dims) — data is random F32 of length = product(dims).
struct Tensor {
    name: String,
    dims: Vec<u64>,
    data: Vec<f32>,
}

fn build_model_gguf() -> Vec<u8> {
    let (c, nh, nkv, ff, vocab, n_layer, ctx) = (16usize, 4usize, 2usize, 32usize, 16usize, 2usize, 8usize);
    let kv = nkv * (c / nh); // kv_dim
    let mut rng = rand::thread_rng();
    let mut rand_vec = |n: usize| (0..n).map(|_| rng.gen_range(-0.5f32..0.5)).collect::<Vec<_>>();

    // Tensor table. ne dims = [cols, rows] for a row-major [rows, cols] weight.
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut push = |name: String, dims: Vec<u64>, data: Vec<f32>| {
        tensors.push(Tensor { name, dims, data });
    };
    push("token_embd.weight".into(), vec![c as u64, vocab as u64], rand_vec(vocab * c));
    push("output_norm.weight".into(), vec![c as u64], rand_vec(c));
    push("output.weight".into(), vec![c as u64, vocab as u64], rand_vec(vocab * c));
    for l in 0..n_layer {
        push(format!("blk.{l}.attn_norm.weight"), vec![c as u64], rand_vec(c));
        push(format!("blk.{l}.attn_q.weight"), vec![c as u64, c as u64], rand_vec(c * c));
        push(format!("blk.{l}.attn_k.weight"), vec![c as u64, kv as u64], rand_vec(kv * c));
        push(format!("blk.{l}.attn_v.weight"), vec![c as u64, kv as u64], rand_vec(kv * c));
        push(format!("blk.{l}.attn_output.weight"), vec![c as u64, c as u64], rand_vec(c * c));
        push(format!("blk.{l}.ffn_norm.weight"), vec![c as u64], rand_vec(c));
        push(format!("blk.{l}.ffn_gate.weight"), vec![c as u64, ff as u64], rand_vec(ff * c));
        push(format!("blk.{l}.ffn_up.weight"), vec![c as u64, ff as u64], rand_vec(ff * c));
        push(format!("blk.{l}.ffn_down.weight"), vec![ff as u64, c as u64], rand_vec(c * ff));
    }

    let tokens: Vec<String> = (0..vocab).map(|i| format!("t{i}")).collect();

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    let kv_count = 15u64;
    buf.extend_from_slice(&kv_count.to_le_bytes());

    kv_str(&mut buf, "general.architecture", "llama");
    kv_u32(&mut buf, "llama.block_count", n_layer as u32);
    kv_u32(&mut buf, "llama.attention.head_count", nh as u32);
    kv_u32(&mut buf, "llama.attention.head_count_kv", nkv as u32);
    kv_u32(&mut buf, "llama.embedding_length", c as u32);
    kv_u32(&mut buf, "llama.feed_forward_length", ff as u32);
    kv_u32(&mut buf, "llama.context_length", ctx as u32);
    kv_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    kv_u32(&mut buf, "llama.rope.dimension_count", (c / nh) as u32);
    kv_f32(&mut buf, "llama.rope.freq_base", 10000.0);
    kv_arr_str(&mut buf, "tokenizer.ggml.tokens", &tokens);
    kv_arr_str(&mut buf, "tokenizer.ggml.merges", &[]);
    kv_u32(&mut buf, "tokenizer.ggml.unknown_token_id", 0);
    kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
    kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 2);

    // Offsets relative to data start, each tensor padded to ALIGN.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for t in &tensors {
        offsets.push(off);
        off = (off + t.data.len() * 4).div_ceil(ALIGN) * ALIGN;
    }

    for (t, &o) in tensors.iter().zip(&offsets) {
        put_string(&mut buf, &t.name);
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&(o as u64).to_le_bytes());
    }

    while buf.len() % ALIGN != 0 {
        buf.push(0);
    }
    let data_start = buf.len();
    for (t, &o) in tensors.iter().zip(&offsets) {
        let target = data_start + o;
        while buf.len() < target {
            buf.push(0);
        }
        for &x in &t.data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf
}

#[test]
fn forward_gpu_matches_cpu() {
    let bytes = build_model_gguf();
    let mut path = std::env::temp_dir();
    path.push(format!("talos_m82_{}.gguf", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();
    }

    let mut model = Model::load(&path).expect("load synthetic model");
    let vocab = model.cfg.vocab_size as u32;

    // A multi-token sequence so the KV cache grows past one entry.
    let prompt: [u32; 6] = [1, 3, 5, 7, 2, 4];

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut gpu_run1 = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let tok = tok % vocab;
        let cpu = model.forward(tok, pos);
        let gpu = model.forward_gpu(tok, pos);
        assert_eq!(cpu.len(), gpu.len());
        for (a, b) in gpu.iter().zip(&cpu) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / b.abs().max(1.0));
        }
        if pos == prompt.len() - 1 {
            gpu_run1 = gpu;
        }
    }
    eprintln!("forward GPU vs CPU: max abs {max_abs:e}, max rel {max_rel:e}");
    assert!(max_rel <= 1e-3, "GPU forward diverged from CPU: max rel {max_rel:e}");

    // Determinism: re-run the same sequence on the GPU; the last logits must be
    // bit-identical (a race in the command buffer would break this).
    model.reset();
    let mut gpu_run2 = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        gpu_run2 = model.forward_gpu(tok % vocab, pos);
    }
    assert_eq!(gpu_run1, gpu_run2, "GPU forward is nondeterministic (likely a race)");

    std::fs::remove_file(&path).ok();
}
