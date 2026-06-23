//! GQA correctness. A grouped-query model (n_head_kv < n_head) is mathematically
//! equivalent to a plain multi-head model whose K/V "group-leader" heads are
//! replicated across each group. We build both from the same random weights via
//! a minimal inline GGUF writer and assert Talos produces identical logits — so
//! this verifies the query-head -> kv-head mapping with no external fixtures.

use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// --- model dimensions ---
const EMBD: usize = 16;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2; // group size 2
const HD: usize = EMBD / N_HEAD; // 4
const KV_DIM: usize = N_HEAD_KV * HD; // 8
const FF: usize = 32;
const VOCAB: usize = 16;
const N_LAYER: usize = 2;
const CTX: usize = 32;
const GROUP: usize = N_HEAD / N_HEAD_KV;

// --- minimal GGUF v3 writer (mirrors the byte layout Talos reads) ---
#[derive(Default)]
struct Gguf {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

impl Gguf {
    fn u32(&mut self, k: &str, v: u32) {
        put_str(&mut self.kv, k);
        self.kv.extend_from_slice(&4u32.to_le_bytes());
        self.kv.extend_from_slice(&v.to_le_bytes());
        self.n_kv += 1;
    }
    fn f32(&mut self, k: &str, v: f32) {
        put_str(&mut self.kv, k);
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&v.to_le_bytes());
        self.n_kv += 1;
    }
    fn str(&mut self, k: &str, v: &str) {
        put_str(&mut self.kv, k);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut self.kv, v);
        self.n_kv += 1;
    }
    fn arr_str(&mut self, k: &str, v: &[String]) {
        put_str(&mut self.kv, k);
        self.kv.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        self.kv.extend_from_slice(&8u32.to_le_bytes()); // of STRING
        self.kv.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for s in v {
            put_str(&mut self.kv, s);
        }
        self.n_kv += 1;
    }
    /// `dims` is GGUF ne order (`[cols, rows]`); `data` is row-major `[rows, cols]`.
    fn tensor(&mut self, name: &str, dims: Vec<u64>, data: Vec<f32>) {
        self.tensors.push((name.to_string(), dims, data));
    }

    fn finish(self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.n_kv.to_le_bytes());
        buf.extend_from_slice(&self.kv);

        let align = |x: usize| x.div_ceil(32) * 32;
        let mut off = 0usize;
        let mut offsets = Vec::new();
        for (_, _, data) in &self.tensors {
            offsets.push(off as u64);
            off = align(off + data.len() * 4);
        }
        for ((name, dims, _), &offset) in self.tensors.iter().zip(&offsets) {
            put_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for ((_, _, data), &offset) in self.tensors.iter().zip(&offsets) {
            let target = data_start + offset as usize;
            buf.resize(target, 0);
            for &x in data {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf
    }
}

fn base_meta(g: &mut Gguf, n_head_kv: usize) {
    g.str("general.architecture", "llama");
    g.u32("llama.context_length", CTX as u32);
    g.u32("llama.embedding_length", EMBD as u32);
    g.u32("llama.block_count", N_LAYER as u32);
    g.u32("llama.feed_forward_length", FF as u32);
    g.u32("llama.attention.head_count", N_HEAD as u32);
    g.u32("llama.attention.head_count_kv", n_head_kv as u32);
    g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    g.u32("llama.rope.dimension_count", HD as u32);
    g.f32("llama.rope.freq_base", 10000.0);
    let tokens: Vec<String> = (0..VOCAB)
        .map(|i| if i == 0 { "<unk>".into() } else { format!("t{i}") })
        .collect();
    g.arr_str("tokenizer.ggml.tokens", &tokens);
    g.arr_str("tokenizer.ggml.merges", &[]);
    g.u32("tokenizer.ggml.unknown_token_id", 0);
    g.u32("tokenizer.ggml.bos_token_id", 0);
    g.u32("tokenizer.ggml.eos_token_id", 0);
}

/// One layer's shared (group-independent) weights, plus full per-head K and V.
struct Layer {
    attn_norm: Vec<f32>,
    q: Vec<f32>,
    k_full: Vec<f32>, // [N_HEAD*HD, EMBD]
    v_full: Vec<f32>,
    attn_output: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

/// Take head `src_head`'s `HD` rows (each EMBD wide) out of a full projection.
fn head_rows(full: &[f32], head: usize) -> &[f32] {
    &full[head * HD * EMBD..(head + 1) * HD * EMBD]
}

fn build(path: &Path, gqa: bool, layers: &[Layer], embd: &[f32], out_norm: &[f32], out: &[f32]) {
    let mut g = Gguf::default();
    base_meta(&mut g, if gqa { N_HEAD_KV } else { N_HEAD });
    g.tensor("token_embd.weight", vec![EMBD as u64, VOCAB as u64], embd.to_vec());
    g.tensor("output_norm.weight", vec![EMBD as u64], out_norm.to_vec());
    g.tensor("output.weight", vec![EMBD as u64, VOCAB as u64], out.to_vec());

    for (l, ly) in layers.iter().enumerate() {
        // K/V either keep n_head_kv group-leader heads (GQA) or replicate each
        // group leader across the whole group (equivalent MHA).
        let (kv_rows, mut k, mut v) = if gqa {
            (N_HEAD_KV, Vec::new(), Vec::new())
        } else {
            (N_HEAD, Vec::new(), Vec::new())
        };
        for r in 0..kv_rows {
            let leader = if gqa { r * GROUP } else { (r / GROUP) * GROUP };
            k.extend_from_slice(head_rows(&ly.k_full, leader));
            v.extend_from_slice(head_rows(&ly.v_full, leader));
        }
        let kv_dim = kv_rows * HD;

        let p = |n: &str| format!("blk.{l}.{n}");
        g.tensor(&p("attn_norm.weight"), vec![EMBD as u64], ly.attn_norm.clone());
        g.tensor(&p("attn_q.weight"), vec![EMBD as u64, EMBD as u64], ly.q.clone());
        g.tensor(&p("attn_k.weight"), vec![EMBD as u64, kv_dim as u64], k);
        g.tensor(&p("attn_v.weight"), vec![EMBD as u64, kv_dim as u64], v);
        g.tensor(&p("attn_output.weight"), vec![EMBD as u64, EMBD as u64], ly.attn_output.clone());
        g.tensor(&p("ffn_norm.weight"), vec![EMBD as u64], ly.ffn_norm.clone());
        g.tensor(&p("ffn_gate.weight"), vec![EMBD as u64, FF as u64], ly.ffn_gate.clone());
        g.tensor(&p("ffn_up.weight"), vec![EMBD as u64, FF as u64], ly.ffn_up.clone());
        g.tensor(&p("ffn_down.weight"), vec![FF as u64, EMBD as u64], ly.ffn_down.clone());
    }
    std::fs::write(path, g.finish()).unwrap();
}

fn run(path: &Path) -> Vec<f32> {
    let mut m = talos::model::Model::load(path).expect("load");
    let mut logits = Vec::new();
    for (pos, tok) in [1u32, 2, 3, 4, 5].into_iter().enumerate() {
        logits = m.forward(tok, pos);
    }
    logits
}

#[test]
fn gqa_matches_replicated_mha() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut rv = |n: usize| (0..n).map(|_| rng.gen_range(-0.5f32..0.5)).collect::<Vec<_>>();

    let embd = rv(VOCAB * EMBD);
    let out_norm = rv(EMBD);
    let out = rv(VOCAB * EMBD);
    let layers: Vec<Layer> = (0..N_LAYER)
        .map(|_| Layer {
            attn_norm: rv(EMBD),
            q: rv(EMBD * EMBD),
            k_full: rv(EMBD * EMBD),
            v_full: rv(EMBD * EMBD),
            attn_output: rv(EMBD * EMBD),
            ffn_norm: rv(EMBD),
            ffn_gate: rv(FF * EMBD),
            ffn_up: rv(FF * EMBD),
            ffn_down: rv(EMBD * FF),
        })
        .collect();

    let dir = std::env::temp_dir();
    let gqa_path = dir.join("talos_gqa.gguf");
    let mha_path = dir.join("talos_gqa_as_mha.gguf");
    build(&gqa_path, true, &layers, &embd, &out_norm, &out);
    build(&mha_path, false, &layers, &embd, &out_norm, &out);

    let gqa = run(&gqa_path);
    let mha = run(&mha_path);
    let max_diff = gqa.iter().zip(&mha).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("gqa vs replicated-mha max abs logit diff = {max_diff:e}");
    assert!(max_diff <= 1e-5, "GQA diverged from replicated MHA: {max_diff}");
}
