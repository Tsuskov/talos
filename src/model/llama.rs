//! The Llama-style forward pass. Owner: lead (M2 — the keystone).
//!
//! Per token at position `pos`:
//!   embed -> for each layer { rmsnorm, q/k/v projections, RoPE(q,k),
//!   append k,v to cache, causal attention over cache, attn_output, residual,
//!   rmsnorm, SwiGLU MLP, residual } -> final rmsnorm -> output projection.
//!
//! Must match Hephaistos numerically (tests/parity.rs).

use anyhow::{bail, Result};
use std::path::Path;

use crate::gguf::GgufFile;
use crate::kv_cache::KvCache;
use crate::math::ops::{rmsnorm, rope, softmax, swiglu};
use crate::model::{Config, Weights};
use crate::tokenizer::Tokenizer;

pub struct Model {
    pub cfg: Config,
    pub tokenizer: Tokenizer,
    gguf: GgufFile,
    kv: KvCache,
}

impl Model {
    /// Load a GGUF model from disk (config + tokenizer + weights + fresh cache).
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        if cfg.n_head_kv != cfg.n_head {
            bail!(
                "grouped-query attention not yet supported (head_count_kv={} != head_count={})",
                cfg.n_head_kv,
                cfg.n_head
            );
        }
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let kv = KvCache::new(cfg.n_layer, cfg.n_head_kv, cfg.head_dim(), cfg.context_length);
        Ok(Self { cfg, tokenizer, gguf, kv })
    }

    /// Run one decode step for `token` at sequence position `pos`, returning
    /// logits over the vocabulary. Appends this step's keys/values to the cache.
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // Disjoint field borrows: `w`/`cfg` shared, `kv` mutable.
        let cfg = &self.cfg;
        let kv = &mut self.kv;
        let w = Weights::from_gguf(&self.gguf, cfg).expect("weights bind");

        let c = cfg.n_embd;
        let nh = cfg.n_head;
        let hd = cfg.head_dim();
        let eps = cfg.rms_eps;
        let scale = 1.0 / (hd as f32).sqrt();

        // Residual stream, initialized to the token embedding.
        let mut x = vec![0.0f32; c];
        w.token_embd.dequant_row(token as usize, &mut x);

        let mut normed = vec![0.0f32; c];
        for (l, lw) in w.layers.iter().enumerate() {
            // --- attention block ---
            rmsnorm(&x, lw.attn_norm, eps, &mut normed);

            let mut q = vec![0.0f32; c];
            let mut k = vec![0.0f32; c];
            let mut v = vec![0.0f32; c];
            lw.attn_q.matvec(&normed, &mut q);
            lw.attn_k.matvec(&normed, &mut k);
            lw.attn_v.matvec(&normed, &mut v);

            rope(&mut q, pos, nh, hd, cfg.rope_freq_base);
            rope(&mut k, pos, nh, hd, cfg.rope_freq_base);

            kv.append(l, &k, &v);
            let keys = kv.keys(l);
            let vals = kv.values(l);
            let seq = kv.len(); // = pos + 1

            // Per-head causal attention over the cache.
            let mut atty = vec![0.0f32; c];
            let mut scores = vec![0.0f32; seq];
            for h in 0..nh {
                let qh = &q[h * hd..h * hd + hd];
                for (t, score) in scores.iter_mut().enumerate() {
                    let kh = &keys[t * c + h * hd..t * c + h * hd + hd];
                    let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                    *score = dot * scale;
                }
                softmax(&mut scores);
                let oh = &mut atty[h * hd..h * hd + hd];
                for (t, &a) in scores.iter().enumerate() {
                    let vh = &vals[t * c + h * hd..t * c + h * hd + hd];
                    for (o, &vi) in oh.iter_mut().zip(vh) {
                        *o += a * vi;
                    }
                }
            }

            let mut attproj = vec![0.0f32; c];
            lw.attn_output.matvec(&atty, &mut attproj);
            for (xi, &a) in x.iter_mut().zip(&attproj) {
                *xi += a;
            }

            // --- feed-forward (SwiGLU) block ---
            rmsnorm(&x, lw.ffn_norm, eps, &mut normed);
            let f = cfg.n_ff;
            let mut gate = vec![0.0f32; f];
            let mut up = vec![0.0f32; f];
            lw.ffn_gate.matvec(&normed, &mut gate);
            lw.ffn_up.matvec(&normed, &mut up);
            let mut glu = vec![0.0f32; f];
            swiglu(&gate, &up, &mut glu);
            let mut down = vec![0.0f32; c];
            lw.ffn_down.matvec(&glu, &mut down);
            for (xi, &d) in x.iter_mut().zip(&down) {
                *xi += d;
            }
        }

        // Final norm + output projection -> logits.
        let mut xf = vec![0.0f32; c];
        rmsnorm(&x, w.output_norm, eps, &mut xf);
        let mut logits = vec![0.0f32; cfg.vocab_size];
        w.output.matvec(&xf, &mut logits);
        logits
    }

    /// Reset the KV cache to start a fresh sequence.
    pub fn reset(&mut self) {
        self.kv.clear();
    }
}
