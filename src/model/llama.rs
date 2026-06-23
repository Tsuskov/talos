//! The Llama-style forward pass. Owner: lead (M2 — the keystone).
//!
//! Per token at position `pos`:
//!   embed -> for each layer { rmsnorm, q/k/v projections, RoPE(q,k),
//!   append k,v to cache, masked attention over cache, attn_output,
//!   residual, rmsnorm, SwiGLU MLP, residual } -> final rmsnorm -> output proj.
//!
//! Must match Hephaistos numerically (tests/parity.rs).

use anyhow::Result;
use std::path::Path;

use crate::gguf::GgufFile;
use crate::kv_cache::KvCache;
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
    pub fn load(_path: &Path) -> Result<Self> {
        todo!("M2")
    }

    /// Run one decode step for `token` at sequence position `pos`, returning
    /// logits over the vocabulary. Updates the KV cache.
    pub fn forward(&mut self, _token: u32, _pos: usize) -> Vec<f32> {
        todo!("M2 — borrow Weights::from_gguf(&self.gguf, &self.cfg) internally")
    }

    /// Reset the KV cache to start a fresh sequence.
    pub fn reset(&mut self) {
        todo!()
    }
}
