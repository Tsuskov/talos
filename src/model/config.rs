//! Model hyperparameters, read from GGUF `llama.*` metadata. Owner: lead.
//!
//! Keys written by Hephaistos export_gguf:
//!   llama.context_length, llama.embedding_length, llama.block_count,
//!   llama.feed_forward_length, llama.attention.head_count,
//!   llama.attention.head_count_kv, llama.attention.layer_norm_rms_epsilon,
//!   llama.rope.dimension_count, llama.rope.freq_base

use anyhow::{anyhow, Result};

use crate::gguf::GgufFile;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Read all hyperparameters from a loaded GGUF file. (vocab_size from the
    /// tokenizer.ggml.tokens length.)
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.get_str("general.architecture").unwrap_or("");
        if arch != "llama" {
            return Err(anyhow!("unsupported architecture {arch:?} (expected \"llama\")"));
        }

        let u = |k: &str| g.get_u32(k).ok_or_else(|| anyhow!("missing metadata key {k}"));

        let vocab_size = g
            .get_arr_str("tokenizer.ggml.tokens")
            .map(|t| t.len())
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens (needed for vocab size)"))?;

        Ok(Config {
            n_layer: u("llama.block_count")? as usize,
            n_head: u("llama.attention.head_count")? as usize,
            n_head_kv: u("llama.attention.head_count_kv")? as usize,
            n_embd: u("llama.embedding_length")? as usize,
            n_ff: u("llama.feed_forward_length")? as usize,
            context_length: u("llama.context_length")? as usize,
            vocab_size,
            rms_eps: g
                .get_f32("llama.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_dim: u("llama.rope.dimension_count")? as usize,
            rope_freq_base: g.get_f32("llama.rope.freq_base").unwrap_or(10000.0),
        })
    }
}
