//! Model hyperparameters, read from GGUF `llama.*` metadata. Owner: lead.
//!
//! Keys written by Hephaistos export_gguf:
//!   llama.context_length, llama.embedding_length, llama.block_count,
//!   llama.feed_forward_length, llama.attention.head_count,
//!   llama.attention.head_count_kv, llama.attention.layer_norm_rms_epsilon,
//!   llama.rope.dimension_count, llama.rope.freq_base

use anyhow::Result;

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
    /// tokenizer.ggml.tokens length / token_embd rows.)
    pub fn from_gguf(_g: &GgufFile) -> Result<Self> {
        todo!("M2: read llama.* keys; vocab from tokens array length")
    }
}
