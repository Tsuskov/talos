//! Borrowed views of model weights (zero-copy into the GGUF mmap). Owner: lead.
//!
//! Tensor names emitted by Hephaistos:
//!   token_embd.weight, output_norm.weight, output.weight
//!   blk.{l}.attn_norm.weight, attn_q/attn_k/attn_v/attn_output.weight,
//!   blk.{l}.ffn_norm.weight, ffn_gate/ffn_up/ffn_down.weight

use anyhow::Result;

use crate::gguf::GgufFile;
use crate::model::Config;

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub attn_q: &'a [f32],
    pub attn_k: &'a [f32],
    pub attn_v: &'a [f32],
    pub attn_output: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub ffn_gate: &'a [f32],
    pub ffn_up: &'a [f32],
    pub ffn_down: &'a [f32],
}

pub struct Weights<'a> {
    pub token_embd: &'a [f32],
    pub output_norm: &'a [f32],
    pub output: &'a [f32],
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> Weights<'a> {
    pub fn from_gguf(_g: &'a GgufFile, _cfg: &Config) -> Result<Self> {
        todo!("M2: bind tensor_f32 views by name for each layer")
    }
}
