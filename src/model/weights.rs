//! Borrowed views of model weights (zero-copy into the GGUF mmap). Owner: lead.
//!
//! Matrix weights are wrapped in `QTensor`, which keeps them in their on-disk
//! dtype (F32 or block-quantized) and dequantizes on demand. RMSNorm weights are
//! 1-D and always F32, so they stay plain `&[f32]`.
//!
//! Tensor names emitted by Hephaistos:
//!   token_embd.weight, output_norm.weight, output.weight
//!   blk.{l}.attn_norm.weight, attn_q/attn_k/attn_v/attn_output.weight,
//!   blk.{l}.ffn_norm.weight, ffn_gate/ffn_up/ffn_down.weight

use anyhow::Result;

use crate::gguf::{GgmlType, GgufFile};
use crate::math::{matmul, quant};
use crate::model::Config;

/// A 2-D weight `[rows, cols]` (row-major, `cols` = input features), in whatever
/// dtype it was stored as.
pub struct QTensor<'a> {
    bytes: &'a [u8],
    dtype: GgmlType,
    rows: usize,
    cols: usize,
}

impl<'a> QTensor<'a> {
    fn bind(g: &'a GgufFile, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let (bytes, dtype) = g.tensor_raw(name)?;
        Ok(QTensor { bytes, dtype, rows, cols })
    }

    /// `out[m] = <row m, x>`, dequantizing on the fly for quantized dtypes.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        match self.dtype {
            GgmlType::F32 => {
                let w: &[f32] = bytemuck::cast_slice(self.bytes);
                matmul::matvec(w, x, out, self.rows, self.cols);
            }
            _ => quant::matvec(self.bytes, self.dtype, x, out, self.rows, self.cols),
        }
    }

    /// Dequantize a single row into `out` (length `cols`). Used for the embedding.
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let row_bytes = (self.cols / self.dtype.block_elems()) * self.dtype.block_bytes();
        self.dtype
            .dequantize(&self.bytes[r * row_bytes..(r + 1) * row_bytes], out);
    }
}

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub attn_q: QTensor<'a>,
    pub attn_k: QTensor<'a>,
    pub attn_v: QTensor<'a>,
    pub attn_output: QTensor<'a>,
    pub ffn_norm: &'a [f32],
    pub ffn_gate: QTensor<'a>,
    pub ffn_up: QTensor<'a>,
    pub ffn_down: QTensor<'a>,
}

pub struct Weights<'a> {
    pub token_embd: QTensor<'a>,
    pub output_norm: &'a [f32],
    pub output: QTensor<'a>,
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> Weights<'a> {
    pub fn from_gguf(g: &'a GgufFile, cfg: &Config) -> Result<Self> {
        let (c, ff, v, kv) = (cfg.n_embd, cfg.n_ff, cfg.vocab_size, cfg.kv_dim());
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            layers.push(LayerWeights {
                attn_norm: g.tensor_f32(&format!("blk.{l}.attn_norm.weight"))?,
                attn_q: QTensor::bind(g, &format!("blk.{l}.attn_q.weight"), c, c)?,
                attn_k: QTensor::bind(g, &format!("blk.{l}.attn_k.weight"), kv, c)?,
                attn_v: QTensor::bind(g, &format!("blk.{l}.attn_v.weight"), kv, c)?,
                attn_output: QTensor::bind(g, &format!("blk.{l}.attn_output.weight"), c, c)?,
                ffn_norm: g.tensor_f32(&format!("blk.{l}.ffn_norm.weight"))?,
                ffn_gate: QTensor::bind(g, &format!("blk.{l}.ffn_gate.weight"), ff, c)?,
                ffn_up: QTensor::bind(g, &format!("blk.{l}.ffn_up.weight"), ff, c)?,
                ffn_down: QTensor::bind(g, &format!("blk.{l}.ffn_down.weight"), c, ff)?,
            });
        }
        Ok(Weights {
            token_embd: QTensor::bind(g, "token_embd.weight", v, c)?,
            output_norm: g.tensor_f32("output_norm.weight")?,
            output: QTensor::bind(g, "output.weight", v, c)?,
            layers,
        })
    }
}
