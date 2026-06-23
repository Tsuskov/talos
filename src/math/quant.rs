//! Matrix-vector product against a block-quantized weight matrix (M4/M5).
//!
//! Weights stay quantized in memory. Each output row is dotted with `x` one
//! 32-element block at a time: a block is dequantized into a small stack buffer
//! and immediately consumed by the SIMD dot, so there is no per-row heap
//! allocation and no full-row dequant pass. Parallel over output rows.

use rayon::prelude::*;

use crate::gguf::dtype::QK;
use crate::gguf::GgmlType;
use crate::math::matmul::dot;

/// `out[m] = <dequant(row m), x>` for a quantized weight `[rows, cols]` stored
/// row-major as block-quantized bytes. `cols` must be a multiple of the block
/// size. F32 should use `matmul::matvec` instead (no dequant needed).
pub fn matvec(bytes: &[u8], dtype: GgmlType, x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    let bb = dtype.block_bytes();
    let row_bytes = (cols / QK) * bb;
    debug_assert_eq!(bytes.len(), rows * row_bytes);
    debug_assert_eq!(cols % QK, 0);

    out.par_iter_mut().enumerate().for_each(|(m, o)| {
        let row = &bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut sum = 0.0f32;
        let mut block = [0.0f32; QK];
        for (b, xb) in row.chunks_exact(bb).zip(x.chunks_exact(QK)) {
            dtype.dequantize(b, &mut block);
            sum += dot(&block, xb);
        }
        *o = sum;
    });
}
