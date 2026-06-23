//! Matrix-vector product against a block-quantized weight matrix (M4).
//!
//! Weights stay quantized in memory; each row is dequantized on the fly into a
//! small scratch buffer and dotted with `x`. Parallel over output rows.

use rayon::prelude::*;

use crate::gguf::GgmlType;

/// `out[m] = <dequant(row m), x>` for a quantized weight `[rows, cols]` stored
/// row-major as block-quantized bytes. `cols` must be a multiple of the dtype's
/// block size. F32 should use `matmul::matvec` instead (no dequant needed).
pub fn matvec(bytes: &[u8], dtype: GgmlType, x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(cols % dtype.block_elems(), 0);
    let row_bytes = (cols / dtype.block_elems()) * dtype.block_bytes();
    debug_assert_eq!(bytes.len(), rows * row_bytes);

    out.par_iter_mut().enumerate().for_each(|(m, o)| {
        let mut row = vec![0.0f32; cols];
        dtype.dequantize(&bytes[m * row_bytes..(m + 1) * row_bytes], &mut row);
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    });
}
