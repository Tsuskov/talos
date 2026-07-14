//! Matrix-vector product against a block-quantized weight matrix (M4/M5).
//!
//! Weights stay quantized in memory. Each output row is dotted with `x` one
//! 32-element block at a time: a block is dequantized into a small stack buffer
//! and immediately consumed by the SIMD dot, so there is no per-row heap
//! allocation and no full-row dequant pass. Parallel over output rows.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::gguf::dtype::QK_K;
use crate::gguf::GgmlType;
use crate::math::matmul::dot;

/// `out[m] = <dequant(row m), x>` for a quantized weight `[rows, cols]` stored
/// row-major as block-quantized bytes. `cols` must be a multiple of the block
/// size. F32 should use `matmul::matvec` instead (no dequant needed).
pub fn matvec(bytes: &[u8], dtype: GgmlType, x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    let bb = dtype.block_bytes();
    let be = dtype.block_elems();
    let row_bytes = (cols / be) * bb;
    debug_assert_eq!(bytes.len(), rows * row_bytes);
    debug_assert_eq!(cols % be, 0);

    // wasm-SIMD: fused dequant+dot for Q4_0, without the stack round-trip
    // through a dequantized block buffer (M9.1).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dtype == GgmlType::Q4_0 {
        out.iter_mut().enumerate().for_each(|(m, o)| {
            *o = wasm_simd::dot_q4_row(&bytes[m * row_bytes..(m + 1) * row_bytes], x, bb);
        });
        return;
    }

    let per_row = |(m, o): (usize, &mut f32)| {
        let row = &bytes[m * row_bytes..(m + 1) * row_bytes];
        let mut sum = 0.0f32;
        let mut buf = [0.0f32; QK_K]; // sized for the largest block (Q6_K)
        let block = &mut buf[..be];
        for (b, xb) in row.chunks_exact(bb).zip(x.chunks_exact(be)) {
            dtype.dequantize(b, block);
            sum += dot(block, xb);
        }
        *o = sum;
    };
    #[cfg(not(target_arch = "wasm32"))]
    out.par_iter_mut().enumerate().for_each(per_row);
    // wasm has no threads: same kernel, sequential over output rows.
    #[cfg(target_arch = "wasm32")]
    out.iter_mut().enumerate().for_each(per_row);
}

/// Q4_0-Zeilen-Dot direkt auf den gepackten Nibbles (simd128). Ein Block sind
/// f16-Skala d + 16 Bytes; Nibble j low = Element j, high = Element j+16,
/// Wert = d · (Nibble − 8) — exakt `GgmlType::dequantize`, nur ohne den Umweg
/// über einen dequantisierten Blockpuffer.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use core::arch::wasm32::*;

    use half::f16;

    use crate::gguf::dtype::QK;

    #[inline]
    pub fn dot_q4_row(row: &[u8], x: &[f32], bb: usize) -> f32 {
        let mut acc = f32x4_splat(0.0);
        for (b, xb) in row.chunks_exact(bb).zip(x.chunks_exact(QK)) {
            let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
            // 16 gepackte Bytes → zwei u8x16 mit den Low-/High-Nibbles.
            let qs = unsafe { v128_load(b[2..].as_ptr() as *const v128) };
            let lo = v128_and(qs, u8x16_splat(0x0F));
            let hi = u8x16_shr(qs, 4);
            // Nibbles stufenweise auf f32x4-Quads weiten: Elemente 0..16 aus
            // lo, 16..32 aus hi — dieselbe Reihenfolge wie `dequantize`.
            let mut block = f32x4_splat(0.0);
            for (q16, xq) in [
                (u16x8_extend_low_u8x16(lo), &xb[0..]),
                (u16x8_extend_high_u8x16(lo), &xb[8..]),
                (u16x8_extend_low_u8x16(hi), &xb[16..]),
                (u16x8_extend_high_u8x16(hi), &xb[24..]),
            ] {
                let qa = f32x4_convert_u32x4(u32x4_extend_low_u16x8(q16));
                let qb = f32x4_convert_u32x4(u32x4_extend_high_u16x8(q16));
                let (xa, xb4) = unsafe {
                    let p = xq.as_ptr();
                    (v128_load(p as *const v128), v128_load(p.add(4) as *const v128))
                };
                let eight = f32x4_splat(8.0);
                block = f32x4_add(block, f32x4_mul(f32x4_sub(qa, eight), xa));
                block = f32x4_add(block, f32x4_mul(f32x4_sub(qb, eight), xb4));
            }
            acc = f32x4_add(acc, f32x4_mul(block, f32x4_splat(d)));
        }
        f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc)
    }
}
