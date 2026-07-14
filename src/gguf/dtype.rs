//! GGML tensor dtypes. Talos supports F32, the two simplest block-quantized
//! formats (Q8_0, Q4_0), and Q6_K. Q8_0/Q4_0 quantize the contiguous
//! (row/`cols`) dimension in blocks of 32 elements, each carrying its own f16
//! scale `d`:
//!   Q8_0 block: f16 d + 32 × i8       => 34 bytes, x[i] = d · q[i]
//!   Q4_0 block: f16 d + 16 × packed   => 18 bytes, x[i] = d · (nibble[i] − 8)
//! Q6_K is a k-quant super-block of 256 elements (6-bit quants + per-16 int8
//! sub-scales + one f16 super-scale); llama.cpp stores the output projection of
//! otherwise-Q4_0 models in it, so a real Mistral GGUF needs it.

use half::f16;

pub const QK: usize = 32; // elements per Q8_0/Q4_0 block
pub const QK_K: usize = 256; // elements per k-quant super-block

impl GgmlType {
    /// Number of bytes in one Q6_K super-block: ql[128] + qh[64] + scales[16] + d.
    const Q6K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210
}

/// A GGML tensor element type, as stored in each tensor info's type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    Q8_0,
    Q4_0,
    Q6K,
}

impl GgmlType {
    /// Map the GGUF tensor type tag to a `GgmlType`. F32 = 0, Q4_0 = 2, Q8_0 = 8,
    /// Q6_K = 14. Returns `None` for tags Talos does not (yet) support.
    pub fn from_u32(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(GgmlType::F32),
            8 => Some(GgmlType::Q8_0),
            2 => Some(GgmlType::Q4_0),
            14 => Some(GgmlType::Q6K),
            _ => None,
        }
    }

    /// Number of elements per quantization block. F32 = 1 (unquantized).
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32 => 1,
            GgmlType::Q8_0 | GgmlType::Q4_0 => QK,
            GgmlType::Q6K => QK_K,
        }
    }

    /// Number of bytes per quantization block. F32 = 4.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::Q8_0 => 2 + QK,     // f16 scale + 32 × i8
            GgmlType::Q4_0 => 2 + QK / 2, // f16 scale + 32 × 4-bit
            GgmlType::Q6K => Self::Q6K_BYTES,
        }
    }

    /// Dequantize `raw` (exactly `out.len()/block_elems` blocks) into `out`.
    /// `out.len()` must be a multiple of `block_elems`.
    pub fn dequantize(self, raw: &[u8], out: &mut [f32]) {
        match self {
            GgmlType::F32 => {
                out.copy_from_slice(bytemuck::cast_slice(raw));
            }
            GgmlType::Q8_0 => {
                let bb = self.block_bytes();
                for (blk, chunk) in out.chunks_mut(QK).enumerate() {
                    let b = &raw[blk * bb..blk * bb + bb];
                    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                    for (o, &q) in chunk.iter_mut().zip(&b[2..]) {
                        *o = d * (q as i8) as f32;
                    }
                }
            }
            GgmlType::Q4_0 => {
                let bb = self.block_bytes();
                for (blk, chunk) in out.chunks_mut(QK).enumerate() {
                    let b = &raw[blk * bb..blk * bb + bb];
                    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                    let qs = &b[2..];
                    for j in 0..QK / 2 {
                        let lo = (qs[j] & 0x0F) as i32 - 8;
                        let hi = (qs[j] >> 4) as i32 - 8;
                        chunk[j] = d * lo as f32;
                        chunk[j + QK / 2] = d * hi as f32;
                    }
                }
            }
            GgmlType::Q6K => {
                // Super-block layout: ql[128] (low 4 bits), qh[64] (high 2 bits),
                // scales[16] (int8), d (f16). Two halves of 128 elements each use
                // 8 sub-scales; the 6-bit quant is biased by −32. Mirrors ggml's
                // `dequantize_row_q6_K`.
                let bb = self.block_bytes();
                for (sb, chunk) in out.chunks_mut(QK_K).enumerate() {
                    let b = &raw[sb * bb..sb * bb + bb];
                    let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                    let d = f16::from_le_bytes([b[208], b[209]]).to_f32();
                    for n in 0..2 {
                        let (ql, qh, sc) = (&ql[n * 64..], &qh[n * 32..], &sc[n * 8..]);
                        let y = &mut chunk[n * 128..];
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0xF) | ((qh[l] & 0x03) << 4)) as i32 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
                            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block() -> Vec<f32> {
        (0..QK).map(|i| (i as f32 - 16.0) * 0.37 + 0.1).collect()
    }

    #[test]
    fn q8_0_round_trip() {
        let x = sample_block();
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let mut raw = f16::from_f32(d).to_le_bytes().to_vec();
        for &v in &x {
            raw.push((v / d).round().clamp(-128.0, 127.0) as i8 as u8);
        }
        let mut out = vec![0.0f32; QK];
        GgmlType::Q8_0.dequantize(&raw, &mut out);
        for (a, b) in x.iter().zip(&out) {
            assert!((a - b).abs() <= d + 1e-3, "q8 err {a} vs {b}");
        }
    }

    #[test]
    fn q4_0_round_trip() {
        let x = sample_block();
        // signed value of largest magnitude -> d = max / -8 (ggml convention).
        let max = x.iter().copied().fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut raw = f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..QK / 2 {
            let xi0 = ((x[j] * id + 8.5) as i32).min(15) as u8;
            let xi1 = ((x[j + QK / 2] * id + 8.5) as i32).min(15) as u8;
            raw.push(xi0 | (xi1 << 4));
        }
        let mut out = vec![0.0f32; QK];
        GgmlType::Q4_0.dequantize(&raw, &mut out);
        // 4-bit step is |d|; allow one step of error.
        for (a, b) in x.iter().zip(&out) {
            assert!((a - b).abs() <= d.abs() + 1e-3, "q4 err {a} vs {b}");
        }
    }
}
