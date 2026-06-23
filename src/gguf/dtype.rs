//! GGML tensor dtypes. Talos supports F32 plus the two simplest block-quantized
//! formats, Q8_0 and Q4_0. Both quantize the contiguous (row/`cols`) dimension
//! in blocks of 32 elements, each block carrying its own f16 scale `d`:
//!   Q8_0 block: f16 d + 32 × i8       => 34 bytes, x[i] = d · q[i]
//!   Q4_0 block: f16 d + 16 × packed   => 18 bytes, x[i] = d · (nibble[i] − 8)

use half::f16;

pub const QK: usize = 32; // elements per quantized block

/// A GGML tensor element type, as stored in each tensor info's type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    Q8_0,
    Q4_0,
}

impl GgmlType {
    /// Map the GGUF tensor type tag to a `GgmlType`. F32 = 0, Q4_0 = 2, Q8_0 = 8.
    /// Returns `None` for tags Talos does not (yet) support.
    pub fn from_u32(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(GgmlType::F32),
            8 => Some(GgmlType::Q8_0),
            2 => Some(GgmlType::Q4_0),
            _ => None,
        }
    }

    /// Number of elements per quantization block. F32 = 1 (unquantized).
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32 => 1,
            GgmlType::Q8_0 | GgmlType::Q4_0 => QK,
        }
    }

    /// Number of bytes per quantization block. F32 = 4.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::Q8_0 => 2 + QK,     // f16 scale + 32 × i8
            GgmlType::Q4_0 => 2 + QK / 2, // f16 scale + 32 × 4-bit
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
