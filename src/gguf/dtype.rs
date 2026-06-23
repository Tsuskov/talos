//! GGML tensor dtypes. Wave 1 only needs F32 (Hephaistos exports
//! `general.file_type = 0`, ALL_F32). Q8_0 / Q4_0 land in M4.

/// A GGML tensor element type, as stored in each tensor info's type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    Q8_0,
    Q4_0,
}

impl GgmlType {
    /// Map the GGUF tensor type tag to a `GgmlType`. F32 = 0.
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
            // M4: real block sizes (Q8_0/Q4_0 are 32-element blocks).
            GgmlType::Q8_0 | GgmlType::Q4_0 => todo!("quantized block sizes — M4"),
        }
    }

    /// Number of bytes per quantization block. F32 = 4.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::Q8_0 | GgmlType::Q4_0 => todo!("quantized block byte sizes — M4"),
        }
    }
}
