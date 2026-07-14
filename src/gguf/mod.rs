//! GGUF v3 container: header, typed metadata key/values, tensor index, and a
//! 32-byte-aligned tensor data section. This mirrors the *writer* in
//! `Hephaistos/src/gguf.rs`; here we only read.
//!
//! Byte layout (little-endian) we must parse, per the Hephaistos writer:
//!   "GGUF" magic, u32 version (=3), u64 tensor_count, u64 kv_count
//!   kv_count × (string key, u32 value-type tag, value)
//!   tensor_count × (string name, u32 n_dims, n_dims×u64 dims, u32 type, u64 offset)
//!   padding to ALIGNMENT (32), then the data section; each tensor's `offset`
//!   is relative to the data-section start and 32-byte aligned.
//!
//! Metadata value-type tags: UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8,
//! ARRAY=9 (array carries an inner type tag then a u64 length).
//! A GGUF string is a u64 length followed by that many UTF-8 bytes.

pub mod dtype;
pub mod reader;

pub use dtype::GgmlType;
pub use reader::{GgufFile, TensorInfo};

/// A parsed GGUF metadata value. Covers the variants Hephaistos emits plus the
/// `ArrF32` scores array that SentencePiece (llama-kind) tokenizers carry.
#[derive(Clone, Debug)]
pub enum MetaValue {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
}
