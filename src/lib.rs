//! Talos — a minimal LLM inference engine.
//!
//! Loads a GGUF model (as forged by the sister project `Hephaistos`), runs a
//! Llama-style forward pass with a KV cache, and samples tokens.
//!
//! Module ownership (see BUILD.md):
//!   gguf      — wave 1, "gguf-reader" agent
//!   tokenizer — wave 1, "tokenizer" agent
//!   math      — wave 1, "math-ops" agent
//!   model     — lead (M2), depends on the three above
//!   kv_cache  — lead (M2)
//!   sample    — lead (M3)

pub mod gguf;
pub mod kv_cache;
pub mod math;
pub mod model;
pub mod sample;
pub mod tokenizer;
