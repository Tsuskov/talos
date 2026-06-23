//! Llama-style model: config, weight handles, and the forward pass.
//! Owner: lead (M2). Depends on `gguf`, `tokenizer`, `math`, `kv_cache`.

pub mod config;
pub mod llama;
pub mod weights;

pub use config::Config;
pub use llama::Model;
pub use weights::Weights;
