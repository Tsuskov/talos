//! Numeric kernels for inference. Owner: "math-ops" agent (M2 building blocks).
//!
//! These are single-token (1-D) kernels — inference runs one position at a time,
//! not a `[B, T]` batch like training. Keep them allocation-free where possible
//! (write into caller-provided `out` slices); the forward loop calls them in a
//! hot path.

pub mod matmul;
pub mod ops;
pub mod quant;
