//! Per-layer key/value cache. Owner: lead (M2).
//!
//! Training attends over a full `[B, T]` block every step; inference attends
//! position `pos` against a growing cache. Layout per layer: `[context, n_head_kv
//! * head_dim]`, appended one row per decode step.

pub struct KvCache {
    // n_layer, n_head_kv, head_dim, context; flat k/v buffers per layer; len.
    _placeholder: std::marker::PhantomData<Vec<f32>>,
}

impl KvCache {
    pub fn new(_n_layer: usize, _n_head_kv: usize, _head_dim: usize, _context: usize) -> Self {
        todo!()
    }

    /// Append this step's key/value rows for `layer` (each `n_head_kv*head_dim`).
    pub fn append(&mut self, _layer: usize, _k: &[f32], _v: &[f32]) {
        todo!()
    }

    /// Cached keys for `layer`: `[len * n_head_kv * head_dim]`.
    pub fn keys(&self, _layer: usize) -> &[f32] {
        todo!()
    }
    pub fn values(&self, _layer: usize) -> &[f32] {
        todo!()
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        todo!()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        todo!()
    }
}
