//! Per-layer key/value cache. Owner: lead (M2).
//!
//! Training attends over a full `[B, T]` block every step; inference attends
//! position `pos` against a growing cache. Each layer stores keys and values as
//! flat buffers, one `n_head_kv * head_dim` row appended per decode step.

pub struct KvCache {
    row: usize, // n_head_kv * head_dim, the floats appended per step per layer
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layer: usize, n_head_kv: usize, head_dim: usize, context: usize) -> Self {
        let row = n_head_kv * head_dim;
        let mk = || (0..n_layer).map(|_| Vec::with_capacity(context * row)).collect();
        Self { row, k: mk(), v: mk() }
    }

    /// Append this step's key/value rows for `layer` (each `n_head_kv*head_dim`).
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.row);
        debug_assert_eq!(v.len(), self.row);
        self.k[layer].extend_from_slice(k);
        self.v[layer].extend_from_slice(v);
    }

    /// Cached keys for `layer`: `[len * n_head_kv * head_dim]`.
    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }
    pub fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Number of positions currently cached (read from layer 0).
    pub fn len(&self) -> usize {
        if self.row == 0 {
            0
        } else {
            self.k[0].len() / self.row
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        for buf in self.k.iter_mut().chain(self.v.iter_mut()) {
            buf.clear();
        }
    }
}
