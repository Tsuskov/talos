//! Elementwise / reduction kernels (M2 building blocks). Owner: "math-ops" agent.

/// RMSNorm: `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
/// `x`, `weight`, `out` all have length = n_embd.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place softmax over `x`.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU / swish: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU feed-forward activation: `out[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Apply rotary position embedding in-place to a `[n_head * head_dim]` vector
/// at sequence position `pos`.
///
/// IMPORTANT — convention: Hephaistos's `export_gguf` permutes q/k weights
/// ("HF rotate-half -> GGUF llama interleaved layout"), so the loaded weights
/// expect RoPE applied to **adjacent pairs** `(x[2i], x[2i+1])` within each
/// head (interleaved), using `theta_i = pos * freq_base^(-2i/head_dim)`. This
/// pairing must match the weights or the M2 parity test will fail — see
/// BUILD.md "RoPE convention".
pub fn rope(vec: &mut [f32], pos: usize, n_head: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    for h in 0..n_head {
        let base = h * head_dim;
        for i in 0..half {
            let theta = pos as f32 * freq_base.powf(-2.0 * i as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let a = vec[base + 2 * i];
            let b = vec[base + 2 * i + 1];
            vec[base + 2 * i] = a * cos - b * sin;
            vec[base + 2 * i + 1] = a * sin + b * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    #[test]
    fn rmsnorm_hand() {
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 0.
        // mean(x^2) = (1 + 4 + 9) / 3 = 14/3, scale = 1/sqrt(14/3).
        let x = [1.0, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        let mut out = [0.0; 3];
        rmsnorm(&x, &w, 0.0, &mut out);
        let scale = 1.0 / (14.0f32 / 3.0).sqrt();
        close(out[0], 1.0 * scale);
        close(out[1], 2.0 * scale);
        close(out[2], 3.0 * scale);
    }

    #[test]
    fn rmsnorm_weighted() {
        let x = [2.0, 0.0];
        let w = [3.0, 5.0];
        let mut out = [0.0; 2];
        rmsnorm(&x, &w, 0.0, &mut out);
        // mean(x^2) = 2, scale = 1/sqrt(2).
        let scale = 1.0 / 2.0f32.sqrt();
        close(out[0], 2.0 * scale * 3.0);
        close(out[1], 0.0);
    }

    #[test]
    fn softmax_sums_to_one_and_known() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        close(x.iter().sum::<f32>(), 1.0);
        // known: exp shifted by max=3 -> [e^-2, e^-1, 1] / sum.
        let denom = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
        close(x[0], (-2.0f32).exp() / denom);
        close(x[1], (-1.0f32).exp() / denom);
        close(x[2], 1.0 / denom);
    }

    #[test]
    fn softmax_uniform() {
        let mut x = [5.0, 5.0, 5.0, 5.0];
        softmax(&mut x);
        for &v in &x {
            close(v, 0.25);
        }
    }

    #[test]
    fn silu_zero() {
        close(silu(0.0), 0.0);
        // silu(x) = x * sigmoid(x); silu(2) = 2 * sigmoid(2).
        close(silu(2.0), 2.0 / (1.0 + (-2.0f32).exp()));
    }

    #[test]
    fn swiglu_small() {
        let gate = [0.0, 2.0];
        let up = [10.0, 3.0];
        let mut out = [0.0; 2];
        swiglu(&gate, &up, &mut out);
        close(out[0], 0.0); // silu(0) * 10 = 0
        close(out[1], silu(2.0) * 3.0);
    }

    #[test]
    fn rope_preserves_pair_norm() {
        let n_head = 2;
        let head_dim = 4;
        let mut v: Vec<f32> = (0..n_head * head_dim).map(|i| (i as f32) + 1.0).collect();
        let orig = v.clone();
        rope(&mut v, 7, n_head, head_dim, 10000.0);
        // Each adjacent pair (2i, 2i+1) within each head preserves its norm.
        for h in 0..n_head {
            for i in 0..head_dim / 2 {
                let b = h * head_dim;
                let n0 = orig[b + 2 * i].powi(2) + orig[b + 2 * i + 1].powi(2);
                let n1 = v[b + 2 * i].powi(2) + v[b + 2 * i + 1].powi(2);
                close(n1, n0);
            }
        }
    }

    #[test]
    fn rope_pos_zero_identity() {
        let n_head = 3;
        let head_dim = 6;
        let mut v: Vec<f32> = (0..n_head * head_dim).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let orig = v.clone();
        rope(&mut v, 0, n_head, head_dim, 10000.0);
        for (a, b) in v.iter().zip(&orig) {
            close(*a, *b);
        }
    }
}
