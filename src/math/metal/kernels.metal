#include <metal_stdlib>
using namespace metal;

// M7.0 gate: trivial element-wise add. Only here to prove the
// Rust -> Metal -> Rust round trip (device, library, pipeline, dispatch,
// shared-memory read-back) before the real matvec kernels land.
kernel void vadd(device const float* a   [[buffer(0)]],
                 device const float* b   [[buffer(1)]],
                 device float*       out [[buffer(2)]],
                 uint                i   [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

// M8.1: out[row] = <row of w, x>, F32 weights row-major as [rows, cols].
// One simdgroup (threadgroup of `W` = simd execution width) per output row: the
// `W` lanes stride over the columns — so at each step adjacent lanes read
// adjacent w[] addresses (coalesced) — then simd_sum reduces the partials.
kernel void matvec_f32(device const float* w    [[buffer(0)]],
                       device const float* x    [[buffer(1)]],
                       device float*       out  [[buffer(2)]],
                       constant uint&      cols [[buffer(3)]],
                       uint  row  [[threadgroup_position_in_grid]],
                       uint  lane [[thread_position_in_threadgroup]],
                       uint  W    [[threads_per_threadgroup]]) {
    device const float* wr = w + (uint)row * cols;
    float acc = 0.0f;
    for (uint k = lane; k < cols; k += W) {
        acc += wr[k] * x[k];
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// Decode the 2-byte little-endian f16 scale at bp[0..2] (matches f16::from_le_bytes).
static inline float block_scale(device const uchar* bp) {
    ushort bits = (ushort)bp[0] | ((ushort)bp[1] << 8);
    return (float)as_type<half>(bits);
}

// M8.1: out[row] = <dequant(row of w), x>, Q8_0 blocks (f16 d + 32 i8,
// 34 bytes/block; x[i] = d * q[i]). Matches dtype.rs. One simdgroup per row;
// lanes split the row's blocks, then simd_sum reduces.
kernel void matvec_q8_0(device const uchar* w    [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant uint&      cols [[buffer(3)]],
                        uint  row  [[threadgroup_position_in_grid]],
                        uint  lane [[thread_position_in_threadgroup]],
                        uint  W    [[threads_per_threadgroup]]) {
    const uint QK = 32, BB = 34;
    uint nblocks = cols / QK;
    device const uchar* rp = w + (uint)row * nblocks * BB;
    float acc = 0.0f;
    for (uint b = lane; b < nblocks; b += W) {
        device const uchar* bp = rp + b * BB;
        float d = block_scale(bp);
        device const float* xb = x + b * QK;
        for (uint j = 0; j < QK; ++j) {
            int q = (int)bp[2 + j];            // explicit signed i8 (don't trust char)
            if (q > 127) q -= 256;
            acc += d * (float)q * xb[j];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// M8.1: Q4_0 blocks (f16 d + 16 packed bytes, 18 bytes/block; low nibble at j,
// high nibble at j+16, both minus 8). Matches dtype.rs. One simdgroup per row;
// lanes split the row's blocks, then simd_sum reduces.
kernel void matvec_q4_0(device const uchar* w    [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant uint&      cols [[buffer(3)]],
                        uint  row  [[threadgroup_position_in_grid]],
                        uint  lane [[thread_position_in_threadgroup]],
                        uint  W    [[threads_per_threadgroup]]) {
    const uint QK = 32, BB = 18, HALF = 16;
    uint nblocks = cols / QK;
    device const uchar* rp = w + (uint)row * nblocks * BB;
    float acc = 0.0f;
    for (uint b = lane; b < nblocks; b += W) {
        device const uchar* bp = rp + b * BB;
        float d = block_scale(bp);
        device const uchar* qs = bp + 2;
        device const float* xb = x + b * QK;
        for (uint j = 0; j < HALF; ++j) {
            int lo = (int)(qs[j] & 0x0F) - 8;
            int hi = (int)(qs[j] >> 4) - 8;
            acc += d * (float)lo * xb[j];
            acc += d * (float)hi * xb[j + HALF];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// M10: Q6_K super-block (256 elements, 210 bytes: ql[128] low 4 bits, qh[64]
// high 2 bits, scales[16] int8, d f16). Two halves of 128 elements, each with 8
// int8 sub-scales; the assembled 6-bit quant is biased by -32. Mirrors
// dtype.rs `dequantize` for Q6_K, accumulating the dot instead of storing.
// One simdgroup per row; lanes split the row's super-blocks, then simd_sum.
kernel void matvec_q6_k(device const uchar* w    [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant uint&      cols [[buffer(3)]],
                        uint  row  [[threadgroup_position_in_grid]],
                        uint  lane [[thread_position_in_threadgroup]],
                        uint  W    [[threads_per_threadgroup]]) {
    const uint QK_K = 256, BB = 210;
    uint nsb = cols / QK_K;
    device const uchar* rp = w + (uint)row * nsb * BB;
    float acc = 0.0f;
    for (uint sb = lane; sb < nsb; sb += W) {
        device const uchar* bp = rp + sb * BB;
        float d = block_scale(bp + 208);
        device const float* xb = x + sb * QK_K;
        for (uint n = 0; n < 2; ++n) {
            device const uchar* ql = bp + n * 64;
            device const uchar* qh = bp + 128 + n * 32;
            device const uchar* sc = bp + 192 + n * 8;
            uint yb = n * 128;
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16;
                int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]       >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                int s1 = (int)sc[is];     if (s1 > 127) s1 -= 256;
                int s2 = (int)sc[is + 2]; if (s2 > 127) s2 -= 256;
                int s3 = (int)sc[is + 4]; if (s3 > 127) s3 -= 256;
                int s4 = (int)sc[is + 6]; if (s4 > 127) s4 -= 256;
                acc += d * (float)s1 * (float)q1 * xb[yb + l];
                acc += d * (float)s2 * (float)q2 * xb[yb + l + 32];
                acc += d * (float)s3 * (float)q3 * xb[yb + l + 64];
                acc += d * (float)s4 * (float)q4 * xb[yb + l + 96];
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// ---- M8.2: the rest of the forward pass, mirroring math::ops + the attention
// in model::llama. One simdgroup-wide threadgroup is used for the reductions
// (rmsnorm, softmax); the elementwise kernels are one thread per element.

// out[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i]. One simdgroup, lanes stride.
kernel void rmsnorm(device const float* x      [[buffer(0)]],
                    device const float* weight [[buffer(1)]],
                    device float*       out    [[buffer(2)]],
                    constant uint&      n      [[buffer(3)]],
                    constant float&     eps    [[buffer(4)]],
                    uint lane [[thread_position_in_threadgroup]],
                    uint W    [[threads_per_threadgroup]]) {
    float ss = 0.0f;
    for (uint i = lane; i < n; i += W) ss += x[i] * x[i];
    ss = simd_sum(ss);
    float scale = rsqrt(ss / (float)n + eps);
    for (uint i = lane; i < n; i += W) out[i] = x[i] * scale * weight[i];
}

// In-place RoPE on a [n_head*head_dim] vector: rotate adjacent pairs
// (vec[2i], vec[2i+1]) within each head. Matches ops::rope (interleaved).
kernel void rope(device float*    vec       [[buffer(0)]],
                 constant uint&   n_head    [[buffer(1)]],
                 constant uint&   head_dim  [[buffer(2)]],
                 constant uint&   pos       [[buffer(3)]],
                 constant float&  freq_base [[buffer(4)]],
                 uint gid [[thread_position_in_grid]]) {
    uint halfd = head_dim / 2;
    uint h = gid / halfd;
    uint i = gid % halfd;
    if (h >= n_head) return;
    uint base = h * head_dim;
    float theta = (float)pos * pow(freq_base, -2.0f * (float)i / (float)head_dim);
    float s = sin(theta), c = cos(theta);
    float a = vec[base + 2 * i];
    float b = vec[base + 2 * i + 1];
    vec[base + 2 * i]     = a * c - b * s;
    vec[base + 2 * i + 1] = a * s + b * c;
}

// scores[h*seq + t] = scale * <q_head h, key_t of kv head h/group>.
// Grid = n_head * seq.
kernel void attn_scores(device const float* q      [[buffer(0)]],
                        device const float* keys   [[buffer(1)]],
                        device float*       scores [[buffer(2)]],
                        constant uint&      hd     [[buffer(3)]],
                        constant uint&      kv_dim [[buffer(4)]],
                        constant uint&      group  [[buffer(5)]],
                        constant uint&      seq    [[buffer(6)]],
                        constant float&     scale  [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    uint h = gid / seq;
    uint t = gid % seq;
    uint kvh = h / group;
    device const float* qh = q + h * hd;
    device const float* kh = keys + t * kv_dim + kvh * hd;
    float d = 0.0f;
    for (uint i = 0; i < hd; i++) d += qh[i] * kh[i];
    scores[h * seq + t] = d * scale;
}

// Stable softmax over each head's `seq` scores, in place. One simdgroup/head.
kernel void attn_softmax(device float*  scores [[buffer(0)]],
                         constant uint& seq    [[buffer(1)]],
                         uint h    [[threadgroup_position_in_grid]],
                         uint lane [[thread_position_in_threadgroup]],
                         uint W    [[threads_per_threadgroup]]) {
    device float* s = scores + h * seq;
    float m = -INFINITY;
    for (uint t = lane; t < seq; t += W) m = max(m, s[t]);
    m = simd_max(m);
    float sum = 0.0f;
    for (uint t = lane; t < seq; t += W) { float e = exp(s[t] - m); s[t] = e; sum += e; }
    sum = simd_sum(sum);
    for (uint t = lane; t < seq; t += W) s[t] /= sum;
}

// out[h*hd + d] = sum_t scores[h,t] * value_t[kvh, d]. Grid = n_head * hd.
kernel void attn_output(device const float* scores [[buffer(0)]],
                        device const float* values [[buffer(1)]],
                        device float*       out    [[buffer(2)]],
                        constant uint&      hd     [[buffer(3)]],
                        constant uint&      kv_dim [[buffer(4)]],
                        constant uint&      group  [[buffer(5)]],
                        constant uint&      seq    [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    uint h = gid / hd;
    uint d = gid % hd;
    uint kvh = h / group;
    device const float* s = scores + h * seq;
    float acc = 0.0f;
    for (uint t = 0; t < seq; t++) acc += s[t] * values[t * kv_dim + kvh * hd + d];
    out[h * hd + d] = acc;
}

// out[i] = silu(gate[i]) * up[i].
kernel void swiglu(device const float* gate [[buffer(0)]],
                   device const float* up   [[buffer(1)]],
                   device float*       out  [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

// x[i] += y[i] (residual add, in place).
kernel void add_inplace(device float*       x [[buffer(0)]],
                        device const float* y [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    x[i] += y[i];
}

// dst[offset + i] = src[i] (append k/v row into the KV cache).
kernel void copy_to(device const float* src    [[buffer(0)]],
                    device float*       dst    [[buffer(1)]],
                    constant uint&      offset [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    dst[offset + i] = src[i];
}
