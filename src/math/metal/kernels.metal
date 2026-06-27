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

// M7.1: out[row] = <row of w, x>, F32 weights row-major as [rows, cols].
// One thread per output row; grid width == rows (dispatch_threads => no overshoot).
kernel void matvec_f32(device const float* w    [[buffer(0)]],
                       device const float* x    [[buffer(1)]],
                       device float*       out  [[buffer(2)]],
                       constant uint&      cols [[buffer(3)]],
                       uint                row  [[thread_position_in_grid]]) {
    device const float* wr = w + (uint)row * cols;
    float acc = 0.0f;
    for (uint k = 0; k < cols; ++k) {
        acc += wr[k] * x[k];
    }
    out[row] = acc;
}

// Decode the 2-byte little-endian f16 scale at bp[0..2] (matches f16::from_le_bytes).
static inline float block_scale(device const uchar* bp) {
    ushort bits = (ushort)bp[0] | ((ushort)bp[1] << 8);
    return (float)as_type<half>(bits);
}

// M7.3: out[row] = <dequant(row of w), x>, weights stored as Q8_0 blocks
// (f16 d + 32 i8, 34 bytes/block; x[i] = d * q[i]). Matches dtype.rs.
kernel void matvec_q8_0(device const uchar* w    [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant uint&      cols [[buffer(3)]],
                        uint                row  [[thread_position_in_grid]]) {
    const uint QK = 32, BB = 34;
    uint nblocks = cols / QK;
    device const uchar* rp = w + (uint)row * nblocks * BB;
    float acc = 0.0f;
    for (uint b = 0; b < nblocks; ++b) {
        device const uchar* bp = rp + b * BB;
        float d = block_scale(bp);
        device const float* xb = x + b * QK;
        for (uint j = 0; j < QK; ++j) {
            int q = (int)bp[2 + j];            // explicit signed i8 (don't trust char)
            if (q > 127) q -= 256;
            acc += d * (float)q * xb[j];
        }
    }
    out[row] = acc;
}

// M7.3: Q4_0 blocks (f16 d + 16 packed bytes, 18 bytes/block; low nibble at j,
// high nibble at j+16, both minus 8). Matches dtype.rs.
kernel void matvec_q4_0(device const uchar* w    [[buffer(0)]],
                        device const float* x    [[buffer(1)]],
                        device float*       out  [[buffer(2)]],
                        constant uint&      cols [[buffer(3)]],
                        uint                row  [[thread_position_in_grid]]) {
    const uint QK = 32, BB = 18, HALF = 16;
    uint nblocks = cols / QK;
    device const uchar* rp = w + (uint)row * nblocks * BB;
    float acc = 0.0f;
    for (uint b = 0; b < nblocks; ++b) {
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
    out[row] = acc;
}
