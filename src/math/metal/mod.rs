//! GPU matvec backend via Apple Metal (M7). Owner: lead.
//!
//! Mirrors the CPU hot path (`math::matmul` for F32, `math::quant` for
//! block-quantized weights), but runs each output row's dot product on the GPU:
//! decode is matrix-×-vector (one token at a time), so every kernel launches one
//! thread per output row.
//!
//! This is a *correctness* milestone. Only matvec moves to the GPU;
//! rmsnorm/rope/attention stay on the CPU, so each layer pays a CPU<->GPU
//! round-trip and per-call buffer uploads — it is not yet faster than the SIMD
//! CPU path. Keeping weights resident and porting the rest of the forward pass
//! (the actual throughput win) is M8.
//!
//! Verified against the CPU implementations in unit tests on random inputs
//! (`cargo test --features metal`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use metal::{
    CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

thread_local! {
    /// One context per thread that touches the GPU. Inference runs the forward
    /// pass on a single thread, so in practice this is built once.
    static CTX: MetalCtx = MetalCtx::new();
}

struct MetalCtx {
    device: Device,
    queue: metal::CommandQueue,
    library: Library,
    /// Compiled pipelines, keyed by kernel name (built on first use).
    pipelines: RefCell<HashMap<&'static str, ComputePipelineState>>,
    /// M8.0: weight buffers uploaded once and reused across tokens, keyed by
    /// tensor name. Stores byte length too, so a reload with a different-sized
    /// tensor of the same name re-uploads rather than serving a stale buffer.
    /// Assumes one model per process (names are unique within a model).
    weights: RefCell<HashMap<String, (usize, metal::Buffer)>>,
}

impl MetalCtx {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device available");
        let queue = device.new_command_queue();
        let src = include_str!("kernels.metal");
        let library = device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("compile kernels.metal");
        MetalCtx {
            device,
            queue,
            library,
            pipelines: RefCell::new(HashMap::new()),
            weights: RefCell::new(HashMap::new()),
        }
    }

    /// Return the resident GPU buffer for weight tensor `key`, uploading it on
    /// first use (or if a different-sized tensor was previously cached there).
    fn resident_weight(&self, key: &str, bytes: &[u8]) -> metal::Buffer {
        if let Some((len, buf)) = self.weights.borrow().get(key) {
            if *len == bytes.len() {
                return buf.clone();
            }
        }
        let buf = self.buffer_from(bytes);
        self.weights.borrow_mut().insert(key.to_string(), (bytes.len(), buf.clone()));
        buf
    }

    fn pipeline(&self, name: &'static str) -> ComputePipelineState {
        if let Some(p) = self.pipelines.borrow().get(name) {
            return p.clone();
        }
        let func = self.library.get_function(name, None).expect("kernel function");
        let pipe = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");
        self.pipelines.borrow_mut().insert(name, pipe.clone());
        pipe
    }

    /// Upload bytes into a shared (unified-memory) buffer the GPU can read.
    fn buffer_from(&self, bytes: &[u8]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            bytes.as_ptr() as *const c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// A zeroed output buffer of `len` f32s in shared memory.
    fn out_buffer(&self, len: usize) -> metal::Buffer {
        self.device
            .new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared)
    }
}

/// Read `len` f32s back out of a shared buffer.
fn read_f32(buf: &metal::Buffer, len: usize) -> &[f32] {
    // SAFETY: shared-storage buffer holds `len` contiguous f32s written by the
    // kernel; the borrow is bounded by the caller's use before the buffer drops.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len) }
}

/// Launch a 1-D grid of `n` threads for `pipeline`, capping the threadgroup at
/// the pipeline's max.
fn dispatch_1d(ctx: &MetalCtx, pipeline: &ComputePipelineState, n: usize, setup: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    setup(enc);
    let tg = pipeline.max_total_threads_per_threadgroup().min(n as u64).max(1);
    enc.dispatch_threads(
        MTLSize { width: n as u64, height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// M7.0 gate: `out = a + b` on the GPU. Exists only to prove the round trip.
pub fn vadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    CTX.with(|ctx| {
        let pipe = ctx.pipeline("vadd");
        let abuf = ctx.buffer_from(bytemuck::cast_slice(a));
        let bbuf = ctx.buffer_from(bytemuck::cast_slice(b));
        let obuf = ctx.out_buffer(n);
        dispatch_1d(ctx, &pipe, n, |enc| {
            enc.set_buffer(0, Some(&abuf), 0);
            enc.set_buffer(1, Some(&bbuf), 0);
            enc.set_buffer(2, Some(&obuf), 0);
        });
        out.copy_from_slice(read_f32(&obuf, n));
    });
}

/// The matvec launch itself, given a weight buffer (resident or freshly
/// uploaded). `x`/`out` are small, so they stay per-call buffers.
fn run_matvec(
    ctx: &MetalCtx,
    kernel: &'static str,
    wbuf: &metal::Buffer,
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let pipe = ctx.pipeline(kernel);
    let xbuf = ctx.buffer_from(bytemuck::cast_slice(x));
    let obuf = ctx.out_buffer(rows);
    let cols_u32 = cols as u32;
    // One threadgroup per output row, sized to exactly one simdgroup so the
    // kernel's `simd_sum` reduces the whole group (M8.1).
    let width = pipe.thread_execution_width();
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(wbuf), 0);
    enc.set_buffer(1, Some(&xbuf), 0);
    enc.set_buffer(2, Some(&obuf), 0);
    enc.set_bytes(3, 4, &cols_u32 as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize { width: rows as u64, height: 1, depth: 1 },
        MTLSize { width, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    out.copy_from_slice(read_f32(&obuf, rows));
}

/// Kernel name for a block-quantized dtype.
fn quant_kernel(dtype: crate::gguf::GgmlType) -> &'static str {
    use crate::gguf::GgmlType;
    match dtype {
        GgmlType::Q8_0 => "matvec_q8_0",
        GgmlType::Q4_0 => "matvec_q4_0",
        GgmlType::F32 => unreachable!("F32 uses matvec_f32"),
    }
}

/// M7.1: `out[m] = <row m of w, x>` for an F32 weight `[rows, cols]` (row-major).
/// Uploads the weight each call — used by tests; inference uses the `_resident`
/// variant. Mirrors `math::matmul::matvec`, on the GPU.
pub fn matvec_f32(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    CTX.with(|ctx| {
        let wbuf = ctx.buffer_from(bytemuck::cast_slice(w));
        run_matvec(ctx, "matvec_f32", &wbuf, x, out, rows, cols);
    });
}

/// M7.3: fused dequant + matvec for a block-quantized weight (per-call upload).
/// Mirrors `math::quant::matvec`, on the GPU.
pub fn matvec_quant(
    bytes: &[u8],
    dtype: crate::gguf::GgmlType,
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % dtype.block_elems(), 0);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    CTX.with(|ctx| {
        let wbuf = ctx.buffer_from(bytes);
        run_matvec(ctx, quant_kernel(dtype), &wbuf, x, out, rows, cols);
    });
}

/// M8.0: F32 matvec using the weight resident on the GPU (uploaded once, keyed
/// by tensor `name`). This is the inference path — no per-token weight upload.
pub fn matvec_f32_resident(name: &str, w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    CTX.with(|ctx| {
        let wbuf = ctx.resident_weight(name, bytemuck::cast_slice(w));
        run_matvec(ctx, "matvec_f32", &wbuf, x, out, rows, cols);
    });
}

/// M8.0: fused dequant + matvec using the resident quantized weight.
pub fn matvec_quant_resident(
    name: &str,
    bytes: &[u8],
    dtype: crate::gguf::GgmlType,
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % dtype.block_elems(), 0);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    CTX.with(|ctx| {
        let wbuf = ctx.resident_weight(name, bytes);
        run_matvec(ctx, quant_kernel(dtype), &wbuf, x, out, rows, cols);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// GPU vs CPU differ only in float accumulation order, so compare with a
    /// relative tolerance (the 1e-5 abs used CPU-only would false-fail).
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * b.abs().max(1.0)
    }

    #[test]
    fn matvec_f32_matches_cpu() {
        let mut rng = rand::thread_rng();
        for (rows, cols) in [(64usize, 128usize), (200, 257), (1, 64), (333, 32)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut cpu = vec![0.0f32; rows];
            crate::math::matmul::matvec(&w, &x, &mut cpu, rows, cols);
            let mut gpu = vec![0.0f32; rows];
            matvec_f32(&w, &x, &mut gpu, rows, cols);
            for m in 0..rows {
                assert!(close(gpu[m], cpu[m]), "row {m}: gpu {} vs cpu {}", gpu[m], cpu[m]);
            }
        }
    }

    // Quantize one 32-element block to Q8_0 bytes (ggml convention, as in
    // dtype.rs tests): f16 scale + 32 i8.
    fn quant_q8_0(block: &[f32]) -> Vec<u8> {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let mut raw = half::f16::from_f32(d).to_le_bytes().to_vec();
        for &v in block {
            let q = if d != 0.0 { (v / d).round().clamp(-128.0, 127.0) } else { 0.0 };
            raw.push(q as i8 as u8);
        }
        raw
    }

    // Quantize one 32-element block to Q4_0 bytes: f16 scale + 16 packed bytes.
    fn quant_q4_0(block: &[f32]) -> Vec<u8> {
        let max = block.iter().copied().fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut raw = half::f16::from_f32(d).to_le_bytes().to_vec();
        for j in 0..16 {
            let xi0 = ((block[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((block[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
            raw.push(xi0 | (xi1 << 4));
        }
        raw
    }

    fn quant_rows(w: &[f32], rows: usize, cols: usize, q: impl Fn(&[f32]) -> Vec<u8>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for r in 0..rows {
            for blk in w[r * cols..(r + 1) * cols].chunks_exact(32) {
                bytes.extend(q(blk));
            }
        }
        bytes
    }

    fn check_quant(dtype: crate::gguf::GgmlType, q: impl Fn(&[f32]) -> Vec<u8>) {
        let mut rng = rand::thread_rng();
        for (rows, cols) in [(64usize, 128usize), (200, 256), (1, 64), (129, 32)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let bytes = quant_rows(&w, rows, cols, &q);
            let mut cpu = vec![0.0f32; rows];
            crate::math::quant::matvec(&bytes, dtype, &x, &mut cpu, rows, cols);
            let mut gpu = vec![0.0f32; rows];
            matvec_quant(&bytes, dtype, &x, &mut gpu, rows, cols);
            for m in 0..rows {
                assert!(close(gpu[m], cpu[m]), "{dtype:?} row {m}: gpu {} vs cpu {}", gpu[m], cpu[m]);
            }
        }
    }

    #[test]
    fn resident_weight_is_cached() {
        // Upload wa under a key, then call again with *different* bytes wb under
        // the SAME key: the result must reflect the cached wa, proving the weight
        // was reused and not re-uploaded.
        let mut rng = rand::thread_rng();
        let (rows, cols) = (64usize, 128usize);
        let wa: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let wb: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut cpu_a = vec![0.0f32; rows];
        crate::math::matmul::matvec(&wa, &x, &mut cpu_a, rows, cols);

        let key = "test::resident_weight_is_cached";
        let mut g1 = vec![0.0f32; rows];
        matvec_f32_resident(key, &wa, &x, &mut g1, rows, cols); // uploads wa
        let mut g2 = vec![0.0f32; rows];
        matvec_f32_resident(key, &wb, &x, &mut g2, rows, cols); // must reuse wa

        for m in 0..rows {
            assert!(close(g1[m], cpu_a[m]), "first call should match wa");
            assert!(close(g2[m], cpu_a[m]), "cached call should still use wa, not wb");
        }
    }

    #[test]
    fn matvec_quant_resident_matches_cpu() {
        let mut rng = rand::thread_rng();
        let (rows, cols) = (96usize, 128usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bytes = quant_rows(&w, rows, cols, quant_q8_0);
        let dtype = crate::gguf::GgmlType::Q8_0;
        let mut cpu = vec![0.0f32; rows];
        crate::math::quant::matvec(&bytes, dtype, &x, &mut cpu, rows, cols);
        let mut gpu = vec![0.0f32; rows];
        matvec_quant_resident("test::quant_resident", &bytes, dtype, &x, &mut gpu, rows, cols);
        for m in 0..rows {
            assert!(close(gpu[m], cpu[m]), "row {m}: gpu {} vs cpu {}", gpu[m], cpu[m]);
        }
    }

    #[test]
    fn matvec_q8_0_matches_cpu() {
        check_quant(crate::gguf::GgmlType::Q8_0, quant_q8_0);
    }

    #[test]
    fn matvec_q4_0_matches_cpu() {
        check_quant(crate::gguf::GgmlType::Q4_0, quant_q4_0);
    }

    #[test]
    fn vadd_matches_cpu() {
        let mut rng = rand::thread_rng();
        let n = 1000;
        let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut out = vec![0.0f32; n];
        vadd(&a, &b, &mut out);
        for i in 0..n {
            assert!((out[i] - (a[i] + b[i])).abs() < 1e-6, "mismatch at {i}");
        }
    }
}
