//! Matrix-vector product (the inference hot path). Owner: "math-ops" agent.
//!
//! Weights are stored row-major as `[rows, cols]` = `[out_features, in_features]`
//! (this is the raw data Hephaistos writes; its GGUF `ne` dims are the reverse,
//! `[cols, rows]`, but the bytes are row-major). So row `m` is the contiguous
//! slice `w[m*cols .. (m+1)*cols]`.
//!
//! M2: a correct, rayon-parallel f32 implementation. M5: SIMD + tiling.

/// `out[m] = sum_k w[m*cols + k] * x[k]` for m in 0..rows.
/// `w.len() == rows*cols`, `x.len() == cols`, `out.len() == rows`.
pub fn matvec(w: &[f32], x: &[f32], out: &mut [f32], _rows: usize, cols: usize) {
    use rayon::prelude::*;
    out.par_iter_mut()
        .zip(w.par_chunks(cols))
        .for_each(|(o, row)| {
            *o = row.iter().zip(x).map(|(&wv, &xv)| wv * xv).sum();
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn naive(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for m in 0..rows {
            let mut acc = 0.0f32;
            for k in 0..cols {
                acc += w[m * cols + k] * x[k];
            }
            out[m] = acc;
        }
        out
    }

    fn check(rows: usize, cols: usize) {
        let mut rng = rand::thread_rng();
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let expected = naive(&w, &x, rows, cols);
        let mut out = vec![0.0f32; rows];
        matvec(&w, &x, &mut out, rows, cols);
        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    #[test]
    fn matvec_square() {
        check(8, 8);
    }

    #[test]
    fn matvec_non_square() {
        check(5, 13);
        check(13, 5);
    }

    #[test]
    fn matvec_hand() {
        // w = [[1,2],[3,4],[5,6]], x = [1, 1].
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0];
        let mut out = [0.0; 3];
        matvec(&w, &x, &mut out, 3, 2);
        assert_eq!(out, [3.0, 7.0, 11.0]);
    }
}
