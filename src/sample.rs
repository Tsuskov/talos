//! Token sampling from logits. Owner: lead (M3).
//! Ported/extended from Hephaistos/src/sample.rs (adds top-p).

use rand::Rng;

/// Greedy: index of the maximum logit.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .expect("empty logits")
}

/// Sample one token id with temperature, optional top-k, and optional top-p
/// (nucleus). `temperature <= 0.0` is greedy (== `argmax`). Filters are applied
/// in order: temperature scale -> top-k -> top-p -> renormalize -> sample.
pub fn sample<R: Rng>(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut R,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let n = logits.len();

    // Indices sorted by logit descending; truncate to top-k.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let keep_k = top_k.map_or(n, |k| k.clamp(1, n));
    let kept = &idx[..keep_k];

    // Softmax (temperature-scaled, max-shifted) over the kept set.
    let maxv = logits[kept[0]];
    let mut probs: Vec<f32> = kept
        .iter()
        .map(|&i| ((logits[i] - maxv) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // top-p: smallest prefix whose cumulative probability reaches p (>= 1 token).
    let cutoff = match top_p {
        Some(p) => {
            let mut c = 0.0;
            let mut m = keep_k;
            for (j, &pr) in probs.iter().enumerate() {
                c += pr;
                if c >= p {
                    m = j + 1;
                    break;
                }
            }
            m
        }
        None => keep_k,
    };

    // Sample from the (renormalized) surviving prefix.
    let total: f32 = probs[..cutoff].iter().sum();
    let r = rng.gen::<f32>() * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if acc >= r {
            return kept[j] as u32;
        }
    }
    kept[cutoff - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[0.1, 9.0, -3.0, 2.0]), 1);
    }

    #[test]
    fn temp_zero_is_greedy() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample(&logits, 0.0, None, None, &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            assert_eq!(sample(&logits, 1.0, Some(1), None, &mut rng), 1);
        }
    }

    #[test]
    fn tiny_top_p_is_greedy() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            assert_eq!(sample(&logits, 1.0, None, Some(1e-6), &mut rng), 1);
        }
    }

    #[test]
    fn sampled_token_stays_in_top_k() {
        // With top-k = 2 on these logits, only ids 1 and 3 may ever be drawn.
        let logits = [0.1, 9.0, -3.0, 2.0];
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..200 {
            let t = sample(&logits, 1.0, Some(2), None, &mut rng);
            assert!(t == 1 || t == 3, "drew {t} outside top-2");
        }
    }
}
