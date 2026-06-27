//! Perplexity — the honest measure of model quality, and the metric that makes
//! the quantization claims testable: a Q4_0 model that "still works" should have
//! a perplexity within a few percent of the F32 original, not just a matching
//! argmax on one prompt.
//!
//! Perplexity is `exp` of the mean per-token negative log-likelihood under
//! teacher forcing: feed the real tokens in order and, at each position, ask how
//! much probability mass the model put on the token that actually came next.
//! Lower is better; a model that assigned probability 1 to every next token
//! would score 1, and a model no better than a uniform guess over a vocabulary
//! of `V` tokens scores `V`.

use crate::model::Model;

/// Negative log-likelihood (in nats) the logits assign to `target`:
/// `-log softmax(logits)[target]`. Computed via the log-sum-exp trick in `f64`
/// so a confident logit can't overflow `exp`.
pub fn token_nll(logits: &[f32], target: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sumexp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_prob = (logits[target as usize] - max) as f64 - sumexp.ln();
    -log_prob as f32
}

/// Teacher-forced perplexity of `tokens` under `model`.
///
/// Feeds `tokens[0..n-1]` in order (resetting the KV cache first) and, at each
/// position, accumulates the NLL the model assigns to the *next* token. Returns
/// `exp(mean NLL)`. Requires at least two tokens; panics otherwise, since
/// perplexity is undefined without a single prediction to score.
pub fn perplexity(model: &mut Model, tokens: &[u32]) -> f32 {
    assert!(tokens.len() >= 2, "perplexity needs at least 2 tokens");
    model.reset();
    let mut nll = 0.0f64;
    for (pos, pair) in tokens.windows(2).enumerate() {
        let logits = model.forward(pair[0], pos);
        nll += token_nll(&logits, pair[1]) as f64;
    }
    (nll / tokens.windows(2).count() as f64).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_score_ln_vocab() {
        // All logits equal => softmax is uniform over V => p(target) = 1/V for
        // any target, so the NLL is exactly ln(V).
        let v = 50usize;
        let logits = vec![0.0f32; v];
        for target in [0u32, 7, (v - 1) as u32] {
            let nll = token_nll(&logits, target);
            assert!((nll - (v as f32).ln()).abs() < 1e-4, "got {nll}");
        }
    }

    #[test]
    fn confident_correct_prediction_scores_near_zero() {
        // A huge logit on the target drives softmax(target) -> 1, NLL -> 0.
        let mut logits = vec![0.0f32; 16];
        logits[5] = 100.0;
        assert!(token_nll(&logits, 5) < 1e-3);
    }

    #[test]
    fn nll_matches_hand_computed_softmax() {
        // logits [1, 2, 3]: p(target=2) = e^3 / (e^1 + e^2 + e^3).
        let logits = [1.0f32, 2.0, 3.0];
        let denom = 1f64.exp() + 2f64.exp() + 3f64.exp();
        let expected = -(3f64.exp() / denom).ln() as f32;
        assert!((token_nll(&logits, 2) - expected).abs() < 1e-5);
    }
}
