//! Token sampling from logits. Owner: lead (M3).
//! Ported/extended from Hephaistos/src/sample.rs (adds top-p).

use rand::Rng;

/// Greedy: index of the maximum logit.
pub fn argmax(_logits: &[f32]) -> u32 {
    todo!()
}

/// Sample one token id with temperature, optional top-k and top-p (nucleus).
/// `temperature == 0.0` should behave like `argmax`.
pub fn sample<R: Rng>(
    _logits: &[f32],
    _temperature: f32,
    _top_k: Option<usize>,
    _top_p: Option<f32>,
    _rng: &mut R,
) -> u32 {
    todo!("M3")
}
