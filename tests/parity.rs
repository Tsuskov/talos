//! THE CONTRACT (M2). The single test that makes Talos rigorous rather than
//! vibes: Talos's first-token logits must match Hephaistos's for the same model
//! and prompt, within tolerance.
//!
//! Until M2 lands this is `#[ignore]`d so the suite stays green. To run:
//!   1. Train/export a small model from Hephaistos to `models/tiny.gguf`.
//!   2. Capture Hephaistos's logits for a fixed prompt into
//!      `tests/fixtures/tiny_logits.json` (one f32 array).
//!   3. Remove `#[ignore]` and: `cargo test --test parity`.
//!
//! Tolerance: 1e-4 max abs diff for F32. Loosen for quantized models (M4).

use std::path::Path;

const MODEL: &str = "models/tiny.gguf";
const FIXTURE: &str = "tests/fixtures/tiny_logits.json";
const TOL: f32 = 1e-4;

#[test]
#[ignore = "enable once M2 forward pass + fixtures exist"]
fn first_token_logits_match_hephaistos() {
    let model_path = Path::new(MODEL);
    assert!(model_path.exists(), "missing {MODEL} — export one from Hephaistos");

    let mut model = talos::model::Model::load(model_path).expect("load model");

    // Fixed prompt token ids the fixture was captured with.
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        logits = model.forward(tok, pos);
    }

    let expected: Vec<f32> = {
        let raw = std::fs::read_to_string(FIXTURE).expect("missing logits fixture");
        // minimal JSON array parse without pulling serde into the test
        raw.trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<f32>().expect("f32"))
            .collect()
    };

    assert_eq!(logits.len(), expected.len(), "vocab size mismatch");
    let max_diff = logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= TOL, "max abs logit diff {max_diff} > {TOL}");
}
