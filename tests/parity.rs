//! THE CONTRACT (M2). The single test that makes Talos rigorous rather than
//! vibes: Talos's last-position logits must match Hephaistos's for the same
//! model and prompt, within tolerance.
//!
//! `models/tiny.gguf` is git-ignored (binary), so regenerate the fixture from
//! the Hephaistos trainer if it's absent — the test skips cleanly when it is,
//! keeping the suite green, and enforces parity when present. Tolerance: 1e-4
//! max abs diff for F32 (loosen for quantized models in M4).

use std::path::Path;

const MODEL: &str = "models/tiny.gguf";
const FIXTURE: &str = "tests/fixtures/tiny_logits.json";
const TOL: f32 = 1e-4;

#[test]
fn last_position_logits_match_hephaistos() {
    let model_path = Path::new(MODEL);
    if !model_path.exists() {
        eprintln!("skipping parity: {MODEL} not present (regenerate from Hephaistos)");
        return;
    }

    let mut model = talos::model::Model::load(model_path).expect("load model");

    // Fixed prompt token ids the fixture was captured with.
    let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
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
    eprintln!("parity max abs logit diff = {max_diff:e} (tol {TOL:e})");
    assert!(max_diff <= TOL, "max abs logit diff {max_diff} > {TOL}");
}
