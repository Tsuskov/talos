//! M4: quantized inference matches the F32 reference within the error budget of
//! each format. Skips cleanly when the (git-ignored) model files are absent.

use std::path::Path;

const F32_MODEL: &str = "models/tiny32_f32.gguf";
const FIXTURE: &str = "tests/fixtures/tiny32_logits.json";
const PROMPT: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];

fn last_logits(model_path: &str) -> Option<Vec<f32>> {
    if !Path::new(model_path).exists() {
        return None;
    }
    let mut model = talos::model::Model::load(Path::new(model_path)).expect("load");
    let mut logits = Vec::new();
    for (pos, &tok) in PROMPT.iter().enumerate() {
        logits = model.forward(tok, pos);
    }
    Some(logits)
}

fn read_fixture() -> Vec<f32> {
    let raw = std::fs::read_to_string(FIXTURE).expect("fixture");
    raw.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect()
}

/// Max abs diff and argmax agreement vs the F32 reference.
fn report(name: &str, got: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(got.len(), reference.len());
    let max_diff = got
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let argmax = |v: &[f32]| {
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
    };
    eprintln!(
        "{name}: max abs diff {max_diff:.5}, argmax {} (ref {})",
        argmax(got),
        argmax(reference)
    );
    max_diff
}

#[test]
fn quantized_logits_track_f32() {
    if !Path::new(F32_MODEL).exists() {
        eprintln!("skipping quant test: {F32_MODEL} absent (regenerate from Hephaistos)");
        return;
    }

    // Sanity: our F32 path reproduces the captured reference exactly.
    let f32_logits = last_logits(F32_MODEL).unwrap();
    let reference = read_fixture();
    let f32_diff = report("f32", &f32_logits, &reference);
    assert!(f32_diff <= 1e-4, "f32 path drifted from reference: {f32_diff}");

    if let Some(q8) = last_logits("models/tiny32_q8.gguf") {
        let d = report("q8_0", &q8, &reference);
        assert!(d <= 0.05, "q8_0 diff {d} too large");
    }
    if let Some(q4) = last_logits("models/tiny32_q4.gguf") {
        let d = report("q4_0", &q4, &reference);
        assert!(d <= 0.5, "q4_0 diff {d} too large");
    }
}
