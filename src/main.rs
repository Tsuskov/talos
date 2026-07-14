//! Talos CLI. Owner: lead.
//!
//!   talos inspect <model.gguf>
//!   talos run <model.gguf> --prompt "…" [-n N] [--temp T] [--top-k K] [--top-p P] [--seed S]
//!   talos perplexity <model.gguf> <text-file>

use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use talos::gguf::GgufFile;
use talos::model::Model;
use talos::sample::sample;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => {
            let path = args.get(1).map(Path::new).ok_or_else(usage_err)?;
            inspect(path)
        }
        Some("run") => run(&args[1..]),
        Some("perplexity") => perplexity(&args[1..]),
        _ => Err(usage_err()),
    }
}

/// `talos inspect` — print metadata + tensor index.
fn inspect(path: &Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    println!("# tensors");
    for t in g.tensors() {
        println!("  {:<32} {:?} {:?}", t.name, t.dims, t.dtype);
    }
    Ok(())
}

/// `talos perplexity <model.gguf> <text-file>` — teacher-forced perplexity of
/// the file's text under the model. The honest quality number: run it on the
/// same text for the F32 and quantized exports to see what quantization costs.
fn perplexity(args: &[String]) -> Result<()> {
    let model_path = args.first().ok_or_else(usage_err)?;
    let text_path = args.get(1).ok_or_else(usage_err)?;

    let mut model = Model::load(Path::new(model_path))?;
    let text = std::fs::read_to_string(text_path)
        .with_context(|| format!("reading {text_path}"))?;

    // Prepend BOS so the first real token is scored with context.
    let mut tokens = vec![model.tokenizer.bos()];
    tokens.extend(model.tokenizer.encode(&text));
    if tokens.len() < 2 {
        bail!("text encoded to fewer than 2 tokens; nothing to score");
    }

    let ppl = talos::eval::perplexity(&mut model, &tokens);
    println!("perplexity {ppl:.4}  ({} tokens)", tokens.len());
    Ok(())
}

/// `talos run` — load a model, encode the prompt, and stream a continuation.
fn run(args: &[String]) -> Result<()> {
    let model_path = args.first().ok_or_else(usage_err)?;
    let opts = Opts::parse(&args[1..])?;
    let prompt = opts.prompt.as_deref().ok_or_else(|| anyhow!("--prompt is required"))?;

    let mut model = Model::load(Path::new(model_path))?;
    model.cap_context(opts.ctx);
    let mut rng = StdRng::seed_from_u64(opts.seed.unwrap_or_else(rand::random));

    let mut prompt_ids = Vec::new();
    if model.tokenizer.add_bos() {
        prompt_ids.push(model.tokenizer.bos());
    }
    prompt_ids.extend(model.tokenizer.encode(prompt));
    if prompt_ids.is_empty() {
        bail!("prompt encoded to zero tokens");
    }

    let mut stdout = std::io::stdout();
    print!("{prompt}");
    stdout.flush().ok();

    // Prefill: feed the prompt, keeping the logits after the last token.
    model.reset();
    let mut pos = 0usize;
    let mut logits = Vec::new();
    for &t in &prompt_ids {
        logits = model.step(t, pos);
        pos += 1;
    }

    // Decode loop.
    let eos = model.tokenizer.eos();
    let mut out_ids = Vec::new();
    let mut printed = 0usize;
    for _ in 0..opts.n {
        if pos >= model.cfg.context_length {
            break;
        }
        let next = sample(&logits, opts.temp, opts.top_k, opts.top_p, &mut rng);
        if next == eos {
            break;
        }
        out_ids.push(next);
        printed = flush_new(&model.tokenizer.decode(&out_ids), printed, &mut stdout);
        logits = model.step(next, pos);
        pos += 1;
    }
    println!();
    Ok(())
}

/// Print whatever decoded text is newly complete since `printed` chars, holding
/// back a trailing replacement char (an incomplete multibyte sequence) until the
/// next token completes it. Returns the new printed-char count.
fn flush_new(text: &str, printed: usize, out: &mut impl Write) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let end = if chars.last() == Some(&'\u{FFFD}') {
        chars.len().saturating_sub(1)
    } else {
        chars.len()
    };
    for c in &chars[printed.min(end)..end] {
        print!("{c}");
    }
    out.flush().ok();
    end
}

struct Opts {
    prompt: Option<String>,
    n: usize,
    temp: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    ctx: usize,
}

impl Opts {
    fn parse(args: &[String]) -> Result<Self> {
        // Cap the KV cache at 4096 positions by default: a model's own context
        // (e.g. Mistral's 32768) sizes the GPU KV buffers to ~8.6 GB, far more
        // than a CLI run needs. Raise it with --ctx for longer generations.
        let mut o = Opts { prompt: None, n: 64, temp: 0.8, top_k: None, top_p: None, seed: None, ctx: 4096 };
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let mut val = || it.next().ok_or_else(|| anyhow!("{flag} needs a value"));
            match flag.as_str() {
                "--prompt" | "-p" => o.prompt = Some(val()?.clone()),
                "-n" | "--tokens" => o.n = val()?.parse().context("-n")?,
                "--temp" => o.temp = val()?.parse().context("--temp")?,
                "--top-k" => o.top_k = Some(val()?.parse().context("--top-k")?),
                "--top-p" => o.top_p = Some(val()?.parse().context("--top-p")?),
                "--seed" => o.seed = Some(val()?.parse().context("--seed")?),
                "--ctx" => o.ctx = val()?.parse().context("--ctx")?,
                other => bail!("unknown option {other}"),
            }
        }
        Ok(o)
    }
}

fn usage_err() -> anyhow::Error {
    anyhow::anyhow!(
        "usage:\n  talos inspect <model.gguf>\n  talos run <model.gguf> --prompt \"…\" [-n N] [--temp T] [--top-k K] [--top-p P] [--seed S] [--ctx N]\n  talos perplexity <model.gguf> <text-file>"
    )
}
