//! Talos CLI. Owner: lead.
//!
//!   talos inspect <model.gguf>          — print metadata + tensor index (M0)
//!   talos run <model.gguf> --prompt "…" [-n N] [--temp T] [--top-k K] [--top-p P]

use std::path::Path;

use anyhow::Result;

use talos::gguf::GgufFile;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => {
            let path = args.get(1).map(Path::new).ok_or_else(usage_err)?;
            inspect(path)
        }
        Some("run") => {
            // Wired up in M2/M3 once Model::forward and sampling land.
            todo!("run: load model, encode prompt, decode loop, stream tokens")
        }
        _ => Err(usage_err()),
    }
}

/// `talos inspect` — usable as soon as the GGUF reader (M0) lands.
fn inspect(path: &Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    println!("# tensors");
    for t in g.tensors() {
        println!("  {:<32} {:?} {:?}", t.name, t.dims, t.dtype);
    }
    Ok(())
}

fn usage_err() -> anyhow::Error {
    anyhow::anyhow!("usage: talos <inspect|run> <model.gguf> [options]")
}
