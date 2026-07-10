//! Byte-level BPE tokenizer (M1). Owner: "tokenizer" agent.
//!
//! Reads the GGUF `gpt2`-kind tokenizer metadata that Hephaistos writes:
//!   tokenizer.ggml.tokens      (ArrStr)  — id -> token piece
//!   tokenizer.ggml.token_type  (ArrI32)  — 1 NORMAL, 2 UNKNOWN, 3 CONTROL
//!   tokenizer.ggml.merges      (ArrStr)  — "a b" merge rules, in priority order
//!   tokenizer.ggml.unknown_token_id / bos_token_id / eos_token_id (U32)
//!   tokenizer.ggml.add_bos_token (Bool)
//!
//! This is byte-level BPE (GPT-2 style): bytes are mapped into a printable
//! unicode alphabet before merging, and reversed on decode. Match that mapping
//! so `decode(encode(s)) == s` for arbitrary UTF-8.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::gguf::GgufFile;

/// A byte-level BPE tokenizer built from GGUF metadata.
pub struct Tokenizer {
    /// id -> token piece (in the byte-level unicode alphabet).
    id_to_token: Vec<String>,
    /// token piece -> id.
    token_to_id: HashMap<String, u32>,
    /// (left piece, right piece) -> merge rank (lower = higher priority).
    merge_rank: HashMap<(String, String), u32>,
    /// byte -> alphabet char.
    byte_encoder: [char; 256],
    /// alphabet char -> byte.
    byte_decoder: HashMap<char, u8>,
    bos: u32,
    eos: u32,
    unk: u32,
}

/// GPT-2's reversible bytes<->unicode table: maps every byte to a printable
/// unicode codepoint so BPE operates over strings with no control/space chars.
fn bytes_to_unicode() -> [char; 256] {
    // Bytes that are already printable map to themselves.
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);

    let mut cs: Vec<u32> = bs.clone();
    // Remaining bytes get mapped to codepoints 256, 257, ... in order.
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

impl Tokenizer {
    fn build(
        tokens: Vec<String>,
        merges: &[String],
        bos: u32,
        eos: u32,
        unk: u32,
    ) -> Result<Self> {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            let a = it
                .next()
                .ok_or_else(|| anyhow!("malformed merge rule: {m:?}"))?;
            let b = it
                .next()
                .ok_or_else(|| anyhow!("malformed merge rule: {m:?}"))?;
            merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        Ok(Self {
            id_to_token: tokens,
            token_to_id,
            merge_rank,
            byte_encoder,
            byte_decoder,
            bos,
            eos,
            unk,
        })
    }

    /// Build from a loaded GGUF file's tokenizer metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let tokens = g
            .get_arr_str("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?
            .to_vec();
        let merges = g
            .get_arr_str("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.merges"))?
            .to_vec();
        let unk = g
            .get_u32("tokenizer.ggml.unknown_token_id")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.unknown_token_id"))?;
        let bos = g
            .get_u32("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.bos_token_id"))?;
        let eos = g
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.eos_token_id"))?;

        Self::build(tokens, &merges, bos, eos, unk)
    }

    /// Apply BPE merges to a single word (already mapped to the byte alphabet),
    /// returning the resulting pieces.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if pieces.len() < 2 {
            return pieces;
        }

        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..pieces.len() - 1 {
                if let Some(&rank) =
                    self.merge_rank.get(&(pieces[i].clone(), pieces[i + 1].clone()))
                {
                    if best.is_none_or(|(_, br)| rank < br) {
                        best = Some((i, rank));
                    }
                }
            }

            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", pieces[i], pieces[i + 1]);
            pieces.splice(i..i + 2, [merged]);
            if pieces.len() < 2 {
                break;
            }
        }
        pieces
    }

    /// Encode UTF-8 text to token ids (no BOS/EOS added; caller decides).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Pre-tokenize like Cadmus (split at spaces, a leading space attaching
        // to the word that follows), then run BPE per word. The merge loop is
        // quadratic in unit length, so running it over the whole input made
        // encode hang on megabyte files; per-word units keep it linear. For
        // vocabs trained with this splitter (Cadmus) no merge crosses a word
        // boundary, so the ids are identical to whole-text BPE.
        let mut ids = Vec::new();
        for word in pretokenize(text) {
            let mapped: String = word
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            for piece in self.bpe(&mapped) {
                match self.token_to_id.get(&piece) {
                    Some(&id) => ids.push(id),
                    None => ids.push(self.unk),
                }
            }
        }
        ids
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for c in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn bos(&self) -> u32 {
        self.bos
    }
    pub fn eos(&self) -> u32 {
        self.eos
    }
}

/// Split text into BPE units, a leading space attaching to the word that
/// follows (" world" is one unit) — the same rule as Cadmus's pretokenizer.
/// Only splits, never drops a byte, so decode(encode(s)) == s is preserved.
fn pretokenize(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == ' ' && i > start {
            words.push(&text[start..i]);
            start = i;
        }
    }
    if start < text.len() {
        words.push(&text[start..]);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic byte-level vocab that covers the full byte alphabet
    /// (so any UTF-8 input is encodable), plus a handful of merges to exercise
    /// the BPE loop.
    fn make_tokenizer() -> Tokenizer {
        let byte_alphabet = bytes_to_unicode();

        // One token per alphabet char => round-trip is always possible.
        let mut tokens: Vec<String> = byte_alphabet.iter().map(|c| c.to_string()).collect();

        // A few multi-char merge targets to make sure merging actually fires.
        // "he", "ll", and "hell" should be reachable for the string "hello".
        let h = byte_alphabet[b'h' as usize];
        let e = byte_alphabet[b'e' as usize];
        let l = byte_alphabet[b'l' as usize];
        let o = byte_alphabet[b'o' as usize];

        let he = format!("{h}{e}");
        let ll = format!("{l}{l}");
        let hell = format!("{he}{ll}");
        let hello = format!("{hell}{o}");
        tokens.push(he.clone());
        tokens.push(ll.clone());
        tokens.push(hell.clone());
        tokens.push(hello.clone());

        // <unk> token at the end.
        let unk_id = tokens.len() as u32;
        tokens.push("<unk>".to_string());

        // Merges in priority order: he, ll, he+ll -> hell, hell+o -> hello.
        let merges = vec![
            format!("{h} {e}"),
            format!("{l} {l}"),
            format!("{he} {ll}"),
            format!("{hell} {o}"),
        ];

        Tokenizer::build(tokens, &merges, unk_id, unk_id, unk_id).unwrap()
    }

    #[test]
    fn round_trip_various() {
        let tok = make_tokenizer();
        let cases = [
            "",
            "hello",
            "hello world",
            "héllo 世界",
            "emoji: 😀🚀✨",
            "tabs\tand\nnewlines",
            "mixed ascii / 日本語 / Ελληνικά / 🎉",
            "\u{0}\u{1}\u{7f}", // control bytes
        ];
        for s in cases {
            let ids = tok.encode(s);
            let back = tok.decode(&ids);
            assert_eq!(back, s, "round trip failed for {s:?}");
        }
    }

    #[test]
    fn merges_apply() {
        let tok = make_tokenizer();
        // "hello" should collapse to the single "hello" merge target.
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 1, "expected merges to collapse 'hello' to 1 token");
        assert_eq!(tok.decode(&ids), "hello");
    }

    #[test]
    fn pretokenize_splits_like_cadmus() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize(" leading"), vec![" leading"]);
        // Runs of spaces become single-space units (Cadmus convention).
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
        // Newlines do not split; they stay inside their unit.
        assert_eq!(pretokenize("a\nb c"), vec!["a\nb", " c"]);
        assert_eq!(pretokenize(""), Vec::<&str>::new());
    }

    /// Regression test for the O(n^2) whole-text merge loop: encoding a few
    /// hundred KB must terminate (it used to effectively hang on MB inputs).
    #[test]
    fn encode_scales_to_large_inputs() {
        let tok = make_tokenizer();
        let text = "hello world, hello again.\n".repeat(12_000); // ~300 KB
        let ids = tok.encode(&text);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn byte_alphabet_is_reversible() {
        let enc = bytes_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            assert!(seen.insert(enc[b as usize]), "duplicate alphabet char for byte {b}");
        }
        assert_eq!(seen.len(), 256);
    }
}
