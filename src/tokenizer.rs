//! Tokenizers (M1). Owner: "tokenizer" agent.
//!
//! Talos loads two GGUF tokenizer kinds, dispatched on `tokenizer.ggml.model`:
//!   - `gpt2` (or absent): byte-level BPE, as Hephaistos/Cadmus export it.
//!   - `llama`: SentencePiece, as Mistral/Llama GGUFs carry it.
//! The [`Tokenizer`] enum wraps whichever the file declares; `Model::load`
//! builds it and callers use `encode`/`decode`/`bos`/`eos` without caring which.

use std::collections::{BinaryHeap, HashMap};

use anyhow::{anyhow, bail, Result};

use crate::gguf::{GgufFile, MetaValue};

/// A loaded tokenizer, one of the two GGUF kinds Talos supports.
pub enum Tokenizer {
    Bpe(Bpe),
    Spm(Spm),
}

impl Tokenizer {
    /// Build from a loaded GGUF, dispatching on the declared tokenizer model.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        match g.get_str("tokenizer.ggml.model") {
            Some("llama") => Ok(Tokenizer::Spm(Spm::from_gguf(g)?)),
            _ => Ok(Tokenizer::Bpe(Bpe::from_gguf(g)?)),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Tokenizer::Bpe(t) => t.encode(text),
            Tokenizer::Spm(t) => t.encode(text),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::Bpe(t) => t.decode(ids),
            Tokenizer::Spm(t) => t.decode(ids),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bpe(t) => t.vocab_size(),
            Tokenizer::Spm(t) => t.vocab_size(),
        }
    }

    pub fn bos(&self) -> u32 {
        match self {
            Tokenizer::Bpe(t) => t.bos(),
            Tokenizer::Spm(t) => t.bos(),
        }
    }

    pub fn eos(&self) -> u32 {
        match self {
            Tokenizer::Bpe(t) => t.eos(),
            Tokenizer::Spm(t) => t.eos(),
        }
    }

    /// Whether a BOS token should be prepended to an encoded prompt. SPM models
    /// set `add_bos_token`; the BPE path never adds one (preserving the behavior
    /// the Hephaistos/Cadmus models were exercised with).
    pub fn add_bos(&self) -> bool {
        match self {
            Tokenizer::Bpe(_) => false,
            Tokenizer::Spm(t) => t.add_bos,
        }
    }
}

// ===========================================================================
// Byte-level BPE (gpt2 kind) — Hephaistos/Cadmus exports.
// ===========================================================================

/// A byte-level BPE tokenizer built from GGUF metadata.
pub struct Bpe {
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

impl Bpe {
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
    fn from_gguf(g: &GgufFile) -> Result<Self> {
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
    fn encode(&self, text: &str) -> Vec<u32> {
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
    fn decode(&self, ids: &[u32]) -> String {
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

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    fn bos(&self) -> u32 {
        self.bos
    }
    fn eos(&self) -> u32 {
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

// ===========================================================================
// SentencePiece (llama kind) — Mistral/Llama GGUFs.
// ===========================================================================

/// token_type tags used by llama-kind GGUFs.
const TYPE_CONTROL: i32 = 3;
const TYPE_BYTE: i32 = 6;
/// SentencePiece space marker (U+2581, "▁").
const SPACE: char = '\u{2581}';

/// A SentencePiece tokenizer: a vocab of scored pieces, merged by the classic
/// SPM bigram algorithm (highest-scoring adjacent pair first), with `▁` marking
/// spaces and `<0xXX>` byte-fallback for pieces the vocab doesn't contain.
pub struct Spm {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    token_type: Vec<i32>,
    /// byte value -> id of its `<0xXX>` fallback token.
    byte_to_id: [Option<u32>; 256],
    bos: u32,
    eos: u32,
    unk: u32,
    add_bos: bool,
}

/// One symbol in the SPM merge list: a byte range of the normalized text plus
/// doubly-linked-list neighbors (`-1` = none). `len == 0` marks a merged-away
/// symbol.
struct Symbol {
    prev: i32,
    next: i32,
    start: usize,
    len: usize,
}

/// A candidate merge of two adjacent symbols. Ordered for a max-heap: highest
/// score first, ties broken toward the left-most position.
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.left == o.left
    }
}
impl Eq for Bigram {}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::Equal;
        match self.score.partial_cmp(&o.score).unwrap_or(Equal) {
            Equal => o.left.cmp(&self.left), // smaller left index = higher priority
            ord => ord,
        }
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Spm {
    fn build(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_type: Vec<i32>,
        bos: u32,
        eos: u32,
        unk: u32,
        add_bos: bool,
    ) -> Result<Self> {
        if scores.len() != tokens.len() || token_type.len() != tokens.len() {
            bail!(
                "tokenizer arrays disagree on length (tokens {}, scores {}, types {})",
                tokens.len(),
                scores.len(),
                token_type.len()
            );
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = [None; 256];
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
            if token_type[id] == TYPE_BYTE {
                if let Some(b) = parse_byte_token(tok) {
                    byte_to_id[b as usize] = Some(id as u32);
                }
            }
        }
        Ok(Spm {
            id_to_token: tokens,
            token_to_id,
            scores,
            token_type,
            byte_to_id,
            bos,
            eos,
            unk,
            add_bos,
        })
    }

    fn from_gguf(g: &GgufFile) -> Result<Self> {
        let tokens = g
            .get_arr_str("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?
            .to_vec();
        let scores = g
            .get_arr_f32("tokenizer.ggml.scores")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.scores"))?
            .to_vec();
        let token_type = g
            .get_arr_i32("tokenizer.ggml.token_type")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.token_type"))?
            .to_vec();
        let bos = g
            .get_u32("tokenizer.ggml.bos_token_id")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.bos_token_id"))?;
        let eos = g
            .get_u32("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.eos_token_id"))?;
        let unk = g.get_u32("tokenizer.ggml.unknown_token_id").unwrap_or(0);
        let add_bos = matches!(
            g.metadata("tokenizer.ggml.add_bos_token"),
            Some(MetaValue::Bool(true))
        );
        Self::build(tokens, scores, token_type, bos, eos, unk, add_bos)
    }

    /// Encode UTF-8 text to token ids (no BOS/EOS added; caller decides).
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // SentencePiece preprocessing: prepend a space (the "dummy prefix"), then
        // replace every space with the ▁ marker.
        let mut norm = String::with_capacity(text.len() + 2 * SPACE.len_utf8());
        norm.push(SPACE);
        for c in text.chars() {
            norm.push(if c == ' ' { SPACE } else { c });
        }

        // Seed one symbol per UTF-8 char, linked in order.
        let mut syms: Vec<Symbol> = Vec::new();
        let bytes = norm.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let len = utf8_len(bytes[i]);
            let idx = syms.len() as i32;
            syms.push(Symbol { prev: idx - 1, next: idx + 1, start: i, len });
            i += len;
        }
        if let Some(last) = syms.last_mut() {
            last.next = -1;
        }

        // Seed the heap with every adjacent bigram that is a known token.
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
        for j in 1..syms.len() {
            self.try_bigram(&mut heap, &syms, &norm, j as i32 - 1, j as i32);
        }

        // Merge the highest-scoring bigram until none remain.
        while let Some(b) = heap.pop() {
            let (ll, rl) = (syms[b.left as usize].len, syms[b.right as usize].len);
            if ll == 0 || rl == 0 || ll + rl != b.size {
                continue; // one side was already merged away; stale entry
            }
            syms[b.left as usize].len += rl;
            syms[b.right as usize].len = 0;
            let rnext = syms[b.right as usize].next;
            syms[b.left as usize].next = rnext;
            if rnext != -1 {
                syms[rnext as usize].prev = b.left;
            }
            let lprev = syms[b.left as usize].prev;
            self.try_bigram(&mut heap, &syms, &norm, lprev, b.left);
            self.try_bigram(&mut heap, &syms, &norm, b.left, rnext);
        }

        // Walk the surviving symbols; emit ids, falling back to bytes.
        let mut ids = Vec::new();
        let mut idx = 0i32;
        while idx != -1 {
            let s = &syms[idx as usize];
            if s.len > 0 {
                let piece = &norm[s.start..s.start + s.len];
                match self.token_to_id.get(piece) {
                    Some(&id) => ids.push(id),
                    None => {
                        for &byte in piece.as_bytes() {
                            ids.push(self.byte_to_id[byte as usize].unwrap_or(self.unk));
                        }
                    }
                }
            }
            idx = s.next;
        }
        ids
    }

    /// Push the bigram formed by symbols `left`+`right` if their concatenation
    /// is a known token.
    fn try_bigram(
        &self,
        heap: &mut BinaryHeap<Bigram>,
        syms: &[Symbol],
        norm: &str,
        left: i32,
        right: i32,
    ) {
        if left == -1 || right == -1 {
            return;
        }
        let (l, r) = (&syms[left as usize], &syms[right as usize]);
        let piece = &norm[l.start..r.start + r.len];
        if let Some(&id) = self.token_to_id.get(piece) {
            heap.push(Bigram { left, right, score: self.scores[id as usize], size: piece.len() });
        }
    }

    /// Decode token ids back to a string. Byte tokens emit their raw byte,
    /// control tokens (`<s>`, `</s>`) render to nothing, and `▁` becomes a space.
    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.id_to_token.len() {
                continue;
            }
            match self.token_type[i] {
                TYPE_BYTE => {
                    if let Some(b) = parse_byte_token(&self.id_to_token[i]) {
                        bytes.push(b);
                    }
                }
                TYPE_CONTROL => {}
                _ => {
                    for c in self.id_to_token[i].chars() {
                        if c == SPACE {
                            bytes.push(b' ');
                        } else {
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    fn bos(&self) -> u32 {
        self.bos
    }
    fn eos(&self) -> u32 {
        self.eos
    }
}

/// Length in bytes of the UTF-8 sequence starting with lead byte `b`.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Parse a `<0xXX>` byte-fallback token into its byte value.
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic byte-level BPE vocab that covers the full byte alphabet
    /// (so any UTF-8 input is encodable), plus a handful of merges to exercise
    /// the BPE loop.
    fn make_bpe() -> Bpe {
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

        Bpe::build(tokens, &merges, unk_id, unk_id, unk_id).unwrap()
    }

    #[test]
    fn round_trip_various() {
        let tok = make_bpe();
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
        let tok = make_bpe();
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
        let tok = make_bpe();
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

    /// Build a minimal SentencePiece vocab: control + unknown tokens, the ▁
    /// marker and a couple of ▁-prefixed pieces (for merging), and all 256
    /// `<0xXX>` byte tokens (so any UTF-8 input round-trips via byte fallback).
    fn make_spm() -> Spm {
        let mut tokens: Vec<String> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut types: Vec<i32> = Vec::new();
        let mut push = |t: &str, s: f32, ty: i32, tokens: &mut Vec<String>, scores: &mut Vec<f32>, types: &mut Vec<i32>| {
            tokens.push(t.to_string());
            scores.push(s);
            types.push(ty);
        };
        push("<unk>", 0.0, 2, &mut tokens, &mut scores, &mut types); // id 0
        push("<s>", 0.0, TYPE_CONTROL, &mut tokens, &mut scores, &mut types); // id 1
        push("</s>", 0.0, TYPE_CONTROL, &mut tokens, &mut scores, &mut types); // id 2
        // ▁ marker and a stepwise-reachable word "▁ab".
        push("\u{2581}", -1.0, 1, &mut tokens, &mut scores, &mut types);
        push("\u{2581}a", -1.0, 1, &mut tokens, &mut scores, &mut types);
        push("\u{2581}ab", -0.5, 1, &mut tokens, &mut scores, &mut types);
        // All 256 byte-fallback tokens.
        for b in 0u8..=255 {
            push(&format!("<0x{b:02X}>"), -10.0, TYPE_BYTE, &mut tokens, &mut scores, &mut types);
        }
        Spm::build(tokens, scores, types, 1, 2, 0, true).unwrap()
    }

    #[test]
    fn spm_leading_space_and_byte_fallback() {
        let tok = make_spm();
        // The dummy prefix ▁ decodes to a leading space; the rest reconstructs
        // exactly via byte fallback, for ASCII and arbitrary UTF-8 alike.
        for s in ["hello", "héllo 世界", "emoji 😀", "tab\tnl\n"] {
            assert_eq!(tok.decode(&tok.encode(s)), format!(" {s}"), "spm round trip {s:?}");
        }
    }

    #[test]
    fn spm_merges_highest_score_stepwise() {
        let tok = make_spm();
        // "ab" -> normalized "▁ab": merges ▁+a -> "▁a", then "▁a"+b -> "▁ab".
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![tok.token_to_id["\u{2581}ab"]]);
        assert_eq!(tok.decode(&ids), " ab");
    }

    #[test]
    fn spm_control_tokens_render_empty() {
        let tok = make_spm();
        // bos/eos are control tokens: they contribute no text on decode.
        let mut ids = vec![tok.bos()];
        ids.extend(tok.encode("hi"));
        ids.push(tok.eos());
        assert_eq!(tok.decode(&ids), " hi");
    }
}
