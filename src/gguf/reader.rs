//! GGUF reader (M0). Owner: "gguf-reader" agent.
//!
//! Implement `GgufFile::open` to mmap the file and parse the header, metadata
//! KV map, and tensor index. Keep the mmap alive for the file's lifetime so
//! `tensor_f32` can return a zero-copy `&[f32]` view into the data section.
//!
//! Alignment note: the data section starts on a 32-byte boundary and every
//! tensor offset is 32-byte aligned, so f32 views are always 4-byte aligned —
//! `bytemuck::cast_slice::<u8, f32>` is safe on each tensor's byte range.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;

use super::dtype::GgmlType;
use super::MetaValue;

const ALIGNMENT: usize = 32;

// GGUF metadata value type tags.
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

/// One entry in the tensor index.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF `ne` order (innermost/contiguous first). A row-major
    /// `[rows, cols]` weight is stored as `dims = [cols, rows]`.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count = product of dims.
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// A GGUF file with its metadata and tensor index parsed.
pub struct GgufFile {
    backing: Backing,
    meta: HashMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    tensor_by_name: HashMap<String, usize>,
    data_start: usize,
}

/// Bytes backing the file: an mmap on native targets, an owned buffer when
/// parsed from bytes (the only option on wasm, which has no filesystem).
enum Backing {
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Backing::Mmap(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// Cursor over the mmap'd bytes that bounds-checks every read.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow!("offset overflow"))?;
        if end > self.buf.len() {
            bail!("unexpected end of file (need {n} bytes at {})", self.pos);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).context("metadata string is not valid UTF-8")
    }
}

/// Parse one metadata value given its type tag.
fn read_value(c: &mut Cursor, tag: u32) -> Result<MetaValue> {
    match tag {
        T_UINT32 => Ok(MetaValue::U32(c.u32()?)),
        T_FLOAT32 => Ok(MetaValue::F32(c.f32()?)),
        T_BOOL => Ok(MetaValue::Bool(c.u8()? != 0)),
        T_STRING => Ok(MetaValue::Str(c.string()?)),
        T_ARRAY => {
            let inner = c.u32()?;
            let count = c.u64()? as usize;
            match inner {
                T_STRING => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(c.string()?);
                    }
                    Ok(MetaValue::ArrStr(v))
                }
                T_INT32 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        v.push(c.i32()?);
                    }
                    Ok(MetaValue::ArrI32(v))
                }
                other => bail!("unsupported array element type tag {other}"),
            }
        }
        other => bail!("unsupported metadata value type tag {other}"),
    }
}

impl GgufFile {
    /// Mmap and parse `path`. Errors on bad magic, unsupported version, or a
    /// truncated/malformed index.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: we keep the mmap alive for the GgufFile's lifetime and only
        // expose immutable views into it.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmapping {}", path.display()))?;
        Self::parse(Backing::Mmap(mmap))
    }

    /// Parse a GGUF file already loaded into memory — the path used on wasm,
    /// where the model arrives as fetched bytes instead of a file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Backing::Owned(bytes))
    }

    fn parse(backing: Backing) -> Result<Self> {
        let mut c = Cursor::new(&backing);

        let magic = c.take(4)?;
        if magic != b"GGUF" {
            bail!("not a GGUF file (bad magic {magic:?})");
        }
        let version = c.u32()?;
        if version != 3 {
            bail!("unsupported GGUF version {version} (expected 3)");
        }
        let tensor_count = c.u64()? as usize;
        let kv_count = c.u64()? as usize;

        let mut meta = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = c.string()?;
            let tag = c.u32()?;
            let value = read_value(&mut c, tag)
                .with_context(|| format!("reading metadata key {key:?}"))?;
            meta.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_by_name = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let type_tag = c.u32()?;
            let dtype = GgmlType::from_u32(type_tag)
                .ok_or_else(|| anyhow!("tensor {name:?} has unsupported ggml type {type_tag}"))?;
            let offset = c.u64()?;
            tensor_by_name.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }

        // Pad to the 32-byte-aligned data section start.
        let data_start = c.pos.div_ceil(ALIGNMENT) * ALIGNMENT;
        if data_start > backing.len() {
            bail!("data section start {data_start} past end of file {}", backing.len());
        }

        Ok(Self {
            backing,
            meta,
            tensors,
            tensor_by_name,
            data_start,
        })
    }

    /// Raw metadata lookup.
    pub fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.meta.get(key)? {
            MetaValue::U32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.meta.get(key)? {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key)? {
            MetaValue::Str(v) => Some(v.as_str()),
            _ => None,
        }
    }
    pub fn get_arr_str(&self, key: &str) -> Option<&[String]> {
        match self.meta.get(key)? {
            MetaValue::ArrStr(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    pub fn get_arr_i32(&self, key: &str) -> Option<&[i32]> {
        match self.meta.get(key)? {
            MetaValue::ArrI32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// All tensors, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor's metadata by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Zero-copy F32 view into the data section for tensor `name`. Errors if the
    /// tensor is missing or not F32.
    pub fn tensor_f32(&self, name: &str) -> Result<&[f32]> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| anyhow!("no tensor named {name:?}"))?;
        if info.dtype != GgmlType::F32 {
            bail!("tensor {name:?} is {:?}, not F32", info.dtype);
        }
        let elems = info.elem_count();
        let start = self
            .data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| anyhow!("tensor offset overflow"))?;
        let byte_len = elems
            .checked_mul(4)
            .ok_or_else(|| anyhow!("tensor byte length overflow"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| anyhow!("tensor byte range overflow"))?;
        if end > self.backing.len() {
            bail!("tensor {name:?} data range past end of file");
        }
        Ok(bytemuck::cast_slice(&self.backing[start..end]))
    }

    /// Raw tensor bytes plus dtype, for any supported type (F32 or quantized).
    /// Byte length is `block_bytes · elem_count / block_elems`.
    pub fn tensor_raw(&self, name: &str) -> Result<(&[u8], GgmlType)> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| anyhow!("no tensor named {name:?}"))?;
        let elems = info.elem_count();
        let blk = info.dtype.block_elems();
        if elems % blk != 0 {
            bail!("tensor {name:?} has {elems} elems, not a multiple of block {blk}");
        }
        let byte_len = (elems / blk)
            .checked_mul(info.dtype.block_bytes())
            .ok_or_else(|| anyhow!("tensor byte length overflow"))?;
        let start = self
            .data_start
            .checked_add(info.offset as usize)
            .ok_or_else(|| anyhow!("tensor offset overflow"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| anyhow!("tensor byte range overflow"))?;
        if end > self.backing.len() {
            bail!("tensor {name:?} data range past end of file");
        }
        Ok((&self.backing[start..end], info.dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const T_UINT32: u32 = 4;
    const T_INT32: u32 = 5;
    const T_FLOAT32: u32 = 6;
    const T_BOOL: u32 = 7;
    const T_STRING: u32 = 8;
    const T_ARRAY: u32 = 9;
    const GGML_TYPE_F32: u32 = 0;
    const ALIGN: usize = 32;

    fn put_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a synthetic GGUF v3 file replicating the Hephaistos writer layout.
    fn build_gguf() -> Vec<u8> {
        // Tensors: row-major data, dims in ne order.
        let t1: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // dims [3, 2]
        let t2: &[f32] = &[-1.5, 0.0, 42.25]; // dims [3]

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // kv_count

        // KV: u32
        put_string(&mut buf, "answer.u32");
        buf.extend_from_slice(&T_UINT32.to_le_bytes());
        buf.extend_from_slice(&42u32.to_le_bytes());

        // KV: f32
        put_string(&mut buf, "scale.f32");
        buf.extend_from_slice(&T_FLOAT32.to_le_bytes());
        buf.extend_from_slice(&2.5f32.to_le_bytes());

        // KV: bool
        put_string(&mut buf, "flag.bool");
        buf.extend_from_slice(&T_BOOL.to_le_bytes());
        buf.push(1);

        // KV: array of strings
        put_string(&mut buf, "names.arr");
        buf.extend_from_slice(&T_ARRAY.to_le_bytes());
        buf.extend_from_slice(&T_STRING.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        for s in ["alpha", "beta", "gamma"] {
            put_string(&mut buf, s);
        }

        // KV: array of i32
        put_string(&mut buf, "types.arr");
        buf.extend_from_slice(&T_ARRAY.to_le_bytes());
        buf.extend_from_slice(&T_INT32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        for x in [-7i32, 0, 9] {
            buf.extend_from_slice(&x.to_le_bytes());
        }

        // Tensor data offsets (relative to data start, each padded to ALIGN).
        let off1 = 0usize;
        let off2 = (off1 + t1.len() * 4).div_ceil(ALIGN) * ALIGN;

        // Tensor infos.
        put_string(&mut buf, "tensor.one");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&(off1 as u64).to_le_bytes());

        put_string(&mut buf, "tensor.two");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&(off2 as u64).to_le_bytes());

        // Pad to data section.
        while buf.len() % ALIGN != 0 {
            buf.push(0);
        }
        let data_start = buf.len();

        // Write each tensor at data_start + offset.
        for (off, data) in [(off1, t1), (off2, t2)] {
            let target = data_start + off;
            while buf.len() < target {
                buf.push(0);
            }
            for &x in data {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        buf
    }

    #[test]
    fn round_trips_metadata_and_tensors() {
        let bytes = build_gguf();
        let mut path = std::env::temp_dir();
        path.push(format!("talos_gguf_test_{}.gguf", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
            f.flush().unwrap();
        }

        let g = GgufFile::open(&path).unwrap();

        // Metadata accessors.
        assert_eq!(g.get_u32("answer.u32"), Some(42));
        assert_eq!(g.get_f32("scale.f32"), Some(2.5));
        assert_eq!(
            g.get_arr_str("names.arr"),
            Some(["alpha".to_string(), "beta".to_string(), "gamma".to_string()].as_slice())
        );
        assert_eq!(g.get_arr_i32("types.arr"), Some([-7i32, 0, 9].as_slice()));
        assert!(matches!(g.metadata("flag.bool"), Some(MetaValue::Bool(true))));
        // Wrong-type access returns None.
        assert_eq!(g.get_u32("scale.f32"), None);
        assert_eq!(g.get_str("missing"), None);

        // Tensor index.
        let tensors = g.tensors();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "tensor.one");
        assert_eq!(tensors[0].dims, vec![3, 2]);
        assert_eq!(tensors[0].dtype, GgmlType::F32);
        assert_eq!(tensors[0].elem_count(), 6);
        assert_eq!(tensors[1].name, "tensor.two");
        assert_eq!(tensors[1].dims, vec![3]);

        assert!(g.tensor_info("nope").is_none());

        // Tensor data is exact.
        assert_eq!(g.tensor_f32("tensor.one").unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(g.tensor_f32("tensor.two").unwrap(), &[-1.5, 0.0, 42.25]);
        assert!(g.tensor_f32("missing").is_err());

        std::fs::remove_file(&path).ok();
    }
}
