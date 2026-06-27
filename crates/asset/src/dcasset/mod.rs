//! `.dcasset` — a cooked-asset container (Phase 12).
//!
//! A cooked asset (mesh + material + textures, an SDF / albedo volume, or a scene
//! level) is serialized into one self-describing binary so the runtime can load it
//! directly instead of re-parsing glTF / re-baking on every launch.
//!
//! ## Layout (manual little-endian; a chunk-directory container)
//!
//! ```text
//! [Header]  magic[8]="DCASSET\0" | version u32 | flags u32
//!           source_hash u64 | cook_params_hash u64 | chunk_count u32
//! [Dir]     chunk_count × { type u32, offset u64, size u64 }   // offset = file start
//! [Payload] chunks…
//! ```
//!
//! Every field is written explicit-little-endian and the cook is pure CPU, so the
//! bytes are **backend-independent and deterministic** — two cooks of the same
//! input produce byte-identical output, and the same `.dcasset` is produced on
//! Vulkan, D3D12, or Metal. The chunk directory (type/offset/size) keeps the
//! format extensible: new payloads attach as new chunk types without breaking
//! older readers, which skip unknown types.
//!
//! This module is the container core (header, the LE [`Writer`]/[`Reader`], the
//! invalidation-key hashing); the per-payload chunk codecs live in [`mesh`]
//! (mesh + textures), [`volume`] (SDF + albedo), and [`level`] (scene/level).

mod level;
mod mesh;
mod volume;

pub use level::{read_level, write_level};
pub use mesh::{read, write};
pub use volume::{read_albedo, read_sdf, write_albedo, write_sdf};

use dreamcoast_core::EngineError;

/// Container magic. The trailing NUL keeps it a fixed 8 bytes and ASCII-greppable.
pub const MAGIC: [u8; 8] = *b"DCASSET\0";

/// Format version. **Bump on any layout change** — the loader treats a mismatch as
/// a cache miss and re-cooks, so an old `.dcasset` is never misread as a new one.
pub const VERSION: u32 = 1;

// Chunk type tags (directory `type` field). Stable across versions; new payloads
// get new tags. Unknown tags are skipped by the readers (forward compatibility).
pub(crate) const CHUNK_MESH: u32 = 1;
pub(crate) const CHUNK_TEXTURE: u32 = 2;
/// SDF volume chunk: `dim`, `aabb_min`/`aabb_max`, then `dim³` R32F voxels (M2).
pub(crate) const CHUNK_SDF: u32 = 3;
/// Albedo volumes chunk: `dim`, then three `dim³` R32F channels (R,G,B) (M2 ext).
pub(crate) const CHUNK_ALBEDO: u32 = 4;
/// Level / scene chunk: entities + lights + camera + environment (item 2).
pub(crate) const CHUNK_LEVEL: u32 = 5;

/// Header byte size: magic(8) + version(4) + flags(4) + source_hash(8) +
/// cook_params_hash(8) + chunk_count(4).
pub(crate) const HEADER_SIZE: usize = 8 + 4 + 4 + 8 + 8 + 4;
/// Directory entry byte size: type(4) + offset(8) + size(8).
pub(crate) const DIR_ENTRY_SIZE: usize = 4 + 8 + 8;

// FNV-1a 64-bit — a dependency-free content hash, mirroring the constants the shader
// cook cache uses (`crates/shader/build.rs`, Phase 12 M4). A build script cannot be
// shared as a library, so the constants are re-stated here; this is the single
// definition for the asset crate. Collision risk is irrelevant for a cache key.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into the running FNV-1a hash `h` (seed with [`FNV_OFFSET`]).
fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Content hash of the source asset bytes — half of the invalidation key.
pub fn source_hash(bytes: &[u8]) -> u64 {
    fnv1a(bytes, FNV_OFFSET)
}

/// Seed for an incremental content hash; fold parts in with [`hash_update`]. Use
/// for assets whose identity spans several buffers (e.g. the scene SDF keyed on its
/// fused geometry + grid dims + AABB) so the key never has to concatenate them.
pub fn hash_begin() -> u64 {
    FNV_OFFSET
}

/// Fold `bytes` into the running hash `h`.
pub fn hash_update(h: u64, bytes: &[u8]) -> u64 {
    fnv1a(bytes, h)
}

/// Hash of the cook parameters that affect the produced bytes — the other half of
/// the invalidation key. **Single source of truth:** every knob that changes the
/// output is folded in here so a parameter change invalidates the cache. Currently
/// folds the format version (a belt-and-suspenders alongside the header `version`).
pub fn cook_params_hash() -> u64 {
    fnv1a(&VERSION.to_le_bytes(), FNV_OFFSET)
}

/// Parsed container header. The loader compares these against the live source to
/// decide hit vs. miss (see the cook orchestration in [`crate::cook`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub source_hash: u64,
    pub cook_params_hash: u64,
}

/// Parse just the header of a `.dcasset` buffer (cheap hit/miss check before a
/// full decode). Verifies the magic; does not validate `source_hash` (the caller
/// compares that against the live source).
pub fn read_header(bytes: &[u8]) -> Result<Header, EngineError> {
    let mut r = Reader::new(bytes);
    if r.take(8)? != MAGIC {
        return Err(EngineError::Asset("dcasset: bad magic".into()));
    }
    let version = r.u32()?;
    let _flags = r.u32()?;
    let source_hash = r.u64()?;
    let cook_params_hash = r.u64()?;
    Ok(Header {
        version,
        source_hash,
        cook_params_hash,
    })
}

/// Read a chunk directory after the fixed header. Returns the `(type, offset, size)`
/// entries; shared by every `read_*`.
pub(crate) fn read_directory(bytes: &[u8]) -> Result<Vec<(u32, usize, usize)>, EngineError> {
    let mut r = Reader::at(bytes, HEADER_SIZE - 4);
    let chunk_count = r.u32()?;
    let mut dir = Vec::with_capacity(chunk_count as usize);
    for _ in 0..chunk_count {
        let ty = r.u32()?;
        let offset = r.u64()? as usize;
        let size = r.u64()? as usize;
        dir.push((ty, offset, size));
    }
    Ok(dir)
}

/// Frame a single-chunk container (the SDF / albedo / level assets): header + a
/// one-entry directory + the payload. Shared by their `write_*`.
pub(crate) fn write_single_chunk(chunk_type: u32, payload: &[u8], src_hash: u64) -> Vec<u8> {
    let payload_start = HEADER_SIZE + DIR_ENTRY_SIZE;
    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.u32(VERSION);
    w.u32(0);
    w.u64(src_hash);
    w.u64(cook_params_hash());
    w.u32(1); // chunk_count
    w.u32(chunk_type);
    w.u64(payload_start as u64);
    w.u64(payload.len() as u64);
    w.bytes(payload);
    w.buf
}

/// Find the chunk of type `chunk_type` and return a [`Reader`] positioned at its
/// payload (bounds-checked). Shared by the single-chunk `read_*`.
pub(crate) fn open_chunk<'a>(
    bytes: &'a [u8],
    chunk_type: u32,
    what: &str,
) -> Result<(Header, Reader<'a>), EngineError> {
    let header = read_header(bytes)?;
    for (ty, offset, size) in read_directory(bytes)? {
        if ty != chunk_type {
            continue;
        }
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| EngineError::Asset(format!("dcasset: {what} chunk out of bounds")))?;
        return Ok((header, Reader::at(&bytes[..end], offset)));
    }
    Err(EngineError::Asset(format!("dcasset: missing {what} chunk")))
}

// --- little-endian writer ---------------------------------------------------

/// Append-only little-endian byte sink. Keeps serialization explicit and
/// platform-independent (no `repr`-punning), which is what makes the cook
/// deterministic and cross-backend byte-identical.
#[derive(Default)]
pub(crate) struct Writer {
    pub(crate) buf: Vec<u8>,
}

impl Writer {
    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub(crate) fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub(crate) fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    /// A length-prefixed UTF-8 string (`u32` byte length + bytes).
    pub(crate) fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

// --- little-endian reader ---------------------------------------------------

/// Bounds-checked little-endian cursor. Every read validates length and returns a
/// descriptive [`EngineError::Asset`] on truncation, so a corrupt `.dcasset`
/// degrades to a cache miss (re-cook) rather than a panic or a wrong read.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(crate) fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| EngineError::Asset("dcasset: unexpected end of data".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub(crate) fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    /// A length-prefixed UTF-8 string written by [`Writer::str`].
    pub(crate) fn str(&mut self) -> Result<String, EngineError> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| EngineError::Asset("dcasset: invalid utf-8 string".into()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn f32(&mut self) -> Result<f32, EngineError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}
