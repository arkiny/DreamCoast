//! `.dcasset` — a cooked-asset container (Phase 12 M1).
//!
//! A `MeshData` (geometry + material factors + textures) is serialized into one
//! self-describing binary so the runtime can load it directly instead of
//! re-parsing glTF and re-decoding textures every launch.
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
//! format extensible: later payloads (SDF volumes in M2, BVH, lightmaps) attach
//! as new chunk types without breaking older readers, which skip unknown types.

use dreamcoast_core::EngineError;

use crate::{ImageData, Material, MeshData, MeshVertex};

/// Container magic. The trailing NUL keeps it a fixed 8 bytes and ASCII-greppable.
pub const MAGIC: [u8; 8] = *b"DCASSET\0";

/// Format version. **Bump on any layout change** — the loader treats a mismatch as
/// a cache miss and re-cooks, so an old `.dcasset` is never misread as a new one.
pub const VERSION: u32 = 1;

// Chunk type tags (directory `type` field). Stable across versions; new payloads
// get new tags. Unknown tags are skipped by the reader (forward compatibility).
const CHUNK_MESH: u32 = 1;
const CHUNK_TEXTURE: u32 = 2;

// Texture slot tags (texture chunk `slot` field) — which `Material` field a decoded
// image fills. Kept distinct from chunk tags so the two namespaces never collide.
const TEX_BASE_COLOR: u32 = 0;
const TEX_METALLIC_ROUGHNESS: u32 = 1;
const TEX_NORMAL: u32 = 2;
const TEX_EMISSIVE: u32 = 3;

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

/// Hash of the cook parameters that affect the produced bytes — the other half of
/// the invalidation key. **Single source of truth:** every knob that changes the
/// output is folded in here so a parameter change invalidates the cache. The mesh
/// cook is parameter-free for M1, so this currently folds only the format version
/// (a belt-and-suspenders alongside the header `version` field); M2's SDF
/// resolution and friends join here.
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

// --- little-endian writer ---------------------------------------------------

/// Append-only little-endian byte sink. Keeps serialization explicit and
/// platform-independent (no `repr`-punning), which is what makes the cook
/// deterministic and cross-backend byte-identical.
#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
}

// --- little-endian reader ---------------------------------------------------

/// Bounds-checked little-endian cursor. Every read validates length and returns a
/// descriptive [`EngineError::Asset`] on truncation, so a corrupt `.dcasset`
/// degrades to a cache miss (re-cook) rather than a panic or a wrong read.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| EngineError::Asset("dcasset: unexpected end of data".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, EngineError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

// --- chunk encoders ---------------------------------------------------------

/// Encode the mesh chunk payload: material factors followed by the vertex and
/// index arrays. Textures live in their own chunks (see [`encode_texture`]).
fn encode_mesh(mesh: &MeshData) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(mesh.vertices.len() as u32);
    w.u32(mesh.indices.len() as u32);

    let m = &mesh.material;
    for c in m.base_color_factor {
        w.f32(c);
    }
    w.f32(m.metallic_factor);
    w.f32(m.roughness_factor);
    for c in m.emissive_factor {
        w.f32(c);
    }

    for v in &mesh.vertices {
        for c in v.pos {
            w.f32(c);
        }
        for c in v.normal {
            w.f32(c);
        }
        for c in v.uv {
            w.f32(c);
        }
    }
    for &i in &mesh.indices {
        w.u32(i);
    }
    w.buf
}

/// Encode one texture chunk payload: `slot`, dimensions, then RGBA8 pixels.
fn encode_texture(slot: u32, img: &ImageData) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(slot);
    w.u32(img.width);
    w.u32(img.height);
    w.bytes(&img.rgba8);
    w.buf
}

/// Collect every chunk for `mesh` as `(type, payload)` pairs, in write order.
/// Textures are emitted only when present, each as its own slot-tagged chunk.
fn collect_chunks(mesh: &MeshData) -> Vec<(u32, Vec<u8>)> {
    let mut chunks = vec![(CHUNK_MESH, encode_mesh(mesh))];
    let m = &mesh.material;
    for (slot, tex) in [
        (TEX_BASE_COLOR, &m.base_color),
        (TEX_METALLIC_ROUGHNESS, &m.metallic_roughness),
        (TEX_NORMAL, &m.normal),
        (TEX_EMISSIVE, &m.emissive),
    ] {
        if let Some(img) = tex {
            chunks.push((CHUNK_TEXTURE, encode_texture(slot, img)));
        }
    }
    chunks
}

// --- public API -------------------------------------------------------------

/// Header byte size: magic(8) + version(4) + flags(4) + source_hash(8) +
/// cook_params_hash(8) + chunk_count(4).
const HEADER_SIZE: usize = 8 + 4 + 4 + 8 + 8 + 4;
/// Directory entry byte size: type(4) + offset(8) + size(8).
const DIR_ENTRY_SIZE: usize = 4 + 8 + 8;

/// Serialize `mesh` into a `.dcasset` byte buffer. `src_hash` is the
/// [`source_hash`] of the source asset (glTF bytes), embedded for invalidation.
pub fn write(mesh: &MeshData, src_hash: u64) -> Vec<u8> {
    let chunks = collect_chunks(mesh);
    let dir_size = DIR_ENTRY_SIZE * chunks.len();
    let payload_start = HEADER_SIZE + dir_size;

    let mut w = Writer::default();
    // Header.
    w.bytes(&MAGIC);
    w.u32(VERSION);
    w.u32(0); // flags (reserved)
    w.u64(src_hash);
    w.u64(cook_params_hash());
    w.u32(chunks.len() as u32);

    // Directory: type/offset/size, offsets relative to file start.
    let mut offset = payload_start as u64;
    for (ty, payload) in &chunks {
        w.u32(*ty);
        w.u64(offset);
        w.u64(payload.len() as u64);
        offset += payload.len() as u64;
    }

    // Payloads, in the same order as the directory.
    for (_, payload) in &chunks {
        w.bytes(payload);
    }
    w.buf
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

/// Decode a `.dcasset` buffer into its [`Header`] and [`MeshData`]. Unknown chunk
/// types are skipped (forward compatibility). Returns an error on a bad magic,
/// truncation, or a missing mesh chunk.
pub fn read(bytes: &[u8]) -> Result<(Header, MeshData), EngineError> {
    let header = read_header(bytes)?;

    // Re-read past the fixed header to the directory.
    let mut r = Reader::at(bytes, HEADER_SIZE - 4);
    let chunk_count = r.u32()?;

    // Directory entries.
    let mut dir = Vec::with_capacity(chunk_count as usize);
    for _ in 0..chunk_count {
        let ty = r.u32()?;
        let offset = r.u64()? as usize;
        let size = r.u64()? as usize;
        dir.push((ty, offset, size));
    }

    let mut mesh: Option<(Vec<MeshVertex>, Vec<u32>, Material)> = None;
    let mut pending_textures: Vec<(u32, ImageData)> = Vec::new();

    for (ty, offset, size) in dir {
        // Validate the slice the directory points at before reading it.
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| EngineError::Asset("dcasset: chunk out of bounds".into()))?;
        let mut cr = Reader::at(&bytes[..end], offset);
        match ty {
            CHUNK_MESH => mesh = Some(decode_mesh(&mut cr)?),
            CHUNK_TEXTURE => pending_textures.push(decode_texture(&mut cr)?),
            _ => {} // unknown chunk: skip (forward compatibility)
        }
    }

    let (vertices, indices, mut material) =
        mesh.ok_or_else(|| EngineError::Asset("dcasset: missing mesh chunk".into()))?;
    for (slot, img) in pending_textures {
        match slot {
            TEX_BASE_COLOR => material.base_color = Some(img),
            TEX_METALLIC_ROUGHNESS => material.metallic_roughness = Some(img),
            TEX_NORMAL => material.normal = Some(img),
            TEX_EMISSIVE => material.emissive = Some(img),
            _ => {} // unknown slot: ignore
        }
    }

    Ok((
        header,
        MeshData {
            vertices,
            indices,
            material,
        },
    ))
}

/// Decode a mesh chunk payload (factors + geometry). Textures are merged in by the
/// caller from separate chunks.
fn decode_mesh(r: &mut Reader) -> Result<(Vec<MeshVertex>, Vec<u32>, Material), EngineError> {
    let vtx_count = r.u32()? as usize;
    let idx_count = r.u32()? as usize;

    let base_color_factor = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
    let metallic_factor = r.f32()?;
    let roughness_factor = r.f32()?;
    let emissive_factor = [r.f32()?, r.f32()?, r.f32()?];

    let mut vertices = Vec::with_capacity(vtx_count);
    for _ in 0..vtx_count {
        vertices.push(MeshVertex {
            pos: [r.f32()?, r.f32()?, r.f32()?],
            normal: [r.f32()?, r.f32()?, r.f32()?],
            uv: [r.f32()?, r.f32()?],
        });
    }
    let mut indices = Vec::with_capacity(idx_count);
    for _ in 0..idx_count {
        indices.push(r.u32()?);
    }

    let material = Material {
        base_color_factor,
        metallic_factor,
        roughness_factor,
        emissive_factor,
        ..Material::default()
    };
    Ok((vertices, indices, material))
}

/// Decode a texture chunk payload into its slot tag and RGBA8 image.
fn decode_texture(r: &mut Reader) -> Result<(u32, ImageData), EngineError> {
    let slot = r.u32()?;
    let width = r.u32()?;
    let height = r.u32()?;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| EngineError::Asset("dcasset: texture size overflow".into()))?;
    let rgba8 = r.take(expected)?.to_vec();
    Ok((
        slot,
        ImageData {
            width,
            height,
            rgba8,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small mesh with non-trivial material factors but no textures (M1.1 scope).
    fn sample_mesh() -> MeshData {
        MeshData {
            vertices: vec![
                MeshVertex {
                    pos: [1.0, 2.0, 3.0],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.25, 0.75],
                },
                MeshVertex {
                    pos: [-1.5, 0.0, 4.25],
                    normal: [1.0, 0.0, 0.0],
                    uv: [1.0, 0.0],
                },
                MeshVertex {
                    pos: [0.5, -2.5, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.5, 0.5],
                },
            ],
            indices: vec![0, 1, 2],
            material: Material {
                base_color_factor: [0.2, 0.4, 0.6, 1.0],
                metallic_factor: 0.3,
                roughness_factor: 0.7,
                emissive_factor: [0.1, 0.0, 0.05],
                ..Material::default()
            },
        }
    }

    fn assert_mesh_eq(a: &MeshData, b: &MeshData) {
        assert_eq!(a.vertices.len(), b.vertices.len(), "vertex count");
        for (x, y) in a.vertices.iter().zip(&b.vertices) {
            assert_eq!(x.pos, y.pos);
            assert_eq!(x.normal, y.normal);
            assert_eq!(x.uv, y.uv);
        }
        assert_eq!(a.indices, b.indices, "indices");
        let (ma, mb) = (&a.material, &b.material);
        assert_eq!(ma.base_color_factor, mb.base_color_factor);
        assert_eq!(ma.metallic_factor, mb.metallic_factor);
        assert_eq!(ma.roughness_factor, mb.roughness_factor);
        assert_eq!(ma.emissive_factor, mb.emissive_factor);
    }

    #[test]
    fn roundtrip_geometry_and_factors() {
        let mesh = sample_mesh();
        let bytes = write(&mesh, 0xdead_beef);
        let (header, decoded) = read(&bytes).expect("decode");
        assert_eq!(header.version, VERSION);
        assert_eq!(header.source_hash, 0xdead_beef);
        assert_eq!(header.cook_params_hash, cook_params_hash());
        assert_mesh_eq(&mesh, &decoded);
    }

    #[test]
    fn cook_is_deterministic() {
        // Two cooks of the same input must be byte-identical (cross-backend gate).
        let mesh = sample_mesh();
        assert_eq!(write(&mesh, 7), write(&mesh, 7));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = write(&sample_mesh(), 0);
        bytes[0] = b'X';
        assert!(read(&bytes).is_err());
        assert!(read_header(&bytes).is_err());
    }

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let bytes = write(&sample_mesh(), 0);
        // Lop off the tail: the directory promises more than the buffer holds.
        assert!(read(&bytes[..bytes.len() - 8]).is_err());
    }
}
