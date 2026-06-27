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

use crate::bc::BcFormat;
use crate::sdf::{AlbedoVolumes, SdfVolume};
use crate::{ImageData, Material, MeshData, MeshVertex, TexData};

/// Container magic. The trailing NUL keeps it a fixed 8 bytes and ASCII-greppable.
pub const MAGIC: [u8; 8] = *b"DCASSET\0";

/// Format version. **Bump on any layout change** — the loader treats a mismatch as
/// a cache miss and re-cooks, so an old `.dcasset` is never misread as a new one.
pub const VERSION: u32 = 1;

// Chunk type tags (directory `type` field). Stable across versions; new payloads
// get new tags. Unknown tags are skipped by the reader (forward compatibility).
const CHUNK_MESH: u32 = 1;
const CHUNK_TEXTURE: u32 = 2;
/// SDF volume chunk: `dim`, `aabb_min`/`aabb_max`, then `dim³` R32F voxels (M2).
const CHUNK_SDF: u32 = 3;
/// Albedo volumes chunk: `dim`, then three `dim³` R32F channels (R,G,B) (M2 ext).
const CHUNK_ALBEDO: u32 = 4;

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

// Texture-encoding kinds (the `kind` field of a texture chunk).
const TEX_KIND_RGBA8: u32 = 0;
const TEX_KIND_BC: u32 = 1;
// BcFormat tags (stored in a BC texture chunk).
const BC_FMT_BC1: u32 = 0;
const BC_FMT_BC5: u32 = 1;
const BC_FMT_BC3: u32 = 2;
const BC_FMT_BC4: u32 = 3;

/// Encode one texture chunk payload: `slot`, a kind tag, dimensions, then either
/// RGBA8 pixels or the BCn block mips (Phase 12 M3).
fn encode_texture(slot: u32, tex: &TexData) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(slot);
    match tex {
        TexData::Rgba8(img) => {
            w.u32(TEX_KIND_RGBA8);
            w.u32(img.width);
            w.u32(img.height);
            w.bytes(&img.rgba8);
        }
        TexData::Bc {
            format,
            srgb,
            width,
            height,
            mips,
        } => {
            w.u32(TEX_KIND_BC);
            w.u32(width.to_owned());
            w.u32(height.to_owned());
            w.u32(match format {
                BcFormat::Bc1 => BC_FMT_BC1,
                BcFormat::Bc5 => BC_FMT_BC5,
                BcFormat::Bc3 => BC_FMT_BC3,
                BcFormat::Bc4 => BC_FMT_BC4,
            });
            w.u32(u32::from(*srgb));
            w.u32(mips.len() as u32);
            for mip in mips {
                w.u32(mip.len() as u32);
                w.bytes(mip);
            }
        }
    }
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
    let mut pending_textures: Vec<(u32, TexData)> = Vec::new();

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
    for (slot, tex) in pending_textures {
        match slot {
            TEX_BASE_COLOR => material.base_color = Some(tex),
            TEX_METALLIC_ROUGHNESS => material.metallic_roughness = Some(tex),
            TEX_NORMAL => material.normal = Some(tex),
            TEX_EMISSIVE => material.emissive = Some(tex),
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

/// Serialize an SDF volume into a `.dcasset` byte buffer (a container with a single
/// SDF chunk — no mesh). `src_hash` is the invalidation key the loader compares
/// against the live source (here, the fused-geometry + grid hash).
pub fn write_sdf(vol: &SdfVolume, src_hash: u64) -> Vec<u8> {
    let payload = encode_sdf(vol);
    let payload_start = HEADER_SIZE + DIR_ENTRY_SIZE; // one chunk

    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.u32(VERSION);
    w.u32(0);
    w.u64(src_hash);
    w.u64(cook_params_hash());
    w.u32(1); // chunk_count
    w.u32(CHUNK_SDF);
    w.u64(payload_start as u64);
    w.u64(payload.len() as u64);
    w.bytes(&payload);
    w.buf
}

/// Decode a `.dcasset` buffer's SDF chunk into its [`Header`] and [`SdfVolume`].
/// Errors on a bad magic, truncation, or a missing SDF chunk.
pub fn read_sdf(bytes: &[u8]) -> Result<(Header, SdfVolume), EngineError> {
    let header = read_header(bytes)?;
    let mut r = Reader::at(bytes, HEADER_SIZE - 4);
    let chunk_count = r.u32()?;
    let mut dir = Vec::with_capacity(chunk_count as usize);
    for _ in 0..chunk_count {
        let ty = r.u32()?;
        let offset = r.u64()? as usize;
        let size = r.u64()? as usize;
        dir.push((ty, offset, size));
    }
    for (ty, offset, size) in dir {
        if ty != CHUNK_SDF {
            continue;
        }
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| EngineError::Asset("dcasset: sdf chunk out of bounds".into()))?;
        let mut cr = Reader::at(&bytes[..end], offset);
        return Ok((header, decode_sdf(&mut cr)?));
    }
    Err(EngineError::Asset("dcasset: missing sdf chunk".into()))
}

/// Serialize the per-voxel albedo volumes into a `.dcasset` (single albedo chunk).
/// `src_hash` is the invalidation key (the fused-geometry + per-triangle albedo +
/// grid hash).
pub fn write_albedo(vol: &AlbedoVolumes, src_hash: u64) -> Vec<u8> {
    let payload = encode_albedo(vol);
    let payload_start = HEADER_SIZE + DIR_ENTRY_SIZE;
    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.u32(VERSION);
    w.u32(0);
    w.u64(src_hash);
    w.u64(cook_params_hash());
    w.u32(1);
    w.u32(CHUNK_ALBEDO);
    w.u64(payload_start as u64);
    w.u64(payload.len() as u64);
    w.bytes(&payload);
    w.buf
}

/// Decode a `.dcasset` buffer's albedo chunk into its [`Header`] and [`AlbedoVolumes`].
pub fn read_albedo(bytes: &[u8]) -> Result<(Header, AlbedoVolumes), EngineError> {
    let header = read_header(bytes)?;
    let mut r = Reader::at(bytes, HEADER_SIZE - 4);
    let chunk_count = r.u32()?;
    let mut dir = Vec::with_capacity(chunk_count as usize);
    for _ in 0..chunk_count {
        let ty = r.u32()?;
        let offset = r.u64()? as usize;
        let size = r.u64()? as usize;
        dir.push((ty, offset, size));
    }
    for (ty, offset, size) in dir {
        if ty != CHUNK_ALBEDO {
            continue;
        }
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| EngineError::Asset("dcasset: albedo chunk out of bounds".into()))?;
        let mut cr = Reader::at(&bytes[..end], offset);
        return Ok((header, decode_albedo(&mut cr)?));
    }
    Err(EngineError::Asset("dcasset: missing albedo chunk".into()))
}

/// Encode the albedo chunk payload: `dim`, then the R, G, B channels in order.
fn encode_albedo(vol: &AlbedoVolumes) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(vol.dim);
    for ch in &vol.channels {
        for &v in ch {
            w.f32(v);
        }
    }
    w.buf
}

/// Decode an albedo chunk payload (`dim` + three `dim³` channels).
fn decode_albedo(r: &mut Reader) -> Result<AlbedoVolumes, EngineError> {
    let dim = r.u32()?;
    let count = (dim as usize)
        .checked_pow(3)
        .ok_or_else(|| EngineError::Asset("dcasset: albedo dim overflow".into()))?;
    let mut channels = [
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
    ];
    for ch in &mut channels {
        for _ in 0..count {
            ch.push(r.f32()?);
        }
    }
    Ok(AlbedoVolumes { dim, channels })
}

/// Encode the SDF chunk payload: `dim`, `aabb_min`, `aabb_max`, then the voxels.
fn encode_sdf(vol: &SdfVolume) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(vol.dim);
    for c in vol.aabb_min {
        w.f32(c);
    }
    for c in vol.aabb_max {
        w.f32(c);
    }
    for &v in &vol.voxels {
        w.f32(v);
    }
    w.buf
}

/// Decode an SDF chunk payload, validating the voxel count against `dim³`.
fn decode_sdf(r: &mut Reader) -> Result<SdfVolume, EngineError> {
    let dim = r.u32()?;
    let aabb_min = [r.f32()?, r.f32()?, r.f32()?];
    let aabb_max = [r.f32()?, r.f32()?, r.f32()?];
    let count = (dim as usize)
        .checked_pow(3)
        .ok_or_else(|| EngineError::Asset("dcasset: sdf dim overflow".into()))?;
    let mut voxels = Vec::with_capacity(count);
    for _ in 0..count {
        voxels.push(r.f32()?);
    }
    Ok(SdfVolume {
        dim,
        aabb_min,
        aabb_max,
        voxels,
    })
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
fn decode_texture(r: &mut Reader) -> Result<(u32, TexData), EngineError> {
    let slot = r.u32()?;
    let kind = r.u32()?;
    let width = r.u32()?;
    let height = r.u32()?;
    let tex = match kind {
        TEX_KIND_RGBA8 => {
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| EngineError::Asset("dcasset: texture size overflow".into()))?;
            let rgba8 = r.take(expected)?.to_vec();
            TexData::Rgba8(ImageData {
                width,
                height,
                rgba8,
            })
        }
        TEX_KIND_BC => {
            let format = match r.u32()? {
                BC_FMT_BC1 => BcFormat::Bc1,
                BC_FMT_BC5 => BcFormat::Bc5,
                BC_FMT_BC3 => BcFormat::Bc3,
                BC_FMT_BC4 => BcFormat::Bc4,
                other => {
                    return Err(EngineError::Asset(format!(
                        "dcasset: unknown bc format {other}"
                    )));
                }
            };
            let srgb = r.u32()? != 0;
            let mip_count = r.u32()? as usize;
            let mut mips = Vec::with_capacity(mip_count);
            for _ in 0..mip_count {
                let len = r.u32()? as usize;
                mips.push(r.take(len)?.to_vec());
            }
            TexData::Bc {
                format,
                srgb,
                width,
                height,
                mips,
            }
        }
        other => {
            return Err(EngineError::Asset(format!(
                "dcasset: unknown texture kind {other}"
            )));
        }
    };
    Ok((slot, tex))
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

    fn rgba8(tex: &TexData) -> &ImageData {
        match tex {
            TexData::Rgba8(im) => im,
            TexData::Bc { .. } => panic!("expected uncompressed texture"),
        }
    }

    #[test]
    fn roundtrip_with_textures() {
        // A 2x1 base-color + a 1x1 normal texture; the other two slots stay None.
        let mut mesh = sample_mesh();
        mesh.material.base_color = Some(TexData::Rgba8(ImageData {
            width: 2,
            height: 1,
            rgba8: vec![10, 20, 30, 40, 50, 60, 70, 80],
        }));
        mesh.material.normal = Some(TexData::Rgba8(ImageData {
            width: 1,
            height: 1,
            rgba8: vec![128, 128, 255, 255],
        }));

        let bytes = write(&mesh, 1);
        let (_, decoded) = read(&bytes).expect("decode");
        assert_mesh_eq(&mesh, &decoded);

        let bc = rgba8(decoded.material.base_color.as_ref().expect("base_color"));
        assert_eq!((bc.width, bc.height), (2, 1));
        assert_eq!(bc.rgba8, vec![10, 20, 30, 40, 50, 60, 70, 80]);
        let n = rgba8(decoded.material.normal.as_ref().expect("normal"));
        assert_eq!((n.width, n.height), (1, 1));
        assert_eq!(n.rgba8, vec![128, 128, 255, 255]);
        // Slots that were None must round-trip back to None (no stray chunks).
        assert!(decoded.material.metallic_roughness.is_none());
        assert!(decoded.material.emissive.is_none());
    }

    #[test]
    fn roundtrip_compressed_texture() {
        // A BC-compressed base-color slot (2 mips) must round-trip byte-exact.
        let mut mesh = sample_mesh();
        mesh.material.base_color = Some(TexData::Bc {
            format: BcFormat::Bc1,
            srgb: true,
            width: 8,
            height: 8,
            mips: vec![vec![1u8; 8 * 2 * 2], vec![2u8; 8]],
        });
        let bytes = write(&mesh, 1);
        let (_, decoded) = read(&bytes).expect("decode");
        match decoded.material.base_color.expect("base_color") {
            TexData::Bc {
                format,
                srgb,
                width,
                height,
                mips,
            } => {
                assert_eq!(format, BcFormat::Bc1);
                assert!(srgb);
                assert_eq!((width, height), (8, 8));
                assert_eq!(mips.len(), 2);
                assert_eq!(mips[0], vec![1u8; 32]);
                assert_eq!(mips[1], vec![2u8; 8]);
            }
            TexData::Rgba8(_) => panic!("expected compressed"),
        }
    }

    #[test]
    fn only_present_textures_become_chunks() {
        // No textures -> exactly one chunk (the mesh); the chunk_count u32 sits at
        // the end of the fixed header.
        let bytes = write(&sample_mesh(), 0);
        let count = u32::from_le_bytes(bytes[HEADER_SIZE - 4..HEADER_SIZE].try_into().unwrap());
        assert_eq!(count, 1);
    }

    #[test]
    fn sdf_chunk_roundtrip() {
        let vol = SdfVolume {
            dim: 2,
            aabb_min: [-1.0, 0.0, 0.5],
            aabb_max: [1.0, 2.0, 1.5],
            voxels: vec![-0.3, 0.1, 0.0, 0.7, -0.5, 0.2, 0.9, -0.1],
        };
        let bytes = write_sdf(&vol, 0xabc);
        let (header, decoded) = read_sdf(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0xabc);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.aabb_min, vol.aabb_min);
        assert_eq!(decoded.aabb_max, vol.aabb_max);
        assert_eq!(decoded.voxels, vol.voxels);
    }

    #[test]
    fn albedo_chunk_roundtrip() {
        let vol = AlbedoVolumes {
            dim: 2,
            channels: [
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            ],
        };
        let bytes = write_albedo(&vol, 0x5151);
        let (header, decoded) = read_albedo(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0x5151);
        assert_eq!(decoded.dim, 2);
        assert_eq!(decoded.channels, vol.channels);
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
