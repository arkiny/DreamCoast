//! Mesh + texture chunk codec — the core `.dcasset` payload (M1).

use dreamcoast_core::EngineError;

use super::{
    CHUNK_MESH, CHUNK_TEXTURE, DIR_ENTRY_SIZE, HEADER_SIZE, Header, MAGIC, Reader, VERSION, Writer,
    cook_params_hash, read_directory, read_header,
};
use crate::bc::BcFormat;
use crate::{ImageData, Material, MeshData, MeshVertex, TexData};

// Texture slot tags (texture chunk `slot` field) — which `Material` field a decoded
// image fills. Kept distinct from chunk tags so the two namespaces never collide.
const TEX_BASE_COLOR: u32 = 0;
const TEX_METALLIC_ROUGHNESS: u32 = 1;
const TEX_NORMAL: u32 = 2;
const TEX_EMISSIVE: u32 = 3;

// Texture-encoding kinds (the `kind` field of a texture chunk).
const TEX_KIND_RGBA8: u32 = 0;
const TEX_KIND_BC: u32 = 1;
// BcFormat tags (stored in a BC texture chunk).
const BC_FMT_BC1: u32 = 0;
const BC_FMT_BC5: u32 = 1;
const BC_FMT_BC3: u32 = 2;
const BC_FMT_BC4: u32 = 3;
const BC_FMT_BC7: u32 = 4;

/// Serialize `mesh` into a `.dcasset` byte buffer. `src_hash` is the
/// [`super::source_hash`] of the source asset (glTF bytes), embedded for invalidation.
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

/// Decode a `.dcasset` buffer into its [`Header`] and [`MeshData`]. Unknown chunk
/// types are skipped (forward compatibility). Returns an error on a bad magic,
/// truncation, or a missing mesh chunk.
pub fn read(bytes: &[u8]) -> Result<(Header, MeshData), EngineError> {
    let header = read_header(bytes)?;
    let dir = read_directory(bytes)?;

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
                BcFormat::Bc7 => BC_FMT_BC7,
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

/// Decode a texture chunk payload into its slot tag and texture data.
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
                BC_FMT_BC7 => BcFormat::Bc7,
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

    /// A small mesh with non-trivial material factors but no textures.
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

    fn rgba8(tex: &TexData) -> &ImageData {
        match tex {
            TexData::Rgba8(im) => im,
            TexData::Bc { .. } => panic!("expected uncompressed texture"),
        }
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
