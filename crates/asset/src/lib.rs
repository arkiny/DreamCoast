//! Asset loading: glTF meshes + base-color textures, plus procedural fallbacks.
//!
//! RHI-agnostic — returns plain CPU data; the caller uploads it to the GPU.

use std::path::Path;

use dreamcoast_core::EngineError;

/// A mesh vertex (matches `rhi::VertexLayout::Mesh`: 32-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Decoded RGBA8 image.
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// CPU-side mesh ready for upload.
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub base_color: Option<ImageData>,
}

/// Load the first mesh primitive of a glTF/GLB file (positions, normals,
/// texcoords, indices) plus its base-color texture (if any).
pub fn load_gltf(path: impl AsRef<Path>) -> Result<MeshData, EngineError> {
    let (doc, buffers, images) =
        gltf::import(path).map_err(|e| EngineError::Asset(format!("gltf import: {e}")))?;

    let mesh = doc
        .meshes()
        .next()
        .ok_or_else(|| EngineError::Asset("glTF has no meshes".into()))?;
    let prim = mesh
        .primitives()
        .next()
        .ok_or_else(|| EngineError::Asset("mesh has no primitives".into()))?;

    let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| EngineError::Asset("primitive has no positions".into()))?
        .collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|n| n.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|i| i.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let vertices = positions
        .iter()
        .enumerate()
        .map(|(i, p)| MeshVertex {
            pos: *p,
            normal: normals[i],
            uv: uvs[i],
        })
        .collect();

    let base_color = prim
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .map(image_to_rgba8);

    Ok(MeshData {
        vertices,
        indices,
        base_color,
    })
}

/// Convert a decoded glTF image into RGBA8.
fn image_to_rgba8(img: &gltf::image::Data) -> ImageData {
    use gltf::image::Format;
    let (w, h) = (img.width, img.height);
    let px = &img.pixels;
    let rgba8 = match img.format {
        Format::R8G8B8A8 => px.clone(),
        Format::R8G8B8 => px
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        Format::R8G8 => px
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        Format::R8 => px.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        // Uncommon formats: fall back to opaque white so the model still draws.
        _ => vec![255u8; (w * h * 4) as usize],
    };
    ImageData {
        width: w,
        height: h,
        rgba8,
    }
}

/// A unit cube centered at the origin with per-face normals and UVs. Fallback
/// when no glTF file is available.
pub fn unit_cube() -> MeshData {
    // The 4 corner positions of each face (CCW).
    type Quad = ([f32; 3], [f32; 3], [f32; 3], [f32; 3]);
    const FACES: [Quad; 6] = [
        // +X
        (
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ),
        // -X
        (
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ),
        // +Y
        (
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ),
        // -Y
        (
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ),
        // +Z
        (
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
        // -Z
        (
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ),
    ];
    const NORMALS: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    const UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (f, face) in FACES.iter().enumerate() {
        let base = vertices.len() as u32;
        let corners = [face.0, face.1, face.2, face.3];
        for (c, pos) in corners.iter().enumerate() {
            vertices.push(MeshVertex {
                pos: *pos,
                normal: NORMALS[f],
                uv: UVS[c],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }
    MeshData {
        vertices,
        indices,
        base_color: None,
    }
}
