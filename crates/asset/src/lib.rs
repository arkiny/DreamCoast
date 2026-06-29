//! Asset loading: glTF meshes + base-color textures, plus procedural fallbacks.
//!
//! RHI-agnostic — returns plain CPU data; the caller uploads it to the GPU.

use std::path::Path;

use dreamcoast_core::EngineError;

pub mod bc;
pub mod cook;
pub mod dcasset;
pub mod gltf_scene;
pub mod level;
pub mod level_graph;
pub mod primitives;
pub mod sdf;

pub use gltf_scene::{
    ChannelData, GltfAnimation, GltfChannel, GltfMaterial, GltfNode, GltfPrimitive, GltfScene,
    Interpolation, load_gltf_scene,
};
pub use level::LevelData;
pub use level_graph::{LevelGraph, WorldChunk};
pub use primitives::{cornell_box, unit_cube, uv_sphere};

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

impl ImageData {
    /// A representative linear-space average colour (sRGB-decoded) — the RGBA8 path of
    /// [`TexData::average_linear`], exposed so the glTF/level importer can derive a
    /// constant albedo for a textured material from its raw decoded image.
    pub fn average_linear(&self) -> [f32; 3] {
        average_rgba8_linear(&self.rgba8)
    }
}

/// A material texture as carried by a cooked asset: either raw RGBA8 (mips are
/// generated at GPU upload) or **pre-cooked BCn block data** with its full mip
/// chain (Phase 12 M3). Block-compressed data is sampled GPU-natively, so it costs
/// nothing to decompress at runtime; the cook decides per slot which to use (color
/// → BC, data textures like metallic-roughness → RGBA8).
pub enum TexData {
    /// Uncompressed RGBA8, single level. The upload generates the mip chain.
    Rgba8(ImageData),
    /// Pre-cooked BCn blocks, one entry per mip (level 0 full-res).
    Bc {
        format: bc::BcFormat,
        /// Whether the colour data is sRGB-encoded (BC1 base/emissive) — picks the
        /// `_SRGB` GPU format at upload.
        srgb: bool,
        width: u32,
        height: u32,
        mips: Vec<Vec<u8>>,
    },
}

impl TexData {
    /// Image dimensions (level 0).
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            TexData::Rgba8(im) => (im.width, im.height),
            TexData::Bc { width, height, .. } => (*width, *height),
        }
    }

    /// A representative linear-space average colour — used by the GI to derive a
    /// constant albedo for a textured object. For RGBA8 it averages every texel;
    /// for BCn it decodes the **smallest mip** (already the box-filtered average) so
    /// the cost is one block, not the whole image.
    pub fn average_linear(&self) -> [f32; 3] {
        match self {
            TexData::Rgba8(im) => average_rgba8_linear(&im.rgba8),
            TexData::Bc { format, mips, .. } => {
                let Some(last) = mips.last() else {
                    return [0.0; 3];
                };
                let rgba = bc::decode_block_rgba8(*format, last);
                average_rgba8_linear(&rgba)
            }
        }
    }
}

/// Average an RGBA8 buffer's RGB channels in linear space (sRGB-decoded).
fn average_rgba8_linear(rgba8: &[u8]) -> [f32; 3] {
    let mut acc = [0f64; 3];
    let n = (rgba8.len() / 4).max(1) as f64;
    for px in rgba8.chunks_exact(4) {
        for (c, a) in acc.iter_mut().enumerate() {
            let s = px[c] as f64 / 255.0;
            *a += if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
    }
    [
        (acc[0] / n) as f32,
        (acc[1] / n) as f32,
        (acc[2] / n) as f32,
    ]
}

/// A metallic-roughness PBR material: scalar factors plus optional textures.
/// Textures that are `None` mean "use the factor". Color-space note: `base_color`
/// and `emissive` are sRGB-encoded (sample as sRGB → linear); `metallic_roughness`
/// (G=roughness, B=metallic per glTF) and `normal` carry linear data.
pub struct Material {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub base_color: Option<TexData>,
    pub metallic_roughness: Option<TexData>,
    pub normal: Option<TexData>,
    pub emissive: Option<TexData>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.6,
            emissive_factor: [0.0, 0.0, 0.0],
            base_color: None,
            metallic_roughness: None,
            normal: None,
            emissive: None,
        }
    }
}

/// CPU-side mesh ready for upload.
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: Material,
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

    let mat = prim.material();
    let pbr = mat.pbr_metallic_roughness();
    let tex = |info: Option<gltf::texture::Info>| {
        info.and_then(|i| images.get(i.texture().source().index()))
            .map(|d| TexData::Rgba8(image_to_rgba8(d)))
    };
    let material = Material {
        base_color_factor: pbr.base_color_factor(),
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        emissive_factor: mat.emissive_factor(),
        base_color: tex(pbr.base_color_texture()),
        metallic_roughness: tex(pbr.metallic_roughness_texture()),
        normal: mat
            .normal_texture()
            .and_then(|n| images.get(n.texture().source().index()))
            .map(|d| TexData::Rgba8(image_to_rgba8(d))),
        emissive: tex(mat.emissive_texture()),
    };

    Ok(MeshData {
        vertices,
        indices,
        material,
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
