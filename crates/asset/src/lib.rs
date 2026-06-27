//! Asset loading: glTF meshes + base-color textures, plus procedural fallbacks.
//!
//! RHI-agnostic — returns plain CPU data; the caller uploads it to the GPU.

use std::path::Path;

use dreamcoast_core::EngineError;

pub mod bc;
pub mod cook;
pub mod dcasset;
pub mod sdf;

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
        material: Material::default(),
    }
}

/// A single quad (two CCW triangles) from four corner positions and a normal.
fn quad(p: [[f32; 3]; 4], normal: [f32; 3]) -> MeshData {
    let v = |pos: [f32; 3], uv: [f32; 2]| MeshVertex { pos, normal, uv };
    MeshData {
        vertices: vec![
            v(p[0], [0.0, 0.0]),
            v(p[1], [1.0, 0.0]),
            v(p[2], [1.0, 1.0]),
            v(p[3], [0.0, 1.0]),
        ],
        indices: vec![0, 1, 2, 2, 3, 0],
        material: Material::default(),
    }
}

/// An axis-aligned box from `min` to `max` (inward-agnostic; the path tracer is
/// two-sided), as one mesh with 6 quad faces.
fn axis_box(min: [f32; 3], max: [f32; 3]) -> MeshData {
    let faces = [
        // +X / -X
        (
            [
                [max[0], min[1], min[2]],
                [max[0], min[1], max[2]],
                [max[0], max[1], max[2]],
                [max[0], max[1], min[2]],
            ],
            [1.0, 0.0, 0.0],
        ),
        (
            [
                [min[0], min[1], max[2]],
                [min[0], min[1], min[2]],
                [min[0], max[1], min[2]],
                [min[0], max[1], max[2]],
            ],
            [-1.0, 0.0, 0.0],
        ),
        // +Y / -Y
        (
            [
                [min[0], max[1], min[2]],
                [max[0], max[1], min[2]],
                [max[0], max[1], max[2]],
                [min[0], max[1], max[2]],
            ],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                [min[0], min[1], max[2]],
                [max[0], min[1], max[2]],
                [max[0], min[1], min[2]],
                [min[0], min[1], min[2]],
            ],
            [0.0, -1.0, 0.0],
        ),
        // +Z / -Z
        (
            [
                [max[0], min[1], max[2]],
                [min[0], min[1], max[2]],
                [min[0], max[1], max[2]],
                [max[0], max[1], max[2]],
            ],
            [0.0, 0.0, 1.0],
        ),
        (
            [
                [min[0], min[1], min[2]],
                [max[0], min[1], min[2]],
                [max[0], max[1], min[2]],
                [min[0], max[1], min[2]],
            ],
            [0.0, 0.0, -1.0],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (corners, n) in faces {
        let base = vertices.len() as u32;
        for (i, c) in corners.iter().enumerate() {
            vertices.push(MeshVertex {
                pos: *c,
                normal: n,
                uv: if i == 1 || i == 2 {
                    [1.0, 0.0]
                } else {
                    [0.0, 0.0]
                },
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }
    MeshData {
        vertices,
        indices,
        material: Material::default(),
    }
}

/// A Cornell box for the path-tracer GI demo: a white floor/ceiling/back, a red
/// left wall, a green right wall, a large emissive ceiling light, and two white
/// boxes inside. The interior spans x,z in [-1, 1] and y in [0, 2]; geometry is
/// in world space (instances use the identity transform). Each entry pairs a mesh
/// with `[albedo_r, albedo_g, albedo_b, emissive_scale]` for the instance table.
pub fn cornell_box() -> Vec<(MeshData, [f32; 4])> {
    let white = [0.73, 0.73, 0.73, 0.0];
    let red = [0.65, 0.05, 0.05, 0.0];
    let green = [0.12, 0.45, 0.15, 0.0];
    // Large area light: bright emissive white covering most of the ceiling (keeps
    // variance low without next-event estimation).
    let light = [1.0, 0.95, 0.85, 12.0];
    vec![
        // Floor (y=0) and ceiling (y=2).
        (
            quad(
                [
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                [0.0, 1.0, 0.0],
            ),
            white,
        ),
        (
            quad(
                [
                    [-1.0, 2.0, -1.0],
                    [-1.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 2.0, -1.0],
                ],
                [0.0, -1.0, 0.0],
            ),
            white,
        ),
        // Back wall (z=-1).
        (
            quad(
                [
                    [-1.0, 0.0, -1.0],
                    [-1.0, 2.0, -1.0],
                    [1.0, 2.0, -1.0],
                    [1.0, 0.0, -1.0],
                ],
                [0.0, 0.0, 1.0],
            ),
            white,
        ),
        // Left wall red (x=-1), right wall green (x=1).
        (
            quad(
                [
                    [-1.0, 0.0, 1.0],
                    [-1.0, 2.0, 1.0],
                    [-1.0, 2.0, -1.0],
                    [-1.0, 0.0, -1.0],
                ],
                [1.0, 0.0, 0.0],
            ),
            red,
        ),
        (
            quad(
                [
                    [1.0, 0.0, -1.0],
                    [1.0, 2.0, -1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0],
                ],
                [-1.0, 0.0, 0.0],
            ),
            green,
        ),
        // Emissive ceiling light (just below the ceiling).
        (
            quad(
                [
                    [-0.5, 1.98, -0.5],
                    [-0.5, 1.98, 0.5],
                    [0.5, 1.98, 0.5],
                    [0.5, 1.98, -0.5],
                ],
                [0.0, -1.0, 0.0],
            ),
            light,
        ),
        // Tall box (back-left) and short box (front-right).
        (axis_box([-0.55, 0.0, -0.55], [-0.05, 1.2, -0.05]), white),
        (axis_box([0.1, 0.0, 0.1], [0.6, 0.6, 0.6]), white),
    ]
}

/// A unit UV sphere centered at the origin (radius 1) with smooth outward
/// normals. Good for showing off PBR / image-based reflections.
pub fn uv_sphere(segments: u32, rings: u32) -> MeshData {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI; // 0 (top) .. PI (bottom)
        let (sin_phi, cos_phi) = phi.sin_cos();
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            // Radius 1, so the position doubles as the outward normal.
            let pos = [sin_phi * cos_theta, cos_phi, sin_phi * sin_theta];
            vertices.push(MeshVertex {
                pos,
                normal: pos,
                uv: [u, v],
            });
        }
    }
    let stride = segments + 1;
    let mut indices = Vec::with_capacity((segments * rings * 6) as usize);
    for r in 0..rings {
        for s in 0..segments {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    MeshData {
        vertices,
        indices,
        material: Material::default(),
    }
}
