//! Geometry + asset helpers extracted from `main.rs`: mesh byte views, the
//! path-tracer instance table, mesh/texture uploads, the ground quad, model
//! normalization, and the fallback checker texture. No render-loop state.

use dreamcoast_asset::bc::BcFormat;
use dreamcoast_asset::{MeshData, MeshVertex, TexData};
use rhi::{
    Buffer, BufferDesc, BufferUsage, Device, Format, StorageBufferDesc, Texture, TextureDesc,
};

use crate::NO_TEXTURE;

/// Raw bytes of a mesh's vertex array (32-byte vertices), for uploading geometry
/// into a ray-tracing storage buffer the path tracer reads (Phase 8 M4).
pub(crate) fn vertex_bytes(m: &MeshData) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            m.vertices.as_ptr() as *const u8,
            std::mem::size_of_val(m.vertices.as_slice()),
        )
    }
}

/// Raw bytes of a mesh's u32 index array (Phase 8 M4).
pub(crate) fn index_bytes(m: &MeshData) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            m.indices.as_ptr() as *const u8,
            std::mem::size_of_val(m.indices.as_slice()),
        )
    }
}

/// Per-instance material for the path tracer's hit shading (mirrors the glTF
/// metallic-roughness model used by the rasterizer). `base_color.a` is the
/// emissive scale; `tex` holds bindless indices for base-color / metallic-
/// roughness / normal / emissive maps (`NO_TEXTURE` if absent).
#[derive(Clone, Copy)]
pub(crate) struct PtMaterial {
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) ao: f32,
    pub(crate) tex: [u32; 4],
}

impl PtMaterial {
    /// A matte diffuse material (no metallic/specular, no textures); `base_color.a`
    /// is the emissive scale.
    pub(crate) fn diffuse(base_color: [f32; 4]) -> Self {
        Self {
            base_color,
            metallic: 0.0,
            roughness: 1.0,
            ao: 1.0,
            tex: [NO_TEXTURE; 4],
        }
    }
}

pub(crate) fn build_pt_instance_table(
    device: &Device,
    entries: &[(&MeshData, PtMaterial)],
) -> anyhow::Result<(rhi::StorageBuffer, Vec<rhi::StorageBuffer>)> {
    let mut geometry: Vec<rhi::StorageBuffer> = Vec::with_capacity(entries.len() * 2);
    let mut records: Vec<u8> = Vec::with_capacity(entries.len() * 64);
    for (mesh, mat) in entries {
        let vb = vertex_bytes(mesh);
        let ib = index_bytes(mesh);
        let vsb = device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: vb.len() as u64,
                stride: 32,
                indirect: false,
            },
            vb,
        )?;
        let isb = device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: ib.len() as u64,
                stride: 4,
                indirect: false,
            },
            ib,
        )?;
        // 64-byte record matching `Instance` in rt_common.slang.
        records.extend_from_slice(&vsb.storage_index().to_le_bytes()); // vtx
        records.extend_from_slice(&isb.storage_index().to_le_bytes()); // idx
        records.extend_from_slice(&mat.tex[0].to_le_bytes()); // tex_base
        records.extend_from_slice(&mat.tex[1].to_le_bytes()); // tex_mr
        for c in mat.base_color {
            records.extend_from_slice(&c.to_le_bytes()); // base_color (16)
        }
        records.extend_from_slice(&mat.metallic.to_le_bytes()); // params.x
        records.extend_from_slice(&mat.roughness.to_le_bytes()); // params.y
        records.extend_from_slice(&mat.ao.to_le_bytes()); // params.z
        records.extend_from_slice(&0f32.to_le_bytes()); // params.w
        records.extend_from_slice(&mat.tex[2].to_le_bytes()); // tex_normal
        records.extend_from_slice(&mat.tex[3].to_le_bytes()); // tex_emissive
        let prim_count = (ib.len() / 4 / 3) as u32; // triangle count (PrimitiveIndex bound)
        records.extend_from_slice(&prim_count.to_le_bytes()); // prim_count
        records.extend_from_slice(&0u32.to_le_bytes()); // pad1
        geometry.push(vsb);
        geometry.push(isb);
    }
    let table = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: records.len() as u64,
            stride: 64,
            indirect: false,
        },
        &records,
    )?;
    Ok((table, geometry))
}

pub(crate) fn upload_mesh(
    device: &Device,
    model: &MeshData,
) -> anyhow::Result<(Buffer, Buffer, u32)> {
    upload_geometry(device, &model.vertices, &model.indices)
}

/// View a `MeshVertex` slice as raw bytes (the GPU vertex-buffer layout) — used to
/// re-write a vertex buffer, e.g. the per-frame CPU-morphed vertices (Stage C).
pub(crate) fn vertex_slice_bytes(vertices: &[MeshVertex]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            vertices.as_ptr() as *const u8,
            std::mem::size_of_val(vertices),
        )
    }
}

/// Phase 16 E (Hit Lighting): build the CONSOLIDATED content geometry + material table the
/// hardware-ray-traced reflection reads to shade a hit with the real material (instead of the low-res
/// surface cache) for off-screen reflections. Unlike `build_pt_instance_table` (one vertex + one
/// index storage buffer PER instance → overflows the 64-slot bindless table on a 400+ mesh scene),
/// this packs ALL unique meshes into ONE vertex buffer + ONE index buffer (indices rebased to be
/// absolute into the shared vertex buffer), plus ONE per-drawable record buffer — three bindless
/// slots total, regardless of mesh count. Returns `(vtx, idx, table)`; the caller keeps them alive
/// and passes their bindless indices to the shader.
///
/// Per-drawable record (48 B, `custom_index` order = the TLAS instance order):
///   [0] idx_base (u32, offset into `idx` in u32 units)  [4] prim_count (u32, triangle bound)
///   [8] tex_albedo (u32 bindless, NO_TEXTURE = untextured)  [12] tex_mr (u32)
///   [16..32] base_color (float4)  [32] metallic (f32)  [36] roughness (f32)  [40] ao (f32)  [44] pad
pub(crate) fn build_content_hit_table(
    device: &Device,
    drawables: &[dreamcoast_scene::Drawable],
    meshes: &crate::registry::MeshRegistry,
    materials: &crate::registry::MaterialRegistry,
) -> anyhow::Result<(rhi::StorageBuffer, rhi::StorageBuffer, rhi::StorageBuffer)> {
    use std::collections::HashMap;
    // Unique meshes in first-seen draw order (same dedup as the BLAS build) → geometry is stored
    // once and shared across instances. `mesh_geo[handle] = (idx_base, prim_count)`.
    let mut vtx_bytes: Vec<u8> = Vec::new();
    let mut idx_bytes: Vec<u8> = Vec::new();
    let mut mesh_geo: HashMap<dreamcoast_scene::MeshHandle, (u32, u32)> = HashMap::new();
    for d in drawables {
        mesh_geo.entry(d.mesh).or_insert_with(|| {
            let cpu = meshes.cpu(d.mesh);
            let vtx_base = (vtx_bytes.len() / 32) as u32; // vertex offset (32 B/vertex)
            vtx_bytes.extend_from_slice(vertex_slice_bytes(&cpu.vertices));
            let idx_base = (idx_bytes.len() / 4) as u32; // index offset (u32 units)
            // Rebase each index to be ABSOLUTE into the shared vertex buffer, so the shader indexes
            // `vtx[i]` directly without a per-vertex base add.
            for &i in &cpu.indices {
                idx_bytes.extend_from_slice(&(i + vtx_base).to_le_bytes());
            }
            let prim_count = (cpu.indices.len() / 3) as u32;
            (idx_base, prim_count)
        });
    }
    // Per-drawable records (custom_index order = draw list order = TLAS instance order).
    let mut records: Vec<u8> = Vec::with_capacity(drawables.len() * 48);
    for d in drawables {
        let (idx_base, prim_count) = mesh_geo[&d.mesh];
        let m = materials.get(d.material);
        records.extend_from_slice(&idx_base.to_le_bytes());
        records.extend_from_slice(&prim_count.to_le_bytes());
        records.extend_from_slice(&m.tex[0].to_le_bytes()); // base-color texture (NO_TEXTURE if none)
        records.extend_from_slice(&m.tex[1].to_le_bytes()); // metallic-roughness texture
        for c in m.base_color {
            records.extend_from_slice(&c.to_le_bytes());
        }
        records.extend_from_slice(&m.metallic.to_le_bytes());
        records.extend_from_slice(&m.roughness.to_le_bytes());
        records.extend_from_slice(&0f32.to_le_bytes()); // ao (unused for now)
        // params.w: decal opacity (0 = not a decal). A `MaterialKind::Decal` drawable is a
        // coplanar alpha-blended tint mesh (the raster's deferred decal pass); the path tracer
        // reads this to composite it stochastically over the surface behind instead of treating
        // it as an opaque hit. Store the material's base-color alpha here — the raster's
        // DecalAlbedo blend uses `base_color.a × texture.a`, so the path tracer needs this factor
        // (the base_color.a slot in the record itself is reused as the emissive scale downstream).
        let decal = if m.kind == dreamcoast_asset::MaterialKind::Decal {
            m.base_color[3]
        } else {
            0.0f32
        };
        records.extend_from_slice(&decal.to_le_bytes()); // → 48 B
    }
    let vtx = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: vtx_bytes.len() as u64,
            stride: 32,
            indirect: false,
        },
        &vtx_bytes,
    )?;
    let idx = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: idx_bytes.len() as u64,
            stride: 4,
            indirect: false,
        },
        &idx_bytes,
    )?;
    let table = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: records.len() as u64,
            stride: 48,
            indirect: false,
        },
        &records,
    )?;
    Ok((vtx, idx, table))
}

/// Upload raw vertex/index slices into GPU vertex/index buffers (the inner of
/// [`upload_mesh`]; also used by the registry-based glTF primitive upload).
pub(crate) fn upload_geometry(
    device: &Device,
    vertices: &[MeshVertex],
    indices: &[u32],
) -> anyhow::Result<(Buffer, Buffer, u32)> {
    let vbytes = unsafe {
        std::slice::from_raw_parts(
            vertices.as_ptr() as *const u8,
            std::mem::size_of_val(vertices),
        )
    };
    let ibytes = unsafe {
        std::slice::from_raw_parts(
            indices.as_ptr() as *const u8,
            std::mem::size_of_val(indices),
        )
    };
    let vbuf = device.create_buffer(&BufferDesc {
        size: vbytes.len() as u64,
        usage: BufferUsage::Vertex,
    })?;
    vbuf.write(vbytes)?;
    let ibuf = device.create_buffer(&BufferDesc {
        size: ibytes.len() as u64,
        usage: BufferUsage::Index,
    })?;
    ibuf.write(ibytes)?;
    Ok((vbuf, ibuf, indices.len() as u32))
}

/// Upload a material texture (bindless). Uncompressed `Rgba8` uses `rgba8_format`
/// (the slot's colour space) and generates mips at upload; pre-cooked `Bc` data
/// (Phase 12 M3) uploads its block mips via the GPU-native path — no decompression.
pub(crate) fn upload_texture(
    device: &Device,
    store: &mut Vec<Texture>,
    tex: &TexData,
    rgba8_format: Format,
) -> anyhow::Result<u32> {
    let t = match tex {
        TexData::Rgba8(img) => device.create_texture(
            &TextureDesc {
                width: img.width,
                height: img.height,
                format: rgba8_format,
            },
            &img.rgba8,
        )?,
        TexData::Bc {
            format,
            srgb,
            width,
            height,
            mips,
        } => {
            let gpu_format = match (format, srgb) {
                (BcFormat::Bc1, true) => Format::Bc1Srgb,
                (BcFormat::Bc1, false) => Format::Bc1Unorm,
                (BcFormat::Bc3, true) => Format::Bc3Srgb,
                (BcFormat::Bc3, false) => Format::Bc3Unorm,
                (BcFormat::Bc4, _) => Format::Bc4Unorm,
                (BcFormat::Bc5, _) => Format::Bc5Unorm,
                (BcFormat::Bc7, true) => Format::Bc7Srgb,
                (BcFormat::Bc7, false) => Format::Bc7Unorm,
            };
            device.create_texture_compressed(
                &TextureDesc {
                    width: *width,
                    height: *height,
                    format: gpu_format,
                },
                mips,
            )?
        }
    };
    let idx = t.bindless_index();
    store.push(t);
    Ok(idx)
}

/// A large horizontal quad at height `y` (normal up, +Y), used as a shadow
/// receiver. `half` is half its side length. Built on the mesh vertex layout so
/// it shares the G-buffer / shadow pipelines.
pub(crate) fn ground_mesh(half: f32, y: f32) -> MeshData {
    let v = |x: f32, z: f32, u: f32, w: f32| MeshVertex {
        pos: [x, y, z],
        normal: [0.0, 1.0, 0.0],
        uv: [u, w],
    };
    MeshData {
        vertices: vec![
            v(-half, -half, 0.0, 0.0),
            v(half, -half, 1.0, 0.0),
            v(half, half, 1.0, 1.0),
            v(-half, half, 0.0, 1.0),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: dreamcoast_asset::Material::default(),
    }
}

/// Framing bounds of the normalized model.
pub(crate) struct ModelBounds {
    /// Bounding-sphere radius (always 1.0 after normalization) — the unit the
    /// camera, ground, lights, and shadow box are sized in.
    pub(crate) radius: f32,
}

/// Normalize a mesh into canonical units: recenter its footprint on the origin,
/// rest its base on `y = 0`, and uniformly scale so its bounding-sphere radius is
/// 1.0. glTF models vary wildly in authored scale/placement (this avocado is
/// sub-0.1 units, off the origin); normalizing keeps the camera/near-far planes,
/// ground, lights, and shadow box in comfortable, model-independent units.
pub(crate) fn normalize_on_ground(model: &mut MeshData) -> ModelBounds {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &model.vertices {
        for i in 0..3 {
            min[i] = min[i].min(v.pos[i]);
            max[i] = max[i].max(v.pos[i]);
        }
    }
    let cx = (min[0] + max[0]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let base = min[1];
    let (sx, sy, sz) = (max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    let radius = (0.5 * (sx * sx + sy * sy + sz * sz).sqrt()).max(1e-6);
    let s = 1.0 / radius; // normalize the bounding-sphere radius to 1.0
    for v in &mut model.vertices {
        v.pos[0] = (v.pos[0] - cx) * s;
        v.pos[1] = (v.pos[1] - base) * s;
        v.pos[2] = (v.pos[2] - cz) * s;
    }
    ModelBounds { radius: 1.0 }
}

/// 8x8 magenta/grey checker (fallback base color).
pub(crate) fn make_checker_texture(device: &Device) -> anyhow::Result<Texture> {
    const N: u32 = 8;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let on = (x + y) % 2 == 0;
            pixels.extend_from_slice(if on {
                &[220, 60, 200, 255]
            } else {
                &[40, 40, 48, 255]
            });
        }
    }
    Ok(device.create_texture(
        &TextureDesc {
            width: N,
            height: N,
            format: Format::Rgba8Srgb,
        },
        &pixels,
    )?)
}
