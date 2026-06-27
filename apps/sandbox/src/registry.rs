//! Renderer-side registries that back the scene's opaque handles (Phase 12 P2).
//!
//! `dreamcoast-scene` references geometry/materials by `MeshHandle` / `MaterialHandle`
//! (plain indices, no GPU types). These registries own the actual GPU mesh buffers
//! and material descriptors those handles point at, and [`build_scene`] resolves a
//! `World`'s draw list into the renderer's [`SceneObject`] draw records.
//!
//! Meshes are shared via `Rc` so multiple instances of the same geometry (e.g. two
//! spheres) upload once and each `SceneObject` just holds a cheap clone — the P2
//! generalization the Stage B glTF importer reuses for N-primitive / M-material
//! assets.

use std::collections::HashMap;
use std::rc::Rc;

use dreamcoast_asset::{GltfScene, Material, MeshData, MeshVertex};
use dreamcoast_core::glam::{Mat4, Quat, Vec3};
use dreamcoast_scene::{MaterialHandle, MeshHandle, World};
use rhi::{Buffer, Device, Format, Texture};

use crate::mesh::{upload_geometry, upload_image_rgba8, upload_mesh};
use crate::{NO_TEXTURE, SceneObject};

/// An uploaded mesh: vertex/index buffers + their counts.
pub(crate) struct GpuMesh {
    pub(crate) vbuf: Buffer,
    pub(crate) ibuf: Buffer,
    pub(crate) index_count: u32,
    pub(crate) vertex_count: u32,
}

/// Owns every uploaded [`GpuMesh`], addressed by [`MeshHandle`].
#[derive(Default)]
pub(crate) struct MeshRegistry {
    meshes: Vec<Rc<GpuMesh>>,
}

impl MeshRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Upload `mesh` to the GPU and return its handle. Each call is one upload; call
    /// once per unique geometry and reuse the handle across instances.
    pub(crate) fn upload(
        &mut self,
        device: &Device,
        mesh: &MeshData,
    ) -> anyhow::Result<MeshHandle> {
        let (vbuf, ibuf, index_count) = upload_mesh(device, mesh)?;
        Ok(self.push(vbuf, ibuf, index_count, mesh.vertices.len() as u32))
    }

    /// Upload raw vertex/index slices (a glTF primitive) and return its handle.
    pub(crate) fn upload_geometry(
        &mut self,
        device: &Device,
        vertices: &[MeshVertex],
        indices: &[u32],
    ) -> anyhow::Result<MeshHandle> {
        let (vbuf, ibuf, index_count) = upload_geometry(device, vertices, indices)?;
        Ok(self.push(vbuf, ibuf, index_count, vertices.len() as u32))
    }

    fn push(
        &mut self,
        vbuf: Buffer,
        ibuf: Buffer,
        index_count: u32,
        vertex_count: u32,
    ) -> MeshHandle {
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(Rc::new(GpuMesh {
            vbuf,
            ibuf,
            index_count,
            vertex_count,
        }));
        handle
    }

    /// Resolve a handle to a shared [`GpuMesh`] (cheap `Rc` clone).
    pub(crate) fn get(&self, handle: MeshHandle) -> Rc<GpuMesh> {
        self.meshes[handle.0 as usize].clone()
    }
}

/// A material's shading parameters (mirrors the glTF metallic-roughness model). `tex`
/// holds bindless indices for base-color / metallic-roughness / normal / emissive
/// (`NO_TEXTURE` if absent).
#[derive(Clone, Copy)]
pub(crate) struct MaterialDesc {
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) tex: [u32; 4],
}

/// Owns every material descriptor, addressed by [`MaterialHandle`].
#[derive(Default)]
pub(crate) struct MaterialRegistry {
    materials: Vec<MaterialDesc>,
}

impl MaterialRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register a material and return its handle.
    pub(crate) fn add(&mut self, desc: MaterialDesc) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(desc);
        handle
    }

    /// Resolve a handle to its descriptor.
    pub(crate) fn get(&self, handle: MaterialHandle) -> &MaterialDesc {
        &self.materials[handle.0 as usize]
    }
}

/// Materialize a `World`'s draw list into the renderer's [`SceneObject`] records by
/// resolving each drawable's mesh/material handles against the registries. Order
/// matches the draw list (deterministic), so the produced list is stable frame to
/// frame and across the rasterizer / RT / GDF consumers.
pub(crate) fn build_scene(
    world: &World,
    meshes: &MeshRegistry,
    materials: &MaterialRegistry,
) -> Vec<SceneObject> {
    world
        .draw_list()
        .into_iter()
        .map(|d| {
            let mat = materials.get(d.material);
            SceneObject {
                mesh: meshes.get(d.mesh),
                transform: d.world,
                base_color: mat.base_color,
                metallic: mat.metallic,
                roughness: mat.roughness,
                tex: mat.tex,
                casts_shadow: d.casts_shadow,
            }
        })
        .collect()
}

/// Per glTF mesh index, the `(mesh, material)` handle of each of its primitives.
pub(crate) type PrimitiveHandles = Vec<Vec<(MeshHandle, MaterialHandle)>>;

/// Upload a whole imported [`GltfScene`] into the registries (Phase 12 Stage B, the
/// P2 multi-material generalization): every primitive's geometry → a `MeshHandle`,
/// every material → a `MaterialHandle` (with shared images deduplicated by index and
/// colour space). Returns, per glTF mesh index, the `(mesh, material)` handle of each
/// primitive — plus a CPU-geometry list aligned with the mesh handles for the RT
/// instance table.
pub(crate) fn upload_gltf_scene(
    device: &Device,
    scene: &GltfScene,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    textures: &mut Vec<Texture>,
) -> anyhow::Result<(PrimitiveHandles, Vec<MeshData>)> {
    // Dedup images by (glTF image index, sRGB) so a shared texture uploads once. The
    // same image is rarely used across colour spaces, so keying on both is safe.
    let mut image_cache: HashMap<(usize, bool), u32> = HashMap::new();
    let mut resolve =
        |textures: &mut Vec<Texture>, slot: Option<usize>, srgb: bool| -> anyhow::Result<u32> {
            let Some(idx) = slot else {
                return Ok(NO_TEXTURE);
            };
            if let Some(&bindless) = image_cache.get(&(idx, srgb)) {
                return Ok(bindless);
            }
            let format = if srgb {
                Format::Rgba8Srgb
            } else {
                Format::Rgba8Unorm
            };
            let bindless = upload_image_rgba8(device, textures, &scene.images[idx], format)?;
            image_cache.insert((idx, srgb), bindless);
            Ok(bindless)
        };

    // Materials: base-color / emissive are sRGB; metallic-roughness / normal linear.
    let mut material_handles: Vec<MaterialHandle> = Vec::with_capacity(scene.materials.len());
    for m in &scene.materials {
        let tex = [
            resolve(textures, m.base_color, true)?,
            resolve(textures, m.metallic_roughness, false)?,
            resolve(textures, m.normal, false)?,
            resolve(textures, m.emissive, true)?,
        ];
        material_handles.push(materials.add(MaterialDesc {
            base_color: m.base_color_factor,
            metallic: m.metallic_factor,
            roughness: m.roughness_factor,
            tex,
        }));
    }
    // Fallback for primitives with no material (glTF default material).
    let default_material = materials.add(MaterialDesc {
        base_color: Material::default().base_color_factor,
        metallic: Material::default().metallic_factor,
        roughness: Material::default().roughness_factor,
        tex: [NO_TEXTURE; 4],
    });

    let mut per_mesh: Vec<Vec<(MeshHandle, MaterialHandle)>> =
        Vec::with_capacity(scene.meshes.len());
    let mut cpu_meshes: Vec<MeshData> = Vec::new();
    for primitives in &scene.meshes {
        let mut row = Vec::with_capacity(primitives.len());
        for prim in primitives {
            let mesh = meshes.upload_geometry(device, &prim.vertices, &prim.indices)?;
            // Keep CPU geometry aligned with the mesh handle for the RT instance table.
            debug_assert_eq!(mesh.0 as usize, cpu_meshes.len());
            cpu_meshes.push(MeshData {
                vertices: prim.vertices.clone(),
                indices: prim.indices.clone(),
                material: Material::default(),
            });
            let material = prim
                .material
                .map(|i| material_handles[i])
                .unwrap_or(default_material);
            row.push((mesh, material));
        }
        per_mesh.push(row);
    }
    Ok((per_mesh, cpu_meshes))
}

/// The world-space AABB of an imported glTF scene at its **native** (authored) scale,
/// walking the node hierarchy with accumulated transforms. Returns `None` if the scene
/// has no geometry. The engine treats glTF native units as **metres** (1 unit = 1 m) —
/// assets are placed at this scale, not rescaled, so a building stays building-sized.
pub(crate) fn gltf_bounds(scene: &GltfScene) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut stack: Vec<(usize, Mat4)> = scene.roots.iter().map(|&r| (r, Mat4::IDENTITY)).collect();
    while let Some((idx, parent)) = stack.pop() {
        let n = &scene.nodes[idx];
        let local = Mat4::from_scale_rotation_translation(
            Vec3::from(n.scale),
            Quat::from_array(n.rotation),
            Vec3::from(n.translation),
        );
        let world = parent * local;
        if let Some(mesh_idx) = n.mesh {
            for prim in &scene.meshes[mesh_idx] {
                for v in &prim.vertices {
                    let p = world.transform_point3(Vec3::from(v.pos));
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
        for &child in &n.children {
            stack.push((child, world));
        }
    }
    (min.x <= max.x).then_some((min, max))
}
