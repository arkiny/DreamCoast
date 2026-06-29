//! CPU vertex skinning (animation Stage B.1).
//!
//! Linear-blend skinning done on the CPU: each frame, after the ECS transforms
//! propagate, build each skinned primitive's joint-matrix palette
//! (`joint_world × inverse_bind`), deform its bind-pose vertices
//! (`Σ wᵢ · paletteᵢ · v`), and write the result into a per-frame-in-flight vertex
//! buffer that the existing g-buffer pipeline draws unchanged. Keeping skinning on
//! the CPU leaves the GPU vertex format / pipelines / shaders untouched, so
//! non-skinned output stays byte-identical and there is no cross-backend risk; a GPU
//! skinning path is a later optimization (Stage B.2).
//!
//! Skinned vertices land in skeleton/scene space (the joint world matrices already
//! include the scene-root placement), so a skinned drawable's model matrix is the
//! identity — its glTF mesh-node transform is ignored, per the glTF spec.

use std::rc::Rc;

use dreamcoast_asset::GltfScene;
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_scene::{Entity, MaterialHandle, MeshHandle, World, WorldTransform};
use rhi::Device;

use crate::SceneObject;
use crate::registry::{GpuMesh, MeshRegistry};
use crate::{FRAMES_IN_FLIGHT, mesh};

/// One skinned primitive: its bind-pose geometry + skin binding, plus the
/// per-frame-in-flight ring of vertex buffers the skinned output is written into.
pub(crate) struct SkinnedMesh {
    /// The static (bind-pose) mesh `build_scene` references; drawables are matched to
    /// it by `Rc` identity and swapped to the live ring buffer for this frame.
    base_mesh: Rc<GpuMesh>,
    /// Per-fif skinned vertex buffers (each owns its index buffer); written each frame.
    ring: Vec<Rc<GpuMesh>>,
    bind: Vec<dreamcoast_asset::MeshVertex>,
    /// Per-vertex joint indices (into `joint_entities`) + weights.
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    /// The skin's joint entities (parallel to `inverse_bind`).
    joint_entities: Vec<Entity>,
    inverse_bind: Vec<Mat4>,
    /// Scratch for this frame's skinned vertices (reused; same length as `bind`).
    out: Vec<dreamcoast_asset::MeshVertex>,
}

/// Build the skinned-mesh table for an imported glTF scene: one [`SkinnedMesh`] per
/// (skinned node, primitive-with-weights). `node_map` is the node-index → entity map
/// from `instantiate_gltf_mapped`; `prim_handles` the renderer-resolved per-primitive
/// handles from `upload_gltf_scene`.
pub(crate) fn build_skinned_meshes(
    device: &Device,
    gscene: &GltfScene,
    prim_handles: &[Vec<(MeshHandle, MaterialHandle)>],
    node_map: &[Option<Entity>],
    registry: &MeshRegistry,
) -> anyhow::Result<Vec<SkinnedMesh>> {
    let mut skinned = Vec::new();
    for node in &gscene.nodes {
        let (Some(mesh_idx), Some(skin_idx)) = (node.mesh, node.skin) else {
            continue;
        };
        let skin = &gscene.skins[skin_idx];
        let joint_entities: Vec<Entity> = skin
            .joints
            .iter()
            .filter_map(|&n| node_map.get(n).copied().flatten())
            .collect();
        if joint_entities.len() != skin.joints.len() {
            continue; // a joint node was not instantiated — skip this skin
        }
        let inverse_bind: Vec<Mat4> = (0..skin.joints.len())
            .map(|i| {
                skin.inverse_bind
                    .get(i)
                    .map(Mat4::from_cols_array)
                    .unwrap_or(Mat4::IDENTITY)
            })
            .collect();

        for (prim_idx, prim) in gscene.meshes[mesh_idx].iter().enumerate() {
            let (Some(joints), Some(weights)) = (prim.joints.as_ref(), prim.weights.as_ref())
            else {
                continue;
            };
            let handle = prim_handles[mesh_idx][prim_idx].0;
            let mut ring = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let (vbuf, ibuf, index_count) =
                    mesh::upload_geometry(device, &prim.vertices, &prim.indices)?;
                ring.push(Rc::new(GpuMesh {
                    vbuf,
                    ibuf,
                    index_count,
                    vertex_count: prim.vertices.len() as u32,
                }));
            }
            skinned.push(SkinnedMesh {
                base_mesh: registry.get(handle),
                ring,
                bind: prim.vertices.clone(),
                joints: joints.clone(),
                weights: weights.clone(),
                joint_entities: joint_entities.clone(),
                inverse_bind: inverse_bind.clone(),
                out: prim.vertices.clone(),
            });
        }
    }
    Ok(skinned)
}

/// Skin every mesh for this frame and write the result into its `fif` ring buffer.
/// Call after the transforms propagate and after this slot's fence has been waited
/// (so the ring buffer is no longer read by an in-flight frame).
pub(crate) fn skin_and_upload(
    skinned: &mut [SkinnedMesh],
    world: &World,
    fif: usize,
) -> anyhow::Result<()> {
    for sk in skinned.iter_mut() {
        let palette: Vec<Mat4> = sk
            .joint_entities
            .iter()
            .zip(&sk.inverse_bind)
            .map(|(&je, ib)| {
                let joint_world = world
                    .get::<WorldTransform>(je)
                    .map(|w| w.0)
                    .unwrap_or(Mat4::IDENTITY);
                joint_world * *ib
            })
            .collect();

        for (vi, v) in sk.bind.iter().enumerate() {
            let (j, w) = (sk.joints[vi], sk.weights[vi]);
            // Linear blend: the weighted sum of joint matrices (Σ wᵢ · paletteᵢ).
            let mut m = Mat4::ZERO;
            for k in 0..4 {
                if w[k] != 0.0 {
                    m += palette[j[k] as usize] * w[k];
                }
            }
            let pos = m.transform_point3(Vec3::from_array(v.pos));
            let normal = m
                .transform_vector3(Vec3::from_array(v.normal))
                .normalize_or_zero();
            sk.out[vi] = dreamcoast_asset::MeshVertex {
                pos: pos.to_array(),
                normal: normal.to_array(),
                uv: v.uv,
            };
        }
        sk.ring[fif].vbuf.write(mesh::vertex_slice_bytes(&sk.out))?;
    }
    Ok(())
}

/// Swap each skinned drawable to its `fif` ring buffer and reset its model matrix to
/// the identity (skinned vertices are already in world space). Run on the freshly
/// built scene list each frame, after [`skin_and_upload`].
pub(crate) fn patch_scene(skinned: &[SkinnedMesh], scene: &mut [SceneObject], fif: usize) {
    for obj in scene.iter_mut() {
        if let Some(sk) = skinned.iter().find(|s| Rc::ptr_eq(&obj.mesh, &s.base_mesh)) {
            obj.mesh = sk.ring[fif].clone();
            obj.transform = Mat4::IDENTITY;
        }
    }
}
