//! CPU morph-target blending (animation Stage C).
//!
//! A primitive with morph targets is deformed each frame on the CPU —
//! `vertex = base + Σ wᵢ · targetᵢ` (position + normal deltas) — from the node's
//! animated [`MorphWeights`], then written into a per-frame-in-flight vertex buffer
//! the existing g-buffer/shadow pipelines draw unchanged (no shader/pipeline change,
//! all backends, no parity gate). Unlike skinning, morphed vertices stay in the mesh
//! node's local space, so the drawable keeps its node transform.

use std::rc::Rc;

use dreamcoast_asset::{GltfScene, MeshVertex};
use dreamcoast_core::glam::Vec3;
use dreamcoast_scene::{Entity, MaterialHandle, MeshHandle, MorphWeights, World};

use crate::registry::{GpuMesh, MeshRegistry};
use crate::{FRAMES_IN_FLIGHT, mesh};

/// One morph target's per-vertex deltas (parallel to the primitive's bind vertices).
struct TargetDeltas {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
}

/// One morph-able primitive: bind geometry + its targets + the per-fif output ring.
pub(crate) struct MorphMesh {
    /// The bind (target-0/base) mesh `build_scene` references; matched by `Rc` identity.
    base_mesh: Rc<GpuMesh>,
    ring: Vec<Rc<GpuMesh>>,
    base: Vec<MeshVertex>,
    targets: Vec<TargetDeltas>,
    /// The mesh node entity carrying the animated [`MorphWeights`].
    node: Entity,
    out: Vec<MeshVertex>,
}

/// Build the morph table for an imported glTF scene: one [`MorphMesh`] per (node with
/// a mesh that has morph targets, primitive-with-targets). `node_map` resolves node
/// indices to entities (the morph-weight channel writes [`MorphWeights`] there).
pub(crate) fn build_morph_meshes(
    device: &rhi::Device,
    gscene: &GltfScene,
    prim_handles: &[Vec<(MeshHandle, MaterialHandle)>],
    node_map: &[Option<Entity>],
    registry: &MeshRegistry,
) -> anyhow::Result<Vec<MorphMesh>> {
    let mut morphed = Vec::new();
    for (node_idx, node) in gscene.nodes.iter().enumerate() {
        let Some(mesh_idx) = node.mesh else { continue };
        let Some(node_entity) = node_map.get(node_idx).copied().flatten() else {
            continue;
        };
        for (prim_idx, prim) in gscene.meshes[mesh_idx].iter().enumerate() {
            if prim.morph_targets.is_empty() {
                continue;
            }
            let targets: Vec<TargetDeltas> = prim
                .morph_targets
                .iter()
                .map(|t| TargetDeltas {
                    positions: t.positions.clone(),
                    normals: t.normals.clone(),
                })
                .collect();
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
            morphed.push(MorphMesh {
                base_mesh: registry.get(prim_handles[mesh_idx][prim_idx].0),
                ring,
                out: prim.vertices.clone(),
                base: prim.vertices.clone(),
                targets,
                node: node_entity,
            });
        }
    }
    Ok(morphed)
}

/// Blend each morph mesh by its node's current weights and write the result into this
/// frame's ring buffer. Call after the animation advances and after this slot's fence
/// has been waited (the per-frame vertex write reuses the ring buffer).
pub(crate) fn apply_morph(
    morphed: &mut [MorphMesh],
    world: &World,
    fif: usize,
) -> anyhow::Result<()> {
    for m in morphed.iter_mut() {
        let weights = world.get::<MorphWeights>(m.node).map(|w| w.0.as_slice());
        for (i, base) in m.base.iter().enumerate() {
            let mut pos = Vec3::from_array(base.pos);
            let mut nrm = Vec3::from_array(base.normal);
            if let Some(weights) = weights {
                for (t, target) in m.targets.iter().enumerate() {
                    let w = weights.get(t).copied().unwrap_or(0.0);
                    if w != 0.0 {
                        pos += w * Vec3::from_array(target.positions[i]);
                        if let Some(nd) = &target.normals {
                            nrm += w * Vec3::from_array(nd[i]);
                        }
                    }
                }
            }
            m.out[i] = MeshVertex {
                pos: pos.to_array(),
                normal: nrm.normalize_or_zero().to_array(),
                uv: base.uv,
            };
        }
        m.ring[fif].vbuf.write(mesh::vertex_slice_bytes(&m.out))?;
    }
    Ok(())
}

/// Swap each morphed drawable to this frame's ring buffer (its node transform is kept
/// — morphed vertices are in local space). Run on the freshly built scene list each
/// frame, after [`apply_morph`].
pub(crate) fn patch_scene(morphed: &[MorphMesh], scene: &mut [crate::SceneObject], fif: usize) {
    for obj in scene.iter_mut() {
        if let Some(m) = morphed.iter().find(|m| Rc::ptr_eq(&obj.mesh, &m.base_mesh)) {
            obj.mesh = m.ring[fif].clone();
        }
    }
}
