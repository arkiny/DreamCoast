//! Morph-target blending (animation Stage C).
//!
//! A primitive with morph targets is deformed each frame by `vertex = base + Σ wᵢ·targetᵢ`
//! (position + normal deltas) from the node's animated [`MorphWeights`]. Two paths share
//! this module:
//!
//! * **GPU vertex-pulling** (Stage C optimization, the fast path): the bind-pose buffer
//!   is uploaded **once** and the per-target deltas live in a static bindless storage
//!   buffer; per frame only a tiny weights buffer (one `f32` per target) is written, and
//!   `gbuffer.slang`/`shadow.slang`'s `vsMainMorphed` blends in the shader by `SV_VertexID`
//!   — so the CPU does **no per-vertex work** and there is no per-frame vertex re-upload.
//!   Mirrors the GPU skinning design ([`crate::skin`]); needs the same host-writable
//!   storage buffer (`create_storage_buffer_host`, added on all 3 backends in Stage B.2c).
//! * **CPU blend** (the fallback): on a backend without host storage writes, blend on the
//!   CPU (distributed over the job system) into a per-frame-in-flight vertex ring the
//!   existing pipelines draw unchanged.
//!
//! Unlike skinning, morphed vertices stay in the mesh node's local space, so the drawable
//! keeps its node transform in both paths.

use std::rc::Rc;

use dreamcoast_asset::{GltfScene, MeshVertex};
use dreamcoast_core::glam::Vec3;
use dreamcoast_scene::{Entity, MaterialHandle, MeshHandle, MorphWeights, World};
use rhi::{Device, StorageBuffer, StorageBufferDesc};

use crate::registry::{GpuMesh, MeshRegistry};
use crate::{FRAMES_IN_FLIGHT, mesh};

/// One morph target's per-vertex deltas (parallel to the primitive's bind vertices).
struct TargetDeltas {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
}

/// CPU-blended morph primitive: bind geometry + targets + the per-fif output ring.
struct CpuMorphMesh {
    /// The bind (target-0/base) mesh `build_scene` references; matched by `Rc` identity.
    base_mesh: Rc<GpuMesh>,
    ring: Vec<Rc<GpuMesh>>,
    base: Vec<MeshVertex>,
    targets: Vec<TargetDeltas>,
    /// The mesh node entity carrying the animated [`MorphWeights`].
    node: Entity,
    out: Vec<MeshVertex>,
}

/// GPU vertex-pulling morph primitive: the static delta storage buffer + a per-fif
/// weights ring; the bind-pose buffer is drawn directly (the VS does the blend).
struct GpuMorphMesh {
    /// The bind-pose mesh `build_scene` references; drawables match it by `Rc` identity.
    base_mesh: Rc<GpuMesh>,
    /// Bindless index of the static per-target per-vertex deltas (two `float4`/vertex).
    deltas_idx: u32,
    target_count: u32,
    vertex_count: u32,
    /// The mesh node entity carrying the animated [`MorphWeights`].
    node: Entity,
    /// Per-fif weights storage buffers + their bindless indices; written each frame.
    weight_bufs: Vec<StorageBuffer>,
    weight_idx: Vec<u32>,
    /// Reused per-frame weights byte scratch (16-byte-aligned buffer size).
    weight_scratch: Vec<u8>,
    // Keep the static delta buffer resident.
    _deltas_buf: StorageBuffer,
}

/// Morph-target table for an imported glTF scene: GPU primitives where host storage is
/// available, else CPU ones. Empty unless the scene has morph targets.
#[derive(Default)]
pub(crate) struct MorphSet {
    gpu: Vec<GpuMorphMesh>,
    cpu: Vec<CpuMorphMesh>,
}

impl MorphSet {
    pub(crate) fn is_empty(&self) -> bool {
        self.gpu.is_empty() && self.cpu.is_empty()
    }
    pub(crate) fn gpu_count(&self) -> usize {
        self.gpu.len()
    }
    pub(crate) fn cpu_count(&self) -> usize {
        self.cpu.len()
    }
}

fn storage_desc(size: usize, stride: u32) -> StorageBufferDesc {
    StorageBufferDesc {
        size: size as u64,
        stride,
        indirect: false,
    }
}

/// Build the morph table for an imported glTF scene: one entry per (node with a mesh that
/// has morph targets, primitive-with-targets). `node_map` resolves node indices to entities
/// (the morph-weight channel writes [`MorphWeights`] there). The GPU path is chosen when a
/// host-writable storage buffer is available (probe), else the whole set blends on the CPU.
pub(crate) fn build_morph_meshes(
    device: &Device,
    gscene: &GltfScene,
    prim_handles: &[Vec<(MeshHandle, MaterialHandle)>],
    node_map: &[Option<Entity>],
    registry: &MeshRegistry,
) -> anyhow::Result<MorphSet> {
    // GPU morph needs a host-writable storage buffer for the per-frame weights (same
    // requirement as GPU skinning). On a backend without it — or with `MORPH_CPU=1`, the
    // seam that forces the CPU blend for the GPU↔CPU cross-check — fall back to the CPU
    // path.
    let force_cpu = std::env::var("MORPH_CPU").is_ok();
    let gpu_ok = !force_cpu
        && device
            .create_storage_buffer_host(&storage_desc(16, 4))
            .map(|b| b.write(&[0u8; 16]).is_ok())
            .unwrap_or(false);
    if !gpu_ok {
        tracing::warn!(
            "GPU morph unavailable on this backend (no storage-buffer host-write); \
             morph targets blend on the CPU"
        );
    }

    let mut gpu = Vec::new();
    let mut cpu = Vec::new();
    for (node_idx, node) in gscene.nodes.iter().enumerate() {
        let Some(mesh_idx) = node.mesh else { continue };
        let Some(node_entity) = node_map.get(node_idx).copied().flatten() else {
            continue;
        };
        for (prim_idx, prim) in gscene.meshes[mesh_idx].iter().enumerate() {
            if prim.morph_targets.is_empty() {
                continue;
            }
            let base_mesh = registry.get(prim_handles[mesh_idx][prim_idx].0);
            if gpu_ok {
                gpu.push(build_gpu_mesh(device, prim, base_mesh, node_entity)?);
            } else {
                cpu.push(build_cpu_mesh(device, prim, base_mesh, node_entity)?);
            }
        }
    }
    Ok(MorphSet { gpu, cpu })
}

/// Build a GPU morph primitive: pack the deltas into a static storage buffer (per target,
/// per vertex: two `float4` = pos.xyz0, nrm.xyz0 — 32 bytes, matching the skinning
/// `Load<float4>` access pattern), seed the per-fif weights ring.
fn build_gpu_mesh(
    device: &Device,
    prim: &dreamcoast_asset::GltfPrimitive,
    base_mesh: Rc<GpuMesh>,
    node: Entity,
) -> anyhow::Result<GpuMorphMesh> {
    let vertex_count = prim.vertices.len();
    let target_count = prim.morph_targets.len();
    // Static deltas: target-major, two float4 (pos, nrm) per vertex.
    let mut dbytes = Vec::with_capacity(target_count * vertex_count * 32);
    for t in &prim.morph_targets {
        for v in 0..vertex_count {
            let p = t.positions[v];
            dbytes.extend_from_slice(&p[0].to_le_bytes());
            dbytes.extend_from_slice(&p[1].to_le_bytes());
            dbytes.extend_from_slice(&p[2].to_le_bytes());
            dbytes.extend_from_slice(&0f32.to_le_bytes());
            let n = t.normals.as_ref().map(|n| n[v]).unwrap_or([0.0; 3]);
            dbytes.extend_from_slice(&n[0].to_le_bytes());
            dbytes.extend_from_slice(&n[1].to_le_bytes());
            dbytes.extend_from_slice(&n[2].to_le_bytes());
            dbytes.extend_from_slice(&0f32.to_le_bytes());
        }
    }
    let deltas_buf = device.create_storage_buffer_init(&storage_desc(dbytes.len(), 32), &dbytes)?;

    // Per-fif weights ring (one f32 per target, buffer rounded up to 16 bytes), seeded 0.
    let weight_size = target_count.next_multiple_of(4).max(4) * 4;
    let zeros = vec![0u8; weight_size];
    let mut weight_bufs = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut weight_idx = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        let wb = device.create_storage_buffer_host(&storage_desc(weight_size, 4))?;
        wb.write(&zeros)?; // first frame's update overwrites it
        weight_idx.push(wb.storage_index());
        weight_bufs.push(wb);
    }

    Ok(GpuMorphMesh {
        base_mesh,
        deltas_idx: deltas_buf.storage_index(),
        target_count: target_count as u32,
        vertex_count: vertex_count as u32,
        node,
        weight_bufs,
        weight_idx,
        weight_scratch: vec![0u8; weight_size],
        _deltas_buf: deltas_buf,
    })
}

/// Build a CPU morph primitive: clone the targets + a per-fif output vertex ring.
fn build_cpu_mesh(
    device: &Device,
    prim: &dreamcoast_asset::GltfPrimitive,
    base_mesh: Rc<GpuMesh>,
    node: Entity,
) -> anyhow::Result<CpuMorphMesh> {
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
    Ok(CpuMorphMesh {
        base_mesh,
        ring,
        out: prim.vertices.clone(),
        base: prim.vertices.clone(),
        targets,
        node,
    })
}

/// Advance each morph primitive for this frame: GPU primitives write their per-frame
/// weights buffer (cheap, no per-vertex work); CPU primitives blend into the ring. Call
/// after the animation advances and after this slot's fence has been waited.
pub(crate) fn apply_morph(set: &mut MorphSet, world: &World, fif: usize) -> anyhow::Result<()> {
    for m in set.gpu.iter_mut() {
        m.weight_scratch.iter_mut().for_each(|b| *b = 0);
        if let Some(w) = world.get::<MorphWeights>(m.node) {
            for (t, &wv) in w.0.iter().take(m.target_count as usize).enumerate() {
                m.weight_scratch[t * 4..t * 4 + 4].copy_from_slice(&wv.to_le_bytes());
            }
        }
        m.weight_bufs[fif].write(&m.weight_scratch)?;
    }
    for m in set.cpu.iter_mut() {
        blend_cpu(m, world, fif)?;
    }
    Ok(())
}

/// Blend one CPU morph primitive (`base + Σ wᵢ·targetᵢ`) and upload this frame's ring.
///
/// The per-vertex blend is distributed over the job system (`parallel_for` across the
/// disjoint output vertices — all plain `Send` data, so no `unsafe`). Each output is a
/// pure function of `base[i]` + the targets + weights, so the result is **bit-identical**
/// to the serial loop; the `vbuf.write` upload stays serial.
fn blend_cpu(m: &mut CpuMorphMesh, world: &World, fif: usize) -> anyhow::Result<()> {
    let jobs = dreamcoast_jobs::global();
    let weights: Vec<f32> = world
        .get::<MorphWeights>(m.node)
        .map(|w| w.0.clone())
        .unwrap_or_default();
    // Disjoint borrows so the parallel closure can read the bind geometry/targets while
    // it writes the output vertices.
    let (base, targets, out) = (&m.base, &m.targets, &mut m.out);
    let weights = &weights;
    // ~256 vertices/chunk: enough work per job to amortise scheduling on big blendshape
    // meshes, while tiny ones stay in a single chunk.
    jobs.parallel_for(out, 256, |i, out_v| {
        let mut pos = Vec3::from_array(base[i].pos);
        let mut nrm = Vec3::from_array(base[i].normal);
        for (t, target) in targets.iter().enumerate() {
            let w = weights.get(t).copied().unwrap_or(0.0);
            if w != 0.0 {
                pos += w * Vec3::from_array(target.positions[i]);
                if let Some(nd) = &target.normals {
                    nrm += w * Vec3::from_array(nd[i]);
                }
            }
        }
        *out_v = MeshVertex {
            pos: pos.to_array(),
            normal: nrm.normalize_or_zero().to_array(),
            uv: base[i].uv,
        };
    });
    m.ring[fif].vbuf.write(mesh::vertex_slice_bytes(&m.out))?;
    Ok(())
}

/// Patch each morphed drawable for this frame (its node transform is kept — morphed
/// vertices are in local space). GPU primitives tag the drawable with this frame's morph
/// indices (the g-buffer/shadow pass draws them with the morph pipeline + the bind-pose
/// buffer); CPU primitives swap to this frame's ring buffer. Run on the freshly built
/// scene list each frame, after [`apply_morph`].
pub(crate) fn patch_scene(set: &MorphSet, scene: &mut [crate::SceneObject], fif: usize) {
    for obj in scene.iter_mut() {
        if let Some(m) = set.gpu.iter().find(|m| Rc::ptr_eq(&obj.mesh, &m.base_mesh)) {
            obj.morph = Some([
                m.deltas_idx,
                m.weight_idx[fif],
                m.target_count,
                m.vertex_count,
            ]);
        } else if let Some(m) = set.cpu.iter().find(|m| Rc::ptr_eq(&obj.mesh, &m.base_mesh)) {
            obj.mesh = m.ring[fif].clone();
        }
    }
}
