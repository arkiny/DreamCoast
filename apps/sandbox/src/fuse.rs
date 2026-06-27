//! Draw-list scene fuse (Scalable-GI Stage 0) — the single path that builds the
//! world-space triangle soup the scene GDF/SW-RT bakes from.
//!
//! Previously the fuse was hardcoded to the 4-object gallery in `main.rs`. This walks
//! the ECS draw list instead: every drawable's **CPU geometry** (from [`MeshRegistry`])
//! is transformed to world space and its triangles tagged with the material's
//! representative albedo ([`MaterialDesc::albedo`]) — so the same routine fuses the
//! gallery, an imported glTF scene, or a level. The byte layout (32-byte vertex
//! records, u32 indices, 12-byte/triangle albedo, 10 % AABB padding) is identical to
//! the old gallery fuse, so the gallery's baked field stays byte-for-byte the same.

use dreamcoast_core::glam::Vec3;
use dreamcoast_scene::World;

use crate::registry::{MaterialRegistry, MeshRegistry};

/// The fused scene: one world-space triangle soup ready for the GDF bake, plus the
/// per-drawable AABBs the surface-cache cards are built from.
pub(crate) struct FusedScene {
    /// 32-byte vertex records (`pos`@0, `normal`@12, `uv`@24), world-space.
    pub(crate) vtx: Vec<u8>,
    /// u32 indices into `vtx`, little-endian.
    pub(crate) idx: Vec<u8>,
    /// One linear-albedo `float3` (12 bytes) per triangle, in fused-triangle order.
    pub(crate) tri_albedo: Vec<u8>,
    pub(crate) tri_count: u32,
    /// Scene AABB, **padded** 10 % per axis (≥0.05 m) so the zero-isosurface isn't
    /// clipped at the volume edge — the grid the SDF/albedo volumes bake over.
    pub(crate) aabb_min: [f32; 3],
    pub(crate) aabb_max: [f32; 3],
    /// Per-drawable world-space AABB (**unpadded**), draw-list order — the surface
    /// cache projects its mesh cards from these.
    pub(crate) drawable_aabb: Vec<([f32; 3], [f32; 3])>,
}

/// Fuse `world`'s opaque draw list into one world-space triangle soup. Transforms are
/// translation + uniform scale, so normals carry through the 3×3 (re-normalized);
/// disjoint objects give the union SDF via the closest-triangle sign convention.
///
/// Geometry comes from [`MeshRegistry::cpu`] and albedo from [`MaterialDesc::albedo`]
/// — the single sources — so there is no second hardcoded layout to drift from.
pub(crate) fn fuse_scene(
    world: &World,
    meshes: &MeshRegistry,
    materials: &MaterialRegistry,
) -> FusedScene {
    let mut vtx: Vec<u8> = Vec::new();
    let mut idx: Vec<u8> = Vec::new();
    let mut tri_albedo: Vec<u8> = Vec::new();
    let mut base: u32 = 0;
    let mut amin = [f32::MAX; 3];
    let mut amax = [f32::MIN; 3];
    let mut drawable_aabb: Vec<([f32; 3], [f32; 3])> = Vec::new();

    for d in world.draw_list() {
        let cpu = meshes.cpu(d.mesh);
        let albedo = materials.get(d.material).albedo;
        let xf = d.world;
        let mut omin = [f32::MAX; 3];
        let mut omax = [f32::MIN; 3];
        for v in &cpu.vertices {
            let p = xf.transform_point3(Vec3::from(v.pos));
            let n = xf
                .transform_vector3(Vec3::from(v.normal))
                .normalize_or_zero();
            amin = [amin[0].min(p.x), amin[1].min(p.y), amin[2].min(p.z)];
            amax = [amax[0].max(p.x), amax[1].max(p.y), amax[2].max(p.z)];
            omin = [omin[0].min(p.x), omin[1].min(p.y), omin[2].min(p.z)];
            omax = [omax[0].max(p.x), omax[1].max(p.y), omax[2].max(p.z)];
            vtx.extend_from_slice(&p.x.to_le_bytes());
            vtx.extend_from_slice(&p.y.to_le_bytes());
            vtx.extend_from_slice(&p.z.to_le_bytes());
            vtx.extend_from_slice(&n.x.to_le_bytes());
            vtx.extend_from_slice(&n.y.to_le_bytes());
            vtx.extend_from_slice(&n.z.to_le_bytes());
            vtx.extend_from_slice(&v.uv[0].to_le_bytes());
            vtx.extend_from_slice(&v.uv[1].to_le_bytes());
        }
        for &ix in &cpu.indices {
            idx.extend_from_slice(&(ix + base).to_le_bytes());
        }
        // One albedo record (float3, 12 B) per triangle of this drawable, in the same
        // fused-triangle order the bake indexes.
        for _ in 0..(cpu.indices.len() / 3) {
            for c in albedo {
                tri_albedo.extend_from_slice(&c.to_le_bytes());
            }
        }
        base += cpu.vertices.len() as u32;
        drawable_aabb.push((omin, omax));
    }

    // Pad the AABB by 10 % per axis so the zero-isosurface isn't clipped at the volume
    // edge (≥0.05 world units) — identical to the legacy gallery fuse.
    for i in 0..3 {
        let pad = ((amax[i] - amin[i]) * 0.1).max(0.05);
        amin[i] -= pad;
        amax[i] += pad;
    }
    let tri_count = (idx.len() / 4 / 3) as u32;

    FusedScene {
        vtx,
        idx,
        tri_albedo,
        tri_count,
        aabb_min: amin,
        aabb_max: amax,
        drawable_aabb,
    }
}
