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
use tracing::warn;

use crate::registry::{MaterialRegistry, MeshRegistry};

/// Surface-cache atlas budget: the maximum number of mesh cards. The atlas is
/// `cards · CARD_TILE²` texels across four flat buffers (captured pos + albedo + a
/// radiance ping-pong) re-lit every frame, so cost is linear in card count: at
/// `CARD_TILE = 32` this cap is ~1024·1024·16 B·4 ≈ 67 MB and ~1.05 M texels re-lit/frame.
/// 6 cards / drawable ⇒ ~170 drawables fit; above this the largest-volume drawables (most
/// screen-relevant) keep cards and the rest are logged — never silently dropped.
pub(crate) const MAX_CARDS: u32 = 1024;

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
    /// Per-drawable representative linear albedo (draw-list order), aligned with
    /// `drawable_aabb`. Each drawable (glTF primitive) has exactly one material, so this
    /// is the surface's true color — the mesh-card capture (C) stamps it onto the card so
    /// the GI/reflection cache carries the real albedo instead of the blurred voxel volume.
    pub(crate) drawable_albedo: Vec<[f32; 3]>,
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
    let mut drawable_albedo: Vec<[f32; 3]> = Vec::new();

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
        drawable_albedo.push(albedo);
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
        drawable_albedo,
    }
}

/// Build the Lumen-style surface-cache mesh cards from the per-drawable world AABBs: 6
/// axis-aligned cards per drawable (one per AABB face), 64 bytes each
/// (`center.xyz/trace_depth, normal.xyz, u_axis.xyz, v_axis.xyz`). The capture pass
/// sphere-traces the GDF inward from each card-plane texel to the surface.
///
/// Optimization / scalability: if 6·drawables would exceed [`MAX_CARDS`], only the largest
/// drawables (by AABB volume — the most screen-relevant) keep cards, in their original
/// draw-list order (so a within-budget scene like the gallery is byte-identical), and the
/// dropped count is logged. This bounds the atlas size and the per-frame relight cost.
pub(crate) fn build_surface_cards(
    drawable_aabb: &[([f32; 3], [f32; 3])],
    drawable_albedo: &[[f32; 3]],
) -> (Vec<u8>, Vec<u8>) {
    let max_drawables = (MAX_CARDS / 6) as usize;
    // Pick which drawables get cards (all, unless over budget → largest-volume subset).
    let mut keep: Vec<usize> = (0..drawable_aabb.len()).collect();
    if keep.len() > max_drawables {
        let volume = |i: usize| -> f32 {
            let (mn, mx) = drawable_aabb[i];
            (mx[0] - mn[0]).max(0.0) * (mx[1] - mn[1]).max(0.0) * (mx[2] - mn[2]).max(0.0)
        };
        keep.sort_by(|&a, &b| volume(b).total_cmp(&volume(a)));
        let dropped = keep.len() - max_drawables;
        keep.truncate(max_drawables);
        keep.sort_unstable(); // restore draw-list order for determinism
        warn!(
            "surface cache: {} drawables exceed the {}-card budget — keeping the {} largest, \
             dropping {} (smaller drawables get no cache cards)",
            drawable_aabb.len(),
            MAX_CARDS,
            max_drawables,
            dropped
        );
    }

    let mut cards: Vec<u8> = Vec::with_capacity(keep.len() * 6 * 64);
    // One linear-albedo float3 (12 B) per card, same card order — the capture stamps the
    // drawable's true material color onto its 6 cards (C).
    let mut card_albedo: Vec<u8> = Vec::with_capacity(keep.len() * 6 * 12);
    let push4 = |v: [f32; 3], w: f32, buf: &mut Vec<u8>| {
        for c in v {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf.extend_from_slice(&w.to_le_bytes());
    };
    for &i in &keep {
        let (omin, omax) = drawable_aabb[i];
        let alb = drawable_albedo[i];
        let center = [
            (omin[0] + omax[0]) * 0.5,
            (omin[1] + omax[1]) * 0.5,
            (omin[2] + omax[2]) * 0.5,
        ];
        let half = [
            (omax[0] - omin[0]) * 0.5,
            (omax[1] - omin[1]) * 0.5,
            (omax[2] - omin[2]) * 0.5,
        ];
        for axis in 0..3 {
            for &sign in &[1.0f32, -1.0] {
                let mut normal = [0.0f32; 3];
                normal[axis] = sign;
                let mut fc = center;
                fc[axis] = if sign > 0.0 { omax[axis] } else { omin[axis] };
                let t1 = (axis + 1) % 3;
                let t2 = (axis + 2) % 3;
                let mut u_axis = [0.0f32; 3];
                u_axis[t1] = half[t1];
                let mut v_axis = [0.0f32; 3];
                v_axis[t2] = half[t2];
                let depth = (omax[axis] - omin[axis]).max(1e-4);
                push4(fc, depth, &mut cards);
                push4(normal, 0.0, &mut cards);
                push4(u_axis, 0.0, &mut cards);
                push4(v_axis, 0.0, &mut cards);
                for c in alb {
                    card_albedo.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
    (cards, card_albedo)
}
