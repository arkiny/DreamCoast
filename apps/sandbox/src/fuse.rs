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
use tracing::info;

use crate::registry::{MaterialRegistry, MeshRegistry};

/// Number of axis-aligned cards captured per drawable (one per AABB face).
pub(crate) const CARDS_PER_DRAWABLE: u32 = 6;

/// Surface-cache atlas budget: the maximum number of mesh cards. The atlas is
/// `cards · CARD_TILE²` texels across four flat buffers (captured pos + albedo + a
/// radiance ping-pong) re-lit every frame, so cost is linear in card count: at
/// `CARD_TILE = 32` this cap is ~1024·1024·16 B·4 ≈ 67 MB and ~1.05 M texels re-lit/frame.
/// 6 cards / drawable ⇒ ~170 drawables fit; above this the surface cache virtualizes:
/// drawables are ranked by a deterministic camera-relevance priority, the top `MAX_CARDS/6`
/// keep cards, and the remainder are marked **coarse fallback** (they read the dense
/// distance-field/voxel path instead of a dedicated card) — an explicit residency decision,
/// never a silent drop. Demand-driven page streaming (LRU) is the next increment.
pub(crate) const MAX_CARDS: u32 = 1024;

/// A reference camera pose used to rank drawables for card residency when the scene
/// exceeds the card budget. Built once at scene setup from the content scene's initial
/// framing (authored/level camera, `CAM_EYE`/`CAM_TARGET`, or the orbit focus). Priority
/// is a pure function of this pose + the drawable AABB, so residency is deterministic.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CardCamera {
    /// World-space eye position.
    pub(crate) eye: [f32; 3],
    /// Normalized view forward direction (eye → focus).
    pub(crate) forward: [f32; 3],
}

impl CardCamera {
    /// Build from an eye + focus point. `forward` is normalized; a degenerate (eye==focus)
    /// pair falls back to `-Z` so priority stays well-defined and deterministic.
    pub(crate) fn from_look(eye: Vec3, focus: Vec3) -> Self {
        let fwd = (focus - eye).normalize_or_zero();
        let fwd = if fwd.length_squared() > 0.0 {
            fwd
        } else {
            Vec3::NEG_Z
        };
        Self {
            eye: eye.to_array(),
            forward: fwd.to_array(),
        }
    }
}

/// The outcome of the deterministic budget selection: which drawables keep dedicated
/// surface-cache cards (in ascending draw-list order, so a within-budget scene is
/// byte-identical to the legacy path) and which fall back to the coarse dense field.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CardResidency {
    /// Draw-list indices that keep cards, sorted ascending (draw-list order).
    pub(crate) resident: Vec<usize>,
    /// Draw-list indices with no dedicated card — explicit coarse fallback, sorted ascending.
    pub(crate) coarse_fallback: Vec<usize>,
}

impl CardResidency {
    fn all_resident(count: usize) -> Self {
        Self {
            resident: (0..count).collect(),
            coarse_fallback: Vec::new(),
        }
    }
}

/// Deterministic per-drawable card priority: **higher is more important** (kept first).
///
/// Two well-ordered, camera-relevant terms:
/// - **Proximity** — closer to the camera eye ranks higher (`1 / (1 + dist_to_aabb)`).
/// - **Frustum relevance** — a drawable in front of the camera outranks one behind it
///   (the signed forward projection of the eye→center vector, mapped to `[0, 1]`).
///
/// Solid-angle-ish size (larger AABB projects to more screen) breaks near-ties so big
/// nearby geometry wins over a tiny nearby speck. All arithmetic is on `f64` derived from
/// the stored `f32` inputs, so the ordering is identical run-to-run and machine-to-machine.
fn card_priority(aabb: &([f32; 3], [f32; 3]), cam: &CardCamera) -> f64 {
    let (mn, mx) = aabb;
    let center = [
        0.5 * (mn[0] as f64 + mx[0] as f64),
        0.5 * (mn[1] as f64 + mx[1] as f64),
        0.5 * (mn[2] as f64 + mx[2] as f64),
    ];
    let eye = [cam.eye[0] as f64, cam.eye[1] as f64, cam.eye[2] as f64];
    let fwd = [
        cam.forward[0] as f64,
        cam.forward[1] as f64,
        cam.forward[2] as f64,
    ];
    // Distance from the eye to the AABB (0 when the eye is inside it).
    let clamp_dist = |c: usize| -> f64 {
        let lo = mn[c] as f64;
        let hi = mx[c] as f64;
        let e = eye[c];
        (lo - e).max(e - hi).max(0.0)
    };
    let d2 = clamp_dist(0).powi(2) + clamp_dist(1).powi(2) + clamp_dist(2).powi(2);
    let dist = d2.sqrt();
    let proximity = 1.0 / (1.0 + dist);
    // Forward relevance: eye→center projected on the view forward, normalized to [0, 1].
    let to_center = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
    let len = (to_center[0].powi(2) + to_center[1].powi(2) + to_center[2].powi(2)).sqrt();
    let cos = if len > 0.0 {
        (to_center[0] * fwd[0] + to_center[1] * fwd[1] + to_center[2] * fwd[2]) / len
    } else {
        1.0 // eye at the center ⇒ maximally relevant
    };
    let relevance = 0.5 * (cos + 1.0);
    // Projected-size tie-break: AABB volume, softly compressed so it only breaks ties.
    let volume = ((mx[0] - mn[0]).max(0.0) as f64)
        * ((mx[1] - mn[1]).max(0.0) as f64)
        * ((mx[2] - mn[2]).max(0.0) as f64);
    let size_term = (1.0 + volume).ln();
    // Weighted sum: proximity and frustum relevance dominate; size only disambiguates.
    proximity * 4.0 + relevance * 2.0 + size_term * 0.1
}

/// Deterministically choose which drawables keep surface-cache cards within [`MAX_CARDS`].
///
/// When every drawable fits (`6·N ≤ MAX_CARDS`) all are resident and there is no fallback,
/// so the produced card set is byte-identical to the legacy path (the gallery anchor). When
/// the scene overflows, drawables are ranked by [`card_priority`] and the top
/// `MAX_CARDS/6` are kept; ties (equal priority) resolve by draw-list index so the choice is
/// stable. The overflow drawables are recorded as `coarse_fallback` rather than dropped, and
/// the kept set is returned in ascending draw-list order (byte-stable card layout).
pub(crate) fn select_card_residency(
    drawable_aabb: &[([f32; 3], [f32; 3])],
    cam: &CardCamera,
    max_cards: u32,
) -> CardResidency {
    let n = drawable_aabb.len();
    let max_drawables = (max_cards.max(CARDS_PER_DRAWABLE) / CARDS_PER_DRAWABLE) as usize;
    if n <= max_drawables {
        return CardResidency::all_resident(n);
    }
    // Rank by priority (descending); break ties deterministically by draw-list index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let pa = card_priority(&drawable_aabb[a], cam);
        let pb = card_priority(&drawable_aabb[b], cam);
        pb.total_cmp(&pa).then(a.cmp(&b))
    });
    let mut resident: Vec<usize> = order[..max_drawables].to_vec();
    let mut coarse_fallback: Vec<usize> = order[max_drawables..].to_vec();
    // Restore draw-list order in both lists so the card layout is byte-stable.
    resident.sort_unstable();
    coarse_fallback.sort_unstable();
    CardResidency {
        resident,
        coarse_fallback,
    }
}

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

/// Build the mesh-card surface cache from the per-drawable world AABBs: 6 axis-aligned
/// cards per drawable (one per AABB face), 64 bytes each (`center.xyz/trace_depth,
/// normal.xyz, u_axis.xyz, v_axis.xyz`). The capture pass sphere-traces the GDF inward from
/// each card-plane texel to the surface.
///
/// Scalability (surface-cache virtualization, first increment): if `6·drawables` would
/// exceed [`MAX_CARDS`], drawables are ranked by a deterministic camera-relevance priority
/// ([`select_card_residency`]) and only the top `MAX_CARDS/6` keep cards, emitted in
/// draw-list order (so a within-budget scene like the gallery is byte-identical). Overflow
/// drawables are marked **coarse fallback** — they read the dense distance field instead of
/// a dedicated card, an explicit residency state rather than a silent drop. Returns the card
/// buffers plus the [`CardResidency`] decision for callers/diagnostics.
pub(crate) fn build_surface_cards(
    drawable_aabb: &[([f32; 3], [f32; 3])],
    drawable_albedo: &[[f32; 3]],
    cam: &CardCamera,
    max_cards: u32,
) -> (Vec<u8>, Vec<u8>, CardResidency) {
    let residency = select_card_residency(drawable_aabb, cam, max_cards);
    let keep = &residency.resident;
    if !residency.coarse_fallback.is_empty() {
        info!(
            "surface cache: {} drawables exceed the {}-card budget — {} keep cards \
             (camera-priority residency), {} on coarse fallback (dense-field lit, not dropped)",
            drawable_aabb.len(),
            max_cards,
            keep.len(),
            residency.coarse_fallback.len(),
        );
    }

    let mut cards: Vec<u8> = Vec::with_capacity(keep.len() * 6 * 64);
    // One linear-albedo float3 (12 B) per card, same card order — the capture stamps the
    // drawable's true material color onto its 6 cards (C).
    let mut card_albedo: Vec<u8> = Vec::with_capacity(keep.len() * 6 * 12);
    for &i in keep {
        append_drawable_cards(
            drawable_aabb[i],
            drawable_albedo[i],
            &mut cards,
            &mut card_albedo,
        );
    }
    (cards, card_albedo, residency)
}

/// Append one drawable's 6 axis-aligned cards (64 B each, one per AABB face) + the parallel
/// per-card albedo (12 B each) to the given buffers, in the exact byte layout
/// [`build_surface_cards`] emits. Factored out so the F1 Stage 3 page-pool streaming can rebuild a
/// single drawable-slot's 6 cards in place when the live camera admits a new drawable, without
/// rebuilding the whole card buffer. `cards` grows by 384 B, `card_albedo` by 72 B.
pub(crate) fn append_drawable_cards(
    aabb: ([f32; 3], [f32; 3]),
    albedo: [f32; 3],
    cards: &mut Vec<u8>,
    card_albedo: &mut Vec<u8>,
) {
    let push4 = |v: [f32; 3], w: f32, buf: &mut Vec<u8>| {
        for c in v {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf.extend_from_slice(&w.to_le_bytes());
    };
    let (omin, omax) = aabb;
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
            push4(fc, depth, cards);
            push4(normal, 0.0, cards);
            push4(u_axis, 0.0, cards);
            push4(v_axis, 0.0, cards);
            for c in albedo {
                card_albedo.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
}

/// C2a — distance-driven per-card resolution assignment. Desired texel density is proportional
/// to the card's PROJECTED size from the reference camera (`extent / distance`, the reference
/// engine's MaxProjectedSize rule); the result is quantised to a pow2 in `[min_res, max_res]`
/// and the density scale is binary-searched so the TOTAL texel count stays within `budget`
/// (same memory as the uniform atlas, redistributed by relevance). Deterministic (pure f64
/// arithmetic on the inputs) → run-to-run identical layouts.
pub(crate) fn assign_card_res(
    cards: &[u8],
    cam: &CardCamera,
    min_res: u32,
    max_res: u32,
    budget_texels: u64,
) -> Vec<u32> {
    let n = cards.len() / 64;
    if n == 0 {
        return Vec::new();
    }
    let f = |o: usize| f32::from_le_bytes(cards[o..o + 4].try_into().unwrap()) as f64;
    let eye = [cam.eye[0] as f64, cam.eye[1] as f64, cam.eye[2] as f64];
    // Desired (unscaled) resolution per card: projected size = 2·max_half_extent / distance.
    let mut desired: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let b = i * 64;
        let c = [f(b), f(b + 4), f(b + 8)];
        let len3 = |x: f64, y: f64, z: f64| (x * x + y * y + z * z).sqrt();
        let ua = len3(f(b + 32), f(b + 36), f(b + 40));
        let va = len3(f(b + 48), f(b + 52), f(b + 56));
        let ext = 2.0 * ua.max(va);
        let dist = len3(c[0] - eye[0], c[1] - eye[1], c[2] - eye[2])
            .max(ext * 0.25)
            .max(1e-3);
        desired.push(ext / dist);
    }
    normalize_card_res(&desired, min_res, max_res, budget_texels)
}

/// C2a/C2b — normalise a per-card desired-resolution vector to a texel budget: quantise
/// `scale · desired` to pow2 in `[min_res, max_res]` and binary-search the scale so
/// `Σ res² ≤ budget` (monotone). Deterministic pure-f64 arithmetic.
pub(crate) fn normalize_card_res(
    desired: &[f64],
    min_res: u32,
    max_res: u32,
    budget_texels: u64,
) -> Vec<u32> {
    let quant = |v: f64| -> u32 {
        let v = v.max(1.0);
        let e = v.log2().round().max(0.0) as u32;
        (1u32 << e.min(31)).clamp(min_res, max_res)
    };
    let total = |scale: f64| -> u64 {
        desired
            .iter()
            .map(|d| {
                let r = quant(d * scale) as u64;
                r * r
            })
            .sum()
    };
    // Binary-search the density scale to the texel budget (monotone in scale). The lower bound
    // sits below any useful scale (everything quantises to min_res there) so feedback-style
    // desired vectors that ALREADY exceed the budget at scale 1 shrink correctly.
    let (mut lo, mut hi) = (1e-4f64, 1e7f64);
    if total(lo) > budget_texels {
        // Even the minimum scale (everything at min_res) may exceed the budget for a huge card
        // count — accept it (min_res is the floor).
        return desired.iter().map(|d| quant(d * lo)).collect();
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if total(mid) <= budget_texels {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    desired.iter().map(|d| quant(d * lo)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(center: [f32; 3], half: f32) -> ([f32; 3], [f32; 3]) {
        (
            [center[0] - half, center[1] - half, center[2] - half],
            [center[0] + half, center[1] + half, center[2] + half],
        )
    }

    /// A budget-fitting scene keeps every drawable, in draw-list order, with no fallback —
    /// this is what guarantees the within-budget gallery stays byte-identical.
    #[test]
    fn within_budget_keeps_all_in_order() {
        let cam = CardCamera::from_look(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let aabbs: Vec<_> = (0..10).map(|i| boxed([i as f32, 0.0, 0.0], 0.5)).collect();
        let r = select_card_residency(&aabbs, &cam, MAX_CARDS);
        assert_eq!(r.resident, (0..10).collect::<Vec<_>>());
        assert!(r.coarse_fallback.is_empty());
    }

    /// The exact budget boundary: `MAX_CARDS/6` drawables all fit; one more forces exactly
    /// one drawable onto coarse fallback, and the two lists partition the draw list with no
    /// overlap and no loss.
    #[test]
    fn budget_boundary_partitions_without_loss() {
        let cam = CardCamera::from_look(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let cap = (MAX_CARDS / CARDS_PER_DRAWABLE) as usize;

        // Exactly at the cap: everything resident.
        let at: Vec<_> = (0..cap).map(|i| boxed([i as f32, 0.0, 0.0], 0.5)).collect();
        let r = select_card_residency(&at, &cam, MAX_CARDS);
        assert_eq!(r.resident.len(), cap);
        assert!(r.coarse_fallback.is_empty());

        // One over the cap: exactly one drawable falls back.
        let over: Vec<_> = (0..cap + 1)
            .map(|i| boxed([i as f32, 0.0, 0.0], 0.5))
            .collect();
        let r = select_card_residency(&over, &cam, MAX_CARDS);
        assert_eq!(r.resident.len(), cap);
        assert_eq!(r.coarse_fallback.len(), 1);
        // Partition: resident ∪ fallback == 0..N, disjoint, each sorted ascending.
        let mut all = r.resident.clone();
        all.extend_from_slice(&r.coarse_fallback);
        all.sort_unstable();
        assert_eq!(all, (0..cap + 1).collect::<Vec<_>>());
        assert!(r.resident.windows(2).all(|w| w[0] < w[1]));
        assert!(r.coarse_fallback.windows(2).all(|w| w[0] < w[1]));
    }

    /// Priority ordering is deterministic: the same inputs yield the same residency across
    /// repeated calls, and (crucially) the selection is independent of the input draw-list
    /// permutation up to the identity of the kept set — closest/most-relevant geometry wins.
    #[test]
    fn priority_selection_is_deterministic() {
        let cam = CardCamera::from_look(Vec3::new(0.0, 0.0, 20.0), Vec3::ZERO);
        let cap = (MAX_CARDS / CARDS_PER_DRAWABLE) as usize;
        // cap+2 drawables receding along -Z away from the eye: the two farthest must fall back.
        let aabbs: Vec<_> = (0..cap + 2)
            .map(|i| boxed([0.0, 0.0, 20.0 - i as f32], 0.4))
            .collect();
        let r1 = select_card_residency(&aabbs, &cam, MAX_CARDS);
        let r2 = select_card_residency(&aabbs, &cam, MAX_CARDS);
        assert_eq!(r1, r2, "residency must be run-to-run identical");
        // Farthest two (largest index ⇒ most negative Z ⇒ farthest from eye at +Z) fall back.
        assert_eq!(r1.coarse_fallback, vec![cap, cap + 1]);
    }

    /// Nearer geometry outranks farther geometry: place one drawable close to the eye and the
    /// rest far; over budget, the near one is always resident.
    #[test]
    fn nearer_geometry_outranks_farther() {
        let cam = CardCamera::from_look(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let cap = (MAX_CARDS / CARDS_PER_DRAWABLE) as usize;
        let mut aabbs: Vec<_> = (0..cap)
            .map(|i| boxed([0.0, 0.0, -100.0 - i as f32], 0.5))
            .collect();
        // Insert a near drawable at draw-list index 0.
        aabbs.insert(0, boxed([0.0, 0.0, -2.0], 0.5));
        let r = select_card_residency(&aabbs, &cam, MAX_CARDS);
        assert!(
            r.resident.contains(&0),
            "the near drawable (index 0) must keep its card"
        );
        assert_eq!(r.coarse_fallback.len(), 1);
    }
}
