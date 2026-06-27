//! Camera-centered GDF clipmap level planning (Scalable-GI Stage B).
//!
//! A clipmap is `L` concentric cube levels of the **same** voxel resolution: the
//! coarsest covers the whole scene (global coverage, = the legacy single volume), and
//! each finer level halves the world extent while staying centered on the camera — so
//! near the camera the effective voxel size shrinks geometrically (Sponza pillars stop
//! being blobs). This module is the pure CPU planner: scene AABB + camera + a target
//! near-voxel size → the per-level world AABBs the bake/cook run over. It is the single
//! source of the level scheme (unit-tested), and the seam the future camera-following
//! streaming reuses ([[gdf-streaming-future]] in memory / docs/scalable-gi.md).

/// The planned clipmap: per-level world AABBs ordered **finest → coarsest**. The last
/// entry always covers the whole scene (global fallback). For a scene already fine
/// enough at the base resolution this is a single level (= the legacy single volume).
pub(crate) struct ClipScheme {
    /// `(aabb_min, aabb_max)` per level, finest first, coarsest last.
    pub(crate) levels: Vec<([f32; 3], [f32; 3])>,
}

impl ClipScheme {
    pub(crate) fn level_count(&self) -> usize {
        self.levels.len()
    }
}

/// Plan the clipmap levels for a scene.
///
/// - `scene_min/max`: the scene's (already padded) world AABB — becomes the coarsest
///   level verbatim, so global coverage and the gallery's existing field are unchanged.
/// - `camera`: world position the finer levels center on (clamped so each stays inside
///   the coarsest level — a finer level never pokes outside the global field).
/// - `dim`: voxel grid edge (same for every level).
/// - `target_voxel`: desired near-camera voxel size (metres). The level count is chosen
///   so level 0's voxel ≈ this (capped by `max_levels`).
/// - `max_levels`: hard cap (memory budget).
///
/// Levels share the coarsest's center axis-by-axis only via the camera clamp; each finer
/// level is a cube of half-extent `coarse_half / 2^k`.
pub(crate) fn plan_levels(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    camera: [f32; 3],
    dim: u32,
    target_voxel: f32,
    max_levels: u32,
) -> ClipScheme {
    // Coarsest level = the scene AABB as given (keeps the legacy single-volume field
    // byte-identical). Its voxel size is the largest axis / dim.
    let coarse_half = [
        (scene_max[0] - scene_min[0]) * 0.5,
        (scene_max[1] - scene_min[1]) * 0.5,
        (scene_max[2] - scene_min[2]) * 0.5,
    ];
    let coarse_center = [
        (scene_min[0] + scene_max[0]) * 0.5,
        (scene_min[1] + scene_max[1]) * 0.5,
        (scene_min[2] + scene_max[2]) * 0.5,
    ];
    let max_half = coarse_half[0]
        .max(coarse_half[1])
        .max(coarse_half[2])
        .max(1e-4);
    let coarse_voxel = 2.0 * max_half / dim as f32;

    // How many halvings to reach the target near-voxel size? Each finer level halves the
    // voxel. k = ceil(log2(coarse_voxel / target)); total levels = k + 1, capped.
    let ratio = (coarse_voxel / target_voxel.max(1e-4)).max(1.0);
    let halvings = ratio.log2().ceil().max(0.0) as u32;
    let n = (halvings + 1).clamp(1, max_levels.max(1));

    // Build finest → coarsest. Level index from finest: i in [0, n-1]; its half-extent is
    // coarse_half / 2^(n-1-i) so the last (i = n-1) equals coarse_half.
    let mut levels = Vec::with_capacity(n as usize);
    for i in 0..n {
        let shift = (n - 1 - i) as i32; // n-1 (finest) .. 0 (coarsest)
        let scale = 0.5f32.powi(shift); // 1/2^shift
        if i == n - 1 {
            // Coarsest: verbatim scene AABB (exact, no recompute drift).
            levels.push((scene_min, scene_max));
            continue;
        }
        let half = [
            coarse_half[0] * scale,
            coarse_half[1] * scale,
            coarse_half[2] * scale,
        ];
        // Center on the camera, clamped so the level stays inside the coarsest cube.
        let mut center = [0.0f32; 3];
        for a in 0..3 {
            let lo = coarse_center[a] - coarse_half[a] + half[a];
            let hi = coarse_center[a] + coarse_half[a] - half[a];
            center[a] = camera[a].clamp(lo.min(hi), lo.max(hi));
        }
        levels.push((
            [
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ],
            [
                center[0] + half[0],
                center[1] + half[1],
                center[2] + half[2],
            ],
        ));
    }
    ClipScheme { levels }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_scene_is_single_level() {
        // A gallery-sized scene already fine at 48³ → one level == the scene AABB
        // (byte-identical to the legacy single volume).
        let mn = [-1.0, -1.0, -1.0];
        let mx = [1.0, 1.0, 1.0];
        let s = plan_levels(mn, mx, [0.0; 3], 48, 0.1, 4);
        assert_eq!(s.level_count(), 1);
        assert_eq!(s.levels[0], (mn, mx));
    }

    #[test]
    fn large_scene_adds_finer_levels() {
        // Sponza-ish: ~36 m, 48³ → coarse voxel 0.75 m; target 0.1 m needs 3 halvings → 4
        // levels (capped). Coarsest is verbatim; finer levels are nested and shrinking.
        let mn = [-18.0, -7.0, -11.0];
        let mx = [18.0, 7.0, 11.0];
        let cam = [5.0, 2.0, 0.0];
        let s = plan_levels(mn, mx, cam, 48, 0.1, 4);
        assert_eq!(s.level_count(), 4);
        // Coarsest verbatim.
        assert_eq!(*s.levels.last().unwrap(), (mn, mx));
        // Each finer level has a strictly smaller extent than the next-coarser.
        for w in s.levels.windows(2) {
            let fine = w[0].1[0] - w[0].0[0];
            let coarse = w[1].1[0] - w[1].0[0];
            assert!(
                fine < coarse,
                "finer level must be smaller: {fine} vs {coarse}"
            );
        }
        // Every finer level stays inside the coarsest cube.
        for (lmin, lmax) in &s.levels {
            for a in 0..3 {
                assert!(lmin[a] >= mn[a] - 1e-4 && lmax[a] <= mx[a] + 1e-4);
            }
        }
    }

    #[test]
    fn finest_voxel_meets_target() {
        let mn = [-18.0, -7.0, -11.0];
        let mx = [18.0, 7.0, 11.0];
        let s = plan_levels(mn, mx, [0.0; 3], 48, 0.1, 8);
        let (lmin, lmax) = s.levels[0];
        let voxel = (lmax[0] - lmin[0])
            .max(lmax[1] - lmin[1])
            .max(lmax[2] - lmin[2])
            / 48.0;
        assert!(
            voxel <= 0.1 + 1e-4,
            "finest voxel {voxel} should meet 0.1 m target"
        );
    }

    #[test]
    fn max_levels_caps_count() {
        let mn = [-500.0, -500.0, -500.0];
        let mx = [500.0, 500.0, 500.0];
        let s = plan_levels(mn, mx, [0.0; 3], 48, 0.05, 3);
        assert_eq!(s.level_count(), 3);
    }
}
