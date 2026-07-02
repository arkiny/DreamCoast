//! Shadow atlas + cascaded shadow maps (CSM) — the PR-7 pipeline-realignment skeleton.
//!
//! This generalizes the single directional shadow map (`deferred.rs::record_shadow` +
//! the `light_view_proj` ortho box in `main.rs`) into a **shadow-depth atlas** carved
//! into fixed tiles, plus a **directional CSM** cascade split over the view frustum. The
//! atlas is one large depth texture: `N` cascade tiles are laid out in a grid, and the
//! slot table is typed so future spot / point (cube-face) lights slot in with no re-wire
//! of the sampling side — only the fill side needs each new light type's caster loop.
//!
//! ## Split scheme (canonical, references verified)
//! The cascade split distances use the **practical split scheme** — a blend of the
//! logarithmic and uniform schemes (Zhang et al., "Parallel-Split Shadow Maps", GPU Gems
//! 3 ch.10; also MJP's shadow-map survey):
//! ```text
//!   d_log_i     = near * (far / near) ^ (i / N)
//!   d_uniform_i = near + (far - near) * (i / N)
//!   split_i     = lerp(d_uniform_i, d_log_i, lambda)      // lambda ~ 0.75
//! ```
//! Logarithmic alone over-samples the near field and starves the far; uniform does the
//! opposite. The blend (lambda ~0.5..0.85) gives a moderate density across the whole
//! range, which is the standard game-engine choice.
//!
//! ## Stable cascades (no shimmer)
//! Each cascade is fit to a **bounding sphere** of its frustum slice (Valient's stable
//! CSM). A sphere's extent is rotation-invariant, so the ortho box size stays constant as
//! the camera rotates — the projection never resizes. The ortho origin is then **snapped
//! to shadow-texel increments** so a moving camera slides the shadow texels in whole-texel
//! steps instead of sub-texel jitter (the classic edge-crawl / shimmer fix).
//!
//! ## Cascade select + blend
//! The lighting shader (`pbr.slang`) picks the **tightest cascade that contains** the pixel's
//! world position (iterating near → far and testing the projected tile UV), then blends to
//! the next cascade across a small view-depth band (`blend_frac`) at each boundary so the
//! resolution step is not a hard seam. Containment (not the raw depth split) is used because
//! the fitted sphere overshoots the exact slice and the radial view-depth proxy is not the
//! perspective z — containment always lands the highest-resolution cascade that covers.
//!
//! Everything here is **opt-in** behind `CSM` (see `CsmConfig::from_env`): default OFF
//! reproduces the single-map path byte-for-byte (the gallery anchor). The Rust side owns
//! all matrices + split params (single source) and feeds them to the shader via globals.

use dreamcoast_core::glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use rhi::Rect2D;

/// Max cascades the globals block + shader loop support (the array bound, not a tier knob).
pub(crate) const MAX_CASCADES: usize = 4;

/// Atlas slot kinds. Only `Cascade` is filled this PR; `Spot` / `PointFace` are declared so
/// the slot table + the shader's per-slot view-projection array extend to them with no
/// re-wire of the atlas layout or the sampling side (the fill loop is the only future work).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SlotKind {
    /// A directional-light cascade (this PR).
    Cascade,
    /// A spot light's single perspective frustum (skeleton — not filled this PR).
    #[allow(dead_code)]
    Spot,
    /// One face of a point light's cube (skeleton — not filled this PR).
    #[allow(dead_code)]
    PointFace,
}

/// One atlas tile: its kind, its pixel sub-rect in the atlas, and the world→light-clip
/// matrix used to both rasterize the tile and sample it. `split_far` is the view-space far
/// distance this cascade covers (cascade slots only; unused for spot/point).
#[derive(Clone, Copy)]
pub(crate) struct ShadowSlot {
    pub(crate) kind: SlotKind,
    pub(crate) rect: Rect2D,
    pub(crate) view_proj: Mat4,
    pub(crate) split_far: f32,
}

/// Runtime CSM configuration (the scalability seam). `enabled == false` is the single-map
/// legacy path (byte-identical anchor). `cascades` and `atlas_size` are the tier knobs.
#[derive(Clone, Copy)]
pub(crate) struct CsmConfig {
    pub(crate) enabled: bool,
    pub(crate) cascades: usize,
    /// Side length of the (square) atlas texture in texels.
    pub(crate) atlas_size: u32,
    /// Practical-split blend factor (0 = uniform, 1 = logarithmic). ~0.75 is canonical.
    pub(crate) lambda: f32,
    /// Fraction of each cascade's far extent used as the cross-cascade blend band.
    pub(crate) blend_frac: f32,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cascades: 4,
            atlas_size: 4096,
            lambda: 0.75,
            blend_frac: 0.1,
        }
    }
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

impl CsmConfig {
    /// Parse the opt-in seam: `CSM=<N>` (1..=MAX_CASCADES) enables with N cascades; `CSM=1`
    /// is a single cascade (closest to the legacy map). Unset / `CSM=0` = legacy single map.
    /// `CSM_LAMBDA`, `CSM_ATLAS`, `CSM_BLEND` override the tuning knobs when set.
    pub(crate) fn from_env() -> Self {
        let mut cfg = Self::default();
        match std::env::var("CSM").ok().as_deref() {
            None | Some("") | Some("0") => cfg.enabled = false,
            Some(n) => match n.parse::<usize>() {
                Ok(0) => cfg.enabled = false,
                Ok(n) => {
                    cfg.enabled = true;
                    cfg.cascades = n.clamp(1, MAX_CASCADES);
                }
                // Non-numeric (e.g. `CSM=on`): enable with the default cascade count.
                Err(_) => cfg.enabled = true,
            },
        }
        if let Some(l) = env_f32("CSM_LAMBDA") {
            cfg.lambda = l.clamp(0.0, 1.0);
        }
        if let Some(a) = env_u32("CSM_ATLAS") {
            cfg.atlas_size = a.clamp(1024, 8192);
        }
        if let Some(b) = env_f32("CSM_BLEND") {
            cfg.blend_frac = b.clamp(0.0, 0.5);
        }
        cfg.cascades = cfg.cascades.clamp(1, MAX_CASCADES);
        cfg
    }

    /// Grid layout: cascades are packed into the smallest square-ish grid (1→1x1, 2→2x1,
    /// 3/4→2x2). Returns (cols, rows).
    fn grid(&self) -> (u32, u32) {
        match self.cascades {
            1 => (1, 1),
            2 => (2, 1),
            _ => (2, 2),
        }
    }

    /// Per-cascade tile side length in texels (square tiles fill the atlas grid).
    pub(crate) fn tile_size(&self) -> u32 {
        let (cols, rows) = self.grid();
        self.atlas_size / cols.max(rows)
    }
}

/// The view camera the cascades are fit to. Must match the scene camera exactly (same
/// fov/aspect/near/far), so the shadow cascades tile the same depth range the view samples.
#[derive(Clone, Copy)]
pub(crate) struct ViewCamera {
    pub(crate) eye: Vec3,
    pub(crate) target: Vec3,
    pub(crate) fov_y_rad: f32,
    pub(crate) aspect: f32,
    pub(crate) near: f32,
    pub(crate) far: f32,
}

/// Compute the CSM cascade slots for this frame.
///
/// * `cam` — the view camera (used to rebuild the frustum-slice corners).
/// * `sun_dir` — direction *to* the sun (normalized upstream is fine; re-normalized here).
///
/// The fill matrices are backend-neutral — the sampling side handles the Vulkan/D3D12
/// clip-Y flip in-shader, matching the legacy single map. The returned slots are ordered
/// cascade 0 (tightest / nearest) → N-1 (widest / far).
pub(crate) fn compute_cascades(
    cfg: &CsmConfig,
    cam: &ViewCamera,
    sun_dir: [f32; 3],
) -> Vec<ShadowSlot> {
    let (cam_eye, cam_target) = (cam.eye, cam.target);
    let (fov_y_rad, aspect, near, far) = (cam.fov_y_rad, cam.aspect, cam.near, cam.far);
    let n = cfg.cascades.clamp(1, MAX_CASCADES);
    let tile = cfg.tile_size();
    let (cols, _rows) = cfg.grid();

    // Light basis: look from the sun toward the scene. Guard the degenerate up vector.
    let dir = Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]).normalize_or_zero();
    let dir = if dir == Vec3::ZERO { Vec3::Y } else { dir };
    let up = if dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    // Camera view matrix (RH) to rebuild the world-space frustum corners of each slice.
    let cam_up = if (cam_target - cam_eye)
        .normalize_or_zero()
        .dot(Vec3::Y)
        .abs()
        > 0.99
    {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let cam_view = Mat4::look_at_rh(cam_eye, cam_target, cam_up);
    let inv_cam_view = cam_view.inverse();
    let tan_half_v = (fov_y_rad * 0.5).tan();
    let tan_half_h = tan_half_v * aspect;

    // Practical split scheme (log/uniform blend). split[0] = near, split[n] = far.
    let mut splits = vec![near; n + 1];
    for (i, s) in splits.iter_mut().enumerate() {
        let f = i as f32 / n as f32;
        let d_log = near * (far / near).powf(f);
        let d_uni = near + (far - near) * f;
        *s = d_uni * (1.0 - cfg.lambda) + d_log * cfg.lambda;
    }
    splits[0] = near;
    splits[n] = far;

    let mut slots = Vec::with_capacity(n);
    for i in 0..n {
        let near_d = splits[i];
        let far_d = splits[i + 1];

        // Frustum-slice corners in *view* space, then to world.
        let mut corners_world = [Vec3::ZERO; 8];
        let mut k = 0;
        for &d in &[near_d, far_d] {
            let x = tan_half_h * d;
            let y = tan_half_v * d;
            // RH view space looks down -Z, so the slice sits at z = -d.
            for &(sx, sy) in &[(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let v = Vec4::new(sx * x, sy * y, -d, 1.0);
                corners_world[k] = (inv_cam_view * v).xyz();
                k += 1;
            }
        }

        // Bounding sphere of the slice (stable extent, rotation-invariant). Center =
        // centroid; radius = max distance to it. A sphere is used (not a tight AABB in
        // light space) precisely so camera rotation cannot resize the ortho box.
        let mut center = Vec3::ZERO;
        for c in &corners_world {
            center += *c;
        }
        center /= 8.0;
        let mut radius = 0.0f32;
        for c in &corners_world {
            radius = radius.max((*c - center).length());
        }
        // Round the radius up to a stable value so tiny float wobble doesn't rescale it.
        radius = (radius * 16.0).ceil() / 16.0;

        // Ortho box facing the sun, centered on the slice sphere. Push the light back
        // far enough to include casters between the light and the slice.
        let texels_per_unit = tile as f32 / (radius * 2.0);

        // Snap the sphere center to shadow-texel increments *in light space* to kill
        // shimmer: build a provisional light view, snap center.xy to the texel grid.
        // `dir` is the direction TO the sun, so the light sits at `center + dir * pushback`
        // and looks back toward the slice (matching the single-map `light_view_proj`). The
        // earlier `center - dir` put the light underground → casters projected inverted and
        // nothing shadowed (the missing-contact-shadow bug).
        let light_pos0 = center + dir * (radius * 2.0);
        let light_view0 = Mat4::look_at_rh(light_pos0, center, up);
        let center_ls = (light_view0 * center.extend(1.0)).xyz();
        let snapped_x = (center_ls.x * texels_per_unit).floor() / texels_per_unit;
        let snapped_y = (center_ls.y * texels_per_unit).floor() / texels_per_unit;
        let offset_ls = Vec3::new(snapped_x - center_ls.x, snapped_y - center_ls.y, 0.0);

        // Final ortho projection (snapped). Depth range spans the caster pushback + slice.
        let half = radius;
        let z_near = 0.0;
        let z_far = radius * 4.0;
        let mut proj = Mat4::orthographic_rh(-half, half, -half, half, z_near, z_far);
        // Apply the texel snap as a post-projection translation in NDC (offset scaled to
        // NDC by 1/half; z untouched).
        proj.w_axis.x += offset_ls.x / half;
        proj.w_axis.y += offset_ls.y / half;

        let view_proj = proj * light_view0;

        // Atlas tile rect (row-major grid).
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        let rect = Rect2D {
            x: (col * tile) as i32,
            y: (row * tile) as i32,
            width: tile,
            height: tile,
        };

        slots.push(ShadowSlot {
            kind: SlotKind::Cascade,
            rect,
            view_proj,
            split_far: far_d,
        });
    }
    slots
}
