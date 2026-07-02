//! View Family (render-pipeline re-baseline PR-9, `docs/view-family.md`).
//!
//! A **view family** renders N points-of-view (`SceneView`) in one frame against a
//! single set of **view-independent** scene resources — shadow/CSM atlas, IBL cubes,
//! the global distance field / surface cache, the clustered-light buffers. Those passes
//! depend only on the scene + sun, so they run **once** and every view samples the same
//! result. Only the **view-dependent** passes (depth pre-pass, G-buffer, screen-space AO/GI,
//! lighting, reflections, fog, translucency, the post chain, tonemap) re-run per view,
//! parameterized by that view's camera math.
//!
//! This mirrors the canonical reference-engine split: a view *family* owns the shared scene
//! render, and each *scene view* is a camera + per-view feature set rendering into its own
//! target (screen region, or an offscreen capture texture for a future real-time env probe /
//! editor viewport). See `docs/view-family.md` §"View-dependent vs. view-independent".
//!
//! This module is RHI-free: it speaks only `glam` — the frame loop turns a `SceneView` into
//! graph passes.

use dreamcoast_core::glam::{Mat4, Vec3};
use rhi::BackendKind;

/// Per-view feature toggles. The **default (`SceneViewFeatures::full`) reproduces the legacy
/// single-view path exactly** (byte-identical golden anchor). A secondary/inset view opts into
/// a *simplified* path by clearing the temporal/post toggles — and crucially that simplification
/// is expressed **in the view descriptor**, not by a scattered `if second_view` check, so adding
/// a third view (e.g. a scene-capture probe) is a matter of choosing its feature set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneViewFeatures {
    /// Run the temporal upscale/AA (TAAU) reconstruction for this view. The secondary view turns
    /// this **off**: its temporal history buffers are per-view state (`TaauSystem` owns one set),
    /// so a second view without its own history must not touch the main view's — clearing the flag
    /// keeps the temporal state view-count-safe by construction.
    pub(crate) taau: bool,
    /// Emit the velocity (motion-vector) pass + consume it (motion blur, velocity-aware TAAU).
    /// Off for the secondary view (no temporal consumer → nothing to feed).
    pub(crate) velocity: bool,
    /// Run the ordered post chain (motion blur / bloom / DoF / grading). Off for the secondary
    /// view → its tonemap is the minimal ACES+sRGB encode.
    pub(crate) post: bool,
    /// Run the screen-space indirect passes (GDF AO/GI, GTAO, SSR/reflections). Off for the
    /// secondary view → it lights from IBL + direct only (a documented simplification: the
    /// secondary view is a cheap situational-awareness inset, not a parity reference).
    pub(crate) screen_space_gi: bool,
}

impl SceneViewFeatures {
    /// The full-fat feature set — the default single-view path. Byte-identical to pre-PR-9.
    pub(crate) fn full() -> Self {
        Self {
            taau: true,
            velocity: true,
            post: true,
            screen_space_gi: true,
        }
    }

    /// The simplified secondary/inset feature set: no temporal (TAAU/velocity), no post chain,
    /// no screen-space GI. Keeps the second view cheap and, more importantly, free of any
    /// per-view temporal state that would collide with the primary view.
    pub(crate) fn secondary() -> Self {
        Self {
            taau: false,
            velocity: false,
            post: false,
            screen_space_gi: false,
        }
    }
}

/// A single point-of-view within the frame's view family. Holds every **view-dependent**
/// quantity: the camera pose, the derived (Y-flip-aware) view/projection matrices, the per-view
/// temporal-history matrices, the sub-pixel jitter, this view's globals-UBO slice offset, and its
/// per-view feature toggles. The frame loop builds one per view and drives the view-dependent
/// passes from it; the view-independent passes (shadow/IBL/GDF/cluster) are built once, outside.
#[derive(Clone, Debug)]
pub(crate) struct SceneView {
    /// Debug/label index (0 = primary).
    pub(crate) index: u32,
    /// Camera eye (world space).
    pub(crate) eye: Vec3,
    /// Look-at target (world space).
    pub(crate) focus: Vec3,
    /// Y-flipped (backend-correct) view-projection — the matrix the raster passes use.
    pub(crate) view_proj: Mat4,
    /// The UN-jittered view-projection — the stable grid TAAU history lives on.
    pub(crate) view_proj_stable: Mat4,
    /// World->view row 2 (for clustered froxel view-Z reconstruction) is derived at use; we keep
    /// the raw view matrix so the frame loop can pull the row + the camera basis.
    pub(crate) view: Mat4,
    /// clip->world (skybox ray reconstruction + TAAU world-point reprojection).
    pub(crate) inv_view_proj: [f32; 16],
    /// Previous-frame (stable) view-projection for this view's temporal reprojection (SSR/TAAU).
    pub(crate) prev_view_proj: [f32; 16],
    /// TAA sub-pixel jitter as a UV shift (0 when this view has no jitter).
    pub(crate) jitter_uv: [f32; 2],
    /// This view's globals-UBO byte offset (`(fif * MAX_VIEWS + index) * GLOBALS_SLICE`).
    pub(crate) globals_offset: u64,
    /// Per-view feature toggles.
    pub(crate) features: SceneViewFeatures,
}

impl SceneView {
    /// Build a view's camera math from a `(eye, focus)` pose and the shared projection params.
    /// `backend` applies the Vulkan clip-space Y flip. Jitter is left zero here; the primary
    /// view's TAAU jitter is applied by the caller (which owns the Halton sequence + frame index).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        index: u32,
        eye: Vec3,
        focus: Vec3,
        aspect: f32,
        fov_y_rad: f32,
        z_near: f32,
        z_far: f32,
        backend: BackendKind,
        globals_offset: u64,
        prev_view_proj: [f32; 16],
        features: SceneViewFeatures,
    ) -> Self {
        let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
        let mut proj = Mat4::perspective_rh(fov_y_rad, aspect, z_near, z_far);
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0;
        }
        let view_proj = proj * view;
        Self {
            index,
            eye,
            focus,
            view_proj,
            view_proj_stable: view_proj,
            view,
            inv_view_proj: view_proj.inverse().to_cols_array(),
            prev_view_proj,
            jitter_uv: [0.0, 0.0],
            globals_offset,
            features,
        }
    }

    /// Row 2 of the world->view matrix (clustered froxel view-Z reconstruction). `view` is
    /// column-major (glam), so row 2 is the `.z` of each column.
    pub(crate) fn cluster_view_z_row(&self) -> [f32; 4] {
        [
            self.view.x_axis.z,
            self.view.y_axis.z,
            self.view.z_axis.z,
            self.view.w_axis.z,
        ]
    }
}
