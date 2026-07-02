//! Forward translucency pass (PR-3, render-pipeline re-baseline track,
//! `docs/render-pipeline-reference.md` §1.7 #12 / §3 PR-3). See `docs/translucency-pass.md`.
//!
//! Deferred shading stores one opaque surface per G-buffer texel, so semi-transparent
//! geometry can't go through it. This module adds the canonical hybrid slot: a SEPARATE
//! forward pass drawn AFTER the opaque scene color is fully composited (deferred lighting,
//! reflections, fog) and BEFORE post (TAAU/tonemap). It rasterizes each translucent
//! object into the HDR scene color with: depth-test ON (opaque geometry, via the shared
//! depth buffer, occludes translucency); depth-write OFF (overlapping translucent layers
//! all blend, none Z-rejects another); alpha blend (SRC_ALPHA / ONE_MINUS_SRC_ALPHA).
//! No G-buffer is written. Lighting is simple forward PBR reusing the SAME Cook-Torrance
//! BRDF as the deferred pass (`pbr_brdf.slang`) plus IBL ambient — single source, no drift.
//!
//! The caller CPU-sorts objects back-to-front (deterministic: distance, then object index
//! tie-break) so the alpha-over sequence is correct. OIT / refraction / a dedicated
//! translucency lighting volume are Phase 20 extensions of this slot.
//!
//! The slot costs nothing when there are no translucent objects: `record` returns without
//! adding a pass, so an empty scene is byte-identical (the gallery anchor). glTF
//! `Transparent` (BLEND, non-decal) materials can be routed here by the material-kind
//! branch in `main.rs`; the foliage alpha-cutout path (opaque G-buffer) is untouched.

use std::rc::Rc;

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, Buffer, DepthCompare, Device, Format, GraphicsPipeline,
    GraphicsPipelineDesc, PrimitiveTopology, VertexLayout,
};

use crate::NO_TEXTURE;
use crate::app::load_shader_pair;
use crate::push::translucent_push;
use crate::registry::GpuMesh;

/// One translucent drawable: a shared GPU mesh + world transform + resolved material.
/// Kept apart from `SceneObject` because the translucency pass sorts these back-to-front
/// each frame (the opaque draw list is order-independent).
pub(crate) struct TranslucentObject {
    pub(crate) mesh: Rc<GpuMesh>,
    pub(crate) transform: Mat4,
    /// rgb tint (linear), a = coverage/opacity for the alpha blend.
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    /// Base-color texture bindless index (`NO_TEXTURE` = untinted).
    pub(crate) base_tex: u32,
    /// Object-space centroid, transformed to world for the back-to-front sort key.
    pub(crate) center: Vec3,
}

impl TranslucentObject {
    /// Convert an opaque-list [`crate::SceneObject`] into a translucent drawable — the glTF
    /// `Transparent` (BLEND) routing skeleton (PR-3). Reuses the object's mesh/transform and
    /// its resolved base color / metallic / roughness / base-color texture; the sort centroid
    /// is the object origin in world space. NOTE: the caller must also remove this object from
    /// the opaque G-buffer draw list (else it renders twice) — see `main.rs` / the Phase 20
    /// hookup described in `docs/translucency-pass.md`.
    #[allow(dead_code)] // routing skeleton — wired in the Phase 20 glTF-BLEND hookup
    pub(crate) fn from_scene(obj: &crate::SceneObject) -> Self {
        Self {
            mesh: obj.mesh.clone(),
            transform: obj.transform,
            base_color: obj.base_color,
            metallic: obj.metallic,
            roughness: obj.roughness,
            base_tex: obj.tex[0],
            center: obj.transform.transform_point3(Vec3::ZERO),
        }
    }
}

/// The forward translucency pipeline (alpha blend, depth-test on / write off).
pub(crate) struct TranslucencySystem {
    pipeline: GraphicsPipeline,
}

impl TranslucencySystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        hdr_format: Format,
        depth_format: Format,
    ) -> anyhow::Result<Self> {
        let (vs, fs) = load_shader_pair(
            backend,
            dreamcoast_shader::translucent_vs_spirv,
            dreamcoast_shader::translucent_fs_spirv,
            dreamcoast_shader::translucent_vs_dxil,
            dreamcoast_shader::translucent_fs_dxil,
            dreamcoast_shader::translucent_vs_metallib,
            dreamcoast_shader::translucent_fs_metallib,
            "translucent",
        )?;
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vs,
            fragment_bytes: fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[hdr_format],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::Mesh, // pos + normal + uv
            blend: BlendMode::AlphaBlend,
            push_constant_size: 176, // see push::translucent_push
            bindless: true,
            uniform_buffer: true, // reads the per-frame Globals UBO (sun + IBL + exposure)
            depth_test: true,     // opaque geometry occludes translucency
            depth_write: false,   // overlapping translucent layers all blend
            depth_compare: DepthCompare::Less,
            depth_format: Some(depth_format),
        })?;
        Ok(Self { pipeline })
    }

    /// Record the sorted translucency pass. `objects` is drawn back-to-front (sorted here,
    /// deterministic: distance from `eye`, then original index tie-break) into `hdr` with the
    /// shared opaque `depth` buffer bound read-only. Returns without adding a pass when
    /// `objects` is empty (zero cost, byte-identical baseline).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        depth: ResourceId,
        objects: &'a [TranslucentObject],
        view_proj: Mat4,
        eye: Vec3,
        globals_buffer: &'a Buffer,
        globals_offset: u64,
        flip_y: u32,
        shadow_map: ResourceId,
    ) {
        if objects.is_empty() {
            return;
        }
        // Back-to-front order: farthest first. Squared distance is monotonic (cheaper); the
        // original index is the deterministic tie-break so two coincident planes have a
        // stable, run-to-run identical order (cross-backend determinism).
        let mut order: Vec<usize> = (0..objects.len()).collect();
        order.sort_by(|&a, &b| {
            let da = (objects[a].center - eye).length_squared();
            let db = (objects[b].center - eye).length_squared();
            db.partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let pipeline = &self.pipeline;
        graph.add_pass(
            PassInfo {
                name: "translucency",
                // `None` clear = LOAD the finished opaque/fog HDR (blend over it).
                colors: vec![(hdr, None)],
                depth: Some(depth), // depth-test only (pipeline has depth-write off)
                reads: vec![shadow_map],
            },
            move |ctx| {
                let shadow_index = ctx.sampled_index(shadow_map);
                let cmd = ctx.cmd();
                cmd.set_globals(globals_buffer, globals_offset);
                cmd.bind_graphics_pipeline(pipeline);
                for &i in &order {
                    let obj = &objects[i];
                    let mvp = (view_proj * obj.transform).to_cols_array();
                    let model = obj.transform.to_cols_array();
                    cmd.push_constants(&translucent_push(
                        &mvp,
                        &model,
                        obj.base_color,
                        obj.metallic,
                        obj.roughness,
                        obj.base_tex,
                        flip_y,
                        shadow_index,
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                }
                Ok(())
            },
        );
    }
}

/// Build the `P_TRANSLUCENT_TEST=1` demo: two overlapping tinted glass planes standing in
/// the gallery, tilted so scene geometry behind them shows through the alpha blend and the
/// two panes overlap (to exercise the back-to-front sort). `scene_radius` scales the panes
/// to the current scene; `center` is the scene centroid they stand near.
pub(crate) fn translucent_test_planes(
    device: &Device,
    scene_radius: f32,
    center: Vec3,
) -> anyhow::Result<Vec<TranslucentObject>> {
    use crate::mesh::upload_geometry;
    use dreamcoast_asset::MeshVertex;

    // A unit quad in the XY plane (normal +Z), spanning [-1,1] in x/y. Scaled + placed by
    // the per-instance transform below.
    let quad = [
        MeshVertex {
            pos: [-1.0, -1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        },
        MeshVertex {
            pos: [1.0, -1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
        },
        MeshVertex {
            pos: [1.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
        },
        MeshVertex {
            pos: [-1.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
        },
    ];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let (vbuf, ibuf, index_count) = upload_geometry(device, &quad, &indices)?;
    let mesh = Rc::new(GpuMesh {
        vbuf,
        ibuf,
        index_count,
        vertex_count: quad.len() as u32,
    });

    let s = scene_radius * 0.55; // pane half-size
    let h = center.y.max(scene_radius * 0.5); // stand near the scene's vertical middle
    // Two panes, offset in X and depth, each tilted ~12° off vertical so the camera sees
    // through them at an angle and they overlap in screen space (the near pane covers part
    // of the far pane — proves the back-to-front sort puts the near one on top).
    let tilt = 12f32.to_radians();
    let build = |offset: Vec3, tint: [f32; 4]| {
        let transform = Mat4::from_translation(center + offset)
            * Mat4::from_rotation_y(tilt)
            * Mat4::from_scale(Vec3::new(s, s, 1.0));
        TranslucentObject {
            mesh: mesh.clone(),
            transform,
            base_color: tint,
            metallic: 0.0,
            roughness: 0.08, // smooth glass
            base_tex: NO_TEXTURE,
            center: center + offset,
        }
    };
    Ok(vec![
        // Far pane: cool blue-green glass.
        build(
            Vec3::new(-s * 0.35, h - center.y, -s * 0.30),
            [0.35, 0.75, 0.85, 0.4],
        ),
        // Near pane: warm amber glass, overlapping the far one.
        build(
            Vec3::new(s * 0.35, h - center.y, s * 0.30),
            [0.90, 0.55, 0.25, 0.4],
        ),
    ])
}
