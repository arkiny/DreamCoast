//! Velocity (motion-vector) G-buffer channel — pipeline re-baseline PR-2.
//!
//! Owns the opaque velocity pass: a separate geometry pass (opt-in `P_VELOCITY=1`) that
//! rasterizes the scene into a single `Rg16Float` target holding, per pixel, the screen-space
//! motion `cur_ndc.xy − prev_ndc.xy` of that surface between the previous and current frame.
//! It is a standalone pass (not folded into `record_gbuffer`) so the 4-MRT G-buffer stays
//! byte-identical when velocity is off — the whole feature is gated on this pass simply not
//! being recorded (default off).
//!
//! Prev-transform single source: the caller supplies each drawable's PREVIOUS unjittered
//! `view_proj * prev_model` (static / Spin / node animation), and — for skinned / morphed
//! drawables — the previous-frame joint palette / weights buffer index, so the vertex shader
//! reconstructs the previous surface point and the motion follows the deform, not just the node.
//!
//! Both the current and previous matrices here are UNJITTERED (the TAA sub-pixel jitter must
//! not enter the motion vector; see `velocity.slang`). The `csViz` compute pass colour-codes
//! the target for `DEBUG_VIEW=11`.

use dreamcoast_core::glam::Mat4;
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, ClearColor, ComputePipeline, ComputePipelineDesc, DepthCompare, Device,
    Extent2D, GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, VertexLayout,
};

use crate::app::{load_compute_shader, load_shader_pair};
use crate::{DEPTH_FORMAT, HDR_FORMAT, SceneObject};

/// The velocity target format: RG16Float holds signed NDC motion with ample precision (the
/// canonical motion-vector format).
pub(crate) const VELOCITY_FMT: rhi::Format = rhi::Format::Rg16Float;

pub(crate) struct VelocitySystem {
    /// Static / Spin / node-animation motion pass.
    pipeline: GraphicsPipeline,
    /// GPU-skinned motion (prev palette).
    skinned_pipeline: GraphicsPipeline,
    /// GPU-morph motion (prev weights).
    morphed_pipeline: GraphicsPipeline,
    /// Baked vertex-cache (deform) motion (prev-frame position storage buffer).
    deform_pipeline: GraphicsPipeline,
    /// DEBUG_VIEW=11 colour-code compute.
    viz_pipeline: ComputePipeline,
}

impl VelocitySystem {
    pub(crate) fn new(device: &Device, backend: BackendKind) -> anyhow::Result<Self> {
        let mk = |vs_key,
                  vs_spirv: fn() -> Option<&'static [u8]>,
                  vs_dxil: fn() -> Option<&'static [u8]>,
                  vs_metallib: fn() -> Option<&'static [u8]>,
                  entry: &'static str|
         -> anyhow::Result<GraphicsPipeline> {
            let (vs, fs) = load_shader_pair(
                backend,
                vs_spirv,
                dreamcoast_shader::velocity_fs_spirv,
                vs_dxil,
                dreamcoast_shader::velocity_fs_dxil,
                vs_metallib,
                dreamcoast_shader::velocity_fs_metallib,
                vs_key,
            )?;
            Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: vs,
                fragment_bytes: fs,
                vertex_entry: entry,
                fragment_entry: "fsMain",
                color_formats: &[VELOCITY_FMT],
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::Mesh,
                blend: BlendMode::Opaque,
                push_constant_size: 208, // mvp(64)+prev_mvp(64)+skin(16)+skin_prev(16)+morph(16)+morph_prev(16)+deform_prev(16)
                bindless: true,
                uniform_buffer: false,
                depth_test: true,
                depth_write: false, // depth already written by the G-buffer fill; test-only (no perturb)
                // The velocity pass re-draws the opaque surface at the SAME depth the G-buffer fill
                // wrote, using that shared depth as its z-test to pick the nearest surface per pixel.
                // That is an EQUAL-depth match, so the compare must admit equality — `Less` rejected
                // every coplanar fragment (the pass wrote nothing; the velocity target stayed cleared
                // to zero, so even camera / skin / morph / deform motion never appeared). `LessEqual`
                // is the documented "equal-ish against the shared depth" intent.
                depth_compare: DepthCompare::LessEqual,
                depth_format: Some(DEPTH_FORMAT),
            })?)
        };
        let pipeline = mk(
            "velocity",
            dreamcoast_shader::velocity_vs_spirv,
            dreamcoast_shader::velocity_vs_dxil,
            dreamcoast_shader::velocity_vs_metallib,
            "vsMain",
        )?;
        let skinned_pipeline = mk(
            "velocity_skinned",
            dreamcoast_shader::velocity_skinned_vs_spirv,
            dreamcoast_shader::velocity_skinned_vs_dxil,
            dreamcoast_shader::velocity_skinned_vs_metallib,
            "vsMainSkinned",
        )?;
        let morphed_pipeline = mk(
            "velocity_morphed",
            dreamcoast_shader::velocity_morphed_vs_spirv,
            dreamcoast_shader::velocity_morphed_vs_dxil,
            dreamcoast_shader::velocity_morphed_vs_metallib,
            "vsMainMorphed",
        )?;
        let deform_pipeline = mk(
            "velocity_deform",
            dreamcoast_shader::velocity_deform_vs_spirv,
            dreamcoast_shader::velocity_deform_vs_dxil,
            dreamcoast_shader::velocity_deform_vs_metallib,
            "vsMainDeform",
        )?;
        let viz_cs = load_compute_shader(
            backend,
            dreamcoast_shader::velocity_viz_cs_spirv,
            dreamcoast_shader::velocity_viz_cs_dxil,
            dreamcoast_shader::velocity_viz_cs_metallib,
            "velocity_viz",
        )?;
        let viz_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: viz_cs,
            compute_entry: "csViz",
            push_constant_size: 32, // velocity_index + out_index + w + h + scale + pad(3)
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [8, 8, 1],
        })?;
        Ok(Self {
            pipeline,
            skinned_pipeline,
            morphed_pipeline,
            deform_pipeline,
            viz_pipeline,
        })
    }

    /// Record the velocity pass into `target` (RG16Float, cleared to 0 = no motion). `view_proj`
    /// and `prev_view_proj` are the UNJITTERED current / previous camera matrices; `prev_scene`
    /// carries each drawable's previous model transform + previous skin/morph indices in the same
    /// order as `scene` (single-source prev pose). The ground plane (identity, static) is drawn
    /// with `prev_model == model`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        target: ResourceId,
        depth: ResourceId,
        scene: &'a [SceneObject],
        prev_scene: &'a [PrevPose],
        ground_vbuf: &'a rhi::Buffer,
        ground_ibuf: &'a rhi::Buffer,
        ground_count: u32,
        view_proj: Mat4,
        prev_view_proj: Mat4,
    ) {
        graph.add_pass(
            PassInfo {
                name: "velocity",
                // Clear to zero motion; depth is LOADED from the G-buffer fill (already written)
                // so the velocity pass depth-tests against the opaque surface (Equal-ish via the
                // shared depth). We keep depth_write on but the values match the fill.
                colors: vec![(target, Some(ClearColor::BLACK))],
                depth: Some(depth),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.pipeline);
                for (i, obj) in scene.iter().enumerate() {
                    // Decals tint albedo only (no motion of their own) — skip, matching the
                    // G-buffer fill's decal skip so the velocity of the underlying surface shows.
                    if obj.kind == dreamcoast_asset::MaterialKind::Decal {
                        continue;
                    }
                    let prev = prev_scene.get(i).copied().unwrap_or_default();
                    let mvp = (view_proj * obj.transform).to_cols_array();
                    let prev_mvp = (prev_view_proj * prev.transform).to_cols_array();
                    if obj.skin.is_some() {
                        cmd.bind_graphics_pipeline(&self.skinned_pipeline);
                    } else if obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(&self.morphed_pipeline);
                    } else if obj.deform.is_some() {
                        // Baked vertex-cache: current positions from this frame's ring VB (obj.mesh),
                        // previous from the prev-frame position storage buffer (`obj.deform`). The
                        // static placement is already in `mvp`/`prev_mvp`, so the shader just swaps
                        // the previous position — the deform motion falls out of the subtraction.
                        cmd.bind_graphics_pipeline(&self.deform_pipeline);
                    }
                    cmd.push_constants(&velocity_push(
                        mvp,
                        prev_mvp,
                        obj.skin.unwrap_or([0; 4]),
                        [prev.skin_palette, 0, 0, 0],
                        obj.morph.unwrap_or([0; 4]),
                        [prev.morph_weights, 0, 0, 0],
                        [obj.deform.unwrap_or(0), 0, 0, 0],
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                    if obj.skin.is_some() || obj.morph.is_some() || obj.deform.is_some() {
                        cmd.bind_graphics_pipeline(&self.pipeline); // restore
                    }
                }
                // Ground plane: static (identity transform), prev == curr.
                let g_mvp = view_proj.to_cols_array();
                let g_prev = prev_view_proj.to_cols_array();
                cmd.push_constants(&velocity_push(
                    g_mvp, g_prev, [0; 4], [0; 4], [0; 4], [0; 4], [0; 4],
                ));
                cmd.bind_vertex_buffer(ground_vbuf, 32);
                cmd.bind_index_buffer(ground_ibuf, true);
                cmd.draw_indexed(ground_count, 0, 0);
                Ok(())
            },
        );
    }

    /// DEBUG_VIEW=11: colour-code the velocity target into an HDR storage image the tonemap
    /// displays. `scale` amplifies the small NDC motion to a visible range.
    pub(crate) fn record_viz<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        velocity: ResourceId,
        extent: Extent2D,
        w: u32,
        h: u32,
        scale: f32,
    ) -> ResourceId {
        let out = graph.create_storage_image("velocity_viz", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "velocity_viz",
                storage_writes: vec![out],
                reads: vec![velocity],
            },
            move |ctx| {
                let vel_index = ctx.sampled_index(velocity);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(&self.viz_pipeline);
                cmd.push_constants_compute(&velocity_viz_push(vel_index, out_index, w, h, scale));
                cmd.dispatch(w.div_ceil(8), h.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}

/// Each drawable's PREVIOUS-frame pose — the single source the velocity pass reads to compute
/// per-object motion. Parallel (same index) to the frame's `scene` list.
#[derive(Clone, Copy)]
pub(crate) struct PrevPose {
    /// Previous unjittered world transform (identity for skinned draws — the palette carries it).
    pub(crate) transform: Mat4,
    /// Previous-frame joint-palette bindless index (skinned drawables), else 0.
    pub(crate) skin_palette: u32,
    /// Previous-frame morph-weights bindless index (morphed drawables), else 0.
    pub(crate) morph_weights: u32,
}

impl Default for PrevPose {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            skin_palette: 0,
            morph_weights: 0,
        }
    }
}

/// Pack the velocity push block (208 bytes): mvp(64) + prev_mvp(64) + skin u32x4(16) +
/// skin_prev u32x4(16) + morph u32x4(16) + morph_prev u32x4(16) + deform_prev u32x4(16).
#[allow(clippy::too_many_arguments)]
fn velocity_push(
    mvp: [f32; 16],
    prev_mvp: [f32; 16],
    skin: [u32; 4],
    skin_prev: [u32; 4],
    morph: [u32; 4],
    morph_prev: [u32; 4],
    deform_prev: [u32; 4],
) -> [u8; 208] {
    let mut pc = [0u8; 208];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in prev_mvp.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, s) in skin.iter().enumerate() {
        let o = 128 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in skin_prev.iter().enumerate() {
        let o = 144 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in morph.iter().enumerate() {
        let o = 160 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in morph_prev.iter().enumerate() {
        let o = 176 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in deform_prev.iter().enumerate() {
        let o = 192 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    pc
}

/// Pack the velocity debug-viz push block (32 bytes): velocity + out indices, width, height,
/// and the display scale (the rest is padding to the 16-byte tail).
fn velocity_viz_push(velocity: u32, out: u32, width: u32, height: u32, scale: f32) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&velocity.to_le_bytes());
    pc[4..8].copy_from_slice(&out.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc[16..20].copy_from_slice(&scale.to_le_bytes());
    pc
}
