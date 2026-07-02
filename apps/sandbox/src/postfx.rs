//! Ordered post-process nodes — pipeline re-baseline PR-5
//! (`docs/render-pipeline-reference.md` §3, `docs/post-process-chain.md`).
//!
//! This module owns the post-process node sequence that sits between the finished
//! opaque HDR scene color (lighting + reflections + the atmosphere/fog slot) and the
//! tonemap. The reference ordering is:
//!
//! ```text
//! (opaque HDR) -> motion-blur -> TAA/upscale -> auto-exposure -> bloom -> DoF -> tonemap+grading
//! ```
//!
//! Motion blur / bloom / DoF live here; TAA/upscale is `taau.rs`, auto-exposure +
//! tonemap+grading are `deferred.rs`. Every node follows the graph's "read old, write
//! new, rethread" convention (like `atmosphere.rs`): it reads the current HDR target
//! and writes a NEW one the caller threads forward. Each node is opt-in — the pipeline
//! is always built (so all three backends compile the shader) but the `record_*` call
//! is only issued when the feature flag is on, so leaving it off adds no pass and the
//! golden anchor stays byte-identical.
//!
//! - **Motion blur** (`P_MOTION_BLUR=1`, requires `P_VELOCITY=1`): a per-pixel
//!   velocity-along blur consuming the PR-2 velocity target. `motion_blur.slang`.
//! - **Bloom** (`P_BLOOM=1`): a progressive dual-filter bloom (Jimenez CoD:AW) — a
//!   Karis-averaged bright-pass prefilter, a 13-tap downsample pyramid, and 3x3 tent
//!   upsamples that additively accumulate. Composited additively at the tonemap input.
//!   `bloom.slang`. All tunables live in the `BloomParams` constant block (single
//!   source, ready to split into a RenderQuality tier).
//! - **DoF** (stub): a passthrough node reserving the slot for Phase 20. `dof.slang`.

use dreamcoast_render::{PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, DepthCompare, Device, Extent2D, Format, GraphicsPipeline,
    GraphicsPipelineDesc, PrimitiveTopology, VertexLayout,
};

use crate::app::load_shader_pair;

/// Sentinel bindless index meaning "no texture" for the optional composite inputs
/// (bloom's additive mip, the tonemap bloom slot). Matches the shaders' `0xFFFFFFFF`.
const NO_INDEX: u32 = u32::MAX;

// ----------------------------------------------------------------------------------
// Bloom parameters — single source of truth (CLAUDE.md rule 3/4). Physically-plausible
// defaults: a soft bright-pass knee just above the diffuse mid-grey range so only
// genuine highlights (specular, emissive, sky) bloom, a modest additive intensity, and
// a 5-level pyramid (mip0 at half render-res down to 1/32) — enough spread for a wide,
// smooth glow without the cost of a full mip chain. Ready to become a RenderQuality
// {low: 4 mips, high: 6} tier by swapping these constants.
// ----------------------------------------------------------------------------------
pub(crate) struct BloomParams;
impl BloomParams {
    /// Bright-pass luminance knee (linear HDR). Highlights above this bloom.
    pub(crate) const THRESHOLD: f32 = 1.0;
    /// Additive composite scale at the tonemap input.
    pub(crate) const INTENSITY: f32 = 0.06;
    /// Number of pyramid mips (mip0 = half render-res). 5 -> down to ~1/32 res.
    pub(crate) const MIPS: u32 = 5;
    /// HDR pyramid format (matches the scene HDR target).
    pub(crate) const FMT: Format = crate::HDR_FORMAT;
}

/// The bloom pyramid pipelines: one vertex shader shared by three fragment entries
/// (prefilter / downsample / upsample).
pub(crate) struct BloomSystem {
    prefilter: GraphicsPipeline,
    downsample: GraphicsPipeline,
    upsample: GraphicsPipeline,
}

impl BloomSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        hdr_format: Format,
    ) -> anyhow::Result<Self> {
        let mk = |fs_spirv: fn() -> Option<&'static [u8]>,
                  fs_dxil: fn() -> Option<&'static [u8]>,
                  fs_metallib: fn() -> Option<&'static [u8]>,
                  fs_entry: &'static str|
         -> anyhow::Result<GraphicsPipeline> {
            let (vs, fs) = load_shader_pair(
                backend,
                dreamcoast_shader::bloom_vs_spirv,
                fs_spirv,
                dreamcoast_shader::bloom_vs_dxil,
                fs_dxil,
                dreamcoast_shader::bloom_vs_metallib,
                fs_metallib,
                "bloom",
            )?;
            Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: vs,
                fragment_bytes: fs,
                vertex_entry: "vsMain",
                fragment_entry: fs_entry,
                color_formats: &[hdr_format],
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::None,
                blend: BlendMode::Opaque, // additive is done in-shader (read + add + write)
                push_constant_size: 32,   // see bloom_push
                bindless: true,
                uniform_buffer: false,
                depth_test: false,
                depth_write: false,
                depth_compare: DepthCompare::Less,
                depth_format: None,
            })?)
        };
        let prefilter = mk(
            dreamcoast_shader::bloom_prefilter_fs_spirv,
            dreamcoast_shader::bloom_prefilter_fs_dxil,
            dreamcoast_shader::bloom_prefilter_fs_metallib,
            "fsPrefilter",
        )?;
        let downsample = mk(
            dreamcoast_shader::bloom_downsample_fs_spirv,
            dreamcoast_shader::bloom_downsample_fs_dxil,
            dreamcoast_shader::bloom_downsample_fs_metallib,
            "fsDownsample",
        )?;
        let upsample = mk(
            dreamcoast_shader::bloom_upsample_fs_spirv,
            dreamcoast_shader::bloom_upsample_fs_dxil,
            dreamcoast_shader::bloom_upsample_fs_metallib,
            "fsUpsample",
        )?;
        Ok(Self {
            prefilter,
            downsample,
            upsample,
        })
    }

    /// Record the whole bloom pyramid and return the mip0 (finest) result — the target
    /// the tonemap pass additively composites onto the HDR. Reads `hdr_in` (the finished
    /// opaque scene color); the scene HDR is never mutated (read old, write new).
    ///
    /// Pyramid: prefilter (bright-pass + Karis 13-tap downsample) into mip0, then plain
    /// 13-tap downsamples down the chain, then 3x3 tent upsamples that add each coarser
    /// mip back onto the next-finer one. The returned mip0 holds the accumulated glow.
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr_in: ResourceId,
        render_extent: Extent2D,
        flip_y: u32,
    ) -> ResourceId {
        // Build the descending-extent mip list (mip0 = half render-res). Guard against
        // degenerate tiny extents by stopping at 1px and capping to MIPS.
        static DOWN_NAMES: [&str; 6] = [
            "bloom_d0", "bloom_d1", "bloom_d2", "bloom_d3", "bloom_d4", "bloom_d5",
        ];
        static UP_NAMES: [&str; 6] = [
            "bloom_u0", "bloom_u1", "bloom_u2", "bloom_u3", "bloom_u4", "bloom_u5",
        ];
        let mut extents: Vec<Extent2D> = Vec::new();
        let mut w = render_extent.width.max(1);
        let mut h = render_extent.height.max(1);
        for _ in 0..BloomParams::MIPS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            extents.push(Extent2D::new(w, h));
            if w == 1 && h == 1 {
                break;
            }
        }
        let n = extents.len();

        // Downsample chain: mip0 = prefilter(hdr_in), mip[k] = downsample(mip[k-1]).
        let mut down: Vec<ResourceId> = Vec::with_capacity(n);
        for (k, ext) in extents.iter().enumerate() {
            let dst = graph.create_color(DOWN_NAMES[k], BloomParams::FMT, *ext);
            let (src, src_ext, pipeline) = if k == 0 {
                (hdr_in, render_extent, &self.prefilter)
            } else {
                (down[k - 1], extents[k - 1], &self.downsample)
            };
            let src_texel = [1.0 / src_ext.width as f32, 1.0 / src_ext.height as f32];
            let threshold = if k == 0 { BloomParams::THRESHOLD } else { 0.0 };
            graph.add_pass(
                PassInfo {
                    name: DOWN_NAMES[k],
                    colors: vec![(dst, None)],
                    depth: None,
                    reads: vec![src],
                },
                move |ctx| {
                    let src_index = ctx.sampled_index(src);
                    let cmd = ctx.cmd();
                    cmd.bind_graphics_pipeline(pipeline);
                    cmd.push_constants(&bloom_push(
                        src_index, NO_INDEX, flip_y, src_texel, threshold, 0.0,
                    ));
                    cmd.draw(3, 1);
                    Ok(())
                },
            );
            down.push(dst);
        }

        // Upsample chain: start from the coarsest downsample, tent-upsample it and add
        // the next-finer downsample, walking back up to mip0. Each up[k] is a NEW target
        // at the finer mip's extent. up[n-1] aliases the coarsest down (nothing to add).
        // The returned value is up[0] at the finest (mip0) extent.
        let mut prev_up = down[n - 1];
        let mut prev_ext = extents[n - 1];
        for k in (0..n - 1).rev() {
            let dst = graph.create_color(UP_NAMES[k], BloomParams::FMT, extents[k]);
            let coarse = prev_up;
            let coarse_ext = prev_ext;
            let fine = down[k];
            let src_texel = [
                1.0 / coarse_ext.width as f32,
                1.0 / coarse_ext.height as f32,
            ];
            graph.add_pass(
                PassInfo {
                    name: UP_NAMES[k],
                    colors: vec![(dst, None)],
                    depth: None,
                    reads: vec![coarse, fine],
                },
                move |ctx| {
                    let src_index = ctx.sampled_index(coarse);
                    let add_index = ctx.sampled_index(fine);
                    let cmd = ctx.cmd();
                    cmd.bind_graphics_pipeline(&self.upsample);
                    cmd.push_constants(&bloom_push(
                        src_index, add_index, flip_y, src_texel, 0.0, 0.0,
                    ));
                    cmd.draw(3, 1);
                    Ok(())
                },
            );
            prev_up = dst;
            prev_ext = extents[k];
        }
        prev_up
    }
}

/// Motion blur node — pipeline re-baseline PR-5 post node #13 (`motion_blur.slang`).
pub(crate) struct MotionBlurSystem {
    pipeline: GraphicsPipeline,
}

impl MotionBlurSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        hdr_format: Format,
    ) -> anyhow::Result<Self> {
        let (vs, fs) = load_shader_pair(
            backend,
            dreamcoast_shader::motion_blur_vs_spirv,
            dreamcoast_shader::motion_blur_fs_spirv,
            dreamcoast_shader::motion_blur_vs_dxil,
            dreamcoast_shader::motion_blur_fs_dxil,
            dreamcoast_shader::motion_blur_vs_metallib,
            dreamcoast_shader::motion_blur_fs_metallib,
            "motion_blur",
        )?;
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vs,
            fragment_bytes: fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[hdr_format],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 32, // see motion_blur_push
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;
        Ok(Self { pipeline })
    }

    /// Record the motion-blur pass: reads `hdr_in` + `velocity`, writes `hdr_out` (the
    /// caller threads it forward). `intensity` scales one frame of on-screen travel;
    /// `max_uv` caps the per-pixel streak so a huge motion vector can't smear the frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr_in: ResourceId,
        hdr_out: ResourceId,
        velocity: ResourceId,
        flip_y: u32,
        sample_count: u32,
        intensity: f32,
        max_uv: f32,
    ) {
        graph.add_pass(
            PassInfo {
                name: "motion_blur",
                colors: vec![(hdr_out, None)],
                depth: None,
                reads: vec![hdr_in, velocity],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(hdr_in);
                let velocity_index = ctx.sampled_index(velocity);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.pipeline);
                cmd.push_constants(&motion_blur_push(
                    hdr_index,
                    velocity_index,
                    flip_y,
                    sample_count,
                    intensity,
                    max_uv,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}

/// Depth-of-field node (stub) — pipeline re-baseline PR-5 post node #17 (`dof.slang`).
/// A passthrough that reserves the slot for the Phase 20 CoC + bokeh implementation.
pub(crate) struct DofSystem {
    pipeline: GraphicsPipeline,
}

impl DofSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        hdr_format: Format,
    ) -> anyhow::Result<Self> {
        let (vs, fs) = load_shader_pair(
            backend,
            dreamcoast_shader::dof_vs_spirv,
            dreamcoast_shader::dof_fs_spirv,
            dreamcoast_shader::dof_vs_dxil,
            dreamcoast_shader::dof_fs_dxil,
            dreamcoast_shader::dof_vs_metallib,
            dreamcoast_shader::dof_fs_metallib,
            "dof",
        )?;
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vs,
            fragment_bytes: fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[hdr_format],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 16, // see dof_push
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;
        Ok(Self { pipeline })
    }

    /// Record the DoF stub: passthrough copy `hdr_in` -> `hdr_out`.
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr_in: ResourceId,
        hdr_out: ResourceId,
        flip_y: u32,
    ) {
        graph.add_pass(
            PassInfo {
                name: "dof",
                colors: vec![(hdr_out, None)],
                depth: None,
                reads: vec![hdr_in],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(hdr_in);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.pipeline);
                cmd.push_constants(&dof_push(hdr_index, flip_y));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}

/// Pack the bloom push block (32 bytes): src_index, add_index, flip_y, pad0 (16),
/// src_texel.xy + threshold + intensity (16). See `bloom.slang`.
fn bloom_push(
    src_index: u32,
    add_index: u32,
    flip_y: u32,
    src_texel: [f32; 2],
    threshold: f32,
    intensity: f32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&src_index.to_le_bytes());
    pc[4..8].copy_from_slice(&add_index.to_le_bytes());
    pc[8..12].copy_from_slice(&flip_y.to_le_bytes());
    pc[16..20].copy_from_slice(&src_texel[0].to_le_bytes());
    pc[20..24].copy_from_slice(&src_texel[1].to_le_bytes());
    pc[24..28].copy_from_slice(&threshold.to_le_bytes());
    pc[28..32].copy_from_slice(&intensity.to_le_bytes());
    pc
}

/// Pack the motion-blur push block (32 bytes): hdr_index, velocity_index, flip_y,
/// sample_count (16), intensity, max_uv, pad, pad (16). See `motion_blur.slang`.
fn motion_blur_push(
    hdr_index: u32,
    velocity_index: u32,
    flip_y: u32,
    sample_count: u32,
    intensity: f32,
    max_uv: f32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&velocity_index.to_le_bytes());
    pc[8..12].copy_from_slice(&flip_y.to_le_bytes());
    pc[12..16].copy_from_slice(&sample_count.to_le_bytes());
    pc[16..20].copy_from_slice(&intensity.to_le_bytes());
    pc[20..24].copy_from_slice(&max_uv.to_le_bytes());
    pc
}

/// Pack the DoF-stub push block (16 bytes): hdr_index, flip_y, pad, pad. See `dof.slang`.
fn dof_push(hdr_index: u32, flip_y: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc
}
