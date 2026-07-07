//! The deferred-PBR backbone — extracted from `run()` as R6 of the render-loop
//! decomposition (see docs/refactor-sandbox.md). `DeferredRenderer` owns the four
//! graphics pipelines (shadow depth, G-buffer fill, deferred lighting, tonemap), the
//! compute post-process pipeline, and the per-frame globals uniform buffer.
//!
//! The render graph's transient targets (G-buffer MRTs, shadow map, HDR) are created
//! in `run()` and shared across the other bundles (the path tracer / GDF replace the
//! tonemap source; cull + particles draw over the backbuffer), so the bundle's
//! `record_*` methods take the resource ids as parameters and add one pass each
//! (`&'a self` tied to the graph's lifetime, like the other bundles). `run()` keeps
//! the per-frame globals assembly and the tonemap-source selection.

use dreamcoast_core::glam::Mat4;
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, Buffer, BufferDesc, BufferUsage, ClearColor, ComputePipeline,
    ComputePipelineDesc, DepthBuffer, DepthCompare, Extent2D, Format, GraphicsPipeline,
    GraphicsPipelineDesc, PrimitiveTopology, Recorder, StorageBuffer, StorageBufferDesc,
    VertexLayout,
};

use crate::app::{load_compute_shader, load_shader_pair};
use crate::push::post_push;
use crate::{
    DEPTH_FORMAT, FRAMES_IN_FLIGHT, GB_ALBEDO_FMT, GB_MATERIAL_FMT, GB_NORMAL_FMT, GB_POSITION_FMT,
    GLOBALS_SLICE, GROUND_ALBEDO, HDR_FORMAT, MAX_VIEWS, NO_TEXTURE, SHADOW_SIZE, SceneObject,
};

/// The render graph's G-buffer targets: four MRTs (+ depth) written by the fill pass
/// and sampled by the lighting pass.
#[derive(Clone, Copy)]
pub(crate) struct GBufferTargets {
    pub(crate) albedo: ResourceId,
    pub(crate) normal: ResourceId,
    pub(crate) material: ResourceId,
    pub(crate) position: ResourceId,
    pub(crate) depth: ResourceId,
}

pub(crate) struct DeferredRenderer {
    shadow_pipeline: GraphicsPipeline,
    /// Depth-only shadow fill for GPU-skinned casters (animation Stage B.2b).
    shadow_skinned_pipeline: GraphicsPipeline,
    gbuffer_pipeline: GraphicsPipeline,
    /// G-buffer fill for GPU-skinned meshes (animation Stage B.2): same as
    /// `gbuffer_pipeline` but with the `vsMainSkinned` vertex-pulling entry point.
    gbuffer_skinned_pipeline: GraphicsPipeline,
    /// G-buffer fill for GPU-morphed meshes (animation Stage C optimization): the
    /// `vsMainMorphed` vertex-pulling entry point of the same shader.
    gbuffer_morphed_pipeline: GraphicsPipeline,
    /// Deferred surface-decal fill (decals A3): the `fsDecal` entry of the same shader with
    /// the `DecalAlbedo` per-RT blend (RT0 albedo blend, rest masked) + depth-test/no-write.
    /// Runs after the opaque G-buffer fill to tint already-shaded surfaces (the dirt_decal fix).
    gbuffer_decal_pipeline: GraphicsPipeline,
    /// Depth-only shadow fill for GPU-morphed casters (so the shadow follows the morph).
    shadow_morphed_pipeline: GraphicsPipeline,
    /// Depth pre-pass fill (pipeline rebaseline PR-1, opt-in `DEPTH_PREPASS=1`): three
    /// depth-only pipelines (static / skinned / morphed) that reuse the *G-buffer* vertex
    /// shaders (`vsMain` / `vsMainSkinned` / `vsMainMorphed`) with the depth-only `fsDepth`
    /// fragment, so the pre-pass clip-space position is computed by the identical instruction
    /// sequence as the base pass — the depths match bit-exactly (EQUAL-test premise).
    prepass_pipeline: GraphicsPipeline,
    prepass_skinned_pipeline: GraphicsPipeline,
    prepass_morphed_pipeline: GraphicsPipeline,
    /// EQUAL-depth-test + depth-write-off variants of the three G-buffer fills, used when the
    /// depth pre-pass is active: the pre-pass has already established depth, so the base pass
    /// only shades fragments whose depth equals it (Early-Z overdraw elimination) and does not
    /// re-write depth. Same shaders / formats as the default `Less` fills — only the depth state
    /// differs — so switching does not change shading (the EQUAL byte-identical goal).
    gbuffer_equal_pipeline: GraphicsPipeline,
    gbuffer_equal_skinned_pipeline: GraphicsPipeline,
    gbuffer_equal_morphed_pipeline: GraphicsPipeline,
    pbr_pipeline: GraphicsPipeline,
    post_pipeline: GraphicsPipeline,
    /// Physical-camera auto-exposure: the histogram + resolve compute pipelines, a 256-bin
    /// luminance histogram buffer, and the persistent 1-element exposure buffer (adapted value
    /// read by the lighting pass when auto on).
    ae_histogram_pipeline: Option<ComputePipeline>,
    ae_resolve_pipeline: Option<ComputePipeline>,
    ae_hist_buf: Option<StorageBuffer>,
    exposure_buf: Option<StorageBuffer>,
    /// Per-frame globals uniform buffer (one `GLOBALS_SLICE` slice per frame-in-flight).
    globals_buffer: Buffer,
}

impl DeferredRenderer {
    pub(crate) fn new(
        device: &rhi::Device,
        backend: BackendKind,
        swapchain_format: Format,
    ) -> anyhow::Result<Self> {
        // G-buffer fill pipeline: mesh -> 4 MRT (+ depth).
        let (gb_vs, gb_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::gbuffer_vs_spirv,
            dreamcoast_shader::gbuffer_fs_spirv,
            dreamcoast_shader::gbuffer_vs_dxil,
            dreamcoast_shader::gbuffer_fs_dxil,
            dreamcoast_shader::gbuffer_vs_metallib,
            dreamcoast_shader::gbuffer_fs_metallib,
            "gbuffer",
        )?;
        let gbuffer_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: gb_vs,
            fragment_bytes: gb_fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[
                GB_ALBEDO_FMT,
                GB_NORMAL_FMT,
                GB_MATERIAL_FMT,
                GB_POSITION_FMT,
            ],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::Mesh,
            blend: BlendMode::Opaque,
            push_constant_size: 208, // +skin u32x4(16) + morph u32x4(16) over the 176-byte core
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // GPU-skinning variant of the G-buffer fill (vertex-pulling skinning VS) —
        // the `vsMainSkinned` entry of the same shader, sharing the G-buffer fragment.
        let (gb_skin_vs, gb_skin_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::gbuffer_skinned_vs_spirv,
            dreamcoast_shader::gbuffer_fs_spirv,
            dreamcoast_shader::gbuffer_skinned_vs_dxil,
            dreamcoast_shader::gbuffer_fs_dxil,
            dreamcoast_shader::gbuffer_skinned_vs_metallib,
            dreamcoast_shader::gbuffer_fs_metallib,
            "gbuffer_skinned",
        )?;
        let gbuffer_skinned_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: gb_skin_vs,
            fragment_bytes: gb_skin_fs,
            vertex_entry: "vsMainSkinned",
            fragment_entry: "fsMain",
            color_formats: &[
                GB_ALBEDO_FMT,
                GB_NORMAL_FMT,
                GB_MATERIAL_FMT,
                GB_POSITION_FMT,
            ],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::Mesh,
            blend: BlendMode::Opaque,
            push_constant_size: 208,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // GPU-morph variant of the G-buffer fill (vertex-pulling morph VS) — the
        // `vsMainMorphed` entry of the same shader, sharing the G-buffer fragment.
        let (gb_morph_vs, gb_morph_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::gbuffer_morphed_vs_spirv,
            dreamcoast_shader::gbuffer_fs_spirv,
            dreamcoast_shader::gbuffer_morphed_vs_dxil,
            dreamcoast_shader::gbuffer_fs_dxil,
            dreamcoast_shader::gbuffer_morphed_vs_metallib,
            dreamcoast_shader::gbuffer_fs_metallib,
            "gbuffer_morphed",
        )?;
        let gbuffer_morphed_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: gb_morph_vs,
            fragment_bytes: gb_morph_fs,
            vertex_entry: "vsMainMorphed",
            fragment_entry: "fsMain",
            color_formats: &[
                GB_ALBEDO_FMT,
                GB_NORMAL_FMT,
                GB_MATERIAL_FMT,
                GB_POSITION_FMT,
            ],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::Mesh,
            blend: BlendMode::Opaque,
            push_constant_size: 208,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // Deferred decal fill (decals A3): the `fsDecal` entry of the same shader (shares
        // `vsMain` + the Mesh vertex layout), with the `DecalAlbedo` per-RT blend so it tints
        // the G-buffer albedo while leaving normal / metallic / roughness / world-pos as the
        // underlying surface's. Depth-tests against the opaque depth (occluded by closer
        // geometry) but does NOT write depth (`depth_write: false`) — it must not perturb the
        // opaque depth that downstream passes read. Same 4 MRT formats + push size as the fill.
        let (gb_decal_vs, gb_decal_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::gbuffer_vs_spirv,
            dreamcoast_shader::gbuffer_decal_fs_spirv,
            dreamcoast_shader::gbuffer_vs_dxil,
            dreamcoast_shader::gbuffer_decal_fs_dxil,
            dreamcoast_shader::gbuffer_vs_metallib,
            dreamcoast_shader::gbuffer_decal_fs_metallib,
            "gbuffer_decal",
        )?;
        let gbuffer_decal_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: gb_decal_vs,
            fragment_bytes: gb_decal_fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsDecal",
            color_formats: &[
                GB_ALBEDO_FMT,
                GB_NORMAL_FMT,
                GB_MATERIAL_FMT,
                GB_POSITION_FMT,
            ],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::Mesh,
            blend: BlendMode::DecalAlbedo,
            push_constant_size: 208,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // Shadow pipeline: depth-only, rasterizes the mesh from the light's POV.
        let (shadow_vs, shadow_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::shadow_vs_spirv,
            dreamcoast_shader::shadow_fs_spirv,
            dreamcoast_shader::shadow_vs_dxil,
            dreamcoast_shader::shadow_fs_dxil,
            dreamcoast_shader::shadow_vs_metallib,
            dreamcoast_shader::shadow_fs_metallib,
            "shadow",
        )?;
        let shadow_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: shadow_vs,
            fragment_bytes: shadow_fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[], // depth-only
            topology: PrimitiveTopology::TriangleList,
            // Pos + uv (normal skipped): depth needs position, the masked alpha-test
            // discard needs uv; the shadow VS reads no normal, so declaring it would
            // trip the "location 1 not consumed" validation warning.
            vertex_layout: VertexLayout::MeshPositionUv,
            blend: BlendMode::Opaque,
            push_constant_size: 112, // light_mvp(64) + tex u32 + cutoff f32 + pad8 + skin u32x4(16) + morph u32x4(16)
            bindless: true,          // push constants + the bindless base-color texture (masked)
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // GPU-skinning variant of the shadow fill (skinning shadow VS).
        let (shadow_skin_vs, shadow_skin_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::shadow_skinned_vs_spirv,
            dreamcoast_shader::shadow_fs_spirv,
            dreamcoast_shader::shadow_skinned_vs_dxil,
            dreamcoast_shader::shadow_fs_dxil,
            dreamcoast_shader::shadow_skinned_vs_metallib,
            dreamcoast_shader::shadow_fs_metallib,
            "shadow_skinned",
        )?;
        let shadow_skinned_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: shadow_skin_vs,
            fragment_bytes: shadow_skin_fs,
            vertex_entry: "vsMainSkinned",
            fragment_entry: "fsMain",
            color_formats: &[], // depth-only
            topology: PrimitiveTopology::TriangleList,
            // Pos + uv (normal skipped) — same as the static shadow VS.
            vertex_layout: VertexLayout::MeshPositionUv,
            blend: BlendMode::Opaque,
            push_constant_size: 112,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // GPU-morph variant of the shadow fill (morph shadow VS).
        let (shadow_morph_vs, shadow_morph_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::shadow_morphed_vs_spirv,
            dreamcoast_shader::shadow_fs_spirv,
            dreamcoast_shader::shadow_morphed_vs_dxil,
            dreamcoast_shader::shadow_fs_dxil,
            dreamcoast_shader::shadow_morphed_vs_metallib,
            dreamcoast_shader::shadow_fs_metallib,
            "shadow_morphed",
        )?;
        let shadow_morphed_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: shadow_morph_vs,
            fragment_bytes: shadow_morph_fs,
            vertex_entry: "vsMainMorphed",
            fragment_entry: "fsMain",
            color_formats: &[], // depth-only
            topology: PrimitiveTopology::TriangleList,
            // Pos + uv (normal skipped) — same as the static shadow VS.
            vertex_layout: VertexLayout::MeshPositionUv,
            blend: BlendMode::Opaque,
            push_constant_size: 112,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;

        // Depth pre-pass fill (pipeline rebaseline PR-1). Depth-only pipelines that reuse the
        // G-buffer *vertex* shaders unchanged (so the clip position is bit-identical to the base
        // pass) with the `fsDepth` fragment (depth + the same alpha-test discard). One per deform
        // path (static / skinned / morphed). `Mesh` layout + 208-byte push match the G-buffer
        // fills (the VS reads pos+normal+uv and the full push). `Less` + write-on establishes the
        // scene depth. Built unconditionally (cheap pipeline objects); only recorded when the
        // `DEPTH_PREPASS` seam is on, so the default path is byte-identical.
        type ShaderFn = fn() -> Option<&'static [u8]>;
        let make_prepass = |vs_spirv: ShaderFn,
                            vs_dxil: ShaderFn,
                            vs_metallib: ShaderFn,
                            entry: &'static str,
                            label: &'static str|
         -> anyhow::Result<GraphicsPipeline> {
            let (vs, fs) = load_shader_pair(
                backend,
                vs_spirv,
                dreamcoast_shader::gbuffer_depth_fs_spirv,
                vs_dxil,
                dreamcoast_shader::gbuffer_depth_fs_dxil,
                vs_metallib,
                dreamcoast_shader::gbuffer_depth_fs_metallib,
                label,
            )?;
            Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: vs,
                fragment_bytes: fs,
                vertex_entry: entry,
                fragment_entry: "fsDepth",
                color_formats: &[], // depth-only
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::Mesh,
                blend: BlendMode::Opaque,
                push_constant_size: 208,
                bindless: true,
                uniform_buffer: false,
                depth_test: true,
                depth_write: true,
                depth_compare: DepthCompare::Less,
                depth_format: Some(DEPTH_FORMAT),
            })?)
        };
        let prepass_pipeline = make_prepass(
            dreamcoast_shader::gbuffer_vs_spirv,
            dreamcoast_shader::gbuffer_vs_dxil,
            dreamcoast_shader::gbuffer_vs_metallib,
            "vsMain",
            "prepass",
        )?;
        let prepass_skinned_pipeline = make_prepass(
            dreamcoast_shader::gbuffer_skinned_vs_spirv,
            dreamcoast_shader::gbuffer_skinned_vs_dxil,
            dreamcoast_shader::gbuffer_skinned_vs_metallib,
            "vsMainSkinned",
            "prepass_skinned",
        )?;
        let prepass_morphed_pipeline = make_prepass(
            dreamcoast_shader::gbuffer_morphed_vs_spirv,
            dreamcoast_shader::gbuffer_morphed_vs_dxil,
            dreamcoast_shader::gbuffer_morphed_vs_metallib,
            "vsMainMorphed",
            "prepass_morphed",
        )?;

        // EQUAL-depth-test + depth-write-off variants of the three G-buffer fills (used when the
        // pre-pass is active). Identical shaders / MRT formats / push size to the default `Less`
        // fills — only the depth state differs — so the shading is unchanged; the EQUAL test just
        // rejects the overdrawn fragments the pre-pass already resolved. `depth_write: false`
        // keeps the pre-pass depth (downstream screen-space passes sample it).
        let make_gbuffer_equal = |vs_spirv: ShaderFn,
                                  vs_dxil: ShaderFn,
                                  vs_metallib: ShaderFn,
                                  entry: &'static str,
                                  label: &'static str|
         -> anyhow::Result<GraphicsPipeline> {
            let (vs, fs) = load_shader_pair(
                backend,
                vs_spirv,
                dreamcoast_shader::gbuffer_fs_spirv,
                vs_dxil,
                dreamcoast_shader::gbuffer_fs_dxil,
                vs_metallib,
                dreamcoast_shader::gbuffer_fs_metallib,
                label,
            )?;
            Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: vs,
                fragment_bytes: fs,
                vertex_entry: entry,
                fragment_entry: "fsMain",
                color_formats: &[
                    GB_ALBEDO_FMT,
                    GB_NORMAL_FMT,
                    GB_MATERIAL_FMT,
                    GB_POSITION_FMT,
                ],
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::Mesh,
                blend: BlendMode::Opaque,
                push_constant_size: 208,
                bindless: true,
                uniform_buffer: false,
                depth_test: true,
                depth_write: false,
                depth_compare: DepthCompare::Equal,
                depth_format: Some(DEPTH_FORMAT),
            })?)
        };
        let gbuffer_equal_pipeline = make_gbuffer_equal(
            dreamcoast_shader::gbuffer_vs_spirv,
            dreamcoast_shader::gbuffer_vs_dxil,
            dreamcoast_shader::gbuffer_vs_metallib,
            "vsMain",
            "gbuffer_equal",
        )?;
        let gbuffer_equal_skinned_pipeline = make_gbuffer_equal(
            dreamcoast_shader::gbuffer_skinned_vs_spirv,
            dreamcoast_shader::gbuffer_skinned_vs_dxil,
            dreamcoast_shader::gbuffer_skinned_vs_metallib,
            "vsMainSkinned",
            "gbuffer_equal_skinned",
        )?;
        let gbuffer_equal_morphed_pipeline = make_gbuffer_equal(
            dreamcoast_shader::gbuffer_morphed_vs_spirv,
            dreamcoast_shader::gbuffer_morphed_vs_dxil,
            dreamcoast_shader::gbuffer_morphed_vs_metallib,
            "vsMainMorphed",
            "gbuffer_equal_morphed",
        )?;

        // Deferred lighting pipeline: full-screen, reads G-buffer + globals -> HDR.
        let (pbr_vs, pbr_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::pbr_vs_spirv,
            dreamcoast_shader::pbr_fs_spirv,
            dreamcoast_shader::pbr_vs_dxil,
            dreamcoast_shader::pbr_fs_dxil,
            dreamcoast_shader::pbr_vs_metallib,
            dreamcoast_shader::pbr_fs_metallib,
            "pbr",
        )?;
        let pbr_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: pbr_vs,
            fragment_bytes: pbr_fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[HDR_FORMAT],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 76, // 60-byte core + clustered light bufs (grid, index, light, count)
            bindless: true,
            uniform_buffer: true,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;

        // Tonemap pipeline: HDR -> backbuffer (encodes sRGB in-shader; UNORM target).
        let (post_vs, post_fs) = load_shader_pair(
            backend,
            dreamcoast_shader::post_vs_spirv,
            dreamcoast_shader::post_fs_spirv,
            dreamcoast_shader::post_vs_dxil,
            dreamcoast_shader::post_fs_dxil,
            dreamcoast_shader::post_vs_metallib,
            dreamcoast_shader::post_fs_metallib,
            "post",
        )?;
        let post_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: post_vs,
            fragment_bytes: post_fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[swapchain_format],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            // hdr_index + mode + flip_y + exposure + sharpen + inv_w/h + PR-5 bloom
            // composite slot + ASC-CDL grading (see push::post_push).
            push_constant_size: 96,
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;

        // Physical-camera auto-exposure: a luminance-histogram metering (csHistogram → csResolve)
        // plus a persistent 256-bin histogram buffer (seeded 0) and an 8-byte [exposure,
        // scene_luminance] buffer (seeded with a sensible content exposure so the first frame,
        // before metering runs, isn't black; it adapts within a frame or two). Compute-gated.
        let ae = (|| -> anyhow::Result<_> {
            let hist_cs = load_compute_shader(
                backend,
                dreamcoast_shader::auto_exposure_histogram_cs_spirv,
                dreamcoast_shader::auto_exposure_histogram_cs_dxil,
                dreamcoast_shader::auto_exposure_histogram_cs_metallib,
                "auto_exposure_histogram",
            )?;
            let resolve_cs = load_compute_shader(
                backend,
                dreamcoast_shader::auto_exposure_resolve_cs_spirv,
                dreamcoast_shader::auto_exposure_resolve_cs_dxil,
                dreamcoast_shader::auto_exposure_resolve_cs_metallib,
                "auto_exposure_resolve",
            )?;
            let hist_pipe = device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: hist_cs,
                compute_entry: "csHistogram",
                push_constant_size: 16, // hdr + hist_buf + width + height
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [16, 16, 1],
            })?;
            let resolve_pipe = device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: resolve_cs,
                compute_entry: "csResolve",
                push_constant_size: 48, // hist + expo + num_pixels + key + adapt + min + max + lo + hi
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [256, 1, 1],
            })?;
            let hist = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: 256 * 4,
                    stride: 4,
                    indirect: false,
                },
                &vec![0u8; 256 * 4],
            )?;
            let seed = [1.0e-4f32, 0.0f32];
            let bytes: Vec<u8> = seed.iter().flat_map(|f| f.to_le_bytes()).collect();
            let expo = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: bytes.len() as u64,
                    stride: 4,
                    indirect: false,
                },
                &bytes,
            )?;
            Ok((hist_pipe, resolve_pipe, hist, expo))
        })();
        let (ae_histogram_pipeline, ae_resolve_pipeline, ae_hist_buf, exposure_buf) = match ae {
            Ok((h, r, hb, e)) => (Some(h), Some(r), Some(hb), Some(e)),
            Err(_) => (None, None, None, None),
        };

        // Per-frame globals uniform buffer. PR-9 view family: `FRAMES_IN_FLIGHT * MAX_VIEWS`
        // slices so each view in a frame gets its own camera-matrix slice (offset
        // `(fif * MAX_VIEWS + view) * GLOBALS_SLICE`). The default single-view path only touches
        // slice 0 of each frame, so the extra tail is unused (and zero-cost) when off.
        let globals_buffer = device.create_buffer(&BufferDesc {
            size: GLOBALS_SLICE * FRAMES_IN_FLIGHT as u64 * MAX_VIEWS,
            usage: BufferUsage::Uniform,
        })?;
        device.set_globals_buffer(&globals_buffer, GLOBALS_SLICE);

        Ok(Self {
            shadow_pipeline,
            shadow_skinned_pipeline,
            gbuffer_pipeline,
            gbuffer_skinned_pipeline,
            gbuffer_morphed_pipeline,
            gbuffer_decal_pipeline,
            shadow_morphed_pipeline,
            prepass_pipeline,
            prepass_skinned_pipeline,
            prepass_morphed_pipeline,
            gbuffer_equal_pipeline,
            gbuffer_equal_skinned_pipeline,
            gbuffer_equal_morphed_pipeline,
            pbr_pipeline,
            post_pipeline,
            ae_histogram_pipeline,
            ae_resolve_pipeline,
            ae_hist_buf,
            exposure_buf,
            globals_buffer,
        })
    }

    /// The bindless storage-buffer index of the adapted-exposure buffer (lighting reads it when
    /// auto-exposure is on). `None` if the metering pipeline failed to build.
    pub(crate) fn exposure_buf_index(&self) -> Option<u32> {
        self.exposure_buf.as_ref().map(|b| b.storage_index())
    }

    /// Auto-exposure metering: bin the lit `hdr` into a luminance histogram (pass 1), then resolve
    /// the trimmed-percentile geometric-mean luminance into the time-adapted exposure for next
    /// frame's lighting (pass 2, which also clears the histogram). Runs after the lighting pass
    /// (the `hdr` read orders it). `key` = target grey, `adapt` = this frame's EMA factor, and
    /// `[min_exp,max_exp]` clamp the exposure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_auto_exposure<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        cw: u32,
        ch: u32,
        key: f32,
        adapt: f32,
        min_exp: f32,
        max_exp: f32,
    ) {
        let (Some(hist_pipe), Some(resolve_pipe), Some(hist_buf), Some(expo_buf)) = (
            &self.ae_histogram_pipeline,
            &self.ae_resolve_pipeline,
            &self.ae_hist_buf,
            &self.exposure_buf,
        ) else {
            return;
        };
        let hist_idx = hist_buf.storage_index();
        let expo_idx = expo_buf.storage_index();
        let hist_ext = graph.import_external("ae_histogram");
        let expo_ext = graph.import_external("exposure_buf");
        // Pass 1: per-pixel histogram (reads hdr, writes the histogram buffer).
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ae_histogram",
                storage_writes: vec![hist_ext],
                reads: vec![hdr],
            },
            move |ctx| {
                let hdr_idx = ctx.sampled_index(hdr);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(hist_pipe);
                cmd.push_constants_compute(&ae_hist_push(hdr_idx, hist_idx, cw, ch));
                cmd.dispatch(cw.div_ceil(16), ch.div_ceil(16), 1);
                Ok(())
            },
        );
        // Pass 2: resolve the exposure + clear the histogram (single 256-thread group).
        let num_pixels = cw * ch;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ae_resolve",
                storage_writes: vec![expo_ext, hist_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(resolve_pipe);
                // Full geometric mean over the histogram (lo=0, hi=1): the trimmed-percentile mean
                // shifts on a few sub-ULP-different bright pixels and amplifies into a cross-backend
                // exposure drift; the full mean barely moves (one pixel changing bin shifts it by
                // ~1/N), so DX≡VK holds. Robust to outliers via the log-average, not a hard cutoff.
                cmd.push_constants_compute(&ae_resolve_push(
                    hist_idx, expo_idx, num_pixels, key, adapt, min_exp, max_exp, 0.0, 1.0,
                ));
                cmd.dispatch(1, 1, 1);
                Ok(())
            },
        );
    }

    /// Write this frame's globals slice (packed by `run()` via `globals_bytes`) at
    /// `offset` (= `frame * GLOBALS_SLICE`); the offset is reused by `record_lighting`.
    pub(crate) fn write_globals(&self, offset: u64, bytes: &[u8]) -> anyhow::Result<()> {
        self.globals_buffer.write_at(offset, bytes)?;
        Ok(())
    }

    /// The per-frame globals uniform buffer, so a compute pass (Stage C7 SSR) can bind it
    /// via `cmd.set_globals` to read structured camera data (the reprojection matrices).
    pub(crate) fn globals_buffer(&self) -> &Buffer {
        &self.globals_buffer
    }

    /// Shadow pass: rasterize the shadow-casting scene objects from the light's POV
    /// into the depth-only shadow map (the lighting pass samples it). The ground is a
    /// flat receiver, not a caster.
    pub(crate) fn record_shadow<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        shadow_map: ResourceId,
        scene: &'a [SceneObject],
        light_vp: Mat4,
    ) {
        graph.add_pass(
            PassInfo {
                name: "shadow",
                colors: vec![],
                depth: Some(shadow_map),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.shadow_pipeline);
                for obj in scene {
                    if !obj.casts_shadow {
                        continue;
                    }
                    let lmvp = (light_vp * obj.transform).to_cols_array();
                    // Deformed casters deform on the GPU (their shadow matches the lit mesh)
                    // via the skinned / morph shadow pipeline; static casters keep the bound
                    // one. A drawable is at most one of skinned / morphed.
                    if obj.skin.is_some() {
                        cmd.bind_graphics_pipeline(&self.shadow_skinned_pipeline);
                    } else if obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(&self.shadow_morphed_pipeline);
                    }
                    // Masked casters carry their base-color texture + cutoff so the depth pass
                    // discards the same texels the lit pass does; opaque casters pass cutoff 0
                    // (base-color index unused) -> depth-only, identical to the pre-mask pass.
                    cmd.push_constants(&shadow_push(
                        lmvp,
                        obj.tex[0],
                        obj.alpha_cutoff,
                        obj.skin.unwrap_or([0; 4]),
                        obj.morph.unwrap_or([0; 4]),
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                    if obj.skin.is_some() || obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(&self.shadow_pipeline); // restore
                    }
                }
                Ok(())
            },
        );
    }

    /// Cached-shadow (docs/shadow-cache-cold-start.md S1): rasterize the legacy directional
    /// shadow map DEPTH-ONLY into an **app-owned persistent** `DepthBuffer`, OUTSIDE the render
    /// graph, on a raw `Recorder` — exactly the pattern `ibl.rs` uses to capture the env cube into
    /// app-owned targets. The legacy map is camera-independent (`light_view_proj` reads only the
    /// sun + scene bounds), so its depth persists across frames and a settled frame can SKIP this
    /// call and re-sample last frame's map by bindless index (the in-graph transient pool does NOT
    /// persist depth — see ledger A4). The caster loop is identical to `record_shadow` (static /
    /// skinned / morph pipeline select + masked alpha-test discard) so the cached depth is
    /// bit-identical to the in-graph raster. The depth-only begin clears + stores the depth; the
    /// caller transitions it to sampled afterwards.
    pub(crate) fn record_shadow_direct(
        &self,
        cmd: &dyn Recorder,
        depth: &DepthBuffer,
        scene: &[SceneObject],
        light_vp: Mat4,
    ) {
        cmd.depth_to_render_target(depth);
        cmd.begin_rendering_depth_only(depth);
        // Out-of-graph: no graph pass to set the viewport, so pin it to the shadow map extent.
        cmd.set_viewport_scissor_extent(Extent2D::new(SHADOW_SIZE, SHADOW_SIZE));
        cmd.bind_graphics_pipeline(&self.shadow_pipeline);
        for obj in scene {
            if !obj.casts_shadow {
                continue;
            }
            let lmvp = (light_vp * obj.transform).to_cols_array();
            if obj.skin.is_some() {
                cmd.bind_graphics_pipeline(&self.shadow_skinned_pipeline);
            } else if obj.morph.is_some() {
                cmd.bind_graphics_pipeline(&self.shadow_morphed_pipeline);
            }
            cmd.push_constants(&shadow_push(
                lmvp,
                obj.tex[0],
                obj.alpha_cutoff,
                obj.skin.unwrap_or([0; 4]),
                obj.morph.unwrap_or([0; 4]),
            ));
            cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
            cmd.bind_index_buffer(&obj.mesh.ibuf, true);
            cmd.draw_indexed(obj.mesh.index_count, 0, 0);
            if obj.skin.is_some() || obj.morph.is_some() {
                cmd.bind_graphics_pipeline(&self.shadow_pipeline); // restore
            }
        }
        cmd.end_rendering();
    }

    /// Depth pre-pass (pipeline rebaseline PR-1, opt-in `DEPTH_PREPASS=1`): rasterize the same
    /// opaque scene the G-buffer fill draws (skipping decals) plus the ground, DEPTH-ONLY, into
    /// `targets.depth` **before** the G-buffer pass. This establishes the scene depth so the base
    /// pass can run EQUAL-test + depth-write-off and shade every pixel exactly once (Early-Z
    /// overdraw elimination), and gives the screen-space passes (AO/GI/SSR/reflect) a completed
    /// depth to sample without an ordering race. The pipelines reuse the *G-buffer vertex shaders
    /// unchanged*, so the clip position — and therefore the depth — is bit-identical to the base
    /// pass. The pass clears the depth (it is the first depth writer this frame); the later
    /// G-buffer pass then LOADS it (the render graph's per-depth first-writer clear/load rule).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_prepass<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        depth: ResourceId,
        scene: &'a [SceneObject],
        ground_vbuf: &'a Buffer,
        ground_ibuf: &'a Buffer,
        ground_count: u32,
        view_proj: Mat4,
        override_material: bool,
        metallic_override: f32,
        roughness_override: f32,
        mip_bias: f32,
    ) {
        graph.add_pass(
            PassInfo {
                name: "prepass",
                colors: vec![], // depth-only
                depth: Some(depth),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.prepass_pipeline);
                for obj in scene {
                    // Skip decals exactly like the G-buffer pass: decals do not write opaque
                    // depth (they tint the albedo in the later decal pass). Keeps the pre-pass
                    // depth identical to what the base pass expects.
                    if obj.kind == dreamcoast_asset::MaterialKind::Decal {
                        continue;
                    }
                    let obj_mvp = (view_proj * obj.transform).to_cols_array();
                    // The alpha-test discard needs the base-color texture + cutoff; the mvp +
                    // model + skin/morph indices must match the base pass exactly (same push),
                    // so the shared VS produces the same clip position. Material factors that do
                    // not affect position (metallic/roughness) are the base pass's; here they are
                    // fed the override too so the push is identical to the G-buffer draw.
                    let (m, rgh, mr_tex) = if override_material {
                        (metallic_override, roughness_override, NO_TEXTURE)
                    } else {
                        (obj.metallic, obj.roughness, obj.tex[1])
                    };
                    if obj.skin.is_some() {
                        cmd.bind_graphics_pipeline(&self.prepass_skinned_pipeline);
                    } else if obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(&self.prepass_morphed_pipeline);
                    }
                    cmd.push_constants(&gbuffer_push(
                        obj_mvp,
                        obj.base_color,
                        m,
                        rgh,
                        mip_bias,
                        obj.alpha_cutoff,
                        [obj.tex[0], mr_tex, obj.tex[2], obj.tex[3]],
                        obj.transform.to_cols_array(),
                        obj.skin.unwrap_or([0; 4]),
                        obj.morph.unwrap_or([0; 4]),
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                    if obj.skin.is_some() || obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(&self.prepass_pipeline); // restore
                    }
                }
                // Ground plane — identical push to the G-buffer ground draw so its depth matches.
                cmd.push_constants(&gbuffer_push(
                    view_proj.to_cols_array(),
                    [GROUND_ALBEDO[0], GROUND_ALBEDO[1], GROUND_ALBEDO[2], 1.0],
                    0.0,
                    0.9,
                    0.0,
                    0.0,
                    [NO_TEXTURE; 4],
                    Mat4::IDENTITY.to_cols_array(),
                    [0; 4],
                    [0; 4],
                ));
                cmd.bind_vertex_buffer(ground_vbuf, 32);
                cmd.bind_index_buffer(ground_ibuf, true);
                cmd.draw_indexed(ground_count, 0, 0);
                Ok(())
            },
        );
    }

    /// CSM / shadow-atlas fill (PR-7): rasterize the shadow casters once per cascade slot into
    /// its atlas tile. One depth pass clears the whole atlas, then each slot restricts the
    /// viewport + scissor to its tile (`set_viewport_scissor_rect`) and draws the casters with
    /// that cascade's view-projection. Slots share the same caster loop as `record_shadow` (the
    /// static / skinned / morph pipeline select + masked alpha-test discard), so a caster's
    /// cascade shadow matches its lit-mesh cutout exactly. The atlas texture is the same
    /// resource the lighting pass samples, so the sampling side needs no per-slot wiring — only
    /// the cascade view-projection array in globals. Future spot / point slots reuse this loop.
    pub(crate) fn record_shadow_atlas<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        shadow_atlas: ResourceId,
        scene: &'a [SceneObject],
        slots: &'a [crate::csm::ShadowSlot],
    ) {
        graph.add_pass(
            PassInfo {
                name: "shadow_atlas",
                colors: vec![],
                depth: Some(shadow_atlas),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                for slot in slots {
                    // Only cascade slots carry the directional caster set this PR; spot /
                    // point slots get their own caster loops when those light types land
                    // (the slot table + per-slot view-projections already accommodate them).
                    if slot.kind != crate::csm::SlotKind::Cascade {
                        continue;
                    }
                    // Restrict rasterization to this cascade's atlas tile. The depth clear
                    // covered the whole atlas (first depth writer); the scissor keeps each
                    // cascade's draws inside its own tile so they don't bleed across slots.
                    cmd.set_viewport_scissor_rect(slot.rect);
                    cmd.bind_graphics_pipeline(&self.shadow_pipeline);
                    for obj in scene {
                        if !obj.casts_shadow {
                            continue;
                        }
                        let lmvp = (slot.view_proj * obj.transform).to_cols_array();
                        if obj.skin.is_some() {
                            cmd.bind_graphics_pipeline(&self.shadow_skinned_pipeline);
                        } else if obj.morph.is_some() {
                            cmd.bind_graphics_pipeline(&self.shadow_morphed_pipeline);
                        }
                        cmd.push_constants(&shadow_push(
                            lmvp,
                            obj.tex[0],
                            obj.alpha_cutoff,
                            obj.skin.unwrap_or([0; 4]),
                            obj.morph.unwrap_or([0; 4]),
                        ));
                        cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                        cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                        cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                        if obj.skin.is_some() || obj.morph.is_some() {
                            cmd.bind_graphics_pipeline(&self.shadow_pipeline); // restore
                        }
                    }
                }
                Ok(())
            },
        );
    }

    /// G-buffer fill: rasterize each scene object (with its PBR material) plus the
    /// ground into the four MRTs (+ depth). The UI material override replaces
    /// metallic/roughness and drops the m/r texture so the factors apply directly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gbuffer<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        targets: GBufferTargets,
        scene: &'a [SceneObject],
        ground_vbuf: &'a Buffer,
        ground_ibuf: &'a Buffer,
        ground_count: u32,
        view_proj: Mat4,
        ambient: f32,
        override_material: bool,
        metallic_override: f32,
        roughness_override: f32,
        mip_bias: f32,
        // Depth pre-pass active (`DEPTH_PREPASS=1`): the pre-pass already wrote depth, so use the
        // EQUAL-test + depth-write-off G-buffer pipelines (Early-Z overdraw elimination). `false`
        // = the default `Less` + write-on fills (byte-identical to the pre-pass-less path).
        prepass: bool,
        // Draw the matte ground plane into the G-buffer. `true` on every normal path (gallery
        // byte-identical); `false` only for the P14_VGEO parity reference (a groundless single
        // model, matching the vgeo producer which draws no ground).
        draw_ground: bool,
    ) {
        // Select the base-pass pipeline set: EQUAL-test (pre-pass on) vs the default `Less` fills.
        let (pipe_static, pipe_skinned, pipe_morphed) = if prepass {
            (
                &self.gbuffer_equal_pipeline,
                &self.gbuffer_equal_skinned_pipeline,
                &self.gbuffer_equal_morphed_pipeline,
            )
        } else {
            (
                &self.gbuffer_pipeline,
                &self.gbuffer_skinned_pipeline,
                &self.gbuffer_morphed_pipeline,
            )
        };
        let sky = ClearColor {
            r: ambient,
            g: ambient,
            b: ambient,
            a: 1.0,
        };
        let zero = ClearColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        graph.add_pass(
            PassInfo {
                name: "gbuffer",
                colors: vec![
                    (targets.albedo, Some(sky)),
                    (targets.normal, Some(ClearColor::BLACK)),
                    (targets.material, Some(ClearColor::BLACK)),
                    (targets.position, Some(zero)), // alpha 0 marks "no geometry"
                ],
                depth: Some(targets.depth),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(pipe_static);
                for obj in scene {
                    // Decals are not opaque surfaces: they tint the G-buffer albedo in the
                    // later `record_decals` pass instead of writing their own (diffuse-less)
                    // material here. Skipping them is what stops a decal from overwriting the
                    // surface as opaque (the dirt_decal fix). Non-decal scenes have none →
                    // this loop is byte-identical for them.
                    if obj.kind == dreamcoast_asset::MaterialKind::Decal {
                        continue;
                    }
                    let obj_mvp = (view_proj * obj.transform).to_cols_array();
                    let (m, rgh, mr_tex) = if override_material {
                        (metallic_override, roughness_override, NO_TEXTURE)
                    } else {
                        (obj.metallic, obj.roughness, obj.tex[1])
                    };
                    // GPU-deformed objects use a vertex-pulling pipeline (skin / morph
                    // indices in the push); the bind-pose vertex buffer is the same. Static
                    // objects keep the already-bound pipeline (gallery byte-identical). A
                    // drawable is at most one of skinned / morphed.
                    if obj.skin.is_some() {
                        cmd.bind_graphics_pipeline(pipe_skinned);
                    } else if obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(pipe_morphed);
                    }
                    cmd.push_constants(&gbuffer_push(
                        obj_mvp,
                        obj.base_color,
                        m,
                        rgh,
                        mip_bias,
                        obj.alpha_cutoff,
                        [obj.tex[0], mr_tex, obj.tex[2], obj.tex[3]],
                        obj.transform.to_cols_array(),
                        obj.skin.unwrap_or([0; 4]),
                        obj.morph.unwrap_or([0; 4]),
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                    if obj.skin.is_some() || obj.morph.is_some() {
                        cmd.bind_graphics_pipeline(pipe_static); // restore
                    }
                }
                // Ground plane (plain matte material, no textures). Albedo from the shared
                // GROUND_ALBEDO single source (also fed to the GI / reflection re-light passes).
                if draw_ground {
                    cmd.push_constants(&gbuffer_push(
                        view_proj.to_cols_array(),
                        [GROUND_ALBEDO[0], GROUND_ALBEDO[1], GROUND_ALBEDO[2], 1.0],
                        0.0,
                        0.9,
                        0.0, // ground is untextured -> bias irrelevant
                        0.0, // opaque -> no alpha test
                        [NO_TEXTURE; 4],
                        Mat4::IDENTITY.to_cols_array(),
                        [0; 4], // not skinned
                        [0; 4], // not morphed
                    ));
                    cmd.bind_vertex_buffer(ground_vbuf, 32);
                    cmd.bind_index_buffer(ground_ibuf, true);
                    cmd.draw_indexed(ground_count, 0, 0);
                }
                Ok(())
            },
        );
    }

    /// Deferred surface-decal pass (decals A3): runs after `record_gbuffer`, before lighting.
    /// Each `kind == Decal` drawable is rasterized with the `DecalAlbedo` blend so it tints the
    /// G-buffer albedo it sits on (RT0 RGB alpha-blend) while RT1/RT2/RT3 are write-masked off —
    /// the underlying surface keeps its own normal / metallic / roughness / world-pos, so the
    /// lighting that runs next shades the decal with the *surface's* inputs (the dirt_decal fix:
    /// a diffuse-less BLEND decal can no longer overwrite stone as opaque black metal). The
    /// G-buffer targets are LOADED (no clear) to preserve the opaque fill; depth is bound for
    /// testing only (no write). Returns immediately when the scene has no decals, so non-decal
    /// scenes get no extra pass (byte-identical).
    pub(crate) fn record_decals<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        targets: GBufferTargets,
        scene: &'a [SceneObject],
        view_proj: Mat4,
        mip_bias: f32,
    ) {
        if !scene
            .iter()
            .any(|o| o.kind == dreamcoast_asset::MaterialKind::Decal)
        {
            return;
        }
        graph.add_pass(
            PassInfo {
                name: "decals",
                // Load (no clear) every G-buffer target: the decal blends into the albedo and
                // leaves the rest untouched via the pipeline's per-RT write mask.
                colors: vec![
                    (targets.albedo, None),
                    (targets.normal, None),
                    (targets.material, None),
                    (targets.position, None),
                ],
                depth: Some(targets.depth),
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.gbuffer_decal_pipeline);
                for obj in scene {
                    if obj.kind != dreamcoast_asset::MaterialKind::Decal {
                        continue;
                    }
                    let obj_mvp = (view_proj * obj.transform).to_cols_array();
                    cmd.push_constants(&gbuffer_push(
                        obj_mvp,
                        obj.base_color,
                        obj.metallic,
                        obj.roughness,
                        mip_bias,
                        obj.alpha_cutoff,
                        obj.tex,
                        obj.transform.to_cols_array(),
                        [0; 4], // decals are static geometry (not skinned)
                        [0; 4], // not morphed
                    ));
                    cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                    cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                    cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                }
                Ok(())
            },
        );
    }

    /// Deferred lighting: full-screen pass reading the G-buffer + shadow map + globals
    /// (Cook-Torrance BRDF + IBL) into the HDR target. `gdf_ao` is the Stage-C2 GDF
    /// ambient-occlusion image (multiplied into the ambient term) and `gdf_gi` the
    /// Stage-C3 indirect-irradiance image (added to the ambient); pass `None` for either
    /// to leave that term off (= the pre-C2/C3 behavior). `reflect` is the Stage-C7c hybrid
    /// SW-RT reflection image — when present it replaces the IBL prefilter-cube specular
    /// in `pbr.slang`; `None` keeps the legacy captured-cube IBL specular.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lighting<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        gbuf: GBufferTargets,
        // Shadow map source. `shadow_map` = the in-graph transient depth (CSM atlas / gallery /
        // cache-disabled legacy). `shadow_override` = an app-owned persistent depth's bindless
        // sampled index (the cached-shadow path, docs/shadow-cache-cold-start.md S1) — when set,
        // the graph resource is not read; the lighting shader samples the persistent depth directly.
        shadow_map: Option<ResourceId>,
        shadow_override: Option<u32>,
        gdf_ao: Option<ResourceId>,
        ssao: Option<ResourceId>,
        gdf_gi: Option<ResourceId>,
        reflect: Option<ResourceId>,
        skyvis: Option<ResourceId>,
        skyvis_tint: f32,
        skyvis_min_occ: f32,
        globals_offset: u64,
        flip_y: u32,
        two_sided: bool,
        exposure_buf_index: u32,
        // Clustered light culling (PR-6). `Some((grid_ext, index_ext, grid_idx, index_idx,
        // light_idx, light_count))` binds the froxel light list built by `ClusterSystem`; the
        // *_ext ids order the lighting pass after the cluster-build compute pass. `None` = the
        // brute-force globals.point_pos[] path (default, gallery byte-identical).
        cluster: Option<(ResourceId, ResourceId, u32, u32, u32, u32)>,
        // Opt-in albedo-tinted multi-bounce AO on the diffuse ambient (reference AOMultiBounce).
        // Default false => scalar AO, byte-identical anchor.
        ao_multibounce: bool,
    ) {
        let mut reads = vec![gbuf.albedo, gbuf.normal, gbuf.material, gbuf.position];
        if let Some(sm) = shadow_map {
            reads.push(sm);
        }
        if let Some((grid_ext, index_ext, ..)) = cluster {
            reads.push(grid_ext);
            reads.push(index_ext);
        }
        if let Some(ao) = gdf_ao {
            reads.push(ao);
        }
        if let Some(s) = ssao {
            reads.push(s);
        }
        if let Some(gi) = gdf_gi {
            reads.push(gi);
        }
        if let Some(r) = reflect {
            reads.push(r);
        }
        if let Some(sv) = skyvis {
            reads.push(sv);
        }
        graph.add_pass(
            PassInfo {
                name: "lighting",
                colors: vec![(hdr, Some(ClearColor::BLACK))],
                depth: None,
                reads,
            },
            move |ctx| {
                let indices = [
                    ctx.sampled_index(gbuf.albedo),
                    ctx.sampled_index(gbuf.normal),
                    ctx.sampled_index(gbuf.material),
                    ctx.sampled_index(gbuf.position),
                ];
                // Cached-shadow path: sample the app-owned persistent depth by its bindless index;
                // otherwise the in-graph transient depth's per-frame bindless slot.
                let shadow_index = match shadow_override {
                    Some(idx) => idx,
                    None => ctx.sampled_index(shadow_map.expect("shadow_map or shadow_override")),
                };
                let ao_index = gdf_ao.map(|ao| ctx.sampled_index(ao)).unwrap_or(u32::MAX);
                let ssao_index = ssao.map(|s| ctx.sampled_index(s)).unwrap_or(u32::MAX);
                let gi_index = gdf_gi.map(|gi| ctx.sampled_index(gi)).unwrap_or(u32::MAX);
                let reflect_index = reflect.map(|r| ctx.sampled_index(r)).unwrap_or(u32::MAX);
                let skyvis_index = skyvis.map(|s| ctx.sampled_index(s)).unwrap_or(u32::MAX);
                // Clustered light bufs: their storage indices are stable (persistent buffers),
                // so pass them straight through; `NO_TEXTURE` grid buf disables the path.
                let (cl_grid, cl_index, cl_light, cl_count) = match cluster {
                    Some((_, _, g, i, l, c)) => (g, i, l, c),
                    None => (u32::MAX, u32::MAX, u32::MAX, 0),
                };
                let cmd = ctx.cmd();
                cmd.set_globals(&self.globals_buffer, globals_offset);
                cmd.bind_graphics_pipeline(&self.pbr_pipeline);
                cmd.push_constants(&pbr_push(
                    indices,
                    flip_y,
                    shadow_index,
                    ao_index,
                    ssao_index,
                    gi_index,
                    reflect_index,
                    two_sided as u32,
                    exposure_buf_index,
                    skyvis_index,
                    skyvis_tint,
                    skyvis_min_occ,
                    [cl_grid, cl_index, cl_light, cl_count],
                    ao_multibounce as u32,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }

    /// Phase 7 compute post: blur `hdr` into the `hdr_post` storage image (the tonemap
    /// pass then samples `hdr_post` instead of the raw `hdr`).
    /// Tonemap `src` (the chosen HDR-ish image: rasterized HDR / post-chain / path
    /// trace / SW-RT) to the backbuffer, encoding sRGB in-shader. `exposure` is 1.0
    /// for the rasterized path (exposure already baked into lighting) and the camera
    /// exposure for the raw-radiance RT/SW-RT sources.
    ///
    /// PR-5: the final post node also composites bloom (`bloom` = the pyramid mip0
    /// resource, `None` = off) and applies the ASC-CDL color grade (`grade_on == 0` or
    /// `CDL_NEUTRAL` = identity). The bloom resource, when present, is added to the pass
    /// `reads` so the graph sequences the bloom chain before tonemap.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_tonemap<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        backbuffer: ResourceId,
        src: ResourceId,
        post_mode: u32,
        flip_y: u32,
        exposure: f32,
        sharpen: f32,
        inv_w: f32,
        inv_h: f32,
        bloom: Option<ResourceId>,
        bloom_intensity: f32,
        grade_on: u32,
        cdl_slope: [f32; 3],
        cdl_offset: [f32; 3],
        cdl_power: [f32; 3],
    ) {
        let mut reads = vec![src];
        if let Some(b) = bloom {
            reads.push(b);
        }
        graph.add_pass(
            PassInfo {
                name: "tonemap",
                colors: vec![(backbuffer, Some(ClearColor::BLACK))],
                depth: None,
                reads,
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(src);
                let bloom_index = bloom.map(|b| ctx.sampled_index(b)).unwrap_or(u32::MAX);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.post_pipeline);
                cmd.push_constants(&post_push(
                    hdr_index,
                    post_mode,
                    flip_y,
                    exposure,
                    sharpen,
                    inv_w,
                    inv_h,
                    bloom_index,
                    bloom_intensity,
                    grade_on,
                    cdl_slope,
                    cdl_offset,
                    cdl_power,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }

    /// PR-9 view family: composite a secondary view's HDR as a picture-in-picture inset. Tonemaps
    /// `src` into `viewport` (a sub-rect of the backbuffer), LOADING (not clearing) the backbuffer
    /// so the already-composited primary view is preserved everywhere outside the inset. The
    /// full-screen triangle is scaled into the rect by the viewport transform (the graph sets the
    /// full-backbuffer viewport before this closure; we override it here). Bloom/grading are
    /// disabled (the inset is a plain ACES+sRGB encode). See `docs/view-family.md`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_tonemap_inset<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        backbuffer: ResourceId,
        src: ResourceId,
        post_mode: u32,
        flip_y: u32,
        exposure: f32,
        viewport: rhi::Rect2D,
    ) {
        graph.add_pass(
            PassInfo {
                name: "view_inset",
                // `None` clear = load the backbuffer (preserve the primary-view composite).
                colors: vec![(backbuffer, None)],
                depth: None,
                reads: vec![src],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(src);
                let cmd = ctx.cmd();
                // Restrict the draw to the inset rect (both viewport + scissor).
                cmd.set_viewport_scissor_rect(viewport);
                cmd.bind_graphics_pipeline(&self.post_pipeline);
                let (s, o, p) = crate::push::CDL_NEUTRAL;
                cmd.push_constants(&post_push(
                    hdr_index,
                    post_mode,
                    flip_y,
                    exposure,
                    0.0, // no sharpen
                    0.0,
                    0.0,
                    u32::MAX, // no bloom
                    0.0,
                    0, // grading off
                    s,
                    o,
                    p,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}

/// Pack the G-buffer push block (208 bytes): mvp(64), base_color(16),
/// metallic/roughness(16), texture indices u32x4 (16), model mat4 (64), skin u32x4 (16),
/// morph u32x4 (16). `model` is the object->world transform the vertex shader uses for the
/// world-space position and normal G-buffer outputs (the `mvp` already folds it in for clip
/// space); `skin` carries the GPU-skinning storage-buffer indices (0 on the non-skinned
/// path), `morph` the GPU-morph indices (0 on the non-morphed path).
#[allow(clippy::too_many_arguments)]
fn gbuffer_push(
    mvp: [f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    mip_bias: f32,
    alpha_cutoff: f32,
    tex: [u32; 4],
    model: [f32; 16],
    // GPU skinning (Stage B.2): bindless storage-buffer indices for joints / weights /
    // palette + joint count. `[0; 4]` on the non-skinned path (`vsMain` ignores it).
    skin: [u32; 4],
    // GPU morph (Stage C optimization): deltas / weights bindless indices + target_count +
    // vertex_count. `[0; 4]` on the non-morphed path (`vsMain`/`vsMainSkinned` ignore it).
    morph: [u32; 4],
) -> [u8; 208] {
    let mut pc = [0u8; 208];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in base_color.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[80..84].copy_from_slice(&metallic.to_le_bytes());
    pc[84..88].copy_from_slice(&roughness.to_le_bytes());
    pc[88..92].copy_from_slice(&mip_bias.to_le_bytes()); // mr_factor.z = texture LOD bias
    pc[92..96].copy_from_slice(&alpha_cutoff.to_le_bytes()); // mr_factor.w = alpha-test cutoff
    for (i, t) in tex.iter().enumerate() {
        let o = 96 + i * 4;
        pc[o..o + 4].copy_from_slice(&t.to_le_bytes());
    }
    for (i, f) in model.iter().enumerate() {
        let o = 112 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, s) in skin.iter().enumerate() {
        let o = 176 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in morph.iter().enumerate() {
        let o = 192 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    pc
}

/// Pack the lighting push block: 4 G-buffer indices + flip_y + shadow_index + gdf_ao_index +
/// ssao_index + gdf_gi_index + reflect_index + two_sided + exposure_buf_index + skyvis_index
/// (52 bytes). The GDF / SSAO / reflect / skyvis indices are `u32::MAX` when those images are
/// absent (then an exact no-op — the gallery anchor).
#[allow(clippy::too_many_arguments)]
fn pbr_push(
    indices: [u32; 4],
    flip_y: u32,
    shadow_index: u32,
    gdf_ao_index: u32,
    ssao_index: u32,
    gdf_gi_index: u32,
    reflect_index: u32,
    two_sided: u32,
    exposure_buf_index: u32,
    skyvis_index: u32,
    skyvis_tint: f32,
    skyvis_min_occ: f32,
    // Clustered light bufs: [grid_buf, index_buf, light_buf, light_count]. grid_buf == u32::MAX
    // (0xFFFFFFFF) selects the brute-force point-light loop (byte-identical anchor).
    cluster: [u32; 4],
    ao_multibounce: u32,
) -> [u8; 80] {
    let mut pc = [0u8; 80];
    for (i, v) in indices.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[16..20].copy_from_slice(&flip_y.to_le_bytes());
    pc[20..24].copy_from_slice(&shadow_index.to_le_bytes());
    pc[24..28].copy_from_slice(&gdf_ao_index.to_le_bytes());
    pc[28..32].copy_from_slice(&ssao_index.to_le_bytes());
    pc[32..36].copy_from_slice(&gdf_gi_index.to_le_bytes());
    pc[36..40].copy_from_slice(&reflect_index.to_le_bytes());
    pc[40..44].copy_from_slice(&two_sided.to_le_bytes());
    pc[44..48].copy_from_slice(&exposure_buf_index.to_le_bytes());
    pc[48..52].copy_from_slice(&skyvis_index.to_le_bytes());
    pc[52..56].copy_from_slice(&skyvis_tint.to_le_bytes());
    pc[56..60].copy_from_slice(&skyvis_min_occ.to_le_bytes());
    for (i, v) in cluster.iter().enumerate() {
        let o = 60 + i * 4;
        pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[76..80].copy_from_slice(&ao_multibounce.to_le_bytes());
    pc
}

/// Pack the auto-exposure histogram push block (16 bytes): hdr sampled index, histogram
/// storage-buffer index, and the frame width/height.
fn ae_hist_push(hdr: u32, hist_buf: u32, width: u32, height: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr.to_le_bytes());
    pc[4..8].copy_from_slice(&hist_buf.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc
}

/// Pack the auto-exposure resolve push block (48 bytes): histogram + exposure buffer indices,
/// pixel count, target grey `key`, this-frame EMA `adapt`, the `[min,max]` exposure clamp, and
/// the `[lo,hi]` percentile window kept for the trimmed mean.
#[allow(clippy::too_many_arguments)]
fn ae_resolve_push(
    hist_buf: u32,
    expo_buf: u32,
    num_pixels: u32,
    key: f32,
    adapt: f32,
    min_exp: f32,
    max_exp: f32,
    lo_pct: f32,
    hi_pct: f32,
) -> [u8; 48] {
    let mut pc = [0u8; 48];
    pc[0..4].copy_from_slice(&hist_buf.to_le_bytes());
    pc[4..8].copy_from_slice(&expo_buf.to_le_bytes());
    pc[8..12].copy_from_slice(&num_pixels.to_le_bytes());
    pc[12..16].copy_from_slice(&key.to_le_bytes());
    pc[16..20].copy_from_slice(&adapt.to_le_bytes());
    pc[20..24].copy_from_slice(&min_exp.to_le_bytes());
    pc[24..28].copy_from_slice(&max_exp.to_le_bytes());
    pc[28..32].copy_from_slice(&lo_pct.to_le_bytes());
    pc[32..36].copy_from_slice(&hi_pct.to_le_bytes());
    pc
}

/// Pack the shadow push block (112 bytes): light_mvp (64), base_color bindless index (u32),
/// alpha cutoff (f32), 8 bytes padding, skin u32x4 (16), then morph u32x4 (16). `cutoff == 0`
/// (opaque) leaves the texture index unused; `skin == 0` / `morph == 0` are the non-deformed
/// paths (`vsMain` ignores both).
fn shadow_push(
    light_mvp: [f32; 16],
    base_color_tex: u32,
    alpha_cutoff: f32,
    skin: [u32; 4],
    morph: [u32; 4],
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, f) in light_mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&base_color_tex.to_le_bytes());
    pc[68..72].copy_from_slice(&alpha_cutoff.to_le_bytes());
    for (i, s) in skin.iter().enumerate() {
        let o = 80 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    for (i, s) in morph.iter().enumerate() {
        let o = 96 + i * 4;
        pc[o..o + 4].copy_from_slice(&s.to_le_bytes());
    }
    pc
}
