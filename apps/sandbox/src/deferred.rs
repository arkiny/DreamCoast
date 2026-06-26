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
    ComputePipelineDesc, Format, GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology,
    VertexLayout,
};

use crate::app::{load_compute_shader, load_shader_pair};
use crate::push::{post_compute_push, post_push};
use crate::{
    DEPTH_FORMAT, FRAMES_IN_FLIGHT, GB_ALBEDO_FMT, GB_MATERIAL_FMT, GB_NORMAL_FMT, GB_POSITION_FMT,
    GLOBALS_SLICE, HDR_FORMAT, NO_TEXTURE, SceneObject,
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
    gbuffer_pipeline: GraphicsPipeline,
    pbr_pipeline: GraphicsPipeline,
    post_pipeline: GraphicsPipeline,
    post_compute_pipeline: ComputePipeline,
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
            push_constant_size: 112, // mat4(64) + base_color(16) + mr(16) + tex u32x4(16)
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
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
            vertex_layout: VertexLayout::MeshPosition,
            blend: BlendMode::Opaque,
            push_constant_size: 64, // light_mvp mat4
            bindless: true,         // for the root-constants param (push constants)
            uniform_buffer: false,
            depth_test: true,
            depth_format: Some(DEPTH_FORMAT),
        })?;

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
            push_constant_size: 36, // 4 G-buffer indices + flip_y + shadow + gdf_ao + gdf_gi + reflect
            bindless: true,
            uniform_buffer: true,
            depth_test: false,
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
            push_constant_size: 16, // hdr_index + mode + flip_y + pad
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_format: None,
        })?;

        // Compute post-process pipeline (Phase 7): blurs the HDR target into a
        // storage image between the lighting and tonemap passes.
        let post_compute_cs = load_compute_shader(
            backend,
            dreamcoast_shader::post_compute_cs_spirv,
            dreamcoast_shader::post_compute_cs_dxil,
            dreamcoast_shader::post_compute_cs_metallib,
            "post_compute",
        )?;
        let post_compute_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: post_compute_cs,
            compute_entry: "csMain",
            push_constant_size: 16, // hdr_index + out_index + width + height
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [8, 8, 1],
        })?;

        // Per-frame globals uniform buffer (one 256-byte slice per frame-in-flight).
        let globals_buffer = device.create_buffer(&BufferDesc {
            size: GLOBALS_SLICE * FRAMES_IN_FLIGHT as u64,
            usage: BufferUsage::Uniform,
        })?;
        device.set_globals_buffer(&globals_buffer, GLOBALS_SLICE);

        Ok(Self {
            shadow_pipeline,
            gbuffer_pipeline,
            pbr_pipeline,
            post_pipeline,
            post_compute_pipeline,
            globals_buffer,
        })
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
                    cmd.push_constants(&mat4_bytes(&lmvp));
                    cmd.bind_vertex_buffer(&obj.vbuf, 32);
                    cmd.bind_index_buffer(&obj.ibuf, true);
                    cmd.draw_indexed(obj.index_count, 0, 0);
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
    ) {
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
                cmd.bind_graphics_pipeline(&self.gbuffer_pipeline);
                for obj in scene {
                    let obj_mvp = (view_proj * obj.transform).to_cols_array();
                    let (m, rgh, mr_tex) = if override_material {
                        (metallic_override, roughness_override, NO_TEXTURE)
                    } else {
                        (obj.metallic, obj.roughness, obj.tex[1])
                    };
                    cmd.push_constants(&gbuffer_push(
                        obj_mvp,
                        obj.base_color,
                        m,
                        rgh,
                        [obj.tex[0], mr_tex, obj.tex[2], obj.tex[3]],
                    ));
                    cmd.bind_vertex_buffer(&obj.vbuf, 32);
                    cmd.bind_index_buffer(&obj.ibuf, true);
                    cmd.draw_indexed(obj.index_count, 0, 0);
                }
                // Ground plane (plain matte material, no textures).
                cmd.push_constants(&gbuffer_push(
                    view_proj.to_cols_array(),
                    [0.8, 0.8, 0.8, 1.0],
                    0.0,
                    0.9,
                    [NO_TEXTURE; 4],
                ));
                cmd.bind_vertex_buffer(ground_vbuf, 32);
                cmd.bind_index_buffer(ground_ibuf, true);
                cmd.draw_indexed(ground_count, 0, 0);
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
        shadow_map: ResourceId,
        gdf_ao: Option<ResourceId>,
        gdf_gi: Option<ResourceId>,
        reflect: Option<ResourceId>,
        globals_offset: u64,
        flip_y: u32,
    ) {
        let mut reads = vec![
            gbuf.albedo,
            gbuf.normal,
            gbuf.material,
            gbuf.position,
            shadow_map,
        ];
        if let Some(ao) = gdf_ao {
            reads.push(ao);
        }
        if let Some(gi) = gdf_gi {
            reads.push(gi);
        }
        if let Some(r) = reflect {
            reads.push(r);
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
                let shadow_index = ctx.sampled_index(shadow_map);
                let ao_index = gdf_ao.map(|ao| ctx.sampled_index(ao)).unwrap_or(u32::MAX);
                let gi_index = gdf_gi.map(|gi| ctx.sampled_index(gi)).unwrap_or(u32::MAX);
                let reflect_index = reflect.map(|r| ctx.sampled_index(r)).unwrap_or(u32::MAX);
                let cmd = ctx.cmd();
                cmd.set_globals(&self.globals_buffer, globals_offset);
                cmd.bind_graphics_pipeline(&self.pbr_pipeline);
                cmd.push_constants(&pbr_push(
                    indices,
                    flip_y,
                    shadow_index,
                    ao_index,
                    gi_index,
                    reflect_index,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }

    /// Phase 7 compute post: blur `hdr` into the `hdr_post` storage image (the tonemap
    /// pass then samples `hdr_post` instead of the raw `hdr`).
    pub(crate) fn record_compute_post<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        hdr_post: ResourceId,
        cw: u32,
        ch: u32,
    ) {
        graph.add_compute_pass(
            ComputePassInfo {
                name: "post_compute",
                storage_writes: vec![hdr_post],
                reads: vec![hdr],
            },
            move |ctx| {
                let in_index = ctx.sampled_index(hdr);
                let out_index = ctx.storage_index(hdr_post);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(&self.post_compute_pipeline);
                cmd.push_constants_compute(&post_compute_push(in_index, out_index, cw, ch));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
    }

    /// Tonemap `src` (the chosen HDR-ish image: rasterized HDR / compute-post / path
    /// trace / SW-RT) to the backbuffer, encoding sRGB in-shader. `exposure` is 1.0
    /// for the rasterized path (exposure already baked into lighting) and the camera
    /// exposure for the raw-radiance RT/SW-RT sources.
    pub(crate) fn record_tonemap<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        backbuffer: ResourceId,
        src: ResourceId,
        post_mode: u32,
        flip_y: u32,
        exposure: f32,
    ) {
        graph.add_pass(
            PassInfo {
                name: "tonemap",
                colors: vec![(backbuffer, Some(ClearColor::BLACK))],
                depth: None,
                reads: vec![src],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(src);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.post_pipeline);
                cmd.push_constants(&post_push(hdr_index, post_mode, flip_y, exposure));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}

/// Pack the G-buffer push block: mvp(64) + base_color(16) + metallic/roughness(16)
/// + texture indices u32x4 (16) = 112 bytes.
fn gbuffer_push(
    mvp: [f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    tex: [u32; 4],
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in base_color.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[80..84].copy_from_slice(&metallic.to_le_bytes());
    pc[84..88].copy_from_slice(&roughness.to_le_bytes());
    for (i, t) in tex.iter().enumerate() {
        let o = 96 + i * 4;
        pc[o..o + 4].copy_from_slice(&t.to_le_bytes());
    }
    pc
}

/// Pack the lighting push block: 4 G-buffer indices + flip_y + shadow_index +
/// gdf_ao_index + gdf_gi_index + reflect_index (36 bytes). The GDF / reflect indices are
/// `u32::MAX` when the C2 AO / C3 GI / C7c hybrid-reflection images are absent.
fn pbr_push(
    indices: [u32; 4],
    flip_y: u32,
    shadow_index: u32,
    gdf_ao_index: u32,
    gdf_gi_index: u32,
    reflect_index: u32,
) -> [u8; 36] {
    let mut pc = [0u8; 36];
    for (i, v) in indices.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[16..20].copy_from_slice(&flip_y.to_le_bytes());
    pc[20..24].copy_from_slice(&shadow_index.to_le_bytes());
    pc[24..28].copy_from_slice(&gdf_ao_index.to_le_bytes());
    pc[28..32].copy_from_slice(&gdf_gi_index.to_le_bytes());
    pc[32..36].copy_from_slice(&reflect_index.to_le_bytes());
    pc
}

/// View a column-major 4x4 matrix as push-constant bytes.
fn mat4_bytes(m: &[f32; 16]) -> [u8; 64] {
    let mut pc = [0u8; 64];
    for (i, f) in m.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc
}
