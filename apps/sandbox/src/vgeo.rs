//! Phase 14 renderer integration (I2) — virtual geometry as a deferred G-buffer producer.
//!
//! `VgeoSystem` owns the cooked cluster pages of a single model plus the per-frame scratch (the R64
//! visibility buffer, the cut's visible-cluster list, the indirect-args), and records the compute +
//! resolve passes that write the REAL Phase-6 G-buffer — so the existing deferred lighting consumes
//! virtual geometry unchanged (see `docs/phase-14-vgeo-integration.md`). SW-only for this first step
//! (the M5b HW/SW binning path is a graph follow-up); opt-in behind `P14_VGEO`, so the default
//! renderer is untouched (gallery byte-identical).
//!
//! The cooked clusters are in raw model space; this normalizes them with the SAME arithmetic as
//! `normalize_on_ground` (recenter + base at y=0 + unit radius) so they land in the mesh's
//! normalized space, and the object's `transform` is then the single model matrix — matching the
//! mesh G-buffer fill exactly for the parity gate.

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, DepthCompare, Device, Extent2D,
    GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, StorageBuffer, StorageBufferDesc,
    VertexLayout,
};

use crate::deferred::GBufferTargets;
use crate::{COLOR_FORMAT, DEPTH_FORMAT, FRAMES_IN_FLIGHT};

/// Single-object material for the resolve (mirrors the fields `gbuffer_push` feeds the mesh fill).
#[derive(Clone, Copy)]
pub(crate) struct VgeoMaterial {
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    /// base color, metallic-roughness, normal, emissive bindless indices (`NO_TEXTURE` if absent).
    pub(crate) tex: [u32; 4],
    pub(crate) alpha_cutoff: f32,
}

pub(crate) struct VgeoSystem {
    // Immutable cluster geometry (shared across frames), all in the bindless storage table.
    vtx: StorageBuffer,
    remap: StorageBuffer,
    tri: StorageBuffer,
    rec: StorageBuffer,
    total_clusters: u32,
    // Per-frame-in-flight scratch: R64 visibility buffer (sized to the render extent), the cut's
    // visible-cluster list, and the indirect-args header. Host-reset each frame after the fence.
    visbuf: Vec<StorageBuffer>,
    vislist: Vec<StorageBuffer>,
    args: Vec<StorageBuffer>,
    extent: Extent2D,
    // Pipelines: LOD cut (SW-only), visibility-buffer clear + SW raster, and the G-buffer resolve.
    cut_pipeline: ComputePipeline,
    clear_pipeline: ComputePipeline,
    raster_pipeline: ComputePipeline,
    resolve_pipeline: GraphicsPipeline,
    /// LOD cut pixel-error threshold (`VGEO_TAU`).
    tau: f32,
}

fn vis_bytes(extent: Extent2D) -> u64 {
    (extent.width.max(1) as u64) * (extent.height.max(1) as u64) * 8
}

impl VgeoSystem {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        source: &std::path::Path,
        cache_key: &str,
        cache_dir: &std::path::Path,
        tex: dreamcoast_asset::cook::TexCompress,
        extent: Extent2D,
    ) -> anyhow::Result<Self> {
        if !device.capabilities().mesh_shader {
            anyhow::bail!(
                "P14_VGEO: {backend:?} lacks the capabilities for virtual geometry (mesh_shader=false)"
            );
        }
        let mut dag =
            dreamcoast_asset::cook::load_cooked_clusters(source, cache_key, cache_dir, tex)?;

        // Normalize the cluster data into the mesh's canonical space (identical arithmetic to
        // `normalize_on_ground`): recenter the footprint on the origin, rest the base on y=0, and
        // scale the bounding-sphere radius to 1. The cluster geometry is the same source as the
        // normalized mesh, so this reproduces the mesh's transform exactly — the object's world
        // transform is then the single model matrix (parity with the mesh G-buffer fill).
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &dag.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.pos[i]);
                max[i] = max[i].max(v.pos[i]);
            }
        }
        let c = Vec3::new((min[0] + max[0]) * 0.5, min[1], (min[2] + max[2]) * 0.5);
        let (sx, sy, sz) = (max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        let s = 1.0 / (0.5 * (sx * sx + sy * sy + sz * sz).sqrt()).max(1e-6);
        for v in &mut dag.vertices {
            let p = (Vec3::from(v.pos) - c) * s;
            v.pos = p.to_array();
        }
        for cl in &mut dag.clusters {
            cl.self_center = ((Vec3::from(cl.self_center) - c) * s).to_array();
            cl.self_radius *= s;
            cl.self_error *= s;
            cl.parent_center = ((Vec3::from(cl.parent_center) - c) * s).to_array();
            cl.parent_radius *= s;
            if cl.parent_error < 3.0e38 {
                cl.parent_error *= s;
            }
            cl.bounds_center = ((Vec3::from(cl.bounds_center) - c) * s).to_array();
            cl.bounds_radius *= s;
        }

        // Upload the (immutable) cluster geometry into bindless storage buffers.
        let sb = |bytes: &[u8], stride: u32| -> anyhow::Result<StorageBuffer> {
            Ok(device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: bytes.len().max(1) as u64,
                    stride,
                    indirect: false,
                },
                bytes,
            )?)
        };
        let mut vtx = Vec::with_capacity(dag.vertices.len() * 32);
        for v in &dag.vertices {
            for f in v.pos.iter().chain(&v.normal) {
                vtx.extend_from_slice(&f.to_le_bytes());
            }
            for f in v.uv {
                vtx.extend_from_slice(&f.to_le_bytes());
            }
        }
        let mut remap = Vec::with_capacity(dag.cluster_vertices.len() * 4);
        for &i in &dag.cluster_vertices {
            remap.extend_from_slice(&i.to_le_bytes());
        }
        let mut tri = Vec::with_capacity(dag.cluster_triangles.len() * 4);
        for &b in &dag.cluster_triangles {
            tri.extend_from_slice(&(b as u32).to_le_bytes());
        }
        // Per-cluster GpuCluster records (96 B), all clusters — same layout as the viewer/shaders.
        let mut rec = Vec::with_capacity(dag.clusters.len() * 96);
        let put = |rec: &mut Vec<u8>, f: f32| rec.extend_from_slice(&f.to_le_bytes());
        let put3 = |rec: &mut Vec<u8>, v: [f32; 3]| {
            for f in v {
                rec.extend_from_slice(&f.to_le_bytes());
            }
        };
        for cl in &dag.clusters {
            for field in [
                cl.vertex_offset,
                cl.vertex_count,
                cl.triangle_offset,
                cl.triangle_count,
            ] {
                rec.extend_from_slice(&field.to_le_bytes());
            }
            put(&mut rec, cl.self_error);
            put3(&mut rec, cl.self_center);
            put(&mut rec, cl.self_radius);
            put(&mut rec, cl.parent_error);
            put3(&mut rec, cl.parent_center);
            put(&mut rec, cl.parent_radius);
            put3(&mut rec, cl.bounds_center);
            put(&mut rec, cl.bounds_radius);
            put3(&mut rec, cl.cone_axis);
            put(&mut rec, cl.cone_cutoff);
            rec.extend_from_slice(&[0u8; 8]); // 88..96 pad
        }
        let total_clusters = dag.clusters.len() as u32;
        let vtx = sb(&vtx, 32)?;
        let remap = sb(&remap, 4)?;
        let tri = sb(&tri, 4)?;
        let rec = sb(&rec, 96)?;

        // Per-FIF scratch rings.
        let mut visbuf = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut vislist = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut args = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            visbuf.push(device.create_storage_buffer(&StorageBufferDesc {
                size: vis_bytes(extent),
                stride: 8,
                indirect: false,
            })?);
            vislist.push(sb(&vec![0u8; total_clusters.max(1) as usize * 4], 4)?);
            args.push(sb(&[0u8; 12], 4)?);
        }

        // Pipelines.
        let compute = |spirv: fn() -> Option<&'static [u8]>,
                       dxil: fn() -> Option<&'static [u8]>,
                       metal: fn() -> Option<&'static [u8]>,
                       entry: &'static str,
                       size: u32,
                       threads: [u32; 3]|
         -> anyhow::Result<ComputePipeline> {
            let bytes = match backend {
                BackendKind::Vulkan => spirv(),
                BackendKind::D3d12 => dxil(),
                BackendKind::Metal => metal(),
            }
            .ok_or_else(|| anyhow::anyhow!("{entry} shader unavailable for {backend:?}"))?;
            Ok(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: bytes,
                compute_entry: entry,
                push_constant_size: size,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: threads,
            })?)
        };
        let cut_pipeline = compute(
            dreamcoast_shader::vgeo_cut_cs_spirv,
            dreamcoast_shader::vgeo_cut_cs_dxil,
            dreamcoast_shader::vgeo_cut_cs_metallib,
            "csCut",
            144,
            [64, 1, 1],
        )?;
        let clear_pipeline = compute(
            dreamcoast_shader::vgeo_swraster_clear_cs_spirv,
            dreamcoast_shader::vgeo_swraster_clear_cs_dxil,
            dreamcoast_shader::vgeo_swraster_clear_cs_metallib,
            "csClear",
            96,
            [64, 1, 1],
        )?;
        let raster_pipeline = compute(
            dreamcoast_shader::vgeo_swraster_cs_spirv,
            dreamcoast_shader::vgeo_swraster_cs_dxil,
            dreamcoast_shader::vgeo_swraster_cs_metallib,
            "csRaster",
            96,
            [128, 1, 1],
        )?;
        let (rvs, rfs) = match backend {
            BackendKind::Vulkan => (
                dreamcoast_shader::vgeo_gbuffer_vs_spirv(),
                dreamcoast_shader::vgeo_gbuffer_fs_spirv(),
            ),
            BackendKind::D3d12 => (
                dreamcoast_shader::vgeo_gbuffer_vs_dxil(),
                dreamcoast_shader::vgeo_gbuffer_fs_dxil(),
            ),
            BackendKind::Metal => (
                dreamcoast_shader::vgeo_gbuffer_vs_metallib(),
                dreamcoast_shader::vgeo_gbuffer_fs_metallib(),
            ),
        };
        let resolve_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: rvs.ok_or_else(|| anyhow::anyhow!("vgeo_gbuffer vs unavailable"))?,
            fragment_bytes: rfs.ok_or_else(|| anyhow::anyhow!("vgeo_gbuffer fs unavailable"))?,
            vertex_entry: "vsMain",
            fragment_entry: "fsGBuffer",
            color_formats: &[
                crate::GB_ALBEDO_FMT,
                crate::GB_NORMAL_FMT,
                crate::GB_MATERIAL_FMT,
                crate::GB_POSITION_FMT,
            ],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: rhi::BlendMode::Opaque,
            push_constant_size: 208,
            bindless: true,
            uniform_buffer: false,
            depth_test: true,
            depth_write: true,
            depth_compare: DepthCompare::Less,
            depth_format: Some(DEPTH_FORMAT),
        })?;
        let _ = COLOR_FORMAT;

        let tau: f32 = std::env::var("VGEO_TAU")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8.0);

        Ok(Self {
            vtx,
            remap,
            tri,
            rec,
            total_clusters,
            visbuf,
            vislist,
            args,
            extent,
            cut_pipeline,
            clear_pipeline,
            raster_pipeline,
            resolve_pipeline,
            tau,
        })
    }

    /// Recreate the per-FIF visibility buffers when the render extent changes.
    pub(crate) fn resize(&mut self, device: &Device, extent: Extent2D) -> anyhow::Result<()> {
        if extent.width == self.extent.width && extent.height == self.extent.height {
            return Ok(());
        }
        self.extent = extent;
        for vb in &mut self.visbuf {
            *vb = device.create_storage_buffer(&StorageBufferDesc {
                size: vis_bytes(extent),
                stride: 8,
                indirect: false,
            })?;
        }
        Ok(())
    }

    /// Record the SW-only virtual-geometry G-buffer producer for `fif`: reset args → LOD cut →
    /// clear visbuf → SW raster (indirect) → resolve into the four G-buffer MRTs (+ depth). Replaces
    /// `DeferredRenderer::record_gbuffer` under `P14_VGEO=1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gbuffer<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gbuf: GBufferTargets,
        fif: usize,
        view_proj: Mat4,
        cull_view_proj: Mat4,
        eye: Vec3,
        model: Mat4,
        material: VgeoMaterial,
        extent: Extent2D,
    ) -> anyhow::Result<()> {
        let visbuf = &self.visbuf[fif];
        let vislist = &self.vislist[fif];
        let args = &self.args[fif];
        // Fresh append counter + sentinel-cleared visible list each frame (this FIF's fence was
        // waited at frame start, so the host writes can't race in-flight GPU work). The list is
        // cleared to 0xFFFFFFFF so the fixed-count raster dispatch (no indirect compute in the graph
        // IR) skips the slots the cut didn't fill — see `vgeo_swraster.slang` csRaster.
        args.write(&0u32.to_le_bytes())?;
        vislist.write(&vec![0xFFu8; self.total_clusters.max(1) as usize * 4])?;

        let (w, h) = (extent.width, extent.height);
        let total = self.total_clusters;
        let tau = self.tau;
        let (vtx_i, remap_i, tri_i, rec_i) = (
            self.vtx.storage_index(),
            self.remap.storage_index(),
            self.tri.storage_index(),
            self.rec.storage_index(),
        );

        let vislist_ext = graph.import_external("vgeo_vislist");
        let args_ext = graph.import_external("vgeo_args");
        let visbuf_ext = graph.import_external("vgeo_visbuf");

        // ── LOD cut (model-space): planes from cull_view_proj*model, cam = inverse(model)*eye ──
        // The cluster spheres stay in model space; screen error's err/dist ratio is scale-invariant,
        // so this handles the object transform (incl. uniform scale) exactly.
        let planes = crate::push::frustum_planes(cull_view_proj * model);
        let cam = model.inverse().transform_point3(eye);
        let proj_factor = 0.5 * h as f32 / (30f32.to_radians()).tan();
        let cut = &self.cut_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_cut",
                storage_writes: vec![vislist_ext, args_ext],
                reads: vec![],
            },
            move |ctx| {
                let mut cpc = [0u8; 144];
                for (i, plane) in planes.iter().enumerate() {
                    for (j, f) in plane.iter().enumerate() {
                        cpc[i * 16 + j * 4..i * 16 + j * 4 + 4].copy_from_slice(&f.to_le_bytes());
                    }
                }
                cpc[96..100].copy_from_slice(&cam.x.to_le_bytes());
                cpc[100..104].copy_from_slice(&cam.y.to_le_bytes());
                cpc[104..108].copy_from_slice(&cam.z.to_le_bytes());
                cpc[108..112].copy_from_slice(&proj_factor.to_le_bytes());
                cpc[112..116].copy_from_slice(&tau.to_le_bytes());
                for (i, word) in [total, rec_i, vislist.storage_index(), args.storage_index()]
                    .iter()
                    .enumerate()
                {
                    cpc[116 + i * 4..120 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(cut);
                cmd.push_constants_compute(&cpc);
                cmd.dispatch(total.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(vislist);
                Ok(())
            },
        );

        // ── Clear the R64 visibility buffer, then SW-raster the cut into it (indirect/cluster) ──
        let mvp = (view_proj * model).to_cols_array();
        let clear = &self.clear_pipeline;
        let raster = &self.raster_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_raster",
                storage_writes: vec![visbuf_ext],
                reads: vec![args_ext, vislist_ext],
            },
            move |ctx| {
                let mut spc = [0u8; 96];
                for (i, v) in mvp.iter().enumerate() {
                    spc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                for (i, word) in [
                    vtx_i,
                    remap_i,
                    tri_i,
                    rec_i,
                    visbuf.storage_index(),
                    vislist.storage_index(),
                    w,
                    h,
                ]
                .iter()
                .enumerate()
                {
                    spc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(clear);
                cmd.push_constants_compute(&spc);
                cmd.dispatch((w * h).div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(visbuf);
                cmd.bind_compute_pipeline(raster);
                cmd.push_constants_compute(&spc);
                // Fixed dispatch = one threadgroup per cluster slot; csRaster skips sentinels.
                cmd.dispatch(total, 1, 1);
                cmd.storage_buffer_barrier(visbuf);
                Ok(())
            },
        );

        // ── Resolve: visibility buffer → the four G-buffer MRTs (+ depth), OVERLAY ──
        // The mesh fill runs first (draws the other opaque objects + ground, clearing the G-buffer +
        // depth), so this LOADs every target and overlays the model where covered: `fsGBuffer`
        // discards empty pixels (the cleared/other-object values survive) and emits SV_Depth, and the
        // Less depth test against the loaded depth resolves the model vs the mesh-rastered objects.
        let resolve = &self.resolve_pipeline;
        let model_arr = model.to_cols_array();
        graph.add_pass(
            PassInfo {
                name: "vgeo_resolve",
                colors: vec![
                    (gbuf.albedo, None),
                    (gbuf.normal, None),
                    (gbuf.material, None),
                    (gbuf.position, None),
                ],
                depth: Some(gbuf.depth),
                reads: vec![visbuf_ext],
            },
            move |ctx| {
                let mut rpc = [0u8; 208];
                for (i, v) in mvp.iter().enumerate() {
                    rpc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                for (i, v) in model_arr.iter().enumerate() {
                    rpc[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
                }
                for (i, word) in [
                    vtx_i,
                    remap_i,
                    tri_i,
                    rec_i,
                    visbuf.storage_index(),
                    w,
                    h,
                    0,
                ]
                .iter()
                .enumerate()
                {
                    rpc[128 + i * 4..132 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                for (i, f) in material.base_color.iter().enumerate() {
                    rpc[160 + i * 4..164 + i * 4].copy_from_slice(&f.to_le_bytes());
                }
                let mr = [
                    material.metallic,
                    material.roughness,
                    0.0,
                    material.alpha_cutoff,
                ];
                for (i, f) in mr.iter().enumerate() {
                    rpc[176 + i * 4..180 + i * 4].copy_from_slice(&f.to_le_bytes());
                }
                for (i, word) in material.tex.iter().enumerate() {
                    rpc[192 + i * 4..196 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(resolve);
                cmd.push_constants(&rpc);
                cmd.draw(3, 1);
                Ok(())
            },
        );
        Ok(())
    }
}
