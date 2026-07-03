//! Phase 14 renderer integration — virtual geometry as a deferred G-buffer producer.
//!
//! `VgeoSystem` holds a cluster page **per mesh** (built from the registry's CPU geometry with
//! `build_lod_dag`, so a page matches its uploaded mesh exactly — no cook round-trip, no
//! renormalization) plus shared per-frame scratch, and records the compute + resolve passes that
//! write the REAL Phase-6 G-buffer. Every eligible opaque object is routed through virtual geometry
//! and overlaid on the mesh-rastered remainder (depth-composited), so a whole multi-object scene can
//! render as virtual geometry — the per-static-mesh-asset direction (see
//! `docs/phase-14-vgeo-integration.md`). SW-only for now; opt-in behind `P14_VGEO`, so the default
//! renderer is untouched (gallery byte-identical).
//!
//! One material per page (the mesh's) — matching a single-material mesh; a many-material model is
//! split into per-mesh assets upstream (the scene-cook direction) rather than baking per-cluster
//! materials into one page.

use std::collections::HashMap;
use std::rc::Rc;

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, DepthCompare, Device, Extent2D,
    GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, StorageBuffer, StorageBufferDesc,
    VertexLayout,
};

use crate::deferred::GBufferTargets;
use crate::registry::{GpuMesh, MeshRegistry};
use crate::{DEPTH_FORMAT, FRAMES_IN_FLIGHT};

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

/// One mesh's immutable cluster geometry in the bindless storage table.
struct ClusterPage {
    vtx: StorageBuffer,
    remap: StorageBuffer,
    tri: StorageBuffer,
    rec: StorageBuffer,
    total_clusters: u32,
}

pub(crate) struct VgeoSystem {
    pages: Vec<ClusterPage>,
    /// `Rc<GpuMesh>` pointer → page index, so a scene object routes to its mesh's page.
    mesh_to_page: HashMap<usize, usize>,
    /// Max clusters over all pages (sizes the per-object visible-list slots).
    max_clusters: u32,
    /// Per-FIF shared R64 visibility buffer (sized to the render extent). Objects serialize on it
    /// (each clears → rasters → resolves), so one buffer suffices; inter-object occlusion is
    /// resolved by the G-buffer depth test in the resolve, not the visibility buffer.
    visbuf: Vec<StorageBuffer>,
    /// Per-FIF growable pool of per-object scratch: `(visible list, indirect-append counter)`.
    pool: Vec<Vec<(StorageBuffer, StorageBuffer)>>,
    extent: Extent2D,
    cut_pipeline: ComputePipeline,
    clear_pipeline: ComputePipeline,
    raster_pipeline: ComputePipeline,
    resolve_pipeline: GraphicsPipeline,
    tau: f32,
}

fn vis_bytes(extent: Extent2D) -> u64 {
    (extent.width.max(1) as u64) * (extent.height.max(1) as u64) * 8
}

impl VgeoSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        registry: &MeshRegistry,
        extent: Extent2D,
    ) -> anyhow::Result<Self> {
        use dreamcoast_scene::MeshHandle;

        if !device.capabilities().mesh_shader {
            anyhow::bail!(
                "P14_VGEO: {backend:?} lacks the capabilities for virtual geometry (mesh_shader=false)"
            );
        }

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

        // Build a cluster page per mesh from its CPU geometry (build_lod_dag = what the cook runs).
        let mut pages: Vec<ClusterPage> = Vec::new();
        let mut mesh_to_page: HashMap<usize, usize> = HashMap::new();
        let mut max_clusters = 0u32;
        for i in 0..registry.len() {
            let handle = MeshHandle(i);
            let cpu = registry.cpu(handle);
            if cpu.indices.len() < 3 {
                continue;
            }
            let dag = dreamcoast_asset::vgeo::build_lod_dag(&cpu.vertices, &cpu.indices, 0);
            if dag.clusters.is_empty() {
                continue;
            }
            let page = Self::upload_page(&sb, &dag)?;
            max_clusters = max_clusters.max(page.total_clusters);
            let idx = pages.len();
            pages.push(page);
            mesh_to_page.insert(Rc::as_ptr(&registry.get(handle)) as usize, idx);
        }
        if pages.is_empty() {
            anyhow::bail!("P14_VGEO: no mesh produced clusters");
        }
        tracing::info!(
            "P14_VGEO: built {} cluster page(s) (max {} clusters)",
            pages.len(),
            max_clusters
        );

        let mut visbuf = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            visbuf.push(device.create_storage_buffer(&StorageBufferDesc {
                size: vis_bytes(extent),
                stride: 8,
                indirect: false,
            })?);
        }
        let pool = (0..FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect();

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
            224,
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

        let tau: f32 = std::env::var("VGEO_TAU")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8.0);

        Ok(Self {
            pages,
            mesh_to_page,
            max_clusters,
            visbuf,
            pool,
            extent,
            cut_pipeline,
            clear_pipeline,
            raster_pipeline,
            resolve_pipeline,
            tau,
        })
    }

    /// Serialize a cluster DAG into bindless storage buffers (same layout as the shaders expect).
    fn upload_page(
        sb: &impl Fn(&[u8], u32) -> anyhow::Result<StorageBuffer>,
        dag: &dreamcoast_asset::vgeo::MeshClusters,
    ) -> anyhow::Result<ClusterPage> {
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
        Ok(ClusterPage {
            vtx: sb(&vtx, 32)?,
            remap: sb(&remap, 4)?,
            tri: sb(&tri, 4)?,
            rec: sb(&rec, 96)?,
            total_clusters: dag.clusters.len() as u32,
        })
    }

    /// The page index for a scene object's mesh, or `None` (→ rasterize it via the mesh fill).
    pub(crate) fn page_for(&self, mesh: &Rc<GpuMesh>) -> Option<usize> {
        self.mesh_to_page.get(&(Rc::as_ptr(mesh) as usize)).copied()
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

    /// Grow this FIF's scratch pool to `count` object slots and host-reset the ones in use: each
    /// object gets a zeroed append counter + a sentinel-cleared (`0xFFFFFFFF`) visible list, so the
    /// fixed-count raster dispatch skips the slots the cut didn't fill (the graph IR recorder has no
    /// indirect compute dispatch). Safe: this FIF's fence was waited at frame start.
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        fif: usize,
        count: usize,
    ) -> anyhow::Result<()> {
        let slots = &mut self.pool[fif];
        while slots.len() < count {
            let vislist = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: (self.max_clusters.max(1) as u64) * 4,
                    stride: 4,
                    indirect: false,
                },
                &vec![0xFFu8; self.max_clusters.max(1) as usize * 4],
            )?;
            let args = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: 12,
                    stride: 4,
                    indirect: false,
                },
                &[0u8; 12],
            )?;
            slots.push((vislist, args));
        }
        for (vislist, args) in slots.iter().take(count) {
            args.write(&0u32.to_le_bytes())?;
            vislist.write(&vec![0xFFu8; self.max_clusters.max(1) as usize * 4])?;
        }
        Ok(())
    }

    /// The shared visibility buffer's ordering handle (imported once per frame so every object's
    /// passes serialize on it).
    pub(crate) fn import_visbuf(graph: &mut RenderGraph) -> ResourceId {
        graph.import_external("vgeo_visbuf")
    }

    /// Record one object's virtual-geometry G-buffer contribution (cut → clear → SW raster →
    /// resolve overlay). `slot` indexes this frame's scratch pool; `visbuf_ext` is the shared handle
    /// from [`Self::import_visbuf`]. The mesh fill must run first (it clears the G-buffer + depth);
    /// the resolve LOADs them and Less-tests the model's `SV_Depth`, compositing with the rest.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_object<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gbuf: GBufferTargets,
        fif: usize,
        slot: usize,
        page_idx: usize,
        visbuf_ext: ResourceId,
        view_proj: Mat4,
        cull_view_proj: Mat4,
        eye: Vec3,
        model: Mat4,
        material: VgeoMaterial,
        extent: Extent2D,
    ) {
        let page = &self.pages[page_idx];
        let visbuf = &self.visbuf[fif];
        let (vislist, args) = &self.pool[fif][slot];
        let (w, h) = (extent.width, extent.height);
        let total = page.total_clusters;
        let tau = self.tau;
        let (vtx_i, remap_i, tri_i, rec_i) = (
            page.vtx.storage_index(),
            page.remap.storage_index(),
            page.tri.storage_index(),
            page.rec.storage_index(),
        );

        let vislist_ext = graph.import_external("vgeo_vislist");
        let args_ext = graph.import_external("vgeo_args");

        // ── LOD cut (world-space): WORLD frustum planes + cam; the shader transforms each cluster's
        // local bounds by `model` (handles non-uniform node scale that a local-space test skews). ──
        let planes = crate::push::frustum_planes(cull_view_proj);
        let cam = eye;
        let max_scale = model
            .x_axis
            .truncate()
            .length()
            .max(model.y_axis.truncate().length())
            .max(model.z_axis.truncate().length());
        let model_arr = model.to_cols_array();
        let proj_factor = 0.5 * h as f32 / (30f32.to_radians()).tan();
        let cut = &self.cut_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_cut",
                storage_writes: vec![vislist_ext, args_ext],
                reads: vec![],
            },
            move |ctx| {
                let mut cpc = [0u8; 224];
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
                for (i, v) in model_arr.iter().enumerate() {
                    cpc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
                }
                cpc[208..212].copy_from_slice(&max_scale.to_le_bytes());
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(cut);
                cmd.push_constants_compute(&cpc);
                cmd.dispatch(total.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(vislist);
                Ok(())
            },
        );

        // ── Clear the R64 visibility buffer, then SW-raster the cut into it ──
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
    }
}
