//! Phase 11 software ray tracing + global distance field, extracted from `run()` —
//! R3 of the render-loop decomposition (see docs/refactor-sandbox.md). The largest
//! bundle: it owns both volumes (per-mesh SDF + merged GDF), the bake mesh, the
//! instance table, and every Stage-A/B compute pipeline (analytic sdf trace, volume
//! fill/view, SDF bake, GDF merge, GDF trace).
//!
//! Each `record_*` adds one feature's passes and returns the output storage image;
//! the frame loop keeps the mutual-exclusion gating (only one replaces the HDR) and
//! the build-once flags, passing `build` in (the bake/merge run once, then the
//! persistent volumes are re-viewed). All record methods borrow `&'a self` for the
//! graph's lifetime, like the other bundles.

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, Format, StorageBuffer,
    StorageBufferDesc, Volume, VolumeDesc,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::mesh::{index_bytes, vertex_bytes};
use crate::push::{gdf_merge_push, gdf_trace_push, sdf_bake_push, sdf_trace_push, volume_push};

/// Volume edge length in voxels (cube). The bake/merge/view all share it.
const VOLUME_DIM: u32 = 64;

pub(crate) struct GdfSystem {
    /// Per-mesh SDF bake target (B2) + GDF merge source (B3).
    volume: Option<Volume>,
    /// World-space merged global distance field (B3) + trace source (B4).
    gdf: Option<Volume>,
    fill_pipeline: Option<ComputePipeline>,  // B1 volume fill
    view_pipeline: Option<ComputePipeline>,  // B1 slice view (reused by B2/B3)
    bake_pipeline: Option<ComputePipeline>,  // B2 per-mesh SDF bake
    merge_pipeline: Option<ComputePipeline>, // B3 instance merge
    trace_pipeline: Option<ComputePipeline>, // B4 GDF sphere-march
    sdf_pipeline: Option<ComputePipeline>,   // Stage-A analytic sphere-march
    bake_vtx: Option<StorageBuffer>,
    bake_idx: Option<StorageBuffer>,
    bake_tri_count: u32,
    /// (table, instance count) for the merge; `None` without a volume.
    instances: Option<(StorageBuffer, u32)>,
}

impl GdfSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Self> {
        let make_volume = || -> anyhow::Result<Option<Volume>> {
            if compute_supported {
                Ok(Some(device.create_volume(&VolumeDesc {
                    width: VOLUME_DIM,
                    height: VOLUME_DIM,
                    depth: VOLUME_DIM,
                    format: Format::R32Float,
                })?))
            } else {
                Ok(None)
            }
        };
        let volume = make_volume()?;
        let gdf = make_volume()?;

        let compute = |entry: &'static str,
                       spirv: fn() -> Option<&'static [u8]>,
                       dxil: fn() -> Option<&'static [u8]>,
                       metallib: fn() -> Option<&'static [u8]>,
                       name: &str,
                       pcsize: u32,
                       tg: [u32; 3]|
         -> anyhow::Result<Option<ComputePipeline>> {
            if !compute_supported {
                return Ok(None);
            }
            let cs = load_compute_shader(backend, spirv, dxil, metallib, name)?;
            Ok(Some(device.create_compute_pipeline(
                &ComputePipelineDesc {
                    compute_bytes: cs,
                    compute_entry: entry,
                    push_constant_size: pcsize,
                    bindless: true,
                    threads_per_group: tg,
                },
            )?))
        };
        let fill_pipeline = compute(
            "fillMain",
            dreamcoast_shader::volume_fill_cs_spirv,
            dreamcoast_shader::volume_fill_cs_dxil,
            dreamcoast_shader::volume_fill_cs_metallib,
            "volume_fill",
            32,
            [4, 4, 4],
        )?;
        let view_pipeline = compute(
            "viewMain",
            dreamcoast_shader::volume_view_cs_spirv,
            dreamcoast_shader::volume_view_cs_dxil,
            dreamcoast_shader::volume_view_cs_metallib,
            "volume_view",
            32,
            [8, 8, 1],
        )?;
        let bake_pipeline = compute(
            "bakeMain",
            dreamcoast_shader::sdf_bake_cs_spirv,
            dreamcoast_shader::sdf_bake_cs_dxil,
            dreamcoast_shader::sdf_bake_cs_metallib,
            "sdf_bake",
            64,
            [4, 4, 4],
        )?;
        let merge_pipeline = compute(
            "mergeMain",
            dreamcoast_shader::gdf_merge_cs_spirv,
            dreamcoast_shader::gdf_merge_cs_dxil,
            dreamcoast_shader::gdf_merge_cs_metallib,
            "gdf_merge",
            48,
            [4, 4, 4],
        )?;
        let trace_pipeline = compute(
            "csMain",
            dreamcoast_shader::gdf_trace_cs_spirv,
            dreamcoast_shader::gdf_trace_cs_dxil,
            dreamcoast_shader::gdf_trace_cs_metallib,
            "gdf_trace",
            128,
            [8, 8, 1],
        )?;
        let sdf_pipeline = compute(
            "csMain",
            dreamcoast_shader::sdf_trace_cs_spirv,
            dreamcoast_shader::sdf_trace_cs_dxil,
            dreamcoast_shader::sdf_trace_cs_metallib,
            "sdf_trace",
            112,
            [8, 8, 1],
        )?;

        // B2 bake mesh: a unit uv-sphere scaled to radius 0.3, centred at (0.5,0.5,0.5)
        // so its baked field matches the analytic centred sphere of the B1 smoke test.
        let (bake_vtx, bake_idx, bake_tri_count) = if compute_supported {
            let mut sphere = dreamcoast_asset::uv_sphere(48, 32);
            for v in &mut sphere.vertices {
                v.pos = [
                    v.pos[0] * 0.3 + 0.5,
                    v.pos[1] * 0.3 + 0.5,
                    v.pos[2] * 0.3 + 0.5,
                ];
            }
            let vb = vertex_bytes(&sphere);
            let ib = index_bytes(&sphere);
            let vsb = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: vb.len() as u64,
                    stride: 32,
                    indirect: false,
                },
                vb,
            )?;
            let isb = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: ib.len() as u64,
                    stride: 4,
                    indirect: false,
                },
                ib,
            )?;
            (Some(vsb), Some(isb), (sphere.indices.len() / 3) as u32)
        } else {
            (None, None, 0u32)
        };

        // B3 instance table: place instances of the baked per-mesh SDF into the unit-cube
        // GDF. `P11_GDF_INSTANCES=1` is a single whole-cube instance (reproduces the B2
        // bake exactly — the regression anchor); else three half-size spheres.
        let instances = if let Some(vol) = volume.as_ref() {
            let sampled = vol.sampled_index();
            let single = std::env::var_os("P11_GDF_INSTANCES")
                .map(|v| v == "1")
                .unwrap_or(false);
            let placements: &[([f32; 3], f32)] = if single {
                &[([0.0, 0.0, 0.0], 1.0)]
            } else {
                &[
                    ([0.05, 0.30, 0.25], 0.5),
                    ([0.45, 0.20, 0.25], 0.5),
                    ([0.25, 0.50, 0.25], 0.5),
                ]
            };
            let mut records = Vec::with_capacity(placements.len() * 32);
            for (origin, extent) in placements {
                let inv = 1.0 / extent;
                records.extend_from_slice(&origin[0].to_le_bytes());
                records.extend_from_slice(&origin[1].to_le_bytes());
                records.extend_from_slice(&origin[2].to_le_bytes());
                records.extend_from_slice(&extent.to_le_bytes()); // dist_scale
                records.extend_from_slice(&inv.to_le_bytes());
                records.extend_from_slice(&inv.to_le_bytes());
                records.extend_from_slice(&inv.to_le_bytes());
                records.extend_from_slice(&sampled.to_le_bytes());
            }
            let buf = device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: records.len() as u64,
                    stride: 32,
                    indirect: false,
                },
                &records,
            )?;
            Some((buf, placements.len() as u32))
        } else {
            None
        };

        Ok(Self {
            volume,
            gdf,
            fill_pipeline,
            view_pipeline,
            bake_pipeline,
            merge_pipeline,
            trace_pipeline,
            sdf_pipeline,
            bake_vtx,
            bake_idx,
            bake_tri_count,
            instances,
        })
    }

    // Feature-availability predicates (drive the UI checkboxes + toggle defaults).
    pub(crate) fn has_sdf_trace(&self) -> bool {
        self.sdf_pipeline.is_some()
    }
    pub(crate) fn has_volume(&self) -> bool {
        self.volume.is_some()
    }
    pub(crate) fn has_bake(&self) -> bool {
        self.bake_pipeline.is_some() && self.bake_vtx.is_some()
    }
    pub(crate) fn has_merge(&self) -> bool {
        self.merge_pipeline.is_some() && self.instances.is_some()
    }
    pub(crate) fn has_gdf_trace(&self) -> bool {
        self.trace_pipeline.is_some() && self.instances.is_some()
    }

    /// Stage A: sphere-trace the analytic SDF scene into a fresh storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_sdf_trace<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
    ) -> ResourceId {
        let pipe = self.sdf_pipeline.as_ref().expect("sdf trace pipeline");
        let out = graph.create_storage_image("sdf_out", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_trace",
                storage_writes: vec![out],
                reads: vec![],
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&sdf_trace_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    sun_intensity,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// B1: fill the volume with an analytic radial SDF, then view a Z slice.
    pub(crate) fn record_volume_test<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        cw: u32,
        ch: u32,
    ) -> ResourceId {
        let vol = self.volume.as_ref().expect("volume");
        let fillp = self.fill_pipeline.as_ref().expect("fill pipeline");
        let out = graph.create_storage_image("vol_out", HDR_FORMAT, extent);
        let vol_ext = graph.import_external("volume");
        let storage = vol.storage_index();
        let sampled = vol.sampled_index();
        graph.add_compute_pass(
            ComputePassInfo {
                name: "volume_fill",
                storage_writes: vec![vol_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_storage(vol);
                cmd.bind_compute_pipeline(fillp);
                cmd.push_constants_compute(&volume_push(
                    storage, sampled, VOLUME_DIM, 0, 0, 0, 0.0,
                ));
                let g = VOLUME_DIM.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
        self.view_volume(graph, vol, vol_ext, out, storage, sampled, cw, ch);
        out
    }

    /// B2: bake the per-mesh SDF into the volume (once), then view a slice.
    pub(crate) fn record_bake_view<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        build: bool,
    ) -> ResourceId {
        let vol = self.volume.as_ref().expect("volume");
        let out = graph.create_storage_image("bake_out", HDR_FORMAT, extent);
        let vol_ext = graph.import_external("volume");
        let storage = vol.storage_index();
        let sampled = vol.sampled_index();
        if build {
            self.record_bake(graph, vol_ext);
        }
        self.view_volume(graph, vol, vol_ext, out, storage, sampled, cw, ch);
        out
    }

    /// B3: build the GDF (bake + merge, once), then view a slice of it.
    pub(crate) fn record_gdf_view<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        build: bool,
    ) -> ResourceId {
        let gdf = self.gdf.as_ref().expect("gdf volume");
        let out = graph.create_storage_image("gdf_out", HDR_FORMAT, extent);
        let vol_ext = graph.import_external("volume");
        let gdf_ext = graph.import_external("gdf");
        if build {
            self.build_gdf(graph, vol_ext, gdf_ext);
        }
        let storage = gdf.storage_index();
        let sampled = gdf.sampled_index();
        self.view_volume(graph, gdf, gdf_ext, out, storage, sampled, cw, ch);
        out
    }

    /// B4: build the GDF (bake + merge, once), then SW ray-trace it from a fixed camera
    /// framing the unit-cube scene. `analytic` swaps in the reference field.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gdf_trace<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        flip_y: u32,
        vulkan: bool,
        analytic: bool,
        build: bool,
    ) -> ResourceId {
        let gdf = self.gdf.as_ref().expect("gdf volume");
        let tracep = self.trace_pipeline.as_ref().expect("gdf trace pipeline");
        let out = graph.create_storage_image("gdf_trace_out", HDR_FORMAT, extent);
        let vol_ext = graph.import_external("volume");
        let gdf_ext = graph.import_external("gdf");
        if build {
            self.build_gdf(graph, vol_ext, gdf_ext);
        }
        let gdf_sampled = gdf.sampled_index();
        // Fixed camera framing the unit-cube GDF scene (same Y-flip convention as the
        // orbit camera so VK/DX reconstruct identical world rays).
        let g_eye = Vec3::new(0.5, 0.65, 2.1);
        let g_view = Mat4::look_at_rh(g_eye, Vec3::new(0.5, 0.42, 0.5), Vec3::Y);
        let mut g_proj =
            Mat4::perspective_rh(35f32.to_radians(), cw as f32 / ch as f32, 0.02, 100.0);
        if vulkan {
            g_proj.y_axis.y *= -1.0;
        }
        let g_inv_vp = (g_proj * g_view).inverse().to_cols_array();
        let mode = u32::from(analytic);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_trace",
                storage_writes: vec![out],
                reads: vec![gdf_ext],
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(gdf);
                cmd.bind_compute_pipeline(tracep);
                cmd.push_constants_compute(&gdf_trace_push(
                    &g_inv_vp,
                    g_eye,
                    sun_dir,
                    sun_intensity,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    gdf_sampled,
                    mode,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// The B2 bake pass: brute-force per-mesh SDF into `volume`.
    fn record_bake<'a>(&'a self, graph: &mut RenderGraph<'a>, vol_ext: ResourceId) {
        let vol = self.volume.as_ref().expect("volume");
        let bakep = self.bake_pipeline.as_ref().expect("bake pipeline");
        let vtx = self.bake_vtx.as_ref().expect("bake vtx").storage_index();
        let idx = self.bake_idx.as_ref().expect("bake idx").storage_index();
        let storage = vol.storage_index();
        let tri_count = self.bake_tri_count;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_bake",
                storage_writes: vec![vol_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_storage(vol);
                cmd.bind_compute_pipeline(bakep);
                cmd.push_constants_compute(&sdf_bake_push(
                    storage, VOLUME_DIM, tri_count, vtx, idx,
                ));
                let g = VOLUME_DIM.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
    }

    /// B3 build: bake the per-mesh SDF, then merge its instances into the GDF.
    fn build_gdf<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        vol_ext: ResourceId,
        gdf_ext: ResourceId,
    ) {
        self.record_bake(graph, vol_ext);
        let vol = self.volume.as_ref().expect("volume");
        let gdf = self.gdf.as_ref().expect("gdf volume");
        let mergep = self.merge_pipeline.as_ref().expect("merge pipeline");
        let (insts, inst_count) = self.instances.as_ref().expect("instances");
        let gdf_storage = gdf.storage_index();
        let inst_table = insts.storage_index();
        let inst_n = *inst_count;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_merge",
                storage_writes: vec![gdf_ext],
                reads: vec![vol_ext],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol); // read the baked per-mesh SDF
                cmd.volume_to_storage(gdf); // write the GDF
                cmd.bind_compute_pipeline(mergep);
                cmd.push_constants_compute(&gdf_merge_push(
                    gdf_storage,
                    VOLUME_DIM,
                    inst_table,
                    inst_n,
                ));
                let g = VOLUME_DIM.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
    }

    /// The shared `volume_view` slice pass: trilinear-sample `vol` (read via `read_ext`)
    /// at Z = 0.5 into `out`.
    #[allow(clippy::too_many_arguments)]
    fn view_volume<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        vol: &'a Volume,
        read_ext: ResourceId,
        out: ResourceId,
        storage: u32,
        sampled: u32,
        cw: u32,
        ch: u32,
    ) {
        let viewp = self.view_pipeline.as_ref().expect("view pipeline");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "volume_view",
                storage_writes: vec![out],
                reads: vec![read_ext],
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                cmd.bind_compute_pipeline(viewp);
                cmd.push_constants_compute(&volume_push(
                    storage, sampled, VOLUME_DIM, out_index, cw, ch, 0.5,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
    }
}
