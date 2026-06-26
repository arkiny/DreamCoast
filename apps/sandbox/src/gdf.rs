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
use crate::push::{
    gdf_ao_push, gdf_gi_push, gdf_merge_push, gdf_trace_push, sdf_bake_push, sdf_trace_push,
    volume_push,
};

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
    ao_pipeline: Option<ComputePipeline>,    // C2 GDF ambient occlusion
    gi_pipeline: Option<ComputePipeline>,    // C3 GDF 1-bounce diffuse GI
    sdf_pipeline: Option<ComputePipeline>,   // Stage-A analytic sphere-march
    bake_vtx: Option<StorageBuffer>,
    bake_idx: Option<StorageBuffer>,
    bake_tri_count: u32,
    /// (table, instance count) for the merge; `None` without a volume.
    instances: Option<(StorageBuffer, u32)>,
    /// Stage C1: world-space GDF of the actual sample scene. The scene's object
    /// triangles are fused into one world-space soup and brute-force baked into this
    /// volume over the scene AABB (the per-mesh-SDF + clipmap merge for dynamic
    /// objects is a later refinement); the ground is added analytically at trace time.
    scene_gdf: Option<Volume>,
    scene_vtx: Option<StorageBuffer>,
    scene_idx: Option<StorageBuffer>,
    scene_tri_count: u32,
    /// World-space AABB the `scene_gdf` voxel grid maps to.
    scene_aabb_min: [f32; 3],
    scene_aabb_max: [f32; 3],
}

/// Scene-GDF volume edge length (cube). Coarser than `VOLUME_DIM`: the fused
/// brute-force bake is O(voxels·tris) over the whole scene, so a 48³ grid keeps the
/// one-time bake well under the GPU watchdog while staying ample for low-frequency GI.
const SCENE_DIM: u32 = 48;

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
            160,
            [8, 8, 1],
        )?;
        let ao_pipeline = compute(
            "csMain",
            dreamcoast_shader::gdf_ao_cs_spirv,
            dreamcoast_shader::gdf_ao_cs_dxil,
            dreamcoast_shader::gdf_ao_cs_metallib,
            "gdf_ao",
            144,
            [8, 8, 1],
        )?;
        let gi_pipeline = compute(
            "csMain",
            dreamcoast_shader::gdf_gi_cs_spirv,
            dreamcoast_shader::gdf_gi_cs_dxil,
            dreamcoast_shader::gdf_gi_cs_metallib,
            "gdf_gi",
            176,
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
            ao_pipeline,
            gi_pipeline,
            sdf_pipeline,
            bake_vtx,
            bake_idx,
            bake_tri_count,
            instances,
            scene_gdf: None,
            scene_vtx: None,
            scene_idx: None,
            scene_tri_count: 0,
            scene_aabb_min: [0.0; 3],
            scene_aabb_max: [0.0; 3],
        })
    }

    /// Stage C1: register the fused world-space scene geometry (a single triangle soup
    /// of all opaque scene objects, already transformed to world space) + its world
    /// AABB, and allocate the scene GDF volume. The bake itself runs once on the graph
    /// (`record_scene_build`). No-op when compute is unsupported.
    pub(crate) fn build_scene_sdf(
        &mut self,
        device: &Device,
        fused_vtx: &[u8],
        fused_idx: &[u8],
        tri_count: u32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> anyhow::Result<()> {
        if self.gdf.is_none() {
            return Ok(()); // compute unsupported (no volumes created)
        }
        self.scene_gdf = Some(device.create_volume(&VolumeDesc {
            width: SCENE_DIM,
            height: SCENE_DIM,
            depth: SCENE_DIM,
            format: Format::R32Float,
        })?);
        self.scene_vtx = Some(device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: fused_vtx.len() as u64,
                stride: 32,
                indirect: false,
            },
            fused_vtx,
        )?);
        self.scene_idx = Some(device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: fused_idx.len() as u64,
                stride: 4,
                indirect: false,
            },
            fused_idx,
        )?);
        self.scene_tri_count = tri_count;
        self.scene_aabb_min = aabb_min;
        self.scene_aabb_max = aabb_max;
        Ok(())
    }

    pub(crate) fn has_scene_sdf(&self) -> bool {
        self.scene_gdf.is_some()
    }

    /// Stage C1: build the world-space scene GDF (fused brute-force bake, once) then SW
    /// ray-trace it from the live camera — the validation that the world GDF matches the
    /// rasterized scene. Reuses the Stage-A/B4 trace machinery (now reading the world
    /// volume over the scene AABB, ground at y=0). Returns the output storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_scene_gdf<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        extent: Extent2D,
        eye: Vec3,
        inv_view_proj: [f32; 16],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
        build: bool,
    ) -> ResourceId {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let tracep = self.trace_pipeline.as_ref().expect("gdf trace pipeline");
        let out = graph.create_storage_image("scene_gdf_out", HDR_FORMAT, extent);
        let gdf_ext = graph.import_external("scene_gdf");
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        if build {
            self.record_scene_bake(graph, gdf_ext);
        }
        let sampled = vol.sampled_index();
        // Sample clamp = AABB diagonal: exceeds the field's true max distance so the
        // march never wrongly clamps (the fused bake fills every voxel — no sparse
        // sentinel), while keeping the empty-space step bounded.
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        graph.add_compute_pass(
            ComputePassInfo {
                name: "scene_gdf_trace",
                storage_writes: vec![out],
                reads: vec![gdf_ext],
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                cmd.bind_compute_pipeline(tracep);
                cmd.push_constants_compute(&gdf_trace_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    sun_intensity,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    sampled,
                    0, // mode 0: sample the baked GDF (no analytic reference)
                    aabb_min,
                    aabb_max,
                    0.0, // world ground plane at y = 0
                    diag,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// The fused scene bake pass: brute-force the world-space triangle soup into the
    /// scene GDF over the scene AABB (Stage C1).
    fn record_scene_bake<'a>(&'a self, graph: &mut RenderGraph<'a>, gdf_ext: ResourceId) {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let bakep = self.bake_pipeline.as_ref().expect("bake pipeline");
        let vtx = self.scene_vtx.as_ref().expect("scene vtx").storage_index();
        let idx = self.scene_idx.as_ref().expect("scene idx").storage_index();
        let storage = vol.storage_index();
        let tri_count = self.scene_tri_count;
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "scene_sdf_bake",
                storage_writes: vec![gdf_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_storage(vol);
                cmd.bind_compute_pipeline(bakep);
                cmd.push_constants_compute(&sdf_bake_push(
                    storage, SCENE_DIM, tri_count, vtx, idx, aabb_min, aabb_max,
                ));
                let g = SCENE_DIM.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
    }

    /// Stage C2: compute GDF ambient occlusion for the deferred render. Builds the
    /// fused scene GDF once (shared with the C1 trace via the caller's `build` flag),
    /// then a full-screen compute pass reconstructs each pixel's world surface point
    /// from the depth G-buffer, marches the scene GDF along the world normal, and
    /// writes an AO factor [0,1] into a storage image the lighting pass multiplies into
    /// its ambient term. World position comes from depth (not the object-space position
    /// MRT) so transformed objects line up with the world-space GDF. Returns the AO
    /// storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gdf_ao<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        cw: u32,
        ch: u32,
        flip_y: u32,
        build: bool,
    ) -> ResourceId {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let aop = self.ao_pipeline.as_ref().expect("gdf ao pipeline");
        let out = graph.create_storage_image("gdf_ao_out", HDR_FORMAT, extent);
        let gdf_ext = graph.import_external("scene_gdf");
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        if build {
            self.record_scene_bake(graph, gdf_ext);
        }
        let sampled = vol.sampled_index();
        // The scene extent sets the world-unit AO scale: a fraction of the AABB diagonal
        // for the sampling reach + a small surface bias, with the clamp = full diagonal
        // (exceeds the field's true max, so a query never wrongly clamps).
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let reach = diag * 0.07;
        let bias = diag * 0.004;
        let strength = 1.6;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_ao",
                storage_writes: vec![out],
                reads: vec![depth, normal, gdf_ext],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                cmd.bind_compute_pipeline(aop);
                cmd.push_constants_compute(&gdf_ao_push(
                    &inv_view_proj,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    aabb_min,
                    aabb_max,
                    0.0, // world ground plane at y = 0
                    diag,
                    reach,
                    strength,
                    bias,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// Stage C3: stochastic 1-bounce diffuse GI for the deferred render. Builds the
    /// fused scene GDF once (shared latch with C1/C2), then a full-screen compute pass
    /// reconstructs each pixel's world surface from depth, casts `spp` cosine-hemisphere
    /// rays into the scene GDF, shades the hits (constant albedo + sun + sky), and writes
    /// the mean incoming radiance (indirect irradiance) the lighting pass adds to the
    /// ambient term (× surface albedo × 1-metallic). Returns the GI storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gdf_gi<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
        spp: u32,
        frame: u32,
        build: bool,
    ) -> ResourceId {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let gip = self.gi_pipeline.as_ref().expect("gdf gi pipeline");
        let out = graph.create_storage_image("gdf_gi_out", HDR_FORMAT, extent);
        let gdf_ext = graph.import_external("scene_gdf");
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        if build {
            self.record_scene_bake(graph, gdf_ext);
        }
        let sampled = vol.sampled_index();
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let bias = diag * 0.004;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_gi",
                storage_writes: vec![out],
                reads: vec![depth, normal, gdf_ext],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                cmd.bind_compute_pipeline(gip);
                cmd.push_constants_compute(&gdf_gi_push(
                    &inv_view_proj,
                    sun_dir,
                    sun_intensity,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    spp,
                    frame,
                    aabb_min,
                    aabb_max,
                    0.0,  // world ground plane at y = 0
                    diag, // sample distance clamp
                    diag, // ray max distance (bounce reach = scene diagonal)
                    bias,
                    0.25, // sky fill radiance at the bounce hit
                    0.7, // constant hit albedo (no material in the GDF; bleed = surface cache later)
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    // Feature-availability predicates (drive the UI checkboxes + toggle defaults).
    pub(crate) fn has_gdf_gi(&self) -> bool {
        self.gi_pipeline.is_some() && self.scene_gdf.is_some()
    }
    pub(crate) fn has_gdf_ao(&self) -> bool {
        self.ao_pipeline.is_some() && self.scene_gdf.is_some()
    }
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
                    [0.0, 0.0, 0.0], // unit-cube GDF extent (B4)
                    [1.0, 1.0, 1.0],
                    0.2, // ground plane height
                    0.6, // sample clamp (> unit-cube field max)
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
                    storage,
                    VOLUME_DIM,
                    tri_count,
                    vtx,
                    idx,
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
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
