//! Phase 11 software ray tracing + global distance field — the distance field itself
//! and its construction / debug viz. Owns the volumes (per-mesh SDF + merged GDF + the
//! world scene GDF), the bake mesh + instance table, and the Stage-A/B pipelines
//! (analytic sdf trace, volume fill/view, SDF bake, GDF merge, GDF trace) + the Stage-C1
//! world-scene trace. The real-render *consumers* of the scene GDF moved out: AO / GI /
//! denoise live in `gi.rs` (`GiSystem`), reflections in `reflect.rs` (`ReflectSystem`);
//! both read the scene GDF via `scene_gdf_volume()` / `scene_aabb()` and rely on this
//! bundle's `record_scene_bake()` for the one-time fused-scene bake.
//!
//! Each `record_*` adds one feature's passes and returns the output storage image;
//! the frame loop keeps the mutual-exclusion gating (only one replaces the HDR) and
//! the build-once flags. All record methods borrow `&'a self` for the graph's lifetime,
//! like the other bundles.

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, CommandBuffer, ComputePipeline, ComputePipelineDesc, Device, Extent2D, Format,
    StorageBuffer, StorageBufferDesc, Volume, VolumeDesc,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::mesh::{index_bytes, vertex_bytes};
use crate::push::{
    cache_capture_push, cache_light_push, cache_view_push, cache_vis_push, gdf_merge_push,
    gdf_trace_push, sdf_albedo_bake_push, sdf_bake_push, sdf_trace_push, volume_push,
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
    /// Scene-GDF voxel-grid edge length (cube). Defaults to `SCENE_DIM`; `P11_GDF_DIM=<n>`
    /// (32..=256) overrides it. NOTE: raising it does NOT meaningfully improve reflection/GI
    /// parity on the sample scene (measured 48→128 ≈ 0.02–0.03/ch; the residual isn't voxel-
    /// resolution-bound — see docs/reflection-sdf-resolution.md). Kept as a coverage /
    /// RenderQuality knob for larger future scenes, not a parity lever. The bake is
    /// O(voxels·tris) one-time, so high values only lengthen the one-shot bake + volume memory.
    scene_dim: u32,
    /// World-space AABB the `scene_gdf` voxel grid maps to.
    scene_aabb_min: [f32; 3],
    scene_aabb_max: [f32; 3],
    /// Phase 12 M2: the scene GDF was seeded from a cooked `.dcasset` (CPU bake
    /// uploaded), so the one-time GPU `record_scene_bake` is skipped.
    scene_sdf_cooked: bool,
    /// Phase 12 M2-ext: the C8a albedo volumes were seeded from a cooked `.dcasset`,
    /// so the one-time GPU `record_scene_albedo_bake` is skipped.
    scene_albedo_cooked: bool,
    /// Stage C8a: per-voxel surface albedo (R/G/B as three R32Float volumes sharing the
    /// scene GDF's grid) + the parallel per-triangle albedo buffer the bake reads. Lets the
    /// C3 GI / C6 reflection re-light a hit with the real surface color instead of a constant.
    scene_albedo: Option<[Volume; 3]>,
    scene_tri_albedo: Option<StorageBuffer>,
    albedo_bake_pipeline: Option<ComputePipeline>,
    /// Stage C8b: Lumen-style mesh-card surface cache. `cards` holds the per-object AABB-face
    /// card records; `cache_pos` (hit pos + valid) and `cache_albedo` are the captured-once
    /// geometry atlases (flat storage buffers, one float4 / texel). C8b2 adds the re-lit
    /// radiance ping-pong; C8b3 looks it up at GI / reflection hits.
    cards: Option<StorageBuffer>,
    cache_pos: Option<StorageBuffer>,
    cache_albedo: Option<StorageBuffer>,
    num_cards: u32,
    cache_capture_pipeline: Option<ComputePipeline>,
    cache_view_pipeline: Option<ComputePipeline>,
    /// C8b2: ring of cached radiance (re-lit each frame; the gather reads last frame's for
    /// multibounce). `cache_frame` selects the read/write slot. The sync path uses a 2-slot
    /// ping-pong ([0]/[1], %2); the async-compute path uses a 3-slot ring (%3) so the slot the
    /// async queue writes is never one an in-flight graphics frame still reads (no WAR hazard).
    cache_radiance: [Option<StorageBuffer>; 3],
    cache_frame: u32,
    /// Async-compute relight: the surface-cache relight runs on the compute queue overlapping the
    /// graphics frame, and consumers read the PREVIOUS frame's radiance (1-frame latency). Opt-in.
    cache_async: bool,
    cache_light_pipeline: Option<ComputePipeline>,
    /// Stage D2b: per-card camera-frustum visibility (1 uint/card; 1 = on-screen) driving the
    /// relight budget, + the compute pipeline that fills it. `None` until the cache is built.
    card_vis: Option<StorageBuffer>,
    cache_vis_pipeline: Option<ComputePipeline>,
    /// QHD/UHD track: surface-cache atlas tile edge (texels/card side), runtime so content can use
    /// a smaller tile (cheaper, resolution-INDEPENDENT relight) while the gallery keeps `CARD_TILE`
    /// (byte-identical). Set in `build_surface_cache`; all cache shaders take it as a push param.
    card_tile: u32,
    /// Stage B (clipmap): the scene-GDF descriptor storage buffer the SW-RT shaders read
    /// to select a level — 48 bytes/level `{ aabb_min, aabb_max, sdf_idx, albedo_idx[3] }`,
    /// finest→coarsest. A single-level descriptor (the gallery) reproduces the legacy
    /// single-volume sampling byte-for-byte; `clipmap.slang` consumes it.
    clip_desc: Option<StorageBuffer>,
    clip_count: u32,
    /// Stage B3: the **finer** clipmap levels (camera-centered, higher resolution near the
    /// camera), each its own SDF + albedo volumes over its own AABB. The coarsest level
    /// stays in `scene_gdf`/`scene_albedo` (global coverage). Empty ⇒ single-level (gallery).
    clip_levels: Vec<ClipLevel>,
    /// Stage D: the analytic ground-plane height the SW-RT marches union with the GDF. The
    /// gallery's floor is *analytic* (y = 0, no floor geometry in the fuse); content scenes
    /// (Sponza) carry their floor as real geometry, so the analytic ground is disabled
    /// (a very low Y) to avoid a spurious second floor.
    scene_ground_y: f32,
}

/// A finer clipmap level: its own SDF (+ optional albedo) volumes over a sub-AABB of the
/// scene, sampled when a query point falls inside it (closer ⇒ higher resolution).
struct ClipLevel {
    sdf: Volume,
    albedo: Option<[Volume; 3]>,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

/// A baked finer clipmap level ready for upload: its world AABB + the cooked SDF bytes and
/// optional per-channel albedo bytes (the same little-endian f32 layout the coarsest uses).
pub(crate) struct ClipLevelData<'a> {
    pub(crate) aabb_min: [f32; 3],
    pub(crate) aabb_max: [f32; 3],
    pub(crate) sdf: &'a [u8],
    pub(crate) albedo: Option<[&'a [u8]; 3]>,
}

/// Surface-cache card atlas tile edge (texels per card side). 6 cards / object.
const CARD_TILE: u32 = 32;

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
                    uniform_buffer: false,
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
        // C8a per-voxel albedo bake (nearest-triangle color into 3 R32F volumes).
        let albedo_bake_pipeline = compute(
            "albedoBakeMain",
            dreamcoast_shader::sdf_albedo_bake_cs_spirv,
            dreamcoast_shader::sdf_albedo_bake_cs_dxil,
            dreamcoast_shader::sdf_albedo_bake_cs_metallib,
            "sdf_albedo_bake",
            64,
            [4, 4, 4],
        )?;
        // C8b1 surface-cache capture (GDF-traced geometry+albedo into card atlases) + viz.
        let cache_capture_pipeline = compute(
            "cacheMain",
            dreamcoast_shader::sdf_cache_capture_cs_spirv,
            dreamcoast_shader::sdf_cache_capture_cs_dxil,
            dreamcoast_shader::sdf_cache_capture_cs_metallib,
            "sdf_cache_capture",
            80,
            [64, 1, 1],
        )?;
        let cache_view_pipeline = compute(
            "viewMain",
            dreamcoast_shader::sdf_cache_view_cs_spirv,
            dreamcoast_shader::sdf_cache_view_cs_dxil,
            dreamcoast_shader::sdf_cache_view_cs_metallib,
            "sdf_cache_view",
            32,
            [8, 8, 1],
        )?;
        // C8b2 surface-cache lighting (re-light texels + multibounce gather).
        let cache_light_pipeline = compute(
            "lightMain",
            dreamcoast_shader::sdf_cache_light_cs_spirv,
            dreamcoast_shader::sdf_cache_light_cs_dxil,
            dreamcoast_shader::sdf_cache_light_cs_metallib,
            "sdf_cache_light",
            128,
            [64, 1, 1],
        )?;
        // Stage D2b: per-card camera-frustum visibility for the relight budget.
        let cache_vis_pipeline = compute(
            "csMain",
            dreamcoast_shader::sdf_cache_visibility_cs_spirv,
            dreamcoast_shader::sdf_cache_visibility_cs_dxil,
            dreamcoast_shader::sdf_cache_visibility_cs_metallib,
            "sdf_cache_visibility",
            112,
            [64, 1, 1],
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
            scene_gdf: None,
            scene_vtx: None,
            scene_idx: None,
            scene_tri_count: 0,
            // `P11_GDF_DIM=<n>` overrides the scene-GDF resolution (clamped 32..=256); the
            // brute-force bake is O(voxels·tris) one-time, so very high values lengthen the
            // one-shot bake. Default keeps the watchdog-safe SCENE_DIM.
            scene_dim: std::env::var("P11_GDF_DIM")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|d| d.clamp(32, 256))
                .unwrap_or(SCENE_DIM),
            scene_aabb_min: [0.0; 3],
            scene_aabb_max: [0.0; 3],
            scene_sdf_cooked: false,
            scene_albedo_cooked: false,
            scene_albedo: None,
            scene_tri_albedo: None,
            albedo_bake_pipeline,
            cards: None,
            cache_pos: None,
            cache_albedo: None,
            num_cards: 0,
            cache_capture_pipeline,
            cache_view_pipeline,
            cache_radiance: [None, None, None],
            cache_frame: 0,
            cache_async: false,
            cache_light_pipeline,
            card_vis: None,
            cache_vis_pipeline,
            card_tile: CARD_TILE,
            clip_desc: None,
            clip_count: 0,
            clip_levels: Vec::new(),
            scene_ground_y: 0.0,
        })
    }

    /// Stage D: set the analytic ground-plane height (0 for the gallery; a disabling low Y
    /// for content scenes whose floor is real geometry). Read by the scene-GDF trace today;
    /// the lighting-path passes pick it up at the Stage D lighting flip.
    pub(crate) fn set_scene_ground_y(&mut self, y: f32) {
        self.scene_ground_y = y;
    }

    /// Stage C1: register the fused world-space scene geometry (a single triangle soup
    /// of all opaque scene objects, already transformed to world space) + its world
    /// AABB, and allocate the scene GDF volume. The bake itself runs once on the graph
    /// (`record_scene_build`). No-op when compute is unsupported.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_scene_sdf(
        &mut self,
        device: &Device,
        fused_vtx: &[u8],
        fused_idx: &[u8],
        tri_albedo: &[u8],
        tri_count: u32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        cooked_sdf: Option<&[u8]>,
        cooked_albedo: Option<[&[u8]; 3]>,
    ) -> anyhow::Result<()> {
        if self.gdf.is_none() {
            return Ok(()); // compute unsupported (no volumes created)
        }
        let dim = self.scene_dim;
        let vol_desc = VolumeDesc {
            width: dim,
            height: dim,
            depth: dim,
            format: Format::R32Float,
        };
        // Phase 12 M2: when a cooked SDF is supplied, upload it straight into the
        // volume (deterministic CPU bake) and skip the one-time GPU bake pass; else
        // allocate an empty volume the GPU `record_scene_bake` fills.
        self.scene_gdf = Some(match cooked_sdf {
            Some(bytes) => device.create_volume_init(&vol_desc, bytes)?,
            None => device.create_volume(&vol_desc)?,
        });
        self.scene_sdf_cooked = cooked_sdf.is_some();
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
        // C8a: three R32F color volumes (R/G/B) over the same grid. Only when the
        // albedo bake exists. Phase 12 M2-ext: a cooked albedo (CPU bake) uploads
        // straight into the volumes and skips the GPU bake + the per-triangle buffer;
        // otherwise allocate empty volumes the GPU `record_scene_albedo_bake` fills.
        if self.albedo_bake_pipeline.is_some() {
            let vd = VolumeDesc {
                width: dim,
                height: dim,
                depth: dim,
                format: Format::R32Float,
            };
            if let Some(channels) = cooked_albedo {
                self.scene_albedo = Some([
                    device.create_volume_init(&vd, channels[0])?,
                    device.create_volume_init(&vd, channels[1])?,
                    device.create_volume_init(&vd, channels[2])?,
                ]);
                self.scene_albedo_cooked = true;
            } else {
                self.scene_albedo = Some([
                    device.create_volume(&vd)?,
                    device.create_volume(&vd)?,
                    device.create_volume(&vd)?,
                ]);
                self.scene_tri_albedo = Some(device.create_storage_buffer_init(
                    &StorageBufferDesc {
                        size: tri_albedo.len() as u64,
                        stride: 12,
                        indirect: false,
                    },
                    tri_albedo,
                )?);
            }
        }
        // Stage B (clipmap): build the descriptor the SW-RT shaders sample through. For now
        // this is a single level (the coarsest, = the volume just built) so the gallery
        // reads exactly the legacy field; the finer camera-centered levels are prepended by
        // `set_clip_levels` (B3 / Stage D) for content scenes.
        self.build_clip_descriptor(device)?;
        Ok(())
    }

    /// (Re)build the clipmap descriptor storage buffer from the current scene levels. Each
    /// 48-byte record is `{ float4 aabb_min, float4 aabb_max, uint4(sdf_idx, alb_r, alb_g,
    /// alb_b) }`, ordered **finest→coarsest**: the B3 finer levels first, then the coarsest
    /// (`scene_gdf` over `scene_aabb`). A scene with no finer levels (gallery) gets one
    /// record = the legacy single volume.
    fn build_clip_descriptor(&mut self, device: &Device) -> anyhow::Result<()> {
        if self.scene_gdf.is_none() {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(48 * (self.clip_levels.len() + 1));
        let push_level = |mn: [f32; 3], mx: [f32; 3], idx: [u32; 4], bytes: &mut Vec<u8>| {
            for v in [mn[0], mn[1], mn[2], 0.0] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            for v in [mx[0], mx[1], mx[2], 0.0] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            for u in idx {
                bytes.extend_from_slice(&u.to_le_bytes());
            }
        };
        let alb_idx = |a: Option<&[Volume; 3]>| -> [u32; 3] {
            match a {
                Some(v) => [
                    v[0].sampled_index(),
                    v[1].sampled_index(),
                    v[2].sampled_index(),
                ],
                None => [u32::MAX; 3],
            }
        };
        // Finer levels first (finest = level 0 of the plan).
        for lv in &self.clip_levels {
            let a = alb_idx(lv.albedo.as_ref());
            push_level(
                lv.aabb_min,
                lv.aabb_max,
                [lv.sdf.sampled_index(), a[0], a[1], a[2]],
                &mut bytes,
            );
        }
        // Coarsest = the scene volume (global coverage).
        let sdf = self.scene_gdf.as_ref().unwrap();
        let a = alb_idx(self.scene_albedo.as_ref());
        push_level(
            self.scene_aabb_min,
            self.scene_aabb_max,
            [sdf.sampled_index(), a[0], a[1], a[2]],
            &mut bytes,
        );

        self.clip_count = self.clip_levels.len() as u32 + 1;
        self.clip_desc = Some(device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: bytes.len() as u64,
                stride: 48,
                indirect: false,
            },
            &bytes,
        )?);
        Ok(())
    }

    /// Stage B3: install the finer (camera-centered) clipmap levels from their cooked SDF +
    /// albedo bytes, ordered **finest first**, and rebuild the descriptor. Each becomes its
    /// own GPU volume over its sub-AABB; the coarsest stays in `scene_gdf`. No-op without a
    /// scene GDF / compute support.
    pub(crate) fn set_clip_levels(
        &mut self,
        device: &Device,
        levels: &[ClipLevelData],
    ) -> anyhow::Result<()> {
        if self.scene_gdf.is_none() {
            return Ok(());
        }
        let dim = self.scene_dim;
        let vd = VolumeDesc {
            width: dim,
            height: dim,
            depth: dim,
            format: Format::R32Float,
        };
        let mut out = Vec::with_capacity(levels.len());
        for lv in levels {
            let albedo = match lv.albedo {
                Some(ch) => Some([
                    device.create_volume_init(&vd, ch[0])?,
                    device.create_volume_init(&vd, ch[1])?,
                    device.create_volume_init(&vd, ch[2])?,
                ]),
                None => None,
            };
            out.push(ClipLevel {
                sdf: device.create_volume_init(&vd, lv.sdf)?,
                albedo,
                aabb_min: lv.aabb_min,
                aabb_max: lv.aabb_max,
            });
        }
        self.clip_levels = out;
        self.build_clip_descriptor(device)?;
        Ok(())
    }

    /// Stage B3: the finer-level SDF + albedo volumes that a consumer must transition to
    /// the sampled state before sampling the clipmap (the coarsest is handled separately).
    /// Empty for a single-level (gallery) clipmap.
    pub(crate) fn clip_level_volumes(&self) -> Vec<&Volume> {
        let mut v = Vec::new();
        for lv in &self.clip_levels {
            v.push(&lv.sdf);
            if let Some(a) = &lv.albedo {
                v.extend(a.iter());
            }
        }
        v
    }

    /// Stage B: the clipmap descriptor `(storage-buffer index, level count)` the SW-RT
    /// shaders sample the scene field through, or `None` until a scene GDF is built.
    pub(crate) fn clip_descriptor(&self) -> Option<(u32, u32)> {
        self.clip_desc
            .as_ref()
            .map(|b| (b.storage_index(), self.clip_count))
    }

    pub(crate) fn has_scene_sdf(&self) -> bool {
        self.scene_gdf.is_some()
    }

    /// Scene-GDF voxel-grid edge length — the dim a cooked SDF must be baked at to
    /// match the volume `build_scene_sdf` allocates.
    pub(crate) fn scene_dim(&self) -> u32 {
        self.scene_dim
    }

    /// Whether the scene GDF was seeded from a cooked `.dcasset` (so the GPU bake is
    /// already done and `record_scene_bake` should be skipped).
    pub(crate) fn scene_sdf_is_cooked(&self) -> bool {
        self.scene_sdf_cooked
    }

    /// Whether the C8a albedo volumes were seeded from a cooked `.dcasset` (so the GPU
    /// `record_scene_albedo_bake` should be skipped).
    pub(crate) fn scene_albedo_is_cooked(&self) -> bool {
        self.scene_albedo_cooked
    }

    pub(crate) fn has_scene_albedo(&self) -> bool {
        self.scene_albedo.is_some()
    }

    /// C8a: the three per-voxel albedo channel volumes (R/G/B), `None` until built / when
    /// compute is unsupported. Consumers (`GiSystem`, `ReflectSystem`) sample them at a hit.
    pub(crate) fn scene_albedo(&self) -> Option<&[Volume; 3]> {
        self.scene_albedo.as_ref()
    }

    /// Record the one-time per-voxel albedo bake into the 3 color volumes, writing the
    /// imported `albedo_ext` handle (the caller imports it once + shares it with every
    /// re-lighting consumer so the graph orders bake -> reads, like the distance field).
    pub(crate) fn record_scene_albedo_bake<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        albedo_ext: ResourceId,
    ) {
        let vols = self.scene_albedo.as_ref().expect("scene albedo volumes");
        let bakep = self
            .albedo_bake_pipeline
            .as_ref()
            .expect("albedo bake pipeline");
        let vtx = self.scene_vtx.as_ref().expect("scene vtx").storage_index();
        let idx = self.scene_idx.as_ref().expect("scene idx").storage_index();
        let tri_albedo = self
            .scene_tri_albedo
            .as_ref()
            .expect("scene tri albedo")
            .storage_index();
        let tri_count = self.scene_tri_count;
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        let dim = self.scene_dim;
        let storage = [
            vols[0].storage_index(),
            vols[1].storage_index(),
            vols[2].storage_index(),
        ];
        graph.add_compute_pass(
            ComputePassInfo {
                name: "scene_albedo_bake",
                storage_writes: vec![albedo_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                for v in vols.iter() {
                    cmd.volume_to_storage(v);
                }
                cmd.bind_compute_pipeline(bakep);
                cmd.push_constants_compute(&sdf_albedo_bake_push(
                    storage[0], storage[1], storage[2], dim, tri_count, vtx, idx, tri_albedo,
                    aabb_min, aabb_max,
                ));
                let g = dim.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
    }

    /// Stage C8b1: register the per-object mesh cards + allocate the surface-cache atlas
    /// buffers (captured geometry: `cache_pos` = hit pos + valid, `cache_albedo`). `cards`
    /// is the host-built card-record byte buffer (64 B / card). No-op without the capture
    /// pipeline. The card tile edge is `CARD_TILE`; the atlas is `num_cards * tile²` texels.
    pub(crate) fn build_surface_cache(
        &mut self,
        device: &Device,
        cards: &[u8],
        num_cards: u32,
        tile: u32,
    ) -> anyhow::Result<()> {
        if self.cache_capture_pipeline.is_none() || num_cards == 0 {
            return Ok(());
        }
        self.card_tile = tile.clamp(4, 64);
        self.cards = Some(device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: cards.len() as u64,
                stride: 16,
                indirect: false,
            },
            cards,
        )?);
        let texels = (num_cards * self.card_tile * self.card_tile) as u64;
        let make = || -> anyhow::Result<Option<StorageBuffer>> {
            Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                size: texels * 16,
                stride: 16,
                indirect: false,
            })?))
        };
        self.cache_pos = make()?;
        self.cache_albedo = make()?;
        self.cache_radiance = [make()?, make()?, make()?];
        // Stage D2b: 1 uint / card visibility flag (filled each frame by record_cache_visibility).
        self.card_vis = Some(device.create_storage_buffer(&StorageBufferDesc {
            size: (num_cards as u64) * 4,
            stride: 4,
            indirect: false,
        })?);
        self.num_cards = num_cards;
        self.cache_frame = 0;
        Ok(())
    }

    /// Stage D2b: the per-card visibility buffer's bindless storage index (for the relight pass),
    /// or `None` when the cache / visibility pipeline isn't built.
    pub(crate) fn card_vis_index(&self) -> Option<u32> {
        Some(self.card_vis.as_ref()?.storage_index())
    }
    pub(crate) fn has_cache_visibility(&self) -> bool {
        self.cache_vis_pipeline.is_some() && self.card_vis.is_some()
    }

    pub(crate) fn has_surface_cache(&self) -> bool {
        self.cards.is_some()
    }

    pub(crate) fn has_cache_lighting(&self) -> bool {
        self.cache_light_pipeline.is_some()
    }

    /// Opt into running the surface-cache relight on the async-compute queue (consumers read the
    /// previous frame's radiance). Switches the radiance ring from a 2-slot ping-pong to 3 slots.
    pub(crate) fn set_cache_async(&mut self, on: bool) {
        self.cache_async = on;
    }

    /// Bump the cache radiance ping-pong (end-of-frame), so next frame's gather + the
    /// consumers read the buffer this frame lit.
    pub(crate) fn advance_cache(&mut self) {
        self.cache_frame = self.cache_frame.saturating_add(1);
    }

    /// C8b2/3: the bindless indices a cache *reader* needs — cards, captured positions, the
    /// radiance buffer this frame lit (write slot), plus card count + tile. `None` until the
    /// cache + radiance buffers exist.
    pub(crate) fn surface_cache_read(&self) -> Option<(u32, u32, u32, u32, u32)> {
        let cards = self.cards.as_ref()?.storage_index();
        let pos = self.cache_pos.as_ref()?.storage_index();
        // Sync: consumers read the slot this frame's (graph) relight wrote. Async: the relight runs
        // on the compute queue overlapping this frame, so consumers read the PREVIOUS frame's slot
        // (the most recent completed relight; the cross-queue semaphore guards it).
        let slot = if self.cache_async {
            ((self.cache_frame + 2) % 3) as usize
        } else {
            (self.cache_frame % 2) as usize
        };
        let rad = self.cache_radiance[slot].as_ref()?.storage_index();
        Some((cards, pos, rad, self.num_cards, self.card_tile))
    }

    /// Stage D2b: fill the per-card visibility buffer for this frame's camera (frustum planes,
    /// Y-flip-free so DX≡VK). Records the compute pass and returns the visibility graph handle to
    /// pass into `record_cache_light` (for the read barrier). `None` without the cache / pipeline.
    pub(crate) fn record_cache_visibility<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        planes: [[f32; 4]; 6],
    ) -> Option<ResourceId> {
        let pipe = self.cache_vis_pipeline.as_ref()?;
        let cards = self.cards.as_ref()?.storage_index();
        let out = self.card_vis.as_ref()?.storage_index();
        let num_cards = self.num_cards;
        let vis_ext = graph.import_external("card_vis");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_cache_visibility",
                storage_writes: vec![vis_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&cache_vis_push(&planes, cards, out, num_cards));
                cmd.dispatch(num_cards.div_ceil(64), 1, 1);
                Ok(())
            },
        );
        Some(vis_ext)
    }

    /// C8b2: re-light the surface cache this frame — direct sun (GDF soft-shadow) + sky +
    /// an indirect gather that reads last frame's radiance (multibounce). Reads the captured
    /// geometry (`scene_cache_ext`) + the scene GDF; writes the `lit_ext` handle (the freshly
    /// lit radiance) the consumers / viz read. `reset` ignores the temporal history.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_cache_light<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf_ext: ResourceId,
        scene_cache_ext: ResourceId,
        lit_ext: ResourceId,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        spp: u32,
        frame: u32,
        reset: bool,
        relight_period: u32,
        card_vis_ext: Option<ResourceId>,
        alpha: f32,
    ) {
        // Stage D2b: feed the per-card visibility buffer index to the shader (sentinel = off =
        // uniform period). When present, declare it as a read so the graph barriers the relight
        // after this frame's visibility pass.
        let card_vis_index = match card_vis_ext {
            Some(_) => self.card_vis_index().unwrap_or(u32::MAX),
            None => u32::MAX,
        };
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let pipe = self
            .cache_light_pipeline
            .as_ref()
            .expect("cache light pipeline");
        let cards = self.cards.as_ref().expect("cards").storage_index();
        let cpos = self.cache_pos.as_ref().expect("cache pos").storage_index();
        let calb = self
            .cache_albedo
            .as_ref()
            .expect("cache albedo")
            .storage_index();
        let read = ((self.cache_frame + 1) % 2) as usize;
        let write = (self.cache_frame % 2) as usize;
        let rad_read = self.cache_radiance[read]
            .as_ref()
            .expect("rad")
            .storage_index();
        let rad_write = self.cache_radiance[write]
            .as_ref()
            .expect("rad")
            .storage_index();
        let num_cards = self.num_cards;
        let num_texels = num_cards * self.card_tile * self.card_tile;
        let sampled = vol.sampled_index();
        let clip = self.clip_descriptor().unwrap_or((0, 1));
        let clip_vols = self.clip_level_volumes();
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let mut reads = vec![scene_gdf_ext, scene_cache_ext];
        if let Some(vis) = card_vis_ext {
            reads.push(vis); // barrier the relight after this frame's visibility pass
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_cache_light",
                storage_writes: vec![lit_ext],
                reads,
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                for v in &clip_vols {
                    cmd.volume_to_sampled(v);
                }
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&cache_light_push(
                    cards,
                    cpos,
                    calb,
                    rad_read,
                    rad_write,
                    sampled,
                    num_cards,
                    self.card_tile,
                    num_texels,
                    spp,
                    frame,
                    u32::from(reset),
                    sun_dir,
                    sun_intensity,
                    aabb_min,
                    0.0,
                    aabb_max,
                    diag,
                    0.25,                            // sky fill irradiance
                    if reset { 1.0 } else { alpha }, // temporal alpha (D3: period-aware)
                    diag * 0.01,                     // surface bias
                    diag,                            // gather ray max distance
                    clip.0,
                    clip.1,
                    relight_period.max(1),
                    card_vis_index,
                ));
                cmd.dispatch(num_texels.div_ceil(64), 1, 1);
                Ok(())
            },
        );
    }

    /// Async-compute relight: record the per-card visibility and surface-cache relight directly
    /// onto a compute command buffer (not the render graph) so it runs on the async-compute queue,
    /// overlapping the graphics frame. Writes the current ring slot (`cache_frame % 3`); both the
    /// gather and the graphics consumers read the previous slot `((cache_frame + 2) % 3)`. The
    /// scene GDF, albedo and clip volumes are already in SHADER_READ_ONLY (baked once) so no layout
    /// transition is needed; the radiance/cards/positions storage buffers are CONCURRENT-shared.
    /// Manual barriers order the visibility write before the relight read, and the relight write
    /// before the queue signal.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_cache_async(
        &self,
        cmd: &CommandBuffer,
        planes: [[f32; 4]; 6],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        spp: u32,
        frame: u32,
        reset: bool,
        relight_period: u32,
        alpha: f32,
        feedback: bool,
    ) {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let pipe = self
            .cache_light_pipeline
            .as_ref()
            .expect("cache light pipeline");
        let cards = self.cards.as_ref().expect("cards").storage_index();
        let cpos = self.cache_pos.as_ref().expect("cache pos").storage_index();
        let calb = self
            .cache_albedo
            .as_ref()
            .expect("cache albedo")
            .storage_index();
        // 3-slot ring: write this frame's slot, gather + consumers read the previous slot.
        let read = ((self.cache_frame + 2) % 3) as usize;
        let write = (self.cache_frame % 3) as usize;
        let rad_read_buf = self.cache_radiance[read].as_ref().expect("rad read");
        let rad_write_buf = self.cache_radiance[write].as_ref().expect("rad write");
        let rad_read = rad_read_buf.storage_index();
        let rad_write = rad_write_buf.storage_index();
        let num_cards = self.num_cards;
        let num_texels = num_cards * self.card_tile * self.card_tile;
        let sampled = vol.sampled_index();
        let clip = self.clip_descriptor().unwrap_or((0, 1));
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };

        // Per-card camera visibility (this frame), read by the relight budget. Same as the graph
        // path's record_cache_visibility, recorded inline on the compute queue.
        let card_vis_index = match (
            feedback,
            self.cache_vis_pipeline.as_ref(),
            self.card_vis.as_ref(),
        ) {
            (true, Some(vis_pipe), Some(vis_buf)) => {
                let vis_out = vis_buf.storage_index();
                cmd.bind_compute_pipeline(vis_pipe);
                cmd.push_constants_compute(&cache_vis_push(&planes, cards, vis_out, num_cards));
                cmd.dispatch(num_cards.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier_compute(vis_buf); // visibility write -> relight read
                vis_out
            }
            _ => u32::MAX,
        };

        // The gather reads the previous slot, written by the previous async submit on this queue —
        // order that write before this read.
        cmd.storage_buffer_barrier_compute(rad_read_buf);
        cmd.bind_compute_pipeline(pipe);
        cmd.push_constants_compute(&cache_light_push(
            cards,
            cpos,
            calb,
            rad_read,
            rad_write,
            sampled,
            num_cards,
            self.card_tile,
            num_texels,
            spp,
            frame,
            u32::from(reset),
            sun_dir,
            sun_intensity,
            aabb_min,
            0.0,
            aabb_max,
            diag,
            0.25,
            if reset { 1.0 } else { alpha },
            diag * 0.01,
            diag,
            clip.0,
            clip.1,
            relight_period.max(1),
            card_vis_index,
        ));
        cmd.dispatch(num_texels.div_ceil(64), 1, 1);
        // Make the relight write available before the queue signals (graphics reads it next frame).
        cmd.storage_buffer_barrier_compute(rad_write_buf);
    }

    /// C8b1: capture the surface cache once — per card texel, sphere-trace the scene GDF
    /// inward from the card plane and store the hit's world position + albedo. Reads the
    /// scene GDF (and the C8a albedo volumes when `albedo_ext` is `Some`, for the captured
    /// color); writes the `cache_ext` handle (the caller imports it once + shares it).
    pub(crate) fn record_cache_capture<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf_ext: ResourceId,
        albedo_ext: Option<ResourceId>,
        cache_ext: ResourceId,
    ) {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let pipe = self
            .cache_capture_pipeline
            .as_ref()
            .expect("cache capture pipeline");
        let cards = self.cards.as_ref().expect("cards").storage_index();
        let cpos = self.cache_pos.as_ref().expect("cache pos").storage_index();
        let calb = self
            .cache_albedo
            .as_ref()
            .expect("cache albedo")
            .storage_index();
        let albedo = albedo_ext.and(self.scene_albedo.as_ref());
        let num_cards = self.num_cards;
        let num_texels = num_cards * self.card_tile * self.card_tile;
        let sampled = vol.sampled_index();
        let clip = self.clip_descriptor().unwrap_or((0, 1));
        let clip_vols = self.clip_level_volumes();
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let mut reads = vec![scene_gdf_ext];
        if let Some(ext) = albedo_ext {
            reads.push(ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_cache_capture",
                storage_writes: vec![cache_ext],
                reads,
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(vol);
                for v in &clip_vols {
                    cmd.volume_to_sampled(v);
                }
                let albedo_rgb = if let Some(vols) = albedo {
                    for v in vols.iter() {
                        cmd.volume_to_sampled(v);
                    }
                    [
                        vols[0].sampled_index(),
                        vols[1].sampled_index(),
                        vols[2].sampled_index(),
                    ]
                } else {
                    [u32::MAX; 3]
                };
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&cache_capture_push(
                    cards,
                    cpos,
                    calb,
                    sampled,
                    num_cards,
                    self.card_tile,
                    num_texels,
                    albedo_rgb,
                    clip.0,
                    clip.1,
                    aabb_min,
                    aabb_max,
                    diag,
                ));
                cmd.dispatch(num_texels.div_ceil(64), 1, 1);
                Ok(())
            },
        );
    }

    /// C8b1/2: tile a surface-cache atlas buffer across the screen (validation viz). `src`
    /// is the buffer shown — the captured albedo (C8b1) or the lit radiance (C8b2); `src_ext`
    /// orders the viz after whichever pass produced it.
    pub(crate) fn record_cache_view<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        src: ResourceId,
        src_index: u32,
        extent: Extent2D,
        cw: u32,
        ch: u32,
    ) -> ResourceId {
        let pipe = self
            .cache_view_pipeline
            .as_ref()
            .expect("cache view pipeline");
        let out = graph.create_storage_image("cache_view_out", HDR_FORMAT, extent);
        let cpos = self.cache_pos.as_ref().expect("cache pos").storage_index();
        let num_cards = self.num_cards;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "sdf_cache_view",
                storage_writes: vec![out],
                reads: vec![src],
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&cache_view_push(
                    cpos,
                    src_index,
                    out_index,
                    num_cards,
                    self.card_tile,
                    cw,
                    ch,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// The captured-albedo buffer index (C8b1 viz source).
    pub(crate) fn cache_albedo_index(&self) -> u32 {
        self.cache_albedo
            .as_ref()
            .map(|b| b.storage_index())
            .unwrap_or(u32::MAX)
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
        // Stage B: sample the scene field through the clipmap descriptor (1 level today).
        let clip = self.clip_descriptor().unwrap_or((0, 1));
        let clip_vols = self.clip_level_volumes();
        let ground_y = self.scene_ground_y;
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
                for v in &clip_vols {
                    cmd.volume_to_sampled(v);
                }
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
                    clip.0,
                    clip.1,
                    aabb_min,
                    aabb_max,
                    ground_y, // analytic ground (gallery y=0; disabled for content)
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
    /// Borrow the world scene GDF volume (consumers — `GiSystem`, the reflection track —
    /// sample it; the volume itself stays owned here). `None` without compute support.
    pub(crate) fn scene_gdf_volume(&self) -> Option<&Volume> {
        self.scene_gdf.as_ref()
    }

    /// The world-space AABB the scene GDF voxel grid maps to (consumers scale their
    /// world-unit constants by its diagonal).
    pub(crate) fn scene_aabb(&self) -> ([f32; 3], [f32; 3]) {
        (self.scene_aabb_min, self.scene_aabb_max)
    }

    /// Record the one-time fused-scene bake into the scene GDF, writing the imported
    /// `gdf_ext` handle. The caller imports `scene_gdf` once and shares the handle with
    /// every consumer (AO / GI / reflection) so the graph orders bake → reads.
    pub(crate) fn record_scene_bake<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gdf_ext: ResourceId,
    ) {
        let vol = self.scene_gdf.as_ref().expect("scene gdf volume");
        let bakep = self.bake_pipeline.as_ref().expect("bake pipeline");
        let vtx = self.scene_vtx.as_ref().expect("scene vtx").storage_index();
        let idx = self.scene_idx.as_ref().expect("scene idx").storage_index();
        let storage = vol.storage_index();
        let tri_count = self.scene_tri_count;
        let aabb_min = self.scene_aabb_min;
        let aabb_max = self.scene_aabb_max;
        let dim = self.scene_dim;
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
                    storage, dim, tri_count, vtx, idx, aabb_min, aabb_max,
                ));
                let g = dim.div_ceil(4);
                cmd.dispatch(g, g, g);
                Ok(())
            },
        );
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
                    0,               // clip_desc unused (legacy single-volume path)
                    0,               // clip_count 0 ⇒ sample gdf_sampled directly (B4 unit-cube)
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
