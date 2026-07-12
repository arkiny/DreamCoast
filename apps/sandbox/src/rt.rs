//! Phase 8 hardware ray tracing — extracted from `run()` as R4 of the render-loop
//! decomposition (see docs/refactor-sandbox.md). Owns the three RT pipelines (M3
//! inline ray-query trace, M4 inline path tracer, M5 RT pipeline + SBT), the two
//! built scenes (the sample scene's BLAS/TLAS + per-instance geometry table, and
//! the Cornell-box scene), and the path tracer's progressive-accumulation state.
//!
//! Per-frame mutable state (accum buffer (re)allocation, TLAS rebind on scene
//! switch, accum-frame / key resets) lives in `prepare(&mut self, …)`, run before
//! the graph is built so its fallible allocations stay off the graph's borrow path.
//! The graph passes are added by `record_path` / `record_trace` (`&'a self` tied to
//! the graph's lifetime, like the other bundles); the frame loop keeps the toggle
//! gating and bumps the accumulation counter at end-of-frame via `advance_accum`.

use dreamcoast_asset::MeshData;
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlasGeometry, Buffer, ComputePipeline, ComputePipelineDesc, Device,
    RaytracingPipeline, RaytracingPipelineDesc, RaytracingScene, RtGeometry, StorageBuffer,
    StorageBufferDesc, TlasInstance,
};
use tracing::info;

use std::collections::HashMap;
use std::rc::Rc;

use dreamcoast_scene::{Drawable, MeshHandle};

use crate::SceneObject;
use crate::app::load_compute_shader;
use crate::mesh::{PtMaterial, build_pt_instance_table};
use crate::push::{mat4_to_3x4, rt_path_push, rt_trace_push};
use crate::registry::{GpuMesh, MeshRegistry};

pub(crate) struct RtSystem {
    /// M3 inline ray-query trace (compute + `RayQuery`).
    trace_pipeline: Option<ComputePipeline>,
    /// M4 inline path tracer (diffuse GI bounce loop + progressive accumulation).
    path_pipeline: Option<ComputePipeline>,
    /// M5 the same path tracer via the hardware RT *pipeline* (raygen/miss/chit + SBT).
    pt_pipeline: Option<RaytracingPipeline>,
    /// Sample scene: BLAS-per-mesh + one TLAS (kept alive while bound).
    scene: Option<RaytracingScene>,
    /// Content scene (Phase 16 HWRT hybrid): per-unique-mesh BLAS + one world TLAS built from the
    /// ECS draw list, for the hardware-ray-traced reflection/GI trace. Opt-in (`P_HWRT`); the
    /// gallery uses `scene` above instead. Kept alive while its TLAS is bound.
    content_scene: Option<RaytracingScene>,
    /// Phase 16 E (Hit Lighting): consolidated content geometry + material table — ONE shared vertex
    /// buffer, ONE index buffer (absolute indices), ONE per-drawable record buffer — so a HW hit can
    /// fetch the real material (normal/UV/albedo texture) and re-light off-screen reflections. Three
    /// bindless slots total (no per-primitive overflow). Built alongside the content TLAS.
    content_hit: Option<(StorageBuffer, StorageBuffer, StorageBuffer)>,
    /// Content PT oracle: drawable count of the content TLAS (`cam_pos.w` bounds +
    /// the accumulation key), set by `build_content_accel`.
    content_instance_count: u32,
    /// Path-tracer per-instance table (vertex/index SB indices + material).
    instance_table: Option<StorageBuffer>,
    /// Keeps the per-instance vertex/index storage buffers alive for the table.
    _geometry: Vec<StorageBuffer>,
    instance_count: u32,
    /// Alternate Cornell-box scene (strong color bleeding, area-light GI).
    cornell_scene: Option<RaytracingScene>,
    cornell_table: Option<StorageBuffer>,
    _cornell_geometry: Vec<StorageBuffer>,
    cornell_instance_count: u32,
    /// M4 progressive accumulation: persistent float4-per-pixel sum, reset on change.
    path_accum: Option<StorageBuffer>,
    accum_extent: (u32, u32),
    accum_frame: u32,
    last_pt_key: Option<[u32; 8]>,
    /// Which scene's TLAS is currently bound (the open scene is the startup default).
    bound_cornell: bool,
}

impl RtSystem {
    /// Build the RT pipelines and acceleration structures. `scene_meshes` are the
    /// mesh sources for the path tracer's instance table, aligned 1:1 with `scene`
    /// (the rasterizer's objects, in TLAS instance order); the ground is appended.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        scene: &[SceneObject],
        scene_meshes: &[&MeshData],
        ground: &MeshData,
        ground_vbuf: &Buffer,
        ground_ibuf: &Buffer,
        ground_count: u32,
        // Build the sample-scene BLAS/TLAS + path-tracer instance table. Off for the
        // glTF path: its primitive count (one vertex+index storage buffer each) would
        // overflow the 64-slot bindless storage-buffer table, and HW RT is gallery-only.
        build_scene_accel: bool,
    ) -> anyhow::Result<Self> {
        // Phase 8 M3: inline ray-query trace pipeline (compute + `RayQuery`). Only on
        // RT-capable devices; the bindless block then carries the scene TLAS (binding
        // 5 / `t1088,space1`).
        let trace_pipeline = if device.has_raytracing() {
            let rt_trace_cs = load_compute_shader(
                backend,
                dreamcoast_shader::rt_trace_cs_spirv,
                dreamcoast_shader::rt_trace_cs_dxil,
                dreamcoast_shader::rt_trace_cs_metallib,
                "rt_trace",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: rt_trace_cs,
                compute_entry: "csMain",
                push_constant_size: 112, // inv_view_proj + cam_pos + sun_dir + out/w/h/pad
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };

        // Phase 8 M4: inline path tracer (diffuse GI bounce loop + progressive
        // accumulation). Shares the bindless TLAS + geometry storage buffers.
        let path_pipeline = if device.has_raytracing() {
            let rt_path_cs = load_compute_shader(
                backend,
                dreamcoast_shader::rt_path_cs_spirv,
                dreamcoast_shader::rt_path_cs_dxil,
                dreamcoast_shader::rt_path_cs_metallib,
                "rt_path",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: rt_path_cs,
                compute_entry: "csMain",
                push_constant_size: 144, // inv_view_proj + cam_pos + sun + 2x uint4 + sky float4
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };

        let rt_pipeline_shaders_available = match backend {
            BackendKind::Metal => {
                dreamcoast_shader::rt_pipeline_rgen_metallib().is_some()
                    && dreamcoast_shader::rt_pipeline_miss_metallib().is_some()
                    && dreamcoast_shader::rt_pipeline_chit_metallib().is_some()
                    && dreamcoast_shader::rt_pipeline_dispatch_metallib().is_some()
                    && dreamcoast_shader::rt_pipeline_isect_metallib().is_some()
            }
            _ => true,
        };

        let rt_pipeline_requested = std::env::var_os("P8_PATHTRACE_PIPELINE").is_some();

        // Phase 8 M5: the same path tracer via the hardware ray-tracing *pipeline*
        // (raygen / miss / closest-hit + shader binding table). Reproduces the inline
        // tracer's image so the two RT abstractions can be cross-checked. Gated on
        // `supports_rt_pipeline()`; on Metal the shader bytes are optional because the
        // converter/DXC toolchain may not be installed on every development machine.
        let pt_pipeline = if rt_pipeline_requested
            && device.supports_rt_pipeline()
            && rt_pipeline_shaders_available
        {
            let rgen = load_compute_shader(
                backend,
                dreamcoast_shader::rt_pipeline_rgen_spirv,
                dreamcoast_shader::rt_pipeline_rgen_dxil,
                dreamcoast_shader::rt_pipeline_rgen_metallib,
                "rt_pipeline_rgen",
            )?;
            let miss = load_compute_shader(
                backend,
                dreamcoast_shader::rt_pipeline_miss_spirv,
                dreamcoast_shader::rt_pipeline_miss_dxil,
                dreamcoast_shader::rt_pipeline_miss_metallib,
                "rt_pipeline_miss",
            )?;
            let chit = load_compute_shader(
                backend,
                dreamcoast_shader::rt_pipeline_chit_spirv,
                dreamcoast_shader::rt_pipeline_chit_dxil,
                dreamcoast_shader::rt_pipeline_chit_metallib,
                "rt_pipeline_chit",
            )?;
            Some(device.create_raytracing_pipeline(&RaytracingPipelineDesc {
                raygen_bytes: rgen,
                raygen_entry: "rgMain",
                miss_bytes: miss,
                miss_entry: "msMain",
                closesthit_bytes: chit,
                closesthit_entry: "chMain",
                metal_ray_dispatch_bytes: if backend == BackendKind::Metal {
                    dreamcoast_shader::rt_pipeline_dispatch_metallib()
                } else {
                    None
                },
                metal_ray_dispatch_entry: Some("RaygenIndirection"),
                metal_intersection_bytes: if backend == BackendKind::Metal {
                    dreamcoast_shader::rt_pipeline_isect_metallib()
                } else {
                    None
                },
                metal_intersection_entry: Some(
                    "irconverter.wrapper.intersection.function.triangle",
                ),
                push_constant_size: 144, // matches rt_path / rt_pipeline PushConstants
                // Payload = float3 x4 (48) + uint x3 (12) + float x2 cone state (8) = 68,
                // rounded up to a multiple of 8. Must be >= the shader payload or D3D12
                // CreateStateObject rejects the SHADER_CONFIG with E_INVALIDARG.
                max_payload_size: 72,
                max_attribute_size: 8, // barycentrics (float2)
            })?)
        } else {
            None
        };

        // Hardware ray tracing (Phase 8): build one BLAS per scene mesh + ground and a
        // TLAS over their instances, then register the TLAS in the bindless table so
        // the inline-trace compute pass (M3) can trace it. The scene outlives the
        // frame loop (the TLAS must stay alive while it is bound).
        let scene_rt = if device.has_raytracing() && build_scene_accel {
            let mut geoms: Vec<RtGeometry> = scene
                .iter()
                .map(|o| RtGeometry {
                    vertex_buffer: &o.mesh.vbuf,
                    index_buffer: &o.mesh.ibuf,
                    geometry: BlasGeometry {
                        vertex_count: o.mesh.vertex_count,
                        vertex_stride: 32,
                        index_count: o.mesh.index_count,
                    },
                })
                .collect();
            geoms.push(RtGeometry {
                vertex_buffer: ground_vbuf,
                index_buffer: ground_ibuf,
                geometry: BlasGeometry {
                    vertex_count: ground.vertices.len() as u32,
                    vertex_stride: 32,
                    index_count: ground_count,
                },
            });
            let instances: Vec<TlasInstance> = (0..geoms.len())
                .map(|i| TlasInstance {
                    blas_index: i as u32,
                    transform: mat4_to_3x4(if i < scene.len() {
                        scene[i].transform
                    } else {
                        Mat4::IDENTITY
                    }),
                    custom_index: i as u32,
                    mask: 0xFF,
                })
                .collect();
            match device.build_raytracing_scene(&geoms, &instances) {
                Ok(s) => {
                    device.bind_tlas(&s);
                    info!("ray-tracing scene built: {} BLAS + 1 TLAS", geoms.len());
                    Some(s)
                }
                Err(e) => {
                    tracing::error!("ray-tracing scene build failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Phase 8 M4: per-instance geometry storage buffers + instance table for the
        // path tracer's hit shading. One vertex + one index storage buffer per
        // instance (read as raw byte-address buffers in the shader), plus a table
        // mapping InstanceID -> { vertex SB index, index SB index, albedo }. The order
        // MUST match the TLAS instance custom_index order (scene objects, then ground).
        // `_geometry` keeps the geometry buffers alive for the program's lifetime.
        let (instance_table, geometry, instance_count) = if scene_rt.is_some() {
            // (mesh, material) per instance, in TLAS instance order (objects, then
            // ground). Materials mirror the rasterizer's so the path tracer shades with
            // the same metallic-roughness PBR model.
            // base_color.a is the path tracer's emissive scale (the Cornell light uses
            // it). The sample-scene objects are NOT emitters — their .a is just opacity —
            // so zero it, else e.g. the chrome sphere emits its own base color and reads
            // as a glowing white ball instead of a mirror.
            let mat_of = |o: &SceneObject| PtMaterial {
                base_color: [o.base_color[0], o.base_color[1], o.base_color[2], 0.0],
                metallic: o.metallic,
                roughness: o.roughness,
                ao: 1.0,
                tex: o.tex,
            };
            let mut entries: Vec<(&MeshData, PtMaterial)> = scene_meshes
                .iter()
                .zip(scene.iter())
                .map(|(m, o)| (*m, mat_of(o)))
                .collect();
            entries.push((ground, PtMaterial::diffuse([0.8, 0.8, 0.8, 0.0])));
            let (table, geometry) = build_pt_instance_table(device, &entries)?;
            info!("path-tracer instance table: {} instances", entries.len());
            (Some(table), geometry, entries.len() as u32)
        } else {
            (None, Vec::new(), 0u32)
        };

        // Phase 8 M4: an alternate Cornell-box scene for the path tracer (strong color
        // bleeding from the red/green walls, area-light GI). Built once: its own BLAS
        // per quad/box + TLAS + instance table. The host-visible vertex/index buffers
        // are only needed during the BLAS build, so they drop at the end of this block.
        let (cornell_scene, cornell_table, cornell_geometry, cornell_instance_count) = if device
            .has_raytracing()
        {
            let meshes = dreamcoast_asset::cornell_box();
            let mut hostbufs: Vec<(Buffer, Buffer, u32, u32)> = Vec::with_capacity(meshes.len());
            for (m, _) in &meshes {
                let (vb, ib, ic) = crate::mesh::upload_mesh(device, m)?;
                hostbufs.push((vb, ib, ic, m.vertices.len() as u32));
            }
            let geoms: Vec<RtGeometry> = hostbufs
                .iter()
                .map(|(vb, ib, ic, vc)| RtGeometry {
                    vertex_buffer: vb,
                    index_buffer: ib,
                    geometry: BlasGeometry {
                        vertex_count: *vc,
                        vertex_stride: 32,
                        index_count: *ic,
                    },
                })
                .collect();
            let instances: Vec<TlasInstance> = (0..geoms.len() as u32)
                .map(|i| TlasInstance {
                    blas_index: i,
                    transform: mat4_to_3x4(Mat4::IDENTITY), // geometry already world-space
                    custom_index: i,
                    mask: 0xFF,
                })
                .collect();
            let scene = device.build_raytracing_scene(&geoms, &instances)?;
            // The Cornell box is all matte diffuse (emissive ceiling via base_color.a).
            let entries: Vec<(&MeshData, PtMaterial)> = meshes
                .iter()
                .map(|(m, a)| (m, PtMaterial::diffuse(*a)))
                .collect();
            let (table, geometry) = build_pt_instance_table(device, &entries)?;
            info!("cornell-box scene built: {} instances", meshes.len());
            (Some(scene), Some(table), geometry, meshes.len() as u32)
        } else {
            (None, None, Vec::new(), 0u32)
        };

        Ok(Self {
            trace_pipeline,
            path_pipeline,
            pt_pipeline,
            scene: scene_rt,
            content_scene: None,
            content_hit: None,
            content_instance_count: 0,
            instance_table,
            _geometry: geometry,
            instance_count,
            cornell_scene,
            cornell_table,
            _cornell_geometry: cornell_geometry,
            cornell_instance_count,
            path_accum: None,
            accum_extent: (0, 0),
            accum_frame: 0,
            last_pt_key: None,
            bound_cornell: false,
        })
    }

    /// Phase 16 (HWRT hybrid): build the CONTENT scene's acceleration structures — one BLAS per
    /// unique mesh referenced by the ECS draw list, and a single TLAS over the drawable instances
    /// (world transforms) — then bind the TLAS into the bindless table so the hardware-ray-traced
    /// reflection/GI compute passes can trace `g.tlas`. Unlike the path tracer's per-instance
    /// geometry table (which needs one vertex + one index storage buffer per instance and would
    /// overflow the 64-slot bindless limit on a 400+ mesh scene), this builds ONLY the TLAS (one
    /// bindless slot): the hardware ray returns a hit position/distance, and the hit is shaded from
    /// the surface cache (SurfaceCache mode), so no per-hit geometry lookup is required. Opt-in
    /// (`P_HWRT`); the gallery keeps its own `scene` accel. No-op without an RT-capable device or on
    /// an empty draw list. The `RaytracingScene` is stored so its BLAS/TLAS outlive the frame loop.
    pub(crate) fn build_content_accel(
        &mut self,
        device: &Device,
        drawables: &[Drawable],
        registry: &MeshRegistry,
        materials: &crate::registry::MaterialRegistry,
    ) {
        if !device.has_raytracing() || drawables.is_empty() {
            return;
        }
        // Unique meshes, in first-seen draw order → one BLAS each (deduped so instanced content
        // shares a BLAS). `index_of` maps a mesh handle to its BLAS index in the geometry list.
        let mut order: Vec<MeshHandle> = Vec::new();
        let mut index_of: HashMap<MeshHandle, u32> = HashMap::new();
        for d in drawables {
            index_of.entry(d.mesh).or_insert_with(|| {
                let idx = order.len() as u32;
                order.push(d.mesh);
                idx
            });
        }
        // Hold the `Rc<GpuMesh>` handles alive across the build; `RtGeometry` borrows their buffers.
        let meshes: Vec<Rc<GpuMesh>> = order.iter().map(|&h| registry.get(h)).collect();
        let geoms: Vec<RtGeometry> = meshes
            .iter()
            .map(|m| RtGeometry {
                vertex_buffer: &m.vbuf,
                index_buffer: &m.ibuf,
                geometry: BlasGeometry {
                    vertex_count: m.vertex_count,
                    vertex_stride: 32,
                    index_count: m.index_count,
                },
            })
            .collect();
        // Instance masks route which rays may hit which geometry. Decals are surface tints,
        // not occluders — in the raster they only blend into the G-buffer albedo and cast no
        // shadow. In the path tracer the decal quad sits (slightly) in front of its host
        // surface, so a shadow ray that treats it as opaque geometry darkens the surface behind
        // it (the "decal shadow" board artifact). Give decals a distinct mask bit (0x01 only) so
        // shadow rays (traced with mask 0xFE) skip them, while camera/GI rays (mask 0xFF) still
        // hit them for the stochastic-alpha tint. Non-decals keep 0xFF (hit by every ray).
        let instances: Vec<TlasInstance> = drawables
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let is_decal =
                    materials.get(d.material).kind == dreamcoast_asset::MaterialKind::Decal;
                TlasInstance {
                    blas_index: index_of[&d.mesh],
                    transform: mat4_to_3x4(d.world),
                    custom_index: i as u32,
                    mask: if is_decal { 0x01 } else { 0xFF },
                }
            })
            .collect();
        let started = std::time::Instant::now();
        match device.build_raytracing_scene(&geoms, &instances) {
            Ok(s) => {
                device.bind_tlas(&s);
                info!(
                    "content ray-tracing scene built: {} BLAS + 1 TLAS ({} instances) in {:.1}ms",
                    geoms.len(),
                    instances.len(),
                    started.elapsed().as_secs_f32() * 1000.0
                );
                self.content_scene = Some(s);
            }
            Err(e) => tracing::error!("content ray-tracing scene build failed: {e}"),
        }
        // Phase 16 E: the consolidated geometry + material table for Hit Lighting (only useful once
        // the TLAS exists). A build failure is non-fatal — the reflection falls back to the surface
        // cache for off-screen hits.
        if self.content_scene.is_some() {
            match crate::mesh::build_content_hit_table(device, drawables, registry, materials) {
                Ok(bufs) => self.content_hit = Some(bufs),
                Err(e) => tracing::error!("content hit-lighting table build failed: {e}"),
            }
            self.content_instance_count = drawables.len() as u32;
        }
    }

    /// Content PT oracle: whether the path tracer can run on the content scene —
    /// the content TLAS is built and the consolidated hit table doubles as its
    /// instance/material table (the level equivalent of `has_instance_table`).
    pub(crate) fn has_content_pt(&self) -> bool {
        self.content_scene.is_some() && self.content_hit.is_some()
    }

    /// The path tracer's packed `inst_index` for the content scene: drawable-record
    /// table in the low byte, shared vertex / index storage indices (+1) in bits
    /// 8..15 / 16..23 — nonzero upper bits are what flip the shader into the 48-byte
    /// consolidated record format (`content_instances()` in rt_common.slang).
    /// Bindless storage indices are < 64, so the fields never overflow a byte.
    fn content_pt_index(&self) -> Option<u32> {
        self.content_hit_indices()
            .map(|(v, i, t)| t | ((v + 1) << 8) | ((i + 1) << 16))
    }

    /// Whether the content scene's TLAS was built (Phase 16 HWRT hybrid opt-in).
    pub(crate) fn has_content_scene(&self) -> bool {
        self.content_scene.is_some()
    }

    /// Phase 16 E: the bindless storage indices `(vertices, indices, drawable table)` of the
    /// consolidated content geometry/material table for Hit Lighting, or `None` if not built.
    pub(crate) fn content_hit_indices(&self) -> Option<(u32, u32, u32)> {
        self.content_hit
            .as_ref()
            .map(|(v, i, t)| (v.storage_index(), i.storage_index(), t.storage_index()))
    }

    // Feature-availability predicates (drive the UI + toggle defaults).
    pub(crate) fn has_trace(&self) -> bool {
        self.trace_pipeline.is_some()
    }
    pub(crate) fn has_path(&self) -> bool {
        self.path_pipeline.is_some()
    }
    pub(crate) fn has_pt_pipeline(&self) -> bool {
        self.pt_pipeline.is_some()
    }
    pub(crate) fn has_scene(&self) -> bool {
        self.scene.is_some()
    }
    pub(crate) fn has_cornell(&self) -> bool {
        self.cornell_scene.is_some()
    }
    pub(crate) fn has_instance_table(&self) -> bool {
        self.instance_table.is_some()
    }
    /// Accumulated path-trace sample frames (for the UI's spp readout).
    pub(crate) fn accum_frame(&self) -> u32 {
        self.accum_frame
    }
    /// Bump the progressive-accumulation counter (end-of-frame, after the graph's
    /// `&self` borrows have ended — see the module docs).
    pub(crate) fn advance_accum(&mut self) {
        self.accum_frame += 1;
    }

    /// The Cornell scene's fixed front-facing camera: `(eye, inverse view-proj)`.
    pub(crate) fn cornell_camera(cw: u32, ch: u32, vulkan: bool) -> (Vec3, [f32; 16]) {
        let c_eye = Vec3::new(0.0, 1.0, 3.2);
        let c_view = Mat4::look_at_rh(c_eye, Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        let mut c_proj =
            Mat4::perspective_rh(40f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
        if vulkan {
            c_proj.y_axis.y *= -1.0;
        }
        (c_eye, (c_proj * c_view).inverse().to_cols_array())
    }

    /// Per-frame accumulation management (M4): rebind the TLAS on a scene switch,
    /// (re)allocate the accumulation buffer on a resize, and reset the accumulation
    /// counter when the view/lighting/resolution changes. Run before the graph is
    /// built (its `wait_idle` + fallible alloc must stay off the graph's borrow path).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        pt_active: bool,
        use_cornell: bool,
        cw: u32,
        ch: u32,
        pt_eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
    ) -> anyhow::Result<()> {
        if !pt_active {
            return Ok(());
        }
        // Switch the bound TLAS when toggling scenes (rare → wait_idle is fine).
        if self.bound_cornell != use_cornell {
            device.wait_idle()?;
            if use_cornell {
                device.bind_tlas(self.cornell_scene.as_ref().unwrap());
            } else if let Some(s) = self.scene.as_ref() {
                device.bind_tlas(s);
            }
            self.bound_cornell = use_cornell;
            self.accum_frame = 0;
        }
        if self.accum_extent != (cw, ch) {
            device.wait_idle()?;
            self.path_accum = Some(device.create_storage_buffer(&StorageBufferDesc {
                size: (cw as u64) * (ch as u64) * 16,
                stride: 16,
                indirect: false,
            })?);
            self.accum_extent = (cw, ch);
            self.accum_frame = 0;
        }
        let key = [
            pt_eye.x.to_bits(),
            pt_eye.y.to_bits(),
            pt_eye.z.to_bits(),
            sun_dir[0].to_bits(),
            sun_dir[1].to_bits(),
            sun_dir[2].to_bits(),
            sun_intensity.to_bits(),
            (cw.wrapping_mul(0x9E37_79B1).wrapping_add(ch)) ^ (use_cornell as u32),
        ];
        if self.last_pt_key != Some(key) {
            self.accum_frame = 0;
            self.last_pt_key = Some(key);
        }
        Ok(())
    }

    /// M4 inline path tracer (default) or M5 RT pipeline + SBT (`pipeline`). Writes
    /// `rt_out` (the storage image the tonemap pass displays) and accumulates into
    /// the persistent sum buffer. `prepare` must have run this frame (accum buffer +
    /// bound TLAS). The counter is bumped at end-of-frame via `advance_accum`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_path<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        rt_out: ResourceId,
        use_cornell: bool,
        pipeline: bool,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
        spp: u32,
        sky_gain: f32,
        sky_wb: [f32; 3],
    ) {
        let rt_pipe = self.path_pipeline.as_ref().expect("rt path pipeline");
        // M5: when enabled, drive the same path tracer through the RT pipeline + SBT
        // instead of the inline compute ray query (`None` = inline).
        let rt_pt = if pipeline {
            self.pt_pipeline.as_ref()
        } else {
            None
        };
        // Index only (no borrow held into the graph closure — that would over-extend
        // the graph's lifetime vs. the transient resources).
        let accum_index = self.path_accum.as_ref().unwrap().storage_index();
        // Instance/material source: the Cornell or gallery per-instance table, else
        // the content scene's consolidated table (the level PT oracle).
        let (inst_index, inst_count) = if use_cornell {
            (
                self.cornell_table.as_ref().unwrap().storage_index(),
                self.cornell_instance_count,
            )
        } else if let Some(table) = self.instance_table.as_ref() {
            (table.storage_index(), self.instance_count)
        } else {
            (
                self.content_pt_index().expect("content PT table"),
                self.content_instance_count,
            )
        };
        // External resource so the graph orders the accumulation write (and inserts a
        // barrier before the next frame's read).
        let accum_ext = graph.import_external("rt_accum");
        // bit0 = Vulkan Y-flip, bit1 = Cornell env mode (no sun, black bg).
        let flip = flip_y | if use_cornell { 2 } else { 0 };
        let frame_idx = self.accum_frame;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "rt_path",
                storage_writes: vec![rt_out, accum_ext],
                reads: vec![],
            },
            move |ctx| {
                let out_index = ctx.storage_index(rt_out);
                let cmd = ctx.cmd();
                let push = rt_path_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    sun_intensity,
                    out_index,
                    accum_index,
                    inst_index,
                    inst_count,
                    frame_idx,
                    cw,
                    ch,
                    flip,
                    spp,
                    sky_gain,
                    sky_wb,
                );
                if let Some(rt_pt) = rt_pt {
                    // Full RT pipeline path (raygen/miss/hit + SBT).
                    cmd.bind_raytracing_pipeline(rt_pt);
                    cmd.push_constants_rt(&push);
                    cmd.trace_rays(rt_pt, cw, ch);
                } else {
                    // Inline ray-query compute path.
                    cmd.bind_compute_pipeline(rt_pipe);
                    cmd.push_constants_compute(&push);
                    cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                }
                Ok(())
            },
        );
    }

    /// M3 single-bounce trace viz (primary-hit instance color modulated by a hardware
    /// shadow ray), written to `rt_out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_trace<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        rt_out: ResourceId,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        cw: u32,
        ch: u32,
        flip_y: u32,
    ) {
        let rt_pipe = self.trace_pipeline.as_ref().expect("rt trace pipeline");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "rt_trace",
                storage_writes: vec![rt_out],
                reads: vec![],
            },
            move |ctx| {
                let out_index = ctx.storage_index(rt_out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(rt_pipe);
                cmd.push_constants_compute(&rt_trace_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
    }
}
