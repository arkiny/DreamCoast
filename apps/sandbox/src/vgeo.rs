//! Phase 14 renderer integration — virtual geometry as a deferred G-buffer producer.
//!
//! `VgeoSystem` holds a cluster page **per mesh** (built from the registry's CPU geometry with
//! `build_lod_dag`, so a page matches its uploaded mesh exactly — no cook round-trip, no
//! renormalization) consolidated into FOUR shared bindless buffers, plus per-frame scratch. It
//! records ONE scene-wide cut → clear → SW-raster → (HW mesh-vis) → resolve set that writes the
//! REAL Phase-6 G-buffer for **every** eligible opaque object at once — not one pass chain per
//! object (the ~515-passes-for-Sponza problem). This is the Nanite `NaniteRasterBinning` structure:
//! a per-pixel payload stores a WORK INDEX `t`; `work[t] = (instance, global cluster)` ties the
//! shared cluster geometry to a per-instance transform + material fetched from an instance table
//! (Nanite's `VisibleCluster.InstanceId → GetInstanceSceneData`). See
//! `docs/phase-14-vgeo-unified-pass.md`.
//!
//! Opt-in behind `P14_VGEO` (default ON for non-gallery scenes; gallery byte-identical anchor).
//! `P14_VGEO_BIN=1` (needs mesh shaders) additionally splits each cluster HW/SW by projected screen
//! size: large-triangle clusters → the `vgeo_scene_hwvis` mesh shader, micro-triangle clusters → the
//! compute SW rasterizer, both writing the same visibility buffer (the Sponza-class large-triangle
//! path). SW-only (bin off) covers small/medium-triangle meshes.
//!
//! One material per page (the mesh's) — matching a single-material mesh; a many-material model is
//! split into per-mesh assets upstream (the scene-cook direction) rather than baking per-cluster
//! materials into one page.

use std::collections::HashMap;
use std::rc::Rc;

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ClearColor, ComputePipeline, ComputePipelineDesc, DepthCompare, Device, Extent2D,
    GraphicsPipeline, GraphicsPipelineDesc, MeshPipeline, MeshPipelineDesc, PrimitiveTopology,
    StorageBuffer, StorageBufferDesc, VertexLayout,
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
    /// glTF `doubleSided`: `false` → the raster backface-culls this object's clusters (per-triangle,
    /// Nanite-style). `true` → two-sided (no cull), matching the `CULL_NONE` mesh fill.
    pub(crate) two_sided: bool,
}

/// One vgeo draw resolved from the scene: its page + world transform + material. The unified pass
/// turns the whole slice of these into one instance table + work list (no per-object recording).
pub(crate) type VgeoDraw = (usize, VgeoMaterial, Mat4);

/// All meshes' cluster geometry consolidated into FOUR bindless storage buffers (not four *per
/// mesh*). Every page's vertex pool / remap / triangle / record arrays are concatenated with each
/// page's base offset baked into the indices at upload, so the whole scene uses a fixed 4 slots
/// regardless of mesh count — the fix for Sponza-class scenes (103–405 meshes would otherwise need
/// 400–1600 storage-buffer slots and overflow the bindless table).
struct GlobalGeom {
    vtx: StorageBuffer,
    remap: StorageBuffer,
    tri: StorageBuffer,
    rec: StorageBuffer,
}

/// One mesh's cluster range inside the consolidated [`GlobalGeom`]: its clusters occupy the global
/// record slots `rec_base .. rec_base + total_clusters`, and each record's `vertex_offset`/
/// `tri_offset` already point into the global remap/triangle arrays (baked at upload).
struct ClusterPage {
    rec_base: u32,
    total_clusters: u32,
}

/// Per-FIF scene scratch (grown as the scene / frustum set changes, never shrunk). Host-visible so
/// the CPU rebuilds the instance table + work list and resets the counters/lists each frame (a
/// DEFAULT-heap host write is illegal on D3D12). `hw_list`/`hw_args` exist only under binning.
struct SceneScratch {
    /// Per-instance transforms + material (256 B/instance), rebuilt each frame from the draw list.
    instances: StorageBuffer,
    /// Flat `(instance, global cluster)` work items — W entries, one per instance × cluster.
    work: StorageBuffer,
    /// Visible list (work indices the cut selected) + its indirect counter. SW-only: the whole cut.
    /// Binning: the SW sub-list.
    sw_list: StorageBuffer,
    sw_args: StorageBuffer,
    /// Binning HW sub-list (mesh-shader) + its indirect draw args.
    hw_list: Option<StorageBuffer>,
    hw_args: Option<StorageBuffer>,
    /// Current capacities (elements) so we only reallocate when the scene grows.
    cap_instances: usize,
    cap_work: usize,
}

pub(crate) struct VgeoSystem {
    /// All meshes' cluster geometry in 4 shared buffers (see [`GlobalGeom`]).
    geom: GlobalGeom,
    pages: Vec<ClusterPage>,
    /// `Rc<GpuMesh>` pointer → page index, so a scene object routes to its mesh's page.
    mesh_to_page: HashMap<usize, usize>,
    /// Per-FIF shared R64 visibility buffer (sized to the render extent). The whole scene serializes
    /// on one buffer; inter-object occlusion is resolved by the packed depth key, not separate
    /// buffers.
    visbuf: Vec<StorageBuffer>,
    /// Per-FIF scene scratch (instance table + work list + visible lists).
    scratch: Vec<Option<SceneScratch>>,
    extent: Extent2D,
    cut_pipeline: ComputePipeline,
    clear_pipeline: ComputePipeline,
    raster_pipeline: ComputePipeline,
    resolve_pipeline: GraphicsPipeline,
    tau: f32,
    /// M5b HW/SW binning (opt-in `P14_VGEO_BIN=1`, needs mesh shaders): the binning cut splits each
    /// cluster into an HW (mesh-shader) and a SW (compute-raster) sub-list by projected screen size;
    /// both rasterize the SAME visibility buffer. `None` = SW-only.
    bin: Option<BinPipelines>,
    bin_px: f32,
}

/// The extra pipelines the HW/SW binning path needs (only built when `P14_VGEO_BIN` + mesh shaders).
struct BinPipelines {
    cut_bin: ComputePipeline,
    hwvis: MeshPipeline,
}

fn vis_bytes(extent: Extent2D) -> u64 {
    (extent.width.max(1) as u64) * (extent.height.max(1) as u64) * 8
}

/// Bytes per instance-table record — mirrors `SceneInstance` in `vgeo_scene_common.slang`.
const INSTANCE_STRIDE: usize = 256;

impl VgeoSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        registry: &MeshRegistry,
        extent: Extent2D,
        cache_dir: &std::path::Path,
    ) -> anyhow::Result<Self> {
        use dreamcoast_scene::MeshHandle;

        // The integrated path is SW-only (compute cut / clear / raster + a full-screen resolve):
        // it needs 64-bit buffer atomics for the visibility buffer, NOT mesh shaders. (Gating on
        // mesh_shader was a Metal-ism — that backend reports both together; DX/VK expose them
        // independently, and the mesh-shader HW path is the binning add-on.)
        if !device.capabilities().atomic_int64 {
            anyhow::bail!(
                "P14_VGEO: {backend:?} lacks 64-bit buffer atomics (atomic_int64=false) required by the SW visibility buffer"
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

        // Build a cluster page per mesh from its CPU geometry (build_lod_dag = what the cook runs),
        // appending each into the CONSOLIDATED global arrays with its base offsets baked in.
        let (mut gvtx, mut gremap, mut gtri, mut grec) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut pages: Vec<ClusterPage> = Vec::new();
        let mut mesh_to_page: HashMap<usize, usize> = HashMap::new();

        // Build each mesh's LOD DAG on the job-system workers, then assemble the pages sequentially
        // in mesh order (append order → byte-identical consolidation regardless of build order). The
        // slow part is `load_or_build_dag` (cache read or `build_lod_dag`); it is per-mesh
        // independent and cross-process deterministic, so the parallel result equals the serial one.
        // Gather the CPU geometry slices up front on this thread: `MeshRegistry` holds `Rc<GpuMesh>`
        // (not `Sync`) so it can't cross to a worker, but the plain-data vertex/index slices can.
        let inputs: Vec<Option<(&[dreamcoast_asset::MeshVertex], &[u32])>> = (0..registry.len())
            .map(|i| {
                let cpu = registry.cpu(MeshHandle(i));
                (cpu.indices.len() >= 3).then_some((cpu.vertices.as_slice(), cpu.indices.as_slice()))
            })
            .collect();
        let mut dags: Vec<Option<dreamcoast_asset::vgeo::MeshClusters>> =
            (0..registry.len()).map(|_| None).collect();
        crate::cook_progress::parallel_cook("vgeo cluster DAGs", &mut dags, 1, |i, slot| {
            if let Some((verts, indices)) = inputs[i] {
                let dag = Self::load_or_build_dag(cache_dir, verts, indices);
                if !dag.clusters.is_empty() {
                    *slot = Some(dag);
                }
            }
        });
        for (i, dag) in dags.iter().enumerate() {
            let Some(dag) = dag else { continue };
            let page = Self::append_page(&mut gvtx, &mut gremap, &mut gtri, &mut grec, dag);
            let idx = pages.len();
            pages.push(page);
            mesh_to_page.insert(Rc::as_ptr(&registry.get(MeshHandle(i as u32))) as usize, idx);
        }
        if pages.is_empty() {
            anyhow::bail!("P14_VGEO: no mesh produced clusters");
        }
        let geom = GlobalGeom {
            vtx: sb(&gvtx, 32)?,
            remap: sb(&gremap, 4)?,
            tri: sb(&gtri, 4)?,
            rec: sb(&grec, 96)?,
        };
        tracing::info!(
            "P14_VGEO: {} cluster page(s), {} total clusters (consolidated into 4 buffers)",
            pages.len(),
            grec.len() / 96,
        );

        let mut visbuf = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            visbuf.push(device.create_storage_buffer(&StorageBufferDesc {
                size: vis_bytes(extent),
                stride: 8,
                indirect: false,
            })?);
        }
        let scratch = (0..FRAMES_IN_FLIGHT).map(|_| None).collect();

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
            dreamcoast_shader::vgeo_scene_cut_cs_spirv,
            dreamcoast_shader::vgeo_scene_cut_cs_dxil,
            dreamcoast_shader::vgeo_scene_cut_cs_metallib,
            "csCutScene",
            160,
            [64, 1, 1],
        )?;
        let clear_pipeline = compute(
            dreamcoast_shader::vgeo_scene_clear_cs_spirv,
            dreamcoast_shader::vgeo_scene_clear_cs_dxil,
            dreamcoast_shader::vgeo_scene_clear_cs_metallib,
            "csClearScene",
            40,
            [64, 1, 1],
        )?;
        let raster_pipeline = compute(
            dreamcoast_shader::vgeo_scene_raster_cs_spirv,
            dreamcoast_shader::vgeo_scene_raster_cs_dxil,
            dreamcoast_shader::vgeo_scene_raster_cs_metallib,
            "csRasterScene",
            40,
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
            push_constant_size: 40,
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
        let bin_px: f32 = std::env::var("VGEO_BINPX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16.0);

        // M5b HW/SW binning: needs mesh shaders for the HW path. **Default ON** when the adapter has
        // mesh shaders (the large-triangle path a Sponza-class scene needs); `P14_VGEO_BIN=0` forces
        // SW-only. Without mesh shaders it warns and stays SW-only rather than failing vgeo.
        let bin_requested = match std::env::var("P14_VGEO_BIN").ok().as_deref() {
            Some("0") => false,
            Some(_) => true,
            None => device.capabilities().mesh_shader,
        };
        let bin = if bin_requested {
            if !device.capabilities().mesh_shader {
                tracing::warn!(
                    "P14_VGEO_BIN: {backend:?} lacks mesh shaders — staying SW-only (large-triangle \
                     clusters may be dropped)"
                );
                None
            } else {
                let cut_bin = compute(
                    dreamcoast_shader::vgeo_scene_cut_bin_cs_spirv,
                    dreamcoast_shader::vgeo_scene_cut_bin_cs_dxil,
                    dreamcoast_shader::vgeo_scene_cut_bin_cs_metallib,
                    "csCutSceneBin",
                    160,
                    [64, 1, 1],
                )?;
                let (hms, hfs) = match backend {
                    BackendKind::Vulkan => (
                        dreamcoast_shader::vgeo_scene_hwvis_ms_spirv(),
                        dreamcoast_shader::vgeo_scene_hwvis_fs_spirv(),
                    ),
                    BackendKind::D3d12 => (
                        dreamcoast_shader::vgeo_scene_hwvis_ms_dxil(),
                        dreamcoast_shader::vgeo_scene_hwvis_fs_dxil(),
                    ),
                    BackendKind::Metal => (
                        dreamcoast_shader::vgeo_scene_hwvis_ms_metallib(),
                        dreamcoast_shader::vgeo_scene_hwvis_fs_metallib(),
                    ),
                };
                let hwvis = device.create_mesh_pipeline(&MeshPipelineDesc {
                    object_bytes: None,
                    object_entry: "",
                    mesh_bytes: hms
                        .ok_or_else(|| anyhow::anyhow!("vgeo_scene_hwvis mesh shader unavailable"))?,
                    mesh_entry: "meshMain",
                    fragment_bytes: hfs.ok_or_else(|| {
                        anyhow::anyhow!("vgeo_scene_hwvis fragment shader unavailable")
                    })?,
                    fragment_entry: "fragMain",
                    // Throwaway color target (the fragment's atomicMax writes the visibility buffer;
                    // its colour output is unused). Matches the scratch target `record_scene` makes.
                    color_formats: &[crate::GB_ALBEDO_FMT],
                    depth_format: None,
                    push_constant_size: 40,
                    bindless: true,
                    uniform_buffer: false,
                    object_threads: [1, 1, 1],
                    mesh_threads: [128, 1, 1],
                    depth_test: false,
                    depth_write: false,
                    depth_compare: DepthCompare::Less,
                })?;
                tracing::info!("P14_VGEO_BIN: HW/SW binning enabled (split at {bin_px}px diameter)");
                Some(BinPipelines { cut_bin, hwvis })
            }
        } else {
            None
        };

        Ok(Self {
            geom,
            pages,
            mesh_to_page,
            visbuf,
            scratch,
            extent,
            cut_pipeline,
            clear_pipeline,
            raster_pipeline,
            resolve_pipeline,
            tau,
            bin,
            bin_px,
        })
    }

    /// Load a mesh's LOD DAG from the per-mesh cluster cache, or build it (`build_lod_dag`) and
    /// cache it. The key is the geometry hash + the `.dcasset` format version, so a geometry or
    /// format change re-cooks that one mesh; a hit skips the (seconds-per-mesh) DAG build.
    fn load_or_build_dag(
        cache_dir: &std::path::Path,
        verts: &[dreamcoast_asset::MeshVertex],
        indices: &[u32],
    ) -> dreamcoast_asset::vgeo::MeshClusters {
        use dreamcoast_asset::dcasset;
        let mut bytes = Vec::with_capacity(verts.len() * 32 + indices.len() * 4);
        for v in verts {
            for f in v.pos.iter().chain(&v.normal).chain(&v.uv) {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        for &i in indices {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        let hash = dcasset::source_hash(&bytes);
        let file = cache_dir.join(format!("{hash:016x}.dcasset"));
        if let Ok(b) = std::fs::read(&file)
            && let Ok((h, mc)) = dcasset::read_clusters(&b)
            && h.version == dcasset::VERSION
            && h.source_hash == hash
        {
            return mc;
        }
        let mc = dreamcoast_asset::vgeo::build_lod_dag(verts, indices, 0);
        let _ = std::fs::create_dir_all(cache_dir);
        let _ = std::fs::write(&file, dcasset::write_clusters(&mc, hash));
        mc
    }

    /// Append a cluster DAG into the consolidated global arrays, baking this page's base offsets
    /// into the indices so the records point into the shared buffers directly. Returns the page's
    /// global record range.
    fn append_page(
        gvtx: &mut Vec<u8>,
        gremap: &mut Vec<u8>,
        gtri: &mut Vec<u8>,
        grec: &mut Vec<u8>,
        dag: &dreamcoast_asset::vgeo::MeshClusters,
    ) -> ClusterPage {
        let vtx_base = (gvtx.len() / 32) as u32;
        let remap_base = (gremap.len() / 4) as u32;
        let tri_base = (gtri.len() / 4) as u32;
        let rec_base = (grec.len() / 96) as u32;
        for v in &dag.vertices {
            for f in v.pos.iter().chain(&v.normal) {
                gvtx.extend_from_slice(&f.to_le_bytes());
            }
            for f in v.uv {
                gvtx.extend_from_slice(&f.to_le_bytes());
            }
        }
        for &i in &dag.cluster_vertices {
            gremap.extend_from_slice(&(i + vtx_base).to_le_bytes());
        }
        for &b in &dag.cluster_triangles {
            gtri.extend_from_slice(&(b as u32).to_le_bytes());
        }
        let put = |rec: &mut Vec<u8>, f: f32| rec.extend_from_slice(&f.to_le_bytes());
        let put3 = |rec: &mut Vec<u8>, v: [f32; 3]| {
            for f in v {
                rec.extend_from_slice(&f.to_le_bytes());
            }
        };
        for cl in &dag.clusters {
            for field in [
                cl.vertex_offset + remap_base,
                cl.vertex_count,
                cl.triangle_offset + tri_base,
                cl.triangle_count,
            ] {
                grec.extend_from_slice(&field.to_le_bytes());
            }
            put(grec, cl.self_error);
            put3(grec, cl.self_center);
            put(grec, cl.self_radius);
            put(grec, cl.parent_error);
            put3(grec, cl.parent_center);
            put(grec, cl.parent_radius);
            put3(grec, cl.bounds_center);
            put(grec, cl.bounds_radius);
            put3(grec, cl.cone_axis);
            put(grec, cl.cone_cutoff);
            grec.extend_from_slice(&[0u8; 8]); // 88..96 pad
        }
        ClusterPage {
            rec_base,
            total_clusters: dag.clusters.len() as u32,
        }
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

    /// The shared visibility buffer's ordering handle (imported once per frame so every pass
    /// serializes on it).
    pub(crate) fn import_visbuf(graph: &mut RenderGraph) -> ResourceId {
        graph.import_external("vgeo_visbuf")
    }

    /// Build this frame's instance table + work list from the whole draw slice, host-reset the
    /// visible lists + counters, and (re)size the per-FIF scratch. Returns the work count `W` (=
    /// Σ instance clusters) for the fixed cut/raster dispatch. Call BEFORE the graph (its buffers
    /// must outlive `graph.execute`). `cull_vp` is the flip-free view_proj (SW raster + resolve);
    /// `view_vp` is the Y-flipped view_proj (HW mesh SV_Position).
    pub(crate) fn prepare_scene(
        &mut self,
        device: &Device,
        fif: usize,
        draws: &[VgeoDraw],
        cull_vp: Mat4,
        view_vp: Mat4,
    ) -> anyhow::Result<u32> {
        let binning = self.bin.is_some();

        // Instance table (256 B/instance) + flat work list (uint2/entry): both rebuilt every frame.
        let mut inst_bytes = Vec::with_capacity(draws.len() * INSTANCE_STRIDE);
        let mut work_bytes: Vec<u8> = Vec::new();
        let push_mat = |b: &mut Vec<u8>, m: &Mat4| {
            for f in m.to_cols_array() {
                b.extend_from_slice(&f.to_le_bytes());
            }
        };
        for (inst_idx, (page_idx, mat, model)) in draws.iter().enumerate() {
            let page = &self.pages[*page_idx];
            let max_scale = model
                .x_axis
                .truncate()
                .length()
                .max(model.y_axis.truncate().length())
                .max(model.z_axis.truncate().length());
            // Three CPU-computed matrices (no in-shader matmul → no layout risk): model for world
            // pos/normal + cut bounds, mvp (flip-free) for SW raster + resolve + hwvis cull, mvp_hw
            // (Y-flipped) for the HW rasterizer's SV_Position.
            push_mat(&mut inst_bytes, model);
            push_mat(&mut inst_bytes, &(cull_vp * *model));
            push_mat(&mut inst_bytes, &(view_vp * *model));
            for f in mat.base_color {
                inst_bytes.extend_from_slice(&f.to_le_bytes());
            }
            let cull_backface = if mat.two_sided { 0.0f32 } else { 1.0f32 };
            for f in [mat.metallic, mat.roughness, cull_backface, mat.alpha_cutoff] {
                inst_bytes.extend_from_slice(&f.to_le_bytes());
            }
            for w in mat.tex {
                inst_bytes.extend_from_slice(&w.to_le_bytes());
            }
            inst_bytes.extend_from_slice(&max_scale.to_le_bytes());
            inst_bytes.extend_from_slice(&[0u8; 12]); // 244..256 pad

            // Work items: this instance × each of its page's clusters (global cluster index baked).
            for c in 0..page.total_clusters {
                work_bytes.extend_from_slice(&(inst_idx as u32).to_le_bytes());
                work_bytes.extend_from_slice(&(page.rec_base + c).to_le_bytes());
            }
        }
        let work_count = (work_bytes.len() / 8) as u32;

        // (Re)allocate the scratch buffers if this frame needs more than the current capacity
        // (grow-only — the scene / frustum set stabilizes, so this settles after a few frames).
        let need_inst = draws.len().max(1);
        let need_work = (work_count as usize).max(1);
        let host = |bytes: u64, stride: u32, indirect: bool| -> anyhow::Result<StorageBuffer> {
            Ok(device.create_storage_buffer_host(&StorageBufferDesc {
                size: bytes.max(4),
                stride,
                indirect,
            })?)
        };
        let realloc = match &self.scratch[fif] {
            None => true,
            Some(s) => need_inst > s.cap_instances || need_work > s.cap_work,
        };
        if realloc {
            // Round capacity up so small per-frame wobble doesn't thrash allocations.
            let cap_inst = need_inst.next_power_of_two();
            let cap_work = need_work.next_power_of_two();
            let (hw_list, hw_args) = if binning {
                (
                    Some(host((cap_work * 4) as u64, 4, false)?),
                    Some(host(12, 4, true)?),
                )
            } else {
                (None, None)
            };
            self.scratch[fif] = Some(SceneScratch {
                instances: host((cap_inst * INSTANCE_STRIDE) as u64, INSTANCE_STRIDE as u32, false)?,
                work: host((cap_work * 8) as u64, 8, false)?,
                sw_list: host((cap_work * 4) as u64, 4, false)?,
                sw_args: host(12, 4, false)?,
                hw_list,
                hw_args,
                cap_instances: cap_inst,
                cap_work,
            });
        }
        let s = self.scratch[fif].as_ref().expect("scratch just ensured");

        // Upload the instance table + work list.
        s.instances.write(&inst_bytes)?;
        s.work.write(&work_bytes)?;

        // Reset the counters to the indirect grid `{0,1,1}` (the cut bumps `.x`) and sentinel-clear
        // the visible list(s) to `0xFFFFFFFF` (the fixed-count raster skips unfilled slots). Only
        // the `work_count` region is used, but clearing the full capacity is cheap + simple.
        let args0: Vec<u8> = [0u32, 1, 1].iter().flat_map(|w| w.to_le_bytes()).collect();
        let sentinel = vec![0xFFu8; s.cap_work * 4];
        s.sw_args.write(&args0)?;
        s.sw_list.write(&sentinel)?;
        if let (Some(hl), Some(ha)) = (&s.hw_list, &s.hw_args) {
            ha.write(&args0)?;
            hl.write(&sentinel)?;
        }
        Ok(work_count)
    }

    /// Record the WHOLE scene's virtual-geometry G-buffer contribution in one pass set: scene cut →
    /// clear → SW-raster → (HW mesh-vis) → resolve overlay. `work_count` is [`Self::prepare_scene`]'s
    /// return. The mesh fill must run first (it clears the G-buffer + depth); the resolve LOADs them
    /// and Less-tests each fragment's `SV_Depth`, compositing with the rest.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_scene<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gbuf: GBufferTargets,
        fif: usize,
        work_count: u32,
        visbuf_ext: ResourceId,
        cull_view_proj: Mat4,
        eye: Vec3,
        mip_bias: f32,
        extent: Extent2D,
    ) {
        if work_count == 0 {
            return;
        }
        let visbuf = &self.visbuf[fif];
        let s = self.scratch[fif].as_ref().expect("prepare_scene ran");
        let (w, h) = (extent.width, extent.height);
        let tau = self.tau;
        let bin_px = self.bin_px;
        let binning = self.bin.is_some();
        let (vtx_i, remap_i, tri_i, rec_i) = (
            self.geom.vtx.storage_index(),
            self.geom.remap.storage_index(),
            self.geom.tri.storage_index(),
            self.geom.rec.storage_index(),
        );
        let vis_i = visbuf.storage_index();
        let work_i = s.work.storage_index();
        let inst_i = s.instances.storage_index();
        // SW list = the whole cut (SW-only) or the SW sub-list (binning). HW list = the binning HW
        // sub-list. In SW-only mode the cut writes the SW list as its single output.
        let sw_list_i = s.sw_list.storage_index();
        let sw_args_i = s.sw_args.storage_index();
        let (hw_list_i, hw_args_i) = match (&s.hw_list, &s.hw_args) {
            (Some(l), Some(a)) => (l.storage_index(), a.storage_index()),
            _ => (sw_list_i, sw_args_i),
        };

        let sw_list_ext = graph.import_external("vgeo_sw_list");
        let sw_args_ext = graph.import_external("vgeo_sw_args");
        let hw_list_ext = graph.import_external("vgeo_hw_list");
        let hw_args_ext = graph.import_external("vgeo_hw_args");

        // ── LOD cut (world-space) over the whole work list. SW-only: the cut writes the SW list +
        // args. Binning: `csCutSceneBin` splits into the SW sub-list (compute raster) and the HW
        // sub-list (mesh shader) by projected screen size. ──
        let planes = crate::push::frustum_planes(cull_view_proj);
        let cam = eye;
        let proj_factor = 0.5 * h as f32 / (30f32.to_radians()).tan();
        let cut = self
            .bin
            .as_ref()
            .map(|b| &b.cut_bin)
            .unwrap_or(&self.cut_pipeline);
        let hw_args_ref = s.hw_args.as_ref();
        let mut cut_writes = vec![sw_list_ext, sw_args_ext];
        if binning {
            cut_writes.push(hw_list_ext);
            cut_writes.push(hw_args_ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_cut",
                storage_writes: cut_writes,
                reads: vec![],
            },
            move |ctx| {
                let mut cpc = [0u8; 160];
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
                // vis_buf(124) = HW list (binning) or the whole SW list (SW-only); args_buf(128)
                // matches. sw_list(132)/sw_args(136) are the binning SW sub-list.
                for (i, word) in [
                    work_count,
                    rec_i,
                    hw_list_i,
                    hw_args_i,
                    sw_list_i,
                    sw_args_i,
                ]
                .iter()
                .enumerate()
                {
                    cpc[116 + i * 4..120 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                cpc[140..144].copy_from_slice(&bin_px.to_le_bytes());
                cpc[144..148].copy_from_slice(&work_i.to_le_bytes());
                cpc[148..152].copy_from_slice(&inst_i.to_le_bytes());
                let cmd = ctx.cmd();
                // `hw_args` feeds `draw_mesh_tasks_indirect` (binning): reset last frame's INDIRECT
                // state back to storage before the cut UAV-writes the count.
                if let Some(ha) = hw_args_ref {
                    cmd.storage_buffer_to_storage(ha);
                }
                cmd.bind_compute_pipeline(cut);
                cmd.push_constants_compute(&cpc);
                cmd.dispatch(work_count.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(&s.sw_list);
                if let Some(hl) = &s.hw_list {
                    cmd.storage_buffer_barrier(hl);
                }
                if let Some(ha) = hw_args_ref {
                    cmd.storage_buffer_to_indirect(ha);
                }
                Ok(())
            },
        );

        // ── Clear the R64 visibility buffer, then SW-raster the SW cut into it. The scene shaders
        // read the per-instance flip-free `mvp` from the instance table (the manual NDC→pixel
        // mapping needs the flip-free matrix; see docs). ──
        let clear = &self.clear_pipeline;
        let raster = &self.raster_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_raster",
                storage_writes: vec![visbuf_ext],
                reads: vec![sw_args_ext, sw_list_ext],
            },
            move |ctx| {
                // Scene raster/clear push (40 B): vtx, remap, tri, rec, vis, vis_list, work, inst, w, h.
                let mut spc = [0u8; 40];
                for (i, word) in [
                    vtx_i, remap_i, tri_i, rec_i, vis_i, sw_list_i, work_i, inst_i, w, h,
                ]
                .iter()
                .enumerate()
                {
                    spc[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(clear);
                cmd.push_constants_compute(&spc);
                cmd.dispatch((w * h).div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(visbuf);
                cmd.bind_compute_pipeline(raster);
                cmd.push_constants_compute(&spc);
                // Fixed dispatch = one threadgroup per work slot; csRasterScene skips sentinels.
                cmd.dispatch(work_count, 1, 1);
                cmd.storage_buffer_barrier(visbuf);
                Ok(())
            },
        );

        // ── HW mesh-vis (binning only): rasterize the HW sub-list into the SAME visibility buffer
        // via the hardware mesh pipeline + fragment atomicMax. Y-flipped `mvp_hw` (per instance)
        // lands the exact pixels the flip-free SW raster wrote (DX≡VK). A trailing barrier pass
        // orders its fragment writes before the resolve reads. ──
        if let (Some(binp), Some(hw_args_buf)) = (self.bin.as_ref(), s.hw_args.as_ref()) {
            let hwvis = &binp.hwvis;
            let scratch = graph.create_color("vgeo_hwvis_scratch", crate::GB_ALBEDO_FMT, extent);
            graph.add_pass_with_storage_writes(
                PassInfo {
                    name: "vgeo_hwvis",
                    colors: vec![(
                        scratch,
                        Some(ClearColor {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                    )],
                    depth: None,
                    reads: vec![hw_list_ext, hw_args_ext],
                },
                vec![visbuf_ext],
                move |ctx| {
                    // Scene hwvis push (40 B): vtx, remap, tri, rec, hw_list, vis, work, inst, w, h.
                    let mut hpc = [0u8; 40];
                    for (i, word) in [
                        vtx_i, remap_i, tri_i, rec_i, hw_list_i, vis_i, work_i, inst_i, w, h,
                    ]
                    .iter()
                    .enumerate()
                    {
                        hpc[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    let cmd = ctx.cmd();
                    cmd.bind_mesh_pipeline(hwvis);
                    cmd.push_constants_mesh(&hpc);
                    cmd.draw_mesh_tasks_indirect(hw_args_buf, 0);
                    Ok(())
                },
            );
            graph.add_compute_pass(
                ComputePassInfo {
                    name: "vgeo_hwvis_barrier",
                    storage_writes: vec![visbuf_ext],
                    reads: vec![],
                },
                move |ctx| {
                    ctx.cmd().storage_buffer_barrier(visbuf);
                    Ok(())
                },
            );
        }

        // ── Resolve: visibility buffer → the four G-buffer MRTs (+ depth), OVERLAY. One full-screen
        // pass for the whole scene; each pixel's payload → work index → instance transform/material.
        let resolve = &self.resolve_pipeline;
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
                // Scene resolve push (40 B): vtx, remap, tri, rec, vis, work, inst, w, h, mip_bias.
                let mut rpc = [0u8; 40];
                for (i, word) in [vtx_i, remap_i, tri_i, rec_i, vis_i, work_i, inst_i, w, h]
                    .iter()
                    .enumerate()
                {
                    rpc[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                rpc[36..40].copy_from_slice(&mip_bias.to_le_bytes());
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(resolve);
                cmd.push_constants(&rpc);
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}
