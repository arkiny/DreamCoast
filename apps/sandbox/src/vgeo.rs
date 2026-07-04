//! Phase 14 renderer integration — virtual geometry as a deferred G-buffer producer.
//!
//! `VgeoSystem` holds a cluster page **per mesh** (built from the registry's CPU geometry with
//! `build_lod_dag`, so a page matches its uploaded mesh exactly — no cook round-trip, no
//! renormalization) plus shared per-frame scratch, and records the compute + resolve passes that
//! write the REAL Phase-6 G-buffer. Every eligible opaque object is routed through virtual geometry
//! and overlaid on the mesh-rastered remainder (depth-composited), so a whole multi-object scene can
//! render as virtual geometry — the per-static-mesh-asset direction (see
//! `docs/phase-14-vgeo-integration.md`). Opt-in behind `P14_VGEO`, so the default renderer is
//! untouched (gallery byte-identical). `P14_VGEO_BIN=1` (needs mesh shaders) additionally splits each
//! object HW/SW: `csCutBin` sends large-triangle clusters to the `vgeo_hwvis` mesh shader and
//! micro-triangle clusters to the compute SW rasterizer, both writing the same visibility buffer
//! (the Sponza-class large-triangle path). SW-only (bin off) covers small/medium-triangle meshes.
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

pub(crate) struct VgeoSystem {
    /// All meshes' cluster geometry in 4 shared buffers (see [`GlobalGeom`]).
    geom: GlobalGeom,
    pages: Vec<ClusterPage>,
    /// `Rc<GpuMesh>` pointer → page index, so a scene object routes to its mesh's page.
    mesh_to_page: HashMap<usize, usize>,
    /// Max clusters over all pages (sizes the per-object visible-list slots).
    max_clusters: u32,
    /// Per-FIF shared R64 visibility buffer (sized to the render extent). Objects serialize on it
    /// (each clears → rasters → resolves), so one buffer suffices; inter-object occlusion is
    /// resolved by the G-buffer depth test in the resolve, not the visibility buffer.
    visbuf: Vec<StorageBuffer>,
    /// Per-FIF growable pool of per-object scratch (visible lists + indirect-append counters).
    pool: Vec<Vec<ObjScratch>>,
    extent: Extent2D,
    cut_pipeline: ComputePipeline,
    clear_pipeline: ComputePipeline,
    raster_pipeline: ComputePipeline,
    resolve_pipeline: GraphicsPipeline,
    tau: f32,
    /// M5b HW/SW binning (opt-in `P14_VGEO_BIN=1`, needs mesh shaders): the binning cut splits each
    /// object's clusters into an HW (mesh-shader) sub-list and a SW (compute-raster) sub-list by
    /// projected screen size; both rasterize the SAME visibility buffer. `None` = SW-only (the cut
    /// selects one list, all rasterized in compute). Required for large-triangle (Sponza-class)
    /// clusters whose single triangle covers too many pixels for the per-thread SW rasterizer.
    bin: Option<BinPipelines>,
    bin_px: f32,
}

/// The extra pipelines the HW/SW binning path needs (only built when `P14_VGEO_BIN` + mesh shaders).
struct BinPipelines {
    cut_bin: ComputePipeline,
    hwvis: MeshPipeline,
}

/// Per-object per-frame scratch. `hw_list`/`hw_args` are the whole cut (SW-only) or the HW sub-list
/// (binning); `sw_*` are the SW sub-list, allocated only when binning. Host-visible so `prepare`
/// resets them each frame (a DEFAULT-heap host write is illegal on D3D12).
struct ObjScratch {
    hw_list: StorageBuffer,
    hw_args: StorageBuffer,
    sw_list: Option<StorageBuffer>,
    sw_args: Option<StorageBuffer>,
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

        // The integrated path is SW-only (compute cut / clear / raster + a full-screen resolve):
        // it needs 64-bit buffer atomics for the visibility buffer, NOT mesh shaders. (Gating on
        // mesh_shader was a Metal-ism — that backend reports both together; DX/VK expose them
        // independently, and the mesh-shader HW path is Track B / I4.)
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
        // appending each into the CONSOLIDATED global arrays with its base offsets baked in. This
        // keeps the whole scene at 4 storage-buffer slots (not 4 × mesh count) so Sponza-class
        // scenes (100s of meshes) don't overflow the bindless table.
        let (mut gvtx, mut gremap, mut gtri, mut grec) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
            let page = Self::append_page(&mut gvtx, &mut gremap, &mut gtri, &mut grec, &dag);
            max_clusters = max_clusters.max(page.total_clusters);
            let idx = pages.len();
            pages.push(page);
            mesh_to_page.insert(Rc::as_ptr(&registry.get(handle)) as usize, idx);
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
            "P14_VGEO: {} cluster page(s), {} total clusters (consolidated into 4 buffers, max {} \
             clusters/page)",
            pages.len(),
            grec.len() / 96,
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
            100,
            [64, 1, 1],
        )?;
        let raster_pipeline = compute(
            dreamcoast_shader::vgeo_swraster_cs_spirv,
            dreamcoast_shader::vgeo_swraster_cs_dxil,
            dreamcoast_shader::vgeo_swraster_cs_metallib,
            "csRaster",
            100,
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
        let bin_px: f32 = std::env::var("VGEO_BINPX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16.0);

        // M5b HW/SW binning (opt-in). Needs mesh shaders for the HW path; if requested on an adapter
        // without them, warn and stay SW-only rather than failing the whole vgeo path.
        let bin_requested = std::env::var("P14_VGEO_BIN").ok().as_deref() == Some("1");
        let bin = if bin_requested {
            if !device.capabilities().mesh_shader {
                tracing::warn!(
                    "P14_VGEO_BIN: {backend:?} lacks mesh shaders — staying SW-only (large-triangle \
                     clusters may be dropped)"
                );
                None
            } else {
                let cut_bin = compute(
                    dreamcoast_shader::vgeo_cut_bin_cs_spirv,
                    dreamcoast_shader::vgeo_cut_bin_cs_dxil,
                    dreamcoast_shader::vgeo_cut_bin_cs_metallib,
                    "csCutBin",
                    224,
                    [64, 1, 1],
                )?;
                let (hms, hfs) = match backend {
                    BackendKind::Vulkan => (
                        dreamcoast_shader::vgeo_hwvis_ms_spirv(),
                        dreamcoast_shader::vgeo_hwvis_fs_spirv(),
                    ),
                    BackendKind::D3d12 => (
                        dreamcoast_shader::vgeo_hwvis_ms_dxil(),
                        dreamcoast_shader::vgeo_hwvis_fs_dxil(),
                    ),
                    BackendKind::Metal => (
                        dreamcoast_shader::vgeo_hwvis_ms_metallib(),
                        dreamcoast_shader::vgeo_hwvis_fs_metallib(),
                    ),
                };
                let hwvis = device.create_mesh_pipeline(&MeshPipelineDesc {
                    object_bytes: None,
                    object_entry: "",
                    mesh_bytes: hms
                        .ok_or_else(|| anyhow::anyhow!("vgeo_hwvis mesh shader unavailable"))?,
                    mesh_entry: "meshMain",
                    fragment_bytes: hfs
                        .ok_or_else(|| anyhow::anyhow!("vgeo_hwvis fragment shader unavailable"))?,
                    fragment_entry: "fragMain",
                    // Throwaway color target (the fragment's atomicMax writes the visibility buffer;
                    // its colour output is unused). Matches the scratch target `record_object` makes.
                    color_formats: &[crate::GB_ALBEDO_FMT],
                    depth_format: None,
                    push_constant_size: 164, // mvp (64) + 8 u32 (32) + cull_mvp (64) + cull flag (4)
                    bindless: true,
                    uniform_buffer: false,
                    object_threads: [1, 1, 1],
                    mesh_threads: [128, 1, 1],
                    depth_test: false,
                    depth_write: false,
                    depth_compare: DepthCompare::Less,
                })?;
                tracing::info!(
                    "P14_VGEO_BIN: HW/SW binning enabled (split at {bin_px}px diameter)"
                );
                Some(BinPipelines { cut_bin, hwvis })
            }
        } else {
            None
        };

        Ok(Self {
            geom,
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
            bin,
            bin_px,
        })
    }

    /// Append a cluster DAG into the consolidated global arrays, baking this page's base offsets
    /// into the indices so the records point into the shared buffers directly:
    ///   * `remap` values (indices into the vertex pool) += the page's global vertex base,
    ///   * each record's `vertex_offset` (into remap) += the global remap base,
    ///   * each record's `tri_offset` (into triangles) += the global triangle base.
    ///
    /// The triangle values are LOCAL per-cluster indices (into the cluster's emitted vertices), so
    /// they need no offset. Returns the page's global record range.
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
        let binning = self.bin.is_some();
        let list_bytes = (self.max_clusters.max(1) as u64) * 4;
        // Host-visible (not DEFAULT-heap `_init`): host-`write`-reset every frame below, which a
        // DEFAULT-heap D3D12 buffer rejects. CUSTOM-L0 / HOST_COHERENT is CPU-writable AND a UAV, so
        // the cut compute writes it on the GPU too.
        let host = |bytes: u64, indirect: bool| -> anyhow::Result<StorageBuffer> {
            Ok(device.create_storage_buffer_host(&StorageBufferDesc {
                size: bytes,
                stride: 4,
                indirect,
            })?)
        };
        let slots = &mut self.pool[fif];
        while slots.len() < count {
            let (sw_list, sw_args) = if binning {
                (Some(host(list_bytes, false)?), Some(host(12, false)?))
            } else {
                (None, None)
            };
            slots.push(ObjScratch {
                hw_list: host(list_bytes, false)?,
                // `hw_args` feeds `draw_mesh_tasks_indirect` in binning mode → needs INDIRECT usage.
                hw_args: host(12, binning)?,
                sw_list,
                sw_args,
            });
        }
        // Reset each in-use slot: append counters to `{0,1,1}` (the indirect grid — `csCut`/`csCutBin`
        // bump `.x` to the count; the mesh/SW draw reads `{count,1,1}`), lists to the `0xFFFFFFFF`
        // sentinel (the fixed-count SW raster dispatch skips slots the cut didn't fill).
        let args0: Vec<u8> = [0u32, 1, 1].iter().flat_map(|w| w.to_le_bytes()).collect();
        let sentinel = vec![0xFFu8; list_bytes as usize];
        for s in slots.iter().take(count) {
            s.hw_args.write(&args0)?;
            s.hw_list.write(&sentinel)?;
            if let (Some(sw_list), Some(sw_args)) = (&s.sw_list, &s.sw_args) {
                sw_args.write(&args0)?;
                sw_list.write(&sentinel)?;
            }
        }
        Ok(())
    }

    /// The shared visibility buffer's ordering handle (imported once per frame so every object's
    /// passes serialize on it).
    pub(crate) fn import_visbuf(graph: &mut RenderGraph) -> ResourceId {
        graph.import_external("vgeo_visbuf")
    }

    /// Record one object's virtual-geometry G-buffer contribution (cut → raster → resolve overlay).
    /// `slot` indexes this frame's scratch pool; `visbuf_ext` is the shared handle from
    /// [`Self::import_visbuf`]. The mesh fill must run first (it clears the G-buffer + depth); the
    /// resolve LOADs them and Less-tests the model's `SV_Depth`, compositing with the rest.
    ///
    /// SW-only (`P14_VGEO_BIN` off): `csCut` → clear + `csRaster` the whole cut. Binning
    /// (`P14_VGEO_BIN=1` + mesh shaders): `csCutBin` splits the cut into a SW sub-list (compute
    /// raster) and an HW sub-list (mesh shader, `vgeo_hwvis`); both write the SAME visibility buffer
    /// so the boundary is seamless — the fix for large-triangle (Sponza-class) clusters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_object<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gbuf: GBufferTargets,
        fif: usize,
        slot: usize,
        page_idx: usize,
        visbuf_ext: ResourceId,
        cull_view_proj: Mat4,
        view_proj: Mat4,
        eye: Vec3,
        model: Mat4,
        material: VgeoMaterial,
        extent: Extent2D,
    ) {
        let page = &self.pages[page_idx];
        let visbuf = &self.visbuf[fif];
        let s = &self.pool[fif][slot];
        let (hw_list, hw_args) = (&s.hw_list, &s.hw_args);
        let (w, h) = (extent.width, extent.height);
        let total = page.total_clusters;
        // Global cluster index of this page's first cluster: the cut adds it to the local index so
        // the shared record buffer is addressed correctly; the visible list then carries GLOBAL
        // cluster indices, which the raster / mesh / resolve read against the shared buffers.
        let rec_base = page.rec_base;
        let tau = self.tau;
        let bin_px = self.bin_px;
        // Consolidated geometry: the same 4 shared buffers for every object (offsets baked in).
        let (vtx_i, remap_i, tri_i, rec_i) = (
            self.geom.vtx.storage_index(),
            self.geom.remap.storage_index(),
            self.geom.tri.storage_index(),
            self.geom.rec.storage_index(),
        );

        let hw_list_ext = graph.import_external("vgeo_vislist");
        let hw_args_ext = graph.import_external("vgeo_args");

        // ── LOD cut (world-space): WORLD frustum planes + cam; the shader transforms each cluster's
        // local bounds by `model` (handles non-uniform node scale that a local-space test skews).
        // Binning (`csCutBin`) additionally classifies each selected cluster by projected screen
        // size into the HW (`hw_list`) or SW (`sw_list`) sub-list. ──
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
        // The SW list rasterized in compute (SW-only: the whole cut = hw_list; binning: the SW
        // sub-list); its ordering handle.
        let (sw_list, sw_args) = match (&s.sw_list, &s.sw_args) {
            (Some(l), Some(a)) => (l, a),
            _ => (hw_list, hw_args),
        };
        let sw_list_ext = if self.bin.is_some() {
            graph.import_external("vgeo_sw_list")
        } else {
            hw_list_ext
        };
        let sw_args_ext = if self.bin.is_some() {
            graph.import_external("vgeo_sw_args")
        } else {
            hw_args_ext
        };
        let cut = self
            .bin
            .as_ref()
            .map(|b| &b.cut_bin)
            .unwrap_or(&self.cut_pipeline);
        let bin = self.bin.is_some();
        let mut cut_writes = vec![hw_list_ext, hw_args_ext];
        if bin {
            cut_writes.push(sw_list_ext);
            cut_writes.push(sw_args_ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_cut",
                storage_writes: cut_writes,
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
                for (i, word) in [
                    total,
                    rec_i,
                    hw_list.storage_index(),
                    hw_args.storage_index(),
                ]
                .iter()
                .enumerate()
                {
                    cpc[116 + i * 4..120 + i * 4].copy_from_slice(&word.to_le_bytes());
                }
                if bin {
                    cpc[132..136].copy_from_slice(&sw_list.storage_index().to_le_bytes());
                    cpc[136..140].copy_from_slice(&sw_args.storage_index().to_le_bytes());
                    cpc[140..144].copy_from_slice(&bin_px.to_le_bytes());
                }
                for (i, v) in model_arr.iter().enumerate() {
                    cpc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
                }
                cpc[208..212].copy_from_slice(&max_scale.to_le_bytes());
                cpc[212..216].copy_from_slice(&rec_base.to_le_bytes());
                let cmd = ctx.cmd();
                // `hw_args` feeds the HW mesh draw's indirect grid (binning): reset last frame's
                // INDIRECT_ARGUMENT state back to storage before the cut UAV-writes the count.
                if bin {
                    cmd.storage_buffer_to_storage(hw_args);
                }
                cmd.bind_compute_pipeline(cut);
                cmd.push_constants_compute(&cpc);
                cmd.dispatch(total.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(hw_list);
                if bin {
                    cmd.storage_buffer_barrier(sw_list);
                    // The count feeds `draw_mesh_tasks_indirect`.
                    cmd.storage_buffer_to_indirect(hw_args);
                }
                Ok(())
            },
        );

        // ── Clear the R64 visibility buffer, then SW-raster the SW cut into it ──
        // Use the FLIP-FREE `cull_view_proj` (not `view_proj`): the SW raster and the resolve both
        // do MANUAL `(0.5 - ndc.y*0.5)` NDC→pixel mapping (the D3D/Metal window convention). On
        // Vulkan `view_proj` carries the clip-space Y-flip that fixes the *hardware* pipeline, which
        // would vertically flip this manual mapping — the visbuf rows would then mismatch the
        // resolve's `SV_Position` reads. `cull_view_proj = proj_noflip * view` is identical DX≡VK.
        let mvp = (cull_view_proj * model).to_cols_array();
        // Per-triangle backface cull for single-sided (glTF `doubleSided=false`) materials (both the
        // SW raster and the HW mesh shader use it). Two-sided materials keep both faces = CULL_NONE.
        let cull_backface: u32 = (!material.two_sided) as u32;
        let clear = &self.clear_pipeline;
        let raster = &self.raster_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vgeo_raster",
                storage_writes: vec![visbuf_ext],
                reads: vec![sw_args_ext, sw_list_ext],
            },
            move |ctx| {
                let mut spc = [0u8; 100];
                for (i, v) in mvp.iter().enumerate() {
                    spc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                for (i, word) in [
                    vtx_i,
                    remap_i,
                    tri_i,
                    rec_i,
                    visbuf.storage_index(),
                    sw_list.storage_index(),
                    w,
                    h,
                    cull_backface,
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

        // ── HW mesh-vis (binning only): rasterize the HW sub-list into the SAME visibility buffer
        // via the hardware mesh pipeline + fragment atomicMax. It renders to a throwaway color
        // target (colour unused) and uses the Y-flipped `view_proj` — the HW rasterizer + fragment
        // `SV_Position` land the exact screen pixels the flip-free SW raster wrote, so HW and SW
        // agree per pixel (DX≡VK). A trailing barrier pass orders its fragment writes before the
        // resolve reads (`storage_buffer_barrier` covers the FRAGMENT stage). ──
        if let Some(binp) = self.bin.as_ref() {
            let hwvis = &binp.hwvis;
            let hmvp = (view_proj * model).to_cols_array();
            // Flip-free mvp for the mesh shader's DX≡VK-consistent backface test (see vgeo_hwvis).
            let hcmvp = (cull_view_proj * model).to_cols_array();
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
                    let mut hpc = [0u8; 164];
                    for (i, v) in hmvp.iter().enumerate() {
                        hpc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    for (i, word) in [
                        vtx_i,
                        remap_i,
                        tri_i,
                        rec_i,
                        hw_list.storage_index(),
                        visbuf.storage_index(),
                        w,
                        h,
                    ]
                    .iter()
                    .enumerate()
                    {
                        hpc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
                    }
                    for (i, v) in hcmvp.iter().enumerate() {
                        hpc[96 + i * 4..100 + i * 4].copy_from_slice(&v.to_le_bytes());
                    }
                    hpc[160..164].copy_from_slice(&cull_backface.to_le_bytes());
                    let cmd = ctx.cmd();
                    cmd.bind_mesh_pipeline(hwvis);
                    cmd.push_constants_mesh(&hpc);
                    cmd.draw_mesh_tasks_indirect(hw_args, 0);
                    Ok(())
                },
            );
            // Order the HW-vis fragment atomicMax writes before the resolve's fragment reads. A
            // graphics pass can't issue a UAV barrier mid-render-pass, so a tiny compute pass does.
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
