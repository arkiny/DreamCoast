//! Virtual shadow maps — V1 static core (docs/vsm-shadows-plan.md; UE mechanism notes in
//! docs/research/vsm_*.txt).
//!
//! One directional sun = [`VSM_LEVELS`] clipmap levels; level i is an 8k²-virtual ortho
//! map (64×64 pages × 128² texels) covering radius `2^(i+2)` m around the camera focus,
//! each level's center snapped to its own page grid (V2's cache scrolling needs snap
//! stability, so V1 snaps from day one). Per frame:
//!
//!   vsm_pages (compute):  clear table/requests → mark needed pages from the G-buffer
//!                         world-position → allocate physical pages (linear counter).
//!   vsm_raster (graph compute-kind pass, manual attachment-less raster): every caster,
//!                         once per level, over the full 8k viewport; the fragment stage
//!                         translates virtual→physical through the page table and
//!                         `InterlockedMin`s ortho depth into the pool buffer.
//!   lighting:             `pbr.slang` `vsm_sun_shadow` — level select mirrors marking,
//!                         3×3 receiver-plane PCF with per-tap page translation.
//!
//! Everything lives in bindless storage buffers (no image atomics — the Metal risk in
//! the plan §4 reduces to plain buffer atomics). V1 has NO caching: the table is rebuilt
//! and every requested page re-renders each frame; the physical-page assignment order is
//! GPU-scheduling dependent, but content follows the virtual page wherever it lands, so
//! images stay deterministic and DX≡VK. Opt-in via `VSM=1` (`App::new`); the legacy /
//! CSM paths are untouched when off.

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, ComputePipeline, ComputePipelineDesc, DepthCompare, Device, Extent2D,
    GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, StorageBuffer, StorageBufferDesc,
    VertexLayout,
};

use crate::app::{load_compute_shader, load_shader_pair};
use crate::{FRAMES_IN_FLIGHT, SceneObject};

/// Clipmap level count. Level i covers receivers within `2^(i+2)` m of the origin
/// (4 m .. 128 m). Keep in lockstep with `VSM_LEVELS` in `vsm_common.slang`.
pub(crate) const VSM_LEVELS: usize = 6;
/// Texels per page side (`VSM_PAGE_SIZE` in the shader).
const VSM_PAGE_SIZE: u32 = 128;
/// Pages per level side (`VSM_TABLE_DIM`); virtual resolution = 64 × 128 = 8192.
const VSM_TABLE_DIM: u32 = 64;
const VSM_VIRTUAL_SIZE: u32 = VSM_PAGE_SIZE * VSM_TABLE_DIM;
const VSM_PAGES_PER_LEVEL: u32 = VSM_TABLE_DIM * VSM_TABLE_DIM;
/// Physical pool capacity (`VSM_POOL_PAGES` in the shader): 512 × 128² × 4 B = 32 MiB.
/// The overflow counter (`stats().1`) is the loud signal to grow it.
const VSM_POOL_PAGES: u32 = 512;
/// Per-frame constants blob: 6 mat4 + 6 float4 params + uint4 misc + float4 origin.
const VSM_CONST_SIZE: usize = VSM_LEVELS * 64 + VSM_LEVELS * 16 + 16 + 16;

pub(crate) struct VsmSystem {
    clear_pipeline: ComputePipeline,
    mark_pipeline: ComputePipeline,
    alloc_pipeline: ComputePipeline,
    depth_pipeline: GraphicsPipeline,
    depth_skinned_pipeline: GraphicsPipeline,
    depth_morphed_pipeline: GraphicsPipeline,
    /// Per-frame-in-flight host-written constants (level matrices / params / origin).
    consts: Vec<StorageBuffer>,
    table: StorageBuffer,
    request: StorageBuffer,
    pool: StorageBuffer,
    /// [0] = next free physical page, [1] = overflow count. Host-visible for stats.
    counter: StorageBuffer,
    /// This frame's flip-free level world→clip matrices (raster push consumption).
    level_mats: [Mat4; VSM_LEVELS],
}

impl VsmSystem {
    /// Build the system, or `None` where compute is unavailable.
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Option<Self>> {
        if !compute_supported {
            return Ok(None);
        }
        let compute = |spirv: fn() -> Option<&'static [u8]>,
                       dxil: fn() -> Option<&'static [u8]>,
                       metallib: fn() -> Option<&'static [u8]>,
                       entry: &str,
                       threads: [u32; 3]|
         -> anyhow::Result<ComputePipeline> {
            let cs = load_compute_shader(backend, spirv, dxil, metallib, entry)?;
            Ok(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: entry,
                push_constant_size: 32,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: threads,
            })?)
        };
        let clear_pipeline = compute(
            dreamcoast_shader::vsm_clear_cs_spirv,
            dreamcoast_shader::vsm_clear_cs_dxil,
            dreamcoast_shader::vsm_clear_cs_metallib,
            "csClear",
            [64, 1, 1],
        )?;
        let mark_pipeline = compute(
            dreamcoast_shader::vsm_mark_cs_spirv,
            dreamcoast_shader::vsm_mark_cs_dxil,
            dreamcoast_shader::vsm_mark_cs_metallib,
            "csMark",
            [8, 8, 1],
        )?;
        let alloc_pipeline = compute(
            dreamcoast_shader::vsm_alloc_cs_spirv,
            dreamcoast_shader::vsm_alloc_cs_dxil,
            dreamcoast_shader::vsm_alloc_cs_metallib,
            "csAlloc",
            [64, 1, 1],
        )?;

        let raster = |vs_spirv: fn() -> Option<&'static [u8]>,
                      vs_dxil: fn() -> Option<&'static [u8]>,
                      vs_metallib: fn() -> Option<&'static [u8]>,
                      entry: &str,
                      label: &str|
         -> anyhow::Result<GraphicsPipeline> {
            let (vs, fs) = load_shader_pair(
                backend,
                vs_spirv,
                dreamcoast_shader::vsm_depth_fs_spirv,
                vs_dxil,
                dreamcoast_shader::vsm_depth_fs_dxil,
                vs_metallib,
                dreamcoast_shader::vsm_depth_fs_metallib,
                label,
            )?;
            Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: vs,
                fragment_bytes: fs,
                vertex_entry: entry,
                fragment_entry: "fsMain",
                color_formats: &[], // attachment-less: the PS scatters into the pool buffer
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::MeshPositionUv,
                blend: BlendMode::Opaque,
                push_constant_size: 128,
                bindless: true,
                uniform_buffer: false,
                depth_test: false, // no depth attachment; the pool InterlockedMin IS the test
                depth_write: false,
                depth_compare: DepthCompare::Less,
                depth_format: None,
            })?)
        };
        let depth_pipeline = raster(
            dreamcoast_shader::vsm_depth_vs_spirv,
            dreamcoast_shader::vsm_depth_vs_dxil,
            dreamcoast_shader::vsm_depth_vs_metallib,
            "vsMain",
            "vsm_depth",
        )?;
        let depth_skinned_pipeline = raster(
            dreamcoast_shader::vsm_depth_skinned_vs_spirv,
            dreamcoast_shader::vsm_depth_skinned_vs_dxil,
            dreamcoast_shader::vsm_depth_skinned_vs_metallib,
            "vsMainSkinned",
            "vsm_depth_skinned",
        )?;
        let depth_morphed_pipeline = raster(
            dreamcoast_shader::vsm_depth_morphed_vs_spirv,
            dreamcoast_shader::vsm_depth_morphed_vs_dxil,
            dreamcoast_shader::vsm_depth_morphed_vs_metallib,
            "vsMainMorphed",
            "vsm_depth_morphed",
        )?;

        let entries = (VSM_LEVELS as u32 * VSM_PAGES_PER_LEVEL) as u64;
        let table = device.create_storage_buffer(&StorageBufferDesc {
            size: entries * 4,
            stride: 4,
            indirect: false,
        })?;
        let request = device.create_storage_buffer(&StorageBufferDesc {
            size: entries * 4,
            stride: 4,
            indirect: false,
        })?;
        let pool = device.create_storage_buffer(&StorageBufferDesc {
            size: (VSM_POOL_PAGES * VSM_PAGE_SIZE * VSM_PAGE_SIZE) as u64 * 4,
            stride: 4,
            indirect: false,
        })?;
        // Host-visible so `stats()` can read the allocation/overflow counters (the HZB
        // cull-stats pattern: GPU atomics on a host-coherent buffer).
        let counter = device.create_storage_buffer_host(&StorageBufferDesc {
            size: 16,
            stride: 4,
            indirect: false,
        })?;
        counter.write(&[0u8; 16])?;
        let consts = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                let b = device.create_storage_buffer_host(&StorageBufferDesc {
                    size: VSM_CONST_SIZE as u64,
                    stride: 16,
                    indirect: false,
                })?;
                b.write(&vec![0u8; VSM_CONST_SIZE])?;
                Ok(b)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Some(Self {
            clear_pipeline,
            mark_pipeline,
            alloc_pipeline,
            depth_pipeline,
            depth_skinned_pipeline,
            depth_morphed_pipeline,
            consts,
            table,
            request,
            pool,
            counter,
            level_mats: [Mat4::IDENTITY; VSM_LEVELS],
        }))
    }

    /// Level i's coverage radius in metres (receivers within it select the level).
    fn level_radius(i: usize) -> f32 {
        (1u32 << (i + 2)) as f32
    }

    /// Rebuild this frame's level matrices around `focus` and host-write the constants.
    /// Returns the (consts, table, pool) bindless storage indices for the lighting push.
    pub(crate) fn update(
        &mut self,
        fif: usize,
        sun_dir: [f32; 3],
        focus: Vec3,
    ) -> anyhow::Result<(u32, u32, u32)> {
        // Light basis — identical up-vector guard to `light_view_proj` / the CSM fit.
        let dir = Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]).normalize_or_zero();
        let dir = if dir == Vec3::ZERO { Vec3::Y } else { dir };
        let up = if dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = up.cross(dir).normalize();
        let lup = dir.cross(right).normalize();

        let mut bytes = vec![0u8; VSM_CONST_SIZE];
        let mut put = |off: usize, v: f32| {
            bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        for i in 0..VSM_LEVELS {
            let radius = Self::level_radius(i);
            let half = radius * 2.0; // ortho spans ±2R: casters around the receiver ring
            // Snap the center to this level's page grid in light-plane XY so a moving
            // camera slides the virtual address space in whole pages (V2 cache scrolling
            // scrolls these; V1 just keeps texel assignments stable frame-to-frame).
            let page_world = (half * 2.0) / VSM_TABLE_DIM as f32;
            let lx = focus.dot(right);
            let ly = focus.dot(lup);
            let center = focus
                - right * (lx - (lx / page_world).round() * page_world)
                - lup * (ly - (ly / page_world).round() * page_world);
            let eye = center + dir * (half * 2.0);
            let view = Mat4::look_at_rh(eye, center, up);
            // Flip-free ortho on EVERY backend (vsm_common.slang orientation contract —
            // the depth VS applies the Vulkan clip-Y flip itself).
            let proj = Mat4::orthographic_rh(-half, half, -half, half, 0.0, half * 4.0);
            let m = proj * view;
            self.level_mats[i] = m;
            for (j, f) in m.to_cols_array().iter().enumerate() {
                put(i * 64 + j * 4, *f);
            }
            let params_off = VSM_LEVELS * 64 + i * 16;
            put(params_off, (half * 2.0) / VSM_VIRTUAL_SIZE as f32); // texel size, metres
        }
        let misc_off = VSM_LEVELS * 64 + VSM_LEVELS * 16;
        bytes[misc_off..misc_off + 4].copy_from_slice(&(VSM_LEVELS as u32).to_le_bytes());
        bytes[misc_off + 4..misc_off + 8].copy_from_slice(&VSM_POOL_PAGES.to_le_bytes());
        let origin_off = misc_off + 16;
        for (j, f) in [focus.x, focus.y, focus.z, 0.0].iter().enumerate() {
            bytes[origin_off + j * 4..origin_off + j * 4 + 4].copy_from_slice(&f.to_le_bytes());
        }
        self.consts[fif].write(&bytes)?;
        Ok((
            self.consts[fif].storage_index(),
            self.table.storage_index(),
            self.pool.storage_index(),
        ))
    }

    /// Import the table + pool as external graph resources (once per frame graph); the
    /// pages pass writes them, the raster pass writes the pool, the lighting pass reads.
    pub(crate) fn import(graph: &mut RenderGraph) -> (ResourceId, ResourceId) {
        (
            graph.import_external("vsm_table"),
            graph.import_external("vsm_pool"),
        )
    }

    /// Page management: clear → mark (from the G-buffer world position) → allocate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pages<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        table_ext: ResourceId,
        pool_ext: ResourceId,
        gbuf_position: ResourceId,
        fif: usize,
        screen: (u32, u32),
    ) {
        let consts = &self.consts[fif];
        let table = &self.table;
        let request = &self.request;
        let pool = &self.pool;
        let counter = &self.counter;
        let clear = &self.clear_pipeline;
        let mark = &self.mark_pipeline;
        let alloc = &self.alloc_pipeline;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vsm_pages",
                storage_writes: vec![table_ext, pool_ext],
                reads: vec![gbuf_position],
            },
            move |ctx| {
                let position_index = ctx.sampled_index(gbuf_position);
                let push = vsm_pages_push(
                    consts.storage_index(),
                    table.storage_index(),
                    request.storage_index(),
                    pool.storage_index(),
                    counter.storage_index(),
                    position_index,
                    screen.0,
                    screen.1,
                );
                let entries = VSM_LEVELS as u32 * VSM_PAGES_PER_LEVEL;
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(clear);
                cmd.push_constants_compute(&push);
                cmd.dispatch(entries.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(table);
                cmd.storage_buffer_barrier(request);
                cmd.storage_buffer_barrier(counter);
                cmd.bind_compute_pipeline(mark);
                cmd.push_constants_compute(&push);
                cmd.dispatch(screen.0.div_ceil(8), screen.1.div_ceil(8), 1);
                cmd.storage_buffer_barrier(request);
                cmd.bind_compute_pipeline(alloc);
                cmd.push_constants_compute(&push);
                cmd.dispatch(entries.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(table);
                cmd.storage_buffer_barrier(pool);
                Ok(())
            },
        );
    }

    /// Caster rasterization: every shadow caster, once per level, attachment-less over
    /// the full virtual grid (a compute-kind graph pass so the graph doesn't try to bind
    /// attachments; rendering begins/ends manually inside). V1 draws the whole caster set
    /// into every level — per-level culling is a V1.5 lever once caching (V2) decides how
    /// much raster survives at steady state.
    pub(crate) fn record_raster<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        table_ext: ResourceId,
        pool_ext: ResourceId,
        scene: &'a [SceneObject],
        flip_y: bool,
    ) {
        let table = &self.table;
        let pool = &self.pool;
        let counter = &self.counter;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "vsm_raster",
                storage_writes: vec![pool_ext],
                reads: vec![table_ext],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.begin_rendering_empty(Extent2D::new(VSM_VIRTUAL_SIZE, VSM_VIRTUAL_SIZE));
                cmd.set_viewport_scissor_extent(Extent2D::new(VSM_VIRTUAL_SIZE, VSM_VIRTUAL_SIZE));
                cmd.bind_graphics_pipeline(&self.depth_pipeline);
                for (level, mat) in self.level_mats.iter().enumerate() {
                    for obj in scene {
                        if !obj.casts_shadow {
                            continue;
                        }
                        if obj.skin.is_some() {
                            cmd.bind_graphics_pipeline(&self.depth_skinned_pipeline);
                        } else if obj.morph.is_some() {
                            cmd.bind_graphics_pipeline(&self.depth_morphed_pipeline);
                        }
                        cmd.push_constants(&vsm_depth_push(
                            (*mat * obj.transform).to_cols_array(),
                            obj.tex[0],
                            obj.alpha_cutoff,
                            flip_y as u32,
                            level as u32,
                            table.storage_index(),
                            pool.storage_index(),
                            counter.storage_index(),
                            obj.skin.unwrap_or([0; 4]),
                            obj.morph.unwrap_or([0; 4]),
                        ));
                        cmd.bind_vertex_buffer(&obj.mesh.vbuf, 32);
                        cmd.bind_index_buffer(&obj.mesh.ibuf, true);
                        cmd.draw_indexed(obj.mesh.index_count, 0, 0);
                        if obj.skin.is_some() || obj.morph.is_some() {
                            cmd.bind_graphics_pipeline(&self.depth_pipeline); // restore
                        }
                    }
                }
                cmd.end_rendering();
                cmd.storage_buffer_barrier(pool);
                Ok(())
            },
        );
    }

    /// (allocated pages, overflowed requests) from the last completed frame — the pool
    /// sizing diagnostic (`DIAG` log consumer in `lib.rs`).
    pub(crate) fn stats(&self) -> (u32, u32) {
        let mut b = [0u8; 16];
        if self.counter.read_into(&mut b).is_err() {
            return (0, 0);
        }
        let frag = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let mapped = u32::from_le_bytes(b[12..16].try_into().unwrap());
        tracing::info!("VSM raster probe: {frag} fragments, {mapped} on mapped pages"); // TEMP
        (
            u32::from_le_bytes(b[0..4].try_into().unwrap()),
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
        )
    }
}

/// Pack the vsm_pages push block (32 bytes; mirrors `PushConstants` in vsm_pages.slang).
#[allow(clippy::too_many_arguments)]
fn vsm_pages_push(
    consts_buf: u32,
    table_buf: u32,
    request_buf: u32,
    pool_buf: u32,
    counter_buf: u32,
    position_index: u32,
    screen_w: u32,
    screen_h: u32,
) -> [u8; 32] {
    let mut pc = [0u8; 32];
    for (i, v) in [
        consts_buf,
        table_buf,
        request_buf,
        pool_buf,
        counter_buf,
        position_index,
        screen_w,
        screen_h,
    ]
    .iter()
    .enumerate()
    {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// Pack the vsm_depth push block (128 bytes; mirrors `PushConstants` in vsm_depth.slang).
#[allow(clippy::too_many_arguments)]
fn vsm_depth_push(
    mvp: [f32; 16],
    base_color_tex: u32,
    alpha_cutoff: f32,
    flip_y: u32,
    level: u32,
    table_buf: u32,
    pool_buf: u32,
    counter_buf: u32,
    skin: [u32; 4],
    morph: [u32; 4],
) -> [u8; 128] {
    let mut pc = [0u8; 128];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&base_color_tex.to_le_bytes());
    pc[68..72].copy_from_slice(&alpha_cutoff.to_le_bytes());
    pc[72..76].copy_from_slice(&flip_y.to_le_bytes());
    pc[76..80].copy_from_slice(&level.to_le_bytes());
    pc[80..84].copy_from_slice(&table_buf.to_le_bytes());
    pc[84..88].copy_from_slice(&pool_buf.to_le_bytes());
    pc[88..92].copy_from_slice(&counter_buf.to_le_bytes()); // TEMP PROBE
    // pc[92..96] = _pad
    for (i, v) in skin.iter().enumerate() {
        let o = 96 + i * 4;
        pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in morph.iter().enumerate() {
        let o = 112 + i * 4;
        pc[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}
