//! Virtual shadow maps — V2: persistent page cache + clipmap scrolling on the V1 static
//! core (docs/vsm-shadows-plan.md; UE mechanism notes in docs/research/vsm_*.txt).
//!
//! One directional sun = [`VSM_LEVELS`] clipmap levels; level i is an 8k²-virtual ortho
//! map (64×64 pages × 128² texels) covering radius `2^(i+2)` m around the camera focus,
//! each level's center snapped to its own page grid. Physical pages PERSIST across
//! frames: the page table is rebuilt each frame from per-page metadata with the host's
//! per-level scroll offset applied, so camera motion slides cached pages instead of
//! invalidating them; only new / invalidated pages carry `VSM_PTE_RENDER`, and the depth
//! raster's fragment stage writes nowhere else — a static scene's steady state
//! re-renders ZERO shadow texels (`stats().0 == 0`).
//!
//! Depth stability across scrolling: each level's view is built at an along-sun Z pinned
//! when the level (re)based — XY snaps in whole pages keep clip XY consistent, the
//! pinned Z keeps stored depths comparable (UE's ViewCenterZ pinning). The pin rebases —
//! invalidating the level — only when the focus drifts a full pushback past it, and a
//! sun-direction change invalidates every level (the UE light cache key).
//!
//! Everything lives in bindless storage buffers (no image atomics — the plan §4 Metal
//! risk reduces to buffer atomics). Opt-in via `VSM=1`; legacy/CSM untouched when off.

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
/// Cached pages unreferenced for this many frames are recycled (UE ages at 1000; the
/// dungeon's working set is small, so a tighter clock keeps the free list warm).
const VSM_MAX_PAGE_AGE: u32 = 120;
/// Movers reported to the GPU per frame (old + new footprint spheres); past the cap the
/// frame degrades to a full invalidate. Mirrors `VSM_MAX_INVAL` in vsm_common.slang.
const VSM_MAX_INVAL: usize = 64;
/// Per-frame constants blob: 6 mat4 + 6 float4 params + uint4 misc + float4 origin +
/// 6 int4 scroll entries + the invalidation header/spheres + the SMRT float4. Mirrors
/// the `VSM_CONST_*` offsets in vsm_common.slang.
const VSM_CONST_SIZE: usize = VSM_LEVELS * 64
    + VSM_LEVELS * 16
    + 16
    + 16
    + VSM_LEVELS * 16
    + 16
    + VSM_MAX_INVAL * 16
    + 16
    + 16; // diag tail: [0] coarse-fallback pixel count (see vsm_common.slang)

/// Per-level CPU cache key: where the level's snapped origin sits in its own page space,
/// and the along-sun component its depth basis is pinned to.
#[derive(Clone, Copy, Default)]
struct LevelState {
    page_loc: [i64; 2],
    pinned_along: f32,
    valid: bool,
}

/// Last frame's caster fingerprint (V3 invalidation): transform bits + world sphere.
#[derive(Clone, Copy)]
struct PrevCaster {
    key: [u32; 16],
    sphere: [f32; 4],
}

pub(crate) struct VsmSystem {
    clear_pipeline: ComputePipeline,
    mark_pipeline: ComputePipeline,
    update_pipeline: ComputePipeline,
    compact_pipeline: ComputePipeline,
    alloc_pipeline: ComputePipeline,
    depth_pipeline: GraphicsPipeline,
    depth_skinned_pipeline: GraphicsPipeline,
    depth_morphed_pipeline: GraphicsPipeline,
    /// Per-frame-in-flight host-written constants (level matrices / params / scroll).
    consts: Vec<StorageBuffer>,
    table: StorageBuffer,
    request: StorageBuffer,
    pool: StorageBuffer,
    /// [0] rendered pages (stat), [4] overflow (stat), [12] virgin high-water (persists).
    counter: StorageBuffer,
    /// Physical-page metadata (virtual address / last-request frame / valid).
    meta: StorageBuffer,
    /// Recycled physical pages: [0] count, then indices.
    freelist: StorageBuffer,
    /// This frame's flip-free level world→clip matrices (raster push consumption).
    level_mats: [Mat4; VSM_LEVELS],
    /// `1 / half` per level — the world→NDC radius scale the V1.5 raster cull uses.
    level_inv_half: [f32; VSM_LEVELS],
    levels: [LevelState; VSM_LEVELS],
    sun_key: [u32; 3],
    frame_no: u32,
    /// Coarse-fallback pixels harvested from the last cycled consts slot (the diag
    /// tail): >0 on a frame whose receivers found their marked level unmapped — the
    /// visible one-frame "shadow enlarges" pop. Exposed to DIAG_FRAME_CSV.
    last_fallback_px: u32,
    /// Last frame's shadow casters, scene order (V3 mover detection).
    prev_casters: Vec<PrevCaster>,
    /// `VSM_BOUNDS_CULL=0` seam: disable the render-bounds whole-caster cull (A/B
    /// fallback, the rule every non-trivial lever ships with).
    bounds_cull: bool,
    /// SMRT filter config (V4): rays (0 = 3x3 PCF fallback), samples per ray, ray length
    /// as a fraction of the receiver's camera distance. Env `VSM_SMRT_RAYS` /
    /// `VSM_SMRT_SAMPLES` / `VSM_SMRT_LEN`; rays/samples default from the quality tier,
    /// length defaults 0.3 (recalibrated — see `new`'s comment; 1.5 detached contact
    /// shadows by starving the contact zone of quadratic samples).
    smrt: [f32; 3],
}

impl VsmSystem {
    /// Build the system, or `None` where compute is unavailable.
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
        smrt_defaults: (u32, u32),
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
                push_constant_size: 48,
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
        let update_pipeline = compute(
            dreamcoast_shader::vsm_update_cs_spirv,
            dreamcoast_shader::vsm_update_cs_dxil,
            dreamcoast_shader::vsm_update_cs_metallib,
            "csUpdate",
            [64, 1, 1],
        )?;
        let compact_pipeline = compute(
            dreamcoast_shader::vsm_compact_cs_spirv,
            dreamcoast_shader::vsm_compact_cs_dxil,
            dreamcoast_shader::vsm_compact_cs_metallib,
            "csCompact",
            [64, 1, 1],
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
        let storage = |size: u64| -> anyhow::Result<StorageBuffer> {
            Ok(device.create_storage_buffer(&StorageBufferDesc {
                size,
                stride: 4,
                indirect: false,
            })?)
        };
        let table = storage(entries * 4)?;
        let request = storage(entries * 4)?;
        let pool = storage((VSM_POOL_PAGES * VSM_PAGE_SIZE * VSM_PAGE_SIZE) as u64 * 4)?;
        // Meta + freelist are the PERSISTENT cache state and must start ZEROED (no valid
        // pages, empty free list) — a device-local buffer's initial contents are
        // undefined, and garbage VALID flags / a garbage free-list count hand out
        // colliding physical pages (the V2 bring-up bug: patchy missing shadows as the
        // clipmap scrolled onto freshly allocated pages).
        let zeroed = |size: usize| -> anyhow::Result<StorageBuffer> {
            Ok(device.create_storage_buffer_init(
                &StorageBufferDesc {
                    size: size as u64,
                    stride: 4,
                    indirect: false,
                },
                &vec![0u8; size],
            )?)
        };
        let meta = zeroed(VSM_POOL_PAGES as usize * 16)?;
        let freelist = zeroed(4 + VSM_POOL_PAGES as usize * 4)?;
        // Host-visible so `stats()` can read the counters (the HZB cull-stats pattern:
        // GPU atomics on a host-coherent buffer). [12] = virgin high-water, PERSISTS —
        // it and the free list are the cache's allocator state.
        // 16 B stats/cursor + VSM_LEVELS x 16 B RENDER-page bounds rows (the depth
        // VS's whole-caster cull key — see VSM_BOUNDS_BASE in vsm_common.slang).
        let counter = device.create_storage_buffer_host(&StorageBufferDesc {
            size: (16 + VSM_LEVELS * 16) as u64,
            stride: 4,
            indirect: false,
        })?;
        counter.write(&[0u8; 16 + VSM_LEVELS * 16])?;
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
            update_pipeline,
            compact_pipeline,
            alloc_pipeline,
            depth_pipeline,
            depth_skinned_pipeline,
            depth_morphed_pipeline,
            consts,
            table,
            request,
            pool,
            counter,
            meta,
            freelist,
            level_mats: [Mat4::IDENTITY; VSM_LEVELS],
            level_inv_half: [0.0; VSM_LEVELS],
            levels: [LevelState::default(); VSM_LEVELS],
            sun_key: [0; 3],
            frame_no: 0,
            last_fallback_px: 0,
            prev_casters: Vec::new(),
            bounds_cull: std::env::var("VSM_BOUNDS_CULL").ok().as_deref() != Some("0"),
            smrt: {
                let f = |k: &str, d: f32| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(d)
                };
                // Defaults come from the quality tier (the V4 landing's deferred
                // "quality.rs 티어" knob — SMRT is the lighting pass's biggest term,
                // measured 6.2→3.6 ms going 7x8→4x6 on the Apple tier); the env vars
                // stay the per-run override they always were.
                [
                    f("VSM_SMRT_RAYS", smrt_defaults.0 as f32).clamp(0.0, 16.0),
                    f("VSM_SMRT_SAMPLES", smrt_defaults.1 as f32).clamp(1.0, 32.0),
                    // 0.3 x view depth (recalibrated from 1.5): the march budget is a
                    // handful of QUADRATIC samples, and a ray reaching 150% of the
                    // camera distance (~24 m in play) leaves ONE sample inside the
                    // whole contact zone — character shadows detached from the feet
                    // (measured; PCF attached, len 0.3 attached, wall penumbrae
                    // unchanged). 0.3 still reaches ~5 m of occluders at the game
                    // camera — the dungeon's whole occluder scale.
                    f("VSM_SMRT_LEN", 0.3).clamp(0.01, 16.0),
                ]
            },
        }))
    }

    /// Level i's coverage radius in metres (receivers within it select the level).
    fn level_radius(i: usize) -> f32 {
        (1u32 << (i + 2)) as f32
    }

    /// Rebuild this frame's level matrices around `focus`, compute each level's page
    /// scroll (or invalidation) against the cached state, and host-write the constants.
    /// Returns the (consts, table, pool) bindless storage indices for the lighting push.
    pub(crate) fn update(
        &mut self,
        fif: usize,
        sun_dir: [f32; 3],
        focus: Vec3,
        scene: &[SceneObject],
    ) -> anyhow::Result<(u32, u32, u32)> {
        self.frame_no = self.frame_no.wrapping_add(1);

        // V3 — the track's core: movers invalidate only the pages they overlap. Diff
        // every caster's transform against last frame; a change contributes its OLD and
        // NEW footprint spheres (the UE double-footprint rule — miss either and you get
        // a stale shadow or a shadowless mover). Pose-deforming casters (skin / morph /
        // vertex-cache) re-render their pages every frame (UE HasDeformableMesh policy);
        // caster-set changes or sphere overflow degrade to a full invalidate.
        let mut inval: Vec<[f32; 4]> = Vec::new();
        let mut force_invalidate = false;
        let mut cur = Vec::with_capacity(self.prev_casters.len().max(16));
        for obj in scene {
            if !obj.casts_shadow {
                continue;
            }
            let mut key = [0u32; 16];
            for (k, f) in key.iter_mut().zip(obj.transform.to_cols_array()) {
                *k = f.to_bits();
            }
            let c = (obj.world_aabb[0] + obj.world_aabb[1]) * 0.5;
            let r = (obj.world_aabb[1] - obj.world_aabb[0]).length() * 0.5;
            cur.push(PrevCaster {
                key,
                sphere: [c.x, c.y, c.z, r],
            });
        }
        if self.prev_casters.len() != cur.len() {
            // Casters appeared/disappeared (level swap, prop pickup): re-render it all.
            force_invalidate = !self.prev_casters.is_empty();
        } else {
            let mut deforming_idx = 0usize;
            for obj in scene {
                if !obj.casts_shadow {
                    continue;
                }
                let (p, n) = (&self.prev_casters[deforming_idx], &cur[deforming_idx]);
                let deforming = obj.skin.is_some() || obj.morph.is_some() || obj.deform.is_some();
                if p.key != n.key {
                    if inval.len() + 2 > VSM_MAX_INVAL {
                        force_invalidate = true;
                        break;
                    }
                    inval.push(p.sphere);
                    inval.push(n.sphere);
                } else if deforming {
                    if inval.len() + 1 > VSM_MAX_INVAL {
                        force_invalidate = true;
                        break;
                    }
                    inval.push(n.sphere);
                }
                deforming_idx += 1;
            }
        }
        self.prev_casters = cur;
        // A rigid-rig character is MANY SceneObjects (one per limb/plate), all moving
        // every animated frame — pushing one sphere pair per part blows the GPU budget
        // (7 characters overflowed 64 and degraded to full invalidates in bring-up).
        // Greedy-merge near/overlapping spheres first: parts of one rig collapse to
        // roughly one sphere, and distinct movers stay separate.
        let mut merged: Vec<[f32; 4]> = Vec::new();
        'sphere: for s in inval {
            for m in merged.iter_mut() {
                let d =
                    ((s[0] - m[0]).powi(2) + (s[1] - m[1]).powi(2) + (s[2] - m[2]).powi(2)).sqrt();
                if d < s[3] + m[3] + 1.0 {
                    // Enclosing sphere of the pair.
                    let r = ((d + s[3] + m[3]) * 0.5).max(s[3]).max(m[3]);
                    if d > 1e-4 {
                        let t = ((r - m[3]) / d).clamp(0.0, 1.0);
                        m[0] += (s[0] - m[0]) * t;
                        m[1] += (s[1] - m[1]) * t;
                        m[2] += (s[2] - m[2]) * t;
                    }
                    m[3] = r;
                    continue 'sphere;
                }
            }
            merged.push(s);
        }
        let mut inval = merged;
        if inval.len() > VSM_MAX_INVAL {
            force_invalidate = true;
        }
        if force_invalidate {
            inval.clear();
        }
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
        // A sun-direction change invalidates every level: cached depths were built in the
        // old light basis (the UE per-light cache key).
        let sun_key = [
            sun_dir[0].to_bits(),
            sun_dir[1].to_bits(),
            sun_dir[2].to_bits(),
        ];
        let sun_moved = sun_key != self.sun_key;
        self.sun_key = sun_key;

        let mut bytes = vec![0u8; VSM_CONST_SIZE];
        fn put(bytes: &mut [u8], off: usize, v: f32) {
            bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        fn put_u(bytes: &mut [u8], off: usize, v: u32) {
            bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        fn put_i(bytes: &mut [u8], off: usize, v: i32) {
            bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        let scroll_base = VSM_LEVELS * 64 + VSM_LEVELS * 16 + 16 + 16;
        for i in 0..VSM_LEVELS {
            let radius = Self::level_radius(i);
            let half = radius * 2.0; // ortho spans ±2R: casters around the receiver ring
            let page_world = (half * 2.0) / VSM_TABLE_DIM as f32;
            // The level's location in its own page space (whole pages; i64 so a far-off
            // world position cannot wrap the arithmetic).
            let lx = focus.dot(right);
            let ly = focus.dot(lup);
            let page_loc = [
                (lx / page_world).round() as i64,
                (ly / page_world).round() as i64,
            ];
            let along = focus.dot(dir);

            let st = &mut self.levels[i];
            let mut invalidate = sun_moved || force_invalidate || !st.valid;
            let mut scroll = [0i32, 0i32];
            if !invalidate {
                if (along - st.pinned_along).abs() > half {
                    // Depth-pin guardband: the focus drifted a full pushback past the
                    // basis this level's depths were built in — rebase and re-render.
                    invalidate = true;
                } else {
                    let dx = page_loc[0] - st.page_loc[0];
                    let dy = page_loc[1] - st.page_loc[1];
                    if dx.unsigned_abs() >= VSM_TABLE_DIM as u64
                        || dy.unsigned_abs() >= VSM_TABLE_DIM as u64
                    {
                        invalidate = true; // scrolled the whole table away
                    } else {
                        scroll = [dx as i32, dy as i32];
                    }
                }
            }
            if invalidate {
                st.pinned_along = along;
                st.valid = true;
            }
            st.page_loc = page_loc;

            // Snapped level center: page-grid XY + the PINNED along-sun component, so
            // scroll-only frames change the matrix by whole-page XY translation only and
            // cached depths stay bit-comparable.
            let center = right * (page_loc[0] as f32 * page_world)
                + lup * (page_loc[1] as f32 * page_world)
                + dir * st.pinned_along;
            let eye = center + dir * (half * 2.0);
            let view = Mat4::look_at_rh(eye, center, up);
            // Flip-free ortho on EVERY backend (vsm_common.slang orientation contract —
            // the depth VS applies the Vulkan clip-Y flip itself).
            let proj = Mat4::orthographic_rh(-half, half, -half, half, 0.0, half * 4.0);
            let m = proj * view;
            self.level_mats[i] = m;
            self.level_inv_half[i] = 1.0 / half;
            for (j, f) in m.to_cols_array().iter().enumerate() {
                put(&mut bytes, i * 64 + j * 4, *f);
            }
            let params_off = VSM_LEVELS * 64 + i * 16;
            put(
                &mut bytes,
                params_off,
                (half * 2.0) / VSM_VIRTUAL_SIZE as f32,
            );
            let so = scroll_base + i * 16;
            put_i(&mut bytes, so, scroll[0]);
            put_i(&mut bytes, so + 4, scroll[1]);
            put_u(&mut bytes, so + 8, u32::from(invalidate));
        }
        let misc_off = VSM_LEVELS * 64 + VSM_LEVELS * 16;
        put_u(&mut bytes, misc_off, VSM_LEVELS as u32);
        put_u(&mut bytes, misc_off + 4, VSM_POOL_PAGES);
        put_u(&mut bytes, misc_off + 8, self.frame_no);
        put_u(&mut bytes, misc_off + 12, VSM_MAX_PAGE_AGE);
        let origin_off = misc_off + 16;
        for (j, f) in [focus.x, focus.y, focus.z, 0.0].iter().enumerate() {
            put(&mut bytes, origin_off + j * 4, *f);
        }
        let inval_off = scroll_base + VSM_LEVELS * 16;
        put_u(&mut bytes, inval_off, inval.len() as u32);
        for (i, s) in inval.iter().enumerate() {
            for (j, f) in s.iter().enumerate() {
                put(&mut bytes, inval_off + 16 + i * 16 + j * 4, *f);
            }
        }
        let smrt_off = inval_off + 16 + VSM_MAX_INVAL * 16;
        for (j, f) in self.smrt.iter().enumerate() {
            put(&mut bytes, smrt_off + j * 4, *f);
        }
        // Harvest the diag tail this slot accumulated the last time it was in flight
        // (coarse-fallback pixel count — the "shadow enlarges for a frame" pop signal),
        // then the full-blob write below rezeroes it for this frame's run.
        let mut prev = vec![0u8; VSM_CONST_SIZE];
        self.last_fallback_px = if self.consts[fif].read_into(&mut prev).is_ok() {
            u32::from_le_bytes(
                prev[VSM_CONST_SIZE - 16..VSM_CONST_SIZE - 12]
                    .try_into()
                    .unwrap(),
            )
        } else {
            0
        };
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

    /// Page management: clear → mark (G-buffer world position) → update (scroll/keep/free
    /// the persistent pages) → allocate (new pages from the free list, then virgin).
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
        let meta = &self.meta;
        let freelist = &self.freelist;
        let clear = &self.clear_pipeline;
        let mark = &self.mark_pipeline;
        let update = &self.update_pipeline;
        let compact = &self.compact_pipeline;
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
                    meta.storage_index(),
                    freelist.storage_index(),
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
                cmd.bind_compute_pipeline(update);
                cmd.push_constants_compute(&push);
                cmd.dispatch(VSM_POOL_PAGES.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(table);
                cmd.storage_buffer_barrier(meta);
                cmd.bind_compute_pipeline(compact);
                cmd.push_constants_compute(&push);
                cmd.dispatch(VSM_POOL_PAGES.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(freelist);
                cmd.bind_compute_pipeline(alloc);
                cmd.push_constants_compute(&push);
                cmd.dispatch(entries.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(table);
                cmd.storage_buffer_barrier(pool);
                cmd.storage_buffer_barrier(counter); // bounds rows -> depth VS cull
                Ok(())
            },
        );
    }

    /// Caster rasterization: every shadow caster, once per level, attachment-less over
    /// the full virtual grid (a compute-kind graph pass so the graph doesn't try to bind
    /// attachments; rendering begins/ends manually inside). The PS only writes pages
    /// flagged RENDER, so with the V2 cache a static steady state costs raster/VS work
    /// but zero pool writes. Per-level caster culling is the V1.5 lever if that VS work
    /// shows up in profiles.
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
                        // V1.5 per-level caster culling (the lever the module doc
                        // reserved for "when that VS work shows up in profiles" — it
                        // did: 4.6 ms of every-caster×every-level draws on the dungeon
                        // floor). A caster whose bounding sphere lies outside this
                        // level's ortho window cannot touch any of its pages: the
                        // level matrix maps the window to NDC ±1, so the test is two
                        // ortho dot products against ±(1 + r/half). Conservative in
                        // XY only (the along-sun range is generous by construction),
                        // so a skipped draw is provably contribution-free.
                        let sphere = {
                            let c = (obj.world_aabb[0] + obj.world_aabb[1]) * 0.5;
                            let r = (obj.world_aabb[1] - obj.world_aabb[0]).length() * 0.5;
                            let clip = *mat * c.extend(1.0);
                            let r_ndc = r * self.level_inv_half[level];
                            let margin = 1.0 + r_ndc;
                            if clip.x.abs() > margin || clip.y.abs() > margin {
                                continue;
                            }
                            [clip.x, clip.y, r_ndc]
                        };
                        if obj.skin.is_some() {
                            cmd.bind_graphics_pipeline(&self.depth_skinned_pipeline);
                        } else if obj.morph.is_some() {
                            cmd.bind_graphics_pipeline(&self.depth_morphed_pipeline);
                        }
                        // The level projections are affine (orthographic), so the
                        // matrix' w row is (0,0,0,1) by construction; the VS
                        // reconstructs w = 1 and that dead row carries the caster's
                        // NDC cull sphere instead (vsm_depth.slang PushConstants).
                        let mut mvp = (*mat * obj.transform).to_cols_array();
                        mvp[3] = sphere[0];
                        mvp[7] = sphere[1];
                        // A giant radius makes the VS test always pass — the off seam.
                        mvp[11] = if self.bounds_cull { sphere[2] } else { 1.0e9 };
                        mvp[15] = 1.0;
                        cmd.push_constants(&vsm_depth_push(
                            mvp,
                            obj.tex[0],
                            obj.alpha_cutoff,
                            flip_y as u32,
                            level as u32,
                            table.storage_index(),
                            pool.storage_index(),
                            self.counter.storage_index(),
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

    /// (pages rendered last frame, overflowed requests) — the cache-effectiveness and
    /// pool-sizing diagnostics (`lib.rs` logs them at shutdown).
    /// Coarse-fallback pixels from the last harvested frame (see `last_fallback_px`).
    pub(crate) fn fallback_px(&self) -> u32 {
        self.last_fallback_px
    }

    pub(crate) fn stats(&self) -> (u32, u32) {
        let mut b = [0u8; 16];
        if self.counter.read_into(&mut b).is_err() {
            return (0, 0);
        }
        let freed = u32::from_le_bytes(b[8..12].try_into().unwrap());
        tracing::debug!("VSM cache: {freed} freed last frame");
        (
            u32::from_le_bytes(b[0..4].try_into().unwrap()),
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
        )
    }
}

/// Pack the vsm_pages push block (48 bytes; mirrors `PushConstants` in vsm_pages.slang).
#[allow(clippy::too_many_arguments)]
fn vsm_pages_push(
    consts_buf: u32,
    table_buf: u32,
    request_buf: u32,
    pool_buf: u32,
    counter_buf: u32,
    meta_buf: u32,
    freelist_buf: u32,
    position_index: u32,
    screen_w: u32,
    screen_h: u32,
) -> [u8; 48] {
    let mut pc = [0u8; 48];
    for (i, v) in [
        consts_buf,
        table_buf,
        request_buf,
        pool_buf,
        counter_buf,
        meta_buf,
        freelist_buf,
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
    pc[88..92].copy_from_slice(&counter_buf.to_le_bytes());
    // pc[92..96] = spare (kept so the shader layout is stable)
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
