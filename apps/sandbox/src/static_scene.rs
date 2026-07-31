//! Per-level **static scene** build — the bake chain the renderer's world-space
//! caches are made of.
//!
//! Everything here is derived from ONE level's geometry and must be rebuilt when the
//! level changes (`App::load_level`), as opposed to the per-DEVICE resources
//! (pipelines, samplers, screen-sized targets, the render-graph pools) built once in
//! `App::new`. Before this module the chain lived inline in `App::new`, so a hot-swap
//! kept the PREVIOUS level's distance fields / surface cache / ray-tracing scene and
//! every distance-field consumer (AO, GI, reflections) read the old world.
//!
//! Three entry points, in the order `App::new` runs them (`App::rebuild` calls the
//! same three, so the startup path and the swap path can never drift):
//!
//! 1. [`build_gdf_scene`] — fuse → per-mesh SDF bakes → clipmap levels → per-mesh SDF
//!    atlas + instance grid → dense albedo volumes → surface-cache cards + capture.
//! 2. [`build_content_accel`] — the content BLAS/TLAS + consolidated hit table.
//! 3. [`install_gi_fine_box`] — the camera-anchored fine level of the SH irradiance
//!    volume.
//!
//! The bodies are the code that used to sit inline; the seam is the parameter block
//! below, which carries the `App::new` locals both callers can supply.

use std::sync::Arc;

use dreamcoast_core::glam::Vec3;
use dreamcoast_scene::World;
use rhi::Device;
use rhi::Format;
use tracing::info;

use crate::gdf::GdfSystem;
use crate::gi::GiSystem;
use crate::loading::LoadingState;
use crate::quality::QualityPreset;
use crate::registry::{MaterialRegistry, MeshRegistry};
use crate::rt::RtSystem;

/// The `App::new` locals the bake chain reads. `App::rebuild` re-derives each from its
/// own fields, so a swap resolves them exactly the way startup did.
pub(crate) struct StaticSceneParams<'a> {
    /// The byte-identical path-tracer anchor scene: keeps the legacy fused bake, the
    /// uniform card atlas and the analytic ground.
    pub gallery_scene: bool,
    /// Chunk-streamed world: no single scene AABB, so no static bake at all.
    pub world_mode: bool,
    /// The level's authored camera (eye, target) — the reference pose card residency
    /// and the GI fine box are ranked from.
    pub level_view: Option<(Vec3, Vec3)>,
    pub scene_center: Vec3,
    pub scene_radius: f32,
    /// The resolved quality tier the load-time knobs default to.
    pub base: &'a QualityPreset,
    /// Loading-screen progress the per-mesh cook advances.
    pub loading_state: &'a Arc<LoadingState>,
}

/// What the static build hands back to the frame loop.
#[derive(Default)]
pub(crate) struct StaticScene {
    /// F1 Stage 3 drawable directory for surface-cache page streaming (empty unless
    /// `P11_CACHE_STREAM`).
    pub stream_obj_aabb: Vec<([f32; 3], [f32; 3])>,
    pub stream_obj_albedo: Vec<[f32; 3]>,
}

/// Fuse the level's opaque draw list and (re)build every world-space distance-field
/// resource from it: the scene GDF + albedo volumes, the finer clipmap levels, the
/// per-mesh SDF atlas / instance grid, and the surface-cache cards + mesh capture.
///
/// A no-op (returning an empty [`StaticScene`]) for a streamed world, a device without
/// the GDF trace pipeline, or an empty draw list — the same gate the inline code used.
pub(crate) fn build_gdf_scene(
    device: &Device,
    gdf: &mut GdfSystem,
    world: &World,
    mesh_registry: &MeshRegistry,
    material_registry: &MaterialRegistry,
    p: &StaticSceneParams,
) -> anyhow::Result<StaticScene> {
    let mut out = StaticScene::default();
    if p.world_mode || !gdf.has_gdf_trace() || world.draw_list().is_empty() {
        return Ok(out);
    }
    let StaticScene {
        stream_obj_aabb,
        stream_obj_albedo,
    } = &mut out;
    let fused = crate::fuse::fuse_scene(world, mesh_registry, material_registry);
    let fused_v = fused.vtx;
    let fused_i = fused.idx;
    let tri_albedo = fused.tri_albedo;
    let amin = fused.aabb_min;
    let amax = fused.aabb_max;
    let tri_count = fused.tri_count;
    // Per-drawable world AABBs + representative albedo (for the surface-cache cards).
    let obj_aabb = fused.drawable_aabb;
    let obj_albedo = fused.drawable_albedo;
    // Phase 12 M2: cook the scene SDF (deterministic CPU bake, cached as a
    // `.dcasset` keyed on the fused geometry + grid) and upload it, replacing
    // the one-time GPU bake. A fresh cache loads directly; a miss bakes + saves.
    let sdf_dim = gdf.scene_dim();
    // Stage B (clipmap): plan the camera-centered level scheme. The gallery is the
    // byte-identical regression reference, so it stays single-level by default
    // (= the legacy 48³ volume). `P11_GDF_CLIP_LEVELS=N` opts into an N-level clipmap
    // (B3 multi-level path verification); the finer levels are cooked over their
    // sub-AABBs (Stage A grid bake, cached) and installed. Default activation for
    // content scenes (Sponza) lands in Stage D.
    let clip_center = [
        (amin[0] + amax[0]) * 0.5,
        (amin[1] + amax[1]) * 0.5,
        (amin[2] + amax[2]) * 0.5,
    ];
    // The gallery stays single-level (byte-identical reference); content scenes
    // (Sponza) default to a 4-level clipmap (auto-trimmed by extent in plan_levels)
    // — the camera-centered clipmap is the default for content, per the design.
    let clip_max_levels = std::env::var("P11_GDF_CLIP_LEVELS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(if p.gallery_scene { 1 } else { 4 })
        .max(1);
    let clip = crate::clipmap::plan_levels(amin, amax, clip_center, sdf_dim, 0.1, clip_max_levels);
    info!("GDF clipmap: {} level(s)", clip.level_count());
    // Stage S1 (per-mesh-distance-fields.md): composite per-mesh DFs (baked once per unique
    // mesh at ~5 cm target voxels, cached/instanced) into each clip level instead of the
    // fused whole-scene triangle-soup bake. Per-mesh is now the **DEFAULT for content**: the
    // fused bake is DEPRECATED — its coarse whole-scene voxels (~0.76 m on a 37 m scene) lose
    // thin features (reliefs, thin walls, tracery), so DF-based passes (GI/AO/reflection +
    // the debug view) march straight through them. The gallery keeps the fused bake (it is
    // the byte-identical anchor and a simple scene where per-mesh buys nothing). The first
    // cook of a non-instanced scene (Intel Sponza ~426 unique meshes) is slower but cached;
    // the win compounds on instanced content (a unique asset bakes once, reused per
    // placement). `P11_PERMESH_GDF=0` forces the deprecated fused path (fallback / A-B).
    // `scene_diag` is the "open space" distance for voxels no object covers.
    let use_permesh = !p.gallery_scene && crate::quality::env_bool("P11_PERMESH_GDF", true);
    if !use_permesh && !p.gallery_scene {
        tracing::warn!(
            "GDF: using the DEPRECATED fused whole-scene distance field (P11_PERMESH_GDF=0). \
                 Thin features (reliefs, thin walls) are lost below the coarse voxel size."
        );
    }
    let scene_diag = {
        let d = [amax[0] - amin[0], amax[1] - amin[1], amax[2] - amin[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let mut mesh_sdfs: Vec<dreamcoast_asset::sdf::SdfVolume> = Vec::new();
    // F5 (gi-fidelity-phases.md): one per-mesh albedo volume per unique mesh, parallel to
    // `mesh_sdfs` (same order / dedup), baked over the SAME grid so the albedo tile aligns
    // 1:1 with the SDF tile. Opt-in (heavy: per-mesh bake + 3 atlas volumes); off ⇒ dense.
    let permesh_albedo = use_permesh && crate::quality::env_bool("F5_PERMESH_ALBEDO", false);
    let mut mesh_albedos: Vec<dreamcoast_asset::sdf::AlbedoVolumes> = Vec::new();
    let mut compose_objects: Vec<crate::compose::ComposeObject> = Vec::new();
    if use_permesh {
        use std::collections::HashMap;
        use std::sync::atomic::{AtomicU32, Ordering};
        let cache_dir = crate::app::cooked_cache_dir();
        // G1 (gdf-reference-alignment.md): small-mesh radius cull — drop tiny drawables
        // from the composite (they barely move low-frequency GI/AO but each is a full
        // per-mesh bake). `P11_GDF_MIN_RADIUS` (m); 0 disables.
        let min_radius = std::env::var("P11_GDF_MIN_RADIUS")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(crate::compose::DEFAULT_MIN_MESH_RADIUS);

        // Sequential dedup + radius-cull pass. It touches the `Rc`-holding registries, so it
        // stays on this thread; it encodes each UNIQUE mesh's bake inputs into OWNED bytes so
        // the parallel bake below borrows no registry. `draw_refs` records every surviving
        // drawable's (world, unique-mesh index) in draw-list order for the compose.
        struct BakeSpec {
            mvtx: Vec<u8>,
            midx: Vec<u8>,
            dims: [u32; 3],
            mn: [f32; 3],
            mx: [f32; 3],
            albedo: Option<([f32; 3], usize)>, // (material colour, triangle count)
        }
        let mut specs: Vec<BakeSpec> = Vec::new();
        let mut draw_refs: Vec<(dreamcoast_core::glam::Mat4, usize)> = Vec::new();
        let mut mesh_index: HashMap<u32, usize> = HashMap::new();
        let mut culled = 0u32;
        for d in world.draw_list() {
            let cpu = mesh_registry.cpu(d.mesh);
            let (mn, mx) = dreamcoast_asset::sdf::mesh_local_aabb_padded(&cpu.vertices);
            if min_radius > 0.0 && crate::compose::mesh_world_radius(d.world, mn, mx) < min_radius {
                culled += 1;
                continue;
            }
            let mi = *mesh_index.entry(d.mesh.0).or_insert_with(|| {
                let i = specs.len();
                // F5: the per-mesh albedo (baked over the SAME grid) keys on the mesh + this
                // first drawable's representative material colour (uniform per drawable here).
                let albedo = permesh_albedo.then(|| {
                    (
                        material_registry.get(d.material).albedo,
                        cpu.indices.len() / 3,
                    )
                });
                specs.push(BakeSpec {
                    mvtx: dreamcoast_asset::sdf::encode_vertices_fused(&cpu.vertices),
                    midx: dreamcoast_asset::sdf::encode_indices(&cpu.indices),
                    dims: dreamcoast_asset::sdf::mesh_sdf_dims(mn, mx),
                    mn,
                    mx,
                    albedo,
                });
                i
            });
            draw_refs.push((d.world, mi));
        }

        // Parallel per-mesh bake on the job-system workers — the expensive step (a full CPU
        // SDF bake per unique mesh; a cache hit just reads the `.dcasset`). Each bake is
        // order-independent + cross-process deterministic, so the parallel result equals the
        // serial one; assembling in `specs` order below keeps the consolidation byte-identical.
        let negated = AtomicU32::new(0);
        let mut baked: Vec<
            Option<(
                dreamcoast_asset::sdf::SdfVolume,
                Option<dreamcoast_asset::sdf::AlbedoVolumes>,
            )>,
        > = (0..specs.len()).map(|_| None).collect();
        // The loading thread owns the window; this phase advances the bar smoothly across
        // its [0.30, 0.60] slice (per-mesh SDF is the bulk of a cold cook) + logs.
        let mut sink = crate::loading::PhaseSink::new(Arc::clone(p.loading_state), 0.30, 0.60);
        crate::cook_progress::parallel_cook(
            "per-mesh SDF",
            &mut baked,
            1,
            |i, slot| {
                let s = &specs[i];
                let (mut vol, _) = dreamcoast_asset::cook::load_or_bake_mesh_sdf(
                    &s.mvtx, &s.midx, s.dims, s.mn, s.mx, &cache_dir,
                );
                // F6H (`P_SDF_OPEN_UNSIGNED`): a non-watertight mesh bakes half-space
                // signs — an open sheet's closest-triangle sign paints the whole half-
                // space behind it "inside", and dozens of curtain/banner AABBs composed
                // that into phantom solid filling the atrium air (the sky-visibility
                // field read ~0 where the reference sees ~0.9; the 50%-negative census
                // is the fingerprint, invisible to the auto-flip below). Such a
                // field cannot carry a sign: take |d| — an unsigned shell still stops
                // the march at the surface band, and the space behind stays open.
                let open_frac = dreamcoast_asset::sdf::mesh_open_fraction(&s.midx);
                // F6I flood-resign (`P_SDF_OPEN_UNSIGNED` seam): a non-watertight
                // mesh's closest-triangle sign paints half-spaces "inside" (the V
                // phantom), while plain |d| loses the sub-voxel zero-crossing and made
                // the roof translucent to the sun (plan 2b/2c — the shell dilemma).
                // Flood-filling air from the padded tile boundary re-signs the phantom
                // air positive and preserves true wall interiors negative; a sheet
                // with no enclosed volume floods all-positive, which we detect and
                // erode by half its thin-axis atlas voxel so the band stays detectable
                // by the marches at any sampling resolution.
                if crate::quality::env_bool("P_SDF_OPEN_UNSIGNED", false)
                    && open_frac > dreamcoast_asset::sdf::OPEN_MESH_BOUNDARY_FRAC
                {
                    let mut vox_min = f32::MAX;
                    for a in 0..3 {
                        let ext = (s.mx[a] - s.mn[a]).max(1e-4);
                        vox_min = vox_min.min(ext / s.dims[a].max(1) as f32);
                    }
                    dreamcoast_asset::sdf::flood_resign(&mut vol, 0.75 * vox_min);
                    let neg = vol.voxels.iter().filter(|&&d| d < 0.0).count();
                    if neg * 200 < vol.voxels.len() {
                        // No enclosed volume at this resolution = a sheet: erode so
                        // the |d| band reaches the march epsilon (F6I plan 2b).
                        let cap = std::env::var("P11_ATLAS_MAX_DIM")
                            .ok()
                            .and_then(|v| v.trim().parse::<u32>().ok())
                            .map(|d| d.clamp(dreamcoast_asset::sdf::MESH_SDF_MIN_DIM, 48))
                            .unwrap_or(32);
                        let mut erode = f32::MAX;
                        for a in 0..3 {
                            let ext = (s.mx[a] - s.mn[a]).max(1e-4);
                            let ad = s.dims[a].min(cap).max(1) as f32;
                            erode = erode.min(0.5 * ext / ad);
                        }
                        for v in &mut vol.voxels {
                            *v -= erode;
                        }
                    }
                }
                // Global-inversion flip. A mesh whose normals point inward bakes an
                // inverted field (open space reads "inside"), which poisons the
                // composite and paints spurious AO/GI blotches, so such a field is
                // negated. Deciding WHICH fields those are has two paths:
                //
                // * legacy (default): "> 60 % of voxels negative". That measures how
                //   much of the AABB reads solid, which for a non-watertight mesh is
                //   dominated by half-space contamination (F6H), not by inversion —
                //   the dungeon's chunk meshes are soups of single-sided floor/wall
                //   quads whose negative fraction lands anywhere in 0.16 … 0.85 with
                //   the room layout, so 2-4 chunks per generated dungeon flip at
                //   random and invert occlusion for the room air the quads face.
                // * `P_SDF_SIGN_PROBE` seam: decide from provably-outside samples
                //   (`sdf::field_is_inverted` — the padded grid's outer shell, with
                //   open meshes never flipped). See that function for the full
                //   argument. It is the correct test, but it is NOT the default: it
                //   disagrees with the legacy test on 14 of Intel Sponza's 426 unique
                //   meshes (15 flips today, 1 under the probe), all of them open
                //   shells where the flip is currently acting as a crude mitigation
                //   for the F6H phantom, so flipping the default moves the byte-anchored
                //   content goldens and needs its own measured, reviewed landing.
                let neg = vol.voxels.iter().filter(|&&d| d < 0.0).count();
                let decision = if crate::quality::env_bool("P_SDF_SIGN_PROBE", false) {
                    dreamcoast_asset::sdf::field_is_inverted(&vol, open_frac)
                } else {
                    neg * 5 > vol.voxels.len() * 3
                };
                if crate::quality::env_bool("DIAG_SDF_SIGN", false) {
                    // Per-mesh sign census (the evidence the seam is judged on):
                    // negative fraction (the legacy signal), boundary-edge fraction
                    // (watertightness) and the provably-outside shell census.
                    tracing::info!(
                        "DF sign: mesh {i} tris {} dims {}x{}x{} neg {:.4} open {:.4} \
                             outside_neg {:.4} -> {}",
                        s.midx.len() / 12,
                        vol.dims[0],
                        vol.dims[1],
                        vol.dims[2],
                        neg as f32 / vol.voxels.len().max(1) as f32,
                        open_frac,
                        dreamcoast_asset::sdf::outside_shell_neg_frac(&vol),
                        if decision { "NEGATE" } else { "keep" },
                    );
                }
                if decision {
                    for v in &mut vol.voxels {
                        *v = -*v;
                    }
                    negated.fetch_add(1, Ordering::Relaxed);
                }
                let alb = s.albedo.map(|(colour, tri_count)| {
                    let mut tri_albedo = Vec::with_capacity(tri_count * 12);
                    for _ in 0..tri_count {
                        for c in colour {
                            tri_albedo.extend_from_slice(&c.to_le_bytes());
                        }
                    }
                    let (av, _) = dreamcoast_asset::cook::load_or_bake_mesh_albedo(
                        &s.mvtx,
                        &s.midx,
                        &tri_albedo,
                        s.dims,
                        s.mn,
                        s.mx,
                        &cache_dir,
                    );
                    av
                });
                *slot = Some((vol, alb));
            },
            &mut sink,
        );

        // Assemble in unique-mesh (specs) order — identical to the original first-seen order,
        // so `mesh_sdfs` / `mesh_albedos` and the composite are byte-for-byte the serial bake.
        for b in baked {
            let (vol, alb) = b.expect("every spec baked");
            mesh_sdfs.push(vol);
            if let Some(av) = alb {
                mesh_albedos.push(av);
            }
        }
        for &(world_m, mi) in &draw_refs {
            compose_objects.push(crate::compose::ComposeObject::new(
                world_m,
                mi,
                &mesh_sdfs[mi],
            ));
        }
        info!(
            "per-mesh DF: {} unique meshes, {} instances ({} culled < {:.2} m radius, \
                 {} inverted-meshes negated)",
            mesh_sdfs.len(),
            compose_objects.len(),
            culled,
            min_radius,
            negated.load(Ordering::Relaxed)
        );
    }
    let sdf_bytes = if !use_permesh {
        let (sdf_vol, sdf_outcome) = dreamcoast_asset::cook::load_or_bake_scene_sdf(
            &fused_v,
            &fused_i,
            [sdf_dim; 3],
            amin,
            amax,
            &crate::app::cooked_cache_dir(),
        );
        info!("scene SDF {sdf_dim}^3 ({sdf_outcome:?})");
        sdf_vol.to_le_bytes()
    } else {
        let vol = crate::compose::compose_sdf_level(
            &compose_objects,
            &mesh_sdfs,
            amin,
            amax,
            sdf_dim,
            scene_diag,
        );
        info!(
            "scene SDF {sdf_dim}^3 (composed from {} per-mesh DFs)",
            mesh_sdfs.len()
        );
        vol.to_le_bytes()
    };
    // C8a per-voxel albedo volumes: cooked the same way (CPU bake, cached),
    // uploaded so the one-time GPU albedo bake is skipped too.
    let (albedo_vol, alb_outcome) = dreamcoast_asset::cook::load_or_bake_scene_albedo(
        &fused_v,
        &fused_i,
        &tri_albedo,
        [sdf_dim; 3],
        amin,
        amax,
        &crate::app::cooked_cache_dir(),
    );
    info!("scene albedo {sdf_dim}^3 ({alb_outcome:?})");
    let alb = [
        albedo_vol.channel_le_bytes(0),
        albedo_vol.channel_le_bytes(1),
        albedo_vol.channel_le_bytes(2),
    ];
    gdf.build_scene_sdf(
        device,
        &fused_v,
        &fused_i,
        &tri_albedo,
        tri_count,
        amin,
        amax,
        Some(&sdf_bytes),
        Some([&alb[0], &alb[1], &alb[2]]),
    )?;
    // Stage D: the gallery's floor is analytic (y = 0, no floor geometry); content
    // scenes carry their floor as real geometry, so disable the analytic ground
    // (a very low Y) to avoid a spurious second floor in the SW-RT march.
    gdf.set_scene_ground_y(if p.gallery_scene { 0.0 } else { -1.0e9 });
    // Stage B3: cook + install the finer clipmap levels (every level but the
    // coarsest, which `build_scene_sdf` just created). Each is keyed on its own
    // sub-AABB so the cache stores them separately; off unless P11_GDF_CLIP_LEVELS>1.
    if clip.level_count() > 1 {
        let finer = &clip.levels[..clip.level_count() - 1];
        let mut sdf_store: Vec<Vec<u8>> = Vec::new();
        let mut alb_store: Vec<[Vec<u8>; 3]> = Vec::new();
        for (lmin, lmax) in finer {
            // S1: finer levels compose from per-mesh DFs when opt-in, else the fused
            // bake (this loop only runs for content — the gallery is single-level).
            let sdf_le = if use_permesh {
                crate::compose::compose_sdf_level(
                    &compose_objects,
                    &mesh_sdfs,
                    *lmin,
                    *lmax,
                    sdf_dim,
                    scene_diag,
                )
                .to_le_bytes()
            } else {
                dreamcoast_asset::cook::load_or_bake_scene_sdf(
                    &fused_v,
                    &fused_i,
                    [sdf_dim; 3],
                    *lmin,
                    *lmax,
                    &crate::app::cooked_cache_dir(),
                )
                .0
                .to_le_bytes()
            };
            sdf_store.push(sdf_le);
            let (av, _) = dreamcoast_asset::cook::load_or_bake_scene_albedo(
                &fused_v,
                &fused_i,
                &tri_albedo,
                [sdf_dim; 3],
                *lmin,
                *lmax,
                &crate::app::cooked_cache_dir(),
            );
            alb_store.push([
                av.channel_le_bytes(0),
                av.channel_le_bytes(1),
                av.channel_le_bytes(2),
            ]);
        }
        let level_data: Vec<crate::gdf::ClipLevelData> = finer
            .iter()
            .enumerate()
            .map(|(i, (lmin, lmax))| crate::gdf::ClipLevelData {
                aabb_min: *lmin,
                aabb_max: *lmax,
                sdf: &sdf_store[i],
                albedo: Some([&alb_store[i][0], &alb_store[i][1], &alb_store[i][2]]),
            })
            .collect();
        gdf.set_clip_levels(device, &level_data)?;
    }
    // P3 (per-mesh-sdf-direct-sample-plan.md): pack every unique mesh's field into one
    // atlas volume + build the instance table / cell grid, then switch the SW-RT field
    // source to direct per-mesh sampling — the **content default** (dense loses per-mesh
    // resolution → thin-geo penetration + surface-cache checkerboard). `P11_DIRECT_SDF=0`
    // opts out to the dense-only composite (kept above as the hybrid's coarse field, and
    // as the A/B fallback). Content-only; the gallery keeps the dense anchor untouched.
    let direct_sdf = use_permesh && crate::quality::env_bool("P11_DIRECT_SDF", true);
    if use_permesh && !direct_sdf {
        info!(
            "GDF: per-mesh SDF direct sampling DISABLED (P11_DIRECT_SDF=0) — dense \
                 composite only (loses per-mesh resolution)"
        );
    }
    if direct_sdf {
        // Atlas memory cap: tiles are dense `dim³`, so downsampling the largest meshes
        // (whose extra resolution is low-frequency, covered by the coarse dense field)
        // trims the atlas a lot while thin features — resolved by their tight AABB, not
        // the cube dim — survive. `P11_ATLAS_MAX_DIM` tunes it (native = 48).
        let atlas_cap = std::env::var("P11_ATLAS_MAX_DIM")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|d| d.clamp(dreamcoast_asset::sdf::MESH_SDF_MIN_DIM, 48))
            .unwrap_or(32);
        let atlas = dreamcoast_asset::sdf_atlas::SdfAtlas::pack_capped(&mesh_sdfs, atlas_cap);
        let res = crate::mesh_sdf::grid_res_for(compose_objects.len());
        let build = crate::mesh_sdf::build(&compose_objects, &atlas, amin, amax, res);
        // F5: pack the per-mesh albedo volumes into the SAME tile geometry as the SDF
        // atlas (one `tile_uvw` maps both), so the shader reads hit colour at the hit
        // instance with per-mesh precision. Opt-in (`F5_PERMESH_ALBEDO`); off ⇒ dense.
        let albedo_atlas = if permesh_albedo && mesh_albedos.len() == mesh_sdfs.len() {
            Some(dreamcoast_asset::sdf_atlas::AlbedoAtlas::pack_like(
                &atlas,
                &mesh_albedos,
                [0.7, 0.7, 0.7],
            ))
        } else {
            None
        };
        // F2 S2b: compact f16 atlas storage. Distances are mesh-local and read through
        // the per-instance `dist_scale`, so half precision holds the march bound to
        // ~2^-11 of the stored value — measured PT-residual-neutral (phase doc §3).
        // `P11_ATLAS_F16=0` restores the f32 upload (A/B seam; bytes only, no shader
        // change). Content-only like the whole per-mesh path.
        let atlas_f16 = crate::quality::env_bool("P11_ATLAS_F16", true);
        let (atlas_format, atlas_bytes) = if atlas_f16 {
            (Format::R16Float, atlas.to_le_bytes_f16())
        } else {
            (Format::R32Float, atlas.to_le_bytes())
        };
        let voxel_bytes = if atlas_f16 { 2 } else { 4 };
        info!(
            "per-mesh SDF direct sample: atlas {}x{}x{} ({:.1} MB, {}), {} instances, {}^3 cell grid{}",
            atlas.dim[0],
            atlas.dim[1],
            atlas.dim[2],
            (atlas.voxels.len() * voxel_bytes) as f32 / 1.0e6,
            if atlas_f16 { "f16" } else { "f32" },
            build.instance_count,
            res,
            if albedo_atlas.is_some() {
                " + per-mesh albedo atlas (F5)"
            } else {
                ""
            },
        );
        let alb_ch = albedo_atlas.as_ref().map(|a| {
            if atlas_f16 {
                [
                    a.channel_le_bytes_f16(0),
                    a.channel_le_bytes_f16(1),
                    a.channel_le_bytes_f16(2),
                ]
            } else {
                [
                    a.channel_le_bytes(0),
                    a.channel_le_bytes(1),
                    a.channel_le_bytes(2),
                ]
            }
        });
        let alb_ref = alb_ch
            .as_ref()
            .map(|c| [c[0].as_slice(), c[1].as_slice(), c[2].as_slice()]);
        gdf.install_mesh_sdf(
            device,
            &atlas_bytes,
            atlas.dim,
            atlas_format,
            &build,
            alb_ref,
            crate::quality::env_bool("P11_SDF_DETAIL_REPLACE", p.base.sdf_detail_replace),
        )?;
    }
    // Phase 12 item 3: optional GPU→CPU volume-readback round-trip check. Reads
    // the just-uploaded scene SDF back and confirms it equals the bytes we
    // uploaded — validating `Device::read_volume` on the live backend.
    if std::env::var_os("P12_VERIFY_VOLUME").is_some()
        && let Some(vol) = gdf.scene_gdf_volume()
    {
        let back = device.read_volume(vol, sdf_dim, sdf_dim, sdf_dim, 4)?;
        let mismatches = back.iter().zip(&sdf_bytes).filter(|(a, b)| a != b).count();
        info!(
            "volume readback round-trip ({sdf_dim}^3): {} bytes, {mismatches} mismatch(es)",
            back.len()
        );
    }
    // Stage C/D: the surface-cache atlas (cards + per-card texel buffers, re-lit each
    // frame) feeds the SW-RT reflection/GI. It is the default ambient for any GDF
    // scene now, so build it unless the IBL escape hatch is forced (then it would be
    // unused — skip the ~67 MB atlas + per-frame relight). MAX_CARDS (fuse.rs) bounds
    // it; cards are draw-list-driven.
    // HQ reflection mode (content opt-in): card every drawable + a sharper reflection trace
    // for crisp, uniform large-scene chrome (used at card-build below and the res-div later).
    let hq_reflect = !p.gallery_scene && crate::quality::env_bool("P11_REFLECT_HQ", false);
    let build_cache = std::env::var_os("P11_LEGACY_IBL").is_none();
    if build_cache {
        // F1 (surface-cache virtualization): rank drawables for card residency from a
        // static reference camera resolved once here, matching the per-frame camera's
        // eye/focus precedence (`CAM_EYE`/`CAM_TARGET` → authored level view → orbit
        // framing). Within-budget scenes (the gallery) keep every drawable regardless of
        // this pose, so the anchor stays byte-identical; over-budget scenes select the
        // camera-relevant subset deterministically and mark the rest coarse fallback.
        let (ref_focus, ref_eye) = match (
            crate::parse_vec3_env("CAM_EYE"),
            crate::parse_vec3_env("CAM_TARGET"),
        ) {
            (Some(e), Some(t)) => (t, e),
            (Some(e), None) => (p.scene_center, e),
            _ => match p.level_view {
                Some((e, t)) => (t, e),
                None => (
                    p.scene_center,
                    p.scene_center + Vec3::new(p.scene_radius * 1.6, p.scene_radius * 0.55, 0.0),
                ),
            },
        };
        let card_cam = crate::fuse::CardCamera::from_look(ref_eye, ref_focus);
        // HQ reflection mode (`P11_REFLECT_HQ`): card EVERY drawable so a grazing reflection
        // reads smooth cached radiance instead of the analytic per-voxel albedo (the residual
        // coloured blobs on a large-scene chrome reflection). Costs extra atlas memory +
        // warm-up relight; the default budget stays 1024 (gallery is within budget either way
        // → byte-identical anchor). Pairs with the sharper reflection trace res below.
        let card_budget = if hq_reflect {
            1u32 << 20
        } else {
            crate::fuse::MAX_CARDS
        };
        let (cards, card_albedo, residency) =
            crate::fuse::build_surface_cards(&obj_aabb, &obj_albedo, &card_cam, card_budget);
        let num_cards = (cards.len() / 64) as u32;
        // C: content stamps the drawable's true albedo onto its cards (fine color);
        // the gallery keeps the legacy voxel-volume albedo (byte-identical anchor).
        // `P11_CARD_ALBEDO=0` forces the legacy path (A/B isolation of the cache color).
        let card_albedo = if p.gallery_scene || !crate::quality::env_bool("P11_CARD_ALBEDO", true) {
            None
        } else {
            Some(card_albedo.as_slice())
        };
        // QHD/UHD track: the surface-cache atlas tile is runtime-tunable (`P11_CACHE_TILE`)
        // so content can trade cache cost + atlas memory for reflection-cache sharpness.
        // Default 32 = unchanged (byte-identical). Measured: tile 16 cuts the relight only
        // ~30% (the relight isn't purely texel-bound at spp1/period40) while blurring
        // reflections (max ~94 LSB) — a poor default, so it stays opt-in. Built once here.
        let cache_tile = std::env::var("P11_CACHE_TILE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(32)
            .clamp(4, 64);
        // C2a adaptive card resolution (opt-in): redistribute the SAME texel budget as
        // the uniform atlas by camera relevance — near/large cards up to 64², far/small
        // down to 8² — so the cache carries detail where reflections actually read it.
        // F1 Stage 1 — surface-cache page pool (`P11_CACHE_POOL`, opt-in, default off). A
        // fixed pool of uniform card slots + a reverse slot_to_card table (the streaming/LRU
        // foundation). It overrides the adaptive-res atlas (the pool uses uniform slots), and
        // with Stage 1's identity map it reproduces the uniform arithmetic bit-for-bit — so
        // it may run on ANY scene (the gallery included) without moving the byte anchor,
        // which is exactly how the plumbing is verified. Default off keeps every scene on the
        // legacy path.
        // F1 Stage 3 page-pool streaming (`P11_CACHE_STREAM`, opt-in, content only). Implies
        // the pool. The fixed card slots are re-owned from the live camera each frame.
        let cache_stream = !p.gallery_scene && crate::quality::env_bool("P11_CACHE_STREAM", false);
        let cache_pool = cache_stream || crate::quality::env_bool("P11_CACHE_POOL", false);
        let card_res: Option<Vec<u32>> = if !cache_pool
            && !p.gallery_scene
            && crate::quality::env_bool("P11_CACHE_ADAPTIVE_RES", p.base.cache_adaptive_res)
        {
            Some(crate::fuse::assign_card_res(
                &cards,
                &card_cam,
                8,
                64,
                (num_cards as u64) * (cache_tile as u64) * (cache_tile as u64),
            ))
        } else {
            None
        };
        gdf.build_surface_cache(
            device,
            &cards,
            num_cards,
            cache_tile,
            card_albedo,
            card_res.as_deref(),
            // Track C card grid: pick-identical lookup acceleration (tier default +
            // `P_CACHE_GRID` override; content only — the gallery keeps the legacy scan).
            // Disabled under streaming: the grid maps cells → card indices from the initial
            // card positions, which go stale as slots are re-owned (the consumer full-scans).
            !p.gallery_scene
                && !cache_stream
                && crate::quality::env_bool("P_CACHE_GRID", p.base.cache_grid),
            cache_pool,
            cache_stream,
        )?;
        // F1 Stage 3: seed the streaming slot ownership from the initial residency + keep the
        // drawable directory so the frame loop can re-own slots from the live camera.
        if cache_stream {
            gdf.init_stream(&residency.resident);
            *stream_obj_aabb = obj_aabb.clone();
            *stream_obj_albedo = obj_albedo.clone();
        }
        // C1 mesh-triangle capture (opt-in): consolidated content geometry (the same
        // layout as the HWRT hit-lighting table, built independently of RT capability)
        // + the card→drawable instance map (table row + world→object 3x4 per resident
        // drawable). The capture then reads per-texel interpolated-UV texture albedo
        // (+ opacity) instead of one flat stamped colour per drawable — the surface
        // cache finally carries the curtain patterns a reflection should show.
        if !p.gallery_scene
            && crate::quality::env_bool("P11_CARD_MESH_CAPTURE", p.base.card_mesh_capture)
        {
            let draw_list = world.draw_list();
            match crate::mesh::build_content_hit_table(
                device,
                &draw_list,
                mesh_registry,
                material_registry,
            ) {
                Ok((vtx, idx, table)) => {
                    let mut rec: Vec<u8> = Vec::with_capacity(residency.resident.len() * 64);
                    for &di in &residency.resident {
                        let inv = draw_list[di].world.inverse();
                        rec.extend_from_slice(&(di as u32).to_le_bytes());
                        rec.extend_from_slice(&[0u8; 12]);
                        // world→object as 3 row-major float4 rows (glam is column-major).
                        let m = inv.to_cols_array_2d();
                        for r in 0..3 {
                            for col in m.iter().take(4) {
                                rec.extend_from_slice(&col[r].to_le_bytes());
                            }
                        }
                    }
                    gdf.set_card_mesh_capture(device, vtx, idx, table, &rec)?;
                    info!(
                        "surface cache: C1 mesh-triangle capture on ({} resident drawables)",
                        residency.resident.len()
                    );
                }
                Err(e) => {
                    tracing::warn!("C1 mesh capture table build failed: {e:#} — stamped albedo")
                }
            }
        }
    }
    Ok(out)
}

/// Build the CONTENT scene's acceleration structures (BLAS/TLAS + the consolidated
/// hit table) whenever the device and the scene allow it — not only when a launch knob
/// asks, because the `ReflectMode` UI switches onto this substrate live.
pub(crate) fn build_content_accel(
    device: &Device,
    rt: &mut RtSystem,
    world: &World,
    mesh_registry: &MeshRegistry,
    material_registry: &MaterialRegistry,
    p: &StaticSceneParams,
) {
    let possible = device.has_raytracing()
        && !p.gallery_scene
        && !p.world_mode
        && !world.draw_list().is_empty();
    // The PT oracle (`--raytracing` on a level scene) needs the same content accel +
    // consolidated table.
    let content_pt_want = crate::app::raytracing_enabled() && possible;
    if possible || content_pt_want {
        rt.build_content_accel(device, &world.draw_list(), mesh_registry, material_registry);
    }
}

/// F4: install the camera-anchored FINE level of the SH irradiance volume. The AABB is
/// resolved from the initial camera with the same eye precedence as the surface-card
/// reference camera (`CAM_EYE` → authored level view → orbit framing); recentering
/// (F4B) reconverges it on camera motion. Half-extent = scene max axis / 6, clamped to
/// [4, 12] m: fine enough to beat the coarse ~1.1 m spacing, small enough that 32³
/// probes stay dense. `enabled` is the caller's `gi_volume && P_GI_VOL_CLIP` gate.
pub(crate) fn install_gi_fine_box(
    device: &Device,
    gi: &mut GiSystem,
    gdf: &GdfSystem,
    enabled: bool,
    p: &StaticSceneParams,
) -> anyhow::Result<()> {
    if !enabled {
        return Ok(());
    }
    let (amin, amax) = gdf.scene_aabb();
    let eye = match (crate::parse_vec3_env("CAM_EYE"), p.level_view) {
        (Some(e), _) => e,
        (None, Some((e, _))) => e,
        (None, None) => {
            p.scene_center + Vec3::new(p.scene_radius * 1.6, p.scene_radius * 0.55, 0.0)
        }
    };
    let ext = (amax[0] - amin[0])
        .max(amax[1] - amin[1])
        .max(amax[2] - amin[2]);
    // F4B3: `P_GI_FINE_HALF` (metres) overrides the derived half-extent — the official
    // box-half sweep lever (a smaller box densifies the probe spacing 2*half/32 AND
    // shrinks the fine-covered screen area; the recenter dead-zone, voxel snap and
    // fade margin all derive from the installed box, so they track automatically).
    let half = std::env::var("P_GI_FINE_HALF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|h| h.clamp(2.0, 16.0))
        .unwrap_or((ext / 6.0).clamp(4.0, 12.0));
    let mn = [eye.x - half, eye.y - half, eye.z - half];
    let mx = [eye.x + half, eye.y + half, eye.z + half];
    gi.set_gi_fine_box(device, mn, mx)?;
    info!(
        "GI volume fine level (P_GI_VOL_CLIP): box [{:.1}, {:.1}, {:.1}]..[{:.1}, {:.1}, \
         {:.1}] half {half} m — probe spacing {:.2} m",
        mn[0],
        mn[1],
        mn[2],
        mx[0],
        mx[1],
        mx[2],
        2.0 * half / crate::gi::GI_VOL_DIM as f32
    );
    Ok(())
}
