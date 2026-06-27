//! Scene-bake cooks (Phase 12 M2): the world-space SDF + per-voxel albedo volumes.
//! Unlike [`super::load_cooked`] there is no source file — the "source" is the fused
//! geometry generated in-process, so each is keyed on a content hash of its inputs.

use std::path::Path;

use super::{LoadOutcome, tracing_warn, write_atomic};
use crate::dcasset;
use crate::sdf::{self, AlbedoVolumes, SdfVolume};

/// Load the scene's signed-distance field as cooked data, baking + caching on a
/// miss. The invalidation key is a content hash of `(fused_vtx, fused_idx, dim,
/// aabb)` — any change re-bakes.
///
/// The cook is pure CPU, so the bytes are deterministic and backend-independent;
/// uploading them to the GPU volume gives a Vulkan/D3D12 byte-identical field and
/// makes "loaded without re-bake == direct bake" hold by construction.
pub fn load_or_bake_scene_sdf(
    fused_vtx: &[u8],
    fused_idx: &[u8],
    dim: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cache_dir: &Path,
) -> (SdfVolume, LoadOutcome) {
    // Content key over geometry + grid params (the cook parameters that change the
    // baked bytes). Folded incrementally so the large vertex buffer isn't copied.
    let mut key = dcasset::hash_begin();
    key = dcasset::hash_update(key, fused_vtx);
    key = dcasset::hash_update(key, fused_idx);
    key = dcasset::hash_update(key, &dim.to_le_bytes());
    for c in aabb_min.iter().chain(aabb_max.iter()) {
        key = dcasset::hash_update(key, &c.to_le_bytes());
    }
    let cache_file = cache_dir.join(format!("scene-sdf.{key:016x}.dcasset"));

    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, vol)) = dcasset::read_sdf(&bytes)
    {
        return (vol, LoadOutcome::CacheHit);
    }

    let vol = sdf::bake_sdf_from_fused(fused_vtx, fused_idx, dim, aabb_min, aabb_max);
    if let Err(e) = write_atomic(&cache_file, &dcasset::write_sdf(&vol, key)) {
        tracing_warn(&format!(
            "failed to write cooked scene SDF {}: {e}",
            cache_file.display()
        ));
    }
    (vol, LoadOutcome::Cooked)
}

/// Load the scene's per-voxel albedo volumes as cooked data, baking + caching on a
/// miss (the C8a companion to [`load_or_bake_scene_sdf`]). Keyed on the fused
/// geometry **plus the per-triangle albedo** + grid, so a material-colour change
/// re-bakes. Pure CPU → deterministic, backend-independent, VK≡DX by construction.
pub fn load_or_bake_scene_albedo(
    fused_vtx: &[u8],
    fused_idx: &[u8],
    tri_albedo: &[u8],
    dim: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cache_dir: &Path,
) -> (AlbedoVolumes, LoadOutcome) {
    let mut key = dcasset::hash_begin();
    key = dcasset::hash_update(key, fused_vtx);
    key = dcasset::hash_update(key, fused_idx);
    key = dcasset::hash_update(key, tri_albedo);
    key = dcasset::hash_update(key, &dim.to_le_bytes());
    for c in aabb_min.iter().chain(aabb_max.iter()) {
        key = dcasset::hash_update(key, &c.to_le_bytes());
    }
    let cache_file = cache_dir.join(format!("scene-albedo.{key:016x}.dcasset"));

    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, vol)) = dcasset::read_albedo(&bytes)
        && vol.dim == dim
    {
        return (vol, LoadOutcome::CacheHit);
    }

    let vol =
        sdf::bake_albedo_from_fused(fused_vtx, fused_idx, tri_albedo, dim, aabb_min, aabb_max);
    if let Err(e) = write_atomic(&cache_file, &dcasset::write_albedo(&vol, key)) {
        tracing_warn(&format!(
            "failed to write cooked scene albedo {}: {e}",
            cache_file.display()
        ));
    }
    (vol, LoadOutcome::Cooked)
}
