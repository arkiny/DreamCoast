//! Cook orchestration (Phase 12): lazy glTF → `.dcasset` with hash-keyed
//! invalidation, so the runtime loads a cooked binary instead of re-parsing glTF
//! and re-decoding textures on every launch.
//!
//! The invalidation key is `{version, source_hash, cook_params_hash}` (see
//! [`crate::dcasset`]) — a single source of truth. A fresh cache hit skips the
//! glTF importer (and the expensive image decode) entirely; only a miss (no
//! cache, version/param bump, or a changed source) re-cooks and rewrites.
//!
//! The runtime also works **without the source glTF**: if the source is absent
//! but a `.dcasset` exists, it loads directly — the shipped-asset path.
//!
//! Submodules: [`texture`] (BCn compression policy + tiers), [`scene`] (the scene
//! SDF / albedo bakes).

mod level;
mod scene;
mod texture;

pub use level::load_or_cook_level;
pub use scene::{
    load_or_bake_mesh_albedo, load_or_bake_mesh_sdf, load_or_bake_scene_albedo,
    load_or_bake_scene_sdf,
};
pub use texture::{TexCompress, TexSlot, compress_image_for_slot, slot_format};

use std::path::{Path, PathBuf};

use dreamcoast_core::EngineError;

use crate::vcache::VertexCache;
use crate::{GltfScene, MeshData, dcasset, load_gltf, load_gltf_scene};

/// Which path [`load_cooked`] took, for the caller to log (startup speedup is
/// observable as the second run reporting `CacheHit` instead of `Cooked`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOutcome {
    /// Fresh `.dcasset` loaded directly; the glTF importer was skipped.
    CacheHit,
    /// Source present but cache missing/stale/corrupt → cooked from glTF and saved.
    Cooked,
    /// Source glTF absent but a cached `.dcasset` existed → loaded it (shipped path).
    CacheHitNoSource,
}

/// Deterministic cache file for the logical asset `cache_key` under `cache_dir`.
/// The key is hashed into the name so two assets with the same stem in different
/// folders never collide; the stem is kept as a human-readable prefix.
///
/// **The key must be a stable, cwd-independent identifier** (e.g. the original
/// `assets/model.glb` reference), *not* a resolved filesystem path — otherwise the
/// cook run (source found at `./assets/…` or an absolute exe-relative path) and a
/// later source-absent run (key falls back to the bare reference) would compute
/// different names and miss each other. Separating the key from the read path is
/// what makes the shipped, glTF-absent load find the cooked asset.
fn cache_path(cache_dir: &Path, cache_key: &str) -> PathBuf {
    let stem = Path::new(cache_key)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset");
    let path_key = dcasset::source_hash(cache_key.as_bytes());
    cache_dir.join(format!("{stem}.{path_key:016x}.dcasset"))
}

/// Load the asset at `source` as cooked mesh data, cooking + caching on a miss.
/// `cache_key` is the stable logical identifier the cache file is named for (see
/// [`cache_path`]) — pass the original asset reference, not the resolved path.
///
/// - **Hit:** source bytes hash to the cached header's `source_hash` (and version
///   / cook params match) → load the `.dcasset` (no glTF parse).
/// - **Miss:** import the glTF, cook a `.dcasset`, write it, return the mesh. A
///   write failure is non-fatal (the mesh is still returned; the cook just isn't
///   cached).
/// - **No source:** if the glTF is gone but a `.dcasset` exists, load it directly.
///
/// Returns an error only when neither a usable source nor a cache exists; the
/// caller's procedural fallback handles that.
pub fn load_cooked(
    source: &Path,
    cache_key: &str,
    cache_dir: &Path,
    tex: TexCompress,
) -> Result<(MeshData, LoadOutcome), EngineError> {
    let cache_file = cache_path(cache_dir, cache_key);

    // Shipped path: no source to validate against — trust the cache if present.
    let Ok(src_bytes) = std::fs::read(source) else {
        if let Ok(bytes) = std::fs::read(&cache_file)
            && let Ok((_, mesh)) = dcasset::read(&bytes)
        {
            return Ok((mesh, LoadOutcome::CacheHitNoSource));
        }
        return Err(EngineError::Asset(format!(
            "no source glTF and no cached .dcasset for {}",
            source.display()
        )));
    };

    // Key folds the source bytes + the compression tier, so changing the tier
    // re-cooks (each tier produces different bytes).
    let key = dcasset::hash_update(dcasset::source_hash(&src_bytes), &[tex.tag()]);

    // Hit: a cached header whose key matches the live source → decode it, no parse.
    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, mesh)) = dcasset::read(&bytes)
    {
        return Ok((mesh, LoadOutcome::CacheHit));
    }

    // Miss: cook from glTF (optionally block-compressing eligible textures) and write the cache
    // (write failure is non-fatal). The cooked `.dcasset` bakes BOTH the static-mesh fallback
    // (`CHUNK_MESH` + textures) AND the virtual-geometry LOD DAG (`CHUNK_CLUSTERS`), so a
    // capability-gated runtime can pick either from one file. `read` returns only the fallback,
    // so this is transparent to the existing render path.
    let mut mesh = load_gltf(source)?;
    if tex.enabled() {
        texture::compress_material(&mut mesh.material, tex);
    }
    let clusters = crate::vgeo::build_lod_dag(&mesh.vertices, &mesh.indices, 0);
    let cooked = dcasset::write_with_clusters(&mesh, &clusters, key);
    if let Err(e) = write_atomic(&cache_file, &cooked) {
        tracing_warn(&format!(
            "failed to write cooked asset {}: {e}",
            cache_file.display()
        ));
    }
    Ok((mesh, LoadOutcome::Cooked))
}

/// Load the virtual-geometry cluster pages of the asset at `source`, cooking on a miss (which
/// also writes the fallback mesh). Returns the LOD DAG baked into the cooked `.dcasset`. Errors
/// if the asset can't be cooked or the cooked file carries no cluster chunk (a fallback-only
/// asset). Phase 14: the vgeo render path (and the `--vgeo-test` viewer) consume this instead of
/// rebuilding the DAG at runtime, exercising the real cooked pipeline end to end.
pub fn load_cooked_clusters(
    source: &Path,
    cache_key: &str,
    cache_dir: &Path,
    tex: TexCompress,
) -> Result<crate::vgeo::MeshClusters, EngineError> {
    // Ensure the asset is cooked (writes the combined file on a miss), then read its clusters.
    load_cooked(source, cache_key, cache_dir, tex)?;
    let cache_file = cache_path(cache_dir, cache_key);
    let bytes = std::fs::read(&cache_file)
        .map_err(|e| EngineError::Asset(format!("read cooked {}: {e}", cache_file.display())))?;
    dcasset::read_clusters_opt(&bytes)?
        .ok_or_else(|| EngineError::Asset("cooked asset has no cluster chunk".into()))
}

/// Load a glTF **scene** (multi-mesh/multi-material) as a cooked, block-compressed
/// `.dcasset`, cooking + caching on a miss. This is the level / glTF-import path
/// (`load_cooked` handles the single-mesh gallery model). On a hit the glTF parse,
/// image decode, AND BCn encode are all skipped — only the cached scene is read.
///
/// `tier` block-compresses the texture table per slot usage (gallery anchor passes
/// `Off`). Animated/skinned scenes are compressed in memory but **not cached** (the
/// static scene chunk can't represent skins/morph), so they re-cook each load.
pub fn load_or_cook_gltf_scene(
    source: &Path,
    cache_key: &str,
    cache_dir: &Path,
    tier: TexCompress,
) -> Result<(GltfScene, LoadOutcome), EngineError> {
    let cache_file = cache_path(cache_dir, cache_key);

    // Shipped path: no source to validate against — trust the cache if present.
    let Ok(src_bytes) = std::fs::read(source) else {
        if let Ok(bytes) = std::fs::read(&cache_file)
            && let Ok((_, scene)) = dcasset::read_scene(&bytes)
        {
            return Ok((scene, LoadOutcome::CacheHitNoSource));
        }
        return Err(EngineError::Asset(format!(
            "no source glTF and no cached scene for {}",
            source.display()
        )));
    };

    // Key folds the source bytes + the compression tier (changing either re-cooks).
    let key = dcasset::hash_update(dcasset::source_hash(&src_bytes), &[tier.tag()]);

    // Hit: a cached scene whose key matches the live source → decode it, no glTF parse.
    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, scene)) = dcasset::read_scene(&bytes)
    {
        return Ok((scene, LoadOutcome::CacheHit));
    }

    // Miss: import the glTF, block-compress the texture table, cache when static.
    let mut scene = load_gltf_scene(source)?;
    texture::compress_scene_textures(&mut scene, tier);
    if is_static_scene(&scene) {
        let cooked = dcasset::write_scene(&scene, key);
        if let Err(e) = write_atomic(&cache_file, &cooked) {
            tracing_warn(&format!(
                "failed to write cooked scene {}: {e}",
                cache_file.display()
            ));
        }
    }
    Ok((scene, LoadOutcome::Cooked))
}

/// Load a baked **vertex-animation cache** (`.abc` Alembic or `.usda` USD) as a cooked
/// `.dcasset`, cooking + caching on a miss. This is the separate-asset cook for the knight
/// deformation caches: the level references the *source* path, the first load cooks the
/// decoded [`VertexCache`] to `cache/dcasset/`, and every later load is a `CacheHit` that
/// skips the multi-hundred-MB text/Ogawa decode entirely (the architecture requirement:
/// assets cook as their own file; the level *loads* the cooked asset).
///
/// The invalidation key is the source file's **cheap metadata** (length + mtime + path),
/// not a content hash — hashing a 665 MB / 1.4 GB source on every launch would defeat the
/// point of the cache (fast reload). A changed source (different size or mtime) re-cooks.
pub fn load_or_cook_vcache(
    source: &Path,
    cache_key: &str,
    cache_dir: &Path,
) -> Result<(VertexCache, LoadOutcome), EngineError> {
    let cache_file = cache_path(cache_dir, cache_key);

    // Shipped path: source absent → trust a cached `.dcasset` if present.
    let Some(key) = source_meta_key(source) else {
        if let Ok(bytes) = std::fs::read(&cache_file)
            && let Ok((_, cache)) = dcasset::read_vcache(&bytes)
        {
            return Ok((cache, LoadOutcome::CacheHitNoSource));
        }
        return Err(EngineError::Asset(format!(
            "no source cache and no cooked .dcasset for {}",
            source.display()
        )));
    };

    // Hit: a cached header whose metadata key matches the live source → decode it directly.
    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, cache)) = dcasset::read_vcache(&bytes)
    {
        return Ok((cache, LoadOutcome::CacheHit));
    }

    // Miss: decode the source by extension, cook the `.dcasset`, write it (non-fatal on fail).
    let cache = decode_vcache_source(source)?;
    let cooked = dcasset::write_vcache(&cache, key);
    if let Err(e) = write_atomic(&cache_file, &cooked) {
        tracing_warn(&format!(
            "failed to write cooked vcache {}: {e}",
            cache_file.display()
        ));
    }
    Ok((cache, LoadOutcome::Cooked))
}

/// Decode a vertex-cache source by extension: `.abc` → the Alembic reader, `.usd`/`.usda`
/// → the ASCII USD reader. Both yield the same neutral [`VertexCache`].
fn decode_vcache_source(source: &Path) -> Result<VertexCache, EngineError> {
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "abc" => crate::alembic::read_vertex_cache(source),
        "usd" | "usda" | "usdc" => crate::usd::read_vertex_cache(source),
        other => Err(EngineError::Asset(format!(
            "unsupported vertex-cache source extension '.{other}' ({})",
            source.display()
        ))),
    }
}

/// A cheap, cwd-independent identity key for a large source file: length + mtime + the
/// logical path, folded into the invalidation hash. `None` when the source is absent
/// (shipped-asset path). Deliberately avoids reading the file's bytes.
fn source_meta_key(source: &Path) -> Option<u64> {
    let md = std::fs::metadata(source).ok()?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h = dcasset::hash_begin();
    h = dcasset::hash_update(h, &md.len().to_le_bytes());
    h = dcasset::hash_update(h, &mtime.to_le_bytes());
    h = dcasset::hash_update(h, source.to_string_lossy().as_bytes());
    Some(h)
}

/// Whether a scene has no skin / morph / animation side data — only static scenes are
/// cacheable (the glTF-scene chunk stores geometry + materials + textures, not rigs).
fn is_static_scene(scene: &GltfScene) -> bool {
    scene.animations.is_empty()
        && scene.skins.is_empty()
        && scene
            .meshes
            .iter()
            .flatten()
            .all(|p| p.joints.is_none() && p.weights.is_none() && p.morph_targets.is_empty())
}

/// Write `bytes` to `path`, creating the parent dir. Writes to a temp sibling then
/// renames so a crash mid-write never leaves a torn `.dcasset` that would later be
/// read as a corrupt (and thus discarded) cache. Shared by the scene bakes.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("dcasset.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// The asset crate has no `tracing` dependency; route the rare cook-write warning
/// to stderr so a failed cache write is visible without pulling in a logging dep.
pub(crate) fn tracing_warn(msg: &str) {
    eprintln!("dcasset cook: {msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A per-test scratch dir under the OS temp dir, removed when dropped.
    pub(super) struct TempDir(pub(super) PathBuf);
    impl TempDir {
        pub(super) fn new(tag: &str) -> Self {
            // Thread id keeps parallel tests from colliding without needing a RNG.
            let id = format!("{:?}", std::thread::current().id());
            let id: String = id.chars().filter(|c| c.is_alphanumeric()).collect();
            let dir = std::env::temp_dir().join(format!("dcasset-cook-{tag}-{id}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tiny_mesh() -> MeshData {
        use crate::{Material, MeshVertex};
        MeshData {
            vertices: vec![MeshVertex {
                pos: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
            }],
            indices: vec![0],
            material: Material::default(),
        }
    }

    #[test]
    fn cache_path_is_stable_and_key_dependent() {
        let dir = Path::new("/cache");
        assert_eq!(
            cache_path(dir, "assets/model.glb"),
            cache_path(dir, "assets/model.glb")
        );
        assert_ne!(
            cache_path(dir, "assets/a.glb"),
            cache_path(dir, "assets/b.glb")
        );
    }

    #[test]
    fn loads_cached_asset_without_source() {
        let tmp = TempDir::new("nosrc");
        let key = "assets/ghost.glb";
        // Pre-cook a .dcasset, then load with a source path that does not exist.
        let bytes = dcasset::write(&tiny_mesh(), 123);
        std::fs::write(cache_path(&tmp.0, key), bytes).unwrap();

        let (mesh, outcome) = load_cooked(
            Path::new("does/not/exist.glb"),
            key,
            &tmp.0,
            TexCompress::Off,
        )
        .expect("load");
        assert_eq!(outcome, LoadOutcome::CacheHitNoSource);
        assert_eq!(mesh.vertices.len(), 1);
    }

    #[test]
    fn missing_source_and_cache_is_an_error() {
        let tmp = TempDir::new("empty");
        let r = load_cooked(Path::new("nope.glb"), "nope.glb", &tmp.0, TexCompress::Off);
        assert!(r.is_err());
    }
}
