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
pub use scene::{load_or_bake_mesh_sdf, load_or_bake_scene_albedo, load_or_bake_scene_sdf};
pub use texture::TexCompress;

use std::path::{Path, PathBuf};

use dreamcoast_core::EngineError;

use crate::{MeshData, dcasset, load_gltf};

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

    // Miss: cook from glTF (optionally block-compressing eligible textures) and
    // write the cache (write failure is non-fatal).
    let mut mesh = load_gltf(source)?;
    if tex.enabled() {
        texture::compress_material(&mut mesh.material, tex);
    }
    let cooked = dcasset::write(&mesh, key);
    if let Err(e) = write_atomic(&cache_file, &cooked) {
        tracing_warn(&format!(
            "failed to write cooked asset {}: {e}",
            cache_file.display()
        ));
    }
    Ok((mesh, LoadOutcome::Cooked))
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
