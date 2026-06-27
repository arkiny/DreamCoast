//! Cooked binary levels (Phase 12 Stage E): a `.level` (RON) cooks to a binary
//! `.dcasset` (a `CHUNK_LEVEL`), hash-keyed for invalidation, so the runtime loads
//! the cooked form directly instead of re-parsing RON every launch.
//!
//! This reuses the same cache machinery as the mesh cook ([`super::load_cooked`]):
//! the cache file is keyed on the RON source bytes, and the shipped path (RON absent,
//! cooked present) still loads. The `CHUNK_LEVEL` codec already exists in
//! [`crate::dcasset`]; this only wires cook → cache → load.

use std::path::Path;

use dreamcoast_core::EngineError;

use super::{LoadOutcome, cache_path, tracing_warn, write_atomic};
use crate::{LevelData, dcasset};

/// Load the level at `source` as a cooked [`LevelData`], cooking + caching on a miss.
/// `cache_key` is the stable logical identifier the cache file is named for (pass the
/// original level reference, not a resolved path — see [`super::load_cooked`]).
///
/// - **Hit:** the RON source hashes to the cached header's `source_hash` (version /
///   cook params match) → decode the `.dcasset` (no RON parse).
/// - **Miss:** parse the RON, cook a `.dcasset`, write it (a write failure is
///   non-fatal — the level is still returned).
/// - **No source:** RON gone but a `.dcasset` exists → load it directly.
pub fn load_or_cook_level(
    source: &Path,
    cache_key: &str,
    cache_dir: &Path,
) -> Result<(LevelData, LoadOutcome), EngineError> {
    let cache_file = cache_path(cache_dir, cache_key);

    // Shipped path: no RON to validate against — trust the cooked binary if present.
    let Ok(src_bytes) = std::fs::read(source) else {
        if let Ok(bytes) = std::fs::read(&cache_file)
            && let Ok((_, level)) = dcasset::read_level(&bytes)
        {
            return Ok((level, LoadOutcome::CacheHitNoSource));
        }
        return Err(EngineError::Asset(format!(
            "no source .level and no cooked .dcasset for {}",
            source.display()
        )));
    };

    let key = dcasset::source_hash(&src_bytes);

    // Hit: a cached header whose key matches the live RON → decode it, no parse.
    if let Ok(bytes) = std::fs::read(&cache_file)
        && let Ok(header) = dcasset::read_header(&bytes)
        && header.version == dcasset::VERSION
        && header.source_hash == key
        && header.cook_params_hash == dcasset::cook_params_hash()
        && let Ok((_, level)) = dcasset::read_level(&bytes)
    {
        return Ok((level, LoadOutcome::CacheHit));
    }

    // Miss: parse the RON, cook a CHUNK_LEVEL .dcasset, and cache it.
    let level = LevelData::load_ron(source)?;
    let cooked = dcasset::write_level(&level, key);
    if let Err(e) = write_atomic(&cache_file, &cooked) {
        tracing_warn(&format!(
            "failed to write cooked level {}: {e}",
            cache_file.display()
        ));
    }
    Ok((level, LoadOutcome::Cooked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::{Camera, Entity, Environment};

    fn sample_level() -> LevelData {
        LevelData {
            entities: vec![Entity {
                asset: "assets/Sponza/Sponza.gltf".into(),
                transform: [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                material_override: None,
            }],
            lights: vec![],
            camera: Camera::default(),
            environment: Environment::default(),
        }
    }

    #[test]
    fn cook_then_cache_hit() {
        let tmp = super::super::tests::TempDir::new("level");
        let src = tmp.0.join("sponza.level");
        sample_level().save_ron(&src).unwrap();
        let key = "levels/sponza.level";

        // First load cooks from RON and writes the .dcasset.
        let (a, o1) = load_or_cook_level(&src, key, &tmp.0).unwrap();
        assert_eq!(o1, LoadOutcome::Cooked);
        // Second load hits the cache (no RON parse) and decodes identical data.
        let (b, o2) = load_or_cook_level(&src, key, &tmp.0).unwrap();
        assert_eq!(o2, LoadOutcome::CacheHit);
        assert_eq!(a, b);
        assert_eq!(a, sample_level());
    }

    #[test]
    fn loads_cooked_without_source() {
        let tmp = super::super::tests::TempDir::new("level-nosrc");
        let src = tmp.0.join("ghost.level");
        sample_level().save_ron(&src).unwrap();
        let key = "levels/ghost.level";
        load_or_cook_level(&src, key, &tmp.0).unwrap(); // cook
        std::fs::remove_file(&src).unwrap(); // drop the RON
        let (level, outcome) = load_or_cook_level(&src, key, &tmp.0).unwrap();
        assert_eq!(outcome, LoadOutcome::CacheHitNoSource);
        assert_eq!(level, sample_level());
    }
}
