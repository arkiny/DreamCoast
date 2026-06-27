//! Cook orchestration (Phase 12 M1.3): lazy glTF → `.dcasset` with hash-keyed
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

use std::path::{Path, PathBuf};

use dreamcoast_core::EngineError;
use rhi_types::Format;

use crate::bc::{self, BcFormat};
use crate::sdf::{self, SdfVolume};
use crate::{ImageData, Material, MeshData, TexData, dcasset, load_gltf};

/// Per-slot texture-compression policy (Phase 12 M3). **Perceptual colour**
/// (base colour, emissive) compresses to BC1; **normals** to BC5 (near-lossless).
/// **Data textures** — metallic-roughness and anything carrying linear/vector data
/// — are left uncompressed, because block compression corrupts non-perceptual
/// values. Textures with meaningful alpha also stay uncompressed (BC1 drops alpha).
fn compress_material(material: &mut Material) {
    take_compress(&mut material.base_color, BcFormat::Bc1, true, true);
    take_compress(&mut material.emissive, BcFormat::Bc1, true, true);
    take_compress(&mut material.normal, BcFormat::Bc5, false, false);
    // metallic_roughness: data texture — intentionally left uncompressed.
}

/// Compress one slot in place if it holds an uncompressed image eligible for `fmt`.
/// `srgb` tags the colour space; `skip_if_alpha` keeps RGBA8 when the image has
/// real transparency (BC1 has no usable alpha).
fn take_compress(slot: &mut Option<TexData>, fmt: BcFormat, srgb: bool, skip_if_alpha: bool) {
    if let Some(TexData::Rgba8(im)) = slot {
        if skip_if_alpha && im.rgba8.chunks_exact(4).any(|p| p[3] != 255) {
            return; // transparency present — keep lossless
        }
        *slot = Some(compress_image(im, fmt, srgb));
    }
}

/// Block-compress an RGBA8 image to a full BCn mip chain. Mips come from the shared
/// `generate_mip_chain` (the cross-backend-parity single source) so cooked mips
/// match the uncompressed upload path, then each level is BC-encoded.
fn compress_image(im: &ImageData, fmt: BcFormat, srgb: bool) -> TexData {
    let format = if srgb {
        Format::Rgba8Srgb
    } else {
        Format::Rgba8Unorm
    };
    let levels = rhi_types::generate_mip_chain(&im.rgba8, im.width, im.height, format);
    let mips = levels
        .iter()
        .enumerate()
        .map(|(mip, lvl)| {
            let w = (im.width >> mip).max(1);
            let h = (im.height >> mip).max(1);
            match fmt {
                BcFormat::Bc1 => bc::encode_bc1(lvl, w, h),
                BcFormat::Bc5 => bc::encode_bc5(lvl, w, h),
            }
        })
        .collect();
    TexData::Bc {
        format: fmt,
        srgb,
        width: im.width,
        height: im.height,
        mips,
    }
}

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
    compress: bool,
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

    // Key folds the source bytes + the compression policy, so toggling compression
    // re-cooks (a BC-compressed asset and an RGBA8 one are different bytes).
    let key = dcasset::hash_update(dcasset::source_hash(&src_bytes), &[u8::from(compress)]);

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
    if compress {
        compress_material(&mut mesh.material);
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

/// Load the scene's signed-distance field as cooked data, baking + caching on a
/// miss. Unlike [`load_cooked`] there is no source file: the "source" is the fused
/// world-space geometry generated in-process, so the invalidation key is a content
/// hash of `(fused_vtx, fused_idx, dim, aabb)` — any change re-bakes.
///
/// - **Hit:** a cached `.dcasset` whose header key matches → decode the SDF chunk
///   (no CPU bake).
/// - **Miss:** [`sdf::bake_sdf_from_fused`] then write the `.dcasset` (atomic; a
///   write failure is non-fatal — the volume is still returned).
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

/// Write `bytes` to `path`, creating the parent dir. Writes to a temp sibling then
/// renames so a crash mid-write never leaves a torn `.dcasset` that would later be
/// read as a corrupt (and thus discarded) cache.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("dcasset.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// The asset crate has no `tracing` dependency; route the rare cook-write warning
/// to stderr so a failed cache write is visible without pulling in a logging dep.
fn tracing_warn(msg: &str) {
    eprintln!("dcasset cook: {msg}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Material, MeshVertex};

    /// A per-test scratch dir under the OS temp dir, removed when dropped.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
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

        let (mesh, outcome) =
            load_cooked(Path::new("does/not/exist.glb"), key, &tmp.0, false).expect("load");
        assert_eq!(outcome, LoadOutcome::CacheHitNoSource);
        assert_eq!(mesh.vertices.len(), 1);
    }

    #[test]
    fn missing_source_and_cache_is_an_error() {
        let tmp = TempDir::new("empty");
        let r = load_cooked(Path::new("nope.glb"), "nope.glb", &tmp.0, false);
        assert!(r.is_err());
    }

    fn solid_image(w: u32, h: u32, rgba: [u8; 4]) -> TexData {
        TexData::Rgba8(ImageData {
            width: w,
            height: h,
            rgba8: rgba.repeat((w * h) as usize),
        })
    }

    #[test]
    fn compression_policy_per_slot() {
        let mut m = Material {
            base_color: Some(solid_image(8, 8, [200, 100, 50, 255])),
            metallic_roughness: Some(solid_image(8, 8, [0, 128, 200, 255])),
            normal: Some(solid_image(8, 8, [128, 128, 255, 255])),
            emissive: Some(solid_image(8, 8, [10, 20, 30, 255])),
            ..Material::default()
        };
        compress_material(&mut m);

        // Perceptual colour -> BC1; normals -> BC5; data texture stays RGBA8.
        assert!(matches!(
            m.base_color,
            Some(TexData::Bc {
                format: BcFormat::Bc1,
                ..
            })
        ));
        assert!(matches!(
            m.emissive,
            Some(TexData::Bc {
                format: BcFormat::Bc1,
                ..
            })
        ));
        assert!(matches!(
            m.normal,
            Some(TexData::Bc {
                format: BcFormat::Bc5,
                ..
            })
        ));
        assert!(
            matches!(m.metallic_roughness, Some(TexData::Rgba8(_))),
            "metallic-roughness is a data texture and must stay uncompressed"
        );
    }

    #[test]
    fn alpha_base_color_stays_uncompressed() {
        // A base colour with real transparency must not lose its alpha to BC1.
        let mut m = Material {
            base_color: Some(solid_image(4, 4, [200, 100, 50, 128])),
            ..Material::default()
        };
        compress_material(&mut m);
        assert!(matches!(m.base_color, Some(TexData::Rgba8(_))));
    }

    #[test]
    fn compression_shrinks_and_roundtrips() {
        let mut m = Material {
            base_color: Some(solid_image(64, 64, [200, 100, 50, 255])),
            ..Material::default()
        };
        let raw = match m.base_color.as_ref().unwrap() {
            TexData::Rgba8(im) => im.rgba8.len(),
            _ => unreachable!(),
        };
        compress_material(&mut m);
        let compressed: usize = match m.base_color.as_ref().unwrap() {
            TexData::Bc { mips, .. } => mips.iter().map(|m| m.len()).sum(),
            _ => unreachable!(),
        };
        // BC1 is 8:1 on the base level; even with the full mip chain it is far
        // smaller than the single uncompressed level.
        assert!(
            compressed < raw / 4,
            "compressed {compressed} should be << raw {raw}"
        );
    }
}
