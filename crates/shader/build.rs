//! Compiles `.slang` shaders to SPIR-V (Vulkan) and DXIL (D3D12) at build time.
//!
//! `slangc` is resolved without requiring a full Vulkan SDK install, in order:
//!   1. `SLANGC` environment variable (full path to slangc.exe)
//!   2. vendored `tools/slang/bin/slangc.exe` at the workspace root
//!   3. `slangc` on `PATH`
//!   4. `%VULKAN_SDK%\Bin\slangc.exe` (compatibility)
//!
//! If `slangc` cannot be found the build does NOT fail: it emits a warning and
//! generates stub accessors that return `None`, so the rest of the workspace
//! still builds (e.g. the empty-window milestone). Obtain slangc and rebuild to
//! enable shaders. A nonzero slangc exit (an actual compile error) DOES fail.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const RT_PIPELINE_ISECT_KEY: &str = "rt_pipeline_isect";
const RT_PIPELINE_DISPATCH_KEY: &str = "rt_pipeline_dispatch";

/// Shared `#include` files that are not JOBS themselves but are pulled in by shaders that are.
/// Every one is folded into the cook cache's `base_hash` and emitted as `rerun-if-changed`, so
/// editing any of them recompiles all dependents (they include no per-job tracking otherwise —
/// an omitted entry silently ships stale bytecode). Keep in sync with the `#include`s under
/// `shaders/` (a non-JOB `.slang` — or the RT-pipeline root-sig JSON — belongs here).
const SHARED_INCLUDES: [&str; 12] = [
    "bindless.slang",
    "rt_common.slang",
    "rt_pipeline_metal_rootsig.json",
    "clipmap.slang",
    "mesh_sdf_sample.slang",
    "gdf_bounce.slang",
    "octahedral.slang",
    "sky_common.slang",
    "surface_cache.slang",
    "wrc_common.slang",
    "light_cluster_common.slang",
    "pbr_brdf.slang",
];

// FNV-1a 64-bit — dependency-free content hash for the shader cook cache (Phase 12
// M4, shader-asset-cache.md §4 option A). Collision risk is irrelevant for a build
// cache key; this keeps `crates/shader` at zero build-dependencies.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into the running FNV-1a hash `h` (seed with `FNV_OFFSET`).
fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// HLSL shader profile for a (target, stage, key), or `None` for the Metal target (which
/// derives everything from the stage). **Single source of truth** shared by both the
/// cache key and the slangc `-profile` arg below, so the key can never drift from what
/// is actually compiled. Ray-tracing stages need a DXIL *library* (`lib_6_5`); inline
/// `RayQuery` requires >= 6.5. SPIR-V uses `sm_6_5` for every stage (RT comes from the
/// stage capability). A few DXIL entry points need `sm_6_6` — see `dxil_needs_sm66`.
fn profile_for(target: &str, stage: &str, key: &str) -> Option<&'static str> {
    match target {
        "spirv" => Some("sm_6_5"),
        "dxil" => Some(if is_rt_stage(stage) {
            "lib_6_5"
        } else if dxil_needs_sm66(key) {
            "sm_6_6"
        } else {
            "sm_6_5"
        }),
        _ => None,
    }
}

/// DXIL entry points that require **Shader Model 6.6**. Slang internally auto-upgrades the
/// profile when it sees an SM6.6-only op (it warns E41012), but that upgrade does NOT reach
/// the shader-model DXC targets, so a 64-bit buffer atomic (`InterlockedMax64` — the
/// visibility-buffer primitive of Phase 14 virtual geometry) lowers to `dx.op.atomicBinOp.i64`
/// and DXC rejects it with "64-bit atomic operations should only be used in Shader Model 6.6+".
/// Setting the profile to `sm_6_6` here fixes it. This is **per-entry**, not per-file:
/// `vgeo_swraster`'s `csClear` uses no atomic and stays on 6.5 (only `csRaster` needs 6.6).
/// Keep in sync with any shader that gains a 64-bit atomic or another SM6.6-only op. SPIR-V
/// needs no analogue — Slang infers the `Int64Atomics` capability from the op directly.
///
/// `vgeo_hwvis_fs` is the Track B (HW-path) counterpart of `vgeo_swraster_cs`: its fragment
/// stage records the same `(depthKey<<32)|payload` into the R64 visibility buffer with an
/// `InterlockedMax64`, so it needs SM6.6 too. The mesh stages (`*_ms`) carry no atomic and stay
/// on the default `sm_6_5` (the minimum mesh-shader profile).
fn dxil_needs_sm66(key: &str) -> bool {
    matches!(key, "vgeo_atomic_cs" | "vgeo_swraster_cs" | "vgeo_hwvis_fs")
}

/// Mesh entry points whose DXIL must be produced via the slang→HLSL→patch→DXC workaround
/// below instead of a direct `slangc -target dxil`. Slang 2026.10.2 has a HLSL-emit bug: when a
/// mesh shader declares **all three** output kinds (`vertices` + `indices` + `primitives`) it
/// drops the `vertices` modifier from the vertex-output parameter, so DXC rejects the SV_Position
/// semantic ("invalid for shader model: ms"). Two-output mesh shaders (`vgeo_meshlet`,
/// `vgeo_cluster` = vertices+indices) are unaffected and compile directly. SPIR-V + Metal emit the
/// modifier correctly, so only DXIL needs the patch. Keep in sync with any mesh shader that adds a
/// per-`primitives` output. See `compile_mesh_dxil_via_hlsl_patch`.
fn dxil_mesh_needs_vertices_patch(key: &str) -> bool {
    matches!(key, "vgeo_hwvis_ms")
}

/// Re-insert the `out vertices` modifier Slang drops from a mesh shader's vertex-output parameter
/// (see `dxil_mesh_needs_vertices_patch`). The buggy emission is `<Type>  verts_<N>[<M>U]`; DXC
/// needs `out vertices <Type>  verts_<N>[...]`. The parameter is named `verts` by convention in
/// our mesh shaders, so the patch keys on that. Returns the HLSL unchanged (and the caller's
/// assertion fires) if the pattern is absent — e.g. if a future Slang fixes the bug, the direct
/// path in the emit loop is used instead.
fn patch_mesh_vertices_out(hlsl: &str) -> String {
    // Operate only on the entry function signature line (the body also mentions `verts_`).
    let Some(line_start) = hlsl.find("void meshMain(") else {
        return hlsl.to_string();
    };
    let line_end = hlsl[line_start..]
        .find(')')
        .map(|o| line_start + o)
        .unwrap_or(hlsl.len());
    let sig = &hlsl[line_start..line_end];
    // Find the vertex-output parameter `<Type>  verts_<N>` inside the signature.
    let Some(rel) = sig.find("  verts_") else {
        return hlsl.to_string();
    };
    let abs = line_start + rel; // index of the first of the two spaces before `verts_`
    let bytes = hlsl.as_bytes();
    // Walk back over the two spaces, then the type identifier, to the type-token start.
    let mut i = abs;
    while i > 0 && bytes[i - 1] == b' ' {
        i -= 1;
    }
    while i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
        i -= 1;
    }
    // Already patched / already correct (Slang fixed): leave it.
    if hlsl[..i].trim_end().ends_with("vertices") {
        return hlsl.to_string();
    }
    let mut out = String::with_capacity(hlsl.len() + 13);
    out.push_str(&hlsl[..i]);
    out.push_str("out vertices ");
    out.push_str(&hlsl[i..]);
    out
}

/// Produce DXIL for a three-output mesh shader via slang→HLSL, patch the dropped `vertices`
/// modifier, then compile the patched HLSL with `slangc -pass-through dxc` (Slang's bundled
/// dxcompiler — no standalone dxc.exe needed). See `dxil_mesh_needs_vertices_patch`.
fn compile_mesh_dxil_via_hlsl_patch(
    slangc: &Path,
    tool_home: &Path,
    src_path: &Path,
    job: &Job,
    profile: &str,
    out_path: &Path,
) -> Result<(), String> {
    let hlsl_path = out_path.with_extension("patch.hlsl");
    // 1) slang → HLSL.
    let out = slang_command(slangc, tool_home)
        .arg(src_path)
        .args(["-target", "hlsl"])
        .args(["-entry", job.entry])
        .args(["-stage", job.stage])
        .args(["-profile", profile])
        .arg("-o")
        .arg(&hlsl_path)
        .output()
        .map_err(|e| format!("failed to launch slangc (HLSL): {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "slangc HLSL failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    // 2) patch the dropped `out vertices` modifier.
    let hlsl = std::fs::read_to_string(&hlsl_path).map_err(|e| e.to_string())?;
    let patched = patch_mesh_vertices_out(&hlsl);
    if !patched.contains("out vertices") {
        return Err(format!(
            "mesh-DXIL patch found no vertex-output parameter to fix in {} — the Slang emission \
             may have changed; review patch_mesh_vertices_out",
            hlsl_path.display()
        ));
    }
    std::fs::write(&hlsl_path, &patched).map_err(|e| e.to_string())?;
    // 3) patched HLSL → DXIL via slang's bundled DXC (pass-through).
    let out = slang_command(slangc, tool_home)
        .arg(&hlsl_path)
        .args(["-target", "dxil"])
        .args(["-entry", job.entry])
        .args(["-stage", job.stage])
        .args(["-profile", profile])
        .args(["-pass-through", "dxc"])
        .arg("-o")
        .arg(out_path)
        .output()
        .map_err(|e| format!("failed to launch slangc (pass-through dxc): {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "slangc pass-through dxc failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Preprocessor `-D` defines for a target. **Single source of truth** for both the
/// cache key and the slangc args. Slang's Metal target rejects the
/// `NonUniformResourceIndex` intrinsic (E36107) that SPIR-V/DXIL need for per-ray
/// descriptor selection; `RT_METAL_TARGET` compiles it out (Metal indexes argument
/// buffers non-uniformly without the decoration). This is also why the key MUST hash
/// defines even before permutations exist (M4.4): the same `rt_common.slang` produces
/// different bytecode with vs. without this define.
fn defines_for(target: &str, key: &str) -> &'static [(&'static str, &'static str)] {
    // The `gdf_gi_hwrt` permutation compiles the hardware-ray-tracing gather path in (`HWRT_GI`);
    // the default `gdf_gi` leaves it out so it references no acceleration structure — Slang then
    // omits the TLAS binding, keeping the scalable SW-default GI independent of RT capability and
    // free of the RT path's register pressure. HW-RT is loaded only when opted in (a High tier).
    let hwrt = key == "gdf_gi_hwrt_cs";
    match (target, hwrt) {
        ("metallib", true) => &[("RT_METAL_TARGET", "1"), ("HWRT_GI", "1")],
        ("metallib", false) => &[("RT_METAL_TARGET", "1")],
        (_, true) => &[("HWRT_GI", "1")],
        (_, false) => &[],
    }
}

/// `slangc -v` output (stdout+stderr) — folded into the base hash so a compiler
/// upgrade invalidates every cache entry.
fn slangc_version(slangc: &Path) -> Vec<u8> {
    Command::new(slangc)
        .arg("-v")
        .output()
        .ok()
        .map(|o| {
            let mut v = o.stdout;
            v.extend_from_slice(&o.stderr);
            v
        })
        .unwrap_or_default()
}

/// Load the cook-cache manifest (`<artifact>  <hex-hash>` per line). Missing/corrupt
/// lines are simply skipped — a malformed manifest degrades to a full recompile, never
/// a wrong cache hit.
fn load_manifest(path: &Path) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((name, hex)) = line.split_once(char::is_whitespace)
                && let Ok(h) = u64::from_str_radix(hex.trim(), 16)
            {
                map.insert(name.to_string(), h);
            }
        }
    }
    map
}

/// Write the manifest back, sorted for deterministic diffs.
fn save_manifest(path: &Path, map: &HashMap<String, u64>) {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut text = String::new();
    for (name, hash) in entries {
        text.push_str(&format!("{name}  {hash:016x}\n"));
    }
    let _ = std::fs::write(path, text);
}

/// The cache identity of one `(job, target)` cell: its content-hash key, the artifact
/// file name, the embed/output path, and the resolved profile/defines. Computed once
/// (the `profile_for`/`defines_for` single source) and used by both the parallel
/// pre-pass and the sequential emit loop so the two can never disagree on hit vs. miss.
struct CellKey {
    key_hash: u64,
    artifact_name: String,
    out_path: PathBuf,
    profile: Option<&'static str>,
    defines: &'static [(&'static str, &'static str)],
}

fn cell_key(
    base_hash: u64,
    src_bytes: &[u8],
    job: &Job,
    target: &str,
    ext: &str,
    cache_dir: &Path,
    out_dir: &Path,
) -> CellKey {
    let profile = profile_for(target, job.stage, job.key);
    let defines = defines_for(target, job.key);
    let defines_str: String = defines.iter().map(|(k, v)| format!("{k}={v};")).collect();
    let params = format!(
        "t={target};e={};s={};p={};d={defines_str}",
        job.entry,
        job.stage,
        profile.unwrap_or(""),
    );
    let key_hash = fnv1a(params.as_bytes(), fnv1a(src_bytes, base_hash));
    let artifact_name = format!("{}.{}", job.key, ext);
    // Cached artifacts live in the persistent per-OS cache dir; the Metal RT-pipeline
    // branch is uncached scratch and stays in OUT_DIR.
    let out_path = if target == "metallib" && is_rt_stage(job.stage) {
        out_dir.join(&artifact_name)
    } else {
        cache_dir.join(&artifact_name)
    };
    CellKey {
        key_hash,
        artifact_name,
        out_path,
        profile,
        defines,
    }
}

/// One queued slangc compile (a cache miss). `command` is fully built and ready to run.
struct WorkItem {
    artifact_name: String,
    command: Command,
}

/// The result of running one `WorkItem`'s slangc.
struct Outcome {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Run the queued slangc compiles concurrently (Phase 12 M4.5), keyed by artifact name.
/// Only cache *misses* reach here, so a cold/changed build's wall-clock drops from the
/// sum of all compiles to roughly `ceil(misses / threads)` of the slowest. Threads only
/// change *when* each slangc runs, never its inputs — bytecode is byte-for-byte
/// identical to the sequential path. Dependency-free (`std::thread` + a shared queue).
fn compile_parallel(work: Vec<WorkItem>) -> HashMap<String, Outcome> {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    if work.is_empty() {
        return HashMap::new();
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(work.len());
    let queue = Arc::new(Mutex::new(work.into_iter().collect::<VecDeque<_>>()));
    let results = Arc::new(Mutex::new(HashMap::new()));
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let queue = Arc::clone(&queue);
        let results = Arc::clone(&results);
        handles.push(std::thread::spawn(move || {
            loop {
                let item = queue.lock().unwrap().pop_front();
                let Some(mut item) = item else { break };
                let output = item
                    .command
                    .output()
                    .unwrap_or_else(|e| panic!("failed to launch slangc: {e}"));
                results.lock().unwrap().insert(
                    item.artifact_name,
                    Outcome {
                        success: output.status.success(),
                        stdout: output.stdout,
                        stderr: output.stderr,
                    },
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    Arc::into_inner(results)
        .expect("all worker threads have joined, so this is the last Arc")
        .into_inner()
        .unwrap()
}

/// One shader entry point to compile.
struct Job {
    /// Source file under `shaders/`.
    src: &'static str,
    /// Entry-point function name.
    entry: &'static str,
    /// Slang stage name.
    stage: &'static str,
    /// Key used to name the generated accessor (`<key>_spirv`, `<key>_dxil`).
    key: &'static str,
}

const JOBS: &[Job] = &[
    Job {
        src: "triangle.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "triangle_vs",
    },
    Job {
        src: "triangle.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "triangle_fs",
    },
    Job {
        src: "imgui.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "imgui_vs",
    },
    Job {
        src: "imgui.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "imgui_fs",
    },
    Job {
        src: "mesh.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "mesh_vs",
    },
    Job {
        src: "mesh.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "mesh_fs",
    },
    Job {
        src: "sky.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "sky_vs",
    },
    Job {
        src: "sky.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "sky_fs",
    },
    Job {
        src: "capture.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "capture_vs",
    },
    Job {
        src: "capture.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "capture_fs",
    },
    Job {
        src: "irradiance.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "irradiance_vs",
    },
    Job {
        src: "irradiance.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "irradiance_fs",
    },
    Job {
        src: "prefilter.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "prefilter_vs",
    },
    Job {
        src: "prefilter.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "prefilter_fs",
    },
    Job {
        src: "brdf.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "brdf_vs",
    },
    Job {
        src: "brdf.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "brdf_fs",
    },
    Job {
        src: "gbuffer.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "gbuffer_vs",
    },
    // GPU-skinning variant of the G-buffer vertex shader (animation Stage B.2).
    Job {
        src: "gbuffer.slang",
        entry: "vsMainSkinned",
        stage: "vertex",
        key: "gbuffer_skinned_vs",
    },
    // GPU-morph variant of the G-buffer vertex shader (animation Stage C optimization).
    Job {
        src: "gbuffer.slang",
        entry: "vsMainMorphed",
        stage: "vertex",
        key: "gbuffer_morphed_vs",
    },
    Job {
        src: "gbuffer.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "gbuffer_fs",
    },
    // Depth pre-pass fragment shader (pipeline rebaseline PR-1): depth-only + the
    // G-buffer's alpha-test discard, sharing the G-buffer vertex shaders unchanged so
    // the pre-pass depth is bit-identical to the base pass (EQUAL depth test premise).
    Job {
        src: "gbuffer.slang",
        entry: "fsDepth",
        stage: "fragment",
        key: "gbuffer_depth_fs",
    },
    // Deferred surface-decal fragment shader (decals A3): shares `vsMain`, writes only
    // RT0 = float4(albedo, alpha); the DecalAlbedo blend state alpha-blends it into the
    // G-buffer albedo and masks the other targets.
    Job {
        src: "gbuffer.slang",
        entry: "fsDecal",
        stage: "fragment",
        key: "gbuffer_decal_fs",
    },
    // Velocity (motion-vector) G-buffer channel (pipeline re-baseline PR-2, opt-in
    // `P_VELOCITY=1`): a separate opaque pass into an RG16Float target. Static + skinned +
    // morphed vertex variants (prev-transform single source) share the one motion fragment;
    // `csViz` colour-codes the target for DEBUG_VIEW=11.
    Job {
        src: "velocity.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "velocity_vs",
    },
    Job {
        src: "velocity.slang",
        entry: "vsMainSkinned",
        stage: "vertex",
        key: "velocity_skinned_vs",
    },
    Job {
        src: "velocity.slang",
        entry: "vsMainMorphed",
        stage: "vertex",
        key: "velocity_morphed_vs",
    },
    Job {
        src: "velocity.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "velocity_fs",
    },
    Job {
        src: "velocity.slang",
        entry: "csViz",
        stage: "compute",
        key: "velocity_viz_cs",
    },
    Job {
        src: "shadow.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "shadow_vs",
    },
    // GPU-skinning variant of the shadow vertex shader (animation Stage B.2b).
    Job {
        src: "shadow.slang",
        entry: "vsMainSkinned",
        stage: "vertex",
        key: "shadow_skinned_vs",
    },
    // GPU-morph variant of the shadow vertex shader (animation Stage C optimization).
    Job {
        src: "shadow.slang",
        entry: "vsMainMorphed",
        stage: "vertex",
        key: "shadow_morphed_vs",
    },
    Job {
        src: "shadow.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "shadow_fs",
    },
    Job {
        src: "pbr.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "pbr_vs",
    },
    Job {
        src: "pbr.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "pbr_fs",
    },
    Job {
        src: "post.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "post_vs",
    },
    Job {
        src: "post.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "post_fs",
    },
    Job {
        src: "atmosphere.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "atmosphere_vs",
    },
    Job {
        src: "atmosphere.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "atmosphere_fs",
    },
    // PR-5 post-process nodes (docs/post-process-chain.md). The Phase-5 `blur.slang`
    // demo (separable Gaussian) and the Phase-7 `post_compute.slang` 3x3 box-blur demo
    // were removed here: neither was a real effect (blur.slang had no consumer;
    // post_compute was a compute-graph demo, not bloom). They are superseded by the
    // ordered post sequence below.
    Job {
        src: "translucent.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "translucent_vs",
    },
    Job {
        src: "translucent.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "translucent_fs",
    },
    Job {
        src: "bloom.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "bloom_vs",
    },
    Job {
        src: "bloom.slang",
        entry: "fsPrefilter",
        stage: "fragment",
        key: "bloom_prefilter_fs",
    },
    Job {
        src: "bloom.slang",
        entry: "fsDownsample",
        stage: "fragment",
        key: "bloom_downsample_fs",
    },
    Job {
        src: "bloom.slang",
        entry: "fsUpsample",
        stage: "fragment",
        key: "bloom_upsample_fs",
    },
    Job {
        src: "motion_blur.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "motion_blur_vs",
    },
    Job {
        src: "motion_blur.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "motion_blur_fs",
    },
    Job {
        src: "dof.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "dof_vs",
    },
    Job {
        src: "dof.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "dof_fs",
    },
    Job {
        src: "particle_sim.slang",
        entry: "csMain",
        stage: "compute",
        key: "particle_sim_cs",
    },
    Job {
        src: "particle_draw.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "particle_draw_vs",
    },
    Job {
        src: "particle_draw.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "particle_draw_fs",
    },
    Job {
        src: "cull.slang",
        entry: "csReset",
        stage: "compute",
        key: "cull_reset_cs",
    },
    Job {
        src: "cull.slang",
        entry: "csCull",
        stage: "compute",
        key: "cull_cs",
    },
    Job {
        src: "light_cluster.slang",
        entry: "csBuildClusters",
        stage: "compute",
        key: "light_cluster_build_cs",
    },
    Job {
        src: "cull.slang",
        entry: "csCullHzb",
        stage: "compute",
        key: "cull_hzb_cs",
    },
    Job {
        src: "cull.slang",
        entry: "csClearStats",
        stage: "compute",
        key: "cull_stats_clear_cs",
    },
    Job {
        src: "hzb_build.slang",
        entry: "csCopy",
        stage: "compute",
        key: "hzb_copy_cs",
    },
    Job {
        src: "hzb_build.slang",
        entry: "csReduce",
        stage: "compute",
        key: "hzb_reduce_cs",
    },
    Job {
        src: "cull_draw.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "cull_draw_vs",
    },
    Job {
        src: "cull_draw.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "cull_draw_fs",
    },
    Job {
        src: "rt_trace.slang",
        entry: "csMain",
        stage: "compute",
        key: "rt_trace_cs",
    },
    Job {
        src: "rt_path.slang",
        entry: "csMain",
        stage: "compute",
        key: "rt_path_cs",
    },
    // Phase 11 Stage A (A1): compute software ray tracing via SDF sphere tracing.
    Job {
        src: "sdf_trace.slang",
        entry: "csMain",
        stage: "compute",
        key: "sdf_trace_cs",
    },
    // Phase 11 Stage B (B1): 3D volume texture RHI smoke test (fill + slice view).
    Job {
        src: "volume_test.slang",
        entry: "fillMain",
        stage: "compute",
        key: "volume_fill_cs",
    },
    Job {
        src: "volume_test.slang",
        entry: "viewMain",
        stage: "compute",
        key: "volume_view_cs",
    },
    // Phase 11 Stage B (B2): per-mesh SDF bake into a 3D storage volume.
    Job {
        src: "sdf_bake.slang",
        entry: "bakeMain",
        stage: "compute",
        key: "sdf_bake_cs",
    },
    // Phase 11 Stage C (C8a): per-voxel albedo bake (nearest-triangle color → 3 volumes).
    Job {
        src: "sdf_albedo_bake.slang",
        entry: "albedoBakeMain",
        stage: "compute",
        key: "sdf_albedo_bake_cs",
    },
    // Phase 11 Stage C (C8b1): mesh-card surface-cache capture (GDF-traced geometry+albedo).
    Job {
        src: "sdf_cache_capture.slang",
        entry: "cacheMain",
        stage: "compute",
        key: "sdf_cache_capture_cs",
    },
    // Phase 11 Stage C (C8b1): surface-cache atlas viz.
    Job {
        src: "sdf_cache_view.slang",
        entry: "viewMain",
        stage: "compute",
        key: "sdf_cache_view_cs",
    },
    // Phase 11 Stage C (C8b2): surface-cache lighting (continuous, multibounce).
    Job {
        src: "sdf_cache_light.slang",
        entry: "lightMain",
        stage: "compute",
        key: "sdf_cache_light_cs",
    },
    // Phase 11 Stage B (B3): merge per-mesh SDF instances into a global distance field.
    Job {
        src: "gdf_merge.slang",
        entry: "mergeMain",
        stage: "compute",
        key: "gdf_merge_cs",
    },
    // Phase 11 Stage B (B4): software ray trace against the merged global distance field.
    Job {
        src: "gdf_trace.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_trace_cs",
    },
    // Phase 11 Stage C (C2): GDF ambient occlusion into the deferred render.
    Job {
        src: "gdf_ao.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_ao_cs",
    },
    // Screen-space (HBAO-lite) near-field AO, composed with the GDF AO; + its bilateral blur.
    Job {
        src: "gtao.slang",
        entry: "csMain",
        stage: "compute",
        key: "gtao_cs",
    },
    Job {
        src: "gtao.slang",
        entry: "csBlur",
        stage: "compute",
        key: "gtao_blur_cs",
    },
    // Phase 11 Stage C (C3): stochastic 1-bounce diffuse GI against the GDF. Default = SW march.
    Job {
        src: "gdf_gi.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_gi_cs",
    },
    // F3 permutation: the same shader with the hardware-ray-tracing gather compiled in (`HWRT_GI`,
    // via `defines_for`). Loaded only when HW-RT GI is opted in (High tier); the default variant
    // above references no acceleration structure, so the SW-default GI stays RT-independent.
    Job {
        src: "gdf_gi.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_gi_hwrt_cs",
    },
    // 레퍼런스 엔진 GI-fidelity track: world-space irradiance volume (DDGI-lite radiance cache) update.
    Job {
        src: "gi_volume.slang",
        entry: "csMain",
        stage: "compute",
        key: "gi_volume_cs",
    },
    // Screen-space radiance probes: per-tile probe trace into the octahedral atlas (P1).
    Job {
        src: "screen_probe_trace.slang",
        entry: "csMain",
        stage: "compute",
        key: "screen_probe_trace_cs",
    },
    // Screen-space radiance probes: per-pixel gather of the probe atlas -> indirect irradiance.
    Job {
        src: "screen_probe_integrate.slang",
        entry: "csMain",
        stage: "compute",
        key: "screen_probe_integrate_cs",
    },
    // Screen-space radiance probes: spatial cross-probe joint-bilateral filter of the atlas (P2).
    Job {
        src: "screen_probe_filter.slang",
        entry: "csMain",
        stage: "compute",
        key: "screen_probe_filter_cs",
    },
    // World radiance cache (P4): per-frame update of the camera-following clipmap probe atlas.
    Job {
        src: "wrc_update.slang",
        entry: "csMain",
        stage: "compute",
        key: "wrc_update_cs",
    },
    // Screen-space radiance probes (P5): per-probe radiance -> irradiance pre-integration.
    Job {
        src: "screen_probe_irradiance.slang",
        entry: "csMain",
        stage: "compute",
        key: "screen_probe_irradiance_cs",
    },
    // GI-on-distance-field visualization: march the camera into the GDF, paint hits with the
    // world radiance cache's stored indirect irradiance.
    Job {
        src: "wrc_view.slang",
        entry: "csMain",
        stage: "compute",
        key: "wrc_view_cs",
    },
    // Physical-camera auto-exposure: luminance histogram (pass 1) → adapted exposure (pass 2).
    Job {
        src: "auto_exposure.slang",
        entry: "csHistogram",
        stage: "compute",
        key: "auto_exposure_histogram_cs",
    },
    Job {
        src: "auto_exposure.slang",
        entry: "csResolve",
        stage: "compute",
        key: "auto_exposure_resolve_cs",
    },
    // Stage D1 (Sponza 60fps): joint-bilateral upsample of the half-res GI to full res.
    Job {
        src: "gdf_gi_upsample.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_gi_upsample_cs",
    },
    // Stage D2b (Sponza 60fps): per-card camera-frustum visibility for the relight budget.
    Job {
        src: "sdf_cache_visibility.slang",
        entry: "csMain",
        stage: "compute",
        key: "sdf_cache_visibility_cs",
    },
    // QHD/UHD track: temporal upsampling (TAAU) — low-res jittered render -> full-res accumulation.
    Job {
        src: "taau.slang",
        entry: "csMain",
        stage: "compute",
        key: "taau_cs",
    },
    // QHD/UHD track: HDR-aware FXAA (the Decima FXAA->TAA pre-pass that stabilizes the jitter).
    Job {
        src: "fxaa.slang",
        entry: "csMain",
        stage: "compute",
        key: "fxaa_cs",
    },
    // Phase 11 Stage C (C4): spatio-temporal denoise of the noisy GI.
    Job {
        src: "gdf_temporal.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_temporal_cs",
    },
    Job {
        src: "gdf_atrous.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_atrous_cs",
    },
    // Phase 11 Stage C (C5): screen-space reflections (stochastic half-res trace).
    Job {
        src: "ssr.slang",
        entry: "csMain",
        stage: "compute",
        key: "ssr_cs",
    },
    // Phase 11 Stage C: stochastic SSR temporal resolve (reproject + EMA + clamp).
    Job {
        src: "ssr_resolve.slang",
        entry: "csMain",
        stage: "compute",
        key: "ssr_resolve_cs",
    },
    // Phase 11 Stage C (C6): GDF reflections (off-screen fallback for SSR misses).
    Job {
        src: "gdf_reflect.slang",
        entry: "csMain",
        stage: "compute",
        key: "gdf_reflect_cs",
    },
    // Phase 11 Stage C (C7): hybrid reflection composite (SSR over GDF / sky).
    Job {
        src: "reflect_composite.slang",
        entry: "csMain",
        stage: "compute",
        key: "reflect_composite_cs",
    },
    // Phase 11 Stage C (C8j): temporal resolve of the stochastic GGX GDF reflection.
    Job {
        src: "reflect_temporal.slang",
        entry: "csMain",
        stage: "compute",
        key: "reflect_temporal_cs",
    },
    // Phase 11 Stage C (C7b): lit-color history capture (raw radiance) for SSR reprojection.
    Job {
        src: "lit_history.slang",
        entry: "csMain",
        stage: "compute",
        key: "lit_history_cs",
    },
    // Phase 14 (virtual geometry) M0 capability smokes. `vgeo_atomic` proves the
    // cross-backend 64-bit `atomicMax` path (the visibility-buffer primitive);
    // `vgeo_meshlet` proves the mesh-shader pipeline. Both are opt-in (`--atomic64-test`
    // / `--mesh-shader-test`) and referenced by no default render pass, so the gallery
    // anchor stays byte-identical.
    Job {
        src: "vgeo_atomic.slang",
        entry: "csAtomicMax",
        stage: "compute",
        key: "vgeo_atomic_cs",
    },
    Job {
        src: "vgeo_meshlet.slang",
        entry: "meshMain",
        stage: "mesh",
        key: "vgeo_meshlet_ms",
    },
    Job {
        src: "vgeo_meshlet.slang",
        entry: "fragMain",
        stage: "fragment",
        key: "vgeo_meshlet_fs",
    },
    // Phase 14 M2: resident cluster render via a mesh shader (reads cluster geometry from
    // bindless storage buffers). Opt-in (`--vgeo-mesh`); no default pass references it.
    Job {
        src: "vgeo_cluster.slang",
        entry: "meshMain",
        stage: "mesh",
        key: "vgeo_cluster_ms",
    },
    Job {
        src: "vgeo_cluster.slang",
        entry: "fragMain",
        stage: "fragment",
        key: "vgeo_cluster_fs",
    },
    // Phase 14 M3: view-dependent LOD DAG cut selection (compute → visible list + indirect args).
    Job {
        src: "vgeo_cut.slang",
        entry: "csCut",
        stage: "compute",
        key: "vgeo_cut_cs",
    },
    // Phase 14 M5b: HW/SW binning cut — splits the cut into a HW (mesh-shader) list and a SW
    // (compute-raster) list by projected screen size, each with its own indirect args.
    Job {
        src: "vgeo_cut.slang",
        entry: "csCutBin",
        stage: "compute",
        key: "vgeo_cut_bin_cs",
    },
    // Phase 14 M5b: HW-path mesh shader writing into the same R64 visibility buffer as the SW
    // rasterizer (per-primitive triId + fragment atomicMax → seamless HW/SW boundary).
    Job {
        src: "vgeo_hwvis.slang",
        entry: "meshMain",
        stage: "mesh",
        key: "vgeo_hwvis_ms",
    },
    Job {
        src: "vgeo_hwvis.slang",
        entry: "fragMain",
        stage: "fragment",
        key: "vgeo_hwvis_fs",
    },
    // Phase 14 M5: software rasterizer into an R64 visibility buffer + its visualization.
    Job {
        src: "vgeo_swraster.slang",
        entry: "csClear",
        stage: "compute",
        key: "vgeo_swraster_clear_cs",
    },
    Job {
        src: "vgeo_swraster.slang",
        entry: "csRaster",
        stage: "compute",
        key: "vgeo_swraster_cs",
    },
    Job {
        src: "vgeo_visbuffer.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "vgeo_visbuffer_vs",
    },
    Job {
        src: "vgeo_visbuffer.slang",
        entry: "fragMain",
        stage: "fragment",
        key: "vgeo_visbuffer_fs",
    },
    // Phase 14 M6: material resolve — visibility buffer → analytic-barycentric attributes → shaded
    // surface (the deferred G-buffer stage; here shaded to match the M2 direct render for the gate).
    Job {
        src: "vgeo_resolve.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "vgeo_resolve_vs",
    },
    Job {
        src: "vgeo_resolve.slang",
        entry: "fragMain",
        stage: "fragment",
        key: "vgeo_resolve_fs",
    },
    // Phase 14 renderer integration: visibility buffer → real Phase-6 G-buffer MRT (+ SV_Depth),
    // so the deferred lighting pipeline consumes virtual geometry unchanged (drop-in producer).
    Job {
        src: "vgeo_gbuffer.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "vgeo_gbuffer_vs",
    },
    Job {
        src: "vgeo_gbuffer.slang",
        entry: "fsGBuffer",
        stage: "fragment",
        key: "vgeo_gbuffer_fs",
    },
    // Full ray-tracing pipeline (Phase 8 M5): raygen / miss / closest-hit compiled
    // as separate entry points. On DXIL these emit a shader *library* (lib_6_5);
    // see the profile selection below.
    Job {
        src: "rt_pipeline.slang",
        entry: "rgMain",
        stage: "raygeneration",
        key: "rt_pipeline_rgen",
    },
    Job {
        src: "rt_pipeline.slang",
        entry: "msMain",
        stage: "miss",
        key: "rt_pipeline_miss",
    },
    Job {
        src: "rt_pipeline.slang",
        entry: "chMain",
        stage: "closesthit",
        key: "rt_pipeline_chit",
    },
];

/// Whether `stage` is a ray-tracing stage (compiled to a DXIL library).
fn is_rt_stage(stage: &str) -> bool {
    matches!(
        stage,
        "raygeneration" | "miss" | "closesthit" | "anyhit" | "intersection" | "callable"
    )
}

/// (slang target name, output file extension, generated accessor suffix, required).
///
/// Which targets are actually compiled depends on the build target OS (see
/// `targets_for_os`): Windows builds SPIR-V (Vulkan) + DXIL (D3D12); macOS builds
/// `metallib` (Metal). SPIR-V is required on Windows; DXIL and metallib are
/// optional (a failure — e.g. a missing DXC / Metal toolchain — is downgraded to a
/// warning + `None` accessor rather than failing the build).
const TARGETS: &[(&str, &str, &str, bool)] = &[
    ("spirv", "spv", "spirv", true),
    ("dxil", "dxil", "dxil", false),
    ("metallib", "metallib", "metallib", false),
];

/// Whether `target` should be compiled for the given build `target_os`. Each
/// platform only builds the bytecode its RHI backend consumes (Windows: SPIR-V +
/// DXIL; macOS: metallib); accessors for the rest are emitted as `None`.
fn target_selected(target: &str, target_os: &str) -> bool {
    match target_os {
        "macos" => target == "metallib",
        _ => target == "spirv" || target == "dxil",
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SLANGC");
    for job in JOBS {
        println!("cargo:rerun-if-changed=shaders/{}", job.src);
    }
    // Shared includes are not in JOBS but several shaders `#include` them; watch them
    // explicitly (and fold them into `base_hash` below) so an edit both re-runs this
    // script and invalidates the cache of every shader that includes them. Missing one
    // silently ships stale bytecode — the whole set must be listed.
    for inc in SHARED_INCLUDES {
        println!("cargo:rerun-if-changed=shaders/{inc}");
    }
    println!("cargo:rerun-if-env-changed=DXC");
    println!("cargo:rerun-if-env-changed=METAL_SHADERCONVERTER");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let shader_tool_home = out_dir.join("shader-tool-home");
    std::fs::create_dir_all(shader_tool_home.join(".cache")).unwrap();

    let mut generated = String::new();

    let Some(slangc) = find_slangc(&manifest_dir) else {
        println!(
            "cargo:warning=slangc not found — shaders will be unavailable. Set SLANGC, place \
             slangc (slangc.exe on Windows) in tools/slang/bin/, or add it to PATH. Releases: \
             https://github.com/shader-slang/slang/releases"
        );
        emit_all_none(&mut generated);
        std::fs::write(out_dir.join("shaders.rs"), generated).unwrap();
        return;
    };
    println!("cargo:warning=using slangc at {}", slangc.display());

    // Build target OS (set by Cargo for build scripts) selects which bytecode
    // targets to compile (Windows: spirv+dxil; macOS: metallib).
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut rt_pipeline_isect_emitted = false;
    let mut rt_pipeline_dispatch_emitted = false;

    // --- Shader cook cache (Phase 12 M4.1/M4.2): content-hash + manifest, skip slangc
    // on hit. The base hash folds the compiler version and the shared includes (§3 step
    // 1, conservative) into every job's key, so a compiler upgrade or a shared-include
    // edit invalidates all entries.
    //
    // M4.2: cooked bytecode + manifest live in a persistent per-OS cache dir under the
    // crate (`compiled/<os>/`), embedded via `include_bytes!` from there — so they
    // survive a `cargo clean` of `target/` and a no-change rebuild costs zero slangc.
    // The dir is local-only (gitignored, §5). The Metal RT-pipeline branch stays on
    // OUT_DIR (uncached, macOS-only — see the branch below).
    let cache_dir = manifest_dir.join("compiled").join(if target_os.is_empty() {
        "unknown"
    } else {
        target_os.as_str()
    });
    std::fs::create_dir_all(&cache_dir).unwrap();
    let manifest_path = cache_dir.join("manifest.txt");
    let mut manifest = load_manifest(&manifest_path);
    let mut base_hash = fnv1a(&slangc_version(&slangc), FNV_OFFSET);
    for inc in SHARED_INCLUDES {
        if let Ok(bytes) = std::fs::read(shader_dir.join(inc)) {
            base_hash = fnv1a(&bytes, base_hash);
        }
    }
    let mut cache_compiled = 0usize;
    let mut cache_hits = 0usize;

    // M4.5 pre-pass: collect every main-branch cache MISS and compile them in parallel.
    // Cache hits and the macOS-only Metal RT-pipeline branch are handled inline in the
    // emit loop below; only plain slangc misses are parallelized. The emit loop recomputes
    // the same `cell_key`, so its hit/miss decision matches this pass exactly.
    let mut work: Vec<WorkItem> = Vec::new();
    for job in JOBS {
        let src_bytes = std::fs::read(shader_dir.join(job.src)).unwrap_or_default();
        for (target, ext, _suffix, _required) in TARGETS {
            if !target_selected(target, &target_os)
                || (*target == "metallib" && is_rt_stage(job.stage))
                // The 3-output mesh-DXIL workaround is compiled inline in the emit loop
                // (two slangc invocations + an HLSL patch), not through the single-Command
                // parallel queue — mirror the Metal RT-pipeline branch's inline handling.
                || (*target == "dxil" && dxil_mesh_needs_vertices_patch(job.key))
            {
                continue;
            }
            let ck = cell_key(
                base_hash, &src_bytes, job, target, ext, &cache_dir, &out_dir,
            );
            if manifest.get(&ck.artifact_name) == Some(&ck.key_hash) && ck.out_path.exists() {
                continue; // cache hit — no compile needed
            }
            let mut command = slang_command(&slangc, &shader_tool_home);
            command
                .arg(shader_dir.join(job.src))
                .args(["-target", target])
                .args(["-entry", job.entry])
                .args(["-stage", job.stage]);
            for &(k, v) in ck.defines {
                command.args(["-D", &format!("{k}={v}")]);
            }
            if let Some(p) = ck.profile {
                command.args(["-profile", p]);
            }
            // Preserve the real SPIR-V entry-point name (Slang defaults to "main").
            if *target == "spirv" {
                command.arg("-fvk-use-entrypoint-name");
            }
            command.arg("-o").arg(&ck.out_path);
            work.push(WorkItem {
                artifact_name: ck.artifact_name,
                command,
            });
        }
    }
    let results = compile_parallel(work);

    for job in JOBS {
        let src_path = shader_dir.join(job.src);
        let src_bytes = std::fs::read(&src_path).unwrap_or_default();
        for (target, ext, suffix, required) in TARGETS {
            // Emit a `None` accessor for targets this platform doesn't use, so the
            // generated `<key>_<suffix>()` function always exists.
            if !target_selected(target, &target_os) {
                emit_none(&mut generated, job.key, suffix);
                if *target == "metallib" && is_rt_stage(job.stage) {
                    emit_text_none(&mut generated, job.key, "metal_reflection");
                }
                continue;
            }

            if *target == "metallib" && is_rt_stage(job.stage) {
                // Metal RT-pipeline: uncached scratch in OUT_DIR (macOS-only).
                let out_path = out_dir.join(format!("{}.{}", job.key, ext));
                match compile_rt_pipeline_metal(MetalRtCompileRequest {
                    manifest_dir: &manifest_dir,
                    shader_dir: &shader_dir,
                    src_path: &src_path,
                    slangc: &slangc,
                    shader_tool_home: &shader_tool_home,
                    job,
                    out_dir: &out_dir,
                    out_path: &out_path,
                }) {
                    Ok(output) => {
                        emit_some(&mut generated, job.key, suffix, &out_path);
                        emit_text_some(
                            &mut generated,
                            job.key,
                            "metal_reflection",
                            &output.reflection_path,
                        );
                        if let Some(intersection_path) = output.intersection_path {
                            emit_some(
                                &mut generated,
                                RT_PIPELINE_ISECT_KEY,
                                "metallib",
                                &intersection_path,
                            );
                            rt_pipeline_isect_emitted = true;
                        }
                        if let Some(dispatch_path) = output.dispatch_path {
                            emit_some(
                                &mut generated,
                                RT_PIPELINE_DISPATCH_KEY,
                                "metallib",
                                &dispatch_path,
                            );
                            rt_pipeline_dispatch_emitted = true;
                        }
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning={} [{}/metal-rt-pipeline] skipped (optional target unavailable): {e}",
                            job.src, job.entry
                        );
                        emit_none(&mut generated, job.key, suffix);
                        emit_text_none(&mut generated, job.key, "metal_reflection");
                        if job.stage == "closesthit" {
                            emit_none(&mut generated, RT_PIPELINE_ISECT_KEY, "metallib");
                            rt_pipeline_isect_emitted = true;
                        }
                        if job.stage == "raygeneration" {
                            emit_none(&mut generated, RT_PIPELINE_DISPATCH_KEY, "metallib");
                            rt_pipeline_dispatch_emitted = true;
                        }
                    }
                }
                continue;
            }

            // 3-output mesh-DXIL workaround (Slang codegen bug): compiled inline via
            // slang→HLSL→patch→DXC. Cached like any other cell (manifest + on-disk artifact),
            // just produced by a dedicated two-step path. Skipped in the parallel pre-pass.
            if *target == "dxil" && dxil_mesh_needs_vertices_patch(job.key) {
                let ck = cell_key(
                    base_hash, &src_bytes, job, target, ext, &cache_dir, &out_dir,
                );
                if manifest.get(&ck.artifact_name) == Some(&ck.key_hash) && ck.out_path.exists() {
                    emit_some(&mut generated, job.key, suffix, &ck.out_path);
                    cache_hits += 1;
                    continue;
                }
                match compile_mesh_dxil_via_hlsl_patch(
                    &slangc,
                    &shader_tool_home,
                    &src_path,
                    job,
                    ck.profile.unwrap_or("sm_6_6"),
                    &ck.out_path,
                ) {
                    Ok(()) => {
                        emit_some(&mut generated, job.key, suffix, &ck.out_path);
                        manifest.insert(ck.artifact_name, ck.key_hash);
                        cache_compiled += 1;
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning={} [{}/dxil] mesh-DXIL workaround failed \
                             (optional target unavailable): {e}",
                            job.src, job.entry
                        );
                        emit_none(&mut generated, job.key, suffix);
                    }
                }
                continue;
            }

            // Same cache identity as the pre-pass: emit from the on-disk cache (hit) or
            // from the parallel compile result (miss). cell_key (via profile_for /
            // defines_for) is the single source so a hit can never claim bytecode that
            // was compiled with different flags.
            let ck = cell_key(
                base_hash, &src_bytes, job, target, ext, &cache_dir, &out_dir,
            );
            if manifest.get(&ck.artifact_name) == Some(&ck.key_hash) && ck.out_path.exists() {
                // Identical source + params + compiler → byte-identical bytecode already
                // on disk. The slangc subprocess was skipped entirely.
                emit_some(&mut generated, job.key, suffix, &ck.out_path);
                cache_hits += 1;
                continue;
            }
            let outcome = results
                .get(&ck.artifact_name)
                .expect("every cache miss was queued for compilation in the pre-pass");

            if outcome.success {
                emit_some(&mut generated, job.key, suffix, &ck.out_path);
                manifest.insert(ck.artifact_name, ck.key_hash);
                cache_compiled += 1;
            } else if *required {
                panic!(
                    "slangc failed for {} [{}/{}]:\n{}\n{}",
                    job.src,
                    job.entry,
                    target,
                    String::from_utf8_lossy(&outcome.stdout),
                    String::from_utf8_lossy(&outcome.stderr),
                );
            } else {
                println!(
                    "cargo:warning={} [{}/{}] skipped (optional target unavailable): {}",
                    job.src,
                    job.entry,
                    target,
                    String::from_utf8_lossy(&outcome.stderr).trim()
                );
                emit_none(&mut generated, job.key, suffix);
                if *target == "metallib" && is_rt_stage(job.stage) {
                    emit_text_none(&mut generated, job.key, "metal_reflection");
                }
            }
        }
    }

    if !rt_pipeline_isect_emitted {
        emit_none(&mut generated, RT_PIPELINE_ISECT_KEY, "metallib");
    }
    if !rt_pipeline_dispatch_emitted {
        emit_none(&mut generated, RT_PIPELINE_DISPATCH_KEY, "metallib");
    }

    save_manifest(&manifest_path, &manifest);
    println!("cargo:warning=shader-cache: {cache_compiled} compiled, {cache_hits} cached");

    std::fs::write(out_dir.join("shaders.rs"), generated).unwrap();
}

/// Resolve the slangc executable per the documented search order.
fn find_slangc(manifest_dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SLANGC") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }

    // The slangc binary runs on the host, so its name follows the host OS.
    let exe = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };

    // Workspace root is two levels up from crates/shader.
    if let Some(workspace_root) = manifest_dir.parent().and_then(Path::parent) {
        let vendored = workspace_root.join("tools/slang/bin").join(exe);
        if vendored.is_file() {
            return Some(vendored);
        }
    }

    if Command::new("slangc").arg("-v").output().is_ok() {
        return Some(PathBuf::from("slangc"));
    }

    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let sdk_slangc = PathBuf::from(sdk).join("Bin").join(exe);
        if sdk_slangc.is_file() {
            return Some(sdk_slangc);
        }
    }

    None
}

fn slang_command(slangc: &Path, tool_home: &Path) -> Command {
    let mut command = Command::new(slangc);
    // Apple `metal` (invoked by Slang's `-target metallib`) writes clang module
    // cache files below HOME. Keep that cache inside Cargo's OUT_DIR so sandboxed
    // builds can compile `metal_types` instead of trying `~/.cache/clang`.
    command.env("HOME", tool_home);
    command.env("XDG_CACHE_HOME", tool_home.join(".cache"));
    // On macOS, point slangc's `-target metallib` at a developer dir that actually
    // provides Apple's `metal` compiler, so the build works even when `xcode-select`
    // is left on the Command Line Tools (which lack the Metal Toolchain). No-op on
    // other platforms and when `metal` is already reachable / explicitly configured.
    if let Some(dir) = metal_developer_dir() {
        command.env("DEVELOPER_DIR", dir);
    }
    command
}

/// A `DEVELOPER_DIR` to feed slangc so Apple's `metal` compiler resolves, or `None`
/// when no override is needed (already reachable / `DEVELOPER_DIR` already set) or
/// possible (not macOS / no toolchain found). Computed once and cached.
///
/// The Metal Toolchain (which provides the `metal` CLI that `-target metallib`
/// shells out to) ships under **Xcode.app**, not the standalone Command Line Tools.
/// If `xcode-select` points at the CLT — the default after a CLT-only install — every
/// metallib silently degrades to a `None` accessor and the Metal backend ends up with
/// no shaders. Discovering a working developer dir here makes a checkout build on any
/// macOS box that has Xcode + the Metal Toolchain installed, regardless of the active
/// `xcode-select`.
fn metal_developer_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        // Only relevant when building the metallib target (macOS).
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
            return None;
        }
        // Respect an explicit override and skip if the active toolchain already works.
        if std::env::var_os("DEVELOPER_DIR").is_some() || metal_reachable(None) {
            return None;
        }
        // Probe the current xcode-select target (in case it is a full Xcode whose
        // `metal` just didn't resolve above) and the common Xcode install paths.
        let mut candidates: Vec<PathBuf> = Vec::new();
        let selected = Command::new("xcode-select")
            .arg("-p")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|p| !p.is_empty());
        if let Some(p) = selected {
            candidates.push(PathBuf::from(p));
        }
        candidates.push("/Applications/Xcode.app/Contents/Developer".into());
        candidates.push("/Applications/Xcode-beta.app/Contents/Developer".into());

        for cand in candidates {
            if cand.is_dir() && metal_reachable(Some(&cand)) {
                println!(
                    "cargo:warning=metallib: using DEVELOPER_DIR={} for Apple `metal` \
                     (active xcode-select lacks the Metal Toolchain)",
                    cand.display()
                );
                return Some(cand);
            }
        }
        None
    })
    .as_deref()
}

/// Whether Apple's `metal` compiler resolves via `xcrun`, optionally under a specific
/// `DEVELOPER_DIR`. Used to pick a developer dir that can build metallibs.
fn metal_reachable(developer_dir: Option<&Path>) -> bool {
    let mut cmd = Command::new("xcrun");
    cmd.args(["--find", "metal"]);
    if let Some(dir) = developer_dir {
        cmd.env("DEVELOPER_DIR", dir);
    }
    matches!(cmd.output(), Ok(o) if o.status.success())
}

/// Compile the DXR-style RT pipeline shaders to Metal via Apple's Metal Shader
/// Converter. Slang's Metal target cannot compile DXR pipeline stages directly,
/// but it can emit HLSL per entry point; DXC compiles that HLSL to DXIL, then the
/// converter produces a `.metallib` plus reflection for the converter TLAB.
struct MetalRtCompileOutput {
    reflection_path: PathBuf,
    intersection_path: Option<PathBuf>,
    dispatch_path: Option<PathBuf>,
}

struct MetalRtCompileRequest<'a> {
    manifest_dir: &'a Path,
    shader_dir: &'a Path,
    src_path: &'a Path,
    slangc: &'a Path,
    shader_tool_home: &'a Path,
    job: &'a Job,
    out_dir: &'a Path,
    out_path: &'a Path,
}

fn compile_rt_pipeline_metal(
    request: MetalRtCompileRequest<'_>,
) -> std::result::Result<MetalRtCompileOutput, String> {
    let dxc = find_dxc(request.manifest_dir).ok_or_else(|| {
        "DXC not found (set DXC, build tools/dxc-src with tools/build-dxc.sh, or add dxc to PATH)"
            .to_string()
    })?;
    let converter = find_metal_shaderconverter().ok_or_else(|| {
        "metal-shaderconverter not found (set METAL_SHADERCONVERTER or add it to PATH)".to_string()
    })?;
    let rootsig = request.shader_dir.join("rt_pipeline_metal_rootsig.json");
    if !rootsig.is_file() {
        return Err(format!("{} missing", rootsig.display()));
    }

    let hlsl_path = request.out_dir.join(format!("{}.hlsl", request.job.key));
    let dxil_path = request
        .out_dir
        .join(format!("{}.metal.dxil", request.job.key));
    let reflection_path = request
        .out_dir
        .join(format!("{}.metal.json", request.job.key));

    let slang_output = slang_command(request.slangc, request.shader_tool_home)
        .arg(request.src_path)
        .args(["-target", "hlsl"])
        .args(["-entry", request.job.entry])
        .args(["-stage", request.job.stage])
        .args(["-profile", "lib_6_5"])
        .arg("-o")
        .arg(&hlsl_path)
        .output()
        .map_err(|e| format!("failed to launch slangc for HLSL: {e}"))?;
    if !slang_output.status.success() {
        return Err(format!(
            "slangc HLSL failed:\n{}\n{}",
            String::from_utf8_lossy(&slang_output.stdout),
            String::from_utf8_lossy(&slang_output.stderr)
        ));
    }

    let dxc_output = Command::new(&dxc)
        .arg(&hlsl_path)
        .args(["-T", "lib_6_5"])
        .args(["-E", request.job.entry])
        .arg("-Fo")
        .arg(&dxil_path)
        .output()
        .map_err(|e| format!("failed to launch dxc: {e}"))?;
    if !dxc_output.status.success() {
        return Err(format!(
            "dxc failed:\n{}\n{}",
            String::from_utf8_lossy(&dxc_output.stdout),
            String::from_utf8_lossy(&dxc_output.stderr)
        ));
    }

    let mut converter_cmd = Command::new(&converter);
    converter_cmd
        .arg(&dxil_path)
        .args(["--entry-point", request.job.entry])
        .arg("--root-signature")
        .arg(&rootsig)
        .args(["--rt-maximum-attribute-size-in-bytes", "8"])
        .arg("--rt-enable-function-groups")
        .arg("--output-reflection-file")
        .arg(&reflection_path)
        .arg("-o")
        .arg(request.out_path);
    if request.job.stage == "raygeneration" {
        converter_cmd.arg("--rt-ray-generation-compilation=kernel");
    } else if request.job.stage == "closesthit" {
        converter_cmd.arg("--rt-hit-group-type=triangles");
    }

    let converter_output = converter_cmd
        .output()
        .map_err(|e| format!("failed to launch metal-shaderconverter: {e}"))?;
    if !converter_output.status.success() {
        return Err(format!(
            "metal-shaderconverter failed:\n{}\n{}",
            String::from_utf8_lossy(&converter_output.stdout),
            String::from_utf8_lossy(&converter_output.stderr)
        ));
    }

    let intersection_path = if request.job.stage == "closesthit" {
        let path = request
            .out_dir
            .join(format!("{RT_PIPELINE_ISECT_KEY}.metallib"));
        let synth_output = Command::new(&converter)
            .arg(&dxil_path)
            .args(["--entry-point", request.job.entry])
            .arg("--root-signature")
            .arg(&rootsig)
            .args(["--rt-maximum-attribute-size-in-bytes", "8"])
            .arg("--rt-enable-function-groups")
            .arg("--synthesize-indirect-intersection-function")
            .arg("--rt-hit-group-type=triangles")
            .arg("-o")
            .arg(&path)
            .output()
            .map_err(|e| format!("failed to launch metal-shaderconverter synth: {e}"))?;
        if !synth_output.status.success() {
            return Err(format!(
                "metal-shaderconverter synth failed:\n{}\n{}",
                String::from_utf8_lossy(&synth_output.stdout),
                String::from_utf8_lossy(&synth_output.stderr)
            ));
        }
        Some(path)
    } else {
        None
    };

    let dispatch_path = if request.job.stage == "raygeneration" {
        let path = request
            .out_dir
            .join(format!("{RT_PIPELINE_DISPATCH_KEY}.metallib"));
        let synth_output = Command::new(&converter)
            .arg(&dxil_path)
            .args(["--entry-point", request.job.entry])
            .arg("--root-signature")
            .arg(&rootsig)
            .args(["--rt-maximum-attribute-size-in-bytes", "8"])
            .arg("--rt-enable-function-groups")
            .arg("--synthesize-indirect-ray-dispatch")
            .arg("-o")
            .arg(&path)
            .output()
            .map_err(|e| format!("failed to launch metal-shaderconverter dispatch synth: {e}"))?;
        if !synth_output.status.success() {
            return Err(format!(
                "metal-shaderconverter dispatch synth failed:\n{}\n{}",
                String::from_utf8_lossy(&synth_output.stdout),
                String::from_utf8_lossy(&synth_output.stderr)
            ));
        }
        Some(path)
    } else {
        None
    };

    Ok(MetalRtCompileOutput {
        reflection_path,
        intersection_path,
        dispatch_path,
    })
}

fn find_dxc(manifest_dir: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DXC") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }

    let exe = if cfg!(windows) { "dxc.exe" } else { "dxc" };
    if let Some(workspace_root) = manifest_dir.parent().and_then(Path::parent) {
        let local = workspace_root.join("tools/dxc-src/build/bin").join(exe);
        if local.is_file() {
            return Some(local);
        }
        let vendored = workspace_root.join("tools/dxc/bin").join(exe);
        if vendored.is_file() {
            return Some(vendored);
        }
    }

    if Command::new("dxc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("dxc"));
    }

    None
}

fn find_metal_shaderconverter() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("METAL_SHADERCONVERTER") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }

    if Command::new("metal-shaderconverter")
        .arg("--version")
        .output()
        .is_ok()
    {
        return Some(PathBuf::from("metal-shaderconverter"));
    }

    None
}

/// Emit an accessor that returns the compiled bytes.
fn emit_some(out: &mut String, key: &str, suffix: &str, path: &Path) {
    // include_bytes! accepts forward slashes on Windows; normalize to avoid
    // escaping backslashes in the generated string literal.
    let normalized = path.to_string_lossy().replace('\\', "/");
    out.push_str(&format!(
        "/// Compiled `{key}` ({suffix}) bytecode.\n\
         pub fn {key}_{suffix}() -> Option<&'static [u8]> {{ \
         Some(include_bytes!(\"{normalized}\")) }}\n"
    ));
}

/// Emit an accessor that returns generated UTF-8 text (reflection JSON, etc.).
fn emit_text_some(out: &mut String, key: &str, suffix: &str, path: &Path) {
    let normalized = path.to_string_lossy().replace('\\', "/");
    out.push_str(&format!(
        "/// Compiled `{key}` ({suffix}) text.\n\
         pub fn {key}_{suffix}() -> Option<&'static str> {{ \
         Some(include_str!(\"{normalized}\")) }}\n"
    ));
}

/// Emit a single stub accessor that returns `None`.
fn emit_none(out: &mut String, key: &str, suffix: &str) {
    out.push_str(&format!(
        "/// Compiled `{key}` ({suffix}) bytecode (unavailable).\n\
         pub fn {key}_{suffix}() -> Option<&'static [u8]> {{ None }}\n"
    ));
}

/// Emit a text stub accessor that returns `None`.
fn emit_text_none(out: &mut String, key: &str, suffix: &str) {
    out.push_str(&format!(
        "/// Compiled `{key}` ({suffix}) text (unavailable).\n\
         pub fn {key}_{suffix}() -> Option<&'static str> {{ None }}\n"
    ));
}

/// Emit stub accessors (all `None`) when shaders could not be compiled.
fn emit_all_none(out: &mut String) {
    for job in JOBS {
        for (_, _, suffix, _) in TARGETS {
            emit_none(out, job.key, suffix);
        }
        if is_rt_stage(job.stage) {
            emit_text_none(out, job.key, "metal_reflection");
        }
    }
    emit_none(out, RT_PIPELINE_ISECT_KEY, "metallib");
    emit_none(out, RT_PIPELINE_DISPATCH_KEY, "metallib");
}
