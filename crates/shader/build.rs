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

use std::path::{Path, PathBuf};
use std::process::Command;

const RT_PIPELINE_ISECT_KEY: &str = "rt_pipeline_isect";
const RT_PIPELINE_DISPATCH_KEY: &str = "rt_pipeline_dispatch";

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
    Job {
        src: "gbuffer.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "gbuffer_fs",
    },
    Job {
        src: "shadow.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "shadow_vs",
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
        src: "blur.slang",
        entry: "vsMain",
        stage: "vertex",
        key: "blur_vs",
    },
    Job {
        src: "blur.slang",
        entry: "fsMain",
        stage: "fragment",
        key: "blur_fs",
    },
    Job {
        src: "post_compute.slang",
        entry: "csMain",
        stage: "compute",
        key: "post_compute_cs",
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
    // Shared includes are not in JOBS but several shaders `#include` them; watch
    // them explicitly so edits trigger a recompile of the including shaders.
    println!("cargo:rerun-if-changed=shaders/bindless.slang");
    println!("cargo:rerun-if-changed=shaders/rt_common.slang");
    println!("cargo:rerun-if-changed=shaders/rt_pipeline_metal_rootsig.json");
    println!("cargo:rerun-if-env-changed=DXC");
    println!("cargo:rerun-if-env-changed=METAL_SHADERCONVERTER");

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

    for job in JOBS {
        let src_path = shader_dir.join(job.src);
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

            let out_path = out_dir.join(format!("{}.{}", job.key, ext));

            if *target == "metallib" && is_rt_stage(job.stage) {
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

            let mut command = slang_command(&slangc, &shader_tool_home);
            command
                .arg(&src_path)
                .args(["-target", target])
                .args(["-entry", job.entry])
                .args(["-stage", job.stage]);
            // The HLSL shader profile applies to the SPIR-V / DXIL targets; the
            // Metal target derives everything it needs from the stage. Ray-tracing
            // stages compile to a DXIL *library* (lib_6_5+) — inline `RayQuery`
            // inside a hit shader requires >= 6.5; SPIR-V uses the same `sm_6_5`
            // profile as every other stage (the RT capability comes from the stage).
            if *target == "spirv" {
                command.args(["-profile", "sm_6_5"]);
            } else if *target == "dxil" {
                command.args([
                    "-profile",
                    if is_rt_stage(job.stage) {
                        "lib_6_5"
                    } else {
                        "sm_6_5"
                    },
                ]);
            }
            // By default Slang names the SPIR-V entry point "main"; preserve the
            // real name so the pipeline can bind it by entry name.
            if *target == "spirv" {
                command.arg("-fvk-use-entrypoint-name");
            }
            let output = command
                .arg("-o")
                .arg(&out_path)
                .output()
                .unwrap_or_else(|e| panic!("failed to launch slangc: {e}"));

            if output.status.success() {
                emit_some(&mut generated, job.key, suffix, &out_path);
            } else if *required {
                panic!(
                    "slangc failed for {} [{}/{}]:\n{}\n{}",
                    job.src,
                    job.entry,
                    target,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            } else {
                println!(
                    "cargo:warning={} [{}/{}] skipped (optional target unavailable): {}",
                    job.src,
                    job.entry,
                    target,
                    String::from_utf8_lossy(&output.stderr).trim()
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
    command
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
