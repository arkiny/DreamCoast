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
];

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

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");

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

    for job in JOBS {
        let src_path = shader_dir.join(job.src);
        for (target, ext, suffix, required) in TARGETS {
            // Emit a `None` accessor for targets this platform doesn't use, so the
            // generated `<key>_<suffix>()` function always exists.
            if !target_selected(target, &target_os) {
                emit_none(&mut generated, job.key, suffix);
                continue;
            }

            let out_path = out_dir.join(format!("{}.{}", job.key, ext));

            let mut command = Command::new(&slangc);
            command
                .arg(&src_path)
                .args(["-target", target])
                .args(["-entry", job.entry])
                .args(["-stage", job.stage]);
            // The HLSL shader profile applies to the SPIR-V / DXIL targets; the
            // Metal target derives everything it needs from the stage.
            if *target == "spirv" || *target == "dxil" {
                command.args(["-profile", "sm_6_5"]);
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
            }
        }
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

/// Emit a single stub accessor that returns `None`.
fn emit_none(out: &mut String, key: &str, suffix: &str) {
    out.push_str(&format!(
        "/// Compiled `{key}` ({suffix}) bytecode (unavailable).\n\
         pub fn {key}_{suffix}() -> Option<&'static [u8]> {{ None }}\n"
    ));
}

/// Emit stub accessors (all `None`) when shaders could not be compiled.
fn emit_all_none(out: &mut String) {
    for job in JOBS {
        for (_, _, suffix, _) in TARGETS {
            emit_none(out, job.key, suffix);
        }
    }
}
