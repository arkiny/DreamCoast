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
];

/// (slang target name, output file extension, generated accessor suffix, required).
///
/// SPIR-V is required (Vulkan, the active backend). DXIL is optional for now:
/// Slang emits it via a bundled DXC (`dxcompiler.dll`) which the standalone
/// release may omit, and D3D12 only arrives in Phase 2. A DXIL failure is
/// downgraded to a warning + `None` accessor rather than failing the build.
const TARGETS: &[(&str, &str, &str, bool)] = &[
    ("spirv", "spv", "spirv", true),
    ("dxil", "dxil", "dxil", false),
];

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
             slangc.exe in tools/slang/bin/, or add it to PATH. Releases: \
             https://github.com/shader-slang/slang/releases"
        );
        emit_all_none(&mut generated);
        std::fs::write(out_dir.join("shaders.rs"), generated).unwrap();
        return;
    };
    println!("cargo:warning=using slangc at {}", slangc.display());

    for job in JOBS {
        let src_path = shader_dir.join(job.src);
        for (target, ext, suffix, required) in TARGETS {
            let out_path = out_dir.join(format!("{}.{}", job.key, ext));
            let profile = "sm_6_5";

            let mut command = Command::new(&slangc);
            command
                .arg(&src_path)
                .args(["-target", target])
                .args(["-profile", profile])
                .args(["-entry", job.entry])
                .args(["-stage", job.stage]);
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

    // Workspace root is two levels up from crates/shader.
    if let Some(workspace_root) = manifest_dir.parent().and_then(Path::parent) {
        let vendored = workspace_root.join("tools/slang/bin/slangc.exe");
        if vendored.is_file() {
            return Some(vendored);
        }
    }

    if Command::new("slangc").arg("-v").output().is_ok() {
        return Some(PathBuf::from("slangc"));
    }

    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let sdk_slangc = PathBuf::from(sdk).join("Bin/slangc.exe");
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
