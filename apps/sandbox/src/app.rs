//! Process / platform plumbing extracted from `main.rs`: shader-bytecode selection
//! per backend, screenshot capture + PNG save, and CLI-flag parsing (model path,
//! log file, backend, validation, screenshots). No render-loop state.

use std::path::{Path, PathBuf};

use anyhow::anyhow;
use rhi::{BackendKind, Device, ReadbackLayout, Semaphore};

use crate::MODEL_PATH;

/// Fetch the (vertex, fragment) bytecode for `backend` from a shader's six
/// generated accessors, erroring if unavailable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_shader_pair(
    backend: BackendKind,
    vs_spirv: fn() -> Option<&'static [u8]>,
    fs_spirv: fn() -> Option<&'static [u8]>,
    vs_dxil: fn() -> Option<&'static [u8]>,
    fs_dxil: fn() -> Option<&'static [u8]>,
    vs_metallib: fn() -> Option<&'static [u8]>,
    fs_metallib: fn() -> Option<&'static [u8]>,
    name: &str,
) -> anyhow::Result<(&'static [u8], &'static [u8])> {
    let (vs, fs) = match backend {
        BackendKind::Vulkan => (vs_spirv(), fs_spirv()),
        BackendKind::D3d12 => (vs_dxil(), fs_dxil()),
        BackendKind::Metal => (vs_metallib(), fs_metallib()),
    };
    let vs = vs.ok_or_else(|| anyhow!("{name} vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("{name} fragment shader unavailable for {backend:?}"))?;
    Ok((vs, fs))
}

/// Fetch single-stage (compute) bytecode for `backend`, erroring if unavailable.
pub(crate) fn load_compute_shader(
    backend: BackendKind,
    cs_spirv: fn() -> Option<&'static [u8]>,
    cs_dxil: fn() -> Option<&'static [u8]>,
    cs_metallib: fn() -> Option<&'static [u8]>,
    name: &str,
) -> anyhow::Result<&'static [u8]> {
    let cs = match backend {
        BackendKind::Vulkan => cs_spirv(),
        BackendKind::D3d12 => cs_dxil(),
        BackendKind::Metal => cs_metallib(),
    };
    cs.ok_or_else(|| anyhow!("{name} compute shader unavailable for {backend:?}"))
}

pub(crate) fn build_render_finished(device: &Device, count: u32) -> anyhow::Result<Vec<Semaphore>> {
    (0..count)
        .map(|_| device.create_semaphore().map_err(Into::into))
        .collect()
}

/// A requested screenshot: output path + whether to include the ImGui overlay.
#[derive(Clone)]
pub(crate) struct Capture {
    pub(crate) path: String,
    pub(crate) include_ui: bool,
}

/// Parse `--screenshot <path>` (with UI overlay) and `--screenshot-clean <path>`
/// (3D only) flags into capture requests, in argument order. Presence of any
/// puts the app in headless screenshot mode (render a few frames, capture, exit).
pub(crate) fn screenshot_captures() -> Vec<Capture> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let include_ui = match arg.as_str() {
            "--screenshot" => true,
            "--screenshot-clean" => false,
            _ => continue,
        };
        if let Some(path) = args.next() {
            out.push(Capture { path, include_ui });
        }
    }
    out
}

/// Auto-generated path for an interactive (F2) screenshot.
pub(crate) fn interactive_screenshot_path() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("screenshot_{secs}.png")
}

/// Save BGRA readback bytes (rows padded to `layout.row_pitch`) as a PNG. The
/// swapchain stores sRGB-encoded bytes, so they map straight to a PNG after the
/// B<->R channel swap; padding is dropped per row.
pub(crate) fn save_screenshot(
    path: &str,
    data: &[u8],
    layout: &ReadbackLayout,
) -> anyhow::Result<()> {
    let w = layout.width as usize;
    let h = layout.height as usize;
    let pitch = layout.row_pitch as usize;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let src = &data[y * pitch..y * pitch + w * 4];
        let dst = &mut rgba[y * w * 4..(y + 1) * w * 4];
        for x in 0..w {
            dst[x * 4] = src[x * 4 + 2]; // R <- B
            dst[x * 4 + 1] = src[x * 4 + 1]; // G
            dst[x * 4 + 2] = src[x * 4]; // B <- R
            dst[x * 4 + 3] = src[x * 4 + 3]; // A
        }
    }
    let img = image::RgbaImage::from_raw(layout.width, layout.height, rgba)
        .ok_or_else(|| anyhow!("screenshot buffer size mismatch"))?;
    img.save(path)?;
    Ok(())
}

/// Model path: `--model <path>` or the default `assets/model.glb`.
pub(crate) fn model_path() -> String {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model"
            && let Some(p) = args.next()
        {
            return p;
        }
    }
    MODEL_PATH.to_string()
}

/// Resolve an asset path so the app finds its data regardless of the current
/// working directory (launched from an IDE, a different cwd, or a shipped install).
///
/// Default asset paths like `assets/model.glb` are relative, and resolving them
/// against the cwd breaks the moment the process is started from anywhere but the
/// repo root — the model silently falls back to the procedural cube. Instead, an
/// absolute path is used as-is; a relative path is tried against, in order:
///   1. the current working directory (preserves running from the repo root),
///   2. the executable's directory and each ancestor — covers a shipped layout
///      (`assets/` beside the binary) and `cargo run` from any cwd, where the exe
///      sits at `<repo>/target/<profile>/` so an ancestor is the repo root,
///   3. the crate's compile-time workspace root (a dev-build belt-and-suspenders).
///
/// The first existing candidate wins; if none exist the original path is returned
/// unchanged so the caller's missing-asset handling and error text stay meaningful.
pub(crate) fn resolve_asset_path(path: &str) -> PathBuf {
    let rel = Path::new(path);
    if rel.is_absolute() {
        return rel.to_path_buf();
    }

    let mut bases: Vec<PathBuf> = vec![PathBuf::from(".")];
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        bases.extend(exe_dir.ancestors().map(Path::to_path_buf));
    }
    // `CARGO_MANIFEST_DIR` is `<root>/apps/sandbox`; the workspace root is two up.
    if let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2) {
        bases.push(root.to_path_buf());
    }

    bases
        .iter()
        .map(|base| base.join(rel))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| rel.to_path_buf())
}

/// `--log-file <path>`: mirror logs to a file (see `main`). `None` when absent.
pub(crate) fn log_file_path() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--log-file" {
            return args.next();
        }
    }
    None
}

pub(crate) fn select_backend() -> BackendKind {
    let mut backend = if cfg!(windows) {
        BackendKind::D3d12
    } else if cfg!(target_os = "macos") {
        BackendKind::Metal
    } else {
        BackendKind::Vulkan
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--backend" {
            match args.next().as_deref() {
                Some("vulkan") => backend = BackendKind::Vulkan,
                Some("d3d12") => backend = BackendKind::D3d12,
                Some("metal") => backend = BackendKind::Metal,
                other => tracing::warn!("unknown --backend value {other:?}; using default"),
            }
        }
    }
    backend
}

/// Whether the Vulkan validation layer / debug-utils messenger should be enabled
/// (Phase 9 M3). Defaults on; `--no-validation` turns it off (useful when running
/// under a capture tool that injects its own layers, or to measure layer-free
/// timings). Validation is instance-level, so this is a launch flag, not a live
/// toggle; in release builds it is always compiled out regardless.
pub(crate) fn validation_enabled() -> bool {
    !std::env::args().skip(1).any(|a| a == "--no-validation")
}
