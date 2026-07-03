//! Backend bring-up smoke tests extracted from `main.rs`: the `--clear-test`,
//! `--triangle-test`, and `--mesh-test` standalone loops (Metal M0/M2/M3
//! milestones) plus their flag predicates. `run()` early-returns into these before
//! building the full deferred renderer; they own their own window/swapchain loops.

use std::time::Instant;

use anyhow::anyhow;
use dreamcoast_asset::MeshData;
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use rhi::{
    BackendKind, BlendMode, BufferDesc, BufferUsage, ClearColor, ComputePipelineDesc, DepthCompare,
    Device, Extent2D, Format, GraphicsPipelineDesc, MeshPipelineDesc, PrimitiveTopology,
    StorageBufferDesc, Swapchain, Texture, VertexLayout,
};
use tracing::info;

use crate::app::{save_screenshot, screenshot_captures};
use crate::mesh::{make_checker_texture, upload_mesh, upload_texture};
use crate::{COLOR_FORMAT, DEPTH_FORMAT, FRAMES_IN_FLIGHT, swapchain_desc};

/// Whether `--clear-test` was passed: run the minimal clear-screen loop (M0 of
/// the Metal backend bring-up) instead of the full deferred renderer. Exercises
/// only window + swapchain + command-buffer clear, which is all the Metal backend
/// implements until the triangle/pipeline milestones land.
pub(crate) fn clear_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--clear-test")
}

/// Minimal render loop: acquire → clear to an animated color → present. Used to
/// validate a backend's window/swapchain/command path before pipelines exist.
pub(crate) fn run_clear_test(
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
) -> anyhow::Result<()> {
    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;

    // Optional `--frames N`: render N frames then exit (non-interactive smoke
    // test). Without it, run until ESC / window close.
    let max_frames = clear_test_max_frames();
    info!("running clear-test loop (press ESC or close the window to exit)");
    let mut t = 0.0f32;
    let mut frames = 0u64;
    let _ = window.take_resized();
    while !window.should_close() {
        if let Some(max) = max_frames
            && frames >= max
        {
            break;
        }
        window.pump_events();
        let (w, h) = window.size();
        if w == 0 || h == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if window.take_resized() {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
        }

        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available)? {
            Some(i) => i,
            None => {
                swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
                continue;
            }
        };
        fence.reset()?;

        let color = ClearColor {
            r: 0.15,
            g: 0.5 + 0.35 * t.sin(),
            b: 0.35,
            a: 1.0,
        };
        cmd.begin()?;
        cmd.transition_to_render_target(swapchain, image_index);
        cmd.begin_rendering(swapchain, image_index, Some(color), None);
        cmd.end_rendering();
        cmd.transition_to_present(swapchain, image_index);
        cmd.end()?;
        queue.submit(&cmd, &image_available, &render_finished, &fence)?;
        queue.present(swapchain, image_index, &render_finished)?;
        t += 0.02;
        frames += 1;
    }
    device.wait_idle()?;
    info!("clear-test exited after {frames} frame(s)");
    Ok(())
}

/// Whether `--triangle-test` was passed: run the clear loop plus a single
/// hardcoded-triangle pipeline (M2 of the Metal backend bring-up). Exercises
/// pipeline creation + draw without vertex buffers, push constants, or bindless.
pub(crate) fn triangle_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--triangle-test")
}

/// Minimal render loop that draws the RGB triangle on a clear background. Like
/// [`run_clear_test`] but builds a pipeline from `triangle.slang` and issues a
/// 3-vertex draw each frame. Cross-backend (selects spirv/dxil/metallib).
pub(crate) fn run_triangle_test(
    backend: BackendKind,
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
) -> anyhow::Result<()> {
    let (vs, fs) = match backend {
        BackendKind::Vulkan => (
            dreamcoast_shader::triangle_vs_spirv(),
            dreamcoast_shader::triangle_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            dreamcoast_shader::triangle_vs_dxil(),
            dreamcoast_shader::triangle_fs_dxil(),
        ),
        BackendKind::Metal => (
            dreamcoast_shader::triangle_vs_metallib(),
            dreamcoast_shader::triangle_fs_metallib(),
        ),
    };
    let vs = vs.ok_or_else(|| anyhow!("triangle vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("triangle fragment shader unavailable for {backend:?}"))?;

    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: vs,
        fragment_bytes: fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[COLOR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 0,
        bindless: false,
        uniform_buffer: false,
        depth_test: false,
        depth_write: false,
        depth_compare: DepthCompare::Less,
        depth_format: None,
    })?;

    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;

    // `--screenshot[-clean] <path>`: capture the rendered frame to a PNG after a
    // short warmup, then exit. Lets the triangle path self-verify headlessly.
    let captures = screenshot_captures();
    const CAPTURE_FRAME: u64 = 2;
    let max_frames = clear_test_max_frames();
    info!("running triangle-test loop (press ESC or close the window to exit)");
    let mut frames = 0u64;
    let _ = window.take_resized();
    while !window.should_close() {
        if let Some(max) = max_frames
            && frames >= max
        {
            break;
        }
        window.pump_events();
        let (w, h) = window.size();
        if w == 0 || h == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if window.take_resized() {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
        }

        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available)? {
            Some(i) => i,
            None => {
                swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
                continue;
            }
        };
        fence.reset()?;

        let color = ClearColor {
            r: 0.1,
            g: 0.1,
            b: 0.12,
            a: 1.0,
        };
        cmd.begin()?;
        cmd.transition_to_render_target(swapchain, image_index);
        cmd.begin_rendering(swapchain, image_index, Some(color), None);
        cmd.set_viewport_scissor(swapchain);
        cmd.bind_graphics_pipeline(&pipeline);
        cmd.draw(3, 1);
        cmd.end_rendering();

        // On the capture frame, copy the backbuffer into a readback buffer in this
        // same command buffer (before it ends).
        let capture = (!captures.is_empty() && frames == CAPTURE_FRAME).then(|| &captures[0]);
        let readback = if capture.is_some() {
            let layout = device.swapchain_readback_layout(swapchain);
            let buf = device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?;
            cmd.copy_swapchain_to_buffer(swapchain, image_index, &buf);
            Some((buf, layout))
        } else {
            None
        };

        cmd.transition_to_present(swapchain, image_index);
        cmd.end()?;
        queue.submit(&cmd, &image_available, &render_finished, &fence)?;

        if let (Some(cap), Some((buf, layout))) = (capture, readback.as_ref()) {
            fence.wait()?;
            let mut bytes = vec![0u8; layout.size as usize];
            buf.read_into(&mut bytes)?;
            save_screenshot(&cap.path, &bytes, layout)?;
            info!(
                "saved triangle screenshot {} ({}x{})",
                cap.path, layout.width, layout.height
            );
        }

        queue.present(swapchain, image_index, &render_finished)?;
        frames += 1;

        if capture.is_some() {
            break;
        }
    }
    device.wait_idle()?;
    info!("triangle-test exited after {frames} frame(s)");
    Ok(())
}

/// Whether `--mesh-test` was passed: run the textured bindless mesh + ImGui loop
/// (M3 of the Metal backend bring-up). Exercises the bindless argument buffer,
/// sampled textures, depth testing, and the ImGui overlay.
pub(crate) fn mesh_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--mesh-test")
}

/// Render loop drawing the loaded glTF model as a depth-tested, diffuse-lit mesh
/// with its bindless base-color texture, plus a Dear ImGui overlay. The minimal
/// M3 counterpart to the full deferred renderer — it touches every M3 feature
/// (bindless table, textures, depth, ImGui) and nothing past it. Cross-backend.
pub(crate) fn run_mesh_test(
    backend: BackendKind,
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
    model: &MeshData,
    model_radius: f32,
) -> anyhow::Result<()> {
    let (vs, fs) = match backend {
        BackendKind::Vulkan => (
            dreamcoast_shader::mesh_vs_spirv(),
            dreamcoast_shader::mesh_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            dreamcoast_shader::mesh_vs_dxil(),
            dreamcoast_shader::mesh_fs_dxil(),
        ),
        BackendKind::Metal => (
            dreamcoast_shader::mesh_vs_metallib(),
            dreamcoast_shader::mesh_fs_metallib(),
        ),
    };
    let vs = vs.ok_or_else(|| anyhow!("mesh vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("mesh fragment shader unavailable for {backend:?}"))?;

    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: vs,
        fragment_bytes: fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[COLOR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::Mesh,
        blend: BlendMode::Opaque,
        push_constant_size: 80, // mat4 mvp (64) + tex_index (4), padded to 16
        bindless: true,
        uniform_buffer: false,
        depth_test: true,
        depth_write: true,
        depth_compare: DepthCompare::Less,
        depth_format: Some(DEPTH_FORMAT),
    })?;

    let (vbuf, ibuf, index_count) = upload_mesh(device, model)?;

    // Base-color texture (its bindless index goes in the push constant), or a
    // procedural checker when the model has none.
    let mut textures: Vec<Texture> = Vec::new();
    let tex_index = match &model.material.base_color {
        Some(im) => upload_texture(device, &mut textures, im, Format::Rgba8Srgb)?,
        None => {
            let t = make_checker_texture(device)?;
            let i = t.bindless_index();
            textures.push(t);
            i
        }
    };

    let mut gui = Gui::new(device, swapchain.format(), FRAMES_IN_FLIGHT)?;

    let (mut w, mut h) = window.size();
    let mut depth = device.create_depth_buffer(Extent2D::new(w.max(1), h.max(1)))?;

    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;

    let captures = screenshot_captures();
    const CAPTURE_FRAME: u64 = 3;
    let max_frames = clear_test_max_frames();
    info!("running mesh-test loop (press ESC or close the window to exit)");
    let mut frames = 0u64;
    let mut frame = 0usize;
    let mut last = Instant::now();
    let mut angle = 0.6f32;
    let _ = window.take_resized();
    while !window.should_close() {
        if let Some(max) = max_frames
            && frames >= max
        {
            break;
        }
        window.pump_events();
        let (nw, nh) = window.size();
        (w, h) = (nw, nh);
        if w == 0 || h == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if window.take_resized() {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
            depth = device.create_depth_buffer(Extent2D::new(w, h))?;
        }

        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        angle += dt * 0.5;

        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available)? {
            Some(i) => i,
            None => {
                swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
                depth = device.create_depth_buffer(Extent2D::new(w, h))?;
                continue;
            }
        };
        fence.reset()?;

        // Orbiting camera framing the model (which sits normalized at the origin).
        let focus = Vec3::new(0.0, model_radius * 0.5, 0.0);
        let dist = model_radius * 3.0;
        let eye = focus + Vec3::new(angle.cos() * dist, model_radius * 0.8, angle.sin() * dist);
        let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.05, 100.0);
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0; // Vulkan clip-space Y points down
        }
        let mvp = (proj * view).to_cols_array();

        // Push constant: mat4 mvp (64 bytes) + tex_index (4), zero-padded to 80.
        let mut pc = [0u8; 80];
        for (i, v) in mvp.iter().enumerate() {
            pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        pc[64..68].copy_from_slice(&tex_index.to_le_bytes());

        // Build the ImGui overlay for this frame.
        let ui = gui.new_frame(dt, [w as f32, h as f32], window.input());
        ui.window("DreamCoast — Metal M3")
            .size([300.0, 110.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text(format!("backend: {backend:?}"));
                ui.text(format!("mesh: {index_count} indices"));
                ui.text(format!("base-color bindless slot: {tex_index}"));
                ui.text("bindless argument buffer + ImGui");
            });

        let color = ClearColor {
            r: 0.08,
            g: 0.09,
            b: 0.12,
            a: 1.0,
        };
        cmd.begin()?;
        cmd.transition_to_render_target(swapchain, image_index);
        // Geometry pass: clear color + depth, draw the depth-tested mesh.
        cmd.begin_rendering(swapchain, image_index, Some(color), Some(&depth));
        cmd.set_viewport_scissor(swapchain);
        cmd.bind_graphics_pipeline(&pipeline);
        cmd.push_constants(&pc);
        cmd.bind_vertex_buffer(&vbuf, 32);
        cmd.bind_index_buffer(&ibuf, true);
        cmd.draw_indexed(index_count, 0, 0);
        cmd.end_rendering();
        // UI pass: load the color target (no depth attachment — the ImGui pipeline
        // is depth-less, so it must not run in the depth pass above).
        cmd.begin_rendering(swapchain, image_index, None, None);
        cmd.set_viewport_scissor(swapchain);
        gui.render(device, &cmd, frame)?;
        cmd.end_rendering();

        let capture = (!captures.is_empty() && frames == CAPTURE_FRAME).then(|| &captures[0]);
        let readback = if capture.is_some() {
            let layout = device.swapchain_readback_layout(swapchain);
            let buf = device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?;
            cmd.copy_swapchain_to_buffer(swapchain, image_index, &buf);
            Some((buf, layout))
        } else {
            None
        };

        cmd.transition_to_present(swapchain, image_index);
        cmd.end()?;
        queue.submit(&cmd, &image_available, &render_finished, &fence)?;

        if let (Some(cap), Some((buf, layout))) = (capture, readback.as_ref()) {
            fence.wait()?;
            let mut bytes = vec![0u8; layout.size as usize];
            buf.read_into(&mut bytes)?;
            save_screenshot(&cap.path, &bytes, layout)?;
            info!(
                "saved mesh screenshot {} ({}x{})",
                cap.path, layout.width, layout.height
            );
        }

        queue.present(swapchain, image_index, &render_finished)?;
        frames += 1;
        frame = (frame + 1) % FRAMES_IN_FLIGHT;

        if capture.is_some() {
            break;
        }
    }
    device.wait_idle()?;
    let _ = &depth; // kept alive for the loop
    info!("mesh-test exited after {frames} frame(s)");
    Ok(())
}

// ── Phase 14 (virtual geometry) M0 capability smokes ────────────────────────────────

/// Whether `--atomic64-test` was passed: run the 64-bit `atomicMax` capability smoke
/// (Phase 14 M0). Proves the visibility-buffer primitive end-to-end — a compute kernel
/// atomic-maxes packed `u64` values into a bindless storage buffer via an INDIRECT dispatch,
/// then the CPU reads it back and checks each slot equals the max the shader should have
/// written. Exercises the 64-bit-atomic + indirect-dispatch RHI paths at once.
pub(crate) fn atomic64_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--atomic64-test")
}

/// Knuth multiplicative hash, matching `vgeo_atomic.slang`'s `hi` scatter exactly (u32
/// wrapping). Kept in lockstep with the shader so the CPU expectation is the single source
/// of truth for what the GPU must produce.
fn atomic64_packed(i: u32) -> u64 {
    let hi = i.wrapping_mul(2654435761) & 0xFFFF;
    ((hi as u64) << 32) | (i as u64)
}

pub(crate) fn run_atomic64_test(backend: BackendKind, device: &Device) -> anyhow::Result<()> {
    let caps = device.capabilities();
    if !caps.atomic_int64 {
        return Err(anyhow!(
            "--atomic64-test: {backend:?} adapter lacks 64-bit buffer atomics (DeviceCapabilities::atomic_int64 = false)"
        ));
    }
    if !caps.dispatch_indirect {
        return Err(anyhow!(
            "--atomic64-test: {backend:?} adapter lacks indirect compute dispatch"
        ));
    }

    let cs = match backend {
        BackendKind::Vulkan => dreamcoast_shader::vgeo_atomic_cs_spirv(),
        BackendKind::D3d12 => dreamcoast_shader::vgeo_atomic_cs_dxil(),
        BackendKind::Metal => dreamcoast_shader::vgeo_atomic_cs_metallib(),
    }
    .ok_or_else(|| anyhow!("vgeo_atomic compute shader unavailable for {backend:?}"))?;

    const COUNT: u32 = 4096;
    const SLOTS: u32 = 16;

    let pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_bytes: cs,
        compute_entry: "csAtomicMax",
        push_constant_size: 16,
        bindless: true,
        uniform_buffer: false,
        threads_per_group: [64, 1, 1],
    })?;

    // R64 target: zero-initialised (so every packed write wins the `atomicMax`), host-visible
    // for CPU readback, registered in the bindless storage table.
    let vis = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: (SLOTS as u64) * 8,
            stride: 8,
            indirect: false,
        },
        &vec![0u8; (SLOTS as usize) * 8],
    )?;
    // Indirect dispatch args: threadgroup counts (x, y, z) read on the GPU.
    let groups_x = COUNT.div_ceil(64);
    let mut args_bytes = Vec::with_capacity(12);
    args_bytes.extend_from_slice(&groups_x.to_le_bytes());
    args_bytes.extend_from_slice(&1u32.to_le_bytes());
    args_bytes.extend_from_slice(&1u32.to_le_bytes());
    let args = device.create_storage_buffer_init(
        &StorageBufferDesc {
            size: 12,
            stride: 4,
            indirect: true,
        },
        &args_bytes,
    )?;

    let mut push = Vec::with_capacity(16);
    push.extend_from_slice(&vis.storage_index().to_le_bytes()); // buf_index
    push.extend_from_slice(&COUNT.to_le_bytes()); // count
    push.extend_from_slice(&SLOTS.to_le_bytes()); // slots
    push.extend_from_slice(&0u32.to_le_bytes()); // pad

    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(false)?;

    cmd.begin()?;
    cmd.bind_compute_pipeline(&pipeline);
    cmd.push_constants_compute(&push);
    cmd.dispatch_indirect(&args, 0);
    cmd.end()?;
    queue.submit(&cmd, &image_available, &render_finished, &fence)?;
    fence.wait()?;

    let mut bytes = vec![0u8; (SLOTS as usize) * 8];
    vis.read_into(&mut bytes)?;

    // CPU expectation: the max packed value routed to each slot (write wraps mod SLOTS).
    let mut expected = vec![0u64; SLOTS as usize];
    for i in 0..COUNT {
        let s = (i % SLOTS) as usize;
        expected[s] = expected[s].max(atomic64_packed(i));
    }

    let mut mismatches = 0usize;
    for (s, exp) in expected.iter().enumerate() {
        let got = u64::from_le_bytes(bytes[s * 8..s * 8 + 8].try_into().unwrap());
        if got != *exp {
            mismatches += 1;
            info!("atomic64 slot {s}: got {got:#018x} expected {exp:#018x}");
        }
    }
    if mismatches != 0 {
        return Err(anyhow!(
            "--atomic64-test FAILED: {mismatches}/{SLOTS} slots mismatched (64-bit atomicMax incorrect)"
        ));
    }

    device.wait_idle()?;
    info!(
        "--atomic64-test PASSED on {backend:?}: {COUNT} threads → {SLOTS} slots, 64-bit atomicMax via indirect dispatch, CPU-verified"
    );
    Ok(())
}

/// Whether `--mesh-shader-test` was passed: run the mesh-shader pipeline capability smoke
/// (Phase 14 M0). A single mesh threadgroup emits one hardcoded RGB triangle, proving
/// `create_mesh_pipeline` + `draw_mesh_tasks` before any cluster data exists.
pub(crate) fn mesh_shader_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--mesh-shader-test")
}

pub(crate) fn run_mesh_shader_test(
    backend: BackendKind,
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
) -> anyhow::Result<()> {
    if !device.capabilities().mesh_shader {
        return Err(anyhow!(
            "--mesh-shader-test: {backend:?} adapter lacks mesh shaders (DeviceCapabilities::mesh_shader = false)"
        ));
    }

    let (ms, fs) = match backend {
        BackendKind::Vulkan => (
            dreamcoast_shader::vgeo_meshlet_ms_spirv(),
            dreamcoast_shader::vgeo_meshlet_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            dreamcoast_shader::vgeo_meshlet_ms_dxil(),
            dreamcoast_shader::vgeo_meshlet_fs_dxil(),
        ),
        BackendKind::Metal => (
            dreamcoast_shader::vgeo_meshlet_ms_metallib(),
            dreamcoast_shader::vgeo_meshlet_fs_metallib(),
        ),
    };
    let ms = ms.ok_or_else(|| anyhow!("vgeo_meshlet mesh shader unavailable for {backend:?}"))?;
    let fs =
        fs.ok_or_else(|| anyhow!("vgeo_meshlet fragment shader unavailable for {backend:?}"))?;

    let pipeline = device.create_mesh_pipeline(&MeshPipelineDesc {
        object_bytes: None,
        object_entry: "",
        mesh_bytes: ms,
        mesh_entry: "meshMain",
        fragment_bytes: fs,
        fragment_entry: "fragMain",
        color_formats: &[COLOR_FORMAT],
        depth_format: None,
        push_constant_size: 0,
        bindless: false,
        uniform_buffer: false,
        object_threads: [1, 1, 1],
        mesh_threads: [3, 1, 1],
    })?;

    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;

    let captures = screenshot_captures();
    const CAPTURE_FRAME: u64 = 2;
    let max_frames = clear_test_max_frames();
    info!("running mesh-shader-test loop (press ESC or close the window to exit)");
    let mut frames = 0u64;
    let _ = window.take_resized();
    while !window.should_close() {
        if let Some(max) = max_frames
            && frames >= max
        {
            break;
        }
        window.pump_events();
        let (w, h) = window.size();
        if w == 0 || h == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if window.take_resized() {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
        }

        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available)? {
            Some(i) => i,
            None => {
                swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
                continue;
            }
        };
        fence.reset()?;

        let color = ClearColor {
            r: 0.1,
            g: 0.1,
            b: 0.12,
            a: 1.0,
        };
        cmd.begin()?;
        cmd.transition_to_render_target(swapchain, image_index);
        cmd.begin_rendering(swapchain, image_index, Some(color), None);
        cmd.set_viewport_scissor(swapchain);
        cmd.bind_mesh_pipeline(&pipeline);
        cmd.draw_mesh_tasks(1, 1, 1);
        cmd.end_rendering();

        let capture = (!captures.is_empty() && frames == CAPTURE_FRAME).then(|| &captures[0]);
        let readback = if capture.is_some() {
            let layout = device.swapchain_readback_layout(swapchain);
            let buf = device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?;
            cmd.copy_swapchain_to_buffer(swapchain, image_index, &buf);
            Some((buf, layout))
        } else {
            None
        };

        cmd.transition_to_present(swapchain, image_index);
        cmd.end()?;
        queue.submit(&cmd, &image_available, &render_finished, &fence)?;

        if let (Some(cap), Some((buf, layout))) = (capture, readback.as_ref()) {
            fence.wait()?;
            let mut bytes = vec![0u8; layout.size as usize];
            buf.read_into(&mut bytes)?;
            save_screenshot(&cap.path, &bytes, layout)?;
            info!(
                "saved mesh-shader screenshot {} ({}x{})",
                cap.path, layout.width, layout.height
            );
        }

        queue.present(swapchain, image_index, &render_finished)?;
        frames += 1;

        if capture.is_some() {
            break;
        }
    }
    device.wait_idle()?;
    info!("mesh-shader-test exited after {frames} frame(s)");
    Ok(())
}

/// `--frames N` cap for the clear-test loop (smoke testing); `None` = unlimited.
pub(crate) fn clear_test_max_frames() -> Option<u64> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--frames" {
            return args.next().and_then(|v| v.parse().ok());
        }
    }
    None
}
