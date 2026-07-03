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
    StorageBuffer, StorageBufferDesc, Swapchain, Texture, VertexLayout,
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

    // R64 target: host-visible so the GPU can UAV-atomic into it AND the CPU can read it back
    // (CUSTOM-L0 on D3D12 / HOST_COHERENT on Vulkan / Shared on Metal). A DEFAULT-heap `_init`
    // buffer is NOT host-readable on D3D12 — only Metal's unified memory tolerated that, so this
    // is the DX≡VK fix for the smoke. Zero-initialised (via `write`) so every packed value wins
    // the `atomicMax`; registered in the bindless storage table.
    let vis = device.create_storage_buffer_host(&StorageBufferDesc {
        size: (SLOTS as u64) * 8,
        stride: 8,
        indirect: false,
    })?;
    vis.write(&vec![0u8; (SLOTS as usize) * 8])?;
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
    let fence = device.create_fence(false)?;

    cmd.begin()?;
    // D3D12 `ExecuteIndirect` reads the argument buffer in INDIRECT_ARGUMENT state; transition it
    // (no-op on Vulkan/Metal). Without this the D3D12 debug layer flags a resource-state error.
    cmd.storage_buffer_to_indirect(&args);
    cmd.bind_compute_pipeline(&pipeline);
    cmd.push_constants_compute(&push);
    cmd.dispatch_indirect(&args, 0);
    cmd.end()?;
    // Headless one-shot: no swapchain semaphores (waiting on an unsignaled `image_available`
    // would trip a Vulkan `VUID-vkQueueSubmit-pWaitSemaphores` validation error).
    queue.submit_oneshot(&cmd, &fence)?;
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
        depth_test: false,
        depth_write: false,
        depth_compare: DepthCompare::Less,
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

/// Whether `--vgeo-test` was passed: build the virtual-geometry LOD DAG for the loaded model,
/// log the LOD pyramid, and render a chosen level (`VGEO_LOD=n`, default 0) so LOD transitions
/// can be inspected for cracks (the offline builder is topologically crack-free-tested; this is
/// the visual counterpart). Phase 14 M1e.
pub(crate) fn vgeo_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--vgeo-test")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_vgeo_test(
    backend: BackendKind,
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
    source: &std::path::Path,
    cache_key: &str,
    cache_dir: &std::path::Path,
    tex: dreamcoast_asset::cook::TexCompress,
) -> anyhow::Result<()> {
    use dreamcoast_asset::vgeo;

    // Consume the COOKED cluster pages baked into the `.dcasset` (cooking on a miss), exercising
    // the real cooked pipeline end to end rather than rebuilding the DAG here.
    let mut dag = dreamcoast_asset::cook::load_cooked_clusters(source, cache_key, cache_dir, tex)?;
    let levels = vgeo::lod_levels(&dag);
    info!(
        "cooked vgeo LOD DAG: {} clusters across {} LOD level(s) ({} verts)",
        dag.clusters.len(),
        levels.len(),
        dag.vertices.len(),
    );
    for &l in &levels {
        let (mut cn, mut tn) = (0u32, 0u32);
        let (mut emin, mut emax) = (f32::MAX, 0.0f32);
        for c in dag.clusters.iter().filter(|c| c.lod_level == l) {
            cn += 1;
            tn += c.triangle_count;
            emin = emin.min(c.self_error);
            emax = emax.max(c.self_error);
        }
        info!("  LOD {l}: {cn} clusters, {tn} tris, self_error {emin:.4}..{emax:.4}");
    }

    // The cooked geometry is in raw (pre-normalization) coordinates; recenter it on the origin so
    // the mesh-test camera (which frames a radius around the origin) sees it, and derive the radius
    // from its own bounds.
    let mut center = Vec3::ZERO;
    for v in &dag.vertices {
        center += Vec3::from(v.pos);
    }
    center /= dag.vertices.len().max(1) as f32;
    let mut radius = 0.0f32;
    for v in &mut dag.vertices {
        let p = Vec3::from(v.pos) - center;
        v.pos = p.to_array();
        radius = radius.max(p.length());
    }

    // Pick the LOD to render (clamped to the coarsest available).
    let want: u32 = std::env::var("VGEO_LOD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let lod = want.min(*levels.last().unwrap_or(&0));
    // Colour mode: per-cluster (the meshlet debug view, default) or per-LOD level.
    let by_lod = matches!(std::env::var("VGEO_COLOR").ok().as_deref(), Some("lod"));

    // Build the meshlet debug mesh: a distinct flat colour per cluster (or per LOD), shaded by
    // the real vertex normal so the 3D form still reads. Each cluster's triangles are emitted with
    // a UV that samples that cluster's slot in a hue palette (so we reuse the existing textured
    // mesh pipeline — no debug shader needed). Non-indexed so adjacent clusters never share a hue.
    let debug_mesh = meshlet_debug_mesh(&dag, lod, by_lod);
    info!(
        "meshlet debug: cooked LOD {lod}, {} clusters, {} triangles, colour by {}",
        dag.clusters.iter().filter(|c| c.lod_level == lod).count(),
        debug_mesh.indices.len() / 3,
        if by_lod { "LOD" } else { "cluster" },
    );

    run_mesh_test(
        backend,
        window,
        device,
        swapchain,
        &debug_mesh,
        radius.max(1e-3),
    )
}

/// Whether `--vgeo-mesh` was passed: render the cooked clusters on the GPU via a mesh shader
/// (Phase 14 M2). One mesh threadgroup per cluster reads its geometry from bindless storage
/// buffers. `VGEO_LOD=n` picks the LOD; `VGEO_MATERIAL=1` swaps per-cluster colour for a texture.
pub(crate) fn vgeo_mesh_test_enabled() -> bool {
    std::env::args().skip(1).any(|a| a == "--vgeo-mesh")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_vgeo_mesh(
    backend: BackendKind,
    window: &mut Window,
    device: &Device,
    swapchain: &mut Swapchain,
    source: &std::path::Path,
    cache_key: &str,
    cache_dir: &std::path::Path,
    tex: dreamcoast_asset::cook::TexCompress,
) -> anyhow::Result<()> {
    use dreamcoast_asset::vgeo;

    if !device.capabilities().mesh_shader {
        return Err(anyhow!(
            "--vgeo-mesh: {backend:?} adapter lacks mesh shaders (DeviceCapabilities::mesh_shader = false)"
        ));
    }
    let (ms, fs) = match backend {
        BackendKind::Vulkan => (
            dreamcoast_shader::vgeo_cluster_ms_spirv(),
            dreamcoast_shader::vgeo_cluster_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            dreamcoast_shader::vgeo_cluster_ms_dxil(),
            dreamcoast_shader::vgeo_cluster_fs_dxil(),
        ),
        BackendKind::Metal => (
            dreamcoast_shader::vgeo_cluster_ms_metallib(),
            dreamcoast_shader::vgeo_cluster_fs_metallib(),
        ),
    };
    let ms = ms.ok_or_else(|| anyhow!("vgeo_cluster mesh shader unavailable for {backend:?}"))?;
    let fs =
        fs.ok_or_else(|| anyhow!("vgeo_cluster fragment shader unavailable for {backend:?}"))?;

    // Cooked clusters; recenter on the origin (raw coords) and derive the framing radius.
    let mut dag = dreamcoast_asset::cook::load_cooked_clusters(source, cache_key, cache_dir, tex)?;
    let mut center = Vec3::ZERO;
    for v in &dag.vertices {
        center += Vec3::from(v.pos);
    }
    center /= dag.vertices.len().max(1) as f32;
    let mut radius = 0.0f32;
    for v in &mut dag.vertices {
        let p = Vec3::from(v.pos) - center;
        v.pos = p.to_array();
        radius = radius.max(p.length());
    }
    let radius = radius.max(1e-3);
    let levels = vgeo::lod_levels(&dag);
    let want: u32 = std::env::var("VGEO_LOD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let lod = want.min(*levels.last().unwrap_or(&0));

    // Upload the geometry into bindless storage buffers. The mesh shader indexes these by the
    // slots we stash in the push constant.
    let sb = |device: &Device, bytes: &[u8], stride: u32| -> anyhow::Result<_> {
        Ok(device.create_storage_buffer_init(
            &StorageBufferDesc {
                size: bytes.len().max(1) as u64,
                stride,
                indirect: false,
            },
            bytes,
        )?)
    };
    let mut vtx = Vec::with_capacity(dag.vertices.len() * 32);
    for v in &dag.vertices {
        for f in v.pos.iter().chain(&v.normal) {
            vtx.extend_from_slice(&f.to_le_bytes());
        }
        for f in v.uv {
            vtx.extend_from_slice(&f.to_le_bytes());
        }
    }
    let mut remap = Vec::with_capacity(dag.cluster_vertices.len() * 4);
    for &i in &dag.cluster_vertices {
        remap.extend_from_slice(&i.to_le_bytes());
    }
    let mut tri = Vec::with_capacity(dag.cluster_triangles.len() * 4);
    for &b in &dag.cluster_triangles {
        tri.extend_from_slice(&(b as u32).to_le_bytes());
    }
    // Per-cluster GpuCluster records (96 B), ALL clusters — the mesh shader reads the first 16
    // bytes (offsets/counts); the LOD-cut compute reads the self/parent error+sphere (LOD), the
    // own bounds sphere (frustum + small cull), and the normal cone (backface cull). Spheres are
    // recentered by the same centroid as the vertices so camera-space projection lines up.
    let mut rec = Vec::with_capacity(dag.clusters.len() * 96);
    for c in &dag.clusters {
        for field in [
            c.vertex_offset,
            c.vertex_count,
            c.triangle_offset,
            c.triangle_count,
        ] {
            rec.extend_from_slice(&field.to_le_bytes());
        }
        let put = |rec: &mut Vec<u8>, f: f32| rec.extend_from_slice(&f.to_le_bytes());
        let put3 = |rec: &mut Vec<u8>, v: [f32; 3]| {
            for f in v {
                rec.extend_from_slice(&f.to_le_bytes());
            }
        };
        put(&mut rec, c.self_error); // 16
        put3(&mut rec, (Vec3::from(c.self_center) - center).to_array());
        put(&mut rec, c.self_radius);
        put(&mut rec, c.parent_error); // 36
        put3(&mut rec, (Vec3::from(c.parent_center) - center).to_array());
        put(&mut rec, c.parent_radius);
        put3(&mut rec, (Vec3::from(c.bounds_center) - center).to_array()); // 56
        put(&mut rec, c.bounds_radius);
        put3(&mut rec, c.cone_axis); // 72
        put(&mut rec, c.cone_cutoff);
        rec.extend_from_slice(&[0u8; 8]); // 88..96 pad
    }
    // Direct-path (M2) span of the selected LOD (clusters are contiguous per level).
    let lod_start = dag
        .clusters
        .iter()
        .position(|c| c.lod_level == lod)
        .unwrap_or(0) as u32;
    let lod_count = dag.clusters.iter().filter(|c| c.lod_level == lod).count() as u32;
    let total_clusters = dag.clusters.len() as u32;

    let vtx_buf = sb(device, &vtx, 32)?;
    let remap_buf = sb(device, &remap, 4)?;
    let tri_buf = sb(device, &tri, 4)?;
    let rec_buf = sb(device, &rec, 96)?;

    // M3 cut mode (VGEO_CUT=1): a compute pass selects the view-dependent cut into `vis_buf` and
    // writes the mesh-draw threadgroup count into `args_buf` for the indirect draw.
    // M5 software-raster mode (VGEO_SW=1): rasterize the cut's clusters into an R64 visibility
    // buffer via atomicMax, then visualize it. SW implies the cut (it needs the visible list).
    // M5b HW/SW binning (VGEO_BIN=1): the cut is split into a HW (mesh-shader) list and a SW
    // (compute-raster) list by projected screen size; both write the SAME R64 visibility buffer.
    // `VGEO_BINPX` is the split threshold (projected bounds diameter in px, below → SW).
    let bin = std::env::var("VGEO_BIN").ok().as_deref() == Some("1");
    // M6 (VGEO_RESOLVE=1): resolve the visibility buffer to shaded attributes (the deferred
    // G-buffer stage) instead of the raw cluster-id visualization. Needs a populated visbuf (sw/bin).
    let resolve = std::env::var("VGEO_RESOLVE").ok().as_deref() == Some("1");
    let sw = std::env::var("VGEO_SW").ok().as_deref() == Some("1");
    let cut = sw || bin || std::env::var("VGEO_CUT").ok().as_deref() == Some("1");
    let tau: f32 = std::env::var("VGEO_TAU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    let bin_px: f32 = std::env::var("VGEO_BINPX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);
    // `vis_buf`/`args_buf` are the cut's list/args (M3/M5a); in binning mode they carry the HW
    // sub-list and `sw_list`/`sw_args` carry the SW sub-list.
    let vis_buf = sb(device, &vec![0u8; total_clusters as usize * 4], 4)?;
    let args_buf = sb(device, &[0u8; 12], 4)?; // {countX, 1, 1}
    let sw_list = sb(device, &vec![0u8; total_clusters as usize * 4], 4)?;
    let sw_args = sb(device, &[0u8; 12], 4)?;
    let cut_pipeline = if cut {
        let cs = match (bin, backend) {
            (false, BackendKind::Vulkan) => dreamcoast_shader::vgeo_cut_cs_spirv(),
            (false, BackendKind::D3d12) => dreamcoast_shader::vgeo_cut_cs_dxil(),
            (false, BackendKind::Metal) => dreamcoast_shader::vgeo_cut_cs_metallib(),
            (true, BackendKind::Vulkan) => dreamcoast_shader::vgeo_cut_bin_cs_spirv(),
            (true, BackendKind::D3d12) => dreamcoast_shader::vgeo_cut_bin_cs_dxil(),
            (true, BackendKind::Metal) => dreamcoast_shader::vgeo_cut_bin_cs_metallib(),
        }
        .ok_or_else(|| anyhow!("vgeo_cut compute shader unavailable for {backend:?}"))?;
        Some(device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: cs,
            compute_entry: if bin { "csCutBin" } else { "csCut" },
            push_constant_size: 224,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [64, 1, 1],
        })?)
    } else {
        None
    };

    // A checker texture for material mode (the cooked cluster page carries no material).
    let checker = make_checker_texture(device)?;
    let tex_index = checker.bindless_index();
    let mode: u32 = if std::env::var("VGEO_MATERIAL").ok().as_deref() == Some("1") {
        1
    } else {
        0
    };
    if bin {
        info!(
            "vgeo-mesh BIN: {total_clusters} clusters, tau {tau}px, HW/SW split at {bin_px}px \
             diameter → HW mesh + SW raster into one R64 visibility buffer"
        );
    } else if cut {
        info!(
            "vgeo-mesh CUT: {total_clusters} clusters across all LODs, tau {tau}px, indirect draw"
        );
    } else {
        info!(
            "vgeo-mesh: LOD {lod}, {lod_count} clusters via mesh shader, mode {}",
            if mode == 1 {
                "material"
            } else {
                "cluster-colour"
            }
        );
    }

    let pipeline = device.create_mesh_pipeline(&MeshPipelineDesc {
        object_bytes: None,
        object_entry: "",
        mesh_bytes: ms,
        mesh_entry: "meshMain",
        fragment_bytes: fs,
        fragment_entry: "fragMain",
        color_formats: &[COLOR_FORMAT],
        depth_format: Some(DEPTH_FORMAT),
        push_constant_size: 112,
        bindless: true,
        uniform_buffer: false,
        object_threads: [1, 1, 1],
        mesh_threads: [128, 1, 1],
        depth_test: true,
        depth_write: true,
        depth_compare: DepthCompare::Less,
    })?;

    let (mut w, mut h) = window.size();
    let mut depth = device.create_depth_buffer(Extent2D::new(w.max(1), h.max(1)))?;

    // M5 software-raster resources: an R64 visibility buffer (one u64/pixel), the clear + raster
    // compute pipelines, and the full-screen visualization pipeline. `visbuf` is recreated on
    // resize (its size tracks the render extent).
    let vis_bytes = |w: u32, h: u32| (w.max(1) as u64) * (h.max(1) as u64) * 8;
    let mut visbuf = device.create_storage_buffer(&StorageBufferDesc {
        size: vis_bytes(w, h),
        stride: 8,
        indirect: false,
    })?;
    let (sw_clear_pipeline, sw_raster_pipeline, vis_pipeline) = if sw || bin {
        let cshader = |spirv: fn() -> Option<&'static [u8]>,
                       dxil: fn() -> Option<&'static [u8]>,
                       metal: fn() -> Option<&'static [u8]>,
                       entry: &str,
                       size: u32|
         -> anyhow::Result<_> {
            let bytes = match backend {
                BackendKind::Vulkan => spirv(),
                BackendKind::D3d12 => dxil(),
                BackendKind::Metal => metal(),
            }
            .ok_or_else(|| anyhow!("{entry} shader unavailable for {backend:?}"))?;
            Ok(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: bytes,
                compute_entry: entry,
                push_constant_size: size,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: if entry == "csClear" {
                    [64, 1, 1]
                } else {
                    [128, 1, 1]
                },
            })?)
        };
        let clear = cshader(
            dreamcoast_shader::vgeo_swraster_clear_cs_spirv,
            dreamcoast_shader::vgeo_swraster_clear_cs_dxil,
            dreamcoast_shader::vgeo_swraster_clear_cs_metallib,
            "csClear",
            96,
        )?;
        let raster = cshader(
            dreamcoast_shader::vgeo_swraster_cs_spirv,
            dreamcoast_shader::vgeo_swraster_cs_dxil,
            dreamcoast_shader::vgeo_swraster_cs_metallib,
            "csRaster",
            96,
        )?;
        let (vvs, vfs) = match backend {
            BackendKind::Vulkan => (
                dreamcoast_shader::vgeo_visbuffer_vs_spirv(),
                dreamcoast_shader::vgeo_visbuffer_fs_spirv(),
            ),
            BackendKind::D3d12 => (
                dreamcoast_shader::vgeo_visbuffer_vs_dxil(),
                dreamcoast_shader::vgeo_visbuffer_fs_dxil(),
            ),
            BackendKind::Metal => (
                dreamcoast_shader::vgeo_visbuffer_vs_metallib(),
                dreamcoast_shader::vgeo_visbuffer_fs_metallib(),
            ),
        };
        let vis = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vvs.ok_or_else(|| anyhow!("vgeo_visbuffer vs unavailable"))?,
            fragment_bytes: vfs.ok_or_else(|| anyhow!("vgeo_visbuffer fs unavailable"))?,
            vertex_entry: "vsMain",
            fragment_entry: "fragMain",
            color_formats: &[COLOR_FORMAT],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 16,
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;
        (Some(clear), Some(raster), Some(vis))
    } else {
        (None, None, None)
    };

    // M5b: the HW-path mesh pipeline that writes the shared visibility buffer (per-primitive triId
    // + fragment atomicMax). No depth attachment — occlusion is resolved by the atomicMax, exactly
    // like the SW rasterizer, so the HW/SW boundary agrees per pixel.
    let hwvis_pipeline = if bin {
        let (hms, hfs) = match backend {
            BackendKind::Vulkan => (
                dreamcoast_shader::vgeo_hwvis_ms_spirv(),
                dreamcoast_shader::vgeo_hwvis_fs_spirv(),
            ),
            BackendKind::D3d12 => (
                dreamcoast_shader::vgeo_hwvis_ms_dxil(),
                dreamcoast_shader::vgeo_hwvis_fs_dxil(),
            ),
            BackendKind::Metal => (
                dreamcoast_shader::vgeo_hwvis_ms_metallib(),
                dreamcoast_shader::vgeo_hwvis_fs_metallib(),
            ),
        };
        Some(device.create_mesh_pipeline(&MeshPipelineDesc {
            object_bytes: None,
            object_entry: "",
            mesh_bytes: hms.ok_or_else(|| anyhow!("vgeo_hwvis mesh shader unavailable"))?,
            mesh_entry: "meshMain",
            fragment_bytes: hfs.ok_or_else(|| anyhow!("vgeo_hwvis fragment shader unavailable"))?,
            fragment_entry: "fragMain",
            color_formats: &[COLOR_FORMAT],
            depth_format: None,
            push_constant_size: 96, // mat4 mvp (64) + 8 u32 slots (32)
            bindless: true,
            uniform_buffer: false,
            object_threads: [1, 1, 1],
            mesh_threads: [128, 1, 1],
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
        })?)
    } else {
        None
    };

    // M6: the material-resolve pipeline (full-screen). Reads the visibility buffer, reconstructs
    // analytic-barycentric attributes, and shades like the M2 direct render (the deferred G-buffer
    // stage in the self-contained viewer). Opt-in via VGEO_RESOLVE=1, needs a populated visbuf.
    let resolve_pipeline = if resolve && (sw || bin) {
        let (rvs, rfs) = match backend {
            BackendKind::Vulkan => (
                dreamcoast_shader::vgeo_resolve_vs_spirv(),
                dreamcoast_shader::vgeo_resolve_fs_spirv(),
            ),
            BackendKind::D3d12 => (
                dreamcoast_shader::vgeo_resolve_vs_dxil(),
                dreamcoast_shader::vgeo_resolve_fs_dxil(),
            ),
            BackendKind::Metal => (
                dreamcoast_shader::vgeo_resolve_vs_metallib(),
                dreamcoast_shader::vgeo_resolve_fs_metallib(),
            ),
        };
        Some(device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: rvs.ok_or_else(|| anyhow!("vgeo_resolve vs unavailable"))?,
            fragment_bytes: rfs.ok_or_else(|| anyhow!("vgeo_resolve fs unavailable"))?,
            vertex_entry: "vsMain",
            fragment_entry: "fragMain",
            color_formats: &[COLOR_FORMAT],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 112,
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?)
    } else {
        None
    };

    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;

    let captures = screenshot_captures();
    const CAPTURE_FRAME: u64 = 3;
    let max_frames = clear_test_max_frames();
    info!("running vgeo-mesh loop (press ESC or close the window to exit)");
    let mut frames = 0u64;
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
        (w, h) = window.size();
        if w == 0 || h == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if window.take_resized() {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(w, h)))?;
            depth = device.create_depth_buffer(Extent2D::new(w, h))?;
            if sw || bin {
                visbuf = device.create_storage_buffer(&StorageBufferDesc {
                    size: vis_bytes(w, h),
                    stride: 8,
                    indirect: false,
                })?;
            }
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

        // VGEO_PAN offsets the look-at target so the object (at the origin) slides toward the
        // screen edge — a way to see frustum culling drop the off-screen clusters.
        let pan: f32 = std::env::var("VGEO_PAN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let focus = Vec3::new(pan * radius, 0.0, 0.0);
        let dist = radius * 3.0;
        let eye = focus + Vec3::new(angle.cos() * dist, radius * 0.6, angle.sin() * dist);
        let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.05, 100.0);
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0;
        }
        let mvp = (proj * view).to_cols_array();

        // Mesh push constant: mat4 mvp (64) + 12 u32 slots/flags (48) = 112.
        let mut pc = [0u8; 112];
        for (i, v) in mvp.iter().enumerate() {
            pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let vis_slot = if cut {
            vis_buf.storage_index()
        } else {
            0xFFFF_FFFF
        };
        let words = [
            vtx_buf.storage_index(),
            remap_buf.storage_index(),
            tri_buf.storage_index(),
            rec_buf.storage_index(),
            lod_start, // direct-path base cluster
            mode,
            tex_index,
            vis_slot,
            0,
            0,
            0,
            0,
        ];
        for (i, word) in words.iter().enumerate() {
            pc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
        }

        cmd.begin()?;

        // M3: reset the indirect count and run the LOD-cut compute → visible list + args.
        // M5b (bin): also reset the SW sub-list args; `csCutBin` splits the cut into HW + SW.
        if cut {
            let reset = |b: &StorageBuffer| -> anyhow::Result<()> {
                b.write(
                    &[0u32, 1, 1]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                Ok(())
            };
            reset(&args_buf)?;
            if bin {
                reset(&sw_args)?;
            }
            let proj_factor = 0.5 * h as f32 / (30f32.to_radians()).tan();
            // Cut push (144 B): planes[6] (96) + cam xyz (12) + proj_factor + tau + 4 slots.
            // Frustum planes come from the UNFLIPPED proj*view (the no-flip cull matrix).
            let planes = crate::push::frustum_planes(
                Mat4::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.05, 100.0) * view,
            );
            let mut cpc = [0u8; 224];
            for (i, plane) in planes.iter().enumerate() {
                for (j, f) in plane.iter().enumerate() {
                    cpc[i * 16 + j * 4..i * 16 + j * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
            for (i, f) in eye.to_array().iter().enumerate() {
                cpc[96 + i * 4..100 + i * 4].copy_from_slice(&f.to_le_bytes());
            }
            cpc[108..112].copy_from_slice(&proj_factor.to_le_bytes());
            cpc[112..116].copy_from_slice(&tau.to_le_bytes());
            let cwords = [
                total_clusters,
                rec_buf.storage_index(),
                vis_buf.storage_index(),
                args_buf.storage_index(),
            ];
            for (i, word) in cwords.iter().enumerate() {
                cpc[116 + i * 4..120 + i * 4].copy_from_slice(&word.to_le_bytes());
            }
            // M5b tail (occupies csCut's former pad → zero for csCut): SW sub-list + threshold.
            if bin {
                cpc[132..136].copy_from_slice(&sw_list.storage_index().to_le_bytes());
                cpc[136..140].copy_from_slice(&sw_args.storage_index().to_le_bytes());
                cpc[140..144].copy_from_slice(&bin_px.to_le_bytes());
            }
            // World-space cut (M-integration): the viewer recenters its mesh, so cluster space IS
            // world → `model = identity`, `max_scale = 1` make the shader's transform a no-op.
            let ident = Mat4::IDENTITY.to_cols_array();
            for (i, v) in ident.iter().enumerate() {
                cpc[144 + i * 4..148 + i * 4].copy_from_slice(&v.to_le_bytes());
            }
            cpc[208..212].copy_from_slice(&1.0f32.to_le_bytes());
            cmd.bind_compute_pipeline(cut_pipeline.as_ref().unwrap());
            cmd.push_constants_compute(&cpc);
            cmd.dispatch(total_clusters.div_ceil(64), 1, 1);
        }

        // M5: clear the R64 visibility buffer and rasterize the SW clusters into it (compute), one
        // threadgroup per visible cluster via the indirect count. Single-SW mode rasterizes the
        // whole cut (`vis_buf`/`args_buf`); binning mode rasterizes only the SW sub-list
        // (`sw_list`/`sw_args`) and the HW mesh pass (below) writes the rest of the same buffer.
        if sw || bin {
            let (raster_list, raster_args) = if bin {
                (&sw_list, &sw_args)
            } else {
                (&vis_buf, &args_buf)
            };
            let mut spc = [0u8; 96];
            for (i, v) in mvp.iter().enumerate() {
                spc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let words = [
                vtx_buf.storage_index(),
                remap_buf.storage_index(),
                tri_buf.storage_index(),
                rec_buf.storage_index(),
                visbuf.storage_index(),
                raster_list.storage_index(),
                w,
                h,
            ];
            for (i, word) in words.iter().enumerate() {
                spc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
            }
            cmd.bind_compute_pipeline(sw_clear_pipeline.as_ref().unwrap());
            cmd.push_constants_compute(&spc);
            cmd.dispatch((w * h).div_ceil(64), 1, 1);
            cmd.bind_compute_pipeline(sw_raster_pipeline.as_ref().unwrap());
            cmd.push_constants_compute(&spc);
            cmd.dispatch_indirect(raster_args, 0);
        }

        let color = ClearColor {
            r: 0.08,
            g: 0.09,
            b: 0.12,
            a: 1.0,
        };
        cmd.transition_to_render_target(swapchain, image_index);
        // M5b: HW mesh-vis pass — rasterize the HW sub-list into the SAME visibility buffer via a
        // fragment atomicMax. It runs in its OWN render encoder so the cross-encoder hazard fence
        // orders its visibility writes before the visualization pass reads them (its colour output
        // is unused; the visualization pass overwrites the swapchain). Depth-less: the atomicMax
        // resolves occlusion exactly like the SW rasterizer, so HW and SW agree per pixel.
        if bin {
            cmd.begin_rendering(swapchain, image_index, Some(color), None);
            cmd.set_viewport_scissor(swapchain);
            let mut hpc = [0u8; 96];
            for (i, v) in mvp.iter().enumerate() {
                hpc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let words = [
                vtx_buf.storage_index(),
                remap_buf.storage_index(),
                tri_buf.storage_index(),
                rec_buf.storage_index(),
                vis_buf.storage_index(), // the HW sub-list
                visbuf.storage_index(),  // the shared R64 visibility buffer
                w,
                h,
            ];
            for (i, word) in words.iter().enumerate() {
                hpc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
            }
            cmd.bind_mesh_pipeline(hwvis_pipeline.as_ref().unwrap());
            cmd.push_constants_mesh(&hpc);
            cmd.draw_mesh_tasks_indirect(&args_buf, 0);
            cmd.end_rendering();
        }
        cmd.begin_rendering(
            swapchain,
            image_index,
            Some(color),
            if sw || bin { None } else { Some(&depth) },
        );
        cmd.set_viewport_scissor(swapchain);
        if let Some(rp) = resolve_pipeline.as_ref() {
            // M6: resolve the visibility buffer → attributes → shaded surface. Push (112 B):
            // mvp (64) + geometry slots + visbuf + size + mode/tex.
            let mut rpc = [0u8; 112];
            for (i, v) in mvp.iter().enumerate() {
                rpc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let words = [
                vtx_buf.storage_index(),
                remap_buf.storage_index(),
                tri_buf.storage_index(),
                rec_buf.storage_index(),
                visbuf.storage_index(),
                w,
                h,
                mode,
                tex_index,
            ];
            for (i, word) in words.iter().enumerate() {
                rpc[64 + i * 4..68 + i * 4].copy_from_slice(&word.to_le_bytes());
            }
            cmd.bind_graphics_pipeline(rp);
            cmd.push_constants(&rpc);
            cmd.draw(3, 1);
        } else if sw || bin {
            // Full-screen visualization of the visibility buffer (cluster id → colour).
            let mut vpc = [0u8; 16];
            vpc[0..4].copy_from_slice(&visbuf.storage_index().to_le_bytes());
            vpc[4..8].copy_from_slice(&w.to_le_bytes());
            vpc[8..12].copy_from_slice(&h.to_le_bytes());
            cmd.bind_graphics_pipeline(vis_pipeline.as_ref().unwrap());
            cmd.push_constants(&vpc);
            cmd.draw(3, 1);
        } else {
            cmd.bind_mesh_pipeline(&pipeline);
            cmd.push_constants_mesh(&pc);
            if cut {
                cmd.draw_mesh_tasks_indirect(&args_buf, 0);
            } else {
                cmd.draw_mesh_tasks(lod_count, 1, 1);
            }
        }
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
            info!("saved vgeo-mesh screenshot {}", cap.path);
        }

        queue.present(swapchain, image_index, &render_finished)?;
        frames += 1;
        if capture.is_some() {
            break;
        }
    }
    device.wait_idle()?;
    if cut {
        let read_count = |b: &StorageBuffer| -> anyhow::Result<u32> {
            let mut a = [0u8; 12];
            b.read_into(&mut a)?;
            Ok(u32::from_le_bytes(a[0..4].try_into().unwrap()))
        };
        if bin {
            info!(
                "vgeo-mesh bin: {} HW clusters + {} SW clusters selected (last frame)",
                read_count(&args_buf)?,
                read_count(&sw_args)?
            );
        } else {
            info!(
                "vgeo-mesh cut: {} clusters selected (last frame)",
                read_count(&args_buf)?
            );
        }
    }
    // Keep the uploaded buffers alive for the loop's lifetime.
    let _ = (
        &vtx_buf,
        &remap_buf,
        &tri_buf,
        &rec_buf,
        &vis_buf,
        &args_buf,
        &sw_list,
        &sw_args,
        &checker,
        &cut_pipeline,
        &visbuf,
        &sw_clear_pipeline,
        &sw_raster_pipeline,
        &vis_pipeline,
        &hwvis_pipeline,
        &resolve_pipeline,
    );
    info!("vgeo-mesh exited after {frames} frame(s)");
    Ok(())
}

/// HSV → RGB (`h`,`s`,`v` in `[0,1]`). Used to spread cluster ids across distinct hues.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.fract() * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// A 256×1 palette of well-separated hues (golden-ratio spacing) as sRGB8 RGBA — the colour table
/// the meshlet debug view samples by cluster/LOD id.
fn hue_palette() -> dreamcoast_asset::ImageData {
    let mut rgba8 = Vec::with_capacity(256 * 4);
    for i in 0..256u32 {
        let hue = (i as f32 * 0.618_034).fract();
        let c = hsv_to_rgb(hue, 0.72, 0.95);
        rgba8.extend_from_slice(&[
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
            255,
        ]);
    }
    dreamcoast_asset::ImageData {
        width: 256,
        height: 1,
        rgba8,
    }
}

/// Reconstruct a mesh where every cluster (or LOD level) at `lod` is a distinct flat colour,
/// keeping real vertex normals for shading. Colour is carried in `uv.x` (the palette slot), so the
/// stock textured mesh pipeline renders it; non-indexed so cluster colours never bleed at seams.
fn meshlet_debug_mesh(
    dag: &dreamcoast_asset::vgeo::MeshClusters,
    lod: u32,
    by_lod: bool,
) -> MeshData {
    let mut vertices: Vec<dreamcoast_asset::MeshVertex> = Vec::new();
    for (draw_id, c) in dag
        .clusters
        .iter()
        .filter(|c| c.lod_level == lod)
        .enumerate()
    {
        // Palette slot: the LOD level, or a per-cluster id spread across the 256-entry palette.
        let slot = if by_lod {
            c.lod_level as usize
        } else {
            draw_id
        } % 256;
        let u = (slot as f32 + 0.5) / 256.0;
        let vbase = c.vertex_offset as usize;
        let tbase = c.triangle_offset as usize;
        for k in 0..c.triangle_count as usize * 3 {
            let src = dag.cluster_vertices[vbase + dag.cluster_triangles[tbase + k] as usize];
            let v = dag.vertices[src as usize];
            vertices.push(dreamcoast_asset::MeshVertex {
                pos: v.pos,
                normal: v.normal,
                uv: [u, 0.5],
            });
        }
    }
    let indices: Vec<u32> = (0..vertices.len() as u32).collect();
    let material = dreamcoast_asset::Material {
        base_color: Some(dreamcoast_asset::TexData::Rgba8(hue_palette())),
        ..Default::default()
    };
    MeshData {
        vertices,
        indices,
        material,
    }
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
