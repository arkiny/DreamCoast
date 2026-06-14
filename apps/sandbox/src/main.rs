//! Sandbox: the playground executable.
//!
//! Phase 1 scope: open a window and draw `triangle.slang` through the `rhi`
//! facade (Vulkan backend). Demonstrates the full frame loop — acquire, record
//! (dynamic rendering), submit, present — plus swapchain recreation on resize.

use std::time::Duration;

use anyhow::anyhow;
use engine_core::init_logging;
use engine_platform::Window;
use rhi::{
    BackendKind, ClearColor, Device, Extent2D, Format, GraphicsPipelineDesc, Instance,
    InstanceDesc, PresentMode, PrimitiveTopology, Semaphore, SwapchainDesc,
};
use tracing::info;

/// Number of frames the CPU may record ahead of the GPU.
const FRAMES_IN_FLIGHT: usize = 2;

/// Swapchain color format used by both the swapchain and the pipeline.
const COLOR_FORMAT: Format = Format::Bgra8Srgb;

fn swapchain_desc(extent: Extent2D) -> SwapchainDesc {
    SwapchainDesc {
        extent,
        format: COLOR_FORMAT,
        present_mode: PresentMode::Fifo,
        image_count: 3,
    }
}

fn main() -> anyhow::Result<()> {
    init_logging();

    // Phase 1 requires the compiled triangle shaders (see docs/phase-1-rhi-vulkan.md).
    let vs = engine_shader::triangle_vs_spirv().ok_or_else(|| {
        anyhow!("triangle vertex SPIR-V unavailable — install slangc and rebuild")
    })?;
    let fs = engine_shader::triangle_fs_spirv().ok_or_else(|| {
        anyhow!("triangle fragment SPIR-V unavailable — install slangc and rebuild")
    })?;

    let mut window = Window::new("Engine Sandbox — Vulkan Triangle", 1280, 720)?;
    let (w, h) = window.size();

    let instance = Instance::new(
        BackendKind::Vulkan,
        &window,
        &InstanceDesc {
            app_name: "engine-sandbox".into(),
            validation: true,
        },
    )?;
    info!("backend: {:?}", instance.backend());

    let device = instance.create_device()?;
    let queue = device.queue();

    let mut swapchain = device.create_swapchain(&swapchain_desc(Extent2D::new(w, h)))?;

    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: vs,
        fragment_bytes: fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_format: swapchain.format(),
        topology: PrimitiveTopology::TriangleList,
    })?;

    // Per-frame-in-flight resources.
    let mut command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        command_buffers.push(device.create_command_buffer()?);
        image_available.push(device.create_semaphore()?);
        in_flight.push(device.create_fence(true)?);
    }
    // One render-finished semaphore per swapchain image (avoids WSI reuse hazards).
    let mut render_finished = build_render_finished(&device, swapchain.image_count())?;

    // Discard the resize flag raised by the initial WM_SIZE during window
    // creation; the first swapchain already matches the current size.
    let _ = window.take_resized();

    info!("entering render loop");
    let mut frame = 0usize;
    let mut needs_recreate = false;

    while !window.should_close() {
        window.pump_events();
        if window.take_resized() {
            needs_recreate = true;
        }

        let (cw, ch) = window.size();
        if cw == 0 || ch == 0 {
            // Minimized: skip rendering until restored.
            std::thread::sleep(Duration::from_millis(16));
            continue;
        }

        if needs_recreate {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(cw, ch)))?;
            render_finished = build_render_finished(&device, swapchain.image_count())?;
            needs_recreate = false;
        }

        let fence = &in_flight[frame];
        fence.wait()?;

        let image_index = match swapchain.acquire_next_image(&image_available[frame])? {
            Some(index) => index,
            None => {
                needs_recreate = true;
                continue;
            }
        };
        fence.reset()?;

        let cmd = &command_buffers[frame];
        cmd.begin()?;
        cmd.transition_to_render_target(&swapchain, image_index);
        cmd.begin_rendering(
            &swapchain,
            image_index,
            ClearColor {
                r: 0.02,
                g: 0.02,
                b: 0.06,
                a: 1.0,
            },
        );
        cmd.set_viewport_scissor(&swapchain);
        cmd.bind_graphics_pipeline(&pipeline);
        cmd.draw(3, 1);
        cmd.end_rendering();
        cmd.transition_to_present(&swapchain, image_index);
        cmd.end()?;

        let signal = &render_finished[image_index as usize];
        queue.submit(cmd, &image_available[frame], signal, fence)?;
        if queue.present(&swapchain, image_index, signal)? {
            needs_recreate = true;
        }

        frame = (frame + 1) % FRAMES_IN_FLIGHT;
    }

    device.wait_idle()?;
    info!("shutting down");
    Ok(())
}

fn build_render_finished(device: &Device, count: u32) -> anyhow::Result<Vec<Semaphore>> {
    (0..count)
        .map(|_| device.create_semaphore().map_err(Into::into))
        .collect()
}
