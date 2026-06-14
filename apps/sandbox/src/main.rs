//! Sandbox: the playground executable.
//!
//! Phase 3 scope: draw `triangle.slang` and a Dear ImGui overlay through the
//! `rhi` facade, on either backend (`--backend vulkan|d3d12`).

use std::time::Instant;

use anyhow::anyhow;
use engine_core::init_logging;
use engine_gui::{Gui, imgui};
use engine_platform::Window;
use rhi::{
    BackendKind, BlendMode, ClearColor, Device, Extent2D, Format, GraphicsPipelineDesc, Instance,
    InstanceDesc, PresentMode, PrimitiveTopology, Semaphore, SwapchainDesc, VertexLayout,
};
use tracing::info;

/// Number of frames the CPU may record ahead of the GPU.
const FRAMES_IN_FLIGHT: usize = 2;

/// Swapchain color format used by the swapchain and pipelines.
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

    let backend = select_backend();
    info!("requested backend: {backend:?}");

    let (vs, fs) = match backend {
        BackendKind::Vulkan => (
            engine_shader::triangle_vs_spirv(),
            engine_shader::triangle_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            engine_shader::triangle_vs_dxil(),
            engine_shader::triangle_fs_dxil(),
        ),
    };
    let vs = vs.ok_or_else(|| anyhow!("triangle vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("triangle fragment shader unavailable for {backend:?}"))?;

    let title = format!("Engine Sandbox — {backend:?}");
    let mut window = Window::new(&title, 1280, 720)?;
    let (w, h) = window.size();

    let instance = Instance::new(
        backend,
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
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 0,
        bindless: false,
    })?;

    let mut gui = Gui::new(&device, swapchain.format(), FRAMES_IN_FLIGHT)?;

    // Per-frame-in-flight resources.
    let mut command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        command_buffers.push(device.create_command_buffer()?);
        image_available.push(device.create_semaphore()?);
        in_flight.push(device.create_fence(true)?);
    }
    let mut render_finished = build_render_finished(&device, swapchain.image_count())?;

    // Discard the resize flag raised by the initial WM_SIZE during creation.
    let _ = window.take_resized();

    info!("entering render loop");
    let mut frame = 0usize;
    let mut needs_recreate = false;
    let mut last = Instant::now();
    let mut clear = [0.02f32, 0.02, 0.06];

    while !window.should_close() {
        window.pump_events();
        if window.take_resized() {
            needs_recreate = true;
        }

        let (cw, ch) = window.size();
        if cw == 0 || ch == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }

        if needs_recreate {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(cw, ch)))?;
            render_finished = build_render_finished(&device, swapchain.image_count())?;
            needs_recreate = false;
        }

        // Update + build the UI for this frame.
        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        {
            let ui = gui.new_frame(dt, [cw as f32, ch as f32], window.input());
            ui.window("Engine")
                .size([280.0, 140.0], imgui::Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("backend: {backend:?}"));
                    ui.text(format!("size: {cw} x {ch}"));
                    ui.text(format!(
                        "{:.1} FPS ({:.2} ms)",
                        1.0 / dt.max(1e-4),
                        dt * 1000.0
                    ));
                    ui.separator();
                    ui.color_edit3("clear", &mut clear);
                });
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
                r: clear[0],
                g: clear[1],
                b: clear[2],
                a: 1.0,
            },
        );
        cmd.set_viewport_scissor(&swapchain);
        cmd.bind_graphics_pipeline(&pipeline);
        cmd.draw(3, 1);

        gui.render(&device, cmd, frame)?;

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

/// Pick the backend: OS default (Windows -> D3D12, else Vulkan), overridable
/// with `--backend vulkan|d3d12`.
fn select_backend() -> BackendKind {
    let mut backend = if cfg!(windows) {
        BackendKind::D3d12
    } else {
        BackendKind::Vulkan
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--backend" {
            match args.next().as_deref() {
                Some("vulkan") => backend = BackendKind::Vulkan,
                Some("d3d12") => backend = BackendKind::D3d12,
                other => tracing::warn!("unknown --backend value {other:?}; using default"),
            }
        }
    }
    backend
}
