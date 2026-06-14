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
    BackendKind, BlendMode, BufferDesc, BufferUsage, ClearColor, Device, Extent2D, Format,
    GraphicsPipelineDesc, Instance, InstanceDesc, PresentMode, PrimitiveTopology, Rect2D,
    Semaphore, SwapchainDesc, TextureDesc, VertexLayout,
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

    let backend = select_backend();
    info!("requested backend: {backend:?}");

    // Each backend consumes its own bytecode: SPIR-V for Vulkan, DXIL for D3D12.
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
    let vs = vs.ok_or_else(|| {
        anyhow!(
            "triangle vertex shader unavailable for {backend:?} — install slangc/DXC and rebuild"
        )
    })?;
    let fs = fs.ok_or_else(|| {
        anyhow!(
            "triangle fragment shader unavailable for {backend:?} — install slangc/DXC and rebuild"
        )
    })?;

    let title = format!("Engine Sandbox — {backend:?} Triangle");
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

    // --- Bindless smoke test: a textured quad over the triangle ---
    let (quad_vs, quad_fs) = match backend {
        BackendKind::Vulkan => (
            engine_shader::imgui_vs_spirv(),
            engine_shader::imgui_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            engine_shader::imgui_vs_dxil(),
            engine_shader::imgui_fs_dxil(),
        ),
    };
    let quad_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: quad_vs.ok_or_else(|| anyhow!("imgui vs unavailable"))?,
        fragment_bytes: quad_fs.ok_or_else(|| anyhow!("imgui fs unavailable"))?,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_format: swapchain.format(),
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::ImGui,
        blend: BlendMode::AlphaBlend,
        push_constant_size: 20,
        bindless: true,
    })?;
    let checker = make_checker_texture(&device)?;
    let (quad_vbuf, quad_ibuf) = make_quad_buffers(&device)?;
    let mut quad_pc = [0u8; 20];
    quad_pc[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // scale.x
    quad_pc[4..8].copy_from_slice(&1.0f32.to_le_bytes()); // scale.y
    quad_pc[8..12].copy_from_slice(&0.0f32.to_le_bytes()); // translate.x
    quad_pc[12..16].copy_from_slice(&0.0f32.to_le_bytes()); // translate.y
    quad_pc[16..20].copy_from_slice(&checker.bindless_index().to_le_bytes());

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

        // Bindless smoke test: textured quad.
        cmd.bind_graphics_pipeline(&quad_pipeline);
        cmd.push_constants(&quad_pc);
        cmd.bind_vertex_buffer(&quad_vbuf, 20);
        cmd.bind_index_buffer(&quad_ibuf, false);
        cmd.set_scissor(Rect2D {
            x: 0,
            y: 0,
            width: cw,
            height: ch,
        });
        cmd.draw_indexed(6, 0, 0);

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

/// An 8x8 magenta/black checkerboard RGBA8 texture (bindless smoke test).
fn make_checker_texture(device: &Device) -> anyhow::Result<rhi::Texture> {
    const N: u32 = 8;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let on = (x + y) % 2 == 0;
            let rgba = if on {
                [255, 0, 255, 255]
            } else {
                [16, 16, 16, 255]
            };
            pixels.extend_from_slice(&rgba);
        }
    }
    Ok(device.create_texture(
        &TextureDesc {
            width: N,
            height: N,
            format: Format::Rgba8Unorm,
        },
        &pixels,
    )?)
}

/// A unit quad (two triangles) in NDC with full UVs and white vertex color,
/// in ImGui `ImDrawVert` layout (pos f32x2, uv f32x2, color unorm8x4).
fn make_quad_buffers(device: &Device) -> anyhow::Result<(rhi::Buffer, rhi::Buffer)> {
    fn vtx(out: &mut Vec<u8>, x: f32, y: f32, u: f32, v: f32) {
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    }
    let mut verts = Vec::new();
    vtx(&mut verts, -0.9, -0.9, 0.0, 0.0);
    vtx(&mut verts, -0.4, -0.9, 1.0, 0.0);
    vtx(&mut verts, -0.4, -0.4, 1.0, 1.0);
    vtx(&mut verts, -0.9, -0.4, 0.0, 1.0);
    let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
    let mut idx_bytes = Vec::new();
    for i in indices {
        idx_bytes.extend_from_slice(&i.to_le_bytes());
    }

    let vbuf = device.create_buffer(&BufferDesc {
        size: verts.len() as u64,
        usage: BufferUsage::Vertex,
    })?;
    vbuf.write(&verts)?;
    let ibuf = device.create_buffer(&BufferDesc {
        size: idx_bytes.len() as u64,
        usage: BufferUsage::Index,
    })?;
    ibuf.write(&idx_bytes)?;
    Ok((vbuf, ibuf))
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
