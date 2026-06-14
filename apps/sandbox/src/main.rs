//! Sandbox: the playground executable.
//!
//! Phase 4 scope: load a glTF mesh (or a procedural cube fallback) and render it
//! textured + diffuse-lit with depth, an orbiting camera, and a Dear ImGui
//! overlay — on either backend (`--backend vulkan|d3d12`).

use std::time::Instant;

use anyhow::anyhow;
use engine_asset::MeshData;
use engine_core::glam::{Mat4, Vec3};
use engine_core::init_logging;
use engine_gui::{Gui, imgui};
use engine_platform::Window;
use rhi::{
    BackendKind, BlendMode, Buffer, BufferDesc, BufferUsage, ClearColor, Device, Extent2D, Format,
    GraphicsPipelineDesc, Instance, InstanceDesc, PresentMode, PrimitiveTopology, Semaphore,
    SwapchainDesc, Texture, TextureDesc, VertexLayout,
};
use tracing::info;

const FRAMES_IN_FLIGHT: usize = 2;
const COLOR_FORMAT: Format = Format::Bgra8Srgb;
const DEPTH_FORMAT: Format = Format::Depth32Float;
const MODEL_PATH: &str = "assets/model.glb";

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
            engine_shader::mesh_vs_spirv(),
            engine_shader::mesh_fs_spirv(),
        ),
        BackendKind::D3d12 => (engine_shader::mesh_vs_dxil(), engine_shader::mesh_fs_dxil()),
    };
    let vs = vs.ok_or_else(|| anyhow!("mesh vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("mesh fragment shader unavailable for {backend:?}"))?;

    // Load a glTF model if present, else fall back to a procedural cube.
    let model = match engine_asset::load_gltf(MODEL_PATH) {
        Ok(m) => {
            info!(
                "loaded {MODEL_PATH}: {} verts, {} indices",
                m.vertices.len(),
                m.indices.len()
            );
            m
        }
        Err(e) => {
            info!("no glTF ({e}); using procedural cube");
            engine_asset::unit_cube()
        }
    };
    let model_radius = bounding_radius(&model);

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
    let device = instance.create_device()?;
    let queue = device.queue();

    let mut swapchain = device.create_swapchain(&swapchain_desc(Extent2D::new(w, h)))?;
    let mut depth = device.create_depth_buffer(Extent2D::new(w, h))?;

    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: vs,
        fragment_bytes: fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_format: swapchain.format(),
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::Mesh,
        blend: BlendMode::Opaque,
        push_constant_size: 68, // mat4 (64) + u32 tex_index (4)
        bindless: true,
        depth_test: true,
        depth_format: Some(DEPTH_FORMAT),
    })?;

    // Upload geometry + base-color texture (bindless).
    let (vbuf, ibuf, index_count) = upload_mesh(&device, &model)?;
    let _base_color: Texture;
    let tex_index;
    if let Some(img) = &model.base_color {
        let t = device.create_texture(
            &TextureDesc {
                width: img.width,
                height: img.height,
                format: Format::Rgba8Unorm,
            },
            &img.rgba8,
        )?;
        tex_index = t.bindless_index();
        _base_color = t;
    } else {
        let t = make_checker_texture(&device)?;
        tex_index = t.bindless_index();
        _base_color = t;
    }

    let mut gui = Gui::new(&device, swapchain.format(), FRAMES_IN_FLIGHT)?;

    let mut command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        command_buffers.push(device.create_command_buffer()?);
        image_available.push(device.create_semaphore()?);
        in_flight.push(device.create_fence(true)?);
    }
    let mut render_finished = build_render_finished(&device, swapchain.image_count())?;

    let _ = window.take_resized();
    info!("entering render loop");
    let mut frame = 0usize;
    let mut needs_recreate = false;
    let mut last = Instant::now();
    let mut angle = 0.0f32;

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
            depth = device.create_depth_buffer(Extent2D::new(cw, ch))?;
            render_finished = build_render_finished(&device, swapchain.image_count())?;
            needs_recreate = false;
        }

        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        angle += dt * 0.6;

        // Orbiting camera looking at the model.
        let dist = model_radius * 3.0;
        let eye = Vec3::new(angle.cos() * dist, model_radius * 1.2, angle.sin() * dist);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0; // Vulkan clip-space Y points down
        }
        let mvp = (proj * view).to_cols_array();

        {
            let ui = gui.new_frame(dt, [cw as f32, ch as f32], window.input());
            ui.window("Engine")
                .size([300.0, 150.0], imgui::Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("backend: {backend:?}"));
                    ui.text(format!("size: {cw} x {ch}"));
                    ui.text(format!(
                        "{:.1} FPS ({:.2} ms)",
                        1.0 / dt.max(1e-4),
                        dt * 1000.0
                    ));
                    ui.text(format!("model: {index_count} indices"));
                });
        }

        let fence = &in_flight[frame];
        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available[frame])? {
            Some(i) => i,
            None => {
                needs_recreate = true;
                continue;
            }
        };
        fence.reset()?;

        let cmd = &command_buffers[frame];
        cmd.begin()?;
        cmd.transition_to_render_target(&swapchain, image_index);

        // Pass 1: mesh (clear color + depth).
        cmd.begin_rendering(
            &swapchain,
            image_index,
            Some(ClearColor {
                r: 0.02,
                g: 0.02,
                b: 0.06,
                a: 1.0,
            }),
            Some(&depth),
        );
        cmd.set_viewport_scissor(&swapchain);
        cmd.bind_graphics_pipeline(&pipeline);
        cmd.push_constants(&mesh_push_constants(mvp, tex_index));
        cmd.bind_vertex_buffer(&vbuf, 32);
        cmd.bind_index_buffer(&ibuf, true);
        cmd.draw_indexed(index_count, 0, 0);
        cmd.end_rendering();

        // Pass 2: ImGui overlay (load color, no depth).
        cmd.begin_rendering(&swapchain, image_index, None, None);
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

/// Pack the mesh push-constant block: column-major mvp (64B) + tex_index (4B).
fn mesh_push_constants(mvp: [f32; 16], tex_index: u32) -> [u8; 68] {
    let mut pc = [0u8; 68];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&tex_index.to_le_bytes());
    pc
}

fn upload_mesh(device: &Device, model: &MeshData) -> anyhow::Result<(Buffer, Buffer, u32)> {
    let vbytes = unsafe {
        std::slice::from_raw_parts(
            model.vertices.as_ptr() as *const u8,
            std::mem::size_of_val(model.vertices.as_slice()),
        )
    };
    let ibytes = unsafe {
        std::slice::from_raw_parts(
            model.indices.as_ptr() as *const u8,
            std::mem::size_of_val(model.indices.as_slice()),
        )
    };
    let vbuf = device.create_buffer(&BufferDesc {
        size: vbytes.len() as u64,
        usage: BufferUsage::Vertex,
    })?;
    vbuf.write(vbytes)?;
    let ibuf = device.create_buffer(&BufferDesc {
        size: ibytes.len() as u64,
        usage: BufferUsage::Index,
    })?;
    ibuf.write(ibytes)?;
    Ok((vbuf, ibuf, model.indices.len() as u32))
}

fn bounding_radius(model: &MeshData) -> f32 {
    model
        .vertices
        .iter()
        .map(|v| (v.pos[0] * v.pos[0] + v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2]).sqrt())
        .fold(0.0f32, f32::max)
        .max(0.5)
}

/// 8x8 magenta/grey checker (fallback base color).
fn make_checker_texture(device: &Device) -> anyhow::Result<Texture> {
    const N: u32 = 8;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let on = (x + y) % 2 == 0;
            pixels.extend_from_slice(if on {
                &[220, 60, 200, 255]
            } else {
                &[40, 40, 48, 255]
            });
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

fn build_render_finished(device: &Device, count: u32) -> anyhow::Result<Vec<Semaphore>> {
    (0..count)
        .map(|_| device.create_semaphore().map_err(Into::into))
        .collect()
}

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
