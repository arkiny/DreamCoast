//! Sandbox: the playground executable.
//!
//! Builds a render graph each frame: a glTF mesh (or procedural cube fallback)
//! rendered textured + diffuse-lit with depth into an offscreen target, a
//! three-pass bloom chain, then a composite to the backbuffer plus a Dear ImGui
//! overlay. The ImGui panel toggles the post effect and transient memory
//! aliasing. Runs on either backend (`--backend vulkan|d3d12`).

use std::time::Instant;

use anyhow::anyhow;
use dreamcoast_asset::MeshData;
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_core::init_logging;
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use dreamcoast_render::{PassInfo, RenderGraph, ResourcePool};
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
            dreamcoast_shader::mesh_vs_spirv(),
            dreamcoast_shader::mesh_fs_spirv(),
        ),
        BackendKind::D3d12 => (
            dreamcoast_shader::mesh_vs_dxil(),
            dreamcoast_shader::mesh_fs_dxil(),
        ),
    };
    let vs = vs.ok_or_else(|| anyhow!("mesh vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("mesh fragment shader unavailable for {backend:?}"))?;

    // Load a glTF model if present, else fall back to a procedural cube.
    let model_path = model_path();
    let model = match dreamcoast_asset::load_gltf(&model_path) {
        Ok(m) => {
            info!(
                "loaded {model_path}: {} verts, {} indices",
                m.vertices.len(),
                m.indices.len()
            );
            m
        }
        Err(e) => {
            info!("no glTF at {model_path} ({e}); using procedural cube");
            dreamcoast_asset::unit_cube()
        }
    };
    let model_radius = bounding_radius(&model);

    let title = format!("DreamCoast Sandbox — {backend:?}");
    let mut window = Window::new(&title, 1280, 720)?;
    let (w, h) = window.size();

    let instance = Instance::new(
        backend,
        &window,
        &InstanceDesc {
            app_name: "dreamcoast-sandbox".into(),
            validation: true,
        },
    )?;
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
        vertex_layout: VertexLayout::Mesh,
        blend: BlendMode::Opaque,
        push_constant_size: 68, // mat4 (64) + u32 tex_index (4)
        bindless: true,
        depth_test: true,
        depth_format: Some(DEPTH_FORMAT),
    })?;

    // Full-screen bloom-blur pipeline (one link of the chain; reused H/V/H).
    let (blur_vs, blur_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::blur_vs_spirv,
        dreamcoast_shader::blur_fs_spirv,
        dreamcoast_shader::blur_vs_dxil,
        dreamcoast_shader::blur_fs_dxil,
        "blur",
    )?;
    let blur_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: blur_vs,
        fragment_bytes: blur_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_format: COLOR_FORMAT,
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 20, // u32 src + u32 flip_y + float2 dir + f32 threshold
        bindless: true,
        depth_test: false,
        depth_format: None,
    })?;

    // Composite pipeline (scene + bloom + tonemap -> backbuffer).
    let (post_vs, post_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::post_vs_spirv,
        dreamcoast_shader::post_fs_spirv,
        dreamcoast_shader::post_vs_dxil,
        dreamcoast_shader::post_fs_dxil,
        "post",
    )?;
    let post_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: post_vs,
        fragment_bytes: post_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_format: swapchain.format(),
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 16, // u32 scene + u32 bloom + u32 mode + u32 flip_y
        bindless: true,
        depth_test: false,
        depth_format: None,
    })?;
    // Clip-space Y orientation for the fullscreen-triangle passes (matches the
    // mesh pass's per-backend proj.y flip).
    let post_flip_y: u32 = match backend {
        BackendKind::Vulkan => 1,
        BackendKind::D3d12 => 0,
    };

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

    // One render-graph transient pool per frame-in-flight: a pool's resources are
    // reused only after that frame slot's fence has signaled, so reusing them is
    // free of cross-frame read/write hazards.
    let mut pools: Vec<ResourcePool> = (0..FRAMES_IN_FLIGHT).map(|_| ResourcePool::new()).collect();
    let mut post_mode: usize = 0;
    let mut aliasing = true;
    const POST_EFFECTS: [&str; 3] = ["None", "Grayscale", "Vignette"];

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
            for p in &mut pools {
                p.clear(); // transient extents changed; drop cached targets
            }
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
            ui.window("DreamCoast")
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
                    ui.separator();
                    ui.combo_simple_string("Post effect", &mut post_mode, &POST_EFFECTS);
                    ui.checkbox("Transient aliasing", &mut aliasing);
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

        // Build the frame's render graph:
        //   scene -> bloom_h0 -> bloom_v -> bloom_h1 -> composite -> ImGui
        // The three bloom links create transient targets with partially disjoint
        // lifetimes, which the graph's transient aliasing reuses.
        let extent = Extent2D::new(cw, ch);
        let dir_h = [2.0 / cw as f32, 0.0];
        let dir_v = [0.0, 2.0 / ch as f32];

        let mut graph = RenderGraph::new();
        let backbuffer = graph.import_backbuffer(swapchain.format(), extent);
        let scene = graph.create_color("scene", COLOR_FORMAT, extent);
        let scene_depth = graph.create_depth("scene_depth", extent);
        let bloom_a = graph.create_color("bloom_a", COLOR_FORMAT, extent);
        let bloom_b = graph.create_color("bloom_b", COLOR_FORMAT, extent);
        let bloom_c = graph.create_color("bloom_c", COLOR_FORMAT, extent);

        let scene_clear = ClearColor {
            r: 0.02,
            g: 0.02,
            b: 0.06,
            a: 1.0,
        };
        graph.add_pass(
            PassInfo {
                name: "scene",
                color: Some((scene, Some(scene_clear))),
                depth: Some(scene_depth),
                reads: vec![],
            },
            |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&pipeline);
                cmd.push_constants(&mesh_push_constants(mvp, tex_index));
                cmd.bind_vertex_buffer(&vbuf, 32);
                cmd.bind_index_buffer(&ibuf, true);
                cmd.draw_indexed(index_count, 0, 0);
                Ok(())
            },
        );
        // Three blur links: bright-pass + horizontal, then vertical, then horizontal.
        let blur_pipeline_ref = &blur_pipeline;
        for (name, dst, src, dir, threshold) in [
            ("bloom_h0", bloom_a, scene, dir_h, 0.6f32),
            ("bloom_v", bloom_b, bloom_a, dir_v, 0.0),
            ("bloom_h1", bloom_c, bloom_b, dir_h, 0.0),
        ] {
            graph.add_pass(
                PassInfo {
                    name,
                    color: Some((dst, None)),
                    depth: None,
                    reads: vec![src],
                },
                move |ctx| {
                    let src_index = ctx.sampled_index(src);
                    let cmd = ctx.cmd();
                    cmd.bind_graphics_pipeline(blur_pipeline_ref);
                    cmd.push_constants(&blur_push_constants(
                        src_index,
                        post_flip_y,
                        dir,
                        threshold,
                    ));
                    cmd.draw(3, 1);
                    Ok(())
                },
            );
        }
        graph.add_pass(
            PassInfo {
                name: "composite",
                color: Some((backbuffer, Some(ClearColor::BLACK))),
                depth: None,
                reads: vec![scene, bloom_c],
            },
            |ctx| {
                let scene_index = ctx.sampled_index(scene);
                let bloom_index = ctx.sampled_index(bloom_c);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&post_pipeline);
                cmd.push_constants(&post_push_constants(
                    scene_index,
                    bloom_index,
                    post_mode as u32,
                    post_flip_y,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
        graph.add_pass(
            PassInfo {
                name: "ui",
                color: Some((backbuffer, None)),
                depth: None,
                reads: vec![],
            },
            |ctx| gui.render(&device, ctx.cmd(), frame),
        );
        graph.execute(
            &device,
            &mut pools[frame],
            cmd,
            &swapchain,
            image_index,
            aliasing,
        )?;

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

/// Pack the composite push block: scene_index + bloom_index + mode + flip_y (16B).
fn post_push_constants(scene_index: u32, bloom_index: u32, mode: u32, flip_y: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&scene_index.to_le_bytes());
    pc[4..8].copy_from_slice(&bloom_index.to_le_bytes());
    pc[8..12].copy_from_slice(&mode.to_le_bytes());
    pc[12..16].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the blur push block: src_index + flip_y + direction.xy + threshold (20B).
fn blur_push_constants(src_index: u32, flip_y: u32, dir: [f32; 2], threshold: f32) -> [u8; 20] {
    let mut pc = [0u8; 20];
    pc[0..4].copy_from_slice(&src_index.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc[8..12].copy_from_slice(&dir[0].to_le_bytes());
    pc[12..16].copy_from_slice(&dir[1].to_le_bytes());
    pc[16..20].copy_from_slice(&threshold.to_le_bytes());
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

/// Fetch the (vertex, fragment) bytecode for `backend` from a shader's four
/// generated accessors, erroring if unavailable.
fn load_shader_pair(
    backend: BackendKind,
    vs_spirv: fn() -> Option<&'static [u8]>,
    fs_spirv: fn() -> Option<&'static [u8]>,
    vs_dxil: fn() -> Option<&'static [u8]>,
    fs_dxil: fn() -> Option<&'static [u8]>,
    name: &str,
) -> anyhow::Result<(&'static [u8], &'static [u8])> {
    let (vs, fs) = match backend {
        BackendKind::Vulkan => (vs_spirv(), fs_spirv()),
        BackendKind::D3d12 => (vs_dxil(), fs_dxil()),
    };
    let vs = vs.ok_or_else(|| anyhow!("{name} vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("{name} fragment shader unavailable for {backend:?}"))?;
    Ok((vs, fs))
}

fn build_render_finished(device: &Device, count: u32) -> anyhow::Result<Vec<Semaphore>> {
    (0..count)
        .map(|_| device.create_semaphore().map_err(Into::into))
        .collect()
}

/// Model path: `--model <path>` or the default `assets/model.glb`.
fn model_path() -> String {
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
