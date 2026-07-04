//! Slate-style loading screen — a dedicated thread that presents a procedural progress bar every
//! frame while the main thread runs the cold cook, so the window stays live (never black / "Not
//! Responding"). See docs/loading-screen-thread.md.
//!
//! **D3D12-only for now.** Its command queue is FREE-THREADED, so the loading thread owns its own
//! `Queue` clone + the swapchain (the existing `P15_RHI_THREAD` single-owner handoff — no mutex, no
//! RHI core change) and the cook keeps uploading on the device's queue concurrently. Vulkan needs
//! external queue synchronization, so it keeps the terminal bar until a later step.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rhi::{
    BackendKind, BlendMode, ClearColor, CommandBuffer, DepthCompare, Device, Extent2D, Fence,
    Format, GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, Queue, Semaphore, Swapchain,
    VertexLayout,
};

/// Shared cook progress the loading thread renders: `frac` in permille (0..1000) + a stop flag.
pub(crate) struct LoadingState {
    frac: AtomicU32,
    stop: AtomicBool,
}

impl LoadingState {
    fn new() -> Self {
        Self {
            frac: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        }
    }

    /// Set the progress fraction (0.0..1.0) — called by the cook at phase boundaries.
    pub(crate) fn set(&self, frac: f32) {
        self.frac
            .store((frac.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
    }
}

/// `--loading-test <path>`: render ONE loading-bar frame (progress from `LOADING_FRAC`, default 0.6)
/// and save it as a PNG — a visual check of the bar without needing the live window. Single-threaded
/// (the device is used directly here, not from the loading thread).
pub(crate) fn run_capture(device: &Device, swapchain: &Swapchain, path: &str) -> anyhow::Result<()> {
    use rhi::{BufferDesc, BufferUsage};
    let ext = swapchain.extent_2d();
    let pipeline = build_pipeline(device, swapchain.format())?;
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;
    let frac: f32 = std::env::var("LOADING_FRAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);
    let layout = device.swapchain_readback_layout(swapchain);
    let buf = device.create_buffer(&BufferDesc {
        size: layout.size,
        usage: BufferUsage::Readback,
    })?;

    fence.wait()?;
    let image_index = swapchain
        .acquire_next_image(&image_available)?
        .ok_or_else(|| anyhow::anyhow!("swapchain acquire failed"))?;
    fence.reset()?;
    let pc = push_bytes(frac, 0.5, ext.width, ext.height);
    cmd.begin()?;
    cmd.transition_to_render_target(swapchain, image_index);
    cmd.begin_rendering(
        swapchain,
        image_index,
        Some(ClearColor {
            r: 0.055,
            g: 0.065,
            b: 0.085,
            a: 1.0,
        }),
        None,
    );
    cmd.set_viewport_scissor(swapchain);
    cmd.bind_graphics_pipeline(&pipeline);
    cmd.push_constants(&pc);
    cmd.draw(3, 1);
    cmd.end_rendering();
    cmd.transition_to_present(swapchain, image_index);
    cmd.copy_swapchain_to_buffer(swapchain, image_index, &buf);
    cmd.end()?;
    device
        .queue()
        .submit(&cmd, &image_available, &render_finished, &fence)?;
    fence.wait()?;
    let mut bytes = vec![0u8; layout.size as usize];
    buf.read_into(&mut bytes)?;
    crate::app::save_screenshot(path, &bytes, &layout)?;
    device.wait_idle()?;
    tracing::info!("loading-test: saved {path} (frac {frac})");
    Ok(())
}

/// Build the loading-bar graphics pipeline. `bindless: true` even though the shader reads no
/// bindless resource: the non-bindless root signature is empty (no push-constant slot), whereas the
/// bindless one has 32-bit root constants at b0 (the slot `push_constants` writes).
/// `bind_graphics_pipeline` rebinds the shared descriptor heap; the shader never samples it, so the
/// cook adding textures to the heap concurrently is harmless.
pub(crate) fn build_pipeline(device: &Device, format: Format) -> anyhow::Result<GraphicsPipeline> {
    let vs = dreamcoast_shader::loading_vs_dxil()
        .ok_or_else(|| anyhow::anyhow!("loading vertex shader unavailable"))?;
    let fs = dreamcoast_shader::loading_fs_dxil()
        .ok_or_else(|| anyhow::anyhow!("loading fragment shader unavailable"))?;
    Ok(device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: vs,
        fragment_bytes: fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[format],
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
    })?)
}

/// Pack the loading push constant: progress fraction + a time value + the target size.
pub(crate) fn push_bytes(frac: f32, time: f32, w: u32, h: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&frac.to_le_bytes());
    pc[4..8].copy_from_slice(&time.to_le_bytes());
    pc[8..12].copy_from_slice(&w.to_le_bytes());
    pc[12..16].copy_from_slice(&h.to_le_bytes());
    pc
}

/// A running loading thread. `finish()` stops it and reclaims the swapchain for the render loop.
pub(crate) struct LoadingThread {
    state: Arc<LoadingState>,
    handle: JoinHandle<Swapchain>,
}

impl LoadingThread {
    /// The shared progress the cook bumps as it advances.
    pub(crate) fn state(&self) -> &Arc<LoadingState> {
        &self.state
    }

    /// Signal the thread to stop, join it, and return the swapchain for the real render loop. The
    /// join blocks until the thread has dropped its pipeline/command-buffer (their `Rc<DeviceShared>`
    /// decrements finish before the main thread resumes `Rc` traffic — the single-owner contract).
    pub(crate) fn finish(self) -> Swapchain {
        self.state.stop.store(true, Ordering::Release);
        self.handle.join().expect("loading thread panicked")
    }
}

/// Spawn the loading thread (interactive D3D12 only). On any other backend / headless, hands the
/// swapchain straight back as `Err` so the caller keeps it and uses the terminal bar. Moves the
/// swapchain + a queue clone + a freshly-built loading pipeline + sync into the thread; it solely
/// owns them until `finish()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn(
    device: &Device,
    backend: BackendKind,
    swapchain: Swapchain,
    swap_format: Format,
    extent: Extent2D,
    headless: bool,
) -> anyhow::Result<Result<LoadingThread, Swapchain>> {
    if headless || backend != BackendKind::D3d12 {
        return Ok(Err(swapchain));
    }
    let pipeline = build_pipeline(device, swap_format)?;
    let queue = device.queue();
    let cmd = device.create_command_buffer()?;
    let image_available = device.create_semaphore()?;
    let render_finished = device.create_semaphore()?;
    let fence = device.create_fence(true)?;
    let state = Arc::new(LoadingState::new());

    let thread_state = Arc::clone(&state);
    let handle = std::thread::Builder::new()
        .name("loading".into())
        .spawn(move || {
            run(
                swapchain,
                queue,
                pipeline,
                cmd,
                image_available,
                render_finished,
                fence,
                extent,
                thread_state,
            )
        })?;
    Ok(Ok(LoadingThread { state, handle }))
}

#[allow(clippy::too_many_arguments)]
fn run(
    swapchain: Swapchain,
    queue: Queue,
    pipeline: GraphicsPipeline,
    cmd: CommandBuffer,
    image_available: Semaphore,
    render_finished: Semaphore,
    fence: Fence,
    extent: Extent2D,
    state: Arc<LoadingState>,
) -> Swapchain {
    let start = Instant::now();
    let (w, h) = (extent.width, extent.height);
    let mut warned = false;
    while !state.stop.load(Ordering::Acquire) {
        if let Err(e) = present_frame(
            &swapchain,
            &queue,
            &pipeline,
            &cmd,
            &image_available,
            &render_finished,
            &fence,
            w,
            h,
            &state,
            start,
        ) && !warned
        {
            warned = true;
            tracing::warn!("loading frame failed ({e}); cook continues without the bar");
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    let _ = fence.wait(); // drain the last in-flight frame before handing the swapchain back
    swapchain
}

#[allow(clippy::too_many_arguments)]
fn present_frame(
    swapchain: &Swapchain,
    queue: &Queue,
    pipeline: &GraphicsPipeline,
    cmd: &CommandBuffer,
    image_available: &Semaphore,
    render_finished: &Semaphore,
    fence: &Fence,
    w: u32,
    h: u32,
    state: &LoadingState,
    start: Instant,
) -> anyhow::Result<()> {
    fence.wait()?;
    let image_index = match swapchain.acquire_next_image(image_available)? {
        Some(i) => i,
        None => return Ok(()),
    };
    fence.reset()?;

    let frac = state.frac.load(Ordering::Relaxed) as f32 / 1000.0;
    let pc = push_bytes(frac, start.elapsed().as_secs_f32(), w, h);

    cmd.begin()?;
    cmd.transition_to_render_target(swapchain, image_index);
    cmd.begin_rendering(
        swapchain,
        image_index,
        Some(ClearColor {
            r: 0.055,
            g: 0.065,
            b: 0.085,
            a: 1.0,
        }),
        None,
    );
    cmd.set_viewport_scissor(swapchain);
    cmd.bind_graphics_pipeline(pipeline);
    cmd.push_constants(&pc);
    cmd.draw(3, 1);
    cmd.end_rendering();
    cmd.transition_to_present(swapchain, image_index);
    cmd.end()?;
    queue.submit(cmd, image_available, render_finished, fence)?;
    queue.present(swapchain, image_index, render_finished)?;
    Ok(())
}
