//! Dear ImGui integration: an imgui-rs context plus a renderer that records
//! ImGui draw data through the engine RHI (bindless font texture, dynamic
//! per-frame vertex/index buffers) and a Win32 input bridge.
//!
//! The `imgui` crate is re-exported so consumers build UI with the same version.

pub use imgui;

use dreamcoast_core::EngineError;
use dreamcoast_platform::Input;
use imgui::{Context, DrawCmd, DrawData, DrawIdx, DrawVert};
use rhi::{
    BackendKind, BlendMode, Buffer, BufferDesc, BufferUsage, DepthCompare, Device, Format,
    GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, Recorder, Rect2D, Texture,
    TextureDesc, VertexLayout,
};

/// Per-frame-in-flight dynamic geometry buffers (grown as needed).
#[derive(Default)]
struct FrameBuffers {
    vtx: Option<Buffer>,
    vtx_cap: u64,
    idx: Option<Buffer>,
    idx_cap: u64,
}

/// The ImGui context + its RHI renderer.
pub struct Gui {
    ctx: Context,
    pipeline: GraphicsPipeline,
    _font: Texture,
    /// Occupies bindless slot 0 so the font (and any later texture) never gets imgui's NULL id 0.
    _slot0_guard: Texture,
    backend: BackendKind,
    frames: Vec<FrameBuffers>,
}

impl Gui {
    /// Create the context, upload the font atlas as a bindless texture, and
    /// build the ImGui pipeline. `frames_in_flight` matches the renderer's.
    pub fn new(
        device: &Device,
        color_format: Format,
        frames_in_flight: usize,
    ) -> Result<Self, EngineError> {
        let mut ctx = Context::create();
        ctx.set_ini_filename(None);

        // imgui reserves texture id 0 as the NULL texture (`ImTextureID` 0 = "no texture"). Our
        // bindless allocator hands out slot 0 as a normal slot, so if the font atlas lands there the
        // imgui shader/`NewFrame` treats the font as absent → every frame renders nothing and
        // `igGetDrawData` returns invalid data (a crash in `DrawData::draw_lists`). This surfaced
        // only once `Gui::new` moved before the scene textures (the font became the first texture).
        // Reserve slot 0 with a 1×1 throwaway (kept alive for the Gui's lifetime) so the font — and
        // any later texture — always gets a non-zero id. Costs one of 1024 bindless slots.
        let slot0_guard = device.create_texture(
            &TextureDesc {
                width: 1,
                height: 1,
                format: Format::Rgba8Unorm,
            },
            &[255, 255, 255, 255],
        )?;

        let font = {
            let fonts = ctx.fonts();
            let tex = fonts.build_rgba32_texture();
            let texture = device.create_texture(
                &TextureDesc {
                    width: tex.width,
                    height: tex.height,
                    format: Format::Rgba8Unorm,
                },
                tex.data,
            )?;
            debug_assert!(texture.bindless_index() != 0, "font must not be imgui NULL id 0");
            fonts.tex_id = imgui::TextureId::from(texture.bindless_index() as usize);
            texture
        };

        let backend = device.backend();
        let (vs, fs) = match backend {
            BackendKind::Vulkan => (
                dreamcoast_shader::imgui_vs_spirv(),
                dreamcoast_shader::imgui_fs_spirv(),
            ),
            BackendKind::D3d12 => (
                dreamcoast_shader::imgui_vs_dxil(),
                dreamcoast_shader::imgui_fs_dxil(),
            ),
            BackendKind::Metal => (
                dreamcoast_shader::imgui_vs_metallib(),
                dreamcoast_shader::imgui_fs_metallib(),
            ),
        };
        let vs = vs.ok_or_else(|| EngineError::Shader("imgui vertex shader unavailable".into()))?;
        let fs =
            fs.ok_or_else(|| EngineError::Shader("imgui fragment shader unavailable".into()))?;

        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vs,
            fragment_bytes: fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[color_format],
            topology: PrimitiveTopology::TriangleList,
            vertex_layout: VertexLayout::ImGui,
            blend: BlendMode::AlphaBlend,
            push_constant_size: 20,
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;

        let frames = (0..frames_in_flight)
            .map(|_| FrameBuffers::default())
            .collect();

        Ok(Self {
            ctx,
            pipeline,
            _font: font,
            _slot0_guard: slot0_guard,
            backend,
            frames,
        })
    }

    /// Feed input + timing and begin a new UI frame. Build windows on the
    /// returned `Ui`, then call [`render`](Self::render).
    pub fn new_frame(&mut self, dt: f32, display_size: [f32; 2], input: &Input) -> &mut imgui::Ui {
        let io = self.ctx.io_mut();
        io.display_size = display_size;
        io.delta_time = dt.max(1.0e-4);
        let (mx, my) = input.mouse_position();
        io.mouse_pos = [mx as f32, my as f32];
        io.mouse_down = [
            input.mouse_button(0),
            input.mouse_button(1),
            input.mouse_button(2),
            false,
            false,
        ];
        io.mouse_wheel += input.wheel_delta();
        for &ch in input.chars() {
            io.add_input_character(ch);
        }
        self.ctx.new_frame()
    }

    /// Render the current frame's UI into `cmd`. Must be called inside an active
    /// render pass on `cmd`. `frame_index` selects the in-flight buffer set.
    pub fn render(
        &mut self,
        device: &Device,
        cmd: &dyn Recorder,
        frame_index: usize,
    ) -> Result<(), EngineError> {
        let draw_data: &DrawData = self.ctx.render();
        // imgui's FIRST frame after context init (and any frame with no visible UI) produces no
        // geometry; its `DrawData` has no draw lists and `draw_lists()` would read a garbage
        // `cmd_lists_count`. Skip it — there is nothing to draw. (The main render loop never hits an
        // empty frame, so this only matters for the pre-render-loop cook loading screen.)
        if draw_data.total_vtx_count == 0 {
            return Ok(());
        }
        let [disp_w, disp_h] = draw_data.display_size;
        if disp_w <= 0.0 || disp_h <= 0.0 {
            return Ok(());
        }
        let disp_pos = draw_data.display_pos;
        let fb_scale = draw_data.framebuffer_scale;

        let vsize = std::mem::size_of::<DrawVert>();
        let isize_ = std::mem::size_of::<DrawIdx>();

        // Concatenate every draw list into one vertex + one index blob, tracking
        // per-command offsets and parameters.
        let mut verts: Vec<u8> = Vec::new();
        let mut indices: Vec<u8> = Vec::new();
        let mut items: Vec<DrawItem> = Vec::new();

        for list in draw_data.draw_lists() {
            let vtx = list.vtx_buffer();
            let idx = list.idx_buffer();
            let base_vertex = (verts.len() / vsize) as i32;
            let base_index = (indices.len() / isize_) as u32;

            let vtx_bytes = unsafe {
                std::slice::from_raw_parts(vtx.as_ptr() as *const u8, std::mem::size_of_val(vtx))
            };
            verts.extend_from_slice(vtx_bytes);
            let idx_bytes = unsafe {
                std::slice::from_raw_parts(idx.as_ptr() as *const u8, std::mem::size_of_val(idx))
            };
            indices.extend_from_slice(idx_bytes);

            for command in list.commands() {
                if let DrawCmd::Elements { count, cmd_params } = command {
                    items.push(DrawItem {
                        idx_count: count as u32,
                        first_index: base_index + cmd_params.idx_offset as u32,
                        vertex_offset: base_vertex + cmd_params.vtx_offset as i32,
                        clip: cmd_params.clip_rect,
                        tex: cmd_params.texture_id.id() as u32,
                    });
                }
            }
        }

        if items.is_empty() {
            return Ok(());
        }

        // Upload geometry into this frame's buffers (growing if needed).
        let fb = &mut self.frames[frame_index];
        ensure_buffer(
            device,
            &mut fb.vtx,
            &mut fb.vtx_cap,
            verts.len() as u64,
            BufferUsage::Vertex,
        )?;
        ensure_buffer(
            device,
            &mut fb.idx,
            &mut fb.idx_cap,
            indices.len() as u64,
            BufferUsage::Index,
        )?;
        let vtx_buf = fb.vtx.as_ref().unwrap();
        let idx_buf = fb.idx.as_ref().unwrap();
        vtx_buf.write(&verts)?;
        idx_buf.write(&indices)?;

        // Orthographic scale/translate (y-flip differs per backend).
        let sx: f32 = 2.0 / disp_w;
        let tx: f32 = -1.0;
        let (sy, ty): (f32, f32) = match self.backend {
            BackendKind::Vulkan => (2.0 / disp_h, -1.0),
            // Metal shares D3D12's clip-space convention (top-left origin).
            BackendKind::D3d12 | BackendKind::Metal => (-2.0 / disp_h, 1.0),
        };

        cmd.bind_graphics_pipeline(&self.pipeline);
        cmd.bind_vertex_buffer(vtx_buf, vsize as u32);
        cmd.bind_index_buffer(idx_buf, isize_ == 4);

        for item in &items {
            let mut pc = [0u8; 20];
            pc[0..4].copy_from_slice(&sx.to_le_bytes());
            pc[4..8].copy_from_slice(&sy.to_le_bytes());
            pc[8..12].copy_from_slice(&tx.to_le_bytes());
            pc[12..16].copy_from_slice(&ty.to_le_bytes());
            pc[16..20].copy_from_slice(&item.tex.to_le_bytes());
            cmd.push_constants(&pc);

            // Clip rect (display coords) -> framebuffer-pixel scissor.
            let x0 = ((item.clip[0] - disp_pos[0]) * fb_scale[0]).max(0.0);
            let y0 = ((item.clip[1] - disp_pos[1]) * fb_scale[1]).max(0.0);
            let x1 = ((item.clip[2] - disp_pos[0]) * fb_scale[0]).max(0.0);
            let y1 = ((item.clip[3] - disp_pos[1]) * fb_scale[1]).max(0.0);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            cmd.set_scissor(Rect2D {
                x: x0 as i32,
                y: y0 as i32,
                width: (x1 - x0) as u32,
                height: (y1 - y0) as u32,
            });
            cmd.draw_indexed(item.idx_count, item.first_index, item.vertex_offset);
        }

        Ok(())
    }
}

struct DrawItem {
    idx_count: u32,
    first_index: u32,
    vertex_offset: i32,
    clip: [f32; 4],
    tex: u32,
}

fn ensure_buffer(
    device: &Device,
    slot: &mut Option<Buffer>,
    cap: &mut u64,
    needed: u64,
    usage: BufferUsage,
) -> Result<(), EngineError> {
    if needed == 0 {
        return Ok(());
    }
    if slot.is_none() || *cap < needed {
        let size = needed.next_power_of_two().max(256);
        *slot = Some(device.create_buffer(&BufferDesc { size, usage })?);
        *cap = size;
    }
    Ok(())
}
