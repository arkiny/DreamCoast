//! Command allocator + graphics command list recording for the triangle frame.

use std::cell::Cell;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{ClearColor, Extent2D, Rect2D};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CLEAR_FLAG_DEPTH, D3D12_COMMAND_LIST_TYPE, D3D12_COMMAND_LIST_TYPE_COMPUTE,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_INDEX_BUFFER_VIEW, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RESOURCE_ALIASING_BARRIER, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_ALIASING, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_VERTEX_BUFFER_VIEW, D3D12_VIEWPORT, ID3D12CommandAllocator, ID3D12GraphicsCommandList,
    ID3D12GraphicsCommandList4, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};
use windows::core::Interface;

use crate::buffer::{D3d12Buffer, D3d12StorageBuffer};
use crate::cubemap::D3d12Cubemap;
use crate::depth::D3d12DepthBuffer;
use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::pipeline::{D3d12ComputePipeline, D3d12GraphicsPipeline};
use crate::render_target::D3d12RenderTarget;
use crate::swapchain::D3d12Swapchain;

/// A primary command list (+ its allocator), reset and re-recorded each frame.
pub struct D3d12CommandBuffer {
    device: Rc<DeviceShared>,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    // GPU VA of the per-frame globals slice for the next PBR pipeline bind.
    globals_va: Cell<u64>,
}

impl D3d12CommandBuffer {
    pub(crate) fn new(device: Rc<DeviceShared>) -> Result<Self, EngineError> {
        Self::with_type(device, D3D12_COMMAND_LIST_TYPE_DIRECT)
    }

    /// Allocate a COMPUTE-type command list for the async-compute queue (Phase 7).
    pub(crate) fn new_compute(device: Rc<DeviceShared>) -> Result<Self, EngineError> {
        Self::with_type(device, D3D12_COMMAND_LIST_TYPE_COMPUTE)
    }

    fn with_type(
        device: Rc<DeviceShared>,
        list_type: D3D12_COMMAND_LIST_TYPE,
    ) -> Result<Self, EngineError> {
        unsafe {
            let allocator: ID3D12CommandAllocator = device
                .device
                .CreateCommandAllocator(list_type)
                .map_err(d3d_err)?;
            let list: ID3D12GraphicsCommandList = device
                .device
                .CreateCommandList(0, list_type, &allocator, None)
                .map_err(d3d_err)?;
            // Created open; close so `begin` can reset it uniformly.
            list.Close().map_err(d3d_err)?;
            Ok(Self {
                device,
                allocator,
                list,
                globals_va: Cell::new(0),
            })
        }
    }

    pub(crate) fn list(&self) -> &ID3D12GraphicsCommandList {
        &self.list
    }

    pub fn begin(&self) -> Result<(), EngineError> {
        unsafe {
            self.allocator.Reset().map_err(d3d_err)?;
            self.list.Reset(&self.allocator, None).map_err(d3d_err)?;
            Ok(())
        }
    }

    pub fn end(&self) -> Result<(), EngineError> {
        unsafe { self.list.Close().map_err(d3d_err) }
    }

    pub fn transition_to_render_target(&self, swapchain: &D3d12Swapchain, image_index: u32) {
        self.barrier(
            swapchain.buffer(image_index),
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    pub fn transition_to_present(&self, swapchain: &D3d12Swapchain, image_index: u32) {
        self.barrier(
            swapchain.buffer(image_index),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
    }

    /// Begin a render pass. `color_clear = Some` clears color, `None` loads it
    /// (overlay pass). `depth = Some` binds + clears the depth target.
    pub fn begin_rendering(
        &self,
        swapchain: &D3d12Swapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&D3d12DepthBuffer>,
    ) {
        let rtv = swapchain.rtv_handle(image_index);
        unsafe {
            match depth {
                Some(d) => {
                    let dsv = d.dsv();
                    self.list
                        .OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                    self.list
                        .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                }
                None => self.list.OMSetRenderTargets(1, Some(&rtv), false, None),
            }
            if let Some(c) = color_clear {
                self.list
                    .ClearRenderTargetView(rtv, &[c.r, c.g, c.b, c.a], None);
            }
        }
    }

    /// Begin a render pass into an offscreen color target (+ optional depth). The
    /// target must already be in `RENDER_TARGET` state (see
    /// [`Self::rt_to_render_target`]).
    pub fn begin_rendering_target(
        &self,
        target: &D3d12RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&D3d12DepthBuffer>,
    ) {
        let rtv = target.rtv_handle();
        unsafe {
            match depth {
                Some(d) => {
                    let dsv = d.dsv();
                    self.list
                        .OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                    self.list
                        .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                }
                None => self.list.OMSetRenderTargets(1, Some(&rtv), false, None),
            }
            if let Some(c) = color_clear {
                self.list
                    .ClearRenderTargetView(rtv, &[c.r, c.g, c.b, c.a], None);
            }
        }
    }

    /// Begin a render pass into N offscreen color targets (MRT) plus optional
    /// depth. Each target's `Some(clear)` clears it, `None` loads. All targets
    /// must already be in `RENDER_TARGET` state. `targets` must be non-empty.
    pub fn begin_rendering_targets(
        &self,
        targets: &[(&D3d12RenderTarget, Option<ClearColor>)],
        depth: Option<&D3d12DepthBuffer>,
    ) {
        let rtvs: Vec<_> = targets.iter().map(|(t, _)| t.rtv_handle()).collect();
        unsafe {
            match depth {
                Some(d) => {
                    let dsv = d.dsv();
                    self.list.OMSetRenderTargets(
                        rtvs.len() as u32,
                        Some(rtvs.as_ptr()),
                        false,
                        Some(&dsv),
                    );
                    self.list
                        .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                }
                None => self.list.OMSetRenderTargets(
                    rtvs.len() as u32,
                    Some(rtvs.as_ptr()),
                    false,
                    None,
                ),
            }
            for ((_, clear), rtv) in targets.iter().zip(&rtvs) {
                if let Some(c) = clear {
                    self.list
                        .ClearRenderTargetView(*rtv, &[c.r, c.g, c.b, c.a], None);
                }
            }
        }
    }

    /// Begin a render pass into a depth-only target (a shadow map): no color
    /// targets, depth is cleared. The depth must already be in `DEPTH_WRITE` (see
    /// [`Self::depth_to_render_target`]).
    pub fn begin_rendering_depth_only(&self, depth: &D3d12DepthBuffer) {
        let dsv = depth.dsv();
        unsafe {
            self.list.OMSetRenderTargets(0, None, false, Some(&dsv));
            self.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
        }
    }

    /// Transition a depth buffer into `DEPTH_WRITE` for writing (a shadow map
    /// reused across frames may currently be in shader-read).
    pub fn depth_to_render_target(&self, depth: &D3d12DepthBuffer) {
        let before = depth.state();
        if before == D3D12_RESOURCE_STATE_DEPTH_WRITE {
            return;
        }
        self.barrier(depth.resource(), before, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        depth.set_state(D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }

    /// Transition a whole cubemap into `RENDER_TARGET` for writing its faces/mips
    /// (the IBL generation passes).
    pub fn cube_to_color(&self, cube: &D3d12Cubemap) {
        let before = cube.state();
        if before == D3D12_RESOURCE_STATE_RENDER_TARGET {
            return;
        }
        self.barrier(cube.resource(), before, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cube.set_state(D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    /// Transition a whole cubemap into `PIXEL_SHADER_RESOURCE` for sampling.
    pub fn cube_to_sampled(&self, cube: &D3d12Cubemap) {
        let before = cube.state();
        if before == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
            return;
        }
        self.barrier(
            cube.resource(),
            before,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        cube.set_state(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    /// Begin a render pass into one (face, mip) of a cubemap. The cubemap must
    /// already be in `RENDER_TARGET` (see [`Self::cube_to_color`]).
    pub fn begin_rendering_cube_face(
        &self,
        cube: &D3d12Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        let rtv = cube.rtv_handle(face, mip);
        unsafe {
            self.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            if let Some(c) = clear {
                self.list
                    .ClearRenderTargetView(rtv, &[c.r, c.g, c.b, c.a], None);
            }
        }
    }

    /// Begin rendering into one (face, mip) of a cubemap **with a depth buffer**
    /// (clears depth), for capturing scene geometry. Color is loaded if
    /// `clear = None`. Cube must be in `RENDER_TARGET`, depth in `DEPTH_WRITE`.
    pub fn begin_rendering_cube_face_depth(
        &self,
        cube: &D3d12Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &D3d12DepthBuffer,
    ) {
        let rtv = cube.rtv_handle(face, mip);
        let dsv = depth.dsv();
        unsafe {
            self.list
                .OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            self.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            if let Some(c) = clear {
                self.list
                    .ClearRenderTargetView(rtv, &[c.r, c.g, c.b, c.a], None);
            }
        }
    }

    /// Transition a depth buffer into `PIXEL_SHADER_RESOURCE` for sampling.
    pub fn depth_to_sampled(&self, depth: &D3d12DepthBuffer) {
        let before = depth.state();
        if before == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
            return;
        }
        self.barrier(
            depth.resource(),
            before,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        depth.set_state(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    /// Transition an offscreen target into `RENDER_TARGET` for writing.
    pub fn rt_to_render_target(&self, target: &D3d12RenderTarget) {
        let before = target.state();
        if before == D3D12_RESOURCE_STATE_RENDER_TARGET {
            return;
        }
        self.barrier(
            target.resource(),
            before,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        target.set_state(D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    /// Prepare an aliased target for writing into shared heap memory: an aliasing
    /// barrier that discards the previous tenant, then a transition to
    /// `RENDER_TARGET` (the full-screen draw that follows reinitializes it).
    pub fn aliasing_barrier(&self, target: &D3d12RenderTarget) {
        let aliasing = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: ManuallyDrop::new(None),
                    pResourceAfter: unsafe { std::mem::transmute_copy(target.resource()) },
                }),
            },
        };
        unsafe { self.list.ResourceBarrier(&[aliasing]) };
        let before = target.state();
        if before != D3D12_RESOURCE_STATE_RENDER_TARGET {
            self.barrier(
                target.resource(),
                before,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            target.set_state(D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
    }

    /// Transition an offscreen target into `PIXEL_SHADER_RESOURCE` for sampling.
    pub fn rt_to_sampled(&self, target: &D3d12RenderTarget) {
        let before = target.state();
        if before == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
            return;
        }
        self.barrier(
            target.resource(),
            before,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        target.set_state(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    /// No-op on D3D12 (render targets are set directly; no render pass object).
    pub fn end_rendering(&self) {}

    pub fn set_viewport_scissor(&self, swapchain: &D3d12Swapchain) {
        let extent = swapchain.extent();
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: extent.width as f32,
            Height: extent.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: extent.width as i32,
            bottom: extent.height as i32,
        };
        unsafe {
            self.list.RSSetViewports(&[viewport]);
            self.list.RSSetScissorRects(&[scissor]);
        }
    }

    /// Set viewport and scissor to cover an arbitrary extent (offscreen target).
    pub fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: extent.width as f32,
            Height: extent.height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: extent.width as i32,
            bottom: extent.height as i32,
        };
        unsafe {
            self.list.RSSetViewports(&[viewport]);
            self.list.RSSetScissorRects(&[scissor]);
        }
    }

    /// Copy a rendered swapchain image into a host-readable readback buffer (for
    /// screenshots). The image must be in `PRESENT` (the state the render graph
    /// leaves it in); it is restored to that state afterward. The buffer receives
    /// rows padded to `GetCopyableFootprints`' 256-byte pitch.
    pub fn copy_swapchain_to_buffer(
        &self,
        swapchain: &D3d12Swapchain,
        image_index: u32,
        buffer: &D3d12Buffer,
    ) {
        let src_res = swapchain.buffer(image_index);
        unsafe {
            let desc = src_res.GetDesc();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            self.device.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total),
            );

            self.barrier(
                src_res,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );

            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(buffer.resource()))),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(src_res))),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);

            self.barrier(
                src_res,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        }
    }

    /// Select the per-frame globals slice (GPU VA) used by the next PBR pipeline
    /// bind (root CBV at param 2).
    pub fn set_globals(&self, va: u64) {
        self.globals_va.set(va);
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &D3d12GraphicsPipeline) {
        unsafe {
            if pipeline.is_bindless() {
                let heaps = [Some(self.device.srv_heap.clone())];
                self.list.SetDescriptorHeaps(&heaps);
            }
            self.list
                .SetGraphicsRootSignature(pipeline.root_signature());
            if pipeline.is_bindless() {
                self.list
                    .SetGraphicsRootDescriptorTable(0, self.device.srv_gpu_start());
            }
            if pipeline.uses_uniform() {
                self.list
                    .SetGraphicsRootConstantBufferView(2, self.globals_va.get());
            }
            self.list.SetPipelineState(pipeline.pso());
            self.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        unsafe {
            self.list.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
    }

    /// Transition a storage render target into `UNORDERED_ACCESS` for compute.
    pub fn rt_to_storage(&self, target: &D3d12RenderTarget) {
        let before = target.state();
        if before == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            return;
        }
        self.barrier(
            target.resource(),
            before,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        target.set_state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    /// Transition a storage image from `UNORDERED_ACCESS` into
    /// `PIXEL_SHADER_RESOURCE` for sampling by a later graphics pass.
    pub fn storage_to_sampled(&self, target: &D3d12RenderTarget) {
        self.rt_to_sampled(target);
    }

    /// UAV barrier on a storage buffer: order a compute write before later reads.
    pub fn storage_buffer_barrier(&self, buffer: &D3d12StorageBuffer) {
        self.uav_barrier(buffer.resource());
    }

    /// Transition a storage buffer from `UNORDERED_ACCESS` (compute write) into
    /// `INDIRECT_ARGUMENT` for `draw_indexed_indirect`.
    pub fn storage_buffer_to_indirect(&self, buffer: &D3d12StorageBuffer) {
        let before = buffer.state();
        if before == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
            return;
        }
        self.barrier(
            buffer.resource(),
            before,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        );
        buffer.set_state(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    /// Transition a storage buffer back into `UNORDERED_ACCESS` (e.g. before the
    /// next frame's compute write).
    pub fn storage_buffer_to_storage(&self, buffer: &D3d12StorageBuffer) {
        let before = buffer.state();
        if before == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            return;
        }
        self.barrier(
            buffer.resource(),
            before,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        buffer.set_state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    fn uav_barrier(&self, resource: &ID3D12Resource) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                }),
            },
        };
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// Bind a compute pipeline (root signature + bindless table + PSO).
    pub fn bind_compute_pipeline(&self, pipeline: &D3d12ComputePipeline) {
        unsafe {
            if pipeline.is_bindless() {
                let heaps = [Some(self.device.srv_heap.clone())];
                self.list.SetDescriptorHeaps(&heaps);
            }
            self.list.SetComputeRootSignature(pipeline.root_signature());
            if pipeline.is_bindless() {
                self.list
                    .SetComputeRootDescriptorTable(0, self.device.srv_gpu_start());
            }
            self.list.SetPipelineState(pipeline.pso());
        }
    }

    /// Dispatch the bound compute pipeline over `(x, y, z)` thread groups.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe { self.list.Dispatch(x, y, z) };
    }

    /// Upload root (push) constants for the bound compute pipeline (param 1).
    pub fn push_constants_compute(&self, data: &[u8]) {
        unsafe {
            self.list.SetComputeRoot32BitConstants(
                1,
                (data.len() / 4) as u32,
                data.as_ptr() as *const c_void,
                0,
            );
        }
    }

    /// Bind a ray-tracing state object + its global root signature, the shared
    /// bindless heap/table, and the state object (Phase 8 M5). The RT pipeline
    /// binds through the compute root-signature slots (`DispatchRays` reads them).
    pub fn bind_raytracing_pipeline(&self, pipeline: &crate::rt_pipeline::D3d12RaytracingPipeline) {
        unsafe {
            let heaps = [Some(self.device.srv_heap.clone())];
            self.list.SetDescriptorHeaps(&heaps);
            self.list.SetComputeRootSignature(pipeline.root_signature());
            self.list
                .SetComputeRootDescriptorTable(0, self.device.srv_gpu_start());
            let list4: ID3D12GraphicsCommandList4 =
                self.list.cast().expect("CommandList4 (DXR available)");
            list4.SetPipelineState1(pipeline.state_object());
        }
    }

    /// Upload root (push) constants for the bound RT pipeline (param 1) — same root
    /// constant slot as compute.
    pub fn push_constants_rt(&self, data: &[u8]) {
        unsafe {
            self.list.SetComputeRoot32BitConstants(
                1,
                (data.len() / 4) as u32,
                data.as_ptr() as *const c_void,
                0,
            );
        }
    }

    /// Trace a `width` x `height` grid of rays through the bound RT pipeline's SBT.
    pub fn trace_rays(
        &self,
        pipeline: &crate::rt_pipeline::D3d12RaytracingPipeline,
        width: u32,
        height: u32,
    ) {
        unsafe {
            let desc = pipeline.dispatch_desc(width, height);
            let list4: ID3D12GraphicsCommandList4 =
                self.list.cast().expect("CommandList4 (DXR available)");
            list4.DispatchRays(&desc);
        }
    }

    /// Set the scissor rectangle.
    pub fn set_scissor(&self, rect: Rect2D) {
        let scissor = RECT {
            left: rect.x,
            top: rect.y,
            right: rect.x + rect.width as i32,
            bottom: rect.y + rect.height as i32,
        };
        unsafe { self.list.RSSetScissorRects(&[scissor]) };
    }

    /// Bind a vertex buffer at slot 0 with the given per-vertex `stride`.
    pub fn bind_vertex_buffer(&self, buffer: &D3d12Buffer, stride: u32) {
        let view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: buffer.gpu_va(),
            SizeInBytes: buffer.size() as u32,
            StrideInBytes: stride,
        };
        unsafe { self.list.IASetVertexBuffers(0, Some(&[view])) };
    }

    /// Bind an index buffer (`wide` selects 32-bit indices, else 16-bit).
    pub fn bind_index_buffer(&self, buffer: &D3d12Buffer, wide: bool) {
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: buffer.gpu_va(),
            SizeInBytes: buffer.size() as u32,
            Format: if wide {
                DXGI_FORMAT_R32_UINT
            } else {
                DXGI_FORMAT_R16_UINT
            },
        };
        unsafe { self.list.IASetIndexBuffer(Some(&view)) };
    }

    /// Upload root (push) constants for the bound bindless pipeline (param 1).
    pub fn push_constants(&self, data: &[u8]) {
        unsafe {
            self.list.SetGraphicsRoot32BitConstants(
                1,
                (data.len() / 4) as u32,
                data.as_ptr() as *const c_void,
                0,
            );
        }
    }

    /// Issue an indexed draw.
    pub fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.list
                .DrawIndexedInstanced(index_count, 1, first_index, vertex_offset, 0);
        }
    }

    /// Issue `draw_count` indexed indirect draws (`ExecuteIndirect`) reading
    /// 20-byte `D3D12_DRAW_INDEXED_ARGUMENTS` records from `buffer` at `offset`.
    /// The buffer must be in `INDIRECT_ARGUMENT` state (see
    /// [`Self::storage_buffer_to_indirect`]).
    pub fn draw_indexed_indirect(&self, buffer: &D3d12StorageBuffer, offset: u64, draw_count: u32) {
        unsafe {
            self.list.ExecuteIndirect(
                &self.device.indirect_draw_signature,
                draw_count,
                buffer.resource(),
                offset,
                None,
                0,
            );
        }
    }

    fn barrier(
        &self,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                // transmute_copy borrows the resource pointer without AddRef; the
                // ManuallyDrop means no Release either, so refcounts stay balanced.
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        unsafe {
            self.list.ResourceBarrier(&[barrier]);
        }
    }
}
