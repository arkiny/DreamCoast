//! Command allocator + graphics command list recording for the triangle frame.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{ClearColor, Rect2D};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CLEAR_FLAG_DEPTH, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_INDEX_BUFFER_VIEW,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATES,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_VERTEX_BUFFER_VIEW, D3D12_VIEWPORT,
    ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};

use crate::buffer::D3d12Buffer;
use crate::depth::D3d12DepthBuffer;
use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::pipeline::D3d12GraphicsPipeline;
use crate::swapchain::D3d12Swapchain;

/// A primary command list (+ its allocator), reset and re-recorded each frame.
pub struct D3d12CommandBuffer {
    device: Rc<DeviceShared>,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

impl D3d12CommandBuffer {
    pub(crate) fn new(device: Rc<DeviceShared>) -> Result<Self, EngineError> {
        unsafe {
            let allocator: ID3D12CommandAllocator = device
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(d3d_err)?;
            let list: ID3D12GraphicsCommandList = device
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(d3d_err)?;
            // Created open; close so `begin` can reset it uniformly.
            list.Close().map_err(d3d_err)?;
            Ok(Self {
                device,
                allocator,
                list,
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
