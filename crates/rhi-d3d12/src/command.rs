//! Command allocator + graphics command list recording for the triangle frame.

use std::mem::ManuallyDrop;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::ClearColor;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_PRESENT,
    D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_VIEWPORT, ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::pipeline::D3d12GraphicsPipeline;
use crate::swapchain::D3d12Swapchain;

/// A primary command list (+ its allocator), reset and re-recorded each frame.
pub struct D3d12CommandBuffer {
    #[allow(dead_code)] // keeps the device alive
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

    pub fn begin_rendering(&self, swapchain: &D3d12Swapchain, image_index: u32, clear: ClearColor) {
        let rtv = swapchain.rtv_handle(image_index);
        let color = [clear.r, clear.g, clear.b, clear.a];
        unsafe {
            self.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            self.list.ClearRenderTargetView(rtv, &color, None);
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
            self.list
                .SetGraphicsRootSignature(pipeline.root_signature());
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
