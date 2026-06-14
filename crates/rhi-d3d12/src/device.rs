//! Logical device, command queue, and resource creation.

use std::cell::Cell;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{GraphicsPipelineDesc, SwapchainDesc};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_FENCE_FLAG_NONE, D3D12CreateDevice, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device,
    ID3D12Fence,
};
use windows::Win32::Graphics::Dxgi::IDXGIFactory6;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCWSTR};

use crate::command::D3d12CommandBuffer;
use crate::instance::{D3d12Instance, d3d_err};
use crate::pipeline::D3d12GraphicsPipeline;
use crate::swapchain::D3d12Swapchain;
use crate::sync::{D3d12Fence, D3d12Semaphore};

/// Device-level objects shared (via `Rc`) by every GPU resource.
pub(crate) struct DeviceShared {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub factory: IDXGIFactory6,
    pub hwnd: HWND,
    // A dedicated fence for `wait_idle`.
    idle_fence: ID3D12Fence,
    idle_event: HANDLE,
    idle_value: Cell<u64>,
}

impl DeviceShared {
    pub(crate) fn new(instance: &D3d12Instance) -> Result<Self, EngineError> {
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(
                &instance.shared.adapter,
                D3D_FEATURE_LEVEL_12_0,
                &mut device,
            )
            .map_err(d3d_err)?;
            let device =
                device.ok_or_else(|| EngineError::Rhi("CreateDevice returned null".into()))?;

            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&queue_desc).map_err(d3d_err)?;

            let idle_fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(d3d_err)?;
            let idle_event = CreateEventW(None, false, false, PCWSTR::null()).map_err(d3d_err)?;
            tracing::debug!("D3D12 device + queue ready");

            Ok(Self {
                device,
                queue,
                factory: instance.shared.factory.clone(),
                hwnd: instance.shared.hwnd,
                idle_fence,
                idle_event,
                idle_value: Cell::new(0),
            })
        }
    }

    /// Block until the GPU has finished all previously submitted work.
    pub(crate) fn wait_idle(&self) -> Result<(), EngineError> {
        unsafe {
            let value = self.idle_value.get() + 1;
            self.idle_value.set(value);
            self.queue
                .Signal(&self.idle_fence, value)
                .map_err(d3d_err)?;
            if self.idle_fence.GetCompletedValue() < value {
                self.idle_fence
                    .SetEventOnCompletion(value, self.idle_event)
                    .map_err(d3d_err)?;
                WaitForSingleObject(self.idle_event, INFINITE);
            }
            Ok(())
        }
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        // Ensure the GPU is idle before COM objects release.
        let _ = self.wait_idle();
        unsafe {
            let _ = CloseHandle(self.idle_event);
        }
    }
}

/// A logical D3D12 device: the factory for swapchains, pipelines, command
/// buffers, and synchronization primitives.
pub struct D3d12Device {
    pub(crate) shared: Rc<DeviceShared>,
}

impl D3d12Device {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<D3d12Swapchain, EngineError> {
        D3d12Swapchain::new(self.shared.clone(), desc)
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<D3d12GraphicsPipeline, EngineError> {
        D3d12GraphicsPipeline::new(self.shared.clone(), desc)
    }

    pub fn create_command_buffer(&self) -> Result<D3d12CommandBuffer, EngineError> {
        D3d12CommandBuffer::new(self.shared.clone())
    }

    pub fn create_fence(&self, signaled: bool) -> Result<D3d12Fence, EngineError> {
        D3d12Fence::new(self.shared.clone(), signaled)
    }

    pub fn create_semaphore(&self) -> Result<D3d12Semaphore, EngineError> {
        Ok(D3d12Semaphore::new())
    }

    pub fn queue(&self) -> D3d12Queue {
        D3d12Queue {
            shared: self.shared.clone(),
        }
    }

    pub fn wait_idle(&self) -> Result<(), EngineError> {
        self.shared.wait_idle()
    }
}

/// The device's DIRECT queue.
pub struct D3d12Queue {
    pub(crate) shared: Rc<DeviceShared>,
}

impl D3d12Queue {
    /// Execute a command list, then signal `fence` (semaphores are ignored on
    /// D3D12 — see crate docs).
    pub fn submit(
        &self,
        cmd: &D3d12CommandBuffer,
        _wait: &D3d12Semaphore,
        _signal: &D3d12Semaphore,
        fence: &D3d12Fence,
    ) -> Result<(), EngineError> {
        unsafe {
            let list: ID3D12CommandList = cmd.list().cast().map_err(d3d_err)?;
            self.shared.queue.ExecuteCommandLists(&[Some(list)]);
            let value = fence.next_value();
            self.shared
                .queue
                .Signal(fence.raw(), value)
                .map_err(d3d_err)?;
            fence.set_target(value);
            Ok(())
        }
    }

    /// Present the swapchain (vsync). Returns `true` if it should be recreated.
    pub fn present(
        &self,
        swapchain: &D3d12Swapchain,
        _image_index: u32,
        _wait: &D3d12Semaphore,
    ) -> Result<bool, EngineError> {
        swapchain.present()
    }
}
