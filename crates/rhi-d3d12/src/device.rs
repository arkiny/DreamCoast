//! Logical device, command queue, and resource creation.

use std::cell::Cell;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{BufferDesc, Format, GraphicsPipelineDesc, SwapchainDesc, TextureDesc};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_FENCE_FLAG_NONE, D3D12_GPU_DESCRIPTOR_HANDLE,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_TEX2D_SRV, D3D12CreateDevice, ID3D12CommandList,
    ID3D12CommandQueue, ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::IDXGIFactory6;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCWSTR};

use crate::command::D3d12CommandBuffer;
use crate::instance::{D3d12Instance, d3d_err};
use crate::pipeline::D3d12GraphicsPipeline;
use crate::swapchain::D3d12Swapchain;
use crate::sync::{D3d12Fence, D3d12Semaphore};
use crate::texture::D3d12Texture;
use crate::{D3d12Buffer, to_dxgi_format};

/// Size of the bindless SRV table.
pub(crate) const BINDLESS_COUNT: u32 = 1024;

/// Device-level objects shared (via `Rc`) by every GPU resource.
pub(crate) struct DeviceShared {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub factory: IDXGIFactory6,
    pub hwnd: HWND,
    // Shader-visible CBV_SRV_UAV heap holding bindless texture SRVs.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_size: u32,
    srv_next: Cell<u32>,
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

            // Shader-visible bindless SRV heap.
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: BINDLESS_COUNT,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&heap_desc).map_err(d3d_err)?;
            let srv_size =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            tracing::debug!("D3D12 device + queue ready");

            Ok(Self {
                device,
                queue,
                factory: instance.shared.factory.clone(),
                hwnd: instance.shared.hwnd,
                srv_heap,
                srv_size,
                srv_next: Cell::new(0),
                idle_fence,
                idle_event,
                idle_value: Cell::new(0),
            })
        }
    }

    /// GPU handle to the start of the bindless SRV heap (for the root table).
    pub(crate) fn srv_gpu_start(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        unsafe { self.srv_heap.GetGPUDescriptorHandleForHeapStart() }
    }

    /// Create an SRV for `resource` at the next bindless slot; returns its index.
    pub(crate) fn register_texture(&self, resource: &ID3D12Resource, format: Format) -> u32 {
        let index = self.srv_next.get();
        self.srv_next.set(index + 1);
        let cpu = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (index as usize) * (self.srv_size as usize),
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&srv), handle)
        };
        index
    }

    /// Record + submit a one-time command list and wait for completion.
    pub(crate) fn immediate_submit(
        &self,
        record: impl FnOnce(&windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList),
    ) -> Result<(), EngineError> {
        use windows::Win32::Graphics::Direct3D12::{
            ID3D12CommandAllocator, ID3D12GraphicsCommandList,
        };
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(d3d_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(d3d_err)?;
            record(&list);
            list.Close().map_err(d3d_err)?;
            let cl: ID3D12CommandList = list.cast().map_err(d3d_err)?;
            self.queue.ExecuteCommandLists(&[Some(cl)]);
            self.wait_idle()?;
            Ok(())
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

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<D3d12Buffer, EngineError> {
        D3d12Buffer::new(self.shared.clone(), desc)
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        pixels: &[u8],
    ) -> Result<D3d12Texture, EngineError> {
        D3d12Texture::new(self.shared.clone(), desc, pixels)
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
