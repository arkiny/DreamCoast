//! DXGI flip-model swapchain with an RTV descriptor heap.

use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{Extent2D, Format, SwapchainDesc};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_RENDER_TARGET_VIEW_DESC, D3D12_RENDER_TARGET_VIEW_DESC_0,
    D3D12_RTV_DIMENSION_TEXTURE2D, D3D12_TEX2D_RTV, ID3D12DescriptorHeap, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    DXGI_MWA_NO_ALT_ENTER, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGISwapChain1,
    IDXGISwapChain3,
};
use windows::core::Interface;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::{to_dxgi_format, to_dxgi_swapchain_format};

/// A window swapchain plus its render-target views.
pub struct D3d12Swapchain {
    device: Rc<DeviceShared>,
    swapchain: IDXGISwapChain3,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_size: usize,
    buffers: Vec<ID3D12Resource>,
    rtv_handles: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    format: Format,
    rtv_format: DXGI_FORMAT,
    base_format: DXGI_FORMAT,
    extent: Extent2D,
    image_count: u32,
}

impl D3d12Swapchain {
    pub(crate) fn new(device: Rc<DeviceShared>, desc: &SwapchainDesc) -> Result<Self, EngineError> {
        unsafe {
            let image_count = desc.image_count.max(2); // flip model requires >= 2
            let base_format = to_dxgi_swapchain_format(desc.format);
            let rtv_format = to_dxgi_format(desc.format);

            let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: desc.extent.width,
                Height: desc.extent.height,
                Format: base_format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: image_count,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };

            let sc1: IDXGISwapChain1 = device
                .factory
                .CreateSwapChainForHwnd(&device.queue, device.hwnd, &sc_desc, None, None)
                .map_err(d3d_err)?;
            // We drive fullscreen transitions ourselves.
            device
                .factory
                .MakeWindowAssociation(device.hwnd, DXGI_MWA_NO_ALT_ENTER)
                .map_err(d3d_err)?;
            let swapchain: IDXGISwapChain3 = sc1.cast().map_err(d3d_err)?;

            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: image_count,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let rtv_heap: ID3D12DescriptorHeap = device
                .device
                .CreateDescriptorHeap(&heap_desc)
                .map_err(d3d_err)?;
            let rtv_size = device
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
                as usize;

            let (buffers, rtv_handles) = build_rtvs(
                &device,
                &swapchain,
                &rtv_heap,
                rtv_size,
                image_count,
                rtv_format,
            )?;
            tracing::debug!(
                "D3D12 swapchain {}x{} ({} images)",
                desc.extent.width,
                desc.extent.height,
                rtv_handles.len()
            );

            Ok(Self {
                device,
                swapchain,
                rtv_heap,
                rtv_size,
                buffers,
                rtv_handles,
                format: desc.format,
                rtv_format,
                base_format,
                extent: desc.extent,
                image_count,
            })
        }
    }

    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<(), EngineError> {
        self.device.wait_idle()?;
        // Release back-buffer references before resizing.
        self.buffers.clear();
        self.rtv_handles.clear();
        unsafe {
            self.swapchain
                .ResizeBuffers(
                    self.image_count,
                    desc.extent.width,
                    desc.extent.height,
                    self.base_format,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .map_err(d3d_err)?;
        }
        let (buffers, rtv_handles) = build_rtvs(
            &self.device,
            &self.swapchain,
            &self.rtv_heap,
            self.rtv_size,
            self.image_count,
            self.rtv_format,
        )?;
        self.buffers = buffers;
        self.rtv_handles = rtv_handles;
        self.extent = desc.extent;
        Ok(())
    }

    /// Current back-buffer index. `signal` is unused on D3D12 (see crate docs).
    pub fn acquire_next_image(
        &self,
        _signal: &crate::D3d12Semaphore,
    ) -> Result<Option<u32>, EngineError> {
        Ok(Some(unsafe { self.swapchain.GetCurrentBackBufferIndex() }))
    }

    pub(crate) fn present(&self) -> Result<bool, EngineError> {
        unsafe {
            self.swapchain
                .Present(1, DXGI_PRESENT(0))
                .ok()
                .map_err(d3d_err)?;
        }
        Ok(false)
    }

    pub(crate) fn buffer(&self, index: u32) -> &ID3D12Resource {
        &self.buffers[index as usize]
    }

    pub(crate) fn rtv_handle(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.rtv_handles[index as usize]
    }

    pub(crate) fn extent(&self) -> Extent2D {
        self.extent
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent_2d(&self) -> Extent2D {
        self.extent
    }

    pub fn image_count(&self) -> u32 {
        self.image_count
    }
}

fn build_rtvs(
    device: &DeviceShared,
    swapchain: &IDXGISwapChain3,
    rtv_heap: &ID3D12DescriptorHeap,
    rtv_size: usize,
    count: u32,
    rtv_format: DXGI_FORMAT,
) -> Result<(Vec<ID3D12Resource>, Vec<D3D12_CPU_DESCRIPTOR_HANDLE>), EngineError> {
    unsafe {
        let start = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let mut buffers = Vec::with_capacity(count as usize);
        let mut handles = Vec::with_capacity(count as usize);
        for i in 0..count {
            let buffer: ID3D12Resource = swapchain.GetBuffer(i).map_err(d3d_err)?;
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i as usize) * rtv_size,
            };
            let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: rtv_format,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_RTV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            };
            device
                .device
                .CreateRenderTargetView(&buffer, Some(&rtv_desc), handle);
            buffers.push(buffer);
            handles.push(handle);
        }
        Ok((buffers, handles))
    }
}
