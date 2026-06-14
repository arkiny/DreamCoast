//! DXGI factory, adapter selection, and (optional) D3D12 debug layer.

use std::rc::Rc;

use engine_core::EngineError;
use engine_platform::Window;
use rhi_types::InstanceDesc;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, D3D12GetDebugInterface, ID3D12Debug,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_DEBUG,
    DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1, IDXGIFactory6,
};
use windows::core::Interface;

use crate::device::{D3d12Device, DeviceShared};

/// Instance-level objects: the DXGI factory, the chosen adapter, and the target
/// window handle. Shared with the device for swapchain creation.
pub(crate) struct InstanceShared {
    pub factory: IDXGIFactory6,
    pub adapter: IDXGIAdapter1,
    pub hwnd: HWND,
}

/// A D3D12 "instance": factory + adapter bound to a window.
pub struct D3d12Instance {
    pub(crate) shared: Rc<InstanceShared>,
}

impl D3d12Instance {
    pub fn new(window: &Window, desc: &InstanceDesc) -> Result<Self, EngineError> {
        unsafe {
            // Optional debug layer (requires the "Graphics Tools" feature).
            let mut debug_enabled = false;
            if desc.validation {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok() {
                    if let Some(debug) = debug {
                        debug.EnableDebugLayer();
                        debug_enabled = true;
                    }
                } else {
                    tracing::warn!(
                        "D3D12 debug layer requested but unavailable (install the \
                         'Graphics Tools' optional feature); continuing without it"
                    );
                }
            }

            let factory_flags = if debug_enabled {
                DXGI_CREATE_FACTORY_DEBUG
            } else {
                DXGI_CREATE_FACTORY_FLAGS(0)
            };
            let factory: IDXGIFactory6 = CreateDXGIFactory2(factory_flags).map_err(d3d_err)?;

            let adapter = pick_adapter(&factory)?;
            let desc1 = adapter.GetDesc1().map_err(d3d_err)?;
            let name = String::from_utf16_lossy(
                &desc1.Description[..desc1
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc1.Description.len())],
            );
            tracing::info!("selected D3D12 adapter: {name}");

            Ok(Self {
                shared: Rc::new(InstanceShared {
                    factory,
                    adapter,
                    hwnd: window.hwnd(),
                }),
            })
        }
    }

    pub fn create_device(&self) -> Result<D3d12Device, EngineError> {
        let shared = DeviceShared::new(self)?;
        Ok(D3d12Device {
            shared: Rc::new(shared),
        })
    }

    pub fn backend(&self) -> rhi_types::BackendKind {
        rhi_types::BackendKind::D3d12
    }
}

/// Pick the highest-performance hardware adapter that can create a 12_0 device.
fn pick_adapter(factory: &IDXGIFactory6) -> Result<IDXGIAdapter1, EngineError> {
    unsafe {
        for index in 0.. {
            let adapter: IDXGIAdapter1 = match factory
                .EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            {
                Ok(a) => a,
                Err(_) => break, // DXGI_ERROR_NOT_FOUND: enumerated all adapters
            };
            let desc = adapter.GetDesc1().map_err(d3d_err)?;
            if DXGI_ADAPTER_FLAG(desc.Flags as i32) & DXGI_ADAPTER_FLAG_SOFTWARE
                != DXGI_ADAPTER_FLAG(0)
            {
                continue; // skip WARP/software
            }
            // Test 12_0 support without creating a real device.
            let unknown = adapter.cast::<windows::core::IUnknown>().map_err(d3d_err)?;
            if D3D12CreateDevice(
                &unknown,
                D3D_FEATURE_LEVEL_12_0,
                std::ptr::null_mut::<Option<windows::Win32::Graphics::Direct3D12::ID3D12Device>>(),
            )
            .is_ok()
            {
                return Ok(adapter);
            }
        }
        Err(EngineError::Rhi(
            "no D3D12-capable hardware adapter found".into(),
        ))
    }
}

/// Map a windows-rs error into the engine error type.
pub(crate) fn d3d_err(e: windows::core::Error) -> EngineError {
    EngineError::Rhi(format!("d3d12: {e}"))
}
