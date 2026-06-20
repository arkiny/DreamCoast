//! Vulkan instance, debug messenger, Win32 surface, and physical-device choice.

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use dreamcoast_platform::Window;
use rhi_types::InstanceDesc;

use crate::device::{DeviceShared, VulkanDevice};
use crate::{debug_callback, vk_err};

/// Optional debug-utils messenger state.
struct DebugState {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

/// Instance-level objects shared (via `Arc`) with the device and kept alive for
/// as long as any device derived from it lives.
pub(crate) struct InstanceShared {
    // Owns the dynamically loaded Vulkan library; must outlive everything else.
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    debug: Option<DebugState>,
}

impl Drop for InstanceShared {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(debug) = &self.debug {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan instance bound to a window surface, plus the chosen physical device.
pub struct VulkanInstance {
    pub(crate) shared: Arc<InstanceShared>,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue_family_index: u32,
}

impl VulkanInstance {
    /// Create an instance + surface and select a suitable physical device.
    pub fn new(window: &Window, desc: &InstanceDesc) -> Result<Self, EngineError> {
        unsafe {
            // Validation is a development aid only: shipping (release) builds
            // compile it out entirely. `cfg!(debug_assertions)` is const-false in
            // release, so the layer setup, manifest probing, and debug messenger
            // below are statically unreachable and stripped from the binary.
            let want_validation = cfg!(debug_assertions) && desc.validation;

            // Point the loader at the locally-fetched standalone validation layer
            // (tools/vulkan-layers/, see tools/fetch-vulkan-layers.py) before it
            // enumerates layers, so validation works without a system SDK install.
            if want_validation {
                add_local_layer_path();
            }

            let entry = ash::Entry::load()
                .map_err(|e| EngineError::Rhi(format!("failed to load Vulkan loader: {e}")))?;

            let app_name = std::ffi::CString::new(desc.app_name.as_str())
                .map_err(|e| EngineError::Rhi(e.to_string()))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .api_version(vk::API_VERSION_1_3);

            // Validation layer is enabled only if wanted AND present.
            let validation_name = c"VK_LAYER_KHRONOS_validation";
            let has_validation = want_validation
                && entry
                    .enumerate_instance_layer_properties()
                    .map_err(vk_err)?
                    .iter()
                    .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
            if want_validation && !has_validation {
                tracing::warn!(
                    "validation requested but VK_LAYER_KHRONOS_validation not found; \
                     run tools/fetch-vulkan-layers.py or set VK_LAYER_PATH"
                );
            }

            let mut layers = Vec::new();
            let mut extensions = vec![
                ash::khr::surface::NAME.as_ptr(),
                ash::khr::win32_surface::NAME.as_ptr(),
            ];
            if has_validation {
                layers.push(validation_name.as_ptr());
                extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            }

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);
            let instance = entry.create_instance(&create_info, None).map_err(vk_err)?;

            let debug = if has_validation {
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback));
                let messenger = loader
                    .create_debug_utils_messenger(&ci, None)
                    .map_err(vk_err)?;
                Some(DebugState { loader, messenger })
            } else {
                None
            };

            // Win32 surface from the window's HWND/HINSTANCE.
            let win32_loader = ash::khr::win32_surface::Instance::new(&entry, &instance);
            let surface_ci = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(window.hinstance().0 as _)
                .hwnd(window.hwnd().0 as _);
            let surface = win32_loader
                .create_win32_surface(&surface_ci, None)
                .map_err(vk_err)?;
            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

            let (physical_device, queue_family_index) =
                pick_physical_device(&instance, &surface_loader, surface)?;

            Ok(Self {
                shared: Arc::new(InstanceShared {
                    entry,
                    instance,
                    surface_loader,
                    surface,
                    debug,
                }),
                physical_device,
                queue_family_index,
            })
        }
    }

    /// Create a logical device (and its single graphics+present queue).
    pub fn create_device(&self) -> Result<VulkanDevice, EngineError> {
        let shared = DeviceShared::new(self)?;
        Ok(VulkanDevice {
            shared: Arc::new(shared),
        })
    }

    /// The backend this instance dispatches to.
    pub fn backend(&self) -> rhi_types::BackendKind {
        rhi_types::BackendKind::Vulkan
    }
}

/// If a standalone validation layer manifest is present in a known location, add
/// its directory to `VK_ADD_LAYER_PATH` so the loader discovers the layer. Looks
/// at `$ENGINE_VK_LAYER_DIR`, then the in-repo `tools/vulkan-layers/` (resolved
/// both from the build-time workspace root and the current directory).
fn add_local_layer_path() {
    const MANIFEST: &str = "VkLayer_khronos_validation.json";
    let candidates = [
        std::env::var_os("ENGINE_VK_LAYER_DIR").map(PathBuf::from),
        Some(PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/vulkan-layers"
        ))),
        Some(PathBuf::from("tools/vulkan-layers")),
    ];

    let Some(dir) = candidates
        .into_iter()
        .flatten()
        .find(|d| d.join(MANIFEST).is_file())
    else {
        return;
    };

    // Prepend our directory, preserving any existing search path.
    let mut paths = vec![dir.clone()];
    if let Some(existing) = std::env::var_os("VK_ADD_LAYER_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    match std::env::join_paths(&paths) {
        Ok(joined) => {
            // SAFE: called once at startup before any threads touch the env or the
            // Vulkan loader reads it.
            unsafe { std::env::set_var("VK_ADD_LAYER_PATH", &joined) };
            tracing::debug!("using local Vulkan layer at {}", dir.display());
        }
        Err(e) => tracing::warn!("could not set VK_ADD_LAYER_PATH: {e}"),
    }
}

/// Pick a physical device with a queue family that supports graphics + present,
/// preferring a discrete GPU. The swapchain extension and Vulkan 1.3 dynamic
/// rendering are enabled at device creation (any modern desktop GPU qualifies).
fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), EngineError> {
    unsafe {
        let devices = instance.enumerate_physical_devices().map_err(vk_err)?;
        let mut fallback: Option<(vk::PhysicalDevice, u32)> = None;

        for pd in devices {
            let families = instance.get_physical_device_queue_family_properties(pd);
            for (index, family) in families.iter().enumerate() {
                let index = index as u32;
                let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let present = surface_loader
                    .get_physical_device_surface_support(pd, index, surface)
                    .unwrap_or(false);
                if graphics && present {
                    let props = instance.get_physical_device_properties(pd);
                    let candidate = (pd, index);
                    if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
                        let v = props.api_version;
                        tracing::info!(
                            "selected discrete GPU: {name} (Vulkan {}.{}.{})",
                            vk::api_version_major(v),
                            vk::api_version_minor(v),
                            vk::api_version_patch(v),
                        );
                        return Ok(candidate);
                    }
                    fallback.get_or_insert(candidate);
                    break;
                }
            }
        }

        fallback.ok_or_else(|| {
            EngineError::Rhi("no Vulkan device with graphics+present support".into())
        })
    }
}
