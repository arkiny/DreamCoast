//! Offscreen color render target: a device-local image usable as both a color
//! attachment (render-graph passes write it) and a bindless sampled texture
//! (later passes read it). Its current layout is tracked so the render graph can
//! emit the right barriers between writing and sampling.
//!
//! A target's image memory is either **owned** (a dedicated allocation) or
//! **aliased** into a [`VulkanTransientHeap`] at a graph-computed offset, so
//! transient targets with non-overlapping lifetimes can share storage.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{MemoryRequirements, RenderTargetDesc};

use crate::device::DeviceShared;
use crate::{color_subresource_range, to_vk_format, vk_err};

/// A shared transient-heap allocation; freed once the heap and every aliased
/// target referencing it are dropped.
pub(crate) struct HeapMemory {
    device: Arc<DeviceShared>,
    memory: vk::DeviceMemory,
}

impl Drop for HeapMemory {
    fn drop(&mut self) {
        unsafe { self.device.device.free_memory(self.memory, None) };
    }
}

/// How a render target's image memory is owned.
enum Memory {
    /// A dedicated allocation, freed with the target.
    Owned(vk::DeviceMemory),
    /// A slice of a transient heap; the `Arc` keeps the shared allocation alive
    /// until every aliased target is dropped (read only via its `Drop`).
    Aliased(#[allow(dead_code)] Arc<HeapMemory>),
}

/// A color render target + view, registered in the bindless table.
pub struct VulkanRenderTarget {
    device: Arc<DeviceShared>,
    image: vk::Image,
    memory: Memory,
    view: vk::ImageView,
    index: u32,
    /// Bindless storage-image (UAV) index, when created with `storage` (Phase 7).
    storage_index: Option<u32>,
    extent: vk::Extent2D,
    /// Current image layout, updated by the barrier helpers.
    layout: Cell<vk::ImageLayout>,
}

impl VulkanRenderTarget {
    /// Create a target with its own dedicated allocation.
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &RenderTargetDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let (image, extent) = create_image(&device, desc, false)?;
            let req = device.device.get_image_memory_requirements(image);
            // VRAM-aware allocation: device-local first, host-visible spill on exhaustion.
            let memory = device.allocate_resource_memory(req.size, req.memory_type_bits)?;
            device
                .device
                .bind_image_memory(image, memory, 0)
                .map_err(vk_err)?;
            Self::finish(device, image, Memory::Owned(memory), extent, desc)
        }
    }

    /// Create a target aliased into `heap` at `offset`.
    pub(crate) fn new_aliased(
        device: Arc<DeviceShared>,
        heap: &VulkanTransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let (image, extent) = create_image(&device, desc, true)?;
            device
                .device
                .bind_image_memory(image, heap.mem.memory, offset)
                .map_err(vk_err)?;
            Self::finish(
                device,
                image,
                Memory::Aliased(heap.mem.clone()),
                extent,
                desc,
            )
        }
    }

    unsafe fn finish(
        device: Arc<DeviceShared>,
        image: vk::Image,
        memory: Memory,
        extent: vk::Extent2D,
        desc: &RenderTargetDesc,
    ) -> Result<Self, EngineError> {
        let format = to_vk_format(desc.format);
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_subresource_range());
        let view = unsafe { device.device.create_image_view(&view_ci, None) }.map_err(vk_err)?;
        // The bindless descriptor records SHADER_READ_ONLY_OPTIMAL; the graph
        // guarantees the image is in that layout before a sampling pass.
        let index = device.register_sampled_image(view);
        // Storage images additionally get a UAV descriptor (GENERAL layout) so a
        // compute pass can write them.
        let storage_index = if desc.storage {
            Some(device.register_storage_image(view))
        } else {
            None
        };
        Ok(Self {
            device,
            image,
            memory,
            view,
            index,
            storage_index,
            extent,
            layout: Cell::new(vk::ImageLayout::UNDEFINED),
        })
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }

    /// Index of this target in the device's bindless table.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    /// Tag this target's image with a debug name (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        self.device.set_image_name(self.image, name);
    }

    /// Bindless storage-image (UAV) index, if created with `storage`.
    pub fn storage_index(&self) -> Option<u32> {
        self.storage_index
    }
}

impl Drop for VulkanRenderTarget {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            self.device.device.destroy_image(self.image, None);
            if let Memory::Owned(m) = self.memory {
                self.device.device.free_memory(m, None);
            }
        }
    }
}

/// Create a color-attachment + sampled image (optionally aliasable).
unsafe fn create_image(
    device: &DeviceShared,
    desc: &RenderTargetDesc,
    alias: bool,
) -> Result<(vk::Image, vk::Extent2D), EngineError> {
    let format = to_vk_format(desc.format);
    let extent = vk::Extent2D {
        width: desc.width.max(1),
        height: desc.height.max(1),
    };
    let flags = if alias {
        vk::ImageCreateFlags::ALIAS
    } else {
        vk::ImageCreateFlags::empty()
    };
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    if desc.storage {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    let image_ci = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.device.create_image(&image_ci, None) }.map_err(vk_err)?;
    Ok((image, extent))
}

/// Query the memory footprint of an aliasable render target.
pub(crate) fn render_target_memory(
    device: &DeviceShared,
    desc: &RenderTargetDesc,
) -> Result<MemoryRequirements, EngineError> {
    unsafe {
        let (image, _) = create_image(device, desc, true)?;
        let req = device.device.get_image_memory_requirements(image);
        device.device.destroy_image(image, None);
        Ok(MemoryRequirements {
            size: req.size,
            alignment: req.alignment,
        })
    }
}

/// A pool of device-local memory that transient render targets alias into at
/// graph-computed offsets.
pub struct VulkanTransientHeap {
    mem: Arc<HeapMemory>,
}

impl VulkanTransientHeap {
    pub(crate) fn new(device: Arc<DeviceShared>, size: u64) -> Result<Self, EngineError> {
        unsafe {
            // Derive a memory type compatible with aliasable color-attachment
            // images (all transient targets share this type).
            let sample = create_image(
                &device,
                &RenderTargetDesc {
                    width: 16,
                    height: 16,
                    format: rhi_types::Format::Rgba8Unorm,
                    storage: false,
                },
                true,
            )?;
            let req = device.device.get_image_memory_requirements(sample.0);
            device.device.destroy_image(sample.0, None);
            // VRAM-aware allocation for the whole transient heap: device-local first,
            // host-visible spill on exhaustion. (`req` only supplies the type mask here.)
            let memory = device.allocate_resource_memory(size.max(1), req.memory_type_bits)?;
            Ok(Self {
                mem: Arc::new(HeapMemory { device, memory }),
            })
        }
    }
}
