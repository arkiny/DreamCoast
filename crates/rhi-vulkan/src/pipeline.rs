//! Graphics pipeline (dynamic rendering): vertex layout, blending, push
//! constants, and the bindless descriptor set layout.

use std::ffi::CString;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{
    BlendMode, ComputePipelineDesc, GraphicsPipelineDesc, MeshPipelineDesc, PrimitiveTopology,
    VertexLayout,
};

use crate::device::DeviceShared;
use crate::{to_vk_format, vk_err};

/// A compiled graphics pipeline and its layout.
pub struct VulkanGraphicsPipeline {
    device: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    bindless: bool,
    uniform_buffer: bool,
}

impl VulkanGraphicsPipeline {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &GraphicsPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let vs = create_shader_module(&device.device, desc.vertex_bytes)?;
            let fs = create_shader_module(&device.device, desc.fragment_bytes)?;
            let result = build(&device, desc, vs, fs);
            device.device.destroy_shader_module(vs, None);
            device.device.destroy_shader_module(fs, None);
            let (pipeline, layout) = result?;
            Ok(Self {
                device,
                pipeline,
                layout,
                bindless: desc.bindless,
                uniform_buffer: desc.uniform_buffer,
            })
        }
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub(crate) fn is_bindless(&self) -> bool {
        self.bindless
    }

    pub(crate) fn uses_uniform(&self) -> bool {
        self.uniform_buffer
    }
}

impl Drop for VulkanGraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

unsafe fn build(
    device: &DeviceShared,
    desc: &GraphicsPipelineDesc,
    vs: vk::ShaderModule,
    fs: vk::ShaderModule,
) -> Result<(vk::Pipeline, vk::PipelineLayout), EngineError> {
    unsafe {
        let vs_entry =
            CString::new(desc.vertex_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
        let fs_entry =
            CString::new(desc.fragment_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(&vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs)
                .name(&fs_entry),
        ];

        // Vertex input (ImGui layout or none).
        let vtx_bindings;
        let vtx_attrs;
        let vertex_input = match desc.vertex_layout {
            VertexLayout::None => vk::PipelineVertexInputStateCreateInfo::default(),
            VertexLayout::ImGui => {
                vtx_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(20)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                vtx_attrs = [
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(0),
                    vk::VertexInputAttributeDescription::default()
                        .location(1)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(8),
                    vk::VertexInputAttributeDescription::default()
                        .location(2)
                        .binding(0)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .offset(16),
                ];
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vtx_bindings)
                    .vertex_attribute_descriptions(&vtx_attrs)
            }
            VertexLayout::Mesh => {
                vtx_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(32)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                vtx_attrs = [
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(0),
                    vk::VertexInputAttributeDescription::default()
                        .location(1)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(12),
                    vk::VertexInputAttributeDescription::default()
                        .location(2)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(24),
                ];
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vtx_bindings)
                    .vertex_attribute_descriptions(&vtx_attrs)
            }
            VertexLayout::MeshPosition => {
                vtx_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(32)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                // Same interleaved mesh buffer, but only POSITION is declared
                // (the shadow VS consumes nothing else). `vtx_attrs` shares a type
                // across arms, so build all three and bind just the first.
                vtx_attrs = [
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(0),
                    vk::VertexInputAttributeDescription::default()
                        .location(1)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(12),
                    vk::VertexInputAttributeDescription::default()
                        .location(2)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(24),
                ];
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vtx_bindings)
                    .vertex_attribute_descriptions(&vtx_attrs[..1])
            }
            VertexLayout::MeshPosNormal => {
                vtx_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(32)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                // Position + normal only (the capture VS skips uv).
                vtx_attrs = [
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(0),
                    vk::VertexInputAttributeDescription::default()
                        .location(1)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(12),
                    vk::VertexInputAttributeDescription::default()
                        .location(2)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(24),
                ];
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vtx_bindings)
                    .vertex_attribute_descriptions(&vtx_attrs[..2])
            }
            VertexLayout::MeshPositionUv => {
                vtx_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(32)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                // Position (location 0) + uv (location 1) over the shared 32-byte
                // mesh buffer; the normal bytes at offset 12 are simply not read.
                // The shadow VS omits NORMAL from its input struct, so SPIR-V packs
                // uv at location 1 (no gap) — exactly these two attributes.
                vtx_attrs = [
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32B32_SFLOAT)
                        .offset(0),
                    vk::VertexInputAttributeDescription::default()
                        .location(1)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(24),
                    // Unused third slot (the shared `vtx_attrs` array is 3 long);
                    // only the first two are bound below.
                    vk::VertexInputAttributeDescription::default()
                        .location(2)
                        .binding(0)
                        .format(vk::Format::R32G32_SFLOAT)
                        .offset(24),
                ];
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vtx_bindings)
                    .vertex_attribute_descriptions(&vtx_attrs[..2])
            }
        };

        let topology = match desc.topology {
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // One blend state per color attachment. Opaque/AlphaBlend replicate a single state
        // across every MRT output; DecalAlbedo is per-attachment (RT0 blends, rest masked).
        let opaque_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let alpha_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> = match desc.blend {
            BlendMode::Opaque => vec![opaque_attachment; desc.color_formats.len()],
            BlendMode::AlphaBlend => vec![alpha_attachment; desc.color_formats.len()],
            // Deferred-decal preset (see `BlendMode::DecalAlbedo`): attachment 0 alpha-blends
            // its RGB into the G-buffer albedo (write mask RGB preserves A = baked AO); every
            // other attachment is masked off so the decal leaves the rest of the G-buffer
            // (normal / metallic / roughness / world-pos) untouched.
            BlendMode::DecalAlbedo => (0..desc.color_formats.len())
                .map(|i| {
                    if i == 0 {
                        // RT0 albedo: alpha-blend RGB (A = baked AO preserved).
                        vk::PipelineColorBlendAttachmentState::default()
                            .color_write_mask(
                                vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B,
                            )
                            .blend_enable(true)
                            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .color_blend_op(vk::BlendOp::ADD)
                            .src_alpha_blend_factor(vk::BlendFactor::ONE)
                            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .alpha_blend_op(vk::BlendOp::ADD)
                    } else if i == 2 {
                        // RT2 material: alpha-blend roughness into G only (R metallic + B AO
                        // stay the surface's). (A4)
                        vk::PipelineColorBlendAttachmentState::default()
                            .color_write_mask(vk::ColorComponentFlags::G)
                            .blend_enable(true)
                            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .color_blend_op(vk::BlendOp::ADD)
                            .src_alpha_blend_factor(vk::BlendFactor::ONE)
                            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .alpha_blend_op(vk::BlendOp::ADD)
                    } else {
                        // RT1 normal, RT3 world-pos: untouched.
                        vk::PipelineColorBlendAttachmentState::default()
                            .color_write_mask(vk::ColorComponentFlags::empty())
                            .blend_enable(false)
                    }
                })
                .collect(),
        };
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Pipeline layout: bindless set (0) + optional globals set (1) + optional
        // push constants.
        let set_layouts: Vec<vk::DescriptorSetLayout> = if desc.uniform_buffer {
            vec![device.bindless_layout, device.globals_layout]
        } else {
            vec![device.bindless_layout]
        };
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(desc.push_constant_size)];
        let mut layout_ci = vk::PipelineLayoutCreateInfo::default();
        if desc.bindless {
            layout_ci = layout_ci.set_layouts(&set_layouts);
        }
        if desc.push_constant_size > 0 {
            layout_ci = layout_ci.push_constant_ranges(&push_ranges);
        }
        let layout = device
            .device
            .create_pipeline_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_test)
            .depth_write_enable(desc.depth_write)
            .depth_compare_op(match desc.depth_compare {
                rhi_types::DepthCompare::Less => vk::CompareOp::LESS,
                rhi_types::DepthCompare::Equal => vk::CompareOp::EQUAL,
            });

        let color_formats: Vec<vk::Format> = desc
            .color_formats
            .iter()
            .copied()
            .map(to_vk_format)
            .collect();
        let depth_format = desc
            .depth_format
            .map(to_vk_format)
            .unwrap_or(vk::Format::UNDEFINED);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering);

        let pipelines = device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            .map_err(|(_, e)| vk_err(e))?;

        Ok((pipelines[0], layout))
    }
}

/// A compiled compute pipeline and its layout (Phase 7).
pub struct VulkanComputePipeline {
    device: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    bindless: bool,
    uniform_buffer: bool,
}

/// A mesh-shader pipeline (Phase 14 virtual geometry Track B): a `mesh`(+`task`)+`fragment`
/// pipeline built through `VK_EXT_mesh_shader`, drawn with `vkCmdDrawMeshTasksEXT`. Uses
/// dynamic rendering + the shared bindless/globals descriptor sets, exactly like the graphics
/// pipeline, but with no vertex input / input-assembly (the mesh stage produces primitives).
pub struct VulkanMeshPipeline {
    device: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    bindless: bool,
    uniform_buffer: bool,
    /// The stage set the push-constant range was declared with (MESH|FRAGMENT, plus TASK when an
    /// object stage exists). The command buffer pushes with exactly these stages.
    push_stages: vk::ShaderStageFlags,
}

impl VulkanMeshPipeline {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &MeshPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let ms = create_shader_module(&device.device, desc.mesh_bytes)?;
            let fs = create_shader_module(&device.device, desc.fragment_bytes)?;
            let task = match desc.object_bytes {
                Some(bytes) => Some(create_shader_module(&device.device, bytes)?),
                None => None,
            };
            let result = build_mesh(&device, desc, task, ms, fs);
            device.device.destroy_shader_module(ms, None);
            device.device.destroy_shader_module(fs, None);
            if let Some(t) = task {
                device.device.destroy_shader_module(t, None);
            }
            let (pipeline, layout, push_stages) = result?;
            Ok(Self {
                device,
                pipeline,
                layout,
                bindless: desc.bindless,
                uniform_buffer: desc.uniform_buffer,
                push_stages,
            })
        }
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub(crate) fn is_bindless(&self) -> bool {
        self.bindless
    }

    pub(crate) fn uses_uniform(&self) -> bool {
        self.uniform_buffer
    }

    pub(crate) fn push_stages(&self) -> vk::ShaderStageFlags {
        self.push_stages
    }
}

impl Drop for VulkanMeshPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// Build a mesh-shader pipeline: `task`(optional)+`mesh`+`fragment` stages, no vertex input,
/// opaque blend, dynamic rendering to the given color/depth formats. Push constants are visible
/// to all three stages (TASK|MESH|FRAGMENT), mirroring the graphics builder's VERTEX|FRAGMENT.
unsafe fn build_mesh(
    device: &DeviceShared,
    desc: &MeshPipelineDesc,
    task: Option<vk::ShaderModule>,
    ms: vk::ShaderModule,
    fs: vk::ShaderModule,
) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::ShaderStageFlags), EngineError> {
    unsafe {
        let ms_entry =
            CString::new(desc.mesh_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
        let fs_entry =
            CString::new(desc.fragment_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
        let task_entry =
            CString::new(desc.object_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;

        let mut stages = Vec::with_capacity(3);
        if let Some(t) = task {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::TASK_EXT)
                    .module(t)
                    .name(&task_entry),
            );
        }
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(ms)
                .name(&ms_entry),
        );
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs)
                .name(&fs_entry),
        );

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let opaque_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let attachments = vec![opaque_attachment; desc.color_formats.len()];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Pipeline layout: bindless set (0) + optional globals set (1) + push constants
        // (visible to task/mesh/fragment).
        let set_layouts: Vec<vk::DescriptorSetLayout> = if desc.uniform_buffer {
            vec![device.bindless_layout, device.globals_layout]
        } else {
            vec![device.bindless_layout]
        };
        // Push constants are visible to the stages that actually exist in the pipeline. Declaring
        // TASK_EXT here when there is no task shader makes the RADV/NV drivers deliver nothing to
        // the mesh stage on some paths, so only include TASK_EXT when an object stage is present.
        // `push_constants_mesh` on the command buffer pushes the matching stage set.
        let mut push_stage_flags = vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT;
        if desc.object_bytes.is_some() {
            push_stage_flags |= vk::ShaderStageFlags::TASK_EXT;
        }
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(push_stage_flags)
            .offset(0)
            .size(desc.push_constant_size)];
        let mut layout_ci = vk::PipelineLayoutCreateInfo::default();
        if desc.bindless {
            layout_ci = layout_ci.set_layouts(&set_layouts);
        }
        if desc.push_constant_size > 0 {
            layout_ci = layout_ci.push_constant_ranges(&push_ranges);
        }
        let layout = device
            .device
            .create_pipeline_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_test)
            .depth_write_enable(desc.depth_write)
            .depth_compare_op(match desc.depth_compare {
                rhi_types::DepthCompare::Less => vk::CompareOp::LESS,
                rhi_types::DepthCompare::Equal => vk::CompareOp::EQUAL,
            });

        let color_formats: Vec<vk::Format> = desc
            .color_formats
            .iter()
            .copied()
            .map(to_vk_format)
            .collect();
        let depth_format = desc
            .depth_format
            .map(to_vk_format)
            .unwrap_or(vk::Format::UNDEFINED);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        // No pVertexInputState / pInputAssemblyState: a mesh pipeline has no input assembler
        // (the mesh stage emits primitives directly). Both are ignored by the driver here.
        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering);

        let pipelines = device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            .map_err(|(_, e)| vk_err(e))?;

        Ok((pipelines[0], layout, push_stage_flags))
    }
}

impl VulkanComputePipeline {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &ComputePipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let cs = create_shader_module(&device.device, desc.compute_bytes)?;
            let result = build_compute(&device, desc, cs);
            device.device.destroy_shader_module(cs, None);
            let (pipeline, layout) = result?;
            Ok(Self {
                device,
                pipeline,
                layout,
                bindless: desc.bindless,
                uniform_buffer: desc.uniform_buffer,
            })
        }
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub(crate) fn is_bindless(&self) -> bool {
        self.bindless
    }

    pub(crate) fn uses_uniform(&self) -> bool {
        self.uniform_buffer
    }
}

impl Drop for VulkanComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

unsafe fn build_compute(
    device: &DeviceShared,
    desc: &ComputePipelineDesc,
    cs: vk::ShaderModule,
) -> Result<(vk::Pipeline, vk::PipelineLayout), EngineError> {
    unsafe {
        let entry =
            CString::new(desc.compute_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cs)
            .name(&entry);

        // Bindless set (0) + the optional per-frame globals set (1), the same layout the
        // graphics PBR pipelines use — a compute pass opts in via `uniform_buffer` to read
        // structured per-frame camera data (Stage C7 reflection reprojection).
        let set_layouts: Vec<vk::DescriptorSetLayout> = if desc.uniform_buffer {
            vec![device.bindless_layout, device.globals_layout]
        } else {
            vec![device.bindless_layout]
        };
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(desc.push_constant_size)];
        let mut layout_ci = vk::PipelineLayoutCreateInfo::default();
        if desc.bindless {
            layout_ci = layout_ci.set_layouts(&set_layouts);
        }
        if desc.push_constant_size > 0 {
            layout_ci = layout_ci.push_constant_ranges(&push_ranges);
        }
        let layout = device
            .device
            .create_pipeline_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipelines = device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            .map_err(|(_, e)| vk_err(e))?;
        Ok((pipelines[0], layout))
    }
}

pub(crate) fn create_shader_module(
    device: &ash::Device,
    bytes: &[u8],
) -> Result<vk::ShaderModule, EngineError> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|e| EngineError::Rhi(format!("invalid SPIR-V: {e}")))?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&ci, None).map_err(vk_err) }
}
