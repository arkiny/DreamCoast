//! Graphics pipeline (dynamic rendering): vertex layout, blending, push
//! constants, and the bindless descriptor set layout.

use std::ffi::CString;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{BlendMode, GraphicsPipelineDesc, PrimitiveTopology, VertexLayout};

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

        let color_blend_attachment = match desc.blend {
            BlendMode::Opaque => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false),
            BlendMode::AlphaBlend => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD),
        };
        // One blend state per color attachment (same blend for all MRT outputs).
        let attachments = vec![color_blend_attachment; desc.color_formats.len()];
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
            .depth_write_enable(desc.depth_test)
            .depth_compare_op(vk::CompareOp::LESS);

        let color_formats: Vec<vk::Format> =
            desc.color_formats.iter().copied().map(to_vk_format).collect();
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

fn create_shader_module(
    device: &ash::Device,
    bytes: &[u8],
) -> Result<vk::ShaderModule, EngineError> {
    let code = ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|e| EngineError::Rhi(format!("invalid SPIR-V: {e}")))?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&ci, None).map_err(vk_err) }
}
