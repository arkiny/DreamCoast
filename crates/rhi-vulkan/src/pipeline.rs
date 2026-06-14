//! Graphics pipeline for the triangle, using dynamic rendering (no render pass).

use std::ffi::CString;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;
use engine_core::EngineError;
use rhi_types::{GraphicsPipelineDesc, PrimitiveTopology};

use crate::device::DeviceShared;
use crate::{to_vk_format, vk_err};

/// A compiled graphics pipeline and its (empty) layout.
pub struct VulkanGraphicsPipeline {
    device: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl VulkanGraphicsPipeline {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &GraphicsPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let vs = create_shader_module(&device.device, desc.vertex_bytes)?;
            let fs = create_shader_module(&device.device, desc.fragment_bytes)?;
            // Modules can be destroyed once the pipeline is built; ensure cleanup
            // even on the error paths below.
            let result = build(&device, desc, vs, fs);
            device.device.destroy_shader_module(vs, None);
            device.device.destroy_shader_module(fs, None);
            let (pipeline, layout) = result?;
            Ok(Self {
                device,
                pipeline,
                layout,
            })
        }
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
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

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let attachments = [color_blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout = device
            .device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .map_err(vk_err)?;

        let color_formats = [to_vk_format(desc.color_format)];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
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
