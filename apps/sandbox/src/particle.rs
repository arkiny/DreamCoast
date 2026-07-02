//! GPU particle system (Phase 7) extracted from `run()` — the first feature-bundle
//! of the render-loop decomposition (see docs/refactor-sandbox.md, R1).
//!
//! Owns its sim + draw pipelines, a ping-pong pair of persistent storage buffers,
//! and the parity. `new` seeds both buffers; `record_sim` / `record_draw` add the
//! graph passes (borrowing `&'a self` for the graph's lifetime). The async-compute
//! submission path stays in the frame loop — it rewrites the frame's *submit*, not a
//! graph pass — and drives this bundle through the accessors below.

use dreamcoast_core::glam::Vec3;
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, ComputePipeline, ComputePipelineDesc, DepthCompare, Device, Format,
    GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, StorageBuffer, StorageBufferDesc,
    VertexLayout,
};

use crate::PARTICLE_COUNT;
use crate::app::{load_compute_shader, load_shader_pair};
use crate::push::{particle_draw_push, particle_sim_push};

pub(crate) struct ParticleSystem {
    sim_pipeline: ComputePipeline,
    /// `None` where compute is unavailable (Metal until M5); the feature stays off.
    draw_pipeline: Option<GraphicsPipeline>,
    /// Two buffers: the sim reads one and writes the other so a frame's compute never
    /// clobbers the buffer a still-in-flight previous draw is reading.
    buffers: [StorageBuffer; 2],
    parity: usize,
}

impl ParticleSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
        color_format: Format,
    ) -> anyhow::Result<Self> {
        let particle_sim_cs = load_compute_shader(
            backend,
            dreamcoast_shader::particle_sim_cs_spirv,
            dreamcoast_shader::particle_sim_cs_dxil,
            dreamcoast_shader::particle_sim_cs_metallib,
            "particle_sim",
        )?;
        let sim_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: particle_sim_cs,
            compute_entry: "csMain",
            push_constant_size: 24, // read_index + write_index + count + dt + time + init
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [64, 1, 1],
        })?;
        // The draw pipeline vertex-pulls from the compute-written buffer, so it only
        // exists where compute does (`None` on Metal; the feature flag stays off).
        let draw_pipeline = if compute_supported {
            let (pd_vs, pd_fs) = load_shader_pair(
                backend,
                dreamcoast_shader::particle_draw_vs_spirv,
                dreamcoast_shader::particle_draw_fs_spirv,
                dreamcoast_shader::particle_draw_vs_dxil,
                dreamcoast_shader::particle_draw_fs_dxil,
                dreamcoast_shader::particle_draw_vs_metallib,
                dreamcoast_shader::particle_draw_fs_metallib,
                "particle_draw",
            )?;
            Some(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: pd_vs,
                fragment_bytes: pd_fs,
                vertex_entry: "vsMain",
                fragment_entry: "fsMain",
                color_formats: &[color_format],
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::None, // vertex-pull from the storage buffer
                blend: BlendMode::AlphaBlend,
                push_constant_size: 112, // view_proj + cam_right + cam_up + buffer/count/size/pad
                bindless: true,
                uniform_buffer: false,
                depth_test: false,
                depth_write: false,
                depth_compare: DepthCompare::Less,
                depth_format: None,
            })?)
        } else {
            None
        };
        let buffers = [
            device.create_storage_buffer(&StorageBufferDesc {
                size: (PARTICLE_COUNT * 32) as u64,
                stride: 32,
                indirect: false,
            })?,
            device.create_storage_buffer(&StorageBufferDesc {
                size: (PARTICLE_COUNT * 32) as u64,
                stride: 32,
                indirect: false,
            })?,
        ];
        // Seed both buffers once (init dispatch into each) so the first frame's read
        // source is valid whichever parity it starts on. Skipped on Metal.
        if compute_supported {
            let init_cmd = device.create_command_buffer()?;
            init_cmd.begin()?;
            init_cmd.bind_compute_pipeline(&sim_pipeline);
            for buf in &buffers {
                let idx = buf.storage_index();
                init_cmd.push_constants_compute(&particle_sim_push(
                    idx,
                    idx,
                    PARTICLE_COUNT as u32,
                    0.0,
                    0.0,
                    1,
                ));
                init_cmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
            }
            init_cmd.end()?;
            let fence = device.create_fence(false)?;
            device.queue().submit_oneshot(&init_cmd, &fence)?;
            fence.wait()?;
        }
        Ok(Self {
            sim_pipeline,
            draw_pipeline,
            buffers,
            parity: 0,
        })
    }

    /// This frame's source buffer index (the previous write).
    pub(crate) fn read_index(&self) -> usize {
        self.parity ^ 1
    }
    /// This frame's destination buffer index (the draw reads this one).
    pub(crate) fn write_index(&self) -> usize {
        self.parity
    }
    /// Swap source/destination for the next simulated frame.
    pub(crate) fn advance(&mut self) {
        self.parity ^= 1;
    }
    /// The sim pipeline, for the async-compute submission path that stays in `run()`.
    pub(crate) fn sim_pipeline(&self) -> &ComputePipeline {
        &self.sim_pipeline
    }
    /// Bindless storage index of buffer `i`, for the async-compute path.
    pub(crate) fn buffer_storage_index(&self, i: usize) -> u32 {
        self.buffers[i].storage_index()
    }

    /// Add the graphics-queue sim compute pass (the non-async path). `particles_ext`
    /// is the external graph resource sequencing the write before the draw's read.
    pub(crate) fn record_sim<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        particles_ext: ResourceId,
        dt: f32,
        time: f32,
    ) {
        let sim = &self.sim_pipeline;
        let src = &self.buffers[self.read_index()];
        let dst = &self.buffers[self.write_index()];
        graph.add_compute_pass(
            ComputePassInfo {
                name: "particle_sim",
                storage_writes: vec![particles_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(sim);
                cmd.push_constants_compute(&particle_sim_push(
                    src.storage_index(),
                    dst.storage_index(),
                    PARTICLE_COUNT as u32,
                    dt,
                    time,
                    0,
                ));
                cmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
                // Order the write before the draw pass's vertex-stage read.
                cmd.storage_buffer_barrier(dst);
                Ok(())
            },
        );
    }

    /// Add the instanced-billboard draw pass over `backbuffer` (alpha blend), reading
    /// this frame's written buffer in the vertex stage.
    pub(crate) fn record_draw<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        backbuffer: ResourceId,
        particles_ext: ResourceId,
        view_proj: [f32; 16],
        cam_right: Vec3,
        cam_up: Vec3,
    ) {
        let draw = self
            .draw_pipeline
            .as_ref()
            .expect("particles require compute support");
        let buf = &self.buffers[self.write_index()];
        graph.add_pass(
            PassInfo {
                name: "particle_draw",
                colors: vec![(backbuffer, None)],
                depth: None,
                reads: vec![particles_ext],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(draw);
                cmd.push_constants(&particle_draw_push(
                    &view_proj,
                    cam_right,
                    cam_up,
                    buf.storage_index(),
                    PARTICLE_COUNT as u32,
                    0.05,
                ));
                cmd.draw(6, PARTICLE_COUNT as u32);
                Ok(())
            },
        );
    }
}
