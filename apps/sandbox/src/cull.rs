//! GPU frustum culling (Phase 7) extracted from `run()` — R2 of the render-loop
//! decomposition (see docs/refactor-sandbox.md).
//!
//! A compute pass tests a cube instance grid against the frustum and writes an
//! indirect draw; the draw renders only the visible instances. The bundle owns the
//! args/visible buffers, the reset/cull/draw pipelines, and the grid's cube mesh.
//! `record_cull` adds the reset + cull compute passes, `record_draw` the indirect
//! draw — both borrow `&'a self` for the graph's lifetime.

use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, Buffer, ComputePipeline, ComputePipelineDesc, DepthCompare, Device,
    Extent2D, Format, GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology, StorageBuffer,
    StorageBufferDesc, VertexLayout,
};

use crate::app::{load_compute_shader, load_shader_pair};
use crate::mesh::upload_mesh;
use crate::push::{cull_draw_push, cull_push};
use crate::{DEPTH_FORMAT, GRID_COUNT, GRID_DIM};

/// Per-frame grid placement derived from the scene radius (the grid floats above the
/// scene so orbiting the camera culls cubes off the frustum edges).
pub(crate) struct CullGrid {
    pub(crate) spacing: f32,
    pub(crate) height: f32,
    /// Per-instance cube scale (draw).
    pub(crate) cube_scale: f32,
    /// Bounding-sphere radius of a scaled cube (cull test).
    pub(crate) cube_radius: f32,
}

pub(crate) struct CullSystem {
    args: StorageBuffer,
    visible: StorageBuffer,
    reset_pipeline: ComputePipeline,
    cull_pipeline: ComputePipeline,
    /// `None` where compute is unavailable (Metal until M5); the feature stays off.
    draw_pipeline: Option<GraphicsPipeline>,
    cube_vbuf: Buffer,
    cube_ibuf: Buffer,
    cube_index_count: u32,
}

impl CullSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
        color_format: Format,
    ) -> anyhow::Result<Self> {
        let cull_grid_cube = dreamcoast_asset::unit_cube();
        let (cube_vbuf, cube_ibuf, cube_index_count) = upload_mesh(device, &cull_grid_cube)?;
        let args = device.create_storage_buffer(&StorageBufferDesc {
            size: 32, // 5 u32 args (+pad), used as draw_indexed_indirect source
            stride: 4,
            indirect: true,
        })?;
        let visible = device.create_storage_buffer(&StorageBufferDesc {
            size: (GRID_COUNT * 4) as u64,
            stride: 4,
            indirect: false,
        })?;
        let cull_reset_cs = load_compute_shader(
            backend,
            dreamcoast_shader::cull_reset_cs_spirv,
            dreamcoast_shader::cull_reset_cs_dxil,
            dreamcoast_shader::cull_reset_cs_metallib,
            "cull_reset",
        )?;
        let reset_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: cull_reset_cs,
            compute_entry: "csReset",
            push_constant_size: 128,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [1, 1, 1],
        })?;
        let cull_cs = load_compute_shader(
            backend,
            dreamcoast_shader::cull_cs_spirv,
            dreamcoast_shader::cull_cs_dxil,
            dreamcoast_shader::cull_cs_metallib,
            "cull",
        )?;
        let cull_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: cull_cs,
            compute_entry: "csCull",
            push_constant_size: 128,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [64, 1, 1],
        })?;
        // The cull-draw pipeline draws the GPU-culled list indirectly, so it is
        // compute-only (`None` on Metal; the feature flag stays off there).
        let draw_pipeline = if compute_supported {
            let (cd_vs, cd_fs) = load_shader_pair(
                backend,
                dreamcoast_shader::cull_draw_vs_spirv,
                dreamcoast_shader::cull_draw_fs_spirv,
                dreamcoast_shader::cull_draw_vs_dxil,
                dreamcoast_shader::cull_draw_fs_dxil,
                dreamcoast_shader::cull_draw_vs_metallib,
                dreamcoast_shader::cull_draw_fs_metallib,
                "cull_draw",
            )?;
            Some(device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_bytes: cd_vs,
                fragment_bytes: cd_fs,
                vertex_entry: "vsMain",
                fragment_entry: "fsMain",
                color_formats: &[color_format],
                topology: PrimitiveTopology::TriangleList,
                vertex_layout: VertexLayout::MeshPosNormal,
                blend: BlendMode::Opaque,
                push_constant_size: 112, // view_proj + sun_dir + grid params
                bindless: true,
                uniform_buffer: false,
                depth_test: true,
                depth_write: true,
                depth_compare: DepthCompare::Less,
                depth_format: Some(DEPTH_FORMAT),
            })?)
        } else {
            None
        };
        Ok(Self {
            args,
            visible,
            reset_pipeline,
            cull_pipeline,
            draw_pipeline,
            cube_vbuf,
            cube_ibuf,
            cube_index_count,
        })
    }

    /// Import the args + visible buffers as external graph resources sequencing the
    /// reset → cull → draw passes. Returns their ids for the caller to thread through.
    pub(crate) fn import(graph: &mut RenderGraph) -> (ResourceId, ResourceId) {
        (
            graph.import_external("cull_args"),
            graph.import_external("cull_visible"),
        )
    }

    /// The indirect-args + visible-list buffers, for the HZB cull path (PR-8), which
    /// runs its own occlusion-aware cull compute over the same buffers so the indirect
    /// draw is unchanged.
    pub(crate) fn buffers(&self) -> (&StorageBuffer, &StorageBuffer) {
        (&self.args, &self.visible)
    }

    /// Cube index count written into the indirect draw args header.
    pub(crate) fn index_count(&self) -> u32 {
        self.cube_index_count
    }

    /// Record the reset pass only (clear the indirect args header). The HZB path
    /// substitutes its own occlusion-aware cull for `record_cull`'s frustum cull but
    /// still needs the header reset first.
    pub(crate) fn record_reset<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        args_ext: ResourceId,
        visible_ext: ResourceId,
        planes: [[f32; 4]; 6],
        grid: &CullGrid,
    ) {
        let reset = &self.reset_pipeline;
        let args = &self.args;
        let visible = &self.visible;
        let icount = self.cube_index_count;
        let (spacing, radius, height) = (grid.spacing, grid.cube_radius, grid.height);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "cull_reset",
                storage_writes: vec![args_ext, visible_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.storage_buffer_to_storage(args); // INDIRECT (prev frame) -> UAV
                cmd.bind_compute_pipeline(reset);
                cmd.push_constants_compute(&cull_push(
                    &planes,
                    args.storage_index(),
                    visible.storage_index(),
                    GRID_COUNT,
                    GRID_DIM,
                    spacing,
                    radius,
                    height,
                    icount,
                ));
                cmd.dispatch(1, 1, 1);
                cmd.storage_buffer_barrier(args); // order reset before cull
                Ok(())
            },
        );
    }

    /// Add the reset + cull compute passes (clear the indirect args, then frustum-cull
    /// the grid into the visible list + draw count).
    pub(crate) fn record_cull<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        args_ext: ResourceId,
        visible_ext: ResourceId,
        planes: [[f32; 4]; 6],
        grid: &CullGrid,
    ) {
        let reset = &self.reset_pipeline;
        let cull = &self.cull_pipeline;
        let args = &self.args;
        let visible = &self.visible;
        let icount = self.cube_index_count;
        let (spacing, radius, height) = (grid.spacing, grid.cube_radius, grid.height);
        // Reset pass: clear the args header (and recycle args from last frame's
        // indirect state back to UAV).
        graph.add_compute_pass(
            ComputePassInfo {
                name: "cull_reset",
                storage_writes: vec![args_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.storage_buffer_to_storage(args); // INDIRECT (prev frame) -> UAV
                cmd.bind_compute_pipeline(reset);
                cmd.push_constants_compute(&cull_push(
                    &planes,
                    args.storage_index(),
                    visible.storage_index(),
                    GRID_COUNT,
                    GRID_DIM,
                    spacing,
                    radius,
                    height,
                    icount,
                ));
                cmd.dispatch(1, 1, 1);
                cmd.storage_buffer_barrier(args); // order reset before cull
                Ok(())
            },
        );
        // Cull pass: append visible instances + atomically bump InstanceCount.
        graph.add_compute_pass(
            ComputePassInfo {
                name: "cull",
                storage_writes: vec![args_ext, visible_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(cull);
                cmd.push_constants_compute(&cull_push(
                    &planes,
                    args.storage_index(),
                    visible.storage_index(),
                    GRID_COUNT,
                    GRID_DIM,
                    spacing,
                    radius,
                    height,
                    icount,
                ));
                cmd.dispatch(GRID_COUNT.div_ceil(64), 1, 1);
                // Order the writes before the indirect draw / vertex read.
                cmd.storage_buffer_barrier(visible);
                cmd.storage_buffer_to_indirect(args);
                Ok(())
            },
        );
    }

    /// Add the indirect, instanced draw of the visible cube grid over `backbuffer`
    /// (its own depth attachment orders cube-vs-cube). `scene_depth` = the scene's
    /// depth buffer, sampled in the fragment shader as a manual depth test (PR-8):
    /// a grid cube behind a wall renders nothing, which is exactly the set the HZB
    /// occlusion cull removes — so culling on/off is image-identical by construction.
    /// `render_extent` (the scene/depth resolution) maps display pixels to depth
    /// texels (1:1 at native; smaller under the TAAU upscale path).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_draw<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        backbuffer: ResourceId,
        extent: Extent2D,
        args_ext: ResourceId,
        visible_ext: ResourceId,
        view_proj: [f32; 16],
        sun_dir: [f32; 3],
        grid: &CullGrid,
        scene_depth: ResourceId,
        render_extent: Extent2D,
    ) {
        let cull_depth = graph.create_depth("cull_depth", extent);
        let draw = self
            .draw_pipeline
            .as_ref()
            .expect("cull requires compute support");
        let args = &self.args;
        let visible = &self.visible;
        let vbuf = &self.cube_vbuf;
        let ibuf = &self.cube_ibuf;
        let (spacing, scale, height) = (grid.spacing, grid.cube_scale, grid.height);
        let depth_scale = [
            render_extent.width as f32 / extent.width as f32,
            render_extent.height as f32 / extent.height as f32,
        ];
        graph.add_pass(
            PassInfo {
                name: "cull_draw",
                colors: vec![(backbuffer, None)],
                depth: Some(cull_depth),
                reads: vec![args_ext, visible_ext, scene_depth],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(scene_depth);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(draw);
                cmd.push_constants(&cull_draw_push(
                    &view_proj,
                    sun_dir,
                    visible.storage_index(),
                    GRID_DIM,
                    spacing,
                    scale,
                    height,
                    depth_index,
                    depth_scale,
                ));
                cmd.bind_vertex_buffer(vbuf, 32);
                cmd.bind_index_buffer(ibuf, true);
                cmd.draw_indexed_indirect(args, 0, 1);
                Ok(())
            },
        );
    }
}
