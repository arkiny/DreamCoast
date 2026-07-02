//! Clustered light culling (PR-6 · render-pipeline-reference §2 #8).
//!
//! A compute pass (`light_cluster.slang::csBuildClusters`) splits the view frustum into a
//! fixed 3D froxel grid and, per cluster, bins the point lights whose sphere of influence
//! overlaps that cluster's world-space AABB into a flat index list. The deferred lighting
//! pass then reads only its pixel's cluster list instead of looping every scene light — the
//! shading cost scales with lights-per-cluster, not lights-in-scene.
//!
//! Opt-in (`CLUSTERED_LIGHTS=1`): default off keeps the brute-force `globals.point_pos[]`
//! loop, byte-identical to the pre-PR-6 renderer. See docs/clustered-lighting.md for the
//! froxel-vs-Z-binning design rationale.
//!
//! Buffers (all in the bindless storage-buffer table):
//!  - `lights`  (per-fif, host-written): packed `Light[]` (2×float4 = 32 B each).
//!  - `grid`    (device-local, UAV): per-cluster light COUNT, `u32[CLUSTER_COUNT]`.
//!  - `index`   (device-local, UAV): flat index list, `u32[CLUSTER_COUNT * MAX_PER_CLUSTER]`.

use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, StorageBuffer, StorageBufferDesc,
};

use crate::app::load_compute_shader;
use crate::{
    CLUSTER_COUNT, CLUSTER_X, CLUSTER_Y, CLUSTER_Z, FRAMES_IN_FLIGHT, MAX_LIGHTS_PER_CLUSTER,
};

/// One point light as uploaded to the cluster/light buffer: position + radius, then color +
/// intensity (candela). Mirrors `Light` in `light_cluster_common.slang` (2×float4, 32 bytes).
#[derive(Clone, Copy)]
pub(crate) struct ClusterLight {
    pub(crate) position: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) intensity: f32,
}

pub(crate) struct ClusterSystem {
    build_pipeline: ComputePipeline,
    /// Per-frame-in-flight host-writable packed light buffer (avoids stomping in-flight data).
    lights: Vec<StorageBuffer>,
    grid: StorageBuffer,
    index: StorageBuffer,
    /// Capacity of `lights` in light records (grows via reallocation when a frame needs more).
    capacity: usize,
}

impl ClusterSystem {
    /// Build the cluster system, or `None` where compute is unavailable (the feature stays off).
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Option<Self>> {
        if !compute_supported {
            return Ok(None);
        }
        let build_cs = load_compute_shader(
            backend,
            dreamcoast_shader::light_cluster_build_cs_spirv,
            dreamcoast_shader::light_cluster_build_cs_dxil,
            dreamcoast_shader::light_cluster_build_cs_metallib,
            "light_cluster_build",
        )?;
        let build_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: build_cs,
            compute_entry: "csBuildClusters",
            push_constant_size: 144,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [64, 1, 1],
        })?;
        let capacity = 256usize; // initial light capacity; grows on demand
        let lights = (0..FRAMES_IN_FLIGHT)
            .map(|_| Self::alloc_lights(device, capacity))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let grid = device.create_storage_buffer(&StorageBufferDesc {
            size: (CLUSTER_COUNT * 4) as u64,
            stride: 4,
            indirect: false,
        })?;
        let index = device.create_storage_buffer(&StorageBufferDesc {
            size: (CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4) as u64,
            stride: 4,
            indirect: false,
        })?;
        Ok(Some(Self {
            build_pipeline,
            lights,
            grid,
            index,
            capacity,
        }))
    }

    fn alloc_lights(device: &Device, capacity: usize) -> anyhow::Result<StorageBuffer> {
        // Host-visible: `upload` host-writes this buffer every frame via `StorageBuffer::write`,
        // which the D3D12/VK RHI forbids on a device-local (`_init`) buffer. Mirrors the GPU-skin
        // joint-palette pattern (`skin.rs`). Seed with zeros; the first frame's upload overwrites it.
        let buf = device.create_storage_buffer_host(&StorageBufferDesc {
            size: (capacity * 32) as u64,
            stride: 32,
            indirect: false,
        })?;
        buf.write(&vec![0u8; capacity * 32])?;
        Ok(buf)
    }

    /// Host-write this frame's lights into the fif's light buffer (reallocating if the light
    /// count exceeds the current capacity), returning the (grid, index, light) bindless indices
    /// and the light count for the lighting-pass push. Call once per frame before `record_build`.
    pub(crate) fn upload(
        &mut self,
        device: &Device,
        fif: usize,
        lights: &[ClusterLight],
    ) -> anyhow::Result<(u32, u32, u32, u32)> {
        if lights.len() > self.capacity {
            // Grow every fif buffer so the storage index set stays consistent frame to frame.
            let new_cap = lights.len().next_power_of_two();
            for b in self.lights.iter_mut() {
                *b = Self::alloc_lights(device, new_cap)?;
            }
            self.capacity = new_cap;
        }
        let mut bytes = vec![0u8; lights.len().max(1) * 32];
        for (i, l) in lights.iter().enumerate() {
            let o = i * 32;
            let vals = [
                l.position[0],
                l.position[1],
                l.position[2],
                l.radius,
                l.color[0],
                l.color[1],
                l.color[2],
                l.intensity,
            ];
            for (j, f) in vals.iter().enumerate() {
                bytes[o + j * 4..o + j * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        self.lights[fif].write(&bytes)?;
        Ok((
            self.grid.storage_index(),
            self.index.storage_index(),
            self.lights[fif].storage_index(),
            lights.len() as u32,
        ))
    }

    /// Import the grid + index buffers as external graph resources so the cluster-build pass
    /// sequences before the lighting pass that reads them. Returns their ids.
    pub(crate) fn import(graph: &mut RenderGraph) -> (ResourceId, ResourceId) {
        (
            graph.import_external("cluster_grid"),
            graph.import_external("cluster_index"),
        )
    }

    /// Add the cluster-build compute pass: one thread per cluster tests every light against
    /// the cluster's world-space AABB and appends survivors (global-index order) to that
    /// cluster's slot in the flat index list.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_build<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        grid_ext: ResourceId,
        index_ext: ResourceId,
        fif: usize,
        light_count: u32,
        view_z_row: [f32; 4],
        inv_view_proj: [f32; 16],
        camera_pos: [f32; 3],
        z_near: f32,
        z_far: f32,
        screen_w: u32,
        screen_h: u32,
    ) {
        let pipe = &self.build_pipeline;
        let lights = &self.lights[fif];
        let grid = &self.grid;
        let index = &self.index;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "light_cluster",
                storage_writes: vec![grid_ext, index_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&cluster_build_push(
                    lights.storage_index(),
                    grid.storage_index(),
                    index.storage_index(),
                    light_count,
                    view_z_row,
                    inv_view_proj,
                    camera_pos,
                    z_near,
                    z_far,
                    screen_w,
                    screen_h,
                ));
                cmd.dispatch(CLUSTER_COUNT.div_ceil(64), 1, 1);
                // Order the grid/index writes before the lighting pass's storage reads.
                cmd.storage_buffer_barrier(grid);
                cmd.storage_buffer_barrier(index);
                Ok(())
            },
        );
    }
}

/// Pack the cluster-build push block (144 bytes). Layout mirrors `PushConstants` in
/// `light_cluster.slang`: 4×u32 buf indices, view_z_row float4, (z_near, z_far, num_x, num_y),
/// (num_z, screen_w, screen_h, pad), inv_view_proj mat4, camera_pos float4.
#[allow(clippy::too_many_arguments)]
fn cluster_build_push(
    light_buf: u32,
    grid_buf: u32,
    index_buf: u32,
    light_count: u32,
    view_z_row: [f32; 4],
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    z_near: f32,
    z_far: f32,
    screen_w: u32,
    screen_h: u32,
) -> [u8; 144] {
    let mut pc = [0u8; 144];
    pc[0..4].copy_from_slice(&light_buf.to_le_bytes());
    pc[4..8].copy_from_slice(&grid_buf.to_le_bytes());
    pc[8..12].copy_from_slice(&index_buf.to_le_bytes());
    pc[12..16].copy_from_slice(&light_count.to_le_bytes());
    for (i, f) in view_z_row.iter().enumerate() {
        let o = 16 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[32..36].copy_from_slice(&z_near.to_le_bytes());
    pc[36..40].copy_from_slice(&z_far.to_le_bytes());
    pc[40..44].copy_from_slice(&CLUSTER_X.to_le_bytes());
    pc[44..48].copy_from_slice(&CLUSTER_Y.to_le_bytes());
    pc[48..52].copy_from_slice(&CLUSTER_Z.to_le_bytes());
    pc[52..56].copy_from_slice(&screen_w.to_le_bytes());
    pc[56..60].copy_from_slice(&screen_h.to_le_bytes());
    // pc[60..64] pad0 = 0
    for (i, f) in inv_view_proj.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in camera_pos.iter().enumerate() {
        let o = 128 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc
}
