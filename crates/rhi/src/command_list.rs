//! Phase 15 M4 (option B) — a **backend-agnostic command-list IR**.
//!
//! The render-graph (record) thread builds a [`CommandList`] — a flat, `Send`
//! vector of [`RhiCommand`]s mirroring the [`CommandBuffer`] API — instead of
//! calling the backend command buffer directly. A single RHI thread later
//! [`CommandList::translate`]s it onto a real [`CommandBuffer`] and submits. Because
//! the list is pure data, it crosses threads freely and independent passes can
//! build separate lists in parallel (M4 B4) — none of which requires the backend's
//! `Rc<DeviceShared>` to become thread-safe.
//!
//! **Resource references.** Commands that bind a resource store a [`ResPtr`] — a
//! raw `*const` to the borrowed facade resource, wrapped `Send`. This is sound
//! under the M4 handoff contract: the record thread keeps every referenced resource
//! alive until the RHI thread signals the frame done (the per-frame fence), so the
//! pointer is always valid when `translate` dereferences it, and only one thread
//! touches the list at a time (ownership passes with the list). In B1/B2 record and
//! translate run on the same thread, so validity is trivial.
//!
//! **Status (B1):** this module is *additive* — the renderer still records directly
//! into a `CommandBuffer`. B2 routes the passes through [`CommandList`]; B3 moves
//! `translate` + submit onto the RHI thread.

use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::{
    Buffer, CommandBuffer, ComputePipeline, Cubemap, DepthBuffer, GraphicsPipeline, QueryHeap,
    RaytracingPipeline, RenderTarget, Result, StorageBuffer, Swapchain, Volume,
};

/// A `Send` raw pointer to a borrowed facade resource, valid for the frame under
/// the M4 handoff contract (the recorder keeps the resource alive until the RHI
/// thread finishes the frame). Copy so commands can hold it inline.
pub struct ResPtr<T>(*const T);

impl<T> Clone for ResPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ResPtr<T> {}

// SAFETY: the pointee outlives every use of the pointer per the handoff contract
// (see the module docs); the list is owned by exactly one thread at a time, so the
// raw pointer is never used to alias across threads concurrently.
unsafe impl<T> Send for ResPtr<T> {}

impl<T> ResPtr<T> {
    #[inline]
    fn new(r: &T) -> Self {
        Self(r as *const T)
    }
    /// # Safety
    /// The pointee must still be alive (guaranteed by the handoff contract).
    #[inline]
    unsafe fn get<'a>(self) -> &'a T {
        unsafe { &*self.0 }
    }
}

/// One recorded command, mirroring a [`CommandBuffer`] method. Holds only `Send`
/// data (primitives, `Copy` descriptors, [`ResPtr`]s, and offsets into the list's
/// push-constant / target arenas).
pub enum RhiCommand {
    Begin,
    End,
    ResetQueries {
        heap: ResPtr<QueryHeap>,
        first: u32,
        count: u32,
    },
    WriteTimestamp {
        heap: ResPtr<QueryHeap>,
        index: u32,
    },
    ResolveQueries {
        heap: ResPtr<QueryHeap>,
        count: u32,
    },
    BeginDebugLabel {
        off: u32,
        len: u32,
    },
    EndDebugLabel,
    TransitionToRenderTarget {
        swapchain: ResPtr<Swapchain>,
        image_index: u32,
    },
    TransitionToPresent {
        swapchain: ResPtr<Swapchain>,
        image_index: u32,
    },
    BeginRendering {
        swapchain: ResPtr<Swapchain>,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<ResPtr<DepthBuffer>>,
    },
    BeginRenderingTarget {
        target: ResPtr<RenderTarget>,
        color_clear: Option<ClearColor>,
        depth: Option<ResPtr<DepthBuffer>>,
    },
    BeginRenderingTargets {
        /// Range into [`CommandList::targets`].
        off: u32,
        len: u32,
        depth: Option<ResPtr<DepthBuffer>>,
    },
    SetGlobals {
        buffer: ResPtr<Buffer>,
        offset: u64,
    },
    BeginRenderingDepthOnly {
        depth: ResPtr<DepthBuffer>,
    },
    DepthToRenderTarget {
        depth: ResPtr<DepthBuffer>,
    },
    DepthToSampled {
        depth: ResPtr<DepthBuffer>,
    },
    CubeToColor {
        cube: ResPtr<Cubemap>,
    },
    CubeToSampled {
        cube: ResPtr<Cubemap>,
    },
    BeginRenderingCubeFace {
        cube: ResPtr<Cubemap>,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    },
    BeginRenderingCubeFaceDepth {
        cube: ResPtr<Cubemap>,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: ResPtr<DepthBuffer>,
    },
    EndRendering,
    CopySwapchainToBuffer {
        swapchain: ResPtr<Swapchain>,
        image_index: u32,
        buffer: ResPtr<Buffer>,
    },
    SetViewportScissor {
        swapchain: ResPtr<Swapchain>,
    },
    SetViewportScissorExtent {
        extent: Extent2D,
    },
    RtToRenderTarget {
        target: ResPtr<RenderTarget>,
    },
    RtToSampled {
        target: ResPtr<RenderTarget>,
    },
    AliasingBarrier {
        target: ResPtr<RenderTarget>,
    },
    BindGraphicsPipeline {
        pipeline: ResPtr<GraphicsPipeline>,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    BindComputePipeline {
        pipeline: ResPtr<ComputePipeline>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    PushConstantsCompute {
        off: u32,
        len: u32,
    },
    BindRaytracingPipeline {
        pipeline: ResPtr<RaytracingPipeline>,
    },
    PushConstantsRt {
        off: u32,
        len: u32,
    },
    TraceRays {
        pipeline: ResPtr<RaytracingPipeline>,
        width: u32,
        height: u32,
    },
    RtToStorage {
        target: ResPtr<RenderTarget>,
    },
    VolumeToStorage {
        volume: ResPtr<Volume>,
    },
    VolumeToSampled {
        volume: ResPtr<Volume>,
    },
    StorageToSampled {
        target: ResPtr<RenderTarget>,
    },
    StorageBufferBarrier {
        buffer: ResPtr<StorageBuffer>,
    },
    StorageBufferBarrierCompute {
        buffer: ResPtr<StorageBuffer>,
    },
    StorageBufferToIndirect {
        buffer: ResPtr<StorageBuffer>,
    },
    StorageBufferToStorage {
        buffer: ResPtr<StorageBuffer>,
    },
    DrawIndexedIndirect {
        buffer: ResPtr<StorageBuffer>,
        offset: u64,
        draw_count: u32,
    },
    SetScissor {
        rect: Rect2D,
    },
    BindVertexBuffer {
        buffer: ResPtr<Buffer>,
        stride: u32,
    },
    BindIndexBuffer {
        buffer: ResPtr<Buffer>,
        wide: bool,
    },
    PushConstants {
        off: u32,
        len: u32,
    },
    DrawIndexed {
        index_count: u32,
        first_index: u32,
        vertex_offset: i32,
    },
}

/// A flat, `Send` list of [`RhiCommand`]s plus side arenas for variable-length
/// data (push-constant blobs, MRT target lists, debug-label strings). The
/// recording methods mirror [`CommandBuffer`] so passes migrate by swapping the
/// receiver (M4 B2).
#[derive(Default)]
pub struct CommandList {
    cmds: Vec<RhiCommand>,
    /// Arena for push-constant byte blobs; commands store `(off, len)` ranges.
    push: Vec<u8>,
    /// Arena for `begin_rendering_targets` MRT lists.
    targets: Vec<(ResPtr<RenderTarget>, Option<ClearColor>)>,
    /// Arena for debug-label strings.
    labels: Vec<u8>,
}

// SAFETY: every field is `Send` (the `ResPtr`s are `Send` by the handoff contract).
unsafe impl Send for CommandList {}

impl CommandList {
    /// A fresh, empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear for reuse next frame (keeps allocations).
    pub fn clear(&mut self) {
        self.cmds.clear();
        self.push.clear();
        self.targets.clear();
        self.labels.clear();
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.cmds.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    #[inline]
    fn push_blob(&mut self, data: &[u8]) -> (u32, u32) {
        let off = self.push.len() as u32;
        self.push.extend_from_slice(data);
        (off, data.len() as u32)
    }

    // --- recording API (mirrors CommandBuffer) -------------------------------

    pub fn begin(&mut self) {
        self.cmds.push(RhiCommand::Begin);
    }
    pub fn end(&mut self) {
        self.cmds.push(RhiCommand::End);
    }
    pub fn reset_queries(&mut self, heap: &QueryHeap, first: u32, count: u32) {
        self.cmds.push(RhiCommand::ResetQueries {
            heap: ResPtr::new(heap),
            first,
            count,
        });
    }
    pub fn write_timestamp(&mut self, heap: &QueryHeap, index: u32) {
        self.cmds.push(RhiCommand::WriteTimestamp {
            heap: ResPtr::new(heap),
            index,
        });
    }
    pub fn resolve_queries(&mut self, heap: &QueryHeap, count: u32) {
        self.cmds.push(RhiCommand::ResolveQueries {
            heap: ResPtr::new(heap),
            count,
        });
    }
    pub fn begin_debug_label(&mut self, name: &str) {
        let off = self.labels.len() as u32;
        self.labels.extend_from_slice(name.as_bytes());
        let len = name.len() as u32;
        self.cmds.push(RhiCommand::BeginDebugLabel { off, len });
    }
    pub fn end_debug_label(&mut self) {
        self.cmds.push(RhiCommand::EndDebugLabel);
    }
    pub fn transition_to_render_target(&mut self, swapchain: &Swapchain, image_index: u32) {
        self.cmds.push(RhiCommand::TransitionToRenderTarget {
            swapchain: ResPtr::new(swapchain),
            image_index,
        });
    }
    pub fn transition_to_present(&mut self, swapchain: &Swapchain, image_index: u32) {
        self.cmds.push(RhiCommand::TransitionToPresent {
            swapchain: ResPtr::new(swapchain),
            image_index,
        });
    }
    pub fn begin_rendering(
        &mut self,
        swapchain: &Swapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        self.cmds.push(RhiCommand::BeginRendering {
            swapchain: ResPtr::new(swapchain),
            image_index,
            color_clear,
            depth: depth.map(ResPtr::new),
        });
    }
    pub fn begin_rendering_target(
        &mut self,
        target: &RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        self.cmds.push(RhiCommand::BeginRenderingTarget {
            target: ResPtr::new(target),
            color_clear,
            depth: depth.map(ResPtr::new),
        });
    }
    pub fn begin_rendering_targets(
        &mut self,
        targets: &[(&RenderTarget, Option<ClearColor>)],
        depth: Option<&DepthBuffer>,
    ) {
        let off = self.targets.len() as u32;
        for (t, c) in targets {
            self.targets.push((ResPtr::new(*t), *c));
        }
        let len = targets.len() as u32;
        self.cmds.push(RhiCommand::BeginRenderingTargets {
            off,
            len,
            depth: depth.map(ResPtr::new),
        });
    }
    pub fn set_globals(&mut self, buffer: &Buffer, offset: u64) {
        self.cmds.push(RhiCommand::SetGlobals {
            buffer: ResPtr::new(buffer),
            offset,
        });
    }
    pub fn begin_rendering_depth_only(&mut self, depth: &DepthBuffer) {
        self.cmds.push(RhiCommand::BeginRenderingDepthOnly {
            depth: ResPtr::new(depth),
        });
    }
    pub fn depth_to_render_target(&mut self, depth: &DepthBuffer) {
        self.cmds.push(RhiCommand::DepthToRenderTarget {
            depth: ResPtr::new(depth),
        });
    }
    pub fn depth_to_sampled(&mut self, depth: &DepthBuffer) {
        self.cmds.push(RhiCommand::DepthToSampled {
            depth: ResPtr::new(depth),
        });
    }
    pub fn cube_to_color(&mut self, cube: &Cubemap) {
        self.cmds.push(RhiCommand::CubeToColor {
            cube: ResPtr::new(cube),
        });
    }
    pub fn cube_to_sampled(&mut self, cube: &Cubemap) {
        self.cmds.push(RhiCommand::CubeToSampled {
            cube: ResPtr::new(cube),
        });
    }
    pub fn begin_rendering_cube_face(
        &mut self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        self.cmds.push(RhiCommand::BeginRenderingCubeFace {
            cube: ResPtr::new(cube),
            face,
            mip,
            clear,
        });
    }
    pub fn begin_rendering_cube_face_depth(
        &mut self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &DepthBuffer,
    ) {
        self.cmds.push(RhiCommand::BeginRenderingCubeFaceDepth {
            cube: ResPtr::new(cube),
            face,
            mip,
            clear,
            depth: ResPtr::new(depth),
        });
    }
    pub fn end_rendering(&mut self) {
        self.cmds.push(RhiCommand::EndRendering);
    }
    pub fn copy_swapchain_to_buffer(
        &mut self,
        swapchain: &Swapchain,
        image_index: u32,
        buffer: &Buffer,
    ) {
        self.cmds.push(RhiCommand::CopySwapchainToBuffer {
            swapchain: ResPtr::new(swapchain),
            image_index,
            buffer: ResPtr::new(buffer),
        });
    }
    pub fn set_viewport_scissor(&mut self, swapchain: &Swapchain) {
        self.cmds.push(RhiCommand::SetViewportScissor {
            swapchain: ResPtr::new(swapchain),
        });
    }
    pub fn set_viewport_scissor_extent(&mut self, extent: Extent2D) {
        self.cmds
            .push(RhiCommand::SetViewportScissorExtent { extent });
    }
    pub fn rt_to_render_target(&mut self, target: &RenderTarget) {
        self.cmds.push(RhiCommand::RtToRenderTarget {
            target: ResPtr::new(target),
        });
    }
    pub fn rt_to_sampled(&mut self, target: &RenderTarget) {
        self.cmds.push(RhiCommand::RtToSampled {
            target: ResPtr::new(target),
        });
    }
    pub fn aliasing_barrier(&mut self, target: &RenderTarget) {
        self.cmds.push(RhiCommand::AliasingBarrier {
            target: ResPtr::new(target),
        });
    }
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        self.cmds.push(RhiCommand::BindGraphicsPipeline {
            pipeline: ResPtr::new(pipeline),
        });
    }
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.cmds.push(RhiCommand::Draw {
            vertex_count,
            instance_count,
        });
    }
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.cmds.push(RhiCommand::BindComputePipeline {
            pipeline: ResPtr::new(pipeline),
        });
    }
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.cmds.push(RhiCommand::Dispatch { x, y, z });
    }
    pub fn push_constants_compute(&mut self, data: &[u8]) {
        let (off, len) = self.push_blob(data);
        self.cmds.push(RhiCommand::PushConstantsCompute { off, len });
    }
    pub fn bind_raytracing_pipeline(&mut self, pipeline: &RaytracingPipeline) {
        self.cmds.push(RhiCommand::BindRaytracingPipeline {
            pipeline: ResPtr::new(pipeline),
        });
    }
    pub fn push_constants_rt(&mut self, data: &[u8]) {
        let (off, len) = self.push_blob(data);
        self.cmds.push(RhiCommand::PushConstantsRt { off, len });
    }
    pub fn trace_rays(&mut self, pipeline: &RaytracingPipeline, width: u32, height: u32) {
        self.cmds.push(RhiCommand::TraceRays {
            pipeline: ResPtr::new(pipeline),
            width,
            height,
        });
    }
    pub fn rt_to_storage(&mut self, target: &RenderTarget) {
        self.cmds.push(RhiCommand::RtToStorage {
            target: ResPtr::new(target),
        });
    }
    pub fn volume_to_storage(&mut self, volume: &Volume) {
        self.cmds.push(RhiCommand::VolumeToStorage {
            volume: ResPtr::new(volume),
        });
    }
    pub fn volume_to_sampled(&mut self, volume: &Volume) {
        self.cmds.push(RhiCommand::VolumeToSampled {
            volume: ResPtr::new(volume),
        });
    }
    pub fn storage_to_sampled(&mut self, target: &RenderTarget) {
        self.cmds.push(RhiCommand::StorageToSampled {
            target: ResPtr::new(target),
        });
    }
    pub fn storage_buffer_barrier(&mut self, buffer: &StorageBuffer) {
        self.cmds.push(RhiCommand::StorageBufferBarrier {
            buffer: ResPtr::new(buffer),
        });
    }
    pub fn storage_buffer_barrier_compute(&mut self, buffer: &StorageBuffer) {
        self.cmds.push(RhiCommand::StorageBufferBarrierCompute {
            buffer: ResPtr::new(buffer),
        });
    }
    pub fn storage_buffer_to_indirect(&mut self, buffer: &StorageBuffer) {
        self.cmds.push(RhiCommand::StorageBufferToIndirect {
            buffer: ResPtr::new(buffer),
        });
    }
    pub fn storage_buffer_to_storage(&mut self, buffer: &StorageBuffer) {
        self.cmds.push(RhiCommand::StorageBufferToStorage {
            buffer: ResPtr::new(buffer),
        });
    }
    pub fn draw_indexed_indirect(&mut self, buffer: &StorageBuffer, offset: u64, draw_count: u32) {
        self.cmds.push(RhiCommand::DrawIndexedIndirect {
            buffer: ResPtr::new(buffer),
            offset,
            draw_count,
        });
    }
    pub fn set_scissor(&mut self, rect: Rect2D) {
        self.cmds.push(RhiCommand::SetScissor { rect });
    }
    pub fn bind_vertex_buffer(&mut self, buffer: &Buffer, stride: u32) {
        self.cmds.push(RhiCommand::BindVertexBuffer {
            buffer: ResPtr::new(buffer),
            stride,
        });
    }
    pub fn bind_index_buffer(&mut self, buffer: &Buffer, wide: bool) {
        self.cmds.push(RhiCommand::BindIndexBuffer {
            buffer: ResPtr::new(buffer),
            wide,
        });
    }
    pub fn push_constants(&mut self, data: &[u8]) {
        let (off, len) = self.push_blob(data);
        self.cmds.push(RhiCommand::PushConstants { off, len });
    }
    pub fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.cmds.push(RhiCommand::DrawIndexed {
            index_count,
            first_index,
            vertex_offset,
        });
    }

    // --- translation (RHI thread) --------------------------------------------

    /// Replay every recorded command onto a real backend [`CommandBuffer`].
    ///
    /// # Safety contract
    /// Every resource referenced by the list must still be alive (guaranteed by the
    /// M4 handoff: the recorder keeps them alive until the frame's fence signals).
    pub fn translate(&self, cmd: &CommandBuffer) -> Result<()> {
        let blob = |off: u32, len: u32| &self.push[off as usize..(off + len) as usize];
        for c in &self.cmds {
            match *c {
                RhiCommand::Begin => cmd.begin()?,
                RhiCommand::End => cmd.end()?,
                RhiCommand::ResetQueries { heap, first, count } => {
                    cmd.reset_queries(unsafe { heap.get() }, first, count)
                }
                RhiCommand::WriteTimestamp { heap, index } => {
                    cmd.write_timestamp(unsafe { heap.get() }, index)
                }
                RhiCommand::ResolveQueries { heap, count } => {
                    cmd.resolve_queries(unsafe { heap.get() }, count)
                }
                RhiCommand::BeginDebugLabel { off, len } => {
                    let s = std::str::from_utf8(&self.labels[off as usize..(off + len) as usize])
                        .unwrap_or("");
                    cmd.begin_debug_label(s)
                }
                RhiCommand::EndDebugLabel => cmd.end_debug_label(),
                RhiCommand::TransitionToRenderTarget {
                    swapchain,
                    image_index,
                } => cmd.transition_to_render_target(unsafe { swapchain.get() }, image_index),
                RhiCommand::TransitionToPresent {
                    swapchain,
                    image_index,
                } => cmd.transition_to_present(unsafe { swapchain.get() }, image_index),
                RhiCommand::BeginRendering {
                    swapchain,
                    image_index,
                    color_clear,
                    depth,
                } => cmd.begin_rendering(
                    unsafe { swapchain.get() },
                    image_index,
                    color_clear,
                    depth.map(|d| unsafe { d.get() }),
                ),
                RhiCommand::BeginRenderingTarget {
                    target,
                    color_clear,
                    depth,
                } => cmd.begin_rendering_target(
                    unsafe { target.get() },
                    color_clear,
                    depth.map(|d| unsafe { d.get() }),
                ),
                RhiCommand::BeginRenderingTargets { off, len, depth } => {
                    let slice = &self.targets[off as usize..(off + len) as usize];
                    let resolved: Vec<(&RenderTarget, Option<ClearColor>)> = slice
                        .iter()
                        .map(|(t, c)| (unsafe { t.get() }, *c))
                        .collect();
                    cmd.begin_rendering_targets(&resolved, depth.map(|d| unsafe { d.get() }))
                }
                RhiCommand::SetGlobals { buffer, offset } => {
                    cmd.set_globals(unsafe { buffer.get() }, offset)
                }
                RhiCommand::BeginRenderingDepthOnly { depth } => {
                    cmd.begin_rendering_depth_only(unsafe { depth.get() })
                }
                RhiCommand::DepthToRenderTarget { depth } => {
                    cmd.depth_to_render_target(unsafe { depth.get() })
                }
                RhiCommand::DepthToSampled { depth } => {
                    cmd.depth_to_sampled(unsafe { depth.get() })
                }
                RhiCommand::CubeToColor { cube } => cmd.cube_to_color(unsafe { cube.get() }),
                RhiCommand::CubeToSampled { cube } => cmd.cube_to_sampled(unsafe { cube.get() }),
                RhiCommand::BeginRenderingCubeFace {
                    cube,
                    face,
                    mip,
                    clear,
                } => cmd.begin_rendering_cube_face(unsafe { cube.get() }, face, mip, clear),
                RhiCommand::BeginRenderingCubeFaceDepth {
                    cube,
                    face,
                    mip,
                    clear,
                    depth,
                } => cmd.begin_rendering_cube_face_depth(
                    unsafe { cube.get() },
                    face,
                    mip,
                    clear,
                    unsafe { depth.get() },
                ),
                RhiCommand::EndRendering => cmd.end_rendering(),
                RhiCommand::CopySwapchainToBuffer {
                    swapchain,
                    image_index,
                    buffer,
                } => cmd.copy_swapchain_to_buffer(unsafe { swapchain.get() }, image_index, unsafe {
                    buffer.get()
                }),
                RhiCommand::SetViewportScissor { swapchain } => {
                    cmd.set_viewport_scissor(unsafe { swapchain.get() })
                }
                RhiCommand::SetViewportScissorExtent { extent } => {
                    cmd.set_viewport_scissor_extent(extent)
                }
                RhiCommand::RtToRenderTarget { target } => {
                    cmd.rt_to_render_target(unsafe { target.get() })
                }
                RhiCommand::RtToSampled { target } => cmd.rt_to_sampled(unsafe { target.get() }),
                RhiCommand::AliasingBarrier { target } => {
                    cmd.aliasing_barrier(unsafe { target.get() })
                }
                RhiCommand::BindGraphicsPipeline { pipeline } => {
                    cmd.bind_graphics_pipeline(unsafe { pipeline.get() })
                }
                RhiCommand::Draw {
                    vertex_count,
                    instance_count,
                } => cmd.draw(vertex_count, instance_count),
                RhiCommand::BindComputePipeline { pipeline } => {
                    cmd.bind_compute_pipeline(unsafe { pipeline.get() })
                }
                RhiCommand::Dispatch { x, y, z } => cmd.dispatch(x, y, z),
                RhiCommand::PushConstantsCompute { off, len } => {
                    cmd.push_constants_compute(blob(off, len))
                }
                RhiCommand::BindRaytracingPipeline { pipeline } => {
                    cmd.bind_raytracing_pipeline(unsafe { pipeline.get() })
                }
                RhiCommand::PushConstantsRt { off, len } => cmd.push_constants_rt(blob(off, len)),
                RhiCommand::TraceRays {
                    pipeline,
                    width,
                    height,
                } => cmd.trace_rays(unsafe { pipeline.get() }, width, height),
                RhiCommand::RtToStorage { target } => cmd.rt_to_storage(unsafe { target.get() }),
                RhiCommand::VolumeToStorage { volume } => {
                    cmd.volume_to_storage(unsafe { volume.get() })
                }
                RhiCommand::VolumeToSampled { volume } => {
                    cmd.volume_to_sampled(unsafe { volume.get() })
                }
                RhiCommand::StorageToSampled { target } => {
                    cmd.storage_to_sampled(unsafe { target.get() })
                }
                RhiCommand::StorageBufferBarrier { buffer } => {
                    cmd.storage_buffer_barrier(unsafe { buffer.get() })
                }
                RhiCommand::StorageBufferBarrierCompute { buffer } => {
                    cmd.storage_buffer_barrier_compute(unsafe { buffer.get() })
                }
                RhiCommand::StorageBufferToIndirect { buffer } => {
                    cmd.storage_buffer_to_indirect(unsafe { buffer.get() })
                }
                RhiCommand::StorageBufferToStorage { buffer } => {
                    cmd.storage_buffer_to_storage(unsafe { buffer.get() })
                }
                RhiCommand::DrawIndexedIndirect {
                    buffer,
                    offset,
                    draw_count,
                } => cmd.draw_indexed_indirect(unsafe { buffer.get() }, offset, draw_count),
                RhiCommand::SetScissor { rect } => cmd.set_scissor(rect),
                RhiCommand::BindVertexBuffer { buffer, stride } => {
                    cmd.bind_vertex_buffer(unsafe { buffer.get() }, stride)
                }
                RhiCommand::BindIndexBuffer { buffer, wide } => {
                    cmd.bind_index_buffer(unsafe { buffer.get() }, wide)
                }
                RhiCommand::PushConstants { off, len } => cmd.push_constants(blob(off, len)),
                RhiCommand::DrawIndexed {
                    index_count,
                    first_index,
                    vertex_offset,
                } => cmd.draw_indexed(index_count, first_index, vertex_offset),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_commands_and_arenas() {
        let mut list = CommandList::new();
        assert!(list.is_empty());
        list.begin();
        list.draw(3, 1);
        list.push_constants(&[1, 2, 3, 4]);
        list.dispatch(8, 1, 1);
        list.push_constants_compute(&[9, 9]);
        list.draw_indexed(36, 0, 0);
        list.end();
        // 7 commands recorded, push arena holds both blobs back to back.
        assert_eq!(list.len(), 7);
        assert_eq!(list.push, vec![1, 2, 3, 4, 9, 9]);
    }

    #[test]
    fn clear_resets_but_keeps_capacity() {
        let mut list = CommandList::new();
        list.begin();
        list.push_constants(&[1, 2, 3]);
        list.end();
        assert_eq!(list.len(), 3);
        list.clear();
        assert!(list.is_empty());
        assert!(list.push.is_empty());
    }

    #[test]
    fn command_list_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CommandList>();
        assert_send::<RhiCommand>();
    }
}
