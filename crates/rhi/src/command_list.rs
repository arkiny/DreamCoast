//! Phase 15 M4 (option B) — a **backend-agnostic command-list IR** + the
//! [`Recorder`] trait that lets a pass record either *immediately* onto a real
//! [`CommandBuffer`] or *deferred* into a `Send` [`CommandList`].
//!
//! The render-graph (record) thread records into a [`CommandList`] — a flat, `Send`
//! vector of [`RhiCommand`]s mirroring the [`CommandBuffer`] API — and a single RHI
//! thread later [`CommandList::translate`]s it onto a real [`CommandBuffer`] and
//! submits. Because the list is pure data it crosses threads freely and independent
//! passes can build separate lists in parallel (M4 B4) — none of which requires the
//! backend's `Rc<DeviceShared>` to become thread-safe.
//!
//! **One recording API, two targets.** Passes and helpers take `&dyn Recorder`.
//! [`CommandBuffer`] implements it by forwarding to its inherent immediate methods
//! (used by the compute-queue / IBL-capture paths that submit directly); a
//! [`CommandList`] implements it by appending IR (used by the render graph). The
//! trait takes `&self` (with interior mutability in `CommandList`) so a shared
//! `&CommandBuffer` — which is all the direct paths hold — satisfies it too.
//!
//! **Resource references.** Commands store a [`ResPtr`] — a `Send`-wrapped `*const`
//! to the borrowed facade resource — valid for the frame under the M4 handoff
//! contract: the recorder keeps every referenced resource alive until the RHI
//! thread signals the frame done, so the pointer is always valid at `translate`.

use std::cell::RefCell;

use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::{
    Buffer, CommandBuffer, ComputePipeline, Cubemap, DepthBuffer, GraphicsPipeline, MeshPipeline,
    QueryHeap, RaytracingPipeline, RenderTarget, Result, StorageBuffer, Swapchain, Volume,
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

/// The recording API shared by immediate ([`CommandBuffer`]) and deferred
/// ([`CommandList`]) targets. Mirrors the [`CommandBuffer`] command surface (minus
/// `begin`/`end`, which the frame loop calls directly on the real buffer). `&self`
/// so a shared `&CommandBuffer` satisfies it.
pub trait Recorder {
    fn reset_queries(&self, heap: &QueryHeap, first: u32, count: u32);
    fn write_timestamp(&self, heap: &QueryHeap, index: u32);
    fn resolve_queries(&self, heap: &QueryHeap, count: u32);
    fn begin_debug_label(&self, name: &str);
    fn end_debug_label(&self);
    fn transition_to_render_target(&self, swapchain: &Swapchain, image_index: u32);
    fn transition_to_present(&self, swapchain: &Swapchain, image_index: u32);
    fn begin_rendering(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    );
    fn begin_rendering_target(
        &self,
        target: &RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    );
    fn begin_rendering_targets(
        &self,
        targets: &[(&RenderTarget, Option<ClearColor>)],
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    );
    fn set_globals(&self, buffer: &Buffer, offset: u64);
    fn begin_rendering_depth_only(&self, depth: &DepthBuffer);
    fn depth_to_render_target(&self, depth: &DepthBuffer);
    fn depth_to_sampled(&self, depth: &DepthBuffer);
    fn cube_to_color(&self, cube: &Cubemap);
    fn cube_to_sampled(&self, cube: &Cubemap);
    fn begin_rendering_cube_face(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    );
    fn begin_rendering_cube_face_depth(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &DepthBuffer,
    );
    fn end_rendering(&self);
    fn copy_swapchain_to_buffer(&self, swapchain: &Swapchain, image_index: u32, buffer: &Buffer);
    fn set_viewport_scissor(&self, swapchain: &Swapchain);
    fn set_viewport_scissor_extent(&self, extent: Extent2D);
    fn rt_to_render_target(&self, target: &RenderTarget);
    fn rt_to_sampled(&self, target: &RenderTarget);
    fn aliasing_barrier(&self, target: &RenderTarget);
    fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline);
    fn draw(&self, vertex_count: u32, instance_count: u32);
    fn bind_compute_pipeline(&self, pipeline: &ComputePipeline);
    fn dispatch(&self, x: u32, y: u32, z: u32);
    fn push_constants_compute(&self, data: &[u8]);
    fn bind_raytracing_pipeline(&self, pipeline: &RaytracingPipeline);
    fn push_constants_rt(&self, data: &[u8]);
    fn trace_rays(&self, pipeline: &RaytracingPipeline, width: u32, height: u32);
    fn rt_to_storage(&self, target: &RenderTarget);
    fn volume_to_storage(&self, volume: &Volume);
    fn volume_to_sampled(&self, volume: &Volume);
    fn storage_to_sampled(&self, target: &RenderTarget);
    fn storage_buffer_barrier(&self, buffer: &StorageBuffer);
    fn storage_buffer_barrier_compute(&self, buffer: &StorageBuffer);
    fn storage_buffer_to_indirect(&self, buffer: &StorageBuffer);
    fn storage_buffer_to_storage(&self, buffer: &StorageBuffer);
    fn draw_indexed_indirect(&self, buffer: &StorageBuffer, offset: u64, draw_count: u32);
    fn dispatch_indirect(&self, buffer: &StorageBuffer, offset: u64);
    fn set_scissor(&self, rect: Rect2D);
    fn set_viewport_scissor_rect(&self, rect: Rect2D);
    fn bind_vertex_buffer(&self, buffer: &Buffer, stride: u32);
    fn bind_index_buffer(&self, buffer: &Buffer, wide: bool);
    fn push_constants(&self, data: &[u8]);
    fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32);
    // Phase 14 Track B: mesh-shader (HW virtual-geometry) commands.
    fn bind_mesh_pipeline(&self, pipeline: &MeshPipeline);
    fn draw_mesh_tasks(&self, x: u32, y: u32, z: u32);
    fn push_constants_mesh(&self, data: &[u8]);
    fn draw_mesh_tasks_indirect(&self, buffer: &StorageBuffer, offset: u64);
}

/// [`CommandBuffer`] records immediately by forwarding to its inherent methods.
impl Recorder for CommandBuffer {
    fn reset_queries(&self, heap: &QueryHeap, first: u32, count: u32) {
        CommandBuffer::reset_queries(self, heap, first, count)
    }
    fn write_timestamp(&self, heap: &QueryHeap, index: u32) {
        CommandBuffer::write_timestamp(self, heap, index)
    }
    fn resolve_queries(&self, heap: &QueryHeap, count: u32) {
        CommandBuffer::resolve_queries(self, heap, count)
    }
    fn begin_debug_label(&self, name: &str) {
        CommandBuffer::begin_debug_label(self, name)
    }
    fn end_debug_label(&self) {
        CommandBuffer::end_debug_label(self)
    }
    fn transition_to_render_target(&self, swapchain: &Swapchain, image_index: u32) {
        CommandBuffer::transition_to_render_target(self, swapchain, image_index)
    }
    fn transition_to_present(&self, swapchain: &Swapchain, image_index: u32) {
        CommandBuffer::transition_to_present(self, swapchain, image_index)
    }
    fn begin_rendering(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        CommandBuffer::begin_rendering(self, swapchain, image_index, color_clear, depth)
    }
    fn begin_rendering_target(
        &self,
        target: &RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    ) {
        CommandBuffer::begin_rendering_target(self, target, color_clear, depth, depth_clear)
    }
    fn begin_rendering_targets(
        &self,
        targets: &[(&RenderTarget, Option<ClearColor>)],
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    ) {
        CommandBuffer::begin_rendering_targets(self, targets, depth, depth_clear)
    }
    fn set_globals(&self, buffer: &Buffer, offset: u64) {
        CommandBuffer::set_globals(self, buffer, offset)
    }
    fn begin_rendering_depth_only(&self, depth: &DepthBuffer) {
        CommandBuffer::begin_rendering_depth_only(self, depth)
    }
    fn depth_to_render_target(&self, depth: &DepthBuffer) {
        CommandBuffer::depth_to_render_target(self, depth)
    }
    fn depth_to_sampled(&self, depth: &DepthBuffer) {
        CommandBuffer::depth_to_sampled(self, depth)
    }
    fn cube_to_color(&self, cube: &Cubemap) {
        CommandBuffer::cube_to_color(self, cube)
    }
    fn cube_to_sampled(&self, cube: &Cubemap) {
        CommandBuffer::cube_to_sampled(self, cube)
    }
    fn begin_rendering_cube_face(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        CommandBuffer::begin_rendering_cube_face(self, cube, face, mip, clear)
    }
    fn begin_rendering_cube_face_depth(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &DepthBuffer,
    ) {
        CommandBuffer::begin_rendering_cube_face_depth(self, cube, face, mip, clear, depth)
    }
    fn end_rendering(&self) {
        CommandBuffer::end_rendering(self)
    }
    fn copy_swapchain_to_buffer(&self, swapchain: &Swapchain, image_index: u32, buffer: &Buffer) {
        CommandBuffer::copy_swapchain_to_buffer(self, swapchain, image_index, buffer)
    }
    fn set_viewport_scissor(&self, swapchain: &Swapchain) {
        CommandBuffer::set_viewport_scissor(self, swapchain)
    }
    fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        CommandBuffer::set_viewport_scissor_extent(self, extent)
    }
    fn rt_to_render_target(&self, target: &RenderTarget) {
        CommandBuffer::rt_to_render_target(self, target)
    }
    fn rt_to_sampled(&self, target: &RenderTarget) {
        CommandBuffer::rt_to_sampled(self, target)
    }
    fn aliasing_barrier(&self, target: &RenderTarget) {
        CommandBuffer::aliasing_barrier(self, target)
    }
    fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        CommandBuffer::bind_graphics_pipeline(self, pipeline)
    }
    fn draw(&self, vertex_count: u32, instance_count: u32) {
        CommandBuffer::draw(self, vertex_count, instance_count)
    }
    fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        CommandBuffer::bind_compute_pipeline(self, pipeline)
    }
    fn dispatch(&self, x: u32, y: u32, z: u32) {
        CommandBuffer::dispatch(self, x, y, z)
    }
    fn push_constants_compute(&self, data: &[u8]) {
        CommandBuffer::push_constants_compute(self, data)
    }
    fn bind_raytracing_pipeline(&self, pipeline: &RaytracingPipeline) {
        CommandBuffer::bind_raytracing_pipeline(self, pipeline)
    }
    fn push_constants_rt(&self, data: &[u8]) {
        CommandBuffer::push_constants_rt(self, data)
    }
    fn trace_rays(&self, pipeline: &RaytracingPipeline, width: u32, height: u32) {
        CommandBuffer::trace_rays(self, pipeline, width, height)
    }
    fn rt_to_storage(&self, target: &RenderTarget) {
        CommandBuffer::rt_to_storage(self, target)
    }
    fn volume_to_storage(&self, volume: &Volume) {
        CommandBuffer::volume_to_storage(self, volume)
    }
    fn volume_to_sampled(&self, volume: &Volume) {
        CommandBuffer::volume_to_sampled(self, volume)
    }
    fn storage_to_sampled(&self, target: &RenderTarget) {
        CommandBuffer::storage_to_sampled(self, target)
    }
    fn storage_buffer_barrier(&self, buffer: &StorageBuffer) {
        CommandBuffer::storage_buffer_barrier(self, buffer)
    }
    fn storage_buffer_barrier_compute(&self, buffer: &StorageBuffer) {
        CommandBuffer::storage_buffer_barrier_compute(self, buffer)
    }
    fn storage_buffer_to_indirect(&self, buffer: &StorageBuffer) {
        CommandBuffer::storage_buffer_to_indirect(self, buffer)
    }
    fn storage_buffer_to_storage(&self, buffer: &StorageBuffer) {
        CommandBuffer::storage_buffer_to_storage(self, buffer)
    }
    fn draw_indexed_indirect(&self, buffer: &StorageBuffer, offset: u64, draw_count: u32) {
        CommandBuffer::draw_indexed_indirect(self, buffer, offset, draw_count)
    }
    fn dispatch_indirect(&self, buffer: &StorageBuffer, offset: u64) {
        CommandBuffer::dispatch_indirect(self, buffer, offset)
    }
    fn set_scissor(&self, rect: Rect2D) {
        CommandBuffer::set_scissor(self, rect)
    }
    fn set_viewport_scissor_rect(&self, rect: Rect2D) {
        CommandBuffer::set_viewport_scissor_rect(self, rect)
    }
    fn bind_vertex_buffer(&self, buffer: &Buffer, stride: u32) {
        CommandBuffer::bind_vertex_buffer(self, buffer, stride)
    }
    fn bind_index_buffer(&self, buffer: &Buffer, wide: bool) {
        CommandBuffer::bind_index_buffer(self, buffer, wide)
    }
    fn push_constants(&self, data: &[u8]) {
        CommandBuffer::push_constants(self, data)
    }
    fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        CommandBuffer::draw_indexed(self, index_count, first_index, vertex_offset)
    }
    fn bind_mesh_pipeline(&self, pipeline: &MeshPipeline) {
        CommandBuffer::bind_mesh_pipeline(self, pipeline)
    }
    fn draw_mesh_tasks(&self, x: u32, y: u32, z: u32) {
        CommandBuffer::draw_mesh_tasks(self, x, y, z)
    }
    fn push_constants_mesh(&self, data: &[u8]) {
        CommandBuffer::push_constants_mesh(self, data)
    }
    fn draw_mesh_tasks_indirect(&self, buffer: &StorageBuffer, offset: u64) {
        CommandBuffer::draw_mesh_tasks_indirect(self, buffer, offset)
    }
}

/// One recorded command, mirroring a [`Recorder`] method. Holds only `Send` data
/// (primitives, `Copy` descriptors, [`ResPtr`]s, and offsets into the list arenas).
pub enum RhiCommand {
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
    // The backbuffer is a *frame-global* target: the swapchain + acquired image
    // index are the same for every backbuffer command in a frame, so they aren't
    // stored per-command — [`CommandList::translate`] supplies them from its
    // context. This is what lets the RHI thread (M4 B3) own the swapchain and
    // resolve the image index at translate time, so the record thread never needs
    // the acquired index to build the IR.
    TransitionToRenderTarget,
    TransitionToPresent,
    BeginRendering {
        color_clear: Option<ClearColor>,
        depth: Option<ResPtr<DepthBuffer>>,
    },
    BeginRenderingTarget {
        target: ResPtr<RenderTarget>,
        color_clear: Option<ClearColor>,
        depth: Option<ResPtr<DepthBuffer>>,
        depth_clear: bool,
    },
    BeginRenderingTargets {
        off: u32,
        len: u32,
        depth: Option<ResPtr<DepthBuffer>>,
        depth_clear: bool,
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
        buffer: ResPtr<Buffer>,
    },
    SetViewportScissor,
    SetViewportScissorExtent {
        extent: Extent2D,
    },
    SetViewportScissorRect {
        rect: Rect2D,
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
    DispatchIndirect {
        buffer: ResPtr<StorageBuffer>,
        offset: u64,
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
    // Phase 14 Track B: mesh-shader commands.
    BindMeshPipeline {
        pipeline: ResPtr<MeshPipeline>,
    },
    DrawMeshTasks {
        x: u32,
        y: u32,
        z: u32,
    },
    PushConstantsMesh {
        off: u32,
        len: u32,
    },
    DrawMeshTasksIndirect {
        buffer: ResPtr<StorageBuffer>,
        offset: u64,
    },
}

#[derive(Default)]
struct Inner {
    cmds: Vec<RhiCommand>,
    /// Arena for push-constant byte blobs; commands store `(off, len)` ranges.
    push: Vec<u8>,
    /// Arena for `begin_rendering_targets` MRT lists.
    targets: Vec<(ResPtr<RenderTarget>, Option<ClearColor>)>,
    /// Arena for debug-label strings.
    labels: Vec<u8>,
}

impl Inner {
    #[inline]
    fn push_blob(&mut self, data: &[u8]) -> (u32, u32) {
        let off = self.push.len() as u32;
        self.push.extend_from_slice(data);
        (off, data.len() as u32)
    }
}

/// A flat, `Send` list of [`RhiCommand`]s the render-graph thread records (via the
/// [`Recorder`] impl) and the RHI thread [`translate`](CommandList::translate)s onto
/// a real [`CommandBuffer`]. Interior mutability (`RefCell`) lets it record through
/// `&self`, matching the trait and the direct-path `&CommandBuffer`.
#[derive(Default)]
pub struct CommandList {
    inner: RefCell<Inner>,
}

// SAFETY: `Inner` is `Send` (the `ResPtr`s are `Send` by the handoff contract); the
// `RefCell` only blocks `Sync`, which is fine — the list is owned by one thread at a
// time and never shared concurrently.
unsafe impl Send for CommandList {}

impl CommandList {
    /// A fresh, empty list.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Backbuffer commands (M4 B3) ---------------------------------------
    //
    // The backbuffer's swapchain + acquired image index are frame-global and
    // resolved at [`translate`](Self::translate). These inherent recorders let the
    // record thread build the IR with *no* swapchain/index in hand — the enabler
    // for the RHI thread owning the swapchain and acquiring off the record thread.
    // (The `Recorder` trait keeps the swapchain-taking signatures so the immediate
    // `CommandBuffer` path still satisfies it; the list just ignores those args.)

    /// Record a backbuffer transition to the render-target state.
    pub fn backbuffer_to_render_target(&self) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::TransitionToRenderTarget);
    }

    /// Record a backbuffer transition to the present state.
    pub fn backbuffer_to_present(&self) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::TransitionToPresent);
    }

    /// Begin rendering into the backbuffer (optional color clear + depth).
    pub fn begin_backbuffer_rendering(
        &self,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRendering {
                color_clear,
                depth: depth.map(ResPtr::new),
            });
    }

    /// Set the viewport + scissor to the backbuffer extent.
    pub fn set_backbuffer_viewport(&self) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::SetViewportScissor);
    }

    /// Append `other`'s commands onto this list, rebasing its arena offsets so the
    /// result is identical to having recorded `other`'s commands directly after this
    /// list's. Used to concatenate per-pass IR buckets — recorded independently (in
    /// parallel on job workers, M4 B4) — back into render-graph schedule order.
    pub fn append(&self, other: CommandList) {
        let mut dst = self.inner.borrow_mut();
        let src = other.inner.into_inner();
        // Offsets in `src`'s commands index `src`'s arenas; after we append those
        // arenas onto `dst`'s, every `src` offset shifts by `dst`'s current length.
        let push_base = dst.push.len() as u32;
        let labels_base = dst.labels.len() as u32;
        let targets_base = dst.targets.len() as u32;
        for mut c in src.cmds {
            match &mut c {
                RhiCommand::PushConstants { off, .. }
                | RhiCommand::PushConstantsCompute { off, .. }
                | RhiCommand::PushConstantsRt { off, .. }
                | RhiCommand::PushConstantsMesh { off, .. } => *off += push_base,
                RhiCommand::BeginDebugLabel { off, .. } => *off += labels_base,
                RhiCommand::BeginRenderingTargets { off, .. } => *off += targets_base,
                _ => {}
            }
            dst.cmds.push(c);
        }
        dst.push.extend_from_slice(&src.push);
        dst.labels.extend_from_slice(&src.labels);
        dst.targets.extend_from_slice(&src.targets);
    }

    /// Clear for reuse next frame (keeps allocations).
    pub fn clear(&self) {
        let mut i = self.inner.borrow_mut();
        i.cmds.clear();
        i.push.clear();
        i.targets.clear();
        i.labels.clear();
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.inner.borrow().cmds.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.inner.borrow().cmds.is_empty()
    }

    /// Replay every recorded command onto a real backend [`CommandBuffer`].
    ///
    /// `swapchain` + `image_index` resolve the frame's backbuffer commands (the IR
    /// stores neither per-command — they're frame-global). The inline path passes
    /// the same swapchain/index it recorded with (behaviour-identical); the RHI
    /// thread (M4 B3) passes its own owned swapchain + freshly acquired index.
    ///
    /// # Safety contract
    /// Every resource referenced by the list must still be alive (guaranteed by the
    /// M4 handoff: the recorder keeps them alive until the frame's fence signals).
    pub fn translate(
        &self,
        cmd: &CommandBuffer,
        swapchain: &Swapchain,
        image_index: u32,
    ) -> Result<()> {
        let inner = self.inner.borrow();
        let blob = |off: u32, len: u32| &inner.push[off as usize..(off + len) as usize];
        for c in &inner.cmds {
            match *c {
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
                    let s = std::str::from_utf8(&inner.labels[off as usize..(off + len) as usize])
                        .unwrap_or("");
                    cmd.begin_debug_label(s)
                }
                RhiCommand::EndDebugLabel => cmd.end_debug_label(),
                RhiCommand::TransitionToRenderTarget => {
                    cmd.transition_to_render_target(swapchain, image_index)
                }
                RhiCommand::TransitionToPresent => {
                    cmd.transition_to_present(swapchain, image_index)
                }
                RhiCommand::BeginRendering { color_clear, depth } => cmd.begin_rendering(
                    swapchain,
                    image_index,
                    color_clear,
                    depth.map(|d| unsafe { d.get() }),
                ),
                RhiCommand::BeginRenderingTarget {
                    target,
                    color_clear,
                    depth,
                    depth_clear,
                } => cmd.begin_rendering_target(
                    unsafe { target.get() },
                    color_clear,
                    depth.map(|d| unsafe { d.get() }),
                    depth_clear,
                ),
                RhiCommand::BeginRenderingTargets {
                    off,
                    len,
                    depth,
                    depth_clear,
                } => {
                    let slice = &inner.targets[off as usize..(off + len) as usize];
                    let resolved: Vec<(&RenderTarget, Option<ClearColor>)> = slice
                        .iter()
                        .map(|(t, c)| (unsafe { t.get() }, *c))
                        .collect();
                    cmd.begin_rendering_targets(
                        &resolved,
                        depth.map(|d| unsafe { d.get() }),
                        depth_clear,
                    )
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
                RhiCommand::CopySwapchainToBuffer { buffer } => {
                    cmd.copy_swapchain_to_buffer(swapchain, image_index, unsafe { buffer.get() })
                }
                RhiCommand::SetViewportScissor => cmd.set_viewport_scissor(swapchain),
                RhiCommand::SetViewportScissorExtent { extent } => {
                    cmd.set_viewport_scissor_extent(extent)
                }
                RhiCommand::SetViewportScissorRect { rect } => cmd.set_viewport_scissor_rect(rect),
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
                RhiCommand::DispatchIndirect { buffer, offset } => {
                    cmd.dispatch_indirect(unsafe { buffer.get() }, offset)
                }
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
                RhiCommand::BindMeshPipeline { pipeline } => {
                    cmd.bind_mesh_pipeline(unsafe { pipeline.get() })
                }
                RhiCommand::DrawMeshTasks { x, y, z } => cmd.draw_mesh_tasks(x, y, z),
                RhiCommand::PushConstantsMesh { off, len } => {
                    cmd.push_constants_mesh(blob(off, len))
                }
                RhiCommand::DrawMeshTasksIndirect { buffer, offset } => {
                    cmd.draw_mesh_tasks_indirect(unsafe { buffer.get() }, offset)
                }
            }
        }
        Ok(())
    }
}

/// [`CommandList`] records by appending IR (interior-mutable, so `&self`).
impl Recorder for CommandList {
    fn reset_queries(&self, heap: &QueryHeap, first: u32, count: u32) {
        self.inner.borrow_mut().cmds.push(RhiCommand::ResetQueries {
            heap: ResPtr::new(heap),
            first,
            count,
        });
    }
    fn write_timestamp(&self, heap: &QueryHeap, index: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::WriteTimestamp {
                heap: ResPtr::new(heap),
                index,
            });
    }
    fn resolve_queries(&self, heap: &QueryHeap, count: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::ResolveQueries {
                heap: ResPtr::new(heap),
                count,
            });
    }
    fn begin_debug_label(&self, name: &str) {
        let mut i = self.inner.borrow_mut();
        let off = i.labels.len() as u32;
        i.labels.extend_from_slice(name.as_bytes());
        let len = name.len() as u32;
        i.cmds.push(RhiCommand::BeginDebugLabel { off, len });
    }
    fn end_debug_label(&self) {
        self.inner.borrow_mut().cmds.push(RhiCommand::EndDebugLabel);
    }
    fn transition_to_render_target(&self, _swapchain: &Swapchain, _image_index: u32) {
        // swapchain + image_index are frame-global (resolved at translate).
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::TransitionToRenderTarget);
    }
    fn transition_to_present(&self, _swapchain: &Swapchain, _image_index: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::TransitionToPresent);
    }
    fn begin_rendering(
        &self,
        _swapchain: &Swapchain,
        _image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRendering {
                color_clear,
                depth: depth.map(ResPtr::new),
            });
    }
    fn begin_rendering_target(
        &self,
        target: &RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    ) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRenderingTarget {
                target: ResPtr::new(target),
                color_clear,
                depth: depth.map(ResPtr::new),
                depth_clear,
            });
    }
    fn begin_rendering_targets(
        &self,
        targets: &[(&RenderTarget, Option<ClearColor>)],
        depth: Option<&DepthBuffer>,
        depth_clear: bool,
    ) {
        let mut i = self.inner.borrow_mut();
        let off = i.targets.len() as u32;
        for (t, c) in targets {
            i.targets.push((ResPtr::new(*t), *c));
        }
        let len = targets.len() as u32;
        i.cmds.push(RhiCommand::BeginRenderingTargets {
            off,
            len,
            depth: depth.map(ResPtr::new),
            depth_clear,
        });
    }
    fn set_globals(&self, buffer: &Buffer, offset: u64) {
        self.inner.borrow_mut().cmds.push(RhiCommand::SetGlobals {
            buffer: ResPtr::new(buffer),
            offset,
        });
    }
    fn begin_rendering_depth_only(&self, depth: &DepthBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRenderingDepthOnly {
                depth: ResPtr::new(depth),
            });
    }
    fn depth_to_render_target(&self, depth: &DepthBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DepthToRenderTarget {
                depth: ResPtr::new(depth),
            });
    }
    fn depth_to_sampled(&self, depth: &DepthBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DepthToSampled {
                depth: ResPtr::new(depth),
            });
    }
    fn cube_to_color(&self, cube: &Cubemap) {
        self.inner.borrow_mut().cmds.push(RhiCommand::CubeToColor {
            cube: ResPtr::new(cube),
        });
    }
    fn cube_to_sampled(&self, cube: &Cubemap) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::CubeToSampled {
                cube: ResPtr::new(cube),
            });
    }
    fn begin_rendering_cube_face(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRenderingCubeFace {
                cube: ResPtr::new(cube),
                face,
                mip,
                clear,
            });
    }
    fn begin_rendering_cube_face_depth(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &DepthBuffer,
    ) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BeginRenderingCubeFaceDepth {
                cube: ResPtr::new(cube),
                face,
                mip,
                clear,
                depth: ResPtr::new(depth),
            });
    }
    fn end_rendering(&self) {
        self.inner.borrow_mut().cmds.push(RhiCommand::EndRendering);
    }
    fn copy_swapchain_to_buffer(&self, _swapchain: &Swapchain, _image_index: u32, buffer: &Buffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::CopySwapchainToBuffer {
                buffer: ResPtr::new(buffer),
            });
    }
    fn set_viewport_scissor(&self, _swapchain: &Swapchain) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::SetViewportScissor);
    }
    fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::SetViewportScissorExtent { extent });
    }
    fn rt_to_render_target(&self, target: &RenderTarget) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::RtToRenderTarget {
                target: ResPtr::new(target),
            });
    }
    fn rt_to_sampled(&self, target: &RenderTarget) {
        self.inner.borrow_mut().cmds.push(RhiCommand::RtToSampled {
            target: ResPtr::new(target),
        });
    }
    fn aliasing_barrier(&self, target: &RenderTarget) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::AliasingBarrier {
                target: ResPtr::new(target),
            });
    }
    fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindGraphicsPipeline {
                pipeline: ResPtr::new(pipeline),
            });
    }
    fn draw(&self, vertex_count: u32, instance_count: u32) {
        self.inner.borrow_mut().cmds.push(RhiCommand::Draw {
            vertex_count,
            instance_count,
        });
    }
    fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindComputePipeline {
                pipeline: ResPtr::new(pipeline),
            });
    }
    fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::Dispatch { x, y, z });
    }
    fn push_constants_compute(&self, data: &[u8]) {
        let mut i = self.inner.borrow_mut();
        let (off, len) = i.push_blob(data);
        i.cmds.push(RhiCommand::PushConstantsCompute { off, len });
    }
    fn bind_raytracing_pipeline(&self, pipeline: &RaytracingPipeline) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindRaytracingPipeline {
                pipeline: ResPtr::new(pipeline),
            });
    }
    fn push_constants_rt(&self, data: &[u8]) {
        let mut i = self.inner.borrow_mut();
        let (off, len) = i.push_blob(data);
        i.cmds.push(RhiCommand::PushConstantsRt { off, len });
    }
    fn trace_rays(&self, pipeline: &RaytracingPipeline, width: u32, height: u32) {
        self.inner.borrow_mut().cmds.push(RhiCommand::TraceRays {
            pipeline: ResPtr::new(pipeline),
            width,
            height,
        });
    }
    fn rt_to_storage(&self, target: &RenderTarget) {
        self.inner.borrow_mut().cmds.push(RhiCommand::RtToStorage {
            target: ResPtr::new(target),
        });
    }
    fn volume_to_storage(&self, volume: &Volume) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::VolumeToStorage {
                volume: ResPtr::new(volume),
            });
    }
    fn volume_to_sampled(&self, volume: &Volume) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::VolumeToSampled {
                volume: ResPtr::new(volume),
            });
    }
    fn storage_to_sampled(&self, target: &RenderTarget) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::StorageToSampled {
                target: ResPtr::new(target),
            });
    }
    fn storage_buffer_barrier(&self, buffer: &StorageBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::StorageBufferBarrier {
                buffer: ResPtr::new(buffer),
            });
    }
    fn storage_buffer_barrier_compute(&self, buffer: &StorageBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::StorageBufferBarrierCompute {
                buffer: ResPtr::new(buffer),
            });
    }
    fn storage_buffer_to_indirect(&self, buffer: &StorageBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::StorageBufferToIndirect {
                buffer: ResPtr::new(buffer),
            });
    }
    fn storage_buffer_to_storage(&self, buffer: &StorageBuffer) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::StorageBufferToStorage {
                buffer: ResPtr::new(buffer),
            });
    }
    fn draw_indexed_indirect(&self, buffer: &StorageBuffer, offset: u64, draw_count: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DrawIndexedIndirect {
                buffer: ResPtr::new(buffer),
                offset,
                draw_count,
            });
    }
    fn dispatch_indirect(&self, buffer: &StorageBuffer, offset: u64) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DispatchIndirect {
                buffer: ResPtr::new(buffer),
                offset,
            });
    }
    fn set_scissor(&self, rect: Rect2D) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::SetScissor { rect });
    }
    fn set_viewport_scissor_rect(&self, rect: Rect2D) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::SetViewportScissorRect { rect });
    }
    fn bind_vertex_buffer(&self, buffer: &Buffer, stride: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindVertexBuffer {
                buffer: ResPtr::new(buffer),
                stride,
            });
    }
    fn bind_index_buffer(&self, buffer: &Buffer, wide: bool) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindIndexBuffer {
                buffer: ResPtr::new(buffer),
                wide,
            });
    }
    fn push_constants(&self, data: &[u8]) {
        let mut i = self.inner.borrow_mut();
        let (off, len) = i.push_blob(data);
        i.cmds.push(RhiCommand::PushConstants { off, len });
    }
    fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.inner.borrow_mut().cmds.push(RhiCommand::DrawIndexed {
            index_count,
            first_index,
            vertex_offset,
        });
    }
    fn bind_mesh_pipeline(&self, pipeline: &MeshPipeline) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::BindMeshPipeline {
                pipeline: ResPtr::new(pipeline),
            });
    }
    fn draw_mesh_tasks(&self, x: u32, y: u32, z: u32) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DrawMeshTasks { x, y, z });
    }
    fn push_constants_mesh(&self, data: &[u8]) {
        let mut i = self.inner.borrow_mut();
        let (off, len) = i.push_blob(data);
        i.cmds.push(RhiCommand::PushConstantsMesh { off, len });
    }
    fn draw_mesh_tasks_indirect(&self, buffer: &StorageBuffer, offset: u64) {
        self.inner
            .borrow_mut()
            .cmds
            .push(RhiCommand::DrawMeshTasksIndirect {
                buffer: ResPtr::new(buffer),
                offset,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_commands_and_arenas() {
        let list = CommandList::new();
        assert!(list.is_empty());
        list.draw(3, 1);
        list.push_constants(&[1, 2, 3, 4]);
        list.dispatch(8, 1, 1);
        list.push_constants_compute(&[9, 9]);
        list.draw_indexed(36, 0, 0);
        assert_eq!(list.len(), 5);
        assert_eq!(list.inner.borrow().push, vec![1, 2, 3, 4, 9, 9]);
    }

    #[test]
    fn records_mesh_commands_into_push_arena() {
        // Phase 14 Track B: the mesh commands record like the others, and
        // push_constants_mesh shares the `push` arena with the graphics/compute pushes.
        let list = CommandList::new();
        list.draw_mesh_tasks(4, 1, 1);
        list.push_constants_mesh(&[7, 7, 7, 7]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.inner.borrow().push, vec![7, 7, 7, 7]);
        // The mesh push offset is rebased on append, like the other push kinds.
        let a = CommandList::new();
        a.push_constants(&[1, 2]);
        let b = CommandList::new();
        b.push_constants_mesh(&[3, 4]);
        a.append(b);
        let inner = a.inner.borrow();
        match inner.cmds.last().unwrap() {
            RhiCommand::PushConstantsMesh { off, len } => {
                assert_eq!(&inner.push[*off as usize..(*off + *len) as usize], &[3, 4]);
            }
            _ => panic!("expected PushConstantsMesh last"),
        }
    }

    #[test]
    fn clear_resets() {
        let list = CommandList::new();
        list.draw(1, 1);
        list.push_constants(&[1, 2, 3]);
        assert_eq!(list.len(), 2);
        list.clear();
        assert!(list.is_empty());
        assert!(list.inner.borrow().push.is_empty());
    }

    #[test]
    fn command_list_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CommandList>();
        assert_send::<RhiCommand>();
    }

    #[test]
    fn append_concatenates_with_rebased_arenas() {
        // Recording into one list vs into two buckets + append must produce the same
        // command stream and arena bytes (with the bucket's offsets rebased).
        let seq = CommandList::new();
        seq.push_constants(&[1, 2]);
        seq.draw(3, 1);
        seq.push_constants_compute(&[9]);
        seq.push_constants(&[3, 4]);

        let a = CommandList::new();
        a.push_constants(&[1, 2]);
        a.draw(3, 1);
        let b = CommandList::new();
        b.push_constants_compute(&[9]);
        b.push_constants(&[3, 4]);
        a.append(b);

        assert_eq!(a.len(), seq.len());
        // Arena bytes match the sequential recording.
        assert_eq!(a.inner.borrow().push, seq.inner.borrow().push);
        assert_eq!(a.inner.borrow().push, vec![1, 2, 9, 3, 4]);
        // The appended `push_constants(&[3,4])` offset was rebased to point past the
        // first list's arena (so it resolves to [3, 4], not [1, 2]).
        let inner = a.inner.borrow();
        match inner.cmds.last().unwrap() {
            RhiCommand::PushConstants { off, len } => {
                assert_eq!(&inner.push[*off as usize..(*off + *len) as usize], &[3, 4]);
            }
            _ => panic!("expected PushConstants last"),
        }
    }
}
