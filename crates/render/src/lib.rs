//! A portable render graph (frame graph) over the engine RHI.
//!
//! Passes declare the resources they **write** (a color attachment, optional
//! depth) and **read** (bindless-sampled textures). The graph compiles those
//! declarations into a dependency DAG, schedules them with a topological sort,
//! culls passes that don't contribute to the backbuffer, computes each transient
//! resource's lifetime, then executes: it realizes physical render targets,
//! inserts the RT<->sampled barriers automatically, and invokes each pass's
//! record closure inside the right render pass.
//!
//! The graph depends only on the `rhi` facade (like `dreamcoast-gui`); all GPU code
//! lives in the backends. Transient *memory aliasing* (Phase 5.3) layers on top
//! of the lifetime analysis computed here.
//!
//! ```ignore
//! let mut graph = RenderGraph::new();
//! let backbuffer = graph.import_backbuffer(swapchain.format(), extent);
//! let scene = graph.create_color("scene", scene_format, extent);
//! let depth = graph.create_depth("depth", extent);
//! graph.add_pass(PassInfo { name: "scene", colors: vec![(scene, Some(clear))],
//!                           depth: Some(depth), reads: vec![] },
//!                |ctx| { /* draw mesh */ });
//! graph.add_pass(PassInfo { name: "post", colors: vec![(backbuffer, None)],
//!                           depth: None, reads: vec![scene] },
//!                |ctx| { /* sample ctx.sampled_index(scene) */ });
//! graph.execute(&device, &mut pool, &cmd, &swapchain, image_index, true, None)?;
//! ```

use std::collections::HashMap;

use dreamcoast_core::EngineError;
use rhi::{
    ClearColor, CommandBuffer, CommandList, DepthBuffer, Device, Extent2D, Format, QueryHeap,
    Recorder, RenderTarget, RenderTargetDesc, Swapchain, TransientHeap,
};

/// Collects per-pass GPU timestamps during [`RenderGraph::execute`].
///
/// The caller supplies a [`QueryHeap`] (sized `>= scheduled_passes + 1`). The
/// graph writes a timestamp at the start of each scheduled pass plus one final
/// boundary, so pass `i`'s GPU duration is `ticks[i + 1] - ticks[i]`. After the
/// frame's fence signals, read the heap and pair the ticks with [`Self::names`]
/// (scheduled order). Pass `Some` to profile, `None` to skip (zero overhead).
pub struct GraphProfiler<'a> {
    heap: &'a QueryHeap,
    /// Scheduled pass names, in order; index `i` is bracketed by query `i..i+1`.
    pub names: Vec<&'static str>,
}

impl<'a> GraphProfiler<'a> {
    /// Create a profiler writing into `heap`.
    pub fn new(heap: &'a QueryHeap) -> Self {
        Self {
            heap,
            names: Vec::new(),
        }
    }
}

/// A pass's record closure: records draw commands, may fail (e.g. per-frame uploads).
type RecordFn<'a> = Box<dyn FnMut(&mut PassContext) -> Result<(), EngineError> + 'a>;

/// A virtual resource handle, valid within the graph that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResourceKind {
    Color,
    Depth,
    /// An app-owned resource (e.g. a persistent storage buffer) tracked only for
    /// scheduling/culling — never realized or barriered by the graph. Its real
    /// barriers are issued explicitly in record closures (Phase 7).
    External,
}

/// Whether a pass runs on the graphics or compute pipeline (Phase 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Graphics,
    Compute,
}

/// A virtual resource: a transient texture the graph allocates, or the imported
/// swapchain backbuffer.
struct Resource {
    #[allow(dead_code)]
    name: &'static str,
    kind: ResourceKind,
    extent: Extent2D,
    format: Format,
    /// `true` for the imported swapchain image (realized from the swapchain, not
    /// the pool).
    backbuffer: bool,
    /// `true` for a compute-writable storage image (UAV); the realized render
    /// target gets a storage bindless index and the graph emits UAV barriers.
    storage: bool,
}

/// What a pass does to a resource.
struct PassNode<'a> {
    name: &'static str,
    kind: PassKind,
    /// Color attachments written by the pass, in attachment order. Each carries
    /// its load behavior (`Some(clear)` clears, `None` loads). Multiple entries
    /// drive MRT (e.g. a G-buffer fill). A backbuffer pass has exactly one.
    colors: Vec<(ResourceId, Option<ClearColor>)>,
    depth: Option<ResourceId>,
    reads: Vec<ResourceId>,
    /// Resources written via UAV (storage images / external storage buffers).
    storage_writes: Vec<ResourceId>,
    record: RecordFn<'a>,
}

/// Declarative description of a pass (everything but the record closure).
pub struct PassInfo {
    pub name: &'static str,
    /// Color attachments written by the pass, in attachment order. `Some(clear)`
    /// clears, `None` loads. One element is the common case; multiple drive MRT.
    pub colors: Vec<(ResourceId, Option<ClearColor>)>,
    /// Depth attachment written by the pass (always cleared).
    pub depth: Option<ResourceId>,
    /// Resources sampled by the pass.
    pub reads: Vec<ResourceId>,
}

/// Declarative description of a compute pass (Phase 7).
pub struct ComputePassInfo {
    pub name: &'static str,
    /// Storage images / external storage buffers written via UAV by this pass.
    pub storage_writes: Vec<ResourceId>,
    /// Sampled textures + external storage buffers read by this pass.
    pub reads: Vec<ResourceId>,
}

/// Per-pass context handed to a record closure during execution.
pub struct PassContext<'a> {
    cmd: &'a dyn Recorder,
    sampled: HashMap<ResourceId, u32>,
    storage: HashMap<ResourceId, u32>,
    extent: Extent2D,
}

impl<'a> PassContext<'a> {
    /// The frame recorder (for a graphics pass a render pass is already open; for a
    /// compute pass no render pass is open — bind a compute pipeline). Backed by the
    /// IR [`CommandList`] (M4): passes record into it, the RHI thread translates it
    /// onto a real command buffer.
    pub fn cmd(&self) -> &dyn Recorder {
        self.cmd
    }

    /// Bindless sampled index of a resource this pass declared as a read.
    pub fn sampled_index(&self, id: ResourceId) -> u32 {
        self.sampled[&id]
    }

    /// Bindless storage-image (UAV) index of a storage resource this pass writes.
    pub fn storage_index(&self, id: ResourceId) -> u32 {
        self.storage[&id]
    }

    /// Extent of the pass's color attachment, or of a compute pass's first storage
    /// write target (viewport/scissor are already set for graphics passes).
    pub fn extent(&self) -> Extent2D {
        self.extent
    }
}

/// A render graph for a single frame. Rebuilt each frame; record closures borrow
/// `'a` frame data.
#[derive(Default)]
pub struct RenderGraph<'a> {
    resources: Vec<Resource>,
    passes: Vec<PassNode<'a>>,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Import the swapchain backbuffer as a graph resource.
    pub fn import_backbuffer(&mut self, format: Format, extent: Extent2D) -> ResourceId {
        self.push_resource(Resource {
            name: "backbuffer",
            kind: ResourceKind::Color,
            extent,
            format,
            backbuffer: true,
            storage: false,
        })
    }

    /// Declare a transient color target.
    pub fn create_color(
        &mut self,
        name: &'static str,
        format: Format,
        extent: Extent2D,
    ) -> ResourceId {
        self.push_resource(Resource {
            name,
            kind: ResourceKind::Color,
            extent,
            format,
            backbuffer: false,
            storage: false,
        })
    }

    /// Declare a transient compute-writable storage image (UAV + sampled). A
    /// compute pass writes it (declared in `storage_writes`); a later graphics
    /// pass samples it like any color target (Phase 7).
    pub fn create_storage_image(
        &mut self,
        name: &'static str,
        format: Format,
        extent: Extent2D,
    ) -> ResourceId {
        self.push_resource(Resource {
            name,
            kind: ResourceKind::Color,
            extent,
            format,
            backbuffer: false,
            storage: true,
        })
    }

    /// Declare a transient depth target.
    pub fn create_depth(&mut self, name: &'static str, extent: Extent2D) -> ResourceId {
        self.push_resource(Resource {
            name,
            kind: ResourceKind::Depth,
            extent,
            format: Format::Depth32Float,
            backbuffer: false,
            storage: false,
        })
    }

    /// Import an app-owned resource (e.g. a persistent storage buffer) for
    /// dependency tracking only. The graph schedules passes that read/write it in
    /// order and keeps them from being culled, but never realizes or barriers it —
    /// the record closures issue its UAV/indirect barriers explicitly (Phase 7).
    pub fn import_external(&mut self, name: &'static str) -> ResourceId {
        self.push_resource(Resource {
            name,
            kind: ResourceKind::External,
            extent: Extent2D::default(),
            format: Format::Rgba8Unorm,
            backbuffer: false,
            storage: false,
        })
    }

    /// Add a pass with its record closure (which records draw commands and may
    /// fail, e.g. when uploading per-frame buffers).
    pub fn add_pass(
        &mut self,
        info: PassInfo,
        record: impl FnMut(&mut PassContext) -> Result<(), EngineError> + 'a,
    ) {
        self.passes.push(PassNode {
            name: info.name,
            kind: PassKind::Graphics,
            colors: info.colors,
            depth: info.depth,
            reads: info.reads,
            storage_writes: Vec::new(),
            record: Box::new(record),
        });
    }

    /// Add a graphics pass that ALSO writes external storage buffers via UAV (Phase 14 Track B:
    /// the HW mesh-vis pass rasterizes into the shared R64 visibility buffer from its fragment
    /// stage while rendering to a scratch color attachment). Identical to [`Self::add_pass`] plus
    /// `storage_writes`, which the scheduler uses for WAW/RAW ordering against the other visibility
    /// writers/readers; the actual UAV barrier for an *external* buffer is emitted by the closure
    /// (or a following barrier pass), since the graph doesn't own external resources.
    pub fn add_pass_with_storage_writes(
        &mut self,
        info: PassInfo,
        storage_writes: Vec<ResourceId>,
        record: impl FnMut(&mut PassContext) -> Result<(), EngineError> + 'a,
    ) {
        self.passes.push(PassNode {
            name: info.name,
            kind: PassKind::Graphics,
            colors: info.colors,
            depth: info.depth,
            reads: info.reads,
            storage_writes,
            record: Box::new(record),
        });
    }

    /// Add a compute pass. It writes `storage_writes` (storage images / external
    /// storage buffers via UAV) and reads `reads` (sampled textures + external
    /// storage buffers). No attachments; the record closure binds a compute
    /// pipeline and dispatches (Phase 7).
    pub fn add_compute_pass(
        &mut self,
        info: ComputePassInfo,
        record: impl FnMut(&mut PassContext) -> Result<(), EngineError> + 'a,
    ) {
        self.passes.push(PassNode {
            name: info.name,
            kind: PassKind::Compute,
            colors: Vec::new(),
            depth: None,
            reads: info.reads,
            storage_writes: info.storage_writes,
            record: Box::new(record),
        });
    }

    fn push_resource(&mut self, r: Resource) -> ResourceId {
        let id = ResourceId(self.resources.len());
        self.resources.push(r);
        id
    }

    /// Compile the declared passes into a culled, topologically-ordered schedule
    /// plus each transient's lifetime over that schedule.
    ///
    /// `scratch` supplies the working Vecs/HashMaps; it is cleared (not
    /// reallocated) at the top of every call, so calling this every frame with the
    /// same `scratch` keeps the backing allocations' capacity across frames instead
    /// of rebuilding them from empty. The compiled result (`Compiled`) is still a
    /// fresh, independently-owned value — only the scratch's storage is reused.
    fn compile(&self, scratch: &mut CompileScratch) -> Compiled {
        let n = self.passes.len();
        scratch.clear();
        let CompileScratch {
            edges,
            sampled,
            last_writer,
            readers_since,
            indegree,
            succ,
            order,
            ready,
            preds,
            required,
            stack,
        } = scratch;

        // Dependency edges: producer -> consumer (RAW), plus WAW/WAR. Because an
        // edge always points from an earlier-declared pass to a later one, the
        // graph is acyclic by construction.
        for (i, pass) in self.passes.iter().enumerate() {
            for &r in &pass.reads {
                if let Some(&w) = last_writer.get(&r) {
                    edges.push((w, i));
                }
                readers_since.entry(r).or_default().push(i);
            }
            for w in pass.writes() {
                if let Some(&prev) = last_writer.get(&w) {
                    edges.push((prev, i)); // WAW
                }
                if let Some(readers) = readers_since.get(&w) {
                    for &reader in readers {
                        edges.push((reader, i)); // WAR
                    }
                }
                last_writer.insert(w, i);
                // Reset "readers since last write" for `w`: clear the existing
                // Vec in place (same end state as `insert(w, Vec::new())`) so its
                // backing allocation survives for the next frame's reuse instead of
                // being replaced by a fresh, empty `Vec` every write.
                readers_since.entry(w).or_default().clear();
            }
        }

        // Kahn topological sort, picking the lowest pass index among ready nodes
        // so the schedule stays close to declaration order.
        indegree.resize(n, 0);
        succ.resize_with(n, Vec::new);
        for &(a, b) in edges.iter() {
            succ[a].push(b);
            indegree[b] += 1;
        }
        order.reserve(n);
        ready.extend((0..n).filter(|&i| indegree[i] == 0));
        while let Some(pos) = ready
            .iter()
            .enumerate()
            .min_by_key(|&(_, &v)| v)
            .map(|(p, _)| p)
        {
            let node = ready.remove(pos);
            order.push(node);
            for &m in &succ[node] {
                indegree[m] -= 1;
                if indegree[m] == 0 {
                    ready.push(m);
                }
            }
        }

        // Dead-pass culling: keep only passes that contribute to the backbuffer.
        preds.resize_with(n, Vec::new);
        for &(a, b) in edges.iter() {
            preds[b].push(a);
        }
        required.resize(n, false);
        for (i, pass) in self.passes.iter().enumerate() {
            // Root at backbuffer-writers and at external-writers (cross-frame side effects,
            // e.g. the Stage C7b lit-color history written this frame, read the next).
            if pass.writes_backbuffer(&self.resources) || pass.writes_external(&self.resources) {
                required[i] = true;
                stack.push(i);
            }
        }
        while let Some(i) = stack.pop() {
            for &p in &preds[i] {
                if !required[p] {
                    required[p] = true;
                    stack.push(p);
                }
            }
        }

        // Capacity hints only (upper bounds — `schedule` is at most `order.len()`
        // after culling, `lifetimes` has at most one entry per resource): the
        // schedule/lifetimes maps are `Compiled`'s owned output, freshly allocated
        // every frame by design (they outlive this function), so unlike the
        // scratch above there is no prior-frame allocation to reuse — but sizing
        // them up front still avoids the several incremental reallocations a
        // grow-from-empty `Vec`/`HashMap` would otherwise do as it fills.
        let mut schedule: Vec<usize> = Vec::with_capacity(order.len());
        schedule.extend(order.iter().copied().filter(|&i| required[i]));

        // Lifetime of each resource over the final schedule (first/last position
        // that references it). Used by transient aliasing. The third field is the
        // tile-only (memoryless) eligibility — folded into this map (instead of a
        // parallel set) so `compile()` keeps its irreducible two owned
        // allocations (see `compile_reuses_scratch_allocations_across_frames`).
        let mut lifetimes: HashMap<ResourceId, Lifetime> =
            HashMap::with_capacity(self.resources.len());
        for (pos, &pass_idx) in schedule.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for r in pass.references() {
                lifetimes
                    .entry(r)
                    .and_modify(|l| l.last = pos)
                    .or_insert(Lifetime {
                        first: pos,
                        last: pos,
                        memoryless: false,
                    });
            }
        }

        // Tile-only (memoryless) eligibility, derived purely from graph lifetime.
        // A transient is eligible only if it is written and consumed entirely
        // within its producing render pass — i.e. NO scheduled pass ever samples
        // it. That is exactly "the resource never appears in any pass's `reads`
        // list": if nothing reads it in a later pass, its contents never have to
        // leave tile memory (no cross-pass sample, no CPU readback, no copy). The
        // deferred G-buffer + depth are read by the later lighting / SW-RT passes,
        // so they appear in `reads` and are correctly excluded here.
        //
        // Backbuffer (presented), external (app-owned / cross-frame), storage
        // images (a UAV whose whole purpose is to be sampled by a later pass), and
        // depth are never eligible: depth is only used cross-pass in this engine,
        // and a memoryless depth buffer would still be a hazard if any future pass
        // sampled it — so we keep the criterion to plain, unread color transients.
        // `sampled` is dense per-resource scratch (reused across frames like the
        // other scheduling scratch — no per-compile allocation once warm).
        sampled.resize(self.resources.len(), false);
        for &pass_idx in &schedule {
            for &r in &self.passes[pass_idx].reads {
                sampled[r.0] = true;
            }
        }
        for (&id, life) in lifetimes.iter_mut() {
            let res = &self.resources[id.0];
            life.memoryless = res.kind == ResourceKind::Color
                && !res.backbuffer
                && !res.storage
                && !sampled[id.0];
        }

        Compiled {
            schedule,
            lifetimes,
        }
    }

    /// Compile and run the graph against the RHI for one frame: build the IR and
    /// translate it onto `cmd` (same thread). The backbuffer is taken from
    /// `swapchain`/`image_index`. See [`Self::record`] for the threaded split.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        self,
        device: &Device,
        pool: &mut ResourcePool,
        cmd: &CommandBuffer,
        swapchain: &Swapchain,
        image_index: u32,
        aliasing: bool,
        profiler: Option<&mut GraphProfiler>,
    ) -> Result<(), EngineError> {
        let list = self.record(device, pool, aliasing, profiler)?;
        // Replay the recorded IR onto the real command buffer (B2: same thread,
        // behaviour-identical to direct recording). The backbuffer's swapchain +
        // image index are frame-global, supplied here at translate.
        list.translate(cmd, swapchain, image_index)?;
        Ok(())
    }

    /// Like [`Self::record`] but records the graph's passes **in parallel** on `jobs`
    /// (M4 B4): each scheduled pass builds its own IR bucket on a worker, then the
    /// buckets are concatenated in schedule order. Recording order is irrelevant to
    /// the result — GPU ordering lives in the barrier commands, stitched in schedule
    /// order at concat — so the output IR is identical to sequential recording.
    /// Profiling is unsupported here (it is inherently sequential); pass the threaded
    /// path, which doesn't profile.
    pub fn record_parallel(
        self,
        device: &Device,
        pool: &mut ResourcePool,
        aliasing: bool,
        jobs: &dreamcoast_jobs::JobSystem,
    ) -> Result<CommandList, EngineError> {
        let list = CommandList::new();
        self.record_into(&list, device, pool, aliasing, None, Some(jobs))?;
        Ok(list)
    }

    /// Compile and record the frame into a backend-agnostic, `Send` [`CommandList`]
    /// — *without* translating or touching the swapchain. This is the record-thread
    /// half of the M4 B3 split: the RHI thread later [`CommandList::translate`]s the
    /// returned list onto a command buffer with its own owned swapchain + freshly
    /// acquired image index, so the record thread never needs to acquire.
    ///
    /// `pool` caches realized targets across frames. When `aliasing` is set,
    /// transient color targets are placed into a shared heap by
    /// [`Self::plan_aliasing`] so targets with non-overlapping lifetimes reuse
    /// memory; otherwise each gets its own dedicated allocation.
    pub fn record(
        self,
        device: &Device,
        pool: &mut ResourcePool,
        aliasing: bool,
        profiler: Option<&mut GraphProfiler>,
    ) -> Result<CommandList, EngineError> {
        let list = CommandList::new();
        self.record_into(&list, device, pool, aliasing, profiler, None)?;
        Ok(list)
    }

    /// Append the frame's commands onto an existing [`CommandList`]. Lets the caller
    /// prepend pre-graph imperative work (e.g. the IBL environment capture) into the
    /// same `Send` list that ships to the RHI thread (M4 B3), preserving record order.
    pub fn record_into(
        mut self,
        list: &CommandList,
        device: &Device,
        pool: &mut ResourcePool,
        aliasing: bool,
        mut profiler: Option<&mut GraphProfiler>,
        jobs: Option<&dreamcoast_jobs::JobSystem>,
    ) -> Result<(), EngineError> {
        let compiled = self.compile(&mut pool.compile_scratch);
        if tracing::enabled!(tracing::Level::TRACE) {
            let names: Vec<&str> = compiled
                .schedule
                .iter()
                .map(|&i| self.passes[i].name)
                .collect();
            tracing::trace!("render graph schedule: {}", names.join(" -> "));
        }
        pool.begin_frame();

        // Plan transient aliasing (per-resource heap offsets + which targets need
        // an aliasing/discard barrier before their first write).
        let mut alias_barrier: HashMap<ResourceId, bool> = HashMap::new();
        if aliasing {
            let plan = self.plan_aliasing(&compiled, device)?;
            for p in &plan.placements {
                alias_barrier.insert(p.id, p.needs_alias_barrier);
            }
            pool.realize_aliased(device, plan)?;
        }

        // Phase 1 — realize physical resources for every transient referenced by
        // the schedule (mutable pool access). Color transients come from the alias
        // set when aliasing, else dedicated pooled allocations; depth is always
        // pooled (depth is not aliased).
        let mut color_locs: HashMap<ResourceId, ColorLoc> = HashMap::new();
        let mut depth_slots: HashMap<ResourceId, usize> = HashMap::new();
        for &pass_idx in &compiled.schedule {
            let pass = &self.passes[pass_idx];
            for r in pass.references() {
                let res = &self.resources[r.0];
                if res.backbuffer {
                    continue;
                }
                match res.kind {
                    ResourceKind::Color => {
                        if let std::collections::hash_map::Entry::Vacant(e) = color_locs.entry(r) {
                            let memoryless =
                                compiled.lifetimes.get(&r).is_some_and(|l| l.memoryless);
                            // Storage images need a dedicated (UAV-capable) allocation;
                            // memoryless targets need a dedicated (unbacked, un-heapable)
                            // allocation; only plain, sampled color transients alias.
                            if aliasing && !res.storage && !memoryless {
                                e.insert(ColorLoc::Aliased(r));
                            } else {
                                let desc = render_target_desc(res, memoryless);
                                e.insert(ColorLoc::Pooled(pool.acquire_color(device, desc)?));
                            }
                        }
                    }
                    ResourceKind::Depth => {
                        if let std::collections::hash_map::Entry::Vacant(e) = depth_slots.entry(r) {
                            e.insert(pool.acquire_depth(device, res.extent)?);
                        }
                    }
                    // External resources are app-owned: never realized.
                    ResourceKind::External => {}
                }
            }
        }

        // Phase 2 — record (immutable pool access).
        //
        // Profiling: write a timestamp at the start of each scheduled pass plus
        // one final boundary, so pass i's GPU time is ticks[i+1]-ticks[i]. Only
        // when the heap has room for every boundary; reset the pool first (Vulkan
        // requires it, outside any render pass — we are between phases here).
        let profile = profiler
            .as_ref()
            .map(|p| p.heap.count() as usize > compiled.schedule.len())
            .unwrap_or(false);
        if let (true, Some(p)) = (profile, profiler.as_mut()) {
            p.names.clear();
            // Reset the whole pool (not just the boundaries we write): the host
            // reads all `count` slots back, and Vulkan requires every queried slot
            // to have been reset since creation.
            list.reset_queries(p.heap, 0, p.heap.count());
        }

        // The backbuffer's first-use transition is the only cross-pass sequential
        // bit; precompute which scheduled pass owns it (the first backbuffer writer)
        // so per-pass recording needs no running flag (and can run out of order).
        let first_backbuffer = compiled.schedule.iter().copied().find(|&i| {
            let c = &self.passes[i].colors;
            !c.is_empty() && self.resources[c[0].0.0].backbuffer
        });

        // Per depth resource, the first scheduled pass that attaches it: that pass CLEARS
        // the depth, every later pass attaching the same depth LOADS it (preserving the
        // earlier writes). Without this a later pass (the deferred decal pass) would clear
        // the G-buffer depth the SW-RT passes still sample (AO / GI / reflections). Computed
        // here (sequential) so the parallel per-pass recording stays a pure read.
        let mut depth_first_writer: HashMap<ResourceId, usize> = HashMap::new();
        for &i in &compiled.schedule {
            if let Some(d) = self.passes[i].depth {
                depth_first_writer.entry(d).or_insert(i);
            }
        }

        if let Some(jobs) = jobs.filter(|_| !profile) {
            // M4 B4 — parallel recording: each scheduled pass builds its own bucket on
            // a worker, then we concatenate the buckets in schedule order. GPU ordering
            // is unaffected (it lives in the barrier commands, stitched in schedule
            // order here), so the IR is identical to the sequential path.
            let pool_ref: &ResourcePool = pool;
            let mut slots: Vec<Option<&mut PassNode>> = self.passes.iter_mut().map(Some).collect();
            let mut pass_jobs: Vec<PassJob> = compiled
                .schedule
                .iter()
                .map(|&i| {
                    let pass = slots[i]
                        .take()
                        .expect("each pass is scheduled at most once");
                    let depth_clear = pass
                        .depth
                        .is_none_or(|d| depth_first_writer.get(&d) == Some(&i));
                    PassJob {
                        pass,
                        resources: &self.resources,
                        pool: pool_ref,
                        color_locs: &color_locs,
                        depth_slots: &depth_slots,
                        alias_barrier: &alias_barrier,
                        is_first_backbuffer: first_backbuffer == Some(i),
                        depth_clear,
                        bucket: CommandList::new(),
                        result: Ok(()),
                    }
                })
                .collect();
            // The closure captures nothing external (all context is in `job`), so it is
            // trivially `Send + Sync`; `PassJob` is `Send` by the handoff contract above.
            jobs.parallel_for(&mut pass_jobs, 1, |_, job| {
                job.result = record_pass(
                    job.pass,
                    job.resources,
                    job.pool,
                    job.color_locs,
                    job.depth_slots,
                    job.alias_barrier,
                    job.is_first_backbuffer,
                    job.depth_clear,
                    &job.bucket,
                );
            });
            // Concatenate in schedule order; surface the first pass that failed.
            for job in pass_jobs {
                job.result?;
                list.append(job.bucket);
            }
        } else {
            for &pass_idx in &compiled.schedule {
                // Timestamp this pass's start boundary (query index = passes seen so far).
                if let (true, Some(p)) = (profile, profiler.as_mut()) {
                    let idx = p.names.len() as u32;
                    list.write_timestamp(p.heap, idx);
                    p.names.push(self.passes[pass_idx].name);
                }
                let depth_clear = self.passes[pass_idx]
                    .depth
                    .is_none_or(|d| depth_first_writer.get(&d) == Some(&pass_idx));
                record_pass(
                    &mut self.passes[pass_idx],
                    &self.resources,
                    pool,
                    &color_locs,
                    &depth_slots,
                    &alias_barrier,
                    first_backbuffer == Some(pass_idx),
                    depth_clear,
                    list,
                )?;
            }
        }

        // Final timestamp boundary, then resolve the whole heap (D3D12) for readback.
        if let (true, Some(p)) = (profile, profiler.as_mut()) {
            let count = p.names.len() as u32;
            list.write_timestamp(p.heap, count);
            list.resolve_queries(p.heap, count + 1);
        }

        if first_backbuffer.is_some() {
            list.backbuffer_to_present();
        }

        Ok(())
    }

    /// Compute a transient-aliasing plan: assign each color transient a heap
    /// offset, sharing offsets between targets whose lifetimes don't overlap
    /// (greedy first-fit over lifetime intervals).
    fn plan_aliasing(
        &self,
        compiled: &Compiled,
        device: &Device,
    ) -> Result<AliasPlan, EngineError> {
        struct Item {
            id: ResourceId,
            desc: RenderTargetDesc,
            first: usize,
            last: usize,
            size: u64,
            align: u64,
        }
        let mut items = Vec::new();
        for (&id, &Lifetime { first, last, .. }) in &compiled.lifetimes {
            let res = &self.resources[id.0];
            // Storage images carry a UAV and are not RT-DS-only, so they can't live
            // in the aliasing heap; they get dedicated pooled allocations. A
            // memoryless (tile-only) target has no system-memory backing to place
            // at a heap offset, so it also bypasses the heap and is pooled.
            if res.backbuffer
                || res.kind != ResourceKind::Color
                || res.storage
                || compiled.lifetimes.get(&id).is_some_and(|l| l.memoryless)
            {
                continue;
            }
            let desc = render_target_desc(res, false);
            let req = device.render_target_memory(&desc)?;
            items.push(Item {
                id,
                desc,
                first,
                last,
                size: req.size,
                align: req.alignment.max(1),
            });
        }
        // Deterministic across frames: order by first-use, then id.
        items.sort_by_key(|it| (it.first, it.id.0));

        struct Slot {
            last: usize,
            size: u64,
            align: u64,
            members: usize,
        }
        let mut slots: Vec<Slot> = Vec::new();
        let mut slot_of: HashMap<ResourceId, usize> = HashMap::new();
        for it in &items {
            // First slot whose previous tenant's lifetime ended before this one
            // begins can be reused.
            let chosen = slots.iter().position(|s| s.last < it.first);
            let si = match chosen {
                Some(si) => {
                    let s = &mut slots[si];
                    s.last = it.last;
                    s.size = s.size.max(it.size);
                    s.align = s.align.max(it.align);
                    s.members += 1;
                    si
                }
                None => {
                    slots.push(Slot {
                        last: it.last,
                        size: it.size,
                        align: it.align,
                        members: 1,
                    });
                    slots.len() - 1
                }
            };
            slot_of.insert(it.id, si);
        }

        // Lay slots out back-to-back, each aligned to its requirement.
        let mut cursor = 0u64;
        let slot_offset: Vec<u64> = slots
            .iter()
            .map(|s| {
                let off = align_up(cursor, s.align);
                cursor = off + align_up(s.size, s.align);
                off
            })
            .collect();
        let heap_size = cursor;

        let committed: u64 = items.iter().map(|it| it.size).sum();
        tracing::trace!(
            "transient aliasing: {} color targets -> {} heap slots, {} KiB (dedicated would be {} KiB)",
            items.len(),
            slots.len(),
            heap_size / 1024,
            committed / 1024,
        );

        let mut placements: Vec<Placement> = items
            .iter()
            .map(|it| {
                let si = slot_of[&it.id];
                Placement {
                    id: it.id,
                    desc: it.desc,
                    offset: slot_offset[si],
                    needs_alias_barrier: slots[si].members > 1,
                }
            })
            .collect();
        placements.sort_by_key(|p| p.id.0);
        Ok(AliasPlan {
            heap_size,
            placements,
        })
    }
}

/// Record one scheduled pass's full IR sub-sequence — debug label, read→sampled
/// barriers, attachment transitions + `begin_rendering`, the pass's record closure,
/// and the closing `end_rendering`/label — onto `bucket`.
///
/// Free of `&self` and of any cross-pass sequential state (the one such bit, the
/// backbuffer's first-use transition, is passed in as `is_first_backbuffer`), so it
/// can run on a job worker into a private bucket and be concatenated back in
/// schedule order (M4 B4). The profiler timestamp is recorded by the caller (it is
/// inline-only and inherently sequential), just before this call.
#[allow(clippy::too_many_arguments)]
fn record_pass(
    pass: &mut PassNode,
    resources: &[Resource],
    pool: &ResourcePool,
    color_locs: &HashMap<ResourceId, ColorLoc>,
    depth_slots: &HashMap<ResourceId, usize>,
    alias_barrier: &HashMap<ResourceId, bool>,
    is_first_backbuffer: bool,
    // Whether this pass is the first to attach its depth (clear) vs a later user (load).
    depth_clear: bool,
    bucket: &CommandList,
) -> Result<(), EngineError> {
    // Name this pass's region for GPU captures (RenderDoc/PIX/NSight); the barriers +
    // draws below land inside the marker, closed at the end of both pass kinds.
    bucket.begin_debug_label(pass.name);
    // Barriers: reads -> sampled. A storage image read after a compute write
    // transitions from UAV/GENERAL; plain color/depth from attachment.
    for &r in &pass.reads {
        if let Some(loc) = color_locs.get(&r) {
            if resources[r.0].storage {
                bucket.storage_to_sampled(pool.color_target(loc));
            } else {
                bucket.rt_to_sampled(pool.color_target(loc));
            }
        } else if let Some(&slot) = depth_slots.get(&r) {
            bucket.depth_to_sampled(pool.depth(slot));
        }
    }

    // Sampled-index map (color targets + depth/shadow maps), shared by both kinds.
    let sampled: HashMap<ResourceId, u32> = pass
        .reads
        .iter()
        .filter_map(|&r| {
            if let Some(loc) = color_locs.get(&r) {
                Some((r, pool.color_target(loc).bindless_index()))
            } else {
                depth_slots
                    .get(&r)
                    .map(|&slot| (r, pool.depth(slot).bindless_index()))
            }
        })
        .collect();

    // Compute pass: transition storage-image writes to UAV, then dispatch (no render
    // pass). External storage-buffer writes barrier explicitly in the closure.
    if pass.kind == PassKind::Compute {
        for &w in &pass.storage_writes {
            if let Some(loc) = color_locs.get(&w) {
                bucket.rt_to_storage(pool.color_target(loc));
            }
        }
        let storage: HashMap<ResourceId, u32> = pass
            .storage_writes
            .iter()
            .filter_map(|&w| {
                color_locs
                    .get(&w)
                    .and_then(|loc| pool.color_target(loc).storage_index().map(|i| (w, i)))
            })
            .collect();
        let extent = pass
            .storage_writes
            .iter()
            .find_map(|&w| color_locs.get(&w).map(|_| resources[w.0].extent))
            .unwrap_or_default();
        let mut ctx = PassContext {
            cmd: bucket,
            sampled,
            storage,
            extent,
        };
        (pass.record)(&mut ctx)?;
        bucket.end_debug_label();
        return Ok(());
    }

    // Transition this pass's depth attachment for writing (a shadow map reused across
    // frames may be in shader-read from the prior frame).
    let depth_ref = pass.depth.map(|d| pool.depth(depth_slots[&d]));
    if let Some(d) = depth_ref {
        bucket.depth_to_render_target(d);
    }

    // Resolve the color attachments, transition them for writing, and begin the
    // render pass. A depth-only pass (no colors) is a shadow map; the backbuffer (when
    // written) is always a single attachment; other offscreen passes may write 1..N.
    let colors = &pass.colors;
    let extent = if colors.is_empty() {
        let depth_id = pass
            .depth
            .expect("depth-only pass needs a depth attachment");
        let d = depth_ref.expect("depth-only pass needs a depth attachment");
        bucket.begin_rendering_depth_only(d);
        let extent = resources[depth_id.0].extent;
        bucket.set_viewport_scissor_extent(extent);
        extent
    } else {
        let first_res = &resources[colors[0].0.0];
        let extent = first_res.extent;
        if first_res.backbuffer {
            let clear = colors[0].1;
            if is_first_backbuffer {
                bucket.backbuffer_to_render_target();
            }
            bucket.begin_backbuffer_rendering(clear, depth_ref);
            bucket.set_backbuffer_viewport();
        } else {
            for &(id, _) in colors {
                let target = pool.color_target(&color_locs[&id]);
                if alias_barrier.get(&id).copied().unwrap_or(false) {
                    bucket.aliasing_barrier(target);
                } else {
                    bucket.rt_to_render_target(target);
                }
            }
            let targets: Vec<(&RenderTarget, Option<ClearColor>)> = colors
                .iter()
                .map(|&(id, clear)| (pool.color_target(&color_locs[&id]), clear))
                .collect();
            bucket.begin_rendering_targets(&targets, depth_ref, depth_clear);
            bucket.set_viewport_scissor_extent(extent);
        }
        extent
    };

    // Run the graphics pass's record closure.
    let mut ctx = PassContext {
        cmd: bucket,
        sampled,
        storage: HashMap::new(),
        extent,
    };
    (pass.record)(&mut ctx)?;
    bucket.end_rendering();
    bucket.end_debug_label();
    Ok(())
}

/// One pass's parallel-recording work unit (M4 B4): exclusive access to its pass
/// node, the read-only graph context it records against, its private output
/// [`CommandList`] bucket, and where the result lands. The context is carried inline
/// (rather than captured by the worker closure) so the closure captures nothing —
/// Rust 2021 disjoint capture would otherwise reach through any shared wrapper and
/// capture the individual `!Sync` field references.
struct PassJob<'a, 'r> {
    pass: &'r mut PassNode<'a>,
    resources: &'r [Resource],
    pool: &'r ResourcePool,
    color_locs: &'r HashMap<ResourceId, ColorLoc>,
    depth_slots: &'r HashMap<ResourceId, usize>,
    alias_barrier: &'r HashMap<ResourceId, bool>,
    is_first_backbuffer: bool,
    depth_clear: bool,
    bucket: CommandList,
    result: Result<(), EngineError>,
}

// SAFETY: each `PassJob` is handed to exactly one worker — its `&mut PassNode` is
// non-aliasing (distinct pass index) and its `bucket` is written only by that worker.
// The pass node + context hold `!Send`/`!Sync` backend handles, but during the
// parallel region every worker only *reads* the shared context and only *borrows* the
// pass's resources to bake `ResPtr`s — none clones/drops an `Rc<DeviceShared>`,
// creates a resource, or writes shared memory (audited: pass closures are
// recording-only; per-frame uploads + resource creation happen before `execute`). So
// no backend refcount/state is mutated off the record thread.
unsafe impl Send for PassJob<'_, '_> {}

/// Round `value` up to a multiple of `align` (a power-of-two or any `>= 1`).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

fn render_target_desc(res: &Resource, memoryless: bool) -> RenderTargetDesc {
    RenderTargetDesc {
        width: res.extent.width,
        height: res.extent.height,
        format: res.format,
        storage: res.storage,
        memoryless,
    }
}

/// Where a color transient's physical storage lives this frame.
enum ColorLoc {
    /// A dedicated pooled allocation (slot index).
    Pooled(usize),
    /// An aliased target in the pool's transient heap (by resource id).
    Aliased(ResourceId),
}

/// A transient-aliasing plan: total heap size and per-target placement.
#[derive(Clone, PartialEq, Eq, Default)]
struct AliasPlan {
    heap_size: u64,
    placements: Vec<Placement>,
}

#[derive(Clone, PartialEq, Eq)]
struct Placement {
    id: ResourceId,
    desc: RenderTargetDesc,
    offset: u64,
    /// True when this target shares its heap region with another (it must issue
    /// an aliasing/discard barrier before its first write).
    needs_alias_barrier: bool,
}

impl PassNode<'_> {
    /// Resources this pass writes (color attachments + depth + UAV storage).
    fn writes(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.colors
            .iter()
            .map(|(id, _)| *id)
            .chain(self.depth)
            .chain(self.storage_writes.iter().copied())
    }

    /// All resources this pass touches (writes + reads).
    fn references(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.writes().chain(self.reads.iter().copied())
    }

    fn writes_backbuffer(&self, resources: &[Resource]) -> bool {
        self.colors.iter().any(|(id, _)| resources[id.0].backbuffer)
    }

    /// Whether this pass writes an imported external resource. Such a write is an
    /// observable side effect that may be consumed outside this graph or in a later
    /// frame (e.g. a ping-pong history buffer written this frame, read the next), so the
    /// pass must be kept even when nothing in *this* frame's graph reads its output.
    fn writes_external(&self, resources: &[Resource]) -> bool {
        self.storage_writes
            .iter()
            .any(|id| matches!(resources[id.0].kind, ResourceKind::External))
    }
}

/// Output of compilation: the culled, ordered pass schedule and resource
/// lifetimes (first/last schedule position).
struct Compiled {
    schedule: Vec<usize>,
    /// Per-resource schedule lifetime (drives aliasing) + the tile-only flag.
    lifetimes: HashMap<ResourceId, Lifetime>,
}

/// One resource's compiled lifetime: first/last schedule position it is
/// referenced, plus tile-only (memoryless) eligibility — a color transient no
/// scheduled pass ever samples. On a tile-based GPU its descriptor gets
/// `memoryless = true`; it bypasses the aliasing heap (a memoryless texture has
/// no system-memory backing to place at a heap offset).
#[derive(Clone, Copy)]
struct Lifetime {
    first: usize,
    last: usize,
    memoryless: bool,
}

/// Reusable working storage for [`RenderGraph::compile`]. The graph itself is
/// rebuilt every frame, but these Vecs/HashMaps are pure scheduling scratch with
/// no cross-frame semantics — owning one instance across frames (in
/// [`ResourcePool`], which already persists per frame-in-flight) and `clear`ing
/// it at the top of each `compile` call keeps their backing allocations' capacity
/// instead of reallocating 8 containers from empty every frame. `clear()` never
/// changes the compile algorithm: every field is emptied before use, same as a
/// freshly-`Default`ed value.
#[derive(Default)]
struct CompileScratch {
    edges: Vec<(usize, usize)>,
    /// Dense per-resource "some scheduled pass reads it" flags (memoryless derivation).
    sampled: Vec<bool>,
    last_writer: HashMap<ResourceId, usize>,
    readers_since: HashMap<ResourceId, Vec<usize>>,
    indegree: Vec<usize>,
    succ: Vec<Vec<usize>>,
    order: Vec<usize>,
    ready: Vec<usize>,
    preds: Vec<Vec<usize>>,
    required: Vec<bool>,
    stack: Vec<usize>,
}

impl CompileScratch {
    /// Empty every field for a new `compile()` call. `succ`/`preds` (indexed by
    /// pass, `Vec<Vec<usize>>`) and `readers_since`'s values (per-resource
    /// `Vec<usize>`) are cleared **in place** — each nested `Vec`'s contents are
    /// dropped but its backing allocation is kept — instead of via the outer
    /// container's own `clear()`, which would drop the nested `Vec`s entirely and
    /// force them to reallocate on the next frame's first `push`. That in-place
    /// nested clear is the difference between "8 containers reused" and "8
    /// containers reused but their N per-pass/per-resource nested Vecs still
    /// reallocate every frame".
    fn clear(&mut self) {
        let Self {
            edges,
            sampled,
            last_writer,
            readers_since,
            indegree,
            succ,
            order,
            ready,
            preds,
            required,
            stack,
        } = self;
        edges.clear();
        sampled.clear();
        last_writer.clear();
        for v in readers_since.values_mut() {
            v.clear();
        }
        indegree.clear();
        for v in succ.iter_mut() {
            v.clear();
        }
        order.clear();
        ready.clear();
        for v in preds.iter_mut() {
            v.clear();
        }
        required.clear();
        stack.clear();
    }
}

struct PooledColor {
    desc: RenderTargetDesc,
    rt: RenderTarget,
    used: bool,
}

struct PooledDepth {
    extent: Extent2D,
    depth: DepthBuffer,
    used: bool,
}

/// A realized transient-aliasing plan: a heap plus the placed targets, cached
/// while the plan is unchanged. `targets` is declared before `heap` so the
/// placed targets drop first.
struct AliasedSet {
    plan: AliasPlan,
    targets: HashMap<ResourceId, RenderTarget>,
    #[allow(dead_code)] // owns the heap memory the targets are placed into
    heap: TransientHeap,
}

/// Caches realized render targets / depth buffers across frames so the graph
/// doesn't reallocate every frame. Use one pool per frame-in-flight: a pool's
/// resources are reused only after that frame slot's fence has signaled. Call
/// [`ResourcePool::clear`] when the swapchain is resized.
#[derive(Default)]
pub struct ResourcePool {
    colors: Vec<PooledColor>,
    depths: Vec<PooledDepth>,
    aliased: Option<AliasedSet>,
    /// Scheduling scratch for [`RenderGraph::compile`], reused across frames. Not
    /// touched by [`Self::clear`] — it holds no GPU resources or extent-dependent
    /// state, only index bookkeeping cleared at the top of every `compile` call.
    compile_scratch: CompileScratch,
}

impl ResourcePool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all cached resources (e.g. after a resize changes target extents).
    pub fn clear(&mut self) {
        self.colors.clear();
        self.depths.clear();
        self.aliased = None;
    }

    /// Ensure the pool's transient heap + placed targets match `plan`, rebuilding
    /// (after a GPU idle) only when the plan changed.
    fn realize_aliased(&mut self, device: &Device, plan: AliasPlan) -> Result<(), EngineError> {
        if self.aliased.as_ref().is_some_and(|set| set.plan == plan) {
            return Ok(());
        }
        // The old heap/targets may still be referenced by in-flight frames.
        device.wait_idle()?;
        self.aliased = None;
        tracing::debug!(
            "building transient heap: {} KiB for {} aliased targets",
            plan.heap_size / 1024,
            plan.placements.len(),
        );
        let heap = device.create_transient_heap(plan.heap_size)?;
        let mut targets = HashMap::new();
        for p in &plan.placements {
            targets.insert(
                p.id,
                device.create_aliased_target(&heap, p.offset, &p.desc)?,
            );
        }
        self.aliased = Some(AliasedSet {
            plan,
            targets,
            heap,
        });
        Ok(())
    }

    fn begin_frame(&mut self) {
        for c in &mut self.colors {
            c.used = false;
        }
        for d in &mut self.depths {
            d.used = false;
        }
    }

    fn acquire_color(
        &mut self,
        device: &Device,
        desc: RenderTargetDesc,
    ) -> Result<usize, EngineError> {
        if let Some(i) = self.colors.iter().position(|c| !c.used && c.desc == desc) {
            self.colors[i].used = true;
            return Ok(i);
        }
        let rt = device.create_render_target(&desc)?;
        self.colors.push(PooledColor {
            desc,
            rt,
            used: true,
        });
        Ok(self.colors.len() - 1)
    }

    fn acquire_depth(&mut self, device: &Device, extent: Extent2D) -> Result<usize, EngineError> {
        if let Some(i) = self
            .depths
            .iter()
            .position(|d| !d.used && d.extent == extent)
        {
            self.depths[i].used = true;
            return Ok(i);
        }
        let depth = device.create_depth_buffer(extent)?;
        self.depths.push(PooledDepth {
            extent,
            depth,
            used: true,
        });
        Ok(self.depths.len() - 1)
    }

    /// Resolve a color transient's physical target (pooled or aliased).
    fn color_target(&self, loc: &ColorLoc) -> &RenderTarget {
        match loc {
            ColorLoc::Pooled(slot) => &self.colors[*slot].rt,
            ColorLoc::Aliased(id) => {
                &self.aliased.as_ref().expect("aliased set realized").targets[id]
            }
        }
    }

    fn depth(&self, slot: usize) -> &DepthBuffer {
        &self.depths[slot].depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};

    /// Counts heap allocations made *by the current thread* while a
    /// [`count_allocs`] window is active, so a test can measure allocations made
    /// by a specific call without instrumenting the call itself. The counter is
    /// thread-local on purpose: `cargo test` runs tests (and the libtest harness
    /// itself) on concurrent threads, and a process-global counter picks up their
    /// unrelated allocations — a flaky off-by-a-few. Both cells are
    /// const-initialized so touching them inside `alloc` cannot itself allocate;
    /// `try_with` guards against TLS teardown.
    struct CountingAlloc;
    thread_local! {
        static TL_COUNT_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static TL_ALLOC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if TL_COUNT_ACTIVE
                .try_with(std::cell::Cell::get)
                .unwrap_or(false)
            {
                let _ = TL_ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    fn count_allocs(f: impl FnOnce()) -> usize {
        TL_ALLOC_COUNT.with(|c| c.set(0));
        TL_COUNT_ACTIVE.with(|a| a.set(true));
        f();
        TL_COUNT_ACTIVE.with(|a| a.set(false));
        TL_ALLOC_COUNT.with(std::cell::Cell::get)
    }

    /// Builds a graph shaped like the sandbox's default deferred+SW-RT frame: a
    /// G-buffer MRT pass, a handful of compute passes (GDF AO/GI/reflect-style
    /// chain), and a backbuffer-writing lighting/tonemap tail — enough passes and
    /// resources to exercise `compile`'s scheduling in a representative way.
    fn build_graph(n_compute: usize) -> RenderGraph<'static> {
        let mut graph = RenderGraph::new();
        let extent = Extent2D::new(1920, 1080);
        let backbuffer = graph.import_backbuffer(Format::Rgba8Unorm, extent);
        let g_albedo = graph.create_color("g_albedo", Format::Rgba8Unorm, extent);
        let depth = graph.create_depth("depth", extent);
        graph.add_pass(
            PassInfo {
                name: "gbuffer",
                colors: vec![(
                    g_albedo,
                    Some(ClearColor {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                )],
                depth: Some(depth),
                reads: vec![],
            },
            |_ctx| Ok(()),
        );
        let mut prev = g_albedo;
        for i in 0..n_compute {
            let out = graph.create_storage_image(
                ["gi", "ao", "reflect", "temporal", "atrous"][i % 5],
                Format::Rgba8Unorm,
                extent,
            );
            graph.add_compute_pass(
                ComputePassInfo {
                    name: ["p_gi", "p_ao", "p_reflect", "p_temporal", "p_atrous"][i % 5],
                    storage_writes: vec![out],
                    reads: vec![prev],
                },
                |_ctx| Ok(()),
            );
            prev = out;
        }
        graph.add_pass(
            PassInfo {
                name: "lighting",
                colors: vec![(backbuffer, None)],
                depth: None,
                reads: vec![prev],
            },
            |_ctx| Ok(()),
        );
        graph
    }

    /// Regression guard for the render-graph alloc-churn fix: once a
    /// [`CompileScratch`] has been warmed up by a prior `compile()` call (as it is
    /// every frame after the first, since `ResourcePool` — and the `CompileScratch`
    /// it now owns — persists across frames), a same-shaped graph must compile
    /// with zero heap allocations from the scheduling scratch itself. This directly
    /// measures the fix in crates/render/src/lib.rs: `compile` reused a
    /// caller-owned `CompileScratch` instead of allocating ~8 fresh Vec/HashMap
    /// scheduling structures every frame.
    #[test]
    fn compile_reuses_scratch_allocations_across_frames() {
        let mut scratch = CompileScratch::default();

        // First call warms up the scratch's capacity (and the graph itself, plus
        // Compiled's own schedule/lifetimes maps, still allocate — this call is not
        // the one under test).
        let warmup = build_graph(10);
        let _ = warmup.compile(&mut scratch);

        // Second call, same shape: with the scratch reused, the only allocations
        // left are the two owned by `Compiled` (`schedule: Vec`, `lifetimes:
        // HashMap`) — nothing from the eliminated per-frame String names (fixed
        // separately by switching Resource/PassNode names to `&'static str`) and
        // nothing from the 8 scheduling scratch containers this test targets.
        let steady = build_graph(10);
        let allocs = count_allocs(|| {
            let _compiled = steady.compile(&mut scratch);
        });

        // Exactly 2: `Compiled.schedule` (Vec) + `Compiled.lifetimes` (HashMap).
        // Before the fix this call also allocated `edges`, `last_writer`,
        // `readers_since` (+ N nested Vecs, one per resource with readers),
        // `indegree`, `succ` (+ N nested Vecs), `order`, `ready`, `preds` (+ N
        // nested Vecs), `required`, `stack` — double digits of allocations for a
        // 12-pass graph. This asserts the steady-state count stays at the
        // irreducible minimum.
        assert_eq!(
            allocs, 2,
            "compile() should only allocate Compiled's own schedule+lifetimes \
             once CompileScratch is warm; got {allocs} allocations"
        );
    }

    /// Same graph, but using a fresh [`CompileScratch`] every call (the pre-fix
    /// shape) — documents how many allocations the fix removes for a
    /// representative ~12-pass frame, so the win is a concrete number rather than
    /// a vibe. Measured on this suite: 132 allocations from empty vs. 2 once
    /// `CompileScratch` is warm (~65x fewer for `compile`'s own scheduling work).
    #[test]
    fn compile_without_scratch_reuse_allocates_far_more() {
        let steady = build_graph(10);
        let allocs = count_allocs(|| {
            let mut fresh_scratch = CompileScratch::default();
            let _compiled = steady.compile(&mut fresh_scratch);
        });
        // Sanity bound: this is the "every frame from empty" baseline the fix
        // avoids. Not asserting an exact number (HashMap growth strategy isn't a
        // stable contract) — just that it is well above the warm-scratch count of 2.
        assert!(
            allocs > 10,
            "expected the from-empty compile to cost noticeably more than the \
             warm-scratch steady state (2); got {allocs}"
        );
    }
}
