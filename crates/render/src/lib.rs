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
//! The graph depends only on the `rhi` facade (like `engine-gui`); all GPU code
//! lives in the backends. Transient *memory aliasing* (Phase 5.3) layers on top
//! of the lifetime analysis computed here.
//!
//! ```ignore
//! let mut graph = RenderGraph::new();
//! let backbuffer = graph.import_backbuffer(swapchain.format(), extent);
//! let scene = graph.create_color("scene", scene_format, extent);
//! let depth = graph.create_depth("depth", extent);
//! graph.add_pass(PassInfo { name: "scene", color: Some((scene, Some(clear))),
//!                           depth: Some(depth), reads: vec![] },
//!                |ctx| { /* draw mesh */ });
//! graph.add_pass(PassInfo { name: "post", color: Some((backbuffer, None)),
//!                           depth: None, reads: vec![scene] },
//!                |ctx| { /* sample ctx.sampled_index(scene) */ });
//! graph.execute(&device, &mut pool, &cmd, &swapchain, image_index, true)?;
//! ```

use std::collections::HashMap;

use engine_core::EngineError;
use rhi::{
    ClearColor, CommandBuffer, DepthBuffer, Device, Extent2D, Format, RenderTarget,
    RenderTargetDesc, Swapchain, TransientHeap,
};

/// A pass's record closure: records draw commands, may fail (e.g. per-frame uploads).
type RecordFn<'a> = Box<dyn FnMut(&mut PassContext) -> Result<(), EngineError> + 'a>;

/// A virtual resource handle, valid within the graph that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResourceKind {
    Color,
    Depth,
}

/// A virtual resource: a transient texture the graph allocates, or the imported
/// swapchain backbuffer.
struct Resource {
    #[allow(dead_code)]
    name: String,
    kind: ResourceKind,
    extent: Extent2D,
    format: Format,
    /// `true` for the imported swapchain image (realized from the swapchain, not
    /// the pool).
    backbuffer: bool,
}

/// What a pass does to a resource.
struct PassNode<'a> {
    name: String,
    /// Single color attachment + its load behavior (`Some(clear)` clears,
    /// `None` loads).
    color: Option<(ResourceId, Option<ClearColor>)>,
    depth: Option<ResourceId>,
    reads: Vec<ResourceId>,
    record: RecordFn<'a>,
}

/// Declarative description of a pass (everything but the record closure).
pub struct PassInfo<'a> {
    pub name: &'a str,
    /// Color attachment written by the pass, with `Some(clear)` to clear or
    /// `None` to load.
    pub color: Option<(ResourceId, Option<ClearColor>)>,
    /// Depth attachment written by the pass (always cleared).
    pub depth: Option<ResourceId>,
    /// Resources sampled by the pass.
    pub reads: Vec<ResourceId>,
}

/// Per-pass context handed to a record closure during execution.
pub struct PassContext<'a> {
    cmd: &'a CommandBuffer,
    sampled: HashMap<ResourceId, u32>,
    extent: Extent2D,
}

impl<'a> PassContext<'a> {
    /// The frame command buffer (a render pass is already open).
    pub fn cmd(&self) -> &CommandBuffer {
        self.cmd
    }

    /// Bindless table index of a resource this pass declared as a read.
    pub fn sampled_index(&self, id: ResourceId) -> u32 {
        self.sampled[&id]
    }

    /// Extent of the pass's color attachment (viewport/scissor are already set).
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
            name: "backbuffer".to_string(),
            kind: ResourceKind::Color,
            extent,
            format,
            backbuffer: true,
        })
    }

    /// Declare a transient color target.
    pub fn create_color(&mut self, name: &str, format: Format, extent: Extent2D) -> ResourceId {
        self.push_resource(Resource {
            name: name.to_string(),
            kind: ResourceKind::Color,
            extent,
            format,
            backbuffer: false,
        })
    }

    /// Declare a transient depth target.
    pub fn create_depth(&mut self, name: &str, extent: Extent2D) -> ResourceId {
        self.push_resource(Resource {
            name: name.to_string(),
            kind: ResourceKind::Depth,
            extent,
            format: Format::Depth32Float,
            backbuffer: false,
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
            name: info.name.to_string(),
            color: info.color,
            depth: info.depth,
            reads: info.reads,
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
    fn compile(&self) -> Compiled {
        let n = self.passes.len();

        // Dependency edges: producer -> consumer (RAW), plus WAW/WAR. Because an
        // edge always points from an earlier-declared pass to a later one, the
        // graph is acyclic by construction.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut last_writer: HashMap<ResourceId, usize> = HashMap::new();
        let mut readers_since: HashMap<ResourceId, Vec<usize>> = HashMap::new();

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
                readers_since.insert(w, Vec::new());
            }
        }

        // Kahn topological sort, picking the lowest pass index among ready nodes
        // so the schedule stays close to declaration order.
        let mut indegree = vec![0usize; n];
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &edges {
            succ[a].push(b);
            indegree[b] += 1;
        }
        let mut order = Vec::with_capacity(n);
        let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
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
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &edges {
            preds[b].push(a);
        }
        let mut required = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for (i, pass) in self.passes.iter().enumerate() {
            if pass.writes_backbuffer(&self.resources) {
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

        let schedule: Vec<usize> = order.into_iter().filter(|&i| required[i]).collect();

        // Lifetime of each resource over the final schedule (first/last position
        // that references it). Used by transient aliasing.
        let mut lifetimes: HashMap<ResourceId, (usize, usize)> = HashMap::new();
        for (pos, &pass_idx) in schedule.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for r in pass.references() {
                lifetimes
                    .entry(r)
                    .and_modify(|(_, last)| *last = pos)
                    .or_insert((pos, pos));
            }
        }

        Compiled {
            schedule,
            lifetimes,
        }
    }

    /// Compile and run the graph against the RHI for one frame.
    ///
    /// `pool` caches realized targets across frames; the backbuffer is taken from
    /// `swapchain`/`image_index`. When `aliasing` is set, transient color targets
    /// are placed into a shared heap by [`Self::plan_aliasing`] so targets with
    /// non-overlapping lifetimes reuse memory; otherwise each gets its own
    /// dedicated allocation.
    pub fn execute(
        mut self,
        device: &Device,
        pool: &mut ResourcePool,
        cmd: &CommandBuffer,
        swapchain: &Swapchain,
        image_index: u32,
        aliasing: bool,
    ) -> Result<(), EngineError> {
        let compiled = self.compile();
        if tracing::enabled!(tracing::Level::TRACE) {
            let names: Vec<&str> = compiled
                .schedule
                .iter()
                .map(|&i| self.passes[i].name.as_str())
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
                            if aliasing {
                                e.insert(ColorLoc::Aliased(r));
                            } else {
                                let desc = render_target_desc(res);
                                e.insert(ColorLoc::Pooled(pool.acquire_color(device, desc)?));
                            }
                        }
                    }
                    ResourceKind::Depth => {
                        if let std::collections::hash_map::Entry::Vacant(e) = depth_slots.entry(r) {
                            e.insert(pool.acquire_depth(device, res.extent)?);
                        }
                    }
                }
            }
        }

        // Phase 2 — record (immutable pool access).
        let mut backbuffer_is_rt = false;
        for &pass_idx in &compiled.schedule {
            // Barriers: reads -> sampled.
            for &r in &self.passes[pass_idx].reads {
                if let Some(loc) = color_locs.get(&r) {
                    cmd.rt_to_sampled(pool.color_target(loc));
                }
            }

            // Resolve the color attachment and transition it for writing.
            let (color, clear) = self.passes[pass_idx]
                .color
                .expect("pass has no color target");
            let color_res = &self.resources[color.0];
            let extent = color_res.extent;
            if color_res.backbuffer {
                if !backbuffer_is_rt {
                    cmd.transition_to_render_target(swapchain, image_index);
                    backbuffer_is_rt = true;
                }
            } else {
                let target = pool.color_target(&color_locs[&color]);
                if alias_barrier.get(&color).copied().unwrap_or(false) {
                    cmd.aliasing_barrier(target);
                } else {
                    cmd.rt_to_render_target(target);
                }
            }

            // Begin the render pass.
            let depth_ref = self.passes[pass_idx]
                .depth
                .map(|d| pool.depth(depth_slots[&d]));
            if color_res.backbuffer {
                cmd.begin_rendering(swapchain, image_index, clear, depth_ref);
                cmd.set_viewport_scissor(swapchain);
            } else {
                cmd.begin_rendering_target(
                    pool.color_target(&color_locs[&color]),
                    clear,
                    depth_ref,
                );
                cmd.set_viewport_scissor_extent(extent);
            }

            // Build the sampled-index map for the pass and run its record closure.
            let sampled: HashMap<ResourceId, u32> = self.passes[pass_idx]
                .reads
                .iter()
                .filter_map(|&r| {
                    color_locs
                        .get(&r)
                        .map(|loc| (r, pool.color_target(loc).bindless_index()))
                })
                .collect();
            let mut ctx = PassContext {
                cmd,
                sampled,
                extent,
            };
            (self.passes[pass_idx].record)(&mut ctx)?;
            cmd.end_rendering();
        }

        if backbuffer_is_rt {
            cmd.transition_to_present(swapchain, image_index);
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
        for (&id, &(first, last)) in &compiled.lifetimes {
            let res = &self.resources[id.0];
            if res.backbuffer || res.kind != ResourceKind::Color {
                continue;
            }
            let desc = render_target_desc(res);
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

/// Round `value` up to a multiple of `align` (a power-of-two or any `>= 1`).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

fn render_target_desc(res: &Resource) -> RenderTargetDesc {
    RenderTargetDesc {
        width: res.extent.width,
        height: res.extent.height,
        format: res.format,
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
    /// Resources this pass writes (color + depth).
    fn writes(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.color.map(|(id, _)| id).into_iter().chain(self.depth)
    }

    /// All resources this pass touches (writes + reads).
    fn references(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.writes().chain(self.reads.iter().copied())
    }

    fn writes_backbuffer(&self, resources: &[Resource]) -> bool {
        self.color
            .map(|(id, _)| resources[id.0].backbuffer)
            .unwrap_or(false)
    }
}

/// Output of compilation: the culled, ordered pass schedule and resource
/// lifetimes (first/last schedule position).
struct Compiled {
    schedule: Vec<usize>,
    /// First/last schedule position each resource is referenced (drives aliasing).
    lifetimes: HashMap<ResourceId, (usize, usize)>,
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
