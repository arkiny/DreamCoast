//! Deferred **command buffer** for structural changes.
//!
//! [`WorldCell`](crate::WorldCell) deliberately offers no spawn/despawn/remove:
//! those touch shared `World` bookkeeping (the entity allocator, the storage map)
//! and would race between workers. A system that wants to create or destroy
//! entities therefore *records* the intent into a [`Commands`] buffer and the caller
//! [`apply`](Commands::apply)s it single-threaded once the parallel region has
//! closed — the same "declare now, execute at a safe point" shape the render graph
//! uses for barriers.
//!
//! **Determinism.** Commands replay in exactly the order they were recorded, so a
//! given buffer always produces the same entity ids and the same insertion order in
//! every storage — which is what keeps the draw list (and thus TLAS instance order)
//! stable frame to frame.
//!
//! **Deferred entity ids.** A recorded spawn cannot return a real [`Entity`]: the
//! generational allocator lives in `World`, hands out slots one at a time, and the
//! recorder holds no borrow of it (that is the whole point). Reserving a range up
//! front would have to be done per buffer, would burn ids for spawns that are later
//! cancelled, and would make two buffers' reservations order-dependent. Instead
//! [`Commands::spawn`] returns a [`DeferredEntity`] — an index into *this buffer's*
//! spawn list — which `apply` resolves to the real id as it replays. Deferred and
//! already-live ids are interchangeable as command targets via [`CommandTarget`], so
//! `insert`/`despawn` read the same at the call site either way.
//!
//! **Dead targets are dropped, never resurrected.** `World::insert` does not check
//! liveness (it is the hot single-threaded path), so a command aimed at an entity
//! that died between recording and apply would otherwise plant a zombie component
//! into a recycled slot and into every query. `apply` checks
//! [`is_alive`](World::is_alive) for every resolved target and skips the command
//! instead.

use crate::ecs::{Entity, World};

/// A placeholder for an entity that does not exist yet: the index of a recorded
/// [`Commands::spawn`] within its own buffer. Only meaningful to the buffer that
/// produced it, and only until that buffer is applied.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeferredEntity(u32);

impl DeferredEntity {
    /// The spawn's position in the recording buffer.
    #[inline]
    pub fn slot(self) -> u32 {
        self.0
    }
}

/// What a command acts on: an entity that already exists, or one this buffer is
/// about to spawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommandTarget {
    /// An id obtained from [`World::spawn`] before recording.
    Existing(Entity),
    /// An id obtained from [`Commands::spawn`] during recording.
    Deferred(DeferredEntity),
}

impl From<Entity> for CommandTarget {
    fn from(e: Entity) -> Self {
        CommandTarget::Existing(e)
    }
}

impl From<DeferredEntity> for CommandTarget {
    fn from(d: DeferredEntity) -> Self {
        CommandTarget::Deferred(d)
    }
}

/// Type-erased "move this value into the world" step.
///
/// A boxed trait object rather than a closure so the component value is stored
/// inline and moved (not cloned) into its storage on apply. `Send` because buffers
/// are filled on worker threads; see [`Commands::insert`].
trait InsertCommand: Send {
    fn insert(self: Box<Self>, world: &mut World, e: Entity);
}

struct TypedInsert<T>(T);

impl<T: 'static + Send> InsertCommand for TypedInsert<T> {
    fn insert(self: Box<Self>, world: &mut World, e: Entity) {
        world.insert(e, self.0);
    }
}

/// One recorded structural operation. Removal needs no boxed payload — a
/// monomorphised `fn` pointer carries the component type and is `Send + Sync` and
/// allocation-free.
enum Command {
    Spawn(DeferredEntity),
    Despawn(CommandTarget),
    Insert(CommandTarget, Box<dyn InsertCommand>),
    Remove(CommandTarget, fn(&mut World, Entity)),
}

/// Monomorphised body of a deferred `remove::<T>` — the component is dropped on the
/// applying thread, so `T` needs no `Send` bound.
fn remove_typed<T: 'static>(world: &mut World, e: Entity) {
    world.remove::<T>(e);
}

/// A recorder of structural changes to replay against a [`World`] later.
///
/// The buffer is `Send`, so a worker can fill one and hand it back for the caller to
/// apply. Reusing one buffer across frames is fine and cheap: [`apply`](Self::apply)
/// drains it and keeps the allocation.
#[derive(Default)]
pub struct Commands {
    cmds: Vec<Command>,
    /// Number of [`spawn`](Self::spawn)s recorded — also the next deferred slot.
    spawns: u32,
}

impl Commands {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an entity creation and return the handle to address it with until the
    /// buffer is applied.
    pub fn spawn(&mut self) -> DeferredEntity {
        let d = DeferredEntity(self.spawns);
        self.spawns += 1;
        self.cmds.push(Command::Spawn(d));
        d
    }

    /// Record a despawn. A target that is already dead (or a deferred id whose spawn
    /// is not in this buffer) is a no-op at apply time.
    pub fn despawn(&mut self, target: impl Into<CommandTarget>) {
        self.cmds.push(Command::Despawn(target.into()));
    }

    /// Record attaching `value` to `target` (replacing any existing `T`).
    ///
    /// `T: Send` because the value is stored in the buffer, which is typically built
    /// on a worker thread and moved back to the applying thread.
    pub fn insert<T: 'static + Send>(&mut self, target: impl Into<CommandTarget>, value: T) {
        self.cmds
            .push(Command::Insert(target.into(), Box::new(TypedInsert(value))));
    }

    /// Record removing component `T` from `target`. The removed value is dropped
    /// during apply.
    pub fn remove<T: 'static>(&mut self, target: impl Into<CommandTarget>) {
        self.cmds
            .push(Command::Remove(target.into(), remove_typed::<T>));
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.cmds.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// Discard every recorded command without touching a world.
    pub fn clear(&mut self) {
        self.cmds.clear();
        self.spawns = 0;
    }

    /// Replay every command against `world` **in recording order**, then empty the
    /// buffer for reuse.
    ///
    /// Commands whose target cannot be resolved to a live entity are skipped (see
    /// the module docs), so a despawn recorded — or performed directly on the world
    /// — before a later `insert` on the same entity wins.
    pub fn apply(&mut self, world: &mut World) {
        // Deferred slot -> real id, filled as Spawn commands replay. `None` means
        // "not spawned yet, or spawned and since despawned".
        let mut spawned: Vec<Option<Entity>> = vec![None; self.spawns as usize];

        for cmd in self.cmds.drain(..) {
            match cmd {
                Command::Spawn(d) => {
                    spawned[d.0 as usize] = Some(world.spawn());
                }
                Command::Despawn(t) => {
                    if let Some(e) = resolve(t, &spawned, world) {
                        world.despawn(e);
                        if let CommandTarget::Deferred(d) = t {
                            // Drop the mapping so later commands on this handle
                            // cannot touch the (now recyclable) slot.
                            spawned[d.0 as usize] = None;
                        }
                    }
                }
                Command::Insert(t, payload) => {
                    if let Some(e) = resolve(t, &spawned, world) {
                        payload.insert(world, e);
                    }
                }
                Command::Remove(t, remove) => {
                    if let Some(e) = resolve(t, &spawned, world) {
                        remove(world, e);
                    }
                }
            }
        }
        self.spawns = 0;
    }
}

/// Resolve a target to a **live** entity, or `None` if it is unspawned, already
/// despawned, or a stale id.
fn resolve(t: CommandTarget, spawned: &[Option<Entity>], world: &World) -> Option<Entity> {
    let e = match t {
        CommandTarget::Existing(e) => e,
        CommandTarget::Deferred(d) => (*spawned.get(d.0 as usize)?)?,
    };
    world.is_alive(e).then_some(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Hp(u32);
    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Tag;

    #[test]
    fn deferred_spawn_and_insert() {
        let mut w = World::new();
        let mut c = Commands::new();
        let d = c.spawn();
        c.insert(d, Hp(10));
        c.insert(d, Tag);
        assert_eq!(c.len(), 3);
        c.apply(&mut w);

        assert!(c.is_empty(), "apply must drain the buffer");
        let hits: Vec<Hp> = w.iter::<Hp>().map(|(_, h)| *h).collect();
        assert_eq!(hits, vec![Hp(10)]);
        let (e, _) = w.iter::<Hp>().next().unwrap();
        assert!(w.is_alive(e));
        assert_eq!(w.get::<Tag>(e), Some(&Tag));
    }

    #[test]
    fn commands_apply_in_recorded_order() {
        let mut w = World::new();
        let e = w.spawn();
        let mut c = Commands::new();
        // Last write wins only if replay is ordered.
        c.insert(e, Hp(1));
        c.insert(e, Hp(2));
        c.insert(e, Hp(3));
        c.apply(&mut w);
        assert_eq!(w.get::<Hp>(e), Some(&Hp(3)));

        // Insert-then-remove and remove-then-insert must differ.
        c.insert(e, Hp(9));
        c.remove::<Hp>(e);
        c.apply(&mut w);
        assert_eq!(w.get::<Hp>(e), None);

        c.remove::<Hp>(e);
        c.insert(e, Hp(9));
        c.apply(&mut w);
        assert_eq!(w.get::<Hp>(e), Some(&Hp(9)));
    }

    #[test]
    fn deferred_spawn_order_is_stable() {
        // Two identical buffers must produce identical ids and iteration order.
        let mut w = World::new();
        let mut c = Commands::new();
        for i in 0..4u32 {
            let d = c.spawn();
            c.insert(d, Hp(i));
        }
        c.apply(&mut w);
        let order: Vec<u32> = w.iter::<Hp>().map(|(_, h)| h.0).collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
        let ids: Vec<u32> = w.iter::<Hp>().map(|(e, _)| e.index()).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn despawn_before_apply_drops_later_commands() {
        // The parallel-region hazard: a system records work for an entity that the
        // main thread (or an earlier command) kills first.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Hp(5));

        let mut c = Commands::new();
        c.insert(e, Hp(99));
        c.remove::<Hp>(e);
        w.despawn(e); // dies between recording and apply
        c.apply(&mut w);

        assert!(!w.is_alive(e));
        // No zombie component may survive in the recycled slot.
        assert_eq!(w.iter::<Hp>().count(), 0);
        let reused = w.spawn();
        assert_eq!(reused.index(), e.index());
        assert_eq!(w.get::<Hp>(reused), None);
    }

    #[test]
    fn despawn_recorded_before_insert_wins() {
        let mut w = World::new();
        let e = w.spawn();
        let mut c = Commands::new();
        c.despawn(e);
        c.insert(e, Hp(1)); // aimed at a corpse -> dropped
        c.apply(&mut w);
        assert!(!w.is_alive(e));
        assert_eq!(w.iter::<Hp>().count(), 0);
    }

    #[test]
    fn deferred_entity_can_be_despawned_in_the_same_buffer() {
        let mut w = World::new();
        let mut c = Commands::new();
        let keep = c.spawn();
        let doomed = c.spawn();
        c.insert(keep, Hp(1));
        c.insert(doomed, Hp(2));
        c.despawn(doomed);
        c.insert(doomed, Hp(3)); // after death -> dropped
        c.apply(&mut w);

        let live: Vec<u32> = w.iter::<Hp>().map(|(_, h)| h.0).collect();
        assert_eq!(live, vec![1]);
    }

    #[test]
    fn commands_on_unspawned_deferred_id_are_skipped() {
        // A handle from another buffer (or one whose spawn was cleared) resolves to
        // nothing rather than panicking or hitting a wrong entity.
        let mut w = World::new();
        let mut a = Commands::new();
        let d = a.spawn();
        a.clear();

        let mut b = Commands::new();
        b.insert(d, Hp(7));
        b.despawn(d);
        b.apply(&mut w);
        assert_eq!(w.iter::<Hp>().count(), 0);
    }

    #[test]
    fn stale_generation_target_is_skipped() {
        let mut w = World::new();
        let old = w.spawn();
        w.despawn(old);
        let fresh = w.spawn(); // same slot, bumped generation

        let mut c = Commands::new();
        c.insert(old, Hp(1));
        c.apply(&mut w);
        assert_eq!(w.get::<Hp>(fresh), None, "stale id must not write the slot");
    }

    #[test]
    fn reused_buffer_resets_deferred_slots() {
        let mut w = World::new();
        let mut c = Commands::new();
        let d0 = c.spawn();
        c.insert(d0, Hp(1));
        c.apply(&mut w);
        // Second round starts numbering from zero again.
        let d1 = c.spawn();
        assert_eq!(d1.slot(), 0);
        c.insert(d1, Hp(2));
        c.apply(&mut w);
        let all: Vec<u32> = w.iter::<Hp>().map(|(_, h)| h.0).collect();
        assert_eq!(all, vec![1, 2]);
    }

    #[test]
    fn buffer_is_send() {
        // Compile-time contract: worker threads fill these.
        fn assert_send<T: Send>() {}
        assert_send::<Commands>();
        let mut c = Commands::new();
        let d = c.spawn();
        c.insert(d, Hp(1));
        let mut c = std::thread::spawn(move || {
            c.insert(d, Tag);
            c
        })
        .join()
        .unwrap();
        let mut w = World::new();
        c.apply(&mut w);
        assert_eq!(w.iter::<Tag>().count(), 1);
    }
}
