//! A thin tree-style builder over the ECS.
//!
//! [`NodeRef`] makes scene authoring *read* like a tree (`.with(...)`, `.child()`),
//! but it is pure sugar: every call spawns/inserts directly into the one [`World`].
//! There is no second store — no retained tree to keep in sync with the ECS.

use crate::components::Name;
use crate::ecs::{Entity, World};
use crate::transform::{Children, LocalTransform, Parent};

/// A builder cursor at one entity. Chain `.with(component)` to attach data and
/// `.child()` to spawn a parented sub-entity.
pub struct NodeRef<'w> {
    world: &'w mut World,
    entity: Entity,
}

impl<'w> NodeRef<'w> {
    pub(crate) fn new(world: &'w mut World, entity: Entity) -> Self {
        Self { world, entity }
    }

    /// The entity this cursor points at.
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Attach (or replace) a component, returning the cursor for chaining.
    pub fn with<T: 'static>(self, component: T) -> Self {
        self.world.insert(self.entity, component);
        self
    }

    /// Set this node's local transform (shorthand for `.with(local)`).
    pub fn set_local(self, local: LocalTransform) -> Self {
        self.with(local)
    }

    /// Name this node (shorthand for `.with(Name(..))`).
    pub fn named(self, name: impl Into<String>) -> Self {
        self.with(Name(name.into()))
    }

    /// Spawn a child entity parented to this node, returning a cursor at the child.
    /// Maintains both the child's [`Parent`] link and this node's [`Children`] list.
    pub fn child(&mut self) -> NodeRef<'_> {
        let parent = self.entity;
        let child = self.world.spawn();
        self.world.insert(child, Parent(parent));
        match self.world.get_mut::<Children>(parent) {
            Some(children) => children.0.push(child),
            None => self.world.insert(parent, Children(vec![child])),
        }
        NodeRef::new(self.world, child)
    }
}
