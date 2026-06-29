//! `dreamcoast-scene` — the engine's scene representation (Phase 12).
//!
//! A from-scratch, single-threaded **ECS** is the scene core (no external ECS
//! crate, mirroring the engine's from-scratch RHI/render-graph philosophy). The
//! "scene graph" is not a separate tree but a *transform hierarchy expressed as
//! components* on the ECS: [`transform::Parent`] + [`transform::LocalTransform`]
//! propagate to [`transform::WorldTransform`] via
//! [`transform::propagate_transforms`].
//!
//! This crate is **RHI-agnostic**: it speaks only `glam`. Meshes and materials are
//! referenced by opaque [`MeshHandle`] / [`MaterialHandle`] indices into registries
//! the renderer owns — that handle indirection is the seam that keeps the scene
//! free of GPU types. The renderer turns a [`draw_list::Drawable`] list into actual
//! draw calls.

mod animation;
mod components;
mod draw_list;
mod ecs;
mod gltf_instance;
mod node;
mod schedule;
mod transform;

pub use animation::{AnimationClip, AnimationPlayer, MorphWeights, advance_animation};
pub use components::{MaterialHandle, MeshHandle, MeshInstance, Name};
pub use draw_list::Drawable;
pub use ecs::{Entity, World, WorldCell};
pub use gltf_instance::{instantiate_gltf, instantiate_gltf_mapped};
pub use node::NodeRef;
pub use schedule::{Access, SystemSchedule};
pub use transform::{
    Children, LocalTransform, Parent, Spin, WorldTransform, advance_spin, propagate_transforms,
    propagate_transforms_parallel, propagate_transforms_system,
};
