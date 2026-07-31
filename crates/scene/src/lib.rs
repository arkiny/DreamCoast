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
//!
//! Around the entity storage sit the three pieces gameplay code needs but the
//! renderer does not: [`Resources`] (typed world singletons), [`Events`] (a
//! double-buffered message channel) and [`Commands`] (a deferred structural-change
//! buffer, the sanctioned way to spawn/despawn from a parallel system). All three
//! preserve the crate's determinism rule — ordered replay, insertion-ordered
//! iteration — because the draw list's stability depends on it.
//!
//! Animation is split along the same seam: [`sample_clip`] evaluates an
//! [`AnimationClip`] into an [`AnimPose`], [`blend_poses`] crossfades two poses, and
//! [`apply_pose`] commits one to the ECS — so gameplay can blend clips rather than
//! only play one. [`advance_animation`] is that pipeline wired to a looping clock.

mod animation;
mod commands;
mod components;
mod draw_list;
mod ecs;
mod events;
mod gltf_instance;
mod node;
mod pose;
mod resources;
mod schedule;
mod transform;

pub use animation::{
    AnimationClip, AnimationPlayer, ClipBuilder, LoopMode, MorphWeights, advance_animation,
    sample_clip, sample_clip_into,
};
pub use commands::{CommandTarget, Commands, DeferredEntity};
pub use components::{MaterialHandle, MeshHandle, MeshInstance, Name};
pub use draw_list::Drawable;
/// glTF keyframe interpolation mode — re-exported because [`ClipBuilder`] takes it.
pub use dreamcoast_asset::Interpolation;
pub use ecs::{Entity, World, WorldCell};
pub use events::Events;
pub use gltf_instance::{instantiate_gltf, instantiate_gltf_mapped};
pub use node::NodeRef;
pub use pose::{AnimPose, PoseEntry, Trs, apply_pose, blend_poses, blend_poses_into};
pub use resources::Resources;
pub use schedule::{Access, SystemSchedule};
pub use transform::{
    Children, LocalTransform, Parent, Spin, WorldTransform, advance_spin, propagate_transforms,
    propagate_transforms_parallel, propagate_transforms_system,
};
