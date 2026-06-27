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

mod components;
mod draw_list;
mod ecs;
mod gltf_instance;
mod node;
mod transform;

pub use components::{MaterialHandle, MeshHandle, MeshInstance, Name};
pub use draw_list::Drawable;
pub use ecs::{Entity, World};
pub use gltf_instance::instantiate_gltf;
pub use node::NodeRef;
pub use transform::{Children, LocalTransform, Parent, WorldTransform, propagate_transforms};
