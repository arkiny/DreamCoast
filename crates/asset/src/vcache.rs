//! Baked **vertex-animation cache** — the neutral, importer-agnostic type.
//!
//! A vertex cache is a set of meshes, each with constant topology (a triangle-index
//! list) and one deformed position array **per frame**. It is the natural target for a
//! baked deformation deliverable that carries no rig — the Intel Sponza knight ships as
//! exactly this, in two formats that decode to this same type:
//! [`crate::alembic`] (Ogawa `.abc`) and [`crate::usd`] (ASCII `.usda` point cache).
//!
//! Keeping the type here (not in either importer) is the single source of truth the
//! separate-asset cook ([`crate::dcasset::write_vcache`] / [`crate::cook::load_or_cook_vcache`])
//! serializes, so both importers feed one cooked-`.dcasset` path and the runtime plays
//! either without a live 665 MB / 1.4 GB decode.

/// One mesh of a [`VertexCache`]: a constant triangle-index list plus a position array
/// per frame (all frames share topology). Positions are in the engine's metres/Y-up.
pub struct VcMesh {
    pub name: String,
    pub indices: Vec<u32>,
    /// `frames[f]` = per-vertex positions for frame `f` (all the same length).
    pub frames: Vec<Vec<[f32; 3]>>,
}

/// A baked vertex-animation cache: a set of meshes each carrying every frame's deformed
/// positions. Playback = pick `frames[frame]`. Decoded from an Alembic `.abc`
/// ([`crate::alembic::read_vertex_cache`]) or an ASCII USD `.usda`
/// ([`crate::usd::read_vertex_cache`]); cooked to / loaded from a `.dcasset`.
pub struct VertexCache {
    pub meshes: Vec<VcMesh>,
    pub num_frames: usize,
    pub fps: f32,
}
