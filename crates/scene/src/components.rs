//! Scene components beyond the transform hierarchy.
//!
//! Meshes and materials are referenced by opaque handles — indices into registries
//! the *renderer* owns. This indirection is what keeps `dreamcoast-scene` free of any
//! GPU/RHI types: the scene says "draw mesh #2 with material #5 here", and the
//! renderer resolves those to buffers and pipeline state.

/// Index of an uploaded mesh in the renderer's mesh registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshHandle(pub u32);

/// Index of a material in the renderer's material registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialHandle(pub u32);

/// A renderable: which mesh, which material, and whether it casts a shadow. The
/// transform comes from the entity's [`crate::WorldTransform`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub casts_shadow: bool,
}

impl MeshInstance {
    /// A shadow-casting instance of `mesh` with `material`.
    pub fn new(mesh: MeshHandle, material: MaterialHandle) -> Self {
        Self {
            mesh,
            material,
            casts_shadow: true,
        }
    }

    /// Builder: set whether this instance casts a shadow.
    pub fn with_shadow(mut self, casts_shadow: bool) -> Self {
        self.casts_shadow = casts_shadow;
        self
    }
}

/// An optional human-readable name (debug UI, level authoring, glTF node names).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name(pub String);
