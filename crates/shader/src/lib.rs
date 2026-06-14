//! Compiled shader access.
//!
//! The build script ([`build.rs`]) compiles every `.slang` source to both
//! SPIR-V and DXIL and writes the body of this module into `OUT_DIR`. Each entry
//! point yields two accessors, `<key>_spirv()` and `<key>_dxil()`, returning the
//! bytecode for the matching backend (or `None` when slangc was unavailable at
//! build time).
//!
//! Example: [`triangle_vs_spirv`] / [`triangle_fs_dxil`].

/// Bytecode format expected by a given RHI backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderFormat {
    /// SPIR-V, consumed by the Vulkan backend.
    SpirV,
    /// DXIL, consumed by the D3D12 backend.
    Dxil,
}

// Generated accessors (`pub fn <key>_spirv()/_dxil() -> Option<&'static [u8]>`).
include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
