//! Baked **vertex-cache deform** playback — general runtime infra (not knight-specific;
//! reusable for cloth / destruction / any pre-baked per-frame deformation).
//!
//! A [`DeformPlayer`] holds a decoded [`VertexCache`] (positions per frame, constant
//! topology — see [`crate::alembic`]/USD import) and, per mesh part, a per-frame-in-flight
//! **ring** of GPU vertex buffers. Each tick it rewrites THIS frame's ring buffer to the
//! current animation frame and [`DeformPlayer::patch_scene`] swaps the drawable to it — so a
//! frame still in flight reading ring slot `N-1` is never overwritten by the CPU writing
//! slot `N`. This is the real-time-safe form of the earlier single-buffer `vbuf.write` (which
//! was correct only for the deterministic headless capture). The ring + patch mirror the
//! CPU-morph path in [`crate::morph`]; no new shader/RHI — the normal static mesh pipeline
//! draws the deformed geometry.
//!
//! Deterministic (CPU-driven from the fixed-timestep clock) so headless captures reproduce.

use std::rc::Rc;

use dreamcoast_asset::{MaterialKind, MeshVertex, VcMesh, VertexCache};
use dreamcoast_core::glam::{Quat, Vec3};
use dreamcoast_scene::{LocalTransform, MeshInstance, Name, Parent, World};
use rhi::Device;
use tracing::{info, warn};

use crate::FRAMES_IN_FLIGHT;
use crate::mesh::{upload_geometry, vertex_slice_bytes};
use crate::registry::{
    GpuMesh, MaterialDesc, MaterialRegistry, MeshRegistry, representative_albedo,
};

/// Where + how big a deforming character/prop stands in the level (world metres; `scale` also
/// converts a source authored in cm to metres). Shared by the glTF/FBX overlay
/// ([`crate::character`]) and this vertex-cache path.
pub(crate) struct Placement {
    pub translation: Vec3,
    pub rotation_y_deg: f32,
    pub scale: f32,
}

impl Placement {
    /// Parse `"x,y,z,rotDeg,scale"` from an env override, falling back to `self`.
    pub(crate) fn with_env(mut self, var: &str) -> Self {
        if let Ok(s) = std::env::var(var) {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if v.len() == 5 {
                self.translation = Vec3::new(v[0], v[1], v[2]);
                self.rotation_y_deg = v[3];
                self.scale = v[4];
            } else {
                warn!("deform: {var} must be 'x,y,z,rotDeg,scale' — ignoring '{s}'");
            }
        }
        self
    }
}

/// Per-vertex normals from a position array + triangle indices (area-weighted face normals
/// accumulated then normalized) — recomputed each frame so the deforming surface shades
/// correctly.
fn compute_normals(pos: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut n = vec![[0f32; 3]; pos.len()];
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cr = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &i in t {
            let m = &mut n[i as usize];
            m[0] += cr[0];
            m[1] += cr[1];
            m[2] += cr[2];
        }
    }
    for v in &mut n {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 1e-8 {
            *v = [v[0] / l, v[1] / l, v[2] / l];
        } else {
            *v = [0.0, 1.0, 0.0];
        }
    }
    n
}

/// Build a mesh part's `MeshVertex` list for one frame (positions + recomputed normals, no
/// UVs — the cache carries none).
fn frame_vertices(m: &VcMesh, frame: usize) -> Vec<MeshVertex> {
    let f = frame.min(m.frames.len().saturating_sub(1));
    let pos = &m.frames[f];
    let nrm = compute_normals(pos, &m.indices);
    pos.iter()
        .zip(&nrm)
        .map(|(p, n)| MeshVertex {
            pos: *p,
            normal: *n,
            uv: [0.0, 0.0],
        })
        .collect()
}

/// A brushed-metal material for a texture-less deform cache (the knight default; the level can
/// override per entity). Two-sided since the baked shells aren't guaranteed watertight.
pub(crate) fn brushed_metal() -> MaterialDesc {
    let base = [0.58, 0.58, 0.60, 1.0];
    MaterialDesc {
        base_color: base,
        metallic: 0.6,
        roughness: 0.45,
        tex: [crate::NO_TEXTURE; 4],
        albedo: representative_albedo(None, base),
        alpha_cutoff: 0.0,
        kind: MaterialKind::Opaque,
        two_sided: true,
    }
}

/// One animated mesh part: the frame-0 `base_mesh` the scene list references (matched by `Rc`
/// identity), plus a per-frame-in-flight ring of vertex buffers rewritten each tick.
struct DeformPart {
    /// The registry mesh `build_scene` references for this part's drawable; drawables are
    /// matched to it by `Rc` identity, then swapped to `ring[fif]` in [`DeformPlayer::patch_scene`].
    base_mesh: Rc<GpuMesh>,
    /// Per-fif vertex buffers (frame geometry); `ring[fif]` is rewritten + drawn this frame.
    ring: Vec<Rc<GpuMesh>>,
    /// Index of this part in [`VertexCache::meshes`] (the source of per-frame positions).
    vc_index: usize,
}

/// Plays a decoded [`VertexCache`]: holds the cache + each part's per-fif buffer ring, and
/// rewrites this frame's ring buffer to the current animation frame each tick.
pub(crate) struct DeformPlayer {
    cache: VertexCache,
    /// One entry per `cache.meshes` part; `None` for an empty/degenerate part.
    parts: Vec<Option<DeformPart>>,
    time: f32,
}

impl DeformPlayer {
    /// Advance the clock by `dt` and rewrite THIS frame-in-flight's ring buffer for every part
    /// to the new animation frame. Real-time-safe: only `ring[fif]` is written, so an in-flight
    /// frame reading another slot is untouched. Call after this slot's fence has been waited.
    pub(crate) fn update(&mut self, fif: usize, dt: f32) -> anyhow::Result<()> {
        if self.cache.num_frames == 0 {
            return Ok(());
        }
        self.time += dt;
        let frame = ((self.time * self.cache.fps) as usize) % self.cache.num_frames;
        for part in self.parts.iter().flatten() {
            let m = &self.cache.meshes[part.vc_index];
            if m.frames.is_empty() {
                continue;
            }
            let verts = frame_vertices(m, frame);
            part.ring[fif].vbuf.write(vertex_slice_bytes(&verts))?;
        }
        Ok(())
    }

    /// Swap each animated drawable to this frame's ring buffer (matched by `Rc` identity to the
    /// part's `base_mesh`). Run on the freshly built scene list each frame, after [`Self::update`].
    /// Mirrors [`crate::morph::patch_scene`]'s CPU path.
    pub(crate) fn patch_scene(&self, scene: &mut [crate::SceneObject], fif: usize) {
        for obj in scene.iter_mut() {
            if let Some(part) = self
                .parts
                .iter()
                .flatten()
                .find(|p| Rc::ptr_eq(&obj.mesh, &p.base_mesh))
            {
                obj.mesh = part.ring[fif].clone();
            }
        }
    }
}

/// Spawn a decoded vertex cache into the level: upload each part's frame-0 geometry (the
/// registry base + a per-fif ring), spawn a drawable per part under a placement wrapper, and
/// return the player that animates them. The cache's parts are pre-assembled in one metre/Y-up
/// space, so a single wrapper transform places the whole thing. `material` is the shared
/// surface (the cache carries no textures); `label` names the wrapper for the scene graph.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn(
    device: &Device,
    world: &mut World,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    cache: VertexCache,
    place: &Placement,
    material: MaterialDesc,
    label: &str,
) -> anyhow::Result<DeformPlayer> {
    let root = world.spawn();
    world.insert(
        root,
        LocalTransform {
            translation: place.translation,
            rotation: Quat::from_rotation_y(place.rotation_y_deg.to_radians()),
            scale: Vec3::splat(place.scale),
        },
    );
    world.insert(root, Name(format!("deform:{label}")));

    let material = materials.add(material);

    let mut parts = Vec::with_capacity(cache.meshes.len());
    let mut n_parts = 0usize;
    for (vc_index, m) in cache.meshes.iter().enumerate() {
        if m.frames.is_empty() || m.indices.is_empty() {
            parts.push(None);
            continue;
        }
        // Frame-0 geometry for the registry base (the scene references it) + the per-fif ring.
        // The base carries an always-visible AABB: `build_scene` reads it once for the CPU
        // frustum cull, but the part's positions deform per frame beyond the frame-0 box.
        let verts = frame_vertices(m, 0);
        const ALWAYS_VISIBLE: [[f32; 3]; 2] = [[-1.0e9; 3], [1.0e9; 3]];
        let base_handle =
            meshes.upload_geometry_aabb(device, &verts, &m.indices, ALWAYS_VISIBLE)?;
        let base_mesh = meshes.get(base_handle);
        let mut ring = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let (vbuf, ibuf, index_count) = upload_geometry(device, &verts, &m.indices)?;
            ring.push(Rc::new(GpuMesh {
                vbuf,
                ibuf,
                index_count,
                vertex_count: verts.len() as u32,
                // Positions deform per frame (bounds vary); never frustum-cull a deform part.
                local_aabb: ALWAYS_VISIBLE,
            }));
        }
        parts.push(Some(DeformPart {
            base_mesh,
            ring,
            vc_index,
        }));

        let e = world.spawn();
        world.insert(e, LocalTransform::IDENTITY);
        world.insert(e, MeshInstance::new(base_handle, material));
        world.insert(e, Parent(root));
        n_parts += 1;
    }
    info!(
        "deform '{label}': {n_parts} parts, {} frames @ {} fps",
        cache.num_frames, cache.fps
    );
    // `DEFORM_START_S` (legacy `ABC_START_S`) seeds the playback clock (seconds) so headless
    // captures can sample different animation phases (otherwise deterministic at one frame).
    let time = std::env::var("DEFORM_START_S")
        .or_else(|_| std::env::var("ABC_START_S"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    Ok(DeformPlayer { cache, parts, time })
}

/// Default placement for the knight deform cache in Intel Sponza (metres, feet at y=0). Tunable
/// via `CHAR_KNIGHT_ABC` (`"x,y,z,rotDeg,scale"`).
pub(crate) fn knight_placement() -> Placement {
    Placement {
        translation: Vec3::new(3.5, 0.0, 0.0),
        rotation_y_deg: 90.0,
        scale: 1.0,
    }
    .with_env("CHAR_KNIGHT_ABC")
}
