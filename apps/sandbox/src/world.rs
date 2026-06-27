//! Level-chunk streaming (Phase 12 Stage D).
//!
//! A `.world` ([`LevelGraph`]) is a graph of chunks, each a `.level` placed at a
//! world-space origin. As the camera moves, chunks within `stream_radius` are
//! streamed in and those outside are dropped. Each loaded chunk owns a **self-
//! contained arena** — its own ECS `World` + mesh/material registries + textures — so
//! unloading is just dropping the arena: its entities vanish and its GPU resources
//! free (shared geometry via `Rc` refcount). No per-material PSO is created on
//! stream-in (this engine is bindless — all chunks reuse the same pipelines), so
//! there is no pipeline hitch and no async-PSO machinery is needed here.

use std::path::PathBuf;

use dreamcoast_asset::{LevelGraph, WorldChunk};
use dreamcoast_core::glam::Vec3;
use dreamcoast_scene::{World, propagate_transforms};
use rhi::{Device, Texture};

use crate::SceneObject;
use crate::level::build_level;
use crate::registry::{MaterialRegistry, MeshRegistry, build_scene};

/// A resident chunk's arena. Dropping it despawns the chunk (the `World` drops) and
/// frees its GPU resources.
struct LoadedChunk {
    index: usize,
    world: World,
    meshes: MeshRegistry,
    materials: MaterialRegistry,
    _textures: Vec<Texture>,
}

/// The streaming manager: the level graph + the currently resident chunks.
pub(crate) struct Streaming {
    graph: LevelGraph,
    /// Base directory for resolving a chunk's `.level` path.
    dir: PathBuf,
    loaded: Vec<LoadedChunk>,
}

impl Streaming {
    pub(crate) fn new(graph: LevelGraph, dir: PathBuf) -> Self {
        Self {
            graph,
            dir,
            loaded: Vec::new(),
        }
    }

    pub(crate) fn chunk_count(&self) -> usize {
        self.graph.chunks.len()
    }

    pub(crate) fn stream_radius(&self) -> f32 {
        self.graph.stream_radius
    }

    /// The chunk indices currently resident, sorted (for the UI + verification logs).
    pub(crate) fn loaded_indices(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.loaded.iter().map(|c| c.index).collect();
        v.sort_unstable();
        v
    }

    /// A chunk's display name.
    pub(crate) fn chunk_name(&self, i: usize) -> &str {
        &self.graph.chunks[i].name
    }

    /// Stream chunks in/out around camera position `cam`: drop every resident chunk
    /// beyond the radius (with hysteresis to avoid boundary thrash) and load the
    /// single nearest missing in-range chunk (a per-frame budget of one load, so a
    /// stream-in can't stall the frame). Returns whether the resident set changed.
    pub(crate) fn update(&mut self, device: &Device, cam: Vec3) -> anyhow::Result<bool> {
        let radius = self.graph.stream_radius;
        let origins: Vec<Vec3> = self
            .graph
            .chunks
            .iter()
            .map(|c| Vec3::from(c.origin))
            .collect();

        let before = self.loaded.len();
        self.loaded
            .retain(|c| origins[c.index].distance(cam) <= radius * 1.25);
        let mut changed = self.loaded.len() != before;

        // Nearest missing in-range chunk.
        let next = (0..self.graph.chunks.len())
            .filter(|&i| !self.loaded.iter().any(|c| c.index == i))
            .map(|i| (i, origins[i].distance(cam)))
            .filter(|&(_, d)| d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = next {
            self.load_chunk(device, i)?;
            changed = true;
        }
        Ok(changed)
    }

    fn load_chunk(&mut self, device: &Device, i: usize) -> anyhow::Result<()> {
        let WorldChunk { level, origin, .. } = &self.graph.chunks[i];
        let level_data = crate::level::load(&self.dir.join(level))?;
        let mut world = World::new();
        let mut meshes = MeshRegistry::new();
        let mut materials = MaterialRegistry::new();
        let mut textures: Vec<Texture> = Vec::new();
        build_level(
            device,
            &level_data,
            &mut world,
            &mut meshes,
            &mut materials,
            &mut textures,
            Vec3::from(*origin),
        )?;
        propagate_transforms(&mut world);
        self.loaded.push(LoadedChunk {
            index: i,
            world,
            meshes,
            materials,
            _textures: textures,
        });
        Ok(())
    }

    /// The frame's draw list: the union of every resident chunk's draw list (each
    /// chunk's entity transforms already include its world origin).
    pub(crate) fn build_scene(&self) -> Vec<SceneObject> {
        let mut out = Vec::new();
        for c in &self.loaded {
            out.extend(build_scene(&c.world, &c.meshes, &c.materials));
        }
        out
    }
}

/// The built-in demo world: three Lantern chunks in a row along X, linked into a line.
pub(crate) fn demo_world() -> LevelGraph {
    let chunk = |name: &str, x: f32| WorldChunk {
        name: name.to_owned(),
        level: "lanterns.level".to_owned(),
        origin: [x, 0.0, 0.0],
    };
    LevelGraph {
        chunks: vec![
            chunk("west", -6.0),
            chunk("center", 0.0),
            chunk("east", 6.0),
        ],
        edges: vec![(0, 1), (1, 2)],
        stream_radius: 12.0,
    }
}

/// Write the built-in demo `.world` into `dir` if missing; return its path.
pub(crate) fn ensure_world_file(dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("demo.world");
    if !path.exists() {
        demo_world().save_ron(&path)?;
    }
    Ok(path)
}
