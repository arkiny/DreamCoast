//! The streaming level graph (Phase 12 Stage D): an on-disk `.world` describing a
//! graph of level **chunks**.
//!
//! A [`LevelGraph`] is named to stay distinct from the runtime ECS `World`: this is
//! the *streaming data* (what content exists where), deserialized from a `.world`
//! file; the ECS `World` is the live container the renderer queries. Each
//! [`WorldChunk`] references a `.level` file and a world-space origin; edges record
//! adjacency. Like [`crate::LevelData`], the model is serde-ready so the same data
//! cooks to a binary `.dcasset` later.

use std::path::Path;

use dreamcoast_core::EngineError;
use serde::{Deserialize, Serialize};

/// One chunk: a level placed at a world-space origin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldChunk {
    pub name: String,
    /// Path to the chunk's `.level` file (its content).
    pub level: String,
    /// World-space offset applied to every entity in the chunk's level.
    pub origin: [f32; 3],
}

/// A graph of level chunks + the streaming radius.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct LevelGraph {
    pub chunks: Vec<WorldChunk>,
    /// Adjacency between chunk indices (undirected portals/links).
    pub edges: Vec<(usize, usize)>,
    /// Chunks within this distance of the camera are streamed in.
    pub stream_radius: f32,
}

impl LevelGraph {
    /// Load a level graph from a RON `.world` file.
    pub fn load_ron(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| EngineError::Asset(format!("world read: {e}")))?;
        ron::from_str(&text).map_err(|e| EngineError::Asset(format!("world parse: {e}")))
    }

    /// Serialize this level graph to RON text (pretty-printed for hand-editing).
    pub fn to_ron(&self) -> Result<String, EngineError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| EngineError::Asset(format!("world serialize: {e}")))
    }

    /// Save this level graph to a RON `.world` file.
    pub fn save_ron(&self, path: impl AsRef<Path>) -> Result<(), EngineError> {
        std::fs::write(path, self.to_ron()?)
            .map_err(|e| EngineError::Asset(format!("world write: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ron_roundtrip() {
        let graph = LevelGraph {
            chunks: vec![
                WorldChunk {
                    name: "a".into(),
                    level: "lanterns.level".into(),
                    origin: [-6.0, 0.0, 0.0],
                },
                WorldChunk {
                    name: "b".into(),
                    level: "lanterns.level".into(),
                    origin: [0.0, 0.0, 0.0],
                },
            ],
            edges: vec![(0, 1)],
            stream_radius: 8.0,
        };
        let text = graph.to_ron().expect("serialize");
        let parsed: LevelGraph = ron::from_str(&text).expect("parse");
        assert_eq!(parsed, graph);
    }
}
