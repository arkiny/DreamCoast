//! Vertex-cache chunk codec — a cooked baked deformation cache (Track B).
//!
//! Serializes a [`VertexCache`] (from an Alembic `.abc` or an ASCII USD `.usda`) into one
//! `.dcasset` chunk so the runtime plays the knight animation without a live 665 MB /
//! 1.4 GB decode each launch. Pure little-endian + CPU, so the bytes are deterministic and
//! backend-independent (the cache is geometry only — no textures/rig).
//!
//! Layout (one `CHUNK_VCACHE`): `num_frames u32`, `fps f32`, `mesh_count u32`, then per
//! mesh: `name` (len-prefixed), `index_count u32` + `u32` indices, `frame_count u32`,
//! `vert_count u32`, then `frame_count × vert_count × 3` `f32` positions (all frames share
//! the vertex count).

use dreamcoast_core::EngineError;

use super::{CHUNK_VCACHE, Header, Reader, Writer, open_chunk, write_single_chunk};
use crate::vcache::{VcMesh, VertexCache};

/// Serialize a [`VertexCache`] into a `.dcasset` buffer (one vertex-cache chunk).
/// `src_hash` is the invalidation key (the source file's cheap metadata key — see
/// [`crate::cook::load_or_cook_vcache`]).
pub fn write_vcache(cache: &VertexCache, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_VCACHE, &encode(cache), src_hash)
}

/// Decode a vertex-cache `.dcasset` buffer into its [`Header`] and [`VertexCache`].
pub fn read_vcache(bytes: &[u8]) -> Result<(Header, VertexCache), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_VCACHE, "vcache")?;
    Ok((header, decode(&mut r)?))
}

fn encode(cache: &VertexCache) -> Vec<u8> {
    let mut w = Writer::default();
    w.u32(cache.num_frames as u32);
    w.f32(cache.fps);
    w.u32(cache.meshes.len() as u32);
    for m in &cache.meshes {
        w.str(&m.name);
        w.u32(m.indices.len() as u32);
        for &i in &m.indices {
            w.u32(i);
        }
        w.u32(m.frames.len() as u32);
        // All frames share the vertex count (constant topology); 0 for an empty mesh.
        let vert_count = m.frames.first().map(Vec::len).unwrap_or(0);
        w.u32(vert_count as u32);
        for frame in &m.frames {
            for p in frame {
                for c in p {
                    w.f32(*c);
                }
            }
        }
    }
    w.buf
}

fn decode(r: &mut Reader) -> Result<VertexCache, EngineError> {
    let num_frames = r.u32()? as usize;
    let fps = r.f32()?;
    let mesh_count = r.u32()? as usize;
    let mut meshes = Vec::with_capacity(mesh_count);
    for _ in 0..mesh_count {
        let name = r.str()?;
        let idx_count = r.u32()? as usize;
        let mut indices = Vec::with_capacity(idx_count);
        for _ in 0..idx_count {
            indices.push(r.u32()?);
        }
        let frame_count = r.u32()? as usize;
        let vert_count = r.u32()? as usize;
        let mut frames = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            let mut verts = Vec::with_capacity(vert_count);
            for _ in 0..vert_count {
                verts.push([r.f32()?, r.f32()?, r.f32()?]);
            }
            frames.push(verts);
        }
        meshes.push(VcMesh {
            name,
            indices,
            frames,
        });
    }
    Ok(VertexCache {
        meshes,
        num_frames,
        fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VertexCache {
        VertexCache {
            meshes: vec![
                VcMesh {
                    name: "quad".into(),
                    indices: vec![0, 1, 2, 0, 2, 3],
                    frames: vec![
                        vec![
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        vec![
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                    ],
                },
                VcMesh {
                    name: "empty".into(),
                    indices: vec![],
                    frames: vec![],
                },
            ],
            num_frames: 2,
            fps: 24.0,
        }
    }

    #[test]
    fn vcache_roundtrips() {
        let cache = sample();
        let bytes = write_vcache(&cache, 0xfeed);
        let (header, decoded) = read_vcache(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0xfeed);
        assert_eq!(decoded.num_frames, 2);
        assert_eq!(decoded.fps, 24.0);
        assert_eq!(decoded.meshes.len(), 2);
        assert_eq!(decoded.meshes[0].name, "quad");
        assert_eq!(decoded.meshes[0].indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(decoded.meshes[0].frames[1][1], [2.0, 0.0, 0.0]);
        assert!(decoded.meshes[1].frames.is_empty());
    }

    #[test]
    fn cook_is_deterministic() {
        assert_eq!(write_vcache(&sample(), 7), write_vcache(&sample(), 7));
    }
}
