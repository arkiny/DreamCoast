//! Cluster page codec — the per-mesh virtual-geometry payload (Phase 14 M1a).
//!
//! One [`CHUNK_CLUSTERS`] chunk holds a mesh's [`MeshClusters`]: the shared vertex remap, the
//! packed `u8` triangle indices, and the cluster records (window offsets + culling bounds).
//! Serialized per mesh so pages can later stream independently; the whole stream stays resident
//! in M1. Uses the same explicit little-endian [`Writer`]/[`Reader`] as the other chunks, so the
//! cook is deterministic and byte-identical across platforms.

use dreamcoast_core::EngineError;

use super::{
    CHUNK_CLUSTERS, Header, Reader, Writer, open_chunk, read_directory, write_single_chunk,
};
use crate::MeshVertex;
use crate::vgeo::{Cluster, MeshClusters};

/// Encode a [`MeshClusters`] into a raw chunk payload (no container framing) — the bytes of a
/// `CHUNK_CLUSTERS` chunk. Shared by the standalone [`write_clusters`] and the combined
/// mesh+clusters cook ([`super::mesh::write_with_clusters`]).
pub(crate) fn encode_clusters(mc: &MeshClusters) -> Vec<u8> {
    let mut w = Writer::default();

    // Source vertex pool (position/normal/uv), so the page is self-contained.
    w.u32(mc.vertices.len() as u32);
    for v in &mc.vertices {
        for f in v.pos {
            w.f32(f);
        }
        for f in v.normal {
            w.f32(f);
        }
        for f in v.uv {
            w.f32(f);
        }
    }

    // Shared vertex remap.
    w.u32(mc.cluster_vertices.len() as u32);
    for &v in &mc.cluster_vertices {
        w.u32(v);
    }
    // Packed u8 triangle indices (length-prefixed; already 3-per-triangle).
    w.u32(mc.cluster_triangles.len() as u32);
    w.bytes(&mc.cluster_triangles);

    // Cluster records.
    w.u32(mc.clusters.len() as u32);
    for c in &mc.clusters {
        w.u32(c.vertex_offset);
        w.u32(c.vertex_count);
        w.u32(c.triangle_offset);
        w.u32(c.triangle_count);
        for f in c.bounds_center {
            w.f32(f);
        }
        w.f32(c.bounds_radius);
        for f in c.cone_axis {
            w.f32(f);
        }
        w.f32(c.cone_cutoff);
        w.u32(c.material);
        // LOD DAG (M1d).
        w.u32(c.lod_level);
        w.u32(c.group);
        w.f32(c.self_error);
        for f in c.self_center {
            w.f32(f);
        }
        w.f32(c.self_radius);
        w.f32(c.parent_error);
        for f in c.parent_center {
            w.f32(f);
        }
        w.f32(c.parent_radius);
    }

    w.buf
}

/// Serialize `mc` into a standalone single-chunk `.dcasset` buffer. `src_hash` is the source
/// mesh's [`super::source_hash`], embedded for cache invalidation.
pub fn write_clusters(mc: &MeshClusters, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_CLUSTERS, &encode_clusters(mc), src_hash)
}

/// Decode a `CHUNK_CLUSTERS` payload (positioned reader) into a [`MeshClusters`]. Shared by the
/// standalone and combined readers.
pub(crate) fn decode_clusters(r: &mut Reader) -> Result<MeshClusters, EngineError> {
    let pool = r.u32()? as usize;
    let mut vertices = Vec::with_capacity(pool);
    for _ in 0..pool {
        vertices.push(MeshVertex {
            pos: [r.f32()?, r.f32()?, r.f32()?],
            normal: [r.f32()?, r.f32()?, r.f32()?],
            uv: [r.f32()?, r.f32()?],
        });
    }

    let vcount = r.u32()? as usize;
    let mut cluster_vertices = Vec::with_capacity(vcount);
    for _ in 0..vcount {
        cluster_vertices.push(r.u32()?);
    }

    let tcount = r.u32()? as usize;
    let cluster_triangles = r.take(tcount)?.to_vec();

    let ccount = r.u32()? as usize;
    let mut clusters = Vec::with_capacity(ccount);
    for _ in 0..ccount {
        clusters.push(Cluster {
            vertex_offset: r.u32()?,
            vertex_count: r.u32()?,
            triangle_offset: r.u32()?,
            triangle_count: r.u32()?,
            bounds_center: [r.f32()?, r.f32()?, r.f32()?],
            bounds_radius: r.f32()?,
            cone_axis: [r.f32()?, r.f32()?, r.f32()?],
            cone_cutoff: r.f32()?,
            material: r.u32()?,
            lod_level: r.u32()?,
            group: r.u32()?,
            self_error: r.f32()?,
            self_center: [r.f32()?, r.f32()?, r.f32()?],
            self_radius: r.f32()?,
            parent_error: r.f32()?,
            parent_center: [r.f32()?, r.f32()?, r.f32()?],
            parent_radius: r.f32()?,
        });
    }

    Ok(MeshClusters {
        vertices,
        cluster_vertices,
        cluster_triangles,
        clusters,
    })
}

/// Decode a standalone cluster-page `.dcasset` buffer into its [`Header`] and [`MeshClusters`].
/// Errors on bad magic, truncation, or a missing cluster chunk.
pub fn read_clusters(bytes: &[u8]) -> Result<(Header, MeshClusters), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_CLUSTERS, "clusters")?;
    Ok((header, decode_clusters(&mut r)?))
}

/// Find a `CHUNK_CLUSTERS` chunk in ANY `.dcasset` container (e.g. a combined mesh+clusters cook)
/// and decode it, or `None` if the file has no cluster chunk (a fallback-only asset). The mesh
/// readers already skip this chunk, so a combined file loads its mesh regardless.
pub fn read_clusters_opt(bytes: &[u8]) -> Result<Option<MeshClusters>, EngineError> {
    for (ty, offset, size) in read_directory(bytes)? {
        if ty != CHUNK_CLUSTERS {
            continue;
        }
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| EngineError::Asset("dcasset: clusters chunk out of bounds".into()))?;
        let mut r = Reader::at(&bytes[..end], offset);
        return Ok(Some(decode_clusters(&mut r)?));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshVertex;
    use crate::vgeo::build_lod_dag;

    fn grid(n: u32) -> (Vec<MeshVertex>, Vec<u32>) {
        let mut verts = Vec::new();
        for y in 0..n {
            for x in 0..n {
                verts.push(MeshVertex {
                    pos: [x as f32, y as f32, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        let mut idx = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let a = y * n + x;
                idx.extend_from_slice(&[a, a + n, a + 1, a + 1, a + n, a + n + 1]);
            }
        }
        (verts, idx)
    }

    #[test]
    fn clusters_round_trip_byte_stable() {
        let (verts, idx) = grid(32);
        // Full LOD DAG so the round-trip exercises the lod/error/sphere fields too.
        let mc = build_lod_dag(&verts, &idx, 3);
        assert!(mc.clusters.len() > 1);
        assert!(
            mc.clusters.iter().any(|c| c.lod_level > 0),
            "multi-LOD expected"
        );

        let bytes = write_clusters(&mc, 0xDEAD_BEEF);
        let (header, back) = read_clusters(&bytes).expect("read clusters");
        assert_eq!(header.source_hash, 0xDEAD_BEEF);
        // The page is self-contained: the source vertex pool round-trips too.
        assert_eq!(back.vertices.len(), verts.len());
        assert_eq!(mc.vertices, back.vertices);
        // Full struct equality: the decode reconstructs the builder output exactly.
        assert_eq!(mc, back);
        // Re-encoding the decoded value is byte-identical (deterministic cook).
        assert_eq!(bytes, write_clusters(&back, 0xDEAD_BEEF));
    }

    #[test]
    fn combined_mesh_and_clusters_round_trip() {
        use crate::{Material, MeshData};
        let (verts, idx) = grid(24);
        let clusters = build_lod_dag(&verts, &idx, 0);
        let mesh = MeshData {
            vertices: verts.clone(),
            indices: idx.clone(),
            material: Material::default(),
        };

        // The cooked form: fallback mesh + cluster pages in one container.
        let bytes = super::super::mesh::write_with_clusters(&mesh, &clusters, 42);

        // The fallback mesh reads back through the plain reader (cluster chunk ignored).
        let (_, back_mesh) = super::super::mesh::read(&bytes).expect("read mesh");
        assert_eq!(back_mesh.vertices.len(), verts.len());
        assert_eq!(back_mesh.indices, idx);

        // The cluster pages read back and match the builder output exactly.
        let back = read_clusters_opt(&bytes)
            .expect("read")
            .expect("clusters present");
        assert_eq!(back, clusters);

        // A mesh-only cook (no clusters) reports no cluster chunk.
        let mesh_only = super::super::mesh::write(&mesh, 42);
        assert!(read_clusters_opt(&mesh_only).expect("read").is_none());
    }
}
