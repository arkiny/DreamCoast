//! Binary glTF (`.glb`) **writer** — the export half of the importer in
//! [`crate::gltf_scene`].
//!
//! Its reason to exist is runtime-generated geometry (game-framework M1: a dungeon
//! meshed from a tile grid). The engine already has a complete, well-tested road from
//! "a file on disk" to "fully static scene geometry": cook → content-hash-keyed
//! `.dcasset` → level instantiation → per-mesh SDF bake → GDF / surface cache / GI /
//! reflections / TLAS. Nothing on that road is glTF-specific *except its entrance*, so
//! the cheapest way to give generated geometry the full static treatment is to give it
//! a file — not a second, parallel injection path that would have to be kept in step
//! with the first forever.
//!
//! Writing the generator's output as a `.glb` therefore buys, for free:
//!
//! * **cook caching keyed by generator output** — the cook key is a hash of the source
//!   bytes, and the bytes are a pure function of the generator (seed, version, params),
//!   so re-running an unchanged generator is a cache hit and a changed one re-cooks. No
//!   hand-rolled "seed + generator version" key to get wrong.
//! * every downstream bake, because the geometry arrives through the same door as an
//!   authored asset.
//!
//! **Determinism is a contract here**, not a nicety: the output must be a pure function
//! of the input, or the cook cache thrashes. So the encoder writes fixed-order JSON with
//! no timestamps, no HashMap iteration, and explicit little-endian scalars.
//!
//! Scope: static geometry with per-primitive scalar PBR materials — positions, normals,
//! UVs, `u32` indices, one primitive per mesh. No textures, skins, animations, or morph
//! targets (a generator that needs them should grow this module, not fork it).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;

use dreamcoast_core::EngineError;
use gltf::json;
use json::validation::Checked::Valid;
use json::validation::USize64;

use crate::MeshVertex;

/// Bytes per vertex in the written buffer — position, normal, UV as little-endian
/// `f32`s. Matches [`MeshVertex`] (and so `rhi::VertexLayout::Mesh`), but the encoder
/// writes the fields explicitly rather than casting the struct, because glTF fixes the
/// byte order and a `#[repr(C)]` cast would silently depend on the host's.
const VERTEX_STRIDE: usize = 32;

/// A scalar metallic-roughness material (no textures).
#[derive(Clone, Debug, PartialEq)]
pub struct GlbMaterial {
    /// Material name, written to the glTF (the importer's `classify_material` reads it,
    /// so it is also the routing tag — keep generated names free of the decal/foliage
    /// keywords unless that routing is intended).
    pub name: String,
    /// Linear base colour + alpha.
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// Disables back-face culling. Generated shells that are only ever seen from one
    /// side should leave this `false` — a two-sided wall is a two-sided SDF sign too.
    pub double_sided: bool,
}

impl Default for GlbMaterial {
    fn default() -> Self {
        Self {
            name: "material".into(),
            base_color_factor: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            double_sided: false,
        }
    }
}

/// One generated mesh: a named node holding a single triangle-list primitive.
///
/// One of these per merged chunk is the intended granularity — the engine bakes a
/// per-mesh SDF per *unique mesh* and draws one call per instance, so chunk-sized
/// meshes are what keeps both counts sane.
#[derive(Clone, Debug, PartialEq)]
pub struct GlbMesh {
    /// Node + mesh name (e.g. `chunk_0_3`), preserved through the importer.
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    /// Triangle-list indices into `vertices`.
    pub indices: Vec<u32>,
    /// Index into the material list passed to [`write_glb`].
    pub material: usize,
}

/// Encode meshes + materials as a self-contained binary glTF.
///
/// The result is a pure function of the arguments (see the module note on determinism).
/// Errors on an out-of-range material index, a non-triangle index count, an index that
/// does not address a vertex, or a buffer past the `u32` GLB size limit — all of which
/// are generator bugs that would otherwise surface much later as a corrupt bake.
pub fn write_glb(meshes: &[GlbMesh], materials: &[GlbMaterial]) -> Result<Vec<u8>, EngineError> {
    let mut root = json::Root {
        asset: json::Asset {
            // Fixed string: no version/timestamp, so the bytes stay content-addressable.
            generator: Some("dreamcoast-asset".into()),
            ..Default::default()
        },
        ..Default::default()
    };

    let material_indices: Vec<json::Index<json::Material>> = materials
        .iter()
        .map(|m| {
            root.push(json::Material {
                name: Some(m.name.clone()),
                double_sided: m.double_sided,
                alpha_mode: Valid(json::material::AlphaMode::Opaque),
                pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                    base_color_factor: json::material::PbrBaseColorFactor(m.base_color_factor),
                    metallic_factor: json::material::StrengthFactor(m.metallic),
                    roughness_factor: json::material::StrengthFactor(m.roughness),
                    ..Default::default()
                },
                ..Default::default()
            })
        })
        .collect();

    // One BIN chunk holds every mesh's vertex block followed by its index block. Both
    // are 4-byte-aligned by construction (32-byte vertices, 4-byte indices), which is
    // what glTF requires of an accessor offset.
    let mut bin: Vec<u8> = Vec::new();
    let mut nodes: Vec<json::Index<json::Node>> = Vec::with_capacity(meshes.len());

    for mesh in meshes {
        let material = *material_indices.get(mesh.material).ok_or_else(|| {
            EngineError::Asset(format!(
                "glb: mesh '{}' references material {} of {}",
                mesh.name,
                mesh.material,
                materials.len()
            ))
        })?;
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Err(EngineError::Asset(format!(
                "glb: mesh '{}' is empty",
                mesh.name
            )));
        }
        if !mesh.indices.len().is_multiple_of(3) {
            return Err(EngineError::Asset(format!(
                "glb: mesh '{}' has {} indices — not whole triangles",
                mesh.name,
                mesh.indices.len()
            )));
        }
        let vertex_count = mesh.vertices.len();
        if let Some(&bad) = mesh.indices.iter().find(|&&i| i as usize >= vertex_count) {
            return Err(EngineError::Asset(format!(
                "glb: mesh '{}' index {bad} is past its {vertex_count} vertices",
                mesh.name
            )));
        }

        let (min, max) = position_bounds(&mesh.vertices);

        let vertex_offset = bin.len();
        for v in &mesh.vertices {
            push_f32s(&mut bin, &v.pos);
            push_f32s(&mut bin, &v.normal);
            push_f32s(&mut bin, &v.uv);
        }
        let vertex_bytes = bin.len() - vertex_offset;

        let index_offset = bin.len();
        for i in &mesh.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let index_bytes = bin.len() - index_offset;

        // Buffer index 0 — pushed once after the loop, but its index is known now.
        let buffer = json::Index::new(0);
        let vertex_view = root.push(json::buffer::View {
            buffer,
            byte_length: USize64::from(vertex_bytes),
            byte_offset: Some(USize64::from(vertex_offset)),
            byte_stride: Some(json::buffer::Stride(VERTEX_STRIDE)),
            name: Some(format!("{}_vertices", mesh.name)),
            target: Some(Valid(json::buffer::Target::ArrayBuffer)),
            extensions: Default::default(),
            extras: Default::default(),
        });
        let index_view = root.push(json::buffer::View {
            buffer,
            byte_length: USize64::from(index_bytes),
            byte_offset: Some(USize64::from(index_offset)),
            // Index views carry no stride (tightly packed scalars).
            byte_stride: None,
            name: Some(format!("{}_indices", mesh.name)),
            target: Some(Valid(json::buffer::Target::ElementArrayBuffer)),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let vertex_accessor = |root: &mut json::Root,
                               offset: usize,
                               ty: json::accessor::Type,
                               min: Option<Vec<f32>>,
                               max: Option<Vec<f32>>| {
            root.push(json::Accessor {
                buffer_view: Some(vertex_view),
                byte_offset: Some(USize64::from(offset)),
                count: USize64::from(vertex_count),
                component_type: Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                type_: Valid(ty),
                min: min.map(json::Value::from),
                max: max.map(json::Value::from),
                normalized: false,
                sparse: None,
                name: None,
                extensions: Default::default(),
                extras: Default::default(),
            })
        };
        // POSITION carries the required min/max; NORMAL and TEXCOORD_0 do not.
        let positions = vertex_accessor(
            &mut root,
            0,
            json::accessor::Type::Vec3,
            Some(min.to_vec()),
            Some(max.to_vec()),
        );
        let normals = vertex_accessor(&mut root, 12, json::accessor::Type::Vec3, None, None);
        let uvs = vertex_accessor(&mut root, 24, json::accessor::Type::Vec2, None, None);
        let indices = root.push(json::Accessor {
            buffer_view: Some(index_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(mesh.indices.len()),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::U32,
            )),
            type_: Valid(json::accessor::Type::Scalar),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
            name: None,
            extensions: Default::default(),
            extras: Default::default(),
        });

        let mut attributes = BTreeMap::new();
        attributes.insert(Valid(json::mesh::Semantic::Positions), positions);
        attributes.insert(Valid(json::mesh::Semantic::Normals), normals);
        attributes.insert(Valid(json::mesh::Semantic::TexCoords(0)), uvs);
        let primitive = json::mesh::Primitive {
            attributes,
            indices: Some(indices),
            material: Some(material),
            mode: Valid(json::mesh::Mode::Triangles),
            targets: None,
            extensions: Default::default(),
            extras: Default::default(),
        };
        let gltf_mesh = root.push(json::Mesh {
            name: Some(mesh.name.clone()),
            primitives: vec![primitive],
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
        });
        nodes.push(root.push(json::Node {
            name: Some(mesh.name.clone()),
            mesh: Some(gltf_mesh),
            ..Default::default()
        }));
    }

    // GLB pads the BIN chunk to 4 bytes; the buffer's declared length must match the
    // padded chunk the container writes, or strict readers reject the file.
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    root.push(json::Buffer {
        byte_length: USize64::from(bin.len()),
        // No URI: the data is the GLB's own BIN chunk.
        uri: None,
        name: None,
        extensions: Default::default(),
        extras: Default::default(),
    });
    root.push(json::Scene {
        name: Some("scene".into()),
        nodes,
        extensions: Default::default(),
        extras: Default::default(),
    });
    root.scene = Some(json::Index::new(0));

    let json_string = json::serialize::to_string(&root)
        .map_err(|e| EngineError::Asset(format!("glb: serialize json: {e}")))?;
    // The JSON chunk is padded to 4 bytes too (with spaces, per the spec).
    let mut json_bytes = json_string.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    // 12-byte header + two 8-byte chunk headers + both payloads.
    let length = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let length = u32::try_from(length)
        .map_err(|_| EngineError::Asset("glb: file exceeds the 4 GiB container limit".into()))?;
    let glb = gltf::binary::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length,
        },
        json: Cow::Owned(json_bytes),
        bin: Some(Cow::Owned(bin)),
    };
    glb.to_vec()
        .map_err(|e| EngineError::Asset(format!("glb: encode: {e}")))
}

/// [`write_glb`] straight to a file, creating the parent directory.
///
/// **Only rewrites the file when the bytes differ.** An unchanged generator must not
/// touch the file's mtime or contents — a rewrite with identical bytes is harmless to
/// the cook (which keys on content, not mtime), but leaving it alone keeps the whole
/// pipeline honestly no-op on a re-run.
pub fn save_glb(
    path: impl AsRef<Path>,
    meshes: &[GlbMesh],
    materials: &[GlbMaterial],
) -> Result<(), EngineError> {
    let path = path.as_ref();
    let bytes = write_glb(meshes, materials)?;
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| EngineError::Asset(format!("glb: create {}: {e}", dir.display())))?;
    }
    std::fs::write(path, &bytes)
        .map_err(|e| EngineError::Asset(format!("glb: write {}: {e}", path.display())))
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// The component-wise min/max of a vertex list's positions (glTF requires them on the
/// POSITION accessor).
fn position_bounds(vertices: &[MeshVertex]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for a in 0..3 {
            min[a] = min[a].min(v.pos[a]);
            max[a] = max[a].max(v.pos[a]);
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A self-cleaning scratch directory (the writer's tests need real files, because
    /// the round trip goes through `gltf::import`, which takes a path).
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let id = format!("{:?}", std::thread::current().id());
            let id: String = id.chars().filter(|c| c.is_alphanumeric()).collect();
            let dir = std::env::temp_dir().join(format!("dcasset-glb-{tag}-{id}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Two triangles forming a quad on y = 0, with distinct normals/UVs so the reader
    /// has something to get wrong.
    fn quad() -> GlbMesh {
        let v = |x: f32, z: f32, u: f32, w: f32| MeshVertex {
            pos: [x, 0.0, z],
            normal: [0.0, 1.0, 0.0],
            uv: [u, w],
        };
        GlbMesh {
            name: "quad".into(),
            vertices: vec![
                v(-1.0, -1.0, 0.0, 0.0),
                v(1.0, -1.0, 1.0, 0.0),
                v(1.0, 1.0, 1.0, 1.0),
                v(-1.0, 1.0, 0.0, 1.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: 0,
        }
    }

    fn stone() -> GlbMaterial {
        GlbMaterial {
            name: "stone".into(),
            base_color_factor: [0.4, 0.42, 0.45, 1.0],
            metallic: 0.0,
            roughness: 0.85,
            double_sided: false,
        }
    }

    /// The round trip that actually matters: what this writes, the engine's own
    /// importer must read back with the geometry and material intact.
    #[test]
    fn round_trips_through_the_engine_importer() {
        let mesh = quad();
        let bytes = write_glb(std::slice::from_ref(&mesh), &[stone()]).expect("write");
        let dir = TempDir::new("glb");
        let path = dir.0.join("quad.glb");
        std::fs::write(&path, &bytes).unwrap();

        let scene = crate::load_gltf_scene(&path).expect("import");
        assert_eq!(scene.meshes.len(), 1);
        let prim = &scene.meshes[0][0];
        assert_eq!(prim.indices, mesh.indices);
        assert_eq!(prim.vertices, mesh.vertices);
        let material = &scene.materials[prim.material.expect("material index")];
        assert_eq!(material.base_color_factor, [0.4, 0.42, 0.45, 1.0]);
        assert_eq!(material.roughness_factor, 0.85);
        // The node keeps its name, which is how a generated chunk stays identifiable.
        assert_eq!(scene.nodes[0].name.as_deref(), Some("quad"));
    }

    /// Byte-for-byte determinism — the property the cook cache key rests on.
    #[test]
    fn output_is_deterministic() {
        let a = write_glb(&[quad()], &[stone()]).unwrap();
        let b = write_glb(&[quad()], &[stone()]).unwrap();
        assert_eq!(a, b);
        // ...and content-sensitive: a moved vertex must produce different bytes.
        let mut moved = quad();
        moved.vertices[0].pos[1] = 0.5;
        assert_ne!(write_glb(&[moved], &[stone()]).unwrap(), a);
    }

    /// Several meshes share one buffer; each must still address its own slice.
    #[test]
    fn multiple_meshes_keep_separate_geometry() {
        let mut second = quad();
        second.name = "chunk_1".into();
        second.vertices.truncate(3);
        second.indices = vec![0, 1, 2];
        second.material = 1;
        let bytes = write_glb(
            &[quad(), second.clone()],
            &[
                stone(),
                GlbMaterial {
                    name: "wood".into(),
                    ..GlbMaterial::default()
                },
            ],
        )
        .expect("write");
        let dir = TempDir::new("glb-multi");
        let path = dir.0.join("two.glb");
        std::fs::write(&path, &bytes).unwrap();

        let scene = crate::load_gltf_scene(&path).expect("import");
        assert_eq!(scene.meshes.len(), 2);
        assert_eq!(scene.meshes[0][0].vertices.len(), 4);
        assert_eq!(scene.meshes[1][0].vertices, second.vertices);
        assert_eq!(
            scene.materials[scene.meshes[1][0].material.unwrap()].base_color_factor[0],
            0.8
        );
    }

    /// Generator bugs are rejected at the writer, not at the bake.
    #[test]
    fn rejects_malformed_input() {
        let mut bad_material = quad();
        bad_material.material = 7;
        assert!(write_glb(&[bad_material], &[stone()]).is_err());

        let mut bad_indices = quad();
        bad_indices.indices = vec![0, 1];
        assert!(write_glb(&[bad_indices], &[stone()]).is_err());

        let mut out_of_range = quad();
        out_of_range.indices = vec![0, 1, 9];
        assert!(write_glb(&[out_of_range], &[stone()]).is_err());
    }

    /// `save_glb` leaves an already-correct file untouched (the no-op re-run contract).
    #[test]
    fn save_is_a_no_op_when_the_bytes_match() {
        let dir = TempDir::new("glb-save");
        let path = dir.0.join("room.glb");
        save_glb(&path, &[quad()], &[stone()]).unwrap();
        let first = std::fs::metadata(&path).unwrap().modified().unwrap();
        save_glb(&path, &[quad()], &[stone()]).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), first);
    }
}
