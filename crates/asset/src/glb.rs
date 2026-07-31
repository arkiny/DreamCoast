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
//! Scope: geometry with per-primitive scalar PBR materials — positions, normals, UVs,
//! `u32` indices, one primitive per mesh — placed on an arbitrary **node hierarchy**
//! ([`GlbNode`]) and driven by node-TRS **animations** ([`GlbAnimation`]). No textures,
//! skins, or morph targets (a generator that needs them should grow this module, not
//! fork it).
//!
//! Two entry points, one encoder:
//!
//! * [`write_glb`] / [`save_glb`] — the flat form: one node per mesh at identity. This
//!   is what the M1 dungeon writer uses, and its **bytes are pinned by a test**, because
//!   every generated dungeon in the cook cache is keyed on them.
//! * [`write_glb_scene`] / [`save_glb_scene`] — the general form: an explicit node tree
//!   plus animation clips. Rigid per-node "bones" animated by TRS channels is the M2
//!   character rig (no skinning), which rides [`crate::gltf_scene`] → the scene crate's
//!   `instantiate_gltf_mapped` → `AnimationClip::from_gltf` exactly like an authored
//!   glTF, for the same reason the geometry does: one door, not two.
//!
//! The flat form is the general one with generated nodes, so there is a single encoder
//! to keep deterministic.

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

/// glTF's default node rotation — the identity quaternion in `[x, y, z, w]` order.
pub const IDENTITY_ROTATION: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
/// glTF's default node scale.
pub const UNIT_SCALE: [f32; 3] = [1.0, 1.0, 1.0];

/// One node of the written hierarchy: a named TRS frame that may carry a mesh and may
/// hang off another node.
///
/// `parent` is an index into the same node slice — the tree is expressed by parent
/// links rather than child lists so a rig can be authored top-down as a flat array (a
/// node's index is its stable id, which is also what an animation channel targets and
/// what the importer hands the scene crate's node → entity map). Children are derived
/// in node order, so the written child lists are a pure function of the input.
///
/// The transform is glTF-native TRS: `rotation` is the `[x, y, z, w]` quaternion (must
/// be unit length), and the composition is `T * R * S`, applied about the node's own
/// origin. For a rigid "bone", that origin is the joint — author the node at the joint
/// and its mesh in joint-local space, and rotating the node swings the limb.
#[derive(Clone, Debug, PartialEq)]
pub struct GlbNode {
    /// Node name, preserved through the importer (and into the ECS `Name` component),
    /// so gameplay can find a bone — e.g. the hand a weapon is parented to.
    pub name: String,
    /// Index of this node's parent in the same slice; `None` makes it a scene root.
    pub parent: Option<usize>,
    pub translation: [f32; 3],
    /// `[x, y, z, w]`, unit length.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    /// Index into the mesh list passed to [`write_glb_scene`]; `None` for a pure
    /// transform node (a rig root, or a joint whose geometry lives on its children).
    pub mesh: Option<usize>,
}

impl Default for GlbNode {
    fn default() -> Self {
        Self {
            name: "node".into(),
            parent: None,
            translation: [0.0; 3],
            rotation: IDENTITY_ROTATION,
            scale: UNIT_SCALE,
            mesh: None,
        }
    }
}

impl GlbNode {
    /// A node at `translation` under `parent`, identity rotation/scale, no mesh —
    /// the common case when laying out a skeleton (set `.mesh` with struct update).
    pub fn new(name: impl Into<String>, parent: Option<usize>, translation: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            parent,
            translation,
            ..Self::default()
        }
    }

    /// Builder form of [`GlbNode::mesh`].
    #[must_use]
    pub fn with_mesh(mut self, mesh: usize) -> Self {
        self.mesh = Some(mesh);
        self
    }
}

/// Which TRS property of a node a channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GlbPath {
    Translation,
    Rotation,
    Scale,
}

/// How a channel's samples are interpolated between keyframes.
///
/// Only glTF's two *keyframe-authored* modes: a generator writes poses, and both of
/// these reconstruct exactly what it wrote. `CUBICSPLINE` is deliberately absent — it
/// needs a tangent per key (`3 ×` the outputs), which is a curve-baking format, not an
/// authoring one. Adding it later is a variant here plus the `3 ×` output layout in
/// [`write_animation`]; the importer and the scene crate's sampler already handle it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlbInterpolation {
    /// Hold each key's value until the next.
    Step,
    /// Linear between adjacent keys (spherical for rotations).
    Linear,
}

/// One keyframe's value. Must match its channel's [`GlbPath`]: `Vec3` for
/// translation/scale, `Quat` for rotation (checked by the writer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlbValue {
    Vec3([f32; 3]),
    /// `[x, y, z, w]`, unit length.
    Quat([f32; 4]),
}

impl GlbValue {
    /// The value's components, in write order.
    fn components(&self) -> &[f32] {
        match self {
            Self::Vec3(v) => v,
            Self::Quat(q) => q,
        }
    }
}

/// One animation channel: sparse keyframes for a single TRS property of a single node.
///
/// Keys are `(time in seconds, value)` and must be **strictly increasing in time**
/// (glTF requires it, and a repeated time is a zero-length segment no sampler can
/// resolve). Author poses, not baked curves — the sampler interpolates.
#[derive(Clone, Debug, PartialEq)]
pub struct GlbChannel {
    /// Index into the node slice passed to [`write_glb_scene`].
    pub node: usize,
    pub path: GlbPath,
    pub interpolation: GlbInterpolation,
    pub keys: Vec<(f32, GlbValue)>,
}

impl GlbChannel {
    /// A linear translation channel from `(time, xyz)` keys.
    pub fn translation(node: usize, keys: impl IntoIterator<Item = (f32, [f32; 3])>) -> Self {
        Self::linear(node, GlbPath::Translation, keys, GlbValue::Vec3)
    }

    /// A linear rotation channel from `(time, [x, y, z, w])` keys.
    pub fn rotation(node: usize, keys: impl IntoIterator<Item = (f32, [f32; 4])>) -> Self {
        Self::linear(node, GlbPath::Rotation, keys, GlbValue::Quat)
    }

    /// A linear scale channel from `(time, xyz)` keys.
    pub fn scale(node: usize, keys: impl IntoIterator<Item = (f32, [f32; 3])>) -> Self {
        Self::linear(node, GlbPath::Scale, keys, GlbValue::Vec3)
    }

    /// Switch this channel to [`GlbInterpolation::Step`] (hold each key).
    #[must_use]
    pub fn stepped(mut self) -> Self {
        self.interpolation = GlbInterpolation::Step;
        self
    }

    fn linear<T>(
        node: usize,
        path: GlbPath,
        keys: impl IntoIterator<Item = (f32, T)>,
        wrap: impl Fn(T) -> GlbValue,
    ) -> Self {
        Self {
            node,
            path,
            interpolation: GlbInterpolation::Linear,
            keys: keys.into_iter().map(|(t, v)| (t, wrap(v))).collect(),
        }
    }
}

/// One named animation clip: a set of node-TRS channels.
///
/// The clip's duration is implicit — the importer derives it as the largest keyframe
/// time across channels, so a clip that must end at a particular time needs a key
/// there (which is also what makes a loop close on itself).
#[derive(Clone, Debug, PartialEq)]
pub struct GlbAnimation {
    /// Clip name, preserved through the importer — how gameplay selects a clip.
    pub name: String,
    pub channels: Vec<GlbChannel>,
}

/// Encode meshes + materials as a self-contained binary glTF: one node per mesh, at
/// identity, all of them scene roots.
///
/// The flat form of [`write_glb_scene`] — and byte-identical to what it wrote before
/// nodes and animations existed, which is the contract every cook-cached generated
/// asset rests on (`flat_output_bytes_are_pinned`).
pub fn write_glb(meshes: &[GlbMesh], materials: &[GlbMaterial]) -> Result<Vec<u8>, EngineError> {
    let nodes: Vec<GlbNode> = meshes
        .iter()
        .enumerate()
        .map(|(i, m)| GlbNode {
            name: m.name.clone(),
            mesh: Some(i),
            ..GlbNode::default()
        })
        .collect();
    write_glb_scene(&nodes, meshes, materials, &[])
}

/// Encode a node hierarchy + meshes + materials + animations as a self-contained
/// binary glTF.
///
/// The result is a pure function of the arguments (see the module note on
/// determinism). Errors on any input a generator could only produce by being wrong —
/// an out-of-range material/mesh/node index, a node cycle, a non-unit rotation, a
/// non-triangle index count, an index that does not address a vertex, keyframe times
/// that do not increase, a value whose type disagrees with its channel's path, or a
/// buffer past the `u32` GLB size limit. All of those would otherwise surface much
/// later as a corrupt bake or a silently dead animation.
pub fn write_glb_scene(
    nodes: &[GlbNode],
    meshes: &[GlbMesh],
    materials: &[GlbMaterial],
    animations: &[GlbAnimation],
) -> Result<Vec<u8>, EngineError> {
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

    // The node tree is validated up front (and its child lists derived) so a bad rig is
    // rejected before a single byte of geometry is encoded.
    let children = node_children(nodes, meshes.len())?;

    // One BIN chunk holds every mesh's vertex block followed by its index block, then
    // the animation keyframes. All are 4-byte-aligned by construction (32-byte vertices,
    // 4-byte indices, `f32` keys), which is what glTF requires of an accessor offset.
    let mut bin: Vec<u8> = Vec::new();
    let mut mesh_indices: Vec<json::Index<json::Mesh>> = Vec::with_capacity(meshes.len());

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
        mesh_indices.push(root.push(json::Mesh {
            name: Some(mesh.name.clone()),
            primitives: vec![primitive],
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
        }));
    }

    // Nodes in input order, so a node's glTF index *is* its index here — which is what
    // the animation channels below and the importer's node → entity map both address.
    // A TRS component equal to glTF's default is omitted rather than written: the
    // importer decomposes an absent component to exactly that default, so this is the
    // same transform in fewer bytes (and it is what keeps the flat form's bytes pinned).
    for (i, node) in nodes.iter().enumerate() {
        root.push(json::Node {
            name: Some(node.name.clone()),
            mesh: node.mesh.map(|m| mesh_indices[m]),
            children: (!children[i].is_empty()).then(|| children[i].clone()),
            translation: (node.translation != [0.0; 3]).then_some(node.translation),
            rotation: (node.rotation != IDENTITY_ROTATION)
                .then_some(json::scene::UnitQuaternion(node.rotation)),
            scale: (node.scale != UNIT_SCALE).then_some(node.scale),
            ..Default::default()
        });
    }

    for animation in animations {
        write_animation(&mut root, &mut bin, animation, nodes.len())?;
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
        // Scene roots: the parentless nodes, in node order.
        nodes: (0..nodes.len())
            .filter(|&i| nodes[i].parent.is_none())
            .map(|i| json::Index::new(i as u32))
            .collect(),
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

/// [`write_glb`] straight to a file, creating the parent directory. Reports whether it
/// wrote.
///
/// **Only rewrites the file when the bytes differ.** An unchanged generator must not
/// touch the file's mtime or contents — a rewrite with identical bytes is harmless to
/// the cook (which keys on content, not mtime), but leaving it alone keeps the whole
/// pipeline honestly no-op on a re-run.
pub fn save_glb(
    path: impl AsRef<Path>,
    meshes: &[GlbMesh],
    materials: &[GlbMaterial],
) -> Result<bool, EngineError> {
    write_if_changed(path.as_ref(), &write_glb(meshes, materials)?)
}

/// [`write_glb_scene`] straight to a file, with the same write-if-different contract as
/// [`save_glb`]. Reports whether it wrote.
pub fn save_glb_scene(
    path: impl AsRef<Path>,
    nodes: &[GlbNode],
    meshes: &[GlbMesh],
    materials: &[GlbMaterial],
    animations: &[GlbAnimation],
) -> Result<bool, EngineError> {
    write_if_changed(
        path.as_ref(),
        &write_glb_scene(nodes, meshes, materials, animations)?,
    )
}

/// Write `bytes` to `path` (creating the parent directory) unless an identical file is
/// already there; reports whether it wrote.
fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<bool, EngineError> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| EngineError::Asset(format!("glb: create {}: {e}", dir.display())))?;
    }
    std::fs::write(path, bytes)
        .map_err(|e| EngineError::Asset(format!("glb: write {}: {e}", path.display())))?;
    Ok(true)
}

/// Validate the node tree and derive each node's child list (in node order).
///
/// The checks are the ones a parent-link tree can get wrong and a reader cannot
/// recover from: an index that addresses nothing, a node that is its own ancestor
/// (which would make the derived child lists describe a graph, not a tree, and hang a
/// depth-first consumer), a mesh index past the mesh list, and a rotation that is not a
/// unit quaternion (glTF requires it; a scaled one silently scales the whole subtree).
fn node_children(
    nodes: &[GlbNode],
    mesh_count: usize,
) -> Result<Vec<Vec<json::Index<json::Node>>>, EngineError> {
    u32::try_from(nodes.len())
        .map_err(|_| EngineError::Asset("glb: more nodes than glTF can index".into()))?;
    let mut children = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        if let Some(mesh) = node.mesh
            && mesh >= mesh_count
        {
            return Err(EngineError::Asset(format!(
                "glb: node '{}' references mesh {mesh} of {mesh_count}",
                node.name
            )));
        }
        let norm_sq: f32 = node.rotation.iter().map(|c| c * c).sum();
        if !(0.999..=1.001).contains(&norm_sq) {
            return Err(EngineError::Asset(format!(
                "glb: node '{}' rotation {:?} is not a unit quaternion (|q|² = {norm_sq})",
                node.name, node.rotation
            )));
        }
        if let Some(parent) = node.parent {
            if parent >= nodes.len() {
                return Err(EngineError::Asset(format!(
                    "glb: node '{}' has parent {parent} of {}",
                    node.name,
                    nodes.len()
                )));
            }
            // Walk to a root; more steps than there are nodes means a cycle.
            let (mut at, mut steps) = (parent, 0usize);
            while let Some(next) = nodes[at].parent {
                steps += 1;
                if steps > nodes.len() {
                    return Err(EngineError::Asset(format!(
                        "glb: node '{}' is in a parent cycle",
                        node.name
                    )));
                }
                at = next;
            }
            children[parent].push(json::Index::new(i as u32));
        }
    }
    Ok(children)
}

/// Encode one clip: two accessors (times, values) and a sampler per channel, plus the
/// channel list that binds each sampler to a node property.
fn write_animation(
    root: &mut json::Root,
    bin: &mut Vec<u8>,
    animation: &GlbAnimation,
    node_count: usize,
) -> Result<(), EngineError> {
    let mut samplers = Vec::with_capacity(animation.channels.len());
    let mut channels = Vec::with_capacity(animation.channels.len());
    // glTF forbids two channels of one clip driving the same node property — and a
    // duplicate is an authoring bug either way (one of the two silently wins).
    let mut targets: Vec<(usize, GlbPath)> = Vec::with_capacity(animation.channels.len());

    for channel in &animation.channels {
        let what = || {
            format!(
                "glb: animation '{}' channel {}",
                animation.name, channel.node
            )
        };
        if channel.node >= node_count {
            return Err(EngineError::Asset(format!(
                "{}: node {} of {node_count}",
                what(),
                channel.node
            )));
        }
        if !targets.contains(&(channel.node, channel.path)) {
            targets.push((channel.node, channel.path));
        } else {
            return Err(EngineError::Asset(format!(
                "{}: a second {:?} channel for the same node",
                what(),
                channel.path
            )));
        }
        if channel.keys.is_empty() {
            return Err(EngineError::Asset(format!("{}: no keyframes", what())));
        }
        let component_count = match channel.path {
            GlbPath::Translation | GlbPath::Scale => 3,
            GlbPath::Rotation => 4,
        };
        for (i, (time, value)) in channel.keys.iter().enumerate() {
            if !time.is_finite() {
                return Err(EngineError::Asset(format!(
                    "{}: key {i} time {time}",
                    what()
                )));
            }
            if i > 0 && *time <= channel.keys[i - 1].0 {
                return Err(EngineError::Asset(format!(
                    "{}: key times must strictly increase ({} then {time})",
                    what(),
                    channel.keys[i - 1].0
                )));
            }
            if value.components().len() != component_count {
                return Err(EngineError::Asset(format!(
                    "{}: key {i} value {value:?} does not fit {:?}",
                    what(),
                    channel.path
                )));
            }
            if !value.components().iter().all(|c| c.is_finite()) {
                return Err(EngineError::Asset(format!(
                    "{}: key {i} value {value:?} is not finite",
                    what()
                )));
            }
        }

        let buffer = json::Index::new(0);
        let key_count = channel.keys.len();

        let time_offset = bin.len();
        for (time, _) in &channel.keys {
            bin.extend_from_slice(&time.to_le_bytes());
        }
        let time_view = root.push(json::buffer::View {
            buffer,
            byte_length: USize64::from(bin.len() - time_offset),
            byte_offset: Some(USize64::from(time_offset)),
            byte_stride: None,
            name: Some(format!("{}_{}_times", animation.name, channel.node)),
            // Animation data is neither vertex nor index data: no buffer target.
            target: None,
            extensions: Default::default(),
            extras: Default::default(),
        });
        // Sampler *input* accessors are one of the two places glTF requires min/max.
        let input = root.push(json::Accessor {
            buffer_view: Some(time_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(key_count),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            type_: Valid(json::accessor::Type::Scalar),
            min: Some(json::Value::from(vec![channel.keys[0].0])),
            max: Some(json::Value::from(vec![channel.keys[key_count - 1].0])),
            normalized: false,
            sparse: None,
            name: None,
            extensions: Default::default(),
            extras: Default::default(),
        });

        let value_offset = bin.len();
        for (_, value) in &channel.keys {
            push_f32s(bin, value.components());
        }
        let value_view = root.push(json::buffer::View {
            buffer,
            byte_length: USize64::from(bin.len() - value_offset),
            byte_offset: Some(USize64::from(value_offset)),
            byte_stride: None,
            name: Some(format!("{}_{}_values", animation.name, channel.node)),
            target: None,
            extensions: Default::default(),
            extras: Default::default(),
        });
        let output = root.push(json::Accessor {
            buffer_view: Some(value_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(key_count),
            component_type: Valid(json::accessor::GenericComponentType(
                json::accessor::ComponentType::F32,
            )),
            type_: Valid(match channel.path {
                GlbPath::Rotation => json::accessor::Type::Vec4,
                _ => json::accessor::Type::Vec3,
            }),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
            name: None,
            extensions: Default::default(),
            extras: Default::default(),
        });

        channels.push(json::animation::Channel {
            sampler: json::Index::new(samplers.len() as u32),
            target: json::animation::Target {
                node: json::Index::new(channel.node as u32),
                path: Valid(match channel.path {
                    GlbPath::Translation => json::animation::Property::Translation,
                    GlbPath::Rotation => json::animation::Property::Rotation,
                    GlbPath::Scale => json::animation::Property::Scale,
                }),
                extensions: Default::default(),
                extras: Default::default(),
            },
            extensions: Default::default(),
            extras: Default::default(),
        });
        samplers.push(json::animation::Sampler {
            input,
            interpolation: Valid(match channel.interpolation {
                GlbInterpolation::Step => json::animation::Interpolation::Step,
                GlbInterpolation::Linear => json::animation::Interpolation::Linear,
            }),
            output,
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    root.push(json::Animation {
        name: Some(animation.name.clone()),
        channels,
        samplers,
        extensions: Default::default(),
        extras: Default::default(),
    });
    Ok(())
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

    /// A three-node articulated chain: a meshless root, a hip, and a forearm rotated
    /// 45° about Z and stretched 2× in Y — every TRS component non-default, so an
    /// importer that dropped or reordered one would show.
    fn articulated() -> (Vec<GlbNode>, Vec<GlbMesh>, Vec<GlbAnimation>) {
        // 45° about +Z, as `[x, y, z, w]`.
        let bent = [0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, 0.5f32.sqrt()];
        let nodes = vec![
            GlbNode::new("root", None, [0.0, 0.0, 0.0]),
            GlbNode::new("hip", Some(0), [0.0, 0.9, 0.0]).with_mesh(0),
            GlbNode {
                rotation: bent,
                scale: [1.0, 2.0, 1.0],
                ..GlbNode::new("forearm", Some(1), [0.2, 0.3, 0.0]).with_mesh(0)
            },
        ];
        let animations = vec![GlbAnimation {
            name: "wave".into(),
            channels: vec![
                GlbChannel::rotation(
                    2,
                    [
                        (0.0, IDENTITY_ROTATION),
                        (0.5, bent),
                        (1.25, IDENTITY_ROTATION),
                    ],
                ),
                GlbChannel::translation(1, [(0.0, [0.0, 0.9, 0.0]), (0.25, [0.0, 1.0, 0.0])])
                    .stepped(),
            ],
        }];
        (nodes, vec![quad()], animations)
    }

    /// Part 1's definition of done: the hierarchy and the clips this writes must come
    /// back out of **the engine's own importer** — the one the scene crate instantiates
    /// from and `AnimationClip::from_gltf` consumes.
    #[test]
    fn scene_hierarchy_and_animation_round_trip() {
        let (nodes, meshes, animations) = articulated();
        let bytes = write_glb_scene(&nodes, &meshes, &[stone()], &animations).expect("write");
        let dir = TempDir::new("glb-scene");
        let path = dir.0.join("rig.glb");
        std::fs::write(&path, &bytes).unwrap();
        let scene = crate::load_gltf_scene(&path).expect("import");

        // Tree: one root, parent links reconstructed as child lists.
        assert_eq!(scene.roots, vec![0]);
        assert_eq!(scene.nodes.len(), 3);
        assert_eq!(scene.nodes[0].children, vec![1]);
        assert_eq!(scene.nodes[1].children, vec![2]);
        assert!(scene.nodes[2].children.is_empty());
        let names: Vec<&str> = scene
            .nodes
            .iter()
            .map(|n| n.name.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(names, ["root", "hip", "forearm"]);

        // TRS, exactly — including the defaults that were never written out.
        assert_eq!(scene.nodes[0].translation, [0.0, 0.0, 0.0]);
        assert_eq!(scene.nodes[0].rotation, IDENTITY_ROTATION);
        assert_eq!(scene.nodes[0].scale, UNIT_SCALE);
        assert_eq!(scene.nodes[0].mesh, None);
        assert_eq!(scene.nodes[1].translation, [0.0, 0.9, 0.0]);
        assert_eq!(scene.nodes[1].mesh, Some(0));
        assert_eq!(scene.nodes[2].translation, [0.2, 0.3, 0.0]);
        assert_eq!(scene.nodes[2].rotation, nodes[2].rotation);
        assert_eq!(scene.nodes[2].scale, [1.0, 2.0, 1.0]);

        // Clips: name, derived duration, and both channels with their keys intact.
        assert_eq!(scene.animations.len(), 1);
        let clip = &scene.animations[0];
        assert_eq!(clip.name.as_deref(), Some("wave"));
        assert_eq!(clip.duration, 1.25, "duration = the largest keyframe time");
        assert_eq!(clip.channels.len(), 2);

        let rot = clip
            .channels
            .iter()
            .find(|c| matches!(c.data, crate::ChannelData::Rotation(_)))
            .expect("rotation channel");
        assert_eq!(rot.target_node, 2);
        assert_eq!(rot.interpolation, crate::Interpolation::Linear);
        assert_eq!(rot.times, vec![0.0, 0.5, 1.25]);
        let crate::ChannelData::Rotation(values) = &rot.data else {
            unreachable!()
        };
        assert_eq!(
            values,
            &[IDENTITY_ROTATION, nodes[2].rotation, IDENTITY_ROTATION]
        );

        let trs = clip
            .channels
            .iter()
            .find(|c| matches!(c.data, crate::ChannelData::Translation(_)))
            .expect("translation channel");
        assert_eq!(trs.target_node, 1);
        assert_eq!(trs.interpolation, crate::Interpolation::Step);
        assert_eq!(trs.times, vec![0.0, 0.25]);
        let crate::ChannelData::Translation(values) = &trs.data else {
            unreachable!()
        };
        assert_eq!(values, &[[0.0, 0.9, 0.0], [0.0, 1.0, 0.0]]);
    }

    /// The general path is deterministic too — animations included.
    #[test]
    fn scene_output_is_deterministic() {
        let (nodes, meshes, animations) = articulated();
        let a = write_glb_scene(&nodes, &meshes, &[stone()], &animations).unwrap();
        let b = write_glb_scene(&nodes, &meshes, &[stone()], &animations).unwrap();
        assert_eq!(a, b);
        // ...and sensitive to a moved keyframe.
        let mut moved = animations.clone();
        moved[0].channels[0].keys[1].0 = 0.6;
        assert_ne!(
            write_glb_scene(&nodes, &meshes, &[stone()], &moved).unwrap(),
            a
        );
    }

    /// Rig bugs are rejected at the writer, where the message can name the node.
    #[test]
    fn rejects_malformed_rigs() {
        let (nodes, meshes, animations) = articulated();
        let write = |n: &[GlbNode], a: &[GlbAnimation]| write_glb_scene(n, &meshes, &[stone()], a);
        assert!(write(&nodes, &animations).is_ok());

        let bad_parent = |parent| {
            let mut n = nodes.clone();
            n[1].parent = Some(parent);
            n
        };
        assert!(write(&bad_parent(9), &[]).is_err(), "parent out of range");
        assert!(write(&bad_parent(1), &[]).is_err(), "self-parent");
        // A two-node cycle: hip ← forearm ← hip.
        let mut cycle = nodes.clone();
        cycle[1].parent = Some(2);
        assert!(write(&cycle, &[]).is_err(), "parent cycle");

        let mut bad_mesh = nodes.clone();
        bad_mesh[1].mesh = Some(7);
        assert!(write(&bad_mesh, &[]).is_err(), "mesh out of range");

        let mut unnormalized = nodes.clone();
        unnormalized[2].rotation = [0.0, 0.0, 0.7, 0.7];
        assert!(write(&unnormalized, &[]).is_err(), "non-unit quaternion");

        let bad_anim = |channels| {
            vec![GlbAnimation {
                name: "x".into(),
                channels,
            }]
        };
        assert!(
            write(
                &nodes,
                &bad_anim(vec![GlbChannel::rotation(9, [(0.0, IDENTITY_ROTATION)])])
            )
            .is_err(),
            "channel targets a node that does not exist"
        );
        assert!(
            write(&nodes, &bad_anim(vec![GlbChannel::rotation(1, [])])).is_err(),
            "channel with no keyframes"
        );
        assert!(
            write(
                &nodes,
                &bad_anim(vec![GlbChannel::rotation(
                    1,
                    [(0.5, IDENTITY_ROTATION), (0.5, IDENTITY_ROTATION)]
                )])
            )
            .is_err(),
            "key times that do not strictly increase"
        );
        assert!(
            write(
                &nodes,
                &bad_anim(vec![
                    GlbChannel::rotation(1, [(0.0, IDENTITY_ROTATION)]),
                    GlbChannel::rotation(1, [(0.0, IDENTITY_ROTATION)]),
                ])
            )
            .is_err(),
            "two channels driving one node property"
        );
        // A translation channel whose keys are quaternions (path/value disagreement).
        assert!(
            write(
                &nodes,
                &bad_anim(vec![GlbChannel {
                    node: 1,
                    path: GlbPath::Translation,
                    interpolation: GlbInterpolation::Linear,
                    keys: vec![(0.0, GlbValue::Quat(IDENTITY_ROTATION))],
                }])
            )
            .is_err(),
            "value type disagrees with the channel path"
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

    /// FNV-1a over the encoded bytes — a stable, dependency-free digest for the pin below.
    fn digest(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// The flat API's **bytes** are pinned, not merely its meaning.
    ///
    /// Every generated dungeon on disk is cook-cached on a hash of exactly these bytes,
    /// so a change here silently invalidates every cached bake for every seed anyone has
    /// played. That is sometimes the right thing (a new attribute, a `gltf` upgrade) and
    /// never a thing to do by accident — hence a pin rather than a round trip: if this
    /// fails, decide deliberately and re-pin.
    #[test]
    fn flat_output_bytes_are_pinned() {
        let mut second = quad();
        second.name = "chunk_1".into();
        second.material = 1;
        let bytes = write_glb(
            &[quad(), second],
            &[
                stone(),
                GlbMaterial {
                    name: "wood".into(),
                    ..GlbMaterial::default()
                },
            ],
        )
        .unwrap();
        assert_eq!(
            (bytes.len(), digest(&bytes)),
            (2288, 0xd485_473a_1a45_2497),
            "the flat writer's bytes moved — every cook-cached generated asset re-cooks"
        );
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
