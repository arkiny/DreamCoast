//! FBX import via **ufbx** (Phase 13 Stage E).
//!
//! Maps an FBX file to the *same* neutral [`GltfScene`] the glTF importer produces, so
//! everything downstream (the scene-graph instantiation, GPU skin cache, HW-RT BLAS,
//! SW-RT GDF) is format-agnostic — only the parser differs. ufbx is compiled from
//! source by the `ufbx` binding crate (`cc`, no bindgen); this module is gated behind
//! the `fbx` cargo feature so the default glTF-only build needs no C compiler.
//!
//! Coordinate/unit alignment (the highest-risk part) is delegated to ufbx's load options
//! (`target_axes = right-handed Y-up`, `target_unit_meters = 1.0`) so the import lands in
//! the engine's convention (1 unit = 1 m, Y-up, RH), matching the glTF path. The unit/axis
//! bake mode depends on skinning — see [`load_opts`] (ufbx can't both rewrite geometry into
//! the target space and preserve skin clusters). FBX's bottom-left UV origin is flipped to
//! glTF's top-left.
//!
//! Animation lives in a *separate* FBX for the Intel Sponza knight (mesh + skeleton in
//! one file, animation curves in another), so [`load_fbx_scene`] takes an optional anim
//! path: its baked node-TRS tracks are matched to the mesh skeleton **by bone name**.

#![cfg(feature = "fbx")]

use std::collections::HashMap;
use std::path::Path;

use dreamcoast_core::EngineError;

use crate::{
    AlphaMode, ChannelData, GltfAnimation, GltfChannel, GltfMaterial, GltfNode, GltfPrimitive,
    GltfScene, GltfSkin, ImageData, Interpolation, MaterialKind, MeshVertex, TexData,
};

/// The maximum bone influences per vertex the skin stream carries (matches the glTF
/// path + the skinning shader). ufbx sorts weights descending, so the top 4 are kept.
const MAX_INFLUENCES: usize = 4;

fn asset_err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::Asset(format!("fbx: {msg}"))
}

/// ufbx load options, chosen by whether the file is **skinned**. The two modes trade off
/// because ufbx can't both rewrite geometry into the target space *and* keep skinning:
/// - `skinned` → `AdjustTransforms` + `HelperNodes`: preserves skin clusters. The joint
///   world transforms come out in metres/Y-up and each cluster's `geometry_to_bone`
///   carries the residual cm→m scale, so the skinned (palette·vertex) output lands in
///   metre world space even though the raw bind-pose vertices stay in the source's cm.
/// - static → `ModifyGeometry`: bakes the axis+unit conversion straight into the vertices
///   so geometry *and* node transforms share metres (self-consistent for a plain draw).
///   `ModifyGeometry` drops skinning, so it's only used when there are no skin clusters.
fn load_opts(skinned: bool) -> ufbx::LoadOpts<'static> {
    let (space_conversion, geometry_transform_handling) = if skinned {
        (
            ufbx::SpaceConversion::AdjustTransforms,
            ufbx::GeometryTransformHandling::HelperNodes,
        )
    } else {
        (
            ufbx::SpaceConversion::ModifyGeometry,
            ufbx::GeometryTransformHandling::ModifyGeometry,
        )
    };
    ufbx::LoadOpts {
        target_axes: ufbx::axes_right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion,
        generate_missing_normals: true,
        geometry_transform_handling,
        ..Default::default()
    }
}

/// Convert a ufbx 3×4 affine [`ufbx::Matrix`] (column vectors `m{row}{col}`) into a
/// column-major `[f32; 16]` 4×4 with the implicit `(0,0,0,1)` bottom row — the layout
/// [`GltfSkin::inverse_bind`] uses (read back via `Mat4::from_cols_array`).
fn matrix_to_cols(m: &ufbx::Matrix) -> [f32; 16] {
    [
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0, // column 0
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0, // column 1
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0, // column 2
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0, // column 3 (translation)
    ]
}

/// Import an FBX file (optionally merging a second FBX of animation-only curves) into a
/// neutral [`GltfScene`]. `mesh_path` supplies geometry + skeleton + skin (bind pose);
/// `anim_path`, if given, supplies the animation clip whose baked tracks are matched to
/// the skeleton by bone name.
pub fn load_fbx_scene(
    mesh_path: impl AsRef<Path>,
    anim_path: Option<impl AsRef<Path>>,
) -> Result<GltfScene, EngineError> {
    let mesh_path = mesh_path.as_ref();
    let mesh_dir = mesh_path.parent().unwrap_or_else(|| Path::new("."));
    let path_str = mesh_path
        .to_str()
        .ok_or_else(|| asset_err("mesh path is not valid UTF-8"))?;
    // Load once assuming skinned (preserves skin clusters); if the file has none, reload
    // baking units into the geometry so a static mesh's vertices + transforms agree.
    let mut scene = ufbx::load_file(path_str, load_opts(true))
        .map_err(|e| asset_err(format!("load {}: {e:?}", mesh_path.display())))?;
    if scene.skin_clusters.is_empty() {
        scene = ufbx::load_file(path_str, load_opts(false))
            .map_err(|e| asset_err(format!("reload {}: {e:?}", mesh_path.display())))?;
    }

    // Dense node index space: `scene.nodes[i]` → GltfScene.nodes[i]. Reference lookups
    // (children / bone_node / mesh owner) go through the globally-unique element_id.
    let mut elem_to_node: HashMap<u32, usize> = HashMap::new();
    for (i, node) in (&scene.nodes).into_iter().enumerate() {
        elem_to_node.insert(node.element.element_id, i);
    }
    let mut mesh_elem_to_idx: HashMap<u32, usize> = HashMap::new();
    for (i, mesh) in (&scene.meshes).into_iter().enumerate() {
        mesh_elem_to_idx.insert(mesh.element.element_id, i);
    }

    // Node name → GltfScene node index, for resolving the separate animation file's
    // bone tracks against this skeleton.
    let mut name_to_node: HashMap<String, usize> = HashMap::new();

    // --- Nodes ------------------------------------------------------------------
    let mut nodes: Vec<GltfNode> = Vec::with_capacity(scene.nodes.len());
    for node in &scene.nodes {
        let t = &node.local_transform;
        let name = node.element.name.as_ref();
        if !name.is_empty() {
            name_to_node.entry(name.to_owned()).or_insert(nodes.len());
        }
        let children = (&node.children)
            .into_iter()
            .filter_map(|c| elem_to_node.get(&c.element.element_id).copied())
            .collect();
        let mesh = node
            .mesh
            .as_ref()
            .and_then(|m| mesh_elem_to_idx.get(&m.element.element_id).copied());
        nodes.push(GltfNode {
            name: (!name.is_empty()).then(|| name.to_owned()),
            translation: [
                t.translation.x as f32,
                t.translation.y as f32,
                t.translation.z as f32,
            ],
            rotation: [
                t.rotation.x as f32,
                t.rotation.y as f32,
                t.rotation.z as f32,
                t.rotation.w as f32,
            ],
            scale: [t.scale.x as f32, t.scale.y as f32, t.scale.z as f32],
            children,
            mesh,
            skin: None, // filled in below for skinned meshes
        });
    }
    // Roots = the synthetic root node's children (the real top-level nodes).
    let roots: Vec<usize> = (&scene.root_node.children)
        .into_iter()
        .filter_map(|c| elem_to_node.get(&c.element.element_id).copied())
        .collect();

    // --- Materials + images -----------------------------------------------------
    let mut images: Vec<TexData> = Vec::new();
    let mut image_cache: HashMap<String, usize> = HashMap::new();
    let materials: Vec<GltfMaterial> = (&scene.materials)
        .into_iter()
        .map(|m| build_material(m, mesh_dir, &mut images, &mut image_cache))
        .collect();
    let mut material_elem_to_idx: HashMap<u32, usize> = HashMap::new();
    for (i, m) in (&scene.materials).into_iter().enumerate() {
        material_elem_to_idx.insert(m.element.element_id, i);
    }

    // --- Meshes (primitives, split by material) + skins -------------------------
    let mut meshes: Vec<Vec<GltfPrimitive>> = Vec::with_capacity(scene.meshes.len());
    let mut skins: Vec<GltfSkin> = Vec::new();
    // meshidx → Some(skin index) once built, so every node referencing it gets `skin`.
    let mut mesh_skin: Vec<Option<usize>> = vec![None; scene.meshes.len()];

    for (mesh_idx, mesh) in (&scene.meshes).into_iter().enumerate() {
        let deformer = (&mesh.skin_deformers).into_iter().next();
        if let Some(def) = deformer {
            // Build the skin: joint order MUST stay parallel to `def.clusters` because a
            // vertex weight's `cluster_index` indexes into it. `bone_node` is nullable, so a
            // boneless cluster keeps its slot with a fallback node index (it carries little/no
            // weight) rather than dropping the whole skin.
            let mut joints = Vec::with_capacity(def.clusters.len());
            let mut inverse_bind = Vec::with_capacity(def.clusters.len());
            let mut boneless = 0usize;
            for cluster in &def.clusters {
                let idx = cluster
                    .bone_node
                    .as_ref()
                    .and_then(|b| elem_to_node.get(&b.element.element_id).copied());
                joints.push(idx.unwrap_or(0));
                if idx.is_none() {
                    boneless += 1;
                }
                inverse_bind.push(matrix_to_cols(&cluster.geometry_to_bone));
            }
            // A skin is real only if at least one cluster resolved to a bone.
            if joints.len() > boneless {
                mesh_skin[mesh_idx] = Some(skins.len());
                skins.push(GltfSkin {
                    joints,
                    inverse_bind,
                });
                if boneless > 0 {
                    eprintln!(
                        "fbx: mesh {mesh_idx}: {boneless} boneless skin cluster(s) → fallback joint 0"
                    );
                }
            } else {
                eprintln!(
                    "fbx: mesh {mesh_idx}: skin deformer has {} clusters but none resolved to a bone",
                    def.clusters.len()
                );
            }
        }
        meshes.push(build_primitives(mesh, deformer, &material_elem_to_idx));
    }

    // Tag every node that draws a skinned mesh with its skin index.
    for (node, src) in nodes.iter_mut().zip(&scene.nodes) {
        if let Some(m) = src
            .mesh
            .as_ref()
            .and_then(|m| mesh_elem_to_idx.get(&m.element.element_id).copied())
        {
            node.skin = mesh_skin[m];
        }
    }

    // --- Animation (optional separate file) -------------------------------------
    let mut animations: Vec<GltfAnimation> = Vec::new();
    if let Some(anim_path) = anim_path {
        match load_animation(anim_path.as_ref(), &name_to_node) {
            Ok(Some(clip)) => animations.push(clip),
            Ok(None) => {}
            Err(e) => eprintln!("fbx: animation import failed: {e}"),
        }
    }

    eprintln!(
        "fbx '{}': {} nodes, {} meshes, {} materials, {} images, {} skins, {} clips",
        mesh_path.display(),
        nodes.len(),
        meshes.len(),
        materials.len(),
        images.len(),
        skins.len(),
        animations.len(),
    );

    Ok(GltfScene {
        nodes,
        roots,
        meshes,
        materials,
        images,
        animations,
        skins,
    })
}

/// Triangulate an FBX mesh into one [`GltfPrimitive`] per material, expanding to a
/// non-indexed vertex list (indices `0..n`) — simplest correct path; matches the glTF
/// fallback. Per-vertex skin joints/weights (top-4, renormalized) come from `deformer`.
fn build_primitives(
    mesh: &ufbx::Mesh,
    deformer: Option<&ufbx::SkinDeformer>,
    material_elem_to_idx: &HashMap<u32, usize>,
) -> Vec<GltfPrimitive> {
    // Group faces by local material slot (index into `mesh.materials`); `face_material`
    // is empty for a single-material mesh.
    let num_slots = mesh.materials.len().max(1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_slots];
    for fi in 0..mesh.num_faces {
        let slot = mesh.face_material.get(fi).copied().unwrap_or(0) as usize;
        groups[slot.min(num_slots - 1)].push(fi);
    }

    let mut tri: Vec<u32> = Vec::new();
    let mut prims = Vec::new();
    for (slot, faces) in groups.iter().enumerate() {
        if faces.is_empty() {
            continue;
        }
        let material = mesh
            .materials
            .get(slot)
            .and_then(|m| material_elem_to_idx.get(&m.element.element_id).copied());

        let mut vertices: Vec<MeshVertex> = Vec::new();
        let mut joints: Vec<[u16; 4]> = Vec::new();
        let mut weights: Vec<[f32; 4]> = Vec::new();

        for &fi in faces {
            let face = mesh.faces[fi];
            ufbx::triangulate_face_vec(&mut tri, mesh, face);
            for &ci in tri.iter() {
                let ci = ci as usize;
                let p = mesh.vertex_position[ci];
                let n = if mesh.vertex_normal.exists {
                    mesh.vertex_normal[ci]
                } else {
                    ufbx::Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    }
                };
                let uv = if mesh.vertex_uv.exists {
                    mesh.vertex_uv[ci]
                } else {
                    ufbx::Vec2 { x: 0.0, y: 0.0 }
                };
                vertices.push(MeshVertex {
                    pos: [p.x as f32, p.y as f32, p.z as f32],
                    normal: [n.x as f32, n.y as f32, n.z as f32],
                    // FBX UV origin is bottom-left; glTF/our sampler is top-left → flip V.
                    uv: [uv.x as f32, 1.0 - uv.y as f32],
                });
                if let Some(def) = deformer {
                    let (js, ws) = vertex_influences(def, mesh.vertex_indices[ci] as usize);
                    joints.push(js);
                    weights.push(ws);
                }
            }
        }

        let n = vertices.len() as u32;
        prims.push(GltfPrimitive {
            vertices,
            indices: (0..n).collect(),
            material,
            joints: deformer.map(|_| joints),
            weights: deformer.map(|_| weights),
            morph_targets: Vec::new(),
        });
    }
    prims
}

/// The top-4 (joint, weight) influences of a unique vertex, renormalized. ufbx sorts a
/// vertex's weights descending, so the first 4 are the most significant; any beyond 4
/// are dropped and the remainder renormalized (documented limit, matches the glTF path).
fn vertex_influences(def: &ufbx::SkinDeformer, vertex: usize) -> ([u16; 4], [f32; 4]) {
    let mut js = [0u16; 4];
    let mut ws = [0f32; 4];
    let Some(sv) = def.vertices.get(vertex) else {
        return (js, ws);
    };
    let n = (sv.num_weights as usize).min(MAX_INFLUENCES);
    let base = sv.weight_begin as usize;
    let mut sum = 0.0f32;
    for k in 0..n {
        let w = def.weights[base + k];
        js[k] = w.cluster_index as u16;
        ws[k] = w.weight as f32;
        sum += ws[k];
    }
    if sum > 0.0 {
        for w in &mut ws {
            *w /= sum;
        }
    } else {
        // Degenerate (no weight) — bind fully to the first joint so the vertex still
        // follows the skeleton rather than collapsing to the origin.
        ws[0] = 1.0;
    }
    (js, ws)
}

/// Build a neutral [`GltfMaterial`] from a ufbx material's normalized PBR maps, loading
/// any referenced texture files (best-effort — a missing file logs and falls back to the
/// scalar factor).
fn build_material(
    m: &ufbx::Material,
    mesh_dir: &Path,
    images: &mut Vec<TexData>,
    cache: &mut HashMap<String, usize>,
) -> GltfMaterial {
    let pbr = &m.pbr;
    let base = pbr.base_color.value_vec4;
    let base_color_factor = if pbr.base_color.has_value {
        [base.x as f32, base.y as f32, base.z as f32, base.w as f32]
    } else {
        [1.0, 1.0, 1.0, 1.0]
    };
    let metallic_factor = if pbr.metalness.has_value {
        pbr.metalness.value_vec4.x as f32
    } else {
        0.0
    };
    let roughness_factor = if pbr.roughness.has_value {
        pbr.roughness.value_vec4.x as f32
    } else {
        0.6
    };
    let emissive_factor = if pbr.emission_color.has_value {
        let e = pbr.emission_color.value_vec4;
        [e.x as f32, e.y as f32, e.z as f32]
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut tex = |map: &ufbx::MaterialMap| -> Option<usize> {
        let t = map.texture.as_ref()?;
        resolve_texture(t.as_ref(), mesh_dir, images, cache)
    };

    GltfMaterial {
        base_color_factor,
        metallic_factor,
        roughness_factor,
        emissive_factor,
        base_color: tex(&pbr.base_color),
        metallic_roughness: None, // FBX rarely packs metal/rough together; use factors
        normal: tex(&pbr.normal_map),
        emissive: tex(&pbr.emission_color),
        alpha_cutoff: 0.0,
        alpha_mode: AlphaMode::Opaque,
        kind: MaterialKind::Opaque,
        double_sided: m.features.double_sided.enabled,
    }
}

/// Resolve a ufbx texture to an image index, loading + decoding the PNG/file. Tries the
/// absolute path, then the FBX-relative path, then a case-insensitive basename search in
/// the FBX directory and a sibling `Textures/` folder (authoring paths are often stale
/// absolute Maya paths). Deduplicates by resolved path. Returns `None` (logged) if the
/// file can't be found or decoded.
fn resolve_texture(
    t: &ufbx::Texture,
    mesh_dir: &Path,
    images: &mut Vec<TexData>,
    cache: &mut HashMap<String, usize>,
) -> Option<usize> {
    let candidates = texture_candidates(t, mesh_dir);
    let found = candidates.iter().find(|p| p.exists())?;
    let key = found.to_string_lossy().into_owned();
    if let Some(&idx) = cache.get(&key) {
        return Some(idx);
    }
    match image::open(found) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            let idx = images.len();
            images.push(TexData::Rgba8(ImageData {
                width,
                height,
                rgba8: rgba.into_raw(),
            }));
            cache.insert(key, idx);
            Some(idx)
        }
        Err(e) => {
            eprintln!("fbx: texture '{}' decode failed: {e}", found.display());
            None
        }
    }
}

/// Candidate on-disk paths for a texture, most-specific first.
fn texture_candidates(t: &ufbx::Texture, mesh_dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut push = |p: std::path::PathBuf| {
        if !p.as_os_str().is_empty() {
            out.push(p);
        }
    };
    let abs = t.absolute_filename.as_ref();
    if !abs.is_empty() {
        push(std::path::PathBuf::from(abs));
    }
    let rel = t.relative_filename.as_ref();
    if !rel.is_empty() {
        push(mesh_dir.join(rel));
    }
    // Basename fallbacks: the pack keeps textures in a sibling `Textures/` folder, and
    // the FBX sits in `Exports/FBX/`, so search a few likely roots by file name.
    let name = t.filename.as_ref();
    let base = Path::new(name)
        .file_name()
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    if !base.as_os_str().is_empty() {
        push(mesh_dir.join(&base));
        push(mesh_dir.join("Textures").join(&base));
        for up in [mesh_dir.parent(), mesh_dir.parent().and_then(Path::parent)]
            .into_iter()
            .flatten()
        {
            push(up.join("Textures").join(&base));
            push(up.join(&base));
        }
    }
    out
}

/// Load an animation-only FBX and bake its node-TRS tracks into a [`GltfAnimation`]
/// whose channels target this scene's node indices (matched by bone name). Returns
/// `Ok(None)` if the file has no animation.
fn load_animation(
    path: &Path,
    name_to_node: &HashMap<String, usize>,
) -> Result<Option<GltfAnimation>, EngineError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| asset_err("anim path is not valid UTF-8"))?;
    // Skinned opts so the animation skeleton shares the mesh's metre/Y-up space.
    let scene = ufbx::load_file(path_str, load_opts(true))
        .map_err(|e| asset_err(format!("load anim {}: {e:?}", path.display())))?;

    let anim: &ufbx::Anim = &scene.anim;
    if anim.time_end <= anim.time_begin {
        return Ok(None);
    }
    // Bake the curves to dense keyframes (linear-samplable). `trim_start_time` rebases
    // the timeline to 0, matching glTF clips + our looping `AnimationPlayer`.
    let baked = ufbx::bake_anim(
        &scene,
        anim,
        ufbx::BakeOpts {
            trim_start_time: true,
            ..Default::default()
        },
    )
    .map_err(|e| asset_err(format!("bake anim: {e:?}")))?;

    let mut channels: Vec<GltfChannel> = Vec::new();
    let mut matched = 0usize;
    let mut unmatched = 0usize;
    for bn in &baked.nodes {
        let src = &scene.nodes[bn.typed_id as usize];
        let name = src.element.name.as_ref();
        let Some(&target_node) = name_to_node.get(name) else {
            unmatched += 1;
            continue;
        };
        matched += 1;
        if !bn.translation_keys.is_empty() {
            channels.push(GltfChannel {
                target_node,
                interpolation: Interpolation::Linear,
                times: bn.translation_keys.iter().map(|k| k.time as f32).collect(),
                data: ChannelData::Translation(
                    bn.translation_keys
                        .iter()
                        .map(|k| [k.value.x as f32, k.value.y as f32, k.value.z as f32])
                        .collect(),
                ),
            });
        }
        if !bn.rotation_keys.is_empty() {
            channels.push(GltfChannel {
                target_node,
                interpolation: Interpolation::Linear,
                times: bn.rotation_keys.iter().map(|k| k.time as f32).collect(),
                data: ChannelData::Rotation(
                    bn.rotation_keys
                        .iter()
                        .map(|k| {
                            [
                                k.value.x as f32,
                                k.value.y as f32,
                                k.value.z as f32,
                                k.value.w as f32,
                            ]
                        })
                        .collect(),
                ),
            });
        }
        if !bn.scale_keys.is_empty() {
            channels.push(GltfChannel {
                target_node,
                interpolation: Interpolation::Linear,
                times: bn.scale_keys.iter().map(|k| k.time as f32).collect(),
                data: ChannelData::Scale(
                    bn.scale_keys
                        .iter()
                        .map(|k| [k.value.x as f32, k.value.y as f32, k.value.z as f32])
                        .collect(),
                ),
            });
        }
    }

    eprintln!(
        "fbx anim '{}': {matched} bones matched, {unmatched} unmatched, {:.2}s, {} channels",
        path.display(),
        baked.playback_duration,
        channels.len(),
    );
    if channels.is_empty() {
        return Ok(None);
    }
    Ok(Some(GltfAnimation {
        name: Some("fbx".to_owned()),
        channels,
        duration: baked.playback_duration as f32,
    }))
}
