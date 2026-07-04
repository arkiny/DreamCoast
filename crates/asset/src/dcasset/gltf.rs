//! glTF-scene chunk codec — a cooked, multi-mesh/multi-material scene (levels,
//! glTF imports). Stores the node hierarchy, every primitive's geometry, materials,
//! and the **block-compressed** texture table, so a cached scene loads without
//! re-parsing glTF, re-decoding images, or re-encoding BCn (the expensive cook work).
//!
//! Scope: **static** scenes only (no skins / morph targets / animations) — the cook
//! refuses to cache an animated scene, so those fields are always empty here. Reusing
//! the mesh chunk's `encode_texdata` keeps the on-disk texture encoding single-source.

use dreamcoast_core::EngineError;

use super::mesh::{decode_texdata, encode_texdata};
use super::{CHUNK_GLTF_SCENE, Header, Reader, Writer, open_chunk, write_single_chunk};
use crate::gltf_scene::{AlphaMode, GltfMaterial, GltfNode, GltfPrimitive, MaterialKind};
use crate::{GltfScene, MeshVertex};

// AlphaMode / MaterialKind tags (stored as u32 so the enums can grow without a format bump).
const ALPHA_OPAQUE: u32 = 0;
const ALPHA_MASK: u32 = 1;
const ALPHA_BLEND: u32 = 2;
const KIND_OPAQUE: u32 = 0;
const KIND_MASKED: u32 = 1;
const KIND_DECAL: u32 = 2;
const KIND_TRANSPARENT: u32 = 3;

/// Serialize a **static** `scene` into a `.dcasset` buffer (one glTF-scene chunk).
/// `src_hash` is the invalidation key (source glTF bytes + compression tier).
pub fn write_scene(scene: &GltfScene, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_GLTF_SCENE, &encode_scene(scene), src_hash)
}

/// Decode a glTF-scene `.dcasset` buffer into its [`Header`] and [`GltfScene`].
pub fn read_scene(bytes: &[u8]) -> Result<(Header, GltfScene), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_GLTF_SCENE, "gltf-scene")?;
    Ok((header, decode_scene(&mut r)?))
}

// --- small helpers ----------------------------------------------------------------

/// `Option<usize>` as a 0/1 flag + (when set) a `u32` value.
fn put_opt(w: &mut Writer, v: Option<usize>) {
    match v {
        Some(i) => {
            w.u32(1);
            w.u32(i as u32);
        }
        None => w.u32(0),
    }
}
fn get_opt(r: &mut Reader) -> Result<Option<usize>, EngineError> {
    if r.u32()? != 0 {
        Ok(Some(r.u32()? as usize))
    } else {
        Ok(None)
    }
}

/// `Vec<usize>` as a count + `u32` values.
fn put_usizes(w: &mut Writer, v: &[usize]) {
    w.u32(v.len() as u32);
    for &i in v {
        w.u32(i as u32);
    }
}
fn get_usizes(r: &mut Reader) -> Result<Vec<usize>, EngineError> {
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.u32()? as usize);
    }
    Ok(out)
}

/// `Option<String>` as a 0/1 flag + (when set) a length-prefixed string.
fn put_opt_str(w: &mut Writer, s: &Option<String>) {
    match s {
        Some(s) => {
            w.u32(1);
            w.str(s);
        }
        None => w.u32(0),
    }
}
fn get_opt_str(r: &mut Reader) -> Result<Option<String>, EngineError> {
    if r.u32()? != 0 {
        Ok(Some(r.str()?))
    } else {
        Ok(None)
    }
}

// --- scene ------------------------------------------------------------------------

fn encode_scene(scene: &GltfScene) -> Vec<u8> {
    let mut w = Writer::default();

    w.u32(scene.nodes.len() as u32);
    for n in &scene.nodes {
        encode_node(&mut w, n);
    }
    put_usizes(&mut w, &scene.roots);

    w.u32(scene.meshes.len() as u32);
    for prims in &scene.meshes {
        w.u32(prims.len() as u32);
        for p in prims {
            encode_primitive(&mut w, p);
        }
    }

    w.u32(scene.materials.len() as u32);
    for m in &scene.materials {
        encode_material(&mut w, m);
    }

    w.u32(scene.images.len() as u32);
    for tex in &scene.images {
        encode_texdata(&mut w, tex);
    }
    w.buf
}

fn decode_scene(r: &mut Reader) -> Result<GltfScene, EngineError> {
    let node_count = r.u32()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(decode_node(r)?);
    }
    let roots = get_usizes(r)?;

    let mesh_count = r.u32()? as usize;
    let mut meshes = Vec::with_capacity(mesh_count);
    for _ in 0..mesh_count {
        let prim_count = r.u32()? as usize;
        let mut prims = Vec::with_capacity(prim_count);
        for _ in 0..prim_count {
            prims.push(decode_primitive(r)?);
        }
        meshes.push(prims);
    }

    let mat_count = r.u32()? as usize;
    let mut materials = Vec::with_capacity(mat_count);
    for _ in 0..mat_count {
        materials.push(decode_material(r)?);
    }

    let img_count = r.u32()? as usize;
    let mut images = Vec::with_capacity(img_count);
    for _ in 0..img_count {
        images.push(decode_texdata(r)?);
    }

    Ok(GltfScene {
        nodes,
        roots,
        meshes,
        materials,
        images,
        animations: Vec::new(), // static scene: never cached with animations
        skins: Vec::new(),
    })
}

fn encode_node(w: &mut Writer, n: &GltfNode) {
    put_opt_str(w, &n.name);
    for c in n.translation {
        w.f32(c);
    }
    for c in n.rotation {
        w.f32(c);
    }
    for c in n.scale {
        w.f32(c);
    }
    put_usizes(w, &n.children);
    put_opt(w, n.mesh);
    put_opt(w, n.skin);
}

fn decode_node(r: &mut Reader) -> Result<GltfNode, EngineError> {
    let name = get_opt_str(r)?;
    let translation = [r.f32()?, r.f32()?, r.f32()?];
    let rotation = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
    let scale = [r.f32()?, r.f32()?, r.f32()?];
    let children = get_usizes(r)?;
    let mesh = get_opt(r)?;
    let skin = get_opt(r)?;
    Ok(GltfNode {
        name,
        translation,
        rotation,
        scale,
        children,
        mesh,
        skin,
    })
}

fn encode_primitive(w: &mut Writer, p: &GltfPrimitive) {
    w.u32(p.vertices.len() as u32);
    w.u32(p.indices.len() as u32);
    put_opt(w, p.material);
    for v in &p.vertices {
        for c in v.pos {
            w.f32(c);
        }
        for c in v.normal {
            w.f32(c);
        }
        for c in v.uv {
            w.f32(c);
        }
    }
    for &i in &p.indices {
        w.u32(i);
    }
}

fn decode_primitive(r: &mut Reader) -> Result<GltfPrimitive, EngineError> {
    let vtx_count = r.u32()? as usize;
    let idx_count = r.u32()? as usize;
    let material = get_opt(r)?;
    let mut vertices = Vec::with_capacity(vtx_count);
    for _ in 0..vtx_count {
        vertices.push(MeshVertex {
            pos: [r.f32()?, r.f32()?, r.f32()?],
            normal: [r.f32()?, r.f32()?, r.f32()?],
            uv: [r.f32()?, r.f32()?],
        });
    }
    let mut indices = Vec::with_capacity(idx_count);
    for _ in 0..idx_count {
        indices.push(r.u32()?);
    }
    // Static cook: no skin / morph side data.
    Ok(GltfPrimitive {
        vertices,
        indices,
        material,
        joints: None,
        weights: None,
        morph_targets: Vec::new(),
    })
}

fn encode_material(w: &mut Writer, m: &GltfMaterial) {
    for c in m.base_color_factor {
        w.f32(c);
    }
    w.f32(m.metallic_factor);
    w.f32(m.roughness_factor);
    for c in m.emissive_factor {
        w.f32(c);
    }
    put_opt(w, m.base_color);
    put_opt(w, m.metallic_roughness);
    put_opt(w, m.normal);
    put_opt(w, m.emissive);
    w.f32(m.alpha_cutoff);
    w.u32(match m.alpha_mode {
        AlphaMode::Opaque => ALPHA_OPAQUE,
        AlphaMode::Mask => ALPHA_MASK,
        AlphaMode::Blend => ALPHA_BLEND,
    });
    w.u32(match m.kind {
        MaterialKind::Opaque => KIND_OPAQUE,
        MaterialKind::Masked => KIND_MASKED,
        MaterialKind::Decal => KIND_DECAL,
        MaterialKind::Transparent => KIND_TRANSPARENT,
    });
    w.u32(m.double_sided as u32);
}

fn decode_material(r: &mut Reader) -> Result<GltfMaterial, EngineError> {
    let base_color_factor = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
    let metallic_factor = r.f32()?;
    let roughness_factor = r.f32()?;
    let emissive_factor = [r.f32()?, r.f32()?, r.f32()?];
    let base_color = get_opt(r)?;
    let metallic_roughness = get_opt(r)?;
    let normal = get_opt(r)?;
    let emissive = get_opt(r)?;
    let alpha_cutoff = r.f32()?;
    let alpha_mode = match r.u32()? {
        ALPHA_MASK => AlphaMode::Mask,
        ALPHA_BLEND => AlphaMode::Blend,
        _ => AlphaMode::Opaque,
    };
    let kind = match r.u32()? {
        KIND_MASKED => MaterialKind::Masked,
        KIND_DECAL => MaterialKind::Decal,
        KIND_TRANSPARENT => MaterialKind::Transparent,
        _ => MaterialKind::Opaque,
    };
    let double_sided = r.u32()? != 0;
    Ok(GltfMaterial {
        base_color_factor,
        metallic_factor,
        roughness_factor,
        emissive_factor,
        base_color,
        metallic_roughness,
        normal,
        emissive,
        alpha_cutoff,
        alpha_mode,
        kind,
        double_sided,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::BcFormat;
    use crate::{ImageData, MeshVertex, TexData};

    fn sample_scene() -> GltfScene {
        GltfScene {
            nodes: vec![
                GltfNode {
                    name: Some("root".into()),
                    translation: [1.0, 2.0, 3.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    children: vec![1],
                    mesh: None,
                    skin: None,
                },
                GltfNode {
                    name: None,
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.1, 0.2, 0.3, 0.9],
                    scale: [2.0, 2.0, 2.0],
                    children: vec![],
                    mesh: Some(0),
                    skin: None,
                },
            ],
            roots: vec![0],
            meshes: vec![vec![GltfPrimitive {
                vertices: vec![
                    MeshVertex {
                        pos: [1.0, 2.0, 3.0],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.25, 0.75],
                    },
                    MeshVertex {
                        pos: [-1.5, 0.0, 4.25],
                        normal: [1.0, 0.0, 0.0],
                        uv: [1.0, 0.0],
                    },
                ],
                indices: vec![0, 1, 0],
                material: Some(0),
                joints: None,
                weights: None,
                morph_targets: Vec::new(),
            }]],
            materials: vec![GltfMaterial {
                base_color_factor: [0.2, 0.4, 0.6, 1.0],
                metallic_factor: 0.3,
                roughness_factor: 0.7,
                emissive_factor: [0.1, 0.0, 0.05],
                base_color: Some(0),
                metallic_roughness: None,
                normal: Some(1),
                emissive: None,
                alpha_cutoff: 0.5,
                alpha_mode: AlphaMode::Mask,
                kind: MaterialKind::Masked,
                double_sided: true,
            }],
            images: vec![
                TexData::Bc {
                    format: BcFormat::Bc1,
                    srgb: true,
                    width: 8,
                    height: 8,
                    mips: vec![vec![1u8; 32], vec![2u8; 8]],
                },
                TexData::Rgba8(ImageData {
                    width: 1,
                    height: 1,
                    rgba8: vec![128, 128, 255, 255],
                }),
            ],
            animations: Vec::new(),
            skins: Vec::new(),
        }
    }

    #[test]
    fn scene_roundtrips() {
        let scene = sample_scene();
        let bytes = write_scene(&scene, 0xabcd);
        let (header, decoded) = read_scene(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0xabcd);
        assert_eq!(decoded.nodes.len(), 2);
        assert_eq!(decoded.nodes[0].name.as_deref(), Some("root"));
        assert_eq!(decoded.nodes[1].name, None);
        assert_eq!(decoded.nodes[1].mesh, Some(0));
        assert_eq!(decoded.roots, vec![0]);
        assert_eq!(decoded.meshes[0][0].indices, vec![0, 1, 0]);
        assert_eq!(decoded.meshes[0][0].vertices[1].pos, [-1.5, 0.0, 4.25]);
        let m = &decoded.materials[0];
        assert_eq!(m.base_color, Some(0));
        assert_eq!(m.normal, Some(1));
        assert_eq!(m.alpha_mode, AlphaMode::Mask);
        assert_eq!(m.kind, MaterialKind::Masked);
        assert!(matches!(
            decoded.images[0],
            TexData::Bc {
                format: BcFormat::Bc1,
                ..
            }
        ));
    }

    #[test]
    fn cook_is_deterministic() {
        let scene = sample_scene();
        assert_eq!(write_scene(&scene, 7), write_scene(&scene, 7));
    }
}
