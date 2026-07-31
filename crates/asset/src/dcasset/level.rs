//! Scene/level chunk codec (Phase 12 item 2). A single-chunk `.dcasset` describing a
//! whole scene; see [`crate::level`] for the data model.

use dreamcoast_core::EngineError;

use super::{CHUNK_LEVEL, Header, Reader, Writer, open_chunk, write_single_chunk};
use crate::level::{
    Camera, DeformEntity, Entity, Environment, LevelData, Light, LightKind, MaterialOverride,
};

// Light kind tags stored in a level chunk.
const LIGHT_DIRECTIONAL: u32 = 0;
const LIGHT_POINT: u32 = 1;

/// Serialize a level/scene into a `.dcasset`. `src_hash` is the invalidation key
/// (e.g. a hash of the authored scene description).
pub fn write_level(level: &LevelData, src_hash: u64) -> Vec<u8> {
    write_single_chunk(CHUNK_LEVEL, &encode_level(level), src_hash)
}

/// Decode a `.dcasset` buffer's level chunk into its [`Header`] and [`LevelData`].
pub fn read_level(bytes: &[u8]) -> Result<(Header, LevelData), EngineError> {
    let (header, mut r) = open_chunk(bytes, CHUNK_LEVEL, "level")?;
    Ok((header, decode_level(&mut r)?))
}

fn encode_level(level: &LevelData) -> Vec<u8> {
    let mut w = Writer::default();
    // Entities.
    w.u32(level.entities.len() as u32);
    for e in &level.entities {
        w.str(&e.asset);
        for f in e.transform {
            w.f32(f);
        }
        match &e.material_override {
            Some(o) => {
                w.u32(1);
                for c in o.base_color_factor {
                    w.f32(c);
                }
                w.f32(o.metallic);
                w.f32(o.roughness);
            }
            None => w.u32(0),
        }
    }
    // Lights.
    w.u32(level.lights.len() as u32);
    for l in &level.lights {
        w.u32(match l.kind {
            LightKind::Directional => LIGHT_DIRECTIONAL,
            LightKind::Point => LIGHT_POINT,
        });
        for c in l.vec {
            w.f32(c);
        }
        for c in l.color {
            w.f32(c);
        }
        w.f32(l.intensity);
    }
    // Camera.
    let c = &level.camera;
    for v in c.position {
        w.f32(v);
    }
    for v in c.target {
        w.f32(v);
    }
    w.f32(c.fov_y_deg);
    w.f32(c.znear);
    w.f32(c.zfar);
    // Environment.
    let env = &level.environment;
    for v in env.sun_dir {
        w.f32(v);
    }
    w.f32(env.sun_intensity);
    for v in env.sky_white_balance {
        w.f32(v);
    }
    // Deforms (v9). Appended after the environment so a pre-v9 chunk is a strict prefix
    // of this layout.
    w.u32(level.deforms.len() as u32);
    for d in &level.deforms {
        w.str(&d.source);
        for f in d.transform {
            w.f32(f);
        }
        match &d.material_override {
            Some(o) => {
                w.u32(1);
                for c in o.base_color_factor {
                    w.f32(c);
                }
                w.f32(o.metallic);
                w.f32(o.roughness);
            }
            None => w.u32(0),
        }
    }
    // Entity names (v10). Appended as a trailing parallel block — not interleaved into
    // the entity records — so a pre-v10 chunk stays a strict prefix of this layout and
    // the same "read what is there, default the rest" decode works for both. Written
    // unconditionally (one length + one string per entity, empty = unnamed) so the
    // encoding is a pure function of the level, not of which entities happen to be named.
    w.u32(level.entities.len() as u32);
    for e in &level.entities {
        w.str(e.name.as_deref().unwrap_or(""));
    }
    // Light ranges (v11). Same trailing-parallel-block pattern as the entity names above,
    // for the same reason: a pre-v11 chunk stays a strict prefix of this layout. Written
    // unconditionally (one float per light, 0 = no cutoff) so the encoding is a pure
    // function of the level.
    w.u32(level.lights.len() as u32);
    for l in &level.lights {
        w.f32(l.range);
    }
    w.buf
}

fn decode_level(r: &mut Reader) -> Result<LevelData, EngineError> {
    let vec3 =
        |r: &mut Reader| -> Result<[f32; 3], EngineError> { Ok([r.f32()?, r.f32()?, r.f32()?]) };

    let entity_count = r.u32()?;
    let mut entities = Vec::with_capacity(entity_count as usize);
    for _ in 0..entity_count {
        let asset = r.str()?;
        let mut transform = [0.0f32; 16];
        for t in &mut transform {
            *t = r.f32()?;
        }
        let material_override = if r.u32()? != 0 {
            Some(MaterialOverride {
                base_color_factor: [r.f32()?, r.f32()?, r.f32()?, r.f32()?],
                metallic: r.f32()?,
                roughness: r.f32()?,
            })
        } else {
            None
        };
        entities.push(Entity {
            asset,
            // Filled from the v10 trailing name block below; a pre-v10 chunk has none,
            // which leaves every entity unnamed (the old behavior exactly).
            name: None,
            transform,
            material_override,
        });
    }

    let light_count = r.u32()?;
    let mut lights = Vec::with_capacity(light_count as usize);
    for _ in 0..light_count {
        let kind = match r.u32()? {
            LIGHT_DIRECTIONAL => LightKind::Directional,
            LIGHT_POINT => LightKind::Point,
            other => {
                return Err(EngineError::Asset(format!(
                    "dcasset: unknown light kind {other}"
                )));
            }
        };
        lights.push(Light {
            kind,
            vec: vec3(r)?,
            color: vec3(r)?,
            intensity: r.f32()?,
            // Filled from the v11 trailing range block below; a pre-v11 chunk has none,
            // which leaves every light at 0 = no cutoff (the old behavior exactly).
            range: 0.0,
        });
    }

    let camera = Camera {
        position: vec3(r)?,
        target: vec3(r)?,
        fov_y_deg: r.f32()?,
        znear: r.f32()?,
        zfar: r.f32()?,
    };
    let environment = Environment {
        sun_dir: vec3(r)?,
        sun_intensity: r.f32()?,
        sky_white_balance: vec3(r)?,
    };

    // Deforms (v9). A v8 chunk ends at `environment`; reading the count off its end yields EOF,
    // which we treat as zero deforms (so an old shipped `.dcasset` decodes cleanly). A live cook
    // always writes this block, and the VERSION gate re-cooks a stale cache from RON regardless.
    let deform_count = r.u32().unwrap_or(0);
    let mut deforms = Vec::with_capacity(deform_count as usize);
    for _ in 0..deform_count {
        let source = r.str()?;
        let mut transform = [0.0f32; 16];
        for t in &mut transform {
            *t = r.f32()?;
        }
        let material_override = if r.u32()? != 0 {
            Some(MaterialOverride {
                base_color_factor: [r.f32()?, r.f32()?, r.f32()?, r.f32()?],
                metallic: r.f32()?,
                roughness: r.f32()?,
            })
        } else {
            None
        };
        deforms.push(DeformEntity {
            source,
            transform,
            material_override,
        });
    }

    // Entity names (v10). A v9 chunk ends at the deform block, so the count read is EOF ⇒
    // zero names ⇒ every entity keeps the `None` set above (an old shipped `.dcasset`
    // decodes cleanly). The VERSION gate re-cooks a stale cache from RON regardless, so
    // this only matters for the source-absent shipped path. An empty string is the
    // encoding of "unnamed", not a name.
    let name_count = r.u32().unwrap_or(0);
    for i in 0..name_count as usize {
        let name = r.str()?;
        if let Some(e) = entities.get_mut(i)
            && !name.is_empty()
        {
            e.name = Some(name);
        }
    }

    // Light ranges (v11). A v10 chunk ends at the name block, so this count read is EOF ⇒
    // zero ranges ⇒ every light keeps the `0.0` (no cutoff) set above.
    let range_count = r.u32().unwrap_or(0);
    for i in 0..range_count as usize {
        let range = r.f32()?;
        if let Some(l) = lights.get_mut(i) {
            l.range = range;
        }
    }

    Ok(LevelData {
        entities,
        lights,
        camera,
        environment,
        deforms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_chunk_roundtrip() {
        let level = LevelData {
            entities: vec![
                Entity {
                    asset: "assets/model.glb".into(),
                    name: None,
                    transform: [
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0,
                        1.0,
                    ],
                    material_override: Some(MaterialOverride {
                        base_color_factor: [0.2, 0.4, 0.6, 1.0],
                        metallic: 0.3,
                        roughness: 0.7,
                    }),
                },
                Entity {
                    asset: "assets/sphere".into(),
                    name: Some("player".into()),
                    transform: [0.0; 16],
                    material_override: None,
                },
            ],
            lights: vec![
                Light {
                    kind: LightKind::Directional,
                    vec: [-0.4, -1.0, -0.3],
                    color: [1.0, 0.95, 0.9],
                    intensity: 3.0,
                    range: 0.0,
                },
                Light {
                    kind: LightKind::Point,
                    vec: [1.0, 2.0, 3.0],
                    color: [0.5, 0.6, 1.0],
                    intensity: 8.0,
                    range: 11.5,
                },
            ],
            camera: Camera {
                position: [0.0, 1.5, 4.0],
                target: [0.0, 0.5, 0.0],
                fov_y_deg: 50.0,
                znear: 0.05,
                zfar: 200.0,
            },
            environment: Environment {
                sun_dir: [-0.3, -0.9, -0.2],
                sun_intensity: 4.0,
                sky_white_balance: [1.0, 0.95, 0.9],
            },
            deforms: vec![
                DeformEntity {
                    source: "assets/Knight/knight.usda".into(),
                    transform: [
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.5, 0.0, 0.0,
                        1.0,
                    ],
                    material_override: Some(MaterialOverride {
                        base_color_factor: [0.58, 0.58, 0.6, 1.0],
                        metallic: 0.6,
                        roughness: 0.45,
                    }),
                },
                DeformEntity {
                    source: "assets/cloth.abc".into(),
                    transform: [0.0; 16],
                    material_override: None,
                },
            ],
        };
        let bytes = write_level(&level, 0x1e7e1);
        let (header, decoded) = read_level(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0x1e7e1);
        assert_eq!(decoded, level);
        assert_eq!(decoded.entities[1].name.as_deref(), Some("player"));
        // Deterministic.
        assert_eq!(write_level(&level, 0x1e7e1), bytes);
    }

    /// A **v9** chunk (written before entity names existed) still decodes: the trailing
    /// name block is simply absent, so every entity reads back unnamed and nothing else
    /// shifts. Built by lopping the v10 (names) and v11 (light ranges) blocks off an
    /// all-unnamed, light-free payload — with names `None` the name block is exactly
    /// `u32 count` + one empty `str` (a bare `u32 0`) per entity, and with no lights the
    /// range block is a bare `u32 0`, so the remainder *is* the v9 byte layout.
    #[test]
    fn pre_name_chunk_decodes_with_no_names() {
        let mut level = LevelData {
            entities: vec![
                Entity {
                    asset: "assets/model.glb".into(),
                    name: None,
                    transform: [0.0; 16],
                    material_override: None,
                },
                Entity {
                    asset: "sphere".into(),
                    name: None,
                    transform: [1.0; 16],
                    material_override: None,
                },
            ],
            ..LevelData::default()
        };
        let payload = encode_level(&level);
        assert!(level.lights.is_empty(), "range block must be a bare count");
        let v9_len = payload.len() - 4 - (4 + 4 * level.entities.len());
        let v9 = write_single_chunk(CHUNK_LEVEL, &payload[..v9_len], 0xbeef);

        let (_, decoded) = read_level(&v9).expect("decode a pre-name chunk");
        assert_eq!(decoded, level);

        // And the same level *with* a name is not silently equal to the v9 read — the
        // block is what carries it.
        level.entities[0].name = Some("door".into());
        let (_, named) = read_level(&write_level(&level, 0xbeef)).expect("decode");
        assert_eq!(named.entities[0].name.as_deref(), Some("door"));
        assert_eq!(named.entities[1].name, None);
    }

    /// A **v10** chunk (written before point-light `range`) still decodes: the trailing
    /// range block is absent, so every light reads back at `range = 0.0` — which the
    /// renderer reads as "no cutoff", the pre-range falloff exactly. Built by lopping the
    /// v11 block (a `u32 count` + one `f32` per light) off a v11 payload.
    #[test]
    fn pre_range_chunk_decodes_with_zero_ranges() {
        let level = LevelData {
            lights: vec![
                Light {
                    kind: LightKind::Directional,
                    vec: [0.0, -1.0, 0.0],
                    color: [1.0; 3],
                    intensity: 2.0,
                    range: 0.0,
                },
                Light {
                    kind: LightKind::Point,
                    vec: [1.0, 2.0, 3.0],
                    color: [1.0, 0.5, 0.25],
                    intensity: 8.0,
                    range: 9.5,
                },
            ],
            ..LevelData::default()
        };
        let payload = encode_level(&level);
        let v10_len = payload.len() - (4 + 4 * level.lights.len());
        let v10 = write_single_chunk(CHUNK_LEVEL, &payload[..v10_len], 0xbeef);
        let (_, decoded) = read_level(&v10).expect("decode a pre-range chunk");
        assert!(decoded.lights.iter().all(|l| l.range == 0.0));

        // And a v11 chunk round-trips the authored range — the block is what carries it.
        let (_, fresh) = read_level(&write_level(&level, 0xbeef)).expect("decode");
        assert_eq!(fresh.lights[1].range, 9.5);
    }
}
