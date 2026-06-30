//! Scene/level chunk codec (Phase 12 item 2). A single-chunk `.dcasset` describing a
//! whole scene; see [`crate::level`] for the data model.

use dreamcoast_core::EngineError;

use super::{CHUNK_LEVEL, Header, Reader, Writer, open_chunk, write_single_chunk};
use crate::level::{Camera, Entity, Environment, LevelData, Light, LightKind, MaterialOverride};

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

    Ok(LevelData {
        entities,
        lights,
        camera,
        environment,
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
                },
                Light {
                    kind: LightKind::Point,
                    vec: [1.0, 2.0, 3.0],
                    color: [0.5, 0.6, 1.0],
                    intensity: 8.0,
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
        };
        let bytes = write_level(&level, 0x1e7e1);
        let (header, decoded) = read_level(&bytes).expect("decode");
        assert_eq!(header.source_hash, 0x1e7e1);
        assert_eq!(decoded, level);
        // Deterministic.
        assert_eq!(write_level(&level, 0x1e7e1), bytes);
    }
}
