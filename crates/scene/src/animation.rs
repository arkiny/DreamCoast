//! glTF node-TRS animation playback on the ECS.
//!
//! An [`AnimationClip`] is a parsed glTF animation with its channels resolved to the
//! ECS entities that [`crate::instantiate_gltf_mapped`] created for the targeted
//! nodes. An [`AnimationPlayer`] component holds a clip + a playback clock;
//! [`advance_animation`] is the [`crate::advance_spin`] analogue — it advances every
//! player by `dt`, samples each channel, and writes the result into the targeted
//! entities' [`LocalTransform`]. Run [`crate::propagate_transforms`] afterwards to
//! push the new locals out to `WorldTransform`.
//!
//! Pure CPU, deterministic given the same `dt` sequence (the engine drives it from
//! the fixed-timestep accumulator), so headless capture sequences reproduce exactly.

use dreamcoast_asset::{ChannelData, GltfAnimation, Interpolation};
use glam::{Quat, Vec3};

use crate::ecs::{Entity, World};
use crate::transform::LocalTransform;

/// A keyframe track: a node's translation / rotation / scale, or a mesh's
/// morph-target weights (`num_targets` values per keyframe, flattened).
enum Track {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
    Weights {
        num_targets: usize,
        values: Vec<f32>,
    },
}

/// The current morph-target weights of a mesh node, written by [`advance_animation`]
/// from a morph-weight channel (Stage C). The renderer blends the primitive's morph
/// targets by these (`vertex += Σ wᵢ · targetᵢ`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphWeights(pub Vec<f32>);

/// One animation channel resolved to a target entity.
struct Channel {
    target: Entity,
    interpolation: Interpolation,
    times: Vec<f32>,
    track: Track,
}

/// A playable animation clip: node-TRS channels resolved to ECS entities + its
/// duration in seconds.
pub struct AnimationClip {
    channels: Vec<Channel>,
    /// Clip length in seconds (largest keyframe time across channels).
    pub duration: f32,
}

impl AnimationClip {
    /// Resolve a parsed glTF animation against a node-index → entity map (from
    /// [`crate::instantiate_gltf_mapped`]). Channels whose target node was not
    /// instantiated are dropped.
    pub fn from_gltf(anim: &GltfAnimation, node_to_entity: &[Option<Entity>]) -> Self {
        let channels = anim
            .channels
            .iter()
            .filter_map(|ch| {
                let target = node_to_entity.get(ch.target_node).copied().flatten()?;
                let track = match &ch.data {
                    ChannelData::Translation(v) => {
                        Track::Translation(v.iter().map(|a| Vec3::from_array(*a)).collect())
                    }
                    ChannelData::Rotation(v) => {
                        Track::Rotation(v.iter().map(|a| Quat::from_array(*a)).collect())
                    }
                    ChannelData::Scale(v) => {
                        Track::Scale(v.iter().map(|a| Vec3::from_array(*a)).collect())
                    }
                    ChannelData::Weights(v) => {
                        // values = num_targets per keyframe (×3 for cubic-spline tangents).
                        let keys = ch.times.len().max(1);
                        let per_key = if ch.interpolation == Interpolation::CubicSpline {
                            v.len() / (3 * keys)
                        } else {
                            v.len() / keys
                        };
                        Track::Weights {
                            num_targets: per_key,
                            values: v.clone(),
                        }
                    }
                };
                Some(Channel {
                    target,
                    interpolation: ch.interpolation,
                    times: ch.times.clone(),
                    track,
                })
            })
            .collect();
        Self {
            channels,
            duration: anim.duration,
        }
    }

    /// Whether the clip has any resolved channels.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

/// Plays an [`AnimationClip`], looping. Attach to any entity; the clip's channels
/// target their own entities, so the player entity need not be one of them.
pub struct AnimationPlayer {
    clip: AnimationClip,
    /// Current playback time in seconds, in `[0, duration)`.
    pub time: f32,
    /// Playback rate multiplier (1.0 = real time).
    pub speed: f32,
}

impl AnimationPlayer {
    pub fn new(clip: AnimationClip) -> Self {
        Self {
            clip,
            time: 0.0,
            speed: 1.0,
        }
    }
}

/// One sampled TRS write to apply to a target entity.
struct TrsWrite {
    target: Entity,
    value: TrsValue,
}

enum TrsValue {
    Translation(Vec3),
    Rotation(Quat),
    Scale(Vec3),
    Weights(Vec<f32>),
}

/// Advance every [`AnimationPlayer`] by `dt` (looping over the clip duration), sample
/// its channels, and write the results into the targeted entities' [`LocalTransform`].
///
/// Two passes (like [`crate::advance_spin`]): read the players to compute the new
/// clocks + sampled writes, then apply — so no player-storage borrow is held across
/// the `LocalTransform` write-back.
pub fn advance_animation(world: &mut World, dt: f32) {
    struct Update {
        player: Entity,
        new_time: f32,
        writes: Vec<TrsWrite>,
    }

    let updates: Vec<Update> = world
        .iter::<AnimationPlayer>()
        .map(|(e, p)| {
            let new_time = if p.clip.duration > 0.0 {
                (p.time + p.speed * dt).rem_euclid(p.clip.duration)
            } else {
                0.0
            };
            let writes = p
                .clip
                .channels
                .iter()
                .filter_map(|ch| sample_channel(ch, new_time))
                .collect();
            Update {
                player: e,
                new_time,
                writes,
            }
        })
        .collect();

    for u in updates {
        if let Some(p) = world.get_mut::<AnimationPlayer>(u.player) {
            p.time = u.new_time;
        }
        for w in u.writes {
            // Morph weights live in their own component; TRS goes to LocalTransform.
            if let TrsValue::Weights(weights) = w.value {
                world.insert(w.target, MorphWeights(weights));
            } else if let Some(lt) = world.get_mut::<LocalTransform>(w.target) {
                match w.value {
                    TrsValue::Translation(t) => lt.translation = t,
                    TrsValue::Rotation(r) => lt.rotation = r,
                    TrsValue::Scale(s) => lt.scale = s,
                    TrsValue::Weights(_) => unreachable!(),
                }
            }
        }
    }
}

/// Sample one channel at time `t` into a [`TrsWrite`] (`None` if the track is empty).
fn sample_channel(ch: &Channel, t: f32) -> Option<TrsWrite> {
    let value = match &ch.track {
        Track::Translation(v) => {
            TrsValue::Translation(sample_vec3(&ch.times, v, ch.interpolation, t)?)
        }
        Track::Scale(v) => TrsValue::Scale(sample_vec3(&ch.times, v, ch.interpolation, t)?),
        Track::Rotation(v) => TrsValue::Rotation(sample_quat(&ch.times, v, ch.interpolation, t)?),
        Track::Weights {
            num_targets,
            values,
        } => TrsValue::Weights(sample_weights(
            &ch.times,
            values,
            *num_targets,
            ch.interpolation,
            t,
        )?),
    };
    Some(TrsWrite {
        target: ch.target,
        value,
    })
}

/// Sample a morph-weight track at time `t`: each of the `num_targets` weights is
/// interpolated independently (the output buffer is `num_targets`-major per key).
fn sample_weights(
    times: &[f32],
    values: &[f32],
    num_targets: usize,
    interp: Interpolation,
    t: f32,
) -> Option<Vec<f32>> {
    if num_targets == 0 {
        return Some(Vec::new());
    }
    let (i0, i1, s) = segment(times, t)?;
    // Value of weight `w` at key `k` (CubicSpline stores [in,val,out] per key → val at
    // the middle of each key's `3*num_targets` block).
    let val = |k: usize, w: usize| -> f32 {
        let base = match interp {
            Interpolation::CubicSpline => (3 * k + 1) * num_targets,
            _ => k * num_targets,
        };
        values[base + w]
    };
    Some(
        (0..num_targets)
            .map(|w| match interp {
                Interpolation::Step => val(i0, w),
                Interpolation::Linear => val(i0, w) + (val(i1, w) - val(i0, w)) * s,
                Interpolation::CubicSpline => {
                    if i0 == i1 {
                        val(i0, w)
                    } else {
                        let dt = times[i1] - times[i0];
                        let p0 = values[(3 * i0 + 1) * num_targets + w];
                        let m0 = values[(3 * i0 + 2) * num_targets + w] * dt;
                        let p1 = values[(3 * i1 + 1) * num_targets + w];
                        let m1 = values[(3 * i1) * num_targets + w] * dt;
                        let (s2, s3) = (s * s, s * s * s);
                        (2.0 * s3 - 3.0 * s2 + 1.0) * p0
                            + (s3 - 2.0 * s2 + s) * m0
                            + (-2.0 * s3 + 3.0 * s2) * p1
                            + (s3 - s2) * m1
                    }
                }
            })
            .collect(),
    )
}

/// Locate the keyframe segment for time `t`: returns `(i0, i1, s)` where `s` is the
/// normalized position in `[0, 1]` within `[times[i0], times[i1]]`. Before the first
/// / after the last key it clamps to that key (`i0 == i1`, `s == 0`).
fn segment(times: &[f32], t: f32) -> Option<(usize, usize, f32)> {
    let n = times.len();
    if n == 0 {
        return None;
    }
    if t <= times[0] {
        return Some((0, 0, 0.0));
    }
    if t >= times[n - 1] {
        return Some((n - 1, n - 1, 0.0));
    }
    let mut i = 0;
    while i + 1 < n && times[i + 1] <= t {
        i += 1;
    }
    let (t0, t1) = (times[i], times[i + 1]);
    let s = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    Some((i, i + 1, s))
}

/// The value keyframe index into the output buffer: `CubicSpline` outputs are laid
/// out `[in-tangent, value, out-tangent]` per key (stride 3), so the value is at
/// `3*k + 1`; the other modes store one value per key.
#[inline]
fn value_index(k: usize, interp: Interpolation) -> usize {
    match interp {
        Interpolation::CubicSpline => 3 * k + 1,
        _ => k,
    }
}

fn sample_vec3(times: &[f32], vals: &[Vec3], interp: Interpolation, t: f32) -> Option<Vec3> {
    let (i0, i1, s) = segment(times, t)?;
    let v = |k: usize| vals[value_index(k, interp)];
    Some(match interp {
        Interpolation::Step => v(i0),
        Interpolation::Linear => v(i0).lerp(v(i1), s),
        Interpolation::CubicSpline => {
            if i0 == i1 {
                v(i0)
            } else {
                let dt = times[i1] - times[i0];
                let p0 = vals[3 * i0 + 1];
                let m0 = vals[3 * i0 + 2] * dt; // out-tangent of i0
                let p1 = vals[3 * i1 + 1];
                let m1 = vals[3 * i1] * dt; // in-tangent of i1
                hermite_vec3(p0, m0, p1, m1, s)
            }
        }
    })
}

fn sample_quat(times: &[f32], vals: &[Quat], interp: Interpolation, t: f32) -> Option<Quat> {
    let (i0, i1, s) = segment(times, t)?;
    let v = |k: usize| vals[value_index(k, interp)];
    Some(match interp {
        Interpolation::Step => v(i0),
        Interpolation::Linear => v(i0).slerp(v(i1), s),
        Interpolation::CubicSpline => {
            if i0 == i1 {
                v(i0)
            } else {
                let dt = times[i1] - times[i0];
                let q0 = vals[3 * i0 + 1];
                let m0 = scale_quat(vals[3 * i0 + 2], dt);
                let q1 = vals[3 * i1 + 1];
                let m1 = scale_quat(vals[3 * i1], dt);
                hermite_quat(q0, m0, q1, m1, s)
            }
        }
    })
}

/// Cubic Hermite basis applied to a `Vec3` (`p0,p1` endpoints, `m0,m1` tangents).
fn hermite_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, s: f32) -> Vec3 {
    let (s2, s3) = (s * s, s * s * s);
    let (h00, h10, h01, h11) = (
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    );
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Per-component scale of a quaternion's `[x,y,z,w]` (for tangent scaling).
fn scale_quat(q: Quat, k: f32) -> Quat {
    Quat::from_xyzw(q.x * k, q.y * k, q.z * k, q.w * k)
}

/// Cubic Hermite on quaternion components, then renormalize (glTF's cubic-spline
/// rotation interpolation).
fn hermite_quat(q0: Quat, m0: Quat, q1: Quat, m1: Quat, s: f32) -> Quat {
    let (s2, s3) = (s * s, s * s * s);
    let (h00, h10, h01, h11) = (
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    );
    let x = h00 * q0.x + h10 * m0.x + h01 * q1.x + h11 * m1.x;
    let y = h00 * q0.y + h10 * m0.y + h01 * q1.y + h11 * m1.y;
    let z = h00 * q0.z + h10 * m0.z + h01 * q1.z + h11 * m1.z;
    let w = h00 * q0.w + h10 * m0.w + h01 * q1.w + h11 * m1.w;
    Quat::from_xyzw(x, y, z, w).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dreamcoast_asset::{ChannelData, GltfAnimation, GltfChannel};

    use crate::transform::propagate_transforms;

    // A 1-channel translation clip: x goes 0 -> 10 over 1s (linear).
    fn translate_clip() -> GltfAnimation {
        GltfAnimation {
            name: Some("t".into()),
            duration: 1.0,
            channels: vec![GltfChannel {
                target_node: 0,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                data: ChannelData::Translation(vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            }],
        }
    }

    #[test]
    fn linear_sampling_midpoint() {
        let v = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let times = [0.0, 1.0];
        assert_eq!(
            sample_vec3(&times, &v, Interpolation::Linear, 0.5).unwrap(),
            Vec3::new(5.0, 0.0, 0.0)
        );
        // Clamps before first / after last.
        assert_eq!(
            sample_vec3(&times, &v, Interpolation::Linear, -1.0).unwrap(),
            v[0]
        );
        assert_eq!(
            sample_vec3(&times, &v, Interpolation::Linear, 2.0).unwrap(),
            v[1]
        );
    }

    #[test]
    fn step_holds_previous() {
        let v = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let times = [0.0, 1.0];
        // At 0.999 the step value is still the first key.
        assert_eq!(
            sample_vec3(&times, &v, Interpolation::Step, 0.999).unwrap(),
            v[0]
        );
    }

    #[test]
    fn advance_drives_local_transform_and_loops_deterministically() {
        let build = || {
            let mut w = World::new();
            let e = w.spawn();
            w.insert(e, LocalTransform::IDENTITY);
            let clip = AnimationClip::from_gltf(&translate_clip(), &[Some(e)]);
            let player = w.spawn();
            w.insert(player, AnimationPlayer::new(clip));
            (w, e)
        };
        let step = |w: &mut World, e: Entity, n: usize| {
            for _ in 0..n {
                advance_animation(w, 1.0 / 60.0);
            }
            propagate_transforms(w);
            w.get::<LocalTransform>(e).unwrap().translation
        };
        // 30 steps = 0.5s -> x ~ 5.
        let (mut a, ea) = build();
        let xa = step(&mut a, ea, 30);
        assert!(
            (xa.x - 5.0).abs() < 1e-3,
            "midway translation, got {}",
            xa.x
        );
        // Determinism: same dt sequence -> identical result.
        let (mut b, eb) = build();
        let xb = step(&mut b, eb, 30);
        assert_eq!(xa, xb);
        // Looping: 90 steps = 1.5s wraps to 0.5s -> same x.
        let (mut c, ec) = build();
        let xc = step(&mut c, ec, 90);
        assert!(
            (xc.x - 5.0).abs() < 1e-3,
            "looped translation, got {}",
            xc.x
        );
    }
}
