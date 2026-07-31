//! glTF node-TRS animation playback on the ECS.
//!
//! An [`AnimationClip`] is a parsed glTF animation with its channels resolved to the
//! ECS entities that [`crate::instantiate_gltf_mapped`] created for the targeted
//! nodes (or one built in code with [`ClipBuilder`]).
//!
//! Evaluation is split in two, so clips can be **blended** and not just played:
//!
//! 1. [`sample_clip`] / [`sample_clip_into`] evaluate a clip at a time into an
//!    [`AnimPose`] — pure math, no world access.
//! 2. [`crate::blend_poses`] crossfades two poses, and [`apply_pose`] commits one to
//!    the targeted entities' [`crate::LocalTransform`] / [`MorphWeights`].
//!
//! An [`AnimationPlayer`] component holds a clip + a playback clock;
//! [`advance_animation`] is the [`crate::advance_spin`] analogue and is now exactly
//! `advance clock -> sample_clip -> apply_pose` for every player. Run
//! [`crate::propagate_transforms`] afterwards to push the new locals out to
//! `WorldTransform`.
//!
//! Pure CPU, deterministic given the same `dt` sequence (the engine drives it from
//! the fixed-timestep accumulator), so headless capture sequences reproduce exactly.

use dreamcoast_asset::{ChannelData, GltfAnimation, Interpolation};
use glam::{Quat, Vec3};

use crate::ecs::{Entity, World};
use crate::pose::{AnimPose, apply_pose};

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

    /// Start building a clip in code — see [`ClipBuilder`].
    pub fn builder() -> ClipBuilder {
        ClipBuilder::new()
    }

    /// Number of channels (one per animated node × path).
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

/// Builds an [`AnimationClip`] from keyframes authored in code, for gameplay clips
/// that have no glTF behind them (and for tests). Channels are evaluated in the order
/// they are added, which is the order the resulting [`AnimPose`] lists them in.
///
/// ```
/// # use dreamcoast_scene::{AnimationClip, Interpolation, LoopMode, World, sample_clip};
/// # use glam::Quat;
/// # let mut world = World::new();
/// # let joint = world.spawn();
/// let q = Quat::from_rotation_y(1.0);
/// let clip = AnimationClip::builder()
///     .rotation(joint, Interpolation::Linear, &[0.0, 0.5], &[Quat::IDENTITY, q])
///     .build();
/// assert_eq!(clip.duration, 0.5);
/// let pose = sample_clip(&clip, 0.5, LoopMode::Clamp);
/// assert_eq!(pose.get(joint).unwrap().trs.rotation, Some(q));
/// ```
#[derive(Default)]
pub struct ClipBuilder {
    channels: Vec<Channel>,
    duration: Option<f32>,
}

impl ClipBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a translation channel targeting `target`.
    pub fn translation(
        self,
        target: Entity,
        interpolation: Interpolation,
        times: &[f32],
        values: &[Vec3],
    ) -> Self {
        debug_assert_key_count(times.len(), values.len(), 1, interpolation);
        self.push(
            target,
            interpolation,
            times,
            Track::Translation(values.to_vec()),
        )
    }

    /// Add a rotation channel targeting `target`.
    pub fn rotation(
        self,
        target: Entity,
        interpolation: Interpolation,
        times: &[f32],
        values: &[Quat],
    ) -> Self {
        debug_assert_key_count(times.len(), values.len(), 1, interpolation);
        self.push(
            target,
            interpolation,
            times,
            Track::Rotation(values.to_vec()),
        )
    }

    /// Add a scale channel targeting `target`.
    pub fn scale(
        self,
        target: Entity,
        interpolation: Interpolation,
        times: &[f32],
        values: &[Vec3],
    ) -> Self {
        debug_assert_key_count(times.len(), values.len(), 1, interpolation);
        self.push(target, interpolation, times, Track::Scale(values.to_vec()))
    }

    /// Add a morph-weight channel targeting `target`: `values` is `num_targets`
    /// weights per keyframe, flattened (×3 for `CubicSpline` tangents).
    pub fn weights(
        self,
        target: Entity,
        interpolation: Interpolation,
        times: &[f32],
        num_targets: usize,
        values: &[f32],
    ) -> Self {
        debug_assert_key_count(times.len(), values.len(), num_targets, interpolation);
        self.push(
            target,
            interpolation,
            times,
            Track::Weights {
                num_targets,
                values: values.to_vec(),
            },
        )
    }

    /// Override the clip duration (default: the largest keyframe time).
    pub fn duration(mut self, seconds: f32) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Finish the clip.
    pub fn build(self) -> AnimationClip {
        let duration = self.duration.unwrap_or_else(|| {
            self.channels
                .iter()
                .filter_map(|c| c.times.last().copied())
                .fold(0.0f32, f32::max)
        });
        AnimationClip {
            channels: self.channels,
            duration,
        }
    }

    fn push(
        mut self,
        target: Entity,
        interpolation: Interpolation,
        times: &[f32],
        track: Track,
    ) -> Self {
        self.channels.push(Channel {
            target,
            interpolation,
            times: times.to_vec(),
            track,
        });
        self
    }
}

/// Debug-only shape check: `CubicSpline` stores `[in-tangent, value, out-tangent]`
/// per key, the other modes one value per key.
#[inline]
fn debug_assert_key_count(keys: usize, values: usize, per_key: usize, interp: Interpolation) {
    let stride = if interp == Interpolation::CubicSpline {
        3
    } else {
        1
    };
    debug_assert_eq!(
        values,
        keys * per_key * stride,
        "channel value count must be keys x {per_key} x {stride}"
    );
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

/// How a sample time outside the clip is interpreted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    /// Wrap into `[0, duration)` — what [`AnimationPlayer`] does (a locomotion clip).
    #[default]
    Loop,
    /// Clamp to `[0, duration]`: before the start holds the first keyframe, after the
    /// end holds the last (a one-shot attack clip parked on its final pose).
    Clamp,
}

/// Map a raw playback time onto the clip under `mode`.
///
/// A zero/negative duration (or a non-finite time — unreachable from
/// [`advance_animation`], whose clock starts at 0 and advances by a finite `dt`)
/// samples time 0 rather than propagating NaN into every transform.
fn normalize_time(time: f32, duration: f32, mode: LoopMode) -> f32 {
    if duration > 0.0 && time.is_finite() {
        match mode {
            LoopMode::Loop => time.rem_euclid(duration),
            LoopMode::Clamp => time.clamp(0.0, duration),
        }
    } else {
        0.0
    }
}

/// Evaluate `clip` at `time` into a fresh [`AnimPose`].
///
/// The channels already carry their target entities (resolved by
/// [`AnimationClip::from_gltf`] or given to [`ClipBuilder`]), so no binding argument
/// is needed. Use [`sample_clip_into`] on a hot path to reuse the pose's allocations.
pub fn sample_clip(clip: &AnimationClip, time: f32, mode: LoopMode) -> AnimPose {
    let mut pose = AnimPose::new();
    sample_clip_into(clip, time, mode, &mut pose);
    pose
}

/// [`sample_clip`] into a reusable pose buffer (cleared first).
///
/// Entries land in channel order (first touch wins the slot), so a clip's pose order
/// is fixed by the clip, not by the sampling call.
pub fn sample_clip_into(clip: &AnimationClip, time: f32, mode: LoopMode, pose: &mut AnimPose) {
    pose.clear();
    let t = normalize_time(time, clip.duration, mode);
    for ch in &clip.channels {
        sample_channel_into(ch, t, pose);
    }
}

/// Advance every [`AnimationPlayer`] by `dt` (looping over the clip duration), sample
/// its channels, and write the results into the targeted entities'
/// [`crate::LocalTransform`] / [`MorphWeights`].
///
/// Two passes (like [`crate::advance_spin`]): read the players to compute the new
/// clocks + sampled poses, then apply — so no player-storage borrow is held across
/// the write-back.
pub fn advance_animation(world: &mut World, dt: f32) {
    struct Update {
        player: Entity,
        new_time: f32,
        pose: AnimPose,
    }

    let updates: Vec<Update> = world
        .iter::<AnimationPlayer>()
        .map(|(e, p)| {
            let new_time = normalize_time(p.time + p.speed * dt, p.clip.duration, LoopMode::Loop);
            // `new_time` is already in `[0, duration)`, so the re-normalisation inside
            // `sample_clip` is the identity (`x.rem_euclid(d) == x` exactly for
            // `0 <= x < d`) — the sampled values are bit-identical to sampling it raw.
            Update {
                player: e,
                new_time,
                pose: sample_clip(&p.clip, new_time, LoopMode::Loop),
            }
        })
        .collect();

    for u in updates {
        if let Some(p) = world.get_mut::<AnimationPlayer>(u.player) {
            p.time = u.new_time;
        }
        apply_pose(world, &u.pose);
    }
}

/// Sample one channel at time `t` into `pose` (no-op if the track has no keyframes).
fn sample_channel_into(ch: &Channel, t: f32, pose: &mut AnimPose) {
    match &ch.track {
        Track::Translation(v) => {
            if let Some(x) = sample_vec3(&ch.times, v, ch.interpolation, t) {
                pose.set_translation(ch.target, x);
            }
        }
        Track::Scale(v) => {
            if let Some(x) = sample_vec3(&ch.times, v, ch.interpolation, t) {
                pose.set_scale(ch.target, x);
            }
        }
        Track::Rotation(v) => {
            if let Some(x) = sample_quat(&ch.times, v, ch.interpolation, t) {
                pose.set_rotation(ch.target, x);
            }
        }
        Track::Weights {
            num_targets,
            values,
        } => {
            if let Some(x) = sample_weights(&ch.times, values, *num_targets, ch.interpolation, t) {
                pose.set_weights(ch.target, x);
            }
        }
    }
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

    use crate::pose::{Trs, blend_poses};
    use crate::transform::{LocalTransform, propagate_transforms};

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

    // ---- sample / blend / apply split -------------------------------------------------

    /// A clip exercising every interpolation mode and every channel path:
    /// cubic-spline translation + rotation, linear scale, step morph weights on `a`,
    /// and a rotation-only channel on `b` (the partial-channel case).
    fn kitchen_sink_clip(a: Entity, b: Entity) -> AnimationClip {
        let cubic_t = [
            // key 0: [in-tangent, value, out-tangent]
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(3.0, 0.0, 0.0),
            // key 1
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(4.0, 1.0, 0.0),
            Vec3::ZERO,
        ];
        let cubic_r = [
            Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            Quat::from_rotation_y(0.1),
            Quat::from_xyzw(0.2, 0.0, 0.1, 0.0),
            Quat::from_xyzw(0.05, 0.0, 0.0, 0.0),
            Quat::from_rotation_y(1.3),
            Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        ];
        AnimationClip::builder()
            .translation(a, Interpolation::CubicSpline, &[0.0, 0.8], &cubic_t)
            .rotation(a, Interpolation::CubicSpline, &[0.0, 0.8], &cubic_r)
            .scale(
                a,
                Interpolation::Linear,
                &[0.0, 0.8],
                &[Vec3::ONE, Vec3::splat(2.5)],
            )
            .weights(
                a,
                Interpolation::Step,
                &[0.0, 0.4, 0.8],
                2,
                &[0.0, 1.0, 0.25, 0.75, 1.0, 0.0],
            )
            .rotation(
                b,
                Interpolation::Linear,
                &[0.0, 0.8],
                &[Quat::IDENTITY, Quat::from_rotation_z(0.9)],
            )
            .build()
    }

    /// The pre-split evaluation path, reproduced verbatim: advance the clock, then
    /// sample each channel in channel order and write it straight to the target.
    /// `advance_animation` must stay bit-identical to this.
    fn legacy_advance(world: &mut World, clip: &AnimationClip, time: &mut f32, dt: f32) {
        let new_time = if clip.duration > 0.0 {
            (*time + 1.0 * dt).rem_euclid(clip.duration)
        } else {
            0.0
        };
        *time = new_time;
        for ch in &clip.channels {
            match &ch.track {
                Track::Translation(v) => {
                    if let Some(x) = sample_vec3(&ch.times, v, ch.interpolation, new_time)
                        && let Some(lt) = world.get_mut::<LocalTransform>(ch.target)
                    {
                        lt.translation = x;
                    }
                }
                Track::Scale(v) => {
                    if let Some(x) = sample_vec3(&ch.times, v, ch.interpolation, new_time)
                        && let Some(lt) = world.get_mut::<LocalTransform>(ch.target)
                    {
                        lt.scale = x;
                    }
                }
                Track::Rotation(v) => {
                    if let Some(x) = sample_quat(&ch.times, v, ch.interpolation, new_time)
                        && let Some(lt) = world.get_mut::<LocalTransform>(ch.target)
                    {
                        lt.rotation = x;
                    }
                }
                Track::Weights {
                    num_targets,
                    values,
                } => {
                    if let Some(x) =
                        sample_weights(&ch.times, values, *num_targets, ch.interpolation, new_time)
                    {
                        world.insert(ch.target, MorphWeights(x));
                    }
                }
            }
        }
    }

    /// Every float of an entity's animated state, as raw bits (so the comparison is
    /// exact: no epsilon, no `-0.0 == 0.0` slack).
    fn state_bits(world: &World, e: Entity) -> Vec<u32> {
        let lt = world.get::<LocalTransform>(e).unwrap();
        let mut bits: Vec<u32> = lt
            .translation
            .to_array()
            .into_iter()
            .chain(lt.rotation.to_array())
            .chain(lt.scale.to_array())
            .map(f32::to_bits)
            .collect();
        if let Some(w) = world.get::<MorphWeights>(e) {
            bits.extend(w.0.iter().copied().map(f32::to_bits));
        }
        bits
    }

    /// The refactor's safety net: the new `sample -> apply` path must reproduce the
    /// old direct-write path bit-for-bit, including the cubic-spline math.
    #[test]
    fn advance_animation_matches_pre_split_path_bit_for_bit() {
        // World A: driven by the real `advance_animation` (player component).
        let mut wa = World::new();
        let (a0, b0) = (wa.spawn(), wa.spawn());
        wa.insert(a0, LocalTransform::IDENTITY);
        wa.insert(b0, LocalTransform::IDENTITY);
        let player = wa.spawn();
        wa.insert(player, AnimationPlayer::new(kitchen_sink_clip(a0, b0)));

        // World B: same entities (same spawn order -> same ids), driven by the legacy path.
        let mut wb = World::new();
        let (a1, b1) = (wb.spawn(), wb.spawn());
        wb.insert(a1, LocalTransform::IDENTITY);
        wb.insert(b1, LocalTransform::IDENTITY);
        let clip = kitchen_sink_clip(a1, b1);
        let mut time = 0.0f32;

        // 120 steps = 2 s over a 0.8 s clip: covers both loop wraps and every segment.
        for step in 0..120 {
            advance_animation(&mut wa, 1.0 / 60.0);
            legacy_advance(&mut wb, &clip, &mut time, 1.0 / 60.0);
            assert_eq!(
                wa.get::<AnimationPlayer>(player).unwrap().time.to_bits(),
                time.to_bits(),
                "clock diverged at step {step}"
            );
            assert_eq!(
                state_bits(&wa, a0),
                state_bits(&wb, a1),
                "node a, step {step}"
            );
            assert_eq!(
                state_bits(&wa, b0),
                state_bits(&wb, b1),
                "node b, step {step}"
            );
        }
    }

    #[test]
    fn sample_clip_reproduces_advance_animation_state() {
        let mut w = World::new();
        let (a, b) = (w.spawn(), w.spawn());
        w.insert(a, LocalTransform::IDENTITY);
        w.insert(b, LocalTransform::IDENTITY);
        let player = w.spawn();
        w.insert(player, AnimationPlayer::new(kitchen_sink_clip(a, b)));
        for _ in 0..17 {
            advance_animation(&mut w, 1.0 / 60.0);
        }
        let t = w.get::<AnimationPlayer>(player).unwrap().time;

        // Same clip, same time, sampled + applied by hand: identical world state.
        let mut w2 = World::new();
        let (a2, b2) = (w2.spawn(), w2.spawn());
        w2.insert(a2, LocalTransform::IDENTITY);
        w2.insert(b2, LocalTransform::IDENTITY);
        let pose = sample_clip(&kitchen_sink_clip(a2, b2), t, LoopMode::Loop);
        apply_pose(&mut w2, &pose);
        assert_eq!(state_bits(&w, a), state_bits(&w2, a2));
        assert_eq!(state_bits(&w, b), state_bits(&w2, b2));
        // Pose order follows channel order, one entry per entity.
        let targets: Vec<Entity> = pose.entries().iter().map(|e| e.target).collect();
        assert_eq!(targets, vec![a2, b2]);
        // `b` is rotation-only: its other channels stay unauthored.
        let tb = pose.get(b2).unwrap().trs;
        assert!(tb.rotation.is_some() && tb.translation.is_none() && tb.scale.is_none());
    }

    #[test]
    fn sample_clip_into_reuses_the_buffer() {
        let mut w = World::new();
        let (a, b) = (w.spawn(), w.spawn());
        let clip = kitchen_sink_clip(a, b);
        let mut pose = AnimPose::new();
        sample_clip_into(&clip, 0.3, LoopMode::Loop, &mut pose);
        let first = pose.clone();
        // A second sample clears the buffer instead of appending duplicates.
        sample_clip_into(&clip, 0.3, LoopMode::Loop, &mut pose);
        assert_eq!(pose.len(), 2);
        assert_eq!(pose, first);
        assert_eq!(pose, sample_clip(&clip, 0.3, LoopMode::Loop));
    }

    #[test]
    fn loop_and_clamp_disagree_only_outside_the_clip() {
        let mut w = World::new();
        let e = w.spawn();
        let clip = AnimationClip::builder()
            .translation(
                e,
                Interpolation::Linear,
                &[0.0, 1.0],
                &[Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
            )
            .build();
        assert_eq!(clip.duration, 1.0);
        let x = |t: f32, m: LoopMode| {
            sample_clip(&clip, t, m)
                .get(e)
                .unwrap()
                .trs
                .translation
                .unwrap()
                .x
        };
        // Inside the clip the modes agree.
        assert_eq!(x(0.25, LoopMode::Loop), x(0.25, LoopMode::Clamp));
        // Past the end: loop wraps to the start, clamp holds the last key.
        assert_eq!(x(1.25, LoopMode::Loop), 2.5);
        assert_eq!(x(1.25, LoopMode::Clamp), 10.0);
        assert_eq!(x(1.0, LoopMode::Clamp), 10.0);
        // Exactly at the duration, loop wraps to 0.
        assert_eq!(x(1.0, LoopMode::Loop), 0.0);
        // Before the start: loop wraps from the end, clamp holds the first key.
        assert_eq!(x(-0.25, LoopMode::Loop), 7.5);
        assert_eq!(x(-0.25, LoopMode::Clamp), 0.0);
        // Degenerate inputs sample time 0 instead of poisoning the pose.
        assert_eq!(x(f32::NAN, LoopMode::Loop), 0.0);
        let zero_len = AnimationClip::builder()
            .translation(
                e,
                Interpolation::Linear,
                &[0.0],
                &[Vec3::new(4.0, 0.0, 0.0)],
            )
            .build();
        assert_eq!(zero_len.duration, 0.0);
        assert_eq!(
            sample_clip(&zero_len, 9.0, LoopMode::Loop)
                .get(e)
                .unwrap()
                .trs
                .translation
                .unwrap()
                .x,
            4.0
        );
    }

    #[test]
    fn crossfade_between_clips_with_different_channel_subsets() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(
            e,
            LocalTransform {
                translation: Vec3::new(0.0, 7.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(3.0),
            },
        );
        // Clip A rotates; clip B translates. Neither authors scale.
        let rot = AnimationClip::builder()
            .rotation(
                e,
                Interpolation::Linear,
                &[0.0, 1.0],
                &[Quat::IDENTITY, Quat::from_rotation_y(1.0)],
            )
            .build();
        let trn = AnimationClip::builder()
            .translation(
                e,
                Interpolation::Linear,
                &[0.0, 1.0],
                &[Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0)],
            )
            .build();
        let pose = blend_poses(
            &sample_clip(&rot, 0.5, LoopMode::Loop),
            &sample_clip(&trn, 0.5, LoopMode::Loop),
            0.5,
        );
        apply_pose(&mut w, &pose);
        let lt = *w.get::<LocalTransform>(e).unwrap();
        // Rotation from A and translation from B both land at full strength (union
        // rule) and the un-animated scale survives untouched.
        assert!(lt.rotation.dot(Quat::from_rotation_y(0.5)).abs() > 0.9999);
        assert_eq!(lt.translation, Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(lt.scale, Vec3::splat(3.0));
    }

    #[test]
    fn builder_clip_matches_the_gltf_import_of_the_same_keys() {
        let mut w = World::new();
        let e = w.spawn();
        let imported = AnimationClip::from_gltf(&translate_clip(), &[Some(e)]);
        let built = AnimationClip::builder()
            .translation(
                e,
                Interpolation::Linear,
                &[0.0, 1.0],
                &[Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
            )
            .duration(1.0)
            .build();
        assert_eq!(imported.channel_count(), built.channel_count());
        assert_eq!(imported.duration, built.duration);
        for t in [0.0, 0.1, 0.37, 0.9, 1.0] {
            assert_eq!(
                sample_clip(&imported, t, LoopMode::Clamp),
                sample_clip(&built, t, LoopMode::Clamp),
                "time {t}"
            );
        }
    }

    #[test]
    fn empty_and_channel_less_clips_sample_to_an_empty_pose() {
        let clip = AnimationClip::builder().build();
        assert!(clip.is_empty());
        assert!(sample_clip(&clip, 0.5, LoopMode::Loop).is_empty());
        // Applying an empty pose is a no-op, not a panic.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, LocalTransform::IDENTITY);
        apply_pose(&mut w, &AnimPose::new());
        assert_eq!(
            *w.get::<LocalTransform>(e).unwrap(),
            LocalTransform::IDENTITY
        );
        assert_eq!(Trs::default(), Trs::EMPTY);
    }
}
