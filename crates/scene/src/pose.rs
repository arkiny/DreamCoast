//! Animation **poses**: the buffer between sampling a clip and writing the ECS.
//!
//! [`crate::animation`] used to sample a clip straight into [`LocalTransform`], which
//! makes exactly one thing possible — play one clip. Blending needs an intermediate
//! value, so sampling now produces an [`AnimPose`] (a set of per-entity [`Trs`] +
//! morph weights) that can be [`blend_poses`]-ed with another pose before
//! [`apply_pose`] commits it. `sample -> blend -> apply` is the whole contract; blend
//! trees, IK, retargeting and root motion are deliberately out of scope.
//!
//! **Partial channels are first class.** A glTF clip animates a *subset* of each
//! node's channels (a rotation-only clip is the norm for skeletons), and a pose that
//! carried a full TRS would stomp the translation/scale the rig authored in its bind
//! pose. So [`Trs`]'s three fields are `Option`: `None` means "this clip says nothing
//! about this channel", and both blending and application skip it. That is also what
//! keeps a crossfade between clips animating *different* channel subsets from
//! snapping — see [`blend_poses`].
//!
//! **Order is deterministic.** Entries are stored in a `Vec` in first-touch order
//! (channel order, for a sampled clip) and only ever appended; the `HashMap` beside
//! it is a lookup accelerator that is never iterated. Application order therefore
//! reproduces exactly, which the headless capture sequences depend on.

use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::animation::MorphWeights;
use crate::ecs::{Entity, World};
use crate::transform::LocalTransform;

/// A pose's value for one entity: each channel is `Some` only if some clip authored
/// it. `None` channels are left untouched by [`apply_pose`] and pass through
/// [`blend_poses`] unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Trs {
    pub translation: Option<Vec3>,
    pub rotation: Option<Quat>,
    pub scale: Option<Vec3>,
}

impl Trs {
    /// No channel authored — the neutral element of [`apply_pose`].
    pub const EMPTY: Self = Self {
        translation: None,
        rotation: None,
        scale: None,
    };

    /// All three channels authored.
    pub const fn full(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation: Some(translation),
            rotation: Some(rotation),
            scale: Some(scale),
        }
    }

    /// The entity's current local transform as a fully-authored pose value.
    pub const fn from_local(local: &LocalTransform) -> Self {
        Self::full(local.translation, local.rotation, local.scale)
    }

    /// Whether no channel is authored.
    pub const fn is_empty(&self) -> bool {
        self.translation.is_none() && self.rotation.is_none() && self.scale.is_none()
    }

    /// Write the authored channels into `local`, leaving the others as they were.
    #[inline]
    pub fn apply_to(&self, local: &mut LocalTransform) {
        if let Some(t) = self.translation {
            local.translation = t;
        }
        if let Some(r) = self.rotation {
            local.rotation = r;
        }
        if let Some(s) = self.scale {
            local.scale = s;
        }
    }

    /// `base` with the authored channels overridden (the non-mutating [`Self::apply_to`]).
    pub fn resolve(&self, base: LocalTransform) -> LocalTransform {
        let mut out = base;
        self.apply_to(&mut out);
        out
    }
}

/// One entity's slot in an [`AnimPose`]: its sampled TRS channels plus, for a mesh
/// node driven by a morph-weight channel, the sampled weight vector.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseEntry {
    /// The entity the channels target.
    pub target: Entity,
    /// The authored TRS channels.
    pub trs: Trs,
    /// Sampled morph-target weights, or `None` if no weight channel targets this entity.
    pub weights: Option<Vec<f32>>,
}

/// A sampled animation pose: per-entity [`Trs`] (+ morph weights) in a deterministic
/// order.
///
/// Reuse one across frames via [`AnimPose::clear`] +
/// [`crate::sample_clip_into`] to keep sampling allocation-free in steady state.
#[derive(Clone, Debug, Default)]
pub struct AnimPose {
    entries: Vec<PoseEntry>,
    /// `target -> index into entries`. Lookup only — never iterated, so it cannot
    /// leak `HashMap` ordering into the pose.
    index: HashMap<Entity, usize>,
}

impl PartialEq for AnimPose {
    /// Poses are equal when their entries are equal *and in the same order* — the
    /// index is a derived accelerator, so it is not compared.
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl AnimPose {
    /// An empty pose.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every entry, keeping the allocations for reuse.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
    }

    /// Number of entities in the pose.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pose touches no entity.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, in application order.
    pub fn entries(&self) -> &[PoseEntry] {
        &self.entries
    }

    /// This pose's entry for `target`, if any.
    pub fn get(&self, target: Entity) -> Option<&PoseEntry> {
        self.index.get(&target).map(|&i| &self.entries[i])
    }

    /// The entry for `target`, appending an empty one (at the end, preserving order)
    /// if this is its first channel.
    fn entry_mut(&mut self, target: Entity) -> &mut PoseEntry {
        let i = match self.index.get(&target) {
            Some(&i) => i,
            None => {
                let i = self.entries.len();
                self.entries.push(PoseEntry {
                    target,
                    trs: Trs::EMPTY,
                    weights: None,
                });
                self.index.insert(target, i);
                i
            }
        };
        &mut self.entries[i]
    }

    /// Author `target`'s translation channel.
    pub fn set_translation(&mut self, target: Entity, translation: Vec3) {
        self.entry_mut(target).trs.translation = Some(translation);
    }

    /// Author `target`'s rotation channel.
    pub fn set_rotation(&mut self, target: Entity, rotation: Quat) {
        self.entry_mut(target).trs.rotation = Some(rotation);
    }

    /// Author `target`'s scale channel.
    pub fn set_scale(&mut self, target: Entity, scale: Vec3) {
        self.entry_mut(target).trs.scale = Some(scale);
    }

    /// Author `target`'s morph-target weights.
    pub fn set_weights(&mut self, target: Entity, weights: Vec<f32>) {
        self.entry_mut(target).weights = Some(weights);
    }

    /// Author the channels `trs` carries; channels it leaves `None` keep whatever the
    /// pose already held for `target`.
    pub fn set_trs(&mut self, target: Entity, trs: Trs) {
        let e = self.entry_mut(target);
        if trs.translation.is_some() {
            e.trs.translation = trs.translation;
        }
        if trs.rotation.is_some() {
            e.trs.rotation = trs.rotation;
        }
        if trs.scale.is_some() {
            e.trs.scale = trs.scale;
        }
    }
}

/// Write `pose` into the world: each entry's authored channels go to the target's
/// [`LocalTransform`] (targets without one are skipped) and its weights, if any, to
/// [`MorphWeights`].
///
/// Per-channel granularity is the point: a rotation-only pose leaves translation and
/// scale exactly as they were. Run [`crate::propagate_transforms`] afterwards.
pub fn apply_pose(world: &mut World, pose: &AnimPose) {
    for entry in &pose.entries {
        // Morph weights live in their own component; TRS goes to LocalTransform.
        if let Some(w) = &entry.weights {
            world.insert(entry.target, MorphWeights(w.clone()));
        }
        if !entry.trs.is_empty()
            && let Some(local) = world.get_mut::<LocalTransform>(entry.target)
        {
            entry.trs.apply_to(local);
        }
    }
}

/// Crossfade `a` towards `b` by `alpha` (0 = `a`, 1 = `b`): translation/scale/weights
/// lerp, rotation slerps along the shortest arc.
///
/// **Union rule.** A channel (or a whole entity) present in only one input is taken
/// from that input at full strength rather than being dropped or blended against a
/// default. Two clips animating different channel subsets — the normal case, e.g. an
/// upper-body attack fading over a locomotion clip — therefore never snap a channel to
/// identity mid-fade. The consequence to know: `blend_poses(a, b, 0.0)` equals `a`
/// only where both agree on the entity/channel set; `b`-only entries are present from
/// alpha 0 onwards (there is nothing in `a` to fade them from).
///
/// `alpha` is clamped to `[0, 1]`; NaN is treated as 0. The endpoints are *exact* (the
/// input value is copied, not run through an interpolator that only approximately
/// reproduces it).
pub fn blend_poses(a: &AnimPose, b: &AnimPose, alpha: f32) -> AnimPose {
    let mut out = AnimPose::new();
    blend_poses_into(a, b, alpha, &mut out);
    out
}

/// [`blend_poses`] into a reusable buffer (cleared first).
pub fn blend_poses_into(a: &AnimPose, b: &AnimPose, alpha: f32, out: &mut AnimPose) {
    out.clear();
    let s = if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    };

    // `a` order first, then the `b`-only entries in `b` order — deterministic given
    // deterministic inputs.
    for ea in &a.entries {
        let entry = match b.get(ea.target) {
            Some(eb) => blend_entry(ea, eb, s),
            None => ea.clone(),
        };
        push_entry(out, entry);
    }
    for eb in &b.entries {
        if a.get(eb.target).is_none() {
            push_entry(out, eb.clone());
        }
    }
}

fn push_entry(out: &mut AnimPose, entry: PoseEntry) {
    out.index.insert(entry.target, out.entries.len());
    out.entries.push(entry);
}

fn blend_entry(a: &PoseEntry, b: &PoseEntry, s: f32) -> PoseEntry {
    PoseEntry {
        target: a.target,
        trs: Trs {
            translation: blend_opt(a.trs.translation, b.trs.translation, s, lerp_vec3),
            rotation: blend_opt(a.trs.rotation, b.trs.rotation, s, slerp_shortest),
            scale: blend_opt(a.trs.scale, b.trs.scale, s, lerp_vec3),
        },
        weights: match (&a.weights, &b.weights) {
            (Some(wa), Some(wb)) => Some(blend_weights(wa, wb, s)),
            (Some(w), None) | (None, Some(w)) => Some(w.clone()),
            (None, None) => None,
        },
    }
}

/// The union rule for one channel: blend when both author it, else take the one that does.
#[inline]
fn blend_opt<T: Copy>(a: Option<T>, b: Option<T>, s: f32, f: fn(T, T, f32) -> T) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b, s)),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

/// Element-wise weight blend, union rule per element (clips whose weight vectors
/// differ in length keep the longer one's tail rather than truncating to zero).
fn blend_weights(a: &[f32], b: &[f32], s: f32) -> Vec<f32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| match (a.get(i), b.get(i)) {
            (Some(&x), Some(&y)) => lerp_f32(x, y, s),
            (Some(&v), None) | (None, Some(&v)) => v,
            (None, None) => 0.0, // unreachable: n = max(len)
        })
        .collect()
}

#[inline]
fn lerp_f32(a: f32, b: f32, s: f32) -> f32 {
    if s <= 0.0 {
        a
    } else if s >= 1.0 {
        b
    } else {
        a + (b - a) * s
    }
}

#[inline]
fn lerp_vec3(a: Vec3, b: Vec3, s: f32) -> Vec3 {
    if s <= 0.0 {
        a
    } else if s >= 1.0 {
        b
    } else {
        a.lerp(b, s)
    }
}

/// Slerp along the shortest arc: `q` and `-q` are the same rotation but slerping to
/// the far representative takes the long way round, so negate `b` when the dot is
/// negative (hemisphere correction).
#[inline]
fn slerp_shortest(a: Quat, b: Quat, s: f32) -> Quat {
    if s <= 0.0 {
        return a;
    }
    if s >= 1.0 {
        return b;
    }
    if a.dot(b) < 0.0 {
        a.slerp(-b, s)
    } else {
        a.slerp(b, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with_entity() -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, LocalTransform::IDENTITY);
        (w, e)
    }

    #[test]
    fn apply_pose_touches_only_authored_channels() {
        let (mut w, e) = world_with_entity();
        w.insert(
            e,
            LocalTransform {
                translation: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(4.0),
            },
        );
        // A rotation-only pose must not stomp translation or scale.
        let mut pose = AnimPose::new();
        let r = Quat::from_rotation_y(0.5);
        pose.set_rotation(e, r);
        apply_pose(&mut w, &pose);
        let lt = *w.get::<LocalTransform>(e).unwrap();
        assert_eq!(lt.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(lt.scale, Vec3::splat(4.0));
        assert_eq!(lt.rotation, r);
    }

    #[test]
    fn apply_pose_skips_entities_without_local_transform() {
        let mut w = World::new();
        let e = w.spawn(); // no LocalTransform
        let mut pose = AnimPose::new();
        pose.set_translation(e, Vec3::X);
        pose.set_weights(e, vec![0.25, 0.75]);
        apply_pose(&mut w, &pose);
        assert!(w.get::<LocalTransform>(e).is_none());
        // Weights are a separate component and are still written.
        assert_eq!(w.get::<MorphWeights>(e).unwrap().0, vec![0.25, 0.75]);
    }

    #[test]
    fn pose_order_is_first_touch_and_stable() {
        let mut w = World::new();
        let (a, b, c) = (w.spawn(), w.spawn(), w.spawn());
        let build = || {
            let mut p = AnimPose::new();
            p.set_rotation(c, Quat::IDENTITY);
            p.set_translation(a, Vec3::X);
            p.set_scale(c, Vec3::ONE); // second touch: no new entry, no reorder
            p.set_translation(b, Vec3::Y);
            p
        };
        let targets: Vec<Entity> = build().entries().iter().map(|e| e.target).collect();
        assert_eq!(targets, vec![c, a, b]);
        // Same construction sequence -> same order, every time.
        for _ in 0..8 {
            let again: Vec<Entity> = build().entries().iter().map(|e| e.target).collect();
            assert_eq!(again, targets);
        }
    }

    #[test]
    fn blend_endpoints_are_exact() {
        let mut w = World::new();
        let e = w.spawn();
        let (mut a, mut b) = (AnimPose::new(), AnimPose::new());
        let (ta, tb) = (Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.7, -1.5, 9.25));
        let (ra, rb) = (Quat::from_rotation_x(0.3), Quat::from_rotation_z(1.9));
        a.set_translation(e, ta);
        a.set_rotation(e, ra);
        a.set_scale(e, Vec3::splat(0.5));
        a.set_weights(e, vec![0.1, 0.9]);
        b.set_translation(e, tb);
        b.set_rotation(e, rb);
        b.set_scale(e, Vec3::splat(2.0));
        b.set_weights(e, vec![0.4, 0.6]);

        assert_eq!(blend_poses(&a, &b, 0.0), a);
        assert_eq!(blend_poses(&a, &b, 1.0), b);
        // Out-of-range alpha clamps to the endpoints; NaN behaves as 0.
        assert_eq!(blend_poses(&a, &b, -3.0), a);
        assert_eq!(blend_poses(&a, &b, 4.0), b);
        assert_eq!(blend_poses(&a, &b, f32::NAN), a);
    }

    #[test]
    fn blend_midpoint_interpolates_each_channel() {
        let mut w = World::new();
        let e = w.spawn();
        let (mut a, mut b) = (AnimPose::new(), AnimPose::new());
        a.set_translation(e, Vec3::ZERO);
        a.set_scale(e, Vec3::splat(1.0));
        a.set_weights(e, vec![0.0, 1.0]);
        b.set_translation(e, Vec3::new(10.0, 0.0, 0.0));
        b.set_scale(e, Vec3::splat(3.0));
        b.set_weights(e, vec![1.0, 0.0]);
        let m = blend_poses(&a, &b, 0.5);
        let entry = m.get(e).unwrap();
        assert_eq!(entry.trs.translation.unwrap(), Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(entry.trs.scale.unwrap(), Vec3::splat(2.0));
        assert_eq!(entry.weights.as_deref().unwrap(), &[0.5, 0.5]);
        assert!(entry.trs.rotation.is_none());
    }

    #[test]
    fn blend_rotation_takes_shortest_arc() {
        let mut w = World::new();
        let e = w.spawn();
        let qa = Quat::from_rotation_y(0.4);
        // Same rotation as `qa` rotated a little further, but expressed as its negated
        // representative -> dot(qa, qb) < 0 and a naive slerp would sweep ~2π the wrong way.
        let target = Quat::from_rotation_y(0.8);
        let qb = -target;
        assert!(qa.dot(qb) < 0.0, "test needs a negative dot");

        let (mut a, mut b) = (AnimPose::new(), AnimPose::new());
        a.set_rotation(e, qa);
        b.set_rotation(e, qb);
        let mid = blend_poses(&a, &b, 0.5)
            .get(e)
            .unwrap()
            .trs
            .rotation
            .unwrap();
        let expected = Quat::from_rotation_y(0.6);
        // The corrected result is the short-arc midpoint (up to quaternion double cover).
        assert!(
            mid.dot(expected).abs() > 0.9999,
            "short-arc midpoint, got {mid:?}"
        );
        // ... and not what an *uncorrected* interpolation gives. At s = 0.5 slerp and
        // nlerp agree up to normalisation, so `normalize(qa + qb)` is exactly the
        // hemisphere-blind midpoint: a rotation ~145° away from the intended one.
        let naive = Quat::from_xyzw(qa.x + qb.x, qa.y + qb.y, qa.z + qb.z, qa.w + qb.w).normalize();
        assert!(
            naive.dot(expected).abs() < 0.5,
            "uncorrected midpoint must differ, got {naive:?}"
        );
    }

    #[test]
    fn blend_union_keeps_channels_present_in_one_pose() {
        let mut w = World::new();
        let (shared, only_a, only_b) = (w.spawn(), w.spawn(), w.spawn());
        let (mut a, mut b) = (AnimPose::new(), AnimPose::new());
        // Shared entity, disjoint channels: translation only in `a`, scale only in `b`.
        a.set_translation(shared, Vec3::new(4.0, 0.0, 0.0));
        b.set_scale(shared, Vec3::splat(7.0));
        a.set_translation(only_a, Vec3::Y);
        b.set_translation(only_b, Vec3::Z);

        let m = blend_poses(&a, &b, 0.25);
        // Channel present in one pose only -> taken at full strength (no snap to identity).
        let e = m.get(shared).unwrap();
        assert_eq!(e.trs.translation.unwrap(), Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(e.trs.scale.unwrap(), Vec3::splat(7.0));
        assert_eq!(m.get(only_a).unwrap().trs.translation.unwrap(), Vec3::Y);
        assert_eq!(m.get(only_b).unwrap().trs.translation.unwrap(), Vec3::Z);
        // Order: `a`'s entries, then the `b`-only ones.
        let targets: Vec<Entity> = m.entries().iter().map(|e| e.target).collect();
        assert_eq!(targets, vec![shared, only_a, only_b]);
    }

    #[test]
    fn blended_pose_applies_per_channel() {
        let (mut w, e) = world_with_entity();
        w.insert(
            e,
            LocalTransform {
                translation: Vec3::new(0.0, 5.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(2.0),
            },
        );
        // Both clips animate rotation only: the bind translation/scale survive the fade.
        let (mut a, mut b) = (AnimPose::new(), AnimPose::new());
        a.set_rotation(e, Quat::from_rotation_y(0.0));
        b.set_rotation(e, Quat::from_rotation_y(1.0));
        apply_pose(&mut w, &blend_poses(&a, &b, 0.5));
        let lt = *w.get::<LocalTransform>(e).unwrap();
        assert_eq!(lt.translation, Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(lt.scale, Vec3::splat(2.0));
        assert!(lt.rotation.dot(Quat::from_rotation_y(0.5)).abs() > 0.9999);
    }

    #[test]
    fn trs_resolve_overrides_only_authored_channels() {
        let base = LocalTransform {
            translation: Vec3::new(1.0, 1.0, 1.0),
            rotation: Quat::from_rotation_x(0.2),
            scale: Vec3::splat(3.0),
        };
        let trs = Trs {
            scale: Some(Vec3::splat(9.0)),
            ..Trs::EMPTY
        };
        let out = trs.resolve(base);
        assert_eq!(out.translation, base.translation);
        assert_eq!(out.rotation, base.rotation);
        assert_eq!(out.scale, Vec3::splat(9.0));
        assert!(Trs::EMPTY.is_empty());
        assert!(!Trs::from_local(&base).is_empty());
    }
}
