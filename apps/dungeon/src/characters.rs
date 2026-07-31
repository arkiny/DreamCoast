//! Binding an authored rig ([`crate::rigs`]) to the ECS sub-tree the level loader built
//! for it, and turning the rig's clips into **per-instance** [`AnimationClip`]s
//! (`docs/game-framework-plan.md` §4.2).
//!
//! # The seam gap this module closes
//!
//! An [`AnimationClip`] does not target *nodes*, it targets **entities**: its channels
//! carry [`Entity`] ids, resolved once by [`AnimationClip::from_gltf`] against the
//! node → entity map [`instantiate_gltf_mapped`](sandbox::scene::instantiate_gltf_mapped)
//! returns. That is the right design — sampling is then a flat walk with no lookups —
//! but it means a clip belongs to **one instance**: six grunts need six copies of
//! `walk`, each bound to its own bones.
//!
//! The engine hands that map back at instantiation time... to whoever called
//! `instantiate_gltf_mapped`. For a level, that caller is the engine's own
//! `level::build_level`, deep inside `App::new`, and the [`GameHooks`](sandbox::GameHooks)
//! seam has no hook there: a game sees the world for the first time in `fixed_update`,
//! long after the map was dropped. **The node map is not reachable from the hooks side.**
//!
//! Rather than open a fourth hook (a new engine call-in, on the path the golden gates
//! guard, for information the world already contains), this resolves the map **from the
//! world**, by name:
//!
//! * a level entity's authored `name` lands on the placement wrapper the loader spawns
//!   (`crate::level::PLAYER_NAME`, `grunt_0`, …), which is how [`crate::game`] already
//!   finds the player;
//! * every glTF node becomes an entity carrying its node `Name`, and
//!   `rig_geometry_is_grounded_and_outward_facing` proves a rig's node names are
//!   **unique within the rig** — so inside one character's sub-tree, a name identifies
//!   exactly one bone.
//!
//! So [`RigBinding::resolve`] walks a character's sub-tree, indexes it by `Name`, and
//! looks each rig node up. The result is exactly the `Vec<Option<Entity>>` the animation
//! API wants, rebuilt per instance, with no engine change at all.
//!
//! ## Why the walk follows `Parent` and not `Children`
//!
//! [`Children`](sandbox::scene::Children) is documented as a convenience — "the
//! authoritative link is `Parent`" — and the level loader takes that at its word: it
//! spawns the placement wrapper, sets `Parent(wrapper)` on the imported root, and never
//! writes `Children` on the wrapper. A top-down walk from a level entity over `Children`
//! therefore finds *nothing*, silently. This module builds its own parent → children
//! index from the `Parent` components instead, which is correct for both shapes.
//!
//! # Where the clips come from
//!
//! From [`crate::rigs`]' authoring data, not from re-reading the `.glb` it wrote.
//!
//! The alternative — `load_gltf_scene(cache/generated/warrior.glb)` at start-up and
//! [`AnimationClip::from_gltf`] — was weighed and rejected. It re-parses a file to
//! recover keyframes this binary produced moments earlier and still holds in memory,
//! adds start-up I/O to every run, and drags file state into every test that wants a
//! posed character. (Reusing the engine's *cooked* scene, which the level really loads,
//! is not on offer either: the cook cache directory is `pub(crate)` in `sandbox`.)
//!
//! What that trade costs is a conversion — [`clip_from_glb`], twenty lines — and what it
//! must not cost is fidelity: the pose the game samples has to be the pose the renderer
//! draws, which comes through the writer and the importer. That equivalence is exactly
//! what `rigs::tests::rigs_round_trip_through_the_engine_importer` pins (node names,
//! order, channel targets, key counts, interpolation) and what
//! [`tests::an_instance_clip_matches_the_imported_glb_pose`] extends to the sampled
//! values themselves. Binding is by *name*, so even a cook that re-indexed nodes could
//! not desynchronise it.

use std::collections::HashMap;

use dreamcoast_asset::glb::{GlbAnimation, GlbInterpolation, GlbPath, GlbValue};
use glam::{Quat, Vec3};
use sandbox::scene::{
    AnimPose, AnimationClip, Entity, Interpolation, LoopMode, Name, Parent, World, apply_pose,
    blend_poses_into, sample_clip_into,
};

use crate::rigs::Rig;

/// Parent → children, built from the authoritative [`Parent`] links (see the module
/// docs on why `Children` is not enough).
///
/// Built once per acquisition pass and shared by every character resolved from it, so a
/// floor of seven characters costs one O(entities) sweep rather than seven.
pub struct ChildIndex {
    by_parent: HashMap<Entity, Vec<Entity>>,
}

impl ChildIndex {
    /// Index `world`'s whole hierarchy.
    pub fn build(world: &World) -> Self {
        let mut by_parent: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (child, parent) in world.iter::<Parent>() {
            by_parent.entry(parent.0).or_default().push(child);
        }
        Self { by_parent }
    }

    fn children_of(&self, entity: Entity) -> &[Entity] {
        self.by_parent.get(&entity).map_or(&[], Vec::as_slice)
    }
}

/// One character instance's rig nodes, resolved to the entities the level built.
pub struct RigBinding {
    /// Indexed by [`Rig::nodes`] index — the id an authored channel targets.
    node_to_entity: Vec<Option<Entity>>,
    /// How many of them resolved.
    resolved: usize,
}

impl RigBinding {
    /// Resolve `rig`'s nodes against the sub-tree rooted at `root` (the level entity the
    /// character was placed as).
    ///
    /// Never fails: a node with no matching entity resolves to `None`, its channels are
    /// dropped from every clip, and [`Self::is_complete`] reports it. That is the right
    /// shape for a *binding* — a missing bone should cost that bone's motion, not the
    /// character — and it is what lets a test drive the whole game loop against a world
    /// that has the named roots but no imported geometry at all.
    pub fn resolve(world: &World, index: &ChildIndex, root: Entity, rig: &Rig) -> Self {
        let names = subtree_names(world, index, root);
        let node_to_entity: Vec<Option<Entity>> = rig
            .nodes
            .iter()
            .map(|node| names.get(node.name.as_str()).copied())
            .collect();
        let resolved = node_to_entity.iter().filter(|e| e.is_some()).count();
        Self {
            node_to_entity,
            resolved,
        }
    }

    /// Number of rig nodes that found an entity.
    pub fn resolved(&self) -> usize {
        self.resolved
    }

    /// Whether every rig node found one.
    pub fn is_complete(&self) -> bool {
        self.resolved == self.node_to_entity.len()
    }

    /// Whether no rig node found one — the character is a name with nothing under it.
    pub fn is_empty(&self) -> bool {
        self.resolved == 0
    }

    /// The entity a rig node index resolved to.
    pub fn entity(&self, node: usize) -> Option<Entity> {
        self.node_to_entity.get(node).copied().flatten()
    }
}

/// Every `Name` in the sub-tree rooted at `root`, including `root` itself.
///
/// First occurrence wins on a duplicate: rig node names are unique within a rig, and
/// preferring the shallower entity keeps a hypothetical clash resolving to the outer
/// one rather than to whichever the walk reached last. Iterative, so a malformed cycle
/// in `Parent` cannot blow the stack — the visited set terminates it.
fn subtree_names(world: &World, index: &ChildIndex, root: Entity) -> HashMap<String, Entity> {
    let mut names: HashMap<String, Entity> = HashMap::new();
    let mut queue = vec![root];
    let mut seen: Vec<Entity> = vec![root];
    while let Some(entity) = queue.pop() {
        if let Some(name) = world.get::<Name>(entity)
            && !names.contains_key(&name.0)
        {
            names.insert(name.0.clone(), entity);
        }
        for &child in index.children_of(entity) {
            if !seen.contains(&child) {
                seen.push(child);
                queue.push(child);
            }
        }
    }
    names
}

/// One character instance's clips, keyed by the rig's own clip name.
///
/// Owned per instance because the clips are: their channels carry this character's
/// entity ids. A warrior costs eight clips, a grunt five; every one is a handful of
/// sparse keyframes, so seven characters is a few tens of kilobytes.
pub struct ClipSet {
    clips: HashMap<String, AnimationClip>,
    /// Sampling scratch, reused across ticks: [`sample_clip`] allocates a fresh
    /// [`AnimPose`] per call, and this runs for every character every fixed step.
    scratch: PoseScratch,
}

/// The three pose buffers one [`ClipSet::apply`] needs, kept for reuse.
#[derive(Default)]
struct PoseScratch {
    current: AnimPose,
    from: AnimPose,
    blended: AnimPose,
}

impl ClipSet {
    /// Retarget every clip `rig` authors onto one instance's bones.
    pub fn build(rig: &Rig, binding: &RigBinding) -> Self {
        let clips = rig
            .animations
            .iter()
            .map(|anim| (anim.name.clone(), clip_from_glb(anim, binding)))
            .collect();
        Self {
            clips,
            scratch: PoseScratch::default(),
        }
    }

    /// Look a clip up by its rig name (`"attack1"`, `"walk"`, …).
    ///
    /// Test-only introspection: the game never needs a clip in the hand, only
    /// [`Self::apply`]d, and an accessor the game does not use is API that drifts.
    #[cfg(test)]
    pub fn get(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.get(name)
    }

    /// Number of clips retargeted.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Whether the set is empty (no clip retargeted).
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Sample `current` (cross-fading `fade` into it when one is running) and commit the
    /// result to this instance's bone transforms.
    ///
    /// `fade` is `(clip, time, alpha)` in the state machine's own convention — `alpha` is
    /// the weight of **`current`**, so the outgoing clip carries `1 - alpha` and a fade
    /// that has just started (`alpha == 0`) shows the outgoing pose unchanged. That is
    /// exactly `blend_poses(from, current, alpha)`.
    ///
    /// A clip name with no clip (an unbound rig, a graph naming something the rig does
    /// not author) leaves the bones where they were rather than snapping them to rest:
    /// holding the last pose is the failure the player is least likely to see.
    pub fn apply(&mut self, world: &mut World, current: ClipSample, fade: Option<ClipSample>) {
        let Some(clip) = self.clips.get(current.clip) else {
            return;
        };
        sample_clip_into(
            clip,
            current.time,
            current.mode(),
            &mut self.scratch.current,
        );

        let faded = fade.and_then(|f| self.clips.get(f.clip).map(|c| (c, f)));
        let pose = match faded {
            Some((clip, request)) => {
                sample_clip_into(clip, request.time, request.mode(), &mut self.scratch.from);
                blend_poses_into(
                    &self.scratch.from,
                    &self.scratch.current,
                    current.weight,
                    &mut self.scratch.blended,
                );
                &self.scratch.blended
            }
            None => &self.scratch.current,
        };
        apply_pose(world, pose);
    }
}

/// One clip to evaluate: which clip, at what time, wrapping or clamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipSample<'a> {
    /// The rig's clip name.
    pub clip: &'a str,
    /// Playback time in seconds, already wrapped/clamped by the state machine.
    pub time: f32,
    /// Whether the clip loops. Locomotion does; a one-shot does not, and **clamping is
    /// what parks a finished clip on its last keyframe** — which is how a death pose
    /// becomes a corpse that stays down instead of snapping back upright.
    pub looping: bool,
    /// Weight of *this* sample when it is the incoming half of a cross-fade.
    pub weight: f32,
}

impl<'a> ClipSample<'a> {
    /// A sample at full weight.
    pub fn new(clip: &'a str, time: f32, looping: bool) -> Self {
        Self {
            clip,
            time,
            looping,
            weight: 1.0,
        }
    }

    /// The same sample carrying a cross-fade weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// The [`LoopMode`] this sample asks for.
    pub fn mode(self) -> LoopMode {
        if self.looping {
            LoopMode::Loop
        } else {
            LoopMode::Clamp
        }
    }
}

/// Convert one authored clip into a playable, instance-bound [`AnimationClip`].
///
/// Channels whose node did not resolve are dropped (the same rule
/// [`AnimationClip::from_gltf`] applies to an uninstantiated node), but the clip's
/// **duration is taken from the authored keyframes, not from what survived** — a
/// dropped bone must not shorten a swing, because the combat clock is aligned to that
/// length (`crate::warrior::clip_aligned_chain`).
fn clip_from_glb(anim: &GlbAnimation, binding: &RigBinding) -> AnimationClip {
    let duration = anim
        .channels
        .iter()
        .filter_map(|c| c.keys.last().map(|(t, _)| *t))
        .fold(0.0f32, f32::max);

    let mut builder = AnimationClip::builder();
    for channel in &anim.channels {
        let Some(target) = binding.entity(channel.node) else {
            continue;
        };
        let interpolation = match channel.interpolation {
            GlbInterpolation::Step => Interpolation::Step,
            GlbInterpolation::Linear => Interpolation::Linear,
        };
        let times: Vec<f32> = channel.keys.iter().map(|(t, _)| *t).collect();
        let values = channel.keys.iter().map(|(_, v)| *v);
        builder = match channel.path {
            GlbPath::Translation => {
                builder.translation(target, interpolation, &times, &vec3_keys(values))
            }
            GlbPath::Scale => builder.scale(target, interpolation, &times, &vec3_keys(values)),
            GlbPath::Rotation => {
                builder.rotation(target, interpolation, &times, &quat_keys(values))
            }
        };
    }
    builder.duration(duration).build()
}

/// `Vec3` keys from a translation/scale channel. A `Quat` value on such a channel is
/// unrepresentable — [`dreamcoast_asset::glb::write_glb_scene`] rejects the mismatch —
/// so it collapses to the rest value rather than growing an error path for a state the
/// writer already refuses to produce.
fn vec3_keys(values: impl Iterator<Item = GlbValue>) -> Vec<Vec3> {
    values
        .map(|v| match v {
            GlbValue::Vec3(v) => Vec3::from(v),
            GlbValue::Quat(_) => Vec3::ZERO,
        })
        .collect()
}

/// `Quat` keys from a rotation channel (see [`vec3_keys`] on the mismatch case).
fn quat_keys(values: impl Iterator<Item = GlbValue>) -> Vec<Quat> {
    values
        .map(|v| match v {
            GlbValue::Quat(q) => Quat::from_array(q),
            GlbValue::Vec3(_) => Quat::IDENTITY,
        })
        .collect()
}

/// Test-only fixtures, shared with [`crate::game`]'s integration tests: a rig placed in
/// an ECS world **exactly** the way the engine's level loader places one.
///
/// Shared rather than duplicated because the shape is the load-bearing part — a fixture
/// that linked the sub-tree with `Children` would pass while the real thing failed.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;
    use sandbox::scene::{LocalTransform, MaterialHandle, MeshHandle, instantiate_gltf_mapped};
    use std::sync::OnceLock;

    /// The rig through the real writer + the real importer, parsed once per process.
    ///
    /// Cached because a test world holds seven characters and the round trip is a file
    /// write and a glTF parse; the result is immutable, so one copy serves everybody.
    pub(crate) fn import(rig: &Rig) -> &'static dreamcoast_asset::GltfScene {
        static WARRIOR: OnceLock<dreamcoast_asset::GltfScene> = OnceLock::new();
        static GRUNT: OnceLock<dreamcoast_asset::GltfScene> = OnceLock::new();
        let slot = if rig.name == crate::rigs::WARRIOR_RIG {
            &WARRIOR
        } else {
            &GRUNT
        };
        slot.get_or_init(|| round_trip(rig))
    }

    fn round_trip(rig: &Rig) -> dreamcoast_asset::GltfScene {
        let id = format!("{:?}", std::process::id());
        let dir = std::env::temp_dir().join(format!("dungeon-characters-{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.glb", rig.name));
        let bytes = dreamcoast_asset::glb::write_glb_scene(
            &rig.nodes,
            &rig.meshes,
            &rig.materials,
            &rig.animations,
        )
        .expect("write");
        std::fs::write(&path, &bytes).unwrap();
        let scene = dreamcoast_asset::load_gltf_scene(&path).expect("import");
        let _ = std::fs::remove_file(&path);
        scene
    }

    /// Spawn a rig into `world` the way `sandbox`'s level loader does: a named placement
    /// wrapper, with the imported sub-tree parented under it via `Parent` **only** — no
    /// `Children` on the wrapper, exactly as `level::build_level` leaves it (which is the
    /// whole reason [`ChildIndex`] exists).
    pub(crate) fn spawn_like_the_level_loader(
        world: &mut World,
        rig: &Rig,
        name: &str,
        translation: Vec3,
    ) -> Entity {
        let scene = import(rig);
        let handles: Vec<Vec<(MeshHandle, MaterialHandle)>> = (0..scene.meshes.len())
            .map(|i| vec![(MeshHandle(i as u32), MaterialHandle(0))])
            .collect();
        let (imported, _) = instantiate_gltf_mapped(world, scene, &handles);
        let root = world.spawn();
        world.insert(
            root,
            LocalTransform {
                translation,
                ..LocalTransform::IDENTITY
            },
        );
        world.insert(root, Name(name.to_owned()));
        world.insert(imported, Parent(root));
        root
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{import, spawn_like_the_level_loader};
    use super::*;
    use crate::rigs::{WARRIOR_ATTACK1_HIT_TIME, WARRIOR_ATTACK1_LEN, grunt, warrior};
    use sandbox::scene::{
        LocalTransform, MaterialHandle, MeshHandle, instantiate_gltf_mapped, propagate_transforms,
        sample_clip,
    };

    /// The binding finds every bone of a level-placed rig — over `Parent`, which is the
    /// only link the level loader actually writes.
    #[test]
    fn a_level_placed_rig_binds_every_bone() {
        for rig in [warrior(), grunt()] {
            let mut world = World::new();
            let root = spawn_like_the_level_loader(&mut world, &rig, "player", Vec3::ZERO);
            let index = ChildIndex::build(&world);
            let binding = RigBinding::resolve(&world, &index, root, &rig);
            assert!(
                binding.is_complete(),
                "{}: only {}/{} bones bound",
                rig.name,
                binding.resolved(),
                rig.nodes.len()
            );
            // And they are distinct entities, not one bone found under many names.
            let mut bound: Vec<Entity> = (0..rig.nodes.len())
                .filter_map(|i| binding.entity(i))
                .collect();
            bound.sort_by_key(|e| (e.index(), e.generation()));
            bound.dedup();
            assert_eq!(bound.len(), rig.nodes.len(), "{}", rig.name);
        }
    }

    /// Two instances of the same rig bind to **disjoint** bones — the property that makes
    /// per-instance clips necessary in the first place.
    #[test]
    fn two_instances_of_a_rig_bind_disjoint_bones() {
        let rig = grunt();
        let mut world = World::new();
        let a = spawn_like_the_level_loader(&mut world, &rig, "grunt_0", Vec3::ZERO);
        let b = spawn_like_the_level_loader(&mut world, &rig, "grunt_1", Vec3::X);
        let index = ChildIndex::build(&world);
        let (ba, bb) = (
            RigBinding::resolve(&world, &index, a, &rig),
            RigBinding::resolve(&world, &index, b, &rig),
        );
        assert!(ba.is_complete() && bb.is_complete());
        for i in 0..rig.nodes.len() {
            assert_ne!(
                ba.entity(i),
                bb.entity(i),
                "node {i} shared between instances"
            );
        }

        // Posing one leaves the other in its rest pose.
        let mut clips_a = ClipSet::build(&rig, &ba);
        clips_a.apply(&mut world, ClipSample::new("walk", 0.45, true), None);
        let torso = rig.nodes.iter().position(|n| n.name == "torso").unwrap();
        let posed = world
            .get::<LocalTransform>(ba.entity(torso).unwrap())
            .unwrap()
            .rotation;
        let rest = world
            .get::<LocalTransform>(bb.entity(torso).unwrap())
            .unwrap()
            .rotation;
        assert!(
            posed.angle_between(rest) > 0.01,
            "the walk pose did not move"
        );
    }

    /// A name with nothing under it binds to nothing, and everything downstream stays a
    /// no-op instead of panicking — the shape a headless test world takes.
    #[test]
    fn an_unbound_root_yields_empty_clips_and_no_panic() {
        let rig = warrior();
        let mut world = World::new();
        let root = world.spawn();
        world.insert(root, LocalTransform::IDENTITY);
        world.insert(root, Name("player".to_owned()));
        let index = ChildIndex::build(&world);
        let binding = RigBinding::resolve(&world, &index, root, &rig);
        assert!(binding.is_empty() && !binding.is_complete());

        let mut clips = ClipSet::build(&rig, &binding);
        assert_eq!(clips.len(), rig.animations.len(), "clips still exist");
        assert!(clips.get("attack1").unwrap().is_empty(), "but bind nothing");
        // Duration survives an unbound rig: the combat clock is aligned to it.
        assert_eq!(clips.get("attack1").unwrap().duration, WARRIOR_ATTACK1_LEN);
        clips.apply(&mut world, ClipSample::new("attack1", 0.2, false), None);
        clips.apply(&mut world, ClipSample::new("nonesuch", 0.2, false), None);
    }

    /// **The retarget claim.** An instance's `attack1`, sampled at the authored hit time,
    /// puts the named bone in the same pose the rig test asserts — and the same pose the
    /// engine's own `from_gltf` path produces from the written `.glb`.
    ///
    /// Both halves matter: the first says the conversion preserved the authoring, the
    /// second says the authoring survives the writer + importer the renderer draws
    /// through. Together they are the licence to build clips from authoring data instead
    /// of re-parsing the file (see the module docs).
    #[test]
    fn an_instance_clip_matches_the_imported_glb_pose() {
        let rig = warrior();
        let scene = import(&rig);
        let mut world = World::new();

        // Instance A: bound by name from a level-shaped placement, clips converted here.
        let root = spawn_like_the_level_loader(&mut world, &rig, "player", Vec3::ZERO);
        let index = ChildIndex::build(&world);
        let binding = RigBinding::resolve(&world, &index, root, &rig);
        let mut clips = ClipSet::build(&rig, &binding);
        assert_eq!(clips.get("attack1").unwrap().duration, WARRIOR_ATTACK1_LEN);
        clips.apply(
            &mut world,
            ClipSample::new("attack1", WARRIOR_ATTACK1_HIT_TIME, false),
            None,
        );

        // Instance B: the engine's road — instantiate the imported scene, resolve the
        // clip with `AnimationClip::from_gltf` against the map it hands back.
        let handles: Vec<Vec<(MeshHandle, MaterialHandle)>> = (0..scene.meshes.len())
            .map(|i| vec![(MeshHandle(i as u32), MaterialHandle(0))])
            .collect();
        let (_, map) = instantiate_gltf_mapped(&mut world, scene, &handles);
        let imported_clip = AnimationClip::from_gltf(
            scene
                .animations
                .iter()
                .find(|a| a.name.as_deref() == Some("attack1"))
                .unwrap(),
            &map,
        );
        apply_pose(
            &mut world,
            &sample_clip(&imported_clip, WARRIOR_ATTACK1_HIT_TIME, LoopMode::Clamp),
        );
        propagate_transforms(&mut world);

        // Every bone: same local TRS, to the bit.
        for (node, authored) in rig.nodes.iter().enumerate() {
            let mine = *world
                .get::<LocalTransform>(binding.entity(node).unwrap())
                .unwrap();
            let theirs = *world.get::<LocalTransform>(map[node].unwrap()).unwrap();
            assert_eq!(
                (
                    mine.translation.to_array().map(f32::to_bits),
                    mine.rotation.to_array().map(f32::to_bits),
                    mine.scale.to_array().map(f32::to_bits),
                ),
                (
                    theirs.translation.to_array().map(f32::to_bits),
                    theirs.rotation.to_array().map(f32::to_bits),
                    theirs.scale.to_array().map(f32::to_bits),
                ),
                "bone '{}' retargeted to a different pose",
                authored.name
            );
        }

        // ...and it is the pose `rigs.rs` pins for this instant, not just a consistent one.
        let elbow = rig
            .nodes
            .iter()
            .position(|n| n.name == "arm_r_lower")
            .unwrap();
        let posed = world
            .get::<LocalTransform>(binding.entity(elbow).unwrap())
            .unwrap()
            .rotation;
        let expected = Quat::from_axis_angle(Vec3::X, -0.25);
        assert!(
            posed.angle_between(expected) < 0.05,
            "elbow at the hit is {posed:?}, expected {expected:?}"
        );
    }

    /// A clamped one-shot holds its final keyframe forever — which is what makes a
    /// corpse stay down (the death clip's last two keys are equal, `rigs.rs` proves it).
    #[test]
    fn a_clamped_death_clip_holds_the_corpse_pose() {
        for rig in [warrior(), grunt()] {
            let mut world = World::new();
            let root = spawn_like_the_level_loader(&mut world, &rig, "corpse", Vec3::ZERO);
            let index = ChildIndex::build(&world);
            let binding = RigBinding::resolve(&world, &index, root, &rig);
            let mut clips = ClipSet::build(&rig, &binding);
            let death = clips.get("death").unwrap().duration;

            let bones: Vec<Entity> = (0..rig.nodes.len())
                .filter_map(|i| binding.entity(i))
                .collect();
            let snapshot = |world: &World| -> Vec<LocalTransform> {
                bones
                    .iter()
                    .map(|&e| *world.get::<LocalTransform>(e).unwrap())
                    .collect()
            };

            clips.apply(&mut world, ClipSample::new("death", death, false), None);
            let at_end = snapshot(&world);
            // Ten seconds later — long past the clip — nothing has moved.
            clips.apply(
                &mut world,
                ClipSample::new("death", death + 10.0, false),
                None,
            );
            assert_eq!(
                snapshot(&world),
                at_end,
                "{}: the corpse got back up",
                rig.name
            );
            // And the pose is not the rest pose it started from.
            let mut fresh = World::new();
            let fresh_root = spawn_like_the_level_loader(&mut fresh, &rig, "rest", Vec3::ZERO);
            let fresh_index = ChildIndex::build(&fresh);
            let fresh_binding = RigBinding::resolve(&fresh, &fresh_index, fresh_root, &rig);
            let rest: Vec<LocalTransform> = (0..rig.nodes.len())
                .filter_map(|i| fresh_binding.entity(i))
                .map(|e| *fresh.get::<LocalTransform>(e).unwrap())
                .collect();
            assert_ne!(
                rest, at_end,
                "{}: the death pose is the rest pose",
                rig.name
            );
        }
    }

    /// A cross-fade at `alpha = 0` is the outgoing pose and at `alpha = 1` the incoming
    /// one — the state machine's convention, pinned so a sign flip cannot hide.
    #[test]
    fn a_crossfade_runs_from_the_outgoing_pose_to_the_incoming_one() {
        let rig = grunt();
        let mut world = World::new();
        let root = spawn_like_the_level_loader(&mut world, &rig, "grunt_0", Vec3::ZERO);
        let index = ChildIndex::build(&world);
        let binding = RigBinding::resolve(&world, &index, root, &rig);
        let mut clips = ClipSet::build(&rig, &binding);
        let bones: Vec<Entity> = (0..rig.nodes.len())
            .filter_map(|i| binding.entity(i))
            .collect();
        let snapshot = |world: &World| -> Vec<LocalTransform> {
            bones
                .iter()
                .map(|&e| *world.get::<LocalTransform>(e).unwrap())
                .collect()
        };

        let walk = ClipSample::new("walk", 0.3, true);
        let attack = ClipSample::new("attack", 0.2, false);

        clips.apply(&mut world, walk, None);
        let pure_walk = snapshot(&world);
        clips.apply(&mut world, attack, None);
        let pure_attack = snapshot(&world);

        clips.apply(&mut world, attack.with_weight(0.0), Some(walk));
        assert_eq!(snapshot(&world), pure_walk, "alpha 0 is the outgoing clip");
        clips.apply(&mut world, attack.with_weight(1.0), Some(walk));
        assert_eq!(
            snapshot(&world),
            pure_attack,
            "alpha 1 is the incoming clip"
        );
        // And the middle is neither.
        clips.apply(&mut world, attack.with_weight(0.5), Some(walk));
        let mid = snapshot(&world);
        assert_ne!(mid, pure_walk);
        assert_ne!(mid, pure_attack);
    }
}
