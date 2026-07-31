//! The dungeon's characters — **authored from code** as articulated glTF, the same way
//! [`crate::level`] authors its geometry (`docs/game-framework-plan.md` §4.1).
//!
//! # Why rigid bones instead of skinning
//!
//! The plan's M2 asset was "an AI-generated rigged humanoid glTF". This is the same
//! *road* with the risk taken out: a character here is a **node hierarchy of rigid box
//! meshes** — a pelvis, a torso, limb segments, a sword — driven by node-TRS animation
//! channels. That is a strict subset of what the engine already imports and plays:
//!
//! * `dreamcoast_asset::load_gltf_scene` reads the node tree and the clips,
//! * `dreamcoast_scene::instantiate_gltf_mapped` turns the tree into an ECS sub-tree and
//!   hands back the node → entity map,
//! * `AnimationClip::from_gltf` resolves the channels against that map, and
//!   `advance_animation` writes `LocalTransform`s that `propagate_transforms` composes.
//!
//! No skin, no joint palette, no GPU skinning path, no vertex-weight quality risk (plan
//! §5-R4) — and no new engine code at all: the characters arrive through the *same door*
//! as an authored glTF, which is the property the whole generated-asset design rests on.
//! A skinned character later replaces the data, not the road.
//!
//! # Conventions
//!
//! * **+Z is forward, +Y is up, origin between the feet.** The character's *left* is +X
//!   (`forward × up = (0,0,1) × (0,1,0) = (-1,0,0)` is its right), which is why the left
//!   limbs sit at positive x.
//! * Every node's origin **is its joint**, and its mesh is authored in joint-local
//!   metres, so a rotation channel swings the limb about the joint with no offset maths.
//! * The rest pose is all-identity rotations (arms hanging, blade forward from the fist).
//!   Animation channels are **absolute**, not additive — a clip that does not touch a
//!   node leaves it in that rest pose, so each clip poses every bone it needs.
//! * Angles are authored in **radians** through [`euler`], which composes axis-angle
//!   quaternions (yaw about +Y, then pitch about +X, then roll about +Z).
//! * Clips are **in place**: no node ever translates in X or Z (plan §4.1 — the mover
//!   owns position, root motion is out of scope). `clips_are_in_place` enforces it.
//! * Poses are sparse **keyframes**, not baked curves: the sampler interpolates.
//!
//! # Readability from above
//!
//! The camera is a fixed ~55° top-down (`crate::game::camera_offset`), so a pose is read
//! mostly from the *plan* silhouette: shoulders, the sword's line, and the gap between
//! the legs. Motion is therefore exaggerated in the horizontal plane — torso yaw against
//! pelvis yaw, wide arm sweeps, a long blade — where a character-height side view would
//! instead lean on vertical arcs that foreshorten to nothing from up here. The helm
//! crest is the one deliberately vertical detail: it is the only part of a knight that
//! stays visible when the body is directly beneath the camera.
//!
//! # Scope
//!
//! Authoring only. [`ensure_rigs`] writes the two `.glb` files next to the generated
//! levels; **nothing here is wired into a level or into gameplay** — spawning, the
//! animation state machine and the combat hookup are the integrator's, and the clip
//! timing they need is exported as consts here ([`WARRIOR_ATTACK1_HIT_TIME`] and
//! friends) so the numbers have exactly one home.

// Nothing in the game binary calls this module yet — by design: it is the authoring
// half of M2, and the integrator wave wires the spawn and the state machine. The tests
// below exercise every public item.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use dreamcoast_asset::MeshVertex;
use dreamcoast_asset::glb::{
    GlbAnimation, GlbChannel, GlbMaterial, GlbMesh, GlbNode, save_glb_scene,
};
use glam::{Quat, Vec3};

/// A character ready for the glTF writer: the node tree, the rigid meshes those nodes
/// carry, the materials, and the clips.
pub struct Rig {
    /// File stem inside [`crate::level::GENERATED_DIR`] (`warrior` → `warrior.glb`).
    pub name: &'static str,
    pub nodes: Vec<GlbNode>,
    pub meshes: Vec<GlbMesh>,
    pub materials: Vec<GlbMaterial>,
    pub animations: Vec<GlbAnimation>,
}

impl Rig {
    /// Total triangle count (debug/logging — a budget the top-down camera cares about).
    pub fn triangles(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len() / 3).sum()
    }
}

// --- Clip metadata: the single source the combat layer aligns to ----------------------
//
// Authored *with* the clips below (each clip's last keyframe is its `_LEN`), so the
// integrator reads a number that cannot drift from the animation it describes — a hit
// window transcribed into a state-machine RON would.

/// Warrior clip names, in authoring order.
pub const WARRIOR_CLIPS: [&str; 8] = [
    "idle", "run", "attack1", "attack2", "attack3", "dodge", "hit", "death",
];
/// Grunt clip names, in authoring order.
pub const GRUNT_CLIPS: [&str; 5] = ["idle", "walk", "attack", "hit", "death"];

pub const WARRIOR_IDLE_LEN: f32 = 2.5;
pub const WARRIOR_RUN_LEN: f32 = 0.7;
pub const WARRIOR_ATTACK1_LEN: f32 = 0.55;
pub const WARRIOR_ATTACK2_LEN: f32 = 0.5;
pub const WARRIOR_ATTACK3_LEN: f32 = 0.85;
pub const WARRIOR_DODGE_LEN: f32 = 0.35;
pub const WARRIOR_HIT_LEN: f32 = 0.3;
pub const WARRIOR_DEATH_LEN: f32 = 1.2;

/// When the blade crosses the arc's centre in each attack — the instant the combat
/// layer should test its damage cone. Placed at the pose where the sword is in front of
/// the character, not at the swing's start or end.
pub const WARRIOR_ATTACK1_HIT_TIME: f32 = 0.28;
pub const WARRIOR_ATTACK2_HIT_TIME: f32 = 0.24;
pub const WARRIOR_ATTACK3_HIT_TIME: f32 = 0.46;

/// The dodge's airborne middle — the window the roll is meant to be invulnerable for,
/// authored to match the crouch the clip actually holds.
pub const WARRIOR_DODGE_IFRAME_START: f32 = 0.07;
pub const WARRIOR_DODGE_IFRAME_END: f32 = 0.26;

/// Time by which the death clip has reached its final pose; every later keyframe repeats
/// it, so a player that stops here or runs to the end shows the same thing. `death` is
/// the one clip that must **not** loop.
pub const WARRIOR_DEATH_HOLD_TIME: f32 = 0.95;

pub const GRUNT_IDLE_LEN: f32 = 2.0;
pub const GRUNT_WALK_LEN: f32 = 0.9;
pub const GRUNT_ATTACK_LEN: f32 = 0.7;
pub const GRUNT_HIT_LEN: f32 = 0.25;
pub const GRUNT_DEATH_LEN: f32 = 1.0;
/// When the grunt's claw reaches full extension.
pub const GRUNT_ATTACK_HIT_TIME: f32 = 0.34;
/// As [`WARRIOR_DEATH_HOLD_TIME`], for the grunt.
pub const GRUNT_DEATH_HOLD_TIME: f32 = 0.8;

// --- Materials -----------------------------------------------------------------------

/// Warrior material slots, in write order.
const MAT_STEEL: usize = 0;
const MAT_DARK: usize = 1;
const MAT_BLADE: usize = 2;
const MAT_ACCENT: usize = 3;

/// The knight's four surfaces. Chosen so the *silhouette* separates from a stone dungeon
/// under a low sun: bright rough-metal plate for the big shapes, a near-black dielectric
/// underlayer so joints read as gaps rather than as more plate, a smoother and more
/// reflective blade so the sword catches the sun and draws the swing arc, and one
/// saturated accent reserved for the helm crest — the only part guaranteed to be visible
/// from directly above.
fn warrior_materials() -> Vec<GlbMaterial> {
    vec![
        GlbMaterial {
            name: "warrior_plate".into(),
            base_color_factor: [0.62, 0.64, 0.68, 1.0],
            metallic: 0.8,
            roughness: 0.45,
            double_sided: false,
        },
        GlbMaterial {
            name: "warrior_underlayer".into(),
            base_color_factor: [0.14, 0.13, 0.15, 1.0],
            metallic: 0.25,
            roughness: 0.7,
            double_sided: false,
        },
        GlbMaterial {
            name: "warrior_blade".into(),
            base_color_factor: [0.82, 0.84, 0.88, 1.0],
            metallic: 1.0,
            roughness: 0.25,
            double_sided: false,
        },
        GlbMaterial {
            name: "warrior_crest".into(),
            base_color_factor: [0.52, 0.11, 0.13, 1.0],
            metallic: 0.0,
            roughness: 0.55,
            double_sided: false,
        },
    ]
}

/// Grunt material slots, in write order.
const MAT_BONE: usize = 0;
const MAT_GRUNT_DARK: usize = 1;

/// Bone-white for the whole minion plus one dark core, so a crowd of them reads as a
/// crowd (bright, uniform, unambiguously *not* the player) with a single dark anchor
/// that tells you which way each one faces.
fn grunt_materials() -> Vec<GlbMaterial> {
    vec![
        GlbMaterial {
            name: "grunt_bone".into(),
            base_color_factor: [0.78, 0.75, 0.66, 1.0],
            metallic: 0.0,
            roughness: 0.65,
            double_sided: false,
        },
        GlbMaterial {
            name: "grunt_core".into(),
            base_color_factor: [0.09, 0.09, 0.11, 1.0],
            metallic: 0.1,
            roughness: 0.8,
            double_sided: false,
        },
    ]
}

// --- Rig construction ----------------------------------------------------------------

/// An axis-aligned box in joint-local metres, as `(min, max)`.
type BoxSpan = ([f32; 3], [f32; 3]);

/// Mirror a box across the YZ plane — the right-side counterpart of a left-side part.
fn flip_x((min, max): BoxSpan) -> BoxSpan {
    ([-max[0], min[1], min[2]], [-min[0], max[1], max[2]])
}

/// Accumulates a skeleton: nodes and the rigid meshes they carry, wired together as they
/// are added so a mesh index can never drift from the node that owns it.
struct RigBuilder {
    nodes: Vec<GlbNode>,
    meshes: Vec<GlbMesh>,
}

impl RigBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            meshes: Vec::new(),
        }
    }

    /// A transform-only node (the rig root, or a joint whose geometry is on its
    /// children). Returns its node index — the id an animation channel targets.
    fn joint(&mut self, name: &str, parent: Option<usize>, translation: [f32; 3]) -> usize {
        self.nodes.push(GlbNode::new(name, parent, translation));
        self.nodes.len() - 1
    }

    /// A joint plus the rigid geometry welded to it: `boxes` are `(min, max)` spans in
    /// **joint-local** metres, so the joint sits wherever the limb pivots.
    fn bone(
        &mut self,
        name: &str,
        parent: usize,
        translation: [f32; 3],
        material: usize,
        boxes: &[BoxSpan],
    ) -> usize {
        let mut mesh = BoxMesh::new(name, material);
        for &(min, max) in boxes {
            mesh.add_box(Vec3::from(min), Vec3::from(max));
        }
        self.meshes.push(mesh.finish());
        let mesh = self.meshes.len() - 1;
        self.nodes
            .push(GlbNode::new(name, Some(parent), translation).with_mesh(mesh));
        self.nodes.len() - 1
    }
}

/// Accumulates axis-aligned boxes into one indexed triangle mesh with outward per-face
/// normals, wound counter-clockwise about the normal — the workspace's front-face
/// convention (see `crate::level`'s winding test for why a mixed mesh is worse than a
/// backwards one: the per-mesh SDF signs by vertex normals).
struct BoxMesh {
    name: String,
    material: usize,
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl BoxMesh {
    fn new(name: &str, material: usize) -> Self {
        Self {
            name: name.to_owned(),
            material,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn finish(self) -> GlbMesh {
        GlbMesh {
            name: self.name,
            vertices: self.vertices,
            indices: self.indices,
            material: self.material,
        }
    }

    /// Append a box as six unsubdivided quads. Unlike the dungeon's chunk boxes these
    /// are *not* subdivided: a character is small, close to the camera and lit by the
    /// pixel shader, and the per-mesh SDF that interior vertices exist to feed is a
    /// static-geometry bake a moving character never enters anyway (plan §0 constraint 2).
    fn add_box(&mut self, min: Vec3, max: Vec3) {
        let e = max - min;
        let (x, y, z) = (
            Vec3::new(e.x, 0.0, 0.0),
            Vec3::new(0.0, e.y, 0.0),
            Vec3::new(0.0, 0.0, e.z),
        );
        // (face origin, du, dv) with `du × dv` outward, which makes the quad below
        // counter-clockwise seen from outside.
        for (origin, du, dv) in [
            (min + x, y, z), // +X
            (min, z, y),     // -X
            (min + y, z, x), // +Y
            (min, x, z),     // -Y
            (min + z, x, y), // +Z
            (min, y, x),     // -Z
        ] {
            self.add_face(origin, du, dv);
        }
    }

    /// One planar quad. UVs are the face-local position in metres (a world-planar
    /// projection, matching the dungeon's convention).
    fn add_face(&mut self, origin: Vec3, du: Vec3, dv: Vec3) {
        let normal = du.cross(dv).normalize().to_array();
        let (lu, lv) = (du.length(), dv.length());
        let base = self.vertices.len() as u32;
        for (fu, fv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            self.vertices.push(MeshVertex {
                pos: (origin + du * fu + dv * fv).to_array(),
                normal,
                uv: [fu * lu, fv * lv],
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// --- Pose authoring ------------------------------------------------------------------

/// A rotation from radian angles, composed from axis-angle quaternions in yaw → pitch →
/// roll order (`Y * X * Z`), returned as glTF's `[x, y, z, w]`.
///
/// With the conventions above: **pitch** (+X) swings a hanging limb *backwards* and tips
/// a standing body's head *forwards*; **yaw** (+Y) turns the character to its left;
/// **roll** (+Z) abducts a left limb away from the body.
fn euler(pitch: f32, yaw: f32, roll: f32) -> [f32; 4] {
    (Quat::from_axis_angle(Vec3::Y, yaw)
        * Quat::from_axis_angle(Vec3::X, pitch)
        * Quat::from_axis_angle(Vec3::Z, roll))
    .to_array()
}

/// A `(time, [pitch, yaw, roll])` keyframe table in radians.
type Pose = [(f32, [f32; 3])];

/// The same cycle half a period later — the opposite limb of a walk or run.
///
/// Deriving the second limb instead of transcribing it is what keeps a gait symmetric:
/// a mistyped digit in a mirrored table is invisible in code review and obvious only as
/// a limp. Requires a keyframe exactly at the half-period (so the shifted cycle still
/// starts at t = 0), which every gait table below has.
fn half_phase(keys: &Pose, period: f32) -> Vec<(f32, [f32; 3])> {
    let half = period * 0.5;
    let mut out: Vec<(f32, [f32; 3])> = keys
        .iter()
        .filter(|(t, _)| *t < period)
        .map(|&(t, v)| ((t + half) % period, v))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(
        out[0].0 == 0.0,
        "a cycle table needs a keyframe at its half-period to be phase-shifted"
    );
    let opening = out[0].1;
    out.push((period, opening));
    out
}

/// The mirror-image pose table: yaw and roll flip, pitch does not — the left limb's
/// motion on the right side of the body.
fn mirrored(keys: &Pose) -> Vec<(f32, [f32; 3])> {
    keys.iter()
        .map(|&(t, [pitch, yaw, roll])| (t, [pitch, -yaw, -roll]))
        .collect()
}

/// Assembles one clip's channels.
struct Clip {
    name: &'static str,
    channels: Vec<GlbChannel>,
}

impl Clip {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            channels: Vec::new(),
        }
    }

    /// A linear rotation channel from radian `(time, [pitch, yaw, roll])` keys.
    #[must_use]
    fn rot(mut self, node: usize, keys: &Pose) -> Self {
        self.channels.push(GlbChannel::rotation(
            node,
            keys.iter().map(|&(t, e)| (t, euler(e[0], e[1], e[2]))),
        ));
        self
    }

    /// A linear translation channel. Only ever used for the vertical bob — the in-place
    /// rule means x and z must equal the node's rest position (`clips_are_in_place`).
    #[must_use]
    fn pos(mut self, node: usize, keys: &[(f32, [f32; 3])]) -> Self {
        self.channels
            .push(GlbChannel::translation(node, keys.iter().copied()));
        self
    }

    fn finish(self) -> GlbAnimation {
        GlbAnimation {
            name: self.name.to_owned(),
            channels: self.channels,
        }
    }
}

// --- The warrior ---------------------------------------------------------------------

/// Standing height of the knight's body, metres (the crest reaches a little higher).
const WARRIOR_HEIGHT: f32 = 1.66;
/// Rest height of the pelvis joint, metres — the hips the whole body hangs from.
const WARRIOR_HIP_Y: f32 = 0.90;

/// The warrior's node indices, bound as the skeleton is built so the clips below cannot
/// address the wrong bone.
struct WarriorBones {
    root: usize,
    pelvis: usize,
    torso: usize,
    head: usize,
    arm_l_upper: usize,
    arm_l_lower: usize,
    hand_l: usize,
    arm_r_upper: usize,
    arm_r_lower: usize,
    hand_r: usize,
    leg_l_upper: usize,
    leg_l_lower: usize,
    foot_l: usize,
    leg_r_upper: usize,
    leg_r_lower: usize,
    foot_r: usize,
}

/// The blocky knight: ~1.7 m, origin between the feet, facing +Z, sword in the right
/// hand.
///
/// Proportions are deliberately chunky (wide pauldrons, a short thick torso, a long
/// blade) rather than realistic: at a 55° top-down angle the readable information is the
/// plan-view outline, and a naturalistically proportioned figure collapses into a blob
/// of head and shoulders.
pub fn warrior() -> Rig {
    let mut b = RigBuilder::new();

    let root = b.joint("warrior_root", None, [0.0, 0.0, 0.0]);

    // Hips and spine. Each mesh is authored around its own joint: the pelvis straddles
    // it, the torso rises from it.
    let pelvis = b.bone(
        "pelvis",
        root,
        [0.0, WARRIOR_HIP_Y, 0.0],
        MAT_STEEL,
        &[([-0.16, -0.09, -0.10], [0.16, 0.08, 0.10])],
    );
    let torso = b.bone(
        "torso",
        pelvis,
        [0.0, 0.08, 0.0],
        MAT_STEEL,
        &[([-0.20, 0.0, -0.125], [0.20, 0.44, 0.125])],
    );
    let head = b.bone(
        "head",
        torso,
        [0.0, 0.44, 0.0],
        MAT_STEEL,
        &[([-0.115, 0.0, -0.115], [0.115, 0.24, 0.115])],
    );
    // The crest: the top-down camera's tell for which way the knight is facing (it runs
    // front-to-back) and the only accent-coloured part of the rig.
    b.bone(
        "crest",
        head,
        [0.0, 0.24, 0.0],
        MAT_ACCENT,
        &[([-0.028, 0.0, -0.10], [0.028, 0.09, 0.10])],
    );

    // Arms. The pauldron rides the upper arm rather than the torso, so a swing carries
    // the shoulder plate with it — which is most of what the swing reads as from above.
    let pauldron_l: BoxSpan = ([-0.07, -0.05, -0.115], [0.11, 0.11, 0.115]);
    let upper_arm_l: BoxSpan = ([-0.055, -0.30, -0.055], [0.055, 0.0, 0.055]);
    let lower_arm_l: BoxSpan = ([-0.05, -0.26, -0.05], [0.05, 0.0, 0.05]);
    let hand_box_l: BoxSpan = ([-0.055, -0.11, -0.06], [0.055, 0.0, 0.06]);

    let arm_l_upper = b.bone(
        "arm_l_upper",
        torso,
        [0.21, 0.38, 0.0],
        MAT_STEEL,
        &[upper_arm_l, pauldron_l],
    );
    let arm_l_lower = b.bone(
        "arm_l_lower",
        arm_l_upper,
        [0.0, -0.30, 0.0],
        MAT_DARK,
        &[lower_arm_l],
    );
    let hand_l = b.bone(
        "hand_l",
        arm_l_lower,
        [0.0, -0.26, 0.0],
        MAT_DARK,
        &[hand_box_l],
    );

    let arm_r_upper = b.bone(
        "arm_r_upper",
        torso,
        [-0.21, 0.38, 0.0],
        MAT_STEEL,
        &[flip_x(upper_arm_l), flip_x(pauldron_l)],
    );
    let arm_r_lower = b.bone(
        "arm_r_lower",
        arm_r_upper,
        [0.0, -0.30, 0.0],
        MAT_DARK,
        &[flip_x(lower_arm_l)],
    );
    // The right hand carries the grip and the crossguard as well as the fist: they are
    // the same dark material, and welding them here keeps the sword itself a single
    // stretched box on its own node.
    let hand_r = b.bone(
        "hand_r",
        arm_r_lower,
        [0.0, -0.26, 0.0],
        MAT_DARK,
        &[
            flip_x(hand_box_l),
            ([-0.032, -0.09, -0.10], [0.032, -0.02, 0.06]), // grip, through the fist
            ([-0.14, -0.09, 0.06], [0.14, -0.04, 0.10]),    // crossguard
        ],
    );
    // The blade: one stretched box reaching forward out of the fist. Long on purpose —
    // it is the swing's whole readable arc from a top-down camera.
    b.bone(
        "sword",
        hand_r,
        [0.0, -0.065, 0.10],
        MAT_BLADE,
        &[([-0.035, -0.014, 0.0], [0.035, 0.014, 0.78])],
    );

    // Legs. Knee and ankle land where the geometry ends, so the sole is exactly y = 0.
    let upper_leg_l: BoxSpan = ([-0.075, -0.39, -0.075], [0.075, 0.0, 0.075]);
    let lower_leg_l: BoxSpan = ([-0.065, -0.38, -0.065], [0.065, 0.0, 0.065]);
    let foot_box_l: BoxSpan = ([-0.075, -0.09, -0.08], [0.075, 0.0, 0.17]);

    let leg_l_upper = b.bone(
        "leg_l_upper",
        pelvis,
        [0.10, -0.04, 0.0],
        MAT_STEEL,
        &[upper_leg_l],
    );
    let leg_l_lower = b.bone(
        "leg_l_lower",
        leg_l_upper,
        [0.0, -0.39, 0.0],
        MAT_DARK,
        &[lower_leg_l],
    );
    let foot_l = b.bone(
        "foot_l",
        leg_l_lower,
        [0.0, -0.38, 0.0],
        MAT_DARK,
        &[foot_box_l],
    );
    let leg_r_upper = b.bone(
        "leg_r_upper",
        pelvis,
        [-0.10, -0.04, 0.0],
        MAT_STEEL,
        &[flip_x(upper_leg_l)],
    );
    let leg_r_lower = b.bone(
        "leg_r_lower",
        leg_r_upper,
        [0.0, -0.39, 0.0],
        MAT_DARK,
        &[flip_x(lower_leg_l)],
    );
    let foot_r = b.bone(
        "foot_r",
        leg_r_lower,
        [0.0, -0.38, 0.0],
        MAT_DARK,
        &[flip_x(foot_box_l)],
    );

    let bones = WarriorBones {
        root,
        pelvis,
        torso,
        head,
        arm_l_upper,
        arm_l_lower,
        hand_l,
        arm_r_upper,
        arm_r_lower,
        hand_r,
        leg_l_upper,
        leg_l_lower,
        foot_l,
        leg_r_upper,
        leg_r_lower,
        foot_r,
    };
    let animations = warrior_clips(&bones);

    Rig {
        name: "warrior",
        nodes: b.nodes,
        meshes: b.meshes,
        materials: warrior_materials(),
        animations,
    }
}

/// The knight's eight clips. Order matches [`WARRIOR_CLIPS`].
fn warrior_clips(b: &WarriorBones) -> Vec<GlbAnimation> {
    vec![
        warrior_idle(b),
        warrior_run(b),
        warrior_attack1(b),
        warrior_attack2(b),
        warrior_attack3(b),
        warrior_dodge(b),
        warrior_hit(b),
        warrior_death(b),
    ]
}

/// A breathing stance: the hips rise and fall, the torso counter-yaws against the
/// pelvis, and the sword arm bobs so the blade's tip — the longest lever in the
/// silhouette — is never quite still.
fn warrior_idle(b: &WarriorBones) -> GlbAnimation {
    let hip = |y: f32| [0.0, y, 0.0];
    Clip::new("idle")
        .pos(
            b.pelvis,
            &[
                (0.0, hip(0.900)),
                (0.625, hip(0.907)),
                (1.25, hip(0.912)),
                (1.875, hip(0.906)),
                (WARRIOR_IDLE_LEN, hip(0.900)),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, -0.02, 0.0]),
                (1.25, [0.0, 0.02, 0.0]),
                (WARRIOR_IDLE_LEN, [0.0, -0.02, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.0, 0.03, 0.0]),
                (0.625, [0.015, 0.0, 0.0]),
                (1.25, [0.025, -0.03, 0.0]),
                (1.875, [0.015, 0.0, 0.0]),
                (WARRIOR_IDLE_LEN, [0.0, 0.03, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.06, 0.0]),
                (1.25, [0.02, -0.06, 0.0]),
                (WARRIOR_IDLE_LEN, [0.0, 0.06, 0.0]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.05, 0.0, 0.07]),
                (1.25, [0.10, 0.0, 0.09]),
                (WARRIOR_IDLE_LEN, [0.05, 0.0, 0.07]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.18, 0.0, 0.0]),
                (1.25, [-0.24, 0.0, 0.0]),
                (WARRIOR_IDLE_LEN, [-0.18, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.12, 0.0, -0.10]),
                (1.25, [-0.06, 0.0, -0.12]),
                (WARRIOR_IDLE_LEN, [-0.12, 0.0, -0.10]),
            ],
        )
        // The sword bob rides the elbow, on its own slower beat than the breath so the
        // two never quite line up and the stance does not look metronomic.
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (0.9, [-0.62, 0.0, 0.0]),
                (1.8, [-0.50, 0.0, 0.0]),
                (WARRIOR_IDLE_LEN, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, -0.06]),
                (1.25, [0.05, -0.16, -0.02]),
                (WARRIOR_IDLE_LEN, [0.0, -0.20, -0.06]),
            ],
        )
        .finish()
}

/// A run cycle: left leg forward at t = 0, right at the half-beat (derived, not
/// transcribed), with the torso yawing against the pelvis. The counter-rotation is the
/// part that reads from above — the legs themselves are mostly hidden by the shoulders.
fn warrior_run(b: &WarriorBones) -> GlbAnimation {
    let t = WARRIOR_RUN_LEN;
    let (q1, q2, q3) = (t * 0.25, t * 0.5, t * 0.75);
    let hip = |y: f32| [0.0, y, 0.0];

    let thigh = [
        (0.0, [-0.62, 0.0, 0.0]),
        (q1, [-0.10, 0.0, 0.0]),
        (q2, [0.45, 0.0, 0.0]),
        (q3, [-0.30, 0.0, 0.0]),
        (t, [-0.62, 0.0, 0.0]),
    ];
    let knee = [
        (0.0, [0.18, 0.0, 0.0]),
        (q1, [0.10, 0.0, 0.0]),
        (q2, [0.30, 0.0, 0.0]),
        (q3, [0.95, 0.0, 0.0]),
        (t, [0.18, 0.0, 0.0]),
    ];
    let ankle = [
        (0.0, [-0.20, 0.0, 0.0]),
        (q1, [0.0, 0.0, 0.0]),
        (q2, [0.45, 0.0, 0.0]),
        (q3, [-0.25, 0.0, 0.0]),
        (t, [-0.20, 0.0, 0.0]),
    ];
    // The free arm swings opposite its own leg; the sword arm keeps the blade tucked
    // and angled outward instead of windmilling it through the legs.
    let free_arm = [
        (0.0, [0.62, 0.0, 0.10]),
        (q1, [0.0, 0.0, 0.12]),
        (q2, [-0.80, 0.0, 0.14]),
        (q3, [0.0, 0.0, 0.12]),
        (t, [0.62, 0.0, 0.10]),
    ];

    Clip::new("run")
        // Two bobs per cycle — one per footfall.
        .pos(
            b.pelvis,
            &[
                (0.0, hip(0.868)),
                (q1, hip(0.900)),
                (q2, hip(0.868)),
                (q3, hip(0.900)),
                (t, hip(0.868)),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, -0.20, 0.0]),
                (q2, [0.0, 0.20, 0.0]),
                (t, [0.0, -0.20, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.16, 0.26, 0.0]),
                (q2, [0.16, -0.26, 0.0]),
                (t, [0.16, 0.26, 0.0]),
            ],
        )
        // Head level against the forward lean, so the crest stays a readable arrow.
        .rot(b.head, &[(0.0, [-0.14, 0.0, 0.0]), (t, [-0.14, 0.0, 0.0])])
        .rot(b.leg_l_upper, &thigh)
        .rot(b.leg_l_lower, &knee)
        .rot(b.foot_l, &ankle)
        .rot(b.leg_r_upper, &half_phase(&thigh, t))
        .rot(b.leg_r_lower, &half_phase(&knee, t))
        .rot(b.foot_r, &half_phase(&ankle, t))
        .rot(b.arm_l_upper, &half_phase(&free_arm, t))
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (q2, [-0.95, 0.0, 0.0]),
                (t, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.45, 0.0, -0.12]),
                (q2, [0.20, 0.0, -0.14]),
                (t, [-0.45, 0.0, -0.12]),
            ],
        )
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-1.00, 0.0, 0.0]),
                (q2, [-1.25, 0.0, 0.0]),
                (t, [-1.00, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[(0.0, [0.0, -0.30, 0.0]), (t, [0.0, -0.30, 0.0])],
        )
        .finish()
}

/// Combo opener: a horizontal slash from the knight's right to its left.
///
/// The sweep is carried by torso **yaw** with the shoulder raised to roughly horizontal,
/// because at that pitch a shoulder yaw sweeps the arm through the horizontal plane —
/// the plane the camera sees. [`WARRIOR_ATTACK1_HIT_TIME`] is the pose where the blade
/// crosses centre.
fn warrior_attack1(b: &WarriorBones) -> GlbAnimation {
    let (wind, hit, follow, end) = (0.16, WARRIOR_ATTACK1_HIT_TIME, 0.40, WARRIOR_ATTACK1_LEN);
    Clip::new("attack1")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (wind, [0.0, 0.875, 0.0]),
                (hit, [0.0, 0.885, 0.0]),
                (end, [0.0, 0.900, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, -0.10, 0.0]),
                (wind, [0.0, -0.30, 0.0]),
                (hit, [0.0, 0.05, 0.0]),
                (follow, [0.0, 0.22, 0.0]),
                (end, [0.0, -0.02, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.05, -0.20, 0.0]),
                (wind, [0.02, -0.62, 0.0]),
                (hit, [0.10, 0.15, 0.0]),
                (follow, [0.14, 0.55, 0.0]),
                (end, [0.03, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, -0.10, 0.0]),
                (wind, [0.0, -0.25, 0.0]),
                (hit, [0.0, 0.10, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.35, 0.0, -0.15]),
                (wind, [-1.15, -0.75, -0.30]),
                (hit, [-1.40, 0.10, -0.25]),
                (follow, [-1.20, 0.80, -0.20]),
                (end, [-0.30, 0.0, -0.12]),
            ],
        )
        // Coiled at the wind-up, snapping straight through the hit.
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.60, 0.0, 0.0]),
                (wind, [-1.30, 0.0, 0.0]),
                (hit, [-0.25, 0.0, 0.0]),
                (follow, [-0.15, 0.0, 0.0]),
                (end, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, -0.15]),
                (hit, [0.0, 0.0, 0.10]),
                (end, [0.0, -0.20, -0.15]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.10, 0.0, 0.12]),
                (wind, [0.45, 0.0, 0.35]),
                (hit, [0.15, 0.0, 0.55]),
                (follow, [-0.10, 0.0, 0.45]),
                (end, [0.08, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.25, 0.0, 0.0]),
                (hit, [-0.70, 0.0, 0.0]),
                (end, [-0.25, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (wind, [-0.18, 0.0, 0.0]),
                (follow, [0.12, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (wind, [0.14, 0.0, 0.0]),
                (follow, [-0.10, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .finish()
}

/// Combo second beat: the blade comes straight back the other way, left to right, with
/// the wrist rolled over into a backhand. Shorter than the opener so the combo
/// accelerates.
fn warrior_attack2(b: &WarriorBones) -> GlbAnimation {
    let (wind, hit, follow, end) = (0.14, WARRIOR_ATTACK2_HIT_TIME, 0.34, WARRIOR_ATTACK2_LEN);
    Clip::new("attack2")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (wind, [0.0, 0.882, 0.0]),
                (hit, [0.0, 0.890, 0.0]),
                (end, [0.0, 0.900, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, 0.06, 0.0]),
                (wind, [0.0, 0.28, 0.0]),
                (hit, [0.0, -0.05, 0.0]),
                (follow, [0.0, -0.20, 0.0]),
                (end, [0.0, 0.02, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.03, 0.20, 0.0]),
                (wind, [0.02, 0.62, 0.0]),
                (hit, [0.10, -0.10, 0.0]),
                (follow, [0.12, -0.50, 0.0]),
                (end, [0.03, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.08, 0.0]),
                (wind, [0.0, 0.22, 0.0]),
                (hit, [0.0, -0.10, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.30, 0.0, -0.12]),
                (wind, [-1.25, 0.85, -0.15]),
                (hit, [-1.40, 0.05, -0.25]),
                (follow, [-1.15, -0.70, -0.35]),
                (end, [-0.30, 0.0, -0.12]),
            ],
        )
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (wind, [-1.15, 0.0, 0.0]),
                (hit, [-0.30, 0.0, 0.0]),
                (follow, [-0.20, 0.0, 0.0]),
                (end, [-0.55, 0.0, 0.0]),
            ],
        )
        // The backhand: the wrist rolls the blade over during the wind-up and holds it
        // inverted through the sweep.
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, -0.15]),
                (wind, [0.0, -0.05, 0.90]),
                (follow, [0.0, -0.05, 0.90]),
                (end, [0.0, -0.20, -0.15]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.08, 0.0, 0.10]),
                (wind, [-0.20, 0.0, 0.50]),
                (follow, [0.40, 0.0, 0.30]),
                (end, [0.08, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.25, 0.0, 0.0]),
                (hit, [-0.60, 0.0, 0.0]),
                (end, [-0.25, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (wind, [-0.16, 0.0, 0.0]),
                (follow, [0.12, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (wind, [0.12, 0.0, 0.0]),
                (follow, [-0.10, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .finish()
}

/// Combo finisher: a two-handed overhead slam.
///
/// The shoulder travels *through the front* on the way up (pitch going negative past
/// −π/2 = horizontal-forward, on to ≈ −3.25 = above and slightly behind) and retraces
/// the same arc coming down, which is the path an overhead swing actually takes; taking
/// the short way round would swing the blade up the character's back. Slow, with the
/// longest wind-up of the three — the tell that makes it dodgeable.
fn warrior_attack3(b: &WarriorBones) -> GlbAnimation {
    let (raise, top, hit, low, end) = (
        0.16,
        0.30,
        WARRIOR_ATTACK3_HIT_TIME,
        0.58,
        WARRIOR_ATTACK3_LEN,
    );
    Clip::new("attack3")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (top, [0.0, 0.915, 0.0]),
                (hit, [0.0, 0.845, 0.0]),
                (low, [0.0, 0.860, 0.0]),
                (end, [0.0, 0.900, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (top, [0.0, -0.10, 0.0]),
                (hit, [0.0, 0.04, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.02, 0.0, 0.0]),
                (top, [-0.35, -0.10, 0.0]),
                (hit, [0.55, 0.02, 0.0]),
                (low, [0.45, 0.0, 0.0]),
                (end, [0.03, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (top, [-0.22, 0.0, 0.0]),
                (hit, [0.25, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.35, 0.0, -0.12]),
                (raise, [-1.45, 0.0, -0.20]),
                (top, [-3.25, 0.0, -0.25]),
                (hit, [-1.20, 0.0, -0.15]),
                (low, [-0.30, 0.0, -0.10]),
                (end, [-0.32, 0.0, -0.12]),
            ],
        )
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (top, [-1.55, 0.0, 0.0]),
                (hit, [-0.15, 0.0, 0.0]),
                (low, [-0.20, 0.0, 0.0]),
                (end, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, 0.0]),
                (top, [0.0, -0.05, 0.0]),
                (hit, [0.0, -0.05, 0.0]),
                (end, [0.0, -0.20, 0.0]),
            ],
        )
        // The off hand comes along for the ride — the read that says "two-handed".
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.08, 0.0, 0.10]),
                (raise, [-1.20, 0.0, 0.25]),
                (top, [-2.90, 0.0, 0.30]),
                (hit, [-1.10, 0.0, 0.20]),
                (low, [-0.25, 0.0, 0.12]),
                (end, [0.08, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.25, 0.0, 0.0]),
                (top, [-1.35, 0.0, 0.0]),
                (hit, [-0.20, 0.0, 0.0]),
                (end, [-0.25, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (top, [0.20, 0.0, 0.0]),
                (hit, [-0.30, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (hit, [0.30, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (top, [-0.15, 0.0, 0.0]),
                (hit, [0.22, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (hit, [0.20, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .finish()
}

/// The evasive roll: pitch forward, tuck, come up.
///
/// The character does not travel here — the mover does that (in-place rule) — so the
/// whole read is the crouch depth and the tuck. Fast: the i-frame window
/// ([`WARRIOR_DODGE_IFRAME_START`]..[`WARRIOR_DODGE_IFRAME_END`]) is the crouch the clip
/// holds, not a number invented next to it.
fn warrior_dodge(b: &WarriorBones) -> GlbAnimation {
    let (drop, low, end) = (0.12, 0.22, WARRIOR_DODGE_LEN);
    Clip::new("dodge")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (drop, [0.0, 0.660, 0.0]),
                (low, [0.0, 0.620, 0.0]),
                (end, [0.0, 0.880, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [0.55, 0.0, 0.0]),
                (low, [0.75, 0.0, 0.0]),
                (end, [0.10, 0.0, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.05, 0.0, 0.0]),
                (drop, [0.65, 0.0, 0.0]),
                (low, [0.80, 0.0, 0.0]),
                (end, [0.10, 0.0, 0.0]),
            ],
        )
        // Chin tucked, so the crest sweeps forward instead of the helm burying itself.
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (low, [-0.55, 0.0, 0.0]),
                (end, [-0.05, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [-0.85, 0.0, 0.0]),
                (low, [-1.10, 0.0, 0.0]),
                (end, [-0.10, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [1.10, 0.0, 0.0]),
                (low, [1.35, 0.0, 0.0]),
                (end, [0.12, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [-0.75, 0.0, 0.0]),
                (low, [-1.05, 0.0, 0.0]),
                (end, [-0.08, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [1.00, 0.0, 0.0]),
                (low, [1.30, 0.0, 0.0]),
                (end, [0.10, 0.0, 0.0]),
            ],
        )
        .rot(
            b.foot_l,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (low, [0.45, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.foot_r,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (low, [0.42, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.05, 0.0, 0.08]),
                (low, [-0.55, 0.0, 0.30]),
                (end, [0.05, 0.0, 0.08]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.25, 0.0, 0.0]),
                (low, [-1.45, 0.0, 0.0]),
                (end, [-0.25, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.30, 0.0, -0.10]),
                (low, [-0.65, 0.0, -0.35]),
                (end, [-0.30, 0.0, -0.10]),
            ],
        )
        // Blade swept out to the side while tucked, so it does not read as stabbing the
        // floor from above.
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (low, [-1.55, 0.0, 0.0]),
                (end, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, 0.0]),
                (low, [0.0, -0.75, 0.0]),
                (end, [0.0, -0.20, 0.0]),
            ],
        )
        .finish()
}

/// A flinch: the hit lands almost immediately, the recovery takes the rest. Short enough
/// to interrupt anything without stealing a whole beat of control.
fn warrior_hit(b: &WarriorBones) -> GlbAnimation {
    let (snap, settle, end) = (0.07, 0.16, WARRIOR_HIT_LEN);
    Clip::new("hit")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (snap, [0.0, 0.878, 0.0]),
                (end, [0.0, 0.900, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (snap, [0.0, 0.10, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.03, 0.0, 0.0]),
                (snap, [-0.32, 0.12, 0.0]),
                (settle, [0.10, -0.05, 0.0]),
                (end, [0.02, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (snap, [-0.40, 0.15, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.08, 0.0, 0.10]),
                (snap, [0.45, 0.0, 0.40]),
                (end, [0.08, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.30, 0.0, -0.12]),
                (snap, [0.15, 0.0, -0.40]),
                (end, [-0.30, 0.0, -0.12]),
            ],
        )
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (snap, [-0.30, 0.0, 0.0]),
                (end, [-0.55, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (snap, [0.18, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (snap, [-0.14, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .finish()
}

/// The collapse: the whole rig rolls off its feet onto its side and stays there.
///
/// The fall is a **root roll** — a rotation, never a translation, so even death obeys the
/// in-place rule and the corpse ends up wherever the mover last put it. Every channel's
/// last two keyframes are identical, so the final pose is held from
/// [`WARRIOR_DEATH_HOLD_TIME`] onward: a player that keeps looping still rests on the
/// same corpse instead of snapping upright halfway through the loop.
fn warrior_death(b: &WarriorBones) -> GlbAnimation {
    let (buckle, fall, land, hold, end) =
        (0.22, 0.55, 0.85, WARRIOR_DEATH_HOLD_TIME, WARRIOR_DEATH_LEN);
    Clip::new("death")
        .rot(
            b.root,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [0.0, 0.0, -0.12]),
                (fall, [0.0, 0.0, -0.70]),
                (land, [0.0, 0.0, -1.42]),
                (hold, [0.0, 0.0, -1.50]),
                (end, [0.0, 0.0, -1.50]),
            ],
        )
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.900, 0.0]),
                (buckle, [0.0, 0.820, 0.0]),
                (fall, [0.0, 0.740, 0.0]),
                (hold, [0.0, 0.680, 0.0]),
                (end, [0.0, 0.680, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.02, 0.0, 0.0]),
                (buckle, [-0.25, 0.0, 0.0]),
                (fall, [0.35, 0.15, 0.0]),
                (hold, [0.55, 0.25, 0.10]),
                (end, [0.55, 0.25, 0.10]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [-0.35, 0.0, 0.0]),
                (fall, [0.30, -0.20, 0.0]),
                (hold, [0.45, -0.30, 0.0]),
                (end, [0.45, -0.30, 0.0]),
            ],
        )
        .rot(
            b.arm_l_upper,
            &[
                (0.0, [0.08, 0.0, 0.10]),
                (buckle, [0.55, 0.0, 0.35]),
                (fall, [0.20, 0.0, 0.60]),
                (hold, [-0.10, 0.0, 0.75]),
                (end, [-0.10, 0.0, 0.75]),
            ],
        )
        .rot(
            b.arm_l_lower,
            &[
                (0.0, [-0.25, 0.0, 0.0]),
                (fall, [-0.55, 0.0, 0.0]),
                (hold, [-0.15, 0.0, 0.0]),
                (end, [-0.15, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r_upper,
            &[
                (0.0, [-0.30, 0.0, -0.12]),
                (buckle, [0.10, 0.0, -0.30]),
                (fall, [0.35, 0.0, -0.55]),
                (hold, [0.20, 0.0, -0.70]),
                (end, [0.20, 0.0, -0.70]),
            ],
        )
        // The grip goes slack — the blade ends up lying beside the body, which is the
        // clearest "this one is done" read from above.
        .rot(
            b.arm_r_lower,
            &[
                (0.0, [-0.55, 0.0, 0.0]),
                (fall, [-0.20, 0.0, 0.0]),
                (hold, [-0.05, 0.0, 0.0]),
                (end, [-0.05, 0.0, 0.0]),
            ],
        )
        .rot(
            b.hand_r,
            &[
                (0.0, [0.0, -0.20, 0.0]),
                (fall, [0.0, -0.55, 0.0]),
                (hold, [0.0, -0.80, 0.0]),
                (end, [0.0, -0.80, 0.0]),
            ],
        )
        .rot(
            b.leg_l_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [-0.45, 0.0, 0.10]),
                (fall, [-0.85, 0.0, 0.20]),
                (hold, [-0.95, 0.0, 0.22]),
                (end, [-0.95, 0.0, 0.22]),
            ],
        )
        .rot(
            b.leg_l_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [0.60, 0.0, 0.0]),
                (fall, [1.05, 0.0, 0.0]),
                (hold, [1.15, 0.0, 0.0]),
                (end, [1.15, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r_upper,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [-0.30, 0.0, -0.08]),
                (fall, [-0.65, 0.0, -0.14]),
                (hold, [-0.75, 0.0, -0.16]),
                (end, [-0.75, 0.0, -0.16]),
            ],
        )
        .rot(
            b.leg_r_lower,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (buckle, [0.45, 0.0, 0.0]),
                (fall, [0.85, 0.0, 0.0]),
                (hold, [0.95, 0.0, 0.0]),
                (end, [0.95, 0.0, 0.0]),
            ],
        )
        .rot(
            b.foot_l,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (hold, [0.35, 0.0, 0.0]),
                (end, [0.35, 0.0, 0.0]),
            ],
        )
        .rot(
            b.foot_r,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (hold, [0.28, 0.0, 0.0]),
                (end, [0.28, 0.0, 0.0]),
            ],
        )
        .finish()
}

// --- The grunt -----------------------------------------------------------------------

/// Standing height of the grunt, metres — a head shorter than the knight, which is the
/// cheapest way to make "this one is not you" legible in a crowd.
const GRUNT_HEIGHT: f32 = 1.47;
/// Rest height of the grunt's pelvis joint, metres.
const GRUNT_HIP_Y: f32 = 0.70;

/// The grunt's node indices.
struct GruntBones {
    root: usize,
    pelvis: usize,
    torso: usize,
    head: usize,
    arm_l: usize,
    arm_r: usize,
    leg_l: usize,
    leg_r: usize,
}

/// The skeletal minion: ~1.5 m, eight boxes, single-segment limbs.
///
/// No knees or elbows on purpose — a minion is seen in numbers, from above, for a couple
/// of seconds each. Limbs swinging from the hip and shoulder read perfectly well at that
/// distance, and the halved bone count is halved draw calls per monster.
pub fn grunt() -> Rig {
    let mut b = RigBuilder::new();

    let root = b.joint("grunt_root", None, [0.0, 0.0, 0.0]);
    let pelvis = b.bone(
        "pelvis",
        root,
        [0.0, GRUNT_HIP_Y, 0.0],
        MAT_BONE,
        &[([-0.13, -0.07, -0.09], [0.13, 0.07, 0.09])],
    );
    let torso = b.bone(
        "torso",
        pelvis,
        [0.0, 0.07, 0.0],
        MAT_BONE,
        &[([-0.155, 0.0, -0.10], [0.155, 0.40, 0.10])],
    );
    // The dark core sticks out fore and aft: the one asymmetric detail that tells you
    // which way a bone-white silhouette is facing from directly overhead.
    b.bone(
        "core",
        torso,
        [0.0, 0.18, 0.0],
        MAT_GRUNT_DARK,
        &[([-0.05, -0.06, -0.115], [0.05, 0.06, 0.115])],
    );
    let head = b.bone(
        "head",
        torso,
        [0.0, 0.42, 0.0],
        MAT_BONE,
        &[([-0.105, 0.0, -0.11], [0.105, 0.28, 0.11])],
    );

    let arm: BoxSpan = ([-0.045, -0.36, -0.045], [0.045, 0.0, 0.045]);
    let leg: BoxSpan = ([-0.055, -0.65, -0.055], [0.055, 0.0, 0.055]);
    let arm_l = b.bone("arm_l", torso, [0.17, 0.36, 0.0], MAT_BONE, &[arm]);
    let arm_r = b.bone("arm_r", torso, [-0.17, 0.36, 0.0], MAT_BONE, &[flip_x(arm)]);
    let leg_l = b.bone("leg_l", pelvis, [0.085, -0.05, 0.0], MAT_BONE, &[leg]);
    let leg_r = b.bone(
        "leg_r",
        pelvis,
        [-0.085, -0.05, 0.0],
        MAT_BONE,
        &[flip_x(leg)],
    );

    let bones = GruntBones {
        root,
        pelvis,
        torso,
        head,
        arm_l,
        arm_r,
        leg_l,
        leg_r,
    };
    let animations = grunt_clips(&bones);

    Rig {
        name: "grunt",
        nodes: b.nodes,
        meshes: b.meshes,
        materials: grunt_materials(),
        animations,
    }
}

/// The grunt's five clips. Order matches [`GRUNT_CLIPS`].
fn grunt_clips(b: &GruntBones) -> Vec<GlbAnimation> {
    vec![
        grunt_idle(b),
        grunt_walk(b),
        grunt_attack(b),
        grunt_hit(b),
        grunt_death(b),
    ]
}

/// A slow sway — deliberately looser and lower-frequency than the knight's breathing, so
/// an idle crowd of these does not pulse in unison with the player.
fn grunt_idle(b: &GruntBones) -> GlbAnimation {
    let (mid, end) = (1.0, GRUNT_IDLE_LEN);
    Clip::new("idle")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.700, 0.0]),
                (0.5, [0.0, 0.712, 0.0]),
                (mid, [0.0, 0.718, 0.0]),
                (1.5, [0.0, 0.710, 0.0]),
                (end, [0.0, 0.700, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, -0.05, 0.0]),
                (mid, [0.0, 0.05, 0.0]),
                (end, [0.0, -0.05, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.03, 0.06, 0.0]),
                (mid, [0.06, -0.06, 0.0]),
                (end, [0.03, 0.06, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.05, 0.10, 0.0]),
                (mid, [0.10, -0.12, 0.0]),
                (end, [0.05, 0.10, 0.0]),
            ],
        )
        .rot(
            b.arm_l,
            &[
                (0.0, [0.05, 0.0, 0.10]),
                (mid, [0.18, 0.0, 0.16]),
                (end, [0.05, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_r,
            &mirrored(&[
                (0.0, [0.05, 0.0, 0.10]),
                (mid, [0.16, 0.0, 0.14]),
                (end, [0.05, 0.0, 0.10]),
            ]),
        )
        .finish()
}

/// A walk cycle at half the knight's cadence — the speed difference is the tell that
/// says you can outrun it.
fn grunt_walk(b: &GruntBones) -> GlbAnimation {
    let t = GRUNT_WALK_LEN;
    let (q1, q2, q3) = (t * 0.25, t * 0.5, t * 0.75);
    let leg = [
        (0.0, [-0.55, 0.0, 0.0]),
        (q1, [-0.05, 0.0, 0.0]),
        (q2, [0.45, 0.0, 0.0]),
        (q3, [-0.20, 0.0, 0.0]),
        (t, [-0.55, 0.0, 0.0]),
    ];
    let arm = [
        (0.0, [0.45, 0.0, 0.12]),
        (q2, [-0.50, 0.0, 0.16]),
        (t, [0.45, 0.0, 0.12]),
    ];
    Clip::new("walk")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.686, 0.0]),
                (q1, [0.0, 0.706, 0.0]),
                (q2, [0.0, 0.686, 0.0]),
                (q3, [0.0, 0.706, 0.0]),
                (t, [0.0, 0.686, 0.0]),
            ],
        )
        .rot(
            b.pelvis,
            &[
                (0.0, [0.0, -0.14, 0.0]),
                (q2, [0.0, 0.14, 0.0]),
                (t, [0.0, -0.14, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.14, 0.18, 0.0]),
                (q2, [0.14, -0.18, 0.0]),
                (t, [0.14, 0.18, 0.0]),
            ],
        )
        .rot(b.head, &[(0.0, [-0.10, 0.0, 0.0]), (t, [-0.10, 0.0, 0.0])])
        .rot(b.leg_l, &leg)
        .rot(b.leg_r, &half_phase(&leg, t))
        .rot(b.arm_l, &half_phase(&arm, t))
        .rot(b.arm_r, &mirrored(&arm))
        .finish()
}

/// A lunge and a two-armed claw: coil back, throw the whole body forward, both arms
/// reaching past the hips at [`GRUNT_ATTACK_HIT_TIME`].
fn grunt_attack(b: &GruntBones) -> GlbAnimation {
    let (coil, hit, recover, end) = (0.16, GRUNT_ATTACK_HIT_TIME, 0.5, GRUNT_ATTACK_LEN);
    Clip::new("attack")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.700, 0.0]),
                (coil, [0.0, 0.660, 0.0]),
                (hit, [0.0, 0.740, 0.0]),
                (recover, [0.0, 0.700, 0.0]),
                (end, [0.0, 0.700, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.05, 0.0, 0.0]),
                (coil, [-0.30, 0.15, 0.0]),
                (hit, [0.55, -0.10, 0.0]),
                (recover, [0.30, 0.0, 0.0]),
                (end, [0.05, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (coil, [-0.35, 0.0, 0.0]),
                (hit, [0.40, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_r,
            &[
                (0.0, [0.05, 0.0, -0.10]),
                (coil, [0.85, 0.0, -0.55]),
                (hit, [-1.55, 0.0, -0.25]),
                (recover, [-0.90, 0.0, -0.15]),
                (end, [0.05, 0.0, -0.10]),
            ],
        )
        .rot(
            b.arm_l,
            &[
                (0.0, [0.05, 0.0, 0.10]),
                (coil, [0.55, 0.0, 0.45]),
                (hit, [-1.20, 0.0, 0.20]),
                (recover, [-0.70, 0.0, 0.14]),
                (end, [0.05, 0.0, 0.10]),
            ],
        )
        .rot(
            b.leg_l,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (coil, [0.25, 0.0, 0.0]),
                (hit, [-0.35, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.leg_r,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (coil, [0.15, 0.0, 0.0]),
                (hit, [0.30, 0.0, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .finish()
}

/// A short flinch — long enough to be seen from above, short enough that a swarm still
/// feels dangerous while it is being hit.
fn grunt_hit(b: &GruntBones) -> GlbAnimation {
    let (snap, settle, end) = (0.06, 0.14, GRUNT_HIT_LEN);
    Clip::new("hit")
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.700, 0.0]),
                (snap, [0.0, 0.678, 0.0]),
                (end, [0.0, 0.700, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.03, 0.0, 0.0]),
                (snap, [-0.40, 0.14, 0.0]),
                (settle, [0.12, -0.05, 0.0]),
                (end, [0.03, 0.0, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (snap, [-0.45, 0.18, 0.0]),
                (end, [0.0, 0.0, 0.0]),
            ],
        )
        .rot(
            b.arm_l,
            &[
                (0.0, [0.05, 0.0, 0.10]),
                (snap, [0.50, 0.0, 0.42]),
                (end, [0.05, 0.0, 0.10]),
            ],
        )
        .rot(
            b.arm_r,
            &mirrored(&[
                (0.0, [0.05, 0.0, 0.10]),
                (snap, [0.45, 0.0, 0.38]),
                (end, [0.05, 0.0, 0.10]),
            ]),
        )
        .finish()
}

/// The crumple: legs fold, the body pitches forward and settles on its face, held from
/// [`GRUNT_DEATH_HOLD_TIME`] on. Forward rather than sideways, so a knot of dead grunts
/// reads differently from a dead knight lying on its side.
fn grunt_death(b: &GruntBones) -> GlbAnimation {
    let (fold, drop, hold, end) = (0.25, 0.55, GRUNT_DEATH_HOLD_TIME, GRUNT_DEATH_LEN);
    Clip::new("death")
        .rot(
            b.root,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (fold, [0.0, 0.0, 0.10]),
                (drop, [0.35, 0.0, 0.30]),
                (hold, [0.55, 0.0, 0.42]),
                (end, [0.55, 0.0, 0.42]),
            ],
        )
        .pos(
            b.pelvis,
            &[
                (0.0, [0.0, 0.700, 0.0]),
                (fold, [0.0, 0.520, 0.0]),
                (drop, [0.0, 0.280, 0.0]),
                (hold, [0.0, 0.160, 0.0]),
                (end, [0.0, 0.160, 0.0]),
            ],
        )
        .rot(
            b.torso,
            &[
                (0.0, [0.03, 0.0, 0.0]),
                (fold, [0.35, 0.0, 0.0]),
                (drop, [0.85, 0.15, 0.0]),
                (hold, [1.05, 0.20, 0.0]),
                (end, [1.05, 0.20, 0.0]),
            ],
        )
        .rot(
            b.head,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (drop, [0.45, -0.20, 0.0]),
                (hold, [0.65, -0.25, 0.0]),
                (end, [0.65, -0.25, 0.0]),
            ],
        )
        .rot(
            b.arm_l,
            &[
                (0.0, [0.05, 0.0, 0.10]),
                (fold, [0.40, 0.0, 0.30]),
                (drop, [0.10, 0.0, 0.55]),
                (hold, [-0.15, 0.0, 0.65]),
                (end, [-0.15, 0.0, 0.65]),
            ],
        )
        .rot(
            b.arm_r,
            &[
                (0.0, [0.05, 0.0, -0.10]),
                (fold, [0.35, 0.0, -0.28]),
                (drop, [0.05, 0.0, -0.50]),
                (hold, [-0.20, 0.0, -0.60]),
                (end, [-0.20, 0.0, -0.60]),
            ],
        )
        .rot(
            b.leg_l,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (fold, [-0.55, 0.0, 0.12]),
                (drop, [-1.15, 0.0, 0.25]),
                (hold, [-1.35, 0.0, 0.28]),
                (end, [-1.35, 0.0, 0.28]),
            ],
        )
        .rot(
            b.leg_r,
            &[
                (0.0, [0.0, 0.0, 0.0]),
                (fold, [-0.45, 0.0, -0.10]),
                (drop, [-1.00, 0.0, -0.22]),
                (hold, [-1.20, 0.0, -0.25]),
                (end, [-1.20, 0.0, -0.25]),
            ],
        )
        .finish()
}

// --- Files ---------------------------------------------------------------------------

/// Path of a rig's `.glb` inside the generated-asset directory.
pub fn rig_asset_path(name: &str) -> PathBuf {
    Path::new(crate::level::GENERATED_DIR).join(format!("{name}.glb"))
}

/// Author both characters and write them next to the generated levels, rewriting only
/// what changed.
///
/// The same road the dungeon geometry takes (`crate::level`): a real file, cooked and
/// content-hash cached, so an unchanged authoring pass is a pure cache hit. Returns the
/// written paths — placing them in a level and driving their clips is the integrator's
/// job, not this module's.
pub fn ensure_rigs() -> anyhow::Result<Vec<PathBuf>> {
    let started = std::time::Instant::now();
    let mut paths = Vec::new();
    for rig in [warrior(), grunt()] {
        let path = rig_asset_path(rig.name);
        let wrote = save_glb_scene(
            &path,
            &rig.nodes,
            &rig.meshes,
            &rig.materials,
            &rig.animations,
        )?;
        tracing::info!(
            "dungeon: rig '{}' — {} nodes, {} meshes, {} triangles, {} clips, {} in {:.1} ms",
            path.display(),
            rig.nodes.len(),
            rig.meshes.len(),
            rig.triangles(),
            rig.animations.len(),
            if wrote { "written" } else { "unchanged" },
            started.elapsed().as_secs_f64() * 1e3,
        );
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sandbox::scene::{
        AnimationClip, AnimationPlayer, LocalTransform, MaterialHandle, MeshHandle, World,
        advance_animation, instantiate_gltf_mapped,
    };
    use std::collections::BTreeSet;

    /// A self-cleaning scratch directory (the round trip goes through `gltf::import`,
    /// which takes a path).
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let id = format!("{:?}", std::thread::current().id());
            let id: String = id.chars().filter(|c| c.is_alphanumeric()).collect();
            let dir = std::env::temp_dir().join(format!("dungeon-rigs-{tag}-{id}"));
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

    fn rigs() -> [Rig; 2] {
        [warrior(), grunt()]
    }

    /// Write a rig and read it back with the engine's own importer.
    fn round_trip(rig: &Rig, dir: &TempDir) -> dreamcoast_asset::GltfScene {
        let path = dir.0.join(format!("{}.glb", rig.name));
        let bytes = dreamcoast_asset::glb::write_glb_scene(
            &rig.nodes,
            &rig.meshes,
            &rig.materials,
            &rig.animations,
        )
        .expect("write");
        std::fs::write(&path, &bytes).unwrap();
        dreamcoast_asset::load_gltf_scene(&path).expect("import")
    }

    /// The definition of done: both rigs survive the engine's importer with their
    /// skeleton, their clip list and their keyframes intact.
    #[test]
    fn rigs_round_trip_through_the_engine_importer() {
        let dir = TempDir::new("roundtrip");
        for (rig, clips) in [
            (warrior(), WARRIOR_CLIPS.as_slice()),
            (grunt(), GRUNT_CLIPS.as_slice()),
        ] {
            let scene = round_trip(&rig, &dir);
            let what = rig.name;

            // Node names and order survive — they are how gameplay finds a bone.
            let names: Vec<&str> = scene
                .nodes
                .iter()
                .map(|n| n.name.as_deref().unwrap_or(""))
                .collect();
            let authored: Vec<&str> = rig.nodes.iter().map(|n| n.name.as_str()).collect();
            assert_eq!(names, authored, "{what}: node names");
            assert_eq!(scene.roots, vec![0], "{what}: one rig root");
            assert_eq!(scene.meshes.len(), rig.meshes.len(), "{what}: mesh count");

            // The tree: every authored parent link came back as a child entry.
            for (i, node) in rig.nodes.iter().enumerate() {
                match node.parent {
                    Some(parent) => assert!(
                        scene.nodes[parent].children.contains(&i),
                        "{what}: '{}' lost its parent link",
                        node.name
                    ),
                    None => assert_eq!(i, 0, "{what}: only the root is parentless"),
                }
                assert_eq!(scene.nodes[i].translation, node.translation, "{what}");
                assert_eq!(scene.nodes[i].mesh, node.mesh, "{what}");
            }

            // Clips: the authored list, in order, each with a duration that matches the
            // exported `_LEN` const (the clip's own last keyframe).
            let imported: Vec<&str> = scene
                .animations
                .iter()
                .map(|a| a.name.as_deref().unwrap_or(""))
                .collect();
            assert_eq!(imported, clips, "{what}: clip list");
            for (clip, authored) in scene.animations.iter().zip(&rig.animations) {
                assert!(!clip.channels.is_empty(), "{what}: empty clip");
                assert_eq!(
                    clip.channels.len(),
                    authored.channels.len(),
                    "{what}: channel count of '{}'",
                    authored.name
                );
                let expected: f32 = authored
                    .channels
                    .iter()
                    .map(|c| c.keys.last().unwrap().0)
                    .fold(0.0, f32::max);
                assert_eq!(
                    clip.duration, expected,
                    "{what}: duration of '{}'",
                    authored.name
                );
                // Every channel targets a node that exists, and keeps its keyframes.
                for (imported, authored) in clip.channels.iter().zip(&authored.channels) {
                    assert_eq!(imported.target_node, authored.node, "{what}");
                    assert_eq!(imported.times.len(), authored.keys.len(), "{what}");
                    assert_eq!(
                        imported.interpolation,
                        dreamcoast_asset::Interpolation::Linear,
                        "{what}: clips are authored linear"
                    );
                }
            }
        }
    }

    /// The exported durations are the clips' own — a const that drifted from its
    /// animation would be worse than no const at all.
    #[test]
    fn clip_metadata_matches_the_clips() {
        let warrior = warrior();
        let expected = |name: &str| -> f32 {
            warrior
                .animations
                .iter()
                .find(|a| a.name == name)
                .unwrap()
                .channels
                .iter()
                .map(|c| c.keys.last().unwrap().0)
                .fold(0.0, f32::max)
        };
        for (name, len) in [
            ("idle", WARRIOR_IDLE_LEN),
            ("run", WARRIOR_RUN_LEN),
            ("attack1", WARRIOR_ATTACK1_LEN),
            ("attack2", WARRIOR_ATTACK2_LEN),
            ("attack3", WARRIOR_ATTACK3_LEN),
            ("dodge", WARRIOR_DODGE_LEN),
            ("hit", WARRIOR_HIT_LEN),
            ("death", WARRIOR_DEATH_LEN),
        ] {
            assert_eq!(expected(name), len, "warrior '{name}' duration");
        }

        let grunt = grunt();
        let grunt_len = |name: &str| -> f32 {
            grunt
                .animations
                .iter()
                .find(|a| a.name == name)
                .unwrap()
                .channels
                .iter()
                .map(|c| c.keys.last().unwrap().0)
                .fold(0.0, f32::max)
        };
        for (name, len) in [
            ("idle", GRUNT_IDLE_LEN),
            ("walk", GRUNT_WALK_LEN),
            ("attack", GRUNT_ATTACK_LEN),
            ("hit", GRUNT_HIT_LEN),
            ("death", GRUNT_DEATH_LEN),
        ] {
            assert_eq!(grunt_len(name), len, "grunt '{name}' duration");
        }

        // Every marker falls strictly inside its clip — a hit time at or past the end
        // would never fire.
        for (marker, len) in [
            (WARRIOR_ATTACK1_HIT_TIME, WARRIOR_ATTACK1_LEN),
            (WARRIOR_ATTACK2_HIT_TIME, WARRIOR_ATTACK2_LEN),
            (WARRIOR_ATTACK3_HIT_TIME, WARRIOR_ATTACK3_LEN),
            (WARRIOR_DODGE_IFRAME_START, WARRIOR_DODGE_LEN),
            (WARRIOR_DODGE_IFRAME_END, WARRIOR_DODGE_LEN),
            (WARRIOR_DEATH_HOLD_TIME, WARRIOR_DEATH_LEN),
            (GRUNT_ATTACK_HIT_TIME, GRUNT_ATTACK_LEN),
            (GRUNT_DEATH_HOLD_TIME, GRUNT_DEATH_LEN),
        ] {
            assert!(marker > 0.0 && marker < len, "marker {marker} vs {len}");
        }
        const { assert!(WARRIOR_DODGE_IFRAME_START < WARRIOR_DODGE_IFRAME_END) };
    }

    /// **In place**: no bone ever moves in X or Z. Only the vertical bob is animated,
    /// and only away from a node's own rest position — the mover owns world position
    /// (plan §4.1, root motion is out of scope).
    #[test]
    fn clips_are_in_place() {
        for rig in rigs() {
            for clip in &rig.animations {
                for channel in &clip.channels {
                    let rest = rig.nodes[channel.node].translation;
                    for (time, value) in &channel.keys {
                        if let dreamcoast_asset::glb::GlbValue::Vec3(v) = value {
                            assert_eq!(
                                (v[0], v[2]),
                                (rest[0], rest[2]),
                                "{}/{} at t={time}: '{}' translated horizontally",
                                rig.name,
                                clip.name,
                                rig.nodes[channel.node].name
                            );
                        }
                    }
                }
            }
            // The root itself is never translated at all, in any axis.
            for clip in &rig.animations {
                for channel in &clip.channels {
                    assert!(
                        channel.node != 0
                            || channel.path != dreamcoast_asset::glb::GlbPath::Translation,
                        "{}/{}: the rig root must not translate",
                        rig.name,
                        clip.name
                    );
                }
            }
        }
    }

    /// The death clips hold their final pose: every channel's last two keys are equal,
    /// so playback past the settle time shows the same corpse.
    #[test]
    fn death_clips_hold_their_final_pose() {
        for (rig, hold) in [
            (warrior(), WARRIOR_DEATH_HOLD_TIME),
            (grunt(), GRUNT_DEATH_HOLD_TIME),
        ] {
            let death = rig
                .animations
                .iter()
                .find(|a| a.name == "death")
                .expect("a death clip");
            for channel in &death.channels {
                let n = channel.keys.len();
                assert!(n >= 2, "{}: single-key death channel", rig.name);
                assert_eq!(
                    channel.keys[n - 2].1,
                    channel.keys[n - 1].1,
                    "{}: '{}' still moves after the settle",
                    rig.name,
                    rig.nodes[channel.node].name
                );
                assert!(
                    channel.keys[n - 2].0 <= hold,
                    "{}: settle later than the exported hold time",
                    rig.name
                );
            }
        }
    }

    /// Looping clips close on themselves: the first and last keyframe of every channel
    /// of a looping clip carry the same value, or the loop point pops.
    #[test]
    fn looping_clips_close_on_themselves() {
        for (rig, looping) in [
            (warrior(), ["idle", "run"].as_slice()),
            (grunt(), ["idle", "walk"].as_slice()),
        ] {
            for clip in rig
                .animations
                .iter()
                .filter(|a| looping.contains(&&*a.name))
            {
                for channel in &clip.channels {
                    let (first, last) =
                        (channel.keys.first().unwrap(), channel.keys.last().unwrap());
                    assert_eq!(
                        first.1, last.1,
                        "{}/{}: '{}' does not close its loop",
                        rig.name, clip.name, rig.nodes[channel.node].name
                    );
                }
            }
        }
    }

    /// The rigs stand on the floor at the sizes the game's collision assumes, and every
    /// triangle is wound counter-clockwise about its own normal (the workspace's
    /// front-face convention — see `crate::level`'s winding test for the consequences of
    /// mixing).
    #[test]
    fn rig_geometry_is_grounded_and_outward_facing() {
        for (rig, height, hip) in [
            (warrior(), WARRIOR_HEIGHT, WARRIOR_HIP_Y),
            (grunt(), GRUNT_HEIGHT, GRUNT_HIP_Y),
        ] {
            let what = rig.name;
            // Names are unique, so a bone lookup by name is unambiguous.
            let names: BTreeSet<&str> = rig.nodes.iter().map(|n| n.name.as_str()).collect();
            assert_eq!(
                names.len(),
                rig.nodes.len(),
                "{what}: node names are unique"
            );

            for mesh in &rig.meshes {
                assert!(mesh.material < rig.materials.len(), "{what}/{}", mesh.name);
                for tri in mesh.indices.chunks_exact(3) {
                    let p: Vec<Vec3> = tri
                        .iter()
                        .map(|&i| Vec3::from(mesh.vertices[i as usize].pos))
                        .collect();
                    let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
                    assert!(
                        geometric.length() > 1e-9,
                        "{what}/{}: degenerate triangle",
                        mesh.name
                    );
                    let authored = Vec3::from(mesh.vertices[tri[0] as usize].normal);
                    assert!(
                        geometric.normalize().dot(authored) > 0.99,
                        "{what}/{}: face winding disagrees with its normal",
                        mesh.name
                    );
                }
            }

            // Rest-pose extent, from the world position of each node's geometry.
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            for (i, node) in rig.nodes.iter().enumerate() {
                let Some(mesh) = node.mesh else { continue };
                // Rest rotations are identity, so a node's world offset is the sum of
                // its ancestors' translations.
                let mut offset = Vec3::ZERO;
                let mut at = Some(i);
                while let Some(n) = at {
                    offset += Vec3::from(rig.nodes[n].translation);
                    at = rig.nodes[n].parent;
                }
                for v in &rig.meshes[mesh].vertices {
                    let p = offset + Vec3::from(v.pos);
                    min = min.min(p);
                    max = max.max(p);
                }
            }
            // Within f32 rounding of the joint chain that sums to it (hip − thigh −
            // shin − ankle − sole), which lands ~3e-8 m off zero.
            assert!(
                min.y.abs() < 1e-6,
                "{what}: the soles stand on y = 0, not {}",
                min.y
            );
            assert!(
                (max.y - height).abs() < 0.12,
                "{what}: stands {} m, expected ~{height}",
                max.y
            );
            assert_eq!(rig.nodes[1].translation[1], hip, "{what}: hip height");
            assert!(
                max.x < 0.6 && min.x > -0.6,
                "{what}: unexpectedly wide ({min} .. {max})"
            );
        }
    }

    /// The whole M2 road in one test: written glTF → imported scene → ECS sub-tree →
    /// `AnimationClip::from_gltf` → `advance_animation` moving a real bone entity.
    #[test]
    fn clips_drive_the_ecs_through_the_scene_crate() {
        let dir = TempDir::new("ecs");
        let rig = warrior();
        let scene = round_trip(&rig, &dir);

        let mut world = World::new();
        let handles: Vec<Vec<(MeshHandle, MaterialHandle)>> = (0..scene.meshes.len())
            .map(|i| vec![(MeshHandle(i as u32), MaterialHandle(0))])
            .collect();
        let (root, map) = instantiate_gltf_mapped(&mut world, &scene, &handles);
        assert!(
            map.iter().all(Option::is_some),
            "every node became an entity"
        );
        assert_eq!(map[0], Some(root), "the rig root is the sub-tree root");

        // The sword arm's elbow: a bone `attack1` definitely animates.
        let elbow_node = rig
            .nodes
            .iter()
            .position(|n| n.name == "arm_r_lower")
            .unwrap();
        let elbow = map[elbow_node].unwrap();
        let rest = world.get::<LocalTransform>(elbow).unwrap().rotation;

        let attack = scene
            .animations
            .iter()
            .find(|a| a.name.as_deref() == Some("attack1"))
            .unwrap();
        let clip = AnimationClip::from_gltf(attack, &map);
        assert!(!clip.is_empty(), "channels resolved to entities");
        assert_eq!(clip.duration, WARRIOR_ATTACK1_LEN);

        let player = world.spawn();
        world.insert(player, AnimationPlayer::new(clip));
        // Advance to the hit pose and confirm the elbow really moved there.
        let steps = (WARRIOR_ATTACK1_HIT_TIME * 60.0).round() as usize;
        for _ in 0..steps {
            advance_animation(&mut world, 1.0 / 60.0);
        }
        let posed = world.get::<LocalTransform>(elbow).unwrap().rotation;
        assert!(
            posed.angle_between(rest) > 0.1,
            "the elbow did not move: {posed:?}"
        );
        // ...and it is the authored hit pose, not some other keyframe.
        let expected = Quat::from_array(euler(-0.25, 0.0, 0.0));
        assert!(
            posed.angle_between(expected) < 0.05,
            "elbow at the hit is {posed:?}, expected {expected:?}"
        );
    }

    /// Authoring is deterministic — the property the cook cache keys on.
    #[test]
    fn authoring_is_deterministic() {
        for (a, b) in [(warrior(), warrior()), (grunt(), grunt())] {
            let bytes = |r: &Rig| {
                dreamcoast_asset::glb::write_glb_scene(
                    &r.nodes,
                    &r.meshes,
                    &r.materials,
                    &r.animations,
                )
                .unwrap()
            };
            assert_eq!(bytes(&a), bytes(&b), "{}", a.name);
        }
        let (w, g) = (warrior(), grunt());
        assert_ne!(w.nodes.len(), g.nodes.len());
    }

    /// `half_phase` really is the same cycle, half a period later.
    #[test]
    fn half_phase_shifts_a_cycle() {
        let cycle = [
            (0.0, [1.0, 0.0, 0.0]),
            (0.25, [2.0, 0.0, 0.0]),
            (0.5, [3.0, 0.0, 0.0]),
            (0.75, [4.0, 0.0, 0.0]),
            (1.0, [1.0, 0.0, 0.0]),
        ];
        let shifted = half_phase(&cycle, 1.0);
        let times: Vec<f32> = shifted.iter().map(|k| k.0).collect();
        assert_eq!(times, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let pitches: Vec<f32> = shifted.iter().map(|k| k.1[0]).collect();
        assert_eq!(pitches, vec![3.0, 4.0, 1.0, 2.0, 3.0]);
    }
}
