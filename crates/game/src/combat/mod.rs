//! Combat: health, factions, the three-phase melee clock, and the damage flow
//! (`docs/game-framework-plan.md` §4.3).
//!
//! This is the *engine-independent* half of a fight. It knows nothing about
//! meshes, animations, HUDs or monsters — it answers four questions and stops:
//!
//! 1. **What phase is this attacker in?** [`AttackState`] runs the classic
//!    windup → active → recovery clock over a [`ComboChain`], including the combo
//!    buffering rules, and reports [`AttackEvent`]s.
//! 2. **Who did the swing connect with?** [`AttackState::resolve_hits`] tests the
//!    arc against the targets the caller offers, once per target per swing.
//! 3. **What does a hit do?** [`DamageEvent`] → [`apply_damage_events`] →
//!    [`Health`], honouring [`IFrames`].
//! 4. **Who died?** [`DeathEvent`], emitted exactly once per entity.
//!
//! Everything else — knockback, hitstun, VFX, aggro, loot — reads those events
//! and is the game's business. Nothing here allocates in the idle path, reads a
//! clock, or touches the renderer, so a whole fight can be simulated in a unit
//! test.
//!
//! # Coordinates
//!
//! 2D on the **XZ plane**, the same convention as [`crate::physics`]:
//! `Vec2::x` is world X and `Vec2::y` is world Z. Positions are always **passed
//! in** rather than read from transform components — see [`BodyCircle`].
//!
//! # A tick, end to end
//!
//! ```
//! use dreamcoast_game::combat::{
//!     AttackState, BodyCircle, ClassDef, DamageEvent, DeathEvent, Health, Team,
//!     apply_damage_events, tick_iframes,
//! };
//! use dreamcoast_scene::{Events, World};
//! use glam::Vec2;
//!
//! const DT: f32 = 1.0 / 60.0;
//!
//! let class = ClassDef::warrior();
//! let mut world = World::new();
//! let mut damage: Events<DamageEvent> = Events::new();
//! let mut deaths: Events<DeathEvent> = Events::new();
//!
//! let player = world.spawn();
//! world.insert(player, class.health());
//! world.insert(player, Team::PLAYER);
//!
//! let skeleton = world.spawn();
//! world.insert(skeleton, Health::new(20.0));
//! world.insert(skeleton, Team::ENEMY);
//! world.insert(skeleton, BodyCircle::new(0.45));
//!
//! // The player is at the origin facing +X; the skeleton stands 1.3 units away.
//! let (player_pos, skeleton_pos) = (Vec2::ZERO, Vec2::new(1.3, 0.0));
//! let facing = Vec2::X;
//!
//! let mut attack = AttackState::new();
//! attack.request(&class.combo); // the attack button
//!
//! for _ in 0..120 {
//!     tick_iframes(&mut world, DT);
//!
//!     // While the hit window is open, test the arc and turn hits into damage.
//!     let struck = attack.resolve_hits(
//!         &class.combo,
//!         player_pos,
//!         facing,
//!         Team::PLAYER,
//!         [(skeleton, skeleton_pos, 0.45, Team::ENEMY)],
//!     );
//!     if let Some(spec) = attack.spec(&class.combo) {
//!         for target in struck {
//!             damage.send(DamageEvent::new(player, target, spec.damage, facing));
//!         }
//!     }
//!
//!     attack.tick(&class.combo, DT);
//!     apply_damage_events(&mut world, damage.iter(), &mut deaths);
//!     damage.update();
//!     deaths.update();
//! }
//!
//! // One swing, one hit, 12 damage off a 20-point skeleton.
//! assert_eq!(world.get::<Health>(skeleton).unwrap().current, 8.0);
//! assert!(!attack.is_attacking(), "the combo window expired");
//! ```

mod attack;
mod class;
mod damage;
mod health;
mod hit;

pub use attack::{
    AttackEvent, AttackEvents, AttackPhase, AttackSpec, AttackState, ComboChain, MAX_ATTACK_EVENTS,
};
pub use class::{ClassDef, CombatError, DodgeDef, WARRIOR_CLASS_RON};
pub use damage::{DamageEvent, DeathEvent, apply_damage_events, tick_iframes};
pub use health::{BodyCircle, Health, IFrames, Team};
pub use hit::resolve_arc_hits;
