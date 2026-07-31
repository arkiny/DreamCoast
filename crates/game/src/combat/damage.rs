//! The damage flow: [`DamageEvent`] in, [`Health`] down, [`DeathEvent`] out.
//!
//! Damage is a **message, not a function call**. A swing does not reach into its
//! victim's health: it sends an event, and one place — [`apply_damage_events`] —
//! resolves every event that tick. That single choke point is what makes
//! invulnerability, armour, and "died exactly once" tractable; a dozen call sites
//! each subtracting hit points could not agree on any of them.

use dreamcoast_scene::{Entity, Events, World};
use glam::Vec2;

use super::{Health, IFrames};

/// "`attacker` hit `target` for `amount`."
///
/// `direction` is the hit's push direction on the **XZ plane** (`.x` = world X,
/// `.y` = world Z), normally the attacker→target vector or the swing's facing.
/// This crate only carries it; knockback, hit sparks and camera shake are the
/// game's business.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageEvent {
    /// Who swung. Kept even when the target dies, so the game can award credit.
    pub attacker: Entity,
    /// Who was hit.
    pub target: Entity,
    /// Hit points requested. The pool clamps; [`Health::damage`] reports what
    /// actually landed.
    pub amount: f32,
    /// Push direction on XZ, for knockback and hit reactions.
    pub direction: Vec2,
}

impl DamageEvent {
    /// A hit of `amount` from `attacker` to `target`, pushed along `direction`.
    pub fn new(attacker: Entity, target: Entity, amount: f32, direction: Vec2) -> Self {
        Self {
            attacker,
            target,
            amount,
            direction,
        }
    }
}

/// "`entity`'s health reached zero." Emitted **exactly once** per entity by
/// [`apply_damage_events`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeathEvent {
    /// The entity that just died. Still alive in the [`World`] — despawning is
    /// the game's decision (a corpse may need to play an animation, drop loot,
    /// or stay as scenery).
    pub entity: Entity,
}

/// Resolve this tick's damage into [`Health`], emitting a [`DeathEvent`] for each
/// entity that crosses from alive to dead. Returns the number of deaths.
///
/// An event is **dropped** — silently, it is not an error — when the target is
/// despawned, has no `Health`, is already dead, or is inside an
/// [`IFrames`] window. The already-dead check is what bounds death emission to
/// one: the second sword in the same tick finds an empty pool and stops there.
///
/// Call **once per fixed-step tick**, after every producer and before
/// [`Events::update`]. `damage` is taken as an iterator rather than the channel
/// itself so that the channel stays readable by other consumers (floating
/// numbers, hit sparks, an aggro table) in the same tick:
///
/// ```ignore
/// let deaths = apply_damage_events(&mut world, damage_events.iter(), &mut death_events);
/// ```
pub fn apply_damage_events<'a, I>(
    world: &mut World,
    damage: I,
    deaths: &mut Events<DeathEvent>,
) -> usize
where
    I: IntoIterator<Item = &'a DamageEvent>,
{
    let mut died = 0;
    for event in damage {
        if !world.is_alive(event.target) {
            continue;
        }
        if world
            .get::<IFrames>(event.target)
            .is_some_and(IFrames::is_active)
        {
            continue;
        }
        let Some(health) = world.get_mut::<Health>(event.target) else {
            continue;
        };
        if health.is_dead() {
            continue;
        }
        health.damage(event.amount);
        if health.is_dead() {
            deaths.send(DeathEvent {
                entity: event.target,
            });
            died += 1;
        }
    }
    died
}

/// Advance every [`IFrames`] window in the world by `dt`.
///
/// Expired windows are left in place at zero rather than removed: a component
/// that survives costs one float, while removing and re-adding it on every dodge
/// churns the storage for no gain.
///
/// Run **before** [`apply_damage_events`] in the tick, so a window that expires
/// this tick stops protecting this tick.
pub fn tick_iframes(world: &mut World, dt: f32) {
    if !dt.is_finite() || dt <= 0.0 {
        return;
    }
    let active: Vec<Entity> = world
        .iter::<IFrames>()
        .filter(|(_, f)| f.is_active())
        .map(|(e, _)| e)
        .collect();
    for entity in active {
        if let Some(frames) = world.get_mut::<IFrames>(entity) {
            frames.tick(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::Team;

    fn world_with_two() -> (World, Entity, Entity) {
        let mut world = World::new();
        let attacker = world.spawn();
        world.insert(attacker, Team::PLAYER);
        let target = world.spawn();
        world.insert(target, Health::new(30.0));
        world.insert(target, Team::ENEMY);
        (world, attacker, target)
    }

    #[test]
    fn damage_lands_on_health() {
        let (mut world, a, t) = world_with_two();
        let mut deaths = Events::new();
        let events = [DamageEvent::new(a, t, 12.0, Vec2::X)];
        assert_eq!(apply_damage_events(&mut world, &events, &mut deaths), 0);
        assert_eq!(world.get::<Health>(t).unwrap().current, 18.0);
        assert!(deaths.is_empty());
    }

    #[test]
    fn iframes_block_damage_entirely() {
        let (mut world, a, t) = world_with_two();
        world.insert(t, IFrames::new(0.3));
        let mut deaths = Events::new();
        let events = [DamageEvent::new(a, t, 12.0, Vec2::X)];
        apply_damage_events(&mut world, &events, &mut deaths);
        assert_eq!(world.get::<Health>(t).unwrap().current, 30.0);

        // Once the window runs out, the same hit lands.
        for _ in 0..20 {
            tick_iframes(&mut world, 1.0 / 60.0);
        }
        assert!(!world.get::<IFrames>(t).unwrap().is_active());
        apply_damage_events(&mut world, &events, &mut deaths);
        assert_eq!(world.get::<Health>(t).unwrap().current, 18.0);
    }

    #[test]
    fn death_is_emitted_exactly_once() {
        let (mut world, a, t) = world_with_two();
        let mut deaths = Events::new();
        // Three overkill hits in the same tick.
        let events = [
            DamageEvent::new(a, t, 40.0, Vec2::X),
            DamageEvent::new(a, t, 40.0, Vec2::X),
            DamageEvent::new(a, t, 40.0, Vec2::X),
        ];
        assert_eq!(apply_damage_events(&mut world, &events, &mut deaths), 1);
        assert_eq!(deaths.len(), 1);
        assert_eq!(deaths.iter().next(), Some(&DeathEvent { entity: t }));

        // And nothing more on later ticks, even though the corpse still exists.
        deaths.update();
        assert_eq!(apply_damage_events(&mut world, &events, &mut deaths), 0);
        assert!(deaths.is_empty());
    }

    #[test]
    fn damage_accumulates_across_events_before_the_kill() {
        let (mut world, a, t) = world_with_two();
        let mut deaths = Events::new();
        let events = [
            DamageEvent::new(a, t, 12.0, Vec2::X),
            DamageEvent::new(a, t, 12.0, Vec2::X),
            DamageEvent::new(a, t, 12.0, Vec2::X),
        ];
        assert_eq!(apply_damage_events(&mut world, &events, &mut deaths), 1);
        assert_eq!(world.get::<Health>(t).unwrap().current, 0.0);
        assert_eq!(deaths.len(), 1);
    }

    #[test]
    fn events_for_gone_or_healthless_targets_are_dropped() {
        let (mut world, a, t) = world_with_two();
        let prop = world.spawn(); // no Health component
        let mut deaths = Events::new();
        world.despawn(t);
        let events = [
            DamageEvent::new(a, t, 12.0, Vec2::X),
            DamageEvent::new(a, prop, 12.0, Vec2::X),
        ];
        assert_eq!(apply_damage_events(&mut world, &events, &mut deaths), 0);
        assert!(deaths.is_empty());
    }

    #[test]
    fn iframes_tick_ignores_junk_dt_and_expired_windows() {
        let (mut world, _, t) = world_with_two();
        world.insert(t, IFrames::new(0.2));
        tick_iframes(&mut world, f32::NAN);
        tick_iframes(&mut world, -1.0);
        assert_eq!(world.get::<IFrames>(t).unwrap().remaining, 0.2);
        tick_iframes(&mut world, 5.0);
        assert_eq!(world.get::<IFrames>(t).unwrap().remaining, 0.0);
        tick_iframes(&mut world, 5.0); // no-op, no panic
        assert_eq!(world.get::<IFrames>(t).unwrap().remaining, 0.0);
    }

    #[test]
    fn the_damage_channel_stays_readable_for_other_consumers() {
        let (mut world, a, t) = world_with_two();
        let mut damage: Events<DamageEvent> = Events::new();
        let mut deaths = Events::new();
        damage.send(DamageEvent::new(a, t, 5.0, Vec2::X));
        apply_damage_events(&mut world, damage.iter(), &mut deaths);
        // A HUD reading the same channel after resolution still sees the hit.
        assert_eq!(damage.len(), 1);
        assert_eq!(world.get::<Health>(t).unwrap().current, 25.0);
    }
}
