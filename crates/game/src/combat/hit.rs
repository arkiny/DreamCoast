//! Arc hit resolution: which entities a swing connects with.

use dreamcoast_scene::Entity;
use glam::Vec2;

use super::{AttackSpec, Team};
use crate::physics::sector_hit;

/// Every hostile target whose body overlaps the swing's arc.
///
/// `targets` yields `(entity, position on XZ, body radius, team)`. Positions are
/// passed in by the caller rather than read from components — see
/// [`BodyCircle`](super::BodyCircle) for why. The geometry is
/// [`sector_hit`](crate::physics::sector_hit), the *exact* disk-vs-sector
/// predicate, so a target whose centre sits outside the wedge still counts when
/// its body reaches in; at melee range that is the difference between "clearly
/// hit" and "the game says no".
///
/// This function is **stateless**: it re-answers the question every tick and will
/// happily report the same target on eight consecutive ticks of one hit window.
/// The one-hit-per-swing rule lives in [`AttackState::resolve_hits`](super::AttackState::resolve_hits),
/// which filters against the already-hit set and is what gameplay code should
/// normally call. Use this directly for stateless one-shot arcs (a shockwave, an
/// AoE tick) where there is no swing to remember.
///
/// Results follow `targets` order, so a deterministic iterator in gives a
/// deterministic list out.
pub fn resolve_arc_hits<I>(
    origin: Vec2,
    facing: Vec2,
    spec: &AttackSpec,
    targets: I,
    attacker_team: Team,
) -> Vec<Entity>
where
    I: IntoIterator<Item = (Entity, Vec2, f32, Team)>,
{
    targets
        .into_iter()
        .filter(|&(_, pos, radius, team)| {
            attacker_team.hostile_to(team)
                && sector_hit(origin, facing, spec.half_angle_rad, spec.range, pos, radius)
        })
        .map(|(entity, ..)| entity)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dreamcoast_scene::World;

    fn swing() -> AttackSpec {
        AttackSpec {
            name: "test".to_string(),
            damage: 10.0,
            range: 2.0,
            half_angle_rad: 55f32.to_radians(),
            windup: 0.2,
            active: 0.1,
            recovery: 0.3,
            stagger: 0.0,
        }
    }

    #[test]
    fn hits_hostiles_in_the_arc_only() {
        let mut world = World::new();
        let (front, behind, far, ally) =
            (world.spawn(), world.spawn(), world.spawn(), world.spawn());
        let targets = vec![
            (front, Vec2::new(1.2, 0.3), 0.4, Team::ENEMY),
            (behind, Vec2::new(-1.2, 0.0), 0.4, Team::ENEMY),
            (far, Vec2::new(6.0, 0.0), 0.4, Team::ENEMY),
            (ally, Vec2::new(1.0, 0.0), 0.4, Team::PLAYER),
        ];
        let hits = resolve_arc_hits(Vec2::ZERO, Vec2::X, &swing(), targets, Team::PLAYER);
        assert_eq!(hits, vec![front]);
    }

    #[test]
    fn the_attacker_cannot_hit_itself_or_its_team() {
        let mut world = World::new();
        let me = world.spawn();
        let hits = resolve_arc_hits(
            Vec2::ZERO,
            Vec2::X,
            &swing(),
            [(me, Vec2::ZERO, 0.4, Team::PLAYER)],
            Team::PLAYER,
        );
        assert!(hits.is_empty());
    }

    #[test]
    fn a_body_reaching_into_the_arc_counts() {
        let mut world = World::new();
        let e = world.spawn();
        // Centre at 70 degrees (15 outside a 55-degree half-angle), body wide
        // enough to reach the edge.
        let a = 70f32.to_radians();
        let pos = Vec2::new(a.cos(), a.sin()) * 1.5;
        let gap = 1.5 * (15f32.to_radians()).sin();
        assert_eq!(
            resolve_arc_hits(
                Vec2::ZERO,
                Vec2::X,
                &swing(),
                [(e, pos, gap + 0.02, Team::ENEMY)],
                Team::PLAYER
            ),
            vec![e]
        );
        assert!(
            resolve_arc_hits(
                Vec2::ZERO,
                Vec2::X,
                &swing(),
                [(e, pos, gap - 0.02, Team::ENEMY)],
                Team::PLAYER
            )
            .is_empty()
        );
    }

    #[test]
    fn results_follow_input_order() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        let at = |e| (e, Vec2::new(1.0, 0.0), 0.3, Team::ENEMY);
        assert_eq!(
            resolve_arc_hits(
                Vec2::ZERO,
                Vec2::X,
                &swing(),
                [at(c), at(a), at(b)],
                Team::PLAYER
            ),
            vec![c, a, b]
        );
    }

    #[test]
    fn an_empty_target_set_hits_nothing() {
        assert!(resolve_arc_hits(Vec2::ZERO, Vec2::X, &swing(), [], Team::PLAYER).is_empty());
    }
}
