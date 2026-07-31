//! The plain-data combat components: a hit-point pool, a faction tag, a body
//! circle, and an invulnerability window.
//!
//! All four are ordinary structs. This crate never registers them with a
//! [`World`](dreamcoast_scene::World) itself — the game does, because the game
//! owns the spawn code and therefore owns the decision of which entities are
//! damageable at all.

use serde::{Deserialize, Serialize};

/// A hit-point pool.
///
/// `current` is clamped into `[0, max]` by every method here; writing the field
/// directly is allowed (it is a component, not an invariant fortress) but then
/// the writer owns the clamp.
///
/// **Death is `current <= 0`, not a separate flag.** A dead pool absorbs no
/// further damage and cannot be healed, which is what makes
/// [`apply_damage_events`](super::apply_damage_events) able to emit exactly one
/// [`DeathEvent`](super::DeathEvent) per entity: the alive → dead edge happens
/// once.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Health {
    /// Remaining hit points. `<= 0` means dead.
    pub current: f32,
    /// The full pool, i.e. the cap [`heal`](Self::heal) restores to.
    pub max: f32,
}

impl Health {
    /// A full pool of `max` points. Negative input is clamped to zero (which is
    /// a corpse, not a panic — bad data must not crash a spawn).
    pub fn new(max: f32) -> Self {
        let max = if max.is_finite() { max.max(0.0) } else { 0.0 };
        Self { current: max, max }
    }

    /// Subtract `amount`, saturating at zero. Returns the points actually
    /// removed (0 when already dead, or when `amount` is not a positive finite
    /// number — a NaN must never be able to kill).
    pub fn damage(&mut self, amount: f32) -> f32 {
        if !amount.is_finite() || amount <= 0.0 || self.is_dead() {
            return 0.0;
        }
        let applied = amount.min(self.current);
        self.current -= applied;
        applied
    }

    /// Add `amount`, saturating at [`max`](Self::max). Returns the points
    /// actually restored. **A dead pool is not healed** — resurrection is a game
    /// decision, made by writing `current` directly.
    pub fn heal(&mut self, amount: f32) -> f32 {
        if !amount.is_finite() || amount <= 0.0 || self.is_dead() {
            return 0.0;
        }
        let applied = amount.min(self.max - self.current).max(0.0);
        self.current += applied;
        applied
    }

    /// Whether the pool is empty.
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }

    /// Whether the pool still has points left.
    #[inline]
    pub fn is_alive(&self) -> bool {
        !self.is_dead()
    }

    /// Empty the pool outright (execution, falling out of the level, …).
    pub fn kill(&mut self) {
        self.current = 0.0;
    }

    /// `current / max` in `[0, 1]` — the number a health bar wants. An empty
    /// `max` reads as 0.
    pub fn fraction(&self) -> f32 {
        if self.max > 0.0 {
            (self.current / self.max).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Faction tag. Two entities are enemies when their ids differ — full stop.
///
/// One `u8` rather than a bitmask of "attitudes": a top-down crawler needs
/// "can I hit this" and nothing more, and a mask would invite an alliance system
/// nobody asked for. Destructible props get their own id (they are hostile to
/// everyone, including each other, which is exactly right for a barrel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Team(pub u8);

impl Team {
    /// The player's team.
    pub const PLAYER: Team = Team(0);
    /// The dungeon's team.
    pub const ENEMY: Team = Team(1);

    /// Whether `other` is a legal target for this team. Friendly fire is off by
    /// construction, so a swing can never clip an ally or the attacker itself.
    #[inline]
    pub fn hostile_to(self, other: Team) -> bool {
        self.0 != other.0
    }
}

/// The entity's collision/hurt circle on the **XZ plane**, in world units.
///
/// Only the radius lives here. The position is deliberately *not* cached in this
/// component and is instead **passed in explicitly** by whoever resolves hits
/// (see [`resolve_arc_hits`](super::resolve_arc_hits)). Reading the position
/// from a component would bind combat to transform-propagation timing: a hit
/// resolved before [`propagate_transforms`](dreamcoast_scene::propagate_transforms)
/// would use last tick's world matrix, and one resolved after would use this
/// tick's — a frame of lag that only shows up as "the swing that clearly
/// connected did nothing". The caller reads
/// [`LocalTransform`](dreamcoast_scene::LocalTransform) (or whatever it uses for
/// authority over position) and hands the value over, so the timing is visible
/// at the call site instead of hidden here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyCircle {
    /// Radius in world units on the XZ plane.
    pub radius: f32,
}

impl BodyCircle {
    /// A body of `radius` world units.
    pub fn new(radius: f32) -> Self {
        Self {
            radius: radius.max(0.0),
        }
    }
}

/// An invulnerability window: while `remaining > 0` the entity takes no damage.
///
/// The dodge roll is the canonical source (see
/// [`DodgeDef::iframes`](super::DodgeDef)), but a hit reaction or a scripted
/// cutscene can grant one just as well. Ticked by
/// [`tick_iframes`](super::tick_iframes) and honoured by
/// [`apply_damage_events`](super::apply_damage_events).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IFrames {
    /// Seconds of invulnerability left.
    pub remaining: f32,
}

impl IFrames {
    /// A window of `seconds`.
    pub fn new(seconds: f32) -> Self {
        Self {
            remaining: if seconds.is_finite() {
                seconds.max(0.0)
            } else {
                0.0
            },
        }
    }

    /// Extend the window to at least `seconds` (never shortens an active one —
    /// a second dodge must not cancel the first one's protection).
    pub fn refresh(&mut self, seconds: f32) {
        if seconds.is_finite() {
            self.remaining = self.remaining.max(seconds.max(0.0));
        }
    }

    /// Advance by `dt`, clamping at zero. Returns whether the window is *still*
    /// active afterwards.
    pub fn tick(&mut self, dt: f32) -> bool {
        if dt.is_finite() && dt > 0.0 {
            self.remaining = (self.remaining - dt).max(0.0);
        }
        self.is_active()
    }

    /// Whether damage is currently being ignored.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.remaining > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damage_saturates_and_reports_what_it_took() {
        let mut h = Health::new(30.0);
        assert_eq!(h.damage(12.0), 12.0);
        assert_eq!(h.current, 18.0);
        // Overkill reports only the points that existed.
        assert_eq!(h.damage(50.0), 18.0);
        assert!(h.is_dead());
        // A corpse absorbs nothing more — this is what keeps death single-shot.
        assert_eq!(h.damage(5.0), 0.0);
    }

    #[test]
    fn damage_rejects_junk_amounts() {
        let mut h = Health::new(10.0);
        assert_eq!(h.damage(f32::NAN), 0.0);
        assert_eq!(h.damage(f32::INFINITY), 0.0);
        assert_eq!(h.damage(-5.0), 0.0);
        assert_eq!(h.damage(0.0), 0.0);
        assert_eq!(h.current, 10.0, "junk must not move the pool");
    }

    #[test]
    fn heal_caps_at_max_and_never_resurrects() {
        let mut h = Health::new(20.0);
        h.damage(15.0);
        assert_eq!(h.heal(4.0), 4.0);
        assert_eq!(h.current, 9.0);
        assert_eq!(h.heal(100.0), 11.0);
        assert_eq!(h.current, 20.0);
        h.kill();
        assert_eq!(h.heal(100.0), 0.0);
        assert!(h.is_dead());
    }

    #[test]
    fn fraction_is_a_health_bar() {
        let mut h = Health::new(50.0);
        assert_eq!(h.fraction(), 1.0);
        h.damage(25.0);
        assert_eq!(h.fraction(), 0.5);
        h.kill();
        assert_eq!(h.fraction(), 0.0);
        assert_eq!(Health::new(-3.0).fraction(), 0.0);
    }

    #[test]
    fn teams_are_hostile_when_they_differ() {
        assert!(Team::PLAYER.hostile_to(Team::ENEMY));
        assert!(Team::ENEMY.hostile_to(Team::PLAYER));
        assert!(!Team::ENEMY.hostile_to(Team::ENEMY));
        assert!(Team(7).hostile_to(Team(8)));
    }

    #[test]
    fn iframes_count_down_and_clamp() {
        let mut f = IFrames::new(0.3);
        assert!(f.is_active());
        assert!(f.tick(0.1));
        assert!((f.remaining - 0.2).abs() < 1e-6);
        assert!(!f.tick(10.0));
        assert_eq!(f.remaining, 0.0);
        assert!(!f.is_active());
    }

    #[test]
    fn iframes_refresh_never_shortens() {
        let mut f = IFrames::new(0.3);
        f.refresh(0.1);
        assert_eq!(f.remaining, 0.3);
        f.refresh(0.5);
        assert_eq!(f.remaining, 0.5);
        assert_eq!(IFrames::new(f32::NAN).remaining, 0.0);
    }

    #[test]
    fn body_radius_is_non_negative() {
        assert_eq!(BodyCircle::new(-1.0).radius, 0.0);
        assert_eq!(BodyCircle::new(0.45).radius, 0.45);
    }
}
