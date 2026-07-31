//! [`ClassDef`] — a playable class as data (`docs/game-framework-plan.md` §4.3).
//!
//! The whole point of putting stats and the moveset in RON is that the *second*
//! class costs a file, not a code change. Nothing in this crate branches on
//! which class an entity is: the game loads a `ClassDef`, reads
//! [`max_health`](ClassDef::max_health) when it spawns the avatar, feeds
//! [`combo`](ClassDef::combo) to an [`AttackState`](super::AttackState), and that
//! is the entire coupling.
//!
//! [`ClassDef::warrior`] is the built-in baseline — a sane, tuned starting point
//! and the reference every other class is balanced against. A game ships its own
//! `.ron` files; `assets/warrior.ron` in this crate is the same numbers on disk,
//! kept in lockstep by a test, so the format always has a worked example.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{AttackSpec, ComboChain, Health};

/// Something went wrong loading or validating combat data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CombatError {
    /// The file could not be read or written.
    Io(String),
    /// The text is not valid RON for this type.
    Ron(String),
    /// The data parsed but is not usable (see the message).
    Invalid(String),
}

impl std::fmt::Display for CombatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "class file I/O failed: {msg}"),
            Self::Ron(msg) => write!(f, "class file is not valid RON: {msg}"),
            Self::Invalid(msg) => write!(f, "class definition is invalid: {msg}"),
        }
    }
}

impl std::error::Error for CombatError {}

/// The dodge roll: how far, how long, and how much of it is invulnerable.
///
/// `iframes` is normally a little shorter than `duration`, so the roll ends with
/// a few vulnerable frames — that recovery tail is what stops dodge-spam from
/// being a strictly dominant defence.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DodgeDef {
    /// Distance covered, in world units.
    pub distance: f32,
    /// Duration of the roll, in seconds.
    pub duration: f32,
    /// Seconds of invulnerability, measured from the start of the roll.
    pub iframes: f32,
}

impl DodgeDef {
    /// Average roll speed in world units per second — the number a character
    /// mover wants.
    pub fn speed(&self) -> f32 {
        if self.duration > 0.0 {
            self.distance / self.duration
        } else {
            0.0
        }
    }
}

/// A playable class: stats plus the moveset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassDef {
    /// Class identifier (`"warrior"`), used for asset lookup and save data.
    pub name: String,
    /// Starting and maximum hit points.
    pub max_health: f32,
    /// Walk speed in world units per second.
    pub move_speed: f32,
    /// Multiplier applied to [`move_speed`](Self::move_speed) while sprinting.
    pub sprint_mult: f32,
    /// The dodge roll.
    pub dodge: DodgeDef,
    /// The melee chain, step 0 first.
    pub combo: ComboChain,
}

impl ClassDef {
    /// The built-in warrior: the M2 baseline.
    ///
    /// | | |
    /// |---|---|
    /// | health | 100 |
    /// | walk / sprint | 4.5 / 6.975 units per second (×1.55) |
    /// | dodge | 3.2 units in 0.35 s, 0.30 s invulnerable |
    /// | chain | 12 → 14 → 22 damage, 1.9–2.1 reach, 110–130° arcs |
    ///
    /// The chain's phases lengthen as it goes (0.25/0.15/0.35 → 0.34/0.20/0.55):
    /// the opener is a quick poke, the finisher is a commitment. Damage per
    /// second is nearly flat across the three, so the payoff for landing the full
    /// chain is the burst on the last hit, not raw throughput — and the finisher's
    /// 0.55 s recovery is the window the dungeon gets to punish a greedy player.
    pub fn warrior() -> Self {
        Self {
            name: "warrior".to_string(),
            max_health: 100.0,
            move_speed: 4.5,
            sprint_mult: 1.55,
            dodge: DodgeDef {
                distance: 3.2,
                duration: 0.35,
                iframes: 0.3,
            },
            combo: ComboChain::new(vec![
                AttackSpec {
                    name: "slash_left".to_string(),
                    damage: 12.0,
                    range: 1.9,
                    half_angle_rad: 55f32.to_radians(),
                    windup: 0.25,
                    active: 0.15,
                    recovery: 0.35,
                    stagger: 0.18,
                },
                AttackSpec {
                    name: "slash_right".to_string(),
                    damage: 14.0,
                    range: 1.9,
                    half_angle_rad: 55f32.to_radians(),
                    windup: 0.28,
                    active: 0.16,
                    recovery: 0.38,
                    stagger: 0.2,
                },
                AttackSpec {
                    name: "overhead".to_string(),
                    damage: 22.0,
                    range: 2.1,
                    half_angle_rad: 65f32.to_radians(),
                    windup: 0.34,
                    active: 0.2,
                    recovery: 0.55,
                    stagger: 0.35,
                },
            ]),
        }
    }

    /// A full [`Health`] pool for this class.
    pub fn health(&self) -> Health {
        Health::new(self.max_health)
    }

    /// Sprint speed in world units per second.
    pub fn sprint_speed(&self) -> f32 {
        self.move_speed * self.sprint_mult
    }

    /// Parse RON text, rejecting anything [`validate`](Self::validate) refuses.
    pub fn from_ron(text: &str) -> Result<Self, CombatError> {
        let parsed: Self = ron::from_str(text).map_err(|e| CombatError::Ron(e.to_string()))?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Serialise to pretty RON text (hand-editable, stable field order).
    pub fn to_ron(&self) -> Result<String, CombatError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| CombatError::Ron(e.to_string()))
    }

    /// Load a class file.
    pub fn load_ron(path: impl AsRef<Path>) -> Result<Self, CombatError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| CombatError::Io(format!("read: {e}")))?;
        Self::from_ron(&text)
    }

    /// Write a class file.
    pub fn save_ron(&self, path: impl AsRef<Path>) -> Result<(), CombatError> {
        std::fs::write(path, self.to_ron()?).map_err(|e| CombatError::Io(format!("write: {e}")))
    }

    /// Check the invariants the runtime relies on.
    ///
    /// This is not balance review — it rejects data that would make the phase
    /// clock or the mover misbehave: non-finite numbers, a class that cannot
    /// attack, a hit window of zero length (which could fall between two fixed
    /// steps and hit nothing), or an arc with no opening.
    pub fn validate(&self) -> Result<(), CombatError> {
        let bad = |msg: String| Err(CombatError::Invalid(msg));
        if self.name.trim().is_empty() {
            return bad("class name is empty".to_string());
        }
        for (label, value) in [
            ("max_health", self.max_health),
            ("move_speed", self.move_speed),
            ("sprint_mult", self.sprint_mult),
            ("dodge.distance", self.dodge.distance),
            ("dodge.duration", self.dodge.duration),
            ("dodge.iframes", self.dodge.iframes),
        ] {
            if !value.is_finite() || value < 0.0 {
                return bad(format!("{}: {label} must be finite and >= 0", self.name));
            }
        }
        if self.max_health <= 0.0 {
            return bad(format!("{}: max_health must be > 0", self.name));
        }
        if self.sprint_mult < 1.0 {
            return bad(format!("{}: sprint_mult must be >= 1", self.name));
        }
        if self.dodge.duration <= 0.0 {
            return bad(format!("{}: dodge.duration must be > 0", self.name));
        }
        if self.combo.is_empty() {
            return bad(format!("{}: combo chain has no steps", self.name));
        }
        for (i, step) in self.combo.iter().enumerate() {
            let where_ = format!("{}: combo step {i}", self.name);
            if step.name.trim().is_empty() {
                return bad(format!("{where_} has no name"));
            }
            for (label, value) in [
                ("damage", step.damage),
                ("range", step.range),
                ("half_angle_rad", step.half_angle_rad),
                ("windup", step.windup),
                ("active", step.active),
                ("recovery", step.recovery),
                ("stagger", step.stagger),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return bad(format!("{where_}: {label} must be finite and >= 0"));
                }
            }
            if step.active <= 0.0 {
                return bad(format!("{where_}: active window must be > 0"));
            }
            if step.half_angle_rad <= 0.0 || step.half_angle_rad > std::f32::consts::PI {
                return bad(format!("{where_}: half_angle_rad must be in (0, PI]"));
            }
            if step.range <= 0.0 {
                return bad(format!("{where_}: range must be > 0"));
            }
        }
        Ok(())
    }
}

/// The shipped warrior class file, exported so games can consume the fixture without
/// an `include_str!` across the workspace by relative path (the same rationale as
/// [`crate::anim::WARRIOR_ANIM_RON`]). Kept byte-parseable and value-identical to
/// [`ClassDef::warrior`] by `fixture_matches_the_builtin_warrior`.
pub const WARRIOR_CLASS_RON: &str = include_str!("../../assets/warrior.ron");

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped example file (see [`WARRIOR_CLASS_RON`]).
    const WARRIOR_RON: &str = WARRIOR_CLASS_RON;

    #[test]
    fn warrior_baseline_is_valid_and_tuned_as_documented() {
        let w = ClassDef::warrior();
        w.validate().unwrap();
        assert_eq!(w.max_health, 100.0);
        assert_eq!(w.move_speed, 4.5);
        assert!((w.sprint_speed() - 6.975).abs() < 1e-4);
        assert_eq!(w.dodge.distance, 3.2);
        assert!((w.dodge.speed() - 3.2 / 0.35).abs() < 1e-4);
        assert!(w.dodge.iframes < w.dodge.duration, "roll ends vulnerable");

        let damage: Vec<f32> = w.combo.iter().map(|s| s.damage).collect();
        assert_eq!(damage, vec![12.0, 14.0, 22.0]);
        // Arcs are the documented 55/55/65 degree half-angles.
        for (step, degrees) in w.combo.iter().zip([55.0, 55.0, 65.0]) {
            assert!(
                (step.half_angle_rad - (degrees as f32).to_radians()).abs() < 1e-6,
                "{}",
                step.name
            );
        }
        // Phases lengthen along the chain: the finisher is the commitment.
        let mut previous = 0.0;
        for step in w.combo.iter() {
            assert!(step.duration() > previous, "{} is not slower", step.name);
            previous = step.duration();
            assert!(
                step.active > 1.0 / 60.0,
                "{}: hit window must outlast a fixed step",
                step.name
            );
        }
        assert_eq!(w.health(), Health::new(100.0));
    }

    #[test]
    fn ron_round_trips() {
        let w = ClassDef::warrior();
        let text = w.to_ron().unwrap();
        let parsed = ClassDef::from_ron(&text).unwrap();
        assert_eq!(parsed, w);
        // Serialisation is stable, so the second pass is byte-identical.
        assert_eq!(parsed.to_ron().unwrap(), text);
    }

    #[test]
    fn fixture_matches_the_builtin_warrior() {
        let from_file = ClassDef::from_ron(WARRIOR_RON).unwrap();
        assert_eq!(
            from_file,
            ClassDef::warrior(),
            "assets/warrior.ron drifted from ClassDef::warrior(); regenerate it \
             with ClassDef::warrior().to_ron()"
        );
    }

    #[test]
    fn hand_written_ron_parses_with_optional_fields_omitted() {
        // `stagger` defaults to 0; everything else is explicit.
        let text = r#"(
            name: "rogue",
            max_health: 70.0,
            move_speed: 5.4,
            sprint_mult: 1.7,
            dodge: (distance: 3.8, duration: 0.3, iframes: 0.26),
            combo: [
                (
                    name: "stab",
                    damage: 9.0,
                    range: 1.4,
                    half_angle_rad: 0.6,
                    windup: 0.14,
                    active: 0.08,
                    recovery: 0.2,
                ),
            ],
        )"#;
        let rogue = ClassDef::from_ron(text).unwrap();
        assert_eq!(rogue.name, "rogue");
        assert_eq!(rogue.combo.len(), 1);
        assert_eq!(rogue.combo.get(0).unwrap().stagger, 0.0);
    }

    #[test]
    fn file_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("dreamcoast-game-class-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("warrior.ron");
        ClassDef::warrior().save_ron(&path).unwrap();
        assert_eq!(ClassDef::load_ron(&path).unwrap(), ClassDef::warrior());
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(
            ClassDef::load_ron(dir.join("missing.ron")).unwrap_err(),
            CombatError::Io(_)
        ));
        assert!(matches!(
            ClassDef::from_ron("not ron").unwrap_err(),
            CombatError::Ron(_)
        ));
    }

    #[test]
    fn validation_rejects_unusable_data() {
        /// One way to break a class definition.
        type Break = Box<dyn Fn(&mut ClassDef)>;

        let cases: Vec<(&str, Break)> = vec![
            ("empty name", Box::new(|c: &mut ClassDef| c.name.clear())),
            (
                "zero health",
                Box::new(|c: &mut ClassDef| c.max_health = 0.0),
            ),
            (
                "nan speed",
                Box::new(|c: &mut ClassDef| c.move_speed = f32::NAN),
            ),
            (
                "slow sprint",
                Box::new(|c: &mut ClassDef| c.sprint_mult = 0.9),
            ),
            (
                "instant dodge",
                Box::new(|c: &mut ClassDef| c.dodge.duration = 0.0),
            ),
            (
                "no moveset",
                Box::new(|c: &mut ClassDef| c.combo = ComboChain::default()),
            ),
            (
                "unnamed step",
                Box::new(|c: &mut ClassDef| c.combo.0[1].name.clear()),
            ),
            (
                "zero-length hit window",
                Box::new(|c: &mut ClassDef| c.combo.0[0].active = 0.0),
            ),
            (
                "closed arc",
                Box::new(|c: &mut ClassDef| c.combo.0[0].half_angle_rad = 0.0),
            ),
            (
                "over-wide arc",
                Box::new(|c: &mut ClassDef| c.combo.0[0].half_angle_rad = 4.0),
            ),
            (
                "zero range",
                Box::new(|c: &mut ClassDef| c.combo.0[2].range = 0.0),
            ),
            (
                "negative windup",
                Box::new(|c: &mut ClassDef| c.combo.0[2].windup = -0.1),
            ),
        ];
        for (label, break_it) in cases {
            let mut class = ClassDef::warrior();
            break_it(&mut class);
            let err = class.validate().unwrap_err();
            assert!(matches!(err, CombatError::Invalid(_)), "{label}");
            // The same rejection happens on the parse path.
            let text = ron::ser::to_string(&class).unwrap();
            assert!(ClassDef::from_ron(&text).is_err(), "{label}");
        }
    }

    #[test]
    fn errors_describe_themselves() {
        let err = ClassDef::from_ron("not ron").unwrap_err();
        assert!(err.to_string().contains("RON"));
        let err = CombatError::Invalid("boom".to_string());
        assert!(err.to_string().contains("boom"));
        assert!(CombatError::Io("x".to_string()).to_string().contains("I/O"));
    }

    #[test]
    fn dodge_speed_handles_a_zero_duration() {
        let d = DodgeDef {
            distance: 3.0,
            duration: 0.0,
            iframes: 0.0,
        };
        assert_eq!(d.speed(), 0.0);
    }
}
