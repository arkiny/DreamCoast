//! Top-down spatialization — pure functions on the GAME side (docs/game-audio-plan.md
//! §2). The mixer never sees positions: the game computes (gain, pan) here each fixed
//! step and sends plain targets, which keeps the DSP dumb and these rules unit-tested.

/// Default audible range in metres for dungeon-scale point sounds (2 m tiles).
pub const DEFAULT_RANGE: f32 = 18.0;

/// Distance attenuation: `1 / (1 + (d/r·3)²)` — full volume at the listener, ~1/10 at
/// r/2, effectively silent at the range edge. Smooth (no hard cutoff pop), monotonic.
pub fn attenuation(distance: f32, range: f32) -> f32 {
    let x = 3.0 * distance / range.max(0.01);
    1.0 / (1.0 + x * x)
}

/// Stereo pan in [-1, 1] from the listener-relative lateral offset. The top-down
/// camera is world-axis aligned, so "lateral" is world X by contract with the game's
/// camera (`level.rs` single source); sounds close to the listener centre out to
/// avoid hard-panned feet.
pub fn pan(lateral: f32, range: f32) -> f32 {
    (lateral / (0.5 * range.max(0.01))).clamp(-1.0, 1.0)
}

/// (gain, pan) for an emitter at `pos` heard by `listener`, world metres (x, z).
pub fn params(listener: [f32; 2], pos: [f32; 2], base_gain: f32, range: f32) -> (f32, f32) {
    let dx = pos[0] - listener[0];
    let dz = pos[1] - listener[1];
    let d = (dx * dx + dz * dz).sqrt();
    (base_gain * attenuation(d, range), pan(dx, range))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attenuation_is_monotonic_and_bounded() {
        let mut prev = attenuation(0.0, DEFAULT_RANGE);
        assert!((prev - 1.0).abs() < 1e-6);
        for i in 1..100 {
            let a = attenuation(i as f32 * 0.5, DEFAULT_RANGE);
            assert!(a <= prev && a >= 0.0);
            prev = a;
        }
        assert!(attenuation(DEFAULT_RANGE, DEFAULT_RANGE) < 0.11);
    }

    #[test]
    fn pan_tracks_lateral_side() {
        let (_, p) = params([0.0, 0.0], [5.0, 0.0], 1.0, DEFAULT_RANGE);
        assert!(p > 0.4, "east emitter pans right: {p}");
        let (_, p) = params([0.0, 0.0], [-5.0, 0.0], 1.0, DEFAULT_RANGE);
        assert!(p < -0.4, "west emitter pans left: {p}");
        let (g, p) = params([3.0, 4.0], [3.0, 4.0], 0.7, DEFAULT_RANGE);
        assert!(
            (g - 0.7).abs() < 1e-5 && p.abs() < 1e-6,
            "on-listener is centred"
        );
    }
}
