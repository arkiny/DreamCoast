//! Grid-free overlap primitives: circle-vs-circle and sector(arc)-vs-circle.
//!
//! These are the combat shapes (M2): a melee swing is an arc in front of the
//! attacker, a projectile or a body is a circle. Pure math, no map, no state.

use glam::Vec2;

/// Whether two circles overlap. Exact tangency counts as a hit.
///
/// Positions are on the XZ plane: `.x` is world X, `.y` is world Z.
#[inline]
pub fn circle_hit(a_pos: Vec2, a_r: f32, b_pos: Vec2, b_r: f32) -> bool {
    let r = a_r.max(0.0) + b_r.max(0.0);
    (b_pos - a_pos).length_squared() <= r * r
}

/// Whether a circular sector (an arc "swing") overlaps a target circle.
///
/// The sector is the set of points within `range` of `origin` whose direction is
/// within `half_angle_rad` of `facing` — a pie slice of total opening angle
/// `2 * half_angle_rad`. Positions are on the XZ plane (`.x` = X, `.y` = Z);
/// `facing` need not be normalized.
///
/// # Fidelity
///
/// This is the **exact** disk-vs-sector predicate, not a center-in-arc
/// approximation: a target whose center falls outside the arc still counts when
/// any part of its body reaches inside. That matters at melee range, where the
/// enemy radius is a large fraction of the swing — the cheap test drops hits
/// that visibly connect, and the miss rate is worst for targets right next to
/// the attacker, which is exactly when a player is sure they hit.
///
/// It is decided in three cases:
/// 1. `origin` inside the target circle, or the target center inside the arc's
///    angular wedge and within `range + target_r` — hit.
/// 2. Farther than `range + target_r` — miss.
/// 3. Otherwise the nearest sector point to an off-wedge center lies on one of
///    the two straight edges, so the answer is a segment-vs-point distance test
///    against both edges.
///
/// Degenerate inputs behave sensibly: `half_angle_rad >= PI` is a full circle,
/// `half_angle_rad <= 0` is a line segment of length `range`, `range <= 0`
/// collapses to a point at `origin`, and a zero-length `facing` never hits
/// (there is no direction to swing in).
pub fn sector_hit(
    origin: Vec2,
    facing: Vec2,
    half_angle_rad: f32,
    range: f32,
    target_pos: Vec2,
    target_r: f32,
) -> bool {
    let target_r = target_r.max(0.0);
    let range = range.max(0.0);
    if !origin.is_finite() || !target_pos.is_finite() || !facing.is_finite() {
        return false;
    }
    let facing_len = facing.length();
    if facing_len <= 0.0 {
        return false;
    }
    let f = facing / facing_len;

    let d = target_pos - origin;
    let dist_sq = d.length_squared();
    // The origin itself is inside the target: any non-degenerate sector touches it.
    if dist_sq <= target_r * target_r {
        return true;
    }
    // Out of reach entirely.
    let reach = range + target_r;
    if dist_sq > reach * reach {
        return false;
    }
    let dist = dist_sq.sqrt();

    // Inside the angular wedge? Compare cosines instead of taking an acos.
    // (`half_angle_rad >= PI` makes cos_half <= -1, so every direction passes.)
    let cos_half = if half_angle_rad >= std::f32::consts::PI {
        -1.0
    } else {
        half_angle_rad.max(0.0).cos()
    };
    if d.dot(f) / dist >= cos_half {
        // Radially reachable (checked above) and angularly inside: the segment
        // from the origin toward the target enters the disk at `dist - target_r`,
        // which is <= range.
        return true;
    }

    // Off-wedge: the closest sector point is on one of the two straight edges.
    let (sin_h, cos_h) = half_angle_rad.max(0.0).sin_cos();
    let e0 = Vec2::new(f.x * cos_h - f.y * sin_h, f.x * sin_h + f.y * cos_h) * range;
    let e1 = Vec2::new(f.x * cos_h + f.y * sin_h, -f.x * sin_h + f.y * cos_h) * range;
    let r2 = target_r * target_r;
    dist_sq_point_segment(d, e0) <= r2 || dist_sq_point_segment(d, e1) <= r2
}

/// Squared distance from point `p` to the segment `[0, b]` (both relative to the
/// segment's start).
#[inline]
fn dist_sq_point_segment(p: Vec2, b: Vec2) -> f32 {
    let len_sq = b.length_squared();
    if len_sq <= 0.0 {
        return p.length_squared();
    }
    let t = (p.dot(b) / len_sq).clamp(0.0, 1.0);
    (p - b * t).length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn circle_hit_basics() {
        assert!(circle_hit(Vec2::ZERO, 1.0, Vec2::new(1.5, 0.0), 1.0));
        assert!(!circle_hit(Vec2::ZERO, 1.0, Vec2::new(2.5, 0.0), 1.0));
        // Concentric and fully contained.
        assert!(circle_hit(Vec2::ZERO, 3.0, Vec2::new(0.1, 0.1), 0.2));
    }

    #[test]
    fn circle_hit_tangency_counts() {
        // Exactly touching: distance == r0 + r1.
        assert!(circle_hit(Vec2::ZERO, 1.0, Vec2::new(3.0, 0.0), 2.0));
        // A hair apart.
        assert!(!circle_hit(Vec2::ZERO, 1.0, Vec2::new(3.0001, 0.0), 2.0));
        // Diagonal tangency (3-4-5).
        assert!(circle_hit(Vec2::ZERO, 2.0, Vec2::new(3.0, 4.0), 3.0));
    }

    #[test]
    fn circle_hit_zero_radius_is_a_point_test() {
        assert!(circle_hit(
            Vec2::new(1.0, 1.0),
            0.0,
            Vec2::new(1.0, 1.4),
            0.5
        ));
        assert!(!circle_hit(
            Vec2::new(1.0, 1.0),
            0.0,
            Vec2::new(1.0, 1.6),
            0.5
        ));
    }

    #[test]
    fn sector_hits_straight_ahead() {
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(1.0, 0.0),
            0.3
        ));
    }

    #[test]
    fn sector_misses_behind_the_origin() {
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(-1.0, 0.0),
            0.3
        ));
        // Also behind but close enough that a naive distance test would pass.
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(-0.5, 0.0),
            0.3
        ));
    }

    /// The fidelity case: center outside the arc, body reaching in.
    #[test]
    fn sector_hits_a_target_whose_center_is_outside_the_arc() {
        let origin = Vec2::ZERO;
        let half = FRAC_PI_4; // 45 degrees
        let range = 3.0;
        // Center at 60 degrees, distance 2 => 15 degrees outside the wedge.
        let a = 60.0f32.to_radians();
        let center = Vec2::new(a.cos(), a.sin()) * 2.0;
        // Distance from the center to the +45-degree edge line: 2*sin(15 deg).
        let gap = 2.0 * (15.0f32.to_radians()).sin();
        assert!(sector_hit(origin, Vec2::X, half, range, center, gap + 0.01));
        assert!(!sector_hit(
            origin,
            Vec2::X,
            half,
            range,
            center,
            gap - 0.01
        ));
    }

    #[test]
    fn sector_range_boundary_is_inclusive() {
        let origin = Vec2::ZERO;
        // Center at exactly range + target_r along the facing direction.
        assert!(sector_hit(
            origin,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(2.5, 0.0),
            0.5
        ));
        assert!(!sector_hit(
            origin,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(2.5001, 0.0),
            0.5
        ));
    }

    #[test]
    fn sector_range_boundary_off_axis() {
        let origin = Vec2::ZERO;
        let half = FRAC_PI_4;
        let range = 2.0;
        // Just past the arc tip, along the +45-degree edge.
        let tip = Vec2::new(FRAC_PI_4.cos(), FRAC_PI_4.sin()) * range;
        let out = tip + Vec2::new(FRAC_PI_4.cos(), FRAC_PI_4.sin()) * 0.4;
        assert!(sector_hit(origin, Vec2::X, half, range, out, 0.41));
        assert!(!sector_hit(origin, Vec2::X, half, range, out, 0.39));
    }

    #[test]
    fn sector_hits_when_the_origin_is_inside_the_target() {
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            0.1,
            1.0,
            Vec2::new(0.0, 0.2),
            0.5
        ));
    }

    #[test]
    fn sector_full_circle_is_a_plain_range_test() {
        for deg in (0..360).step_by(15) {
            let a = (deg as f32).to_radians();
            let p = Vec2::new(a.cos(), a.sin()) * 1.5;
            assert!(sector_hit(Vec2::ZERO, Vec2::X, PI, 2.0, p, 0.1), "{deg}");
            assert!(!sector_hit(Vec2::ZERO, Vec2::X, PI, 1.0, p, 0.1), "{deg}");
        }
    }

    #[test]
    fn sector_half_circle_edges() {
        // 90-degree half-angle: the wedge is the entire +X half plane.
        let half = FRAC_PI_2;
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            half,
            2.0,
            Vec2::new(0.0, 1.0),
            0.01
        ));
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            half,
            2.0,
            Vec2::new(-0.2, 1.0),
            0.1
        ));
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            half,
            2.0,
            Vec2::new(-0.2, 1.0),
            0.25
        ));
    }

    #[test]
    fn sector_degenerate_inputs() {
        // Zero half-angle: a line segment along `facing`.
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            0.0,
            2.0,
            Vec2::new(1.0, 0.2),
            0.3
        ));
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            0.0,
            2.0,
            Vec2::new(1.0, 0.4),
            0.3
        ));
        // Zero range: a point at the origin.
        assert!(sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            0.0,
            Vec2::new(0.2, 0.0),
            0.3
        ));
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            0.0,
            Vec2::new(0.4, 0.0),
            0.3
        ));
        // No facing direction, no swing.
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::ZERO,
            FRAC_PI_4,
            2.0,
            Vec2::new(1.0, 0.0),
            0.3
        ));
        // Non-finite input never hits.
        assert!(!sector_hit(
            Vec2::ZERO,
            Vec2::X,
            FRAC_PI_4,
            2.0,
            Vec2::new(f32::NAN, 0.0),
            0.3
        ));
    }

    /// Brute-force cross-check against a dense sampling of the sector region.
    ///
    /// Sampling is one-sided (a grid can miss a sliver of true overlap), so it is
    /// used in both directions with a margin: any sampled overlap must be
    /// reported, and any reported hit must show up once the target is grown by
    /// more than the sample spacing.
    #[test]
    fn sector_agrees_with_brute_force_sampling() {
        const NA: usize = 200;
        const NR: usize = 150;
        const MARGIN: f32 = 0.08;
        let origin = Vec2::new(1.0, -2.0);
        let facing = Vec2::new(0.6, 0.8);
        let range = 2.5;
        let base = facing.to_angle();

        let sampled = |target: Vec2, tr: f32, half: f32| -> bool {
            for i in 0..=NA {
                let a = base - half + 2.0 * half * (i as f32 / NA as f32);
                let dir = Vec2::from_angle(a);
                for j in 0..=NR {
                    let p = origin + dir * (range * j as f32 / NR as f32);
                    if (p - target).length_squared() <= tr * tr {
                        return true;
                    }
                }
            }
            false
        };

        let mut seed = 0x9E37_79B9u32;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 8_388_608.0 - 1.0
        };
        for half in [0.2f32, 0.7, 1.2, 2.0, 3.0] {
            for _ in 0..100 {
                let target = origin + Vec2::new(rand(), rand()) * 4.0;
                let tr = (rand().abs() * 1.2).max(0.02);
                let exact = sector_hit(origin, facing, half, range, target, tr);
                if sampled(target, tr, half) {
                    assert!(exact, "missed a sampled overlap: {target:?} r={tr}");
                }
                if exact {
                    assert!(
                        sampled(target, tr + MARGIN, half),
                        "reported a hit the sampling cannot find: {target:?} r={tr}"
                    );
                }
            }
        }
    }
}
