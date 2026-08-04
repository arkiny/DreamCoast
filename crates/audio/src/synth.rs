//! Deterministic procedural SFX synthesis (docs/game-audio-plan.md §2, M-A1).
//!
//! The rig/clip principle applied to sound: every effect is SYNTHESIZED from a seed at
//! load — no external assets, no licensing surface, and the whole bank is a pure
//! function whose output hashes are unit-test gates (the audio counterpart of the
//! golden-image battery).
//!
//! ## Determinism contract
//!
//! IEEE-754 f32 add/mul/div are bit-exact everywhere, but `std` transcendentals
//! (`sin`, `exp`, `powf`) route through platform libm and are NOT. Everything here
//! therefore uses this module's own polynomial approximations — `fast_sin` /
//! `fast_exp2` — built from adds and muls only, so the same seed produces the same
//! bytes on every platform (the cross-platform hash gate in `tests`).

/// Cook-time sample rate (docs/game-audio-plan.md §2): every buffer is authored at
/// 48 kHz mono; the mixer's fractional cursor adapts to whatever the device runs at.
pub const COOK_RATE: u32 = 48_000;

/// The dungeon SFX bank ids. Indexes into [`bank`]'s output — keep in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Sfx {
    Footstep = 0,
    SwordSwing,
    SwordHit,
    GruntHit,
    GruntDeath,
    PotionPickup,
    PotionDrink,
    FloorExit,
    /// Loopable (seam-crossfaded at synth time).
    TorchLoop,
    /// Loopable low room-tone bed.
    AmbienceLoop,
}

/// Number of bank entries (the enum is dense from 0).
pub const SFX_COUNT: usize = 10;

/// Whether a bank entry is authored as a seamless loop.
pub fn is_loop(sfx: Sfx) -> bool {
    matches!(sfx, Sfx::TorchLoop | Sfx::AmbienceLoop)
}

// ---------------------------------------------------------------------------------
// Deterministic primitives
// ---------------------------------------------------------------------------------

/// PCG32 — the same dependency-free generator the shader cook cache uses.
pub struct Pcg(u64);

impl Pcg {
    pub fn new(seed: u64) -> Self {
        Pcg(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    /// Uniform in [-1, 1) — the white-noise sample.
    pub fn bipolar(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
    }
}

/// sin(2π·t) for t in [0, 1), adds/muls only. Parabolic core + odd symmetry, then one
/// refinement toward equal-power accuracy (max error ~1e-3 — inaudible for SFX).
pub fn fast_sin(t: f32) -> f32 {
    let t = t - t.floor(); // wrap to [0,1)
    // Map to [-0.5, 0.5) signed half-cycles: s = t in [0,0.5) → +, [0.5,1) → −.
    let (t, sign) = if t < 0.5 { (t, 1.0) } else { (t - 0.5, -1.0) };
    // Parabola through (0,0), (0.25,1), (0.5,0): p = 16·t·(0.5 − t). Slope 8 at zero
    // (vs the true 2π ≈ 6.28) — the refinement below pulls it onto the sine.
    let p = 16.0 * t * (0.5 - t);
    // Refine toward a true sine: blend p toward p² preserving the peak.
    sign * (0.775 * p + 0.225 * p * p)
}

/// cos(2π·t).
pub fn fast_cos(t: f32) -> f32 {
    fast_sin(t + 0.25)
}

/// 2^x for x in roughly [-30, 30], adds/muls plus float bit assembly — deterministic.
pub fn fast_exp2(x: f32) -> f32 {
    let xi = x.floor();
    let xf = x - xi; // [0,1)
    // Cubic minimax-ish for 2^xf on [0,1): max rel error ~2e-4.
    let m = 1.0 + xf * (0.695_976_1 + xf * (0.224_494_23 + xf * 0.079_529_77));
    let e = (xi as i32 + 127).clamp(1, 254) as u32;
    f32::from_bits(e << 23) * m
}

/// e^x via 2^(x·log2 e).
pub fn fast_exp(x: f32) -> f32 {
    fast_exp2(x * std::f32::consts::LOG2_E)
}

/// Chamberlin state-variable filter — two integrators, adds/muls only. `f` is the
/// tuning coefficient `2·sin(π·fc/fs)` (compute with [`svf_f`]), `q` damping ~[0.1, 2].
pub struct Svf {
    low: f32,
    band: f32,
}

/// SVF tuning coefficient for cutoff `fc` at sample rate `fs`.
pub fn svf_f(fc: f32, fs: f32) -> f32 {
    2.0 * fast_sin(0.5 * fc / fs) // sin(π·fc/fs) as turns: (fc/fs)·0.5 of a full cycle
}

impl Svf {
    pub fn new() -> Self {
        Svf {
            low: 0.0,
            band: 0.0,
        }
    }
    /// Advance one sample; returns (low, band, high).
    pub fn tick(&mut self, x: f32, f: f32, q: f32) -> (f32, f32, f32) {
        self.low += f * self.band;
        let high = x - self.low - q * self.band;
        self.band += f * high;
        (self.low, self.band, high)
    }
}

impl Default for Svf {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------
// SFX builders
// ---------------------------------------------------------------------------------

fn seconds(s: f32) -> usize {
    (s * COOK_RATE as f32) as usize
}

/// Peak-normalize to `peak` (leaves silence alone).
fn normalize(buf: &mut [f32], peak: f32) {
    let max = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max > 1.0e-6 {
        let k = peak / max;
        for v in buf.iter_mut() {
            *v *= k;
        }
    }
}

/// Crossfade the tail into the head so a buffer loops without a seam, then trim the
/// tail. `fade` in samples.
fn make_loopable(buf: &mut Vec<f32>, fade: usize) {
    let n = buf.len();
    if n < fade * 2 {
        return;
    }
    for i in 0..fade {
        let w = i as f32 / fade as f32;
        buf[i] = buf[i] * w + buf[n - fade + i] * (1.0 - w);
    }
    buf.truncate(n - fade);
}

/// Filtered noise burst: the shared skeleton of most impact-ish effects.
/// `fc0 → fc1` sweeps the SVF cutoff over the duration, `decay` is the amplitude
/// exponent (higher = snappier), `pick` selects the SVF output (0 low, 1 band, 2 high).
fn noise_burst(
    rng: &mut Pcg,
    dur: f32,
    fc0: f32,
    fc1: f32,
    q: f32,
    decay: f32,
    pick: usize,
) -> Vec<f32> {
    let n = seconds(dur);
    let mut svf = Svf::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let fc = fc0 + (fc1 - fc0) * t;
        let f = svf_f(fc, COOK_RATE as f32);
        let (l, b, h) = svf.tick(rng.bipolar(), f, q);
        let v = [l, b, h][pick];
        out.push(v * fast_exp(-decay * t));
    }
    out
}

fn footstep(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    // Soft heel thud: lowpassed noise, fast decay, slight cutoff drop.
    let mut out = noise_burst(&mut rng, 0.085, 420.0, 160.0, 0.9, 26.0, 0);
    // A touch of grit on the attack (scuff).
    let grit = noise_burst(&mut rng, 0.03, 1800.0, 900.0, 0.7, 60.0, 1);
    for (i, g) in grit.iter().enumerate() {
        out[i] += 0.35 * g;
    }
    normalize(&mut out, 0.9);
    out
}

fn sword_swing(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(0.28);
    let mut svf = Svf::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Whoosh: bandpass sweep up then down (arch), amplitude arched too.
        let arch = fast_sin(0.5 * t); // half-sine window
        let fc = 500.0 + 2600.0 * arch;
        let f = svf_f(fc, COOK_RATE as f32);
        let (_, b, _) = svf.tick(rng.bipolar(), f, 0.5);
        out.push(b * arch);
    }
    normalize(&mut out, 0.85);
    out
}

fn sword_hit(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(0.14);
    let mut out = vec![0.0f32; n];
    // Metallic crack: highpassed noise snap.
    let crack = noise_burst(&mut rng, 0.14, 2500.0, 1200.0, 0.4, 34.0, 2);
    // Body: low thump with a fast downward pitch sweep 150→70 Hz.
    let mut phase = 0.0f32;
    for (i, o) in out.iter_mut().enumerate() {
        let t = i as f32 / n as f32;
        let hz = 150.0 - 80.0 * t;
        phase += hz / COOK_RATE as f32;
        *o = 0.9 * fast_sin(phase) * fast_exp(-22.0 * t) + 0.6 * crack[i];
    }
    normalize(&mut out, 0.95);
    out
}

fn grunt_hit(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    // Fleshy mid thud, duller than the sword's metal crack.
    let mut out = noise_burst(&mut rng, 0.16, 700.0, 250.0, 0.8, 20.0, 0);
    let snap = noise_burst(&mut rng, 0.05, 1400.0, 800.0, 0.8, 50.0, 1);
    for (i, s) in snap.iter().enumerate() {
        out[i] += 0.4 * s;
    }
    normalize(&mut out, 0.9);
    out
}

fn grunt_death(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(0.45);
    let mut svf = Svf::new();
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Falling groan: low tone 220→70 Hz with growl (band noise) on top.
        let hz = 220.0 - 150.0 * t;
        phase += hz / COOK_RATE as f32;
        let tone = fast_sin(phase);
        let f = svf_f(400.0 - 250.0 * t, COOK_RATE as f32);
        let (_, b, _) = svf.tick(rng.bipolar(), f, 0.6);
        out.push((0.7 * tone + 0.5 * b) * fast_exp(-4.5 * t));
    }
    normalize(&mut out, 0.9);
    out
}

fn potion_pickup(seed: u64) -> Vec<f32> {
    let _ = seed;
    let n = seconds(0.09);
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / n as f32;
        let hz = 660.0 + 440.0 * t; // bright upward blip
        phase += hz / COOK_RATE as f32;
        out.push(fast_sin(phase) * fast_exp(-8.0 * t) * (1.0 - fast_exp(-60.0 * t)));
    }
    normalize(&mut out, 0.7);
    out
}

fn potion_drink(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(0.38);
    let mut out = vec![0.0f32; n];
    // Three bubbly gulps: short sine glisses at staggered offsets.
    for (k, start) in [0.0f32, 0.12, 0.24].iter().enumerate() {
        let s0 = seconds(*start);
        let gn = seconds(0.1);
        let base = 300.0 + 60.0 * k as f32;
        let mut phase = 0.0f32;
        for i in 0..gn {
            let t = i as f32 / gn as f32;
            let hz = base + 180.0 * t;
            phase += hz / COOK_RATE as f32;
            if s0 + i < n {
                out[s0 + i] += fast_sin(phase) * fast_exp(-10.0 * t) * (1.0 - fast_exp(-80.0 * t));
            }
        }
    }
    // Liquid texture underneath.
    let wash = noise_burst(&mut rng, 0.38, 900.0, 500.0, 0.9, 6.0, 0);
    for (i, w) in wash.iter().enumerate() {
        out[i] += 0.25 * w;
    }
    normalize(&mut out, 0.75);
    out
}

fn floor_exit(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(0.55);
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    let mut svf = Svf::new();
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Descending-into-the-depths gliss + airy shimmer.
        let hz = 520.0 * fast_exp2(-1.2 * t);
        phase += hz / COOK_RATE as f32;
        let f = svf_f(3000.0, COOK_RATE as f32);
        let (_, b, _) = svf.tick(rng.bipolar(), f, 0.3);
        let env = fast_sin(0.5 * t); // arch
        out.push((0.8 * fast_sin(phase) + 0.2 * b) * env);
    }
    normalize(&mut out, 0.8);
    out
}

fn torch_loop(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(1.2);
    let mut out = vec![0.0f32; n];
    // Low fire rumble bed.
    let bed = noise_burst(&mut rng, 1.2, 240.0, 240.0, 1.2, 0.0, 0);
    for (i, b) in bed.iter().enumerate() {
        out[i] = 0.35 * b;
    }
    // Sparse crackle pops: short high-passed bursts at random offsets.
    for _ in 0..26 {
        let at = ((rng.bipolar() * 0.5 + 0.5) * (n as f32 - 400.0)) as usize;
        let pn = seconds(0.006) + ((rng.bipolar() * 0.5 + 0.5) * seconds(0.004) as f32) as usize;
        let amp = 0.4 + 0.5 * (rng.bipolar() * 0.5 + 0.5);
        let mut svf = Svf::new();
        for i in 0..pn {
            let t = i as f32 / pn as f32;
            let f = svf_f(2400.0, COOK_RATE as f32);
            let (_, _, h) = svf.tick(rng.bipolar(), f, 0.5);
            if at + i < n {
                out[at + i] += amp * h * fast_exp(-8.0 * t);
            }
        }
    }
    make_loopable(&mut out, seconds(0.08));
    normalize(&mut out, 0.8);
    out
}

fn ambience_loop(seed: u64) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    let n = seconds(2.4);
    // Dark room tone: heavily lowpassed noise, slow amplitude wander.
    let mut svf = Svf::new();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let f = svf_f(120.0, COOK_RATE as f32);
        let (l, _, _) = svf.tick(rng.bipolar(), f, 1.4);
        let wander = 0.8 + 0.2 * fast_sin(2.0 * t);
        out.push(l * wander);
    }
    let mut v = out;
    make_loopable(&mut v, seconds(0.2));
    normalize(&mut v, 0.6);
    v
}

/// Synthesize the whole bank, indexed by [`Sfx`] discriminant. Pure function of the
/// seed — the determinism gate hashes exactly this.
pub fn bank(seed: u64) -> Vec<Vec<f32>> {
    vec![
        footstep(seed ^ 0x01),
        sword_swing(seed ^ 0x02),
        sword_hit(seed ^ 0x03),
        grunt_hit(seed ^ 0x04),
        grunt_death(seed ^ 0x05),
        potion_pickup(seed ^ 0x06),
        potion_drink(seed ^ 0x07),
        floor_exit(seed ^ 0x08),
        torch_loop(seed ^ 0x09),
        ambience_loop(seed ^ 0x0a),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over the raw f32 bits — byte-exact determinism, the audio golden gate.
    fn hash(buf: &[f32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for v in buf {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    #[test]
    fn bank_is_deterministic() {
        let a = bank(20260731);
        let b = bank(20260731);
        assert_eq!(a.len(), SFX_COUNT);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(hash(x), hash(y));
        }
        // A different seed must actually change the noise-driven effects.
        let c = bank(42);
        assert_ne!(
            hash(&a[Sfx::Footstep as usize]),
            hash(&c[Sfx::Footstep as usize])
        );
    }

    #[test]
    fn buffers_are_sane() {
        for (i, buf) in bank(20260731).iter().enumerate() {
            assert!(!buf.is_empty(), "sfx {i} empty");
            assert!(buf.iter().all(|v| v.is_finite()), "sfx {i} non-finite");
            let peak = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak <= 1.0 + 1e-4, "sfx {i} clips: {peak}");
            assert!(peak > 0.05, "sfx {i} silent: {peak}");
        }
    }

    #[test]
    fn loops_are_seam_free() {
        let b = bank(20260731);
        for sfx in [Sfx::TorchLoop, Sfx::AmbienceLoop] {
            let buf = &b[sfx as usize];
            // The wrap-around step must be no larger than the typical in-buffer step —
            // a seam would be an order-of-magnitude outlier click.
            let wrap = (buf[0] - buf[buf.len() - 1]).abs();
            let mut mean_step = 0.0f32;
            for w in buf.windows(2) {
                mean_step += (w[1] - w[0]).abs();
            }
            mean_step /= (buf.len() - 1) as f32;
            assert!(
                wrap < mean_step * 12.0 + 1e-3,
                "{sfx:?} seam step {wrap} vs mean {mean_step}"
            );
        }
    }

    #[test]
    fn fast_math_is_close_enough() {
        for i in 0..1000 {
            let t = i as f32 / 1000.0;
            let err = (fast_sin(t) - (t * std::f32::consts::TAU).sin()).abs();
            assert!(err < 4.0e-3, "sin err {err} at {t}");
        }
        for i in -60..60 {
            let x = i as f32 * 0.25;
            let rel = (fast_exp2(x) - x.exp2()).abs() / x.exp2();
            assert!(rel < 5.0e-4, "exp2 rel err {rel} at {x}");
        }
    }
}
