//! The audio-thread mixer (docs/game-audio-plan.md §2, M-A0/M-A3).
//!
//! Owns all playback state and runs ONLY on the audio callback (or an offline test
//! harness — same code, which is what makes the mix hashable). Communication is the
//! SPSC command ring; the render path allocates nothing and locks nothing.
//!
//! Spatialization is deliberately NOT here: the game side computes (gain, pan) from
//! listener/emitter positions with the pure helpers in [`crate::spatial`] and sends
//! plain targets. The mixer stays a dumb voice sum — one-shot voices freeze their
//! params at trigger, loop slots smooth toward retargets (one-pole per sample, ~8 ms)
//! so a 60 Hz update rate can't zipper.

use crate::ring::Consumer;
use crate::synth::{fast_cos, fast_sin};

/// One-shot voice polyphony. Beyond this the weakest (smallest smoothed gain) voice is
/// stolen — the plan's "최약 보이스 스틸" policy.
pub const ONESHOT_VOICES: usize = 32;
/// Persistent loop slots (torches, ambience). Game-assigned indices, never stolen.
pub const LOOP_SLOTS: usize = 32;
/// Command ring capacity.
pub const RING_CAP: usize = 1024;

/// Mix buses (M-A3): every voice belongs to one; master multiplies all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bus {
    Sfx = 0,
    Ambience = 1,
}

/// Game → mixer commands. `Copy` + fixed size for the ring.
#[derive(Clone, Copy, Debug)]
pub enum Command {
    /// Trigger a one-shot: params frozen at trigger time.
    Play {
        sfx: u8,
        gain: f32,
        pan: f32,
    },
    /// Start (or restart) a persistent loop slot.
    LoopStart {
        slot: u8,
        sfx: u8,
        gain: f32,
        pan: f32,
    },
    /// Retarget a running loop's gain/pan (smoothed in the mixer).
    LoopSet {
        slot: u8,
        gain: f32,
        pan: f32,
    },
    /// Stop a loop slot.
    LoopStop {
        slot: u8,
    },
    /// Bus gains (applied immediately; master smoothed).
    BusGain {
        bus: u8,
        gain: f32,
    },
    MasterGain {
        gain: f32,
    },
}

/// Equal-power stereo weights for pan in [-1, 1].
fn pan_weights(pan: f32) -> (f32, f32) {
    let p = pan.clamp(-1.0, 1.0);
    // θ from 0 (hard left) to 1/4 turn (hard right); cos/sin give the √-power law.
    let theta = (p + 1.0) * 0.125; // turns
    (fast_cos(theta), fast_sin(theta))
}

#[derive(Clone, Copy)]
struct Voice {
    buf: u32, // bank index; u32::MAX = free
    cursor: f32,
    gain_l: f32,
    gain_r: f32,
    bus: u8,
    looping: bool,
    // Loop slots smooth toward these targets; one-shots keep target == current.
    target_l: f32,
    target_r: f32,
}

const FREE: u32 = u32::MAX;

impl Voice {
    const IDLE: Voice = Voice {
        buf: FREE,
        cursor: 0.0,
        gain_l: 0.0,
        gain_r: 0.0,
        bus: 0,
        looping: false,
        target_l: 0.0,
        target_r: 0.0,
    };
}

/// The mixer. Constructed with the synthesized bank; `render` fills interleaved
/// stereo f32 at the DEVICE rate (per-voice fractional cursors resample the 48 kHz
/// bank by linear interpolation — cook-rate content, any device rate).
pub struct Mixer {
    bank: Vec<Vec<f32>>,
    oneshots: [Voice; ONESHOT_VOICES],
    loops: [Voice; LOOP_SLOTS],
    bus_gain: [f32; 2],
    master: f32,
    master_target: f32,
    /// Per-sample one-pole coefficient for loop retarget smoothing (~8 ms), derived
    /// from the device rate at construction.
    smooth: f32,
    step: f32, // cook-rate / device-rate cursor increment
    /// Commands dropped because a voice/steal target could not be found (never expected;
    /// observable for diagnostics via [`Self::overflow_count`]).
    overflow: u32,
}

impl Mixer {
    pub fn new(bank: Vec<Vec<f32>>, device_rate: u32) -> Self {
        let rate = device_rate.max(8000) as f32;
        Mixer {
            bank,
            oneshots: [Voice::IDLE; ONESHOT_VOICES],
            loops: [Voice::IDLE; LOOP_SLOTS],
            bus_gain: [1.0, 1.0],
            master: 1.0,
            master_target: 1.0,
            // one-pole: y += k (x - y), k = 1 - exp(-1/(τ·fs)) with τ = 8 ms. The exp is
            // evaluated once here with the deterministic approximation.
            smooth: 1.0 - crate::synth::fast_exp(-1.0 / (0.008 * rate)),
            step: crate::synth::COOK_RATE as f32 / rate,
            overflow: 0,
        }
    }

    /// Drain the command ring. Called at the top of each callback block.
    pub fn apply<const N: usize>(&mut self, rx: &mut Consumer<Command, N>) {
        while let Some(cmd) = rx.pop() {
            self.apply_one(cmd);
        }
    }

    fn apply_one(&mut self, cmd: Command) {
        match cmd {
            Command::Play { sfx, gain, pan } => {
                let Some(buf) = self.bank_index(sfx) else {
                    return;
                };
                let (wl, wr) = pan_weights(pan);
                let v = Voice {
                    buf,
                    cursor: 0.0,
                    gain_l: gain * wl,
                    gain_r: gain * wr,
                    bus: Bus::Sfx as u8,
                    looping: false,
                    target_l: gain * wl,
                    target_r: gain * wr,
                };
                // Free voice, else steal the weakest (smallest stereo gain sum).
                let slot = self
                    .oneshots
                    .iter()
                    .position(|v| v.buf == FREE)
                    .or_else(|| {
                        self.oneshots
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                (a.gain_l + a.gain_r)
                                    .partial_cmp(&(b.gain_l + b.gain_r))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    });
                match slot {
                    Some(i) => self.oneshots[i] = v,
                    None => self.overflow = self.overflow.wrapping_add(1),
                }
            }
            Command::LoopStart {
                slot,
                sfx,
                gain,
                pan,
            } => {
                let Some(buf) = self.bank_index(sfx) else {
                    return;
                };
                let Some(v) = self.loops.get_mut(slot as usize) else {
                    self.overflow = self.overflow.wrapping_add(1);
                    return;
                };
                let (wl, wr) = pan_weights(pan);
                let restart = v.buf != buf;
                // Bus routing: the room-tone bed is the Ambience bus; everything else
                // (including the torch crackle loops — they are diegetic point sounds)
                // is SFX.
                let bus = if sfx == crate::synth::Sfx::AmbienceLoop as u8 {
                    Bus::Ambience as u8
                } else {
                    Bus::Sfx as u8
                };
                if restart {
                    *v = Voice {
                        buf,
                        cursor: 0.0,
                        gain_l: 0.0, // fade in from silence toward the target
                        gain_r: 0.0,
                        bus,
                        looping: true,
                        target_l: gain * wl,
                        target_r: gain * wr,
                    };
                } else {
                    v.target_l = gain * wl;
                    v.target_r = gain * wr;
                }
            }
            Command::LoopSet { slot, gain, pan } => {
                if let Some(v) = self.loops.get_mut(slot as usize)
                    && v.buf != FREE
                {
                    let (wl, wr) = pan_weights(pan);
                    v.target_l = gain * wl;
                    v.target_r = gain * wr;
                }
            }
            Command::LoopStop { slot } => {
                if let Some(v) = self.loops.get_mut(slot as usize) {
                    *v = Voice::IDLE;
                }
            }
            Command::BusGain { bus, gain } => {
                if let Some(g) = self.bus_gain.get_mut(bus as usize) {
                    *g = gain.clamp(0.0, 4.0);
                }
            }
            Command::MasterGain { gain } => {
                self.master_target = gain.clamp(0.0, 4.0);
            }
        }
    }

    fn bank_index(&mut self, sfx: u8) -> Option<u32> {
        if (sfx as usize) < self.bank.len() && !self.bank[sfx as usize].is_empty() {
            Some(sfx as u32)
        } else {
            self.overflow = self.overflow.wrapping_add(1);
            None
        }
    }

    /// Render interleaved stereo into `out`. Wait-free, no allocation.
    pub fn render(&mut self, out: &mut [f32]) {
        out.fill(0.0);
        let frames = out.len() / 2;
        for vi in 0..(ONESHOT_VOICES + LOOP_SLOTS) {
            // Split borrow: voices array and bank are disjoint fields.
            let v = if vi < ONESHOT_VOICES {
                &mut self.oneshots[vi]
            } else {
                &mut self.loops[vi - ONESHOT_VOICES]
            };
            if v.buf == FREE {
                continue;
            }
            let buf = &self.bank[v.buf as usize];
            let n = buf.len();
            let bus = self.bus_gain[v.bus as usize];
            let mut cursor = v.cursor;
            let (mut gl, mut gr) = (v.gain_l, v.gain_r);
            let (tl, tr) = (v.target_l, v.target_r);
            let k = self.smooth;
            let mut ended = false;
            for f in 0..frames {
                let i0 = cursor as usize;
                if i0 + 1 >= n {
                    if v.looping {
                        cursor -= (n - 1) as f32;
                        // fall through with the wrapped cursor next iteration
                    } else {
                        ended = true;
                        break;
                    }
                }
                let i0 = cursor as usize;
                let fr = cursor - i0 as f32;
                let i1 = if i0 + 1 < n { i0 + 1 } else { 0 };
                let s = buf[i0] * (1.0 - fr) + buf[i1] * fr;
                gl += k * (tl - gl);
                gr += k * (tr - gr);
                out[f * 2] += s * gl * bus;
                out[f * 2 + 1] += s * gr * bus;
                cursor += self.step;
            }
            if ended {
                *v = Voice::IDLE;
            } else {
                v.cursor = cursor;
                v.gain_l = gl;
                v.gain_r = gr;
            }
        }
        // Master, smoothed against setting changes.
        let k = self.smooth;
        let mut m = self.master;
        for f in 0..frames {
            m += k * (self.master_target - m);
            out[f * 2] *= m;
            out[f * 2 + 1] *= m;
        }
        self.master = m;
    }

    /// Voices dropped / invalid commands since start (diagnostic; read from tests or
    /// an offline harness — the live path surfaces underruns host-side instead).
    pub fn overflow_count(&self) -> u32 {
        self.overflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::ring;
    use crate::synth::{Sfx, bank};

    fn hash(buf: &[f32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for v in buf {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// The offline render gate: a scripted command sequence mixed twice must be
    /// byte-identical — the audio analogue of a golden image.
    #[test]
    fn offline_render_is_deterministic() {
        let run = || {
            let (mut tx, mut rx) = ring::<Command, 64>();
            let mut mx = Mixer::new(bank(20260731), 48_000);
            tx.push(Command::Play {
                sfx: Sfx::SwordSwing as u8,
                gain: 0.8,
                pan: -0.3,
            });
            tx.push(Command::LoopStart {
                slot: 0,
                sfx: Sfx::TorchLoop as u8,
                gain: 0.5,
                pan: 0.4,
            });
            let mut out = vec![0.0f32; 2 * 4800 * 4];
            for chunk in out.chunks_mut(2 * 512) {
                mx.apply(&mut rx);
                mx.render(chunk);
            }
            tx.push(Command::LoopSet {
                slot: 0,
                gain: 0.2,
                pan: -0.8,
            });
            tx.push(Command::Play {
                sfx: Sfx::SwordHit as u8,
                gain: 1.0,
                pan: 0.0,
            });
            let mut out2 = vec![0.0f32; 2 * 4800 * 2];
            for chunk in out2.chunks_mut(2 * 512) {
                mx.apply(&mut rx);
                mx.render(chunk);
            }
            out.extend_from_slice(&out2);
            out
        };
        let a = run();
        let b = run();
        assert_eq!(hash(&a), hash(&b));
        assert!(
            a.iter().any(|v| v.abs() > 0.01),
            "must actually produce sound"
        );
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 4.0));
    }

    #[test]
    fn pan_law_is_equal_power() {
        for pan in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let (l, r) = pan_weights(pan);
            let p = l * l + r * r;
            assert!((p - 1.0).abs() < 0.02, "power {p} at pan {pan}");
        }
        let (l, r) = pan_weights(-1.0);
        assert!(l > 0.99 && r.abs() < 0.02, "hard left {l}/{r}");
    }

    #[test]
    fn oneshot_ends_and_frees() {
        let (mut tx, mut rx) = ring::<Command, 64>();
        let mut mx = Mixer::new(bank(1), 48_000);
        tx.push(Command::Play {
            sfx: Sfx::PotionPickup as u8,
            gain: 1.0,
            pan: 0.0,
        });
        // 0.09 s effect: after 0.2 s of rendering the voice must be free again.
        let mut out = vec![0.0f32; 2 * 9600];
        mx.apply(&mut rx);
        mx.render(&mut out);
        assert!(mx.oneshots.iter().all(|v| v.buf == FREE));
        let tail = &out[out.len() / 2..];
        assert!(tail.iter().all(|v| v.abs() < 1e-3), "tail must be silent");
    }

    #[test]
    fn loop_keeps_playing_and_steal_works() {
        let (mut tx, mut rx) = ring::<Command, 256>();
        let mut mx = Mixer::new(bank(1), 44_100); // non-cook rate: exercises resampling
        tx.push(Command::LoopStart {
            slot: 3,
            sfx: Sfx::AmbienceLoop as u8,
            gain: 0.6,
            pan: 0.0,
        });
        for _ in 0..(ONESHOT_VOICES + 8) {
            tx.push(Command::Play {
                sfx: Sfx::Footstep as u8,
                gain: 0.5,
                pan: 0.0,
            });
        }
        let mut out = vec![0.0f32; 2 * 44100 * 3];
        for chunk in out.chunks_mut(2 * 441) {
            mx.apply(&mut rx);
            mx.render(chunk);
        }
        // 3 s later the 2.2 s loop has wrapped and must still be audible.
        let tail = &out[out.len() - 2 * 4410..];
        assert!(
            tail.iter().any(|v| v.abs() > 1e-3),
            "loop fell silent after wrap"
        );
        assert_eq!(
            mx.overflow_count(),
            0,
            "steal path must not count as overflow"
        );
    }

    #[test]
    fn master_and_bus_gains_apply() {
        let (mut tx, mut rx) = ring::<Command, 64>();
        let mut mx = Mixer::new(bank(1), 48_000);
        tx.push(Command::MasterGain { gain: 0.0 });
        tx.push(Command::Play {
            sfx: Sfx::SwordHit as u8,
            gain: 1.0,
            pan: 0.0,
        });
        let mut out = vec![0.0f32; 2 * 48_00];
        mx.apply(&mut rx);
        // Let the smoothed master settle to 0 first.
        let mut warm = vec![0.0f32; 2 * 4800];
        mx.render(&mut warm);
        mx.render(&mut out);
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 1e-3, "master 0 must silence the mix, peak {peak}");
    }
}
