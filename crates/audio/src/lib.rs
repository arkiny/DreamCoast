//! DreamCoast audio — device output, lock-free mixer, procedural SFX
//! (docs/game-audio-plan.md, approved M-A0..A3).
//!
//! Layering (bottom-up):
//! - [`ring`] — bounded SPSC command ring (the only game↔audio-thread channel);
//! - [`synth`] — seed-deterministic SFX bank (the audio golden gate hashes it);
//! - [`mixer`] — voices/buses/render, runs on the callback OR an offline harness;
//! - [`spatial`] — pure (gain, pan) rules the game evaluates per fixed step;
//! - [`AudioSystem`] — the game-facing handle: owns the cpal stream (or the Null
//!   fallback) and the producer half of the ring.
//!
//! The engine's render path is untouched by construction: no sandbox coupling, and
//! headless capture passes `enabled = false` so golden batteries cannot be perturbed.

pub mod mixer;
pub mod ring;
pub mod spatial;
pub mod synth;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use mixer::{Command, Mixer, RING_CAP};
use ring::Producer;
pub use synth::Sfx;

/// The game-facing audio handle. Construction NEVER fails: any device error degrades
/// to the Null sink (commands are accepted and dropped) with one log line — a machine
/// with no output device plays the game silently, and CI/headless never touches
/// CoreAudio/WASAPI at all.
pub struct AudioSystem {
    tx: Option<Producer<Command, RING_CAP>>,
    /// Keeps the device stream alive; dropped = stream closed. `None` = Null sink.
    _stream: Option<cpal::Stream>,
    drops: u32,
    warned: bool,
}

impl AudioSystem {
    /// `enabled = false` (headless capture, `NO_AUDIO=1`) skips device init entirely.
    /// The synth bank is seeded once here — same seed, same bytes, every platform.
    pub fn new(enabled: bool, seed: u64) -> Self {
        if !enabled {
            return Self::null();
        }
        match Self::open(seed) {
            Ok(sys) => sys,
            Err(e) => {
                tracing::warn!("audio: no output device, running silent ({e})");
                Self::null()
            }
        }
    }

    fn null() -> Self {
        AudioSystem {
            tx: None,
            _stream: None,
            drops: 0,
            warned: false,
        }
    }

    fn open(seed: u64) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output device".to_string())?;
        // Prefer the cook rate so the per-voice resampler is an identity; otherwise the
        // device default rate drives the fractional cursors (any rate works).
        let default = device.default_output_config().map_err(|e| e.to_string())?;
        let rate = default.sample_rate().0;
        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let (tx, mut rx) = ring::ring::<Command, RING_CAP>();
        let mut mx = Mixer::new(synth::bank(seed), rate);
        let stream = device
            .build_output_stream(
                &config,
                move |out: &mut [f32], _| {
                    // The whole real-time path: drain commands, sum voices. No locks,
                    // no allocation (mixer invariant, enforced by its tests).
                    mx.apply(&mut rx);
                    mx.render(out);
                },
                |e| tracing::warn!("audio stream error: {e}"),
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        tracing::info!(
            "audio: output stream at {rate} Hz (cook rate {})",
            synth::COOK_RATE
        );
        Ok(AudioSystem {
            tx: Some(tx),
            _stream: Some(stream),
            drops: 0,
            warned: false,
        })
    }

    fn send(&mut self, cmd: Command) {
        if let Some(tx) = self.tx.as_mut()
            && !tx.push(cmd)
        {
            self.drops = self.drops.wrapping_add(1);
            if !self.warned {
                self.warned = true;
                tracing::warn!("audio: command ring full, dropping (spam frame?)");
            }
        }
    }

    /// One-shot with explicit gain/pan (UI-ish, non-spatial).
    pub fn play(&mut self, sfx: Sfx, gain: f32, pan: f32) {
        self.send(Command::Play {
            sfx: sfx as u8,
            gain,
            pan,
        });
    }

    /// One-shot heard from `pos` by `listener` (world metres, xz-plane).
    pub fn play_at(&mut self, sfx: Sfx, listener: [f32; 2], pos: [f32; 2], base_gain: f32) {
        let (gain, pan) = spatial::params(listener, pos, base_gain, spatial::DEFAULT_RANGE);
        if gain > 1.0e-3 {
            self.send(Command::Play {
                sfx: sfx as u8,
                gain,
                pan,
            });
        }
    }

    /// Start or retarget a persistent loop slot from listener/emitter positions.
    /// Call every fixed step for moving listeners — the mixer smooths the targets.
    pub fn loop_at(
        &mut self,
        slot: u8,
        sfx: Sfx,
        listener: [f32; 2],
        pos: [f32; 2],
        base_gain: f32,
    ) {
        let (gain, pan) = spatial::params(listener, pos, base_gain, spatial::DEFAULT_RANGE);
        self.send(Command::LoopStart {
            slot,
            sfx: sfx as u8,
            gain,
            pan,
        });
    }

    /// Non-spatial loop (the ambience bed).
    pub fn loop_flat(&mut self, slot: u8, sfx: Sfx, gain: f32) {
        self.send(Command::LoopStart {
            slot,
            sfx: sfx as u8,
            gain,
            pan: 0.0,
        });
    }

    pub fn loop_stop(&mut self, slot: u8) {
        self.send(Command::LoopStop { slot });
    }

    /// Volume settings (env seams `AUDIO_MASTER` / `AUDIO_SFX` / `AUDIO_AMBIENCE` are
    /// read by the caller; this crate stays env-free for testability).
    pub fn set_master(&mut self, gain: f32) {
        self.send(Command::MasterGain { gain });
    }
    pub fn set_bus_sfx(&mut self, gain: f32) {
        self.send(Command::BusGain {
            bus: mixer::Bus::Sfx as u8,
            gain,
        });
    }
    pub fn set_bus_ambience(&mut self, gain: f32) {
        self.send(Command::BusGain {
            bus: mixer::Bus::Ambience as u8,
            gain,
        });
    }

    /// True when a real device stream is live (diagnostics / tests).
    pub fn is_live(&self) -> bool {
        self._stream.is_some()
    }
}
