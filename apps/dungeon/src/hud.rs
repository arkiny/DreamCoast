//! Game HUD compositor (docs/game-ui-plan.md M-U0..M-U3) — windowless presentation
//! drawn on the ImGui background/foreground draw lists, through the engine's existing
//! batched 2D pipeline. No new render pass; no window chrome.
//!
//! Contract: **presentation only, fixed steps only.** Everything here reads simulation
//! state and never writes it, and every animation (damage ghost, border pulse, overhead
//! fades) is keyed to sim STEP COUNTS, not wall time — a `CAPTURE_SEQ` walk reproduces
//! the same pixels every run, which is the determinism gate this module ships under.
//!
//! The module is a toolkit of pure draw helpers plus a small [`HudState`]: the game
//! (`game.rs`) owns the composition — it knows its own fields; this file knows how to
//! draw a bar, a pip, a banner, a menu, and where a world point lands on screen.

use glam::{Mat4, Vec3, Vec4Swizzles};
use sandbox::imgui::{self, DrawListMut, FontId};

/// Atlas sizes for the game face (Alegreya SC, OFL — see `main.rs`): HUD body text
/// and overlay titles, each at two densities — the atlas is baked once, so DPI
/// adaptation is a pick between baked sizes ([`HudFonts::body`]/[`HudFonts::title`]
/// switch on display height), not a runtime rebuild. `GameConfig::ui_fonts` bakes
/// them in this order.
pub const FONT_BODY_PX: f32 = 17.0;
pub const FONT_BODY_LG_PX: f32 = 28.0;
pub const FONT_TITLE_PX: f32 = 44.0;
pub const FONT_TITLE_LG_PX: f32 = 72.0;

/// Display height at and above which the large baked sizes are picked (a 1440p+
/// backing surface; a 720p-900p window keeps the small ones).
const LARGE_DISPLAY_H: f32 = 1000.0;

/// Geometry scale for a display height: HUD rectangles/margins scale continuously
/// (they are vectors), fonts snap to the nearest baked size.
pub fn ui_scale(display_h: f32) -> f32 {
    (display_h / 900.0).clamp(0.75, 2.0)
}

/// Steps the damage ghost holds before it starts to drain (½ s at 60 Hz).
const GHOST_DELAY_STEPS: u64 = 30;
/// Ghost drain per step, in health fraction (full bar empties in ~1.4 s).
const GHOST_RATE: f32 = 0.012;
/// Steps the hit border pulse takes to fade out.
const PULSE_STEPS: u64 = 18;
/// Steps an overhead bar stays after its grunt disengages.
const OVERHEAD_LINGER_STEPS: u64 = 150;
/// Steps an overhead bar takes to fade after its grunt dies.
const OVERHEAD_DEATH_FADE_STEPS: u64 = 45;

// ── Palette ──────────────────────────────────────────────────────────────────────────
// One place, so the HUD reads as one artefact. Alphas are baked where a layer is
// always translucent.

const INK_SHADOW: [f32; 4] = [0.0, 0.0, 0.0, 0.78];
pub const TEXT_MAIN: [f32; 4] = [0.92, 0.89, 0.82, 1.0];
pub const TEXT_DIM: [f32; 4] = [0.68, 0.66, 0.60, 1.0];
pub const TEXT_ACCENT: [f32; 4] = [0.95, 0.85, 0.45, 1.0];
const BAR_BACK: [f32; 4] = [0.06, 0.05, 0.05, 0.82];
const BAR_RIM: [f32; 4] = [0.30, 0.26, 0.20, 0.9];
const BAR_GHOST: [f32; 4] = [0.95, 0.78, 0.35, 0.85];
const HP_HIGH: [f32; 4] = [0.30, 0.78, 0.35, 1.0];
const HP_MID: [f32; 4] = [0.90, 0.68, 0.20, 1.0];
const HP_LOW: [f32; 4] = [0.85, 0.22, 0.20, 1.0];
const POTION_FULL: [f32; 4] = [0.86, 0.36, 0.38, 1.0];
const POTION_EMPTY: [f32; 4] = [0.30, 0.27, 0.25, 0.9];
const PULSE_RED: [f32; 3] = [0.72, 0.10, 0.08];
const OVERLAY_DIM: [f32; 4] = [0.02, 0.015, 0.015, 0.62];
pub const DEATH_RED: [f32; 4] = [0.92, 0.26, 0.24, 1.0];
const MENU_FOCUS_BG: [f32; 4] = [0.32, 0.27, 0.18, 0.85];
const MENU_ITEM_BG: [f32; 4] = [0.10, 0.09, 0.08, 0.72];

/// The player's health-bar colour at a given fraction (the existing green→amber→red
/// readability rule, kept verbatim).
pub fn hp_colour(fraction: f32) -> [f32; 4] {
    if fraction > 0.5 {
        HP_HIGH
    } else if fraction > 0.25 {
        HP_MID
    } else {
        HP_LOW
    }
}

// ── Fonts ────────────────────────────────────────────────────────────────────────────

/// The game fonts delivered once by `GameHooks::register_fonts`, in
/// `GameConfig::ui_fonts` order. `None` until (unless) the engine bakes them —
/// helpers fall back to the default font, so a fontless bring-up still draws.
#[derive(Default)]
pub struct HudFonts {
    body_s: Option<FontId>,
    body_l: Option<FontId>,
    title_s: Option<FontId>,
    title_l: Option<FontId>,
}

impl HudFonts {
    pub fn register(&mut self, ids: &[FontId]) {
        self.body_s = ids.first().copied();
        self.body_l = ids.get(1).copied();
        self.title_s = ids.get(2).copied();
        self.title_l = ids.get(3).copied();
    }

    /// The body face for a display height (falls back across baked sizes).
    pub fn body(&self, display_h: f32) -> Option<FontId> {
        if display_h >= LARGE_DISPLAY_H {
            self.body_l.or(self.body_s)
        } else {
            self.body_s.or(self.body_l)
        }
    }

    pub fn title(&self, display_h: f32) -> Option<FontId> {
        if display_h >= LARGE_DISPLAY_H {
            self.title_l.or(self.title_s)
        } else {
            self.title_s.or(self.title_l)
        }
    }
}

/// Push `font` for the closure if present, else draw with the current one.
fn with_font<R>(ui: &imgui::Ui, font: Option<FontId>, f: impl FnOnce() -> R) -> R {
    match font {
        Some(id) => {
            let tok = ui.push_font(id);
            let r = f();
            tok.pop();
            r
        }
        None => f(),
    }
}

// ── Step-keyed presentation state ───────────────────────────────────────────────────

/// Per-grunt presentation memory: when it last mattered on screen.
#[derive(Clone, Copy, Default)]
struct Glimpse {
    /// Last step the grunt was engaged (chasing/attacking) or took damage.
    last_active: Option<u64>,
    /// Step it died, latched once.
    death: Option<u64>,
}

/// The HUD's own memory between steps — everything derived, everything step-keyed.
#[derive(Default)]
pub struct HudState {
    /// Trailing health fraction: holds on damage, then drains toward the real value —
    /// the classic two-layer bar that makes burst damage legible.
    ghost: f32,
    last_hit_step: Option<u64>,
    grunts: Vec<Glimpse>,
}

impl HudState {
    /// Advance the player-facing state one sim step.
    pub fn step(&mut self, step: u64, hp_fraction: f32, took_hit: bool) {
        if took_hit {
            self.last_hit_step = Some(step);
        }
        if self.ghost < hp_fraction {
            // Heals snap the ghost up — the ghost tells the damage story, not the heal's.
            self.ghost = hp_fraction;
        } else if self
            .last_hit_step
            .is_none_or(|hit| step.saturating_sub(hit) >= GHOST_DELAY_STEPS)
        {
            self.ghost = (self.ghost - GHOST_RATE).max(hp_fraction);
        }
    }

    /// Note one grunt's step. Self-heals its slot count on floor swaps (the cast is
    /// re-acquired per floor, so the count is authoritative each call).
    pub fn note_grunt(&mut self, count: usize, i: usize, step: u64, active: bool, dead: bool) {
        if self.grunts.len() != count {
            self.grunts = vec![Glimpse::default(); count];
        }
        let g = &mut self.grunts[i];
        if dead {
            if g.death.is_none() {
                g.death = Some(step);
            }
            return;
        }
        if active {
            g.last_active = Some(step);
        }
    }

    /// Reset for a fresh warrior (restart): full ghost, no pulse, no glimpses.
    pub fn reset(&mut self) {
        *self = Self {
            ghost: 1.0,
            ..Self::default()
        };
    }

    pub fn ghost(&self) -> f32 {
        self.ghost
    }

    /// Hit border pulse strength `[0,1]` at `step`.
    pub fn pulse(&self, step: u64) -> f32 {
        self.last_hit_step.map_or(0.0, |hit| {
            let age = step.saturating_sub(hit);
            (1.0 - age as f32 / PULSE_STEPS as f32).max(0.0)
        })
    }

    /// Overhead-bar alpha for grunt `i` at `step`: 1 while engaged, linger + fade
    /// after, a short fade-out after death, 0 = don't draw.
    pub fn overhead_alpha(&self, i: usize, step: u64) -> f32 {
        let Some(g) = self.grunts.get(i) else {
            return 0.0;
        };
        if let Some(death) = g.death {
            let age = step.saturating_sub(death);
            return (1.0 - age as f32 / OVERHEAD_DEATH_FADE_STEPS as f32).clamp(0.0, 1.0) * 0.9;
        }
        let Some(active) = g.last_active else {
            return 0.0;
        };
        let idle = step.saturating_sub(active);
        if idle <= OVERHEAD_LINGER_STEPS / 3 {
            1.0
        } else {
            (1.0 - (idle - OVERHEAD_LINGER_STEPS / 3) as f32
                / (OVERHEAD_LINGER_STEPS - OVERHEAD_LINGER_STEPS / 3) as f32)
                .max(0.0)
        }
    }
}

// ── Draw helpers (pure) ─────────────────────────────────────────────────────────────

/// Text with a 1 px drop shadow — the HUD sits over a lit 3D scene, so every string
/// carries its own contrast.
pub fn shadow_text(dl: &DrawListMut, pos: [f32; 2], colour: [f32; 4], text: &str) {
    dl.add_text([pos[0] + 1.0, pos[1] + 1.0], INK_SHADOW, text);
    dl.add_text(pos, colour, text);
}

/// The layered health bar: rim, back, damage ghost, fill, and the `cur / max` figure.
#[allow(clippy::too_many_arguments)] // a draw primitive: one argument per visual layer input
pub fn health_bar(
    ui: &imgui::Ui,
    dl: &DrawListMut,
    pos: [f32; 2],
    size: [f32; 2],
    fraction: f32,
    ghost: f32,
    current: f32,
    max: f32,
    label: &str,
) {
    let (x, y, w, h) = (pos[0], pos[1], size[0], size[1]);
    let frac = fraction.clamp(0.0, 1.0);
    let ghost = ghost.clamp(frac, 1.0);
    dl.add_rect([x - 2.0, y - 2.0], [x + w + 2.0, y + h + 2.0], BAR_RIM)
        .rounding(3.0)
        .build();
    dl.add_rect([x, y], [x + w, y + h], BAR_BACK)
        .filled(true)
        .build();
    if ghost > 0.0 {
        dl.add_rect([x, y], [x + w * ghost, y + h], BAR_GHOST)
            .filled(true)
            .build();
    }
    if frac > 0.0 {
        dl.add_rect([x, y], [x + w * frac, y + h], hp_colour(frac))
            .filled(true)
            .build();
    }
    let th = ui.current_font_size();
    shadow_text(dl, [x + 6.0, y + h * 0.5 - th * 0.5], TEXT_MAIN, label);
    let figure = format!("{current:.0} / {max:.0}");
    let fw = ui.calc_text_size(&figure)[0];
    shadow_text(
        dl,
        [x + w - 8.0 - fw, y + h * 0.5 - th * 0.5],
        TEXT_MAIN,
        &figure,
    );
}

/// One potion pip: a flask silhouette (neck + shoulder + bulb), filled or hollow.
fn pip(dl: &DrawListMut, centre: [f32; 2], r: f32, colour: [f32; 4], filled: bool) {
    let (cx, cy) = (centre[0], centre[1]);
    // Bulb.
    let bulb = dl.add_circle([cx, cy + r * 0.25], r * 0.75, colour);
    bulb.filled(filled).num_segments(14).thickness(1.6).build();
    // Neck.
    let neck = dl.add_rect(
        [cx - r * 0.28, cy - r],
        [cx + r * 0.28, cy - r * 0.2],
        colour,
    );
    neck.filled(filled).thickness(1.4).build();
    // Cork line, only on a full flask (reads as "stoppered").
    if filled {
        dl.add_line([cx - r * 0.34, cy - r], [cx + r * 0.34, cy - r], TEXT_MAIN)
            .thickness(1.4)
            .build();
    }
}

/// The potion pocket: `carried`/`max` pips plus how many are still on the floor.
pub fn potion_pips(
    dl: &DrawListMut,
    pos: [f32; 2],
    s: f32,
    carried: u32,
    max: u32,
    on_floor: usize,
) {
    let r = 9.0 * s;
    for i in 0..max {
        let filled = i < carried;
        pip(
            dl,
            [pos[0] + r + i as f32 * (r * 2.6), pos[1] + r],
            r,
            if filled { POTION_FULL } else { POTION_EMPTY },
            filled,
        );
    }
    let text = format!("{on_floor} on this floor");
    shadow_text(
        dl,
        [pos[0] + max as f32 * (r * 2.6) + 10.0 * s, pos[1] + 2.0],
        TEXT_DIM,
        &text,
    );
}

/// Hit feedback: a red gradient bleeding in from every screen edge, `strength` [0,1].
pub fn border_pulse(dl: &DrawListMut, display: [f32; 2], strength: f32) {
    if strength <= 0.0 {
        return;
    }
    let (w, h) = (display[0], display[1]);
    let a = 0.55 * strength;
    let edge = (h * 0.16).min(140.0);
    let solid = [PULSE_RED[0], PULSE_RED[1], PULSE_RED[2], a];
    let clear = [PULSE_RED[0], PULSE_RED[1], PULSE_RED[2], 0.0];
    // Top, bottom, left, right — each a one-direction gradient.
    dl.add_rect_filled_multicolor([0.0, 0.0], [w, edge], solid, solid, clear, clear);
    dl.add_rect_filled_multicolor([0.0, h - edge], [w, h], clear, clear, solid, solid);
    dl.add_rect_filled_multicolor([0.0, 0.0], [edge, h], solid, clear, clear, solid);
    dl.add_rect_filled_multicolor([w - edge, 0.0], [w, h], clear, solid, solid, clear);
}

/// Full-screen dim + a centred title (+ optional subtitle). The death screen and the
/// pause menu both build on this.
pub fn overlay_title(
    ui: &imgui::Ui,
    dl: &DrawListMut,
    fonts: &HudFonts,
    display: [f32; 2],
    title: &str,
    title_colour: [f32; 4],
    subtitle: Option<&str>,
) {
    dl.add_rect([0.0, 0.0], display, OVERLAY_DIM)
        .filled(true)
        .build();
    let s = ui_scale(display[1]);
    with_font(ui, fonts.title(display[1]), || {
        let size = ui.calc_text_size(title);
        let pos = [display[0] * 0.5 - size[0] * 0.5, display[1] * 0.34];
        dl.add_text([pos[0] + 2.0, pos[1] + 2.0], INK_SHADOW, title);
        dl.add_text(pos, title_colour, title);
    });
    if let Some(sub) = subtitle {
        with_font(ui, fonts.body(display[1]), || {
            let size = ui.calc_text_size(sub);
            shadow_text(
                dl,
                [
                    display[0] * 0.5 - size[0] * 0.5,
                    display[1] * 0.34 + 58.0 * s,
                ],
                TEXT_DIM,
                sub,
            );
        });
    }
}

/// The descent banner: a centred accent line low on the screen (the floor's story is
/// the world's, so it does not cover the middle).
pub fn banner(ui: &imgui::Ui, dl: &DrawListMut, fonts: &HudFonts, display: [f32; 2], text: &str) {
    let s = ui_scale(display[1]);
    with_font(ui, fonts.body(display[1]), || {
        let size = ui.calc_text_size(text);
        let pos = [display[0] * 0.5 - size[0] * 0.5, display[1] * 0.80];
        dl.add_rect(
            [pos[0] - 14.0 * s, pos[1] - 7.0 * s],
            [pos[0] + size[0] + 14.0 * s, pos[1] + size[1] + 7.0 * s],
            MENU_ITEM_BG,
        )
        .filled(true)
        .rounding(4.0)
        .build();
        shadow_text(dl, pos, TEXT_ACCENT, text);
    });
}

/// A small overhead bar at a projected screen point, `alpha` pre-multiplied in.
pub fn overhead_bar(dl: &DrawListMut, centre: [f32; 2], fraction: f32, alpha: f32, s: f32) {
    if alpha <= 0.0 {
        return;
    }
    let (w, h) = (46.0 * s, 6.0 * s);
    let (x, y) = (centre[0] - w * 0.5, centre[1]);
    let fade = |mut c: [f32; 4]| {
        c[3] *= alpha;
        c
    };
    dl.add_rect(
        [x - 1.0, y - 1.0],
        [x + w + 1.0, y + h + 1.0],
        fade(BAR_BACK),
    )
    .filled(true)
    .rounding(2.0)
    .build();
    let frac = fraction.clamp(0.0, 1.0);
    if frac > 0.0 {
        dl.add_rect([x, y], [x + w * frac, y + h], fade(hp_colour(frac)))
            .filled(true)
            .build();
    }
}

/// Where a world point lands on screen, or `None` behind the camera. The projection
/// is rebuilt game-side from the same inputs the engine uses (look-at pose + the
/// engine's 60° vertical FOV); screen x/y do not depend on the clip planes.
pub fn project(
    world: Vec3,
    eye: Vec3,
    target: Vec3,
    fov_y_rad: f32,
    display: [f32; 2],
) -> Option<[f32; 2]> {
    if display[0] <= 0.0 || display[1] <= 0.0 {
        return None;
    }
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh(fov_y_rad, display[0] / display[1], 0.05, 100.0);
    let clip = proj * view * world.extend(1.0);
    if clip.w <= 1.0e-4 {
        return None;
    }
    let ndc = clip.xyz() / clip.w;
    Some([
        (ndc.x * 0.5 + 0.5) * display[0],
        (1.0 - (ndc.y * 0.5 + 0.5)) * display[1],
    ])
}

// ── Menu ────────────────────────────────────────────────────────────────────────────

/// Draw a centred vertical menu and resolve this frame's interaction: hover moves the
/// focus, click or `confirm` activates it. Returns the activated item, if any.
///
/// Keyboard NAVIGATION is the game's (it moves `focus` on fixed steps, where its edge
/// detection lives); the mouse is frame-rate territory and is resolved here.
pub fn menu(
    ui: &imgui::Ui,
    dl: &DrawListMut,
    fonts: &HudFonts,
    display: [f32; 2],
    items: &[&str],
    focus: &mut usize,
    confirm: bool,
) -> Option<usize> {
    let s = ui_scale(display[1]);
    let (w, h) = (260.0 * s, 44.0 * s);
    let gap = 12.0 * s;
    let x = display[0] * 0.5 - w * 0.5;
    // Centred below the screen midline: the overlay's title + subtitle own the band
    // above it (see `overlay_title`), so the two never collide at any item count.
    let y0 = display[1] * 0.56 - (items.len() as f32 * (h + gap) - gap) * 0.5;
    let mouse = ui.io().mouse_pos;
    let clicked = ui.is_mouse_clicked(imgui::MouseButton::Left);
    let mut activated = None;
    for (i, item) in items.iter().enumerate() {
        let y = y0 + i as f32 * (h + gap);
        let hover = mouse[0] >= x && mouse[0] <= x + w && mouse[1] >= y && mouse[1] <= y + h;
        if hover && (ui.io().mouse_delta[0] != 0.0 || ui.io().mouse_delta[1] != 0.0) {
            *focus = i;
        }
        let focused = *focus == i;
        dl.add_rect(
            [x, y],
            [x + w, y + h],
            if focused { MENU_FOCUS_BG } else { MENU_ITEM_BG },
        )
        .filled(true)
        .rounding(5.0)
        .build();
        if focused {
            dl.add_rect([x, y], [x + w, y + h], TEXT_ACCENT)
                .rounding(5.0)
                .thickness(1.5)
                .build();
        }
        with_font(ui, fonts.body(display[1]), || {
            let size = ui.calc_text_size(item);
            shadow_text(
                dl,
                [x + w * 0.5 - size[0] * 0.5, y + h * 0.5 - size[1] * 0.5],
                if focused { TEXT_MAIN } else { TEXT_DIM },
                item,
            );
        });
        if hover && clicked {
            activated = Some(i);
        }
    }
    if confirm {
        activated = Some(*focus);
    }
    activated
}
