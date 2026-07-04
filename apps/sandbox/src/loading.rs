//! A minimal ImGui loading screen shown in the main window while the startup cook runs on the job
//! workers, so the window stays live (pumped + a progress bar) instead of the OS "Not Responding"
//! freeze. It implements [`crate::cook_progress::ProgressSink`]: each `tick` pumps window events and
//! presents one frame — acquire → clear → ImGui progress bar → present. Best-effort: any swapchain
//! error (resize race, out-of-date image) is swallowed so a UI hiccup never blocks the cook.
//!
//! Created after the swapchain + `Gui` (both moved before the cook) and dropped before the real
//! renderer is built, so it only borrows the window/device/swapchain/gui for the cook's duration.

use std::time::Instant;

use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use rhi::{ClearColor, CommandBuffer, Device, Fence, Semaphore, Swapchain};

use crate::cook_progress::ProgressSink;

pub(crate) struct LoadingScreen<'a> {
    window: &'a mut Window,
    device: &'a Device,
    // Shared: the loading frames only `acquire_next_image`/`present` (both take `&`); a resize
    // (`recreate`, `&mut`) is skipped — the frame is just dropped until the next acquire succeeds.
    swapchain: &'a Swapchain,
    gui: &'a mut Gui,
    cmd: CommandBuffer,
    image_available: Semaphore,
    render_finished: Semaphore,
    fence: Fence,
    fif: usize,
    frames_in_flight: usize,
    last: Instant,
    /// Set once if a frame errors, so we don't spam the log for every tick of a broken swapchain.
    warned: bool,
}

impl<'a> LoadingScreen<'a> {
    pub(crate) fn new(
        window: &'a mut Window,
        device: &'a Device,
        swapchain: &'a Swapchain,
        gui: &'a mut Gui,
        frames_in_flight: usize,
        now: Instant,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            cmd: device.create_command_buffer()?,
            image_available: device.create_semaphore()?,
            render_finished: device.create_semaphore()?,
            fence: device.create_fence(true)?,
            window,
            device,
            swapchain,
            gui,
            fif: 0,
            frames_in_flight,
            last: now,
            warned: false,
        })
    }

    /// Present one loading frame. Returns `Err` on a swapchain failure (the caller's `tick` swallows
    /// it) — never propagates so the cook keeps running.
    fn present(&mut self, label: &str, done: usize, total: usize) -> anyhow::Result<()> {
        self.window.pump_events();
        let (w, h) = self.window.size();
        if w == 0 || h == 0 {
            return Ok(()); // minimized — nothing to draw
        }

        // Acquire FIRST — a skipped frame (out-of-date swapchain, minimized) must not leave an ImGui
        // frame half-started: `new_frame` and `render` are always paired below, or neither runs.
        let fence = &self.fence;
        fence.wait()?;
        let image_index = match self.swapchain.acquire_next_image(&self.image_available)? {
            Some(i) => i,
            None => return Ok(()),
        };
        fence.reset()?;

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().clamp(1.0 / 240.0, 0.1);
        self.last = now;

        // Build the ImGui frame: a centred, borderless progress panel.
        let pct = if total == 0 {
            1.0
        } else {
            done.min(total) as f32 / total as f32
        };
        let ui = self
            .gui
            .new_frame(dt, [w as f32, h as f32], self.window.input());
        ui.window("cooking")
            .position([w as f32 * 0.5, h as f32 * 0.5], imgui::Condition::Always)
            .position_pivot([0.5, 0.5])
            .size([460.0, 0.0], imgui::Condition::Always)
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .build(|| {
                ui.text("Cooking DreamCoast assets…");
                ui.spacing();
                ui.text(label);
                imgui::ProgressBar::new(pct)
                    .size([440.0, 24.0])
                    .overlay_text(format!("{:.0}%  ({done}/{total})", pct * 100.0))
                    .build(ui);
            });

        let cmd = &self.cmd;
        cmd.begin()?;
        cmd.transition_to_render_target(self.swapchain, image_index);
        cmd.begin_rendering(
            self.swapchain,
            image_index,
            Some(ClearColor {
                r: 0.02,
                g: 0.02,
                b: 0.03,
                a: 1.0,
            }),
            None,
        );
        self.gui.render(self.device, cmd, self.fif)?;
        cmd.end_rendering();
        cmd.transition_to_present(self.swapchain, image_index);
        cmd.end()?;
        self.device.queue().submit(
            cmd,
            &self.image_available,
            &self.render_finished,
            fence,
        )?;
        self.device
            .queue()
            .present(self.swapchain, image_index, &self.render_finished)?;
        self.fif = (self.fif + 1) % self.frames_in_flight;
        Ok(())
    }
}

impl ProgressSink for LoadingScreen<'_> {
    fn tick(&mut self, label: &str, done: usize, total: usize) {
        if let Err(e) = self.present(label, done, total)
            && !self.warned
        {
            self.warned = true;
            tracing::warn!("loading screen present failed ({e}); cook continues without it");
        }
    }
}

impl Drop for LoadingScreen<'_> {
    fn drop(&mut self) {
        // The cook's last presented frame may still be in flight; let it retire before the borrowed
        // swapchain/device are reused by the real renderer.
        let _ = self.device.wait_idle();
    }
}
