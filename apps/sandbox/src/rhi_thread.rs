//! Phase 15 M4 B3 — the RHI (submit) thread.
//!
//! When `P15_RHI_THREAD=1` a single worker thread *solely owns* the graphics queue,
//! the swapchain, the per-fif command buffers, the frame fences, and the
//! image-acquired / render-finished semaphores. The record (main) thread builds a
//! backend-agnostic, `Send` [`CommandList`] per frame and ships it here; the worker
//! acquires the next image, translates the IR onto its command buffer, submits,
//! presents, and (for capture frames) copies the backbuffer into a persistent
//! per-fif readback buffer and writes the PNG. A rendezvous channel bounds the
//! record thread to ≤1 frame ahead, so it builds frame N+1 while the worker submits
//! frame N (record N+1 ∥ submit N).
//!
//! ## Soundness (M4 single-owner handoff)
//!
//! The boundary objects are `!Send` (backend `Rc<DeviceShared>` + `RefCell`/`Cell`);
//! `rhi`'s `unsafe impl Send` on them is justified by this contract, upheld here:
//!
//! * **Single owner.** The objects are moved into the worker once and back at
//!   [`RhiThread::join`]; only the worker touches them while it runs, so the inner
//!   `RefCell`/`Cell` is never aliased across threads.
//! * **Refcount has one writer.** The worker only *borrows* the backend objects
//!   (acquire/translate/submit/present never `clone`/`drop` an `Rc`), so it never
//!   mutates the shared `Rc<DeviceShared>` refcount. The record thread (which keeps
//!   `Device`, sharing that `Rc`) is the sole refcount mutator — no race.
//! * **No backend drop on the worker.** The per-fif readback buffers are created on
//!   the record thread before the move and returned at `join`, dropped there — never
//!   on the worker. So no backend object is ever dropped off the record thread.
//! * **Bounded overlap.** The rendezvous channel + per-slot fence keep the record
//!   thread ≤1 frame ahead; with `FRAMES_IN_FLIGHT ≥ 2` the fif the record thread
//!   builds next never aliases the one the worker is translating.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use rhi::{Buffer, CommandBuffer, CommandList, Fence, Queue, ReadbackLayout, Semaphore, Swapchain};

use crate::app::save_screenshot;

/// The backend objects the RHI thread owns while running. Moved in at
/// [`RhiThread::spawn`] and returned at [`RhiThread::join`].
pub(crate) struct ThreadObjects {
    pub queue: Queue,
    pub swapchain: Swapchain,
    pub command_buffers: Vec<CommandBuffer>,
    pub image_available: Vec<Semaphore>,
    pub in_flight: Vec<Fence>,
    pub render_finished: Vec<Semaphore>,
    /// Persistent per-fif readback buffers (one per command buffer), sized to
    /// `readback_layout`. Reused every capture; never dropped on the worker.
    pub readback: Vec<Buffer>,
    pub readback_layout: ReadbackLayout,
}

// SAFETY: see the module docs + `rhi`'s `unsafe impl Send` on the boundary types.
// `ThreadObjects` crosses to the worker once and back at join under the single-owner
// handoff; the contained `Buffer`s (readback) are only used on the worker and
// dropped on the record thread, so no `Rc<DeviceShared>` refcount races.
struct SendObjects(ThreadObjects);
unsafe impl Send for SendObjects {}

/// A capture request for one frame: where to write the PNG.
pub(crate) struct CaptureReq {
    pub path: String,
    pub include_ui: bool,
}

/// One unit of work shipped from the record thread to the worker.
enum Msg {
    Submit {
        list: CommandList,
        fif: usize,
        capture: Option<CaptureReq>,
    },
}

/// Handle to the RHI worker thread (lives on the record thread).
pub(crate) struct RhiThread {
    /// `Option` so [`Self::join`] can drop the sender to signal the worker to stop.
    tx: Option<SyncSender<Msg>>,
    handle: Option<JoinHandle<SendObjects>>,
    /// Set by the worker when acquire/present reports the swapchain out-of-date; the
    /// record thread polls it and runs the resize on its side (it owns the swapchain
    /// after `join`).
    recreate: Arc<AtomicBool>,
}

impl RhiThread {
    /// Spawn the worker, moving `objects` into it.
    pub fn spawn(objects: ThreadObjects) -> Self {
        // Rendezvous (capacity 0): `submit` blocks until the worker takes the frame,
        // bounding the record thread to ≤1 frame ahead.
        let (tx, rx) = sync_channel::<Msg>(0);
        let recreate = Arc::new(AtomicBool::new(false));
        let recreate_worker = recreate.clone();
        let send = SendObjects(objects);
        let handle = std::thread::Builder::new()
            .name("rhi-submit".into())
            .spawn(move || run(send, rx, recreate_worker))
            .expect("spawn RHI thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            recreate,
        }
    }

    /// Ship a frame's IR to the worker. Blocks until the worker receives it
    /// (rendezvous → record thread stays ≤1 frame ahead). A dead worker (panic) is
    /// surfaced at the next [`Self::join`].
    pub fn submit(&self, list: CommandList, fif: usize, capture: Option<CaptureReq>) {
        let _ = self
            .tx
            .as_ref()
            .expect("sender live")
            .send(Msg::Submit { list, fif, capture });
    }

    /// Whether the worker flagged the swapchain out-of-date since the last check.
    pub fn take_recreate(&self) -> bool {
        self.recreate.swap(false, Ordering::AcqRel)
    }

    /// Stop the worker (drains any in-flight frame) and reclaim the owned objects.
    pub fn join(mut self) -> ThreadObjects {
        self.tx.take(); // drop the sender → the worker's `recv` returns `Err`
        self.handle
            .take()
            .expect("handle live")
            .join()
            .expect("RHI thread panicked")
            .0
    }
}

/// Worker loop: process frames until the channel closes, then hand the objects back.
fn run(objects: SendObjects, rx: Receiver<Msg>, recreate: Arc<AtomicBool>) -> SendObjects {
    let o = objects.0;
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Submit { list, fif, capture } => {
                if let Err(e) = process_frame(&o, &list, fif, capture, &recreate) {
                    // A submit failure here can't propagate to the record thread mid
                    // frame; log it and keep the worker alive for the next frame.
                    tracing::error!("RHI thread frame failed: {e}");
                }
            }
        }
    }
    SendObjects(o)
}

/// Acquire → translate → submit → (capture) → present for one frame. Mirrors the
/// inline frame loop's normal submit path exactly (same fence/semaphore order), so
/// captures stay byte-identical.
fn process_frame(
    o: &ThreadObjects,
    list: &CommandList,
    fif: usize,
    capture: Option<CaptureReq>,
    recreate: &AtomicBool,
) -> anyhow::Result<()> {
    // Throttle this slot to FRAMES_IN_FLIGHT, then acquire the drawable.
    o.in_flight[fif].wait()?;
    o.in_flight[fif].reset()?;
    let image_index = match o.swapchain.acquire_next_image(&o.image_available[fif])? {
        Some(i) => i,
        None => {
            recreate.store(true, Ordering::Release);
            return Ok(());
        }
    };

    let cmd = &o.command_buffers[fif];
    cmd.begin()?;
    // Translate the record thread's IR onto the real command buffer, resolving the
    // backbuffer to the worker's own swapchain + freshly acquired image index.
    list.translate(cmd, &o.swapchain, image_index)?;
    if capture.is_some() {
        cmd.copy_swapchain_to_buffer(&o.swapchain, image_index, &o.readback[fif]);
    }
    cmd.end()?;

    let signal = &o.render_finished[image_index as usize];
    o.queue
        .submit(cmd, &o.image_available[fif], signal, &o.in_flight[fif])?;

    // Capture readback: wait this frame's fence, read the persistent buffer, save.
    if let Some(cap) = capture {
        o.in_flight[fif].wait()?;
        let mut bytes = vec![0u8; o.readback_layout.size as usize];
        o.readback[fif].read_into(&mut bytes)?;
        save_screenshot(&cap.path, &bytes, &o.readback_layout)?;
        tracing::info!(
            "saved screenshot {} ({}x{}, ui={})",
            cap.path,
            o.readback_layout.width,
            o.readback_layout.height,
            cap.include_ui
        );
    }

    if o.queue.present(&o.swapchain, image_index, signal)? {
        recreate.store(true, Ordering::Release);
    }
    Ok(())
}
