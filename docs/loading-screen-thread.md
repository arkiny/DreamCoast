# Slate-style loading screen — a dedicated present thread during the cold cook

**Status:** design (2026-07-04). Goal: keep the window live (a progress bar, no black / "Not
Responding") during the multi-minute cold cook (Intel New Sponza), inspired by UE's Slate
`MoviePlayer` — the game thread loads (blocking) while a separate thread renders + presents the
loading UI every frame.

## Why the single-threaded approach failed

The loading UI was presented from the MAIN thread between wrapped cook phases. The imgui frames
render correctly (verified: 248 verts), but:

- The main thread presents ONE frame then blocks for seconds (a GPU upload, a system-setup call).
  With no further present, the swapchain image goes stale and Windows paints the window black
  (class `BLACK_BRUSH`) / ghosts it.
- The cook interleaves CPU work (parallelizable) with GPU uploads (`immediate_submit`, main-thread,
  device-bound) across many call sites — wrapping each is whack-a-mole and can't cover the uploads.

The fix must present CONTINUOUSLY (every ~16 ms) regardless of what the cook is doing → a dedicated
thread.

## Constraints (RHI reality)

- `Device` is `!Send` (`Rc<DeviceShared>` + `Cell` idle tracking + COM/ash handles). It CANNOT move
  to another thread.
- `Queue`/`Swapchain`/`CommandBuffer`/`Fence`/`Semaphore` are `Send` (not `Sync`) — the existing
  `P15_RHI_THREAD` (M4 B3) already MOVES a fixed set of them to one owning thread; soundness rests on
  a **single-owner handoff contract**, not on the types being `Sync`.
- `immediate_submit` (every `create_texture`, i.e. the cook's uploads) uses the device's **graphics
  queue** + `wait_idle`. VK `vkQueueSubmit`/`vkQueuePresentKHR` require EXTERNAL synchronization —
  the same queue may not be touched by two threads at once.

## Design

### Ownership split (single-owner handoff — no `Sync` needed except the queue)

- **Loading thread** solely owns (moved in, all `Send`): the **swapchain**, a pre-built
  **progress pipeline** + its resources, one **command buffer**, and its **fences/semaphores**. It
  reads a shared `AtomicU32` progress + `AtomicBool` stop. It renders WITHOUT imgui — a full-screen
  clear + a single quad scaled by the progress fraction (a trivial `loading.slang`: a push-constant
  `{frac, r,g,b}` positions the bar). No `Device` call per frame, so the `!Send` device never leaves
  the main thread.
- **Main (cook) thread** keeps the `Device` and does the entire `App::new` cook, including GPU
  uploads.

### The shared resource: the graphics queue — D3D12 needs NO mutex

Present (loading thread) and `immediate_submit` (cook uploads) both hit the graphics queue.

- **D3D12** — `ID3D12CommandQueue` is FREE-THREADED (`ExecuteCommandLists`/`Present` are safe from
  multiple threads). `device.queue()` returns a fresh COM-cloned `D3d12Queue` (an independent Rust
  handle to the same underlying queue), so the loading thread OWNS its own `Queue` clone and the cook
  keeps the device's — no aliasing, no Rust `Sync` needed, no lock. **This is the P15_RHI_THREAD
  single-owner handoff, nothing more.** No RHI core change.
- **Vulkan** — `vkQueueSubmit`/`vkQueuePresentKHR` need EXTERNAL synchronization, so the same-queue
  concurrency is invalid without a mutex. **VK ships in a later step** (the loading thread is gated
  to D3D12; VK keeps the terminal bar until the queue is mutex-guarded).

The render loop (after the loading thread joins + reclaims the swapchain) is unchanged and
single-threaded, so **DX≡VK is unaffected**.

### Swapchain ownership

The cook only needs the swapchain's `format()`/`extent()` (cached before the cook for pipeline
creation) — the swapchain OBJECT is moved to the loading thread for the cook's duration and returned
by `join()`, then the render loop uses it as today.

### Lifecycle (in `run()` / `App::new`)

1. Create window + device + swapchain + the progress pipeline/resources (main thread).
2. `DisableProcessWindowsGhosting()` (already added) + present one initial frame.
3. Spawn the **loading thread**: move it the swapchain + progress renderer + a clone of the
   `Arc<Mutex<Queue>>` + the shared progress/stop atomics. It loops: acquire → clear + bar → present
   → sleep to ~60 fps, until `stop`.
4. Main thread runs the WHOLE cook (`App::new` body), bumping the progress atomic at phase
   boundaries (level cook, per-mesh SDF, vgeo DAG, …). Uploads go through the mutexed queue.
5. Cook done → set `stop`, `join()` the loading thread (reclaims the swapchain), then enter the real
   render loop unchanged.

### Progress accounting

A coarse phase-weighted fraction (e.g. level-cook 0.0–0.5, per-mesh SDF 0.5–0.8, vgeo DAG 0.8–1.0)
written to the `AtomicU32` as the cook advances. Exactness isn't needed — a moving bar that never
freezes is the goal.

## Risks / verification

- **VK queue external sync** — the mutex must cover every submit/present on the graphics queue
  (uploads + loading present). Audit `immediate_submit` + present. Validation layers on.
- **DX≡VK unaffected** — the render loop is single-threaded post-join; the mutex is uncontended
  there. Re-run the gallery/Sponza byte-identical + DX≡VK gate after.
- **Headless** — screenshot mode skips the loading thread entirely (terminal progress).
- Metal (Mac) — the loading thread is gated to the D3D12/VK path initially; Metal keeps the terminal
  bar until ported.

## Scope note

This replaces the per-call `run_blocking` wraps + `prime` (revert those); `DisableProcessWindowsGhosting`
and `DC_CACHE_DIR` stay. The committed loading screen for the SW `parallel_cook` phases (`5592388`)
is superseded by the continuous loading thread.
</content>
