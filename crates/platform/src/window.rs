//! A thin Win32 window with its own message loop.
//!
//! We talk to Win32 directly (rather than via a windowing crate) both to keep
//! dependencies minimal and because the engine owns its render loop. The window
//! exposes its `HWND`/`HINSTANCE` so the Phase 1 RHI backends can build a
//! swapchain against it.
//!
//! Per-window mutable state lives in a heap-allocated [`WindowState`] whose
//! pointer is stashed in `GWLP_USERDATA`; the window procedure reaches it from
//! there. This is the standard idiom for routing Win32 messages into Rust state.

use std::ffi::c_void;

use dreamcoast_core::EngineError;
use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM,
};
use windows::Win32::Graphics::Gdi::{BLACK_BRUSH, GetStockObject, HBRUSH};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_ESCAPE;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{PCWSTR, w};

const CLASS_NAME: PCWSTR = w!("EngineWindowClass");

#[derive(Default)]
struct WindowState {
    should_close: bool,
    resized: bool,
    size: (u32, u32),
    input: crate::Input,
}

/// An open application window.
pub struct Window {
    hwnd: HWND,
    hinstance: HINSTANCE,
    // Boxed so the address handed to `GWLP_USERDATA` stays stable even if the
    // `Window` itself is moved. Accessed both here and from `wndproc`.
    state: Box<WindowState>,
}

impl Window {
    /// Create and show a window with the given title and client-area size.
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self, EngineError> {
        unsafe {
            let module = GetModuleHandleW(None).map_err(plat)?;
            let hinstance = HINSTANCE(module.0);

            let wc = WNDCLASSEXW {
                cbSize: size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                hInstance: hinstance,
                hCursor: LoadCursorW(None, IDC_ARROW).map_err(plat)?,
                hbrBackground: HBRUSH(GetStockObject(BLACK_BRUSH).0),
                lpszClassName: CLASS_NAME,
                ..Default::default()
            };

            // A zero atom means failure unless the class is simply already
            // registered (we use one shared class name for every window).
            if RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS {
                return Err(EngineError::Platform("RegisterClassExW failed".into()));
            }

            // Grow the requested client size into the full window size.
            let style = WS_OVERLAPPEDWINDOW;
            let mut rect = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            AdjustWindowRect(&mut rect, style, false).map_err(plat)?;
            let win_w = rect.right - rect.left;
            let win_h = rect.bottom - rect.top;

            let mut state = Box::new(WindowState {
                size: (width, height),
                ..Default::default()
            });
            let state_ptr = state.as_mut() as *mut WindowState;

            let title_w: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();

            let hwnd = CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                CLASS_NAME,
                PCWSTR(title_w.as_ptr()),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                win_w,
                win_h,
                None,
                None,
                Some(hinstance),
                Some(state_ptr as *const c_void),
            )
            .map_err(plat)?;

            let _ = ShowWindow(hwnd, SW_SHOW);

            // Stop Windows from replacing the window with the grey "(Not Responding)" ghost when the
            // main thread is busy (a synchronous cold cook). A dedicated loading thread keeps
            // presenting, but this also covers any brief main-thread stall.
            DisableProcessWindowsGhosting();

            Ok(Self {
                hwnd,
                hinstance,
                state,
            })
        }
    }

    /// Drain all pending Win32 messages, updating window and input state.
    ///
    /// Non-blocking: returns immediately when the queue is empty so the caller's
    /// frame loop keeps running.
    pub fn pump_events(&mut self) {
        self.state.input.begin_frame();
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
                if msg.message == WM_QUIT {
                    self.state.should_close = true;
                }
            }
        }
    }

    /// Whether the window has been asked to close (close button, ESC, or quit).
    #[inline]
    pub fn should_close(&self) -> bool {
        self.state.should_close
    }

    /// Current client-area size in pixels.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        self.state.size
    }

    /// Take the "was resized since last checked" flag, clearing it.
    pub fn take_resized(&mut self) -> bool {
        std::mem::take(&mut self.state.resized)
    }

    /// Current input snapshot.
    #[inline]
    pub fn input(&self) -> &crate::Input {
        &self.state.input
    }

    /// The native window handle (for swapchain creation in Phase 1).
    #[inline]
    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    /// The native module handle.
    #[inline]
    pub fn hinstance(&self) -> HINSTANCE {
        self.hinstance
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

/// Map a Win32 error into our platform error variant.
fn plat(e: windows::core::Error) -> EngineError {
    EngineError::Platform(e.to_string())
}

/// The window procedure. Routes messages into the per-window [`WindowState`]
/// recovered from `GWLP_USERDATA`.
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe {
        // On creation Win32 hands us the state pointer via CREATESTRUCTW; stash
        // it before any other message can arrive.
        if msg == WM_NCCREATE {
            let cs = lparam.0 as *const CREATESTRUCTW;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, (*cs).lpCreateParams as isize);
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }

        let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
        if state_ptr.is_null() {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }
        let state = &mut *state_ptr;

        match msg {
            WM_CLOSE => {
                state.should_close = true;
                LRESULT(0)
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            WM_SIZE => {
                let w = (lparam.0 & 0xFFFF) as u32;
                let h = ((lparam.0 >> 16) & 0xFFFF) as u32;
                state.size = (w, h);
                state.resized = true;
                LRESULT(0)
            }
            WM_KEYDOWN => {
                let vk = wparam.0 & 0xFF;
                state.input.set_key(vk, true);
                if vk == VK_ESCAPE.0 as usize {
                    state.should_close = true;
                }
                LRESULT(0)
            }
            WM_KEYUP => {
                state.input.set_key(wparam.0 & 0xFF, false);
                LRESULT(0)
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 & 0xFFFF) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
                state.input.set_mouse_pos(x, y);
                LRESULT(0)
            }
            WM_MOUSEWHEEL => {
                let raw = ((wparam.0 >> 16) & 0xFFFF) as i16;
                state.input.add_wheel(raw as f32 / 120.0);
                LRESULT(0)
            }
            WM_CHAR => {
                if let Some(ch) = char::from_u32(wparam.0 as u32) {
                    state.input.push_char(ch);
                }
                LRESULT(0)
            }
            WM_LBUTTONDOWN => {
                state.input.set_button(0, true);
                LRESULT(0)
            }
            WM_LBUTTONUP => {
                state.input.set_button(0, false);
                LRESULT(0)
            }
            WM_RBUTTONDOWN => {
                state.input.set_button(1, true);
                LRESULT(0)
            }
            WM_RBUTTONUP => {
                state.input.set_button(1, false);
                LRESULT(0)
            }
            WM_MBUTTONDOWN => {
                state.input.set_button(2, true);
                LRESULT(0)
            }
            WM_MBUTTONUP => {
                state.input.set_button(2, false);
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, msg, wparam, lparam),
        }
    }
}
