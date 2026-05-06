//! Safe wrappers around the CUDA driver API entry points.
//!
//! Layered on top of [`crate::sys`]: this module owns the lifecycle of
//! a CUDA driver init + device handles + (primary-style) contexts, and
//! converts every `CUresult != CUDA_SUCCESS` into an [`NvError`].
//!
//! # Lifetime model
//!
//! - [`Cuda`] is a zero-sized handle that proves `cuInit(0)` ran.
//!   The actual library + symbol cache lives in [`crate::sys::vtable`]
//!   under a `OnceLock`, so [`Cuda::init`] is cheap on subsequent calls.
//! - [`CudaDevice`] is just an `i32` ordinal — devices are not
//!   refcounted in the CUDA driver API, so there's nothing to drop.
//! - [`CudaContext`] owns a `CUcontext` and `cuCtxDestroy_v2`s on Drop.
//!   The constructor `Cuda::create_context_for` makes the new context
//!   *current* (push), so calls like `cuvidGetDecoderCaps` that require
//!   "a CUDA context" find one.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

use crate::sys::{
    self, CUcontext, CUdevice, CUresult, Vtable, CUDA_SUCCESS,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};

// ─────────────────────────── Error ────────────────────────────────────────────

/// Error type for the CUDA / NVDEC bridge.
///
/// Wraps a `CUresult` plus a human-readable message obtained from
/// `cuGetErrorString` when available.
#[derive(Debug, Clone)]
pub struct NvError {
    pub code: CUresult,
    pub message: String,
}

impl NvError {
    /// Construct an error directly from a `CUresult`. If a vtable is
    /// available, populates `message` from `cuGetErrorString`; otherwise
    /// uses a generic placeholder.
    pub(crate) fn from_cu(vt: Option<&Vtable>, code: CUresult) -> Self {
        let message = match vt {
            Some(vt) => unsafe {
                let mut p: *const c_char = std::ptr::null();
                let r = (vt.cu_get_error_string)(code, &mut p as *mut _);
                if r == CUDA_SUCCESS && !p.is_null() {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                } else {
                    format!("CUresult {code}")
                }
            },
            None => format!("CUresult {code}"),
        };
        Self { code, message }
    }

    /// Construct a plain error without a `CUresult` (used when the
    /// dlopen step itself failed).
    pub(crate) fn from_str(msg: impl Into<String>) -> Self {
        Self {
            code: -1,
            message: msg.into(),
        }
    }

    /// True if this error indicates the CUDA driver / NVIDIA stack
    /// isn't available on this host (no driver, no GPU, container
    /// without `--gpus all`, etc.). Tests use this to skip cleanly on
    /// non-NVIDIA hosts.
    pub fn is_unavailable(&self) -> bool {
        // code 100 == CUDA_ERROR_NO_DEVICE
        // code 999 == CUDA_ERROR_UNKNOWN (rare)
        // code 3   == CUDA_ERROR_NOT_INITIALIZED (shouldn't see post-init)
        // dlopen failure also returns code -1 with a string.
        self.code == 100
            || self.code == -1
            || self.message.contains("dlopen")
            || self.message.contains("dlsym")
            || self.message.contains("not available")
            || self.message.contains("no CUDA")
    }
}

impl std::fmt::Display for NvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NvError({}): {}", self.code, self.message)
    }
}

impl std::error::Error for NvError {}

/// Wrap a `CUresult` into a `Result<(), NvError>` using the cached
/// vtable for error messages.
fn check(vt: &Vtable, code: CUresult) -> Result<(), NvError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(NvError::from_cu(Some(vt), code))
    }
}

// ─────────────────────────── Cuda handle ──────────────────────────────────────

/// One-shot guard for `cuInit(0)`.
static INIT_DONE: OnceLock<Result<(), NvError>> = OnceLock::new();

/// Zero-sized proof that `cuInit(0)` returned success.
///
/// All subsequent device / context functions on this crate take a
/// `&Cuda` so the driver-init step cannot be skipped.
#[derive(Debug, Clone, Copy)]
pub struct Cuda {
    _priv: (),
}

impl Cuda {
    /// Resolve the vtable, then call `cuInit(0)` exactly once per
    /// process. Subsequent calls reuse the cached result.
    pub fn init() -> Result<Self, NvError> {
        let res = INIT_DONE.get_or_init(|| {
            let vt = sys::vtable().map_err(NvError::from_str)?;
            unsafe {
                let r = (vt.cu_init)(0);
                check(vt, r)
            }
        });
        match res {
            Ok(()) => Ok(Self { _priv: () }),
            Err(e) => Err(e.clone()),
        }
    }

    /// Number of NVIDIA devices visible to the driver.
    pub fn device_count(&self) -> Result<u32, NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut n: i32 = 0;
        unsafe {
            check(vt, (vt.cu_device_get_count)(&mut n))?;
        }
        if n < 0 {
            return Err(NvError {
                code: -1,
                message: format!("cuDeviceGetCount returned negative: {n}"),
            });
        }
        Ok(n as u32)
    }

    /// Acquire a handle to the device at the given ordinal (0-based).
    pub fn device(&self, ordinal: i32) -> Result<CudaDevice, NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut dev: CUdevice = -1;
        unsafe {
            check(vt, (vt.cu_device_get)(&mut dev, ordinal))?;
        }
        Ok(CudaDevice { handle: dev })
    }

    /// Create a CUDA context bound to `device` and push it on the
    /// calling thread's context stack.
    ///
    /// The returned [`CudaContext`] pops + destroys the context on
    /// Drop. Round 2 only needs this for `cuvidGetDecoderCaps` to find
    /// a current context.
    pub fn create_context_for(&self, device: &CudaDevice) -> Result<CudaContext, NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        unsafe {
            check(vt, (vt.cu_ctx_create_v2)(&mut ctx, 0, device.handle))?;
        }
        // cuCtxCreate already makes the context current — no push needed.
        Ok(CudaContext { ctx })
    }
}

// ─────────────────────────── CudaDevice ───────────────────────────────────────

/// A CUDA device ordinal (e.g. `0` for the first GPU).
///
/// Cheap to copy. Methods reuse the cached vtable on every call.
#[derive(Debug, Clone, Copy)]
pub struct CudaDevice {
    pub(crate) handle: CUdevice,
}

impl CudaDevice {
    /// Driver-reported device ordinal.
    pub fn handle(&self) -> CUdevice {
        self.handle
    }

    /// Human-readable device name (e.g. `"NVIDIA GeForce RTX 5080"`).
    pub fn name(&self) -> Result<String, NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut buf = [0i8; 256];
        unsafe {
            check(
                vt,
                (vt.cu_device_get_name)(
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    self.handle,
                ),
            )?;
            // Buffer is NUL-terminated by the driver.
            let cstr = CStr::from_ptr(buf.as_ptr() as *const c_char);
            Ok(cstr.to_string_lossy().into_owned())
        }
    }

    /// Total device-memory in bytes.
    pub fn total_memory_bytes(&self) -> Result<u64, NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut bytes: usize = 0;
        unsafe {
            check(vt, (vt.cu_device_total_mem_v2)(&mut bytes, self.handle))?;
        }
        Ok(bytes as u64)
    }

    /// Compute capability of the device as `(major, minor)` —
    /// e.g. `(12, 0)` for an RTX 5080.
    pub fn compute_capability(&self) -> Result<(u32, u32), NvError> {
        let vt = sys::vtable().map_err(NvError::from_str)?;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            check(
                vt,
                (vt.cu_device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    self.handle,
                ),
            )?;
            check(
                vt,
                (vt.cu_device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    self.handle,
                ),
            )?;
        }
        Ok((major.max(0) as u32, minor.max(0) as u32))
    }
}

// ─────────────────────────── CudaContext ──────────────────────────────────────

/// Owned CUDA context.
///
/// Made current on construction (`cuCtxCreate_v2` pushes the new
/// context implicitly), and destroyed via `cuCtxDestroy_v2` on Drop —
/// which also pops it from the thread-local stack if it's still current.
#[derive(Debug)]
pub struct CudaContext {
    ctx: CUcontext,
}

impl CudaContext {
    /// Raw `CUcontext` for FFI calls that need a context handle.
    pub fn raw(&self) -> CUcontext {
        self.ctx
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        if let Ok(vt) = sys::vtable() {
            // SAFETY: `self.ctx` was returned by `cuCtxCreate_v2` and
            // hasn't been destroyed yet.
            unsafe {
                let _ = (vt.cu_ctx_destroy_v2)(self.ctx);
            }
        }
        self.ctx = std::ptr::null_mut();
    }
}
