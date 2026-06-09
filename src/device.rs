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
    self, CUcontext, CUdevice, CUresult, Vtable, CUDA_ERROR_DEINITIALIZED,
    CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,
    CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED,
    CUDA_ERROR_NO_DEVICE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN, CUDA_SUCCESS,
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

    /// Typed view of the underlying `CUresult`.
    ///
    /// Maps the well-known driver-API status codes to a [`CudaErrorKind`]
    /// variant so callers can `match` on the failure category instead of
    /// substring-matching against [`NvError::message`].
    ///
    /// The synthetic code `-1` (no `CUresult` from the driver — used
    /// when the dlopen / dlsym step itself failed before any CUDA entry
    /// point ran) maps to [`CudaErrorKind::FrameworkLoad`]. Every code
    /// the enum doesn't name explicitly flows through unchanged as
    /// [`CudaErrorKind::Other`] so future driver-status additions
    /// remain observable without a crate update.
    pub fn kind(&self) -> CudaErrorKind {
        CudaErrorKind::from_cu(self.code)
    }
}

/// Typed view of `NvError::code` covering the small set of `CUresult`
/// values this bridge cares about.
///
/// The numeric values are part of the public CUDA driver ABI
/// (`<cuda.h>` `enum cudaError_enum`) — callers can both `match` on the
/// named variants for the well-known cases and recover the raw code via
/// [`CudaErrorKind::as_code`] when they need to forward it to logging or
/// telemetry layers that already understand the driver's numbering.
///
/// `FrameworkLoad` is a bridge-internal synthetic kind (raw code `-1`)
/// used when the dlopen / dlsym step itself failed — the driver never
/// got a chance to return a `CUresult`. It's distinct from
/// `NotInitialized` (the driver loaded but `cuInit` wasn't called).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaErrorKind {
    /// `CUDA_SUCCESS` — included for completeness; an `NvError` whose
    /// `code` is `0` is malformed (success isn't an error), but the
    /// mapping is total so we name it rather than dropping it into
    /// `Other(0)`.
    Success,
    /// Synthetic: the dlopen / dlsym step failed before any CUDA driver
    /// entry point ran. The raw `code` is `-1`.
    FrameworkLoad,
    /// `CUDA_ERROR_INVALID_VALUE` (code `1`).
    InvalidValue,
    /// `CUDA_ERROR_OUT_OF_MEMORY` (code `2`).
    OutOfMemory,
    /// `CUDA_ERROR_NOT_INITIALIZED` (code `3`).
    NotInitialized,
    /// `CUDA_ERROR_DEINITIALIZED` (code `4`).
    Deinitialized,
    /// `CUDA_ERROR_NO_DEVICE` (code `100`) — no NVIDIA-driver-visible
    /// device. Together with `FrameworkLoad`, the canonical "skip this
    /// test on a non-NVIDIA host" signal.
    NoDevice,
    /// `CUDA_ERROR_INVALID_DEVICE` (code `101`) — ordinal outside
    /// `[0, device_count)`.
    InvalidDevice,
    /// `CUDA_ERROR_INVALID_CONTEXT` (code `201`) — no current context
    /// where one is required (e.g. `cuvidGetDecoderCaps`), or the
    /// context handle passed is stale.
    InvalidContext,
    /// `CUDA_ERROR_NOT_PERMITTED` (code `800`) — operation refused by
    /// the driver (container sandbox without `--gpus all` etc.).
    NotPermitted,
    /// `CUDA_ERROR_NOT_SUPPORTED` (code `801`) — operation not
    /// supported on this driver / platform combination.
    NotSupported,
    /// `CUDA_ERROR_UNKNOWN` (code `999`) — driver fall-through.
    Unknown,
    /// Any other `CUresult` the bridge doesn't name explicitly; the raw
    /// value is preserved so callers can still report it.
    Other(CUresult),
}

impl CudaErrorKind {
    /// Map a raw `CUresult` (plus the synthetic `-1` framework-load
    /// code) to a typed kind.
    pub fn from_cu(code: CUresult) -> Self {
        match code {
            CUDA_SUCCESS => Self::Success,
            -1 => Self::FrameworkLoad,
            CUDA_ERROR_INVALID_VALUE => Self::InvalidValue,
            CUDA_ERROR_OUT_OF_MEMORY => Self::OutOfMemory,
            CUDA_ERROR_NOT_INITIALIZED => Self::NotInitialized,
            CUDA_ERROR_DEINITIALIZED => Self::Deinitialized,
            CUDA_ERROR_NO_DEVICE => Self::NoDevice,
            CUDA_ERROR_INVALID_DEVICE => Self::InvalidDevice,
            CUDA_ERROR_INVALID_CONTEXT => Self::InvalidContext,
            CUDA_ERROR_NOT_PERMITTED => Self::NotPermitted,
            CUDA_ERROR_NOT_SUPPORTED => Self::NotSupported,
            CUDA_ERROR_UNKNOWN => Self::Unknown,
            other => Self::Other(other),
        }
    }

    /// Round-trip back to the raw `CUresult` carried by the original
    /// `NvError`. For `Other(code)` this echoes the wrapped value.
    pub fn as_code(self) -> CUresult {
        match self {
            Self::Success => CUDA_SUCCESS,
            Self::FrameworkLoad => -1,
            Self::InvalidValue => CUDA_ERROR_INVALID_VALUE,
            Self::OutOfMemory => CUDA_ERROR_OUT_OF_MEMORY,
            Self::NotInitialized => CUDA_ERROR_NOT_INITIALIZED,
            Self::Deinitialized => CUDA_ERROR_DEINITIALIZED,
            Self::NoDevice => CUDA_ERROR_NO_DEVICE,
            Self::InvalidDevice => CUDA_ERROR_INVALID_DEVICE,
            Self::InvalidContext => CUDA_ERROR_INVALID_CONTEXT,
            Self::NotPermitted => CUDA_ERROR_NOT_PERMITTED,
            Self::NotSupported => CUDA_ERROR_NOT_SUPPORTED,
            Self::Unknown => CUDA_ERROR_UNKNOWN,
            Self::Other(c) => c,
        }
    }

    /// True for the kinds that correspond to "no NVIDIA stack present"
    /// — the typed analogue of [`NvError::is_unavailable`]. Hosts
    /// without a driver or without an NVIDIA device produce one of
    /// these; tests use it to skip cleanly.
    ///
    /// Includes `FrameworkLoad` (no library), `NoDevice` (driver loaded
    /// but reports zero devices), and `NotInitialized` (which the
    /// `Cuda::init` `OnceLock` shouldn't normally let escape — but
    /// covered for robustness in case a caller bypasses the handle).
    pub fn is_unavailable(self) -> bool {
        matches!(
            self,
            Self::FrameworkLoad | Self::NoDevice | Self::NotInitialized
        )
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
