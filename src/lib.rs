#![cfg(target_os = "linux")]
//! Linux NVIDIA NVDEC + NVENC hardware decode/encode bridge.
//!
//! This crate is a **runtime-loaded** bridge to NVIDIA's proprietary
//! Linux video toolchain:
//!
//! * `libcuda.so.1` — CUDA driver API (context, device memory).
//! * `libnvcuvid.so.1` — NVDEC (hardware video decode).
//! * `libnvidia-encode.so.1` — NVENC (hardware video encode).
//!
//! All three are opened via [`libloading`] on first use, so:
//!
//! * Linux builds have **no compile-time link dependency** on the
//!   CUDA SDK; if any library can't be loaded, the registered
//!   factories return `Error::Unsupported` and the framework registry
//!   falls back to the pure-Rust codec implementation.
//! * No bindgen, no `*-sys` crate. Symbol resolution and `CUresult` /
//!   `NVENCSTATUS` propagation is all done by hand.
//!
//! The crate is gated to `cfg(target_os = "linux")` at the source
//! level: on macOS / Windows the entire crate compiles to an empty
//! rlib, and consumers (umbrella `oxideav`) gate the `register` call
//! behind the same cfg.
//!
//! # Status
//!
//! Round 2 (this commit): in addition to the Round 1 scaffolding the
//! crate now exposes safe wrappers for CUDA driver init + device
//! enumeration ([`device::Cuda`], [`device::CudaDevice`],
//! [`device::CudaContext`]), and an NVDEC capability query
//! ([`nvdec::nvdec_caps`]). On a host with the NVIDIA driver and at
//! least one supported GPU, `tests/round2_init.rs` end-to-end calls
//! `cuInit` → `cuDeviceGet` → `cuCtxCreate_v2` → `cuvidGetDecoderCaps`
//! and asserts the H.264 / 4:2:0 / 8-bit combo is reported supported.
//!
//! Round 3 will wire up the NVDEC + NVENC `Decoder` / `Encoder` trait
//! factories so the crate plugs into the framework registry like
//! `oxideav-videotoolbox` does.
//!
//! # Workspace policy
//!
//! Calling a system OS / driver API via FFI is the same shape as
//! calling `libc::malloc` — it's the platform, not a copied
//! algorithm. The workspace's clean-room rule (no embedding source
//! from libvpx, libwebp, libjxl, etc.) doesn't apply here.

pub mod device;
pub mod nvdec;
pub mod sys;

pub use device::{Cuda, CudaContext, CudaDevice, NvError};
pub use nvdec::{nvdec_caps, NvdecCaps};
pub use sys::CudaVideoCodec;

/// Confirm the NVIDIA framework loads, but do not register any codec
/// factories yet (Round 1 scaffolding).
///
/// If the driver isn't installed (no NVIDIA hardware, AMD-only system,
/// container without `--gpus all`, etc.) the function logs and
/// returns — the runtime falls back to the pure-Rust impls.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {
    match sys::framework() {
        Ok(_) => {
            // Round 1: framework loads. No factories wired up yet.
        }
        Err(e) => {
            eprintln!("oxideav-nvidia: library unavailable, skipping registration: {e}");
        }
    }
}

#[cfg(feature = "registry")]
oxideav_core::register!("nvidia", register);
