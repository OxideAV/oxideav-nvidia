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
//! Round 1 (this commit): scaffolding only. The framework load is
//! verified via `sys::framework()`; no codec factories are wired up
//! yet. Round 2 will add H.264 + HEVC decode via NVDEC's
//! `cuvidCreateDecoder` + `cuvidDecodePicture`.
//!
//! # Workspace policy
//!
//! Calling a system OS / driver API via FFI is the same shape as
//! calling `libc::malloc` — it's the platform, not a copied
//! algorithm. The workspace's clean-room rule (no embedding source
//! from libvpx, libwebp, libjxl, etc.) doesn't apply here.

pub mod sys;

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
