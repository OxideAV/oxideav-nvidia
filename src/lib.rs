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
//! Round 3 (this commit): real H.264 hardware decode end-to-end via
//! NVDEC. The new [`decoder::H264NvDecoder`] uses the cuvidParser
//! bitstream layer (`cuvidCreateVideoParser` → `cuvidParseVideoData`
//! with sequence / decode / display callbacks), `cuvidCreateDecoder`,
//! and `cuvidMapVideoFrame64` + `cuMemcpyDtoH_v2` to deliver planar
//! I420 [`oxideav_core::VideoFrame`]s. `register()` now wires the H.264
//! factory into the codec registry with `with_priority(5)`. NVENC
//! encode is still on the roadmap.
//!
//! Round 2: safe wrappers for CUDA driver init + device enumeration
//! ([`device::Cuda`], [`device::CudaDevice`], [`device::CudaContext`])
//! and an NVDEC capability query ([`nvdec::nvdec_caps`]).
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

#[cfg(feature = "registry")]
pub mod decoder;

pub use device::{Cuda, CudaContext, CudaDevice, NvError};
pub use nvdec::{nvdec_caps, NvdecCaps};
pub use sys::CudaVideoCodec;

#[cfg(feature = "registry")]
pub use decoder::H264NvDecoder;

/// Register the NVDEC H.264 decoder factory with the codec registry.
///
/// The factory itself does the runtime driver-availability check and
/// returns `Error::Unsupported` if CUDA / NVDEC isn't usable on the
/// host — but we still gate the registration on a successful framework
/// dlopen at startup so the registry doesn't keep a slot reserved on
/// systems with no NVIDIA hardware at all.
#[cfg(feature = "registry")]
pub fn register(ctx: &mut oxideav_core::RuntimeContext) {
    use oxideav_core::{CodecCapabilities, CodecId, CodecInfo, CodecTag};

    match sys::framework() {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "oxideav-nvidia: library unavailable, skipping registration: {e}"
            );
            return;
        }
    }

    // ── H.264 decoder via NVDEC ────────────────────────────────────────────
    let h264_caps = CodecCapabilities::video("h264_nvdec")
        .with_lossy(true)
        .with_intra_only(false)
        .with_hardware(true)
        // priority 5: lower than the platform-native VideoToolbox (10)
        // when both are available, but well above any pure-software
        // fallback. Adjust if VAAPI/VDPAU sit at a different rank.
        .with_priority(5);

    ctx.codecs.register(
        CodecInfo::new(CodecId::new("h264"))
            .capabilities(h264_caps.with_decode())
            .decoder(decoder::H264NvDecoder::make)
            .tags([
                CodecTag::fourcc(b"H264"),
                CodecTag::fourcc(b"h264"),
                CodecTag::fourcc(b"AVC1"),
                CodecTag::fourcc(b"avc1"),
                CodecTag::fourcc(b"X264"),
                CodecTag::matroska("V_MPEG4/ISO/AVC"),
            ]),
    );
}

#[cfg(feature = "registry")]
oxideav_core::register!("nvidia", register);
