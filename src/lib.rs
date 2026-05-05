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
//! Round 4 (this commit): NVDEC decoders for HEVC and AV1, plus NVENC
//! encoders for H.264 and HEVC. The cuvidParser pipeline from Round 3
//! generalises trivially across codecs — the [`decoder::NvDecoder`]
//! struct is now codec-agnostic and the public [`decoder::H264NvDecoder`],
//! [`decoder::HevcNvDecoder`], and [`decoder::Av1NvDecoder`] are thin
//! wrappers around it. NVENC support follows the standard NVENC pattern:
//! [`encoder::NvEncoder`] resolves the function table via
//! `NvEncodeAPICreateInstance`, opens a CUDA-backed encode session, asks
//! the driver for the default `NV_ENC_CONFIG` for a `P4` preset, and
//! pumps NV12 frames through `nvEncEncodePicture` →
//! `nvEncLockBitstream`.
//!
//! Round 3: real H.264 hardware decode end-to-end via NVDEC. The
//! [`decoder::H264NvDecoder`] uses the cuvidParser bitstream layer
//! (`cuvidCreateVideoParser` → `cuvidParseVideoData` with sequence /
//! decode / display callbacks), `cuvidCreateDecoder`, and
//! `cuvidMapVideoFrame64` + `cuMemcpyDtoH_v2` to deliver planar I420
//! [`oxideav_core::VideoFrame`]s.
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

#[cfg(feature = "registry")]
pub mod encoder;

pub use device::{Cuda, CudaContext, CudaDevice, NvError};
pub use nvdec::{nvdec_caps, NvdecCaps};
pub use sys::CudaVideoCodec;

#[cfg(feature = "registry")]
pub use decoder::{Av1NvDecoder, H264NvDecoder, HevcNvDecoder, NvDecoder};

#[cfg(feature = "registry")]
pub use encoder::{H264NvEncoder, HevcNvEncoder, NvEncoder};

/// Register the NVDEC and NVENC factories with the codec registry.
///
/// Round 4 wires up:
///
/// * NVDEC decoders for H.264, HEVC, and AV1
/// * NVENC encoders for H.264 and HEVC
///
/// Each factory does the runtime driver-availability check and returns
/// `Error::Unsupported` if CUDA / NVDEC / NVENC isn't usable on the
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

    // ── H.264 encoder via NVENC ────────────────────────────────────────────
    ctx.codecs.register(
        CodecInfo::new(CodecId::new("h264"))
            .capabilities(
                CodecCapabilities::video("h264_nvenc")
                    .with_lossy(true)
                    .with_intra_only(false)
                    .with_hardware(true)
                    .with_priority(5)
                    .with_encode(),
            )
            .encoder(encoder::H264NvEncoder::make),
    );

    // ── HEVC decoder via NVDEC ─────────────────────────────────────────────
    let hevc_caps = CodecCapabilities::video("hevc_nvdec")
        .with_lossy(true)
        .with_intra_only(false)
        .with_hardware(true)
        .with_priority(5);

    ctx.codecs.register(
        CodecInfo::new(CodecId::new("hevc"))
            .capabilities(hevc_caps.with_decode())
            .decoder(decoder::HevcNvDecoder::make)
            .tags([
                CodecTag::fourcc(b"hvc1"),
                CodecTag::fourcc(b"hev1"),
                CodecTag::fourcc(b"HEVC"),
                CodecTag::fourcc(b"H265"),
                CodecTag::matroska("V_MPEGH/ISO/HEVC"),
            ]),
    );

    // ── HEVC encoder via NVENC ─────────────────────────────────────────────
    ctx.codecs.register(
        CodecInfo::new(CodecId::new("hevc"))
            .capabilities(
                CodecCapabilities::video("hevc_nvenc")
                    .with_lossy(true)
                    .with_intra_only(false)
                    .with_hardware(true)
                    .with_priority(5)
                    .with_encode(),
            )
            .encoder(encoder::HevcNvEncoder::make),
    );

    // ── AV1 decoder via NVDEC (Blackwell+) ─────────────────────────────────
    let av1_caps = CodecCapabilities::video("av1_nvdec")
        .with_lossy(true)
        .with_intra_only(false)
        .with_hardware(true)
        .with_priority(5);

    ctx.codecs.register(
        CodecInfo::new(CodecId::new("av1"))
            .capabilities(av1_caps.with_decode())
            .decoder(decoder::Av1NvDecoder::make)
            .tags([
                CodecTag::fourcc(b"AV01"),
                CodecTag::fourcc(b"av01"),
                CodecTag::matroska("V_AV1"),
            ]),
    );
}

#[cfg(feature = "registry")]
oxideav_core::register!("nvidia", register);
