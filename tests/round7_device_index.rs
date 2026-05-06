//! Round-7 verification of `CodecParameters::device_index` routing
//! through every NVDEC decoder + NVENC encoder factory.
//!
//! Phase 1 of `oxideav-core` shipped `CodecParameters::device_index:
//! Option<u32>` (None = backend default) plus a builder
//! `with_device_index(u32)`. The decoder + encoder factories in this
//! crate now read `params.device_index.unwrap_or(0)` and bind the
//! resulting CUDA context to the matching ordinal in
//! [`oxideav_nvidia::engine_info`]'s output.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. On a host without
//! the NVIDIA driver / no GPU each test logs and returns instead of
//! panicking, so this file is safe to run on CI workers without a GPU.

#![cfg(all(target_os = "linux", feature = "registry"))]

use oxideav_core::{CodecId, CodecParameters};
use oxideav_nvidia::{
    Av1NvDecoder, Cuda, H264NvDecoder, H264NvEncoder, HevcNvDecoder, HevcNvEncoder,
};

fn cuda_available() -> bool {
    match Cuda::init() {
        Ok(c) => c.device_count().map(|n| n > 0).unwrap_or(false),
        Err(_) => false,
    }
}

fn h264_video_params() -> CodecParameters {
    CodecParameters::video(CodecId::new("h264"))
}

fn hevc_video_params() -> CodecParameters {
    CodecParameters::video(CodecId::new("hevc"))
}

fn av1_video_params() -> CodecParameters {
    CodecParameters::video(CodecId::new("av1"))
}

// ─────────────────────────── default (None) ───────────────────────────────────

/// `device_index = None` must select device 0 (historical behaviour
/// preserved). This exercises every factory in the crate so a
/// regression on any one of them surfaces immediately.
#[test]
fn device_index_none_uses_default_device() {
    if !cuda_available() {
        eprintln!("device_index_none_uses_default_device: skipping — no CUDA / GPU");
        return;
    }

    let h264 = h264_video_params();
    assert!(
        h264.device_index.is_none(),
        "default builder must leave device_index = None"
    );

    H264NvDecoder::make(&h264).expect("H264NvDecoder::make with device_index = None");
    HevcNvDecoder::make(&hevc_video_params())
        .expect("HevcNvDecoder::make with device_index = None");
    Av1NvDecoder::make(&av1_video_params()).expect("Av1NvDecoder::make with device_index = None");
    H264NvEncoder::make(&h264_video_params())
        .expect("H264NvEncoder::make with device_index = None");
    HevcNvEncoder::make(&hevc_video_params())
        .expect("HevcNvEncoder::make with device_index = None");
}

// ─────────────────────────── explicit Some(0) ─────────────────────────────────

/// `device_index = Some(0)` must work the same as the default on every
/// host — there is always at least one CUDA device when
/// `cuda_available()` is true.
#[test]
fn device_index_zero_explicit_works() {
    if !cuda_available() {
        eprintln!("device_index_zero_explicit_works: skipping — no CUDA / GPU");
        return;
    }

    let p_h264 = h264_video_params().with_device_index(0);
    let p_hevc = hevc_video_params().with_device_index(0);
    let p_av1 = av1_video_params().with_device_index(0);
    assert_eq!(p_h264.device_index, Some(0));

    H264NvDecoder::make(&p_h264).expect("H264NvDecoder::make on device 0");
    HevcNvDecoder::make(&p_hevc).expect("HevcNvDecoder::make on device 0");
    Av1NvDecoder::make(&p_av1).expect("Av1NvDecoder::make on device 0");
    H264NvEncoder::make(&p_h264).expect("H264NvEncoder::make on device 0");
    HevcNvEncoder::make(&p_hevc).expect("HevcNvEncoder::make on device 0");
}

// ─────────────────────────── out-of-range ─────────────────────────────────────

/// `device_index` greater than the visible device count must fail
/// cleanly with `Err(...)` from every factory rather than panicking
/// or silently falling back to device 0.
#[test]
fn device_index_out_of_range_errors() {
    if !cuda_available() {
        eprintln!("device_index_out_of_range_errors: skipping — no CUDA / GPU");
        return;
    }

    let bad = 99u32;
    let p_h264 = h264_video_params().with_device_index(bad);
    let p_hevc = hevc_video_params().with_device_index(bad);
    let p_av1 = av1_video_params().with_device_index(bad);

    assert!(
        H264NvDecoder::make(&p_h264).is_err(),
        "H264NvDecoder::make must reject device_index = {bad}"
    );
    assert!(
        HevcNvDecoder::make(&p_hevc).is_err(),
        "HevcNvDecoder::make must reject device_index = {bad}"
    );
    assert!(
        Av1NvDecoder::make(&p_av1).is_err(),
        "Av1NvDecoder::make must reject device_index = {bad}"
    );
    assert!(
        H264NvEncoder::make(&p_h264).is_err(),
        "H264NvEncoder::make must reject device_index = {bad}"
    );
    assert!(
        HevcNvEncoder::make(&p_hevc).is_err(),
        "HevcNvEncoder::make must reject device_index = {bad}"
    );
}
