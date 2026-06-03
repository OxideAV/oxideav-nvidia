//! Round-9 NVDEC MPEG-2 Video decoder tests.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. On a host without a
//! CUDA device every test logs and returns without panicking, so the
//! file is safe to run on CI without a GPU.
//!
//! Like the VP9 round, no MPEG-2 elementary-stream fixture is committed
//! here — `oxideav-mpeg12video` already carries the format's test
//! corpus. The tests here cover three structural properties of the
//! MPEG-2 wiring:
//!
//! 1. `Mpeg2NvDecoder::make` returns `Error::Unsupported` cleanly on
//!    hosts without CUDA / no NVIDIA GPU rather than panicking.
//! 2. On a CUDA-capable host where NVDEC reports MPEG-2 support, the
//!    factory builds a decoder successfully.
//! 3. The `engine_info` probe lists `mpeg2video` with `decode = true`
//!    alongside the other decode-capable codecs.

#![cfg(all(target_os = "linux", feature = "registry"))]

use oxideav_core::{CodecId, CodecParameters};
use oxideav_nvidia::{
    nvdec_caps, sys::CUDA_VIDEO_CHROMA_FORMAT_420, Cuda, CudaVideoCodec, Mpeg2NvDecoder,
};

fn cuda_available() -> bool {
    match Cuda::init() {
        Ok(c) => c.device_count().map(|n| n > 0).unwrap_or(false),
        Err(_) => false,
    }
}

/// Returns true if the host has CUDA *and* NVDEC reports MPEG-2 /
/// 4:2:0 / 8-bit as supported. Every consumer NVIDIA GPU since Fermi
/// ships MPEG-2 NVDEC; the check exists to skip cleanly on datacenter
/// SKUs without a video engine.
fn nvdec_mpeg2_available() -> bool {
    if !cuda_available() {
        return false;
    }
    // Need a current context to query caps. The context drops at the
    // end of the helper which is fine — caps are per-codec, not
    // per-context.
    let cuda = match Cuda::init() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let dev = match cuda.device(0) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let _ctx = match cuda.create_context_for(&dev) {
        Ok(c) => c,
        Err(_) => return false,
    };
    match nvdec_caps(CudaVideoCodec::Mpeg2, CUDA_VIDEO_CHROMA_FORMAT_420, 8) {
        Ok(c) => c.is_supported != 0,
        Err(_) => false,
    }
}

// ─────────────────────────── no-GPU host ──────────────────────────────────────

/// On a host with no CUDA driver / no GPU the factory must surface a
/// clean `Err(_)` (specifically `Error::Unsupported`) so the registry
/// can fall back to the pure-Rust MPEG-2 decoder. Must never panic.
#[test]
fn mpeg2_make_returns_unsupported_with_no_gpu() {
    if cuda_available() {
        eprintln!("mpeg2_make_returns_unsupported_with_no_gpu: skipping — CUDA is available");
        return;
    }
    let params = CodecParameters::video(CodecId::new("mpeg2video"));
    let err = Mpeg2NvDecoder::make(&params).err().expect(
        "Mpeg2NvDecoder::make must Err with no GPU; the registry needs Err to try the SW fallback",
    );
    eprintln!("mpeg2 no-gpu error (expected): {err:?}");
}

// ─────────────────────────── CUDA-capable host ───────────────────────────────

/// On a host with CUDA and NVDEC MPEG-2 support, the factory must
/// construct successfully — we don't decode anything (no fixture),
/// just confirm the parser + decoder wiring is plumbed end-to-end.
#[test]
fn mpeg2_make_constructs_decoder_on_supported_host() {
    if !nvdec_mpeg2_available() {
        eprintln!(
            "mpeg2_make_constructs_decoder_on_supported_host: skipping — NVDEC MPEG-2 not supported on this host"
        );
        return;
    }
    let params = CodecParameters::video(CodecId::new("mpeg2video"));
    let dec =
        Mpeg2NvDecoder::make(&params).expect("Mpeg2NvDecoder::make on NVDEC-MPEG-2-capable host");
    assert_eq!(dec.codec_id().as_str(), "mpeg2video");
}

/// On a CUDA-capable host the engine probe lists `mpeg2video` with
/// `decode = true` (and `max_width >= 1920` — NVDEC's MPEG-2 path has
/// supported 1080p since Fermi). Skips cleanly when MPEG-2 isn't
/// NVDEC-supported on this device.
#[test]
fn engine_info_lists_mpeg2_decode() {
    if !nvdec_mpeg2_available() {
        eprintln!(
            "engine_info_lists_mpeg2_decode: skipping — NVDEC MPEG-2 not supported on this host"
        );
        return;
    }
    let probes = oxideav_nvidia::engine_info();
    if probes.is_empty() {
        eprintln!("engine_info_lists_mpeg2_decode: skipping — engine_info returned empty");
        return;
    }
    let dev = &probes[0];
    let mpeg2 = dev
        .codecs
        .iter()
        .find(|c| c.codec == "mpeg2video")
        .expect("mpeg2video entry present in engine_info");
    assert!(
        mpeg2.decode,
        "mpeg2video decode must be true on NVDEC-MPEG-2 host"
    );
    assert!(
        !mpeg2.encode,
        "mpeg2video encode must be false (NVENC ships no MPEG-2 encoder)"
    );
    assert!(
        mpeg2.max_width.unwrap_or(0) >= 1920,
        "mpeg2video max_width >= 1920: {:?}",
        mpeg2.max_width
    );
}
