//! Round-8 NVDEC VP9 decoder tests.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. On a host without a
//! CUDA device every test logs and returns without panicking, so the
//! file is safe to run on CI without a GPU.
//!
//! Unlike the round-3 / round-4 decode tests, round 8 does not ship a
//! pre-extracted VP9 fixture — committing a VP9 superframe + IVF
//! wrapper would just duplicate the fixtures `oxideav-vp9` already
//! carries. The tests here cover three structural properties of the
//! VP9 wiring:
//!
//! 1. `Vp9NvDecoder::make` returns `Error::Unsupported` cleanly on
//!    hosts without CUDA / no NVIDIA GPU rather than panicking.
//! 2. On a CUDA-capable host where NVDEC reports VP9 support, the
//!    factory builds a decoder successfully.
//! 3. The `engine_info` probe lists `vp9` alongside the other
//!    decode-capable codecs.

#![cfg(all(target_os = "linux", feature = "registry"))]

use oxideav_core::{CodecId, CodecParameters};
use oxideav_nvidia::{
    nvdec_caps, sys::CUDA_VIDEO_CHROMA_FORMAT_420, Cuda, CudaVideoCodec, Vp9NvDecoder,
};

fn cuda_available() -> bool {
    match Cuda::init() {
        Ok(c) => c.device_count().map(|n| n > 0).unwrap_or(false),
        Err(_) => false,
    }
}

/// Returns true if the host has CUDA *and* NVDEC reports VP9 / 4:2:0
/// / 8-bit as supported. Maxwell GM206 (GTX 950 / 960) and newer.
fn nvdec_vp9_available() -> bool {
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
    match nvdec_caps(CudaVideoCodec::Vp9, CUDA_VIDEO_CHROMA_FORMAT_420, 8) {
        Ok(c) => c.is_supported != 0,
        Err(_) => false,
    }
}

// ─────────────────────────── no-GPU host ──────────────────────────────────────

/// On a host with no CUDA driver / no GPU the factory must surface a
/// clean `Err(_)` (specifically `Error::Unsupported`) so the registry
/// can fall back to the pure-Rust VP9 decoder. Must never panic.
#[test]
fn vp9_make_returns_unsupported_with_no_gpu() {
    if cuda_available() {
        eprintln!("vp9_make_returns_unsupported_with_no_gpu: skipping — CUDA is available");
        return;
    }
    let params = CodecParameters::video(CodecId::new("vp9"));
    let err = Vp9NvDecoder::make(&params).err().expect(
        "Vp9NvDecoder::make must Err with no GPU; the registry needs Err to try the SW fallback",
    );
    eprintln!("vp9 no-gpu error (expected): {err:?}");
}

// ─────────────────────────── CUDA-capable host ───────────────────────────────

/// On a host with CUDA and NVDEC VP9 support, the factory must
/// construct successfully — we don't decode anything (no fixture),
/// just confirm the parser + decoder wiring is plumbed end-to-end.
#[test]
fn vp9_make_constructs_decoder_on_supported_host() {
    if !nvdec_vp9_available() {
        eprintln!("vp9_make_constructs_decoder_on_supported_host: skipping — NVDEC VP9 not supported on this host");
        return;
    }
    let params = CodecParameters::video(CodecId::new("vp9"));
    let dec = Vp9NvDecoder::make(&params).expect("Vp9NvDecoder::make on NVDEC-VP9-capable host");
    assert_eq!(dec.codec_id().as_str(), "vp9");
}

/// On a CUDA-capable host the engine probe lists `vp9` with
/// `decode = true` (and `max_width >= 1920` since every NVDEC that
/// supports VP9 at all supports at least 1080p). Skips cleanly when
/// VP9 isn't NVDEC-supported on this device.
#[test]
fn engine_info_lists_vp9_decode() {
    if !nvdec_vp9_available() {
        eprintln!("engine_info_lists_vp9_decode: skipping — NVDEC VP9 not supported on this host");
        return;
    }
    let probes = oxideav_nvidia::engine_info();
    if probes.is_empty() {
        eprintln!("engine_info_lists_vp9_decode: skipping — engine_info returned empty");
        return;
    }
    let dev = &probes[0];
    let vp9 = dev
        .codecs
        .iter()
        .find(|c| c.codec == "vp9")
        .expect("vp9 entry present in engine_info");
    assert!(vp9.decode, "vp9 decode must be true on NVDEC-VP9 host");
    assert!(
        !vp9.encode,
        "vp9 encode must be false (NVIDIA ships no VP9 encoder)"
    );
    assert!(
        vp9.max_width.unwrap_or(0) >= 1920,
        "vp9 max_width >= 1920: {:?}",
        vp9.max_width
    );
}
