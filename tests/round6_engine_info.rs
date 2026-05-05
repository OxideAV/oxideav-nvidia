//! Round-6 engine-probe tests.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. Skip-friendly: when
//! `engine_info()` returns an empty vec we log + `return` rather than
//! panic, so the suite passes on every CI host with no GPU and *also*
//! passes on the developer's RTX 5080 box where it actually exercises
//! the per-codec NVDEC / NVENC capability surface.
//!
//! See `crates/oxideav-core/src/engine.rs` for the [`HwDeviceInfo`] /
//! [`HwCodecCaps`] schema this exercises.

#![cfg(all(target_os = "linux", feature = "registry"))]

#[test]
fn engine_info_smoke_does_not_panic() {
    // Universal skip-friendly invariant: calling `engine_info()` on
    // any host — including hosts with no NVIDIA driver / no GPU —
    // must never panic. An empty vec is the correct "nothing here"
    // response.
    let _probes = oxideav_nvidia::engine_info();
}

#[test]
fn engine_info_finds_rtx_5080_or_skips() {
    let probes = oxideav_nvidia::engine_info();
    if probes.is_empty() {
        eprintln!("engine_info_finds_rtx_5080_or_skips: skipping — no NVIDIA GPU available");
        return;
    }

    let dev = &probes[0];
    eprintln!("device 0: {dev:?}");
    assert!(!dev.name.is_empty(), "device name non-empty");
    assert!(dev.total_memory_bytes.is_some(), "memory reported");
    assert!(
        dev.driver_version.is_some(),
        "driver_version populated from cuDriverGetVersion"
    );
    assert!(
        dev.api_version
            .as_deref()
            .map(|s| s.starts_with("CUDA "))
            .unwrap_or(false),
        "api_version starts with `CUDA `: {:?}",
        dev.api_version
    );
    assert!(
        dev.extra
            .iter()
            .any(|(k, _)| k == "compute_capability"),
        "compute_capability extra reported"
    );

    let h264 = dev.codecs.iter().find(|c| c.codec == "h264");
    assert!(h264.is_some(), "h264 entry present");
    let h264 = h264.unwrap();
    assert!(h264.decode, "h264 decode supported");
    assert!(
        h264.max_width.unwrap_or(0) >= 1920,
        "h264 max_width >= 1920: {:?}",
        h264.max_width
    );
}

#[test]
fn engine_info_reports_modern_codecs_or_skips() {
    let probes = oxideav_nvidia::engine_info();
    if probes.is_empty() {
        eprintln!(
            "engine_info_reports_modern_codecs_or_skips: skipping — no NVIDIA GPU available"
        );
        return;
    }
    let dev = &probes[0];
    // Each codec we report should have a string id we recognise.
    for codec in &dev.codecs {
        eprintln!(
            "codec {:>10}  decode={}  encode={}  max={:?}x{:?}  bit_depth={:?}",
            codec.codec,
            codec.decode,
            codec.encode,
            codec.max_width,
            codec.max_height,
            codec.max_bit_depth,
        );
        assert!(
            matches!(
                codec.codec.as_str(),
                "h264" | "hevc" | "av1" | "vp9" | "vp8" | "mpeg2video"
            ),
            "unexpected codec id: {}",
            codec.codec
        );
    }
}
