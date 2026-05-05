//! End-to-end Round-3 NVDEC H.264 decode test.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. On a host without
//! the NVIDIA driver / no GPU the test logs and returns instead of
//! panicking, so this file is safe to run on CI workers without a GPU.

#![cfg(all(target_os = "linux", feature = "registry"))]

use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_nvidia::{Cuda, H264NvDecoder};

const FIXTURE_REL: &str = "tests/fixtures/h264_baseline_320x240_1frame.h264";

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(FIXTURE_REL);
    p
}

fn cuda_available() -> bool {
    match Cuda::init() {
        Ok(c) => c.device_count().map(|n| n > 0).unwrap_or(false),
        Err(_) => false,
    }
}

fn h264_params() -> CodecParameters {
    CodecParameters::video(CodecId::new("h264"))
}

#[test]
fn nvdec_h264_idr_decodes_to_320x240_i420() {
    if !cuda_available() {
        eprintln!("nvdec_h264_idr_decodes_to_320x240_i420: skipping — no CUDA / GPU");
        return;
    }

    let bytes = std::fs::read(fixture_path()).expect("read H.264 fixture");
    assert!(
        bytes.len() > 200,
        "fixture suspiciously small: {} bytes",
        bytes.len()
    );

    let params = h264_params();
    let mut dec =
        H264NvDecoder::make(&params).expect("H264NvDecoder::make on a CUDA-capable host");

    // Hand over the whole Annex-B stream as a single packet — the
    // cuvidParser handles SPS/PPS/SEI/IDR splitting internally.
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);

    dec.send_packet(&pkt).expect("send_packet on IDR");

    // The display callback fires synchronously inside
    // cuvidParseVideoData for an IDR-only stream, so the frame should
    // be available immediately. Allow one flush in case the parser
    // chooses to delay.
    let frame = match dec.receive_frame() {
        Ok(f) => f,
        Err(_) => {
            dec.flush().expect("flush after IDR");
            dec.receive_frame().expect("receive_frame after flush")
        }
    };

    let v = match frame {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };

    assert_eq!(v.planes.len(), 3, "I420 should have 3 planes");
    let y = &v.planes[0];
    let u = &v.planes[1];
    let v_pl = &v.planes[2];

    // 320×240 Y plane: stride must be at least 320, data must be 320*240 long
    assert!(y.stride >= 320, "Y stride too small: {}", y.stride);
    assert_eq!(
        y.data.len(),
        y.stride * 240,
        "Y plane size mismatch (stride={}, want=stride*240)",
        y.stride
    );
    assert!(u.stride >= 160, "U stride too small: {}", u.stride);
    assert_eq!(u.data.len(), u.stride * 120, "U plane size mismatch");
    assert!(v_pl.stride >= 160, "V stride too small: {}", v_pl.stride);
    assert_eq!(v_pl.data.len(), v_pl.stride * 120, "V plane size mismatch");

    // testsrc2 is a colour-bar pattern — luma should be far from
    // constant. Compute rough std-dev on the Y plane.
    let n = y.data.len() as f64;
    let mean: f64 = y.data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var: f64 = y
        .data
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let stddev = var.sqrt();
    eprintln!(
        "Y mean={mean:.1} stddev={stddev:.1} (320x240, stride={})",
        y.stride
    );
    assert!(
        stddev > 5.0,
        "Y plane is suspiciously flat (stddev={stddev:.2}); decode probably produced black"
    );
}

#[test]
fn nvdec_h264_make_returns_unsupported_with_no_gpu() {
    // This test only checks the negative path *if* there is no GPU.
    // On a GPU host the factory succeeds and we just exit early.
    if cuda_available() {
        eprintln!("nvdec_h264_make_returns_unsupported_with_no_gpu: skipping — GPU is present");
        return;
    }
    let params = h264_params();
    let r = H264NvDecoder::make(&params);
    assert!(
        r.is_err(),
        "expected H264NvDecoder::make to fail on a no-GPU host"
    );
}
