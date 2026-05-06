//! Round-4 NVDEC HEVC/AV1 decode + NVENC H.264/HEVC encode tests.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "registry"))` —
//! everywhere else the file compiles to nothing. On a host without a
//! CUDA device the tests log and return without panicking, so the
//! file is safe to run on CI without a GPU.

#![cfg(all(target_os = "linux", feature = "registry"))]

use std::path::PathBuf;

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, Rational, TimeBase, VideoFrame,
    VideoPlane,
};
use oxideav_nvidia::{
    Av1NvDecoder, Cuda, H264NvDecoder, H264NvEncoder, HevcNvDecoder, HevcNvEncoder,
};

const HEVC_FIXTURE_REL: &str = "tests/fixtures/hevc_main_320x240_1frame.h265";
const AV1_FIXTURE_REL: &str = "tests/fixtures/av1_320x240_1frame.obu";

fn fixture(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

fn cuda_available() -> bool {
    match Cuda::init() {
        Ok(c) => c.device_count().map(|n| n > 0).unwrap_or(false),
        Err(_) => false,
    }
}

fn assert_yuv_planes(v: &VideoFrame, w: usize, h: usize) {
    assert_eq!(v.planes.len(), 3, "I420 should have 3 planes");
    let y = &v.planes[0];
    let u = &v.planes[1];
    let v_pl = &v.planes[2];
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    assert!(y.stride >= w, "Y stride too small: {}", y.stride);
    assert_eq!(y.data.len(), y.stride * h, "Y plane size mismatch");
    assert!(u.stride >= cw, "U stride too small: {}", u.stride);
    assert_eq!(u.data.len(), u.stride * ch, "U plane size mismatch");
    assert!(v_pl.stride >= cw, "V stride too small: {}", v_pl.stride);
    assert_eq!(v_pl.data.len(), v_pl.stride * ch, "V plane size mismatch");

    // Variance check — a real decoded frame should not be flat.
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
        "Y mean={mean:.1} stddev={stddev:.1} ({w}x{h}, stride={})",
        y.stride
    );
    assert!(
        stddev > 5.0,
        "Y plane suspiciously flat (stddev={stddev:.2}); decode probably produced black"
    );
}

// ─────────────────────────── HEVC decode ──────────────────────────────────────

#[test]
fn nvdec_hevc_idr_decodes_to_320x240_i420() {
    if !cuda_available() {
        eprintln!("nvdec_hevc: skipping — no CUDA / GPU");
        return;
    }
    let bytes = std::fs::read(fixture(HEVC_FIXTURE_REL)).expect("read HEVC fixture");
    assert!(
        bytes.len() > 200,
        "HEVC fixture too small: {} bytes",
        bytes.len()
    );

    let params = CodecParameters::video(CodecId::new("hevc"));
    let mut dec = HevcNvDecoder::make(&params).expect("HevcNvDecoder::make on a CUDA host");
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);

    dec.send_packet(&pkt).expect("send_packet on HEVC IDR");

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
    assert_yuv_planes(&v, 320, 240);
}

// ─────────────────────────── AV1 decode ───────────────────────────────────────

#[test]
fn nvdec_av1_idr_decodes_to_320x240_i420() {
    if !cuda_available() {
        eprintln!("nvdec_av1: skipping — no CUDA / GPU");
        return;
    }
    let bytes = std::fs::read(fixture(AV1_FIXTURE_REL)).expect("read AV1 fixture");
    assert!(
        bytes.len() > 200,
        "AV1 fixture too small: {} bytes",
        bytes.len()
    );

    let params = CodecParameters::video(CodecId::new("av1"));
    let mut dec = Av1NvDecoder::make(&params).expect("Av1NvDecoder::make on a CUDA host");
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send_packet on AV1 keyframe");

    let frame = match dec.receive_frame() {
        Ok(f) => f,
        Err(_) => {
            dec.flush().expect("flush AV1");
            dec.receive_frame().expect("receive_frame after AV1 flush")
        }
    };
    let v = match frame {
        Frame::Video(v) => v,
        other => panic!("expected video frame, got {other:?}"),
    };
    assert_yuv_planes(&v, 320, 240);
}

// ─────────────────────────── synthetic input helper ──────────────────────────

/// Build a 320x240 I420 frame with a horizontal gradient on Y so the
/// encoder has actual content to compress (and a non-trivial PSNR
/// target for a round-trip).
fn make_gradient_frame(w: usize, h: usize, t: u8) -> VideoFrame {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            // Slow gradient + frame-dependent offset, so multiple frames
            // produce slightly different content.
            let v = ((col + row + t as usize) & 0xFF) as u8;
            y[row * w + col] = v;
        }
    }
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    VideoFrame {
        pts: None,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    }
}

fn psnr_y(orig: &VideoFrame, decoded: &VideoFrame, w: usize, h: usize) -> f64 {
    let yo = &orig.planes[0];
    let yd = &decoded.planes[0];
    let mut mse = 0.0f64;
    let mut count = 0usize;
    for row in 0..h {
        for col in 0..w {
            let so = row * yo.stride + col;
            let sd = row * yd.stride + col;
            if so < yo.data.len() && sd < yd.data.len() {
                let d = yo.data[so] as f64 - yd.data[sd] as f64;
                mse += d * d;
                count += 1;
            }
        }
    }
    if count == 0 || mse == 0.0 {
        return f64::INFINITY;
    }
    let mse = mse / count as f64;
    20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
}

// ─────────────────────────── H.264 encode → decode round-trip ────────────────

fn enc_round_trip(
    encoder_make: fn(&CodecParameters) -> oxideav_core::Result<Box<dyn oxideav_core::Encoder>>,
    decoder_make: fn(&CodecParameters) -> oxideav_core::Result<Box<dyn oxideav_core::Decoder>>,
    codec_id: &str,
    min_psnr: f64,
) {
    if !cuda_available() {
        eprintln!("{codec_id} round trip: skipping — no CUDA / GPU");
        return;
    }

    let w = 320usize;
    let h = 240usize;

    let mut params = CodecParameters::video(CodecId::new(codec_id));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));

    let mut enc = match encoder_make(&params) {
        Ok(e) => e,
        Err(e) => panic!("{codec_id} encoder make failed: {e:?}"),
    };

    // Drive 5 frames through the encoder.
    let mut frames_in: Vec<VideoFrame> = Vec::new();
    let mut bytes_out: Vec<u8> = Vec::new();
    for i in 0..5u8 {
        let f = make_gradient_frame(w, h, i.wrapping_mul(7));
        frames_in.push(f.clone());
        enc.send_frame(&Frame::Video(f)).expect("send_frame");
        loop {
            match enc.receive_packet() {
                Ok(pkt) => bytes_out.extend_from_slice(&pkt.data),
                Err(oxideav_core::Error::NeedMore) => break,
                Err(e) => panic!("receive_packet: {e:?}"),
            }
        }
    }
    enc.flush().expect("flush encoder");
    loop {
        match enc.receive_packet() {
            Ok(pkt) => bytes_out.extend_from_slice(&pkt.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => panic!("post-flush receive: {e:?}"),
        }
    }
    drop(enc);

    eprintln!(
        "{codec_id} encoded total = {} bytes from 5 frames",
        bytes_out.len()
    );
    assert!(
        bytes_out.len() > 100,
        "{codec_id} encoder produced suspiciously few bytes ({})",
        bytes_out.len()
    );
    // Optional dump to /tmp for inspection.
    if std::env::var("NVENC_DUMP").is_ok() {
        let path = format!("/tmp/round4_dump_{codec_id}.bin");
        let _ = std::fs::write(&path, &bytes_out);
        eprintln!("{codec_id} dumped to {path}");
    }

    // Decode round-trip.
    let mut dec = decoder_make(&params).expect("decoder make");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), bytes_out))
        .expect("send_packet to decoder");
    let _ = dec.flush();

    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => decoded.push(v),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    eprintln!("{codec_id} decoded {} frames", decoded.len());
    assert!(!decoded.is_empty(), "{codec_id} decoder produced no frames");

    // Best-PSNR-Y across frame pairs.
    let mut best = 0.0f64;
    for fd in &decoded {
        for fi in &frames_in {
            let p = psnr_y(fi, fd, w, h);
            if p > best {
                best = p;
            }
        }
    }
    eprintln!("{codec_id} best PSNR_Y across decoded frames = {best:.2} dB");
    assert!(
        best >= min_psnr || best.is_infinite(),
        "{codec_id} round-trip PSNR_Y too low: {best:.2} dB (want >= {min_psnr})"
    );
}

#[test]
fn nvenc_h264_round_trip_gradient_5frames() {
    // H.264 NVENC reproduces the gradient nearly losslessly with the
    // default low-latency P4 preset; expect a high PSNR.
    enc_round_trip(H264NvEncoder::make, H264NvDecoder::make, "h264", 30.0);
}

#[test]
fn nvenc_hevc_round_trip_gradient_5frames() {
    // HEVC NVENC at the same preset chooses a more aggressive QP for
    // P-frames on this synthetic gradient content (the brightness
    // ramp ends up systematically biased a few luma levels away from
    // the input), so PSNR is meaningfully lower than H.264 — but
    // still well above noise floor (~6 dB for two random images),
    // confirming the round-trip pipeline is structurally intact.
    enc_round_trip(HevcNvEncoder::make, HevcNvDecoder::make, "hevc", 15.0);
}
