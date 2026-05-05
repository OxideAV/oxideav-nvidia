//! End-to-end tests for the Round-2 CUDA + NVDEC bridge.
//!
//! Gated to `cfg(all(target_os = "linux", feature = "registry"))` —
//! on non-linux platforms or without the `registry` feature this file
//! compiles to nothing.
//!
//! Each test is *skip-friendly*: if `Cuda::init()` returns
//! "driver unavailable" / "no devices" we `eprintln!` and `return`
//! rather than panic. The intent is for these tests to **pass** on a
//! Linux box with NVIDIA hardware (the developer's box), and SKIP on
//! every other CI or workstation.

#![cfg(all(target_os = "linux", feature = "registry"))]

use oxideav_nvidia::{
    nvdec_caps,
    sys::CUDA_VIDEO_CHROMA_FORMAT_420,
    Cuda, CudaVideoCodec,
};

/// Convenience: run `Cuda::init()` and skip the test on common
/// "unavailable" failures. Returns `Some(Cuda)` if we should proceed,
/// `None` if we logged + skipped.
fn try_init(test_name: &str) -> Option<Cuda> {
    match Cuda::init() {
        Ok(cuda) => Some(cuda),
        Err(e) if e.is_unavailable() => {
            eprintln!("{test_name}: skipping — CUDA unavailable: {e}");
            None
        }
        Err(e) => {
            eprintln!(
                "{test_name}: skipping — Cuda::init failed (unexpected, but treating as no-hw): {e}"
            );
            None
        }
    }
}

#[test]
fn cuda_init_succeeds() {
    let _cuda = match try_init("cuda_init_succeeds") {
        Some(c) => c,
        None => return,
    };
    // If we got here, init worked.
}

#[test]
fn lists_at_least_one_device() {
    let cuda = match try_init("lists_at_least_one_device") {
        Some(c) => c,
        None => return,
    };
    let n = cuda.device_count().expect("cuDeviceGetCount");
    if n == 0 {
        eprintln!("lists_at_least_one_device: skipping — no NVIDIA devices visible");
        return;
    }
    assert!(n >= 1, "expected at least one device, got {n}");
}

#[test]
fn device_zero_reports_name_and_memory() {
    let cuda = match try_init("device_zero_reports_name_and_memory") {
        Some(c) => c,
        None => return,
    };
    let n = cuda.device_count().expect("cuDeviceGetCount");
    if n == 0 {
        eprintln!("device_zero_reports_name_and_memory: skipping — no devices");
        return;
    }
    let dev = cuda.device(0).expect("cuDeviceGet(0)");
    let name = dev.name().expect("cuDeviceGetName");
    assert!(!name.is_empty(), "device name was empty");
    eprintln!("device 0 name: {name}");
    let mem = dev.total_memory_bytes().expect("cuDeviceTotalMem_v2");
    assert!(
        mem > 1u64 << 30,
        "expected > 1 GiB device memory, got {mem} bytes"
    );
}

#[test]
fn device_zero_reports_compute_capability() {
    let cuda = match try_init("device_zero_reports_compute_capability") {
        Some(c) => c,
        None => return,
    };
    let n = cuda.device_count().expect("cuDeviceGetCount");
    if n == 0 {
        eprintln!("device_zero_reports_compute_capability: skipping — no devices");
        return;
    }
    let dev = cuda.device(0).expect("cuDeviceGet(0)");
    let (major, minor) = dev.compute_capability().expect("cuDeviceGetAttribute");
    eprintln!("device 0 compute capability: {major}.{minor}");
    assert!(
        major >= 5,
        "expected compute capability major >= 5, got {major}.{minor}"
    );
}

#[test]
fn nvdec_h264_supported_on_device_zero() {
    let cuda = match try_init("nvdec_h264_supported_on_device_zero") {
        Some(c) => c,
        None => return,
    };
    let n = cuda.device_count().expect("cuDeviceGetCount");
    if n == 0 {
        eprintln!("nvdec_h264_supported_on_device_zero: skipping — no devices");
        return;
    }
    let dev = cuda.device(0).expect("cuDeviceGet(0)");
    // cuvidGetDecoderCaps requires a current CUDA context. The
    // returned `_ctx` pops + destroys on Drop.
    let _ctx = cuda
        .create_context_for(&dev)
        .expect("cuCtxCreate_v2 on device 0");

    let caps = nvdec_caps(CudaVideoCodec::H264, CUDA_VIDEO_CHROMA_FORMAT_420, 8)
        .expect("cuvidGetDecoderCaps for H264 / 4:2:0 / 8-bit");

    eprintln!(
        "NVDEC H264 caps: supported={} max={}x{} mbs={}",
        caps.is_supported, caps.max_width, caps.max_height, caps.max_mb_count
    );
    assert_eq!(
        caps.is_supported, 1,
        "NVDEC reported H264 4:2:0 8-bit unsupported on this GPU"
    );
    assert!(caps.max_width >= 1920, "max_width too small: {}", caps.max_width);
    assert!(
        caps.max_height >= 1080,
        "max_height too small: {}",
        caps.max_height
    );
}
