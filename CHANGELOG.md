# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Round 2

- New module `device` with safe CUDA wrappers:
  - `Cuda::init()` runs `cuInit(0)` once per process via `OnceLock`
    and returns a zero-sized handle that proves the driver is up.
  - `Cuda::device_count()` / `Cuda::device(ordinal)` for enumeration.
  - `CudaDevice::name()` (uses `cuDeviceGetName`, 256-byte buffer),
    `CudaDevice::total_memory_bytes()` (uses `cuDeviceTotalMem_v2`),
    and `CudaDevice::compute_capability()` (uses `cuDeviceGetAttribute`
    with `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR = 75/76`).
  - `Cuda::create_context_for(&CudaDevice)` returning a `CudaContext`
    that owns the `CUcontext` and calls `cuCtxDestroy_v2` on Drop.
  - `NvError` wraps `CUresult` and lazily resolves the error message
    via `cuGetErrorString`. `is_unavailable()` distinguishes
    "no driver / no GPU / no `--gpus all`" from real failures so tests
    can skip cleanly.
- New module `nvdec` with `nvdec_caps(codec, chroma, bit_depth)` calling
  `cuvidGetDecoderCaps`. Returns a public `NvdecCaps` struct mirroring
  the *out* fields of `CUVIDDECODECAPS`
  (`is_supported`, `num_nvdecs`, `output_format_mask`, `max_width`,
  `max_height`, `max_mb_count`, `min_width`, `min_height`,
  `is_histogram_supported`, `counter_bit_depth`, `max_histogram_bins`).
- New `CudaVideoCodec` enum (`Mpeg1=0`, `Mpeg2=1`, …, `H264=4`,
  `Hevc=8`, `Vp9=10`, `Av1=11`) and `cudaVideoChromaFormat` constants
  (`420=1`, `422=2`, `444=3`, `Monochrome=0`).
- New `CUVIDDECODECAPS` `#[repr(C)]` struct mirroring the public
  vendor-supplied layout from `<cuviddec.h>`.
- New libcuda symbols in `sys::Vtable`: `cuDeviceGetName`,
  `cuDriverGetVersion`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute`,
  `cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`.
- `cuvid_get_decoder_caps` retyped from `*mut c_void` to
  `*mut CUVIDDECODECAPS` for sounder callers.
- Integration test `tests/round2_init.rs` with five skip-friendly
  tests:
  - `cuda_init_succeeds`
  - `lists_at_least_one_device`
  - `device_zero_reports_name_and_memory` (>1 GiB)
  - `device_zero_reports_compute_capability` (major >= 5)
  - `nvdec_h264_supported_on_device_zero` (asserts H.264 / 4:2:0 /
    8-bit is supported, max ≥ 1920×1080)
  Each test detects "CUDA driver / GPU not available" and `eprintln!`s
  + `return`s rather than panicking, so the suite passes on the
  developer's RTX 5080 box and skips cleanly elsewhere.
- `lib.rs` re-exports `Cuda`, `CudaDevice`, `CudaContext`, `NvError`,
  `NvdecCaps`, `nvdec_caps`, `CudaVideoCodec`.

### Added

- Initial scaffolding: `#![cfg(target_os = "linux")]` crate that
  dlopens `libcuda.so.1`, `libnvcuvid.so.1`, and
  `libnvidia-encode.so.1` via `libloading` on first use.
- `sys.rs` exposes opaque type aliases (`CUcontext`, `CUstream`,
  `CUresult`, `CUvideodecoder`) and a resolved `Vtable` covering the
  bootstrap symbol set:
  - libcuda: `cuInit`, `cuDeviceGet`, `cuDeviceGetCount`,
    `cuCtxCreate_v2`, `cuCtxDestroy_v2`, `cuMemAlloc_v2`,
    `cuMemFree_v2`, `cuGetErrorString`.
  - libnvcuvid: `cuvidCreateDecoder`, `cuvidDestroyDecoder`,
    `cuvidDecodePicture`, `cuvidMapVideoFrame64`,
    `cuvidUnmapVideoFrame64`, `cuvidGetDecoderCaps`.
  - libnvidia-encode: `NvEncodeAPICreateInstance` (the single
    bootstrap export; the rest of NVENC lives in a function table
    populated by that call, to be wired up in Round 2).
- Process-wide `OnceLock<Result<Vtable, String>>` cache so the
  dlopen + dlsym round-trip happens at most once per process.
- Unified `register(&mut RuntimeContext)` entry point. Round 1: the
  function confirms the libraries load and returns; no codec
  factories are wired up yet. If load fails (no NVIDIA driver, ABI
  mismatch, container without GPU passthrough, etc.) the function
  logs and returns — the pure-Rust codec path remains the only
  resolution candidate.
- Standalone-friendly `registry` feature (default-on) gates the
  `oxideav-core` + `linkme` deps.
- README coverage roadmap and priority explanation.
- Smoke tests: `frameworks_load` and `vtable_resolves` confirm
  symbol resolution on Linux machines that have the NVIDIA driver
  stack installed.
