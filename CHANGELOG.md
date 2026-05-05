# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
