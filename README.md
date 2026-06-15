# oxideav-nvidia

Linux NVIDIA NVDEC + NVENC hardware decode/encode bridge for the [oxideav](https://github.com/OxideAV/oxideav) framework.

## Why a bridge crate?

NVIDIA's NVENC + NVDEC engines deliver the highest absolute throughput on Linux for H.264 / HEVC / AV1 encode + decode on NVIDIA hardware. The toolchain is **proprietary**: bytes flow in and out of the GPU through the CUDA driver API. This crate is the bridge from oxideav's pipeline traits to that toolchain.

It is a **thin runtime-loaded bridge** — no compile-time link dependency on the CUDA SDK, no `*-sys` crate. The libraries are opened via [`libloading`] on first use.

| Library                  | Role                                                              |
|--------------------------|-------------------------------------------------------------------|
| `libcuda.so.1`           | CUDA driver API — context create, device-memory allocation        |
| `libnvcuvid.so.1`        | NVDEC video decode (H.264 / HEVC / VP9 / AV1 / MPEG-2)            |
| `libnvidia-encode.so.1`  | NVENC video encode (H.264 / HEVC / AV1)                           |

## Codec coverage

| Codec        | Decode (NVDEC)           | Encode (NVENC)            |
|--------------|--------------------------|---------------------------|
| H.264        | shipped                  | shipped                   |
| HEVC         | shipped                  | shipped                   |
| AV1          | shipped (Blackwell+)     | planned (Ada Lovelace+)   |
| VP9          | shipped (Maxwell GM206+) | — (no NVENC VP9 encoder)  |
| MPEG-2       | shipped (Fermi+)         | — (no NVENC MPEG-2 encoder) |
| MPEG-4 Pt 2  | planned                  | —                         |
| VC-1         | planned                  | —                         |
| JPEG         | planned (NVJPEG)         | —                         |

The decoder pipeline is built on the cuvidParser bitstream layer and is
codec-agnostic; the public `H264NvDecoder` / `HevcNvDecoder` /
`Av1NvDecoder` / `Vp9NvDecoder` / `Mpeg2NvDecoder` wrappers pick the
`CudaVideoCodec` and the parser's `bAnnexb` flag. The encoder resolves
the NVENC vtable via `NvEncodeAPICreateInstance`, opens a CUDA-backed
encode session, and pumps NV12 frames through `nvEncEncodePicture` →
`nvEncLockBitstream`.

## Error inspection

`NvError::kind()` returns a typed `CudaErrorKind` over the underlying
`CUresult` (named variants for the public CUDA driver-API codes the
bridge inspects, plus a synthetic `FrameworkLoad` for the dlopen / dlsym
failure path and `Other(CUresult)` for unnamed codes). `#[non_exhaustive]`
reserves room for naming further variants. `CudaErrorKind::is_unavailable()`
lights up on the documented "no NVIDIA stack present" set
(`FrameworkLoad` / `NoDevice` / `NotInitialized`).

## Fallback behaviour

Two distinct failure paths fall back automatically to the pure-Rust codec:

1. **Load failure** — driver not installed (no NVIDIA hardware, AMD-only system, container without `--gpus all`), `nvidia-uvm` kernel module not loaded, or `libcuda.so.1` ABI mismatch. `register()` logs and returns without registering.
2. **Init failure** — `cuInit` / `cuCtxCreate` / `cuvidCreateDecoder` / encoder creation return non-zero, the requested codec/profile exceeds the SM-class capability matrix, or the encoder slot cap is reached. The factory returns `Err`; the registry falls back to the next-priority impl.

Pipelines that **require** hardware can opt out of the SW fallback by setting `CodecPreferences { require_hardware: true, .. }`.

## Platform gating

The whole crate is `#![cfg(target_os = "linux")]`. On macOS / Windows it compiles to an empty rlib; the umbrella `oxideav` crate gates the `register` call behind the same cfg. NVDEC / NVENC are also available on Windows, but Windows support is a future cfg axis not yet in scope.

## Priority & opt-out

Hardware factories register with `CodecCapabilities::with_priority(5)` — slightly higher (better) than VA-API's 10, because on machines with both an iGPU and an NVIDIA GPU the NVIDIA path generally has higher absolute throughput and codec coverage. `--no-hwaccel` on the `oxideav` CLI biases dispatch away from HW factories without unregistering them.

## Workspace policy

Calling a system OS / driver API via FFI is the same shape as calling `libc::malloc` — it's the platform, not a copied algorithm. The workspace's clean-room rule does not apply to this crate.

## License

MIT.
