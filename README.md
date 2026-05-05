# oxideav-nvidia

Linux NVIDIA NVDEC + NVENC hardware decode/encode bridge for the [oxideav](https://github.com/OxideAV/oxideav) framework.

## Why a bridge crate?

NVIDIA's NVENC + NVDEC engines deliver the highest absolute throughput on Linux for H.264 / HEVC / AV1 encode + decode on NVIDIA hardware. The toolchain is **proprietary**: there is no Mesa equivalent, and bytes flow in and out of the GPU through the CUDA driver API. This crate is the bridge from oxideav's pipeline traits to that toolchain.

This crate is a **thin runtime-loaded bridge** — no compile-time link dependency on the CUDA SDK, no `*-sys` crate. The three libraries are opened via [`libloading`] on first use.

## Library set

| Library                  | Role                                                              |
|--------------------------|-------------------------------------------------------------------|
| `libcuda.so.1`           | CUDA driver API — context create, device-memory allocation        |
| `libnvcuvid.so.1`        | NVDEC video decode (H.264 / HEVC / VP9 / AV1)                     |
| `libnvidia-encode.so.1`  | NVENC video encode (H.264 / HEVC / AV1)                           |

NVENC's API surface is unusual: only `NvEncodeAPICreateInstance` is exported. Calling it returns a function-table struct (`NV_ENCODE_API_FUNCTION_LIST`) that holds every other NVENC entry. Round 1 resolves only the bootstrap symbol; Round 2 will populate that function table.

## Fallback behaviour

Two distinct failure paths fall back automatically to the pure-Rust codec:

1. **Load failure** — driver not installed (no NVIDIA hardware, AMD-only system, container without `--gpus all`), `nvidia-uvm` kernel module not loaded, or `libcuda.so.1` ABI mismatch with the running driver. `register()` logs and returns without registering.
2. **Init failure** — `cuInit` / `cuCtxCreate` / `cuvidCreateDecoder` / `NvEncoderCreateEncoder` return non-zero, the requested codec/profile exceeds the SM-class capability matrix, or the encoder slot cap is reached (consumer cards limit concurrent NVENC sessions). The factory returns `Err`; the registry falls back to the next-priority impl.

Pipelines that **require** hardware can opt out of the SW fallback by setting `CodecPreferences { require_hardware: true, .. }`.

## Platform gating

The whole crate is `#![cfg(target_os = "linux")]`. On macOS / Windows it compiles to an empty rlib; the umbrella `oxideav` crate gates the `register` call behind the same cfg.

NVDEC / NVENC are also available on Windows — Windows support is a future cfg axis but not in scope here.

## Priority

Hardware factories register with `CodecCapabilities::with_priority(5)` — slightly higher (better) than VA-API's 10, because on machines that have both an iGPU + an NVIDIA GPU the NVIDIA path generally has higher absolute throughput and better codec coverage.

## Opt-out

`--no-hwaccel` on the `oxideav` CLI bias dispatch away from HW factories without unregistering them.

## Coverage roadmap

| Codec        | Decode (NVDEC) | Encode (NVENC) |
|--------------|----------------|----------------|
| H.264        | **shipped (Round 3)** | planned        |
| HEVC         | planned        | planned        |
| AV1          | planned (Ada Lovelace+) | planned (Ada Lovelace+) |
| VP9          | planned        | —              |
| MPEG-2       | planned        | —              |
| MPEG-4 Pt 2  | planned        | —              |
| VC-1         | planned        | —              |
| JPEG         | planned (NVJPEG, separate lib) | — |

Round 3 (this commit): real H.264 decode via NVDEC + the
[cuvidParser](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/nvdec-video-decoder-api-prog-guide/index.html#video-parser) bitstream layer:

- New `H264NvDecoder` implementing `oxideav_core::Decoder`.
- `H264NvDecoder::make` initialises CUDA, picks device 0, creates a
  context, and stands up a `cuvidParser` configured for H.264 with
  three callbacks (sequence, decode, display).
- `send_packet` hands the Annex-B bytes to `cuvidParseVideoData`.
  The parser fires the sequence callback on the first SPS — at which
  point we build a `CUVIDDECODECREATEINFO` (NV12 output, weave
  deinterlace, target = coded resolution) and call
  `cuvidCreateDecoder`. Per-picture, the parser fills a 4280-byte
  `CUVIDPICPARAMS` blob and we forward it verbatim to
  `cuvidDecodePicture`; per-display, we `cuvidMapVideoFrame64`,
  `cuStreamSynchronize`, copy back via `cuMemcpyDtoH_v2`, deinterleave
  the NV12 chroma plane into separate U + V planes, and push an I420
  `VideoFrame`.
- `flush()` sends a `CUVID_PKT_ENDOFSTREAM` packet so the parser
  drains its display queue, after which `receive_frame` returns
  `Error::Eof`.
- `Drop` destroys the parser, then the decoder, then the CUDA
  context (in that order).
- `register()` now wires the H.264 factory into the codec registry
  with `with_priority(5)` and the standard
  H264 / AVC1 / X264 / `V_MPEG4/ISO/AVC` tag set.
- `tests/round3_decode.rs::nvdec_h264_idr_decodes_to_320x240_i420`
  decodes a 1-frame Annex-B IDR fixture (`testsrc2 -> libx264 baseline`)
  and asserts the output is 320×240 I420 with luma std-dev > 5
  (the colour-bar testsrc2 pattern produces stddev ≈ 56).
- All Round 2 tests continue to pass.

NVENC encode is still on the roadmap — Round 3 ships decode only.

Round 2 (previous commit): exposed safe wrappers around the CUDA
driver init + device enumeration, plus an NVDEC capability query.

- `Cuda::init()` runs `cuInit(0)` once and returns a handle.
- `Cuda::device_count()` + `Cuda::device(ordinal)` enumerate GPUs.
- `CudaDevice::{name, total_memory_bytes, compute_capability}` cover the basic device introspection surface.
- `Cuda::create_context_for(&device)` builds a `CudaContext` that pushes itself current on construction and `cuCtxDestroy_v2`s on Drop.
- `nvdec_caps(codec, chroma, bit_depth)` calls `cuvidGetDecoderCaps` and returns a public `NvdecCaps` struct (codec / chroma / bit depth + `is_supported` + `max_width` / `max_height` / `max_mb_count` etc.).

`tests/round2_init.rs` exercises the full path on real NVIDIA hardware: `cuInit` → `cuDeviceGet` → `cuDeviceGetName` / `cuDeviceTotalMem_v2` / `cuDeviceGetAttribute` → `cuCtxCreate_v2` → `cuvidGetDecoderCaps`. Each test detects "no driver / no GPU" and skips with `eprintln!` rather than panicking, so the suite passes on a Linux box with an NVIDIA GPU and skips cleanly on every other host.

## Workspace policy

Calling a system OS / driver API via FFI is the same shape as calling `libc::malloc` — it's the platform, not a copied algorithm. The workspace's clean-room rule (no embedding source from libvpx, libwebp, libjxl, etc.) does not apply to this crate.

## License

MIT.
