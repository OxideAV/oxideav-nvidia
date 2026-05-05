//! Runtime-loaded NVIDIA library handles.
//!
//! Loaded once via `OnceLock` on first use and cached for the process
//! lifetime. If any dlopen fails the cache stores the error so
//! subsequent calls don't repeatedly hammer the dynamic linker.
//!
//! Libraries needed for the NVIDIA bridge:
//!
//! | Library                  | Purpose                                          |
//! |--------------------------|--------------------------------------------------|
//! | libcuda.so.1             | CUDA driver API (`cuInit`, `cuCtxCreate`, …)     |
//! | libnvcuvid.so.1          | NVDEC video decode (`cuvidCreateDecoder`, …)     |
//! | libnvidia-encode.so.1    | NVENC video encode (`NvEncodeAPICreateInstance`) |
//!
//! All three are opened separately because they are distributed by
//! the NVIDIA driver as independent `.so` files; one being absent
//! (e.g. older driver without NVENC support, datacenter SKU without
//! NVENC) shouldn't bring the whole bridge down — but Round 1 is
//! conservative and treats any missing library as "framework not
//! available". Round 2 may relax this to per-engine availability.

use libloading::Library;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::OnceLock;

// ─────────────────────────── opaque CUDA types ───────────────────────────────

/// CUDA context handle. Treated opaquely; we only pass the pointer
/// around.
pub type CUcontext = *mut c_void;

/// CUDA stream handle.
pub type CUstream = *mut c_void;

/// CUDA device pointer. Driver API uses an integer-typed device
/// pointer (`unsigned long long` on 64-bit, `unsigned int` on 32-bit).
/// We use `u64` since this crate is Linux x86_64 / arm64 only.
pub type CUdeviceptr = u64;

/// CUresult — return code for almost every CUDA driver API entry.
pub type CUresult = i32;

/// Success status: `CUDA_SUCCESS == 0`.
pub const CUDA_SUCCESS: CUresult = 0;

/// CUdevice — 32-bit ordinal returned by `cuDeviceGet`.
pub type CUdevice = i32;

/// `CUdevice_attribute` value for "compute capability major version".
/// Vendor-supplied constant from `<cuda.h>`.
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
/// `CUdevice_attribute` value for "compute capability minor version".
/// Vendor-supplied constant from `<cuda.h>`.
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// ─────────────────────────── cudaVideoCodec / cudaVideoChromaFormat ──────────

/// `cudaVideoCodec` enum from the NVIDIA Video Codec SDK
/// (`<cuviddec.h>`). Numeric values are part of the public ABI.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CudaVideoCodec {
    Mpeg1 = 0,
    Mpeg2 = 1,
    Mpeg4 = 2,
    Vc1 = 3,
    H264 = 4,
    Jpeg = 5,
    H264Svc = 6,
    H264Mvc = 7,
    Hevc = 8,
    Vp8 = 9,
    Vp9 = 10,
    Av1 = 11,
}

/// `cudaVideoChromaFormat` enum from `<cuviddec.h>`.
pub const CUDA_VIDEO_CHROMA_FORMAT_MONOCHROME: u32 = 0;
pub const CUDA_VIDEO_CHROMA_FORMAT_420: u32 = 1;
pub const CUDA_VIDEO_CHROMA_FORMAT_422: u32 = 2;
pub const CUDA_VIDEO_CHROMA_FORMAT_444: u32 = 3;

/// Public layout of `CUVIDDECODECAPS` from `<cuviddec.h>`.
///
/// Layout (bytes 0..76, total 80 with the trailing 10×u32 reserved):
/// - eCodecType:          i32 (4)
/// - eChromaFormat:       i32 (4)
/// - nBitDepthMinus8:     u32 (4)
/// - reserved1[3]:        3×u32 (12)
/// - bIsSupported:        u8 (1)
/// - nNumNVDECs:          u8 (1)
/// - nOutputFormatMask:   u16 (2)
/// - nMaxWidth:           u32 (4)
/// - nMaxHeight:          u32 (4)
/// - nMaxMBCount:         u32 (4)
/// - nMinWidth:           u16 (2)
/// - nMinHeight:          u16 (2)
/// - bIsHistogramSupported: u8 (1)
/// - nCounterBitDepth:    u8 (1)
/// - nMaxHistogramBins:   u16 (2)
/// - reserved3[10]:       10×u32 (40)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUVIDDECODECAPS {
    pub e_codec_type: i32,
    pub e_chroma_format: i32,
    pub n_bit_depth_minus_8: u32,
    pub reserved1: [u32; 3],

    pub b_is_supported: u8,
    pub n_num_nvdecs: u8,
    pub n_output_format_mask: u16,
    pub n_max_width: u32,
    pub n_max_height: u32,
    pub n_max_mb_count: u32,
    pub n_min_width: u16,
    pub n_min_height: u16,
    pub b_is_histogram_supported: u8,
    pub n_counter_bit_depth: u8,
    pub n_max_histogram_bins: u16,
    pub reserved3: [u32; 10],
}

// ─────────────────────────── opaque NVDEC types ──────────────────────────────

/// NVDEC video decoder handle.
pub type CUvideodecoder = *mut c_void;

/// NVDEC video context lock.
pub type CUvideoctxlock = *mut c_void;

// ─────────────────────────── function pointer types ──────────────────────────

// libcuda
pub type FnCuInit = unsafe extern "C" fn(flags: u32) -> CUresult;

pub type FnCuDeviceGet =
    unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult;

pub type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> CUresult;

pub type FnCuCtxCreateV2 = unsafe extern "C" fn(
    pctx: *mut CUcontext,
    flags: u32,
    dev: CUdevice,
) -> CUresult;

pub type FnCuCtxDestroyV2 = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;

pub type FnCuMemAllocV2 =
    unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;

pub type FnCuMemFreeV2 = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;

pub type FnCuGetErrorString =
    unsafe extern "C" fn(error: CUresult, str_out: *mut *const c_char) -> CUresult;

pub type FnCuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult;

pub type FnCuDriverGetVersion = unsafe extern "C" fn(version_out: *mut i32) -> CUresult;

pub type FnCuDeviceTotalMemV2 =
    unsafe extern "C" fn(bytes_out: *mut usize, dev: CUdevice) -> CUresult;

pub type FnCuDeviceGetAttribute =
    unsafe extern "C" fn(value_out: *mut i32, attrib: i32, dev: CUdevice) -> CUresult;

pub type FnCuCtxPushCurrentV2 = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
pub type FnCuCtxPopCurrentV2 = unsafe extern "C" fn(ctx_out: *mut CUcontext) -> CUresult;

// libnvcuvid (NVDEC)
//
// The full NVDEC structures (`CUVIDDECODECREATEINFO`, `CUVIDPICPARAMS`,
// `CUVIDPROCPARAMS`) are large and codec-specific; Round 2 will model
// them. For Round 1 the function pointer types use opaque pointers so
// the dlsym + transmute chain is verified without locking us into a
// particular struct layout.
pub type FnCuvidCreateDecoder = unsafe extern "C" fn(
    decoder: *mut CUvideodecoder,
    create_info: *mut c_void,
) -> CUresult;

pub type FnCuvidDestroyDecoder = unsafe extern "C" fn(decoder: CUvideodecoder) -> CUresult;

pub type FnCuvidDecodePicture =
    unsafe extern "C" fn(decoder: CUvideodecoder, pic_params: *mut c_void) -> CUresult;

pub type FnCuvidMapVideoFrame64 = unsafe extern "C" fn(
    decoder: CUvideodecoder,
    pic_idx: i32,
    dev_ptr: *mut u64,
    pitch: *mut u32,
    proc_params: *mut c_void,
) -> CUresult;

pub type FnCuvidUnmapVideoFrame64 =
    unsafe extern "C" fn(decoder: CUvideodecoder, dev_ptr: u64) -> CUresult;

pub type FnCuvidGetDecoderCaps =
    unsafe extern "C" fn(decoder_caps: *mut CUVIDDECODECAPS) -> CUresult;

// libnvidia-encode (NVENC)
//
// Only the single bootstrap entry. `NvEncodeAPICreateInstance` takes a
// `NV_ENCODE_API_FUNCTION_LIST*` whose `version` field is set by the
// caller; on success the rest of the function table is populated.
// Round 2 will model the function-list struct and call this.
pub type FnNvEncodeApiCreateInstance =
    unsafe extern "C" fn(function_list: *mut c_void) -> i32;

// ─────────────────────────── Vtable ───────────────────────────────────────────

/// Resolved function pointers for the bootstrap NVIDIA symbol set.
///
/// All fields are `unsafe extern "C" fn(...)` pointer types — callers
/// are responsible for the FFI invariants.
pub struct Vtable {
    // libcuda
    pub cu_init: FnCuInit,
    pub cu_device_get: FnCuDeviceGet,
    pub cu_device_get_count: FnCuDeviceGetCount,
    pub cu_ctx_create_v2: FnCuCtxCreateV2,
    pub cu_ctx_destroy_v2: FnCuCtxDestroyV2,
    pub cu_mem_alloc_v2: FnCuMemAllocV2,
    pub cu_mem_free_v2: FnCuMemFreeV2,
    pub cu_get_error_string: FnCuGetErrorString,
    pub cu_device_get_name: FnCuDeviceGetName,
    pub cu_driver_get_version: FnCuDriverGetVersion,
    pub cu_device_total_mem_v2: FnCuDeviceTotalMemV2,
    pub cu_device_get_attribute: FnCuDeviceGetAttribute,
    pub cu_ctx_push_current_v2: FnCuCtxPushCurrentV2,
    pub cu_ctx_pop_current_v2: FnCuCtxPopCurrentV2,
    // libnvcuvid (NVDEC)
    pub cuvid_create_decoder: FnCuvidCreateDecoder,
    pub cuvid_destroy_decoder: FnCuvidDestroyDecoder,
    pub cuvid_decode_picture: FnCuvidDecodePicture,
    pub cuvid_map_video_frame_64: FnCuvidMapVideoFrame64,
    pub cuvid_unmap_video_frame_64: FnCuvidUnmapVideoFrame64,
    pub cuvid_get_decoder_caps: FnCuvidGetDecoderCaps,
    // libnvidia-encode (NVENC)
    pub nv_encode_api_create_instance: FnNvEncodeApiCreateInstance,
    // Keep libraries alive
    _libcuda: Library,
    _libnvcuvid: Library,
    _libnvenc: Library,
}

/// Smoke-test wrapper used by tests + by the pre-flight load check
/// in `register()`. Holds the raw `Library` handles so callers can
/// assert that dlopen succeeded without paying the full dlsym tour.
pub struct FrameworkSmoke {
    pub libcuda: Library,
    pub libnvcuvid: Library,
    pub libnvenc: Library,
}

// ─────────────────────────── Caches ───────────────────────────────────────────

static VTABLE: OnceLock<Result<Vtable, String>> = OnceLock::new();
static FRAMEWORK: OnceLock<Result<FrameworkSmoke, String>> = OnceLock::new();

/// Get (or load) the fully-resolved vtable. Returns the cached `Err`
/// if a previous load attempt failed.
pub fn vtable() -> Result<&'static Vtable, &'static str> {
    VTABLE
        .get_or_init(load_vtable)
        .as_ref()
        .map_err(|s| s.as_str())
}

/// Cheap framework-load check used by `register()`. Resolves the
/// three libraries but does no dlsym work.
pub fn framework() -> Result<&'static FrameworkSmoke, &'static str> {
    FRAMEWORK
        .get_or_init(load_smoke)
        .as_ref()
        .map_err(|s| s.as_str())
}

fn load_smoke() -> Result<FrameworkSmoke, String> {
    Ok(FrameworkSmoke {
        libcuda: open("libcuda.so.1")?,
        libnvcuvid: open("libnvcuvid.so.1")?,
        libnvenc: open("libnvidia-encode.so.1")?,
    })
}

fn load_vtable() -> Result<Vtable, String> {
    let libcuda = open("libcuda.so.1")?;
    let libnvcuvid = open("libnvcuvid.so.1")?;
    let libnvenc = open("libnvidia-encode.so.1")?;

    macro_rules! sym {
        ($lib:expr, $name:expr, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = unsafe {
                $lib.get(concat!($name, "\0").as_bytes())
                    .map_err(|e| format!("dlsym {}: {}", $name, e))?
            };
            *s
        }};
    }

    Ok(Vtable {
        cu_init: sym!(libcuda, "cuInit", FnCuInit),
        cu_device_get: sym!(libcuda, "cuDeviceGet", FnCuDeviceGet),
        cu_device_get_count: sym!(libcuda, "cuDeviceGetCount", FnCuDeviceGetCount),
        cu_ctx_create_v2: sym!(libcuda, "cuCtxCreate_v2", FnCuCtxCreateV2),
        cu_ctx_destroy_v2: sym!(libcuda, "cuCtxDestroy_v2", FnCuCtxDestroyV2),
        cu_mem_alloc_v2: sym!(libcuda, "cuMemAlloc_v2", FnCuMemAllocV2),
        cu_mem_free_v2: sym!(libcuda, "cuMemFree_v2", FnCuMemFreeV2),
        cu_get_error_string: sym!(libcuda, "cuGetErrorString", FnCuGetErrorString),
        cu_device_get_name: sym!(libcuda, "cuDeviceGetName", FnCuDeviceGetName),
        cu_driver_get_version: sym!(libcuda, "cuDriverGetVersion", FnCuDriverGetVersion),
        cu_device_total_mem_v2: sym!(libcuda, "cuDeviceTotalMem_v2", FnCuDeviceTotalMemV2),
        cu_device_get_attribute: sym!(libcuda, "cuDeviceGetAttribute", FnCuDeviceGetAttribute),
        cu_ctx_push_current_v2: sym!(libcuda, "cuCtxPushCurrent_v2", FnCuCtxPushCurrentV2),
        cu_ctx_pop_current_v2: sym!(libcuda, "cuCtxPopCurrent_v2", FnCuCtxPopCurrentV2),
        cuvid_create_decoder: sym!(libnvcuvid, "cuvidCreateDecoder", FnCuvidCreateDecoder),
        cuvid_destroy_decoder: sym!(
            libnvcuvid,
            "cuvidDestroyDecoder",
            FnCuvidDestroyDecoder
        ),
        cuvid_decode_picture: sym!(libnvcuvid, "cuvidDecodePicture", FnCuvidDecodePicture),
        cuvid_map_video_frame_64: sym!(
            libnvcuvid,
            "cuvidMapVideoFrame64",
            FnCuvidMapVideoFrame64
        ),
        cuvid_unmap_video_frame_64: sym!(
            libnvcuvid,
            "cuvidUnmapVideoFrame64",
            FnCuvidUnmapVideoFrame64
        ),
        cuvid_get_decoder_caps: sym!(libnvcuvid, "cuvidGetDecoderCaps", FnCuvidGetDecoderCaps),
        nv_encode_api_create_instance: sym!(
            libnvenc,
            "NvEncodeAPICreateInstance",
            FnNvEncodeApiCreateInstance
        ),
        _libcuda: libcuda,
        _libnvcuvid: libnvcuvid,
        _libnvenc: libnvenc,
    })
}

fn open(path: &str) -> Result<Library, String> {
    // SAFETY: dlopen on a soname with no init callbacks; equivalent to
    // a normal program startup load.
    unsafe { Library::new(path) }.map_err(|e| format!("dlopen {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: all three libraries on this machine load cleanly.
    #[test]
    fn frameworks_load() {
        let fw = framework().expect("framework load");
        let _: libloading::Symbol<unsafe extern "C" fn()> =
            unsafe { fw.libcuda.get(b"cuInit\0").expect("cuInit symbol") };
        let _: libloading::Symbol<unsafe extern "C" fn()> = unsafe {
            fw.libnvcuvid
                .get(b"cuvidCreateDecoder\0")
                .expect("cuvidCreateDecoder symbol")
        };
        let _: libloading::Symbol<unsafe extern "C" fn()> = unsafe {
            fw.libnvenc
                .get(b"NvEncodeAPICreateInstance\0")
                .expect("NvEncodeAPICreateInstance symbol")
        };
    }

    /// Verify the vtable resolves all symbols.
    #[test]
    fn vtable_resolves() {
        vtable().expect("vtable load");
    }
}
