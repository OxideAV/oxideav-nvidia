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

/// NVDEC bitstream parser handle.
pub type CUvideoparser = *mut c_void;

/// CUvideotimestamp = `long long` per `<nvcuvid.h>`.
pub type CUvideotimestamp = i64;

// ─────────────────────────── packet flags / chroma constants ─────────────────

pub const CUVID_PKT_ENDOFSTREAM: u32 = 0x01;
pub const CUVID_PKT_TIMESTAMP: u32 = 0x02;
pub const CUVID_PKT_DISCONTINUITY: u32 = 0x04;
pub const CUVID_PKT_ENDOFPICTURE: u32 = 0x08;

/// `cudaVideoCreateFlags_PreferCUVID` — dedicated NVDEC engine path.
pub const CUDA_VIDEO_CREATE_PREFER_CUVID: u32 = 0x04;

/// `cudaVideoSurfaceFormat_NV12` — Y plane + interleaved UV.
pub const CUDA_VIDEO_SURFACE_FORMAT_NV12: i32 = 0;

/// `cudaVideoDeinterlaceMode_Weave` — pass-through (progressive).
pub const CUDA_VIDEO_DEINTERLACE_WEAVE: i32 = 0;

// ─────────────────────────── CUVIDEOFORMAT ───────────────────────────────────

/// Layout of `CUVIDEOFORMAT` from `<nvcuvid.h>`.
///
/// Total size 64 bytes on x86_64 Linux. The struct mixes packed
/// `unsigned char` flag bytes with 32-bit ints so we list the exact
/// layout as commented offsets.
///
/// Offsets (verified against the vendor header):
/// - 0:  codec               (i32)
/// - 4:  frame_rate.numerator   (u32)
/// - 8:  frame_rate.denominator (u32)
/// - 12: progressive_sequence  (u8)
/// - 13: bit_depth_luma_minus8 (u8)
/// - 14: bit_depth_chroma_minus8 (u8)
/// - 15: min_num_decode_surfaces (u8)
/// - 16: coded_width  (u32)
/// - 20: coded_height (u32)
/// - 24: display_area.left   (i32)
/// - 28: display_area.top    (i32)
/// - 32: display_area.right  (i32)
/// - 36: display_area.bottom (i32)
/// - 40: chroma_format (i32)
/// - 44: bitrate (u32)
/// - 48: display_aspect_ratio.x (i32)
/// - 52: display_aspect_ratio.y (i32)
/// - 56: video_signal_description (4 bytes packed)
/// - 60: seqhdr_data_length (u32)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUVIDEOFORMAT {
    pub codec: i32,
    pub frame_rate_numerator: u32,
    pub frame_rate_denominator: u32,
    pub progressive_sequence: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub min_num_decode_surfaces: u8,
    pub coded_width: u32,
    pub coded_height: u32,
    pub display_left: i32,
    pub display_top: i32,
    pub display_right: i32,
    pub display_bottom: i32,
    pub chroma_format: i32,
    pub bitrate: u32,
    pub display_aspect_x: i32,
    pub display_aspect_y: i32,
    pub video_signal_description: [u8; 4],
    pub seqhdr_data_length: u32,
}

// ─────────────────────────── CUVIDDECODECREATEINFO ───────────────────────────

/// Layout of `CUVIDDECODECREATEINFO` from `<cuviddec.h>`.
///
/// Total size 176 bytes on x86_64 Linux (`tcu_ulong = unsigned long =
/// 8 bytes`). The trailing reserved area pads out to that size.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUVIDDECODECREATEINFO {
    pub ul_width: u64,
    pub ul_height: u64,
    pub ul_num_decode_surfaces: u64,
    pub codec_type: i32,
    pub chroma_format: i32,
    pub ul_creation_flags: u64,
    pub bit_depth_minus_8: u64,
    pub ul_intra_decode_only: u64,
    pub ul_max_width: u64,
    pub ul_max_height: u64,
    pub reserved1: u64,
    pub display_left: i16,
    pub display_top: i16,
    pub display_right: i16,
    pub display_bottom: i16,
    pub output_format: i32,
    pub deinterlace_mode: i32,
    pub ul_target_width: u64,
    pub ul_target_height: u64,
    pub ul_num_output_surfaces: u64,
    pub vid_lock: CUvideoctxlock,
    pub target_left: i16,
    pub target_top: i16,
    pub target_right: i16,
    pub target_bottom: i16,
    pub enable_histogram: u64,
    pub reserved2: [u64; 4],
}

impl Default for CUVIDDECODECREATEINFO {
    fn default() -> Self {
        // SAFETY: the struct is plain-data; zeroing is a valid initial state.
        unsafe { std::mem::zeroed() }
    }
}

// ─────────────────────────── CUVIDPICPARAMS ──────────────────────────────────

/// Size of `CUVIDPICPARAMS` (verified via the vendor header). The
/// struct is huge and codec-specific; we treat it as an opaque blob —
/// the parser fills it in via the decode callback and we hand the same
/// pointer straight to `cuvidDecodePicture`. We never construct one
/// from Rust.
pub const CUVIDPICPARAMS_SIZE: usize = 4280;

/// Opaque alias for `CUVIDPICPARAMS`.
pub type CUVIDPICPARAMS = [u8; CUVIDPICPARAMS_SIZE];

// ─────────────────────────── CUVIDPROCPARAMS ─────────────────────────────────

/// Layout of `CUVIDPROCPARAMS` from `<cuviddec.h>`.
///
/// Used to map a decoded frame for display. Total 264 bytes.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUVIDPROCPARAMS {
    pub progressive_frame: i32,
    pub second_field: i32,
    pub top_field_first: i32,
    pub unpaired_field: i32,
    pub reserved_flags: u32,
    pub reserved_zero: u32,
    pub raw_input_dptr: u64,
    pub raw_input_pitch: u32,
    pub raw_input_format: u32,
    pub raw_output_dptr: u64,
    pub raw_output_pitch: u32,
    pub reserved1: u32,
    pub output_stream: CUstream,
    pub reserved: [u32; 46],
    pub histogram_dptr: *mut u64,
    pub reserved2: [*mut c_void; 1],
}

impl Default for CUVIDPROCPARAMS {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ─────────────────────────── CUVIDPARSERDISPINFO ─────────────────────────────

/// `CUVIDPARSERDISPINFO` — passed to the display callback.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUVIDPARSERDISPINFO {
    pub picture_index: i32,
    pub progressive_frame: i32,
    pub top_field_first: i32,
    pub repeat_first_field: i32,
    pub timestamp: CUvideotimestamp,
}

// ─────────────────────────── CUVIDSOURCEDATAPACKET ───────────────────────────

/// `CUVIDSOURCEDATAPACKET` — the payload handed to `cuvidParseVideoData`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUVIDSOURCEDATAPACKET {
    pub flags: u64,
    pub payload_size: u64,
    pub payload: *const u8,
    pub timestamp: CUvideotimestamp,
}

impl Default for CUVIDSOURCEDATAPACKET {
    fn default() -> Self {
        Self {
            flags: 0,
            payload_size: 0,
            payload: std::ptr::null(),
            timestamp: 0,
        }
    }
}

// ─────────────────────────── parser callback typedefs ────────────────────────

pub type PfnVidSequenceCallback =
    unsafe extern "C" fn(user_data: *mut c_void, fmt: *mut CUVIDEOFORMAT) -> i32;

pub type PfnVidDecodeCallback =
    unsafe extern "C" fn(user_data: *mut c_void, pic: *mut CUVIDPICPARAMS) -> i32;

pub type PfnVidDisplayCallback =
    unsafe extern "C" fn(user_data: *mut c_void, disp: *mut CUVIDPARSERDISPINFO) -> i32;

pub type PfnVidOpPointCallback = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

pub type PfnVidSeiMsgCallback = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

// ─────────────────────────── CUVIDPARSERPARAMS ───────────────────────────────

/// `CUVIDPARSERPARAMS` from `<nvcuvid.h>`. Total 136 bytes.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUVIDPARSERPARAMS {
    pub codec_type: i32,
    pub ul_max_num_decode_surfaces: u32,
    pub ul_clock_rate: u32,
    pub ul_error_threshold: u32,
    pub ul_max_display_delay: u32,
    /// `bAnnexb : 1` + `uReserved : 31` packed into one u32.
    pub flags: u32,
    pub u_reserved1: [u32; 4],
    pub user_data: *mut c_void,
    pub pfn_sequence_callback: Option<PfnVidSequenceCallback>,
    pub pfn_decode_picture: Option<PfnVidDecodeCallback>,
    pub pfn_display_picture: Option<PfnVidDisplayCallback>,
    pub pfn_get_operating_point: Option<PfnVidOpPointCallback>,
    pub pfn_get_sei_msg: Option<PfnVidSeiMsgCallback>,
    pub pv_reserved2: [*mut c_void; 5],
    pub p_ext_video_info: *mut c_void,
}

impl Default for CUVIDPARSERPARAMS {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

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

pub type FnCuvidCreateVideoParser = unsafe extern "C" fn(
    parser_out: *mut CUvideoparser,
    params: *mut CUVIDPARSERPARAMS,
) -> CUresult;

pub type FnCuvidParseVideoData = unsafe extern "C" fn(
    parser: CUvideoparser,
    packet: *mut CUVIDSOURCEDATAPACKET,
) -> CUresult;

pub type FnCuvidDestroyVideoParser =
    unsafe extern "C" fn(parser: CUvideoparser) -> CUresult;

pub type FnCuMemcpyDtoHV2 =
    unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;

pub type FnCuStreamSynchronize = unsafe extern "C" fn(stream: CUstream) -> CUresult;

// libnvidia-encode (NVENC)
//
// Only the single bootstrap entry. `NvEncodeAPICreateInstance` takes a
// `NV_ENCODE_API_FUNCTION_LIST*` whose `version` field is set by the
// caller; on success the rest of the function table is populated.
// Round 2 will model the function-list struct and call this.
pub type FnNvEncodeApiCreateInstance =
    unsafe extern "C" fn(function_list: *mut c_void) -> i32;

// ─────────────────────────── compile-time size guards ───────────────────────

// Layout sanity — these are the sizes a C compiler emits for the
// vendor headers on x86_64 Linux. If the host C ABI deviates we'd
// silently send the driver a struct of the wrong size; pinning the
// sizes here turns that into a compile error instead.
const _: () = {
    if std::mem::size_of::<CUVIDEOFORMAT>() != 64 {
        panic!("CUVIDEOFORMAT layout drift");
    }
    if std::mem::size_of::<CUVIDDECODECREATEINFO>() != 176 {
        panic!("CUVIDDECODECREATEINFO layout drift");
    }
    if std::mem::size_of::<CUVIDPARSERPARAMS>() != 136 {
        panic!("CUVIDPARSERPARAMS layout drift");
    }
    if std::mem::size_of::<CUVIDPROCPARAMS>() != 264 {
        panic!("CUVIDPROCPARAMS layout drift");
    }
    if std::mem::size_of::<CUVIDPARSERDISPINFO>() != 24 {
        panic!("CUVIDPARSERDISPINFO layout drift");
    }
    if std::mem::size_of::<CUVIDSOURCEDATAPACKET>() != 32 {
        panic!("CUVIDSOURCEDATAPACKET layout drift");
    }
    if std::mem::size_of::<CUVIDDECODECAPS>() != 88 {
        panic!("CUVIDDECODECAPS layout drift");
    }
};

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
    pub cu_memcpy_dto_h_v2: FnCuMemcpyDtoHV2,
    pub cu_stream_synchronize: FnCuStreamSynchronize,
    // libnvcuvid (NVDEC)
    pub cuvid_create_decoder: FnCuvidCreateDecoder,
    pub cuvid_destroy_decoder: FnCuvidDestroyDecoder,
    pub cuvid_decode_picture: FnCuvidDecodePicture,
    pub cuvid_map_video_frame_64: FnCuvidMapVideoFrame64,
    pub cuvid_unmap_video_frame_64: FnCuvidUnmapVideoFrame64,
    pub cuvid_get_decoder_caps: FnCuvidGetDecoderCaps,
    pub cuvid_create_video_parser: FnCuvidCreateVideoParser,
    pub cuvid_parse_video_data: FnCuvidParseVideoData,
    pub cuvid_destroy_video_parser: FnCuvidDestroyVideoParser,
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
        cu_memcpy_dto_h_v2: sym!(libcuda, "cuMemcpyDtoH_v2", FnCuMemcpyDtoHV2),
        cu_stream_synchronize: sym!(libcuda, "cuStreamSynchronize", FnCuStreamSynchronize),
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
        cuvid_create_video_parser: sym!(
            libnvcuvid,
            "cuvidCreateVideoParser",
            FnCuvidCreateVideoParser
        ),
        cuvid_parse_video_data: sym!(
            libnvcuvid,
            "cuvidParseVideoData",
            FnCuvidParseVideoData
        ),
        cuvid_destroy_video_parser: sym!(
            libnvcuvid,
            "cuvidDestroyVideoParser",
            FnCuvidDestroyVideoParser
        ),
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
