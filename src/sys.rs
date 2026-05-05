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
// Only the single bootstrap entry: `NvEncodeAPICreateInstance` takes a
// `NV_ENCODE_API_FUNCTION_LIST*` whose `version` field is set by the
// caller; on success the rest of the function table is populated. The
// rest of the NVENC entry points are reached via that table —
// `NvEncFunctions` below — so the dynamic-linker stays free of
// per-driver-version symbol drift.
pub type FnNvEncodeApiCreateInstance =
    unsafe extern "C" fn(function_list: *mut c_void) -> i32;

// ─────────────────────────── NVENC types ─────────────────────────────────────

/// NVENCSTATUS — return code for every NVENC function. `0` is success.
pub type NvEncStatus = i32;
pub const NV_ENC_SUCCESS: NvEncStatus = 0;

/// `NV_ENC_DEVICE_TYPE_CUDA` from `<nvEncodeAPI.h>`.
pub const NV_ENC_DEVICE_TYPE_CUDA: u32 = 0x1;

/// 16-byte GUID, matching the platform `GUID` typedef in `<nvEncodeAPI.h>`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl Guid {
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

// ─── codec / preset / profile GUIDs (literal from <nvEncodeAPI.h>) ──────────

/// `NV_ENC_CODEC_H264_GUID = {6BC82762-4E63-4ca4-AA85-1E50F321F6BF}`.
pub const NV_ENC_CODEC_H264_GUID: Guid = Guid::new(
    0x6bc82762,
    0x4e63,
    0x4ca4,
    [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
);

/// `NV_ENC_CODEC_HEVC_GUID = {790CDC88-4522-4d7b-9425-BDA9975F7603}`.
pub const NV_ENC_CODEC_HEVC_GUID: Guid = Guid::new(
    0x790cdc88,
    0x4522,
    0x4d7b,
    [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
);

/// `NV_ENC_CODEC_AV1_GUID = {0A352289-0AA7-4759-862D-5D15CD16D254}`.
pub const NV_ENC_CODEC_AV1_GUID: Guid = Guid::new(
    0x0a352289,
    0x0aa7,
    0x4759,
    [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
);

/// `NV_ENC_PRESET_P4_GUID = {90A7B826-DF06-4862-B9D2-CD6D73A08681}` —
/// medium quality / speed.
pub const NV_ENC_PRESET_P4_GUID: Guid = Guid::new(
    0x90a7b826,
    0xdf06,
    0x4862,
    [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
);

/// `NV_ENC_H264_PROFILE_HIGH_GUID = {E7CBC309-4F7A-4b89-AF2A-D537C92BE310}`.
pub const NV_ENC_H264_PROFILE_HIGH_GUID: Guid = Guid::new(
    0xe7cbc309,
    0x4f7a,
    0x4b89,
    [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
);

/// `NV_ENC_HEVC_PROFILE_MAIN_GUID = {B514C39A-B55B-40fa-878F-F1253B4DFDEC}`.
pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: Guid = Guid::new(
    0xb514c39a,
    0xb55b,
    0x40fa,
    [0x87, 0x8f, 0xf1, 0x25, 0x3b, 0x4d, 0xfd, 0xec],
);

// ─── NVENC version macros ───────────────────────────────────────────────────

/// `NVENCAPI_MAJOR_VERSION` — fixed at 12 in driver 580.x's header.
pub const NVENCAPI_MAJOR_VERSION: u32 = 12;
/// `NVENCAPI_MINOR_VERSION` — fixed at 1 in driver 580.x's header.
pub const NVENCAPI_MINOR_VERSION: u32 = 1;

/// `NVENCAPI_VERSION = MAJOR | (MINOR << 24) = 0x0100_000C`.
pub const NVENCAPI_VERSION: u32 =
    NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);

/// `NVENCAPI_STRUCT_VERSION(ver) = NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)`.
pub const fn nvenc_struct_ver(v: u32) -> u32 {
    NVENCAPI_VERSION | (v << 16) | (0x7u32 << 28)
}

/// Convenience: high-bit flag (`1u<<31`) set on the structs whose
/// version macros use the `| (1u<<31)` form.
const NV_ENC_VER_HIGH: u32 = 1u32 << 31;

pub const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = nvenc_struct_ver(1);
pub const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = nvenc_struct_ver(2);
pub const NV_ENC_INITIALIZE_PARAMS_VER: u32 = nvenc_struct_ver(6) | NV_ENC_VER_HIGH;
pub const NV_ENC_CONFIG_VER: u32 = nvenc_struct_ver(8) | NV_ENC_VER_HIGH;
pub const NV_ENC_PRESET_CONFIG_VER: u32 = nvenc_struct_ver(4) | NV_ENC_VER_HIGH;
pub const NV_ENC_CREATE_INPUT_BUFFER_VER: u32 = nvenc_struct_ver(1);
pub const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = nvenc_struct_ver(1);
pub const NV_ENC_LOCK_INPUT_BUFFER_VER: u32 = nvenc_struct_ver(1);
pub const NV_ENC_LOCK_BITSTREAM_VER: u32 = nvenc_struct_ver(1) | NV_ENC_VER_HIGH;
pub const NV_ENC_PIC_PARAMS_VER: u32 = nvenc_struct_ver(6) | NV_ENC_VER_HIGH;

/// `NV_ENC_BUFFER_FORMAT_NV12 = 0x1`.
pub const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x1;

/// `NV_ENC_PIC_STRUCT_FRAME = 0x1`.
pub const NV_ENC_PIC_STRUCT_FRAME: u32 = 0x1;

/// `NV_ENC_TUNING_INFO_HIGH_QUALITY = 1`.
pub const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;
/// `NV_ENC_TUNING_INFO_LOW_LATENCY = 2`.
pub const NV_ENC_TUNING_INFO_LOW_LATENCY: u32 = 2;

/// `NV_ENC_PIC_FLAGS_FORCEIDR = 1`.
pub const NV_ENC_PIC_FLAGS_FORCEIDR: u32 = 0x1;
/// `NV_ENC_PIC_FLAGS_EOS = 8` — the last-frame-of-stream marker.
pub const NV_ENC_PIC_FLAGS_EOS: u32 = 0x8;

// ─── NVENC structs we touch (verified vs `<nvEncodeAPI.h>` sizes) ───────────

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS` — passed to
/// `nvEncOpenEncodeSessionEx`.
///
/// Layout (`sizeof = 1552`):
/// - 0:    version    (u32)
/// - 4:    deviceType (u32, NV_ENC_DEVICE_TYPE)
/// - 8:    device     (*mut c_void) — CUDA context handle
/// - 16:   reserved   (*mut c_void)
/// - 24:   apiVersion (u32) + 4-byte alignment hole
/// - 32:   reserved1[253] (1012 bytes)
/// - 1044: reserved2[64]  (512 bytes)
#[repr(C)]
pub struct NvEncOpenEncodeSessionExParams {
    pub version: u32,
    pub device_type: u32,
    pub device: *mut c_void,
    pub reserved: *mut c_void,
    pub api_version: u32,
    pub reserved1: [u32; 253],
    pub reserved2: [*mut c_void; 64],
}

impl Default for NvEncOpenEncodeSessionExParams {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_CREATE_INPUT_BUFFER` — sizeof = 776.
#[repr(C)]
pub struct NvEncCreateInputBuffer {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub memory_heap: u32, // deprecated
    pub buffer_fmt: u32,
    pub reserved: u32,
    pub input_buffer: *mut c_void, // out
    pub p_sys_mem_buffer: *mut c_void,
    pub reserved1: [u32; 57],
    pub reserved2: [*mut c_void; 63],
}

impl Default for NvEncCreateInputBuffer {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_CREATE_BITSTREAM_BUFFER` — sizeof = 776.
#[repr(C)]
pub struct NvEncCreateBitstreamBuffer {
    pub version: u32,
    pub size: u32,                     // deprecated
    pub memory_heap: u32,              // deprecated
    pub reserved: u32,
    pub bitstream_buffer: *mut c_void, // out
    pub bitstream_buffer_ptr: *mut c_void, // reserved
    pub reserved1: [u32; 58],
    pub reserved2: [*mut c_void; 64],
}

impl Default for NvEncCreateBitstreamBuffer {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_LOCK_INPUT_BUFFER` — sizeof = 1544.
///
/// Field offsets per `<nvEncodeAPI.h>`:
/// -   0: version (u32)
/// -   4: do_not_wait : 1, reservedBitFields : 31 (u32)
/// -   8: inputBuffer (*mut c_void)
/// -  16: bufferDataPtr (*mut c_void)
/// -  24: pitch (u32)
/// -  28: reserved1[251] (1004 bytes) → ends at 1032
/// - 1032: reserved2[64]  (512 bytes) → ends at 1544
#[repr(C)]
pub struct NvEncLockInputBuffer {
    pub version: u32,
    pub flags: u32,
    pub input_buffer: *mut c_void,
    pub buffer_data_ptr: *mut c_void, // out
    pub pitch: u32,                   // out
    pub reserved1: [u32; 251],
    pub reserved2: [*mut c_void; 64],
}

impl Default for NvEncLockInputBuffer {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_LOCK_BITSTREAM` — sizeof = 1552.
///
/// Field layout per `<nvEncodeAPI.h>`:
/// -   0: version (u32)
/// -   4: do_not_wait : 1, ltrFrame : 1, getRCStats : 1, reservedBitFields : 29 (u32)
/// -   8: outputBitstream (*mut c_void)
/// -  16: sliceOffsets (*mut u32)
/// -  24: frameIdx (u32)
/// -  28: hwEncodeStatus (u32)
/// -  32: numSlices (u32)
/// -  36: bitstreamSizeInBytes (u32)
/// -  40: outputTimeStamp (u64)
/// -  48: outputDuration (u64)
/// -  56: bitstreamBufferPtr (*mut c_void)
/// -  64: pictureType (u32)
/// -  68: pictureStruct (u32)
/// -  72: frameAvgQP (u32)
/// -  76: frameSatd (u32)
/// -  80: ltrFrameIdx (u32)
/// -  84: ltrFrameBitmap (u32)
/// -  88: temporalId (u32)
/// -  92: intraMBCount (u32)
/// -  96: interMBCount (u32)
/// - 100: averageMVX (i32)
/// - 104: averageMVY (i32)
/// - 108: alphaLayerSizeInBytes (u32)
/// - 112: outputStatsPtrSize (u32) + 4-byte alignment hole
/// - 120: outputStatsPtr (*mut c_void)
/// - 128: frameIdxDisplay (u32)
/// - 132: reserved1[220]      (880 bytes)
/// - 1012+ reserved2[63]      (504 bytes)
/// - 1516+ reservedInternal[8]  (32 bytes)  → 1548 (then 4 padding to 1552)
#[repr(C)]
pub struct NvEncLockBitstream {
    pub version: u32,
    pub flags: u32, // packed bitfield word
    pub output_bitstream: *mut c_void,
    pub slice_offsets: *mut u32,
    pub frame_idx: u32,
    pub hw_encode_status: u32,
    pub num_slices: u32,
    pub bitstream_size_in_bytes: u32,
    pub output_time_stamp: u64,
    pub output_duration: u64,
    pub bitstream_buffer_ptr: *mut c_void,
    pub picture_type: u32,
    pub picture_struct: u32,
    pub frame_avg_qp: u32,
    pub frame_satd: u32,
    pub ltr_frame_idx: u32,
    pub ltr_frame_bitmap: u32,
    pub temporal_id: u32,
    pub intra_mb_count: u32,
    pub inter_mb_count: u32,
    pub average_mvx: i32,
    pub average_mvy: i32,
    pub alpha_layer_size_in_bytes: u32,
    pub output_stats_ptr_size: u32,
    pub _pad0: u32,
    pub output_stats_ptr: *mut c_void,
    pub frame_idx_display: u32,
    pub reserved1: [u32; 221],
    pub reserved2: [*mut c_void; 63],
    pub reserved_internal: [u32; 8],
}

impl Default for NvEncLockBitstream {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_PIC_PARAMS` — sizeof = 3360.
///
/// Field layout per `<nvEncodeAPI.h>`. The codec-specific union sits
/// at offset 80 and is 1024 bytes (256 × u32). After that comes
/// per-block ME hints (16 bytes), then ME hint pointers, then a few
/// scalar fields, then large reserved arrays. We model only the prefix
/// + use the trailing `tail` array for the rest.
#[repr(C)]
pub struct NvEncPicParams {
    pub version: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub input_pitch: u32,
    pub encode_pic_flags: u32,
    pub frame_idx: u32,
    pub input_time_stamp: u64,
    pub input_duration: u64,
    pub input_buffer: *mut c_void,
    pub output_bitstream: *mut c_void,
    pub completion_event: *mut c_void,
    pub buffer_fmt: u32,
    pub picture_struct: u32,
    pub picture_type: u32,
    /// `NV_ENC_CODEC_PIC_PARAMS` — opaque blob (256 × u32). Treated as
    /// reserved/zero — the driver applies sensible defaults when the
    /// caller leaves picture-type-decision (PTD) in driver-mode.
    pub codec_pic_params: [u32; 256],
    /// Combined tail (`meHintCountsPerBlock[2]` + `meExternalHints` +
    /// `reserved1[6]` + `reserved2[2]` + `qpDeltaMap` + scalars +
    /// large reserved arrays). Zero-init is safe.
    pub tail: [u8; 3360 - 80 - 1024],
}

impl Default for NvEncPicParams {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_CONFIG` — sizeof = 3584, alignof = 8. Treated as an opaque
/// blob: we only ever construct it from `nvEncGetEncodePresetConfigEx`
/// and pass it straight back to `nvEncInitializeEncoder` (with the
/// version field re-stamped).
#[repr(C, align(8))]
#[derive(Copy, Clone)]
pub struct NvEncConfig {
    pub bytes: [u8; 3584],
}

impl Default for NvEncConfig {
    fn default() -> Self {
        Self { bytes: [0u8; 3584] }
    }
}

/// `NV_ENC_PRESET_CONFIG` — sizeof = 5128. Wraps a single
/// `NV_ENC_CONFIG` plus reserved padding.
///
/// Layout per `<nvEncodeAPI.h>`:
/// -    0: version (u32)
/// -    8: presetCfg (NV_ENC_CONFIG, 3584 bytes, 8-aligned)
/// - 3592: reserved1[255] (1020 bytes)
/// - 4612: 4 bytes of padding for the 8-aligned reserved2 pointer array
/// - 4616: reserved2[64]  (512 bytes) → ends at 5128
#[repr(C)]
pub struct NvEncPresetConfig {
    pub version: u32,
    pub _pad0: u32, // alignment for the 8-aligned NV_ENC_CONFIG payload
    pub preset_cfg: NvEncConfig,
    pub reserved1: [u32; 255],
    pub _pad1: u32,
    pub reserved2: [*mut c_void; 64],
}

impl Default for NvEncPresetConfig {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENC_INITIALIZE_PARAMS` — sizeof = 1808. Exposed prefix is what we
/// actually fill in; the remainder is reserved + zero.
#[repr(C)]
pub struct NvEncInitializeParams {
    pub version: u32,                      // 0
    pub encode_guid: Guid,                 // 4
    pub preset_guid: Guid,                 // 20
    pub encode_width: u32,                 // 36
    pub encode_height: u32,                // 40
    pub dar_width: u32,                    // 44
    pub dar_height: u32,                   // 48
    pub frame_rate_num: u32,               // 52
    pub frame_rate_den: u32,               // 56
    pub enable_encode_async: u32,          // 60
    pub enable_ptd: u32,                   // 64
    /// Packed bitfields: reportSliceOffsets:1, enableSubFrameWrite:1,
    /// enableExternalMEHints:1, enableMEOnlyMode:1,
    /// enableWeightedPrediction:1, splitEncodeMode:4,
    /// enableOutputInVidmem:1, enableReconFrameOutput:1,
    /// enableOutputStats:1, reservedBitFields:20.
    pub flags: u32,                        // 68
    pub priv_data_size: u32,               // 72
    pub _pad0: u32,
    pub priv_data: *mut c_void,            // 80
    pub encode_config: *mut NvEncConfig,   // 88
    pub max_encode_width: u32,             // 96
    pub max_encode_height: u32,            // 100
    /// `maxMEHintCountsPerBlock[2]` — 2 × 16 bytes.
    pub max_me_hint_counts_per_block: [u8; 32], // 104..136
    pub tuning_info: u32,                  // 136
    pub buffer_format: u32,                // 140
    pub num_state_buffers: u32,            // 144
    pub output_stats_level: u32,           // 148
    /// Trailing reserved bytes (`reserved[285]` + `reserved2[64]` +
    /// rounding to 1808). Zero-init is correct.
    pub tail: [u8; 1808 - 152],
}

impl Default for NvEncInitializeParams {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// `NV_ENCODE_API_FUNCTION_LIST` — sizeof = 2552. We keep the leading
/// version + the function pointer slots we actually use; everything
/// else is reserved.
///
/// Field offsets are tabled in this crate's commit message and verified
/// against the host C compiler.
#[repr(C)]
pub struct NvEncodeApiFunctionList {
    pub version: u32,                                                  // 0
    pub reserved: u32,                                                 // 4
    pub nv_enc_open_encode_session: *mut c_void,                       // 8 (unused)
    pub nv_enc_get_encode_guid_count: *mut c_void,                     // 16 (unused)
    pub nv_enc_get_encode_profile_guid_count: *mut c_void,             // 24 (unused)
    pub nv_enc_get_encode_profile_guids: *mut c_void,                  // 32 (unused)
    pub nv_enc_get_encode_guids: *mut c_void,                          // 40 (unused)
    pub nv_enc_get_input_format_count: *mut c_void,                    // 48 (unused)
    pub nv_enc_get_input_formats: *mut c_void,                         // 56 (unused)
    pub nv_enc_get_encode_caps: *mut c_void,                           // 64 (unused)
    pub nv_enc_get_encode_preset_count: *mut c_void,                   // 72 (unused)
    pub nv_enc_get_encode_preset_guids: *mut c_void,                   // 80 (unused)
    pub nv_enc_get_encode_preset_config: *mut c_void,                  // 88 (unused)
    pub nv_enc_initialize_encoder: PfnNvEncInitializeEncoder,          // 96
    pub nv_enc_create_input_buffer: PfnNvEncCreateInputBuffer,         // 104
    pub nv_enc_destroy_input_buffer: PfnNvEncDestroyInputBuffer,       // 112
    pub nv_enc_create_bitstream_buffer: PfnNvEncCreateBitstreamBuffer, // 120
    pub nv_enc_destroy_bitstream_buffer: PfnNvEncDestroyBitstreamBuffer, // 128
    pub nv_enc_encode_picture: PfnNvEncEncodePicture,                  // 136
    pub nv_enc_lock_bitstream: PfnNvEncLockBitstream,                  // 144
    pub nv_enc_unlock_bitstream: PfnNvEncUnlockBitstream,              // 152
    pub nv_enc_lock_input_buffer: PfnNvEncLockInputBuffer,             // 160
    pub nv_enc_unlock_input_buffer: PfnNvEncUnlockInputBuffer,         // 168
    pub nv_enc_get_encode_stats: *mut c_void,                          // 176 (unused)
    pub nv_enc_get_sequence_params: *mut c_void,                       // 184 (unused)
    pub nv_enc_register_async_event: *mut c_void,                      // 192 (unused)
    pub nv_enc_unregister_async_event: *mut c_void,                    // 200 (unused)
    pub nv_enc_map_input_resource: *mut c_void,                        // 208 (unused)
    pub nv_enc_unmap_input_resource: *mut c_void,                      // 216 (unused)
    pub nv_enc_destroy_encoder: PfnNvEncDestroyEncoder,                // 224
    pub nv_enc_invalidate_ref_frames: *mut c_void,                     // 232 (unused)
    pub nv_enc_open_encode_session_ex: PfnNvEncOpenEncodeSessionEx,    // 240
    pub nv_enc_register_resource: *mut c_void,                         // 248 (unused)
    pub nv_enc_unregister_resource: *mut c_void,                       // 256 (unused)
    pub nv_enc_reconfigure_encoder: *mut c_void,                       // 264 (unused)
    pub reserved1: *mut c_void,                                        // 272
    pub nv_enc_create_mv_buffer: *mut c_void,                          // 280 (unused)
    pub nv_enc_destroy_mv_buffer: *mut c_void,                         // 288 (unused)
    pub nv_enc_run_motion_estimation_only: *mut c_void,                // 296 (unused)
    pub nv_enc_get_last_error_string: PfnNvEncGetLastErrorString,      // 304
    pub nv_enc_set_io_cuda_streams: *mut c_void,                       // 312 (unused)
    pub nv_enc_get_encode_preset_config_ex: PfnNvEncGetEncodePresetConfigEx, // 320
    pub nv_enc_get_sequence_param_ex: *mut c_void,                     // 328 (unused)
    pub nv_enc_restore_encoder_state: *mut c_void,                     // 336 (unused)
    pub nv_enc_lookahead_picture: *mut c_void,                         // 344 (unused)
    pub reserved2: [*mut c_void; 275],                                 // 352..2552
}

impl Default for NvEncodeApiFunctionList {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ─── NVENC function pointer typedefs (subset we use) ────────────────────────

pub type PfnNvEncOpenEncodeSessionEx = Option<
    unsafe extern "C" fn(
        params: *mut NvEncOpenEncodeSessionExParams,
        encoder: *mut *mut c_void,
    ) -> NvEncStatus,
>;
pub type PfnNvEncGetEncodePresetConfigEx = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        encode_guid: Guid,
        preset_guid: Guid,
        tuning_info: u32,
        preset_config: *mut NvEncPresetConfig,
    ) -> NvEncStatus,
>;
pub type PfnNvEncInitializeEncoder = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        params: *mut NvEncInitializeParams,
    ) -> NvEncStatus,
>;
pub type PfnNvEncCreateInputBuffer = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        params: *mut NvEncCreateInputBuffer,
    ) -> NvEncStatus,
>;
pub type PfnNvEncDestroyInputBuffer = Option<
    unsafe extern "C" fn(encoder: *mut c_void, input_buffer: *mut c_void) -> NvEncStatus,
>;
pub type PfnNvEncCreateBitstreamBuffer = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        params: *mut NvEncCreateBitstreamBuffer,
    ) -> NvEncStatus,
>;
pub type PfnNvEncDestroyBitstreamBuffer = Option<
    unsafe extern "C" fn(encoder: *mut c_void, bitstream_buffer: *mut c_void) -> NvEncStatus,
>;
pub type PfnNvEncEncodePicture = Option<
    unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncPicParams) -> NvEncStatus,
>;
pub type PfnNvEncLockBitstream = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        params: *mut NvEncLockBitstream,
    ) -> NvEncStatus,
>;
pub type PfnNvEncUnlockBitstream = Option<
    unsafe extern "C" fn(encoder: *mut c_void, output_buffer: *mut c_void) -> NvEncStatus,
>;
pub type PfnNvEncLockInputBuffer = Option<
    unsafe extern "C" fn(
        encoder: *mut c_void,
        params: *mut NvEncLockInputBuffer,
    ) -> NvEncStatus,
>;
pub type PfnNvEncUnlockInputBuffer = Option<
    unsafe extern "C" fn(encoder: *mut c_void, input_buffer: *mut c_void) -> NvEncStatus,
>;
pub type PfnNvEncDestroyEncoder =
    Option<unsafe extern "C" fn(encoder: *mut c_void) -> NvEncStatus>;
pub type PfnNvEncGetLastErrorString =
    Option<unsafe extern "C" fn(encoder: *mut c_void) -> *const c_char>;

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
    if std::mem::size_of::<Guid>() != 16 {
        panic!("Guid layout drift");
    }
    if std::mem::size_of::<NvEncOpenEncodeSessionExParams>() != 1552 {
        panic!("NvEncOpenEncodeSessionExParams layout drift");
    }
    if std::mem::size_of::<NvEncodeApiFunctionList>() != 2552 {
        panic!("NvEncodeApiFunctionList layout drift");
    }
    if std::mem::size_of::<NvEncInitializeParams>() != 1808 {
        panic!("NvEncInitializeParams layout drift");
    }
    if std::mem::size_of::<NvEncConfig>() != 3584 {
        panic!("NvEncConfig layout drift");
    }
    if std::mem::size_of::<NvEncPresetConfig>() != 5128 {
        panic!("NvEncPresetConfig layout drift");
    }
    if std::mem::size_of::<NvEncCreateInputBuffer>() != 776 {
        panic!("NvEncCreateInputBuffer layout drift");
    }
    if std::mem::size_of::<NvEncCreateBitstreamBuffer>() != 776 {
        panic!("NvEncCreateBitstreamBuffer layout drift");
    }
    if std::mem::size_of::<NvEncLockInputBuffer>() != 1544 {
        panic!("NvEncLockInputBuffer layout drift");
    }
    if std::mem::size_of::<NvEncLockBitstream>() != 1552 {
        panic!("NvEncLockBitstream layout drift");
    }
    if std::mem::size_of::<NvEncPicParams>() != 3360 {
        panic!("NvEncPicParams layout drift");
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
