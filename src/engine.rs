//! Per-process NVIDIA engine probe (Phase-1 contract from
//! `oxideav_core::engine`).
//!
//! [`engine_info`] returns one [`HwDeviceInfo`] entry per visible
//! NVIDIA GPU plus the per-codec NVDEC decode + NVENC encode caps for
//! the codec set this crate registers (H.264 / HEVC / AV1 / VP9). The
//! function is wired into every `CodecInfo` registered by [`crate::register`]
//! via `CodecInfo::with_engine_probe`, so consumers can call it via the
//! registry without depending on this crate directly.
//!
//! Skip-friendly: every error path returns `vec![]`. The CLI's `info`
//! pass treats an empty result as "no devices found" — the same outcome
//! as a host with no NVIDIA driver loaded.
//!
//! # NVDEC capability query
//!
//! Decode caps come from `cuvidGetDecoderCaps(codec, chroma=420,
//! bit_depth=8)`. The query needs a *current* CUDA context; we make one
//! on device 0 (and one for each subsequent device when probing
//! multi-GPU setups).
//!
//! # NVENC capability query
//!
//! Encode caps come from the same NVENC function table the encoder
//! module already uses. We:
//!
//! 1. open a throwaway NVENC encode session on the device's CUDA
//!    context (`nvEncOpenEncodeSessionEx`),
//! 2. enumerate the supported codec GUIDs via
//!    `nvEncGetEncodeGUIDCount` + `nvEncGetEncodeGUIDs`,
//! 3. for each known codec (H.264, HEVC, AV1) call
//!    `nvEncGetEncodeCaps(NV_ENC_CAPS_WIDTH_MAX/HEIGHT_MAX/...)`,
//! 4. close the session via `nvEncDestroyEncoder`.
//!
//! NVENC explicitly does not list VP8 / VP9 / MPEG-2 — only encode-
//! capable codecs are returned by `nvEncGetEncodeGUIDs`.

use std::ffi::c_void;

use oxideav_core::engine::{HwCodecCaps, HwDeviceInfo};

use crate::device::{Cuda, CudaContext, CudaDevice};
use crate::nvdec::nvdec_caps;
use crate::sys::{
    self, CudaVideoCodec, Guid, NvEncCapsParam, NvEncOpenEncodeSessionExParams,
    NvEncodeApiFunctionList, CUDA_VIDEO_CHROMA_FORMAT_420, NVENCAPI_VERSION,
    NV_ENCODE_API_FUNCTION_LIST_VER, NV_ENC_CAPS_HEIGHT_MAX, NV_ENC_CAPS_PARAM_VER,
    NV_ENC_CAPS_SUPPORT_10BIT_ENCODE, NV_ENC_CAPS_WIDTH_MAX, NV_ENC_CODEC_AV1_GUID,
    NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_DEVICE_TYPE_CUDA,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_SUCCESS,
};

// ─────────────────────────── codec table ────────────────────────────────────

/// Codecs we report decode capabilities for. NVDEC supports these
/// across the modern Blackwell / Ada / Ampere generations; the per-
/// codec query reports `is_supported = 0` on older silicon for the
/// codecs that aren't physically present.
const DECODE_CODECS: &[(CudaVideoCodec, &str)] = &[
    (CudaVideoCodec::H264, "h264"),
    (CudaVideoCodec::Hevc, "hevc"),
    (CudaVideoCodec::Av1, "av1"),
    (CudaVideoCodec::Vp9, "vp9"),
    (CudaVideoCodec::Vp8, "vp8"),
    (CudaVideoCodec::Mpeg2, "mpeg2video"),
];

/// Codec GUID → string id used in `HwCodecCaps::codec`. Order matches
/// the capability lookup we do for any GUIDs reported by NVENC.
const ENCODE_CODECS: &[(Guid, &str)] = &[
    (NV_ENC_CODEC_H264_GUID, "h264"),
    (NV_ENC_CODEC_HEVC_GUID, "hevc"),
    (NV_ENC_CODEC_AV1_GUID, "av1"),
];

// ─────────────────────────── public probe ───────────────────────────────────

/// Enumerate every NVIDIA GPU visible to the CUDA driver and report
/// per-codec NVDEC / NVENC capabilities for each one.
///
/// Returns `vec![]` if the CUDA driver isn't loaded, no devices are
/// visible, or any of the expected libraries (`libcuda.so.1`,
/// `libnvcuvid.so.1`, `libnvidia-encode.so.1`) failed to dlopen.
///
/// Per-device errors during the probe are folded silently into the
/// returned `HwDeviceInfo`: a device whose NVENC session refuses to
/// open just gets `encode = false` flags across the board, with the
/// decode caps still reported. The probe never panics.
pub fn engine_info() -> Vec<HwDeviceInfo> {
    let cuda = match Cuda::init() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let count = match cuda.device_count() {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as i32 {
        let device = match cuda.device(i) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if let Some(info) = probe_device(&cuda, &device) {
            out.push(info);
        }
    }
    out
}

// ─────────────────────────── per-device probe ───────────────────────────────

fn probe_device(cuda: &Cuda, device: &CudaDevice) -> Option<HwDeviceInfo> {
    let name = device.name().ok()?;
    let total_mem = device.total_memory_bytes().ok();
    let cc = device.compute_capability().ok();

    // Driver version — encoded as 1000*major + 10*minor (e.g. 12060
    // for CUDA 12.6).
    let (driver_version, api_version) = driver_versions().unwrap_or((None, None));

    // Build CUDA context current on this thread for both NVDEC and
    // NVENC queries. Drops at end of function.
    let ctx = cuda.create_context_for(device).ok()?;

    let mut codecs: Vec<HwCodecCaps> = DECODE_CODECS
        .iter()
        .map(|&(codec, id)| probe_decode(codec, id))
        .collect();

    // Encode caps via NVENC. May fail on some setups (driver older
    // than the API our headers target, container without GPU access,
    // etc.); on failure we just leave `encode = false` everywhere.
    if let Err(e) = annotate_encode(&ctx, &mut codecs) {
        eprintln!("oxideav-nvidia: NVENC probe skipped: {e}");
    }

    let mut extra: Vec<(String, String)> = Vec::new();
    if let Some((maj, min)) = cc {
        extra.push(("compute_capability".into(), format!("{maj}.{min}")));
    }

    Some(HwDeviceInfo {
        name,
        driver_version,
        api_version,
        total_memory_bytes: total_mem,
        extra,
        codecs,
    })
}

/// Returns `(driver_version_string, api_version_string)` from
/// `cuDriverGetVersion`. The driver exposes the version as
/// `1000*major + 10*minor`, e.g. `12060 -> "12.6"`.
fn driver_versions() -> Option<(Option<String>, Option<String>)> {
    let vt = sys::vtable().ok()?;
    let mut v: i32 = 0;
    // SAFETY: `cuDriverGetVersion` only writes to its single output
    // parameter; we own the i32.
    let r = unsafe { (vt.cu_driver_get_version)(&mut v) };
    if r != 0 || v <= 0 {
        return Some((None, None));
    }
    let major = v / 1000;
    let minor = (v % 1000) / 10;
    let driver = format!("{major}.{minor}");
    let api = format!("CUDA {major}.{minor}");
    Some((Some(driver), Some(api)))
}

// ─────────────────────────── NVDEC ──────────────────────────────────────────

fn probe_decode(codec: CudaVideoCodec, id: &str) -> HwCodecCaps {
    let mut caps = HwCodecCaps {
        codec: id.to_string(),
        decode: false,
        encode: false,
        max_width: None,
        max_height: None,
        max_bit_depth: None,
        profiles: Vec::new(),
        extra: Vec::new(),
    };

    if let Ok(c) = nvdec_caps(codec, CUDA_VIDEO_CHROMA_FORMAT_420, 8) {
        if c.is_supported != 0 {
            caps.decode = true;
            if c.max_width > 0 {
                caps.max_width = Some(c.max_width);
            }
            if c.max_height > 0 {
                caps.max_height = Some(c.max_height);
            }
            caps.max_bit_depth = Some(8);
            if c.max_mb_count > 0 {
                caps.extra
                    .push(("max_mb_count".into(), c.max_mb_count.to_string()));
            }
            if c.num_nvdecs > 0 {
                caps.extra
                    .push(("num_nvdecs".into(), c.num_nvdecs.to_string()));
            }
            // Probe 10-bit support too — most modern engines support
            // it for HEVC / AV1 / VP9, none for H.264 baseline.
            if let Ok(c10) = nvdec_caps(codec, CUDA_VIDEO_CHROMA_FORMAT_420, 10) {
                if c10.is_supported != 0 {
                    caps.max_bit_depth = Some(10);
                }
            }
        }
    }
    caps
}

// ─────────────────────────── NVENC ──────────────────────────────────────────

/// Annotate the codec list with NVENC encode caps. Mutates `codecs` in
/// place. Returns `Err` if the NVENC bootstrap fails altogether — a
/// per-codec failure is folded silently (the codec just stays
/// `encode = false`).
fn annotate_encode(ctx: &CudaContext, codecs: &mut [HwCodecCaps]) -> Result<(), String> {
    let fns = nvenc_function_table()?;

    // ── open a throwaway encode session ────────────────────────────
    let mut sess = NvEncOpenEncodeSessionExParams {
        version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
        device_type: NV_ENC_DEVICE_TYPE_CUDA,
        device: ctx.raw(),
        api_version: NVENCAPI_VERSION,
        ..Default::default()
    };

    let open = fns
        .nv_enc_open_encode_session_ex
        .ok_or_else(|| "nvEncOpenEncodeSessionEx slot empty".to_string())?;
    let mut encoder: *mut c_void = std::ptr::null_mut();
    // SAFETY: `sess` is a #[repr(C)] struct populated per ABI; the
    // driver writes the resulting handle into `&mut encoder`.
    let r = unsafe { open(&mut sess, &mut encoder) };
    if r != NV_ENC_SUCCESS {
        return Err(format!("nvEncOpenEncodeSessionEx -> {r}"));
    }
    // Always close, even on error in the body below.
    let _guard = SessionGuard { fns, encoder };

    // ── enumerate the codecs the session can encode to ─────────────
    let supported_guids = enumerate_encode_guids(fns, encoder).unwrap_or_default();

    // For each known codec, see if its GUID is in the supported
    // set; if so, query width/height/10-bit caps.
    let get_caps = match fns.nv_enc_get_encode_caps {
        Some(f) => f,
        None => return Err("nvEncGetEncodeCaps slot empty".to_string()),
    };

    for &(guid, id) in ENCODE_CODECS {
        if !supported_guids.contains(&guid) {
            continue;
        }
        // Locate the matching cap entry (decode entry created in
        // probe_decode). If we don't already have a row for this codec
        // we add a fresh one — H.264 / HEVC / AV1 are always in
        // DECODE_CODECS so this only matters for sources we haven't
        // probed for decode.
        let row = match codecs.iter_mut().find(|c| c.codec == id) {
            Some(r) => r,
            None => continue, // shouldn't happen given DECODE_CODECS
        };
        row.encode = true;

        let mut max_w: i32 = 0;
        let mut max_h: i32 = 0;
        let mut ten_bit: i32 = 0;
        unsafe {
            let _ = get_caps(
                encoder,
                guid,
                &mut caps_param(NV_ENC_CAPS_WIDTH_MAX),
                &mut max_w,
            );
            let _ = get_caps(
                encoder,
                guid,
                &mut caps_param(NV_ENC_CAPS_HEIGHT_MAX),
                &mut max_h,
            );
            let _ = get_caps(
                encoder,
                guid,
                &mut caps_param(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE),
                &mut ten_bit,
            );
        }
        // Take the max over decode-side and encode-side reported
        // dimensions — NVDEC and NVENC frequently advertise the same
        // hard limit, but on some SKUs the encoder-side limit is
        // bigger (e.g. 8192 vs decode's 4096) or vice-versa.
        if max_w > 0 {
            row.max_width = Some(row.max_width.unwrap_or(0).max(max_w as u32));
        }
        if max_h > 0 {
            row.max_height = Some(row.max_height.unwrap_or(0).max(max_h as u32));
        }
        if ten_bit > 0 {
            // Only widen, never narrow: a codec we already flagged as
            // "decode 10-bit" stays 10-bit even if the encoder is 8-bit.
            row.max_bit_depth = Some(row.max_bit_depth.unwrap_or(0).max(10));
        } else if row.max_bit_depth.is_none() {
            row.max_bit_depth = Some(8);
        }
    }

    Ok(())
}

fn caps_param(which: u32) -> NvEncCapsParam {
    NvEncCapsParam {
        version: NV_ENC_CAPS_PARAM_VER,
        caps_to_query: which,
        ..Default::default()
    }
}

fn enumerate_encode_guids(
    fns: &NvEncodeApiFunctionList,
    encoder: *mut c_void,
) -> Result<Vec<Guid>, String> {
    let get_count = fns
        .nv_enc_get_encode_guid_count
        .ok_or_else(|| "nvEncGetEncodeGUIDCount slot empty".to_string())?;
    let mut count: u32 = 0;
    // SAFETY: count is a stack u32; encoder pointer was returned by
    // nvEncOpenEncodeSessionEx and is currently open.
    let r = unsafe { get_count(encoder, &mut count) };
    if r != NV_ENC_SUCCESS || count == 0 {
        return Ok(Vec::new());
    }
    let get_guids = fns
        .nv_enc_get_encode_guids
        .ok_or_else(|| "nvEncGetEncodeGUIDs slot empty".to_string())?;
    let mut buf: Vec<Guid> = vec![Guid::default(); count as usize];
    let mut written: u32 = 0;
    // SAFETY: buffer is exactly `count` entries; we ask the driver to
    // fill at most `count` and tell us how many it wrote.
    let r = unsafe { get_guids(encoder, buf.as_mut_ptr(), count, &mut written) };
    if r != NV_ENC_SUCCESS {
        return Ok(Vec::new());
    }
    buf.truncate(written as usize);
    Ok(buf)
}

/// Resolve the NVENC function table (`NvEncodeAPICreateInstance`),
/// independently of the encoder module's `OnceLock` — the encoder
/// module gates this behind the `registry` feature and we want
/// `engine_info` to work even when consumers turn `registry` off.
///
/// We allocate a fresh function-list every call rather than caching:
/// the function table is small (2552 bytes) and the call only happens
/// once per CLI `info` pass. Caching here would duplicate the
/// encoder-module cache; keeping them separate avoids a feature-gated
/// reach across modules.
fn nvenc_function_table() -> Result<&'static NvEncodeApiFunctionList, String> {
    /// Wrapper that makes the function-list pointer `Send + Sync`. The
    /// table only contains function pointers + reserved padding; once
    /// initialised it's read-only — the encoder module uses the same
    /// wrapper for the same reason.
    struct Holder(Box<NvEncodeApiFunctionList>);
    unsafe impl Send for Holder {}
    unsafe impl Sync for Holder {}

    static CACHE: std::sync::OnceLock<Result<Holder, String>> = std::sync::OnceLock::new();

    let res = CACHE.get_or_init(|| {
        let vt = sys::vtable().map_err(|e| format!("vtable: {e}"))?;
        let mut fns = Box::new(NvEncodeApiFunctionList::default());
        fns.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        let ptr = &mut *fns as *mut NvEncodeApiFunctionList as *mut c_void;
        // SAFETY: ptr has the exact size + version stamp the driver
        // expects; ABI verified by the size-guard in sys.rs.
        let r = unsafe { (vt.nv_encode_api_create_instance)(ptr) };
        if r != NV_ENC_SUCCESS {
            return Err(format!("NvEncodeAPICreateInstance -> NVENCSTATUS {r}"));
        }
        Ok(Holder(fns))
    });
    res.as_ref()
        .map(|h| {
            // SAFETY: the cached `Holder` lives for the process
            // lifetime; the function table inside is never mutated
            // after the `OnceLock` first resolves.
            let ptr: *const NvEncodeApiFunctionList = &*h.0;
            unsafe { &*ptr }
        })
        .map_err(|e| e.clone())
}

/// RAII wrapper that calls `nvEncDestroyEncoder` on drop. Used by the
/// engine probe to make sure the throwaway session opened for cap
/// queries is always closed.
struct SessionGuard<'a> {
    fns: &'a NvEncodeApiFunctionList,
    encoder: *mut c_void,
}

impl Drop for SessionGuard<'_> {
    fn drop(&mut self) {
        if self.encoder.is_null() {
            return;
        }
        if let Some(d) = self.fns.nv_enc_destroy_encoder {
            // SAFETY: the encoder pointer was returned by a successful
            // nvEncOpenEncodeSessionEx and hasn't been destroyed yet.
            unsafe {
                let _ = d(self.encoder);
            }
        }
        self.encoder = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `engine_info` must always be safe to call — even on hosts with no
    /// NVIDIA driver / no GPU. The function should swallow load errors
    /// and return an empty vector rather than panicking.
    #[test]
    fn engine_info_never_panics() {
        let _probes = engine_info();
    }
}
