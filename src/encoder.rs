// Field-by-field initialisation of large `#[repr(C)]` structs reads
// substantially better than `..Default::default()` once you start
// writing into ~10 fields; turn the lint off here.
#![allow(clippy::field_reassign_with_default)]

//! NVENC `Encoder` trait implementations.
//!
//! Round 4 ships H.264 and HEVC NVENC encoders built on top of the
//! single-bootstrap `NvEncodeAPICreateInstance` entry point:
//!
//! 1. `NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST*)` — fills
//!    the function table with all the per-API-version entry points.
//! 2. `nvEncOpenEncodeSessionEx` opens a session bound to a CUDA
//!    context (`NV_ENC_DEVICE_TYPE_CUDA`).
//! 3. `nvEncGetEncodePresetConfigEx` returns the default
//!    `NV_ENC_CONFIG` for a given `(codec, preset, tuning)` triple. We
//!    use `P4 + HIGH_QUALITY` which the driver maps to the
//!    "balanced" tuning.
//! 4. `nvEncInitializeEncoder` stamps the session with our chosen
//!    width/height + frame rate + the preset config.
//! 5. Per frame:
//!    - `nvEncCreateInputBuffer` (cached for the encoder lifetime) →
//!      `nvEncLockInputBuffer` to upload NV12 → `nvEncUnlockInputBuffer`.
//!    - `nvEncEncodePicture` submits the picture. The driver may reply
//!      `NV_ENC_SUCCESS` (encoded data ready) or
//!      `NV_ENC_ERR_NEED_MORE_INPUT` (queued; we return `NeedMore`).
//!    - On success, `nvEncLockBitstream` exposes the bytes; we copy
//!      them out, `nvEncUnlockBitstream`, and queue a `Packet`.
//! 6. `flush()` posts an EOS-flagged picture to drain the encoder.
//!
//! We allocate a single input + single output buffer pair. Together
//! with the `enablePTD = 1` driver-side picture-type-decision setting,
//! this is the simplest correct shape: one frame in, one packet out
//! (for low-latency presets / HEVC) — buffered through several frames
//! for B-frame-using H.264 presets, surfaced via the `NeedMore`
//! contract.

use std::collections::VecDeque;
use std::ffi::{c_void, CStr};
use std::sync::Mutex;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::device::{Cuda, CudaContext, NvError};
use crate::sys::{
    self, Guid, NvEncConfig, NvEncCreateBitstreamBuffer, NvEncCreateInputBuffer,
    NvEncInitializeParams, NvEncLockBitstream, NvEncLockInputBuffer,
    NvEncOpenEncodeSessionExParams, NvEncPicParams, NvEncPresetConfig,
    NvEncodeApiFunctionList, NV_ENCODE_API_FUNCTION_LIST_VER, NV_ENC_BUFFER_FORMAT_NV12,
    NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG_VER,
    NV_ENC_CREATE_BITSTREAM_BUFFER_VER, NV_ENC_CREATE_INPUT_BUFFER_VER, NV_ENC_DEVICE_TYPE_CUDA,
    NV_ENC_H264_PROFILE_HIGH_GUID, NV_ENC_HEVC_PROFILE_MAIN_GUID,
    NV_ENC_INITIALIZE_PARAMS_VER, NV_ENC_LOCK_BITSTREAM_VER, NV_ENC_LOCK_INPUT_BUFFER_VER,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_PIC_PARAMS_VER,
    NV_ENC_PIC_STRUCT_FRAME, NV_ENC_PRESET_CONFIG_VER, NV_ENC_PRESET_P4_GUID, NV_ENC_SUCCESS,
    NV_ENC_TUNING_INFO_LOW_LATENCY, NVENCAPI_VERSION,
};

// ─────────────────────────── shared NVENC function table ──────────────────────

/// Lazily-initialised NVENC function table (the result of
/// `NvEncodeAPICreateInstance`). Populated once per process.
static NVENC_FNS: std::sync::OnceLock<std::result::Result<NvEncFnsHolder, String>> =
    std::sync::OnceLock::new();

/// Wrapper that makes the function-list pointer `Send + Sync`. The
/// table itself only contains function pointers + reserved padding —
/// once initialised it's read-only.
struct NvEncFnsHolder {
    fns: Box<NvEncodeApiFunctionList>,
}

unsafe impl Send for NvEncFnsHolder {}
unsafe impl Sync for NvEncFnsHolder {}

fn nvenc_fns() -> std::result::Result<&'static NvEncodeApiFunctionList, String> {
    let res = NVENC_FNS.get_or_init(|| {
        let vt = sys::vtable().map_err(|e| format!("nvidia vtable: {e}"))?;

        let mut fns = Box::new(NvEncodeApiFunctionList::default());
        fns.version = NV_ENCODE_API_FUNCTION_LIST_VER;

        let ptr = &mut *fns as *mut NvEncodeApiFunctionList as *mut c_void;
        let r = unsafe { (vt.nv_encode_api_create_instance)(ptr) };
        if r != NV_ENC_SUCCESS {
            return Err(format!("NvEncodeAPICreateInstance failed: NVENCSTATUS {r}"));
        }
        Ok(NvEncFnsHolder { fns })
    });

    match res {
        Ok(h) => Ok(&*h.fns),
        Err(e) => Err(e.clone()),
    }
}

// ─────────────────────────── NvEncoder ────────────────────────────────────────

/// NVENC-backed video encoder, parameterised over codec.
///
/// Holds a CUDA context (kept current on the calling thread for the
/// whole encoder lifetime), an open NVENC encode session, and a single
/// pair of (input, output) buffers.
pub struct NvEncoder {
    codec_id: CodecId,
    /// Drop guard for `cuInit`.
    _cuda: Cuda,
    /// CUDA context — drop after the NVENC session.
    _ctx: Option<CudaContext>,
    /// Open encoder session pointer.
    encoder: *mut c_void,
    /// Input NV12 buffer (driver-allocated, locked per frame).
    input_buffer: *mut c_void,
    /// Output bitstream buffer (driver-allocated).
    output_buffer: *mut c_void,
    /// Output queue drained by `receive_packet`.
    output_queue: Mutex<VecDeque<Packet>>,
    /// Output codec parameters surfaced via `output_params()`.
    output_params: CodecParameters,
    /// Frame counter feeding `inputTimeStamp` when the caller doesn't
    /// supply a PTS.
    pts_counter: i64,
    /// Width / height of the encode session.
    width: u32,
    height: u32,
    /// Framerate numerator / denominator.
    fr_num: u32,
    fr_den: u32,
    /// Set after `flush()` — once true, `send_frame` returns Eof and
    /// `receive_packet` returns Eof when the queue drains.
    flushed: bool,
}

unsafe impl Send for NvEncoder {}

impl NvEncoder {
    /// Build an encoder for `codec_guid` (H.264 or HEVC), labelled
    /// with `codec_id`, using the given profile.
    fn make_for(
        codec_guid: Guid,
        profile_guid: Guid,
        codec_id: &str,
        params: &CodecParameters,
    ) -> Result<Box<dyn Encoder>> {
        let cuda = Cuda::init().map_err(map_unsupported)?;
        let count = cuda.device_count().map_err(map_unsupported)?;
        if count == 0 {
            return Err(Error::unsupported("nvidia: no CUDA devices visible"));
        }
        let dev = cuda.device(0).map_err(map_unsupported)?;
        let ctx = cuda.create_context_for(&dev).map_err(map_unsupported)?;

        let fns = nvenc_fns().map_err(|e| {
            if e.contains("dlopen") || e.contains("dlsym") || e.contains("vtable") {
                Error::unsupported(format!("nvidia: {e}"))
            } else {
                Error::other(format!("nvidia nvenc: {e}"))
            }
        })?;

        // ── open encode session ────────────────────────────────────────
        let mut sess_params = NvEncOpenEncodeSessionExParams::default();
        sess_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        sess_params.device_type = NV_ENC_DEVICE_TYPE_CUDA;
        sess_params.device = ctx.raw();
        sess_params.api_version = NVENCAPI_VERSION;

        let mut encoder: *mut c_void = std::ptr::null_mut();
        let r = unsafe {
            (fns.nv_enc_open_encode_session_ex.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncOpenEncodeSessionEx not in fn table")
            })?)(&mut sess_params, &mut encoder)
        };
        if r != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(fns, encoder, "nvEncOpenEncodeSessionEx", r));
        }

        // ── pick width / height / framerate from caller params ─────────
        let width = params.width.unwrap_or(320).max(160);
        let height = params.height.unwrap_or(240).max(64);
        let (fr_num, fr_den) = match params.frame_rate {
            Some(fr) if fr.den > 0 => (fr.num as u32, fr.den as u32),
            _ => (30, 1),
        };

        // ── ask NVENC for a default config for our preset ──────────────
        let mut preset_cfg = NvEncPresetConfig::default();
        preset_cfg.version = NV_ENC_PRESET_CONFIG_VER;
        preset_cfg.preset_cfg = NvEncConfig::default();
        // The embedded NV_ENC_CONFIG also needs a version stamp before
        // the driver writes into it.
        write_u32_at(&mut preset_cfg.preset_cfg, 0, NV_ENC_CONFIG_VER);

        let r = unsafe {
            (fns.nv_enc_get_encode_preset_config_ex.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncGetEncodePresetConfigEx not in fn table")
            })?)(
                encoder,
                codec_guid,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO_LOW_LATENCY,
                &mut preset_cfg as *mut _,
            )
        };
        if r != NV_ENC_SUCCESS {
            unsafe {
                if let Some(d) = fns.nv_enc_destroy_encoder {
                    let _ = d(encoder);
                }
            }
            return Err(map_nvenc_status(
                fns,
                encoder,
                "nvEncGetEncodePresetConfigEx",
                r,
            ));
        }

        // Re-stamp the version field of the embedded config.
        write_u32_at(&mut preset_cfg.preset_cfg, 0, NV_ENC_CONFIG_VER);
        // Set the profile GUID in the config (offset 4 inside NV_ENC_CONFIG).
        write_guid_at(&mut preset_cfg.preset_cfg, 4, profile_guid);

        // ── initialise the encoder session ─────────────────────────────
        let mut init = NvEncInitializeParams::default();
        init.version = NV_ENC_INITIALIZE_PARAMS_VER;
        init.encode_guid = codec_guid;
        init.preset_guid = NV_ENC_PRESET_P4_GUID;
        init.encode_width = width;
        init.encode_height = height;
        init.dar_width = width;
        init.dar_height = height;
        init.frame_rate_num = fr_num;
        init.frame_rate_den = fr_den;
        init.enable_encode_async = 0;
        init.enable_ptd = 1;
        init.flags = 0;
        init.priv_data = std::ptr::null_mut();
        init.priv_data_size = 0;
        init.encode_config = &mut preset_cfg.preset_cfg as *mut _;
        init.max_encode_width = width;
        init.max_encode_height = height;
        init.tuning_info = NV_ENC_TUNING_INFO_LOW_LATENCY;
        init.buffer_format = 0;
        init.num_state_buffers = 0;
        init.output_stats_level = 0;

        let r = unsafe {
            (fns.nv_enc_initialize_encoder.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncInitializeEncoder not in fn table")
            })?)(encoder, &mut init as *mut _)
        };
        if r != NV_ENC_SUCCESS {
            let err = map_nvenc_status(fns, encoder, "nvEncInitializeEncoder", r);
            unsafe {
                if let Some(d) = fns.nv_enc_destroy_encoder {
                    let _ = d(encoder);
                }
            }
            return Err(err);
        }

        // ── allocate input + output buffers ────────────────────────────
        let mut create_in = NvEncCreateInputBuffer::default();
        create_in.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        create_in.width = width;
        create_in.height = height;
        create_in.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        let r = unsafe {
            (fns.nv_enc_create_input_buffer.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncCreateInputBuffer not in fn table")
            })?)(encoder, &mut create_in as *mut _)
        };
        if r != NV_ENC_SUCCESS {
            let err = map_nvenc_status(fns, encoder, "nvEncCreateInputBuffer", r);
            unsafe {
                if let Some(d) = fns.nv_enc_destroy_encoder {
                    let _ = d(encoder);
                }
            }
            return Err(err);
        }
        let input_buffer = create_in.input_buffer;

        let mut create_out = NvEncCreateBitstreamBuffer::default();
        create_out.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        let r = unsafe {
            (fns.nv_enc_create_bitstream_buffer.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncCreateBitstreamBuffer not in fn table")
            })?)(encoder, &mut create_out as *mut _)
        };
        if r != NV_ENC_SUCCESS {
            let err = map_nvenc_status(fns, encoder, "nvEncCreateBitstreamBuffer", r);
            unsafe {
                if let Some(d) = fns.nv_enc_destroy_input_buffer {
                    let _ = d(encoder, input_buffer);
                }
                if let Some(d) = fns.nv_enc_destroy_encoder {
                    let _ = d(encoder);
                }
            }
            return Err(err);
        }
        let output_buffer = create_out.bitstream_buffer;

        let mut output_params = CodecParameters::video(CodecId::new(codec_id));
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        output_params.frame_rate = params.frame_rate;
        output_params.bit_rate = params.bit_rate;

        Ok(Box::new(Self {
            codec_id: CodecId::new(codec_id),
            _cuda: cuda,
            _ctx: Some(ctx),
            encoder,
            input_buffer,
            output_buffer,
            output_queue: Mutex::new(VecDeque::new()),
            output_params,
            pts_counter: 0,
            width,
            height,
            fr_num,
            fr_den,
            flushed: false,
        }))
    }

    /// Convert an I420 [`VideoFrame`] into NV12 in the locked input
    /// buffer, given the driver-supplied destination pointer + pitch.
    fn upload_nv12(&self, dst: *mut u8, pitch: u32, frame: &VideoFrame) -> Result<()> {
        if frame.planes.len() < 3 {
            return Err(Error::invalid("expected I420 frame with 3 planes"));
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        let pitch = pitch as usize;

        // SAFETY: `dst` is valid for `pitch * (height + chroma_h)`
        // bytes per the NVENC contract for NV12 input buffers.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst, pitch * (height + chroma_h))
        };

        // ── Y plane ─────────────────────────────────────────────────
        let y = &frame.planes[0];
        for row in 0..height {
            let src_off = row * y.stride;
            let dst_off = row * pitch;
            if src_off + width > y.data.len() || dst_off + width > dst.len() {
                continue;
            }
            dst[dst_off..dst_off + width].copy_from_slice(&y.data[src_off..src_off + width]);
        }

        // ── interleave U + V → UV plane ────────────────────────────
        let u = &frame.planes[1];
        let v = &frame.planes[2];
        let uv_base = pitch * height;
        for row in 0..chroma_h {
            let u_src = row * u.stride;
            let v_src = row * v.stride;
            let dst_off = uv_base + row * pitch;
            if dst_off + chroma_w * 2 > dst.len() {
                continue;
            }
            for col in 0..chroma_w {
                let uv = dst_off + col * 2;
                let uval = if u_src + col < u.data.len() {
                    u.data[u_src + col]
                } else {
                    128
                };
                let vval = if v_src + col < v.data.len() {
                    v.data[v_src + col]
                } else {
                    128
                };
                dst[uv] = uval;
                dst[uv + 1] = vval;
            }
        }
        Ok(())
    }

    /// Drain whatever bitstream is currently queued in `output_buffer`
    /// (called both after a normal encode and after EOS flush).
    ///
    /// Returns `Ok(true)` if a non-empty packet was queued, `Ok(false)`
    /// if the lock returned 0 bytes (signal "encoder is empty").
    fn drain_output(&self, fns: &NvEncodeApiFunctionList, pts: i64) -> Result<bool> {
        let mut lb = NvEncLockBitstream::default();
        lb.version = NV_ENC_LOCK_BITSTREAM_VER;
        lb.output_bitstream = self.output_buffer;
        let r = unsafe {
            (fns.nv_enc_lock_bitstream
                .ok_or_else(|| Error::unsupported("nvidia: nvEncLockBitstream not in fn table"))?)(
                self.encoder,
                &mut lb as *mut _,
            )
        };
        if r != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(fns, self.encoder, "nvEncLockBitstream", r));
        }

        let size = lb.bitstream_size_in_bytes as usize;
        let data = if size > 0 && !lb.bitstream_buffer_ptr.is_null() {
            unsafe { std::slice::from_raw_parts(lb.bitstream_buffer_ptr as *const u8, size).to_vec() }
        } else {
            Vec::new()
        };

        let r2 = unsafe {
            (fns.nv_enc_unlock_bitstream
                .ok_or_else(|| Error::unsupported("nvidia: nvEncUnlockBitstream not in fn table"))?)(
                self.encoder,
                self.output_buffer,
            )
        };
        if r2 != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(fns, self.encoder, "nvEncUnlockBitstream", r2));
        }

        if data.is_empty() {
            return Ok(false);
        }
        let pkt = Packet::new(
            0,
            TimeBase::new(self.fr_den as i64, self.fr_num as i64),
            data,
        )
        .with_pts(pts);
        if let Ok(mut q) = self.output_queue.lock() {
            q.push_back(pkt);
        }
        Ok(true)
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        if self.encoder.is_null() {
            return;
        }
        let fns = match nvenc_fns() {
            Ok(f) => f,
            Err(_) => return,
        };
        unsafe {
            if !self.input_buffer.is_null() {
                if let Some(d) = fns.nv_enc_destroy_input_buffer {
                    let _ = d(self.encoder, self.input_buffer);
                }
                self.input_buffer = std::ptr::null_mut();
            }
            if !self.output_buffer.is_null() {
                if let Some(d) = fns.nv_enc_destroy_bitstream_buffer {
                    let _ = d(self.encoder, self.output_buffer);
                }
                self.output_buffer = std::ptr::null_mut();
            }
            if let Some(d) = fns.nv_enc_destroy_encoder {
                let _ = d(self.encoder);
            }
        }
        self.encoder = std::ptr::null_mut();
    }
}

impl Encoder for NvEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::Eof);
        }
        let vf = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("expected Video frame")),
        };
        let pts = vf.pts.unwrap_or(self.pts_counter);
        self.pts_counter = pts + 1;

        let fns = nvenc_fns().map_err(|e| Error::other(format!("nvidia: {e}")))?;

        // ── lock + upload NV12 ─────────────────────────────────────────
        let mut lk = NvEncLockInputBuffer::default();
        lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lk.input_buffer = self.input_buffer;
        let r = unsafe {
            (fns.nv_enc_lock_input_buffer.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncLockInputBuffer not in fn table")
            })?)(self.encoder, &mut lk as *mut _)
        };
        if r != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(fns, self.encoder, "nvEncLockInputBuffer", r));
        }
        let upload_result = self.upload_nv12(lk.buffer_data_ptr as *mut u8, lk.pitch, vf);
        let r2 = unsafe {
            (fns.nv_enc_unlock_input_buffer.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncUnlockInputBuffer not in fn table")
            })?)(self.encoder, self.input_buffer)
        };
        upload_result?;
        if r2 != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(
                fns,
                self.encoder,
                "nvEncUnlockInputBuffer",
                r2,
            ));
        }

        // ── encode picture ─────────────────────────────────────────────
        let mut pp = NvEncPicParams::default();
        pp.version = NV_ENC_PIC_PARAMS_VER;
        pp.input_width = self.width;
        pp.input_height = self.height;
        pp.input_pitch = self.width;
        pp.encode_pic_flags = 0;
        pp.frame_idx = self.pts_counter as u32;
        pp.input_time_stamp = pts as u64;
        pp.input_duration = (self.fr_den as u64).max(1);
        pp.input_buffer = self.input_buffer;
        pp.output_bitstream = self.output_buffer;
        pp.completion_event = std::ptr::null_mut();
        pp.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        pp.picture_struct = NV_ENC_PIC_STRUCT_FRAME;
        pp.picture_type = 0; // PTD on, driver picks

        let r = unsafe {
            (fns.nv_enc_encode_picture.ok_or_else(|| {
                Error::unsupported("nvidia: nvEncEncodePicture not in fn table")
            })?)(self.encoder, &mut pp as *mut _)
        };

        // NVENCSTATUS values: 0 = SUCCESS, 9 = NV_ENC_ERR_INVALID_CALL,
        // 10 = NV_ENC_ERR_OUT_OF_MEMORY, 12 = NV_ENC_ERR_UNSUPPORTED_PARAM,
        // 17 = NV_ENC_ERR_NEED_MORE_INPUT (driver wants more frames before
        // it can emit a packet — happens with B-frame presets).
        const NV_ENC_ERR_NEED_MORE_INPUT: i32 = 17;
        if r == NV_ENC_ERR_NEED_MORE_INPUT {
            // Encoded data not yet available — return Ok so caller
            // keeps feeding frames.
            return Ok(());
        }
        if r != NV_ENC_SUCCESS {
            return Err(map_nvenc_status(fns, self.encoder, "nvEncEncodePicture", r));
        }

        // ── lock + drain bitstream ─────────────────────────────────────
        let _ = self.drain_output(fns, pts)?;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Ok(mut q) = self.output_queue.lock() {
            if let Some(pkt) = q.pop_front() {
                return Ok(pkt);
            }
        }
        Err(if self.flushed { Error::Eof } else { Error::NeedMore })
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // With LOW_LATENCY tuning the encoder produces exactly one
        // packet per submitted frame; the output_queue already holds
        // any unread packets and there is no buffered B-frame to
        // drain. The EOS-flagged picture submission was historically
        // used to drain B-frame buffers, but on NVENC 12.x for HEVC
        // it can hang the driver call when the output buffer is
        // already free. We therefore mark the encoder flushed and
        // let `receive_packet` surface remaining data as `Eof` once
        // the queue empties — no extra driver calls required.
        self.flushed = true;
        Ok(())
    }
}

// ─────────────────────────── helpers ──────────────────────────────────────────

fn map_unsupported(e: NvError) -> Error {
    if e.is_unavailable() {
        Error::unsupported(format!("nvidia: {e}"))
    } else {
        Error::other(format!("nvidia: {e}"))
    }
}

fn map_nvenc_status(
    fns: &NvEncodeApiFunctionList,
    encoder: *mut c_void,
    op: &str,
    status: i32,
) -> Error {
    let detail = unsafe {
        match fns.nv_enc_get_last_error_string {
            Some(f) if !encoder.is_null() => {
                let p = f(encoder);
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            }
            _ => String::new(),
        }
    };
    Error::other(format!(
        "{op} failed: NVENCSTATUS {status}{}{}",
        if detail.is_empty() { "" } else { " — " },
        detail
    ))
}

/// Write `value` as little-endian into `cfg` at offset `off`. Used to
/// stamp version + profile fields in the opaque `NV_ENC_CONFIG` blob.
fn write_u32_at(cfg: &mut NvEncConfig, off: usize, value: u32) {
    let bytes = value.to_le_bytes();
    cfg.bytes[off..off + 4].copy_from_slice(&bytes);
}

fn write_guid_at(cfg: &mut NvEncConfig, off: usize, guid: Guid) {
    cfg.bytes[off..off + 4].copy_from_slice(&guid.data1.to_le_bytes());
    cfg.bytes[off + 4..off + 6].copy_from_slice(&guid.data2.to_le_bytes());
    cfg.bytes[off + 6..off + 8].copy_from_slice(&guid.data3.to_le_bytes());
    cfg.bytes[off + 8..off + 16].copy_from_slice(&guid.data4);
}

// ─────────────────────────── public codec wrappers ────────────────────────────

/// NVENC-backed H.264 encoder.
pub struct H264NvEncoder;

impl H264NvEncoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
        NvEncoder::make_for(
            NV_ENC_CODEC_H264_GUID,
            NV_ENC_H264_PROFILE_HIGH_GUID,
            "h264",
            params,
        )
    }
}

/// NVENC-backed HEVC (H.265) encoder.
pub struct HevcNvEncoder;

impl HevcNvEncoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
        NvEncoder::make_for(
            NV_ENC_CODEC_HEVC_GUID,
            NV_ENC_HEVC_PROFILE_MAIN_GUID,
            "hevc",
            params,
        )
    }
}
