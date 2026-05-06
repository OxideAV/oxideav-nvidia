// Field-by-field initialisation of large `#[repr(C)]` structs is
// substantially more readable than the inline-record form clippy
// suggests when most fields stay at the type's default zero value;
// turn the lint off here.
#![allow(clippy::field_reassign_with_default)]

//! NVDEC `Decoder` trait implementations.
//!
//! Round 3 ships an H.264 decoder built on the cuvidParser bitstream
//! layer:
//!
//! 1. `cuvidCreateVideoParser` — bitstream is fed by `cuvidParseVideoData`
//!    which fires three callbacks driven by the parsed frames:
//!    * `pfnSequenceCallback` (once per format change) — we create the
//!      `CUvideodecoder` here using the parser-supplied `CUVIDEOFORMAT`.
//!    * `pfnDecodePicture` (per coded picture) — we forward the
//!      parser-built `CUVIDPICPARAMS` directly to `cuvidDecodePicture`.
//!    * `pfnDisplayPicture` (per displayable picture) — we
//!      `cuvidMapVideoFrame64` the surface, copy host-side, convert
//!      NV12 → planar I420, push to the output queue, and unmap.
//!
//! 2. `cuvidDestroyVideoParser` and `cuvidDestroyDecoder` are called on
//!    `Drop`.
//!
//! The CUDA context created by [`Cuda::create_context_for`] is held by
//! the decoder for its lifetime so callbacks (which run synchronously
//! inside `cuvidParseVideoData`) always have a current context on the
//! caller's thread.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Result, VideoFrame, VideoPlane,
};

use crate::device::{Cuda, CudaContext, CudaDevice, NvError};
use crate::sys::{
    self, CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS,
    CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDSOURCEDATAPACKET, CUDA_SUCCESS,
    CUDA_VIDEO_CHROMA_FORMAT_420, CUDA_VIDEO_CREATE_PREFER_CUVID,
    CUDA_VIDEO_DEINTERLACE_WEAVE, CUDA_VIDEO_SURFACE_FORMAT_NV12, CUVID_PKT_ENDOFSTREAM,
    CUVID_PKT_TIMESTAMP, CUvideodecoder, CUvideoparser, CudaVideoCodec,
};

/// Number of decode surfaces we ask NVDEC to keep around. The parser
/// usually requests 4-20 in `min_num_decode_surfaces`; we round up to
/// at least this many to keep behaviour predictable on small fixtures.
const DEFAULT_NUM_DECODE_SURFACES: u32 = 8;

/// Number of mapped output surfaces the host can hold simultaneously.
const DEFAULT_NUM_OUTPUT_SURFACES: u32 = 2;

// ─────────────────────────── shared callback state ────────────────────────────

/// Mutable state shared between the parser callbacks and the public
/// `Decoder` trait methods.
///
/// Holds a raw NVDEC decoder pointer; we manually mark this `Send +
/// Sync` because the callbacks always run on the same thread that
/// drove `cuvidParseVideoData` (so there is no real cross-thread
/// access of the pointer), but `clippy::arc_with_non_send_sync`
/// catches the technical hole.
struct CallbackState {
    /// Created lazily in the sequence callback, dropped when the
    /// `H264NvDecoder` is destroyed.
    decoder: CUvideodecoder,
    /// Coded width/height reported in the most recent sequence callback.
    coded_width: u32,
    coded_height: u32,
    /// Display rectangle reported by the parser (defaults to coded if
    /// the parser leaves it as zeros).
    display_w: u32,
    display_h: u32,
    display_left: u32,
    display_top: u32,
    /// Frames pulled out of the display callback, ready for
    /// `receive_frame` to drain.
    frames: VecDeque<VideoFrame>,
    /// First error seen inside any callback. Surfaces from the next
    /// `Decoder` method.
    error: Option<String>,
}

// The decoder pointer is opaque and only ever accessed under the
// `Mutex<CallbackState>` guard; manual Send/Sync are required because
// `*mut c_void` is otherwise neither.
unsafe impl Send for CallbackState {}
unsafe impl Sync for CallbackState {}

impl CallbackState {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            decoder: std::ptr::null_mut(),
            coded_width: 0,
            coded_height: 0,
            display_w: 0,
            display_h: 0,
            display_left: 0,
            display_top: 0,
            frames: VecDeque::new(),
            error: None,
        }))
    }
}

// ─────────────────────────── parser callbacks ─────────────────────────────────

/// Sequence callback: fired once per format change. Returns 1 on
/// success (with the requested decode-surface count); 0 on failure.
///
/// The parser passes us the freshly-parsed `CUVIDEOFORMAT`; we use it
/// to build a `CUVIDDECODECREATEINFO` and call `cuvidCreateDecoder`.
unsafe extern "C" fn seq_callback(user_data: *mut c_void, fmt: *mut CUVIDEOFORMAT) -> i32 {
    if user_data.is_null() || fmt.is_null() {
        return 0;
    }
    let state_ptr = user_data as *const Mutex<CallbackState>;
    let state = unsafe { &*state_ptr };
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };

    let vt = match sys::vtable() {
        Ok(v) => v,
        Err(e) => {
            guard.error = Some(format!("seq_callback vtable: {e}"));
            return 0;
        }
    };

    let f = unsafe { &*fmt };
    let coded_w = f.coded_width;
    let coded_h = f.coded_height;
    let mut disp_w = (f.display_right - f.display_left).max(0) as u32;
    let mut disp_h = (f.display_bottom - f.display_top).max(0) as u32;
    if disp_w == 0 {
        disp_w = coded_w;
    }
    if disp_h == 0 {
        disp_h = coded_h;
    }

    // Honour parser request, but never fewer than our default.
    let min_surfaces = f.min_num_decode_surfaces as u32;
    let num_surfaces = min_surfaces.max(DEFAULT_NUM_DECODE_SURFACES);

    // If a decoder already exists we destroy it before re-creating —
    // mid-stream resolution change. For Round 3's single-frame tests
    // this branch never fires.
    if !guard.decoder.is_null() {
        unsafe {
            let _ = (vt.cuvid_destroy_decoder)(guard.decoder);
        }
        guard.decoder = std::ptr::null_mut();
    }

    let mut create = CUVIDDECODECREATEINFO::default();
    create.ul_width = coded_w as u64;
    create.ul_height = coded_h as u64;
    create.ul_num_decode_surfaces = num_surfaces as u64;
    create.codec_type = f.codec;
    create.chroma_format = f.chroma_format;
    create.ul_creation_flags = CUDA_VIDEO_CREATE_PREFER_CUVID as u64;
    create.bit_depth_minus_8 = f.bit_depth_luma_minus8 as u64;
    create.ul_intra_decode_only = 0;
    create.ul_max_width = coded_w as u64;
    create.ul_max_height = coded_h as u64;
    create.display_left = f.display_left as i16;
    create.display_top = f.display_top as i16;
    create.display_right = f.display_right as i16;
    create.display_bottom = f.display_bottom as i16;
    create.output_format = CUDA_VIDEO_SURFACE_FORMAT_NV12;
    create.deinterlace_mode = CUDA_VIDEO_DEINTERLACE_WEAVE;
    create.ul_target_width = coded_w as u64;
    create.ul_target_height = coded_h as u64;
    create.ul_num_output_surfaces = DEFAULT_NUM_OUTPUT_SURFACES as u64;
    create.vid_lock = std::ptr::null_mut();
    // target_rect: 0,0,target_w,target_h
    create.target_left = 0;
    create.target_top = 0;
    create.target_right = coded_w as i16;
    create.target_bottom = coded_h as i16;
    create.enable_histogram = 0;

    let mut dec: CUvideodecoder = std::ptr::null_mut();
    let r = unsafe {
        (vt.cuvid_create_decoder)(&mut dec as *mut _, &mut create as *mut _ as *mut c_void)
    };
    if r != CUDA_SUCCESS {
        guard.error = Some(format!("cuvidCreateDecoder failed: CUresult {r}"));
        return 0;
    }

    guard.decoder = dec;
    guard.coded_width = coded_w;
    guard.coded_height = coded_h;
    guard.display_w = disp_w;
    guard.display_h = disp_h;
    guard.display_left = f.display_left.max(0) as u32;
    guard.display_top = f.display_top.max(0) as u32;

    // Returning 1 accepts the format. Returning a value > 1 would
    // override the dpb size with that value; we set it via
    // ulNumDecodeSurfaces in the create-info instead.
    1
}

/// Decode callback: fired per coded picture. The parser hands us a
/// `CUVIDPICPARAMS` it has already filled with reference-list and
/// slice-data pointers; we forward it as-is to `cuvidDecodePicture`.
unsafe extern "C" fn decode_callback(user_data: *mut c_void, pic: *mut CUVIDPICPARAMS) -> i32 {
    if user_data.is_null() || pic.is_null() {
        return 0;
    }
    let state_ptr = user_data as *const Mutex<CallbackState>;
    let state = unsafe { &*state_ptr };
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };

    let vt = match sys::vtable() {
        Ok(v) => v,
        Err(e) => {
            guard.error = Some(format!("decode_callback vtable: {e}"));
            return 0;
        }
    };

    if guard.decoder.is_null() {
        guard.error = Some("decode_callback: decoder not yet created".into());
        return 0;
    }

    let r = unsafe { (vt.cuvid_decode_picture)(guard.decoder, pic as *mut c_void) };
    if r != CUDA_SUCCESS {
        guard.error = Some(format!("cuvidDecodePicture failed: CUresult {r}"));
        return 0;
    }
    1
}

/// Display callback: fired per displayable frame. Maps the frame,
/// copies it host-side, converts NV12 → planar I420, and pushes a
/// `VideoFrame` onto the queue.
unsafe extern "C" fn display_callback(
    user_data: *mut c_void,
    disp: *mut CUVIDPARSERDISPINFO,
) -> i32 {
    if user_data.is_null() || disp.is_null() {
        return 0;
    }
    let state_ptr = user_data as *const Mutex<CallbackState>;
    let state = unsafe { &*state_ptr };
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };

    let vt = match sys::vtable() {
        Ok(v) => v,
        Err(e) => {
            guard.error = Some(format!("display_callback vtable: {e}"));
            return 0;
        }
    };

    if guard.decoder.is_null() {
        return 0;
    }

    let info = unsafe { &*disp };
    let pic_idx = info.picture_index;

    // ── map the frame ──────────────────────────────────────────────
    let mut proc_params = CUVIDPROCPARAMS::default();
    proc_params.progressive_frame = info.progressive_frame;
    proc_params.top_field_first = info.top_field_first;
    proc_params.unpaired_field = if info.repeat_first_field < 0 { 1 } else { 0 };
    proc_params.second_field = 0;
    proc_params.output_stream = std::ptr::null_mut();

    let mut dev_ptr: u64 = 0;
    let mut pitch: u32 = 0;
    let r = unsafe {
        (vt.cuvid_map_video_frame_64)(
            guard.decoder,
            pic_idx,
            &mut dev_ptr,
            &mut pitch,
            &mut proc_params as *mut _ as *mut c_void,
        )
    };
    if r != CUDA_SUCCESS {
        guard.error = Some(format!("cuvidMapVideoFrame64 failed: CUresult {r}"));
        return 0;
    }

    let pitch = pitch as usize;
    let coded_h = guard.coded_height as usize;
    let coded_w = guard.coded_width as usize;
    let disp_w = guard.display_w as usize;
    let disp_h = guard.display_h as usize;
    let disp_left = guard.display_left as usize;
    let disp_top = guard.display_top as usize;
    let chroma_h = coded_h.div_ceil(2);

    // Wait for decode work to finish before reading host-side. NVDEC
    // returns the surface synchronously enough that this is normally a
    // no-op, but `cuStreamSynchronize(NULL)` is the documented way to
    // make sure the default stream has flushed.
    let r_sync = unsafe { (vt.cu_stream_synchronize)(std::ptr::null_mut()) };
    if r_sync != CUDA_SUCCESS {
        guard.error = Some(format!("cuStreamSynchronize failed: CUresult {r_sync}"));
        unsafe {
            let _ = (vt.cuvid_unmap_video_frame_64)(guard.decoder, dev_ptr);
        }
        return 0;
    }

    // ── copy device → host ─────────────────────────────────────────
    // NV12 layout: pitch * coded_h luma plane, immediately followed by
    // pitch * (coded_h / 2) interleaved UV plane.
    let mut y_pitched = vec![0u8; pitch * coded_h];
    let mut uv_pitched = vec![0u8; pitch * chroma_h];

    let r_y = unsafe {
        (vt.cu_memcpy_dto_h_v2)(y_pitched.as_mut_ptr() as *mut c_void, dev_ptr, pitch * coded_h)
    };
    let r_uv = unsafe {
        (vt.cu_memcpy_dto_h_v2)(
            uv_pitched.as_mut_ptr() as *mut c_void,
            dev_ptr + (pitch * coded_h) as u64,
            pitch * chroma_h,
        )
    };

    // ── unmap before any further work / error returns ──────────────
    unsafe {
        let _ = (vt.cuvid_unmap_video_frame_64)(guard.decoder, dev_ptr);
    }

    if r_y != CUDA_SUCCESS {
        guard.error = Some(format!("cuMemcpyDtoH_v2 (Y) failed: CUresult {r_y}"));
        return 0;
    }
    if r_uv != CUDA_SUCCESS {
        guard.error = Some(format!("cuMemcpyDtoH_v2 (UV) failed: CUresult {r_uv}"));
        return 0;
    }

    // ── crop to display rect, deinterleave NV12 → I420 ─────────────
    // Honour the conformance-window-style display rectangle reported
    // by the parser: NVENC HEVC streams of 320×240 typically code
    // 320×256 with `display_top = 0, display_bottom = 240` (bottom
    // crop), but other producers may use a top crop instead.
    let out_w = disp_w.min(coded_w.saturating_sub(disp_left));
    let out_h = disp_h.min(coded_h.saturating_sub(disp_top));
    let chroma_out_w = out_w.div_ceil(2);
    let chroma_out_h = out_h.div_ceil(2);
    let chroma_left = disp_left / 2;
    let chroma_top = disp_top / 2;

    let mut y_out = vec![0u8; out_w * out_h];
    let mut u_out = vec![0u8; chroma_out_w * chroma_out_h];
    let mut v_out = vec![0u8; chroma_out_w * chroma_out_h];

    for row in 0..out_h {
        let src_row = disp_top + row;
        let src_off = src_row * pitch + disp_left;
        let dst_off = row * out_w;
        if src_off + out_w <= y_pitched.len() {
            y_out[dst_off..dst_off + out_w]
                .copy_from_slice(&y_pitched[src_off..src_off + out_w]);
        }
    }

    for row in 0..chroma_out_h {
        let src_row = chroma_top + row;
        let src_off = src_row * pitch + chroma_left * 2;
        let dst_off = row * chroma_out_w;
        if src_off >= uv_pitched.len() {
            break;
        }
        // The chroma row contains (chroma_out_w * 2) interleaved
        // bytes followed by padding to `pitch`. Slice safely.
        let row_data = &uv_pitched[src_off..(src_off + (chroma_out_w * 2)).min(uv_pitched.len())];
        for col in 0..chroma_out_w {
            let i = col * 2;
            if i + 1 < row_data.len() {
                u_out[dst_off + col] = row_data[i];
                v_out[dst_off + col] = row_data[i + 1];
            }
        }
    }

    // ── push frame ─────────────────────────────────────────────────
    let pts = if info.timestamp != 0 {
        Some(info.timestamp)
    } else {
        None
    };

    guard.frames.push_back(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: out_w,
                data: y_out,
            },
            VideoPlane {
                stride: chroma_out_w,
                data: u_out,
            },
            VideoPlane {
                stride: chroma_out_w,
                data: v_out,
            },
        ],
    });

    1
}

// ─────────────────────────── Generic NVDEC decoder ────────────────────────────

/// NVDEC-backed video decoder, parameterised over codec type.
///
/// Holds a CUDA context (kept current on the caller's thread for the
/// whole lifetime), the cuvidParser, and the lazily-created
/// `CUvideodecoder`. Frame output is pushed by the display callback
/// onto a `Mutex<VecDeque<VideoFrame>>` and drained by `receive_frame`.
///
/// The cuvidParser pipeline is codec-agnostic — only the codec_type and
/// the `bAnnexb` flag change between H.264, HEVC, and AV1. The
/// `CUVIDPICPARAMS` blob the parser fills in is opaque to us; we just
/// forward it to `cuvidDecodePicture`.
pub struct NvDecoder {
    codec_id: CodecId,
    /// CUDA driver init guard. Cheap to copy.
    _cuda: Cuda,
    /// CUDA context kept alive (and current) for the whole decoder life.
    /// Drop order: parser/decoder must be destroyed before the context.
    _ctx: Option<CudaContext>,
    /// cuvidParser handle.
    parser: CUvideoparser,
    /// Shared callback state — lifetime is `Arc` so the FFI side can
    /// dereference it from inside the parser callbacks.
    state: Arc<Mutex<CallbackState>>,
    /// Local output queue drained by `receive_frame`.
    output_queue: VecDeque<VideoFrame>,
    /// Set after `flush()` so the next empty `receive_frame` returns
    /// `Eof` instead of `NeedMore`.
    flushed: bool,
}

unsafe impl Send for NvDecoder {}

impl NvDecoder {
    /// Build an `NvDecoder` for `codec` with the given codec id label.
    /// `is_annex_b` selects between Annex-B / start-code framed input
    /// (H.264 / HEVC) and the codec's native framing (AV1 raw OBUs).
    ///
    /// Honours `params.device_index`: `None` selects ordinal 0 (the
    /// historical default); `Some(i)` selects ordinal `i` and matches
    /// the index in the [`crate::engine::engine_info`] result. An
    /// out-of-range index returns `Error::Unsupported`.
    fn make_for(
        codec: CudaVideoCodec,
        codec_id: &str,
        is_annex_b: bool,
        params: &CodecParameters,
    ) -> Result<Box<dyn oxideav_core::Decoder>> {
        let cuda = Cuda::init().map_err(map_unsupported)?;
        let count = cuda.device_count().map_err(map_unsupported)?;
        if count == 0 {
            return Err(Error::unsupported("nvidia: no CUDA devices visible"));
        }
        let device_index = params.device_index.unwrap_or(0);
        if device_index >= count {
            return Err(Error::unsupported(format!(
                "nvidia: device_index {device_index} out of range (0..{count})"
            )));
        }
        let dev = cuda
            .device(device_index as i32)
            .map_err(map_unsupported)?;
        let ctx = cuda.create_context_for(&dev).map_err(map_unsupported)?;

        let state = CallbackState::new();
        let parser = create_parser(codec, is_annex_b, &state).map_err(|e| match e {
            // Driver-load / dlsym errors → unsupported, anything else
            // (e.g. a real CUresult) → other.
            _ if e.is_unavailable() => Error::unsupported(format!("nvidia: {e}")),
            _ => Error::other(format!("nvidia parser: {e}")),
        })?;

        let _ = dev; // device handle is just an ordinal; nothing to keep alive

        Ok(Box::new(Self {
            codec_id: CodecId::new(codec_id),
            _cuda: cuda,
            _ctx: Some(ctx),
            parser,
            state,
            output_queue: VecDeque::new(),
            flushed: false,
        }))
    }

    fn pull_frames(&mut self) {
        if let Ok(mut g) = self.state.lock() {
            while let Some(f) = g.frames.pop_front() {
                self.output_queue.push_back(f);
            }
        }
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        if let Ok(vt) = sys::vtable() {
            if !self.parser.is_null() {
                unsafe {
                    let _ = (vt.cuvid_destroy_video_parser)(self.parser);
                }
                self.parser = std::ptr::null_mut();
            }
            // Destroy decoder if the sequence callback created one.
            if let Ok(mut g) = self.state.lock() {
                if !g.decoder.is_null() {
                    unsafe {
                        let _ = (vt.cuvid_destroy_decoder)(g.decoder);
                    }
                    g.decoder = std::ptr::null_mut();
                }
            }
        }
        // CudaContext drop runs after this.
    }
}

impl oxideav_core::Decoder for NvDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.flushed = false;

        // Surface any error that fired in a previous callback.
        if let Some(e) = self
            .state
            .lock()
            .ok()
            .and_then(|g| g.error.clone())
        {
            return Err(Error::other(e));
        }

        let vt = sys::vtable().map_err(|e| Error::unsupported(format!("nvidia: {e}")))?;

        // Don't set CUVID_PKT_ENDOFPICTURE here — that flag tells the
        // parser the packet contains *exactly* one picture, which
        // breaks when a single packet carries a whole multi-frame
        // bytestream (encoder output, fixture concatenation). The
        // parser delimits pictures via NAL/OBU framing on its own.
        let mut flags: u64 = 0;
        let timestamp = packet.pts.unwrap_or(0);
        if packet.pts.is_some() {
            flags |= CUVID_PKT_TIMESTAMP as u64;
        }

        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags,
            payload_size: packet.data.len() as u64,
            payload: if packet.data.is_empty() {
                std::ptr::null()
            } else {
                packet.data.as_ptr()
            },
            timestamp,
        };

        let r = unsafe {
            (vt.cuvid_parse_video_data)(self.parser, &mut pkt as *mut _)
        };
        if r != CUDA_SUCCESS {
            return Err(Error::other(format!(
                "cuvidParseVideoData failed: CUresult {r}"
            )));
        }

        // Surface any error queued up by callbacks during the parse.
        if let Some(e) = self
            .state
            .lock()
            .ok()
            .and_then(|g| g.error.clone())
        {
            return Err(Error::other(e));
        }

        self.pull_frames();
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pull_frames();
        if let Some(f) = self.output_queue.pop_front() {
            return Ok(Frame::Video(f));
        }
        Err(if self.flushed {
            Error::Eof
        } else {
            Error::NeedMore
        })
    }

    fn flush(&mut self) -> Result<()> {
        // Send an end-of-stream packet so the parser drains its
        // display queue.
        if let Ok(vt) = sys::vtable() {
            let mut pkt = CUVIDSOURCEDATAPACKET {
                flags: CUVID_PKT_ENDOFSTREAM as u64,
                payload_size: 0,
                payload: std::ptr::null(),
                timestamp: 0,
            };
            unsafe {
                let _ = (vt.cuvid_parse_video_data)(self.parser, &mut pkt as *mut _);
            }
        }
        self.pull_frames();
        self.flushed = true;
        Ok(())
    }
}

// ─────────────────────────── helpers ──────────────────────────────────────────

/// Build a cuvidParser configured for `codec` with our three callbacks
/// wired to `state`.
///
/// `is_annex_b`: set to `true` for H.264 / HEVC streams in Annex-B
/// (start-code) framing. AV1 raw OBU streams use their own framing —
/// the parser figures it out, so we leave the flag at 0.
fn create_parser(
    codec: CudaVideoCodec,
    is_annex_b: bool,
    state: &Arc<Mutex<CallbackState>>,
) -> std::result::Result<CUvideoparser, NvError> {
    let vt = sys::vtable().map_err(NvError::from_str)?;

    let mut params = CUVIDPARSERPARAMS::default();
    params.codec_type = codec as i32;
    params.ul_max_num_decode_surfaces = DEFAULT_NUM_DECODE_SURFACES;
    params.ul_clock_rate = 0; // default 10 MHz
    params.ul_error_threshold = 0;
    params.ul_max_display_delay = 0;
    // bAnnexb : 1 occupies the lowest bit of the packed flags word.
    params.flags = if is_annex_b { 1 } else { 0 };
    params.user_data = Arc::as_ptr(state) as *mut c_void;
    params.pfn_sequence_callback = Some(seq_callback);
    params.pfn_decode_picture = Some(decode_callback);
    params.pfn_display_picture = Some(display_callback);
    params.pfn_get_operating_point = None;
    params.pfn_get_sei_msg = None;
    params.p_ext_video_info = std::ptr::null_mut();

    let mut parser: CUvideoparser = std::ptr::null_mut();
    let r = unsafe { (vt.cuvid_create_video_parser)(&mut parser, &mut params) };
    if r != CUDA_SUCCESS {
        return Err(NvError::from_cu(Some(vt), r));
    }
    Ok(parser)
}

// ─────────────────────────── public codec wrappers ────────────────────────────

/// NVDEC-backed H.264 decoder — thin wrapper over [`NvDecoder`].
pub struct H264NvDecoder;

impl H264NvDecoder {
    /// Standard codec-registry factory. Maps any
    /// driver-unavailable / no-device condition to `Error::Unsupported`
    /// so the registry falls back to a software decoder.
    ///
    /// Honours `params.device_index` for CUDA device selection — see
    /// [`NvDecoder::make_for`] for details.
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        NvDecoder::make_for(CudaVideoCodec::H264, "h264", true, params)
    }
}

/// NVDEC-backed HEVC (H.265) decoder.
pub struct HevcNvDecoder;

impl HevcNvDecoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        NvDecoder::make_for(CudaVideoCodec::Hevc, "hevc", true, params)
    }
}

/// NVDEC-backed AV1 decoder.
///
/// Expects a raw OBU bitstream — the cuvidParser parses AV1 OBUs
/// natively, so `bAnnexb` is left at 0.
pub struct Av1NvDecoder;

impl Av1NvDecoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        NvDecoder::make_for(CudaVideoCodec::Av1, "av1", false, params)
    }
}

/// Map every `NvError` from initialisation into `Error::Unsupported`
/// so a missing driver / no GPU host falls back to the pure-Rust
/// decoder rather than panicking.
fn map_unsupported(e: NvError) -> Error {
    Error::unsupported(format!("nvidia: {e}"))
}

// Suppress "unused" warnings on chroma constant — referenced via the
// `CudaVideoCodec::H264` decision path.
#[allow(dead_code)]
const _CHROMA_420_PIN: u32 = CUDA_VIDEO_CHROMA_FORMAT_420;
// Same for CudaDevice — used only inside `make`, but referenced here
// to keep the import in lib.rs unambiguous.
#[allow(dead_code)]
fn _device_marker(_d: &CudaDevice) {}
