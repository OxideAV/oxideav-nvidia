//! Safe wrapper around `cuvidGetDecoderCaps`.
//!
//! NVDEC's capability query (`cuvidGetDecoderCaps`) takes an in/out
//! `CUVIDDECODECAPS` struct: the caller fills in `eCodecType`,
//! `eChromaFormat`, and `nBitDepthMinus8`, and the driver returns
//! `bIsSupported`, `nMaxWidth`, `nMaxHeight`, etc.
//!
//! The query requires a *current* CUDA context, so callers typically:
//!
//! ```ignore
//! let cuda = Cuda::init()?;
//! let dev = cuda.device(0)?;
//! let _ctx = cuda.create_context_for(&dev)?;
//! let caps = nvdec_caps(CudaVideoCodec::H264, CUDA_VIDEO_CHROMA_FORMAT_420, 8)?;
//! ```

use crate::device::NvError;
use crate::sys::{self, CudaVideoCodec, CUVIDDECODECAPS, CUDA_SUCCESS};

/// Public-facing snapshot of an NVDEC capability query result.
///
/// Mirrors the *out* fields of `CUVIDDECODECAPS` plus the inputs we
/// echoed back, with the driver's reserved fields stripped.
#[derive(Debug, Clone, Copy)]
pub struct NvdecCaps {
    /// Echo of the queried codec.
    pub codec: i32,
    /// Echo of the queried chroma format.
    pub chroma_format: i32,
    /// Echo of the queried bit depth (8, 10, 12).
    pub bit_depth: u32,

    /// 1 if the codec/chroma/bit-depth combo is hardware-supported.
    pub is_supported: u8,
    /// Number of NVDEC engines that can handle this combo.
    pub num_nvdecs: u8,
    /// Bitmask of `cudaVideoSurfaceFormat` enums supported as output.
    pub output_format_mask: u16,
    /// Maximum coded width in pixels (0 if unsupported).
    pub max_width: u32,
    /// Maximum coded height in pixels (0 if unsupported).
    pub max_height: u32,
    /// Maximum macroblock count
    /// (`coded_width * coded_height / 256` must be `<= max_mb_count`).
    pub max_mb_count: u32,
    /// Minimum coded width in pixels.
    pub min_width: u16,
    /// Minimum coded height in pixels.
    pub min_height: u16,
    /// 1 if Y-component histogram output is supported on this combo.
    pub is_histogram_supported: u8,
    /// Histogram counter bit depth (only meaningful if histograms supported).
    pub counter_bit_depth: u8,
    /// Maximum number of histogram bins.
    pub max_histogram_bins: u16,
}

/// Query `cuvidGetDecoderCaps` for the given (codec, chroma, bit-depth)
/// triple.
///
/// Requires a CUDA context to be *current* on the calling thread —
/// usually established via [`crate::device::Cuda::create_context_for`]
/// before calling this function.
pub fn nvdec_caps(
    codec: CudaVideoCodec,
    chroma_format: u32,
    bit_depth: u32,
) -> Result<NvdecCaps, NvError> {
    let vt = sys::vtable().map_err(NvError::from_str)?;

    let mut caps = CUVIDDECODECAPS {
        e_codec_type: codec as i32,
        e_chroma_format: chroma_format as i32,
        n_bit_depth_minus_8: bit_depth.saturating_sub(8),
        ..CUVIDDECODECAPS::default()
    };

    // SAFETY: `caps` is a properly sized + zero-init struct matching
    // the layout in `<cuviddec.h>`. The function fills out fields and
    // returns CUDA_SUCCESS on success.
    let r = unsafe { (vt.cuvid_get_decoder_caps)(&mut caps as *mut _) };
    if r != CUDA_SUCCESS {
        return Err(NvError::from_cu(Some(vt), r));
    }

    Ok(NvdecCaps {
        codec: caps.e_codec_type,
        chroma_format: caps.e_chroma_format,
        bit_depth: caps.n_bit_depth_minus_8 + 8,
        is_supported: caps.b_is_supported,
        num_nvdecs: caps.n_num_nvdecs,
        output_format_mask: caps.n_output_format_mask,
        max_width: caps.n_max_width,
        max_height: caps.n_max_height,
        max_mb_count: caps.n_max_mb_count,
        min_width: caps.n_min_width,
        min_height: caps.n_min_height,
        is_histogram_supported: caps.b_is_histogram_supported,
        counter_bit_depth: caps.n_counter_bit_depth,
        max_histogram_bins: caps.n_max_histogram_bins,
    })
}
