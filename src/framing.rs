//! Container (length-prefixed) → Annex-B re-framing for the NVDEC
//! H.264 / HEVC parsers.
//!
//! The cuvidParser is created with `bAnnexb = 1`: it scans its input
//! for start codes and expects the parameter sets in-band. Streams
//! demuxed from ISO-BMFF (MP4 / MOV / HEIF), Matroska, FLV and friends
//! carry the other framing instead:
//!
//! * the parameter sets live out of band in
//!   `CodecParameters::extradata`, as an `AVCDecoderConfigurationRecord`
//!   (`avcC`, ISO/IEC 14496-15 §5.3.3.1) or an
//!   `HEVCDecoderConfigurationRecord` (`hvcC`, §8.3.3.1);
//! * every NAL unit in a sample is preceded by a big-endian length of
//!   `lengthSizeMinusOne + 1` bytes instead of a start code.
//!
//! Handing such packets to the parser verbatim decodes nothing (no
//! start code is ever found, no SPS ever seen). [`NalFraming`] parses
//! the configuration record once at construction, prepends its
//! parameter sets (start-code framed) to the first packet and rewrites
//! every length prefix to a 4-byte start code. Empty extradata, or
//! extradata that already is an Annex-B byte stream, leaves packets
//! untouched.

/// Which configuration-record layout `extradata` uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConfigKind {
    /// `AVCDecoderConfigurationRecord` (H.264).
    Avc,
    /// `HEVCDecoderConfigurationRecord` (H.265).
    Hevc,
}

const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Per-decoder re-framing state.
#[derive(Clone, Debug, Default)]
pub(crate) struct NalFraming {
    /// NAL length-prefix width (1, 2 or 4) when packets are
    /// length-prefixed; `None` = packets already are Annex-B.
    length_size: Option<usize>,
    /// Start-code framed parameter sets from the configuration record,
    /// emitted ahead of the first packet (then cleared).
    prologue: Vec<u8>,
}

impl NalFraming {
    /// Build the re-framer from `extradata`. Anything that is not a
    /// well-formed configuration record (empty, Annex-B parameter
    /// sets, truncated) selects pass-through: the stream is then
    /// assumed to be Annex-B, which is what the parser was given
    /// before this module existed.
    pub(crate) fn from_extradata(kind: ConfigKind, extradata: &[u8]) -> Self {
        if extradata.is_empty() || starts_with_start_code(extradata) {
            // Annex-B parameter sets carried as extradata: feed them
            // ahead of the stream as-is.
            return Self {
                length_size: None,
                prologue: extradata.to_vec(),
            };
        }
        let parsed = match kind {
            ConfigKind::Avc => parse_avcc(extradata),
            ConfigKind::Hevc => parse_hvcc(extradata),
        };
        match parsed {
            Some((length_size, nals)) => {
                let mut prologue = Vec::new();
                for nal in nals {
                    prologue.extend_from_slice(&START_CODE);
                    prologue.extend_from_slice(nal);
                }
                Self {
                    length_size: Some(length_size),
                    prologue,
                }
            }
            None => Self::default(),
        }
    }

    /// Re-frame one packet payload into the Annex-B bytes the parser
    /// consumes. A payload whose length prefixes do not tile it
    /// exactly is passed through unchanged (some muxers write Annex-B
    /// samples despite an `avcC` record).
    pub(crate) fn reframe(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = std::mem::take(&mut self.prologue);
        match self.length_size {
            Some(n) => match length_prefixed_to_annex_b(data, n) {
                Some(annex_b) => out.extend_from_slice(&annex_b),
                None => out.extend_from_slice(data),
            },
            None => out.extend_from_slice(data),
        }
        out
    }

    /// `true` when neither a prologue nor a length-prefix rewrite is
    /// pending — packets can then go to the parser by reference.
    pub(crate) fn is_passthrough(&self) -> bool {
        self.length_size.is_none() && self.prologue.is_empty()
    }
}

fn starts_with_start_code(b: &[u8]) -> bool {
    b.starts_with(&[0, 0, 1]) || b.starts_with(&[0, 0, 0, 1])
}

fn be(b: &[u8]) -> usize {
    b.iter().fold(0usize, |acc, &x| (acc << 8) | x as usize)
}

/// Read `count` `u16`-length-prefixed NAL units starting at `*pos`.
fn read_units<'a>(
    rec: &'a [u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<&'a [u8]>,
) -> Option<()> {
    for _ in 0..count {
        let len = be(rec.get(*pos..*pos + 2)?);
        *pos += 2;
        let nal = rec.get(*pos..*pos + len)?;
        *pos += len;
        if !nal.is_empty() {
            out.push(nal);
        }
    }
    Some(())
}

/// ISO/IEC 14496-15 §5.3.3.1: `configurationVersion` (= 1),
/// profile / compatibility / level, `lengthSizeMinusOne` (low 2 bits of
/// byte 4), `numOfSequenceParameterSets` (low 5 bits of byte 5) SPS
/// units, `numOfPictureParameterSets` PPS units. The trailing
/// high-profile fields (chroma format, bit depths, SPS extensions) are
/// redundant with the SPS and ignored.
fn parse_avcc(rec: &[u8]) -> Option<(usize, Vec<&[u8]>)> {
    if rec.len() < 7 || rec[0] != 1 {
        return None;
    }
    let length_size = (rec[4] & 0x03) as usize + 1;
    if length_size == 3 {
        return None;
    }
    let mut nals = Vec::new();
    let mut pos = 6;
    read_units(rec, &mut pos, (rec[5] & 0x1f) as usize, &mut nals)?;
    let num_pps = *rec.get(pos)? as usize;
    pos += 1;
    read_units(rec, &mut pos, num_pps, &mut nals)?;
    Some((length_size, nals))
}

/// ISO/IEC 14496-15 §8.3.3.1: a 22-byte fixed header whose last byte
/// carries `lengthSizeMinusOne` in its low 2 bits, then `numOfArrays`
/// arrays of (`NAL_unit_type` byte, `u16` `numNalus`, units).
fn parse_hvcc(rec: &[u8]) -> Option<(usize, Vec<&[u8]>)> {
    if rec.len() < 23 || rec[0] != 1 {
        return None;
    }
    let length_size = (rec[21] & 0x03) as usize + 1;
    if length_size == 3 {
        return None;
    }
    let mut nals = Vec::new();
    let mut pos = 23;
    for _ in 0..rec[22] {
        let count = be(rec.get(pos + 1..pos + 3)?);
        pos += 3;
        read_units(rec, &mut pos, count, &mut nals)?;
    }
    Some((length_size, nals))
}

/// Rewrite every `length_size`-byte length prefix to a 4-byte start
/// code. `None` when the prefixes do not tile `data` exactly.
fn length_prefixed_to_annex_b(data: &[u8], length_size: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() + 16);
    let mut pos = 0;
    while pos < data.len() {
        let len = be(data.get(pos..pos + length_size)?);
        pos += length_size;
        let nal = data.get(pos..pos.checked_add(len)?)?;
        pos += len;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPS: &[u8] = &[0x67, 0x42, 0xc0, 0x1e];
    const PPS: &[u8] = &[0x68, 0xce, 0x3c, 0x80];

    fn avcc(length_size_minus_one: u8) -> Vec<u8> {
        let mut r = vec![1, 0x42, 0xc0, 0x1e, 0xfc | length_size_minus_one, 0xe1];
        r.extend_from_slice(&(SPS.len() as u16).to_be_bytes());
        r.extend_from_slice(SPS);
        r.push(1);
        r.extend_from_slice(&(PPS.len() as u16).to_be_bytes());
        r.extend_from_slice(PPS);
        r
    }

    #[test]
    fn avcc_prologue_then_length_prefixes_become_start_codes() {
        let mut f = NalFraming::from_extradata(ConfigKind::Avc, &avcc(3));
        assert!(!f.is_passthrough());
        let pkt = [0, 0, 0, 2, 0x65, 0x88, 0, 0, 0, 1, 0x06];
        let first = f.reframe(&pkt);
        let mut want = vec![0, 0, 0, 1];
        want.extend_from_slice(SPS);
        want.extend_from_slice(&[0, 0, 0, 1]);
        want.extend_from_slice(PPS);
        want.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1, 0x06]);
        assert_eq!(first, want);
        // The prologue is emitted once.
        assert_eq!(f.reframe(&pkt), &want[want.len() - 11..]);
    }

    #[test]
    fn two_byte_length_prefixes() {
        let mut f = NalFraming::from_extradata(ConfigKind::Avc, &avcc(1));
        let _ = f.reframe(&[]);
        assert_eq!(f.reframe(&[0, 1, 0x41]), [0, 0, 0, 1, 0x41]);
    }

    #[test]
    fn untiled_payload_passes_through() {
        let mut f = NalFraming::from_extradata(ConfigKind::Avc, &avcc(3));
        let _ = f.reframe(&[]);
        let annex_b = [0, 0, 0, 1, 0x65, 0x88, 0x84];
        assert_eq!(f.reframe(&annex_b), annex_b);
    }

    #[test]
    fn empty_or_annex_b_extradata_is_passthrough() {
        assert!(NalFraming::from_extradata(ConfigKind::Avc, &[]).is_passthrough());
        let mut f = NalFraming::from_extradata(ConfigKind::Avc, &[0, 0, 0, 1, 0x67]);
        assert_eq!(
            f.reframe(&[0, 0, 1, 0x65]),
            [0, 0, 0, 1, 0x67, 0, 0, 1, 0x65]
        );
        assert!(f.is_passthrough());
    }

    #[test]
    fn hvcc_arrays_feed_the_prologue() {
        let mut r = vec![1u8; 21];
        r.push(0xf0 | 3); // lengthSizeMinusOne = 3
        r.push(2); // numOfArrays
        for (ty, nal) in [(32u8, &[0x40u8, 0x01][..]), (33, &[0x42, 0x01, 0xaa][..])] {
            r.push(ty);
            r.extend_from_slice(&1u16.to_be_bytes());
            r.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            r.extend_from_slice(nal);
        }
        let mut f = NalFraming::from_extradata(ConfigKind::Hevc, &r);
        let out = f.reframe(&[0, 0, 0, 2, 0x26, 0x01]);
        assert_eq!(
            out,
            [0, 0, 0, 1, 0x40, 0x01, 0, 0, 0, 1, 0x42, 0x01, 0xaa, 0, 0, 0, 1, 0x26, 0x01]
        );
    }

    #[test]
    fn malformed_records_fall_back_to_passthrough() {
        assert!(NalFraming::from_extradata(ConfigKind::Avc, &[1, 2, 3]).is_passthrough());
        assert!(NalFraming::from_extradata(ConfigKind::Hevc, &[1; 10]).is_passthrough());
        let mut bad = avcc(3);
        bad.truncate(bad.len() - 2);
        assert!(NalFraming::from_extradata(ConfigKind::Avc, &bad).is_passthrough());
    }
}
