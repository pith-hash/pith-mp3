//! Layers I and II: subband coders without spectral transform.
//!
//! Frame layout after the 4-byte header (ISO/IEC 11172-3 §2.4.1/§2.4.2):
//! an optional 16-bit CRC, then allocation (4 bits per subband for I,
//! `nbal` bits from the selected B.2 table for II), the scfsi section
//! (Layer II only), scalefactors (6 bits each), and finally the coded
//! samples — 12 samples per subband for Layer I, 3×12 grouped for
//! Layer II.
//!
//! CRC coverage per spec §2.4.1.3/§2.4.2.4 and confirmed bit-for-bit by
//! libmad's `mad_layer_I`/`mad_layer_II`: header bytes 2..4 plus the
//! allocation for Layer I; header bytes 2..4 plus allocation *and* scfsi
//! for Layer II. The Layer II protected span ends mid-byte, which is why
//! [`crate::crc::Crc16`] accumulates bit-wise.
//!
//! Section ordering is `subband` outermost, `channel` innermost in both
//! allocation and scfsi; for the scalefactor and sample sections the
//! subband loop is outermost too (spec syntax tables; mpg123 and libmad
//! agree — an earlier draft of this module walked channel-outermost and
//! mis-parsed every stereo frame).
//!
//! The dequantizer normalizes to `[-1, 1)` at the midpoint: `q` coded with
//! `b` bits becomes `(2q + 1 - 2^b) / (2^b - 1)` for both layers, then
//! multiplied by `SF_TABLE[scalefactor]`. Layer II grouped codes unpack
//! least-significant first: `c mod n`, `c/n mod n`, `c/n²`.
//!
//! Joint-stereo layout (both layers): subbands below the mode-extension
//! bound carry per-channel everything; subbands at and above it share one
//! allocation and one sample stream but still carry per-channel
//! scalefactors — the shared code is scaled independently per channel.

use crate::crc::Crc16;
use crate::header::{Header, Layer, Mode};
use crate::tables::{ALLOC_TABLES, QUANT_BITS, QUANT_STEPS, SF_TABLE};
use pith_digest::{BitReader, Error, Result};

/// `sblimit` for each compact allocation table (B.2a..d).
const L2_SBLIMIT: [usize; 4] = [27, 30, 8, 12];

/// Pick the Layer II allocation table — the formula shared by libmad's
/// `mad_layer_II` and ffmpeg's `ff_mpa_l2_select_table` (ISO §2.4.1.6
/// prescribes the same split by bitrate-per-channel and sampling rate).
fn l2_table_index(h: &Header) -> usize {
    let per_ch = h.bitrate_kbps / h.channels() as u32;
    // ffmpeg's `ff_mpa_l2_select_table` verbatim (MPEG-1 branch):
    // table a for high rate / any 48kHz, b for >=96kbit/ch elsewhere,
    // c for <=48kbit/ch except 32kHz, d otherwise.
    if (h.sample_rate == 48000 && per_ch >= 56) || (56..=80).contains(&per_ch) {
        0
    } else if h.sample_rate != 48000 && per_ch >= 96 {
        1
    } else if h.sample_rate != 32000 && per_ch <= 48 {
        2
    } else {
        3
    }
}

/// Non-grouped dequantizer shared by both layers — ISO/IEC 11172-3
/// §2.4.3.2 `s" = (2^nb/(2^nb-1))·(s'" + 2^(-nb+1))` with `s'"` the
/// offset-binary code shifted to two's-complement; in raw-code terms
/// `s" = (2q - 2^nb + 2)/(2^nb - 1)` (libmad I_sample, ffmpeg
/// l1_unscale and mpg123's `(-1<<n) + sample + 1` all agree; the
/// asymmetric +2 is spec-exact, not the midpoint's +1).
fn unquant(code: u32, bits: u32) -> f32 {
    let levels = (1u32 << bits) as f32;
    (2.0 * code as f32 + 2.0 - levels) / (levels - 1.0)
}

/// Layer II grouped quantizer midpoint formula (Table 3-B.4 `C/D`).
fn l2_group_dequant(c: u32, steps: u32) -> f32 {
    (2.0 * c as f32 + 1.0) / steps as f32 - 1.0
}

#[allow(clippy::needless_range_loop)] // bit-field indices, not data iteration
/// Decode the requantized subband samples of one Layer I or II frame into
/// `out[ch][subband][sample]` — 12 samples per subband for L1, 36 for L2.
///
/// `frame` starts at the sync word: the CRC covers header bytes 2..4, so
/// the parse needs the header as well as the body.
pub(crate) fn decode_subbands(h: &Header, frame: &[u8]) -> Result<[[[f32; 36]; 32]; 2]> {
    let nch = h.channels();
    let joint = h.mode == Mode::JointStereo;
    let body = &frame[4..];
    let mut r = BitReader::new(body);
    let mut out = [[[0.0f32; 36]; 32]; 2];
    let mut crc = Crc16::new();
    crc.bytes(&frame[2..4]);

    // CRC word: occupies the first two body bytes when protection is on.
    // It covers header bytes 2..4 plus the allocation (plus scfsi for
    // Layer II) — i.e. data that physically FOLLOWS the field.
    let crc_target = if h.unprotected {
        None
    } else {
        Some(r.bits(16)? as u16)
    };

    // ---- allocation
    let mut alloc = [[0u8; 32]; 2];
    let mut qbits = [[0i8; 32]; 2]; // L1 coded bits per sample
    let mut qidx = [[0u8; 32]; 2]; // L2 quantization class index
    let sblimit;
    let bound; // first jointly-coded subband (== sblimit when not joint)
    if h.layer == Layer::I {
        sblimit = 32;
        bound = if joint { h.js_bound(32) } else { 32 };
        for sb in 0..bound {
            for ch in 0..nch {
                let a = r.bits(4)? as u8;
                if a == 15 {
                    return Err(Error::BadValue("layer I allocation"));
                }
                crc.field(u64::from(a), 4);
                alloc[ch][sb] = a;
                qbits[ch][sb] = if a > 0 { a as i8 + 1 } else { 0 };
            }
        }
        for sb in bound..32 {
            let a = r.bits(4)? as u8;
            if a == 15 {
                return Err(Error::BadValue("layer I allocation"));
            }
            crc.field(u64::from(a), 4);
            alloc[0][sb] = a;
            alloc[1][sb] = a;
            let b = if a > 0 { a as i8 + 1 } else { 0 };
            qbits[0][sb] = b;
            qbits[1][sb] = b;
        }
    } else {
        let t = l2_table_index(h);
        sblimit = L2_SBLIMIT[t];
        bound = if joint { h.js_bound(sblimit) } else { sblimit };
        let tab = ALLOC_TABLES[t];
        let mut j = 0usize;
        for sb in 0..bound {
            let nbal = tab[j] as usize;
            for ch in 0..nch {
                let code = r.bits(nbal)? as usize;
                crc.field(code as u64, nbal);
                alloc[ch][sb] = code as u8;
                if code > 0 {
                    qidx[ch][sb] = tab[j + code];
                }
            }
            j += 1 << nbal;
        }
        for sb in bound..sblimit {
            let nbal = tab[j] as usize;
            let code = r.bits(nbal)? as usize;
            crc.field(code as u64, nbal);
            alloc[0][sb] = code as u8;
            alloc[1][sb] = code as u8;
            if code > 0 {
                qidx[0][sb] = tab[j + code];
                qidx[1][sb] = tab[j + code];
            }
            j += 1 << nbal;
        }
    }

    // ---- Layer II scfsi: 2 bits per allocated subband, subband-outer /
    // channel-inner, and covered by the CRC.
    let mut scfsi = [[0u8; 32]; 2];
    if h.layer == Layer::II {
        for sb in 0..sblimit {
            for ch in 0..nch {
                if alloc[ch][sb] != 0 {
                    let v = r.bits(2)? as u8;
                    crc.field(u64::from(v), 2);
                    scfsi[ch][sb] = v;
                }
            }
        }
    }

    if let Some(target) = crc_target {
        if crc.finish() != target {
            return Err(Error::BadValue("crc"));
        }
    }

    // ---- scalefactors: L1 keeps one per subband; L2 keeps three per
    // subband gated by scfsi. Read order: subband outer, channel inner,
    // and each channel's factor is fully read before the next subband
    // even in the joint region (shared alloc, per-channel scalefactors).
    let mut sf3 = [[[0u8; 3]; 32]; 2];
    if h.layer == Layer::II {
        for sb in 0..sblimit {
            for ch in 0..nch {
                if alloc[ch][sb] == 0 {
                    continue;
                }
                match scfsi[ch][sb] {
                    0 => {
                        for p in &mut sf3[ch][sb] {
                            *p = r.bits(6)? as u8;
                        }
                    }
                    1 => {
                        sf3[ch][sb][0] = r.bits(6)? as u8;
                        sf3[ch][sb][2] = r.bits(6)? as u8;
                        sf3[ch][sb][1] = sf3[ch][sb][0];
                    }
                    2 => {
                        let s = r.bits(6)? as u8;
                        sf3[ch][sb] = [s, s, s];
                    }
                    _ => {
                        sf3[ch][sb][0] = r.bits(6)? as u8;
                        sf3[ch][sb][2] = r.bits(6)? as u8;
                        sf3[ch][sb][1] = sf3[ch][sb][2];
                    }
                }
            }
        }
    } else {
        for sb in 0..32 {
            for ch in 0..nch {
                if alloc[ch][sb] != 0 {
                    let s = r.bits(6)? as u8;
                    if s == 63 {
                        return Err(Error::BadValue("scalefactor"));
                    }
                    sf3[ch][sb] = [s, s, s];
                }
            }
        }
    }

    // ---- samples
    if h.layer == Layer::I {
        // 12 sample groups; within a group the subband loop is outermost.
        for s in 0..12 {
            for sb in 0..bound {
                for ch in 0..nch {
                    let b = qbits[ch][sb];
                    if b <= 0 {
                        continue;
                    }
                    let q = r.bits(b as usize)? as u32;
                    out[ch][sb][s] = unquant(q, b as u32) * SF_TABLE[sf3[ch][sb][0] as usize];
                }
            }
            for sb in bound..32 {
                let b = qbits[0][sb];
                if b <= 0 {
                    continue;
                }
                let q = r.bits(b as usize)? as u32;
                let f = unquant(q, b as u32);
                for ch in 0..nch {
                    out[ch][sb][s] = f * SF_TABLE[sf3[ch][sb][0] as usize];
                }
            }
        }
    } else {
        // 12 granules of 3 samples; granules 4p..4p+3 use scalefactor
        // part p.
        for part in 0..3usize {
            for g in 0..4usize {
                let base = part * 12 + g * 3;
                for sb in 0..bound {
                    for ch in 0..nch {
                        if alloc[ch][sb] == 0 {
                            continue;
                        }
                        read_three(&mut r, qidx[ch][sb], |k, f| {
                            out[ch][sb][base + k] = f * SF_TABLE[sf3[ch][sb][part] as usize];
                        })?;
                    }
                }
                for sb in bound..sblimit {
                    if alloc[0][sb] == 0 {
                        continue;
                    }
                    read_three(&mut r, qidx[0][sb], |k, f| {
                        for ch in 0..nch {
                            out[ch][sb][base + k] = f * SF_TABLE[sf3[ch][sb][part] as usize];
                        }
                    })?;
                }
            }
        }
    }

    Ok(out)
}

/// Read the three samples of one Layer II granule and hand their
/// normalized values to `f(k, value)` in order.
fn read_three(r: &mut BitReader<'_>, qidx: u8, mut f: impl FnMut(usize, f32)) -> Result<()> {
    let qi = qidx as usize;
    let bits = QUANT_BITS[qi];
    if bits < 0 {
        let n = QUANT_STEPS[qi];
        let c = r.bits((-bits) as usize)? as u32;
        f(0, l2_group_dequant(c % n, n));
        f(1, l2_group_dequant((c / n) % n, n));
        f(2, l2_group_dequant(c / (n * n), n));
    } else {
        for k in 0..3 {
            let q = r.bits(bits as usize)? as u32;
            f(k, unquant(q, bits as u32));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;

    /// A 32-bit MPEG-1 audio header word: sync 11 bits, version `11`,
    /// then layer / protection / bri / sri / pad / mode / mode_ext.
    fn header_word(
        layer: u32,
        bri: u32,
        sri: u32,
        mode: u32,
        mode_ext: u32,
        unprotected: bool,
    ) -> u32 {
        0xFFE0_0000
            | (0b11 << 19)
            | (layer << 17)
            | (u32::from(unprotected) << 16)
            | (bri << 12)
            | (sri << 10)
            | (mode << 6)
            | (mode_ext << 4)
    }

    fn header_of(w: u32) -> Header {
        Header::parse(&w.to_be_bytes()).expect("crafted header word must parse")
    }

    /// MSB-first bit writer for hand-built frames (the write-order twin of
    /// the decoder's `BitReader`).
    struct BitWriter {
        out: Vec<u8>,
        acc: u8,
        n: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter {
                out: Vec::new(),
                acc: 0,
                n: 0,
            }
        }
        fn put(&mut self, v: u32, bits: usize) {
            // Values wider than 32 bits only ever pad with zeros here.
            let head = bits.saturating_sub(32);
            for _ in 0..head {
                self.raw(0);
            }
            for i in (0..bits.saturating_sub(head)).rev() {
                self.raw((v >> i) & 1);
            }
        }
        fn raw(&mut self, b: u32) {
            self.acc = (self.acc << 1) | b as u8;
            self.n += 1;
            if self.n == 8 {
                self.out.push(self.acc);
                self.acc = 0;
                self.n = 0;
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.out.push(self.acc << (8 - self.n));
            }
            self.out
        }
    }

    /// Test-side mirror of `Crc16` so a crafted protected frame can carry
    /// the checksum the decoder will recompute.
    struct FrameCrc(u16);

    impl FrameCrc {
        fn new() -> Self {
            FrameCrc(0xFFFF)
        }
        fn bit(&mut self, b: bool) {
            let fb = ((self.0 >> 15) as u8 ^ u8::from(b)) & 1;
            self.0 <<= 1;
            if fb != 0 {
                self.0 ^= 0x8005;
            }
        }
        fn byte(&mut self, b: u8) {
            for i in (0..8).rev() {
                self.bit((b >> i) & 1 != 0);
            }
        }
        fn field(&mut self, v: u32, bits: usize) {
            for i in (0..bits).rev() {
                self.bit((v >> i) & 1 != 0);
            }
        }
    }

    #[test]
    fn l2_table_selection_matches_the_ffmpeg_split() {
        // arm 0: 48 kHz at any rate, or 56..=80 kbit/s per channel.
        assert_eq!(
            l2_table_index(&header_of(header_word(0b10, 8, 1, 0, 0, true))),
            0
        );
        assert_eq!(
            l2_table_index(&header_of(header_word(0b10, 9, 0, 0, 0, true))),
            0
        );
        // arm 1: >= 96 kbit/s per channel away from 48 kHz.
        assert_eq!(
            l2_table_index(&header_of(header_word(0b10, 13, 0, 0, 0, true))),
            1
        );
        // arm 2: <= 48 kbit/s per channel away from 32 kHz and 48 kHz.
        assert_eq!(
            l2_table_index(&header_of(header_word(0b10, 5, 0, 0, 0, true))),
            2
        );
        // arm 3: everything else (32 kHz at low per-channel rate).
        assert_eq!(
            l2_table_index(&header_of(header_word(0b10, 5, 2, 0, 0, true))),
            3
        );
    }

    #[test]
    fn layer1_joint_stereo_shares_upper_subbands() {
        let w = header_word(0b11, 12, 0, 1, 1, true);
        let h = header_of(w);
        let bound = h.js_bound(32);
        assert_eq!(bound, 8);
        // Per-channel factors below the bound; shared subbands carry one
        // stream but per-channel factors, so give both channels the same
        // factor there to keep the shared equality check meaningful.
        let sf = |sb: usize, ch: usize| {
            if sb >= 8 {
                ((sb * 5) % 60 + 1) as u32
            } else {
                ((sb * 2 + ch * 7) % 60 + 1) as u32
            }
        };
        let mut bw = BitWriter::new();
        for _ in 0..bound {
            bw.put(1, 4);
            bw.put(1, 4);
        }
        for _ in bound..32 {
            bw.put(1, 4); // one shared allocation
        }
        for sb in 0..32 {
            for ch in 0..2 {
                bw.put(sf(sb, ch), 6); // per-channel factors even when shared
            }
        }
        for _ in 0..12 {
            for _ in 0..bound {
                bw.put(1, 2);
                bw.put(2, 2);
            }
            for _ in bound..32 {
                bw.put(3, 2); // one shared sample stream
            }
        }
        let mut frame = w.to_be_bytes().to_vec();
        frame.extend_from_slice(&bw.finish());
        frame.resize(h.frame_bytes, 0);
        let out = decode_subbands(&h, &frame).expect("crafted joint frame decodes");
        // Shared upper subbands feed both channels from one bit stream.
        for (a, b) in out[0][bound..32].iter().zip(out[1][bound..32].iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn layer1_reserved_allocation_is_a_named_error() {
        // Mono: the per-channel loop sees the reserved code first.
        let w = header_word(0b11, 12, 0, 3, 0, true);
        let h = header_of(w);
        let mut bw = BitWriter::new();
        bw.put(15, 4);
        bw.put(0, 4 * 31 + 6 * 32 + 2 * 12);
        let mut frame = w.to_be_bytes().to_vec();
        frame.extend_from_slice(&bw.finish());
        frame.resize(h.frame_bytes, 0);
        let err = decode_subbands(&h, &frame).expect_err("reserved allocation");
        assert!(format!("{err:?}").contains("layer I allocation"));

        // Joint stereo: the shared region (bound..32) rejects it too.
        let wj = header_word(0b11, 12, 0, 1, 1, true);
        let hj = header_of(wj);
        let bound = hj.js_bound(32);
        let mut bw = BitWriter::new();
        for _ in 0..bound {
            bw.put(0, 4);
            bw.put(0, 4);
        }
        bw.put(15, 4); // first shared subband
        bw.put(0, 4 * 23 + 6 * 32 + 2 * 12);
        let mut frame = wj.to_be_bytes().to_vec();
        frame.extend_from_slice(&bw.finish());
        frame.resize(hj.frame_bytes, 0);
        let err = decode_subbands(&hj, &frame).expect_err("reserved shared allocation");
        assert!(format!("{err:?}").contains("layer I allocation"));
    }

    #[test]
    fn layer1_scalefactor_63_is_a_named_error() {
        let w = header_word(0b11, 12, 0, 3, 0, true);
        let h = header_of(w);
        let mut bw = BitWriter::new();
        bw.put(1, 4);
        bw.put(0, 4 * 31);
        bw.put(63, 6); // reserved scalefactor index
        bw.put(0, 6 * 31 + 2 * 12);
        let mut frame = w.to_be_bytes().to_vec();
        frame.extend_from_slice(&bw.finish());
        frame.resize(h.frame_bytes, 0);
        let err = decode_subbands(&h, &frame).expect_err("scalefactor 63");
        assert!(format!("{err:?}").contains("scalefactor"));
    }

    #[test]
    fn layer1_crc16_gates_the_frame() {
        let w = header_word(0b11, 12, 0, 3, 0, false); // protected
        let h = header_of(w);
        assert!(!h.unprotected);
        let sf: u32 = 17;
        // Fields exactly as the decoder reads them: 32 allocation nibbles,
        // one scalefactor for the only allocated subband, 12 samples.
        let mut crc = FrameCrc::new();
        for b in &w.to_be_bytes()[2..4] {
            crc.byte(*b);
        }
        let mut bw = BitWriter::new();
        for sb in 0..32usize {
            let a = u32::from(sb == 0);
            bw.put(a, 4);
            crc.field(a, 4);
        }
        bw.put(sf, 6); // scalefactors sit outside the CRC's coverage
        for _ in 0..12 {
            bw.put(1, 2); // samples too
        }
        let mut frame = w.to_be_bytes().to_vec();
        frame.extend_from_slice(&crc.0.to_be_bytes());
        frame.extend_from_slice(&bw.finish());
        frame.resize(h.frame_bytes, 0);
        decode_subbands(&h, &frame).expect("matching CRC accepts the frame");

        // Flip one CRC bit: the frame must be refused by name.
        frame[4] ^= 0x01;
        let err = decode_subbands(&h, &frame).expect_err("corrupt CRC");
        assert!(format!("{err:?}").contains("crc"));
    }

    /// scfsi value per allocated slot; walks all four layout arms.
    const SLOT_SCFSI: [u32; 4] = [0, 1, 2, 3];

    #[test]
    fn layer2_joint_stereo_reads_shared_region_and_all_scfsi_arms() {
        // L2 384k 48 kHz joint stereo, mode_ext 1 -> bound 8 < sblimit 27.
        let w = header_word(0b10, 13, 1, 1, 1, true);
        let h = header_of(w);
        assert_eq!(l2_table_index(&h), 0);
        let t = l2_table_index(&h);
        let sblimit = L2_SBLIMIT[t];
        let bound = h.js_bound(sblimit);
        assert!(bound < sblimit);
        let tab = ALLOC_TABLES[t];

        // Allocation codes: sb0 (per-channel region) and sb=bound (shared
        // region) carry code 1, everything else 0.
        let mut bw = BitWriter::new();
        let mut allocated: Vec<(usize, usize)> = Vec::new(); // (sb, nbal)
        let mut j = 0usize;
        for sb in 0..bound {
            let nbal = tab[j] as usize;
            for _ in 0..2 {
                bw.put(u32::from(sb == 0), nbal);
            }
            if sb == 0 {
                allocated.push((sb, nbal));
            }
            j += 1 << nbal;
        }
        let shared_sb = bound;
        for sb in bound..sblimit {
            let nbal = tab[j] as usize;
            bw.put(u32::from(sb == shared_sb), nbal);
            if sb == shared_sb {
                allocated.push((sb, nbal));
            }
            j += 1 << nbal;
        }
        // scfsi: 2 bits per allocated (subband, channel); assign all four
        // values across the four allocated slots (read order: sb0/ch0,
        // sb0/ch1, shared/ch0, shared/ch1) so every scalefactor layout
        // arm runs.
        let sf_count = [3usize, 2, 1, 2]; // per scfsi arm: 3/2/1/2 factors
        for slot in 0..allocated.len() * 2 {
            bw.put(SLOT_SCFSI[slot % 4], 2);
        }
        // Scalefactors follow the same slots.
        for slot in 0..allocated.len() * 2 {
            for _ in 0..sf_count[SLOT_SCFSI[slot % 4] as usize] {
                bw.put(7, 6);
            }
        }
        // Sample bits: every allocated subband reads three values per
        // granule; zeros are valid codes for every quantizer class.
        bw.put(0, 64 * 12 * 8);

        let mut frame = w.to_be_bytes().to_vec();
        frame.extend_from_slice(&bw.finish());
        frame.resize(h.frame_bytes, 0);
        let out = decode_subbands(&h, &frame).expect("crafted L2 joint frame decodes");
        for (a, b) in out[0][bound..sblimit]
            .iter()
            .zip(out[1][bound..sblimit].iter())
        {
            assert_eq!(a, b);
        }
    }
}
