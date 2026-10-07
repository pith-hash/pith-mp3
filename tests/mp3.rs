//! End-to-end and hostile-input tests for the MP3 decoder.
//!
//! Real fixtures come from ffmpeg 9.0.1's libmp3lame/libtwolame encoders
//! (fixture provenance: the upstream `modhash` kit's `lab/mp3/PROVENANCE.md`); the reference PCM is ffmpeg's own
//! `mp3`/`mp2` decoder at `s32le`, so the comparison is against an
//! independent implementation rather than a self-consistent one.

use pith_mp3::{Decoder, Header, Limits, decode, decode_header};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

/// Compare two interleaved s32 streams with a lag search over ±3 frames.
/// MP3 decoders disagree on encoder delay (the Xing tag's gapless
/// fields), so absolute alignment isn't meaningful — the *shape* of the
/// decoded audio is what must match.
fn compare(name: &str, mine: &[i32], reference: &[i32], channels: usize) {
    compare_n(name, mine, reference, channels, 1152);
}

fn compare_n(name: &str, mine: &[i32], reference: &[i32], channels: usize, spf: usize) {
    // Skip the first two frames (decoder delay region differs between
    // implementations) and search lag at SAMPLE granularity: ffmpeg
    // applies the Xing tag's gapless trim (LAME start delay 1105
    // samples + decoder delay), we emit the raw stream. A
    // frame-granularity search can never find a sub-frame offset —
    // for white-noise fixtures a one-sample shift is fully
    // decorrelated — so lag is in per-channel sample positions.
    assert!(
        mine.len() > 2 * spf * channels && reference.len() > 2 * spf * channels,
        "{name}: not enough decoded audio to compare (mine={}, ref={})",
        mine.len(),
        reference.len()
    );
    let mine_n = mine.len() / channels;
    let ref_n = reference.len() / channels;
    let mut best = (f64::INFINITY, 0i64);
    for lag in -6000i64..=6000 {
        let (a_start, b_start) = if lag >= 0 {
            (2 * spf, 2 * spf + lag as usize)
        } else {
            (2 * spf + (-lag) as usize, 2 * spf)
        };
        if a_start >= mine_n || b_start >= ref_n {
            continue;
        }
        let n = (mine_n - a_start).min(ref_n - b_start).min(8 * spf);
        let mut e = 0.0f64;
        for i in 0..n {
            for c in 0..channels {
                let d = (mine[(a_start + i) * channels + c] as f64
                    - reference[(b_start + i) * channels + c] as f64)
                    / 2147483648.0;
                e += d * d;
            }
        }
        let rmse = (e / (n * channels) as f64).sqrt();
        if rmse < best.0 {
            best = (rmse, lag);
        }
    }
    eprintln!("{name}: rmse {:.5} at lag {} samples", best.0, best.1);
    assert!(
        best.0 < 0.08,
        "{name}: RMSE {:.4} at lag {} samples is too far from the ffmpeg reference",
        best.0,
        best.1
    );
}

#[test]
fn decodes_layer3_stereo() {
    let mp3 = decode(&fixture("l3_stereo.mp3"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 2);
    assert_eq!(mp3.sample_rate, 44100);
    assert_eq!(mp3.samples.len() % (1152 * 2), 0);
    assert!(mp3.frames >= 8);
    compare(
        "l3_stereo",
        &mp3.samples,
        &i32s(&fixture("l3_stereo.ref.pcm")),
        2,
    );
}

#[test]
fn decodes_layer3_mono() {
    let mp3 = decode(&fixture("l3_mono.mp3"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 1);
    compare(
        "l3_mono",
        &mp3.samples,
        &i32s(&fixture("l3_mono.ref.pcm")),
        1,
    );
}

#[test]
fn decodes_layer3_48k() {
    let mp3 = decode(&fixture("l3_48k.mp3"), &Limits::default()).unwrap();
    assert_eq!(mp3.sample_rate, 48000);
    compare("l3_48k", &mp3.samples, &i32s(&fixture("l3_48k.ref.pcm")), 1);
}

#[test]
fn decodes_layer3_short_blocks() {
    // White noise forces short windows into the stream.
    let mp3 = decode(&fixture("l3_short.mp3"), &Limits::default()).unwrap();
    compare(
        "l3_short",
        &mp3.samples,
        &i32s(&fixture("l3_short.ref.pcm")),
        1,
    );
}

#[test]
fn decodes_layer3_joint_stereo() {
    let mp3 = decode(&fixture("l3_joint.mp3"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 2);
    compare(
        "l3_joint",
        &mp3.samples,
        &i32s(&fixture("l3_joint.ref.pcm")),
        2,
    );
}

#[test]
fn decodes_layer3_vbr() {
    let mp3 = decode(&fixture("l3_vbr.mp3"), &Limits::default()).unwrap();
    assert!(mp3.vbr, "the fixture carries a Xing header");
    compare("l3_vbr", &mp3.samples, &i32s(&fixture("l3_vbr.ref.pcm")), 1);
}

#[test]
fn decodes_layer1_stereo() {
    // ffmpeg cannot encode Layer I, so the upstream kit's lab/mp3/make_l1_fixture.py
    // hand-writes a spec-valid stream (384 kbps stereo, alloc=2,
    // per-subband scalefactors, deterministic sample pattern) and
    // ffmpeg's mp1 decoder produced the reference PCM.
    let mp3 = decode(&fixture("l1_stereo.mp1"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 2);
    assert_eq!(mp3.samples.len() % (384 * 2), 0);
    compare_n(
        "l1_stereo",
        &mp3.samples,
        &i32s(&fixture("l1_stereo.ref.pcm")),
        2,
        384,
    );
}

#[test]
fn decodes_layer2_stereo() {
    let mp3 = decode(&fixture("l2_stereo.mp2"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 2);
    assert_eq!(mp3.samples.len() % (1152 * 2), 0);
    compare(
        "l2_stereo",
        &mp3.samples,
        &i32s(&fixture("l2_stereo.ref.pcm")),
        2,
    );
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic PRNG for hostile-input scans.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[test]
fn hostile_inputs_never_panic() {
    // Prefix scans: every truncation of a real stream must be an Err,
    // never a panic, never Ok-with-garbage semantics.
    let src = fixture("l3_stereo.mp3");
    for cut in [0usize, 3, 10, 50, 100, 300, 417, src.len() - 1] {
        let r = decode(&src[..cut], &Limits::default());
        assert!(r.is_err(), "truncated to {cut} must not decode");
    }

    // Single-byte mutations of a middle frame's header + side info.
    let mut rng = 0x1234_5678_9abc_def0u64;
    for _ in 0..400 {
        let mut mutated = src.clone();
        let pos = (splitmix64(&mut rng) as usize) % src.len();
        mutated[pos] ^= (splitmix64(&mut rng) as u8) | 1;
        let _ = decode(&mutated, &Limits::default());
    }

    // Pure-noise streams: invalid magic or a stream of phantom syncs.
    for n in [4usize, 64, 512, 4096] {
        let mut buf = vec![0u8; n];
        let mut s = 42u64;
        for b in &mut buf {
            *b = splitmix64(&mut s) as u8;
        }
        let _ = decode(&buf, &Limits::default());
    }

    // All-0xFF stream: every 4-byte window looks like a sync word with
    // reserved fields.
    let ff = vec![0xFFu8; 2048];
    let _ = decode(&ff, &Limits::default());
}

#[test]
fn crc_corruption_detected() {
    // Locate the first real frame (after the ID3v2 tag), flip a bit in
    // the protected region of a frame that carries a CRC.
    let src = fixture("l3_stereo.mp3");
    let mut at = 0;
    if src.len() >= 10 && &src[..3] == b"ID3" {
        at = 10
            + (((src[6] & 0x7F) as usize) << 21)
            + (((src[7] & 0x7F) as usize) << 14)
            + (((src[8] & 0x7F) as usize) << 7)
            + (src[9] & 0x7F) as usize;
    }
    // Walk frames until a protected one turns up.
    let mut found = false;
    let mut pos = at;
    while pos + 4 < src.len() {
        if src[pos] != 0xFF || (src[pos + 1] & 0xE0) != 0xE0 {
            break;
        }
        let protected = src[pos + 1] & 1 == 0;
        if protected {
            // Corrupt a side-info byte inside the CRC's coverage.
            let mut m = src.clone();
            m[pos + 8] ^= 0x40;
            let r = decode(&m, &Limits::default());
            assert!(
                r.is_err() || r.unwrap().frames == 0,
                "corrupting a protected side-info byte must fail the frame"
            );
            found = true;
            break;
        }
        // Next frame.
        let bri = (src[pos + 2] >> 4) & 0xF;
        let sri = (src[pos + 2] >> 2) & 3;
        let pad = (src[pos + 2] >> 1) & 1;
        let bitrate = [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ][bri as usize]
            * 1000;
        let rate = [44100, 48000, 32000][sri as usize];
        pos += (144 * bitrate / rate + pad as usize).max(4);
    }
    // LAME defaults to protection off; if none found, still exercise the
    // CRC path via the protected-fixture mutation test below.
    eprintln!("protected frame found: {found}");
}

#[test]
fn reservoir_under_run_is_named_error() {
    // main_data_begin pointing before the start of the stream must be
    // Truncated, not a panic or silent garbage.
    let src = fixture("l3_mono.mp3");
    let mut at = 0;
    if src.len() >= 10 && &src[..3] == b"ID3" {
        at = 10
            + (((src[6] & 0x7F) as usize) << 21)
            + (((src[7] & 0x7F) as usize) << 14)
            + (((src[8] & 0x7F) as usize) << 7)
            + (src[9] & 0x7F) as usize;
    }
    // Frame 0 of a LAME stream is the Xing tag: it contributes bytes to
    // the reservoir but decodes no granules. Corrupt frame 1's
    // main_data_begin to 0x1FF so its granule reaches before the start
    // of the stream.
    let first = pith_mp3::Header::parse(&src[at..at + 4]).unwrap();
    let p0 = at + first.frame_bytes; // second frame
    let h1 = pith_mp3::Header::parse(&src[p0..p0 + 4]).unwrap();
    let mut m = src.clone();
    let crc_bytes = if h1.unprotected { 0 } else { 2 };
    let p = p0 + 4 + crc_bytes;
    m[p] = 0xFF;
    m[p + 1] |= 0x80;
    let r = decode(&m, &Limits::default());
    assert!(
        r.is_err(),
        "main_data_begin past reservoir start must error"
    );
}

#[test]
fn reservoir_back_pointer_reads_donor_bytes() {
    // `reservoir_pair.mp3` is a hand-assembled two-frame stream made by
    // the upstream kit's `lab/mp3/make_reservoir_pair.py`: frame 1 is a silent donor whose
    // main-data area carries the last 511 history bytes of
    // `l3_mono.mp3`'s first five frames; frame 2 is `l3_mono`'s frame 4
    // verbatim, whose `main_data_begin = 67` back-pointer reaches 67
    // bytes into the donor's payload. Decoding must succeed (the
    // granule's `part2_3_length` bits exist only because the history
    // bytes were assembled in) and the second frame must emit real
    // PCM — the donor itself is silent.
    let mp3 = decode(&fixture("reservoir_pair.mp3"), &Limits::default()).unwrap();
    assert_eq!(mp3.channels, 1);
    assert_eq!(mp3.frames, 2);
    let donor = &mp3.samples[..1152];
    assert!(
        donor.iter().all(|&s| s == 0),
        "the donor granules are empty: its output must be silence"
    );
    let target = &mp3.samples[1152..];
    assert!(
        target.iter().any(|&s| s != 0),
        "the target frame must produce PCM read out of the donor's \
         reservoir contribution"
    );
    // Pin the exact output: the FNV-1a/64 of the target frame's PCM.
    // The value was recorded from a build verified bit-exact against
    // minimp3's decode of the same constructed stream; any drift in the
    // reservoir assembly, Huffman decode or synthesis changes it.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &s in target {
        h = (h ^ s as u32 as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    assert_eq!(h, 0x8bdf_cc6c_e0c9_ca2d, "target PCM changed");
}

#[test]
fn limits_enforced() {
    let src = fixture("l3_mono.mp3");
    let r = decode(
        &src,
        &Limits {
            max_input: 10,
            ..Limits::default()
        },
    );
    assert!(r.is_err());
    let r = decode(
        &src,
        &Limits {
            max_frames: 1,
            ..Limits::default()
        },
    );
    assert!(r.is_err());
    let r = decode(
        &src,
        &Limits {
            max_output: 8,
            ..Limits::default()
        },
    );
    assert!(r.is_err());
}

#[test]
fn header_probe_matches_stream() {
    let src = fixture("l3_stereo.mp3");
    let info = decode_header(&src).unwrap();
    assert_eq!(info.channels, 2);
    assert_eq!(info.sample_rate, 44100);
    assert!(matches!(info.layer, pith_mp3::Layer::III));
}

#[test]
fn empty_and_tiny_inputs() {
    assert!(decode(&[], &Limits::default()).is_err());
    assert!(decode(b"\xff\xfb\x90", &Limits::default()).is_err());
    assert!(decode_header(&[]).is_err());
    // A lone valid header with no body.
    let header_only = [0xFF, 0xFB, 0x90, 0x00];
    assert!(decode(&header_only, &Limits::default()).is_err());
}

#[test]
fn header_tables_match_spec() {
    // Every (layer, bitrate index) pair the header can name, plus the
    // frame-length formula of §2.4.2.3 — built as raw 32-bit fields so
    // a mis-ordered table entry is pinned by construction, not by
    // comparing against the decoder's own tables.
    use pith_mp3::Header;
    const BR1: [u32; 15] = [
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
    ];
    const BR2: [u32; 15] = [
        32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
    ];
    const BR3: [u32; 15] = [
        32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const SR: [u32; 3] = [44100, 48000, 32000];
    for (layer_code, table) in [(3u32, &BR1), (2, &BR2), (1, &BR3)] {
        for bri in 1..15u32 {
            for (sri, &sr) in SR.iter().enumerate() {
                let w: u32 = 0xFFE0_0000
                    | (3 << 19)
                    | (layer_code << 17)
                    | (1 << 16)
                    | (bri << 12)
                    | ((sri as u32) << 10);
                let h = Header::parse(&w.to_be_bytes()).unwrap();
                assert_eq!(h.bitrate_kbps, table[bri as usize - 1]);
                assert_eq!(h.sample_rate, sr);
                let expected = match layer_code {
                    3 => ((12 * h.bitrate_kbps * 1000 / sr) * 4) as usize,
                    _ => (144 * h.bitrate_kbps * 1000 / sr) as usize,
                };
                assert_eq!(
                    h.frame_bytes, expected,
                    "layer_code {layer_code} bri {bri} sr {sr}"
                );
            }
        }
    }
    // Layer III's reserved-field errors must be named, not panics.
    // `(0 << n)` keeps the bit position readable where the field itself
    // is what fails.
    #[allow(clippy::identity_op)]
    for w in [
        0x0000_0000u32,                      // no sync
        0xFFE0_0000 | (2 << 19),             // reserved version bits
        0xFFE0_0000 | (0 << 17),             // reserved layer bits
        0xFFE0_0000 | (15 << 12),            // "bad" bitrate index
        0xFFE0_0000,                         // free format (bitrate index 0)
        0xFFE0_0000 | (1 << 12) | (3 << 10), // reserved sample rate
        0xFFE0_0000 | (0 << 19),             // MPEG-2.5: named Unsupported
    ] {
        assert!(
            Header::parse(&w.to_be_bytes()).is_err(),
            "header {w:08x} must be rejected"
        );
    }
    // MPEG-2 is a named Unsupported, not a bad-value parse failure.
    let mpeg2 = 0xFFE0_0000u32 | (2 << 19) | (1 << 17) | (6 << 12);
    let e = Header::parse(&mpeg2.to_be_bytes()).unwrap_err();
    let msg = format!("{e}");
    assert!(msg.contains("unsupported"), "got {msg}");
}

#[test]
fn decoder_feed_matches_bulk_decode() {
    // Feeding frames individually through one Decoder must produce
    // bit-identical PCM to `decode` — the reservoir and IMDCT overlap
    // are the only cross-frame state, and both live inside Decoder.
    let src = fixture("l3_mono.mp3");
    let bulk = decode(&src, &Limits::default()).unwrap();
    let mut at = 0;
    if src.len() >= 10 && &src[..3] == b"ID3" {
        at = 10
            + (((src[6] & 0x7F) as usize) << 21)
            + (((src[7] & 0x7F) as usize) << 14)
            + (((src[8] & 0x7F) as usize) << 7)
            + (src[9] & 0x7F) as usize;
    }
    let mut dec = Decoder::new();
    let mut pcm = Vec::new();
    let mut pos = at;
    let mut seen = 0usize;
    while let Ok(h) = pith_mp3::Header::parse(src.get(pos..pos + 4).unwrap_or(&[])) {
        if pos + h.frame_bytes > src.len() {
            break;
        }
        dec.feed(&h, &src[pos..pos + h.frame_bytes], &mut pcm)
            .unwrap();
        seen += 1;
        pos += h.frame_bytes;
    }
    assert_eq!(
        pcm, bulk.samples,
        "per-frame feeding must match the bulk decode byte for byte"
    );
    assert!(seen >= 5);
}

// ---------------------------------------------------------------------------
// Crafted streams: error paths and branch shapes the recorded fixtures
// cannot reach (they are all well-formed encodes).
// ---------------------------------------------------------------------------

/// ISO/IEC 11172-3 CRC-16 over the protected span: poly 0x8005, initial
/// state 0xFFFF, MSB-first, no final XOR (CRC-16/UMTS).
struct FrameCrc(u16);

impl FrameCrc {
    fn new() -> Self {
        FrameCrc(0xFFFF)
    }
    fn byte(&mut self, b: u8) {
        for i in (0..8).rev() {
            let fb = ((self.0 >> 15) as u8 ^ ((b >> i) & 1)) & 1;
            self.0 <<= 1;
            if fb != 0 {
                self.0 ^= 0x8005;
            }
        }
    }
    fn bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.byte(b);
        }
    }
}

/// Builds an MPEG-1 audio header word: sync 11 bits, version `11`, then
/// layer / protection / bri / sri / pad / mode / mode_ext.
fn header_word(layer: u32, bri: u32, sri: u32, mode: u32, mode_ext: u32, unprotected: bool) -> u32 {
    0xFFE0_0000
        | (0b11 << 19)
        | (layer << 17)
        | (u32::from(unprotected) << 16)
        | (bri << 12)
        | (sri << 10)
        | (mode << 6)
        | (mode_ext << 4)
}

/// Byte offset of the first real frame, past a leading ID3v2 tag.
fn first_frame(src: &[u8]) -> usize {
    if src.len() >= 10 && &src[..3] == b"ID3" {
        10 + (((src[6] & 0x7F) as usize) << 21)
            + (((src[7] & 0x7F) as usize) << 14)
            + (((src[8] & 0x7F) as usize) << 7)
            + (src[9] & 0x7F) as usize
    } else {
        0
    }
}

/// Walks a stream's frame spans `(start, len)` until the headers run out.
fn frame_spans(src: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = first_frame(src);
    while pos + 4 <= src.len() {
        let Ok(h) = Header::parse(&src[pos..pos + 4]) else {
            break;
        };
        if pos + h.frame_bytes > src.len() {
            break;
        }
        spans.push((pos, h.frame_bytes));
        pos += h.frame_bytes;
    }
    spans
}

#[test]
fn header_field_validation_is_named() {
    // Baseline: MPEG-1 Layer III, 128 kbit/s, 44.1 kHz, mono, unprotected.
    let base = header_word(0b01, 9, 0, 3, 0, true);
    let parse = |w: u32| Header::parse(&w.to_be_bytes());
    // Reserved version code 01.
    let e = format!("{:?}", parse((base & !(3 << 19)) | (1 << 19)).unwrap_err());
    assert!(e.contains("version bits"), "{e}");
    // Reserved layer code 00.
    let e = format!("{:?}", parse(base & !(3 << 17)).unwrap_err());
    assert!(e.contains("layer bits"), "{e}");
    // Free bitrate (index 0) is unsupported; index 15 is reserved.
    let e = format!("{:?}", parse(base & !(0xF << 12)).unwrap_err());
    assert!(e.contains("free bitrate"), "{e}");
    let e = format!("{:?}", parse(base | (0xF << 12)).unwrap_err());
    assert!(e.contains("bitrate index"), "{e}");
    // Reserved sampling-rate code 11.
    let e = format!("{:?}", parse(base | (3 << 10)).unwrap_err());
    assert!(e.contains("sampling rate index"), "{e}");
    // Broken sync.
    let e = format!("{:?}", parse(base & !(0x7FF << 21)).unwrap_err());
    assert!(e.contains("sync"), "{e}");
    // Short input.
    let e = format!("{:?}", Header::parse(&[0xFF, 0xFB, 0x90]).unwrap_err());
    assert!(e.contains("Truncated"), "{e}");
}

#[test]
fn header_accessors_follow_the_mode_bits() {
    // Layer I joint stereo, mode_ext 2: js_bound = (2+1)*4 = 12, clamped
    // by the caller's subband limit; Layer III is never jointly coded.
    let l1 = Header::parse(&header_word(0b11, 12, 0, 1, 2, true).to_be_bytes()).unwrap();
    assert_eq!(l1.js_bound(32), 12);
    assert_eq!(l1.js_bound(5), 5);
    assert!(l1.intensity_stereo() == ((2 & 1) != 0) && !l1.intensity_stereo());
    assert!(l1.ms_stereo());
    let l3 = Header::parse(&header_word(0b01, 9, 0, 1, 3, true).to_be_bytes()).unwrap();
    assert_eq!(l3.js_bound(32), 0);
}

#[test]
fn id3_tags_and_xing_frames_shape_the_stream() {
    let src = fixture("l3_mono.mp3");
    let plain = decode(&src, &Limits::default()).unwrap();

    // A leading ID3v2 tag (size 0) and a trailing ID3v1 tag are skipped.
    // The fixture itself opens with an ID3v2 tag, so only the frames after
    // it are re-wrapped.
    let body = &src[first_frame(&src)..];
    let mut tagged = Vec::new();
    tagged.extend_from_slice(b"ID3\x04\x00\x00\x00\x00\x00\x00");
    tagged.extend_from_slice(body);
    tagged.extend_from_slice(b"TAG");
    tagged.extend_from_slice(&[0u8; 125]);
    let got = decode(&tagged, &Limits::default()).unwrap();
    assert_eq!(got.samples, plain.samples);

    // Writing the Xing magic into a later frame's ancillary space makes
    // that one frame metadata, not audio: vbr flips on, one frame of PCM
    // disappears, and the stream still decodes.
    let spans = frame_spans(&src);
    let (f1, f1len) = spans[1];
    let h = Header::parse(&src[f1..f1 + 4]).unwrap();
    let side = 4 + usize::from(!h.unprotected) * 2 + if h.channels() == 1 { 17 } else { 32 };
    let mut xing = src.clone();
    xing[f1 + side..f1 + side + 4].copy_from_slice(b"Xing");
    let got = decode(&xing, &Limits::default()).unwrap();
    assert!(got.vbr);
    assert_eq!(got.frames, plain.frames);
    assert_eq!(got.samples.len(), plain.samples.len() - 1152);
    let _ = f1len;
}

#[test]
fn layer3_crc16_protects_side_info() {
    // Rebuild l3_mono's second frame (frame 0 carries LAME's Info tag)
    // with the protection bit on and a correct CRC-16 over header bytes
    // 2..4 plus the 17 side-info bytes that follow the CRC word.
    let src = fixture("l3_mono.mp3");
    let (f1, f1len) = frame_spans(&src)[1];
    let mut w = src[f1..f1 + f1len].to_vec();
    w[1] &= 0xFE; // protection_bit = 0 -> protected
    let mut crc = FrameCrc::new();
    crc.bytes(&w[2..4]);
    crc.bytes(&w[4..4 + 17]);
    let mut protected = Vec::new();
    protected.extend_from_slice(&w[..4]);
    protected.extend_from_slice(&crc.0.to_be_bytes());
    protected.extend_from_slice(&w[4..]);
    let ok = decode(&protected, &Limits::default()).unwrap();
    assert_eq!(ok.frames, 1);
    assert_eq!(ok.samples.len(), 1152);

    // One flipped CRC bit rejects the frame by name.
    protected[4] ^= 0x01;
    let e = format!("{:?}", decode(&protected, &Limits::default()).unwrap_err());
    assert!(e.contains("crc"), "{e}");
}

#[test]
fn crafted_side_info_bit_errors_are_named() {
    let src = fixture("l3_mono.mp3");
    let spans = frame_spans(&src);
    let (f1, _) = spans[1];
    // Side info (mono, unprotected) starts at frame byte 4:
    // mdb 9b, private 5b, scfsi 4b, then per granule part2_3_length 12b,
    // big_values 9b, global_gain 8b, scalefac_compress 4b, then the
    // window-switching fields (side byte 6 holds ws/block_type/mixed).

    // window_switching = 1 with block_type = 00 is the reserved code.
    // Granule fields start at side bit 18: ws is side bit 51 (side byte 6
    // bit 4), block_type the two bits after it (bits 3..2).
    let mut bad = src.clone();
    bad[f1 + 10] = (bad[f1 + 10] | 0x10) & !0x0C;
    let e = format!("{:?}", decode(&bad, &Limits::default()).unwrap_err());
    assert!(e.contains("block_type"), "{e}");

    // part2_3_length reaching past the frame's main data is Truncated.
    // A smallest-rate mono frame (32 kbit/s at 48 kHz -> 96 bytes) leaves
    // only ~78 bytes of main data, so a large-but-legal part2_3_length
    // (big_values still <= 288) overshoots it.
    let mut tiny = header_word(0b01, 1, 1, 3, 0, true).to_be_bytes().to_vec();
    tiny.resize(96, 0);
    tiny[4 + 2] |= 0x3F; // part2_3_length bits 11..6
    tiny[4 + 3] = 0xBE; // pl low = 0b101111; big_values high = 0b10
    tiny[4 + 4] &= 0x01; // big_values low bits = 0 -> 256 <= 288
    let e = format!("{:?}", decode(&tiny, &Limits::default()).unwrap_err());
    assert!(e.contains("main_data") || e.contains("granule"), "{e}");

    // Small part2_3_length and low global_gain sweeps on frame 1: every
    // outcome is a named error or a clean decode, never a panic.
    let end = spans[2].0;
    for v in 0u32..40 {
        let mut m = src[..end].to_vec();
        let v = v.to_be_bytes();
        m[f1 + 6] = (m[f1 + 6] & !0x3F) | (v[1] >> 2) & 0x3F;
        m[f1 + 7] = ((m[f1 + 7] & 0x03) | (v[1] << 6) & 0xC0) | (v[0] >> 2) & 0x3F;
        m[f1 + 8] = (m[f1 + 8] & 0x0F) | (v[0] << 4) & 0xF0;
        let _ = decode(&m, &Limits::default());
    }
    for gg in 0u32..64 {
        let mut m = src[..end].to_vec();
        // global_gain: side byte 4 bit 0 (MSB) + side byte 5 (low 7 bits).
        m[f1 + 8] = (m[f1 + 8] & 0xFE) | ((gg >> 7) & 1) as u8;
        m[f1 + 9] = (gg & 0x7F) as u8;
        let _ = decode(&m, &Limits::default());
    }
}

#[test]
fn intensity_mode_extension_still_decodes() {
    // l3_joint is mid-side; forcing the intensity bit on every frame
    // header routes the stereo pass through the intensity branch. The
    // stream must still decode completely (values differ, structure holds).
    let src = fixture("l3_joint.mp3");
    let mut forced = src.clone();
    for (at, len) in frame_spans(&src) {
        forced[at + 3] |= 0x08; // mode_ext bit 0 -> intensity stereo
        let _ = len;
    }
    let got = decode(&forced, &Limits::default()).unwrap();
    let plain = decode(&src, &Limits::default()).unwrap();
    assert_eq!(got.frames, plain.frames);
    assert_eq!(got.samples.len(), plain.samples.len());
}

#[test]
fn frame_mutation_sieve_never_panics() {
    // Deterministic bit mutations of real frames (the Info-tagged frame 0
    // is skipped by design) exercise the hostile branches of side-info
    // parsing, requantization, Huffman decoding and the PCM clamp: every
    // outcome is Ok or a named Err, never a panic.
    let mut rng: u64 = 0x0DEFACED_12345678;
    let next = |rng: &mut u64| {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    for name in ["l3_mono.mp3", "l3_stereo.mp3", "l3_short.mp3"] {
        let src = fixture(name);
        let spans = frame_spans(&src);
        assert!(spans.len() >= 3, "{name}: need three frames");
        let start = spans[1].0;
        let end = spans[2].0 + spans[2].1;
        for _ in 0..1500 {
            let mut m = src[..end].to_vec();
            let flips = 1 + (next(&mut rng) as usize) % 3;
            for _ in 0..flips {
                let pos = start + (next(&mut rng) as usize) % (end - start);
                if (pos - spans[1].0) % spans[1].1 < 4
                    && pos % spans[1].1 == spans[1].0 % spans[1].1
                {
                    continue; // never break a sync word
                }
                m[pos] ^= 1 << (next(&mut rng) % 8);
            }
            let _ = decode(&m, &Limits::default());
        }
    }
}
