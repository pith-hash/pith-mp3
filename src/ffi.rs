//! The C ABI surface of `pith-mp3`: the entry points the Python
//! (ctypes), Node (koffi) and Go (cgo) SDKs bind through.
//!
//! The suite's FFI convention, defined by the pilot cdylibs and
//! mirrored by every `pith-*` cdylib:
//!
//! * one flat set of `#[unsafe(no_mangle)] pub unsafe extern "C"`
//!   functions — raw pointers plus lengths, no structs across the
//!   boundary;
//! * every function returns a status code (see the constants below),
//!   never a `Result`, never a panic: a `panic = "abort"` cdylib must
//!   not be reachable from a foreign caller;
//! * an operation either hands ownership to the caller (and ships a
//!   matching `_free` — [`pith_mp3_free`] here) or writes into
//!   caller-provided out-parameters;
//! * the `unsafe` allowance is confined to this module; every core
//!   module stays unsafe-free behind the crate-root `#![deny]`.
//!
//! Decoding uses the crate's conservative default [`Limits`] — a
//! hashing pipeline never wants an unbounded decode, and the FFI
//! surface is no exception.
//!
//! # Canonical wire format
//!
//! [`pith_mp3_decode`] hands the caller the canonical decode-output
//! stream the SDKs consume: a 48-byte big-endian header followed by
//! the decoded PCM, interleaved `i32` little-endian — exactly the byte
//! string `pcm_sha256` in `reference.json` covers:
//!
//! | offset | width | field |
//! |-------:|------:|-------|
//! | 0  | u32 | layer (`1`, `2` or `3`) |
//! | 4  | u32 | first-header bitrate, kbit/s |
//! | 8  | u32 | first-header sample rate |
//! | 12 | u32 | first-header channel count |
//! | 16 | u32 | decoded channel count |
//! | 20 | u32 | decoded sample rate |
//! | 24 | u32 | decoded frame count |
//! | 28 | u64 | decoded sample count (interleaved) |
//! | 36 | u32 | `vbr` flag (0/1) |
//! | 40 | u64 | PCM byte length |
//! | 48 | …   | PCM, interleaved `i32` little-endian |

#![allow(unsafe_code)]

use alloc::vec::Vec;

use crate::{Layer, Limits, decode, decode_header};

/// Status: success.
pub const PITH_OK: i32 = 0;
/// Status: a caller argument is invalid — a null pointer.
pub const PITH_E_INVALID: i32 = -1;
/// Status: the core decoder refused the input (malformed MPEG audio:
/// no sync word, bad header fields, CRC mismatch, reservoir under-run,
/// or a truncated stream).
pub const PITH_E_REJECTED: i32 = -2;

/// The canonical stream's header length in bytes.
pub const PITH_HEADER_LEN: usize = 48;

/// Appends one big-endian `u32` field to the canonical header.
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Appends one big-endian `u64` field to the canonical header.
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// The layer's canonical numeric code (1/2/3).
fn layer_code(layer: Layer) -> u32 {
    match layer {
        Layer::I => 1,
        Layer::II => 2,
        Layer::III => 3,
    }
}

/// Decodes an MPEG audio stream into the canonical byte stream the
/// `reference.json` vectors are defined over.
///
/// `data` points at `len` bytes of the complete file. On success the
/// function allocates a buffer, writes its address through `out`, its
/// length through `out_len`, and returns [`PITH_OK`]; the caller owns
/// the buffer and must release it with [`pith_mp3_free`], passing back
/// the same pointer *and* length. The buffer layout is the canonical
/// wire format documented on this module.
///
/// # Safety
///
/// `data` must point to `len` readable bytes; `out` and `out_len` to
/// one writable pointer/`usize` each. All must stay valid for the
/// duration of the call; the function retains nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_mp3_decode(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match decode_and_serialize(bytes) {
        Ok(canonical) => {
            let len = canonical.len();
            // Hand the exact-length buffer to the caller; `pith_mp3_free`
            // reconstructs the boxed slice from the same length.
            let ptr = alloc::boxed::Box::into_raw(canonical.into_boxed_slice());
            unsafe {
                *out = ptr.cast::<u8>();
                *out_len = len;
            }
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Releases a buffer handed out by [`pith_mp3_decode`].
///
/// # Safety
///
/// `ptr` must be a pointer returned by [`pith_mp3_decode`] with the
/// `out_len` value that came back with it, and must not have been
/// released (or otherwise freed) before. Null is accepted and
/// ignored, so callers can free unconditionally on the error path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_mp3_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    drop(unsafe { alloc::boxed::Box::from_raw(slice) });
}

/// The safe core of [`pith_mp3_decode`]: probe the header, decode,
/// then serialize canonically. Any decoding failure maps to
/// [`PITH_E_REJECTED`].
fn decode_and_serialize(bytes: &[u8]) -> Result<Vec<u8>, i32> {
    let info = decode_header(bytes).map_err(|_| PITH_E_REJECTED)?;
    let mp3 = decode(bytes, &Limits::default()).map_err(|_| PITH_E_REJECTED)?;
    let mut out = Vec::with_capacity(PITH_HEADER_LEN + mp3.samples.len() * 4);
    // First-header facts.
    push_u32(&mut out, layer_code(info.layer));
    push_u32(&mut out, info.bitrate_kbps);
    push_u32(&mut out, info.sample_rate);
    push_u32(&mut out, u32::from(info.channels));
    // Decoded facts.
    push_u32(&mut out, u32::from(mp3.channels));
    push_u32(&mut out, mp3.sample_rate);
    push_u32(&mut out, mp3.frames as u32);
    push_u64(&mut out, mp3.samples.len() as u64);
    push_u32(&mut out, u32::from(mp3.vbr));
    push_u64(&mut out, mp3.samples.len() as u64 * 4);
    debug_assert_eq!(out.len(), PITH_HEADER_LEN);
    // The PCM section: interleaved i32 little-endian, the exact bytes
    // pcm_sha256 is computed over.
    for &s in &mp3.samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        PITH_E_INVALID, PITH_E_REJECTED, PITH_HEADER_LEN, PITH_OK, decode_and_serialize,
        pith_mp3_decode, pith_mp3_free,
    };
    use pith_digest::sha256;

    /// The committed Layer I conformance fixture, decoded end-to-end
    /// through the raw FFI: status OK, the 48-byte header carries the
    /// recorded facts, the PCM section digests to the recorded sha256,
    /// and the buffer round-trips through `pith_mp3_free`.
    #[test]
    fn ffi_decode_reproduces_the_canonical_stream() {
        let path = format!(
            "{}/tests/fixtures/l1_stereo.mp1",
            env!("CARGO_MANIFEST_DIR")
        );
        let input = std::fs::read(&path).expect("fixture");
        let expected = decode_and_serialize(&input).expect("decode");

        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status =
            unsafe { pith_mp3_decode(input.as_ptr(), input.len(), &mut out, &mut out_len) };
        assert_eq!(status, PITH_OK);
        assert_eq!(out_len, expected.len());
        let handed_back = unsafe { core::slice::from_raw_parts(out, out_len) };
        assert_eq!(handed_back, expected.as_slice());
        // The header: layer I (1), 384 kbit/s, 44100 Hz stereo, 16
        // frames, 12288 samples, not VBR, 49152 PCM bytes.
        let be32 = |o: usize| u32::from_be_bytes(handed_back[o..o + 4].try_into().unwrap());
        assert_eq!(be32(0), 1);
        assert_eq!(be32(4), 384);
        assert_eq!(be32(8), 44_100);
        assert_eq!(be32(12), 2);
        assert_eq!(be32(16), 2);
        assert_eq!(be32(24), 16);
        assert_eq!(
            u64::from_be_bytes(handed_back[28..36].try_into().unwrap()),
            12_288
        );
        assert_eq!(be32(36), 0);
        // The PCM section digests to the vector's recorded sha256.
        let digest = sha256(&handed_back[PITH_HEADER_LEN..]).expect("sha256");
        assert_eq!(
            digest.as_bytes(),
            &hex_bytes("e705fa74b17d29db9df451fce896f8f0f5089c821fb8f5de4217adbd409c2440")[..]
        );
        unsafe { pith_mp3_free(out, out_len) };
    }

    fn hex_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// The PCM section opens with the bytes `pcm_head_hex` records
    /// (the first 64 decoded bytes).
    #[test]
    fn pcm_head_matches_the_recorded_hex() {
        let path = format!(
            "{}/tests/fixtures/l1_stereo.mp1",
            env!("CARGO_MANIFEST_DIR")
        );
        let input = std::fs::read(&path).expect("fixture");
        let canonical = decode_and_serialize(&input).expect("decode");
        let head = hex_bytes(
            "00000000000000004046000055d5ffff90490000e2d0ffff934b00003fcfffff004e000051ccffffc15300004dc6ffff4c580000d5c2ffff90bc0000b37cffff",
        );
        assert_eq!(&canonical[PITH_HEADER_LEN..PITH_HEADER_LEN + 64], &head[..]);
    }

    /// Null pointers are [`PITH_E_INVALID`]; garbage input is
    /// [`PITH_E_REJECTED`]; a null buffer is a legal free.
    #[test]
    fn ffi_refusals() {
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let null_data = unsafe { pith_mp3_decode(core::ptr::null(), 0, &mut out, &mut out_len) };
        assert_eq!(null_data, PITH_E_INVALID);

        let stream = [0u8; 16];
        let null_out = unsafe {
            pith_mp3_decode(
                stream.as_ptr(),
                stream.len(),
                &mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(null_out, PITH_E_INVALID);

        let garbage =
            unsafe { pith_mp3_decode(stream.as_ptr(), stream.len(), &mut out, &mut out_len) };
        assert_eq!(garbage, PITH_E_REJECTED);

        unsafe { pith_mp3_free(core::ptr::null_mut(), 0) };
    }

    /// The safe core rejects malformed input instead of panicking.
    #[test]
    fn safe_core_rejects_garbage() {
        assert_eq!(
            decode_and_serialize(b"not an mpeg stream"),
            Err(PITH_E_REJECTED)
        );
    }
}
