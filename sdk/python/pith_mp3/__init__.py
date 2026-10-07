# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""pith-mp3 SDK: MPEG audio decoding through ctypes.

Decodes layers I/II/III (including the bit reservoir) through the Rust
cdylib, handing back the canonical byte stream the ``reference.json``
vectors are defined over: a 48-byte big-endian header followed by the
decoded PCM, interleaved ``i32`` little-endian.

The cdylib is located through the suite's discovery chain:

1. ``PITH_CDYLIB`` — an explicit cdylib *file* path;
2. ``PITH_CDYLIB_DIR`` — a *directory* scanned for the cdylib names
   (the CD pipeline points this at ``target/release``);
3. the package directory itself (the built wheel ships the cdylib as
   package data);
4. ``<repo root>/target/release`` — the repository working-tree layout,
   so a source checkout runs against a local cargo build with no
   configuration.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Canonical",
    "FfiError",
    "LibraryNotFoundError",
    "find_cdylib",
    "decode_canonical",
    "parse_canonical",
    "STATUS_OK",
    "STATUS_INVALID",
    "STATUS_REJECTED",
    "HEADER_LEN",
]

#: Status: success.
STATUS_OK = 0
#: Status: a caller argument is invalid (a null pointer).
STATUS_INVALID = -1
#: Status: the core decoder refused the input (malformed MPEG audio).
STATUS_REJECTED = -2

#: The canonical stream's header length in bytes.
HEADER_LEN = 48

#: Every cdylib file name cargo may drop into the build directory, per
#: platform (windows / linux / macOS).
CDYLIB_NAMES = ("pith_mp3.dll", "libpith_mp3.so", "libpith_mp3.dylib")


@dataclass(frozen=True)
class Canonical:
    """A decoded MPEG audio stream, re-expressed from the canonical
    byte stream.

    ``pcm`` is the interleaved ``i32`` little-endian sample buffer —
    exactly the bytes the ``pcm_sha256`` digest covers.
    """

    #: Layer of the first frame: 1, 2 or 3.
    layer: int
    #: Bitrate of the first frame in kbit/s (VBR streams vary after).
    bitrate_kbps: int
    #: Sample rate declared in the first frame header.
    header_sample_rate: int
    #: Channel count declared in the first frame header.
    header_channels: int
    #: Decoded channel count, 1 or 2.
    channels: int
    #: Decoded sample rate per channel.
    sample_rate: int
    #: Frames actually decoded.
    frames: int
    #: Decoded sample count, interleaved across channels.
    samples: int
    #: True when a Xing/Info VBR header frame was seen.
    vbr: bool
    #: The canonical byte stream the digest is computed over.
    raw: bytes

    @property
    def pcm(self) -> bytes:
        """The interleaved ``i32`` little-endian PCM (everything after
        the 48-byte header)."""
        return self.raw[HEADER_LEN:]

    @property
    def pcm_head_hex(self) -> str:
        """The first 64 PCM bytes, lowercase hex — the same string the
        vectors' ``pcm_head_hex`` records."""
        return self.raw[HEADER_LEN : HEADER_LEN + 64].hex()


class LibraryNotFoundError(OSError):
    """No cdylib was found through the discovery chain."""


class FfiError(Exception):
    """A non-zero status code came back from the cdylib."""

    def __init__(self, op: str, status: int) -> None:
        detail = {
            STATUS_INVALID: "invalid argument",
            STATUS_REJECTED: "input rejected",
        }.get(status, "unknown failure")
        super().__init__(f"{op} failed: {detail} (status {status})")
        #: The raw status code the FFI returned.
        self.status = status


def find_cdylib() -> Path:
    """Locates the cdylib through the suite's discovery chain."""
    explicit = os.environ.get("PITH_CDYLIB")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env_dir = os.environ.get("PITH_CDYLIB_DIR")
    candidates: list[Path] = []
    if env_dir:
        env_dir_path = Path(env_dir)
        candidates.append(env_dir_path)
        if not env_dir_path.is_absolute():
            # CD and local runs invoke tools from the repository root or
            # from sdk/<lang>; resolve the env value against both.
            candidates.append(Path.cwd() / env_dir_path)
            candidates.append(Path(__file__).resolve().parents[3] / env_dir_path)
    candidates.append(Path(__file__).resolve().parent)  # packaged wheel
    candidates.append(Path(__file__).resolve().parents[3] / "target" / "release")
    for directory in candidates:
        for name in CDYLIB_NAMES:
            p = directory / name
            if p.is_file():
                return p
    raise LibraryNotFoundError(
        "no pith-mp3 cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, "
        "the package directory and <repo>/target/release); "
        "run `cargo build --release` first"
    )


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_cdylib()))
        lib.pith_mp3_decode.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_size_t,  # len
            ctypes.POINTER(ctypes.c_void_p),  # out buffer
            ctypes.POINTER(ctypes.c_size_t),  # out length
        ]
        lib.pith_mp3_decode.restype = ctypes.c_int32
        lib.pith_mp3_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.pith_mp3_free.restype = None
        _lib = lib
    return _lib


def decode_canonical(data: bytes) -> bytes:
    """Decodes a complete MPEG audio stream into the canonical byte
    stream the ``reference.json`` vectors are defined over.

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for any
    malformed input — no sync word, bad header fields, CRC mismatch or
    a truncated stream; the decoder never panics through this boundary.
    """
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    status = _load().pith_mp3_decode(data, len(data), ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError("pith_mp3_decode", status)
    try:
        return ctypes.string_at(out, out_len.value)
    finally:
        _load().pith_mp3_free(out, out_len.value)


def parse_canonical(raw: bytes) -> Canonical:
    """Re-expresses the canonical byte stream as a :class:`Canonical`."""
    if len(raw) < HEADER_LEN:
        raise ValueError("canonical stream is shorter than the 48-byte header")

    def u32(offset: int) -> int:
        return int.from_bytes(raw[offset : offset + 4], "big")

    def u64(offset: int) -> int:
        return int.from_bytes(raw[offset : offset + 8], "big")

    return Canonical(
        layer=u32(0),
        bitrate_kbps=u32(4),
        header_sample_rate=u32(8),
        header_channels=u32(12),
        channels=u32(16),
        sample_rate=u32(20),
        frames=u32(24),
        samples=u64(28),
        vbr=u32(36) == 1,
        raw=raw,
    )
