# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""Hex-exact conformance: the committed reference vectors through ctypes.

Every vector in the repository-root ``reference.json`` is replayed
through the cdylib and compared byte-exact — the PCM section's SHA-256
against ``pcm_sha256``, the first 64 PCM bytes against ``pcm_head_hex``
and every recorded stream fact. The same vectors the Rust
``gen-reference verify`` gate and the Node/Go SDKs check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pith_mp3 import FfiError, decode_canonical, find_cdylib, parse_canonical

REPO_ROOT = Path(__file__).resolve().parents[3]
VECTORS = json.loads((REPO_ROOT / "reference.json").read_text(encoding="utf-8"))["vectors"]


def test_cdylib_is_discoverable() -> None:
    path = find_cdylib()
    assert path.is_file(), path


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: v["fixture"])
def test_reference_vector_is_reproduced_hex_exact(vector: dict) -> None:
    data = (REPO_ROOT / "tests" / "fixtures" / vector["fixture"]).read_bytes()
    raw = decode_canonical(data)
    canonical = parse_canonical(raw)

    # The PCM section digests to the recorded sha256, byte-exact.
    assert hashlib.sha256(canonical.pcm).hexdigest() == vector["pcm_sha256"], vector["fixture"]

    # The first 64 PCM bytes match the recorded head hex.
    assert canonical.pcm_head_hex == vector["pcm_head_hex"], vector["fixture"]

    # Every stream fact the vector records.
    assert canonical.layer == {"I": 1, "II": 2, "III": 3}[vector["layer"]], vector["fixture"]
    assert canonical.bitrate_kbps == vector["bitrate_kbps"], vector["fixture"]
    assert canonical.header_sample_rate == vector["header_sample_rate"], vector["fixture"]
    assert canonical.header_channels == vector["header_channels"], vector["fixture"]
    assert canonical.channels == vector["channels"], vector["fixture"]
    assert canonical.sample_rate == vector["sample_rate"], vector["fixture"]
    assert canonical.frames == vector["frames"], vector["fixture"]
    assert canonical.samples == vector["samples"], vector["fixture"]
    assert canonical.vbr == vector["vbr"], vector["fixture"]


def test_malformed_input_is_refused_not_crashing() -> None:
    with pytest.raises(FfiError) as err:
        decode_canonical(b"not an mpeg stream at all")
    assert err.value.status == -2


def test_empty_input_is_refused() -> None:
    with pytest.raises(FfiError):
        decode_canonical(b"")


def test_fixture_pcm_matches_a_rust_pinned_value() -> None:
    # l1_stereo's digest, pinned in the committed reference.json and
    # re-derived by the Rust unit tests; this test fails loudly even if
    # reference.json were regenerated wrongly.
    data = (REPO_ROOT / "tests" / "fixtures" / "l1_stereo.mp1").read_bytes()
    raw = decode_canonical(data)
    canonical = parse_canonical(raw)
    assert hashlib.sha256(canonical.pcm).hexdigest() == (
        "e705fa74b17d29db9df451fce896f8f0f5089c821fb8f5de4217adbd409c2440"
    )
    # Header prologue: layer I (1), 384 kbit/s, 44100 Hz stereo.
    assert raw[:8] == bytes([0, 0, 0, 1, 0, 0, 1, 128])
