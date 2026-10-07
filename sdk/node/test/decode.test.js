// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Hex-exact conformance: the committed reference vectors through koffi.
// Every vector in the repository-root reference.json is replayed through
// the cdylib and compared byte-exact — the PCM section's SHA-256 against
// pcm_sha256, the first 64 PCM bytes against pcm_head_hex and every
// recorded stream fact. The same vectors the Rust gen-reference verify
// gate and the Python/Go SDKs check.

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const { FfiError, decodeCanonical, findCdylib, parseCanonical } = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const VECTORS = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8")).vectors;

const LAYER_CODES = { I: 1, II: 2, III: 3 };

test("cdylib is discoverable", () => {
  assert.ok(fs.statSync(findCdylib()).isFile());
});

for (const vector of VECTORS) {
  test(`reference vector ${vector.fixture} is reproduced hex-exact`, () => {
    const data = fs.readFileSync(path.join(REPO_ROOT, "tests", "fixtures", vector.fixture));
    const raw = decodeCanonical(data);
    const canonical = parseCanonical(raw);

    assert.equal(
      crypto.createHash("sha256").update(canonical.pcm).digest("hex"),
      vector.pcm_sha256,
      vector.fixture,
    );
    assert.equal(canonical.pcm.subarray(0, 64).toString("hex"), vector.pcm_head_hex, vector.fixture);

    assert.equal(canonical.layer, LAYER_CODES[vector.layer], vector.fixture);
    assert.equal(canonical.bitrateKbps, vector.bitrate_kbps, vector.fixture);
    assert.equal(canonical.headerSampleRate, vector.header_sample_rate, vector.fixture);
    assert.equal(canonical.headerChannels, vector.header_channels, vector.fixture);
    assert.equal(canonical.channels, vector.channels, vector.fixture);
    assert.equal(canonical.sampleRate, vector.sample_rate, vector.fixture);
    assert.equal(canonical.frames, vector.frames, vector.fixture);
    assert.equal(canonical.samples, vector.samples, vector.fixture);
    assert.equal(canonical.vbr, vector.vbr, vector.fixture);
  });
}

test("malformed input is refused, not crashing", () => {
  assert.throws(() => decodeCanonical(Buffer.from("not an mpeg stream at all")), (err) => {
    assert.ok(err instanceof FfiError);
    assert.equal(err.status, -2);
    return true;
  });
});

test("empty input is refused", () => {
  assert.throws(() => decodeCanonical(Buffer.alloc(0)), FfiError);
});

test("fixture pcm matches a rust-pinned value", () => {
  // l1_stereo's digest, pinned in the committed reference.json and
  // re-derived by the Rust unit tests; this test fails loudly even if
  // reference.json were regenerated wrongly.
  const data = fs.readFileSync(path.join(REPO_ROOT, "tests", "fixtures", "l1_stereo.mp1"));
  const raw = decodeCanonical(data);
  const canonical = parseCanonical(raw);
  assert.equal(
    crypto.createHash("sha256").update(canonical.pcm).digest("hex"),
    "e705fa74b17d29db9df451fce896f8f0f5089c821fb8f5de4217adbd409c2440",
  );
  // Header prologue: layer I (1), 384 kbit/s, 44100 Hz stereo.
  assert.deepEqual([...raw.subarray(0, 8)], [0, 0, 0, 1, 0, 0, 1, 128]);
});
