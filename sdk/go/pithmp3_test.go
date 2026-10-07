// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

package pithmp3

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// repoRoot resolves the repository root relative to this package
// (sdk/go -> sdk -> repo root), the anchor for reference.json and the
// committed fixtures.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(filepath.Join(root, "reference.json")); err != nil || st.IsDir() {
		t.Fatalf("reference.json not found at %s", root)
	}
	return root
}

// mp3Vector mirrors one vector of reference.json.
type mp3Vector struct {
	Fixture          string `json:"fixture"`
	Layer            string `json:"layer"`
	BitrateKbps      uint32 `json:"bitrate_kbps"`
	HeaderSampleRate uint32 `json:"header_sample_rate"`
	HeaderChannels   uint32 `json:"header_channels"`
	Channels         uint32 `json:"channels"`
	SampleRate       uint32 `json:"sample_rate"`
	Frames           uint32 `json:"frames"`
	Samples          uint64 `json:"samples"`
	VBR              bool   `json:"vbr"`
	PcmSha256        string `json:"pcm_sha256"`
	PcmHeadHex       string `json:"pcm_head_hex"`
}

// reference parses the committed reference.json.
func reference(t *testing.T) []mp3Vector {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed struct {
		Vectors []mp3Vector `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	return parsed.Vectors
}

var layerCodes = map[string]uint32{"I": 1, "II": 2, "III": 3}

// TestReferenceVectorsHexExact replays every committed reference.json
// vector through the cdylib and compares byte-exact: the PCM section's
// SHA-256 against pcm_sha256, the first 64 PCM bytes against
// pcm_head_hex and every recorded stream fact — the same vectors the
// Rust gen-reference verify gate and the Python/Node SDKs check.
func TestReferenceVectorsHexExact(t *testing.T) {
	for _, want := range reference(t) {
		t.Run(want.Fixture, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", want.Fixture))
			if err != nil {
				t.Fatal(err)
			}
			raw, err := DecodeCanonical(data)
			if err != nil {
				t.Fatalf("DecodeCanonical(%s): %v", want.Fixture, err)
			}
			canonical, err := ParseCanonical(raw)
			if err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256(canonical.PCM())
			if got := hex.EncodeToString(digest[:]); got != want.PcmSha256 {
				t.Errorf("%s: digest %s, want %s", want.Fixture, got, want.PcmSha256)
			}
			if got := canonical.PCMHeadHex(); got != want.PcmHeadHex {
				t.Errorf("%s: head %s, want %s", want.Fixture, got, want.PcmHeadHex)
			}
			if canonical.Layer != layerCodes[want.Layer] ||
				canonical.BitrateKbps != want.BitrateKbps ||
				canonical.HeaderSampleRate != want.HeaderSampleRate ||
				canonical.HeaderChannels != want.HeaderChannels ||
				canonical.Channels != want.Channels ||
				canonical.SampleRate != want.SampleRate ||
				canonical.Frames != want.Frames ||
				canonical.Samples != want.Samples ||
				canonical.VBR != want.VBR {
				t.Errorf("%s: facts %+v, want layer=%s rate=%d/%d ch=%d/%d frames=%d samples=%d vbr=%v",
					want.Fixture, canonical, want.Layer, want.BitrateKbps, want.SampleRate,
					want.HeaderChannels, want.Channels, want.Frames, want.Samples, want.VBR)
			}
		})
	}
}

// TestPinnedDigest pins one digest the Rust unit tests re-derive, so
// the binding fails loudly even if reference.json were regenerated
// wrongly.
func TestPinnedDigest(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "fixtures", "l1_stereo.mp1"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := DecodeCanonical(data)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := ParseCanonical(raw)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(canonical.PCM())
	const want = "e705fa74b17d29db9df451fce896f8f0f5089c821fb8f5de4217adbd409c2440"
	if got := hex.EncodeToString(digest[:]); got != want {
		t.Errorf("l1_stereo: digest %s, want %s", got, want)
	}
	wantHeader := []byte{0, 0, 0, 1, 0, 0, 1, 128}
	for i, b := range wantHeader {
		if raw[i] != b {
			t.Fatalf("l1_stereo: header byte %d = %d, want %d", i, raw[i], b)
		}
	}
}

// TestMalformedInputIsRefused checks the decoder's refusal path: a
// status code, never a crash.
func TestMalformedInputIsRefused(t *testing.T) {
	_, err := DecodeCanonical([]byte("not an mpeg stream at all"))
	var ffi *FfiError
	if e, ok := err.(*FfiError); ok {
		ffi = e
	} else {
		t.Fatalf("want FfiError, got %v", err)
	}
	if ffi.Status != StatusRejected {
		t.Errorf("want StatusRejected, got %d", ffi.Status)
	}
}
