// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

// Package pithmp3 provides Go bindings for the pith-mp3 Rust cdylib:
// MPEG audio decoding into the canonical vector stream.
//
// The single Rust core (built by `cargo build --release`) is loaded at
// runtime; the package carries zero module dependencies. On unix the
// cdylib is opened with dlopen through cgo, on Windows with
// LoadLibrary through the standard syscall package — both resolve the
// library through the same discovery chain, so `go build ./... &&
// go test ./...` works unchanged on every OS the CD matrix builds.
//
// Discovery order (the suite's cdylib convention):
//
//  1. PITH_CDYLIB — an explicit cdylib file path;
//  2. PITH_CDYLIB_DIR — a directory scanned for the cdylib names (the
//     CD pipeline points this at target/release);
//  3. <repo root>/target/release — the repository working-tree layout,
//     anchored at this package's source directory, so a source
//     checkout runs against a local cargo build unconfigured.
//
// The FFI surface is one decode operation plus one free:
// pith_mp3_decode decodes a whole stream into the canonical byte
// stream the reference.json vectors are defined over (a 48-byte
// big-endian header of stream facts, then the decoded PCM as
// interleaved i32 little-endian), and pith_mp3_free releases the
// handed-out buffer.
package pithmp3

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// Status codes returned by the cdylib's C ABI.
const (
	// StatusOK: success.
	StatusOK int32 = 0
	// StatusInvalid: a caller argument is invalid (a null pointer).
	StatusInvalid int32 = -1
	// StatusRejected: the core decoder refused the input (malformed
	// MPEG audio: no sync word, bad header fields, CRC mismatch,
	// reservoir under-run, truncated stream).
	StatusRejected int32 = -2
)

// HeaderLen is the canonical stream's header length in bytes.
const HeaderLen = 48

// cdylibNames are the file names cargo may drop into the build
// directory, per platform (windows / linux / macOS).
var cdylibNames = []string{"pith_mp3.dll", "libpith_mp3.so", "libpith_mp3.dylib"}

// FfiError reports a non-zero status code from the cdylib.
type FfiError struct {
	// Op is the FFI operation name.
	Op string
	// Status is the raw status code the FFI returned.
	Status int32
}

func (e *FfiError) Error() string {
	kind := "unknown failure"
	switch e.Status {
	case StatusInvalid:
		kind = "invalid argument"
	case StatusRejected:
		kind = "input rejected"
	}
	return fmt.Sprintf("%s failed: %s (status %d)", e.Op, kind, e.Status)
}

// FindCdylib locates the cdylib through the suite's discovery chain.
func FindCdylib() (string, error) {
	if p := os.Getenv("PITH_CDYLIB"); p != "" {
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return filepath.Abs(p)
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("pithmp3: cannot locate the package source directory")
	}
	pkgDir := filepath.Dir(thisFile)
	repoRoot := filepath.Dir(filepath.Dir(pkgDir)) // sdk/go -> sdk -> repo root

	var dirs []string
	if env := os.Getenv("PITH_CDYLIB_DIR"); env != "" {
		dirs = append(dirs, env)
		if !filepath.IsAbs(env) {
			dirs = append(dirs, filepath.Join(repoRoot, env))
		}
	}
	dirs = append(dirs, filepath.Join(repoRoot, "target", "release"))
	for _, dir := range dirs {
		for _, name := range cdylibNames {
			p := filepath.Join(dir, name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"pithmp3: no cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR and <repo>/target/release); run `cargo build --release` first",
	)
}

// locate resolves the cdylib path once per process.
var locate = sync.OnceValues(FindCdylib)

// Canonical is a decoded MPEG audio stream, re-expressed from the
// canonical byte stream: the stream facts and the PCM.
type Canonical struct {
	// Layer of the first frame: 1, 2 or 3.
	Layer uint32
	// BitrateKbps is the bitrate of the first frame in kbit/s (VBR
	// streams vary after).
	BitrateKbps uint32
	// HeaderSampleRate is the sample rate declared in the first frame
	// header.
	HeaderSampleRate uint32
	// HeaderChannels is the channel count declared in the first frame
	// header.
	HeaderChannels uint32
	// Channels is the decoded channel count, 1 or 2.
	Channels uint32
	// SampleRate is the decoded sample rate per channel.
	SampleRate uint32
	// Frames is the number of frames actually decoded.
	Frames uint32
	// Samples is the decoded sample count, interleaved across
	// channels.
	Samples uint64
	// VBR reports a seen Xing/Info VBR header frame.
	VBR bool
	// Raw is the canonical byte stream the digest is computed over.
	Raw []byte
}

// PCM returns the interleaved i32 little-endian sample buffer —
// exactly the bytes the pcm_sha256 digest covers.
func (c *Canonical) PCM() []byte {
	return c.Raw[HeaderLen:]
}

// PCMHeadHex returns the first 64 PCM bytes, lowercase hex — the same
// string the vectors' pcm_head_hex records.
func (c *Canonical) PCMHeadHex() string {
	return hexEncode(c.Raw[HeaderLen : HeaderLen+64])
}

// hexEncode renders bytes as lowercase hex (stdlib-free, like the rest
// of the package).
func hexEncode(b []byte) string {
	const digits = "0123456789abcdef"
	out := make([]byte, len(b)*2)
	for i, v := range b {
		out[i*2] = digits[v>>4]
		out[i*2+1] = digits[v&0x0f]
	}
	return string(out)
}

// DecodeCanonical decodes a complete MPEG audio stream into the
// canonical byte stream the reference.json vectors are defined over.
// The returned slice is a Go copy; the handed-out cdylib buffer is
// released before returning.
func DecodeCanonical(data []byte) ([]byte, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	var dataPtr *byte
	if len(data) > 0 {
		dataPtr = &data[0]
	}
	status, err := ffiDecode(libPath, dataPtr, len(data), &out, &outLen)
	if err != nil {
		return nil, err
	}
	if status != StatusOK {
		return nil, &FfiError{Op: "pith_mp3_decode", Status: status}
	}
	buf := make([]byte, outLen)
	copy(buf, unsafe.Slice(out, outLen))
	ffiFree(libPath, out, outLen)
	return buf, nil
}

// ParseCanonical re-expresses the canonical byte stream as a
// Canonical.
func ParseCanonical(raw []byte) (*Canonical, error) {
	if len(raw) < HeaderLen {
		return nil, fmt.Errorf("pithmp3: canonical stream is shorter than the 48-byte header")
	}
	be32 := func(off int) uint32 {
		return binary.BigEndian.Uint32(raw[off:])
	}
	return &Canonical{
		Layer:            be32(0),
		BitrateKbps:      be32(4),
		HeaderSampleRate: be32(8),
		HeaderChannels:   be32(12),
		Channels:         be32(16),
		SampleRate:       be32(20),
		Frames:           be32(24),
		Samples:          binary.BigEndian.Uint64(raw[28:]),
		VBR:              be32(36) == 1,
		Raw:              raw,
	}, nil
}
