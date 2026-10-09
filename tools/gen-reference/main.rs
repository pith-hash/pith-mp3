//! Regenerates and verifies `reference.json` â€” the hex-exact decoded
//! PCM vectors for every conformance fixture, shared byte-for-byte across
//! the pith suite's Python, Node and Go SDKs.
//!
//! Usage:
//!
//! ```text
//! cargo run --locked --bin gen-reference            # regenerate reference.json
//! cargo run --locked --bin gen-reference -- verify  # check the committed copy is current
//! ```
//!
//! `verify` recomputes every vector and compares the resulting document
//! byte-for-byte against the committed `reference.json`, exiting
//! non-zero on drift. This is the CI gate ("Reference vectors are current").
//!
//! Vector definition per fixture: the fixture is decoded with
//! [`Limits::default`], and the vector records the stream facts plus the
//! SHA-256 digest (via `pith-digest`, the suite's own hash primitive) of the
//! decoded interleaved `i32` PCM in little-endian byte order, and the first
//! [`HEAD_BYTES`] PCM bytes in hex.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pith_digest::sha256;
use pith_mp3::{Layer, Limits, decode, decode_header};

/// Directory holding the conformance fixtures, relative to the crate root.
const FIXTURES_DIR: &str = "tests/fixtures";
/// The committed vector file this tool regenerates and verifies.
const REFERENCE_PATH: &str = "reference.json";
/// How many leading PCM bytes each vector carries in hex.
const HEAD_BYTES: usize = 64;

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match run(&mode, &root) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gen-reference: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Dispatches one CLI invocation against the crate rooted at `root`.
fn run(mode: &str, root: &Path) -> Result<(), String> {
    match mode {
        // Regenerate is the default so a bare `cargo run --bin gen-reference`
        // refreshes the file; CI always passes `verify` explicitly.
        "gen" | "" | "generate" => {
            let doc = generate(&root.join(FIXTURES_DIR))?;
            let path = root.join(REFERENCE_PATH);
            std::fs::write(&path, &doc).map_err(|e| format!("write {}: {e}", path.display()))?;
            println!("wrote {} ({} bytes)", path.display(), doc.len());
            Ok(())
        }
        "verify" => {
            let doc = generate(&root.join(FIXTURES_DIR))?;
            let path = root.join(REFERENCE_PATH);
            let committed = std::fs::read_to_string(&path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            if doc != committed {
                // Keep the diff actionable: report the first divergent line.
                let line = doc
                    .lines()
                    .zip(committed.lines())
                    .position(|(a, b)| a != b)
                    .map(|p| format!("line {}", p + 1))
                    .unwrap_or_else(|| "length".to_string());
                return Err(format!(
                    "{path_display} is stale ({line} differs); rerun \
                     `cargo run --locked --bin gen-reference` and commit",
                    path_display = path.display()
                ));
            }
            println!(
                "reference vectors are current: {} ({} bytes)",
                path.display(),
                doc.len()
            );
            Ok(())
        }
        other => Err(format!(
            "unknown mode {other:?}; expected no argument, `generate` or `verify`"
        )),
    }
}

/// Builds the full reference document: one vector per conformance fixture,
/// fixtures sorted by name so the output is deterministic across machines.
fn generate(fixtures_dir: &Path) -> Result<String, String> {
    let files = fixture_files(fixtures_dir)?;
    let mut doc = String::new();
    doc.push_str("{\n");
    doc.push_str("  \"schema\": \"pith-mp3/reference-vectors-v1\",\n");
    doc.push_str(
        "  \"note\": \"Decoded-PCM vectors per conformance fixture. pcm_sha256 is \
             SHA-256 (pith-digest) over the interleaved i32 samples in \
             little-endian byte order; pcm_head_hex carries the first 64 of \
             those bytes.\",\n",
    );
    doc.push_str("  \"vectors\": [\n");
    for (i, path) in files.iter().enumerate() {
        let vector = vector_for(path, i + 1 == files.len())?;
        for line in vector.lines() {
            let _ = writeln!(doc, "    {line}");
        }
    }
    doc.push_str("  ]\n");
    doc.push_str("}\n");
    Ok(doc)
}

/// Lists the audio fixtures (`.mp1`/`.mp2`/`.mp3`, never the `.ref.pcm`
/// references) sorted by file name.
fn fixture_files(fixtures_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(fixtures_dir)
        .map_err(|e| format!("read_dir {}: {e}", fixtures_dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        if matches!(ext, "mp1" | "mp2" | "mp3") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(format!("no fixtures found in {}", fixtures_dir.display()));
    }
    Ok(files)
}

/// Decodes one fixture and renders its vector as an indented JSON object
/// (opening brace on the first line, closing brace on the last). `is_last`
/// decides whether the closing brace carries a JSON comma.
fn vector_for(path: &Path, is_last: bool) -> Result<String, String> {
    let input = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("non-utf8 fixture name {}", path.display()))?
        .to_owned();

    let info = decode_header(&input).map_err(|e| format!("{name}: {e}"))?;
    let mp3 = decode(&input, &Limits::default()).map_err(|e| format!("{name}: {e}"))?;

    let pcm = pcm_bytes(&mp3.samples);
    let digest = sha256(&pcm).map_err(|e| format!("{name}: sha256: {e}"))?;
    let head_hex = hex(&pcm[..pcm.len().min(HEAD_BYTES)]);

    let mut out = String::new();
    let _ = writeln!(out, "  {{");
    let _ = writeln!(out, "    \"fixture\": \"{}\",", json_str(&name));
    let _ = writeln!(out, "    \"layer\": \"{}\",", layer_name(info.layer));
    let _ = writeln!(out, "    \"bitrate_kbps\": {},", info.bitrate_kbps);
    let _ = writeln!(out, "    \"header_sample_rate\": {},", info.sample_rate);
    let _ = writeln!(out, "    \"header_channels\": {},", info.channels);
    let _ = writeln!(out, "    \"channels\": {},", mp3.channels);
    let _ = writeln!(out, "    \"sample_rate\": {},", mp3.sample_rate);
    let _ = writeln!(out, "    \"frames\": {},", mp3.frames);
    let _ = writeln!(out, "    \"samples\": {},", mp3.samples.len());
    let _ = writeln!(out, "    \"vbr\": {},", mp3.vbr);
    let _ = writeln!(out, "    \"pcm_sha256\": \"{}\",", digest);
    let _ = writeln!(out, "    \"pcm_head_hex\": \"{}\"", head_hex);
    if is_last {
        let _ = write!(out, "  }}");
    } else {
        let _ = write!(out, "  }},");
    }
    Ok(out)
}

/// Interleaved `i32` PCM in little-endian byte order â€” the canonical byte
/// string every SDK hashes.
fn pcm_bytes(samples: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Lowercase hex.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Layer spelling used in the vectors.
fn layer_name(layer: Layer) -> &'static str {
    match layer {
        Layer::I => "I",
        Layer::II => "II",
        Layer::III => "III",
    }
}

/// Escapes a string for a JSON double-quoted literal.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn hex_encodes_lowercase() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn pcm_bytes_are_little_endian() {
        assert_eq!(pcm_bytes(&[1]), vec![1, 0, 0, 0]);
        assert_eq!(pcm_bytes(&[-1]), vec![0xff, 0xff, 0xff, 0xff]);
        assert_eq!(pcm_bytes(&[0x1234_5678]), vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn json_str_escapes() {
        assert_eq!(json_str("a\"b\\c\u{1}"), "a\\\"b\\\\c\\u0001");
        assert_eq!(json_str("plain.mp3"), "plain.mp3");
    }

    #[test]
    fn every_fixture_yields_a_vector() {
        let files = fixture_files(&root().join(FIXTURES_DIR)).unwrap();
        assert!(
            files.len() >= 9,
            "expected the full fixture set, got {files:?}"
        );
        for f in &files {
            vector_for(f, true).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        }
    }

    #[test]
    fn generated_document_is_wellformed_and_deterministic() {
        let a = generate(&root().join(FIXTURES_DIR)).unwrap();
        let b = generate(&root().join(FIXTURES_DIR)).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("{\n  \"schema\": \"pith-mp3/reference-vectors-v1\""));
        assert!(a.ends_with("  ]\n}\n"));
        // Every fixture appears exactly once.
        for f in fixture_files(&root().join(FIXTURES_DIR)).unwrap() {
            let name = f.file_name().unwrap().to_str().unwrap();
            assert_eq!(a.matches(&format!("\"fixture\": \"{name}\"")).count(), 1);
        }
    }

    #[test]
    fn committed_reference_is_current() {
        // Same check CI's `gen-reference verify` performs; runs on every
        // `cargo test` so drift cannot hide between CI runs.
        run("verify", &root()).expect("committed reference.json is current");
    }

    /// A throwaway crate root carrying copies of the real fixtures, so the
    /// CLI paths can be exercised end-to-end without touching the repo.
    fn scratch_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pith-mp3-gen-reference-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join(FIXTURES_DIR)).expect("scratch fixtures dir");
        for f in fixture_files(&root().join(FIXTURES_DIR)).unwrap() {
            std::fs::copy(&f, dir.join(FIXTURES_DIR).join(f.file_name().unwrap()))
                .expect("copy fixture");
        }
        dir
    }

    #[test]
    fn generate_then_verify_round_trip_in_scratch_root() {
        let dir = scratch_root("roundtrip");
        run("generate", &dir).expect("generate into scratch root");
        let written = std::fs::read_to_string(dir.join(REFERENCE_PATH)).unwrap();
        assert_eq!(written, generate(&dir.join(FIXTURES_DIR)).unwrap());
        run("verify", &dir).expect("verify passes right after generate");
        run("", &dir).expect("default mode regenerates");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn verify_detects_stale_vectors() {
        let dir = scratch_root("stale");
        run("generate", &dir).expect("generate");
        let path = dir.join(REFERENCE_PATH);
        let mut stale = std::fs::read_to_string(&path).unwrap();
        stale = stale.replacen("pcm_sha256", "pcm_sha255", 1);
        std::fs::write(&path, stale).unwrap();
        let err = run("verify", &dir).expect_err("stale copy must fail verify");
        assert!(err.contains("is stale"), "unactionable error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn verify_without_a_committed_file_fails_named() {
        let dir = scratch_root("missing");
        let err = run("verify", &dir).expect_err("no reference.json yet");
        assert!(err.contains("read "), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_mode_is_a_named_error() {
        let err = run("publish", &root()).expect_err("unknown mode");
        assert!(err.contains("unknown mode"), "unexpected error: {err}");
    }

    #[test]
    fn empty_fixture_dir_is_a_named_error() {
        let dir = std::env::temp_dir().join(format!(
            "pith-mp3-gen-reference-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join(FIXTURES_DIR)).unwrap();
        let err = generate(&dir.join(FIXTURES_DIR)).expect_err("no fixtures");
        assert!(err.contains("no fixtures found"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}