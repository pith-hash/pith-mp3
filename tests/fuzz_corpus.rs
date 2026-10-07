//! Deterministic replay of the `fuzz/corpus/` seeds through the decoder.
//!
//! Ported from the upstream kit's `modhash/src/bin/fuzz.rs` codec harness:
//! the same splitmix64 PRNG (via `pith-digest`) and the same four mutation
//! modes drive every seed, so a failing iteration reproduces exactly on any
//! machine. Corrupt input is `Err` by contract — the only hard failure is a
//! panic, which `catch_unwind` turns into a located report.
//!
//! Opt-in: CI's fuzz job runs
//! `cargo test --release --locked -p pith-mp3 --features fuzz --test fuzz_corpus`.

#![cfg(feature = "fuzz")]
#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use pith_digest::SplitMix64;
use pith_mp3::{Limits, decode};

/// The four mutation modes every target is fuzzed with.
const MODES: [&str; 4] = ["random", "truncate", "bitflip", "repeat-insert"];

/// Fixed harness parameters: one shared PRNG stream, `CORPUS_ROUNDS` full
/// passes over the corpus in each mode. Deterministic by construction —
/// no time, no filesystem randomness, no environment.
const SEED: u64 = 0x5EED_4D50_33AA_11C7;
const CORPUS_ROUNDS: usize = 64;

/// Returns a value in `0..n`, or `0` when `n` is zero (the upstream
/// harness's `SplitMix64::below`).
fn below(rng: &mut SplitMix64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (rng.next_u64() % n as u64) as usize
}

/// Applies one mutation mode to `input`.
fn apply(mode: &str, rng: &mut SplitMix64, input: &[u8]) -> Vec<u8> {
    match mode {
        "random" => (0..=below(rng, 512))
            .map(|_| rng.next_u64() as u8)
            .collect(),
        "truncate" => {
            let end = below(rng, input.len() + 1);
            input[..end].to_vec()
        }
        "bitflip" => {
            // An empty input has no bit to flip. Returning it unchanged is the
            // honest semantic: this mode corrupts existing bytes and never
            // introduces new ones. Truncate can reach length zero, so this
            // case is reachable, not hypothetical.
            if input.is_empty() {
                return input.to_vec();
            }
            let mut out = input.to_vec();
            let flips = 1 + below(rng, 8);
            for _ in 0..flips {
                let i = below(rng, out.len());
                out[i] ^= 1 << below(rng, 8);
            }
            out
        }
        "repeat-insert" => {
            if input.is_empty() {
                return vec![rng.next_u64() as u8];
            }
            let at = below(rng, input.len());
            let take = 1 + below(rng, input.len());
            let end = at.saturating_add(take).min(input.len());
            let mut out = input[..at].to_vec();
            out.extend_from_slice(&input[at..end]);
            out.extend_from_slice(&input[at..]);
            out
        }
        other => unreachable!("unknown mutation mode {other}"),
    }
}

/// Loads every seed file from `fuzz/corpus/` sorted by name.
fn corpus() -> Vec<Vec<u8>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("corpus dir {} unreadable: {e}", dir.display()))
        .map(|e| e.expect("corpus dir entry").path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "corpus dir {} has no seed files",
        dir.display()
    );
    files
        .into_iter()
        .map(|p| std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
        .collect()
}

#[test]
fn corpus_replay_never_panics() {
    let corpus = corpus();
    let mut rng = SplitMix64::new(SEED);
    let iters = corpus.len() * CORPUS_ROUNDS;
    for i in 0..iters {
        let base = &corpus[i % corpus.len()];
        let input = apply(MODES[i % MODES.len()], &mut rng, base);
        // Panics inside the decoder are the failure the harness exists to
        // catch; catch_unwind turns them into a named crash report.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = decode(&input, &Limits::default());
        }));
        assert!(
            outcome.is_ok(),
            "decode panicked at iter {i} (seed {SEED}, mode {}, input {} bytes)",
            MODES[i % MODES.len()],
            input.len()
        );
    }
}

#[test]
fn mutator_preserves_its_contract() {
    // The same property pins the upstream harness's fuzz_coremode: each mode
    // guarantees a shape, so a broken mutator fails here instead of silently
    // weakening the corpus replay above.
    let mut rng = SplitMix64::new(SEED ^ 0xB07C);
    let base = b"\xff\xfb\x90\x64 resync-me";
    for i in 0..256 {
        match MODES[i % MODES.len()] {
            "random" => {} // arbitrary bytes by design
            "truncate" => {
                let out = apply("truncate", &mut rng, base);
                assert!(out.len() <= base.len());
                assert!(base.starts_with(&out));
            }
            "bitflip" => {
                let out = apply("bitflip", &mut rng, base);
                assert_eq!(out.len(), base.len());
            }
            "repeat-insert" => {
                // out = prefix ++ input[at..end] ++ input[at..]; the tail is
                // re-inserted, so the output is strictly longer than the input.
                let out = apply("repeat-insert", &mut rng, base);
                assert!(out.len() >= base.len() + 1);
            }
            _ => unreachable!(),
        }
    }
}
