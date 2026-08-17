//! CLI subprocess smoke tests (subsystem 10).
//!
//! These drive the real `familiar-connect` binary end-to-end for the
//! discord-free subcommands (`version`, `diagnose`, bare-invocation help),
//! porting the argparse subprocess tests. The `run` subcommand needs a live
//! Discord token + the `discord` feature, so it is exercised only by the
//! in-module unit tests, not here.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[test]
fn version_subcommand_prints_version() {
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .arg("version")
        .assert()
        .success()
        .stdout(contains(VERSION));
}

#[test]
fn version_flag_prints_version() {
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .arg("--version")
        .assert()
        .success()
        .stdout(contains(VERSION));
}

#[test]
fn bare_invocation_shows_usage() {
    //  bare
    // invocation prints help and exits 0.
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .assert()
        .success()
        // clap 4 prints an "Usage:" section in its help output.
        .stdout(contains("Usage:"));
}

#[test]
fn diagnose_reads_stdin_and_shows_placeholder() {
    // `diagnose -` reads stdin; a line with no span markers yields the "no spans"
    // laceholder.
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .args(["diagnose", "-"])
        .write_stdin("nothing here\n")
        .assert()
        .success()
        .stdout(contains("no spans"));
}

#[test]
fn diagnose_aggregates_span_lines_from_stdin() {
    let input = "INFO [span] span=llm ms=100 status=ok\n\
         INFO [span] span=llm ms=200 status=ok\n";
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .args(["diagnose", "-"])
        .write_stdin(input)
        .assert()
        .success()
        .stdout(contains("llm"));
}

#[test]
fn diagnose_span_only_log_prints_no_cache_tables() {
    // The #206 tables are additive: a span-only log is unchanged.
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .args(["diagnose", "-"])
        .write_stdin("INFO [span] span=llm ms=100 status=ok\n")
        .assert()
        .success()
        .stdout(contains("LLM calls by slot / model").not());
}

#[test]
fn diagnose_reports_prompt_cache_from_llm_call_lines() {
    // Two `[LLM call]` lines, one cached and one cold (#206).
    let input = "INFO [LLM call] slot=fast model=anthropic/claude-haiku-4.5 status=ok \
         chars=8000 ttfb_ms=800 ttft_ms=850 total_ms=1200 provider=anthropic \
         est_in_tokens=2000 in_tokens=2100 out_tokens=80 cached=0 cal_ratio=1.050\n\
         INFO [LLM call] slot=prose model=z-ai/glm-5.2 status=ok \
         chars=8000 ttfb_ms=300 ttft_ms=350 total_ms=1200 provider=z-ai \
         est_in_tokens=2000 in_tokens=2100 out_tokens=80 cached=1890 cal_ratio=1.050\n";
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .args(["diagnose", "-"])
        .write_stdin(input)
        .assert()
        .success()
        .stdout(contains("LLM calls by slot / model"))
        .stdout(contains("cache hit vs miss latency (ms)"))
        .stdout(contains("token estimator accuracy by model"))
        .stdout(contains("anthropic/claude-haiku-4.5"))
        // prose reused 1890 of 2100 prompt tokens.
        .stdout(contains("90.0"));
}
