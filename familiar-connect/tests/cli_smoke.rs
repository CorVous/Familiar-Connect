//! CLI subprocess smoke tests (subsystem 10; Python `test_cli.py` /
//! `test_version.py` subprocess halves).
//!
//! These drive the real `familiar-connect` binary end-to-end for the
//! discord-free subcommands (`version`, `diagnose`, bare-invocation help),
//! porting the argparse subprocess tests. The `run` subcommand needs a live
//! Discord token + the `discord` feature, so it is exercised only by the
//! in-module unit tests, not here.

use assert_cmd::Command;
use predicates::str::contains;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[test]
fn version_subcommand_prints_version() {
    // Ported from test_version.py::test_version_subcommand.
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .arg("version")
        .assert()
        .success()
        .stdout(contains(VERSION));
}

#[test]
fn version_flag_prints_version() {
    // Ported from test_version.py::test_version_flag.
    Command::cargo_bin("familiar-connect")
        .expect("binary")
        .arg("--version")
        .assert()
        .success()
        .stdout(contains(VERSION));
}

#[test]
fn bare_invocation_shows_usage() {
    // Ported from test_cli.py::test_parser_no_subcommand_shows_help — a bare
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
    // placeholder (test_diagnose_cmd.py::test_empty_log_shows_placeholder shape).
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
