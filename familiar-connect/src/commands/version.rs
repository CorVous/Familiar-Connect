//! version subcommand (subsystem 10; Python commands/version.py).
//!
//! Prints one styled line carrying the package version. The version string comes
//! from `CARGO_PKG_VERSION` (Python read `familiar_connect.__version__`); the
//! layout mirrors Python's `tag('✨ Version') + word('familiar-connect') +
//! word(__version__)`.

use crate::log_style as ls;

/// The installed crate version (`CARGO_PKG_VERSION`), the Rust analog of
/// Python's `familiar_connect.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build the styled one-line version banner (separated from [`run`] so the
/// content is unit-testable without capturing stdout).
#[must_use]
pub fn version_line() -> String {
    format!(
        "{} {} {}",
        ls::tag("\u{2728} Version", ls::C),
        ls::word("familiar-connect", ls::W),
        ls::word(VERSION, ls::LC),
    )
}

/// Print the version banner; always succeeds (exit code `0`).
#[must_use]
pub fn run() -> i32 {
    println!("{}", version_line());
    0
}

#[cfg(test)]
mod tests {
    use super::{VERSION, run, version_line};
    use regex::Regex;

    fn strip_ansi(s: &str) -> String {
        Regex::new(r"\x1b\[[0-9;]*m")
            .expect("valid ansi regex")
            .replace_all(s, "")
            .into_owned()
    }

    #[test]
    fn version_is_semver_prefixed() {
        // Ported from test_version.py::test_version_format — starts with X.Y.Z.
        assert!(
            Regex::new(r"^\d+\.\d+\.\d+")
                .expect("valid")
                .is_match(VERSION),
            "version {VERSION} does not start with X.Y.Z"
        );
    }

    #[test]
    fn version_line_contains_version_and_name() {
        // Ported from test_version.py::test_version_subcommand (output contains
        // the version) — asserts on the built line rather than captured stdout.
        let stripped = strip_ansi(&version_line());
        assert!(stripped.contains(VERSION));
        assert!(stripped.contains("familiar-connect"));
        assert!(stripped.contains("Version"));
    }

    #[test]
    fn run_returns_zero() {
        assert_eq!(run(), 0);
    }
}
