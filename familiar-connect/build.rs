//! Embed a `git describe`-based version at compile time (consumed by
//! `commands::version`, subsystem 10). Mirrors the rust-template build script;
//! the `.git` rerun hints are dropped because the crate is nested under `rust/`.

use std::env;
use std::process::Command;

fn main() {
    let pkg_version = env::var("CARGO_PKG_VERSION").expect("cargo always sets CARGO_PKG_VERSION");
    let version = git_describe().map_or_else(
        || pkg_version.clone(),
        |describe| {
            let describe = describe.strip_prefix('v').unwrap_or(&describe).to_string();
            if describe.contains('.') {
                describe
            } else {
                format!("{pkg_version}+g{describe}")
            }
        },
    );
    println!("cargo::rustc-env=GIT_DESCRIBE_VERSION={version}");
}

fn git_describe() -> Option<String> {
    let output = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let describe = String::from_utf8(output.stdout).ok()?;
    let describe = describe.trim();
    if describe.is_empty() {
        None
    } else {
        Some(describe.to_string())
    }
}
