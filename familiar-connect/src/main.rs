//! Binary entry point (subsystem 10; Python `__main__.py`).
//!
//! Thin per the scaffold: delegate straight to [`familiar_connect::cli::main`],
//! which parses args, configures logging, and dispatches to the subcommands.
//! `cli::main` owns the exit-code contract (`ExitCode`), so `__main__.py`'s
//! `sys.exit(main())` collapses into returning it.

fn main() -> std::process::ExitCode {
    // Two TLS backends can coexist in the dependency tree (deepgram's newer
    // reqwest pulls rustls/aws-lc-rs; serenity/songbird/tungstenite use
    // rustls/ring). With both crate features enabled, rustls 0.23 refuses to
    // auto-select a process-level CryptoProvider and PANICS at the first
    // ClientConfig built without an explicit provider. Pin ring before any
    // TLS use; Err just means a default was already installed — fine.
    let _ = rustls::crypto::ring::default_provider().install_default();
    familiar_connect::cli::main()
}
