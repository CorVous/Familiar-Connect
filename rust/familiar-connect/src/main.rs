//! Binary entry point (subsystem 10; Python `__main__.py`).
//!
//! Thin per the scaffold: delegate straight to [`familiar_connect::cli::main`],
//! which parses args, configures logging, and dispatches to the subcommands.
//! `cli::main` owns the exit-code contract (`ExitCode`), so `__main__.py`'s
//! `sys.exit(main())` collapses into returning it.

fn main() -> std::process::ExitCode {
    familiar_connect::cli::main()
}
