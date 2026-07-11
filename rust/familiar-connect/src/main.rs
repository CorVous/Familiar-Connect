//! Binary entry point (subsystem 10). The real composition root lands in
//! `familiar_connect::cli` / `familiar_connect::commands::run` during the port;
//! this stub keeps the `familiar-connect` binary compiling.

fn main() {
    // Wired to `familiar_connect::cli::main()` once subsystem 10 is ported.
    std::process::exit(0);
}
