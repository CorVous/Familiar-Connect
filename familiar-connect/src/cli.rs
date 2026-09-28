//! CLI parsing + setup_logging (subsystem 10).
//!
//! `clap` (derive) replaces `argparse`; the subcommand set (`run` / `diagnose` /
//! `prompts` / `version`) and the repeatable `-v/--verbose` counter are a fixed
//! CLI contract, and there is deliberately **no** `sleep` subcommand
//! (test-pinned).
//! [`setup_logging`] installs a `tracing` subscriber whose event formatter
//! reproduces the [`StyledFormatter`](crate::log_style::StyledFormatter) wire
//! format, and pins the two-tier visibility (`warn` root, `familiar_connect` at
//! `info`) via the package logger floor.

use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::LazyLock;

use clap::{Args, CommandFactory, Parser, Subcommand};
use regex::Regex;
use tracing::{Event, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

use crate::commands;
use crate::log_style::{self as ls, LogLevel, LogRecord, StyledFormatter};

/// Top-level parser.
#[derive(Parser, Debug)]
#[command(
    name = "familiar-connect",
    version = crate::commands::version::VERSION,
    about = "familiar-connect CLI tool"
)]
pub struct Cli {
    /// Increase verbosity (can be repeated: -v, -vv, -vvv).
    #[arg(
        short = 'v',
        long = "verbose",
        action = clap::ArgAction::Count,
        global = true
    )]
    pub verbose: u8,

    /// The chosen subcommand; `None` for a bare invocation (prints help).
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// The available subcommands. Deliberately no `sleep` (removed; test-pinned).
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the Discord bot.
    Run(commands::run::RunArgs),
    /// Aggregate span timings from log files.
    Diagnose(DiagnoseArgs),
    /// Print mirrored LLM prompts/responses from `history.db`.
    Prompts(commands::prompts::PromptsArgs),
    /// Display package version.
    Version,
}

/// `diagnose` arguments — one or more log files (`-` for stdin).
#[derive(Args, Debug)]
pub struct DiagnoseArgs {
    /// One or more log files to aggregate (`-` for stdin).
    #[arg(value_name = "LOG_FILE", required = true, num_args = 1..)]
    pub paths: Vec<String>,
}

// ---------------------------------------------------------------------------
// setup_logging
// ---------------------------------------------------------------------------

/// A resolved log level. `Critical` has no `tracing` analog and maps to `error`
/// at the subscriber (tracing's most severe level).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedLevel {
    /// `DEBUG` (verbose ≥ 2).
    Debug,
    /// `INFO` (verbose == 1).
    Info,
    /// `WARNING` (verbose == 0 — the default).
    Warning,
    /// `ERROR` (explicit only).
    Error,
    /// `CRITICAL` (explicit only).
    Critical,
}

impl ResolvedLevel {
    /// The `EnvFilter` directive level string.
    const fn filter_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warning => "warn",
            // tracing has no CRITICAL/fatal; `error` is its most severe level.
            Self::Error | Self::Critical => "error",
        }
    }

    /// The package-logger floor: `min(level, INFO)` so package INFO stays
    /// visible even at root WARNING.
    const fn floored_to_info(self) -> Self {
        match self {
            Self::Debug => Self::Debug,
            _ => Self::Info,
        }
    }
}

/// Resolve the effective level from the verbose counter and optional explicit
/// name.
///
/// `verbose`: 0 → WARNING, 1 → INFO, ≥ 2 → DEBUG. An explicit `level`
/// (case-insensitive) overrides the counter.
///
/// # Errors
/// An unknown `level` name yields `Err` with the byte-stable `"Invalid log
/// level: <name>"` message.
pub fn resolve_log_level(verbose: u8, level: Option<&str>) -> Result<ResolvedLevel, String> {
    if let Some(level) = level {
        return match level.to_ascii_uppercase().as_str() {
            "DEBUG" => Ok(ResolvedLevel::Debug),
            "INFO" => Ok(ResolvedLevel::Info),
            "WARNING" => Ok(ResolvedLevel::Warning),
            "ERROR" => Ok(ResolvedLevel::Error),
            "CRITICAL" => Ok(ResolvedLevel::Critical),
            _ => Err(format!("Invalid log level: {level}")),
        };
    }
    Ok(match verbose {
        0 => ResolvedLevel::Warning,
        1 => ResolvedLevel::Info,
        _ => ResolvedLevel::Debug,
    })
}

static ANSI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").expect("valid ansi regex"));

fn strip_ansi(text: &str) -> String {
    ANSI_RE.replace_all(text, "").into_owned()
}

/// Extracts the `message` field of a `tracing` event into a string.
struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write as _;
            let _ = write!(self.0, "{value:?}");
        }
    }
}

/// A `tracing` event formatter that reproduces the `log_style` wire format via
/// [`StyledFormatter`] — the layout the `diagnose` grep parses.
struct StyledEventFormat;

impl<S, N> FormatEvent<S, N> for StyledEventFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        let level = match *event.metadata().level() {
            tracing::Level::TRACE | tracing::Level::DEBUG => LogLevel::Debug,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::WARN => LogLevel::Warning,
            tracing::Level::ERROR => LogLevel::Error,
        };
        let mut record = LogRecord::new(level, visitor.0);
        let mut line = StyledFormatter::new().format(&mut record);
        if ls::strip_enabled() {
            line = strip_ansi(&line);
        }
        writeln!(writer, "{line}")
    }
}

/// songbird's UDP receive task. Warns per packet on RTCP decrypt failures,
/// which upstream documents as non-fatal by design (the packet is still
/// forwarded with a fallback offset).
const SONGBIRD_UDP_RX_TARGET: &str = "songbird::driver::tasks::udp_rx";

/// Build the `EnvFilter` directive string: root level, `familiar_connect`
/// floor, third-party quieting.
///
/// The root level gates every dependency, and it defaults to `warn` — so
/// songbird's per-packet receive-path spam passes the filter at any verbosity
/// (#199). A target-scoped directive pins that one module to `error`; genuine
/// songbird errors elsewhere still show. `-vv` (root `debug`) is the "show me
/// everything" escape hatch and drops the directive.
fn log_filter_directive(resolved: ResolvedLevel) -> String {
    let base = format!(
        "{},familiar_connect={}",
        resolved.filter_str(),
        resolved.floored_to_info().filter_str()
    );
    if resolved == ResolvedLevel::Debug {
        base
    } else {
        format!("{base},{SONGBIRD_UDP_RX_TARGET}=error")
    }
}

/// Configure logging.
///
/// Installs a process-wide `tracing` subscriber with the `StyledFormatter` wire
/// format and the `EnvFilter` from [`log_filter_directive`].
/// Re-installation across a process is a no-op (`try_init`), so a second
/// call is harmless.
///
/// # Errors
/// Propagates the `resolve_log_level` error on an unknown explicit level.
pub fn setup_logging(verbose: u8, level: Option<&str>) -> Result<(), String> {
    let resolved = resolve_log_level(verbose, level)?;
    // colorama parity: strip ANSI when stderr is not an interactive terminal.
    ls::init(!std::io::stderr().is_terminal());
    let directive = log_filter_directive(resolved);
    let filter = EnvFilter::try_new(&directive).unwrap_or_else(|_| EnvFilter::new("warn"));
    // `try_init` fails only if a global subscriber is already installed; that is
    // benign here (a second setup_logging call), so the error is dropped.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .event_format(StyledEventFormat)
        .with_writer(std::io::stderr)
        .try_init();
    Ok(())
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

fn exit_code(code: i32) -> ExitCode {
    u8::try_from(code).map_or(ExitCode::FAILURE, ExitCode::from)
}

/// Program entry: load `.env`, parse, dispatch.
///
/// A bare invocation (no subcommand) prints help and exits `0`. `clap` handles
/// `--version` (prints and exits `0`) during parse, matching argparse.
#[must_use]
pub fn main() -> ExitCode {
    // Autoload `.env` before parsing; a missing file is
    // not an error.
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    let Some(command) = cli.command else {
        let mut cmd = Cli::command();
        let _ = cmd.print_help();
        println!();
        return ExitCode::SUCCESS;
    };

    if let Err(err) = setup_logging(cli.verbose, None) {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    let code = match command {
        Command::Run(args) => commands::run::run(&args),
        Command::Diagnose(args) => commands::diagnose::diagnose(&args.paths),
        Command::Prompts(args) => commands::prompts::main(&args),
        Command::Version => commands::version::run(),
    };
    exit_code(code)
}

#[cfg(test)]
mod tests {
    use super::{Cli, Command, ResolvedLevel, log_filter_directive, resolve_log_level};
    use crate::commands::version::VERSION;
    use clap::Parser;

    // --- parser shape ---

    #[test]
    fn parser_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn run_subcommand_registered() {
        let cli = Cli::try_parse_from(["familiar-connect", "run"]).expect("parse");
        assert!(matches!(cli.command, Some(Command::Run(_))));
    }

    #[test]
    fn run_subcommand_has_familiar_flag() {
        let cli =
            Cli::try_parse_from(["familiar-connect", "run", "--familiar", "aria"]).expect("parse");
        match cli.command {
            Some(Command::Run(args)) => assert_eq!(args.familiar.as_deref(), Some("aria")),
            _ => panic!("expected run"),
        }
    }

    #[test]
    fn diagnose_subcommand_registered() {
        let cli = Cli::try_parse_from(["familiar-connect", "diagnose", "somefile"]).expect("parse");
        match cli.command {
            Some(Command::Diagnose(args)) => assert_eq!(args.paths, vec!["somefile".to_owned()]),
            _ => panic!("expected diagnose"),
        }
    }

    #[test]
    fn prompts_subcommand_registered() {
        let cli =
            Cli::try_parse_from(["familiar-connect", "prompts", "--turn", "42"]).expect("parse");
        match cli.command {
            Some(Command::Prompts(args)) => {
                assert_eq!(args.turn, Some(42));
                // One call unless asked otherwise: prompts are ~15 KB each.
                assert_eq!(args.limit, 1);
            }
            _ => panic!("expected prompts"),
        }
    }

    #[test]
    fn prompts_subcommand_takes_familiar_slot_and_limit() {
        let cli = Cli::try_parse_from([
            "familiar-connect",
            "prompts",
            "--familiar",
            "aria",
            "--slot",
            "fast",
            "--limit",
            "3",
        ])
        .expect("parse");
        match cli.command {
            Some(Command::Prompts(args)) => {
                assert_eq!(args.familiar.as_deref(), Some("aria"));
                assert_eq!(args.slot.as_deref(), Some("fast"));
                assert_eq!(args.limit, 3);
            }
            _ => panic!("expected prompts"),
        }
    }

    #[test]
    fn version_subcommand_registered() {
        let cli = Cli::try_parse_from(["familiar-connect", "version"]).expect("parse");
        assert!(matches!(cli.command, Some(Command::Version)));
    }

    #[test]
    fn sleep_is_not_a_registered_subcommand() {
        // The runtime sleep path stays the only trigger — no `sleep` CLI verb.
        assert!(Cli::try_parse_from(["familiar-connect", "sleep"]).is_err());
    }

    #[test]
    fn verbose_is_count_based() {
        assert_eq!(
            Cli::try_parse_from(["familiar-connect", "version"])
                .unwrap()
                .verbose,
            0
        );
        assert_eq!(
            Cli::try_parse_from(["familiar-connect", "-v", "version"])
                .unwrap()
                .verbose,
            1
        );
        assert_eq!(
            Cli::try_parse_from(["familiar-connect", "-vv", "version"])
                .unwrap()
                .verbose,
            2
        );
    }

    #[test]
    fn version_flag_reports_version() {
        let err = Cli::try_parse_from(["familiar-connect", "--version"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(err.to_string().contains(VERSION));
    }

    #[test]
    fn bare_invocation_has_no_command() {
        let cli = Cli::try_parse_from(["familiar-connect"]).expect("parse");
        assert!(cli.command.is_none());
    }

    // --- resolve_log_level ---

    #[test]
    fn verbose_ladder() {
        assert_eq!(resolve_log_level(0, None), Ok(ResolvedLevel::Warning));
        assert_eq!(resolve_log_level(1, None), Ok(ResolvedLevel::Info));
        assert_eq!(resolve_log_level(2, None), Ok(ResolvedLevel::Debug));
        assert_eq!(resolve_log_level(5, None), Ok(ResolvedLevel::Debug));
    }

    #[test]
    fn explicit_levels() {
        assert_eq!(
            resolve_log_level(0, Some("DEBUG")),
            Ok(ResolvedLevel::Debug)
        );
        assert_eq!(resolve_log_level(0, Some("INFO")), Ok(ResolvedLevel::Info));
        assert_eq!(
            resolve_log_level(0, Some("WARNING")),
            Ok(ResolvedLevel::Warning)
        );
        assert_eq!(
            resolve_log_level(0, Some("ERROR")),
            Ok(ResolvedLevel::Error)
        );
        assert_eq!(
            resolve_log_level(0, Some("CRITICAL")),
            Ok(ResolvedLevel::Critical)
        );
    }

    #[test]
    fn explicit_level_is_case_insensitive() {
        assert_eq!(
            resolve_log_level(0, Some("debug")),
            Ok(ResolvedLevel::Debug)
        );
        assert_eq!(resolve_log_level(0, Some("InFo")), Ok(ResolvedLevel::Info));
    }

    #[test]
    fn invalid_level_is_error() {
        let err = resolve_log_level(0, Some("INVALID")).unwrap_err();
        assert!(err.contains("Invalid log level"));
    }

    #[test]
    fn explicit_level_overrides_verbose() {
        assert_eq!(
            resolve_log_level(2, Some("WARNING")),
            Ok(ResolvedLevel::Warning)
        );
    }

    #[test]
    fn package_floor_keeps_info_visible() {
        // Root WARNING still shows package INFO; root DEBUG floors to DEBUG.
        assert_eq!(
            ResolvedLevel::Warning.floored_to_info(),
            ResolvedLevel::Info
        );
        assert_eq!(ResolvedLevel::Debug.floored_to_info(), ResolvedLevel::Debug);
    }

    // --- EnvFilter directive (#199) ---

    #[test]
    fn directive_quiets_songbird_udp_rx_below_debug() {
        // Root WARNING lets every dependency's WARN through, so songbird's
        // per-packet "RTCP decryption failed" spam needs its own directive.
        assert_eq!(
            log_filter_directive(ResolvedLevel::Warning),
            "warn,familiar_connect=info,songbird::driver::tasks::udp_rx=error"
        );
        assert_eq!(
            log_filter_directive(ResolvedLevel::Info),
            "info,familiar_connect=info,songbird::driver::tasks::udp_rx=error"
        );
    }

    #[test]
    fn directive_restores_songbird_udp_rx_at_debug() {
        // `-vv` is the "show me everything" escape hatch — no quieting.
        assert_eq!(
            log_filter_directive(ResolvedLevel::Debug),
            "debug,familiar_connect=debug"
        );
    }

    #[test]
    fn directive_parses_as_env_filter() {
        // Guards the target path + syntax: a bad directive would silently fall
        // back to bare `warn` in setup_logging.
        for level in [
            ResolvedLevel::Debug,
            ResolvedLevel::Info,
            ResolvedLevel::Warning,
            ResolvedLevel::Error,
            ResolvedLevel::Critical,
        ] {
            let directive = log_filter_directive(level);
            assert!(
                tracing_subscriber::EnvFilter::try_new(&directive).is_ok(),
                "unparseable directive: {directive}"
            );
        }
    }
}
