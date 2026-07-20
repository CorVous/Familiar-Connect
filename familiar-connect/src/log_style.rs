//! Styled console log primitives + `StyledFormatter` wire format.
//!
//! Port of `familiar_connect/log_style.py`. Each call site composes its own log
//! string — emoji, label, colour choices live next to the log they describe.
//!
//! The rendered forms produced by [`tag`], [`kv`], [`kv_styled`], and [`word`]
//! are **wire formats**: they are regex-parsed downstream by the `diagnose` CLI
//! and by `StyledFormatter`. ANSI codes are single-parameter SGR only
//! (`ESC[<n>m`, colorama `Fore.*` 30–37/90–97, reset `ESC[0m`) — never emit
//! compound sequences (`ESC[1;33m`), they break both the formatter regex and the
//! diagnose parser (spec 01 §37).

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

use regex::Regex;

// ---------------------------------------------------------------------------
// init / strip flag
// ---------------------------------------------------------------------------

static STRIP: AtomicBool = AtomicBool::new(false);

/// Configure ANSI stripping; call once at process start (colorama parity).
///
/// Python delegates to `colorama.init(strip=strip, autoreset=False)`, which
/// wraps the output stream to strip ANSI on write when `strip` is true. The
/// primitives below always *build* ANSI; stripping is a property of the output
/// writer (subsystem 10). This stores the flag for that writer to consult.
pub fn init(strip: bool) {
    STRIP.store(strip, Ordering::Relaxed);
}

/// Whether ANSI output should be stripped by the writer (see [`init`]).
#[must_use]
pub fn strip_enabled() -> bool {
    STRIP.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Public colour constants (colorama Fore.* + Style.RESET_ALL, single-param SGR)
// ---------------------------------------------------------------------------

/// White (`Fore.WHITE`).
pub const W: &str = "\x1b[37m";
/// Cyan (`Fore.CYAN`).
pub const C: &str = "\x1b[36m";
/// Green (`Fore.GREEN`).
pub const G: &str = "\x1b[32m";
/// Yellow (`Fore.YELLOW`).
pub const Y: &str = "\x1b[33m";
/// Blue (`Fore.BLUE`).
pub const B: &str = "\x1b[34m";
/// Magenta (`Fore.MAGENTA`).
pub const M: &str = "\x1b[35m";
/// Red (`Fore.RED`).
pub const R: &str = "\x1b[31m";
/// Light green (`Fore.LIGHTGREEN_EX`).
pub const LG: &str = "\x1b[92m";
/// Light yellow (`Fore.LIGHTYELLOW_EX`).
pub const LY: &str = "\x1b[93m";
/// Light cyan (`Fore.LIGHTCYAN_EX`).
pub const LC: &str = "\x1b[96m";
/// Light magenta (`Fore.LIGHTMAGENTA_EX`).
pub const LM: &str = "\x1b[95m";
/// Light blue (`Fore.LIGHTBLUE_EX`).
pub const LB: &str = "\x1b[94m";
/// Light white (`Fore.LIGHTWHITE_EX`).
pub const LW: &str = "\x1b[97m";
/// Reset all (`Style.RESET_ALL`).
pub const RS: &str = "\x1b[0m";

// ---------------------------------------------------------------------------
// Public primitives
// ---------------------------------------------------------------------------

/// Bracketed label; brackets white, inner text `color`.
///
/// Render: `W + "[" + color + text + W + "]" + RS`.
#[must_use]
pub fn tag(text: &str, color: &str) -> String {
    format!("{W}[{color}{text}{W}]{RS}")
}

/// `key=value` chunk with default (white) key/value colours.
#[must_use]
pub fn kv(key: &str, val: &str) -> String {
    kv_styled(key, val, W, W)
}

/// `key=value` chunk with explicit key colour `kc` and value colour `vc`.
///
/// Render: `kc + key + "=" + RS + vc + val + RS`. The `=` is painted in the key
/// colour and `RS` sits between `=` and the value; the `diagnose` regex depends
/// on this exact shape (spec 01 §37).
#[must_use]
pub fn kv_styled(key: &str, val: &str, kc: &str, vc: &str) -> String {
    format!("{kc}{key}={RS}{vc}{val}{RS}")
}

/// Single coloured word: `color + text + RS`.
#[must_use]
pub fn word(text: &str, color: &str) -> String {
    format!("{color}{text}{RS}")
}

/// Truncate with `…` (U+2026) if longer than `limit` **Unicode scalars**.
///
/// Counts and slices by `char`, never bytes (DESIGN §4.9).
#[must_use]
pub fn trunc(text: &str, limit: usize) -> String {
    if text.chars().count() > limit {
        let head: String = text.chars().take(limit).collect();
        format!("{head}\u{2026}")
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// StyledFormatter
// ---------------------------------------------------------------------------

// Matches a leading `tag(text, color)` render: `W[ COLOR text W] RS`.
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\x1b\[\d+m\[\x1b\[\d+m([^\x1b]+)\x1b\[\d+m\]\x1b\[0m")
        .expect("static tag regex is valid")
});

/// Log severity, mirroring Python `logging` levels used by `StyledFormatter`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LogLevel {
    /// `DEBUG` — routine per-call timing, visible at `-vv`.
    Debug,
    /// `INFO` — package-level informational lines.
    #[default]
    Info,
    /// `WARNING`.
    Warning,
    /// `ERROR`.
    Error,
    /// `CRITICAL`.
    Critical,
}

impl LogLevel {
    /// The uppercase level label (`logging.getLevelName`).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

/// A record to be formatted.
///
/// Mirrors the fields of Python `logging.LogRecord` that `StyledFormatter`
/// consults. `exc_info`/`exc_text` model the traceback caching; in Rust the
/// "traceback" is any pre-formatted string.
#[derive(Clone, Debug)]
pub struct LogRecord {
    /// Severity.
    pub level: LogLevel,
    /// The already-composed message (may contain ANSI from the primitives).
    pub message: String,
    /// Raw exception text (Rust analog of `record.exc_info`); formatted once.
    pub exc_info: Option<String>,
    /// Cached formatted exception (Rust analog of `record.exc_text`).
    pub exc_text: Option<String>,
    /// Stack info appended after any exception text.
    pub stack_info: Option<String>,
}

impl LogRecord {
    /// A record with only a level and message; no exception/stack info.
    #[must_use]
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            exc_info: None,
            exc_text: None,
            stack_info: None,
        }
    }
}

/// Repaint the leading tag + append the level label for WARNING/ERROR/CRITICAL.
///
/// Keeps the DEBUG prefix; passes INFO through. Preserves stdlib append of
/// exception/stack traces (without this, `logger.exception(...)` drops the
/// traceback — pinned by three tests).
#[derive(Debug, Default, Clone, Copy)]
pub struct StyledFormatter;

impl StyledFormatter {
    /// Construct a formatter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Format `record`, mutating its `exc_text` cache like the Python formatter.
    #[allow(clippy::unused_self)] // instance API mirrors `StyledFormatter().format(...)`
    #[must_use]
    pub fn format(&self, record: &mut LogRecord) -> String {
        let msg = record.message.clone();
        let mut out = match record.level {
            LogLevel::Warning | LogLevel::Error | LogLevel::Critical => {
                let color = if record.level == LogLevel::Warning {
                    Y
                } else {
                    R
                };
                let label = record.level.label();
                TAG_RE.captures(&msg).map_or_else(
                    || format!("{color}{label}{RS}: {msg}"),
                    |caps| {
                        let inner = caps.get(1).map_or("", |m| m.as_str());
                        let end = caps.get(0).map_or(0, |m| m.end());
                        let rest = &msg[end..];
                        format!("{W}[{color}{inner}{W}]{RS} {color}{label}{RS}{rest}")
                    },
                )
            }
            LogLevel::Debug => format!("{LW}DEBUG{RS}: {msg}"),
            LogLevel::Info => msg,
        };
        // Mirror logging.Formatter: append exc_info / stack_info when present.
        if record.exc_info.is_some() && record.exc_text.is_none() {
            record.exc_text.clone_from(&record.exc_info);
        }
        if let Some(exc) = record.exc_text.clone() {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&exc);
        }
        if let Some(stack) = record.stack_info.clone() {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&stack);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        G, LC, LM, LogLevel, LogRecord, M, R, RS, StyledFormatter, W, Y, kv, kv_styled, tag, trunc,
        word,
    };
    use regex::Regex;
    use std::sync::LazyLock;

    static ANSI_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").expect("valid"));

    fn strip(s: &str) -> String {
        ANSI_RE.replace_all(s, "").into_owned()
    }

    fn fmt(level: LogLevel, msg: &str) -> String {
        StyledFormatter::new().format(&mut LogRecord::new(level, msg))
    }

    // --- primitive render shapes (wire formats) ---

    #[test]
    fn tag_render_shape() {
        assert_eq!(tag("span", LM), format!("{W}[{LM}span{W}]{RS}"));
    }

    #[test]
    fn kv_render_shape() {
        // `=` painted in key colour, RS between `=` and value.
        assert_eq!(kv("k", "v"), format!("{W}k={RS}{W}v{RS}"));
        assert_eq!(
            kv_styled("ms", "12", W, LC),
            format!("{W}ms={RS}{LC}12{RS}")
        );
    }

    #[test]
    fn word_render_shape() {
        assert_eq!(word("hi", G), format!("{G}hi{RS}"));
    }

    #[test]
    fn trunc_counts_unicode_scalars() {
        assert_eq!(trunc("hello", 200), "hello");
        assert_eq!(trunc("abcdef", 3), "abc\u{2026}");
        // exactly-limit does not truncate.
        assert_eq!(trunc("abc", 3), "abc");
        // emoji are single scalars, not bytes.
        assert_eq!(trunc("😀😀😀", 2), "😀😀\u{2026}");
    }

    // --- StyledFormatter (ported from tests/test_logging.py) ---

    #[test]
    fn formatter_info_no_prefix() {
        let msg = format!("{} body", tag("Hi", G));
        assert_eq!(strip(&fmt(LogLevel::Info, &msg)), "[Hi] body");
    }

    #[test]
    fn formatter_debug_keeps_prefix() {
        assert_eq!(strip(&fmt(LogLevel::Debug, "plain")), "DEBUG: plain");
    }

    #[test]
    fn formatter_warning_moves_label_after_tag_yellow() {
        let msg = format!("{} body", tag("Filter", M));
        let out = fmt(LogLevel::Warning, &msg);
        assert_eq!(strip(&out), "[Filter] WARNING body");
        assert!(out.contains(&format!("{Y}Filter")));
        assert!(out.contains(&format!("{Y}WARNING")));
    }

    #[test]
    fn formatter_error_moves_label_after_tag_red() {
        let msg = format!("{} body", tag("Content", M));
        let out = fmt(LogLevel::Error, &msg);
        assert_eq!(strip(&out), "[Content] ERROR body");
        assert!(out.contains(&format!("{R}Content")));
        assert!(out.contains(&format!("{R}ERROR")));
    }

    #[test]
    fn formatter_critical_uses_red() {
        let msg = format!("{} fatal", tag("Boom", M));
        let out = fmt(LogLevel::Critical, &msg);
        assert_eq!(strip(&out), "[Boom] CRITICAL fatal");
        assert!(out.contains(&format!("{R}CRITICAL")));
    }

    #[test]
    fn formatter_untagged_warning_falls_back_to_prefix() {
        assert_eq!(
            strip(&fmt(LogLevel::Warning, "no tag here")),
            "WARNING: no tag here"
        );
    }

    #[test]
    fn formatter_untagged_error_falls_back_to_prefix() {
        assert_eq!(
            strip(&fmt(LogLevel::Error, "bare error")),
            "ERROR: bare error"
        );
    }

    #[test]
    fn formatter_appends_exception_traceback() {
        // Rust has no Python traceback; model exc_info as pre-formatted text
        // carrying the exact substrings the Python test asserts.
        let exc = "Traceback (most recent call last):\n  ...\nRuntimeError: boom";
        let mut rec = LogRecord::new(LogLevel::Error, "Command failed");
        rec.exc_info = Some(exc.to_string());
        let out = strip(&StyledFormatter::new().format(&mut rec));
        assert!(out.contains("ERROR: Command failed"));
        assert!(out.contains("Traceback (most recent call last):"));
        assert!(out.contains("RuntimeError: boom"));
    }

    #[test]
    fn formatter_appends_exception_traceback_with_tag() {
        let exc = "Traceback (most recent call last):\n  ...\nValueError: kaboom";
        let msg = format!("{} startup failed", tag("Boot", M));
        let mut rec = LogRecord::new(LogLevel::Error, msg);
        rec.exc_info = Some(exc.to_string());
        let out = strip(&StyledFormatter::new().format(&mut rec));
        assert!(out.starts_with("[Boot] ERROR startup failed"));
        assert!(out.contains("Traceback (most recent call last):"));
        assert!(out.contains("ValueError: kaboom"));
    }

    #[test]
    fn formatter_does_not_get_double_traceback() {
        // Cached exc_text must not duplicate the traceback on repeated format.
        let exc = "Traceback (most recent call last):\n  ...\nRuntimeError: once";
        let mut rec = LogRecord::new(LogLevel::Error, "Command failed");
        rec.exc_info = Some(exc.to_string());
        let fmt = StyledFormatter::new();
        let first = strip(&fmt.format(&mut rec));
        let second = strip(&fmt.format(&mut rec));
        assert_eq!(first.matches("RuntimeError: once").count(), 1);
        assert_eq!(second.matches("RuntimeError: once").count(), 1);
    }
}
