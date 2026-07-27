//! Closing "final reminder" block appended to every system prompt (subsystem 05;
//! Python `context/final_reminder.py`).
//!
//! Restates the current time (so the model doesn't drift on long-lived caches)
//! and enumerates text-channel sentinels, the per-mode operating directive, an
//! optional voice tool nudge, a focus + unread digest block, and any
//! post-history etiquette. Responders render it twice per turn (head copy with
//! `include_time=false`; tail copy with the clock, mode instruction, post-history
//! and guild name).
//!
//! The block grammar is a byte-exact prompt-format contract (spec 05 behavior 44).

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use chrono_tz::Tz;

use crate::focus::PRIVATE_MESSAGE_GUILD_NAME;

/// Per-mode operating directive. Intentionally duplicates the strings
/// `OperatingModeLayer` is configured with (spec 05 behavior 8) — keep in sync.
const VOICE_INSTRUCTION: &str =
    "You are speaking aloud. Keep replies short (one or two sentences). Avoid markdown.";
const TEXT_INSTRUCTION: &str =
    "You are chatting in a text channel. Markdown and multi-line replies are fine.";

fn mode_instruction(viewer_mode: &str) -> Option<&'static str> {
    match viewer_mode {
        "voice" => Some(VOICE_INSTRUCTION),
        "text" => Some(TEXT_INSTRUCTION),
        _ => None,
    }
}

/// Parenthetical count/ping suffix for one channel in the unread digest.
fn unread_suffix(unread: i64, pings: i64) -> String {
    if pings == 0 {
        if unread > 1 {
            format!(" ({unread})")
        } else {
            String::new()
        }
    } else {
        let word = if pings == 1 { "ping" } else { "pings" };
        if unread == pings {
            format!(" ({pings} {word})")
        } else {
            format!(" ({unread}, {pings} {word})")
        }
    }
}

/// Render `now` as `YYYY-MM-DD H:MMpm TZ` in `display_tz` (no leading zero on the
/// hour; `%Z` timezone abbreviation).
fn fmt_when(now: DateTime<Utc>, display_tz: &str) -> String {
    let tz: Tz = display_tz.parse().unwrap_or(Tz::UTC);
    let aware = now.with_timezone(&tz);
    let clock = aware.format("%I:%M%p").to_string();
    let clock = clock.trim_start_matches('0');
    format!(
        "{} {clock} {}",
        aware.format("%Y-%m-%d"),
        aware.format("%Z")
    )
}

fn channel_label(names: &HashMap<i64, String>, cid: i64) -> String {
    names
        .get(&cid)
        .map_or_else(|| format!("#{cid}"), |name| format!("#{name}"))
}

/// Unread-list item label: named channels **must** carry the numeric id so the
/// model can pass a valid `channel_id` to `shift_focus`.
///
/// A channel whose `guilds` entry is [`PRIVATE_MESSAGE_GUILD_NAME`] is a DM and
/// renders as `DM from <name> (id <cid>)` (or `DM (id <cid>)` unnamed); the id
/// is preserved either way because `shift_focus` still needs it.
fn channel_label_with_id(
    names: &HashMap<i64, String>,
    guilds: &HashMap<i64, String>,
    cid: i64,
) -> String {
    let name = names.get(&cid);
    if guilds.get(&cid).map(String::as_str) == Some(PRIVATE_MESSAGE_GUILD_NAME) {
        return name.map_or_else(
            || format!("DM (id {cid})"),
            |name| format!("DM from {name} (id {cid})"),
        );
    }
    name.map_or_else(|| format!("#{cid}"), |name| format!("#{name} (id {cid})"))
}

/// The closing "final reminder" block builder.
///
/// Construct with [`FinalReminder::new`], set the desired options, then
/// [`render`](FinalReminder::render). Mirrors Python `build_final_reminder(*,
/// viewer_mode, ...)` — the keyword-only args become builder setters. `now`
/// defaults to the wall clock when `include_time` and unset.
pub struct FinalReminder {
    viewer_mode: String,
    now: Option<DateTime<Utc>>,
    display_tz: String,
    include_time: bool,
    include_mode_instruction: bool,
    tools_enabled: bool,
    post_history_instructions: Option<String>,
    focus_channel_id: Option<i64>,
    unread_digest: Vec<(i64, (i64, i64))>,
    channel_names: HashMap<i64, String>,
    guild_names: HashMap<i64, String>,
    guild_name: Option<String>,
}

impl FinalReminder {
    /// New builder for `viewer_mode` (`"voice"` / `"text"`), `include_time`
    /// defaulting to `true` and `display_tz` to `"UTC"`.
    #[must_use]
    pub fn new(viewer_mode: impl Into<String>) -> Self {
        Self {
            viewer_mode: viewer_mode.into(),
            now: None,
            display_tz: "UTC".to_owned(),
            include_time: true,
            include_mode_instruction: false,
            tools_enabled: false,
            post_history_instructions: None,
            focus_channel_id: None,
            unread_digest: Vec::new(),
            channel_names: HashMap::new(),
            guild_names: HashMap::new(),
            guild_name: None,
        }
    }

    /// Set the reference time for the `It is now:` line.
    #[must_use]
    pub const fn now(mut self, now: DateTime<Utc>) -> Self {
        self.now = Some(now);
        self
    }
    /// Set the IANA display timezone for the clock.
    #[must_use]
    pub fn display_tz(mut self, tz: impl Into<String>) -> Self {
        self.display_tz = tz.into();
        self
    }
    /// Toggle the `It is now:` time line.
    #[must_use]
    pub const fn include_time(mut self, include: bool) -> Self {
        self.include_time = include;
        self
    }
    /// Toggle the per-mode operating directive.
    #[must_use]
    pub const fn include_mode_instruction(mut self, include: bool) -> Self {
        self.include_mode_instruction = include;
        self
    }
    /// Toggle the voice tool nudge (voice mode only).
    #[must_use]
    pub const fn tools_enabled(mut self, enabled: bool) -> Self {
        self.tools_enabled = enabled;
        self
    }
    /// Set the trailing post-history instructions (blank/None omits).
    #[must_use]
    pub fn post_history_instructions(mut self, text: impl Into<String>) -> Self {
        self.post_history_instructions = Some(text.into());
        self
    }
    /// Set the focus channel for the attention directive.
    #[must_use]
    pub const fn focus_channel_id(mut self, channel_id: i64) -> Self {
        self.focus_channel_id = Some(channel_id);
        self
    }
    /// Set the unread digest as `(channel_id, (unread, pings))` in render order.
    #[must_use]
    pub fn unread_digest(mut self, digest: Vec<(i64, (i64, i64))>) -> Self {
        self.unread_digest = digest;
        self
    }
    /// Set the channel-name map (for `#name (id …)` rendering).
    #[must_use]
    pub fn channel_names(mut self, names: HashMap<i64, String>) -> Self {
        self.channel_names = names;
        self
    }
    /// Set the channel→server-name map for the unread digest. An entry equal to
    /// [`PRIVATE_MESSAGE_GUILD_NAME`] marks the channel a DM, rendered as
    /// `DM from <name>` instead of `#<channel>`.
    #[must_use]
    pub fn guild_names(mut self, names: HashMap<i64, String>) -> Self {
        self.guild_names = names;
        self
    }
    /// Set the current server display name for the focus line.
    #[must_use]
    pub fn guild_name(mut self, name: impl Into<String>) -> Self {
        self.guild_name = Some(name.into());
        self
    }

    /// Render the block as a `\n`-joined string starting with `"---"`.
    #[must_use]
    pub fn render(&self) -> String {
        let mut lines: Vec<String> = vec!["---".to_owned()];

        if self.include_time {
            let now = self.now.unwrap_or_else(Utc::now);
            lines.push(String::new());
            lines.push(format!("It is now: {}", fmt_when(now, &self.display_tz)));
        }

        if self.viewer_mode == "text" {
            lines.push(String::new());
            lines.push("Special input:".to_owned());
            lines.push(String::new());
            lines.push("* `[@DisplayName]` - ping user".to_owned());
            lines.push("* `[\u{21a9} <message_id>]` - reply to message".to_owned());
        }

        if self.include_mode_instruction {
            if let Some(instruction) = mode_instruction(&self.viewer_mode) {
                lines.push(String::new());
                lines.push(instruction.to_owned());
            }
        }

        if self.tools_enabled && self.viewer_mode == "voice" {
            lines.push(String::new());
            lines.push(
                "Always speak at least a brief acknowledgement before calling a tool. \
                 Never reply with a tool call alone."
                    .to_owned(),
            );
        }

        if self.focus_channel_id.is_some() || !self.unread_digest.is_empty() {
            let focus_part = self.focus_channel_id.map_or_else(String::new, |cid| {
                let mut where_ = channel_label(&self.channel_names, cid);
                if self.guild_name.as_deref() == Some(PRIVATE_MESSAGE_GUILD_NAME) {
                    where_.push_str(" in a private message");
                } else if let Some(guild) = self.guild_name.as_deref().filter(|s| !s.is_empty()) {
                    use std::fmt::Write as _;
                    let _ = write!(where_, " in the \"{guild}\" server");
                }
                format!("Your attention is currently on {where_}.")
            });

            let active: Vec<(i64, (i64, i64))> = self
                .unread_digest
                .iter()
                .copied()
                .filter(|(_, (unread, _))| *unread > 0)
                .collect();
            let unread_part = if active.is_empty() {
                String::new()
            } else {
                let ch_list = active
                    .iter()
                    .map(|(cid, (unread, pings))| {
                        format!(
                            "{}{}",
                            channel_label_with_id(&self.channel_names, &self.guild_names, *cid),
                            unread_suffix(*unread, *pings)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let total: i64 = active.iter().map(|(_, (unread, _))| *unread).sum();
                let verb = if total == 1 { "is" } else { "are" };
                let noun = if total == 1 {
                    "a new message"
                } else {
                    "new messages"
                };
                format!(
                    "There {verb} {noun} in {ch_list} \
                     \u{2014} use shift_focus if it pulls your attention."
                )
            };

            let block = [focus_part, unread_part]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !block.is_empty() {
                lines.push(String::new());
                lines.push(block);
            }
        }

        if let Some(phi) = self.post_history_instructions.as_deref() {
            let trimmed = phi.trim();
            if !trimmed.is_empty() {
                lines.push(String::new());
                lines.push(trimmed.to_owned());
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::FinalReminder;
    use crate::focus::PRIVATE_MESSAGE_GUILD_NAME;
    use chrono::{TimeZone, Utc};
    use std::collections::HashMap;

    fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .unwrap()
    }

    fn names(pairs: &[(i64, &str)]) -> HashMap<i64, String> {
        pairs.iter().map(|(k, v)| (*k, (*v).to_owned())).collect()
    }

    #[test]
    fn text_mode_lists_ping_and_reply_sentinels() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .render();
        assert!(out.contains("It is now: 2026-05-04 2:30PM UTC"));
        assert!(out.contains("[@DisplayName]"));
        assert!(out.contains("[\u{21a9} <message_id>]"));
    }

    #[test]
    fn text_mode_no_silent_sentinel() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .render();
        assert!(!out.contains("`<silent>`"));
    }

    #[test]
    fn voice_mode_no_sentinels() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 1, 1, 9, 5))
            .render();
        assert!(out.contains("It is now: 2026-01-01 9:05AM UTC"));
        assert!(!out.contains("[@DisplayName]"));
        assert!(!out.contains("message_id"));
    }

    #[test]
    fn display_tz_converts_clock_and_abbrev() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 21, 30))
            .display_tz("America/Los_Angeles")
            .render();
        assert!(out.contains("It is now: 2026-05-04 2:30PM PDT"), "{out}");
    }

    #[test]
    fn display_tz_defaults_to_utc() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 21, 30))
            .render();
        assert!(out.contains("It is now: 2026-05-04 9:30PM UTC"));
    }

    #[test]
    fn starts_with_horizontal_rule() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 0, 0))
            .render();
        assert!(out.starts_with("---"));
    }

    #[test]
    fn include_time_false_omits_timestamp() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .include_time(false)
            .render();
        assert!(!out.contains("It is now:"));
        assert!(out.contains("[@DisplayName]"));
    }

    #[test]
    fn voice_mode_instruction_appended_when_requested() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .include_mode_instruction(true)
            .render();
        assert!(out.contains("You are speaking aloud"));
        assert!(out.contains("Avoid markdown"));
    }

    #[test]
    fn text_mode_instruction_appended_when_requested() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .include_mode_instruction(true)
            .render();
        assert!(out.contains("chatting in a text channel"));
        assert!(out.contains("Markdown"));
    }

    #[test]
    fn mode_instruction_omitted_by_default() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .render();
        assert!(!out.contains("You are speaking aloud"));
    }

    #[test]
    fn unknown_mode_with_instruction_flag_is_silent() {
        let out = FinalReminder::new("other")
            .now(at(2026, 5, 4, 14, 30))
            .include_mode_instruction(true)
            .render();
        assert!(!out.contains("You are speaking aloud"));
        assert!(!out.contains("Markdown"));
    }

    #[test]
    fn post_history_instructions_appended_when_provided() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .post_history_instructions("# Etiquette\n\nBe terse.")
            .render();
        assert!(out.contains("# Etiquette"));
        assert!(out.contains("Be terse."));
    }

    #[test]
    fn post_history_instructions_land_at_tail() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .include_mode_instruction(true)
            .post_history_instructions("ETIQUETTE_MARKER")
            .render();
        assert!(out.trim_end().ends_with("ETIQUETTE_MARKER"));
        assert!(
            out.find("You are speaking aloud").unwrap() < out.find("ETIQUETTE_MARKER").unwrap()
        );
    }

    #[test]
    fn post_history_instructions_omitted_by_default() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .render();
        assert!(!out.contains("Etiquette"));
    }

    #[test]
    fn blank_post_history_instructions_appends_nothing() {
        let out = FinalReminder::new("voice")
            .now(at(2026, 5, 4, 14, 30))
            .post_history_instructions("   ")
            .render();
        assert!(!out.ends_with("\n\n"));
    }

    // --- guild name ---------------------------------------------------------

    #[test]
    fn guild_name_named_alongside_channel_on_focus_line() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .guild_name("My Server")
            .render();
        let focus_line = out
            .lines()
            .find(|l| l.contains("attention is currently on"))
            .unwrap();
        assert!(focus_line.contains("#general"));
        assert!(focus_line.contains("My Server"));
    }

    #[test]
    fn no_guild_name_output_byte_for_byte_unchanged() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .render();
        let expected = "---\n\
             \n\
             It is now: 2026-05-04 2:30PM UTC\n\
             \n\
             Special input:\n\
             \n\
             * `[@DisplayName]` - ping user\n\
             * `[\u{21a9} <message_id>]` - reply to message\n\
             \n\
             Your attention is currently on #general.";
        assert_eq!(out, expected);
        assert!(!out.contains("server"));
    }

    #[test]
    fn guild_name_none_keeps_plain_focus_line() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .render();
        assert!(out.contains("Your attention is currently on #general."));
        assert!(!out.contains("server"));
    }

    #[test]
    fn guild_name_without_focus_channel_leaks_no_server_text() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .guild_name("My Server")
            .render();
        assert!(!out.contains("My Server"));
        assert!(!out.contains("server"));
    }

    #[test]
    fn server_clause_stays_on_focus_sentence_not_unread() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .unread_digest(vec![(20, (2, 0))])
            .channel_names(names(&[(123, "general"), (20, "random")]))
            .guild_name("My Server")
            .render();
        assert!(out.contains("server. There "));
        assert!(out.find("My Server").unwrap() < out.find("There are").unwrap());
    }

    #[test]
    fn empty_guild_name_identical_to_none() {
        let render = |guild: Option<&str>| {
            let mut fr = FinalReminder::new("text")
                .now(at(2026, 5, 4, 14, 30))
                .focus_channel_id(123)
                .channel_names(names(&[(123, "general")]));
            if let Some(g) = guild {
                fr = fr.guild_name(g);
            }
            fr.render()
        };
        assert_eq!(render(Some("")), render(None));
    }

    #[test]
    fn server_clause_exact_wording() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .guild_name("My Server")
            .render();
        let focus_line = out
            .lines()
            .find(|l| l.contains("attention is currently on"))
            .unwrap();
        assert!(focus_line.contains("in the \"My Server\" server."));
    }

    // --- private message ----------------------------------------------------

    #[test]
    fn dm_focus_line_reads_as_private_message() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "dm")]))
            .guild_name(PRIVATE_MESSAGE_GUILD_NAME)
            .render();
        assert!(out.contains("in a private message"));
        assert!(!out.contains("\"Private Message\" server"));
    }

    #[test]
    fn guild_focus_line_unchanged() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .guild_name("Aetheria")
            .render();
        assert!(out.contains("in the \"Aetheria\" server"));
    }

    #[test]
    fn no_guild_focus_line_has_neither_clause() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(123)
            .channel_names(names(&[(123, "general")]))
            .render();
        assert!(out.contains("Your attention is currently on #general."));
        assert!(!out.contains(" server"));
        assert!(!out.contains("private message"));
    }

    // --- focus channel ------------------------------------------------------

    #[test]
    fn focus_channel_directive_rendered() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(42)
            .render();
        assert!(out.contains("attention is currently on #42"));
        assert!(!out.contains("shift_focus"));
    }

    #[test]
    fn no_focus_channel_no_directive() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .render();
        assert!(!out.contains("attention is currently on"));
        assert!(!out.contains("shift_focus"));
    }

    #[test]
    fn focus_channel_before_post_history_instructions() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(7)
            .post_history_instructions("ETIQUETTE")
            .render();
        assert!(out.find("attention is currently on #7").unwrap() < out.find("ETIQUETTE").unwrap());
    }

    // --- unread digest ------------------------------------------------------

    #[test]
    fn unread_digest_rendered() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (3, 0)), (20, (1, 0))])
            .render();
        assert!(out.contains("new message"));
        assert!(out.contains("#10"));
        assert!(out.contains("#20"));
        assert!(out.contains("shift_focus"));
    }

    #[test]
    fn empty_unread_digest_renders_nothing() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![])
            .render();
        assert!(!out.contains("new message"));
        assert!(!out.contains("shift_focus"));
    }

    #[test]
    fn zero_count_channels_excluded() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (5, 0)), (20, (0, 0)), (30, (2, 0))])
            .render();
        assert!(out.contains("#10"));
        assert!(out.contains("#30"));
        assert!(!out.contains("#20"));
    }

    #[test]
    fn unread_digest_before_post_history_instructions() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(5, (2, 0))])
            .post_history_instructions("TAIL_MARKER")
            .render();
        assert!(out.find("new message").unwrap() < out.find("TAIL_MARKER").unwrap());
    }

    #[test]
    fn focus_and_unread_both_present() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .focus_channel_id(3)
            .unread_digest(vec![(10, (4, 0))])
            .render();
        assert!(out.contains("attention is currently on #3"));
        assert!(out.contains("#10 (4)"));
    }

    #[test]
    fn ping_subset_with_higher_unread_count() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (3, 1))])
            .render();
        assert!(out.contains("#10 (3, 1 ping)"));
        assert!(out.contains("shift_focus"));
    }

    #[test]
    fn all_unreads_are_pings_singular() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (1, 1))])
            .render();
        assert!(out.contains("#10 (1 ping)"));
    }

    #[test]
    fn mixed_unread_with_multiple_pings_plural() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (3, 2))])
            .render();
        assert!(out.contains("#10 (3, 2 pings)"));
    }

    #[test]
    fn no_pings_renders_count_only() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (2, 0))])
            .render();
        assert!(out.contains("#10 (2)"));
    }

    #[test]
    fn single_unread_no_ping_has_no_suffix() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(10, (1, 0))])
            .render();
        assert!(out.contains("#10 \u{2014}"));
        assert!(!out.contains("#10 ("));
    }

    #[test]
    fn named_unread_channel_surfaces_numeric_id() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(422_137_955_130_408_970, (2, 0))])
            .channel_names(names(&[(422_137_955_130_408_970, "the-annex")]))
            .render();
        assert!(out.contains("#the-annex"));
        assert!(out.contains("422137955130408970"));
        assert!(out.contains("shift_focus"));
    }

    // --- DM digest labels (PR #194) -----------------------------------------

    #[test]
    fn dm_unread_named_renders_dm_from() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(123, (1, 0))])
            .channel_names(names(&[(123, "Cor")]))
            .guild_names(names(&[(123, PRIVATE_MESSAGE_GUILD_NAME)]))
            .render();
        assert!(out.contains("DM from Cor (id 123)"));
        assert!(!out.contains("#123"));
        assert!(!out.contains("#Cor"));
    }

    #[test]
    fn dm_unread_unnamed_falls_back_to_dm_id() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(123, (1, 0))])
            .guild_names(names(&[(123, PRIVATE_MESSAGE_GUILD_NAME)]))
            .render();
        assert!(out.contains("DM (id 123)"));
    }

    #[test]
    fn guild_unread_entry_unchanged() {
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(20, (2, 0))])
            .channel_names(names(&[(20, "general")]))
            .guild_names(names(&[(20, "My Server")]))
            .render();
        assert!(out.contains("#general (id 20)"));
        assert!(!out.contains("DM from"));
    }

    #[test]
    fn dm_unread_with_pings_keeps_suffix() {
        // Guards the label / suffix boundary: the suffix is appended outside
        // the labeler, so the DM branch must not swallow the ping count.
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(123, (2, 2))])
            .channel_names(names(&[(123, "Cor")]))
            .guild_names(names(&[(123, PRIVATE_MESSAGE_GUILD_NAME)]))
            .render();
        assert!(out.contains("DM from Cor (id 123) (2 pings)"));
    }

    #[test]
    fn mixed_guild_and_dm_in_one_digest() {
        // Per-cid branch selection: guild and DM each keep their own form.
        let out = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(20, (2, 0)), (123, (1, 1))])
            .channel_names(names(&[(20, "general"), (123, "Cor")]))
            .guild_names(names(&[
                (20, "My Server"),
                (123, PRIVATE_MESSAGE_GUILD_NAME),
            ]))
            .render();
        assert!(out.contains("#general (id 20) (2)"));
        assert!(out.contains("DM from Cor (id 123) (1 ping)"));
    }

    #[test]
    fn omitting_guild_names_preserves_todays_output() {
        // New input defaults safely: no guild_names == today's behavior.
        let named = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(20, (2, 0))])
            .channel_names(names(&[(20, "general")]))
            .render();
        assert!(named.contains("#general (id 20)"));
        assert!(!named.contains("DM from"));
        // A DM channel with no guild_names map falls through to the old
        // no-name #{cid} rendering.
        let fallback = FinalReminder::new("text")
            .now(at(2026, 5, 4, 14, 30))
            .unread_digest(vec![(123, (1, 0))])
            .render();
        assert!(fallback.contains("#123"));
        assert!(!fallback.contains("DM"));
    }
}
