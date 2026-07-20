//! Shared channel-history serialization for the focus tools (subsystem 08;
//! Python `tools/channel_view.py`).
//!
//! `read_channel` and `shift_focus` render recent turns through
//! [`serialize_turns`], which surfaces *conversation only*: `role="tool"` turns
//! (the familiar's own serialized bookkeeping) and empty/whitespace turns
//! (tool-call scaffolding husks) are dropped. That exclusion also closes a
//! recursion — a preview's tool dump can never be re-embedded in a later
//! preview's recent-turns window and compound until context overflow.

use serde::Serialize;

use crate::history::store::HistoryTurn;
use crate::support::time::iso_utc;

/// A single serialized conversation turn.
///
/// Field order (`id, role, author, content, timestamp`) is the wire shape; serde
/// serializes struct fields in declaration order, so the emitted JSON key order
/// is stable regardless of the `serde_json` map backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TurnView {
    /// Engine-assigned turn id.
    pub id: i64,
    /// `"user"` / `"assistant"` / ….
    pub role: String,
    /// Display name, falling back to username, else `null`.
    pub author: Option<String>,
    /// Turn text (verbatim).
    pub content: String,
    /// ISO-8601 UTC write time.
    pub timestamp: String,
}

/// Render conversation turns as `{id, role, author, content, timestamp}`.
///
/// Excludes `role="tool"` turns (bookkeeping payloads) and empty-content turns
/// (tool-call scaffolding husks) so previews carry only real user/assistant
/// messages. Surviving turns pass through verbatim. Idempotent.
#[must_use]
pub fn serialize_turns(turns: &[HistoryTurn]) -> Vec<TurnView> {
    let mut result = Vec::new();
    for t in turns {
        if t.role == "tool" || t.content.trim().is_empty() {
            continue;
        }
        // Python: `t.author.display_name or t.author.username` — a non-empty
        // display_name wins, otherwise fall through to the username value *as
        // is* (including an empty string or `None`, never re-filtered).
        let author = t.author.as_ref().and_then(|a| {
            a.display_name
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| a.username.clone())
        });
        result.push(TurnView {
            id: t.id,
            role: t.role.clone(),
            author,
            content: t.content.clone(),
            timestamp: iso_utc(t.timestamp),
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Author;
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    fn turn(id: i64, role: &str, content: &str, author: Option<Author>) -> HistoryTurn {
        HistoryTurn {
            id,
            timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                + chrono::Duration::seconds(id),
            role: role.to_owned(),
            author,
            content: content.to_owned(),
            channel_id: 0,
            mode: None,
            platform_message_id: None,
            reply_to_message_id: None,
            guild_id: None,
            arrived_at: None,
            consumed_at: None,
            pings_bot: false,
        }
    }

    #[test]
    fn tool_and_empty_turns_excluded_only_real_messages() {
        let payload = serde_json::to_string(
            &(0..50)
                .map(|n| json!({"id": n, "content": format!("msg {n}")}))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let window = vec![
            turn(1, "user", "hey", None),
            turn(2, "assistant", "", None),
            turn(3, "tool", &payload, None),
            turn(4, "assistant", "hello!", None),
            turn(5, "user", "   ", None),
        ];
        let out = serialize_turns(&window);
        assert!(out.iter().all(|e| e.role != "tool"));
        assert!(out.iter().all(|e| !e.content.trim().is_empty()));
        let pairs: Vec<(&str, &str)> = out
            .iter()
            .map(|e| (e.role.as_str(), e.content.as_str()))
            .collect();
        assert_eq!(pairs, [("user", "hey"), ("assistant", "hello!")]);
    }

    #[test]
    fn user_and_assistant_content_pass_through_verbatim() {
        let out = serialize_turns(&[
            turn(1, "user", "hello there", None),
            turn(2, "assistant", "general kenobi", None),
        ]);
        assert_eq!(out[0].content, "hello there");
        assert_eq!(out[1].content, "general kenobi");
    }

    #[test]
    fn author_resolution_prefers_display_name() {
        let author = Author::new(
            "discord",
            "42",
            Some("cor_handle".into()),
            Some("Cor".into()),
        );
        let out = serialize_turns(&[turn(1, "user", "hi", Some(author))]);
        assert_eq!(out[0].author.as_deref(), Some("Cor"));
    }

    #[test]
    fn author_resolution_falls_back_to_username() {
        let author = Author::new(
            "discord",
            "42",
            Some("cor_handle".into()),
            Some(String::new()),
        );
        let out = serialize_turns(&[turn(1, "user", "hi", Some(author))]);
        assert_eq!(out[0].author.as_deref(), Some("cor_handle"));
    }

    #[test]
    fn authorless_turn_resolves_to_none() {
        let out = serialize_turns(&[turn(1, "user", "hi", None)]);
        assert!(out[0].author.is_none());
    }

    #[test]
    fn huge_tool_payload_does_not_inflate_output() {
        let huge = "x".repeat(500_000);
        let serialized =
            serde_json::to_string(&serialize_turns(&[turn(1, "tool", &huge, None)])).unwrap();
        assert!(!serialized.contains(&huge));
        assert!(serialized.len() < 1_000);
    }

    #[test]
    fn dict_shape_is_stable() {
        let out = serialize_turns(&[turn(7, "user", "hi", None)]);
        let s = serde_json::to_string(&out[0]).unwrap();
        // Keys emitted in declaration order.
        let id_pos = s.find("\"id\"").unwrap();
        let role_pos = s.find("\"role\"").unwrap();
        let author_pos = s.find("\"author\"").unwrap();
        let content_pos = s.find("\"content\"").unwrap();
        let ts_pos = s.find("\"timestamp\"").unwrap();
        assert!(id_pos < role_pos && role_pos < author_pos);
        assert!(author_pos < content_pos && content_pos < ts_pos);
    }

    #[test]
    fn recursion_closed_when_serialized_dump_re_ingested() {
        let prior: Vec<HistoryTurn> = (0..50)
            .map(|n| turn(n, "user", &format!("msg {n}"), None))
            .collect();
        let dump = serde_json::to_string(&serialize_turns(&prior)).unwrap();
        let tool_turn = turn(99, "tool", &dump, None);
        assert!(serialize_turns(&[tool_turn]).is_empty());
    }

    #[test]
    fn serialize_output_is_idempotent() {
        let turns = vec![
            turn(1, "user", "hi", None),
            turn(2, "tool", "big dump", None),
            turn(3, "assistant", "ok", None),
        ];
        assert_eq!(serialize_turns(&turns), serialize_turns(&turns));
    }
}
