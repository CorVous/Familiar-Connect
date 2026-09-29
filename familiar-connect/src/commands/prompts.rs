//! prompts subcommand: read back mirrored LLM calls (subsystem 10).
//!
//! Answers the question the logs cannot: *what did the assembled prompt on turn
//! X actually contain?* Reads the `llm_calls` table the call mirror writes
//! (subsystems 01 + 03) straight off `history.db` — no Discord, no network, no
//! running bot.
//!
//! Rendered plain, not styled: the payload is a ~15 KB prompt destined for a
//! pager or a `grep`, and colour codes would fight both.

use std::path::Path;

use clap::Args;

use crate::history::store::{HistoryStore, LlmCallRow, NoopFtsIndex};

/// `prompts` arguments — which mirrored LLM calls to print.
#[derive(Args, Debug)]
pub struct PromptsArgs {
    /// Folder name of the character to read (overrides `FAMILIAR_ID`).
    #[arg(long, value_name = "ID")]
    pub familiar: Option<String>,
    /// Print every call made under this `turns.id`.
    #[arg(long, value_name = "TURN_ID")]
    pub turn: Option<i64>,
    /// Only calls from this slot (`fast` / `prose` / `background`).
    #[arg(long, value_name = "SLOT")]
    pub slot: Option<String>,
    /// Newest calls to print when no `--turn` is given.
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub limit: i64,
}

impl PromptsArgs {
    /// The selection these arguments describe.
    #[must_use]
    pub fn query(&self) -> Query {
        Query {
            turn: self.turn,
            slot: self.slot.clone(),
            limit: self.limit,
        }
    }
}

/// Resolve the familiar the same way `run` does, then print its mirrored calls.
///
/// # Panics
/// Never; a bad selection returns exit code `1`.
#[must_use]
pub fn main(args: &PromptsArgs) -> i32 {
    let root = crate::commands::run::default_familiars_root();
    let familiar_root = match crate::commands::run::resolve_familiar_root(
        args.familiar.as_deref(),
        std::env::var("FAMILIAR_ID").ok(),
        &root,
    ) {
        Ok(root) => root,
        Err(msg) => {
            eprintln!("{msg}");
            return 1;
        }
    };
    let id = familiar_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_owned();
    run(&familiar_root, &id, &args.query())
}

/// Selection for one `prompts` run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Query {
    /// `turns.id` to anchor on; `None` reads the newest calls instead.
    pub turn: Option<i64>,
    /// Slot filter (`fast` / `prose` / `background`); ignored with `turn`.
    pub slot: Option<String>,
    /// Rows to read when no turn is named.
    pub limit: i64,
}

/// Fetch the rows `q` selects.
///
/// # Errors
/// Propagates store faults (a missing or unreadable `history.db`).
pub fn fetch(
    store: &HistoryStore,
    familiar_id: &str,
    q: &Query,
) -> Result<Vec<LlmCallRow>, crate::history::StoreError> {
    q.turn.map_or_else(
        || store.recent_llm_calls(familiar_id, q.slot.as_deref(), q.limit),
        |turn_id| store.llm_calls_for_turn(familiar_id, turn_id),
    )
}

/// Render one mirrored call as a plain, greppable block.
#[must_use]
pub fn render_call(row: &LlmCallRow) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== llm_call id={} at={} slot={} model={} status={}",
        row.id,
        crate::support::time::iso_utc(row.created_at),
        row.slot.as_deref().unwrap_or("-"),
        row.model,
        row.status,
    );
    let _ = writeln!(
        out,
        "    turn_id={} turn_scope={} channel_id={} provider={}",
        opt_i64(row.turn_id),
        row.turn_scope.as_deref().unwrap_or("-"),
        opt_i64(row.channel_id),
        row.provider.as_deref().unwrap_or("-"),
    );
    let _ = writeln!(
        out,
        "    ttfb_ms={} ttft_ms={} total_ms={} est_in={} in={} out={} cached={}",
        opt_i64(row.ttfb_ms),
        opt_i64(row.ttft_ms),
        opt_i64(row.total_ms),
        opt_i64(row.est_in_tokens),
        opt_i64(row.in_tokens),
        opt_i64(row.out_tokens),
        opt_i64(row.cached),
    );
    if let Some(text) = &row.turn_content {
        let _ = writeln!(out, "--- turn ---\n{text}");
    }
    let _ = writeln!(out, "--- system prompt ---\n{}", row.system_prompt);
    let _ = writeln!(out, "--- messages ---\n{}", row.messages_json);
    let _ = writeln!(out, "--- response ---\n{}", row.response_text);
    if let Some(calls) = &row.tool_calls_json {
        let _ = writeln!(out, "--- tool calls ---\n{calls}");
    }
    if let Some(results) = &row.tool_results_json {
        let _ = writeln!(out, "--- tool results ---\n{results}");
    }
    out
}

fn opt_i64(v: Option<i64>) -> String {
    v.map_or_else(|| "-".to_owned(), |n| n.to_string())
}

/// Read `familiar_root/history.db` and print the selected calls.
///
/// Returns the process exit code (`0` on success, `1` when the store cannot be
/// opened). An empty selection is a success with a one-line note.
#[must_use]
pub fn run(familiar_root: &Path, familiar_id: &str, q: &Query) -> i32 {
    let db_path = familiar_root.join("history.db");
    if !db_path.exists() {
        eprintln!("No history.db at {}", db_path.display());
        return 1;
    }
    // No tantivy: the mirror lives entirely in SQLite, and taking the FTS
    // writer lock would fight a bot running against the same familiar.
    let store =
        match HistoryStore::open_with_fts(&db_path, Box::new(NoopFtsIndex), Box::new(NoopFtsIndex))
        {
            Ok(store) => store,
            Err(e) => {
                eprintln!("failed to open history.db: {e}");
                return 1;
            }
        };
    let rows = match fetch(&store, familiar_id, q) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("failed to read llm_calls: {e}");
            return 1;
        }
    };
    if rows.is_empty() {
        println!("no mirrored LLM calls matched");
        return 0;
    }
    for row in &rows {
        print!("{}", render_call(row));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{Query, fetch, render_call, run};
    use crate::history::store::{AppendLlmCall, AppendTurn, HistoryStore};

    fn call(system_prompt: &str, slot: &str, turn_id: Option<i64>) -> AppendLlmCall {
        AppendLlmCall {
            familiar_id: "aria".to_owned(),
            turn_id,
            turn_scope: Some("turn-1".to_owned()),
            channel_id: Some(5),
            slot: Some(slot.to_owned()),
            model: "anthropic/claude-haiku-4.5".to_owned(),
            status: "ok".to_owned(),
            system_prompt: system_prompt.to_owned(),
            messages_json: "[]".to_owned(),
            response_text: "Umu.".to_owned(),
            ..AppendLlmCall::default()
        }
    }

    #[test]
    fn fetch_by_turn_returns_only_that_turns_calls() {
        let store = HistoryStore::open(":memory:").expect("store");
        let turn = store
            .append_turn(AppendTurn::new("aria", 5, "user", "who is here?"))
            .expect("turn");
        store
            .append_llm_call(call("In the call: Ada", "fast", Some(turn.id)), 100)
            .expect("call");
        store
            .append_llm_call(call("other", "prose", None), 100)
            .expect("call");
        let rows = fetch(
            &store,
            "aria",
            &Query {
                turn: Some(turn.id),
                ..Query::default()
            },
        )
        .expect("fetch");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].system_prompt, "In the call: Ada");
        assert_eq!(rows[0].turn_content.as_deref(), Some("who is here?"));
    }

    #[test]
    fn fetch_recent_filters_by_slot() {
        let store = HistoryStore::open(":memory:").expect("store");
        store
            .append_llm_call(call("voice prompt", "fast", None), 100)
            .expect("call");
        store
            .append_llm_call(call("text prompt", "prose", None), 100)
            .expect("call");
        let rows = fetch(
            &store,
            "aria",
            &Query {
                turn: None,
                slot: Some("fast".to_owned()),
                limit: 10,
            },
        )
        .expect("fetch");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].system_prompt, "voice prompt");
    }

    #[test]
    fn render_shows_the_whole_system_prompt() {
        let store = HistoryStore::open(":memory:").expect("store");
        store
            .append_llm_call(call("In the call: Ada, Bel", "fast", None), 100)
            .expect("call");
        let rows = fetch(
            &store,
            "aria",
            &Query {
                limit: 1,
                ..Query::default()
            },
        )
        .expect("fetch");
        let text = render_call(&rows[0]);
        assert!(text.contains("--- system prompt ---"));
        assert!(text.contains("In the call: Ada, Bel"));
        assert!(text.contains("--- response ---"));
        assert!(text.contains("Umu."));
        assert!(text.contains("slot=fast"));
    }

    #[test]
    fn missing_database_fails_cleanly() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(run(dir.path(), "aria", &Query::default()), 1);
    }
}
