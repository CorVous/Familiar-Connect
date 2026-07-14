//! Request side of structured LLM output (subsystem 08; Python
//! `structured_request.py`).
//!
//! [`structured_output`](crate::structured_output) owns the PARSE side (raw
//! reply → tolerant JSON). This module owns the REQUEST side: declare a
//! [`Schema`], render its contract text ([`render_contract`]), and run the
//! call/parse/re-ask loop ([`request_structured`]). Domain validation (grounding
//! rails, id filtering, dedup) stays in the caller — this layer only guarantees
//! "you got JSON of the declared root shape, or you got nothing."
//!
//! The strategy seam (contract wording, parser coupling, the retry budget) is
//! isolated here so a swap to a different format or budget changes this module
//! and nothing else.

use crate::llm::{LlmClient, Message};
use crate::log_style as ls;
use crate::structured_output::{Expect, JsonResult, coerce_json};

/// The retry knob: one corrective re-ask after the first shape failure.
///
/// Enough to recover a model that fenced its JSON or added a preamble, without
/// doubling cost on every healthy call. Pinned so a change is deliberate.
pub const DEFAULT_MAX_RETRIES: i64 = 1;

/// Root shape of a structured reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    /// A top-level JSON object (`{...}`).
    Object,
    /// A top-level JSON array (`[...]`).
    Array,
}

impl Root {
    /// The token used in contract/correction wording (`"object"` / `"array"`).
    const fn as_str(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::Array => "array",
        }
    }

    /// The [`Expect`] shape to extract for this root.
    const fn expect(self) -> Expect {
        match self {
            Self::Object => Expect::Object,
            Self::Array => Expect::Array,
        }
    }
}

/// One field of a structured reply, rendered into the shape contract.
///
/// `placeholder` is the token shown verbatim in the JSON skeleton the model
/// copies — pick it to read as the value's type (`"\"<stance>\""`,
/// `"[<id>...]"`, `"<1-10>"`). `desc` is an optional gloss rendered as a bullet;
/// `required=false` only changes the rendered wording (enforcement is the
/// caller's rails, never this layer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// The JSON key.
    pub name: String,
    /// The value placeholder shown in the skeleton.
    pub placeholder: String,
    /// Optional plain-language gloss (rendered as a bullet when non-empty).
    pub desc: String,
    /// Whether the field is required (only affects the rendered wording).
    pub required: bool,
}

impl Field {
    /// A required field with no description.
    #[must_use]
    pub fn new(name: impl Into<String>, placeholder: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            placeholder: placeholder.into(),
            desc: String::new(),
            required: true,
        }
    }

    /// Builder: attach a description bullet.
    #[must_use]
    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = desc.into();
        self
    }

    /// Builder: mark the field optional (renders the ` (optional)` marker).
    #[must_use]
    pub const fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// The explanatory bullet line for this field.
    fn bullet(&self) -> String {
        let opt = if self.required { "" } else { " (optional)" };
        format!("- `{}`{opt}: {}", self.name, self.desc)
    }
}

/// Declarative shape of a structured LLM reply plus how to parse it.
///
/// Covers the three shapes the feature modules need:
/// * top-level ARRAY of items — [`Schema::array`];
/// * OBJECT wrapping one named list — [`Schema::object_container`]
///   (`{"candidates": [ {item}, … ]}`);
/// * flat OBJECT of fields — [`Schema::object`].
///
/// `fields` always describes the *item* (array element, container element, or
/// flat object). `container` is meaningful only for `root = Object`; pairing it
/// with `Array` is a programming error ([`Schema::try_new`] rejects it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    /// The item fields.
    pub fields: Vec<Field>,
    /// Top-level shape.
    pub root: Root,
    /// Container key wrapping the item list (object root only).
    pub container: Option<String>,
    /// Trailing "empty list when …" contract line.
    pub empty_note: String,
    /// Extra trailing constraint lines.
    pub constraints: Vec<String>,
}

impl Schema {
    /// Construct a schema, validating the `container`/`root` pairing.
    ///
    /// # Errors
    /// [`StructuredError::ContainerOnNonObject`] when `container` is set with a
    /// non-object root — a caller bug, not bad data (message contains
    /// `container is only valid`).
    pub fn try_new(
        fields: Vec<Field>,
        root: Root,
        container: Option<String>,
        empty_note: impl Into<String>,
        constraints: Vec<String>,
    ) -> Result<Self, StructuredError> {
        if container.is_some() && root != Root::Object {
            return Err(StructuredError::ContainerOnNonObject);
        }
        Ok(Self {
            fields,
            root,
            container,
            empty_note: empty_note.into(),
            constraints,
        })
    }

    /// A top-level ARRAY of items.
    #[must_use]
    pub const fn array(fields: Vec<Field>) -> Self {
        Self {
            fields,
            root: Root::Array,
            container: None,
            empty_note: String::new(),
            constraints: Vec::new(),
        }
    }

    /// A flat OBJECT of fields (no container).
    #[must_use]
    pub const fn object(fields: Vec<Field>) -> Self {
        Self {
            fields,
            root: Root::Object,
            container: None,
            empty_note: String::new(),
            constraints: Vec::new(),
        }
    }

    /// An OBJECT wrapping the item list under `container`.
    #[must_use]
    pub fn object_container(fields: Vec<Field>, container: impl Into<String>) -> Self {
        Self {
            fields,
            root: Root::Object,
            container: Some(container.into()),
            empty_note: String::new(),
            constraints: Vec::new(),
        }
    }

    /// Builder: set the trailing empty-note line.
    #[must_use]
    pub fn with_empty_note(mut self, note: impl Into<String>) -> Self {
        self.empty_note = note.into();
        self
    }

    /// Builder: set the trailing constraint lines.
    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Render one `{"name": <placeholder>, …}` JSON object skeleton.
    fn item_skeleton(&self) -> String {
        let inner = self
            .fields
            .iter()
            .map(|f| format!("\"{}\": {}", f.name, f.placeholder))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{{inner}}}")
    }
}

/// Outcome of a structured request.
///
/// `ok` is the success signal; on failure (every attempt fumbled the shape)
/// `value` is `None` and the caller degrades to its empty container. `attempts`
/// is how many LLM calls were spent (`1 + retries actually used`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StructuredReply {
    /// The parsed root value on success, else `None`.
    pub value: Option<serde_json::Value>,
    /// Whether a well-shaped reply was obtained.
    pub ok: bool,
    /// Number of LLM calls spent.
    pub attempts: u32,
}

/// Programming-error faults from schema construction (not data errors).
#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    /// A `container` was set on a non-object root.
    #[error("Schema.container is only valid when root='object'")]
    ContainerOnNonObject,
}

/// Render `schema` into the reply-shape contract appended to a prompt.
///
/// Produces a literal JSON skeleton (what models lock onto) plus optional
/// per-field bullets and any trailing constraints / empty-note. This is the
/// single authoritative wording.
#[must_use]
pub fn render_contract(schema: &Schema) -> String {
    let item = schema.item_skeleton();
    let skeleton = if schema.root == Root::Array {
        format!("[{item}, ...]")
    } else if let Some(container) = &schema.container {
        format!("{{\"{container}\": [{item}, ...]}}")
    } else {
        item
    };

    let mut lines = vec![format!(
        "Reply with JSON only, no prose or code fences: {skeleton}"
    )];
    let bullets: Vec<String> = schema
        .fields
        .iter()
        .filter(|f| !f.desc.is_empty())
        .map(Field::bullet)
        .collect();
    if !bullets.is_empty() {
        let per_item = schema.root == Root::Array || schema.container.is_some();
        lines.push(
            if per_item {
                "Each item's fields:"
            } else {
                "Fields:"
            }
            .to_string(),
        );
        lines.extend(bullets);
    }
    lines.extend(schema.constraints.iter().cloned());
    if !schema.empty_note.is_empty() {
        lines.push(schema.empty_note.clone());
    }
    lines.join("\n")
}

/// Ask `llm` for output matching `schema`; re-ask on a wrong shape.
///
/// `messages` is the fully-built prompt (the caller appends
/// [`render_contract`]'s text to its system message). Each attempt is parsed via
/// [`coerce_json`] to the schema's root; an unparseable or wrong-root reply is a
/// shape failure, answered with a corrective follow-up up to `max_retries`
/// times (`retries = max(0, max_retries)`; `retries + 1` attempts total).
///
/// Returns the parsed root value with `ok = true`, or `StructuredReply { ok:
/// false, .. }` when every attempt fumbled. The caller's `messages` slice is
/// never mutated.
///
/// # Errors
/// Transport failures from [`LlmClient::chat`] propagate unchanged — only SHAPE
/// problems are retried or degraded.
pub async fn request_structured(
    llm: &dyn LlmClient,
    messages: &[Message],
    schema: &Schema,
    max_retries: i64,
) -> anyhow::Result<StructuredReply> {
    let retries = max_retries.max(0);
    let mut convo: Vec<Message> = messages.to_vec();
    let mut attempts: u32 = 0;
    let mut last_problem = String::new();

    for attempt in 0..=retries {
        let reply = llm.chat(convo.clone()).await?;
        attempts += 1;
        let content = reply.content_str();
        let (problem, result) = shape_problem(&content, schema);
        match problem {
            None => {
                return Ok(StructuredReply {
                    value: result.value,
                    ok: true,
                    attempts,
                });
            }
            Some(problem) => {
                if attempt < retries {
                    let correction = correction(&problem, schema);
                    convo.push(Message::new("assistant", content));
                    convo.push(Message::new("user", correction));
                }
                last_problem = problem;
            }
        }
    }

    log_degraded(llm, schema, attempts, &last_problem);
    Ok(StructuredReply {
        value: None,
        ok: false,
        attempts,
    })
}

/// Return why `reply` fails `schema`'s root (or `None`) plus the parse.
///
/// The parsed [`JsonResult`] is handed back so the success path reuses it. The
/// problem string doubles as the model-facing correction reason, so it names the
/// expected shape plainly.
fn shape_problem(reply: &str, schema: &Schema) -> (Option<String>, JsonResult) {
    let result = coerce_json(reply, schema.root.expect());
    if !result.parsed_ok {
        return (Some("the reply was not valid JSON".to_string()), result);
    }
    let ok = match schema.root {
        Root::Object => result
            .value
            .as_ref()
            .is_some_and(serde_json::Value::is_object),
        Root::Array => result
            .value
            .as_ref()
            .is_some_and(serde_json::Value::is_array),
    };
    if ok {
        (None, result)
    } else {
        let msg = match schema.root {
            Root::Object => "the top-level value must be a JSON object ({...})",
            Root::Array => "the top-level value must be a JSON array ([...])",
        };
        (Some(msg.to_string()), result)
    }
}

/// Build the corrective follow-up that re-states the contract.
fn correction(problem: &str, schema: &Schema) -> String {
    format!(
        "Your previous reply could not be used: {problem}. Reply again with ONLY \
         the JSON described below — no prose, no code fences, no explanation.\n{}",
        render_contract(schema)
    )
}

/// Warn once when a request degrades to empty after exhausting retries.
fn log_degraded(llm: &dyn LlmClient, schema: &Schema, attempts: u32, problem: &str) {
    let slot = llm.slot().filter(|s| !s.is_empty()).unwrap_or("-");
    let msg = format!(
        "{} {} {} {} {}",
        ls::tag("Structured", ls::R),
        ls::kv_styled("degraded", schema.root.as_str(), ls::W, ls::R),
        ls::kv_styled("slot", slot, ls::W, ls::LC),
        ls::kv_styled("attempts", &attempts.to_string(), ls::W, ls::LW),
        ls::kv_styled("problem", problem, ls::W, ls::LW),
    );
    tracing::warn!(target: "familiar_connect.structured_request", "{msg}");
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_RETRIES, Field, Root, Schema, StructuredError, StructuredReply,
        render_contract, request_structured,
    };
    use crate::llm::{LlmClient, Message};
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use serde_json::{Value, json};
    use std::sync::Mutex;

    /// Minimal LLM stub: pops canned replies, records each prompt sent.
    struct FakeLlm {
        replies: Mutex<Vec<String>>,
        calls: Mutex<Vec<Vec<Message>>>,
        slot: Option<String>,
    }

    impl FakeLlm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().map(|s| (*s).to_string()).collect()),
                calls: Mutex::new(Vec::new()),
                slot: Some("test".to_string()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl LlmClient for FakeLlm {
        async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message> {
            self.calls.lock().unwrap().push(messages);
            let mut replies = self.replies.lock().unwrap();
            let content = if replies.is_empty() {
                "{}".to_string()
            } else {
                replies.remove(0)
            };
            Ok(Message::new("assistant", content))
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<Value>>,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<crate::llm::LlmDelta>>> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn slot(&self) -> Option<&str> {
            self.slot.as_deref()
        }
        fn multimodal(&self) -> bool {
            false
        }
        fn tool_calling_enabled(&self) -> bool {
            false
        }
    }

    /// Stub whose `chat` errors — stands in for a transport failure.
    struct BoomLlm;

    #[async_trait]
    impl LlmClient for BoomLlm {
        async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
            Err(anyhow::anyhow!("network down"))
        }
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<Value>>,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<crate::llm::LlmDelta>>> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn slot(&self) -> Option<&str> {
            Some("boom")
        }
        fn multimodal(&self) -> bool {
            false
        }
        fn tool_calling_enabled(&self) -> bool {
            false
        }
    }

    fn array_schema() -> Schema {
        Schema::array(vec![
            Field::new("text", "\"<one sentence>\"").with_desc("the reflection"),
            Field::new("ids", "[<id>...]")
                .with_desc("turn ids it draws from")
                .optional(),
        ])
        .with_empty_note("Reply [] when nothing stands out.")
    }

    fn container_schema() -> Schema {
        Schema::object_container(
            vec![
                Field::new("text", "\"<stance>\""),
                Field::new("turn_ids", "[<id>...]"),
            ],
            "candidates",
        )
        .with_empty_note("Empty list when nothing stands out.")
    }

    fn flat_schema() -> Schema {
        Schema::object(vec![Field::new("superseded_ids", "[<id>...]")])
            .with_constraints(vec!["Only use ids from the list below.".to_string()])
    }

    fn prompt() -> Vec<Message> {
        vec![
            Message::new("system", "do the thing"),
            Message::new("user", "data"),
        ]
    }

    // --- render_contract ---------------------------------------------------

    #[test]
    fn array_root_renders_skeleton_bullets_and_empty_note() {
        let text = render_contract(&array_schema());
        assert!(text.contains(r#"[{"text": "<one sentence>", "ids": [<id>...]}, ...]"#));
        assert!(text.contains("Each item's fields:"));
        assert!(text.contains("- `text`: the reflection"));
        assert!(text.contains("- `ids` (optional): turn ids it draws from"));
        assert!(text.contains("Reply [] when nothing stands out."));
    }

    #[test]
    fn object_container_wraps_item_list() {
        let text = render_contract(&container_schema());
        assert!(
            text.contains(r#"{"candidates": [{"text": "<stance>", "turn_ids": [<id>...]}, ...]}"#)
        );
        assert!(text.contains("Empty list when nothing stands out."));
    }

    #[test]
    fn flat_object_renders_fields_and_constraints() {
        let text = render_contract(&flat_schema());
        assert!(text.contains(r#"{"superseded_ids": [<id>...]}"#));
        // No field descs ⇒ no bullet block.
        assert!(!text.contains("Fields:"));
        assert!(text.contains("Only use ids from the list below."));
    }

    #[test]
    fn container_on_array_root_is_rejected() {
        let err = Schema::try_new(
            vec![Field::new("x", "1")],
            Root::Array,
            Some("oops".to_string()),
            "",
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, StructuredError::ContainerOnNonObject));
        assert!(err.to_string().contains("container is only valid"));
    }

    // --- request_structured ------------------------------------------------

    #[tokio::test]
    async fn object_success_first_try() {
        let llm = FakeLlm::new(&[r#"{"candidates": [{"text": "x", "turn_ids": [1]}]}"#]);
        let result = request_structured(&llm, &prompt(), &container_schema(), DEFAULT_MAX_RETRIES)
            .await
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.attempts, 1);
        assert_eq!(
            result.value,
            Some(json!({"candidates": [{"text": "x", "turn_ids": [1]}]}))
        );
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn array_success_first_try() {
        let llm = FakeLlm::new(&[r#"[{"text": "a"}]"#]);
        let result = request_structured(&llm, &prompt(), &array_schema(), DEFAULT_MAX_RETRIES)
            .await
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.value, Some(json!([{"text": "a"}])));
    }

    #[tokio::test]
    async fn retries_then_succeeds_with_correction() {
        let llm = FakeLlm::new(&["not json at all", r#"{"superseded_ids": [7]}"#]);
        let result = request_structured(&llm, &prompt(), &flat_schema(), 1)
            .await
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.attempts, 2);
        assert_eq!(result.value, Some(json!({"superseded_ids": [7]})));

        // The second call carried the bad reply + a corrective user turn.
        let second = llm.calls.lock().unwrap()[1].clone();
        let n = second.len();
        assert_eq!(second[n - 2].role, "assistant");
        assert_eq!(second[n - 2].content_str(), "not json at all");
        assert_eq!(second[n - 1].role, "user");
        assert!(second[n - 1].content_str().contains("could not be used"));
        assert!(second[n - 1].content_str().contains("superseded_ids"));
    }

    #[tokio::test]
    async fn degrades_after_exhausting_retries() {
        let llm = FakeLlm::new(&["nope", "still nope"]);
        let result = request_structured(&llm, &prompt(), &flat_schema(), 1)
            .await
            .unwrap();
        assert_eq!(
            result,
            StructuredReply {
                value: None,
                ok: false,
                attempts: 2,
            }
        );
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn wrong_root_type_is_a_shape_failure() {
        // An object site that gets a bare array must re-ask, not accept it.
        let llm = FakeLlm::new(&[r#"["a", "b"]"#, r#"{"superseded_ids": []}"#]);
        let result = request_structured(&llm, &prompt(), &flat_schema(), 1)
            .await
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.value, Some(json!({"superseded_ids": []})));
        assert_eq!(result.attempts, 2);
    }

    #[tokio::test]
    async fn zero_retries_takes_one_attempt_only() {
        let llm = FakeLlm::new(&["garbage"]);
        let result = request_structured(&llm, &prompt(), &flat_schema(), 0)
            .await
            .unwrap();
        assert!(!result.ok);
        assert_eq!(result.attempts, 1);
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn does_not_mutate_caller_messages() {
        let llm = FakeLlm::new(&["bad", r#"{"superseded_ids": []}"#]);
        let messages = prompt();
        let _ = request_structured(&llm, &messages, &flat_schema(), 1)
            .await
            .unwrap();
        // Caller's list is untouched even though a retry happened.
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn fenced_json_is_accepted() {
        let llm = FakeLlm::new(&["```json\n{\"superseded_ids\": [1]}\n```"]);
        let result = request_structured(&llm, &prompt(), &flat_schema(), 1)
            .await
            .unwrap();
        assert!(result.ok);
        assert_eq!(result.attempts, 1);
        assert_eq!(result.value, Some(json!({"superseded_ids": [1]})));
    }

    #[tokio::test]
    async fn transport_errors_propagate() {
        let err = request_structured(&BoomLlm, &prompt(), &flat_schema(), DEFAULT_MAX_RETRIES)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("network down"));
    }

    #[test]
    fn default_max_retries_is_one() {
        assert_eq!(DEFAULT_MAX_RETRIES, 1);
    }
}
