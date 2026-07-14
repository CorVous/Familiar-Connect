//! Vision-based image description using an [`LlmClient`] (subsystem 08; Python
//! `tools/image_describe.py`).
//!
//! Neutral base prompt; per-familiar `constraints` (from
//! `[prompt].image_description_constraints`) append to it, so authors add one
//! sentence rather than re-authoring the whole prompt.

use serde_json::json;

use crate::llm::{Content, LlmClient, Message};

/// Neutral base prompt — no character/persona constraints.
pub const DESCRIBE_PROMPT: &str = "Describe this image concisely for a chat \
    assistant. Focus on the main subject and any notable details.";

/// Call the vision model; return a text description of the image.
///
/// Sends a single user message with `[text, image_url]` blocks. `constraints`
/// append to the base prompt when non-blank; blank → base only (no trailing
/// space).
///
/// # Errors
/// Propagates a transport error from the underlying `chat` call.
pub async fn describe_image(
    llm: &dyn LlmClient,
    jpeg_base64: &str,
    media_type: &str,
    constraints: &str,
) -> anyhow::Result<String> {
    let extra = constraints.trim();
    let prompt = if extra.is_empty() {
        DESCRIBE_PROMPT.to_owned()
    } else {
        format!("{DESCRIBE_PROMPT} {extra}")
    };
    let content = Content::Blocks(vec![
        json!({"type": "text", "text": prompt}),
        json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{media_type};base64,{jpeg_base64}")},
        }),
    ]);
    let reply = llm.chat(vec![Message::new("user", content)]).await?;
    Ok(match reply.content {
        Content::Text(s) => s,
        Content::Blocks(_) => String::new(),
    })
}
