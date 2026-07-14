//! Ported from Python `tests/test_image_describe.py` + `tests/test_image_tool.py`
//! — the vision describe call and the `view_image` tool (fetch injected via the
//! `ImageFetcher` seam, replacing the Python `_fetch_image_bytes` patch).
#![cfg(feature = "images")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::Value;

use familiar_connect::llm::{Content, LlmClient, LlmDelta, Message};
use familiar_connect::tools::image::{ImageFetcher, build_view_image_tool_with_fetcher};
use familiar_connect::tools::image_describe::{DESCRIBE_PROMPT, describe_image};
use familiar_connect::tools::registry::{ToolContext, ToolOutput};

// ---------------------------------------------------------------------------
// Doubles
// ---------------------------------------------------------------------------

struct CaptureLlm {
    reply: String,
    captured: Mutex<Vec<Message>>,
}

impl CaptureLlm {
    fn new(reply: &str) -> Self {
        Self {
            reply: reply.to_owned(),
            captured: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LlmClient for CaptureLlm {
    async fn chat(&self, messages: Vec<Message>) -> anyhow::Result<Message> {
        self.captured.lock().unwrap().extend(messages);
        Ok(Message::new("assistant", self.reply.clone()))
    }
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        _tools: Option<Vec<Value>>,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
        anyhow::bail!("stream not used")
    }
    fn slot(&self) -> Option<&str> {
        None
    }
    fn multimodal(&self) -> bool {
        false
    }
    fn tool_calling_enabled(&self) -> bool {
        false
    }
}

struct CannedFetcher {
    bytes: Vec<u8>,
}

#[async_trait]
impl ImageFetcher for CannedFetcher {
    async fn fetch(&self, _url: &str) -> anyhow::Result<Vec<u8>> {
        Ok(self.bytes.clone())
    }
}

fn tiny_png() -> Vec<u8> {
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    let img = RgbImage::from_pixel(10, 10, Rgb([100, 150, 200]));
    let mut buf = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, ImageFormat::Png)
        .unwrap();
    buf.into_inner()
}

fn last_text_block(m: &Message) -> String {
    match &m.content {
        Content::Blocks(blocks) => blocks
            .iter()
            .find(|b| b["type"] == "text")
            .and_then(|b| b["text"].as_str())
            .unwrap()
            .to_owned(),
        Content::Text(_) => panic!("expected block content"),
    }
}

fn image_url(m: &Message) -> String {
    match &m.content {
        Content::Blocks(blocks) => blocks
            .iter()
            .find(|b| b["type"] == "image_url")
            .and_then(|b| b["image_url"]["url"].as_str())
            .unwrap()
            .to_owned(),
        Content::Text(_) => panic!("expected block content"),
    }
}

// ---------------------------------------------------------------------------
// describe_image
// ---------------------------------------------------------------------------

fn last_message(llm: &CaptureLlm) -> Message {
    llm.captured.lock().unwrap().last().cloned().unwrap()
}

#[tokio::test]
async fn describe_calls_llm_with_vision_block() {
    let llm = CaptureLlm::new("a fluffy cat");
    let result = describe_image(&llm, "abc123", "image/jpeg", "")
        .await
        .unwrap();
    assert_eq!(result, "a fluffy cat");
    let msg = last_message(&llm);
    assert_eq!(msg.role, "user");
    assert!(matches!(msg.content, Content::Blocks(_)));
    assert_eq!(image_url(&msg), "data:image/jpeg;base64,abc123");
}

#[tokio::test]
async fn describe_uses_custom_media_type() {
    let llm = CaptureLlm::new("x");
    describe_image(&llm, "abc123", "image/png", "")
        .await
        .unwrap();
    assert!(image_url(&last_message(&llm)).starts_with("data:image/png;base64,"));
}

#[tokio::test]
async fn describe_base_prompt_is_neutral() {
    let llm = CaptureLlm::new("x");
    describe_image(&llm, "abc123", "image/jpeg", "")
        .await
        .unwrap();
    let text = last_text_block(&last_message(&llm));
    assert_eq!(text, DESCRIBE_PROMPT);
    assert!(!text.to_lowercase().contains("proper noun"));
}

#[tokio::test]
async fn describe_appends_constraints() {
    let llm = CaptureLlm::new("x");
    describe_image(&llm, "abc123", "image/jpeg", "Do not name brands.")
        .await
        .unwrap();
    let text = last_text_block(&last_message(&llm));
    assert!(text.starts_with(DESCRIBE_PROMPT));
    assert!(text.ends_with("Do not name brands."));
}

#[tokio::test]
async fn describe_empty_constraints_no_trailing_space() {
    let llm = CaptureLlm::new("x");
    describe_image(&llm, "abc123", "image/jpeg", "   ")
        .await
        .unwrap();
    assert_eq!(last_text_block(&last_message(&llm)), DESCRIBE_PROMPT);
}

// ---------------------------------------------------------------------------
// view_image tool
// ---------------------------------------------------------------------------

fn ctx_with_images(images: &[(&str, &str)], llm: Option<Arc<dyn LlmClient>>) -> ToolContext {
    let map: HashMap<String, String> = images
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let mut ctx = ToolContext::new("fam-1", 42, "text", "turn-1").with_images(map);
    if let Some(llm) = llm {
        ctx = ctx.with_description_llm(llm);
    }
    ctx
}

fn fetcher(bytes: Vec<u8>) -> Arc<dyn ImageFetcher> {
    Arc::new(CannedFetcher { bytes })
}

#[tokio::test]
async fn view_image_unknown_id_returns_error() {
    let tool = build_view_image_tool_with_fetcher("", fetcher(tiny_png()));
    let ctx = ctx_with_images(&[], None);
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_99"}), &ctx)
        .await
        .unwrap();
    let ToolOutput::Text(s) = out else {
        panic!("expected text error");
    };
    let data: Value = serde_json::from_str(&s).unwrap();
    assert!(data.get("error").is_some());
}

#[tokio::test]
async fn view_image_returns_image_result() {
    let llm: Arc<dyn LlmClient> = Arc::new(CaptureLlm::new("a cat"));
    let ctx = ctx_with_images(&[("img_0", "http://cdn.example.com/cat.png")], Some(llm));
    let tool = build_view_image_tool_with_fetcher("", fetcher(tiny_png()));
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_0"}), &ctx)
        .await
        .unwrap();
    let ToolOutput::Image(img) = out else {
        panic!("expected image result");
    };
    assert_eq!(img.description, "a cat");
    assert!(!img.jpeg_base64.is_empty());
}

#[tokio::test]
async fn view_image_constraints_flow_into_description() {
    let llm = Arc::new(CaptureLlm::new("a cat"));
    let ctx = ctx_with_images(
        &[("img_0", "http://cdn.example.com/cat.png")],
        Some(llm.clone()),
    );
    let tool = build_view_image_tool_with_fetcher("Do not name characters.", fetcher(tiny_png()));
    tool.handler
        .call(serde_json::json!({"image_id": "img_0"}), &ctx)
        .await
        .unwrap();
    let captured = llm.captured.lock().unwrap().clone();
    let found = captured.iter().any(|m| {
        matches!(&m.content, Content::Blocks(_))
            && last_text_block(m).contains("Do not name characters.")
    });
    assert!(found);
}

#[tokio::test]
async fn view_image_no_description_llm_degrades() {
    let ctx = ctx_with_images(&[("img_0", "http://cdn.example.com/img.png")], None);
    let tool = build_view_image_tool_with_fetcher("", fetcher(tiny_png()));
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_0"}), &ctx)
        .await
        .unwrap();
    let ToolOutput::Image(img) = out else {
        panic!("expected image result");
    };
    assert!(!img.description.is_empty());
    assert!(!img.jpeg_base64.is_empty());
}
