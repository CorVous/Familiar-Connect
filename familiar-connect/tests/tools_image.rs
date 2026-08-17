//! The vision describe call and the `view_image` tool (fetch injected via the
//! `ImageFetcher` seam).
#![cfg(feature = "images")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use familiar_connect::llm::{Content, LlmClient, Message};
use familiar_connect::tools::agentic::{serialize_image_result, tool_content_as_text};
use familiar_connect::tools::image::{
    GuardedFetcher, ImageFetcher, build_view_image_tool_with_fetcher,
};
use familiar_connect::tools::image_describe::{DESCRIBE_PROMPT, describe_image};
use familiar_connect::tools::image_policy::{HostResolver, ImageUrlPolicy, UrlGuard};
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

    /// How many `chat` calls this double has served.
    fn calls(&self) -> usize {
        self.captured.lock().unwrap().len()
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
    ) -> anyhow::Result<familiar_connect::llm::LlmStream> {
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

// ---------------------------------------------------------------------------
// Substitution vs. persistence (#204)
// ---------------------------------------------------------------------------

/// A context with both image clients wired and the caller's vision capability
/// declared.
fn ctx_split(
    multimodal: bool,
    substitution: &Arc<CaptureLlm>,
    caption: &Arc<CaptureLlm>,
) -> ToolContext {
    let sub: Arc<dyn LlmClient> = substitution.clone();
    let cap: Arc<dyn LlmClient> = caption.clone();
    ctx_with_images(&[("img_0", "http://cdn.example.com/cat.png")], None)
        .with_description_llm(sub)
        .with_caption_llm(cap)
        .with_multimodal(multimodal)
}

async fn view(ctx: &ToolContext) -> familiar_connect::tools::registry::ImageResult {
    let tool = build_view_image_tool_with_fetcher("", fetcher(tiny_png()));
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_0"}), ctx)
        .await
        .unwrap();
    let ToolOutput::Image(img) = out else {
        panic!("expected image result");
    };
    img
}

#[tokio::test]
async fn a_multimodal_caller_never_pays_for_the_substitution_description() {
    let substitution = Arc::new(CaptureLlm::new("long substitution description"));
    let caption = Arc::new(CaptureLlm::new("a cat"));
    let img = view(&ctx_split(true, &substitution, &caption)).await;
    assert_eq!(
        substitution.calls(),
        0,
        "the caller sees the image itself; describing it again is double-pay"
    );
    assert_eq!(caption.calls(), 1, "exactly one call, for memory");
    assert_eq!(img.description, "a cat");
}

#[tokio::test]
async fn a_multimodal_caller_still_persists_a_caption() {
    let substitution = Arc::new(CaptureLlm::new("sub"));
    let caption = Arc::new(CaptureLlm::new("a tabby on a windowsill"));
    let img = view(&ctx_split(true, &substitution, &caption)).await;
    // The image never survives into history, so the caption is the ONLY thing
    // the fact extractor, summaries, and dossiers will ever see.
    assert_eq!(img.description, "a tabby on a windowsill");
    let content = serialize_image_result(&img, true);
    assert_eq!(
        tool_content_as_text(&content),
        "a tabby on a windowsill",
        "the persisted projection must carry the caption"
    );
    let Content::Blocks(blocks) = content else {
        panic!("multimodal content is blocks");
    };
    assert!(
        blocks.iter().any(|b| b["type"] == "image_url"),
        "and the image still reaches the model natively"
    );
}

#[tokio::test]
async fn a_text_only_caller_makes_one_call_serving_both_roles() {
    let substitution = Arc::new(CaptureLlm::new("a detailed description"));
    let caption = Arc::new(CaptureLlm::new("caption"));
    let img = view(&ctx_split(false, &substitution, &caption)).await;
    assert_eq!(substitution.calls(), 1);
    assert_eq!(
        caption.calls(),
        0,
        "no second call — the description IS both"
    );
    assert_eq!(img.description, "a detailed description");
    // Text-only serialization sends that same string as the tool result.
    assert_eq!(
        serialize_image_result(&img, false),
        Content::Text("a detailed description".to_owned())
    );
}

#[tokio::test]
async fn a_multimodal_caller_falls_back_to_the_substitution_model_for_the_caption() {
    // Legacy profile: only `image_description_model` is set. Memory must not
    // silently lose images just because the slot gained vision.
    let substitution = Arc::new(CaptureLlm::new("a cat"));
    let sub: Arc<dyn LlmClient> = substitution.clone();
    let ctx = ctx_with_images(&[("img_0", "http://cdn.example.com/cat.png")], None)
        .with_description_llm(sub)
        .with_multimodal(true);
    assert_eq!(view(&ctx).await.description, "a cat");
    assert_eq!(substitution.calls(), 1);
}

#[tokio::test]
async fn a_text_only_caller_falls_back_to_the_caption_model() {
    let caption = Arc::new(CaptureLlm::new("a cat"));
    let cap: Arc<dyn LlmClient> = caption.clone();
    let ctx = ctx_with_images(&[("img_0", "http://cdn.example.com/cat.png")], None)
        .with_caption_llm(cap)
        .with_multimodal(false);
    assert_eq!(view(&ctx).await.description, "a cat");
    assert_eq!(caption.calls(), 1);
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

// ---------------------------------------------------------------------------
// Fetch-boundary URL policy (SSRF / IP-disclosure gate)
// ---------------------------------------------------------------------------

/// Records whether the inner fetch was reached — a refusal must never call it.
struct SpyFetcher {
    called: Arc<Mutex<bool>>,
}

#[async_trait]
impl ImageFetcher for SpyFetcher {
    async fn fetch(&self, _url: &str) -> anyhow::Result<Vec<u8>> {
        *self.called.lock().unwrap() = true;
        Ok(tiny_png())
    }
}

struct FakeResolver {
    addrs: Vec<std::net::IpAddr>,
}

#[async_trait]
impl HostResolver for FakeResolver {
    async fn resolve(&self, _host: &str, _port: u16) -> anyhow::Result<Vec<std::net::IpAddr>> {
        Ok(self.addrs.clone())
    }
}

/// Build a guarded tool; returns it plus the "inner fetch happened" flag.
fn guarded_tool(
    allow_untrusted: bool,
    resolves_to: &[&str],
) -> (familiar_connect::tools::registry::Tool, Arc<Mutex<bool>>) {
    let called = Arc::new(Mutex::new(false));
    let inner: Arc<dyn ImageFetcher> = Arc::new(SpyFetcher {
        called: called.clone(),
    });
    let guard = Arc::new(UrlGuard::new(
        ImageUrlPolicy {
            allow_untrusted,
            ..ImageUrlPolicy::default()
        },
        Arc::new(FakeResolver {
            addrs: resolves_to.iter().map(|s| s.parse().unwrap()).collect(),
        }),
    ));
    let fetcher: Arc<dyn ImageFetcher> = Arc::new(GuardedFetcher::new(guard, inner));
    (build_view_image_tool_with_fetcher("", fetcher), called)
}

async fn view_error(tool: &familiar_connect::tools::registry::Tool, url: &str) -> String {
    let ctx = ctx_with_images(&[("img_0", url)], None);
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_0"}), &ctx)
        .await
        .unwrap();
    let ToolOutput::Text(s) = out else {
        panic!("expected a refusal, got an image");
    };
    let data: Value = serde_json::from_str(&s).unwrap();
    data["error"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn view_image_refuses_untrusted_host_by_default() {
    let (tool, called) = guarded_tool(false, &["93.184.216.34"]);
    assert_eq!(
        view_error(&tool, "https://attacker.example/x.png").await,
        "image host 'attacker.example' is not in [tools].trusted_image_hosts — \
         set [tools].allow_untrusted_image_urls = true to allow it"
    );
    assert!(!*called.lock().unwrap(), "no fetch may be attempted");
}

#[tokio::test]
async fn view_image_allows_untrusted_host_when_flag_set() {
    let (tool, called) = guarded_tool(true, &["93.184.216.34"]);
    let ctx = ctx_with_images(&[("img_0", "https://attacker.example/x.png")], None);
    let out = tool
        .handler
        .call(serde_json::json!({"image_id": "img_0"}), &ctx)
        .await
        .unwrap();
    assert!(matches!(out, ToolOutput::Image(_)));
    assert!(*called.lock().unwrap());
}

#[tokio::test]
async fn view_image_allows_discord_cdn_by_default() {
    for url in [
        "https://cdn.discordapp.com/attachments/1/2/cat.png",
        "https://media.discordapp.net/attachments/1/2/cat.png?width=100",
        "https://64.media.tumblr.com/abc/def.jpg",
    ] {
        let (tool, called) = guarded_tool(false, &["162.159.135.232"]);
        let ctx = ctx_with_images(&[("img_0", url)], None);
        let out = tool
            .handler
            .call(serde_json::json!({"image_id": "img_0"}), &ctx)
            .await
            .unwrap();
        assert!(matches!(out, ToolOutput::Image(_)), "{url} should pass");
        assert!(*called.lock().unwrap(), "{url} should be fetched");
    }
}

#[tokio::test]
async fn view_image_refuses_private_addresses_despite_flag() {
    for (url, addr) in [
        ("http://127.0.0.1/x.png", "127.0.0.1"),
        ("http://169.254.169.254/x.png", "169.254.169.254"),
        ("http://192.168.0.4/x.png", "192.168.0.4"),
    ] {
        let (tool, called) = guarded_tool(true, &[]);
        assert_eq!(
            view_error(&tool, url).await,
            format!("image host '{addr}' resolves to non-public address {addr}")
        );
        assert!(!*called.lock().unwrap(), "{url}");
    }
}

#[tokio::test]
async fn view_image_refuses_host_resolving_to_loopback_despite_flag() {
    let (tool, called) = guarded_tool(true, &["127.0.0.1"]);
    assert_eq!(
        view_error(&tool, "https://rebind.example/x.png").await,
        "image host 'rebind.example' resolves to non-public address 127.0.0.1"
    );
    assert!(!*called.lock().unwrap());
}

#[tokio::test]
async fn view_image_refuses_non_http_scheme_despite_flag() {
    let (tool, called) = guarded_tool(true, &["1.1.1.1"]);
    assert_eq!(
        view_error(&tool, "file:///etc/passwd").await,
        "image url scheme 'file' is not allowed — only http/https"
    );
    assert!(!*called.lock().unwrap());
}
