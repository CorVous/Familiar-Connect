//! `view_image` tool: fetch → compress → vision-describe (subsystem 08; Python
//! `tools/image.py`; feature `images`).
//!
//! The image fetch is injected via the [`ImageFetcher`] seam so tests can supply
//! canned bytes (the Python suite patches `_fetch_image_bytes`). The production
//! fetcher uses `reqwest` (feature `net`): a fresh client per call with a 15s
//! timeout, redirect-following, a content-type allow-list, and a hard 20 MiB
//! streaming cap.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{Value, json};

use crate::llm::LlmClient;
use crate::log_style as ls;
use crate::tools::image_compress::{SIZE_CEILING, compress_for_description, compress_to_jpeg};
use crate::tools::image_describe::describe_image;
use crate::tools::registry::{FnHandler, ImageResult, Tool, ToolContext, ToolOutput};

const TOOL_TIMEOUT_S: f64 = 30.0;

/// Fetches raw image bytes from a URL (validating it is an image).
#[async_trait]
pub trait ImageFetcher: Send + Sync {
    /// Fetch and return the raw bytes, or an error describing the failure.
    async fn fetch(&self, url: &str) -> anyhow::Result<Vec<u8>>;
}

/// The production HTTP fetcher (real work only under feature `net`).
pub struct HttpImageFetcher;

#[async_trait]
impl ImageFetcher for HttpImageFetcher {
    async fn fetch(&self, url: &str) -> anyhow::Result<Vec<u8>> {
        #[cfg(feature = "net")]
        {
            fetch_image_bytes(url).await
        }
        #[cfg(not(feature = "net"))]
        {
            let _ = url;
            anyhow::bail!("image fetch requires the `net` feature")
        }
    }
}

#[cfg(feature = "net")]
const ALLOWED_CONTENT_TYPES: [&str; 6] = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
];

#[cfg(feature = "net")]
async fn fetch_image_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    use futures::StreamExt as _;
    const FETCH_TIMEOUT_S: u64 = 15;
    const MAX_DOWNLOAD_BYTES: usize = 20 * 1024 * 1024;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_S))
        .build()?;
    let resp = client.get(url).send().await?.error_for_status()?;
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ct_base = ct.split(';').next().unwrap_or("").trim().to_lowercase();
    if !ALLOWED_CONTENT_TYPES.contains(&ct_base.as_str()) {
        anyhow::bail!("non-image content-type: {ct:?}");
    }

    let mut stream = resp.bytes_stream();
    let mut out: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if out.len() + chunk.len() > MAX_DOWNLOAD_BYTES {
            anyhow::bail!("download exceeds {MAX_DOWNLOAD_BYTES} byte cap");
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

fn text_error(msg: &str) -> ToolOutput {
    ToolOutput::Text(json!({ "error": msg }).to_string())
}

async fn describe_leg(
    llm: &dyn LlmClient,
    raw: &[u8],
    constraints: &str,
) -> anyhow::Result<String> {
    let desc_jpeg = compress_for_description(raw)?;
    let desc_b64 = base64::engine::general_purpose::STANDARD.encode(&desc_jpeg);
    describe_image(llm, &desc_b64, "image/jpeg", constraints).await
}

async fn view_image_handler(
    fetcher: &dyn ImageFetcher,
    constraints: &str,
    args: &Value,
    ctx: &ToolContext,
) -> ToolOutput {
    let Some(img_id) = args.get("image_id").and_then(Value::as_str) else {
        return text_error("image_id must be a string");
    };
    let Some(url) = ctx.images.get(img_id) else {
        return text_error(&format!("unknown image id '{img_id}'"));
    };

    let raw = match fetcher.fetch(url).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                "{} {} {}",
                ls::tag("Image", ls::Y),
                ls::kv_styled("fetch_error", &format!("{e}"), ls::W, ls::LY),
                ls::kv_styled("img_id", img_id, ls::W, ls::LC),
            );
            return text_error(&format!("fetch failed: {e}"));
        }
    };

    // Describe at high quality (result persists in history); degrade gracefully.
    let desc = if let Some(llm) = ctx.description_llm.as_ref() {
        match describe_leg(llm.as_ref(), &raw, constraints).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    "{} {} {}",
                    ls::tag("Image", ls::Y),
                    ls::kv_styled("describe_error", &format!("{e}"), ls::W, ls::LY),
                    ls::kv_styled("img_id", img_id, ls::W, ls::LC),
                );
                "(description unavailable)".to_owned()
            }
        }
    } else {
        "(no description model configured)".to_owned()
    };

    // Tight compress for the prompt payload; this failure IS fatal to the call.
    let jpeg = match compress_to_jpeg(&raw, SIZE_CEILING) {
        Ok(j) => j,
        Err(e) => return text_error(&format!("compression failed: {e}")),
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    ToolOutput::Image(ImageResult::new(desc, b64))
}

/// Build the `view_image` tool with the production HTTP fetcher.
///
/// `describe_constraints` (per-familiar) bind into the handler at construction.
#[must_use]
pub fn build_view_image_tool(describe_constraints: &str) -> Tool {
    build_view_image_tool_with_fetcher(describe_constraints, Arc::new(HttpImageFetcher))
}

/// Build the `view_image` tool with an injected fetcher (test seam).
#[must_use]
pub fn build_view_image_tool_with_fetcher(
    describe_constraints: &str,
    fetcher: Arc<dyn ImageFetcher>,
) -> Tool {
    let constraints = describe_constraints.to_owned();
    Tool::new(
        "view_image",
        "Fetch and look at an image referenced by its [image: img_N (filename)] \
         placeholder. Pass the `image_id` exactly as shown (e.g. `img_0`). Returns \
         a description and the image itself when the model supports vision.",
        json!({
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "Image id from the [image: img_N] placeholder.",
                },
            },
            "required": ["image_id"],
        }),
        Arc::new(FnHandler(move |args: Value, ctx: ToolContext| {
            let fetcher = Arc::clone(&fetcher);
            let constraints = constraints.clone();
            async move { Ok(view_image_handler(fetcher.as_ref(), &constraints, &args, &ctx).await) }
        })),
    )
    .with_timeout_s(TOOL_TIMEOUT_S)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_view_image_tool_shape() {
        let tool = build_view_image_tool("");
        assert_eq!(tool.name, "view_image");
        assert!((tool.timeout_s - 30.0).abs() < f64::EPSILON);
        let props = &tool.parameters["properties"];
        assert!(props.get("image_id").is_some());
        assert_eq!(tool.parameters["required"], json!(["image_id"]));
    }
}
