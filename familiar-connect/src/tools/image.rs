//! `view_image` tool: fetch → compress → vision-describe (subsystem 08;
//! feature `images`).
//!
//! The image fetch is injected via the [`ImageFetcher`] seam so tests can
//! supply canned bytes. The production fetcher uses `reqwest` (feature `net`):
//! a fresh client per call with a 15s timeout, a content-type allow-list, and a
//! hard 20 MiB streaming cap.
//!
//! Every URL — whatever source put it in `ctx.images` — clears
//! [`UrlGuard`](crate::tools::image_policy::UrlGuard) *before* a socket opens.
//! Redirects are followed by hand so each hop faces the same gate, and the
//! connection is pinned to the addresses the gate validated.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{Value, json};

use crate::llm::LlmClient;
use crate::log_style as ls;
use crate::tools::image_compress::{SIZE_CEILING, compress_for_description, compress_to_jpeg};
use crate::tools::image_describe::describe_image;
use crate::tools::image_policy::{UrlGuard, UrlRefusal};
use crate::tools::registry::{FnHandler, ImageResult, Tool, ToolContext, ToolOutput};

const TOOL_TIMEOUT_S: f64 = 30.0;

/// Fetches raw image bytes from a URL (validating it is an image).
#[async_trait]
pub trait ImageFetcher: Send + Sync {
    /// Fetch and return the raw bytes, or an error describing the failure.
    async fn fetch(&self, url: &str) -> anyhow::Result<Vec<u8>>;
}

/// The production HTTP fetcher (real work only under feature `net`).
pub struct HttpImageFetcher {
    guard: Arc<UrlGuard>,
}

impl HttpImageFetcher {
    /// Fetcher gated by `guard` — applied to the first request and every
    /// redirect hop.
    #[must_use]
    pub const fn new(guard: Arc<UrlGuard>) -> Self {
        Self { guard }
    }
}

#[async_trait]
impl ImageFetcher for HttpImageFetcher {
    async fn fetch(&self, url: &str) -> anyhow::Result<Vec<u8>> {
        #[cfg(feature = "net")]
        {
            fetch_image_bytes(url, &self.guard).await
        }
        #[cfg(not(feature = "net"))]
        {
            self.guard.check(url).await?;
            anyhow::bail!("image fetch requires the `net` feature")
        }
    }
}

/// Applies [`UrlGuard`] then delegates — the same gate `HttpImageFetcher` runs
/// per hop, in front of an arbitrary fetcher (tests, future transports).
pub struct GuardedFetcher {
    guard: Arc<UrlGuard>,
    inner: Arc<dyn ImageFetcher>,
}

impl GuardedFetcher {
    /// Wrap `inner` behind `guard`.
    #[must_use]
    pub const fn new(guard: Arc<UrlGuard>, inner: Arc<dyn ImageFetcher>) -> Self {
        Self { guard, inner }
    }
}

#[async_trait]
impl ImageFetcher for GuardedFetcher {
    async fn fetch(&self, url: &str) -> anyhow::Result<Vec<u8>> {
        self.guard.check(url).await?;
        self.inner.fetch(url).await
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

/// Fetch with hand-rolled redirect following: `reqwest`'s own follower would
/// carry a permitted host into a 302 at a private address, and its redirect
/// policy hook is sync so it cannot re-resolve. Each hop re-enters the guard.
#[cfg(feature = "net")]
async fn fetch_image_bytes(url: &str, guard: &UrlGuard) -> anyhow::Result<Vec<u8>> {
    use futures::StreamExt as _;
    const FETCH_TIMEOUT_S: u64 = 15;
    const MAX_DOWNLOAD_BYTES: usize = 20 * 1024 * 1024;
    const MAX_REDIRECTS: usize = 5;

    let mut url = url.to_owned();
    for _ in 0..=MAX_REDIRECTS {
        let checked = guard.check(&url).await?;
        let parsed = reqwest::Url::parse(&url)?;
        // Belt and braces: if reqwest reads a different host than the guard
        // validated, the validation (and the DNS pin below) would not apply.
        let reqwest_host =
            crate::tools::image_policy::normalize_host(parsed.host_str().unwrap_or(""));
        if reqwest_host != checked.host {
            anyhow::bail!(
                "image url host is ambiguous: {reqwest_host:?} vs {:?}",
                checked.host
            );
        }

        let mut builder = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_S))
            .redirect(reqwest::redirect::Policy::none());
        // Connect to exactly the addresses just validated — a second lookup
        // could answer differently (DNS rebinding).
        if checked.host.parse::<std::net::IpAddr>().is_err() {
            builder = builder.resolve_to_addrs(&checked.host, &checked.addrs);
        }
        let resp = builder.build()?.get(url.clone()).send().await?;

        if resp.status().is_redirection() {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| anyhow::anyhow!("redirect without a Location header"))?;
            url = parsed.join(location)?.to_string();
            continue;
        }

        let resp = resp.error_for_status()?;
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
        return Ok(out);
    }
    anyhow::bail!("image fetch exceeded {MAX_REDIRECTS} redirects")
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
            // Policy refusals never reached the network — say so, and say it once.
            let refused = e.downcast_ref::<UrlRefusal>().is_some();
            let key = if refused { "refused" } else { "fetch_error" };
            tracing::warn!(
                "{} {} {}",
                ls::tag("Image", ls::Y),
                ls::kv_styled(key, &format!("{e}"), ls::W, ls::LY),
                ls::kv_styled("img_id", img_id, ls::W, ls::LC),
            );
            return text_error(&if refused {
                format!("{e}")
            } else {
                format!("fetch failed: {e}")
            });
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
/// `describe_constraints` (per-familiar) and the URL `guard` bind into the
/// handler at construction.
#[must_use]
pub fn build_view_image_tool(describe_constraints: &str, guard: Arc<UrlGuard>) -> Tool {
    build_view_image_tool_with_fetcher(describe_constraints, Arc::new(HttpImageFetcher::new(guard)))
}

/// Build the `view_image` tool with an injected fetcher (test seam).
///
/// The fetcher owns the URL gate — production passes [`HttpImageFetcher`], and
/// tests wrap their double in [`GuardedFetcher`] to exercise it.
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
        let guard = Arc::new(UrlGuard::production(
            crate::tools::image_policy::ImageUrlPolicy::default(),
        ));
        let tool = build_view_image_tool("", guard);
        assert_eq!(tool.name, "view_image");
        assert!((tool.timeout_s - 30.0).abs() < f64::EPSILON);
        let props = &tool.parameters["properties"];
        assert!(props.get("image_id").is_some());
        assert_eq!(tool.parameters["required"], json!(["image_id"]));
    }
}
