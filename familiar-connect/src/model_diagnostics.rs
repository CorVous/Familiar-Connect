//! Startup model-configuration diagnostics (#218).
//!
//! Two independent checks, both kept out of config loading so no network
//! dependency is added there:
//!
//! - **Capability audit** — declared `[llm.<slot>]` capability flags vs. the
//!   model metadata OpenRouter serves at `GET /models`. Fetched off a detached
//!   task after boot; never gates readiness.
//! - **Focus reachability** (#221) — `tool_calling = false` on a slot whose
//!   surface has a focus manager wired. Needs no network, so it runs
//!   unconditionally at composition time.
//!
//! The same catalog also *drives* config: [`cache`] persists the last known
//! good copy so the tri-state `multimodal` flag can auto-detect at boot off a
//! local file (#204). Fetching stays on the detached task — the read side never
//! touches the network, so the "never gates readiness" invariant above holds
//! unchanged.
//!
//! Everything except [`fetch_model_catalog`](crate::llm::fetch_model_catalog)
//! is pure and unit-tested without network.

use std::collections::BTreeMap;

use crate::config::LLMSlotConfig;
use crate::log_style as ls;

pub mod cache;

/// `supported_parameters` entry that means "this model accepts a tools array".
const TOOLS_PARAM: &str = "tools";
/// `architecture.input_modalities` entry that means "accepts image input".
const IMAGE_MODALITY: &str = "image";

// ---------------------------------------------------------------------------
// Catalog types + parsing
// ---------------------------------------------------------------------------

/// One model's capability metadata from `GET /models`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// OpenRouter model id (`vendor/name`, optionally `:variant`).
    pub id: String,
    /// `architecture.input_modalities` (`text` / `image` / `audio` / …).
    pub input_modalities: Vec<String>,
    /// `supported_parameters` (`tools`, `temperature`, …).
    pub supported_parameters: Vec<String>,
}

/// Parse a `GET /models` body. `None` when it is not JSON with a `data` array.
///
/// Deliberately total below that point: every field is optional, wrong-typed
/// fields are skipped rather than failing the entry, and entries without a
/// string `id` are dropped. A false alarm is worse than no alarm.
#[must_use]
pub fn parse_model_catalog(body: &str) -> Option<Vec<ModelCapabilities>> {
    let root: serde_json::Value = serde_json::from_str(body).ok()?;
    let data = root.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|entry| {
                let id = entry.get("id")?.as_str()?;
                Some(ModelCapabilities {
                    id: id.to_owned(),
                    input_modalities: string_list(
                        entry
                            .get("architecture")
                            .and_then(|a| a.get("input_modalities")),
                    ),
                    supported_parameters: string_list(entry.get("supported_parameters")),
                })
            })
            .collect(),
    )
}

/// Strings out of a JSON array; anything else (absent, null, wrong type, or a
/// non-string element) contributes nothing.
fn string_list(value: Option<&serde_json::Value>) -> Vec<String> {
    value
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Pure comparison
// ---------------------------------------------------------------------------

/// Direction of a capability disagreement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MismatchKind {
    /// Config declares a capability the model lacks — broken, logs at ERROR.
    DeclaredUnsupported,
    /// Model offers a capability config leaves off — advisory, logs at INFO.
    SupportedNotDeclared,
}

/// One capability disagreement between a slot's config and its model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityMismatch {
    /// `[llm.<slot>]` name.
    pub slot: String,
    /// Configured model string.
    pub model: String,
    /// Config key at fault.
    pub flag: &'static str,
    /// Direction.
    pub kind: MismatchKind,
    /// What the operator should change.
    pub remediation: String,
}

/// Result of auditing every slot against a catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuditReport {
    /// Disagreements, slot order.
    pub mismatches: Vec<CapabilityMismatch>,
    /// `(slot, model)` pairs the catalog does not list — never flagged.
    pub unknown_models: Vec<(String, String)>,
}

/// Compare one slot's declared flags against its model's metadata.
///
/// `multimodal` is tri-state: only an explicit `true` can contradict a
/// text-only model, and an unset one needs no advisory because the catalog
/// already decides it (`cache::apply_detected_multimodal`). What is left to
/// advise on is `image_tools`, which stays a deliberate opt-in.
#[must_use]
pub fn compare_slot(
    slot: &str,
    cfg: &LLMSlotConfig,
    model: &ModelCapabilities,
) -> Vec<CapabilityMismatch> {
    let supports_tools = model.supported_parameters.iter().any(|p| p == TOOLS_PARAM);
    let supports_image = model.input_modalities.iter().any(|m| m == IMAGE_MODALITY);
    let mut out = Vec::new();
    let mut push = |flag, kind, remediation| {
        out.push(CapabilityMismatch {
            slot: slot.to_owned(),
            model: cfg.model.clone(),
            flag,
            kind,
            remediation,
        });
    };

    match (cfg.tool_calling, supports_tools) {
        (true, false) => push(
            "tool_calling",
            MismatchKind::DeclaredUnsupported,
            format!(
                "set [llm.{slot}].tool_calling = false, or pick a model whose \
                 supported_parameters include '{TOOLS_PARAM}'"
            ),
        ),
        (false, true) => push(
            "tool_calling",
            MismatchKind::SupportedNotDeclared,
            format!("model supports tools; set [llm.{slot}].tool_calling = true to use them"),
        ),
        _ => {}
    }

    if supports_image {
        if !cfg.image_tools {
            push(
                "image_tools",
                MismatchKind::SupportedNotDeclared,
                format!(
                    "model accepts image input; set [llm.{slot}].image_tools = true \
                     to register view_image (multimodal auto-detects)"
                ),
            );
        }
    } else {
        for (flag, declared) in [
            ("multimodal", cfg.multimodal == Some(true)),
            ("image_tools", cfg.image_tools),
        ] {
            if declared {
                push(
                    flag,
                    MismatchKind::DeclaredUnsupported,
                    format!(
                        "set [llm.{slot}].{flag} = false, or pick a model whose \
                         input_modalities include '{IMAGE_MODALITY}'"
                    ),
                );
            }
        }
    }
    out
}

/// Audit every configured slot against `catalog`.
///
/// Slots whose model string is empty are skipped; slots whose model the catalog
/// does not list land in `unknown_models` and are never flagged. Variant
/// suffixes (`:free`, `:nitro`, …) fall back to the base id.
#[must_use]
pub fn audit_slots(
    slots: &BTreeMap<String, LLMSlotConfig>,
    catalog: &[ModelCapabilities],
) -> AuditReport {
    let mut report = AuditReport::default();
    for (name, cfg) in slots {
        if cfg.model.is_empty() {
            continue;
        }
        match lookup(catalog, &cfg.model) {
            Some(model) => report.mismatches.extend(compare_slot(name, cfg, model)),
            None => report
                .unknown_models
                .push((name.clone(), cfg.model.clone())),
        }
    }
    report
}

/// Exact id first, then the pre-`:` base id.
fn lookup<'a>(catalog: &'a [ModelCapabilities], model: &str) -> Option<&'a ModelCapabilities> {
    let base = model.split(':').next().unwrap_or(model);
    catalog
        .iter()
        .find(|m| m.id == model)
        .or_else(|| catalog.iter().find(|m| m.id == base))
}

// ---------------------------------------------------------------------------
// Focus reachability (#221)
// ---------------------------------------------------------------------------

/// `Some(message)` when `slot` cannot reach `shift_focus` because tool calling
/// is off on a surface that has a focus manager wired.
///
/// `ping_fallback` marks a surface that can still move focus without a tool —
/// today only text, via the direct-ping fallback in `TextResponder`. Voice has
/// no such path, so its focus really is pinned for the session.
#[must_use]
pub fn focus_unreachable_message(
    surface: &str,
    slot_name: &str,
    slot: &LLMSlotConfig,
    ping_fallback: bool,
) -> Option<String> {
    (!slot.tool_calling).then(|| {
        let consequence = if ping_fallback {
            format!(
                "so shift_focus is unreachable — the familiar can only follow a \
                 direct ping to another {surface} channel, never move on its own \
                 judgement"
            )
        } else {
            format!(
                "so shift_focus is unreachable and the {surface} channel focus is \
                 fixed for the session"
            )
        };
        format!(
            "[llm.{slot_name}].tool_calling = false disables every tool call on the \
             {surface} surface, {consequence} — set [llm.{slot_name}].tool_calling = \
             true to let the familiar change channels"
        )
    })
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

/// Render one mismatch as a `[Config]` line.
#[must_use]
pub fn mismatch_line(m: &CapabilityMismatch) -> String {
    let (state, color) = match m.kind {
        MismatchKind::DeclaredUnsupported => ("unsupported", ls::R),
        MismatchKind::SupportedNotDeclared => ("available", ls::LM),
    };
    format!(
        "{} {} {} {} {}",
        ls::tag("Config", ls::W),
        ls::kv("slot", &m.slot),
        ls::kv("model", &m.model),
        ls::kv_styled(&format!("capability={}", m.flag), state, ls::W, color),
        ls::kv("fix", &m.remediation),
    )
}

/// Emit `report`: declared-but-unsupported at ERROR, the inverse at INFO,
/// unknown model ids at DEBUG.
pub fn log_audit(report: &AuditReport) {
    for m in &report.mismatches {
        let line = mismatch_line(m);
        match m.kind {
            MismatchKind::DeclaredUnsupported => {
                tracing::error!(target: "familiar_connect.llm", "{line}");
            }
            MismatchKind::SupportedNotDeclared => {
                tracing::info!(target: "familiar_connect.llm", "{line}");
            }
        }
    }
    for (slot, model) in &report.unknown_models {
        let line = format!(
            "{} {} {} {}",
            ls::tag("Config", ls::W),
            ls::kv("slot", slot),
            ls::kv("model", model),
            ls::kv(
                "capability_audit",
                "model not in OpenRouter catalog; skipped"
            ),
        );
        tracing::debug!(target: "familiar_connect.llm", "{line}");
    }
}

/// Emit the single line that ends a failed or unusable fetch. One line, then
/// silence — the audit is advisory and must never look like an outage.
pub fn log_audit_skipped(reason: &str) {
    let line = format!(
        "{} {}",
        ls::tag("Config", ls::W),
        ls::kv("capability_audit", &format!("skipped — {reason}")),
    );
    tracing::info!(target: "familiar_connect.llm", "{line}");
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Refresh the on-disk catalog cache, then log the audit.
///
/// Callers detach this, so it can neither block nor fail startup. A cache still
/// inside [`cache::CACHE_TTL_HOURS`] short-circuits the network entirely and
/// audits from disk. What a refresh writes lands on the **next** boot's
/// capability resolution — never this one.
#[cfg(feature = "net")]
pub async fn run_capability_audit(
    api_key: String,
    base_url: String,
    slots: BTreeMap<String, LLMSlotConfig>,
    cache_path: std::path::PathBuf,
) {
    if let Some(cached) = cache::read_cache(&cache_path)
        && !cache::is_stale(
            &cached.fetched_at,
            chrono::Utc::now(),
            cache::CACHE_TTL_HOURS,
        )
    {
        log_audit(&audit_slots(&slots, &cached.models));
        return;
    }
    let body = match crate::llm::fetch_model_catalog(&api_key, &base_url).await {
        Ok(body) => body,
        Err(err) => {
            log_audit_skipped(&format!("model catalog unreachable: {err}"));
            return;
        }
    };
    let Some(catalog) = parse_model_catalog(&body) else {
        log_audit_skipped("model catalog response not understood");
        return;
    };
    if let Err(err) = cache::write_cache(&cache_path, &catalog, chrono::Utc::now()) {
        tracing::debug!(target: "familiar_connect.llm", "model catalog cache write failed: {err}");
    }
    log_audit(&audit_slots(&slots, &catalog));
}

#[cfg(test)]
mod tests;
