//! Prompt assembler: layer compose + per-layer memoization + recent-history slot
//! (subsystem 05; Python `context/assembler.py`).
//!
//! Composes [`Layer`] contributions into one system-prompt string plus the
//! recent-history message list. Each system-prompt layer is independently cached
//! on its own [`Layer::invalidation_key`]; two assemble calls with the same
//! context and unchanged keys reuse the rendered text without re-running
//! [`Layer::build`].
//!
//! Per DESIGN D15 the assembler uses **explicit slots**, not `isinstance`
//! downcasting: recent-history is a distinct slot (not a `Layer`), and the RAG
//! cue is routed through an explicit handle. The layer-order pin (behavior 6)
//! therefore applies to the system-prompt layer `Vec` only.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::layers::{Layer, RagContextLayer, RecentHistoryLayer};
use crate::llm::Message;

/// Inputs the assembler passes to every layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyContext {
    /// Familiar this prompt is being built for.
    pub familiar_id: String,
    /// Active channel; `None` for channel-less contexts.
    pub channel_id: Option<i64>,
    /// `"voice"` or `"text"` — selects the [`super::layers::OperatingModeLayer`]
    /// output and may affect layer order.
    pub viewer_mode: String,
    /// Discord guild scoping per-guild nicknames; `None` for DMs / non-Discord.
    pub guild_id: Option<i64>,
}

impl AssemblyContext {
    /// New context with `viewer_mode = "text"` and no guild.
    #[must_use]
    pub fn new(familiar_id: impl Into<String>, channel_id: Option<i64>) -> Self {
        Self {
            familiar_id: familiar_id.into(),
            channel_id,
            viewer_mode: "text".to_owned(),
            guild_id: None,
        }
    }

    /// Builder: set the viewer mode (`"voice"` / `"text"`).
    #[must_use]
    pub fn with_viewer_mode(mut self, viewer_mode: impl Into<String>) -> Self {
        self.viewer_mode = viewer_mode.into();
        self
    }

    /// Builder: set the Discord guild id.
    #[must_use]
    pub const fn with_guild_id(mut self, guild_id: i64) -> Self {
        self.guild_id = Some(guild_id);
        self
    }
}

/// Output of [`Assembler::assemble`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssembledPrompt {
    /// The composed system prompt (non-empty layer texts joined with `"\n\n"`).
    pub system_prompt: String,
    /// The recent-history message list from the recent-history slot.
    pub recent_history: Vec<Message>,
}

/// Layer composer with per-layer memoization.
///
/// Layer order is preserved from construction. The cache keeps a single slot per
/// layer name (`name -> (key, text)`): this is behavior-equivalent to Python's
/// unbounded `(name, key) -> text` dict for every observable test — nothing reads
/// stale entries — while staying leak-free (DESIGN §5, port notes).
pub struct Assembler {
    layers: Vec<Arc<dyn Layer>>,
    recent_history: Option<RecentHistoryLayer>,
    rag: Option<Arc<RagContextLayer>>,
    cache: Mutex<HashMap<String, (String, String)>>,
}

impl Assembler {
    /// Start building an assembler.
    #[must_use]
    pub fn builder() -> AssemblerBuilder {
        AssemblerBuilder::default()
    }

    /// The system-prompt layer names in construction order (order-pin oracle).
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name()).collect()
    }

    /// Forward *cue* to the RAG layer, if one is wired.
    ///
    /// Uses the explicit handle registered at build time (DESIGN D15) rather than
    /// downcasting a `dyn Layer`.
    pub fn set_rag_cue(&self, cue: &str) {
        if let Some(rag) = &self.rag {
            rag.set_current_cue(cue);
        }
    }

    /// Compose the system prompt + recent history for `ctx`.
    ///
    /// See behaviors 1–5: layers iterate in construction order, each cached on
    /// `(name, invalidation_key)`; non-empty texts join with `"\n\n"`; the
    /// recent-history slot (if present) yields `recent_history`.
    pub async fn assemble(&self, ctx: &AssemblyContext) -> AssembledPrompt {
        let mut sections: Vec<String> = Vec::new();
        for layer in &self.layers {
            let name = layer.name().to_owned();
            let key = layer.invalidation_key(ctx).await;
            let cached = {
                let cache = self.cache.lock().expect("assembler cache mutex");
                cache
                    .get(&name)
                    .filter(|(cached_key, _)| *cached_key == key)
                    .map(|(_, text)| text.clone())
            };
            let text = if let Some(text) = cached {
                text
            } else {
                let text = layer.build(ctx).await;
                self.cache
                    .lock()
                    .expect("assembler cache mutex")
                    .insert(name, (key, text.clone()));
                text
            };
            if !text.is_empty() {
                sections.push(text);
            }
        }
        let recent = match &self.recent_history {
            Some(rh) => rh.recent_messages(ctx).await,
            None => Vec::new(),
        };
        AssembledPrompt {
            system_prompt: sections.join("\n\n"),
            recent_history: recent,
        }
    }
}

/// Fluent builder for [`Assembler`] (mirrors Python's `Assembler(layers=[...])`
/// while keeping recent-history and RAG as explicit slots, DESIGN D15).
#[derive(Default)]
pub struct AssemblerBuilder {
    layers: Vec<Arc<dyn Layer>>,
    recent_history: Option<RecentHistoryLayer>,
    rag: Option<Arc<RagContextLayer>>,
}

impl AssemblerBuilder {
    /// Append one system-prompt layer (order matters — behavior 6).
    #[must_use]
    pub fn layer(mut self, layer: Arc<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Install the recent-history slot.
    #[must_use]
    pub fn recent_history(mut self, recent_history: RecentHistoryLayer) -> Self {
        self.recent_history = Some(recent_history);
        self
    }

    /// Append the RAG layer as a system-prompt layer *and* register it as the
    /// `set_rag_cue` handle.
    #[must_use]
    pub fn rag(mut self, rag: Arc<RagContextLayer>) -> Self {
        self.layers.push(rag.clone());
        self.rag = Some(rag);
        self
    }

    /// Finish building.
    #[must_use]
    pub fn build(self) -> Assembler {
        Assembler {
            layers: self.layers,
            recent_history: self.recent_history,
            rag: self.rag,
            cache: Mutex::new(HashMap::new()),
        }
    }
}
