//! `Familiar` — the per-character DI bundle (subsystem 02; Python `familiar.py`).
//!
//! One process per character (selected by `FAMILIAR_ID`). The bundle carries the
//! parsed config, the history store, the LLM clients (keyed by slot), the bus,
//! the turn router, the subscription registry, and the optional TTS / STT /
//! local-turn handles. [`Familiar::load_from_disk`] is the sole constructor: it
//! walks `data/familiars/<id>/`, loads and validates the config, and opens (thus
//! creates) `history.db` as a side effect.
//!
//! Per DESIGN D3 the two Python deferred registry imports
//! (`known_projectors()` / `known_embedders()`) are injected as `&BTreeSet`
//! parameters rather than reached via a module cycle — the config validator
//! seam the whole port uses.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::bus::in_process::InProcessEventBus;
use crate::bus::protocols::EventBus;
use crate::bus::router::TurnRouter;
use crate::config::{CharacterConfig, load_character_config};
use crate::history::async_store::AsyncHistoryStore;
use crate::history::store::HistoryStore;
use crate::llm::LlmClient;
use crate::stt::Transcriber;
use crate::subscriptions::SubscriptionRegistry;
use crate::tts::TtsClient;
use crate::voice::turn_detection::LocalTurnDetector;

/// Runtime bundle for one character.
///
/// `transcriber == None` means a voice subscription joins for TTS playback only;
/// incoming audio is not transcribed.
pub struct Familiar {
    /// Character id (the `data/familiars/<id>` folder name).
    pub id: String,
    /// The character's on-disk root.
    pub root: PathBuf,
    /// The parsed, validated, immutable character config.
    pub config: CharacterConfig,
    /// The async history store facade.
    pub history_store: Arc<AsyncHistoryStore>,
    /// LLM clients keyed by slot (`"fast"` / `"prose"` / `"background"`).
    pub llm_clients: HashMap<String, Arc<dyn LlmClient>>,
    /// The TTS client, or `None` (voice joins for playback only when set).
    pub tts_client: Option<Arc<dyn TtsClient>>,
    /// The transcriber template (cloned per Discord user), or `None`.
    pub transcriber: Option<Box<dyn Transcriber>>,
    /// The channel-subscription registry sidecar.
    pub subscriptions: SubscriptionRegistry,
    /// The event bus (sources publish, processors consume).
    pub bus: Arc<dyn EventBus>,
    /// Per-session turn routing + cancel-prior-scope.
    pub router: Arc<TurnRouter>,
    /// Discord snowflake for the logged-in bot user (set post-login).
    pub bot_user_id: Option<i64>,
    /// TEN-VAD + Smart Turn local endpointer factory, when configured.
    pub local_turn_detector: Option<LocalTurnDetector>,
}

impl Familiar {
    /// Human-readable name for prompts / the self-subject.
    ///
    /// The first configured alias when present (`"Sapphire"`), else the
    /// title-cased id (`"sapphire"` → `"Sapphire"`).
    #[must_use]
    pub fn display_name(&self) -> String {
        if let Some(first) = self.config.aliases.first() {
            return first.clone();
        }
        title_case(&self.id)
    }

    /// Build a bundle from the on-disk `data/familiars/<id>/` layout.
    ///
    /// `defaults_path` overrides the default profile path (tests use it to skip
    /// staging a sibling `_default/` folder); it defaults to
    /// `root.parent / "_default" / "character.toml"`. `known_projectors` /
    /// `known_embedders` are the injected config validator sets (DESIGN D3).
    ///
    /// Side effect: opens (thus creates) `root / "history.db"`.
    ///
    /// # Errors
    /// [`crate::config::ConfigError`] on a missing default profile or invalid
    /// config; store/subscription open failures surface as `anyhow::Error`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the DI bundle threads many injected handles"
    )]
    pub fn load_from_disk(
        root: &Path,
        llm_clients: HashMap<String, Arc<dyn LlmClient>>,
        tts_client: Option<Arc<dyn TtsClient>>,
        transcriber: Option<Box<dyn Transcriber>>,
        local_turn_detector: Option<LocalTurnDetector>,
        defaults_path: Option<&Path>,
        known_projectors: &BTreeSet<String>,
        known_embedders: &BTreeSet<String>,
    ) -> anyhow::Result<Self> {
        let id = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_owned();
        let default_owned = defaults_path.map_or_else(
            || {
                root.parent()
                    .unwrap_or(root)
                    .join("_default")
                    .join("character.toml")
            },
            Path::to_path_buf,
        );
        let config = load_character_config(
            &root.join("character.toml"),
            &default_owned,
            known_projectors,
            known_embedders,
        )?;

        let store = HistoryStore::open(root.join("history.db"))
            .map_err(|e| anyhow::anyhow!("failed to open history.db: {e}"))?;
        let history_store = Arc::new(AsyncHistoryStore::new(store));
        let subscriptions = SubscriptionRegistry::new(root.join("subscriptions.toml"))
            .map_err(|e| anyhow::anyhow!("failed to open subscriptions: {e}"))?;

        Ok(Self {
            id,
            root: root.to_path_buf(),
            config,
            history_store,
            llm_clients,
            tts_client,
            transcriber,
            subscriptions,
            bus: Arc::new(InProcessEventBus::new()),
            router: Arc::new(TurnRouter::new()),
            bot_user_id: None,
            local_turn_detector,
        })
    }
}

/// Python `str.title()`: uppercase the first letter of each alphabetic run,
/// lowercase the rest. Ids are simple lowercase words in practice
/// (`"sapphire"` → `"Sapphire"`).
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Familiar, title_case};
    use crate::llm::{LlmClient, LlmDelta, Message};
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::Value;
    use std::collections::{BTreeSet, HashMap};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tempfile::tempdir;

    struct FakeLlm(String);

    #[async_trait]
    impl LlmClient for FakeLlm {
        async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
            Ok(Message::new("assistant", "ok"))
        }
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<Value>>,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
            Ok(Box::pin(stream::empty()))
        }
        fn slot(&self) -> Option<&str> {
            Some(&self.0)
        }
        fn multimodal(&self) -> bool {
            false
        }
        fn tool_calling_enabled(&self) -> bool {
            false
        }
    }

    fn fake_clients() -> HashMap<String, Arc<dyn LlmClient>> {
        ["fast", "prose", "background"]
            .into_iter()
            .map(|s| {
                (
                    s.to_owned(),
                    Arc::new(FakeLlm(s.to_owned())) as Arc<dyn LlmClient>,
                )
            })
            .collect()
    }

    fn projectors() -> BTreeSet<String> {
        [
            "rolling_summary",
            "rich_note",
            "people_dossier",
            "reflection",
            "fact_supersede",
            "fact_embedding",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }

    fn embedders() -> BTreeSet<String> {
        ["off", "hash", "fastembed"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    fn default_profile() -> PathBuf {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../data/familiars/_default/character.toml"
        ))
        .to_path_buf()
    }

    fn seed_root(dir: &Path, name: &str) -> PathBuf {
        let root = dir.join(name);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::copy(default_profile(), root.join("character.toml")).unwrap();
        root
    }

    fn load(root: &Path) -> Familiar {
        Familiar::load_from_disk(
            root,
            fake_clients(),
            None,
            None,
            None,
            Some(&default_profile()),
            &projectors(),
            &embedders(),
        )
        .unwrap()
    }

    #[test]
    fn returns_minimal_bundle() {
        let dir = tempdir().unwrap();
        let root = seed_root(dir.path(), "test-familiar");
        let familiar = load(&root);
        assert_eq!(familiar.id, "test-familiar");
        assert_eq!(familiar.root, root);
        assert_eq!(familiar.config.display_tz, "UTC");
        for slot in ["fast", "prose", "background"] {
            assert!(familiar.llm_clients.contains_key(slot));
        }
    }

    #[test]
    fn tts_and_transcriber_default_to_none() {
        let dir = tempdir().unwrap();
        let root = seed_root(dir.path(), "test-familiar");
        let familiar = load(&root);
        assert!(familiar.tts_client.is_none());
        assert!(familiar.transcriber.is_none());
    }

    #[test]
    fn history_db_created_on_load() {
        let dir = tempdir().unwrap();
        let root = seed_root(dir.path(), "test-familiar");
        let _ = load(&root);
        assert!(root.join("history.db").exists());
    }

    #[test]
    fn display_name_falls_back_to_title_cased_id() {
        let dir = tempdir().unwrap();
        let root = seed_root(dir.path(), "sapphire");
        let familiar = load(&root);
        // The shipped default profile has no aliases → title-cased id.
        assert_eq!(familiar.display_name(), "Sapphire");
    }

    #[test]
    fn title_case_matches_python_title() {
        assert_eq!(title_case("sapphire"), "Sapphire");
        assert_eq!(title_case("test-familiar"), "Test-Familiar");
        assert_eq!(title_case("aria"), "Aria");
    }
}
