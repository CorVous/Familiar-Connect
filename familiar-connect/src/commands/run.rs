//! run subcommand: composition root + teardown (subsystem 10; Python
//! commands/run.py).
//!
//! `run` is the failure ladder (token → familiar → config → clients → familiar
//! bundle); the async composition root `async_main` brings up the bus,
//! responders, workers, sources, alarm scheduler, and activity engine, wires the
//! serenity gateway, and tears everything down in the spec's order on a
//! cooperative SIGINT/SIGTERM (spec 10 §55–58).
//!
//! Because the real gateway needs `serenity`, `async_main` and the client-facing
//! half of `run` are `cfg(feature = "discord")`; the default-feature build keeps
//! the pure, unit-tested wiring helpers ([`resolve_familiar_root`],
//! [`default_assembler`], [`build_activity_engine`], [`ShutdownController`]) and
//! reports a clear error if asked to actually connect without the feature.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Args;
use directories::ProjectDirs;
use tokio_util::sync::CancellationToken;

use crate::activities::config::load_activities_config;
use crate::activities::engine::{
    ActivityEngine, DefaultMaintenanceRunner, EngineParams, FocusLike, LcgRng, RealActivityStore,
    SystemClock,
};
use crate::bot::{BotHandle, build_activity_presence_cb};
use crate::budget::TierBudget;
use crate::config::EmbeddingConfig;
use crate::context::layers::ChannelResolver;
use crate::context::{
    Assembler, CharacterCardLayer, ConversationSummaryLayer, LorebookLayer, OperatingModeLayer,
    PeopleDossierLayer, RagContextLayer, RecentHistoryLayer, ReflectionLayer,
};
use crate::embedding::{Embedder, EmbeddingError};
use crate::familiar::Familiar;
use crate::focus::FocusManager;
use crate::sleep::maintenance::SleepPromptText;

/// `run` arguments (Python `add_parser`'s `--familiar`).
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Folder name of the character to run (under `data/familiars/`). Overrides
    /// `FAMILIAR_ID`.
    #[arg(long, value_name = "ID")]
    pub familiar: Option<String>,
}

/// Root holding per-user familiar folders (`<root>/<id>/`).
///
/// Precedence: the `FAMILIARS_ROOT` env override wins; otherwise the platform
/// per-user data directory (`~/.local/share/familiar-connect/familiars` on Linux,
/// the OS-correct analog elsewhere). Home-based storage means a `git clean -fdx`
/// in a repo checkout can no longer wipe live familiars — the reported foot-gun
/// (issue #201). The legacy CWD-relative `data/familiars` survives only as the
/// last-resort fallback when no home directory resolves, and as the source of the
/// one-shot migration in [`run`].
fn default_familiars_root() -> PathBuf {
    resolve_familiars_root(std::env::var("FAMILIARS_ROOT").ok(), home_familiars_root())
}

/// Pure core of [`default_familiars_root`]: the env override wins, else the home
/// fallback. The env value and fallback are injected so tests need not mutate the
/// process environment (`std::env::set_var` is `unsafe` under edition 2024, which
/// the crate forbids).
fn resolve_familiars_root(env_override: Option<String>, home_fallback: PathBuf) -> PathBuf {
    match env_override {
        Some(root) if !root.is_empty() => PathBuf::from(root),
        _ => home_fallback,
    }
}

/// The platform per-user data location for familiars, or the legacy CWD-relative
/// `data/familiars` when no home directory is discoverable.
fn home_familiars_root() -> PathBuf {
    ProjectDirs::from("", "", "familiar-connect").map_or_else(legacy_familiars_root, |dirs| {
        dirs.data_dir().join("familiars")
    })
}

/// The historical CWD-relative `data/familiars` root. Now only the tracked
/// `_default` profile and any not-yet-migrated legacy familiars live here.
fn legacy_familiars_root() -> PathBuf {
    Path::new("data").join("familiars")
}

/// Root holding the tracked `_default/` profile skeleton (character/activities
/// merge defaults).
///
/// `_default` is a repo resource (`.gitignore` un-ignores it), so it is resolved
/// **independently** of where per-user familiars live — it must not migrate to
/// the home data dir with user state (issue #201). Precedence: the
/// `FAMILIAR_DEFAULTS_ROOT` env override wins (point a `cargo install`ed binary
/// at its bundled copy); otherwise the CWD-relative `data/familiars` a repo
/// checkout ships.
fn default_defaults_root() -> PathBuf {
    resolve_defaults_root(std::env::var("FAMILIAR_DEFAULTS_ROOT").ok())
}

/// Pure core of [`default_defaults_root`] (env value injected; see
/// [`resolve_familiars_root`] for why).
fn resolve_defaults_root(env_override: Option<String>) -> PathBuf {
    match env_override {
        Some(root) if !root.is_empty() => PathBuf::from(root),
        _ => legacy_familiars_root(),
    }
}

/// One-shot, best-effort migration of legacy CWD-relative familiars into the
/// resolved root (issue #201).
///
/// For every `legacy_root/<id>` folder other than the tracked `_default`, move it
/// to `new_root/<id>` when no familiar already lives there. Idempotent (a second
/// run finds nothing left to move) and never-clobbering (an existing home-dir
/// familiar is left untouched, its legacy copy kept in place). Best-effort: a
/// failed move logs a hint and leaves the legacy copy behind rather than aborting
/// startup (e.g. a cross-device rename the operator must complete by hand).
fn migrate_legacy_familiars(legacy_root: &Path, new_root: &Path) {
    if legacy_root == new_root || !legacy_root.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(legacy_root) else {
        return;
    };
    for entry in entries.flatten() {
        let src = entry.path();
        if !src.is_dir() {
            continue;
        }
        let Some(name) = src.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // `_default` is a tracked repo resource, resolved via `default_defaults_root`.
        if name == "_default" {
            continue;
        }
        let dest = new_root.join(name);
        if dest.exists() {
            // A familiar already lives at the new root — never clobber it.
            continue;
        }
        if let Err(err) = std::fs::create_dir_all(new_root) {
            tracing::warn!(
                "could not create familiars root {}: {err}",
                new_root.display()
            );
            return;
        }
        match std::fs::rename(&src, &dest) {
            Ok(()) => tracing::info!(
                "migrated familiar '{name}' to {} (issue #201 home-dir storage)",
                dest.display()
            ),
            Err(err) => tracing::warn!(
                "could not migrate familiar '{name}' from {} to {} ({err}); \
                 move it by hand or set FAMILIARS_ROOT",
                src.display(),
                dest.display()
            ),
        }
    }
}

/// Resolve the active familiar's root directory.
///
/// Resolution order: `--familiar` flag → `FAMILIAR_ID` env (Python
/// `_resolve_familiar_root`). The `root` base is injected so tests need not touch
/// the process environment (the Python `_DEFAULT_FAMILIARS_ROOT` monkeypatch).
///
/// # Errors
/// Byte-stable messages: no id selected, or the resolved directory is missing.
pub fn resolve_familiar_root(
    flag: Option<&str>,
    env_id: Option<String>,
    root: &Path,
) -> Result<PathBuf, String> {
    let familiar_id = flag
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| env_id.filter(|s| !s.is_empty()));
    let Some(familiar_id) = familiar_id else {
        return Err("No familiar selected. Set FAMILIAR_ID, pass --familiar, \
             or create data/familiars/<id>/."
            .to_owned());
    };
    let dir = root.join(&familiar_id);
    if !dir.exists() {
        // Show the absolute path: the resolved familiars root is platform- or
        // env-dependent (issue #201), and a relative path in the error would
        // hide *which* directory was searched.
        let shown = std::path::absolute(&dir).unwrap_or_else(|_| dir.clone());
        return Err(format!(
            "Familiar folder does not exist: {} (set FAMILIARS_ROOT to point \
             the familiars root elsewhere)",
            shown.display()
        ));
    }
    Ok(dir)
}

/// Resolve the startup embedder from `config`, or fail.
///
/// The composition root calls this BEFORE opening the history store / FTS: a
/// misconfigured embedding backend (e.g. `fastembed` without the `local-embed`
/// extra) must refuse to start rather than wipe the FTS and then die deep in
/// `create_projectors` under a misleading Discord-token hint. The backend error
/// (which already names the real fix) is propagated, never swallowed into
/// `None`.
///
/// # Errors
/// Whatever [`create_embedder`](crate::embedding::create_embedder) returns —
/// unknown backend, bad dimensionality, or the `fastembed` `local-embed` gap.
#[cfg_attr(
    not(feature = "discord"),
    allow(
        dead_code,
        reason = "the only non-test caller is the `discord`-gated `run_inner`; \
                  the fail-fast contract is unit-tested under default features"
    )
)]
fn resolve_embedder(config: &EmbeddingConfig) -> Result<Option<Arc<dyn Embedder>>, EmbeddingError> {
    crate::embedding::create_embedder(config)
}

/// The two hardcoded operating-mode strings (byte-exact; Python
/// `_default_assembler`).
fn operating_modes() -> HashMap<String, String> {
    let mut modes = HashMap::new();
    modes.insert(
        "voice".to_owned(),
        "You are speaking aloud. Keep replies short (one or two sentences). \
         Avoid markdown."
            .to_owned(),
    );
    modes.insert(
        "text".to_owned(),
        "You are chatting in a text channel. Markdown and multi-line replies \
         are fine."
            .to_owned(),
    );
    modes
}

/// Build the full layer stack with token-aware per-section caps.
///
/// Order is **stability descending** for OpenAI prompt-cache friendliness
/// (Python `_default_assembler`). Per DESIGN D15 recent-history is a distinct
/// slot (not a system-prompt layer), so the layer-order pin applies to the
/// system-prompt `Vec`: `[character_card, operating_mode, lorebook,
/// conversation_summary, reflection, people_dossier, rag_context]` with
/// `rag_context` last, recent-history in its own slot.
#[must_use]
pub fn default_assembler(
    familiar: &Familiar,
    window_size: i64,
    budget: TierBudget,
    silence_gap_fold_seconds: f64,
    embedder: Option<Arc<dyn Embedder>>,
    focus_manager: Option<Arc<FocusManager>>,
) -> Assembler {
    let store = familiar.history_store.clone();
    let display_tz = familiar.config.display_tz.clone();
    let retrieval = &familiar.config.memory_retrieval;

    // Guild-name resolver binds over the focus manager's live map. Channel names
    // have no live getter on the landed `FocusManager` (a getter is a filed
    // shared-file request), so the channel resolver is left unset — matching the
    // responders' current behaviour.
    let guild_resolver: Option<ChannelResolver> = focus_manager.map(|fm| {
        let resolver: ChannelResolver = Arc::new(move |cid: i64| fm.guild_name_for(Some(cid)));
        resolver
    });

    let mut recent = RecentHistoryLayer::builder(store.clone())
        .window_size(window_size)
        .max_tokens(Some(budget.recent_history_tokens))
        .coalesce_max_gap_seconds(familiar.config.recent_history_coalesce_max_gap_seconds)
        .silence_gap_fold_seconds(silence_gap_fold_seconds)
        .display_tz(display_tz.clone());
    if let Some(resolver) = guild_resolver {
        recent = recent.guild_name_resolver(resolver);
    }
    let recent = recent.build();

    let mut rag = RagContextLayer::builder(store.clone())
        .max_results(budget.max_rag_turns)
        .max_facts(budget.max_rag_facts)
        // Match the recent-history window so RAG only surfaces older turns.
        .recent_window_size(window_size)
        .max_tokens(Some(budget.rag_tokens))
        .bm25_weight(retrieval.bm25_weight)
        .recency_weight(retrieval.recency_weight)
        .importance_weight(retrieval.importance_weight)
        .embedding_weight(retrieval.embedding_weight)
        .display_tz(display_tz);
    if let Some(embedder) = embedder {
        rag = rag.embedder(embedder);
    }
    let rag = Arc::new(rag.build());

    Assembler::builder()
        .layer(Arc::new(CharacterCardLayer::new(
            familiar.root.join("character.md"),
        )))
        .layer(Arc::new(OperatingModeLayer::new(operating_modes())))
        .layer(Arc::new(
            LorebookLayer::builder(store.clone(), familiar.root.join("lorebook.toml"))
                .recent_window(window_size)
                .max_entries(budget.max_lorebook_entries)
                .max_tokens(Some(budget.lorebook_tokens))
                .build(),
        ))
        .layer(Arc::new(
            ConversationSummaryLayer::new(store.clone()).with_max_tokens(budget.summary_tokens),
        ))
        .layer(Arc::new(
            ReflectionLayer::new(store.clone())
                .with_max_reflections(budget.max_reflections)
                .with_max_tokens(budget.reflection_tokens),
        ))
        .layer(Arc::new(
            PeopleDossierLayer::builder(store)
                .window_size(window_size)
                .max_people(budget.max_dossier_people)
                .max_tokens(Some(budget.dossier_tokens))
                .familiar_display_name(familiar.display_name())
                .build(),
        ))
        // RAG is the last system-prompt layer; recent-history is a slot.
        .rag(rag)
        .recent_history(recent)
        .build()
}

/// Build an [`ActivityEngine`] from the familiar's `activities.toml`.
///
/// Sidecar merged over the `_default` skeleton (Python `_build_activity_engine`);
/// a missing file or empty catalog yields `None` (no engine, zero behavior
/// change). `voice_active_fn` reads `handle.voice_channels` (the default-feature
/// proxy for the voice runtime map), and `bot_user_id` is late-bound over the
/// shared cell the gateway fills on ready.
#[must_use]
pub fn build_activity_engine(
    familiar: &Familiar,
    focus: Arc<FocusManager>,
    handle: Arc<BotHandle>,
    bot_user_id: Arc<std::sync::Mutex<Option<i64>>>,
) -> Option<Arc<ActivityEngine>> {
    // Activities defaults come from the tracked `_default` skeleton, resolved
    // independently of the per-user familiar root (issue #201).
    let defaults_path = default_defaults_root()
        .join("_default")
        .join("activities.toml");
    let config =
        load_activities_config(&familiar.root.join("activities.toml"), Some(&defaults_path))
            .ok()?;
    if !config.enabled() {
        return None;
    }
    let background_llm = familiar.llm_clients.get("background").cloned()?;

    let voice_handle = handle.clone();
    let voice_active_fn: crate::activities::engine::VoiceActiveFn = Arc::new(move || {
        !voice_handle
            .voice_channels
            .lock()
            .expect("voice_channels mutex")
            .is_empty()
    });
    let bot_user_id_cell = bot_user_id;
    let bot_user_id_fn: crate::activities::engine::BotUserIdFn =
        Arc::new(move || *bot_user_id_cell.lock().expect("bot_user_id mutex"));
    let focus: Arc<dyn FocusLike> = focus;

    Some(ActivityEngine::new(EngineParams {
        store: Arc::new(RealActivityStore(familiar.history_store.clone())),
        config,
        background_llm,
        bus: familiar.bus.clone(),
        focus,
        presence_cb: build_activity_presence_cb(handle),
        familiar_id: familiar.id.clone(),
        display_tz: familiar.config.display_tz.clone(),
        bot_user_id: bot_user_id_fn,
        sleep_window: familiar.config.sleep_window,
        sleep_grace_minutes: familiar.config.sleep_grace_minutes,
        voice_active_fn,
        clock: Arc::new(SystemClock),
        rng: Arc::new(LcgRng::default()),
        nudge_tick: std::time::Duration::from_secs(60),
        familiar_display_name: Some(familiar.display_name()),
        sleep_passes_enabled: true,
        seed_dream_path: Some(familiar.root.join("seed_dream.md")),
        sleep_prompts: SleepPromptText::from_config(
            familiar.config.sleep_consolidation_system.clone(),
            familiar.config.sleep_stance_system.clone(),
            familiar.config.sleep_synthesis_system.clone(),
        ),
        maintenance_runner: Arc::new(DefaultMaintenanceRunner),
    }))
}

// ---------------------------------------------------------------------------
// Cooperative shutdown (Python `_install_shutdown_handlers` / `_wait_for_shutdown`)
// ---------------------------------------------------------------------------

/// What a delivered signal should do: drain (first) or force-exit (second).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownStage {
    /// First signal — begin an orderly drain.
    Drain,
    /// Second signal — the operator wants an immediate force-kill.
    Force,
}

/// Two-stage cooperative shutdown coordinator.
///
/// The first delivered signal flips the [`CancellationToken`] (the run loop's
/// supervisor unwinds and teardown runs in normal task state — the Python
/// `_GracefulShutdown` path). A second signal returns [`ShutdownStage::Force`],
/// the caller's cue to restore the OS default and force-exit so a wedged
/// shutdown stays killable (Python's second-signal handler removal).
#[derive(Debug, Default)]
pub struct ShutdownController {
    cancel: CancellationToken,
    signalled: AtomicBool,
}

impl ShutdownController {
    /// A fresh controller with an un-cancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The cancellation token the run loop waits on.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Record a delivered signal; first cancels + drains, second forces.
    pub fn signal(&self) -> ShutdownStage {
        if self.signalled.swap(true, Ordering::SeqCst) {
            ShutdownStage::Force
        } else {
            self.cancel.cancel();
            ShutdownStage::Drain
        }
    }

    /// Park until the first signal cancels the token (Python `_wait_for_shutdown`).
    pub async fn wait(&self) {
        self.cancel.cancelled().await;
    }
}

// ---------------------------------------------------------------------------
// run (failure ladder)
// ---------------------------------------------------------------------------

/// Start the Discord bot (Python `run`).
///
/// Reads `DISCORD_BOT`, selects the familiar, and — with the `discord` feature —
/// loads the config + clients + familiar bundle and launches the gateway under
/// a tokio runtime. Returns the process exit code (`0` success, `1` on any
/// startup failure).
#[must_use]
pub fn run(args: &RunArgs) -> i32 {
    let token = std::env::var("DISCORD_BOT").unwrap_or_default();
    if token.is_empty() {
        tracing::error!("DISCORD_BOT environment variable is not set");
        return 1;
    }
    let root = default_familiars_root();
    // One-shot: relocate any legacy CWD-relative familiars into the resolved
    // root before we look one up (idempotent, never clobbers — issue #201).
    migrate_legacy_familiars(&legacy_familiars_root(), &root);
    let familiar_root = match resolve_familiar_root(
        args.familiar.as_deref(),
        std::env::var("FAMILIAR_ID").ok(),
        &root,
    ) {
        Ok(root) => root,
        Err(msg) => {
            tracing::error!("{msg}");
            return 1;
        }
    };
    run_inner(&token, &familiar_root)
}

#[cfg(not(feature = "discord"))]
fn run_inner(_token: &str, _familiar_root: &Path) -> i32 {
    tracing::error!(
        "familiar-connect was built without the `discord` feature; rebuild with \
         --features discord,discord-voice to connect to Discord"
    );
    1
}

#[cfg(feature = "discord")]
#[allow(
    clippy::too_many_lines,
    reason = "the composition root threads every subsystem; one cohesive function"
)]
fn run_inner(token: &str, familiar_root: &Path) -> i32 {
    use crate::config::load_character_config;
    use crate::embedding::known_embedders;
    use crate::llm::LlmClient;
    use crate::processors::projectors::known_projectors;

    // `_default` is a tracked repo resource resolved independently of where the
    // (now home-based) per-user familiars live, so it is not stranded by the
    // #201 move — never `familiar_root.parent()/_default`.
    let defaults_path = default_defaults_root()
        .join("_default")
        .join("character.toml");
    let known_proj = known_projectors();
    let known_emb = known_embedders();

    // Load merged config first so the client factories see per-slot models.
    let config = match load_character_config(
        &familiar_root.join("character.toml"),
        &defaults_path,
        &known_proj,
        &known_emb,
    ) {
        Ok(config) => config,
        Err(err) => {
            tracing::error!("Failed to load familiar config: {err}");
            return 1;
        }
    };

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        tracing::error!("OPENROUTER_API_KEY environment variable is required");
        return 1;
    }

    let llm_clients: HashMap<String, Arc<dyn LlmClient>> =
        match crate::llm::create_llm_clients(&api_key, &config) {
            Ok(clients) => clients
                .into_iter()
                .map(|(slot, client)| (slot, Arc::new(client) as Arc<dyn LlmClient>))
                .collect(),
            Err(err) => {
                tracing::error!("Character config is missing an LLM slot: {err}");
                return 1;
            }
        };

    // Degrade-not-fail: TTS / STT / local turn detector unavailability warns.
    let tts_client = match crate::tts::create_tts_client(&config.tts) {
        Ok(kind) => Some(kind.into_dyn()),
        Err(err) => {
            tracing::warn!("TTS client unavailable: {err}");
            None
        }
    };
    let transcriber = match crate::stt::create_transcriber(&config.stt) {
        Ok(transcriber) => Some(transcriber),
        Err(err) => {
            tracing::warn!("Transcriber unavailable: {err}");
            None
        }
    };
    let local_turn_detector = if config.turn_detection.strategy == "ten+smart_turn" {
        crate::voice::turn_detection::create_local_turn_detector(&config.turn_detection.local)
    } else {
        None
    };

    // Resolve the embedder BEFORE loading the familiar (which opens + possibly
    // rebuilds the history store / FTS). A misconfigured backend must refuse to
    // start here, while nothing has been mutated — not fail deep in
    // `create_projectors` after the store is already open. Its error names the
    // real fix (e.g. the `local-embed` extra), so surface it verbatim.
    let embedder = match resolve_embedder(&config.embedding) {
        Ok(embedder) => embedder,
        Err(err) => {
            tracing::error!("Embedding backend unavailable: {err}");
            return 1;
        }
    };

    let familiar = match Familiar::load_from_disk(
        familiar_root,
        llm_clients,
        tts_client,
        transcriber,
        local_turn_detector,
        Some(&defaults_path),
        &known_proj,
        &known_emb,
    ) {
        Ok(familiar) => familiar,
        Err(err) => {
            tracing::error!("Failed to load familiar config: {err}");
            return 1;
        }
    };

    // `load_opus` is a no-op in the Rust port: songbird statically links libopus,
    // so there is no ctypes-style runtime discovery to perform.

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            tracing::error!("failed to build tokio runtime: {err}");
            return 1;
        }
    };
    match runtime.block_on(async_main(token.to_owned(), familiar, embedder)) {
        Ok(()) => 0,
        Err(err) => {
            tracing::error!(
                "Discord bot exited with an error (if this is a login failure, the \
                 DISCORD_BOT token may be invalid or expired — generate a new one at \
                 https://discord.com/developers/applications): {err}"
            );
            1
        }
    }
}

// ---------------------------------------------------------------------------
// async_main (composition root) — cfg(discord)
// ---------------------------------------------------------------------------

/// A [`ResponderLlm`](crate::processors::ResponderLlm) adapter over a type-erased
/// [`LlmClient`](crate::llm::LlmClient).
///
/// The familiar stores clients as `Arc<dyn LlmClient>`, erasing the concrete
/// `image_tools_enabled` accessor the text responder needs; this thin adapter
/// re-attaches it (the Python `getattr(llm, "image_tools_enabled")` seam).
#[cfg(feature = "discord")]
struct ResponderLlmAdapter {
    inner: Arc<dyn crate::llm::LlmClient>,
    image_tools: bool,
}

#[cfg(feature = "discord")]
#[async_trait::async_trait]
impl crate::llm::LlmClient for ResponderLlmAdapter {
    async fn chat(
        &self,
        messages: Vec<crate::llm::Message>,
    ) -> anyhow::Result<crate::llm::Message> {
        self.inner.chat(messages).await
    }
    async fn stream_completion(
        &self,
        messages: Vec<crate::llm::Message>,
        tools: Option<Vec<serde_json::Value>>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<crate::llm::LlmDelta>>>
    {
        self.inner.stream_completion(messages, tools).await
    }
    fn slot(&self) -> Option<&str> {
        self.inner.slot()
    }
    fn multimodal(&self) -> bool {
        self.inner.multimodal()
    }
    fn tool_calling_enabled(&self) -> bool {
        self.inner.tool_calling_enabled()
    }
}

#[cfg(feature = "discord")]
impl crate::processors::ResponderLlm for ResponderLlmAdapter {
    fn image_tools_enabled(&self) -> bool {
        self.image_tools
    }
}

#[cfg(feature = "discord")]
fn responder_llm(
    inner: Arc<dyn crate::llm::LlmClient>,
    image_tools: bool,
) -> Arc<dyn crate::processors::ResponderLlm> {
    Arc::new(ResponderLlmAdapter { inner, image_tools })
}

/// Debug-logger observed topics (Python `_DEBUG_TOPICS`).
#[cfg(feature = "discord")]
const DEBUG_TOPICS: [&str; 4] = [
    crate::bus::topics::TOPIC_DISCORD_TEXT,
    crate::bus::topics::TOPIC_TWITCH_EVENT,
    crate::bus::topics::TOPIC_VOICE_ACTIVITY_START,
    crate::bus::topics::TOPIC_VOICE_TRANSCRIPT_FINAL,
];

/// Spawn the two-stage SIGINT/SIGTERM listener (Python
/// `_install_shutdown_handlers`).
#[cfg(all(feature = "discord", unix))]
fn spawn_signal_listener(controller: Arc<ShutdownController>) {
    use crate::log_style as ls;
    use tokio::signal::unix::{SignalKind, signal};
    tokio::spawn(async move {
        let (Ok(mut sigint), Ok(mut sigterm)) = (
            signal(SignalKind::interrupt()),
            signal(SignalKind::terminate()),
        ) else {
            tracing::debug!("signal handlers unavailable on this platform");
            return;
        };
        loop {
            let name = tokio::select! {
                _ = sigint.recv() => "SIGINT",
                _ = sigterm.recv() => "SIGTERM",
            };
            match controller.signal() {
                ShutdownStage::Drain => tracing::info!(
                    "{} {} draining — {}",
                    ls::tag("Shutdown", ls::Y),
                    ls::kv("signal", name),
                    ls::word("signal again to force", ls::LY),
                ),
                ShutdownStage::Force => {
                    tracing::warn!("{} forced — exiting", ls::tag("Shutdown", ls::Y));
                    std::process::exit(130);
                }
            }
        }
    });
}

/// Drop persisted DM subscription rows whose peer left the allowlist.
///
/// Guild rows (`dm_user_id` is `None`) are untouched. Removal rewrites the
/// sidecar, so a de-allowlisted peer's row cannot resurface on restart or win
/// the seeded text focus (Python `_prune_deallowlisted_dm_subscriptions`).
#[cfg(feature = "discord")]
fn prune_deallowlisted_dm_subscriptions(
    subscriptions: &mut crate::subscriptions::SubscriptionRegistry,
    dm_allowlist: &[i64],
) -> Result<(), crate::subscriptions::SubscriptionError> {
    for sub in subscriptions.all() {
        let Some(dm_user_id) = sub.dm_user_id else {
            continue;
        };
        if !dm_allowlist.contains(&dm_user_id) {
            tracing::info!(
                channel_id = sub.channel_id,
                peer = dm_user_id,
                "pruned DM subscription — peer no longer allowlisted"
            );
            subscriptions.remove(sub.channel_id, sub.kind)?;
        }
    }
    Ok(())
}

/// A DM channel has one human peer; a small window still tolerates stray
/// authors without scanning the whole channel history.
#[cfg(feature = "discord")]
const DM_PEER_AUTHOR_LIMIT: i64 = 5;

/// Restore DM naming for persisted DM subscriptions after a restart.
///
/// Mirrors what `register_dm_channel` records live: the sentinel guild name
/// (DM detection keys off it) and the peer's display name, recovered from
/// history via the author row matching the subscription's `dm_user_id`. When
/// history has no such author, `channel_names` stays unset and the digest falls
/// back to `DM (id <cid>)`. Guild rows (`dm_user_id` is `None`) are untouched
/// (Python `_rehydrate_dm_naming`).
#[cfg(feature = "discord")]
async fn rehydrate_dm_naming(
    focus_manager: &FocusManager,
    subscriptions: &dyn crate::subscriptions::SubscriptionView,
    store: &crate::history::async_store::AsyncHistoryStore,
    familiar_id: &str,
) -> Result<(), crate::history::StoreError> {
    for sub in subscriptions.all() {
        let Some(dm_user_id) = sub.dm_user_id else {
            continue;
        };
        let Ok(channel_id) = i64::try_from(sub.channel_id) else {
            continue;
        };
        focus_manager.set_guild_name(channel_id, crate::focus::PRIVATE_MESSAGE_GUILD_NAME);
        let authors = store
            .recent_distinct_authors(familiar_id.to_owned(), channel_id, DM_PEER_AUTHOR_LIMIT)
            .await?;
        let Some(peer) = authors
            .into_iter()
            .find(|a| a.user_id == dm_user_id.to_string())
        else {
            continue;
        };
        let name = peer
            .display_name
            .filter(|s| !s.is_empty())
            .or_else(|| peer.username.filter(|s| !s.is_empty()));
        if let Some(name) = name {
            focus_manager.set_channel_name(channel_id, name);
        }
    }
    Ok(())
}

/// Boot-time DM validation, naming, and default-focus seeding — the ordering
/// contract, in one testable unit (the tail of Python `_async_main`).
///
/// Order is load-bearing: (1) prune de-allowlisted DM rows so a stale DM can
/// neither survive nor win the focus seed, and so `initialize`'s
/// `keep_if_subscribed` drops a pointer at a de-allowlisted DM; (2) initialize
/// focus from the persisted pointers; (3) rehydrate DM naming for surviving DM
/// rows *before* seeding, so a seeded DM already renders by name; (4) seed the
/// first subscribed channel per modality when that modality has no focus.
///
/// `focus_manager` must be freshly constructed (not yet initialized). Callers
/// keep ownership of the shared registry; this only reads/mutates through the
/// borrow.
#[cfg(feature = "discord")]
async fn boot_dm_focus(
    focus_manager: &FocusManager,
    subscriptions: &std::sync::Mutex<crate::subscriptions::SubscriptionRegistry>,
    store: &crate::history::async_store::AsyncHistoryStore,
    dm_allowlist: &[i64],
    familiar_id: &str,
) -> anyhow::Result<()> {
    use crate::subscriptions::{SubscriptionKind, SubscriptionView};

    prune_deallowlisted_dm_subscriptions(
        &mut subscriptions.lock().expect("subscriptions mutex"),
        dm_allowlist,
    )?;
    focus_manager.initialize().await;
    rehydrate_dm_naming(focus_manager, subscriptions, store, familiar_id).await?;

    for sub in subscriptions.all() {
        let cid = i64::try_from(sub.channel_id).unwrap_or_default();
        match sub.kind {
            SubscriptionKind::Text if focus_manager.get_focus("text").is_none() => {
                focus_manager.set_focus_immediately(cid, "text");
            }
            SubscriptionKind::Voice if focus_manager.get_focus("voice").is_none() => {
                focus_manager.set_focus_immediately(cid, "voice");
            }
            _ => {}
        }
    }
    Ok(())
}

/// Asyncio entry point: bring up the bus, responders, workers, and gateway, then
/// tear down in order (Python `_async_main`).
///
/// # Errors
/// Propagates a serenity client build/start failure (surfaced by `run` with a
/// token-regeneration hint).
#[cfg(feature = "discord")]
#[allow(
    clippy::too_many_lines,
    reason = "the composition root wires every subsystem; splitting obscures the ordering contract"
)]
async fn async_main(
    token: String,
    mut familiar: Familiar,
    embedder: Option<Arc<dyn Embedder>>,
) -> anyhow::Result<()> {
    use std::sync::Mutex;

    use crate::bot::{ActivityResync, AsyncBotStore, BotStore, CreateBotDeps, create_bot};
    use crate::bus::protocols::{BackpressurePolicy, Processor};
    use crate::log_style as ls;
    use crate::processors::debug_logger::DebugLoggerProcessor;
    use crate::processors::projectors::{ProjectorContext, create_projectors};
    use crate::processors::text_responder::TextResponder;
    use crate::processors::voice_responder::VoiceResponder;
    use crate::processors::{ActivityGate, FocusManagerApi, MemberResolver};
    use crate::subscriptions::SubscriptionRegistry;
    use crate::tools::builtins::{build_text_registry, build_voice_registry};
    use crate::tools::registry::{ChannelReadStore, FocusControl, ToolContext};
    use crate::tools::scheduler::AlarmScheduler;
    use crate::tools::start_activity::StartActivityEngine;
    use crate::tools::waker::AlarmWaker;
    use crate::tts_player::{DiscordVoicePlayer, LoggingTTSPlayer, TtsPlayer, VoiceClientLike};

    familiar.bus.start().await;

    // Subscriptions: ONE shared-mutable registry (`Arc<Mutex<…>>`) consumed by
    // both the bot (which mutates it on `/subscribe*`) and the focus manager
    // (which reads it through the `SubscriptionView` seam). Mirrors Python, where
    // `bot` and `focus` share a single registry object, so a runtime
    // `/subscribe` mutation is visible to `is_focused` / `should_wake` /
    // startup-default-focus / `staged_channels` logic without a restart.
    let subs_path = familiar.root.join("subscriptions.toml");
    let subscriptions = Arc::new(Mutex::new(SubscriptionRegistry::new(&subs_path)?));

    let focus_manager = Arc::new(
        FocusManager::new(
            familiar.id.clone(),
            familiar.history_store.clone(),
            subscriptions.clone(),
        )
        .with_unread_nudge_enabled(familiar.config.focus.unread_nudge_enabled)
        .with_nudge_debounce_seconds(familiar.config.focus.nudge_debounce_seconds)
        .with_catch_up_limit(usize::try_from(familiar.config.focus.catch_up_limit).unwrap_or(20)),
    );

    // Boot DM validation + naming + default-focus seeding, in one ordered unit:
    // prune de-allowlisted DM rows, initialize focus, rehydrate DM naming, then
    // seed the default focus (see `boot_dm_focus` for the ordering contract).
    boot_dm_focus(
        focus_manager.as_ref(),
        subscriptions.as_ref(),
        familiar.history_store.as_ref(),
        &familiar.config.dm_allowlist,
        &familiar.id,
    )
    .await?;

    let bot_user_id = Arc::new(Mutex::new(None::<i64>));
    let (handle, client) = create_bot(CreateBotDeps {
        token: token.clone(),
        familiar_id: familiar.id.clone(),
        bot_user_id: bot_user_id.clone(),
        subscriptions: subscriptions.clone(),
        dm_allowlist: familiar.config.dm_allowlist.clone(),
        router: familiar.router.clone(),
        discord_text: familiar.config.discord_text.clone(),
        // Voice-intake prototype: clone the familiar's transcriber so
        // `/subscribe-voice` clones a fresh per-speaker WS from it while the
        // familiar keeps its own copy for teardown. `None` → playback-only join.
        transcriber_template: familiar
            .transcriber
            .as_ref()
            .map(|t| Arc::new(Mutex::new(t.clone_transcriber()))),
        local_turn_detector: familiar.local_turn_detector.take().map(Arc::new),
        store: {
            // Route reaction / edit writes through the async store's
            // blocking-thread facade (off the reactor, DESIGN §4.4) rather than
            // running rusqlite inline on the gateway task; see `AsyncBotStore`.
            let store: Arc<dyn BotStore> =
                Arc::new(AsyncBotStore::new(familiar.history_store.clone()));
            store
        },
        history_store: familiar.history_store.clone(),
        bus: familiar.bus.clone(),
        focus_manager: Some(focus_manager.clone()),
    })
    .await?;

    // Embedder resolved at the composition root (before the store opened); a bad
    // backend would already have aborted startup in `run_inner`.
    let voice_assembler = Arc::new(default_assembler(
        &familiar,
        familiar.config.voice_window_size,
        familiar.config.budget_for("voice"),
        0.0,
        embedder.clone(),
        Some(focus_manager.clone()),
    ));
    let text_assembler = Arc::new(default_assembler(
        &familiar,
        familiar.config.text_window_size,
        familiar.config.budget_for("text"),
        familiar.config.text_silence_gap_fold_seconds,
        embedder.clone(),
        Some(focus_manager.clone()),
    ));

    let tts_player: Arc<dyn TtsPlayer> = if let Some(tts) = familiar.tts_client.clone() {
        // Voice-client retrieval seam: read the live voice client `/subscribe-voice`
        // stored in `handle.voice_runtime`. That map is a `BTreeMap`, so
        // `.values().next()` deterministically yields the lowest-keyed entry and
        // is stable across utterances (v1 supports one voice channel at a time, so
        // the single entry is unambiguous — Python `_first_voice_client`).
        // Under a `discord`-only build (no `discord-voice`), the runtime map does
        // not exist, so playback degrades to no-op.
        let vc_handle = handle.clone();
        Arc::new(DiscordVoicePlayer::new(
            tts,
            move || -> Option<Arc<dyn VoiceClientLike>> {
                #[cfg(feature = "discord-voice")]
                {
                    vc_handle
                        .voice_runtime
                        .lock()
                        .expect("voice_runtime mutex")
                        .values()
                        .next()
                        .map(|rt| rt.voice_client.clone())
                }
                #[cfg(not(feature = "discord-voice"))]
                {
                    let _ = &vc_handle;
                    None
                }
            },
        ))
    } else {
        Arc::new(LoggingTTSPlayer::default())
    };

    let alarm_scheduler = Arc::new(AlarmScheduler::new(
        familiar.history_store.clone(),
        familiar.bus.clone(),
        familiar.id.clone(),
    ));

    let activity_engine = build_activity_engine(
        &familiar,
        focus_manager.clone(),
        handle.clone(),
        bot_user_id.clone(),
    );
    if let Some(engine) = &activity_engine {
        let resync: Arc<dyn ActivityResync> = engine.clone();
        handle.set_activity_engine(resync);
    }

    let prose_image_tools = familiar
        .config
        .llm
        .get("prose")
        .is_some_and(|slot| slot.image_tools);
    let voice_tool_registry = Arc::new(build_voice_registry(&alarm_scheduler, true));
    let text_activity_engine: Option<Arc<dyn StartActivityEngine>> =
        activity_engine.clone().map(|engine| {
            let engine: Arc<dyn StartActivityEngine> = engine;
            engine
        });
    let text_tool_registry = Arc::new(build_text_registry(
        &alarm_scheduler,
        prose_image_tools,
        &familiar.config.image_description_constraints,
        true,
        text_activity_engine,
    ));

    let description_llm = familiar.llm_clients.get("__image_description__").cloned();

    // Per-turn ToolContext factory (Python `_make_tool_context`).
    let make_factory = |channel_kind: &'static str, with_description: bool| {
        let familiar_id = familiar.id.clone();
        let history = familiar.history_store.clone();
        let bus = familiar.bus.clone();
        let scheduler = alarm_scheduler.clone();
        let fm = focus_manager.clone();
        let description = if with_description {
            description_llm.clone()
        } else {
            None
        };
        let factory: crate::processors::ToolContextFactory = Arc::new(
            move |channel_id: i64, turn_id: &str, images: HashMap<String, String>| {
                let focus_control: Arc<dyn FocusControl> = fm.clone();
                let read_store: Arc<dyn ChannelReadStore> = history.clone();
                let mut ctx =
                    ToolContext::new(familiar_id.clone(), channel_id, channel_kind, turn_id)
                        .with_history(history.clone())
                        .with_bus(bus.clone())
                        .with_scheduler(scheduler.clone())
                        .with_images(images)
                        .with_focus_manager(focus_control)
                        .with_store(read_store);
                if let Some(description) = &description {
                    ctx = ctx.with_description_llm(description.clone());
                }
                ctx
            },
        );
        factory
    };
    let voice_factory = make_factory("voice", false);
    let text_factory = make_factory("text", true);

    let alarm_waker = AlarmWaker::new(familiar.id.clone());

    let loop_max = usize::try_from(familiar.config.tools.loop_max_iterations).unwrap_or(5);
    let fast_llm = responder_llm(
        familiar
            .llm_clients
            .get("fast")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing 'fast' LLM slot"))?,
        false,
    );
    let prose_llm = responder_llm(
        familiar
            .llm_clients
            .get("prose")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing 'prose' LLM slot"))?,
        prose_image_tools,
    );

    let member_resolver: Option<MemberResolver> = handle
        .resolve_member
        .lock()
        .expect("resolve_member mutex")
        .clone();
    let mut voice_responder = VoiceResponder::new(
        voice_assembler,
        fast_llm,
        tts_player.clone(),
        familiar.history_store.clone(),
        familiar.router.clone(),
        familiar.id.clone(),
    )
    .with_tools(voice_tool_registry, voice_factory)
    // Filler phrases disabled 2026-06-25 (too chatty in voice).
    .with_tool_filler_phrases(Vec::new())
    .with_post_history_instructions(familiar.config.post_history_instructions.clone())
    .with_display_tz(familiar.config.display_tz.clone())
    .with_focus_manager(focus_manager.clone() as Arc<dyn FocusManagerApi>)
    .with_loop_max_iterations(loop_max);
    if let Some(resolver) = member_resolver {
        voice_responder = voice_responder.with_member_resolver(resolver);
    }

    let mut text_responder = TextResponder::new(
        text_assembler,
        prose_llm,
        handle.send_text.clone(),
        familiar.history_store.clone(),
        familiar.router.clone(),
        familiar.id.clone(),
    )
    .with_tools(text_tool_registry, text_factory)
    .with_post_history_instructions(familiar.config.post_history_instructions.clone())
    .with_display_tz(familiar.config.display_tz.clone())
    .with_focus_manager(focus_manager.clone() as Arc<dyn FocusManagerApi>)
    .with_loop_max_iterations(loop_max);
    if let Some(typing) = &handle.trigger_typing {
        text_responder = text_responder.with_trigger_typing(typing.clone());
    }
    if let Some(handler) = &handle.typing_interrupt {
        text_responder = text_responder.with_typing_handler(handler.clone());
    }
    if let Some(engine) = &activity_engine {
        text_responder =
            text_responder.with_activity_engine(engine.clone() as Arc<dyn ActivityGate>);
    }

    let mut projector_context =
        ProjectorContext::new(familiar.history_store.clone(), familiar.id.clone());
    projector_context.llm_clients = familiar
        .llm_clients
        .iter()
        .map(|(slot, client)| (slot.clone(), client.clone()))
        .collect();
    projector_context.embedder = embedder;
    projector_context.memory = familiar.config.memory_providers.clone();
    projector_context.familiar_display_name = Some(familiar.display_name());
    projector_context.dream_extraction_clause = familiar.config.dream_extraction_clause.clone();
    let projectors = create_projectors(
        &familiar.config.memory_providers.projectors,
        &projector_context,
    )
    .map_err(|err| anyhow::anyhow!("failed to build projectors: {err}"))?;

    // Pending alarms count down before the bot accepts traffic.
    alarm_scheduler
        .start()
        .await
        .map_err(|err| anyhow::anyhow!("alarm scheduler start failed: {err}"))?;
    if let Some(engine) = &activity_engine {
        engine.start().await;
    }

    // -- run loop --------------------------------------------------------
    let controller = Arc::new(ShutdownController::new());
    let cancel = controller.token();
    #[cfg(unix)]
    spawn_signal_listener(controller.clone());

    let mut set = tokio::task::JoinSet::new();

    // All four production subscriptions below use the default BLOCK/64 policy
    // (`maxsize == 0`), matching Python's `bus.subscribe(proc.topics)` (no policy
    // arg → `BackpressurePolicy.BLOCK`, maxsize 64). Spec 01 pins this: the ADR's
    // "unbounded for text/twitch" note describes intent for topics not yet on the
    // bus, so the head-of-line backpressure coupling (spec 01 §6) stays in force.

    // debug-logger
    {
        let bus = familiar.bus.clone();
        let cancel = cancel.clone();
        set.spawn(async move {
            let proc = DebugLoggerProcessor::new(DEBUG_TOPICS);
            let topics = proc.topics();
            let mut sub = bus.subscribe(&topics, BackpressurePolicy::Block, 0);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => break,
                    event = sub.recv() => {
                        let Some(event) = event else { break };
                        let _ = proc.handle(event.as_ref(), bus.as_ref()).await;
                    }
                }
            }
        });
    }

    // voice-responder
    {
        let bus = familiar.bus.clone();
        let cancel = cancel.clone();
        set.spawn(async move {
            let topics = voice_responder.topics();
            let mut sub = bus.subscribe(&topics, BackpressurePolicy::Block, 0);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => break,
                    event = sub.recv() => {
                        let Some(event) = event else { break };
                        if let Err(err) = voice_responder.handle(event.as_ref(), bus.as_ref()).await {
                            tracing::warn!("voice-responder error: {err}");
                        }
                    }
                }
            }
        });
    }

    // text-responder
    {
        let bus = familiar.bus.clone();
        let cancel = cancel.clone();
        set.spawn(async move {
            let topics = text_responder.topics();
            let mut sub = bus.subscribe(&topics, BackpressurePolicy::Block, 0);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => break,
                    event = sub.recv() => {
                        let Some(event) = event else { break };
                        if let Err(err) = text_responder.handle(event.as_ref(), bus.as_ref()).await {
                            tracing::warn!("text-responder error: {err}");
                        }
                    }
                }
            }
        });
    }

    // alarm-waker
    {
        let bus = familiar.bus.clone();
        let cancel = cancel.clone();
        set.spawn(async move {
            let topics: Vec<&str> = alarm_waker.topics().to_vec();
            let mut sub = bus.subscribe(&topics, BackpressurePolicy::Block, 0);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => break,
                    event = sub.recv() => {
                        let Some(event) = event else { break };
                        let _ = alarm_waker.handle(event, bus.as_ref()).await;
                    }
                }
            }
        });
    }

    // one task per projector (each runs its own loop until aborted)
    for projector in projectors {
        set.spawn(async move {
            projector.run().await;
        });
    }

    // discord gateway
    let shard_manager = client.shard_manager.clone();
    let mut bot_task = tokio::spawn(async move {
        let mut client = client;
        client.start().await
    });

    // Supervisor: first of {shutdown signal, bot task ends}.
    let bot_ended = tokio::select! {
        () = cancel.cancelled() => None,
        result = &mut bot_task => Some(result),
    };
    if bot_ended.is_some() {
        // A bot exit means every sibling should drain too.
        cancel.cancel();
    } else {
        tracing::info!(
            "{} {}",
            ls::tag("Shutdown", ls::G),
            ls::word("clean", ls::LG)
        );
    }

    // -- teardown (order pinned, spec 10 §57) ----------------------------
    // 1. (signal handlers auto-drop with the runtime)
    // 2. close the gateway first so serenity's session does not leak.
    shard_manager.shutdown_all().await;
    if bot_ended.is_none() {
        bot_task.abort();
        let _ = (&mut bot_task).await;
    }
    // abort the responder / projector tasks (already unwinding on `cancel`).
    set.shutdown().await;
    // 2b. tear down any live voice-intake pipelines (per-speaker pumps / fan-ins
    //     cancelled, transcriber WSs closed) before stopping the template.
    #[cfg(feature = "discord-voice")]
    {
        let runtimes: Vec<_> =
            std::mem::take(&mut *handle.voice_runtime.lock().expect("voice_runtime mutex"))
                .into_values()
                .collect();
        for rt in runtimes {
            crate::bot::voice_intake::stop_voice_intake(rt).await;
        }
    }
    // 3. stop the transcriber template.
    if let Some(transcriber) = familiar.transcriber.as_mut() {
        transcriber.stop().await;
    }
    // 4. alarm scheduler.
    alarm_scheduler.shutdown().await;
    // 5. activity engine.
    if let Some(engine) = &activity_engine {
        engine.stop().await;
    }
    // 6. router (sync, not suppressed).
    familiar.router.shutdown();
    // 7. bus.
    familiar.bus.shutdown().await;
    // 8. (the LLM client `close()` step has no Rust analog — reqwest connection
    //     pools close on drop.)

    match bot_ended {
        None | Some(Ok(Ok(()))) => Ok(()),
        Some(Ok(Err(err))) => Err(anyhow::anyhow!(err)),
        Some(Err(err)) => Err(anyhow::anyhow!(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ShutdownController, ShutdownStage, build_activity_engine, default_assembler,
        home_familiars_root, migrate_legacy_familiars, resolve_defaults_root, resolve_embedder,
        resolve_familiar_root, resolve_familiars_root,
    };
    use crate::activities::engine::ActivityEngine;
    use crate::bot::{BotHandle, Presence, PresenceSink};
    use crate::budget::TierBudget;
    use crate::config::EmbeddingConfig;
    use crate::familiar::Familiar;
    use crate::focus::FocusManager;
    use crate::processors::SendText;
    use crate::subscriptions::SubscriptionRegistry;
    use async_trait::async_trait;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    // --- resolve_embedder (composition-root fail-fast) ---

    /// The composition root must surface the embedder feature-gap as an error
    /// (which names the `local-embed` fix), not swallow it into `None` and let
    /// startup proceed to mutate the store. The default test build lacks the
    /// `local-embed` extra, so `fastembed` genuinely has no backend here.
    #[cfg(not(feature = "local-embed"))]
    #[test]
    fn resolve_embedder_fastembed_without_extra_errors() {
        let config = EmbeddingConfig {
            backend: "fastembed".to_owned(),
            ..EmbeddingConfig::default()
        };
        let err = resolve_embedder(&config)
            .err()
            .expect("fastembed without the local-embed extra must fail fast");
        assert!(
            err.to_string().contains("local-embed"),
            "error must name the real fix, got: {err}"
        );
    }

    /// A disabled backend resolves to `None` without erroring — fail-fast must
    /// not become fail-always.
    #[test]
    fn resolve_embedder_off_backend_is_none() {
        let config = EmbeddingConfig {
            backend: "off".to_owned(),
            ..EmbeddingConfig::default()
        };
        assert!(resolve_embedder(&config).unwrap().is_none());
    }

    // --- resolve_familiar_root (ported from test_run_cmd.py) ---

    #[test]
    fn flag_overrides_env() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("flag")).unwrap();
        std::fs::create_dir(dir.path().join("env")).unwrap();
        let root = resolve_familiar_root(Some("flag"), Some("env".to_owned()), dir.path()).unwrap();
        assert_eq!(root, dir.path().join("flag"));
    }

    #[test]
    fn env_used_when_no_flag() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("env-chosen")).unwrap();
        let root = resolve_familiar_root(None, Some("env-chosen".to_owned()), dir.path()).unwrap();
        assert_eq!(root, dir.path().join("env-chosen"));
    }

    #[test]
    fn no_id_selected_errors() {
        let dir = TempDir::new().unwrap();
        let err = resolve_familiar_root(None, None, dir.path()).unwrap_err();
        assert!(err.contains("No familiar selected"));
    }

    #[test]
    fn missing_folder_errors() {
        let dir = TempDir::new().unwrap();
        let err = resolve_familiar_root(Some("does-not-exist"), None, dir.path()).unwrap_err();
        assert!(err.contains("Familiar folder does not exist"));
    }

    // --- home-dir familiar storage (issue #201) ---

    #[test]
    fn familiars_root_env_override_wins_else_home() {
        let home = PathBuf::from("/home/x/.local/share/familiar-connect/familiars");
        // An explicit, non-empty override takes top precedence.
        assert_eq!(
            resolve_familiars_root(Some("/custom/root".to_owned()), home.clone()),
            PathBuf::from("/custom/root")
        );
        // An empty override falls through to the home default.
        assert_eq!(
            resolve_familiars_root(Some(String::new()), home.clone()),
            home
        );
        // An unset override falls through to the home default.
        assert_eq!(resolve_familiars_root(None, home.clone()), home);
    }

    #[test]
    fn home_familiars_root_lives_under_the_platform_data_dir() {
        let root = home_familiars_root();
        // The leaf is always `familiars` (both the home and legacy branches).
        assert_eq!(root.file_name().and_then(|s| s.to_str()), Some("familiars"));
        // When a home directory resolves, it sits under the app's data dir.
        if directories::ProjectDirs::from("", "", "familiar-connect").is_some() {
            assert!(
                root.ends_with(Path::new("familiar-connect").join("familiars")),
                "unexpected familiars root: {}",
                root.display()
            );
        }
    }

    #[test]
    fn defaults_root_env_override_wins_else_cwd() {
        // The tracked `_default` root is decoupled from where user familiars live.
        assert_eq!(
            resolve_defaults_root(Some("/opt/fc".to_owned())),
            PathBuf::from("/opt/fc")
        );
        assert_eq!(
            resolve_defaults_root(Some(String::new())),
            Path::new("data").join("familiars")
        );
        assert_eq!(
            resolve_defaults_root(None),
            Path::new("data").join("familiars")
        );
    }

    #[test]
    fn bundled_default_profile_still_resolves() {
        // `_default` is a repo resource: joining it onto the defaults root finds
        // the shipped profile the run loop loads as merge defaults, independent of
        // the (now home-based) per-user familiar root.
        let profile = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("data")
            .join("familiars")
            .join("_default")
            .join("character.toml");
        assert!(
            profile.exists(),
            "missing bundled default profile at {}",
            profile.display()
        );
    }

    #[test]
    fn migration_moves_legacy_familiars_and_spares_default() {
        let tmp = TempDir::new().unwrap();
        let legacy = tmp.path().join("legacy");
        let home = tmp.path().join("home");
        std::fs::create_dir_all(legacy.join("aria")).unwrap();
        std::fs::write(legacy.join("aria").join("character.toml"), "id = 1").unwrap();
        std::fs::create_dir_all(legacy.join("_default")).unwrap();
        std::fs::write(legacy.join("_default").join("character.toml"), "d = 1").unwrap();

        migrate_legacy_familiars(&legacy, &home);

        // The user familiar moved to the new root, contents intact.
        assert!(home.join("aria").join("character.toml").exists());
        assert!(!legacy.join("aria").exists());
        // The tracked `_default` skeleton stays put and is never copied over.
        assert!(legacy.join("_default").join("character.toml").exists());
        assert!(!home.join("_default").exists());

        // Idempotent: a second run has nothing left to move and does not error.
        migrate_legacy_familiars(&legacy, &home);
        assert!(home.join("aria").join("character.toml").exists());
    }

    #[test]
    fn migration_never_clobbers_existing_home_familiar() {
        let tmp = TempDir::new().unwrap();
        let legacy = tmp.path().join("legacy");
        let home = tmp.path().join("home");
        std::fs::create_dir_all(legacy.join("aria")).unwrap();
        std::fs::write(legacy.join("aria").join("character.toml"), "legacy").unwrap();
        std::fs::create_dir_all(home.join("aria")).unwrap();
        std::fs::write(home.join("aria").join("character.toml"), "home").unwrap();

        migrate_legacy_familiars(&legacy, &home);

        // The home copy is authoritative and untouched; the legacy copy is left
        // in place rather than overwriting it.
        assert_eq!(
            std::fs::read_to_string(home.join("aria").join("character.toml")).unwrap(),
            "home"
        );
        assert!(legacy.join("aria").exists());
    }

    // --- test familiar bundle ---

    fn default_profile() -> PathBuf {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../data/familiars/_default/character.toml"
        ))
        .to_path_buf()
    }

    fn fake_clients() -> std::collections::HashMap<String, Arc<dyn crate::llm::LlmClient>> {
        use async_trait::async_trait;
        use futures::stream::{self, BoxStream};
        use serde_json::Value;

        struct FakeLlm(String);
        #[async_trait]
        impl crate::llm::LlmClient for FakeLlm {
            async fn chat(
                &self,
                _messages: Vec<crate::llm::Message>,
            ) -> anyhow::Result<crate::llm::Message> {
                Ok(crate::llm::Message::new("assistant", "ok"))
            }
            async fn stream_completion(
                &self,
                _messages: Vec<crate::llm::Message>,
                _tools: Option<Vec<Value>>,
            ) -> anyhow::Result<BoxStream<'static, anyhow::Result<crate::llm::LlmDelta>>>
            {
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
        ["fast", "prose", "background"]
            .into_iter()
            .map(|slot| {
                (
                    slot.to_owned(),
                    Arc::new(FakeLlm(slot.to_owned())) as Arc<dyn crate::llm::LlmClient>,
                )
            })
            .collect()
    }

    fn projectors_set() -> BTreeSet<String> {
        crate::processors::projectors::known_projectors()
    }

    fn embedders_set() -> BTreeSet<String> {
        crate::embedding::known_embedders()
    }

    const ACTIVITIES_TOML: &str = "[[catalog]]\n\
         id = \"walk\"\n\
         label = \"creek walk\"\n\
         duration_minutes = [20, 40]\n\
         seed = \"A walk along the creek.\"\n";

    fn load_test_familiar(with_activities: bool) -> (Familiar, TempDir) {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("aria");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::copy(default_profile(), root.join("character.toml")).unwrap();
        if with_activities {
            std::fs::write(root.join("activities.toml"), ACTIVITIES_TOML).unwrap();
        }
        let familiar = Familiar::load_from_disk(
            &root,
            fake_clients(),
            None,
            None,
            None,
            Some(&default_profile()),
            &projectors_set(),
            &embedders_set(),
        )
        .unwrap();
        (familiar, dir)
    }

    fn test_focus_manager(familiar: &Familiar) -> Arc<FocusManager> {
        let subs =
            Arc::new(SubscriptionRegistry::new(familiar.root.join("subscriptions.toml")).unwrap());
        Arc::new(FocusManager::new(
            familiar.id.clone(),
            familiar.history_store.clone(),
            subs,
        ))
    }

    struct NoopSend;
    #[async_trait]
    impl SendText for NoopSend {
        async fn send(
            &self,
            _channel_id: i64,
            _content: &str,
            _reply_to_message_id: Option<&str>,
            _mention_user_ids: &[i64],
        ) -> anyhow::Result<Option<String>> {
            Ok(None)
        }
    }

    struct NoopPresence;
    #[async_trait]
    impl PresenceSink for NoopPresence {
        fn is_ready(&self) -> bool {
            false
        }
        async fn set_presence(&self, _presence: Presence) {}
    }

    fn test_handle() -> Arc<BotHandle> {
        Arc::new(BotHandle::new(Arc::new(NoopSend), Arc::new(NoopPresence)))
    }

    // --- default_assembler layer order (ported from
    //     test_run_cmd.py::TestDefaultAssemblerLayerOrder, adapted to D15) ---

    fn layer_order() -> Vec<String> {
        let (familiar, _dir) = load_test_familiar(false);
        let asm = default_assembler(&familiar, 20, TierBudget::default(), 0.0, None, None);
        asm.layer_names().into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn conversation_summary_precedes_reflection() {
        let order = layer_order();
        let summary = order
            .iter()
            .position(|n| n == "conversation_summary")
            .unwrap();
        let reflection = order.iter().position(|n| n == "reflection").unwrap();
        assert!(summary < reflection);
    }

    #[test]
    fn people_dossier_precedes_rag() {
        let order = layer_order();
        let dossier = order.iter().position(|n| n == "people_dossier").unwrap();
        let rag = order.iter().position(|n| n == "rag_context").unwrap();
        assert!(dossier < rag);
    }

    #[test]
    fn rag_is_last_system_prompt_layer() {
        // Per DESIGN D15 recent-history is a slot, so rag_context is the tail of
        // the system-prompt layer vec (recent-history is not among the names).
        let order = layer_order();
        let rag = order.iter().position(|n| n == "rag_context").unwrap();
        assert_eq!(rag, order.len() - 1);
        assert!(!order.iter().any(|n| n == "recent_history"));
    }

    // --- build_activity_engine (ported from
    //     test_run_cmd.py::TestBuildActivityEngine) ---

    #[test]
    fn missing_sidecar_disables_engine() {
        let (familiar, _dir) = load_test_familiar(false);
        let fm = test_focus_manager(&familiar);
        let engine =
            build_activity_engine(&familiar, fm, test_handle(), Arc::new(Mutex::new(None)));
        assert!(engine.is_none());
    }

    #[test]
    fn catalog_enables_engine() {
        let (familiar, _dir) = load_test_familiar(true);
        let fm = test_focus_manager(&familiar);
        let engine =
            build_activity_engine(&familiar, fm, test_handle(), Arc::new(Mutex::new(None)));
        assert!(engine.is_some());
        let _typed: Arc<ActivityEngine> = engine.unwrap();
    }

    #[test]
    fn voice_active_fn_tracks_channel_set() {
        let (familiar, _dir) = load_test_familiar(true);
        let handle = test_handle();
        let engine = build_activity_engine(
            &familiar,
            test_focus_manager(&familiar),
            handle.clone(),
            Arc::new(Mutex::new(None)),
        )
        .expect("engine");
        assert_eq!(
            engine
                .defer_start("walk", None)
                .get("ack")
                .and_then(|v| v.as_str()),
            Some("ok")
        );

        // A live voice channel refuses a fresh engine's activity start.
        handle
            .voice_channels
            .lock()
            .expect("voice_channels")
            .insert(1);
        let other = build_activity_engine(
            &familiar,
            test_focus_manager(&familiar),
            handle,
            Arc::new(Mutex::new(None)),
        )
        .expect("engine");
        let error = other.defer_start("walk", None);
        assert!(
            error
                .get("error")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains("voice"))
        );
    }

    // --- ShutdownController (ports the spirit of _wait_for_shutdown /
    //     _install_shutdown_handlers two-stage semantics) ---

    #[test]
    fn first_signal_drains_second_forces() {
        let controller = ShutdownController::new();
        assert!(!controller.token().is_cancelled());
        assert_eq!(controller.signal(), ShutdownStage::Drain);
        assert!(controller.token().is_cancelled());
        assert_eq!(controller.signal(), ShutdownStage::Force);
    }

    #[tokio::test]
    async fn wait_returns_after_signal() {
        let controller = Arc::new(ShutdownController::new());
        let waiter = {
            let controller = controller.clone();
            tokio::spawn(async move { controller.wait().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        controller.signal();
        waiter.await.unwrap();
    }

    // --- boot DM subscription validation + naming (PR #194) ----------------

    #[cfg(feature = "discord")]
    mod boot_dm {
        use super::super::{
            boot_dm_focus, prune_deallowlisted_dm_subscriptions, rehydrate_dm_naming,
        };
        use crate::context::final_reminder::FinalReminder;
        use crate::focus::{FocusManager, FocusStore, PRIVATE_MESSAGE_GUILD_NAME};
        use crate::history::async_store::AsyncHistoryStore;
        use crate::history::store::{AppendTurn, HistoryStore};
        use crate::identity::Author;
        use crate::subscriptions::{SubscriptionKind, SubscriptionRegistry, SubscriptionView};
        use std::sync::{Arc, Mutex};

        fn store() -> Arc<AsyncHistoryStore> {
            Arc::new(AsyncHistoryStore::new(
                HistoryStore::open(":memory:").expect("open :memory:"),
            ))
        }

        /// The DM peer (user 123) as history recorded it.
        fn peer_author() -> Author {
            Author::new(
                "discord",
                "123",
                Some("cor".to_owned()),
                Some("Cor".to_owned()),
            )
        }

        fn registry() -> (tempfile::TempDir, std::path::PathBuf, SubscriptionRegistry) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("subscriptions.toml");
            let reg = SubscriptionRegistry::new(&path).unwrap();
            (dir, path, reg)
        }

        // --- prune ---------------------------------------------------------

        #[test]
        fn prune_removes_dm_row_whose_peer_left_allowlist() {
            let (_dir, path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(999))
                .unwrap();
            prune_deallowlisted_dm_subscriptions(&mut reg, &[]).unwrap();
            assert!(reg.get(555, SubscriptionKind::Text).is_none());
            assert!(
                !std::fs::read_to_string(&path)
                    .unwrap()
                    .contains("channel_id = 555")
            );
        }

        #[test]
        fn prune_keeps_dm_row_whose_peer_is_allowlisted() {
            let (_dir, path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(999))
                .unwrap();
            prune_deallowlisted_dm_subscriptions(&mut reg, &[999]).unwrap();
            let sub = reg.get(555, SubscriptionKind::Text).unwrap();
            assert_eq!(sub.dm_user_id, Some(999));
            assert!(
                std::fs::read_to_string(&path)
                    .unwrap()
                    .contains("dm_user_id = 999")
            );
        }

        #[test]
        fn prune_leaves_guild_rows_untouched() {
            let (_dir, _path, mut reg) = registry();
            reg.add(42, SubscriptionKind::Text, Some(1), None).unwrap();
            reg.add(43, SubscriptionKind::Voice, Some(1), None).unwrap();
            prune_deallowlisted_dm_subscriptions(&mut reg, &[]).unwrap();
            assert!(reg.get(42, SubscriptionKind::Text).is_some());
            assert!(reg.get(43, SubscriptionKind::Voice).is_some());
        }

        // --- rehydrate -----------------------------------------------------

        fn focus_manager(
            reg: SubscriptionRegistry,
            store: &Arc<AsyncHistoryStore>,
        ) -> (Arc<dyn SubscriptionView>, Arc<FocusManager>) {
            let view: Arc<dyn SubscriptionView> = Arc::new(reg);
            let fm = Arc::new(FocusManager::new(
                "fam",
                Arc::clone(store) as Arc<dyn FocusStore>,
                Arc::clone(&view),
            ));
            (view, fm)
        }

        #[tokio::test]
        async fn rehydrate_dm_row_with_history_restores_guild_and_peer_name() {
            let (_dir, _path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(123))
                .unwrap();
            let store = store();
            store
                .append_turn(AppendTurn::new("fam", 555, "user", "hi").author(peer_author()))
                .await
                .unwrap();
            let (view, fm) = focus_manager(reg, &store);
            rehydrate_dm_naming(fm.as_ref(), view.as_ref(), store.as_ref(), "fam")
                .await
                .unwrap();
            assert_eq!(
                fm.guild_name_for(Some(555)).as_deref(),
                Some(PRIVATE_MESSAGE_GUILD_NAME)
            );
            assert_eq!(
                fm.channel_names().get(&555).map(String::as_str),
                Some("Cor")
            );
        }

        #[tokio::test]
        async fn rehydrate_dm_row_without_history_sets_guild_only() {
            let (_dir, _path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(123))
                .unwrap();
            let store = store();
            let (view, fm) = focus_manager(reg, &store);
            rehydrate_dm_naming(fm.as_ref(), view.as_ref(), store.as_ref(), "fam")
                .await
                .unwrap();
            assert_eq!(
                fm.guild_name_for(Some(555)).as_deref(),
                Some(PRIVATE_MESSAGE_GUILD_NAME)
            );
            assert!(!fm.channel_names().contains_key(&555));
        }

        #[tokio::test]
        async fn rehydrate_leaves_guild_row_naming_untouched() {
            let (_dir, _path, mut reg) = registry();
            reg.add(42, SubscriptionKind::Text, Some(1), None).unwrap();
            let store = store();
            let (view, fm) = focus_manager(reg, &store);
            rehydrate_dm_naming(fm.as_ref(), view.as_ref(), store.as_ref(), "fam")
                .await
                .unwrap();
            assert!(fm.guild_names().is_empty());
            assert!(fm.channel_names().is_empty());
        }

        #[tokio::test]
        async fn rehydrated_maps_feed_unread_digest() {
            let (_dir, _path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(123))
                .unwrap();
            let store = store();
            store
                .append_turn(AppendTurn::new("fam", 555, "user", "hi").author(peer_author()))
                .await
                .unwrap();
            let (view, fm) = focus_manager(reg, &store);
            rehydrate_dm_naming(fm.as_ref(), view.as_ref(), store.as_ref(), "fam")
                .await
                .unwrap();
            let out = FinalReminder::new("text")
                .unread_digest(vec![(555, (1, 0))])
                .channel_names(fm.channel_names())
                .guild_names(fm.guild_names())
                .render();
            assert!(out.contains("DM from Cor (id 555)"), "{out}");
        }

        // --- boot_dm_focus ordering invariants ----------------------------

        /// Build a focus manager sharing one mutable registry with the caller,
        /// so `boot_dm_focus`'s prune/seed observe the same rows the manager does.
        fn shared_boot(
            reg: SubscriptionRegistry,
            store: &Arc<AsyncHistoryStore>,
        ) -> (Arc<Mutex<SubscriptionRegistry>>, Arc<FocusManager>) {
            let subs = Arc::new(Mutex::new(reg));
            let view: Arc<dyn SubscriptionView> = subs.clone();
            let fm = Arc::new(FocusManager::new(
                "fam",
                Arc::clone(store) as Arc<dyn FocusStore>,
                view,
            ));
            (subs, fm)
        }

        #[tokio::test]
        async fn boot_prunes_deallowlisted_dm_before_seeding_focus() {
            // The DM row has a LOWER channel id than the legit guild row, so
            // without prune-before-seed it would win the seeded text focus.
            let (_dir, path, mut reg) = registry();
            reg.add(10, SubscriptionKind::Text, None, Some(999))
                .unwrap(); // peer 999 NOT allowlisted
            reg.add(42, SubscriptionKind::Text, Some(1), None).unwrap(); // legit guild row
            let store = store();
            let (subs, fm) = shared_boot(reg, &store);

            boot_dm_focus(fm.as_ref(), subs.as_ref(), store.as_ref(), &[], "fam")
                .await
                .unwrap();

            // The de-allowlisted DM neither survives nor wins the focus seed.
            assert_eq!(fm.get_focus("text"), Some(42));
            assert!(
                subs.lock()
                    .unwrap()
                    .get(10, SubscriptionKind::Text)
                    .is_none()
            );
            assert!(
                !std::fs::read_to_string(&path)
                    .unwrap()
                    .contains("channel_id = 10")
            );
        }

        #[tokio::test]
        async fn boot_populates_dm_naming_and_seeds_the_dm_focus() {
            let (_dir, _path, mut reg) = registry();
            reg.add(555, SubscriptionKind::Text, None, Some(123))
                .unwrap();
            let store = store();
            store
                .append_turn(AppendTurn::new("fam", 555, "user", "hi").author(peer_author()))
                .await
                .unwrap();
            let (subs, fm) = shared_boot(reg, &store);

            boot_dm_focus(fm.as_ref(), subs.as_ref(), store.as_ref(), &[123], "fam")
                .await
                .unwrap();

            // Naming rehydrated (guild sentinel + peer name) before the seed loop
            // seeded the surviving DM as the text focus.
            assert_eq!(
                fm.guild_names().get(&555).map(String::as_str),
                Some(PRIVATE_MESSAGE_GUILD_NAME)
            );
            assert_eq!(
                fm.channel_names().get(&555).map(String::as_str),
                Some("Cor")
            );
            assert_eq!(fm.get_focus("text"), Some(555));
        }
    }
}
