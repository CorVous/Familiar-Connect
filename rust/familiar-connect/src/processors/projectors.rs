//! Memory-projector registry (subsystem 06; Python `processors/projectors.py`).
//!
//! Lifts the watermark-driven writers (subsystem 07) behind the
//! [`MemoryProjector`] seam so operators can swap or extend strategy via
//! `[providers.memory].projectors`. Per DESIGN §5 (D14) the Python import-time
//! global registration becomes an **explicit** [`ProjectorRegistry`] builder
//! rather than module-level mutable state.
//!
//! Port status: the six built-in factories map their `[providers.memory.<name>]`
//! knob structs onto the subsystem-07 worker constructors and are all wired in
//! [`ProjectorRegistry::with_builtins`] (`rolling_summary`→`SummaryWorker`,
//! `rich_note`→`FactExtractor`, `people_dossier`→`PeopleDossierWorker`,
//! `reflection`→`ReflectionWorker`, `fact_supersede`→`FactSupersedeWorker`,
//! `fact_embedding`→`FactEmbeddingWorker`). Every built-in reads the
//! `"background"` LLM slot; `fact_embedding` is registered but NOT in
//! [`DEFAULT_PROJECTORS`] (opt-in), and its factory errors when
//! `context.embedder` is `None` (byte-compatible message with Python).
//!
//! The subsystem-07 workers expose `run(&self, CancellationToken)` (a
//! cooperative-cancellation forever loop); the [`MemoryProjector`] contract
//! stops a projector by dropping/aborting its spawned task instead (mirroring
//! Python's `tg.create_task(proj.run())` + task cancellation). A small
//! [`WorkerProjector`] adapter bridges the two — see [`worker_projector`].

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

use crate::config::MemoryProvidersConfig;
use crate::embedding::protocol::Embedder;
use crate::history::async_store::AsyncHistoryStore;
use crate::llm::LlmClient;
use crate::processors::fact_embedding_worker::FactEmbeddingWorker;
use crate::processors::fact_extractor::FactExtractor;
use crate::processors::fact_supersede_worker::FactSupersedeWorker;
use crate::processors::people_dossier_worker::PeopleDossierWorker;
use crate::processors::reflection_worker::ReflectionWorker;
use crate::processors::summary_worker::SummaryWorker;

/// Names enabled when `[providers.memory].projectors` is unset.
///
/// `fact_embedding` is registered but **not** in this tuple — it stays opt-in so
/// default deployments need no embedder configured.
pub const DEFAULT_PROJECTORS: [&str; 5] = [
    "rolling_summary",
    "rich_note",
    "people_dossier",
    "reflection",
    "fact_supersede",
];

/// A backend writer projecting `turns` into a side-index.
///
/// Implementations run as a long-lived task; [`run`](MemoryProjector::run) is
/// the forever loop (cancellation stops the task).
#[async_trait]
pub trait MemoryProjector: Send + Sync {
    /// Label for logs / task names.
    fn name(&self) -> &str;
    /// The forever loop.
    async fn run(self: Box<Self>);
}

/// Inputs available to projector factories (built once at wiring, reused across
/// every factory).
#[derive(Clone)]
pub struct ProjectorContext {
    /// The shared history store.
    pub store: Arc<AsyncHistoryStore>,
    /// LLM clients by slot (`"background"` / `"fast"` / `"prose"`).
    pub llm_clients: BTreeMap<String, Arc<dyn LlmClient>>,
    /// The familiar this context serves.
    pub familiar_id: String,
    /// Text → vector seam for the `fact_embedding` projector; `None` when the
    /// embedding backend is `"off"`.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// `[providers.memory]` knob tables.
    pub memory: MemoryProvidersConfig,
    /// Human-readable name for the reserved self-subject.
    pub familiar_display_name: Option<String>,
    /// Config-sourced dream-framing clause for the fact extractor.
    pub dream_extraction_clause: String,
}

impl ProjectorContext {
    /// A context with just the required inputs; everything else defaults.
    #[must_use]
    pub fn new(store: Arc<AsyncHistoryStore>, familiar_id: impl Into<String>) -> Self {
        Self {
            store,
            llm_clients: BTreeMap::new(),
            familiar_id: familiar_id.into(),
            embedder: None,
            memory: MemoryProvidersConfig::default(),
            familiar_display_name: None,
            dream_extraction_clause: String::new(),
        }
    }

    /// The shared `"background"` LLM slot every built-in projector uses.
    ///
    /// Python indexes `ctx.llm_clients["background"]` (a `KeyError` propagates
    /// out of the factory when the slot is absent); the Rust factory surfaces
    /// the same failure as a [`ProjectorError::Config`] instead of panicking.
    fn background_llm(&self) -> Result<Arc<dyn LlmClient>, ProjectorError> {
        self.llm_clients.get("background").cloned().ok_or_else(|| {
            ProjectorError::Config(
                "memory projector requires the \"background\" LLM slot".to_owned(),
            )
        })
    }
}

/// A projector factory: `&ProjectorContext -> MemoryProjector` (or an error, as
/// `fact_embedding` raises when no embedder is configured — mirroring the
/// fallible Python built-in factories).
pub type ProjectorFactory = Arc<
    dyn Fn(&ProjectorContext) -> Result<Box<dyn MemoryProjector>, ProjectorError> + Send + Sync,
>;

/// Errors a projector factory / the registry can surface.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProjectorError {
    /// A requested projector name is not registered.
    #[error("unknown memory projector '{name}'; valid: {valid}")]
    Unknown {
        /// The offending name.
        name: String,
        /// Sorted, `", "`-joined valid names (or `"(none)"`).
        valid: String,
    },
    /// A factory refused to build (e.g. `fact_embedding` without an embedder,
    /// or a missing LLM slot). The message is byte-compatible with Python.
    #[error("{0}")]
    Config(String),
}

/// An explicit projector registry (replaces the Python import-time global).
#[derive(Clone)]
pub struct ProjectorRegistry {
    factories: BTreeMap<String, ProjectorFactory>,
}

impl ProjectorRegistry {
    /// An empty registry (no built-ins).
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: BTreeMap::new(),
        }
    }

    /// A registry pre-populated with the six shipped built-in projectors.
    ///
    /// Every built-in reads the `"background"` LLM slot and threads its
    /// `[providers.memory.<name>]` knobs (P3); `fact_embedding` additionally
    /// requires `context.embedder`.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register("rolling_summary", Arc::new(summary_factory));
        registry.register("rich_note", Arc::new(rich_note_factory));
        registry.register("people_dossier", Arc::new(people_dossier_factory));
        registry.register("reflection", Arc::new(reflection_factory));
        registry.register("fact_supersede", Arc::new(fact_supersede_factory));
        registry.register("fact_embedding", Arc::new(fact_embedding_factory));
        registry
    }

    /// Register `factory` under `name`. Re-registration overwrites.
    pub fn register(&mut self, name: impl Into<String>, factory: ProjectorFactory) {
        self.factories.insert(name.into(), factory);
    }

    /// Names registered today (built-ins + third-party additions), sorted.
    #[must_use]
    pub fn known(&self) -> BTreeSet<String> {
        self.factories.keys().cloned().collect()
    }

    /// Instantiate the selected projectors in `names` order.
    ///
    /// # Errors
    /// [`ProjectorError::Unknown`] when any name is not registered (the valid
    /// list is sorted, matching the Python error text), plus whatever the
    /// selected factory returns ([`ProjectorError::Config`]).
    pub fn create(
        &self,
        names: &[String],
        context: &ProjectorContext,
    ) -> Result<Vec<Box<dyn MemoryProjector>>, ProjectorError> {
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let Some(factory) = self.factories.get(name) else {
                return Err(ProjectorError::Unknown {
                    name: name.clone(),
                    valid: self.valid_names(),
                });
            };
            out.push(factory(context)?);
        }
        Ok(out)
    }

    /// The sorted, comma-joined valid-name list (or `"(none)"` when empty).
    fn valid_names(&self) -> String {
        if self.factories.is_empty() {
            "(none)".to_owned()
        } else {
            self.factories
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

impl Default for ProjectorRegistry {
    /// The default registry carries the built-ins (mirrors `EmbedderRegistry`).
    fn default() -> Self {
        Self::with_builtins()
    }
}

/// Registered built-in names, sorted (convenience over a fresh
/// [`ProjectorRegistry::with_builtins`]).
///
/// Mirrors the Python module-level `known_projectors()`; config parsing (02)
/// injects this set to validate `[providers.memory].projectors`.
#[must_use]
pub fn known_projectors() -> BTreeSet<String> {
    ProjectorRegistry::with_builtins().known()
}

/// Instantiate the selected projectors from the built-ins, in `names` order.
///
/// Mirrors the Python module-level `create_projectors(names=..., context=...)`;
/// the wiring uses a shared [`ProjectorRegistry`] when third-party projectors
/// need registering.
///
/// # Errors
/// See [`ProjectorRegistry::create`].
pub fn create_projectors(
    names: &[String],
    context: &ProjectorContext,
) -> Result<Vec<Box<dyn MemoryProjector>>, ProjectorError> {
    ProjectorRegistry::with_builtins().create(names, context)
}

// ---------------------------------------------------------------------------
// Worker → MemoryProjector adapter
// ---------------------------------------------------------------------------

type RunFuture = BoxFuture<'static, ()>;

/// Adapts a subsystem-07 background worker to the [`MemoryProjector`] seam.
///
/// The worker's forever loop is `run(&self, CancellationToken)`; the
/// [`MemoryProjector`] contract has no token and is stopped by
/// dropping/aborting its spawned task (Python's `create_task` + cancellation).
/// The adapter therefore runs the worker with a fresh token that is never
/// signalled — `JoinHandle::abort()` at the wiring drops the future at its next
/// await point, exactly as asyncio's `CancelledError` unwinds the Python loop.
struct WorkerProjector {
    name: &'static str,
    run: Box<dyn FnOnce(CancellationToken) -> RunFuture + Send + Sync>,
}

#[async_trait]
impl MemoryProjector for WorkerProjector {
    fn name(&self) -> &str {
        self.name
    }
    async fn run(self: Box<Self>) {
        let this = *self;
        (this.run)(CancellationToken::new()).await;
    }
}

/// Box a worker (its display `name` + a `run(CancellationToken)` closure) as a
/// [`MemoryProjector`].
fn worker_projector<F, Fut>(name: &'static str, run: F) -> Box<dyn MemoryProjector>
where
    F: FnOnce(CancellationToken) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    Box::new(WorkerProjector {
        name,
        run: Box::new(move |token| Box::pin(run(token))),
    })
}

// ---------------------------------------------------------------------------
// Built-in factories (thread `[providers.memory.<name>]` knobs — P3)
// ---------------------------------------------------------------------------

fn summary_factory(ctx: &ProjectorContext) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let knobs = &ctx.memory.rolling_summary;
    let worker = SummaryWorker::new(
        Arc::clone(&ctx.store),
        ctx.background_llm()?,
        ctx.familiar_id.clone(),
    )
    .turns_threshold(knobs.turns_threshold)
    .tick_interval_s(knobs.tick_interval_s);
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

fn rich_note_factory(ctx: &ProjectorContext) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let knobs = &ctx.memory.rich_note;
    let mut worker = FactExtractor::new(
        Arc::clone(&ctx.store),
        ctx.background_llm()?,
        ctx.familiar_id.clone(),
    )
    .batch_size(knobs.batch_size)
    .tick_interval_s(knobs.tick_interval_s)
    .participants_max(knobs.participants_max)
    .dream_extraction_clause(ctx.dream_extraction_clause.clone());
    if let Some(display) = &ctx.familiar_display_name {
        worker = worker.familiar_display_name(display.clone());
    }
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

fn people_dossier_factory(
    ctx: &ProjectorContext,
) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let mut worker = PeopleDossierWorker::new(
        Arc::clone(&ctx.store),
        ctx.background_llm()?,
        ctx.familiar_id.clone(),
    )
    .tick_interval_s(ctx.memory.people_dossier.tick_interval_s);
    if let Some(display) = &ctx.familiar_display_name {
        worker = worker.familiar_display_name(display.clone());
    }
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

fn reflection_factory(ctx: &ProjectorContext) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let knobs = &ctx.memory.reflection;
    let worker = ReflectionWorker::new(
        Arc::clone(&ctx.store),
        ctx.background_llm()?,
        ctx.familiar_id.clone(),
    )
    .turns_threshold(knobs.turns_threshold)
    .max_reflections_per_tick(knobs.max_reflections_per_tick)
    .max_turns_per_tick(knobs.max_turns_per_tick)
    .recent_facts_limit(knobs.recent_facts_limit)
    .tick_interval_s(knobs.tick_interval_s);
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

fn fact_supersede_factory(
    ctx: &ProjectorContext,
) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let knobs = &ctx.memory.fact_supersede;
    let worker = FactSupersedeWorker::new(
        Arc::clone(&ctx.store),
        ctx.background_llm()?,
        ctx.familiar_id.clone(),
    )
    .batch_size(knobs.batch_size)
    .tick_interval_s(knobs.tick_interval_s)
    .priors_max(knobs.priors_max);
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

fn fact_embedding_factory(
    ctx: &ProjectorContext,
) -> Result<Box<dyn MemoryProjector>, ProjectorError> {
    let Some(embedder) = ctx.embedder.clone() else {
        return Err(ProjectorError::Config(
            "fact_embedding projector requires a configured embedder. set \
             [providers.embedding].backend to a backend other than \"off\" \
             (e.g. \"hash\") and restart."
                .to_owned(),
        ));
    };
    let worker =
        FactEmbeddingWorker::new(Arc::clone(&ctx.store), embedder, ctx.familiar_id.clone());
    let name = worker.name();
    Ok(worker_projector(name, move |token| async move {
        worker.run(token).await;
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PROJECTORS, MemoryProjector, ProjectorContext, ProjectorError, ProjectorFactory,
        ProjectorRegistry, create_projectors, known_projectors,
    };
    use crate::embedding::hash::HashEmbedder;
    use crate::embedding::protocol::Embedder;
    use crate::history::async_store::AsyncHistoryStore;
    use crate::history::store::HistoryStore;
    use crate::llm::{LlmClient, LlmDelta, Message};
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use serde_json::Value;
    use std::sync::Arc;

    struct CustomProjector {
        name: String,
    }

    #[async_trait]
    impl MemoryProjector for CustomProjector {
        fn name(&self) -> &str {
            &self.name
        }
        async fn run(self: Box<Self>) {}
    }

    /// A no-op LLM stub — the projector wiring tests construct workers but never
    /// drive a tick, so the client methods are never called.
    struct NoopLlm;

    #[async_trait]
    impl LlmClient for NoopLlm {
        async fn chat(&self, _messages: Vec<Message>) -> anyhow::Result<Message> {
            Ok(Message::new("assistant", ""))
        }
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: Option<Vec<Value>>,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<LlmDelta>>> {
            Ok(Box::pin(stream::empty()))
        }
        fn slot(&self) -> Option<&str> {
            Some("background")
        }
        fn multimodal(&self) -> bool {
            false
        }
        fn tool_calling_enabled(&self) -> bool {
            false
        }
    }

    fn store() -> Arc<AsyncHistoryStore> {
        Arc::new(AsyncHistoryStore::new(
            HistoryStore::open(":memory:").unwrap(),
        ))
    }

    /// A bare context (no LLM slots, no embedder) — for registry-shape tests.
    fn ctx() -> ProjectorContext {
        ProjectorContext::new(store(), "fam")
    }

    /// A context carrying the `"background"` LLM slot the built-ins need.
    fn ctx_with_llm() -> ProjectorContext {
        let mut ctx = ProjectorContext::new(store(), "fam");
        ctx.llm_clients.insert(
            "background".to_owned(),
            Arc::new(NoopLlm) as Arc<dyn LlmClient>,
        );
        ctx
    }

    fn custom_factory(label: &'static str) -> ProjectorFactory {
        Arc::new(move |_ctx| {
            Ok(Box::new(CustomProjector {
                name: label.to_owned(),
            }) as Box<dyn MemoryProjector>)
        })
    }

    #[test]
    fn default_projectors_lists_five_builtins() {
        let set: std::collections::BTreeSet<&str> = DEFAULT_PROJECTORS.iter().copied().collect();
        assert_eq!(
            set,
            [
                "fact_supersede",
                "people_dossier",
                "reflection",
                "rich_note",
                "rolling_summary"
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn known_projectors_includes_all_builtins() {
        let known = known_projectors();
        for name in [
            "rolling_summary",
            "rich_note",
            "people_dossier",
            "reflection",
            "fact_supersede",
        ] {
            assert!(known.contains(name), "missing built-in {name}");
        }
    }

    #[test]
    fn empty_names_yields_empty_list() {
        let reg = ProjectorRegistry::with_builtins();
        let out = reg.create(&[], &ctx()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn unknown_projector_raises() {
        let reg = ProjectorRegistry::with_builtins();
        let err = reg
            .create(&["nonexistent".to_owned()], &ctx())
            .err()
            .expect("expected an Unknown error");
        assert!(matches!(err, ProjectorError::Unknown { .. }));
        assert!(err.to_string().contains("unknown memory projector"));
    }

    #[test]
    fn create_projectors_returns_workers_in_order() {
        // Names preserved; each name() uniquely identifies the worker type that
        // the factory constructed (the isinstance() proxy).
        let out = create_projectors(
            &["rich_note".to_owned(), "rolling_summary".to_owned()],
            &ctx_with_llm(),
        )
        .unwrap();
        let names: Vec<&str> = out.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["fact-extractor", "summary-worker"]);
    }

    #[test]
    fn create_default_yields_all_builtins() {
        let names: Vec<String> = DEFAULT_PROJECTORS.iter().map(|s| (*s).to_owned()).collect();
        let out = create_projectors(&names, &ctx_with_llm()).unwrap();
        let got: std::collections::BTreeSet<&str> = out.iter().map(|p| p.name()).collect();
        assert_eq!(
            got,
            [
                "fact-extractor",
                "people-dossier-worker",
                "reflection-worker",
                "summary-worker",
                "fact-supersede-worker",
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn each_builtin_projector_has_a_name() {
        for name in DEFAULT_PROJECTORS {
            let out = create_projectors(&[name.to_owned()], &ctx_with_llm()).unwrap();
            assert_eq!(out.len(), 1);
            assert!(!out[0].name().is_empty());
        }
    }

    #[test]
    fn fact_embedding_known_but_not_default() {
        assert!(known_projectors().contains("fact_embedding"));
        assert!(!DEFAULT_PROJECTORS.contains(&"fact_embedding"));
    }

    #[test]
    fn fact_embedding_factory_requires_embedder() {
        // No embedder configured → the factory refuses with the Python message.
        let err = create_projectors(&["fact_embedding".to_owned()], &ctx_with_llm())
            .err()
            .expect("expected a Config error");
        assert!(
            err.to_string()
                .contains("fact_embedding projector requires"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn fact_embedding_factory_yields_worker_when_wired() {
        let mut ctx = ctx_with_llm();
        ctx.embedder = Some(Arc::new(HashEmbedder::new(64).unwrap()) as Arc<dyn Embedder>);
        let out = create_projectors(&["fact_embedding".to_owned()], &ctx).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name(), "fact-embedding-worker");
    }

    #[test]
    fn missing_background_slot_reports_config_error() {
        // A built-in that needs the "background" slot errors when it is absent
        // (Python raises KeyError; Rust surfaces a Config error).
        let err = create_projectors(&["rolling_summary".to_owned()], &ctx())
            .err()
            .expect("expected a Config error");
        assert!(matches!(err, ProjectorError::Config(_)));
        assert!(err.to_string().contains("background"));
    }

    #[test]
    fn register_projector_adds_to_registry() {
        let mut reg = ProjectorRegistry::new();
        reg.register("custom_test_projector", custom_factory("custom"));
        assert!(reg.known().contains("custom_test_projector"));
        let out = reg
            .create(&["custom_test_projector".to_owned()], &ctx())
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name(), "custom");
    }

    #[test]
    fn register_last_write_wins() {
        let mut reg = ProjectorRegistry::new();
        reg.register("dup", custom_factory("first"));
        reg.register("dup", custom_factory("second"));
        let out = reg.create(&["dup".to_owned()], &ctx()).unwrap();
        assert_eq!(out[0].name(), "second");
    }

    #[test]
    fn create_preserves_names_order() {
        let mut reg = ProjectorRegistry::new();
        reg.register("a", custom_factory("alpha"));
        reg.register("b", custom_factory("beta"));
        let out = reg
            .create(&["b".to_owned(), "a".to_owned()], &ctx())
            .unwrap();
        let names: Vec<&str> = out.iter().map(|p| p.name()).collect();
        assert_eq!(names, ["beta", "alpha"]);
    }

    #[test]
    fn unknown_error_lists_valid_names_sorted() {
        let mut reg = ProjectorRegistry::new();
        reg.register("zebra", custom_factory("z"));
        reg.register("apple", custom_factory("a"));
        let err = reg
            .create(&["nope".to_owned()], &ctx())
            .err()
            .expect("expected an Unknown error");
        assert_eq!(
            err.to_string(),
            "unknown memory projector 'nope'; valid: apple, zebra"
        );
    }

    #[test]
    fn empty_registry_reports_none_valid() {
        let err = ProjectorRegistry::new()
            .create(&["x".to_owned()], &ctx())
            .err()
            .expect("expected an Unknown error");
        assert!(err.to_string().contains("(none)"));
    }
}
