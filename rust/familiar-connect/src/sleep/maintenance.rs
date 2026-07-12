//! Maintenance-pass registry — discrete DB-maintenance actions + order
//! (subsystem 04; Python `sleep/maintenance.py`).
//!
//! A *maintenance pass* is one discrete, one-shot consolidation over the
//! familiar's database (run once per sleep). Today's two: `consolidation` (fact
//! retire/rewrite) and `opinion` (opinion-formation). The registry (DESIGN D14 —
//! an explicit [`PassRegistry::with_builtins`] builder, not an import-time global
//! dict) sequences them; the ONE thing it has that the projector registry lacks
//! is an ordered inter-pass data-flow: consolidation's retirements feed the
//! opinion pass's known-bits deny-list, threaded through a shared
//! [`MaintenanceRun`].
//!
//! The free functions [`execute_consolidation`] / [`execute_opinion_formation`]
//! are the plan→apply orchestrators each pass wraps; ad-hoc callers may use them
//! directly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::SleepError;
use super::apply::apply_consolidation;
use super::consolidation::{
    ConsolidationPlan, DEFAULT_FACTS_MAX, DEFAULT_RETIRE_CAP, DEFAULT_TURNS_MAX, RejectedAction,
    plan_consolidation,
};
use super::opinion_formation::{
    DEFAULT_OPINION_CAP, OpinionPlan, RejectedOpinion, apply_opinions, plan_opinions,
};
use crate::history::async_store::AsyncHistoryStore;
use crate::identity::ego_canonical_key;
use crate::llm::LlmClient;
use crate::log_style as ls;

/// The consolidation pass name (also the `create_passes` selection key).
pub const CONSOLIDATION_PASS: &str = "consolidation";
/// The opinion-formation pass name.
pub const OPINION_PASS: &str = "opinion";
/// Passes run when none explicitly selected — consolidation then opinions
/// (the order is a contract).
pub const DEFAULT_PASSES: [&str; 2] = [CONSOLIDATION_PASS, OPINION_PASS];

// ---------------------------------------------------------------------------
// rejection logging (the surviving audit trail)
// ---------------------------------------------------------------------------

/// Shared shape of a rail-blocked proposal across passes (Python `_Rejection`).
trait Rejection {
    fn rail(&self) -> &str;
    fn detail(&self) -> &str;
    fn payload(&self) -> &Value;
}

impl Rejection for RejectedAction {
    fn rail(&self) -> &str {
        &self.rail
    }
    fn detail(&self) -> &str {
        &self.detail
    }
    fn payload(&self) -> &Value {
        &self.payload
    }
}

impl Rejection for RejectedOpinion {
    fn rail(&self) -> &str {
        &self.rail
    }
    fn detail(&self) -> &str {
        &self.detail
    }
    fn payload(&self) -> &Value {
        &self.payload
    }
}

/// Emit each rail-blocked proposal as a WARNING naming the rail.
///
/// The sleep audit JSON is gone; this log is the surviving record of what the
/// LLM proposed that the code rails refused (rail + truncated payload). A test
/// greps WARNING records for the rail name.
fn log_rejections<R: Rejection>(pass_name: &str, rejected: &[R]) {
    for r in rejected {
        let msg = format!(
            "{} {pass_name} rejected {} {} {}",
            ls::tag("Sleep", ls::G),
            ls::kv_styled("rail", r.rail(), ls::W, ls::LY),
            ls::kv_styled("detail", &ls::trunc(r.detail(), 80), ls::W, ls::LW),
            ls::kv_styled(
                "payload",
                &ls::trunc(&r.payload().to_string(), 160),
                ls::W,
                ls::LW
            ),
        );
        tracing::warn!(target: "familiar_connect.sleep.maintenance", "{msg}");
    }
}

// ---------------------------------------------------------------------------
// free orchestrators (plan → apply)
// ---------------------------------------------------------------------------

/// Plan (always) → apply (if `apply`). Return the plan.
///
/// `plan_consolidation` is read-only; only `apply_consolidation` mutates, so a
/// dry run (`apply == false`) never writes — not even the watermark.
///
/// # Errors
/// Propagates store + LLM transport faults.
#[allow(clippy::too_many_arguments)]
pub async fn execute_consolidation(
    store: &AsyncHistoryStore,
    llm: &dyn LlmClient,
    familiar_id: &str,
    familiar_display_name: Option<&str>,
    apply: bool,
    facts_max: i64,
    turns_max: i64,
    cap: i64,
    system: &str,
) -> anyhow::Result<ConsolidationPlan> {
    let plan =
        plan_consolidation(store, llm, familiar_id, facts_max, turns_max, cap, system).await?;
    log_rejections("consolidation", &plan.rejected);
    if apply {
        apply_consolidation(store, &plan, familiar_display_name).await?;
    }
    Ok(plan)
}

/// Fact ids judged junk/bits this run — feeds the opinion deny-list. Accepted
/// retire ids + rewrite-source ids, in that order (rejected proposals excluded).
#[must_use]
pub fn consolidation_denylist_ids(plan: &ConsolidationPlan) -> Vec<i64> {
    let mut ids: Vec<i64> = Vec::new();
    for a in &plan.retire {
        ids.extend(a.fact_ids.iter().copied());
    }
    for a in &plan.rewrite {
        ids.extend(a.old_fact_ids.iter().copied());
    }
    ids
}

/// Plan opinions → apply (if `apply`). Return the plan.
///
/// Fetches the prior self-dossier and threads its text into the synthesis
/// prompt. `denylist` (consolidation's retired-fact texts) is fed to the stance
/// prompt as known-bits context.
///
/// # Errors
/// Propagates store + LLM transport faults.
#[allow(clippy::too_many_arguments)]
pub async fn execute_opinion_formation(
    store: &AsyncHistoryStore,
    llm: &dyn LlmClient,
    familiar_id: &str,
    familiar_display_name: Option<&str>,
    display_tz: &str,
    apply: bool,
    denylist: &[String],
    cap: i64,
    stance_system: &str,
    synthesis_system: &str,
) -> anyhow::Result<OpinionPlan> {
    let self_key = ego_canonical_key(familiar_id);
    let prior = store
        .get_people_dossier(familiar_id.to_owned(), self_key)
        .await?;
    let self_name = familiar_display_name.unwrap_or(familiar_id);
    let dossier = prior.as_ref().map(|p| p.dossier_text.as_str());
    let plan = plan_opinions(
        store,
        llm,
        familiar_id,
        display_tz,
        self_name,
        denylist,
        dossier,
        cap,
        stance_system,
        synthesis_system,
    )
    .await?;
    log_rejections("opinion", &plan.rejected);
    if apply {
        apply_opinions(store, &plan, familiar_display_name).await?;
    }
    Ok(plan)
}

// ---------------------------------------------------------------------------
// config-sourced prompt text
// ---------------------------------------------------------------------------

/// Config-sourced static instruction text for the sleep passes.
///
/// Built from `CharacterConfig`'s `[prompt]` fields and carried into every pass
/// via [`MaintenanceContext`]. The single-source prose ships in
/// `_default/character.toml`; the empty default keeps that profile the sole
/// source of truth — no in-code copy. Rails stay code-enforced; this text is
/// phrasing only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SleepPromptText {
    /// Consolidation system message text.
    pub consolidation_system: String,
    /// Stance-stage system message text (`{self_name}` filled at build).
    pub stance_system: String,
    /// Synthesis-stage system message text (`{self_name}` filled at build).
    pub synthesis_system: String,
}

impl SleepPromptText {
    /// Build from `[prompt]` strings, relayed verbatim.
    #[must_use]
    pub fn from_config(
        consolidation_system: impl Into<String>,
        stance_system: impl Into<String>,
        synthesis_system: impl Into<String>,
    ) -> Self {
        Self {
            consolidation_system: consolidation_system.into(),
            stance_system: stance_system.into(),
            synthesis_system: synthesis_system.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// per-sleep context + accumulator
// ---------------------------------------------------------------------------

/// Static inputs every maintenance pass needs, built once per sleep and reused
/// across every factory. Adding an input here is the extension point; callers
/// don't know which pass consumes which field.
#[derive(Clone)]
pub struct MaintenanceContext {
    /// The history store.
    pub store: Arc<AsyncHistoryStore>,
    /// The background-slot LLM client.
    pub llm: Arc<dyn LlmClient>,
    /// Familiar id.
    pub familiar_id: String,
    /// Familiar display name (falls back to `familiar_id` where needed).
    pub display_name: Option<String>,
    /// IANA display timezone (opinion-pass day bucketing).
    pub display_tz: String,
    /// Whether the passes mutate (`false` = dry run).
    pub apply: bool,
    /// Consolidation fact-window cap.
    pub facts_max: i64,
    /// Consolidation turn-window cap.
    pub turns_max: i64,
    /// Consolidation retire/rewrite cap.
    pub retire_cap: i64,
    /// Opinion-formation cap.
    pub opinion_cap: i64,
    /// Config-sourced prompt text.
    pub prompts: SleepPromptText,
}

impl MaintenanceContext {
    /// Build a context with the in-code default caps and empty prompt text.
    #[must_use]
    pub fn new(
        store: Arc<AsyncHistoryStore>,
        llm: Arc<dyn LlmClient>,
        familiar_id: impl Into<String>,
        display_name: Option<String>,
        display_tz: impl Into<String>,
        apply: bool,
    ) -> Self {
        Self {
            store,
            llm,
            familiar_id: familiar_id.into(),
            display_name,
            display_tz: display_tz.into(),
            apply,
            facts_max: DEFAULT_FACTS_MAX,
            turns_max: DEFAULT_TURNS_MAX,
            retire_cap: DEFAULT_RETIRE_CAP,
            opinion_cap: DEFAULT_OPINION_CAP,
            prompts: SleepPromptText::default(),
        }
    }

    /// Builder: attach config-sourced prompt text.
    #[must_use]
    pub fn with_prompts(mut self, prompts: SleepPromptText) -> Self {
        self.prompts = prompts;
        self
    }
}

/// Mutable accumulator threading one pass's result to the next.
///
/// `denylist_fact_ids`: consolidation writes the fact ids it retired this run;
/// the opinion pass reads them (resolving to texts) for its known-bits deny-list.
/// `opinion_plan`: the opinion pass's product, surfaced for the engine-owned
/// prose-gen step.
#[derive(Default)]
pub struct MaintenanceRun {
    /// Fact ids consolidation retired/merged this run.
    pub denylist_fact_ids: Vec<i64>,
    /// The opinion pass's product (for engine-owned prose-gen).
    pub opinion_plan: Option<OpinionPlan>,
}

/// One discrete DB-maintenance action, run once per sleep. `run` reads/writes
/// the shared [`MaintenanceRun`] so an earlier pass's result reaches a later one.
#[async_trait]
pub trait MaintenancePass: Send + Sync {
    /// Stable label for logs / selection.
    fn name(&self) -> &str;
    /// Execute the pass; mutate the shared run.
    ///
    /// # Errors
    /// Propagates store + LLM transport faults — [`run_passes`] does NOT guard
    /// them (the engine owns its own catch-everything guard).
    async fn run(&self, run: &mut MaintenanceRun) -> anyhow::Result<()>;
}

// ---------------------------------------------------------------------------
// registry (DESIGN D14 — explicit builder, sorted names, byte-exact errors)
// ---------------------------------------------------------------------------

/// A pass factory: turns the per-sleep context into a boxed pass.
pub type PassFactory = Arc<dyn Fn(MaintenanceContext) -> Box<dyn MaintenancePass> + Send + Sync>;

/// Name-indexed maintenance-pass registry. Keys stay sorted (`BTreeMap`) so
/// [`known`](PassRegistry::known) and the valid-name list in the unknown-pass
/// error are deterministic.
#[derive(Clone)]
pub struct PassRegistry {
    factories: BTreeMap<String, PassFactory>,
}

impl PassRegistry {
    /// An empty registry (no built-ins).
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: BTreeMap::new(),
        }
    }

    /// A registry with the two built-in passes registered.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(
            CONSOLIDATION_PASS,
            Arc::new(|ctx| Box::new(ConsolidationPass { ctx }) as Box<dyn MaintenancePass>),
        );
        registry.register(
            OPINION_PASS,
            Arc::new(|ctx| Box::new(OpinionFormationPass { ctx }) as Box<dyn MaintenancePass>),
        );
        registry
    }

    /// Register `factory` under `name`; re-registration overwrites silently.
    pub fn register(&mut self, name: impl Into<String>, factory: PassFactory) {
        self.factories.insert(name.into(), factory);
    }

    /// Registered pass names, sorted.
    #[must_use]
    pub fn known(&self) -> BTreeSet<String> {
        self.factories.keys().cloned().collect()
    }

    /// Instantiate selected passes in `names` order.
    ///
    /// # Errors
    /// [`SleepError::UnknownPass`] when any name is not registered (message names
    /// the bad name and the sorted valid list).
    pub fn create(
        &self,
        names: &[&str],
        ctx: &MaintenanceContext,
    ) -> Result<Vec<Box<dyn MaintenancePass>>, SleepError> {
        let mut out: Vec<Box<dyn MaintenancePass>> = Vec::new();
        for &name in names {
            match self.factories.get(name) {
                Some(factory) => out.push(factory(ctx.clone())),
                None => {
                    return Err(SleepError::UnknownPass {
                        name: name.to_owned(),
                        valid: self.valid_names(),
                    });
                }
            }
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

impl Default for PassRegistry {
    /// The default registry carries the built-ins.
    fn default() -> Self {
        Self::with_builtins()
    }
}

/// Names registered by the built-in registry today.
#[must_use]
pub fn known_passes() -> BTreeSet<String> {
    PassRegistry::with_builtins().known()
}

/// Instantiate selected passes in `names` order from the built-in registry.
///
/// # Errors
/// [`SleepError::UnknownPass`] when any name is not registered.
pub fn create_passes(
    names: &[&str],
    context: &MaintenanceContext,
) -> Result<Vec<Box<dyn MaintenancePass>>, SleepError> {
    PassRegistry::with_builtins().create(names, context)
}

/// Run `passes` in order, threading the shared [`MaintenanceRun`].
///
/// Strictly sequential and UNGUARDED — a pass failure aborts the remaining
/// passes and propagates to the engine's own guard.
///
/// # Errors
/// Propagates the first pass failure.
pub async fn run_passes(
    passes: &[Box<dyn MaintenancePass>],
    run: Option<MaintenanceRun>,
) -> anyhow::Result<MaintenanceRun> {
    let mut run = run.unwrap_or_default();
    for p in passes {
        p.run(&mut run).await?;
    }
    Ok(run)
}

// ---------------------------------------------------------------------------
// built-in passes
// ---------------------------------------------------------------------------

/// Fact retire/rewrite consolidation. Stashes retired ids for the opinion pass.
struct ConsolidationPass {
    ctx: MaintenanceContext,
}

#[async_trait]
impl MaintenancePass for ConsolidationPass {
    fn name(&self) -> &str {
        CONSOLIDATION_PASS
    }

    async fn run(&self, run: &mut MaintenanceRun) -> anyhow::Result<()> {
        let ctx = &self.ctx;
        let plan = execute_consolidation(
            ctx.store.as_ref(),
            ctx.llm.as_ref(),
            &ctx.familiar_id,
            ctx.display_name.as_deref(),
            ctx.apply,
            ctx.facts_max,
            ctx.turns_max,
            ctx.retire_cap,
            &ctx.prompts.consolidation_system,
        )
        .await?;
        let msg = format!(
            "{} consolidation {} {}",
            ls::tag("Sleep", ls::G),
            ls::kv_styled("retired", &plan.retire.len().to_string(), ls::W, ls::LW),
            ls::kv_styled("merged", &plan.rewrite.len().to_string(), ls::W, ls::LW),
        );
        tracing::info!(target: "familiar_connect.sleep.maintenance", "{msg}");
        // thread retirements forward → opinion deny-list (ordered data-flow).
        run.denylist_fact_ids
            .extend(consolidation_denylist_ids(&plan));
        Ok(())
    }
}

/// Opinion-formation. Reads consolidation's retired ids as its deny-list.
struct OpinionFormationPass {
    ctx: MaintenanceContext,
}

#[async_trait]
impl MaintenancePass for OpinionFormationPass {
    fn name(&self) -> &str {
        OPINION_PASS
    }

    async fn run(&self, run: &mut MaintenanceRun) -> anyhow::Result<()> {
        let ctx = &self.ctx;
        let denylist: Vec<String> = if run.denylist_fact_ids.is_empty() {
            Vec::new()
        } else {
            let facts = ctx
                .store
                .facts_by_ids(ctx.familiar_id.clone(), run.denylist_fact_ids.clone())
                .await?;
            facts.into_iter().map(|f| f.text).collect()
        };
        let plan = execute_opinion_formation(
            ctx.store.as_ref(),
            ctx.llm.as_ref(),
            &ctx.familiar_id,
            ctx.display_name.as_deref(),
            &ctx.display_tz,
            ctx.apply,
            &denylist,
            ctx.opinion_cap,
            &ctx.prompts.stance_system,
            &ctx.prompts.synthesis_system,
        )
        .await?;
        let msg = format!(
            "{} opinions {}",
            ls::tag("Sleep", ls::G),
            ls::kv_styled("formed", &plan.opinions.len().to_string(), ls::W, ls::LW),
        );
        tracing::info!(target: "familiar_connect.sleep.maintenance", "{msg}");
        run.opinion_plan = Some(plan); // surface for engine-owned prose-gen
        Ok(())
    }
}
