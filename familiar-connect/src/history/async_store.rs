//! Async facade over [`HistoryStore`] (subsystem 03; Python
//! `history/async_store.py`).
//!
//! Python's `AsyncHistoryStore` is a `__getattr__` duck-proxy that dispatches
//! every call onto a 4-worker `ThreadPoolExecutor`, keeping Turso/tantivy work
//! off the event loop. Rust cannot transliterate `__getattr__`, so this is an
//! explicit set of `async fn` wrappers (DESIGN §4.4 / port notes). Each wrapper
//! moves its owned arguments onto a `tokio::task::spawn_blocking` thread and
//! runs the synchronous [`HistoryStore`] method there — DB work already lives on
//! the store's single owning actor thread (see [`super::db`]), so this only
//! ensures the *await* never blocks a reactor worker.
//!
//! ## Ordering / concurrency (spec 03 behaviors 2–3)
//!
//! Every SQL statement still serializes onto the one DB actor thread; multiple
//! concurrent `spawn_blocking` tasks may run tantivy searches in genuine
//! parallel (searches take no lock). Whole multi-statement operations
//! (`supersede`, promotions, `bump_reaction`, `append_fact`'s dedup-scan+insert)
//! run inside explicit transactions on the actor, so — unlike Python's
//! per-statement interleaving — an interleaving of two operations' statements
//! is impossible. This is the safe atomicity strengthening the DESIGN sanctions
//! (D5); no test pins the old non-atomicity.
//!
//! Cancelling an awaiting caller drops the `JoinHandle` but leaves the dispatched
//! blocking job running to completion (standard `spawn_blocking` semantics),
//! preserving at-most-once execution (behavior 6).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};

use super::StoreError;
use super::store::{
    AccountProfile, ActivityRecord, AlarmRow, AppendFact, AppendTurn, ChannelUnread, Fact,
    FocusPointers, HistoryStore, HistoryTurn, NewFact, OtherChannelInfo, PeopleDossierEntry,
    Promotion, Reflection, SleepWatermark, SummaryEntry, SupersedeResult, WatermarkEntry,
};
use crate::identity::Author;

/// Async proxy around a shared [`HistoryStore`]. `close()` stays synchronous.
pub struct AsyncHistoryStore {
    inner: Arc<HistoryStore>,
}

impl AsyncHistoryStore {
    /// Wrap an owned store.
    #[must_use]
    pub fn new(store: HistoryStore) -> Self {
        Self {
            inner: Arc::new(store),
        }
    }

    /// Wrap an already-shared store.
    #[must_use]
    pub const fn from_arc(store: Arc<HistoryStore>) -> Self {
        Self { inner: store }
    }

    /// The raw synchronous store — for callers that must run inline (Python's
    /// `.sync` property; used by invalidation-key paths in subsystem 05).
    #[must_use]
    pub fn sync(&self) -> &HistoryStore {
        &self.inner
    }

    /// A cloned handle to the shared store.
    #[must_use]
    pub fn sync_arc(&self) -> Arc<HistoryStore> {
        Arc::clone(&self.inner)
    }

    /// Shut down the underlying DB actor (synchronous, like Python).
    pub fn close(&self) {
        self.inner.close();
    }

    /// Run a synchronous store operation on a blocking thread.
    async fn run<T, F>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&HistoryStore) -> Result<T, StoreError> + Send + 'static,
        T: Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || f(inner.as_ref()))
            .await
            .expect("history store blocking task panicked")
    }

    // -- turns -----------------------------------------------------------

    /// See [`HistoryStore::append_turn`].
    pub async fn append_turn(&self, p: AppendTurn) -> Result<HistoryTurn, StoreError> {
        self.run(move |s| s.append_turn(p)).await
    }

    /// See [`HistoryStore::stage_turn`].
    pub async fn stage_turn(&self, p: AppendTurn) -> Result<HistoryTurn, StoreError> {
        self.run(move |s| s.stage_turn(p)).await
    }

    /// See [`HistoryStore::lookup_turn_by_platform_message_id`].
    pub async fn lookup_turn_by_platform_message_id(
        &self,
        familiar_id: String,
        platform_message_id: String,
    ) -> Result<Option<HistoryTurn>, StoreError> {
        self.run(move |s| s.lookup_turn_by_platform_message_id(&familiar_id, &platform_message_id))
            .await
    }

    /// See [`HistoryStore::update_turn_content_by_message_id`].
    pub async fn update_turn_content_by_message_id(
        &self,
        familiar_id: String,
        platform_message_id: String,
        content: String,
    ) -> Result<(), StoreError> {
        self.run(move |s| {
            s.update_turn_content_by_message_id(&familiar_id, &platform_message_id, &content)
        })
        .await
    }

    /// See [`HistoryStore::turns_by_ids`].
    pub async fn turns_by_ids(
        &self,
        familiar_id: String,
        ids: Vec<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.turns_by_ids(&familiar_id, &ids)).await
    }

    /// See [`HistoryStore::recent`].
    pub async fn recent(
        &self,
        familiar_id: String,
        channel_id: i64,
        limit: i64,
        mode: Option<String>,
        before_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.recent(&familiar_id, channel_id, limit, mode.as_deref(), before_id))
            .await
    }

    /// See [`HistoryStore::turns_around`].
    pub async fn turns_around(
        &self,
        familiar_id: String,
        channel_id: i64,
        turn_id: i64,
        before: i64,
        after: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.turns_around(&familiar_id, channel_id, turn_id, before, after))
            .await
    }

    /// See [`HistoryStore::recent_distinct_authors`].
    pub async fn recent_distinct_authors(
        &self,
        familiar_id: String,
        channel_id: i64,
        limit: i64,
    ) -> Result<Vec<Author>, StoreError> {
        self.run(move |s| s.recent_distinct_authors(&familiar_id, channel_id, limit))
            .await
    }

    /// See [`HistoryStore::older_than`].
    pub async fn older_than(
        &self,
        familiar_id: String,
        max_id: i64,
        channel_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.older_than(&familiar_id, max_id, channel_id, limit))
            .await
    }

    /// See [`HistoryStore::latest_id`].
    pub async fn latest_id(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
    ) -> Result<Option<i64>, StoreError> {
        self.run(move |s| s.latest_id(&familiar_id, channel_id))
            .await
    }

    /// See [`HistoryStore::count`].
    pub async fn count(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
    ) -> Result<i64, StoreError> {
        self.run(move |s| s.count(&familiar_id, channel_id)).await
    }

    /// See [`HistoryStore::turns_in_id_range`].
    pub async fn turns_in_id_range(
        &self,
        familiar_id: String,
        min_id_exclusive: i64,
        max_id_inclusive: i64,
        channel_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| {
            s.turns_in_id_range(&familiar_id, min_id_exclusive, max_id_inclusive, channel_id)
        })
        .await
    }

    /// See [`HistoryStore::all_channel_ids`].
    pub async fn all_channel_ids(&self, familiar_id: String) -> Result<HashSet<i64>, StoreError> {
        self.run(move |s| s.all_channel_ids(&familiar_id)).await
    }

    /// See [`HistoryStore::distinct_other_channels`].
    pub async fn distinct_other_channels(
        &self,
        familiar_id: String,
        exclude_channel_id: i64,
    ) -> Result<Vec<OtherChannelInfo>, StoreError> {
        self.run(move |s| s.distinct_other_channels(&familiar_id, exclude_channel_id))
            .await
    }

    /// See [`HistoryStore::latest_id_at_or_before`].
    pub async fn latest_id_at_or_before(
        &self,
        familiar_id: String,
        ts: DateTime<Utc>,
    ) -> Result<Option<i64>, StoreError> {
        self.run(move |s| s.latest_id_at_or_before(&familiar_id, ts))
            .await
    }

    // -- mentions --------------------------------------------------------

    /// See [`HistoryStore::record_mentions`].
    pub async fn record_mentions(
        &self,
        turn_id: i64,
        canonical_keys: Vec<String>,
    ) -> Result<(), StoreError> {
        self.run(move |s| {
            let refs: Vec<&str> = canonical_keys.iter().map(String::as_str).collect();
            s.record_mentions(turn_id, &refs)
        })
        .await
    }

    /// See [`HistoryStore::mentions_for_turn`].
    pub async fn mentions_for_turn(&self, turn_id: i64) -> Result<Vec<String>, StoreError> {
        self.run(move |s| s.mentions_for_turn(turn_id)).await
    }

    // -- reactions -------------------------------------------------------

    /// See [`HistoryStore::set_reaction`].
    pub async fn set_reaction(
        &self,
        familiar_id: String,
        platform_message_id: String,
        emoji: String,
        count: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_reaction(&familiar_id, &platform_message_id, &emoji, count))
            .await
    }

    /// See [`HistoryStore::bump_reaction`].
    pub async fn bump_reaction(
        &self,
        familiar_id: String,
        platform_message_id: String,
        emoji: String,
        delta: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.bump_reaction(&familiar_id, &platform_message_id, &emoji, delta))
            .await
    }

    /// See [`HistoryStore::clear_reactions`].
    pub async fn clear_reactions(
        &self,
        familiar_id: String,
        platform_message_id: String,
        emoji: Option<String>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.clear_reactions(&familiar_id, &platform_message_id, emoji.as_deref()))
            .await
    }

    /// See [`HistoryStore::reactions_for_messages`].
    pub async fn reactions_for_messages(
        &self,
        familiar_id: String,
        platform_message_ids: Vec<String>,
    ) -> Result<HashMap<String, Vec<(String, i64)>>, StoreError> {
        self.run(move |s| {
            let refs: Vec<&str> = platform_message_ids.iter().map(String::as_str).collect();
            s.reactions_for_messages(&familiar_id, &refs)
        })
        .await
    }

    // -- summaries -------------------------------------------------------

    /// See [`HistoryStore::get_summary`].
    pub async fn get_summary(
        &self,
        familiar_id: String,
        channel_id: i64,
    ) -> Result<Option<SummaryEntry>, StoreError> {
        self.run(move |s| s.get_summary(&familiar_id, channel_id))
            .await
    }

    /// See [`HistoryStore::put_summary`].
    pub async fn put_summary(
        &self,
        familiar_id: String,
        last_summarised_id: i64,
        summary_text: String,
        channel_id: i64,
        last_consumed_at: Option<String>,
    ) -> Result<(), StoreError> {
        self.run(move |s| {
            s.put_summary(
                &familiar_id,
                last_summarised_id,
                &summary_text,
                channel_id,
                last_consumed_at.as_deref(),
            )
        })
        .await
    }

    // -- watermarks ------------------------------------------------------

    /// See [`HistoryStore::get_writer_watermark`].
    pub async fn get_writer_watermark(
        &self,
        familiar_id: String,
    ) -> Result<Option<WatermarkEntry>, StoreError> {
        self.run(move |s| s.get_writer_watermark(&familiar_id))
            .await
    }

    /// See [`HistoryStore::put_writer_watermark`].
    pub async fn put_writer_watermark(
        &self,
        familiar_id: String,
        last_written_id: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.put_writer_watermark(&familiar_id, last_written_id))
            .await
    }

    /// See [`HistoryStore::get_sleep_watermark`].
    pub async fn get_sleep_watermark(
        &self,
        familiar_id: String,
    ) -> Result<Option<SleepWatermark>, StoreError> {
        self.run(move |s| s.get_sleep_watermark(&familiar_id)).await
    }

    /// See [`HistoryStore::advance_sleep_watermark`].
    pub async fn advance_sleep_watermark(
        &self,
        familiar_id: String,
        last_fact_id: Option<i64>,
        last_turn_id: Option<i64>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.advance_sleep_watermark(&familiar_id, last_fact_id, last_turn_id))
            .await
    }

    /// See [`HistoryStore::turns_since_watermark`].
    pub async fn turns_since_watermark(
        &self,
        familiar_id: String,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.turns_since_watermark(&familiar_id, limit))
            .await
    }

    // -- dossiers / identity --------------------------------------------

    /// See [`HistoryStore::get_people_dossier`].
    pub async fn get_people_dossier(
        &self,
        familiar_id: String,
        canonical_key: String,
    ) -> Result<Option<PeopleDossierEntry>, StoreError> {
        self.run(move |s| s.get_people_dossier(&familiar_id, &canonical_key))
            .await
    }

    /// See [`HistoryStore::put_people_dossier`].
    pub async fn put_people_dossier(
        &self,
        familiar_id: String,
        canonical_key: String,
        last_fact_id: i64,
        dossier_text: String,
    ) -> Result<(), StoreError> {
        self.run(move |s| {
            s.put_people_dossier(&familiar_id, &canonical_key, last_fact_id, &dossier_text)
        })
        .await
    }

    /// See [`HistoryStore::put_people_dossier_if_current`].
    pub async fn put_people_dossier_if_current(
        &self,
        familiar_id: String,
        canonical_key: String,
        expected_prev_last_fact_id: Option<i64>,
        new_last_fact_id: i64,
        dossier_text: String,
    ) -> Result<bool, StoreError> {
        self.run(move |s| {
            s.put_people_dossier_if_current(
                &familiar_id,
                &canonical_key,
                expected_prev_last_fact_id,
                new_last_fact_id,
                &dossier_text,
            )
        })
        .await
    }

    /// See [`HistoryStore::subjects_with_facts`].
    pub async fn subjects_with_facts(
        &self,
        familiar_id: String,
    ) -> Result<HashMap<String, i64>, StoreError> {
        self.run(move |s| s.subjects_with_facts(&familiar_id)).await
    }

    /// See [`HistoryStore::facts_for_subject`].
    pub async fn facts_for_subject(
        &self,
        familiar_id: String,
        canonical_key: String,
        min_id_exclusive: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| {
            s.facts_for_subject(
                &familiar_id,
                &canonical_key,
                min_id_exclusive,
                include_superseded,
                as_of,
            )
        })
        .await
    }

    /// See [`HistoryStore::upsert_account`].
    pub async fn upsert_account(&self, author: Author) -> Result<(), StoreError> {
        self.run(move |s| s.upsert_account(&author)).await
    }

    /// See [`HistoryStore::get_account_profile`].
    pub async fn get_account_profile(
        &self,
        canonical_key: String,
    ) -> Result<Option<AccountProfile>, StoreError> {
        self.run(move |s| s.get_account_profile(&canonical_key))
            .await
    }

    /// See [`HistoryStore::upsert_guild_nick`].
    pub async fn upsert_guild_nick(
        &self,
        canonical_key: String,
        guild_id: i64,
        nick: Option<String>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.upsert_guild_nick(&canonical_key, guild_id, nick.as_deref()))
            .await
    }

    /// See [`HistoryStore::resolve_label`].
    pub async fn resolve_label(
        &self,
        canonical_key: String,
        guild_id: Option<i64>,
        familiar_id: Option<String>,
    ) -> Result<String, StoreError> {
        self.run(move |s| s.resolve_label(&canonical_key, guild_id, familiar_id.as_deref()))
            .await
    }

    /// See [`HistoryStore::latest_author_for`].
    pub async fn latest_author_for(
        &self,
        familiar_id: String,
        canonical_key: String,
    ) -> Result<Option<Author>, StoreError> {
        self.run(move |s| s.latest_author_for(&familiar_id, &canonical_key))
            .await
    }

    // -- facts -----------------------------------------------------------

    /// See [`HistoryStore::append_fact`].
    pub async fn append_fact(&self, p: AppendFact) -> Result<Fact, StoreError> {
        self.run(move |s| s.append_fact(p)).await
    }

    /// See [`HistoryStore::facts_by_ids`].
    pub async fn facts_by_ids(
        &self,
        familiar_id: String,
        ids: Vec<i64>,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| s.facts_by_ids(&familiar_id, &ids)).await
    }

    /// See [`HistoryStore::recent_facts`].
    pub async fn recent_facts(
        &self,
        familiar_id: String,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| s.recent_facts(&familiar_id, limit, include_superseded, as_of))
            .await
    }

    /// See [`HistoryStore::latest_fact_id`].
    pub async fn latest_fact_id(&self, familiar_id: String) -> Result<i64, StoreError> {
        self.run(move |s| s.latest_fact_id(&familiar_id)).await
    }

    /// See [`HistoryStore::all_fact_ids`].
    pub async fn all_fact_ids(&self, familiar_id: String) -> Result<HashSet<i64>, StoreError> {
        self.run(move |s| s.all_fact_ids(&familiar_id)).await
    }

    /// See [`HistoryStore::ancestors_of`].
    pub async fn ancestors_of(
        &self,
        familiar_id: String,
        fact_id: i64,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| s.ancestors_of(&familiar_id, fact_id))
            .await
    }

    /// See [`HistoryStore::superseded_fact_ids`].
    pub async fn superseded_fact_ids(
        &self,
        familiar_id: String,
        fact_ids: Vec<i64>,
    ) -> Result<HashSet<i64>, StoreError> {
        self.run(move |s| s.superseded_fact_ids(&familiar_id, &fact_ids))
            .await
    }

    /// See [`HistoryStore::supersede`].
    pub async fn supersede(
        &self,
        familiar_id: String,
        obsolete_facts: Vec<i64>,
        new_fact: NewFact,
    ) -> Result<SupersedeResult, StoreError> {
        self.run(move |s| s.supersede(&familiar_id, &obsolete_facts, new_fact))
            .await
    }

    // -- fact embeddings -------------------------------------------------

    /// See [`HistoryStore::set_fact_embedding`].
    pub async fn set_fact_embedding(
        &self,
        fact_id: i64,
        model: String,
        vector: Vec<f32>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_fact_embedding(fact_id, &model, &vector))
            .await
    }

    /// See [`HistoryStore::get_fact_embeddings`].
    pub async fn get_fact_embeddings(
        &self,
        fact_ids: Vec<i64>,
        model: String,
    ) -> Result<HashMap<i64, Vec<f32>>, StoreError> {
        self.run(move |s| s.get_fact_embeddings(&fact_ids, &model))
            .await
    }

    /// See [`HistoryStore::unembedded_facts`].
    pub async fn unembedded_facts(
        &self,
        familiar_id: String,
        model: String,
        limit: i64,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| s.unembedded_facts(&familiar_id, &model, limit))
            .await
    }

    /// See [`HistoryStore::latest_embedded_fact_id`].
    pub async fn latest_embedded_fact_id(
        &self,
        familiar_id: String,
        model: String,
    ) -> Result<i64, StoreError> {
        self.run(move |s| s.latest_embedded_fact_id(&familiar_id, &model))
            .await
    }

    // -- reflections -----------------------------------------------------

    /// See [`HistoryStore::append_reflection`].
    #[allow(clippy::too_many_arguments)]
    pub async fn append_reflection(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
        text: String,
        cited_turn_ids: Vec<i64>,
        cited_fact_ids: Vec<i64>,
        last_turn_id: i64,
        last_fact_id: i64,
    ) -> Result<Reflection, StoreError> {
        self.run(move |s| {
            s.append_reflection(
                &familiar_id,
                channel_id,
                &text,
                &cited_turn_ids,
                &cited_fact_ids,
                last_turn_id,
                last_fact_id,
            )
        })
        .await
    }

    /// See [`HistoryStore::recent_reflections`].
    pub async fn recent_reflections(
        &self,
        familiar_id: String,
        channel_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<Reflection>, StoreError> {
        self.run(move |s| s.recent_reflections(&familiar_id, channel_id, limit))
            .await
    }

    /// See [`HistoryStore::latest_reflection_watermarks`].
    pub async fn latest_reflection_watermarks(
        &self,
        familiar_id: String,
    ) -> Result<(i64, i64), StoreError> {
        self.run(move |s| s.latest_reflection_watermarks(&familiar_id))
            .await
    }

    /// See [`HistoryStore::set_reflection_watermark`].
    pub async fn set_reflection_watermark(
        &self,
        familiar_id: String,
        last_turn_id: i64,
        last_fact_id: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_reflection_watermark(&familiar_id, last_turn_id, last_fact_id))
            .await
    }

    // -- alarms ----------------------------------------------------------

    /// See [`HistoryStore::insert_alarm`].
    pub async fn insert_alarm(
        &self,
        familiar_id: String,
        channel_id: i64,
        channel_kind: String,
        scheduled_at: String,
        reason: String,
        originating_turn_id: Option<String>,
    ) -> Result<String, StoreError> {
        self.run(move |s| {
            s.insert_alarm(
                &familiar_id,
                channel_id,
                &channel_kind,
                &scheduled_at,
                &reason,
                originating_turn_id.as_deref(),
            )
        })
        .await
    }

    /// See [`HistoryStore::list_pending_alarms`].
    pub async fn list_pending_alarms(
        &self,
        familiar_id: String,
    ) -> Result<Vec<AlarmRow>, StoreError> {
        self.run(move |s| s.list_pending_alarms(&familiar_id)).await
    }

    /// See [`HistoryStore::mark_alarm_fired`].
    pub async fn mark_alarm_fired(
        &self,
        alarm_id: String,
        fired_at: String,
    ) -> Result<bool, StoreError> {
        self.run(move |s| s.mark_alarm_fired(&alarm_id, &fired_at))
            .await
    }

    /// See [`HistoryStore::cancel_alarm`].
    pub async fn cancel_alarm(
        &self,
        alarm_id: String,
        cancelled_at: String,
    ) -> Result<bool, StoreError> {
        self.run(move |s| s.cancel_alarm(&alarm_id, &cancelled_at))
            .await
    }

    // -- attentional stream ---------------------------------------------

    /// See [`HistoryStore::promote_staged_turns`].
    pub async fn promote_staged_turns(
        &self,
        familiar_id: String,
        channel_id: i64,
        catch_up_limit: Option<usize>,
    ) -> Result<Promotion, StoreError> {
        self.run(move |s| s.promote_staged_turns(&familiar_id, channel_id, catch_up_limit))
            .await
    }

    /// See [`HistoryStore::promote_staged_turns_since`].
    pub async fn promote_staged_turns_since(
        &self,
        familiar_id: String,
        after_turn_id: i64,
        catch_up_limit: Option<usize>,
    ) -> Result<Promotion, StoreError> {
        self.run(move |s| s.promote_staged_turns_since(&familiar_id, after_turn_id, catch_up_limit))
            .await
    }

    /// See [`HistoryStore::count_staged`].
    pub async fn count_staged(
        &self,
        familiar_id: String,
        channel_id: i64,
    ) -> Result<i64, StoreError> {
        self.run(move |s| s.count_staged(&familiar_id, channel_id))
            .await
    }

    /// See [`HistoryStore::staged_channels`].
    pub async fn staged_channels(
        &self,
        familiar_id: String,
    ) -> Result<HashMap<i64, ChannelUnread>, StoreError> {
        self.run(move |s| s.staged_channels(&familiar_id)).await
    }

    /// See [`HistoryStore::recent_cross_channel`].
    pub async fn recent_cross_channel(
        &self,
        familiar_id: String,
        limit: i64,
        respect_archive: bool,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.recent_cross_channel(&familiar_id, limit, respect_archive))
            .await
    }

    /// See [`HistoryStore::consumed_turns_after`].
    pub async fn consumed_turns_after(
        &self,
        familiar_id: String,
        after_consumed_at: String,
        after_id: i64,
        limit: i64,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.consumed_turns_after(&familiar_id, &after_consumed_at, after_id, limit))
            .await
    }

    /// See [`HistoryStore::get_focus_pointers`].
    pub async fn get_focus_pointers(
        &self,
        familiar_id: String,
    ) -> Result<Option<FocusPointers>, StoreError> {
        self.run(move |s| s.get_focus_pointers(&familiar_id)).await
    }

    /// See [`HistoryStore::set_focus_pointers`].
    pub async fn set_focus_pointers(
        &self,
        familiar_id: String,
        text_channel_id: Option<i64>,
        voice_channel_id: Option<i64>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_focus_pointers(&familiar_id, text_channel_id, voice_channel_id))
            .await
    }

    /// See [`HistoryStore::get_digest_watermark`].
    pub async fn get_digest_watermark(
        &self,
        familiar_id: String,
    ) -> Result<Option<DateTime<Utc>>, StoreError> {
        self.run(move |s| s.get_digest_watermark(&familiar_id))
            .await
    }

    /// See [`HistoryStore::set_digest_watermark`].
    pub async fn set_digest_watermark(
        &self,
        familiar_id: String,
        watermark_at: DateTime<Utc>,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_digest_watermark(&familiar_id, watermark_at))
            .await
    }

    // -- archive watermark ----------------------------------------------

    /// See [`HistoryStore::set_archive_watermark`].
    pub async fn set_archive_watermark(
        &self,
        familiar_id: String,
        channel_id: i64,
        turn_id: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_archive_watermark(&familiar_id, channel_id, turn_id))
            .await
    }

    /// See [`HistoryStore::set_archive_watermark_all`].
    pub async fn set_archive_watermark_all(
        &self,
        familiar_id: String,
        turn_id: i64,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_archive_watermark_all(&familiar_id, turn_id))
            .await
    }

    /// See [`HistoryStore::get_archive_watermark`].
    pub async fn get_archive_watermark(
        &self,
        familiar_id: String,
        channel_id: i64,
    ) -> Result<Option<i64>, StoreError> {
        self.run(move |s| s.get_archive_watermark(&familiar_id, channel_id))
            .await
    }

    // -- activities ------------------------------------------------------

    /// See [`HistoryStore::create_activity`].
    pub async fn create_activity(
        &self,
        familiar_id: String,
        type_id: String,
        label: String,
        started_at: DateTime<Utc>,
        planned_return_at: DateTime<Utc>,
        note: Option<String>,
    ) -> Result<i64, StoreError> {
        self.run(move |s| {
            s.create_activity(
                &familiar_id,
                &type_id,
                &label,
                started_at,
                planned_return_at,
                note.as_deref(),
            )
        })
        .await
    }

    /// See [`HistoryStore::finish_activity`].
    pub async fn finish_activity(
        &self,
        activity_id: i64,
        status: String,
        actual_return_at: DateTime<Utc>,
        experience_text: Option<String>,
    ) -> Result<(), StoreError> {
        self.run(move |s| {
            s.finish_activity(
                activity_id,
                &status,
                actual_return_at,
                experience_text.as_deref(),
            )
        })
        .await
    }

    /// See [`HistoryStore::set_activity_experience`].
    pub async fn set_activity_experience(
        &self,
        activity_id: i64,
        experience_text: String,
    ) -> Result<(), StoreError> {
        self.run(move |s| s.set_activity_experience(activity_id, &experience_text))
            .await
    }

    /// See [`HistoryStore::active_activity`].
    pub async fn active_activity(
        &self,
        familiar_id: String,
    ) -> Result<Option<ActivityRecord>, StoreError> {
        self.run(move |s| s.active_activity(&familiar_id)).await
    }

    /// See [`HistoryStore::latest_activity`].
    pub async fn latest_activity(
        &self,
        familiar_id: String,
        type_id: String,
    ) -> Result<Option<ActivityRecord>, StoreError> {
        self.run(move |s| s.latest_activity(&familiar_id, &type_id))
            .await
    }

    // -- FTS-backed reads ------------------------------------------------

    /// See [`HistoryStore::search_turns`].
    pub async fn search_turns(
        &self,
        familiar_id: String,
        query: String,
        limit: i64,
        channel_id: Option<i64>,
        max_id: Option<i64>,
    ) -> Result<Vec<HistoryTurn>, StoreError> {
        self.run(move |s| s.search_turns(&familiar_id, &query, limit, channel_id, max_id))
            .await
    }

    /// See [`HistoryStore::rebuild_fts`].
    pub async fn rebuild_fts(&self) -> Result<(), StoreError> {
        self.run(HistoryStore::rebuild_fts).await
    }

    /// See [`HistoryStore::latest_fts_id`].
    pub async fn latest_fts_id(&self, familiar_id: String) -> Result<i64, StoreError> {
        self.run(move |s| s.latest_fts_id(&familiar_id)).await
    }

    /// See [`HistoryStore::search_facts`].
    pub async fn search_facts(
        &self,
        familiar_id: String,
        query: String,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<Fact>, StoreError> {
        self.run(move |s| s.search_facts(&familiar_id, &query, limit, include_superseded, as_of))
            .await
    }

    /// See [`HistoryStore::search_facts_scored`].
    pub async fn search_facts_scored(
        &self,
        familiar_id: String,
        query: String,
        limit: i64,
        include_superseded: bool,
        as_of: Option<DateTime<Utc>>,
    ) -> Result<Vec<(Fact, f32)>, StoreError> {
        self.run(move |s| {
            s.search_facts_scored(&familiar_id, &query, limit, include_superseded, as_of)
        })
        .await
    }
}
