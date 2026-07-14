# 05-context-assembly — port spec

Source modules: `src/familiar_connect/context/__init__.py` (re-exports),
`context/assembler.py` (~100 loc), `context/layers.py` (~1560 loc),
`context/final_reminder.py` (~175 loc), `focus.py` (~235 loc),
`budget.py` (~195 loc). Total ~2300 loc.
Reference docs: `docs/architecture/context-pipeline.md` (whole file),
`docs/architecture/tuning.md` §§ "Prompt assembly budget", "History / context
layers", "Retrieval ranking (M2)", "Attentional focus".
Conformance oracle: `tests/test_context_assembler.py`, `test_context_budget.py`,
`test_context_budget_layers.py`, `test_final_reminder.py`,
`test_focus_manager.py`, `test_attentional_layers.py`, `test_phase3_layers.py`,
`test_rag_merge_facts.py`, `test_rag_embedding_rerank.py`,
`test_reflection_layer.py`, `test_people_dossier_layer.py`,
`test_lorebook_layer.py`, plus the layer-rendering halves of
`test_message_reactions.py` and the layer-order pin in `test_run_cmd.py`
(~4300 test loc relevant).

## Role

Turns a responder's "assemble me a prompt" call into (a) one system-prompt
string composed of independently cached, independently token-capped layers and
(b) the recent-history message list, plus the closing "final reminder" block
both responders append twice (head + tail). Also owns the attentional focus
controller (`FocusManager`: two channel pointers, staged→consumed/missed
promotion trigger, unread-nudge debounce) and the token-budget arithmetic
(`TierBudget` + per-model curves) that sizes every layer. This subsystem is
pure read-path: it never writes turns/facts — it renders what 03 stores and
what 07's workers project.

## Public API surface

### `budget.py` — leaf module (zero runtime imports)

- `estimate_tokens(text: str) -> int` — `ceil(len(text)/4)` on *characters*
  (Python `len` = Unicode scalar count, not bytes); `0` for empty string.
- `estimate_message_tokens(msg: Message) -> int` —
  `estimate_tokens(msg.content_str) + 4` (+ `estimate_tokens(msg.name)` when
  `name` set). `content_str` joins `text` blocks of multimodal lists with
  `\n`; plain strings pass through.
- `estimate_messages_tokens(messages) -> int` — sum.
- `ModelBudgetCurve` — frozen dataclass of 12 `float` multipliers (all default
  `1.0`), field names exactly mirroring `TierBudget`'s 12 configurable caps.
  No `total_tokens` field (test-pinned absent).
- `TierBudget` — frozen dataclass, 12 `int` caps. Defaults = voice tier:
  `recent_history_tokens=3000, rag_tokens=900, dossier_tokens=900,
  summary_tokens=600, reflection_tokens=600, lorebook_tokens=600,
  max_history_turns=200, max_rag_turns=10, max_rag_facts=6,
  max_dossier_people=16, max_reflections=6, max_lorebook_entries=12`.
  - `total_tokens` (derived property, NOT a field): sum of the six `*_tokens`
    caps only — count caps excluded. Reporting only; nothing trims against it.
  - `apply_curve(curve) -> TierBudget` — every field scaled via
    `max(1, round(base * multiplier))`. **Python `round()` is
    banker's rounding (half-to-even)** — see Rust notes.

Consumed by 02 (`config.py` parses `[budget.*]` into these types) and by the
run wiring, which passes cap values into layer constructors. Deliberately at
package root so config can import it without dragging in `context`/`llm`.

### `context/assembler.py`

- `AssemblyContext` (frozen): `familiar_id: str`, `channel_id: int | None`,
  `viewer_mode: str = "text"` ("voice" | "text"), `guild_id: int | None`.
- `AssembledPrompt`: `system_prompt: str = ""`,
  `recent_history: list[Message]`.
- `Assembler(layers: list[Layer])`:
  - `async assemble(ctx) -> AssembledPrompt` — see behaviors 1–5.
  - `set_rag_cue(cue: str)` — forwards to the **first** `RagContextLayer` in
    the stack (by isinstance); no-op when none present.

### `context/layers.py`

- `Layer` — **Protocol seam** (structural): `name: str`,
  `async build(ctx) -> str` (empty string opts the layer out),
  `invalidation_key(ctx) -> str` (synchronous). All eight concrete layers
  are swappable implementations; tests define ad-hoc fakes.
- `CharacterCardLayer(card_path: Path)` — `name="character_card"`.
- `OperatingModeLayer(modes: dict[str, str])` — `name="operating_mode"`.
- `RecentHistoryLayer(store: AsyncHistoryStore, window_size=20,
  max_tokens=None, coalesce_max_gap_seconds=45.0,
  silence_gap_fold_seconds=0.0, display_tz="UTC",
  channel_name_resolver: (int)->str|None = None,
  guild_name_resolver: (int)->str|None = None)` — `name="recent_history"`;
  extra method `async recent_messages(ctx) -> list[Message]` (the assembler
  special-cases this type; `build` always returns `""`).
- `ConversationSummaryLayer(store, max_tokens=None)` —
  `name="conversation_summary"`.
- `PeopleDossierLayer(store, window_size=20, max_people=8, max_tokens=None,
  familiar_display_name: str|None=None)` — `name="people_dossier"`.
- `RagContextLayer(store, max_results=5, max_facts=3, recent_window_size=0,
  max_tokens=None, context_window=1, bm25_weight=1.0, recency_weight=0.0,
  importance_weight=0.0, embedding_weight=0.0, embedder: Embedder|None=None,
  fact_overfetch=12, display_tz="UTC")` — `name="rag_context"`; extra method
  `set_current_cue(cue)` (strips; `None`→`""`).
- `ReflectionLayer(store, max_reflections=3, max_tokens=None)` —
  `name="reflection"`.
- `LorebookEntry` (frozen): `keys: tuple[str,...]`, `content: str`,
  `priority: int = 0`, `selective: bool = False`.
- `LorebookLayer(store, path: Path, recent_window=20, max_entries=10,
  max_tokens=None)` — `name="lorebook"`.
- `_turn_to_message_with_context(...)` — module-private but imported by
  `tests/test_text_responder.py`; keep it callable from tests.

### `context/final_reminder.py`

- `build_final_reminder(*, viewer_mode, now=None, display_tz="UTC",
  include_time=True, include_mode_instruction=False, tools_enabled=False,
  post_history_instructions=None, focus_channel_id=None,
  unread_digest: dict[int, tuple[int,int]]|None=None,
  channel_names: dict[int,str]|None=None, guild_name: str|None=None) -> str`.
  Pure function; called by 06 responders twice per turn (head copy with
  `include_time=False`; tail copy with time + mode instruction +
  post-history + guild name).

### `focus.py`

- `PRIVATE_MESSAGE_GUILD_NAME = "Private Message"` (module constant; also
  imported by `final_reminder` and `bot.py` — DM channels are registered
  under this guild name).
- `FocusManager(*, familiar_id, store: AsyncHistoryStore,
  subscriptions: SubscriptionRegistry, clock: () -> float = time.monotonic,
  unread_nudge_enabled=True, nudge_debounce_seconds=30.0,
  catch_up_limit=20)`:
  - `async initialize()` — load persisted pointers, dropping unsubscribed.
  - `catch_up_limit` (read-only property).
  - `get_focus(modality: str) -> int | None` — `"text"` → text pointer;
    **any other string** → voice pointer.
  - `is_subscribed(channel_id) -> bool`, `subscribed_channels() -> list[int]`
    (sorted, deduped across kinds), `is_focused(channel_id) -> bool`
    (membership in `{text_focus, voice_focus}`).
  - `async shift_now(channel_id)` — apply shift immediately (behaviors 40–44).
  - `should_wake(channel_id) -> bool`, `mark_nudge_pending()`.
  - `async end_turn()` — deliberate no-op (kept so responders can await
    uniformly; test-pinned to not move focus and not fire `on_shift`).
  - `channel_label(channel_id|None) -> str` — `"#name(id)"`, `"#id"`, or
    `"none"` for None.
  - `guild_name_for(channel_id|None) -> str|None`, `presence_guild()`,
    `presence_text()` (`"#name"`, falling back to `"#<id>"` via
    `str(channel_id)` — no parens form here).
  - `set_focus_immediately(channel_id, modality)` — startup default seeding,
    no persist, no lock, no promotion.
  - Public mutable attributes: `channel_names: dict[int,str]`,
    `guild_names: dict[int,str]` (populated by 10's `on_ready`),
    `on_shift: async () -> None | None` (presence refresh hook).

## Behaviors & invariants

### Assembler

1. `assemble` iterates layers **in construction order**. A layer that
   `isinstance`-matches `RecentHistoryLayer` is diverted: its
   `recent_messages(ctx)` result becomes `AssembledPrompt.recent_history` and
   it contributes nothing to the system prompt (if several are present, the
   last one wins). All other layers go through the cache.
2. Cache: in-process `dict[(layer.name, layer.invalidation_key(ctx)) -> str]`.
   On hit, `build` is not called. Empty-string results are cached too (still
   skipped when joining). The dict **never evicts** — unbounded growth across
   distinct keys is current behavior (see risks).
3. Non-empty layer texts join with `"\n\n"` in layer order to form
   `system_prompt`.
4. `invalidation_key` is called on every assemble for every non-history layer,
   synchronously, on the event loop. For `LorebookLayer` and
   `PeopleDossierLayer` computing the key is nearly as expensive as `build`
   (file re-read + parse + scan; N+1 sync DB lookups). Correctness depends
   only on key equality, not on key cheapness.
5. `set_rag_cue` mutates shared layer state; the assembler has **no internal
   locking**. Contract with 06: cue-set + assemble happen in the same task
   with no interleaving (VoiceResponder holds a per-channel `asyncio.Lock`
   around set-cue→assemble→stream→commit; TextResponder is a single bus
   consumer). Two assembler instances exist in production (voice tier, text
   tier) sharing the same store and embedder.
6. Production layer order (pinned by `test_run_cmd.py::TestPromptLayerOrder`,
   stability-descending for provider prompt-cache prefix reuse):
   `CharacterCard, OperatingMode, Lorebook, ConversationSummary, Reflection,
   PeopleDossier, RagContext, RecentHistory`.

### Static layers

7. `CharacterCardLayer.build`: missing file → `""`; else file text UTF-8,
   `.strip()`ed. Key: BLAKE2b(**digest_size=8**) hex of file **bytes**;
   literal `"missing"` when absent. Content hash, not mtime — sub-second
   edits must flip the key (test-pinned).
8. `OperatingModeLayer.build`: `modes.get(ctx.viewer_mode, "")`. Key:
   `ctx.viewer_mode`. Production modes dict (exact strings, duplicated in
   `final_reminder._MODE_INSTRUCTIONS` — keep in sync):
   voice: `"You are speaking aloud. Keep replies short (one or two
   sentences). Avoid markdown."`; text: `"You are chatting in a text channel.
   Markdown and multi-line replies are fine."`.

### RecentHistoryLayer

9. `recent_messages` pipeline order (each step feeds the next):
   1. `store.recent_cross_channel(familiar_id, limit=window_size,
      respect_archive=True)` — the **consumed** cross-channel stream ordered
      by `(arrived_at, id)`; staged and missed turns never appear; per-channel
      activity archive watermark hides pre-departure turns (window shrinks,
      never backfills — other read paths stay unfiltered).
   2. Voice-fragment coalescing (behavior 10).
   3. Silence-gap fold (behavior 11).
   4. `in_window_msg_ids` = set of non-null `platform_message_id` of the
      surviving turns (computed **before** token trim — a parent later
      dropped by the trim still renders in snippet form).
   5. One batched `reactions_for_messages` call for all surviving ids
      (test-pinned: exactly one SQL query per assemble).
   6. Render each turn (behaviors 13–17).
   7. Token trim (behavior 12), then realign `turns` to the surviving tail so
      markers track only what the model sees.
   8. Channel markers (behavior 18).
10. Coalescing (`coalesce_max_gap_seconds`, ≤0 disables): consecutive turns
    merge iff same `role`, both authors non-None with equal `canonical_key`,
    **neither** side has `platform_message_id` or `reply_to_message_id`, and
    `0 ≤ gap ≤ max_gap` seconds. Merged turn keeps the earlier turn's id,
    timestamp, channel, guild; content joined with a single space; merged
    `platform_message_id`/`reply_to_message_id` forced to None. Folding is
    left-to-right: a merged row can absorb further fragments.
11. Silence fold (`silence_gap_fold_seconds`, ≤0 disables; needs ≥2 turns):
    scan oldest-first; drop everything before the turn following the **last**
    gap ≥ threshold. No qualifying gap → keep all. Gap at the last position →
    only the newest turn survives.
12. Token trim (`max_tokens`, None disables): walk newest→oldest summing
    `estimate_message_tokens`; stop before the message that would cross the
    cap — but the **newest message is always kept** even if it alone exceeds
    the cap. Result is a contiguous tail.
13. Rendering — user turn with author:
    `[{HH:MM} {label} #{channel_id}{msg_id_tag}] {reply_prefix}{content}`
    (+ ` {reactions_suffix}` when reactions exist), where `HH:MM` is
    `%H:%M` in the layer's `display_tz` (IANA, resolved via zoneinfo at
    construction), `label` = `HistoryStore.resolve_label(canonical_key,
    guild_id=ctx.guild_id, familiar_id)` (guild nick → global_name → username
    → snapshot → bare user_id), `msg_id_tag` = ` #<platform_message_id>` when
    present. `Message.name` = `sanitize_name(author.canonical_key)`
    (`[^a-zA-Z0-9_-]`→`_`, max 64, strip `_`, empty→None). Assistant turns
    and authorless turns render bare content (+reactions suffix), no name, no
    prefix, **no msg-id tag** (deliberate: the tag on assistant turns causes
    output mimicry).
14. `role="tool"` turns replay as `Message(role="user",
    content="[tool result] {content}")` — never protocol `tool` role (orphan
    `tool` messages 500 on Anthropic), never `assistant` (mimicry). Mention
    rewriting applies; reactions suffix does **not**.
15. Mention rewriting (user-visible content and reply-parent text): regex
    `<@!?(\d+)>` → `[@{resolve_label(f"discord:{id}", guild_id, familiar_id)}]`.
    Unknown ids resolve to the bare user_id; never raises.
16. Reply marker: when `reply_to_message_id` resolves via
    `lookup_turn_by_platform_message_id`, prefix (followed by one space):
    - parent in window: `↩ {parent_label}: {snippet}` — snippet =
      mention-rewritten parent content truncated to **80** chars
      (`text[:79] + "…"` when over).
    - parent outside window: `↩ {parent_label} ({HH:MM}): {full}` — parent
      timestamp in layer tz, content capped at **400** chars (same
      `limit-1 + "…"` rule).
    Unknown parent → no marker (silent). Parent label resolved through
    `resolve_label`; authorless parent falls back to its `role`.
17. Reactions suffix: `[reactions: {emoji} x{count} …]` space-joined in the
    store's order (count desc, emoji asc); empty tuple → no suffix. Applies
    to user **and** assistant turns.
18. Channel markers: only when the surviving turns span **>1 distinct
    channel_id** (checked after trim; a window that collapses to one channel
    emits none, byte-for-byte passthrough). A marker precedes the first turn
    and every channel change: `Message(role="user", content=marker)`, no
    `name`. Marker text: `{guild}/{channel}` when the guild-name resolver
    returns a name for that channel, else bare channel; channel is
    `#{resolved_name}` or `#{channel_id}` fallback. Resolvers are plain
    callables bound over `FocusManager.channel_names.get` /
    `FocusManager.guild_name_for` in production; None resolvers → id-only
    markers.
19. `build` returns `""`; `invalidation_key` returns the constant
    `"always-rebuild"` (never cached in effect — but note the assembler
    diverts this type before consulting the cache anyway).

### ConversationSummaryLayer

20. Reads the single per-familiar focus-stream summary at
    `FOCUS_STREAM_CHANNEL_ID = -1` (from 03); `ctx.channel_id` ignored
    (test-pinned: text and voice tiers see the same summary). Missing row or
    whitespace-only text → `""`. Output: `"## Conversation so far\n\n" +
    body.strip()` truncated to `max_tokens` via `_truncate_to_tokens`
    (char-cap = `max_tokens*4`, `text[:cap-1] + "…"`; `max_tokens<=0` → `""`).
21. Key: `"none"` when no row, else
    `f"focus:{last_consumed_at}:{last_summarised_id}"` (composite watermark —
    consumed_at moves on focus-shift promotion even when ids are old).

### PeopleDossierLayer

22. Candidate order: the ego key `ego:{familiar_id}` **always leads** and is
    exempt from `max_people`. Then, if `ctx.channel_id` is set: walk the
    active channel's last `window_size` turns **newest-first** (store
    `recent()` returns oldest-first; layer reverses); per turn append the
    author's canonical_key then each `mentions_for_turn(turn_id)` key,
    dedupe on first sight; stop scanning once ≥ `max_people` people
    collected; slice to `max_people`. `channel_id=None` → ego only.
23. `build`: per candidate, `get_people_dossier`; missing row or blank text →
    skipped silently. Ego candidate: header display =
    `familiar_display_name or ctx.familiar_id.title()`, **no** profile
    lookup. Others: display via `resolve_label`, profile via
    `get_account_profile`. Header block:
    `### {display}` + optional `@{username} · {pronouns}` line (either alone
    if only one set; omitted if both empty) + optional
    `Bio: {bio truncated to 240 chars}` line. Section =
    `header + "\n\n" + dossier_text.strip()`.
24. Token budgeting (shared pattern with Reflection/Lorebook): running
    `remaining` starts at `max_tokens` (None disables). Per section: if
    `estimate_tokens(section) > remaining` **and at least one section already
    kept** → stop; else truncate section to `remaining` (no-op when it fits)
    and subtract the truncated section's estimate. Net effect: the first
    section is truncated rather than dropped; later sections are dropped
    whole.
25. Output: `"## People in this conversation\n\n"` + sections joined
    `"\n\n"`; `""` when no sections.
26. Key: `f"t{latest_id}|cap{max_people}|"` + per-candidate
    `{key}:f{last_fact_id}` or `{key}:none`, joined `|` (exact composition:
    parts list `[f"t{latest}", f"cap{n}", *per-key]` joined with `|`).
    `latest_id` queried with `channel_id=ctx.channel_id or 0`, `None`→0.
    Any new channel turn flips it; a worker dossier refresh flips the
    per-key watermark.

### RagContextLayer

27. Empty/unset cue → `""` (layer opts out; responder seeds the cue with the
    inbound user text each turn via `Assembler.set_rag_cue`).
28. Recent-window exclusion: when `recent_window_size > 0` and
    `ctx.channel_id` set, `max_id = latest_id(channel) - recent_window_size`
    (None when channel empty); passed to `search_turns` and re-applied to
    neighbour expansion so nothing already shown verbatim by
    RecentHistoryLayer re-surfaces. `recent_window_size=0` (default)
    preserves unfiltered behavior.
29. Turn search: `search_turns(familiar_id, query=cue, limit=max_results,
    max_id=max_id)` (tantivy BM25, OR semantics, English analyzer — owned by
    03).
30. Fact path forks on `_rerank_facts` = `recency_weight>0 or
    importance_weight>0 or (embedding_weight>0 and embedder is not None)`:
    - off: `search_facts(limit=max_facts)` (BM25 order).
    - on: over-fetch `fetch = max(min(fact_overfetch, max_facts*4),
      max_facts)` scored candidates via `search_facts_scored`, optionally
      compute embedding similarities, then rerank (behavior 31) and keep top
      `max_facts`.
31. Rerank fusion — each signal normalized to [0,1] **within the candidate
    batch**, then `score = Σ weight_i * quality_i`:
    - BM25: `(s - min) / (max - min)`; **span 0 → 1.0 for all** (the
      docstring's "ties map to 0.5" is stale FTS5-era text; code and tests
      pin 1.0. Raw scores are tantivy-positive, higher = better).
    - Recency: rank of `fact.id` among the batch's distinct sorted ids,
      `i/(n-1)`; newest = 1.0; single candidate → 1.0.
    - Importance: `importance/10`; `None` → 0.5 (legacy rows neutral).
    - Embedding: `(cosine+1)/2`; facts without a stored vector → 0.5.
    Tie-break: sort key `(score, -candidate_index)` descending — equal scores
    keep BM25 candidate order deterministically.
32. Embedding similarities: skip entirely (`{}`) when weight ≤ 0, embedder
    None, or no candidates. Fetch stored vectors first
    (`get_fact_embeddings(fact_ids, model=embedder.name)`); **only if at
    least one vector exists** call `embedder.embed([cue])` (test-pinned: no
    embed call on a cold side-index). Cosine returns 0.0 on length mismatch
    or zero norm. `embedding_weight>0` with no embedder → log one warning
    (once per layer instance) and fall back to BM25-only; the rerank passes
    `embedding_weight=0` in that case.
33. Fact line render: `- {fact.text}` with optional rename annotation
    ` ({display_at_write} is now known as {current}; …)` — one note per
    distinct subject canonical_key whose `resolve_label` output differs from
    both `display_at_write` and the bare user_id tail of the key (the latter
    means "nothing known", annotate nothing). Original text never rewritten
    (annotation, not substitution).
34. Turn lines render: expand each hit to ids `hit.id ± context_window`
    (`max(0, …)` at construction), fetch via `turns_by_ids`, drop neighbours
    from channels no hit came from and any active-channel turn with
    `id > max_id`. Group by date `%Y-%m-%d` in layer tz; date groups emitted
    in **sorted (chronological) order**, each as `"{date}:"` then per turn
    `> [{h:MMAM/PM} {label}]: {first line}` (12-hour clock, leading zero
    stripped) with continuation lines as `> {line}` (bare `>` for empty
    lines, preserving blockquote spacing); blank line between groups,
    trailing blank popped.
35. Joint token cap (`max_tokens`, None disables): facts first then turns,
    adding whole lines while they fit; the first fact line is always kept
    even over-cap; the first turn line is kept only when **no** fact lines
    were kept. Headers are not counted (flat ~10-token overhead absorbed by
    the budget).
36. Output sections (either may be absent): `"## Possibly relevant facts\n"`
    + fact lines (joined `\n`), and `"## Possibly relevant earlier turns\n"`
    + turn lines; sections joined `"\n\n"`. No hits at all → `""`.
37. Key: `f"{cue}|t{latest_fts_id}|f{latest_fact_id}"` —
    `latest_fact_id` counts superseded rows too, so
    supersession-by-replacement always flips the key.

### ReflectionLayer

38. `max_reflections <= 0` (clamped at construction via `max(0,…)`) → `""`.
    Reads `recent_reflections(familiar_id, channel_id=ctx.channel_id,
    limit=max_reflections)` — channel-scoped, channel-agnostic rows always
    surface (store-side). Stale check is **one batched**
    `superseded_fact_ids` call across every row's citations. Line:
    `- {text.strip()}[ [T#{id}, …, F#{id}, …]][ (stale)]` — turn citations
    before fact citations, comma-space joined inside one bracket; `(stale)`
    when ≥1 cited fact superseded; rows never dropped for staleness. Token
    budget: same first-truncated/rest-dropped pattern as behavior 24. Output
    `"## Recent reflections\n\n"` + lines joined `"\n"`.
39. Key: `f"ch{ctx.channel_id}|r{latest_id}|cap{max_reflections}"` where
    `latest_id` = newest matching reflection's id via
    `recent_reflections(limit=1)`, 0 when none.

### LorebookLayer

40. File loading (every build **and** every invalidation_key — no caching of
    the parse): missing file, TOML parse error, or OSError → no entries
    (empty block, never raises; test-pinned on malformed TOML). Schema:
    top-level `entries` must be a list; per entry (non-dict rows skipped):
    `keys` list → keep non-empty strings only, entry dropped if none survive;
    `content` stringified + stripped, dropped if empty; `priority` kept only
    for true ints (bool explicitly rejected → 0); `selective` via
    truthiness.
41. Matching: scan text = last `recent_window` turns of the **active
    channel** (`ctx.channel_id`; None → no scan → no matches), contents
    joined `\n`, lowercased; key match = case-insensitive **substring**.
    `selective=True` → all keys must match (AND); else any (OR). Matched
    indices sorted by `(-priority, file_index)`, sliced to `max_entries`
    (`max_entries<=0` → `""` before any work).
42. Token budget: same pattern as behavior 24 over entry contents. Output
    `"## Lorebook\n\n"` + contents joined `"\n\n"`.
43. Key: `f"f{file_hash}|ch{ctx.channel_id}|m{i,j,…}|cap{max_entries}"` —
    file content hash (same BLAKE2b-8 helper, `"missing"` when absent) plus
    the matched file-order indices comma-joined (pre-priority-sort,
    pre-cap). Flips on file edit or match-set change.

### build_final_reminder

44. Output is `\n`-joined lines starting with `"---"`. Blocks in order, each
    preceded by one blank line, each independently omitted:
    1. Time (`include_time`): `It is now: {YYYY-MM-DD} {h:MMAM/PM} {TZ}` in
       `display_tz` (12-hour, hour leading-zero stripped, `%Z` abbreviation
       e.g. `EDT`).
    2. Text-mode sentinels (`viewer_mode == "text"` only):
       `Special input:` + bullets `` * `[@DisplayName]` - ping user `` and
       `` * `[↩ <message_id>]` - reply to message ``. Voice mode lists none.
    3. Mode instruction (`include_mode_instruction`): the exact
       `_MODE_INSTRUCTIONS[viewer_mode]` string (voice/text only; unknown
       mode silently omits).
    4. Voice tool nudge (`tools_enabled and viewer_mode == "voice"`):
       `Always speak at least a brief acknowledgement before calling a tool.
       Never reply with a tool call alone.`
    5. Focus + unread block (when `focus_channel_id is not None` **or**
       `unread_digest` truthy): focus clause
       `Your attention is currently on {#name-or-#id}` +
       ` in a private message` when `guild_name ==
       PRIVATE_MESSAGE_GUILD_NAME`, or ` in the "{guild_name}" server` when
       guild_name is a non-empty string (empty string behaves like None —
       plain clause) + `.`. Unread clause over channels with `unread > 0`:
       `There {is/are} {a new message/new messages} in {list} — use
       shift_focus if it pulls your attention.` (singular iff total unread
       == 1). Per-channel list item: `#{name} (id {cid})` when named else
       `#{cid}`, plus suffix from `(unread, pings)`: pings=0 → ` ({unread})`
       only when unread>1; unread==pings → ` ({pings} ping[s])`; mixed →
       ` ({unread}, {pings} ping[s])`. Named channels **must** carry the
       numeric id (model needs it for `shift_focus`). Focus and unread
       clauses joined with a single space into one line; both empty → block
       omitted.
    6. `post_history_instructions` (stripped) last — deepest recency slot;
       None/blank omits.
45. Responder contract (06): head system message =
    `assembled.system_prompt + "\n\n" + [text-only output-controls addendum]
    + "\n\n" + build_final_reminder(include_time=False, focus/unread/names)`
    — time deliberately omitted so the cache prefix stays byte-stable, and
    guild_name deliberately omitted from the head copy. Tail system message
    (after recent history) = full reminder with `display_tz` clock,
    `include_mode_instruction=True`, `post_history_instructions`,
    `guild_name`, and the same focus/unread arguments. Voice responder uses
    a reminder with `tools_enabled` and no focus/unread digest.
    `unread_digest` values are `ChannelUnread(unread, pings)` NamedTuples
    from `staged_channels` — the renderer treats them as plain 2-tuples.

### FocusManager

46. State: two independent pointers (`text`, `voice`), each guarded by its
    **own** `asyncio.Lock`. Pointers start None; `initialize()` loads the
    persisted `focus_pointers` row (via async store) and drops any pointer
    whose channel is no longer in the SubscriptionRegistry (warn log); no
    row → both stay None. Run wiring then seeds unset pointers with the
    first text / first voice subscription via `set_focus_immediately`
    (in-memory only).
47. `shift_now(channel_id)`: modality inferred from
    `subscriptions.kind_for(channel_id)` — `voice` kind → voice; **anything
    else including None (unsubscribed) → text**. Callers (the `shift_focus`
    tool, subsystem 08) are responsible for rejecting unsubscribed targets
    before calling; `shift_now` itself does not guard.
48. Text shift, under the text lock: (a)
    `store.promote_staged_turns(familiar_id, channel_id, catch_up_limit)` —
    03 flips the channel's last `catch_up_limit` staged turns (plus any
    that ping the bot, regardless of age) to consumed
    (`consumed_at = now`) and marks older staged backlog `missed_at`
    (terminal); returns `Promotion(consumed, missed)` counts (logged);
    (b) move the text pointer; (c) persist **both** pointers via
    `set_focus_pointers`. Voice shift, under the voice lock: move pointer +
    persist both (no promotion — test-pinned). `on_shift` (if set) is
    awaited **after** the lock is released, once per shift.
49. Cross-modal note: each shift persists both pointers while holding only
    its own modality lock — two concurrent opposite-modality shifts race on
    the DB row (last write wins with the writer's snapshot of the other
    pointer). Benign under the single-event-loop, effectively serialized
    call pattern, but the Rust port should not make it worse (a single
    state mutex is an acceptable simplification).
50. Shifts are immediate, not deferred: a reply later in the same turn posts
    to the *new* focus, and a turn that goes silent still leaves focus where
    it moved. There is no pending-shift state (`end_turn` is a no-op).
51. Unread nudge: `should_wake(channel_id)` is true iff
    `unread_nudge_enabled` AND channel is not focused (in either modality)
    AND `clock() - last_nudge >= nudge_debounce_seconds`. `last_nudge`
    initializes to `-inf` (first arrival always eligible) and is set to
    `clock()` only by `mark_nudge_pending()` — the caller (06) marks pending
    when it actually emits the synthetic wake event. The nudge never moves
    focus; the arrival itself fires it (no idle requirement); debounce is
    the sole throttle and re-arms after each window. `clock` is injectable
    (default `time.monotonic`) — tests drive a fake.
52. Consumers of the pointer/label API: 06 responders
    (`is_focused` gate for staging, `get_focus("text")` for wake routing and
    digest), 08 tools (`shift_now`, `subscribed_channels`, `channel_label`,
    `catch_up_limit` preview size for `read_channel`/`shift_focus`), 10 bot
    (presence via `presence_guild`/`presence_text`, populates
    `channel_names`/`guild_names`, sets DM guilds to
    `PRIVATE_MESSAGE_GUILD_NAME`), 11 activities engine (structural
    `FocusLike` subset: `get_focus`, `is_focused`, plus its own nudge state
    modeled on `should_wake`).

### Budget

53. Every cap is enforced independently by its layer; there is no combined
    cap and no post-assembly trim. `total_tokens` is derived
    (six `*_tokens` fields only) and nothing may trim against it
    (test-pinned as a property, not a field).
54. `budget_for(tier)` (lives in 02's `CharacterConfig`): base tier budget,
    then the `ModelBudgetCurve` registered for the tier's active model
    (tier→slot: voice→fast, text→prose, background→background) applied via
    `apply_curve`. Curve multipliers are validated positive at config load;
    unknown curve keys are load errors.

## Data formats

- **Rendered message shapes** (the de-facto wire format toward the LLM —
  behaviors 13–18, 33–34 give exact grammar):
  - user turn: `[{HH:MM} {label} #{channel_id}[ #{platform_message_id}]] [↩ …] {content} [[reactions: …]]`, `name` = sanitized canonical key.
  - channel marker: own `role="user"` message, `{Guild}/{#channel}` or `#channel`/`#<id>`, no name.
  - tool replay: `role="user"`, `[tool result] {content}`.
  - RAG turn block: `YYYY-MM-DD:` header + `> [h:MMAM/PM Label]: …`
    blockquote lines.
  - Section headings (exact): `## Conversation so far`, `## People in this
    conversation`, `## Recent reflections`, `## Lorebook`, `## Possibly
    relevant facts`, `## Possibly relevant earlier turns`.
- **`lorebook.toml`** (`data/familiars/<id>/lorebook.toml`):
  `[[entries]]` with `keys: [str, …]` (required non-empty), `content: str`
  (required non-empty), `priority: int = 0`, `selective: bool = false`.
  Hand-authored; file is sole source of truth; malformed → silently empty.
- **`character.md`** (`data/familiars/<id>/character.md`): free-form UTF-8
  persona + operational essentials; consumed verbatim (stripped).
- **`focus_pointers` table** (owned by 03, consumed here):
  `familiar_id TEXT PK, text_channel_id INT NULL, voice_channel_id INT NULL,
  updated_at TEXT (ISO-8601 UTC)`; upsert on every shift.
- **Cross-type contracts** (owned elsewhere, shapes relied on here):
  `HistoryTurn` (id, timestamp tz-aware, role, author, content, channel_id,
  platform_message_id, reply_to_message_id, guild_id);
  `ChannelUnread(unread: int, pings: int)` NamedTuple — unpacks as 2-tuple;
  `Promotion(consumed: int, missed: int)`; `AccountProfile(username?,
  pronouns?, bio?)`; `Fact(id, text, importance?, subjects:
  [{canonical_key, display_at_write}])`; reflection rows (`text`,
  `cited_turn_ids`, `cited_fact_ids`, `id`); summary row (`summary_text`,
  `last_consumed_at`, `last_summarised_id`); `Message(role, content,
  name?, tool_calls?, tool_call_id?)` with `content_str`.
- **Constants**: `FOCUS_STREAM_CHANNEL_ID = -1`;
  `PRIVATE_MESSAGE_GUILD_NAME = "Private Message"`; reply caps 400/80 chars;
  bio cap 240 chars; chars-per-token 4; per-message overhead 4 tokens;
  BLAKE2b digest_size 8; default nudge debounce 30.0 s; default catch-up 20.

## Config knobs

All via `character.toml` deep-merged over `data/familiars/_default/
character.toml` (parsing itself is subsystem 02; values are threaded into
constructors by the run wiring). No env vars are read by this subsystem.

| Key | Default | Feeds |
|---|---|---|
| `display_tz` | `"UTC"` (IANA-validated at load) | RecentHistory/Rag layer clocks, final-reminder tail clock |
| `[providers.history].voice_window_size` | `100` | voice-tier `window_size` (RecentHistory, PeopleDossier, Lorebook scan, RAG exclusion) |
| `[providers.history].text_window_size` | `200` | text-tier `window_size` (same fan-out) |
| `[providers.history].coalesce_max_gap_seconds` | `45.0` | `RecentHistoryLayer.coalesce_max_gap_seconds` (both tiers) |
| `[providers.history].text_silence_gap_fold_seconds` | `0.0` (disabled; ≥0 enforced) | text-tier `silence_gap_fold_seconds` only |
| `[channels.<id>].history_window_size` | unset | per-channel window override (02 resolves) |
| `[budget.voice]` | 3000/900/900/600/600/600 tokens; 200/10/6/16/6/12 counts | voice `TierBudget` |
| `[budget.text]` | 8000/2400/2400/1600/1600/1600; 400/16/10/24/10/20 | text `TierBudget` |
| `[budget.background]` | 24000/8000/8000/4000/4000/4000; 1000/24/16/32/16/32 | background `TierBudget` (workers, subsystem 07) |
| `[budget.model_curves."<model>"].<field>` | `1.0` each; must be > 0; unknown keys reject | `ModelBudgetCurve` per model, applied in `budget_for` |
| `[memory.retrieval].bm25_weight` | `1.0` (shipped default file: 1.0) | RagContextLayer |
| `[memory.retrieval].recency_weight` | `0.0` | RagContextLayer |
| `[memory.retrieval].importance_weight` | dataclass default 0.0; shipped `_default` toml sets `0.6` | RagContextLayer |
| `[memory.retrieval].embedding_weight` | `0.0` (negatives rejected at load) | RagContextLayer |
| `[focus].unread_nudge_enabled` | `true` | FocusManager |
| `[focus].nudge_debounce_seconds` | `30.0` | FocusManager |
| `[focus].catch_up_limit` | `20` (positive int) | FocusManager → promotion + shift_focus preview |
| `[prompt].post_history_instructions` | shipped etiquette text; empty omits | tail final reminder |

Constructor-only knobs (no TOML today): `RagContextLayer.context_window=1`,
`fact_overfetch=12`, `recent_window_size` (wired = the tier's window size),
`OperatingModeLayer.modes` (hardcoded strings in run.py, duplicated in
`_MODE_INSTRUCTIONS`).

## Dependency edges

Imports (05 → other subsystems):

| Module | Imports | Subsystem |
|---|---|---|
| `budget.py` | `llm.Message` (TYPE_CHECKING only — leaf at runtime) | 08 |
| `context/layers.py` | `budget` (05); `history.store.FOCUS_STREAM_CHANNEL_ID`, `HistoryTurn` (+ type-only `HistoryStore`, `AsyncHistoryStore`, `AccountProfile`, `Fact`) | 03 |
| | `identity.ego_canonical_key`, `is_ego_key` (+ type `Author`) | 02 |
| | `llm.Message`, `sanitize_name` | 08 |
| | type-only `embedding.protocol.Embedder` | 04 |
| `context/final_reminder.py` | `focus.PRIVATE_MESSAGE_GUILD_NAME` | 05 (internal) |
| `focus.py` | `log_style` | 01 |
| | `subscriptions.SubscriptionKind` (+ type `SubscriptionRegistry`) | 02 |
| | type-only `history.async_store.AsyncHistoryStore` | 03 |

Imported by (other subsystems → 05):

- 02 `config.py`: `TierBudget`, `ModelBudgetCurve` (budget types are part of
  the config surface).
- 06 responders: `AssemblyContext`, `build_final_reminder`, type
  `Assembler`, type `FocusManager`; tests also import
  `_turn_to_message_with_context`.
- 08 tools (`registry.py`, `builtins.py`, `shift_focus.py`,
  `read_channel.py`): type `FocusManager`; `shift_focus` tool calls
  `shift_now`/`subscribed_channels`/`channel_label`/`catch_up_limit`.
- 10 `bot.py`: `PRIVATE_MESSAGE_GUILD_NAME`, type `FocusManager`; mutates
  `channel_names`/`guild_names`, reads presence helpers.
- 11 `activities/engine.py`: structural `FocusLike` protocol over a
  `FocusManager` subset.
- Wiring `commands/run.py`: constructs `FocusManager`, both `Assembler`
  stacks (`_default_assembler`), seeds startup focus.
- Package `__init__.py` re-exports `FocusManager`.

Store methods this subsystem requires from 03 (async proxy + `.sync` raw):
`recent_cross_channel`, `recent`, `reactions_for_messages`, `resolve_label`,
`lookup_turn_by_platform_message_id`, `get_summary`, `get_people_dossier`,
`get_account_profile`, `mentions_for_turn`, `latest_id`, `latest_fts_id`,
`latest_fact_id`, `search_turns`, `search_facts`, `search_facts_scored`,
`get_fact_embeddings`, `recent_reflections`, `superseded_fact_ids`,
`turns_by_ids`, `get_focus_pointers`, `set_focus_pointers`,
`promote_staged_turns`, `staged_channels`.

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `test_context_assembler.py` (883) | Card read/missing/hash key; mode layer; recent-history rendering (name+`[HH:MM …]` prefix, UTC + display_tz), window size, empty `build`; coalescing rules (speaker/role/msg-id/reply/gap boundaries, knob override); silence fold (last gap, zero disable, tail-only); guild-nick label; reply markers in/out-of-window + unknown parent; mention rewrite + unknown-id fallback; assembler compose order, `\n\n` join, cache hit/miss on key change, history passthrough | logic-portable (in-memory HistoryStore fixture) |
| `test_context_budget.py` (196) | Estimator (ceil/4, +4 overhead, name cost); TierBudget defaults/partial override; derived `total_tokens` excludes counts, is not a field; curve defaults, no total field; apply_curve identity/scale/round/floor-1/count fields | logic-portable except `TestEstimatorPerf` (<1 ms wall-clock — Python-specific-skip or rewrite as bench) |
| `test_context_budget_layers.py` (159) | recent-history drop-oldest under cap, None = full window; RAG joint section cap; dossier trailing-drop; summary truncate | logic-portable |
| `test_final_reminder.py` (271) | `---` opener; text sentinels present, voice absent; tz clock + `%Z` abbrev; include_time=False; mode instruction on request only, unknown mode silent; post-history tail placement + blank omission; guild-name clause exact wording (`in the "X" server`), empty==None, no leak without focus; DM `in a private message` clause | logic-portable |
| `test_focus_manager.py` (605) | kind_for; initialize load / drop-unsubscribed / no-row; is_focused both modalities; shift_now text promotes + persists, voice does not promote; end_turn no-op; should_wake truth table incl. fake-clock debounce expiry/re-arm; modality independence; set_focus_immediately; presence_text/guild fallbacks; guild_name_for; on_shift fires per shift, not on end_turn | logic-portable (needs SubscriptionRegistry + store fixtures, injectable clock) |
| `test_attentional_layers.py` (592) | cross-channel consumed-only window, archive watermark scoping; `#channel_id` tag in every user prefix + msg-id tag coexistence; channel markers (single-channel byte-identical, leading+change markers, resolver naming, id fallback, guild-less, role=user no-name, post-trim realign, trim-to-single-channel emits none); final-reminder focus directive + unread digest wording, zero-count exclusion, ping suffix grammar, name+numeric-id | logic-portable |
| `test_phase3_layers.py` (532) | summary layer sentinel-channel read, tier-agnostic, composite watermark key; RAG empty-cue opt-out, cue matching, familiar scoping, key = cue+watermarks, recent-window exclusion (and 0 = legacy), rename annotation on/off/legacy, date header + 12 h clock + display_tz, multiline blockquote continuation, neighbour context expansion; tool-turn narration role=user | logic-portable |
| `test_rag_merge_facts.py` (152) | facts+turns dual sections, fact-only, fact-watermark key flip; importance rerank beats BM25 order, default weights preserve BM25, NULL importance neutral | logic-portable |
| `test_rag_embedding_rerank.py` (216) | cosine rerank ordering; zero weight disables; missing-embedder warn-once + BM25 fallback (caplog assertion); unembedded neutral 0.5; no embed call when no stored vectors (spy) | mostly logic-portable; warn-once + call-count assertions need Rust log-capture / mock embedder |
| `test_reflection_layer.py` (220) | empty; `[T#…, F#…]` breadcrumbs; `(stale)` on any superseded citation, absent otherwise; caps + zero opt-out; key flips on new row; channel scoping; token truncate | logic-portable |
| `test_people_dossier_layer.py` (340) | empty cases; header + `@user · pronouns` / `Bio:` variants and omissions; mention-sourced candidates; most-recent-first capping; dedupe; channel scoping; ego dossier always injected (alone and alongside); key flips on new turn and on dossier watermark | logic-portable |
| `test_lorebook_layer.py` (269) | missing file; no-match; substring + case-insensitive match; priority desc + file-order ties; selective AND; max_entries; token truncate; scan window bound; empty keys never fire; key flips on match-set and file change; malformed TOML → empty; active-channel scoping | logic-portable |
| `test_message_reactions.py` (classes `TestRecentHistoryReactions`, `TestRecentHistoryReactionsBatch`) | `[reactions: …]` suffix on user and assistant turns, absent when none; single batched query per assemble (execute spy) | rendering logic-portable; the single-query spy needs a Rust store mock/counter |
| `test_run_cmd.py` (`TestPromptLayerOrder`) | production layer ordering pins (summary before reflection, dossier before RAG, etc.) | needs-Rust-mock (wiring-level; port as an assertion on the Rust builder) |
| `test_attentional_responders.py` / `test_attentional_tools.py` | FocusManager contract exercised from 06/08 (staging gate, wake emission + `mark_nudge_pending`, shift promotion + preview) | cross-subsystem — inventoried under 06/08 |

## Rust port notes

- **Layer protocol → trait.** `#[async_trait] trait Layer { fn name(&self)
  -> &str; async fn build(&self, ctx) -> String; fn invalidation_key(&self,
  ctx) -> String; }`. The assembler's `isinstance(RecentHistoryLayer)` /
  `isinstance(RagContextLayer)` special-casing is duck-typing to redesign:
  make recent-history a distinct assembler slot (not a `Layer`), and give
  the assembler an explicit optional handle to the RAG layer (or route the
  cue through `AssemblyContext`) instead of downcasting. `test_run_cmd`'s
  order pin then applies to the system-prompt layer vec only.
- **Sync/async split.** Python layers use the async store for `build` but
  the raw sync store (blocking Turso/tantivy calls **on the event loop**)
  inside `invalidation_key` — a known wart, not a feature. In Rust either
  make `invalidation_key` async, or keep it sync but have the assembler
  compute keys via `spawn_blocking` alongside the store's thread model
  (see spec 03). Do not replicate loop-blocking.
- **Cache.** Keep the `(name, key) -> String` memo per assembler; consider
  bounding it (e.g. keep only the latest key per layer name) — the Python
  dict grows without bound and nothing reads old entries, so per-layer
  single-slot caching is behavior-equivalent and leak-free.
- **Banker's rounding.** `apply_curve` uses Python `round()`
  (half-to-even); Rust `f64::round` is half-away-from-zero. Current tests
  never hit a .5 case, but for parity implement half-to-even explicitly
  (`(x).round_ties_even()` is stable since Rust 1.77) or document the
  divergence.
- **Character counting.** All truncation caps (`limit-1 + "…"`, char/4
  estimates, bio 240, reply 400/80) count Unicode scalars, not bytes —
  use `chars().count()` / `char_indices` for slicing, never byte slicing
  (emoji-heavy chat *will* land mid-codepoint otherwise). The ellipsis is
  U+2026, the reply glyph U+21A9, and the digest renders literal emoji —
  keep everything UTF-8 exact.
- **Time.** `chrono` + `chrono-tz`. Needed formats: `%H:%M`, `%Y-%m-%d`,
  12-hour `%I:%M%p` with leading zero stripped, and `%Z` **abbreviation**
  (chrono-tz gives abbreviations via offset name; verify `EDT`-style output
  matches the test in `test_final_reminder.py`). IANA names are validated
  at config load (02); layers may assume validity and resolve once at
  construction.
- **Hashing.** `blake2` crate, `Blake2bVar` with 8-byte output, lowercase
  hex — the key string must match only itself across runs (no cross-language
  compat needed, but keep it content-based, not mtime).
- **Lorebook TOML.** `toml` crate with manual field extraction mirroring the
  lenient rules (behavior 40) — do not derive-Deserialize strictly; every
  malformed shape degrades to "entry dropped" or "no entries", never an
  error.
- **Regex.** `regex` crate for `<@!?(\d+)>`; replacement is per-match with a
  store lookup — use `replace_all` with a closure (`Replacer`).
- **FocusManager.** Two `tokio::sync::Mutex`es mirror the Python locks, but
  a single `Mutex<FocusState>` is simpler and closes the benign
  double-pointer persist race (behavior 49) — acceptable redesign; keep
  `on_shift` invoked outside the lock. `clock` should stay injectable
  (`Box<dyn Fn() -> f64>` or an `Instant`-based trait) for the debounce
  tests; note `-inf` initial value means "first arrival always wakes".
  `channel_names`/`guild_names` are mutated by the Discord shell from event
  handlers — wrap in the same state lock or `RwLock` rather than exposing
  bare maps.
- **Seams to preserve for tests.** Store trait object (03's store trait
  covers all 23 methods listed above), `Embedder` trait (04), injectable
  clock, `tmp_path`-style file fixtures for card/lorebook. The warn-once
  embedder fallback should use `tracing` so the test can assert via a
  capture layer.
- **What not to transliterate.** The `_turn_to_message` legacy renderer is
  dead in production (only voice-intake era); port only if 06/09 specs
  still reference it — otherwise drop. The stale docstring claims (BM25
  "negative, ties → 0.5") must not leak into the port: implement the code's
  actual normalization (tantivy-positive, span-0 → 1.0).
