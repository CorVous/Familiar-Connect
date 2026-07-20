# 02-config-and-identity — port spec

Source modules: `src/familiar_connect/config.py`, `identity.py`, `familiar.py`,
`subscriptions.py`, `prompt_fill.py`, `structured_output.py`.
Reference doc: `docs/architecture/configuration-model.md`.
Conformance oracle: `tests/test_config.py` (1661 lines, ~160 tests),
`tests/test_identity.py`, `tests/test_familiar.py`, `tests/test_subscriptions.py`,
`tests/test_prompt_fill.py`, `tests/test_structured_output.py`.

## Role

This subsystem is the process-wide "load once, immutable thereafter" layer: it
parses `character.toml` (deep-merged over the checked-in default profile) into
a fully validated, frozen `CharacterConfig`; models speaker identity (`Author`)
with stable storage keys; bundles all per-character runtime state into
`Familiar`; and persists the channel-subscription registry as a TOML sidecar.
Two small pure-utility modules ride along: crash-safe placeholder fill for
config-sourced prompt text, and tolerant JSON coercion for LLM replies. There
is no async code anywhere in this subsystem — everything is synchronous,
startup-time or pure-function work.

## Public API surface

### config.py

- `ConfigError(Exception)` — the single error type for every config problem
  (malformed TOML, unknown key, bad type, out-of-range value, missing default
  profile). Callers `match` on message substrings in tests; keep messages
  stable (see Behaviors for the exact phrases tests pin).
- `load_character_config(path: Path, *, defaults_path: Path) -> CharacterConfig`
  — the only loader. Reads defaults TOML (must exist, else `ConfigError`
  containing "default character profile"), reads target TOML (missing file →
  `{}`; parse error → `ConfigError`), deep-merges target over defaults, then
  parses/validates the merged dict.
- `CharacterConfig` (frozen dataclass) — ~30 fields, see Config knobs. Methods:
  - `for_channel(channel_id: int | None) -> ChannelOverrides` — `None` or
    unknown id → default (all-`None`) `ChannelOverrides`.
  - `voice_window_for(channel_id) -> int` / `text_window_for(channel_id) -> int`
    — channel `history_window_size` override wins over the tier default. Note
    a single channel override applies to BOTH tiers.
  - `budget_for(tier: str) -> TierBudget` — `self.budgets[tier]` (KeyError on
    unknown tier — callers only pass canonical names), then if the tier's slot
    (`voice→fast`, `text→prose`, `background→background`) exists in `self.llm`
    and `budget_curves` has an entry keyed by that slot's *model string*,
    return `base.apply_curve(curve)`; otherwise the base.
- Frozen sub-config dataclasses, all constructible with pure-default values
  (tests build them directly): `LLMSlotConfig`, `TTSConfig`,
  `DiscordTextConfig`, `MemoryRetrievalConfig`, `MemoryProvidersConfig` (+ five
  per-worker knob structs `RollingSummaryConfig`, `RichNoteConfig`,
  `PeopleDossierConfig`, `ReflectionConfig`, `FactSupersedeConfig`),
  `EmbeddingConfig`, `ChannelOverrides`, `TurnDetectionConfig`,
  `LocalTurnConfig`, `STTConfig`, `DeepgramSTTConfig`, `ParakeetSTTConfig`,
  `FasterWhisperSTTConfig`, `FocusConfig`, `ToolsConfig`.
- Constants other modules import: `LLM_SLOT_NAMES = {"fast","prose","background"}`,
  `BUDGET_TIER_NAMES = {"voice","text","background"}`,
  `REASONING_LEVELS = {"off","none","low","medium","high","default"}`,
  `DEFAULT_AZURE_TTS_VOICE = "en-US-AmberNeural"`,
  `DEFAULT_GEMINI_TTS_VOICE = "Kore"`,
  `DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"`.
- `parse_hhmm_range(value, *, key: str) -> (time, time)` — public: shared by
  `[sleep].window` here and the activities catalog (subsystem 04/07 territory)
  for `active_hours`. Format `"HH:MM-HH:MM"`, exactly two 2-digit fields per
  side, hour ≤ 23, minute ≤ 59, may wrap midnight, `start == end` rejected.

### identity.py

- `Author` (frozen dataclass): `platform: "discord"|"twitch"|"ego"`,
  `user_id: str`, `username: str|None`, `display_name: str|None`,
  `global_name: str|None = None`, `guild_nick: str|None = None`,
  `pronouns: str|None = None`, `bio: str|None = None`,
  `aliases: frozenset[str] = ∅`. Properties:
  - `canonical_key -> str` = `f"{platform}:{user_id}"` — the storage-stable id
    used throughout history/memory (subsystem 03 keys rows on it).
  - `slug -> str` — lowercase `canonical_key`, every run of `[^a-z0-9]+`
    collapsed to one `-`, leading/trailing `-` stripped. Basename of
    `people/<slug>.md` in the memory store. (`"discord:42"` → `"discord-42"`,
    `":x:" + ":7:"` → `"x-7"`, `"twitch:U__99!!"` → `"twitch-u-99"`.)
  - `label -> str` = `display_name or username or user_id` (first truthy).
  - `openai_name -> str|None` = `sanitize_name(label) or sanitize_name(user_id)`.
    `sanitize_name` (subsystem 08, `llm.py`) replaces chars outside
    `[a-zA-Z0-9_-]` with `_`, truncates to 64, strips leading/trailing `_`,
    returns `None` when empty. So `"Ada Lovelace!"` → `"Ada_Lovelace"`; an
    all-punctuation display name falls back to the numeric id.
  - `all_known_names -> set[str]` = aliases ∪ {display_name if set} ∪
    {username if set}.
  - `from_discord_member(member)` (classmethod) — duck-typed over a minimal
    Protocol (`id: int`, `name: str`, `display_name: str`); `global_name`,
    `nick`, `pronouns`, `bio` read via `getattr(..., None)` so DM `User`
    objects / older py-cord / `SimpleNamespace` test doubles work.
    `user_id = str(member.id)`, `guild_nick = member.nick`.
  - `from_twitch(*, user_id, user_login, user_name)` (classmethod) —
    `username = user_login` (immutable lowercase login),
    `display_name = user_name`.
- `EGO_PLATFORM = "ego"` — reserved platform for the familiar's own narrative
  subject; can never collide with real `discord:`/`twitch:` keys.
- `ego_canonical_key(familiar_id: str) -> str` = `f"ego:{familiar_id}"`.
- `is_ego_key(canonical_key: str) -> bool` — `partition(":")`: platform part
  == `"ego"`, separator present, remainder non-empty. `"ego"` (no colon) is
  NOT an ego key; `"ego:"` is not either.
- `format_turn_for_transcript(role, author, content) -> str` — user turns with
  an author render `"{role} ({author.label}): {content}"`; everything else
  `"{role}: {content}"`. Shared by the summary provider (05) and memory writer
  (07); the format is a cross-module contract — changing it invalidates stored
  summaries' framing.

### familiar.py

- `Familiar` (mutable dataclass) — the DI bundle handed to nearly everything:
  `id: str`, `root: Path`, `config: CharacterConfig`,
  `history_store: AsyncHistoryStore` (03), `llm_clients: dict[str, LLMClient]`
  keyed by slot name (08), `tts_client: ... | None` (09),
  `transcriber: Transcriber | None` (09), `subscriptions: SubscriptionRegistry`,
  `bus: EventBus = InProcessEventBus()` (01), `router: TurnRouter = TurnRouter()`
  (01), `bot_user_id: int | None = None`,
  `local_turn_detector: LocalTurnDetector | None = None` (09).
  `transcriber=None` means voice subscriptions join for TTS playback only.
- `Familiar.display_name -> str` — first configured alias if any, else
  `id.title()` (Python str.title semantics on the folder name; ids are simple
  lowercase words in practice, `"sapphire"` → `"Sapphire"`).
- `Familiar.load_from_disk(root, *, llm_clients, tts_client=None,
  transcriber=None, local_turn_detector=None, defaults_path=None) -> Familiar`
  — sole constructor. `id = root.name`; `defaults_path` defaults to
  `root.parent / "_default" / "character.toml"`; loads config; opens/creates
  `root / "history.db"` (side effect: DB file exists after load); constructs
  `SubscriptionRegistry(root / "subscriptions.toml")`.

### subscriptions.py

- `SubscriptionKind` enum: `text = "text"`, `voice = "voice"`.
- `Subscription` (frozen dataclass): `channel_id: int`,
  `kind: SubscriptionKind`, `guild_id: int | None = None`.
- `SubscriptionRegistry(path: Path)` — loads sidecar on construction (missing
  file → empty registry). Methods:
  - `all() -> Iterable[Subscription]` (snapshot list).
  - `get(*, channel_id, kind) -> Subscription | None`.
  - `kind_for(channel_id) -> SubscriptionKind | None` — checks text first,
    then voice (enum declaration order); returns the first kind subscribed.
  - `voice_in_guild(guild_id) -> Subscription | None` — first voice row whose
    `guild_id` matches; at most one exists per guild by convention
    (discord.VoiceClient constraint), not enforced here.
  - `add(*, channel_id, kind, guild_id, persist=True) -> Subscription` —
    upsert on `(channel_id, kind)`; re-add updates `guild_id`.
    `persist=False` registers an *ephemeral* row (in-memory only).
  - `remove(*, channel_id, kind) -> None` — no-op if absent; only saves when
    something was actually removed.

### prompt_fill.py

- `fill_placeholders(template, /, **values) -> str` — replace each `{key}`
  (`key` matching `\w+`) present in `values` with `str(value)`; unknown tokens
  and stray braces pass through verbatim; never raises; single pass over the
  original template so injected values are never re-scanned
  (`fill("{a} {b}", a="{b}", b="X")` == `"{b} X"`). Consumers: sleep passes
  and fact extractor fill `{self_name}` / `{self_key}` / `{ids}` into the
  config-sourced prompt strings.

### structured_output.py

- `JsonResult` (frozen dataclass): `value: Any = None`,
  `parsed_ok: bool = False`. The bool keeps "model fumbled the JSON"
  distinguishable from "model returned empty object/array".
- `coerce_json(reply: str, *, expect: "object"|"array"|"any") -> JsonResult`
  — never raises. Pipeline: empty/whitespace reply → failure; strip every
  ```` ``` ````/```` ```json ```` fence token (case-insensitive regex
  substitution, not paired-fence matching); extract blob by shape:
  greedy-DOTALL `\{.*\}` for object, `\[.*\]` for array; for `"any"` search
  both and take whichever match *starts* earlier (tie → object); when no
  shape match, hand the whole cleaned text to the JSON parser verbatim (this
  is how bare scalars/pre-trimmed payloads parse); `json.loads`; any parse
  error → failure.
- `coerce_positive_int_list(raw) -> list[int]` — non-list → `[]`; bools
  rejected outright (Python `True == 1` hazard — in Rust, JSON bools must
  not coerce to ints); ints kept; strings kept only when the *entire trimmed
  string* matches `-?\d+` (so `"--5"`, `"3-"`, `"1.5"` drop); keep only
  `> 0`; de-dup preserving first-occurrence order.
- `coerce_str_list(raw) -> list[str]` — non-list → `[]`; keep non-empty
  (after strip check — but the *original* string is kept, not the stripped
  one), de-dup by exact string, order-preserving.

None of these are Protocol/ABC seams. The one duck-typed seam is
`_DiscordMemberLike` (Protocol) consumed by `Author.from_discord_member`. The
registries consulted for validation (`known_projectors()`,
`known_embedders()`) are swappable-by-registration seams owned by subsystems
07 and 04 respectively.

## Behaviors & invariants

### Loading & deep-merge

1. `load_character_config` fails with `ConfigError` ("default character
   profile not found at …") when `defaults_path` doesn't exist. A missing
   *target* file is fine (treated as `{}` — defaults-only install).
2. TOML parse errors in either file surface as `ConfigError`
   ("failed to parse TOML config at {path}: …"), never as raw
   `tomllib.TOMLDecodeError`.
3. Deep-merge semantics (`_deep_merge(base, override)`): recursive only when
   BOTH sides are tables; any non-table override value (including a list)
   replaces the base value wholesale — lists are never element-merged. Keys
   only in override are appended. Neither input is mutated. Base-key order
   then override-only-key order is preserved (irrelevant to parsing, but keep
   merge pure).
4. All parsing happens on the *merged* dict. Consequence: shipped defaults in
   `_default/character.toml` behave as if typed by the user, so validation
   applies to them too, and dataclass-level Python defaults only matter when
   the merged TOML omits a key entirely (pinned:
   `test_history_window_fallbacks_match_dataclass_defaults` loads with an
   *empty* defaults file and expects dataclass defaults).
5. The default profile is the single source of truth for prompt prose: the
   Python-side defaults for `[prompt]` strings are `""`, and tests assert the
   sleep-prompt prose constants do NOT exist in code
   (`test_no_in_code_sleep_prompt_prose_constants`) and that the merged
   default profile carries the real prose ("memory-consolidation pass",
   "stance-moment", "settled opinions", "dream narration" substrings). The
   Rust port must likewise keep zero in-code copies of that prose.

### Validation discipline (applies across all sections)

6. Every section that is a table is type-checked: a non-table value where a
   table is expected raises `ConfigError` `"[section] must be a table, got
   {typename}"`.
7. Unknown-key rejection is per-section and exact: `[discord.text]`,
   `[prompt]`, `[focus]`, `[tools]`, `[sleep]`, `[budget.<tier>]`,
   `[budget.model_curves.<model>]`, `[memory.retrieval]`,
   `[providers.memory]`, `[providers.embedding]` all reject unknown keys with
   `"... has unknown keys: a, b"` (sorted, comma-joined). Sections WITHOUT
   unknown-key rejection (unknown keys silently ignored): `[tts]`,
   `[providers.stt.*]`, `[providers.turn_detection.local]`,
   `[channels.<id>]` (pinned: `test_channel_total_tokens_key_ignored`), and
   the top level itself.
8. Bool-is-not-int: everywhere an integer or float is validated, `bool` is
   explicitly rejected first (Python `isinstance(True, int)` is true). In
   Rust this comes free from the TOML value enum, but error messages must
   still call out the received type name.
9. Numeric coercion: floats accept TOML ints (`5` → `5.0`); ints never accept
   floats. Positive means `> 0`; non-negative means `>= 0` (only
   `[memory.retrieval]` weights, `coalesce_max_gap_seconds`, and
   `text_silence_gap_fold_seconds` are non-negative-allowing-zero).
10. Everything in `CharacterConfig` and its sub-configs is immutable after
    load (frozen dataclasses; lists `aliases`/`greetings` are the exception in
    Python — treat as immutable in Rust).

### Section-specific rules

11. `display_tz` must be a valid IANA zone (validated by constructing
    `ZoneInfo`); `"PST"` rejected, `"America/Los_Angeles"` accepted. Stored
    as the string, not the zone object.
12. `[sleep]`: absent table → `(window=None, grace=30)` (schedule disarmed).
    `window` parsed by `parse_hhmm_range` (rule set in API section; wrapping
    midnight like `"22:30-06:15"` is legal and preserved as
    `(time(22,30), time(6,15))`). `grace_minutes` positive int, default 30.
13. `[llm]`: two *shared* scalar keys live at table level and are split out
    before slot parsing — `image_description_model` (string, default `""` =
    feature disabled) and `max_concurrent_requests` (positive int, default 4).
    Any other key under `[llm]` must be one of exactly `fast`, `prose`,
    `background`; unknown slot names (including retired `main_prose`) raise
    `"unknown LLM slot {name!r}; valid slots: background, fast, prose"`.
14. `[llm.<slot>]`: `model` required non-empty string (no default —
    a slot table without `model` fails). `temperature` optional, range
    [0, 2]. `top_p` optional [0, 1]; `presence_penalty` optional [-2, 2];
    `top_k` optional int ≥ 1. `think_prepend`, `provider_allow_fallbacks`
    (default true), `tool_calling`, `image_tools`, `multimodal` strict bools.
    `provider_order` optional list of non-empty strings → tuple; `None` when
    omitted. `reasoning` optional string in `REASONING_LEVELS`; the value
    `"default"` is a *sentinel that normalizes to `None`* — its purpose is to
    let a per-familiar file reclaim the model default over a level merged in
    from `_default` (TOML has no null). Pinned by
    `test_reasoning_default_sentinel_overrides_merged_value`.
15. `CharacterConfig.llm` is a plain map: slots present only if the merged
    TOML defines them. The shipped default profile defines all three, so a
    production config always has all three; `CharacterConfig()` (no TOML) has
    `{}`.
16. `[budget]`: tier keys must be in `BUDGET_TIER_NAMES`; each
    `[budget.<tier>]` merges over the *dataclass-default* `TierBudget`
    (voice-shaped) — but because parsing happens post-merge, the shipped
    profile's full tier tables mean production overrides layer over shipped
    values (pinned: `test_partial_override_keeps_other_subcaps`). All 12
    fields positive ints. `total_tokens` is NOT a key (rejected as unknown) —
    it is a derived sum on `TierBudget` (subsystem 05's `budget.py`).
    Missing tiers get the dataclass default. Shipped defaults: voice
    3000/900/900/600/600/600 + 200/10/6/16/6/12; text
    8000/2400/2400/1600/1600/1600 + 400/16/10/24/10/20; background
    24000/8000/8000/4000/4000/4000 + 1000/24/16/32/16/32 (field order:
    recent_history/rag/dossier/summary/reflection/lorebook tokens, then
    max_history_turns/max_rag_turns/max_rag_facts/max_dossier_people/
    max_reflections/max_lorebook_entries).
17. `[budget.model_curves.<model>]`: model name is an arbitrary table key
    (contains `/` in practice, e.g. `"anthropic/claude-haiku-4.5"` — TOML
    quoted key). Fields: same 12 names as TierBudget, each a positive float
    multiplier (ints accepted); unset fields default 1.0; `total_tokens`
    rejected. `apply_curve` scales each int field as
    `max(1, round(base * multiplier))` (Python banker's rounding —
    `round()` half-to-even; use the same in Rust:
    `(x - x.floor() == 0.5)` cases matter only in synthetic tests, but
    `f64::round` rounds half-away-from-zero — match Python or accept a
    documented deviation; safest is `round_ties_even`).
18. `budget_for` curve lookup keys on the *model string of the tier's slot*:
    tier→slot map is `voice→fast`, `text→prose`, `background→background`. No
    slot configured, or no curve for that model → base budget unchanged.
19. `[providers.history]`: retired key `window_size` rejected with a
    migration message ("has been split into voice_window_size and
    text_window_size"). `voice_window_size` (default 100) /
    `text_window_size` (default 200) positive ints.
    `coalesce_max_gap_seconds` ≥ 0 float, default 45.0 (0 disables).
    `text_silence_gap_fold_seconds` ≥ 0 float, default 0.0 (disabled).
20. `[providers.turn_detection]`: `strategy` ∈ {"deepgram",
    "ten+smart_turn"}, default "deepgram". `[providers.turn_detection.local]`
    knobs always parsed (regardless of strategy): two non-empty strings, three
    ints (`silence_ms` 200, `speech_start_ms` 100, `vad_hop_size` 256 — note:
    *not* positivity-checked, only int-checked), three floats
    (`vad_threshold` 0.5, `smart_turn_threshold` 0.5, `idle_fallback_s` 1.5 —
    also not range-checked).
21. `[providers.stt]`: `backend` ∈ {"deepgram", "parakeet",
    "faster_whisper"}, default "deepgram". All three backend sub-tables are
    parsed unconditionally (operators can flip `backend` without retyping),
    each with typed knobs and defaults per the dataclasses; strings must be
    non-empty; numbers type-checked but not range-checked. Deepgram defaults:
    model "nova-3", language "en", endpointing_ms 500, utterance_end_ms 1500,
    smart_format true, punctuate true, keyterms () (list of strings → tuple),
    replay_buffer_s 5.0, keepalive_interval_s 3.0, reconnect_max_attempts 5,
    reconnect_backoff_cap_s 16.0, idle_close_s 30.0.
22. `[providers.memory]`: `projectors` list of strings; each name validated
    against the live registry `known_projectors()` (07) via *deferred import*
    — unknown name → `ConfigError` listing valid names sorted (or "(none)").
    When `projectors` omitted, keep the dataclass default tuple
    `("rolling_summary","rich_note","people_dossier","reflection",
    "fact_supersede")` — which is asserted equal to the registry's
    `DEFAULT_PROJECTORS`. Empty list is legal (disables all projection). The
    five worker knob tables are parsed whether or not their projector is
    listed; knob names/int-vs-float kinds are *derived by reflection from the
    defaults dataclass fields* in Python — in Rust enumerate them explicitly;
    all values positive; unknown knob and unknown sub-table names rejected.
    Worker defaults: rolling_summary (turns_threshold 10, tick_interval_s
    5.0); rich_note (batch_size 10, tick_interval_s 15.0, participants_max
    30); people_dossier (tick_interval_s 20.0); reflection (turns_threshold
    20, max_reflections_per_tick 3, max_turns_per_tick 50, recent_facts_limit
    20, tick_interval_s 60.0); fact_supersede (batch_size 5, tick_interval_s
    60.0, priors_max 20). These are pinned to match "legacy hardcodes"
    (`test_dataclass_defaults_match_legacy_hardcodes`).
23. `[providers.embedding]`: `backend` validated against `known_embedders()`
    (04, deferred import); shipped/default "off". `dim` positive int, default
    256. `fastembed_model` non-empty string, default "BAAI/bge-small-en-v1.5".
    `fastembed_cache_dir`: `None` stays `None`, `""` normalizes to `None`,
    other strings kept, non-string rejected.
24. `[memory.retrieval]`: four float weights, non-negative (0 allowed),
    defaults bm25 1.0 / recency 0.0 / importance 0.0 / embedding 0.0
    (dataclass = pre-M2 BM25-only). Shipped profile differs: importance_weight
    = 0.6 — tests pin BOTH (dataclass default vs merged-shipped default).
25. `[channels.<id>]`: table keys are strings (TOML restriction) coerced via
    `int(key)`; non-numeric key → `ConfigError`. `history_window_size`
    optional positive int; `prompt_layers` optional list of strings → tuple;
    `message_rendering` optional ∈ {"prefixed", "name_only"}. Unknown keys
    ignored. Absent `[channels]` → empty map.
26. `[discord].dm_allowlist`: list of ints (bools rejected) → tuple, default
    empty (= engage no DMs). `[discord.text]`: `respond_to_typing` strict
    bool (default true); `typing_backoff_initial_s` / `typing_backoff_max_s`
    positive floats (defaults 1.0 / 30.0) with cross-field rule
    `max_s >= initial_s` (violation message: "typing_backoff_max_s must
    be >= typing_backoff_initial_s").
27. `[prompt]`: exactly six known string fields (`post_history_instructions`,
    `image_description_constraints`, `sleep_consolidation_system`,
    `sleep_stance_system`, `sleep_synthesis_system`,
    `dream_extraction_clause`); each read as string and **stripped** of
    surrounding whitespace; absent → `""`.
28. `[tts]`: `provider` ∈ {"azure", "cartesia", "gemini"}, default "azure".
    `azure_voice` / `gemini_voice` / `gemini_model` non-empty strings with
    the module-level defaults; `cartesia_voice_id` / `cartesia_model`
    optional strings (may be `None`); the six optional gemini_* strings
    (`scene`, `context`, `audio_profile`, `style`, `pace`, `accent`)
    normalize `""`→`None` (`val or None`). `greetings` list; elements
    stringified via `str()` (not type-rejected).
29. `[focus]`: `unread_nudge_enabled` strict bool (true),
    `nudge_debounce_seconds` positive float (30.0), `catch_up_limit`
    positive int (20). `[tools]`: `loop_max_iterations` positive int (5).
30. `aliases`: top-level list; elements stringified via `str()`; non-list
    rejected.

### SubscriptionRegistry semantics

31. Constructor loads the sidecar; a missing file, a non-list
    `subscription` value, or absent key yields an empty registry.
    Row-level tolerance on load: non-table rows skipped; rows missing/with
    wrong-typed `channel_id` (must be int) or `kind` (must be string)
    skipped; unknown `kind` values skipped; `guild_id` kept only when int,
    else `None`. Load never raises for content problems (a hand-edit must
    not brick startup) — but a TOML syntax error DOES propagate
    (`tomllib.load` raises; no catch). Preserve that asymmetry or document
    the deviation.
32. Rows are keyed by `(channel_id, kind)` — text and voice for the same
    channel coexist. `add` is an idempotent upsert.
33. Ephemeral rows (`persist=False`): queryable via `get`/`all`/`kind_for`
    like any row, never written to disk — including when a *later* persisted
    mutation rewrites the file. A later `persist=True` add of the same key
    promotes it (removes from the ephemeral set, then saves). `remove` of
    any row also clears its ephemeral mark. Pinned by four tests in
    `TestEphemeralSubscriptions`.
34. Every persisting mutation rewrites the entire file synchronously
    (registry is tens of rows; no append log, no atomic-rename — a plain
    `write_text`). Parent directories are created (`mkdir -p`).
35. On-disk ordering: rows sorted by `(channel_id, kind.value)` — stable,
    hand-diff-friendly output.
36. Concurrency: the registry has no locking; in Python it is only touched
    from the single event-loop thread. In Rust, either keep it
    single-threaded-owned or add a mutex; file writes must not interleave.

### prompt_fill / structured_output

37. `fill_placeholders` must be a single-pass regex substitution over the
    original template — see API section for the re-expansion pin. Value
    conversion is `str(value)` (Rust: `Display`).
38. `coerce_json` shape selection: with `expect="object"` a reply containing
    a *leading array and a later object* must return the object (and vice
    versa for `"array"`); with `"any"`, first-starting blob wins. Because
    the regexes are greedy `.*` spans from first opening to LAST closing
    bracket of that type, a reply like `'["b"] and then {...}'` under
    `expect="array"` would span `["b"] and then {` … up to any later `]` —
    the tests only exercise shapes where this resolves correctly; port the
    greedy-span semantics literally, not a "proper" bracket matcher, or you
    will diverge on prose containing brackets.
39. Fence stripping removes fence *tokens* anywhere (```` ``` ```` and
    ```` ```json ````, case-insensitive), not paired blocks.
40. All three coercion functions and `fill_placeholders` never raise/panic on
    any input.

### Async/concurrency summary

41. Nothing in this subsystem spawns tasks, holds async locks, or awaits.
    `load_character_config` and `Familiar.load_from_disk` run once at startup
    on the main thread before the event loop does real work. All products
    are immutable except `Familiar` (mutable fields `bot_user_id`, and the
    injected clients set at construction) and `SubscriptionRegistry`
    (mutated by slash-command handlers, subsystem 10, on the loop thread).

## Data formats

### character.toml (input; deep-merged pair of files)

Full schema per the Config knobs table below. Two files:
`data/familiars/_default/character.toml` (required, checked-in, 539 lines)
and `data/familiars/<FAMILIAR_ID>/character.toml` (optional overrides).
Notable format facts:
- Channel override tables use string keys of snowflakes:
  `[channels.123456789]`.
- Model-curve tables use quoted model names:
  `[budget.model_curves."anthropic/claude-haiku-4.5"]`.
- `[sleep].window` / activity `active_hours` use `"HH:MM-HH:MM"` strings.
- Shipped `[llm.fast]` = anthropic/claude-haiku-4.5, temp 0.7,
  reasoning "off", tool_calling false; `[llm.prose]` and `[llm.background]` =
  z-ai/glm-5.2, temp 0.7, provider_order ["z-ai"], reasoning "medium";
  background has tool_calling true.

### subscriptions.toml (read/write sidecar)

Written by `SubscriptionRegistry._save`, byte-exact format:

```
# Persistent subscription registry.
# Managed by /subscribe-* slash commands; safe to hand-edit while the bot is stopped.

[[subscription]]
channel_id = 42
kind = "text"
guild_id = 999
```

Header comment block, then one `[[subscription]]` array-of-tables entry per
persisted row in `(channel_id, kind.value)` order; `guild_id` line omitted
when `None`; a blank line after each row (rows are joined such that the file
ends with a trailing newline after the last row's blank-line separator —
match by writing each row block followed by "\n"). Round-trip through the
tolerant loader is the real contract; the exact bytes matter only for
hand-editing ergonomics and git diffs.

### Derived string formats

- `canonical_key`: `<platform>:<user_id>` — storage key in history DB (03)
  and memory files. `ego:<familiar_id>` reserved.
- `slug`: see API — filesystem name for `people/<slug>.md`.
- Transcript line: `user (Label): content` / `assistant: content`.

## Config knobs

Environment variables (this subsystem's docs own the model; the *reads*
happen in other modules): `FAMILIAR_ID` (selects `data/familiars/<id>/`; read
in `commands/run.py`, CLI arg wins), `DISCORD_BOT`, `OPENROUTER_API_KEY`,
`DEEPGRAM_API_KEY`, `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`,
`CARTESIA_API_KEY`, `GOOGLE_API_KEY` (fallback `GEMINI_API_KEY`). Rule:
env = secrets + install selector only; every behavioral knob lives in TOML.
`config.py` itself reads NO env vars.

TOML keys (defaults = Python dataclass fallback; "shipped" = value in
`_default/character.toml` where different):

| Key | Type / range | Default (shipped if different) |
|---|---|---|
| `display_tz` | IANA string | "UTC" |
| `aliases` | list of str | [] |
| `[sleep].window` | "HH:MM-HH:MM", wrap ok, start≠end | absent (disarmed) |
| `[sleep].grace_minutes` | int > 0 | 30 |
| `[providers.history].voice_window_size` | int > 0 | 100 |
| `[providers.history].text_window_size` | int > 0 | 200 |
| `[providers.history].coalesce_max_gap_seconds` | float ≥ 0 | 45.0 |
| `[providers.history].text_silence_gap_fold_seconds` | float ≥ 0 | 0.0 |
| `[providers.history].window_size` | — | REJECTED (retired) |
| `[budget.<voice/text/background>].*` | 12 pos-int caps | see Behaviors #16 |
| `[budget.model_curves.<model>].*` | 12 pos-float multipliers | 1.0 each |
| `[memory.retrieval].bm25_weight` | float ≥ 0 | 1.0 |
| `[memory.retrieval].recency_weight` | float ≥ 0 | 0.0 |
| `[memory.retrieval].importance_weight` | float ≥ 0 | 0.0 (shipped 0.6) |
| `[memory.retrieval].embedding_weight` | float ≥ 0 | 0.0 |
| `[providers.memory].projectors` | list of registered names | 5 defaults |
| `[providers.memory.<worker>].*` | pos ints/floats | see Behaviors #22 |
| `[providers.embedding].backend` | registered name | "off" |
| `[providers.embedding].dim` | int > 0 | 256 |
| `[providers.embedding].fastembed_model` | non-empty str | "BAAI/bge-small-en-v1.5" |
| `[providers.embedding].fastembed_cache_dir` | str or absent | None ("" → None) |
| `[providers.turn_detection].strategy` | "deepgram"\|"ten+smart_turn" | "deepgram" |
| `[providers.turn_detection.local].*` | see Behaviors #20 | field-tested V1 values |
| `[providers.stt].backend` | "deepgram"\|"parakeet"\|"faster_whisper" | "deepgram" |
| `[providers.stt.deepgram].*` | see Behaviors #21 | nova-3 etc. |
| `[providers.stt.parakeet].*` | model_name/device/idle_close_s | nvidia/parakeet-tdt-0.6b-v3, "auto", 30.0 |
| `[providers.stt.faster_whisper].*` | model_size/device/compute_type/language/idle_close_s | "small","auto","auto","en",30.0 |
| `[llm].image_description_model` | str | "" (disabled) |
| `[llm].max_concurrent_requests` | int > 0 | 4 |
| `[llm.<fast/prose/background>].model` | non-empty str | required per slot |
| `…temperature` | float in [0,2] or absent | None |
| `…top_p` | float in [0,1] or absent | None |
| `…top_k` | int ≥ 1 or absent | None |
| `…presence_penalty` | float in [-2,2] or absent | None |
| `…think_prepend` | bool | false |
| `…provider_order` | list of non-empty str or absent | None |
| `…provider_allow_fallbacks` | bool | true |
| `…reasoning` | off/none/low/medium/high/default | None ("default"→None) |
| `…tool_calling` / `image_tools` / `multimodal` | bool | false |
| `[tts].provider` | azure\|cartesia\|gemini | "azure" |
| `[tts].azure_voice` | non-empty str | "en-US-AmberNeural" |
| `[tts].cartesia_voice_id` / `cartesia_model` | str or absent | None (shipped values exist) |
| `[tts].gemini_voice` / `gemini_model` | non-empty str | "Kore" / "gemini-3.1-flash-tts-preview" |
| `[tts].gemini_{scene,context,audio_profile,style,pace,accent}` | str, ""→None | None |
| `[tts].greetings` | list (stringified) | [] |
| `[channels.<id>].history_window_size` | int > 0 or absent | None |
| `[channels.<id>].prompt_layers` | list of str or absent | None |
| `[channels.<id>].message_rendering` | "prefixed"\|"name_only" or absent | None |
| `[discord].dm_allowlist` | list of int | () |
| `[discord.text].respond_to_typing` | bool | true |
| `[discord.text].typing_backoff_initial_s` | float > 0 | 1.0 |
| `[discord.text].typing_backoff_max_s` | float ≥ initial | 30.0 |
| `[prompt].<6 fields>` | str, stripped | "" (shipped prose in `_default`) |
| `[focus].unread_nudge_enabled` | bool | true |
| `[focus].nudge_debounce_seconds` | float > 0 | 30.0 |
| `[focus].catch_up_limit` | int > 0 | 20 |
| `[tools].loop_max_iterations` | int > 0 | 5 |

## Dependency edges

Imports FROM other subsystems (runtime unless noted):

- `familiar_connect.budget` (`TierBudget`, `ModelBudgetCurve`) — file lives at
  package root specifically to avoid a config→context circular import, but it
  is context-assembly's budget model → subsystem **05**. The Rust workspace
  must break this the same way: the budget types crate must not depend on the
  config crate (config depends on it).
- `familiar_connect.llm.sanitize_name` (identity.py) → **08**.
- `familiar_connect.processors.projectors.known_projectors` — *deferred/lazy*
  import inside `_parse_memory_providers` → **07**.
- `familiar_connect.embedding.factory.known_embedders` — deferred import
  inside `_parse_embedding_config` → **04**.
- `familiar_connect.bus` (`InProcessEventBus`, `TurnRouter`; `EventBus`
  protocol type-only) (familiar.py) → **01**.
- `familiar_connect.history.async_store` / `history.store` (familiar.py) →
  **03**.
- Type-only (TYPE_CHECKING) in familiar.py: `llm.LLMClient` (**08**),
  `stt.Transcriber`, `tts.*TTSClient`, `voice.turn_detection.LocalTurnDetector`
  (**09**). In Rust these become generics/trait objects; no crate dependency
  on concrete impls is required if traits live in a shared interface crate.

Imported BY (consumers): effectively everything. `config` types are consumed
by 09 (stt/tts/turn_detection/voice), 05 (context layers, budget_for), 06
(responders — slot configs, discord_text, tools, focus), 07 (worker knob
structs), 04 (embedding + sleep configs + prompt strings), 10
(bot.py/run.py — dm_allowlist, subscriptions, Familiar), 11
(twitch/activities — `parse_hhmm_range`, aliases). `identity.Author`
threads through 03 (history rows), 05, 06, 07, 10, 11.
`structured_output` is consumed by `structured_request` (08), sleep passes
(04), fact_supersede (07). `prompt_fill` by 04 sleep passes and 07
fact_extractor. `subscriptions` by 10 (bot, run), and `focus` (05/06).

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| tests/test_config.py | ~160 tests: deep-merge over `_default`; every section's defaults, overrides, unknown-key/type/range rejections listed in Behaviors #6–#30; shipped-default values (budgets, retrieval weights, projector list, LLM slots); `budget_for` curve application; `for_channel`/`*_window_for`; reasoning "default" sentinel; retired-key rejections (`window_size`, `main_prose`); no-in-code sleep prose | logic-portable (fixture: real `_default/character.toml` path + tmp dirs; two tests reflect on Python modules for absent constants — Python-specific-skip those two) |
| tests/test_identity.py | Author construction/frozen-ness; canonical_key; slug normalization (lowercase, collapse, trim); label fallback chain; openai_name sanitize + id fallback; all_known_names; from_discord_member incl. getattr-optional fields; from_twitch; ego key build/membership/non-collision | logic-portable (SimpleNamespace doubles → Rust struct impls of the member trait) |
| tests/test_familiar.py | load_from_disk builds bundle; id=root name; config loaded; history.db created as side effect; tts/transcriber default None; llm_clients keyed by slot | needs-Rust-mock (fake LLM clients; real tmp-dir fs) |
| tests/test_subscriptions.py | empty registry; add/get/upsert-idempotent; remove noop; text+voice coexist; voice_in_guild; persistence file created on add; reload round-trip; remove persists; ephemeral rows queryable-not-written, excluded from later saves, promotable, cleared correctly | logic-portable (tmp dirs only) |
| tests/test_prompt_fill.py | known fill; unknown token pass-through; missing key pass-through; stray braces; single-pass no re-expansion | logic-portable |
| tests/test_structured_output.py | fenced object/array parse; garbage degrades under every expect; `{}` parsed_ok=true; object-expect skips leading array; array-expect skips leading object; any=first-blob; int-list bool/dupe/negative/malformed-string handling; str-list dedup/blank-drop | logic-portable |
| tests/conftest.py | `default_profile_path` fixture → checked-in `_default/character.toml`; `build_fake_llm_clients` | needs-Rust-mock |

Also touching this subsystem indirectly: virtually every other test file
constructs `CharacterConfig(...)` or sub-configs directly with keyword
overrides — the Rust config types need an ergonomic all-defaults constructor
plus per-field override (builder or `..Default::default()` struct-update) to
keep those tests portable.

## Rust port notes

- **Crate layout**: make `config`+`identity` a leaf-ish crate. Two Python
  circular-import dodges must become explicit seams: (a) `budget.py` lives at
  package root so config can use `TierBudget` — in Rust put budget types in a
  core/types crate below both config and context-assembly; (b) the deferred
  imports of `known_projectors()`/`known_embedders()` mean config validation
  *reaches upward* into registries owned by 07/04. Options: pass validator
  callbacks/sets into `load_character_config` (preferred — makes the seam a
  parameter: `fn load(..., known_projectors: &BTreeSet<String>,
  known_embedders: &BTreeSet<String>)`), or a registration-based inventory
  (e.g. `inventory`/`linkme`). Do NOT transliterate the lazy import.
- **TOML**: use the `toml` crate. Deep-merge on `toml::Value` tables
  (recursive only when both sides are tables), then validate the merged
  value manually. Avoid serde-derive for the top-level parse: the error
  behavior (exact `ConfigError` messages, unknown-key policies differing per
  section, bool-not-int, "default" sentinel, ""→None normalizations,
  stringification of alias/greeting elements) is the contract, and serde's
  `deny_unknown_fields` cannot reproduce the per-section mix. Hand-rolled
  visitors over `toml::Value` match the Python structure nearly 1:1.
- **Error type**: single `ConfigError(String)` (thiserror). Tests match on
  substrings — preserve key phrases: "default character profile",
  "unknown LLM slot", "has unknown keys:", "must be a table", "must be a
  positive integer", "display_tz", "'HH:MM-HH:MM'", the `window_size`
  migration text, "unknown memory projector", "valid options:".
- **Timezone**: `chrono-tz` (`Tz::from_str`) replaces `ZoneInfo` validation;
  store the validated *string* to stay wire-compatible. `NaiveTime` for the
  sleep window.
- **Rounding**: `TierBudget::apply_curve` — Python `round()` is
  half-to-even; use `f64::round_ties_even` (stable since 1.77) inside
  `max(1, …)`.
- **Frozen dataclasses** → plain structs, no interior mutability; derive
  `Clone, Debug, PartialEq` (tests compare whole structs) and `Default`
  where Python has field defaults. `LLMSlotConfig` has no Default in
  spirit (`model` required) — but tests construct it with just `model`;
  provide `LLMSlotConfig::new(model)` or Default with empty model used only
  in tests.
- **Author**: `platform` as an enum {Discord, Twitch, Ego} — but note two
  identity tests construct Authors with arbitrary platform strings
  ("Discord", ":x:") purely to exercise slug normalization; either make
  slug a free function over `(platform_str, user_id)` tested directly, or
  accept those two assertions move to unit tests of the slug function.
  `from_discord_member`'s getattr-duck-typing becomes a
  `DiscordMemberLike` trait with `Option`-returning accessors implemented by
  the Discord shell (10) and by test doubles.
- **SubscriptionRegistry**: `BTreeMap<(u64, SubscriptionKind), Subscription>`
  gives the sorted save order for free (derive `Ord` on kind with
  text < voice — matches `kind.value` string ordering). Keep the
  tolerant-row/strict-syntax load asymmetry. Serialize by hand (or
  `toml_edit`) to hit the exact header + `[[subscription]]` layout.
  Wrap in whatever ownership the shell needs (`Mutex`/single-task owner);
  the Python version relies on event-loop single-threading.
- **regex ports**: slug `[^a-z0-9]+`→"-"; prompt_fill token `\{(\w+)\}` —
  Rust `regex` `\w` is Unicode-aware like Python's, fine; structured_output
  `\{.*\}` / `\[.*\]` need DOTALL (`(?s)`) and *greedy* semantics — port
  literally per Behaviors #38; fence regex `(?i)```(?:json)?`; int gate
  `^-?\d+$` via `Regex::is_match` on the trimmed string (use full-match
  anchoring — Python uses `fullmatch`). Beware Python `\d` here matches
  Unicode digits and `int()` would accept them; the `_INT_RE` gate +
  `int()` combination accepts e.g. Arabic-Indic digits in Python. Rust
  `str::parse::<i64>` won't. Use ASCII-only `\d` (`(?-u:\d)` or `[0-9]`)
  and accept the (untested, unobservable-in-practice) deviation.
- **structured_output value type**: `JsonResult.value` is dynamic —
  `Option<serde_json::Value>` + `parsed_ok: bool` (or just
  `Option<Value>`; keep the empty-object-vs-failure distinction: `Some
  (Value::Object(empty))` ≠ `None`).
- **`Familiar`**: in Rust this is the composition root's wiring struct;
  `llm_clients: HashMap<String, Arc<dyn LlmClient>>` keyed "fast"/"prose"/
  "background" (plus run.py's reserved `"__image_description__"` key —
  owned by 08/10 wiring but flows through this map). `bot_user_id` is set
  post-login by the shell — make it `OnceLock<u64>` or set during a
  mutable wiring phase.
- **Redesign candidates** (flag for reviewer sign-off, don't silently do):
  the reflection-driven `_worker_knobs` (enumerate fields explicitly); the
  `str()` stringification of aliases/greetings elements (consider strict
  string lists — deviation visible only for exotic TOML like ints in the
  list); `Familiar.display_name`'s `str.title()` (Rust: uppercase first
  char of each word — ids are single lowercase words, so a simple
  capitalize-first is adequate; document).
