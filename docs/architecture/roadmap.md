# Roadmap

Forward-looking work, ordered by approximate priority. Each item names
the lesson, the gap, and the change. Cross-references point at the
architecture pages already covering the underlying machinery.

When an item ships, fold its description into the matching page and
prune it here.

The 2026 self-hosted-AI-character survey distilled into:

- [Memory strategies](memory-strategies.md) — bi-temporal logs, rich
  notes, reflections, lorebooks; swap points.
- [Voice pipeline](voice-pipeline.md) — cascaded vs full-duplex,
  two-stage turn detection, sentence streaming, swap points.

Existing instinct (event-sourced `turns`, regenerable side-indices,
supersession over overwrite) already aligns with the rigorous end of
the field. Roadmap is about widening seams for A/B testing strategies
and closing the voice-latency gap to sub-1 s.

## Memory

### M2 — Importance-weighted retrieval (shipped)

`FactExtractor` emits a 1–10 importance hint per fact (prompt teaches
the scale: 1 = throwaway, 5 = ordinary, 10 = identity-defining /
safety-critical). Persisted on the `facts` table as a nullable
`importance INTEGER`; legacy rows read back as `None` and rank as
the neutral midpoint. `HistoryStore.search_facts_scored` exposes
BM25 verbatim so `RagContextLayer` can fuse three signals at rank
time:

```toml
[memory.retrieval]
bm25_weight       = 1.0
recency_weight    = 0.0
importance_weight = 0.6
embedding_weight  = 0.0   # M6 placeholder
```

Defaults reproduce pre-M2 ordering when `importance_weight = 0`;
the shipped default raises it to `0.6` so safety-critical facts
beat equally-matched casual ones. See
[Memory strategies — retrieval ranking](memory-strategies.md#3-retrieval-ranking-ragcontextlayer)
and [Tuning — retrieval ranking](tuning.md#retrieval-ranking-m2).

### M3 — Reflection layer (shipped)

`reflections` table with `cited_turn_ids` / `cited_fact_ids` (JSON
arrays) plus `last_turn_id` / `last_fact_id` watermarks snapshotting
the worker's view at write time. Each row is forever-provenance —
never edited, never trimmed.

`ReflectionWorker` (`src/familiar_connect/processors/reflection_worker.py`)
ticks every 60 s (slower than `PeopleDossierWorker`'s 20 s); fires
when at least `turns_threshold` (default 20) new turns have arrived
since the newest reflection. Asks the background-tier LLM "what
high-level questions do recent events raise?" and persists each
answer with at least one cited turn or fact id. Rows whose only
citations the LLM hallucinates are dropped silently; rows where
some ids are valid keep the valid subset.

`ReflectionLayer` renders the most recent rows scoped to the active
channel (channel-agnostic rows always surface) with citation
breadcrumbs `[T#42, F#7]`. Reflections that cite at least one
superseded fact are flagged `(stale)` on read — the row is never
deleted, per the supersession-over-overwrite rule.

Token + count caps live in `[budget.<tier>]` as `reflection_tokens`
and `max_reflections`. See [Memory strategies — reflections](memory-strategies.md#reflections-m3).

### M4 — Lorebook layer (shipped)

`data/familiars/<id>/lorebook.toml` carries `[[entries]]` with
`keys`, `content`, optional `priority` (default 0) and optional
`selective` (default `false`). `LorebookLayer` scans the active
channel's last `recent_window` turns case-insensitively against
each entry's keys; hits render newest-priority-first under a
`## Lorebook` block. `selective = true` flips the per-entry match
from any-key (OR) to all-keys (AND).

No worker — the file is the sole source of truth. Cache key
combines a BLAKE2b hash of the file with the matched entry indices,
so the layer only flips when the file or the activation set
changes.

Token + count caps live in `[budget.<tier>]` as `lorebook_tokens`
and `max_lorebook_entries`. See
[Memory strategies — lorebook](memory-strategies.md#lorebook-m4).

### M5 — Pluggable memory-store backend (shipped)

Existing watermark-driven writers (`SummaryWorker`, `FactExtractor`,
`PeopleDossierWorker`, `ReflectionWorker`) lifted behind a
`MemoryProjector` Protocol (`src/familiar_connect/processors/projectors.py`).
Operators select projectors via TOML:

```toml
[providers.memory]
projectors = ["rolling_summary", "rich_note", "people_dossier", "reflection"]
```

Names map to a registry (`register_projector`); third-party
projectors (Graphiti, Cognee) call `register_projector` at import
time and the same selector picks them up. Unknown names raise at
config load — a typo never silently drops a writer.

Default keeps every shipped projector. Empty list disables all
memory projection. See
[Memory strategies — swap points](memory-strategies.md#swap-points).

### M6 — Embeddings for semantic recall

Today: FTS5 keyword recall only. Misses paraphrase cues.

Change: embedding provider behind a Protocol; default to local
FastEmbed/ONNX. Vectors in `sqlite-vec` alongside FTS.
`RagContextLayer` fuses both. Same TOML weights as M2.

## Voice

### V1 — Local VAD + semantic turn detection

Shipped. TEN-VAD + Smart Turn v3 wrappers (phase 1), per-user
`UtteranceEndpointer` wired into the audio pump (phase 2), VAD-to-STT
telemetry (`voice.vad_to_stt`), TOML-driven strategy selector
(`[providers.turn_detection] strategy = "ten+smart_turn"`), and
audio-fixture integration coverage for complete-sentence / mid-thought /
filler endpointing patterns
(`tests/test_endpointer_audio_fixtures.py`) have all landed. Local VAD
saves 150–200 ms over remote endpointing.

See [Voice pipeline — turn detection](voice-pipeline.md#turn-detection),
[Per-turn budget telemetry](voice-pipeline.md#per-turn-budget-telemetry),
and [Tuning — local turn detection](tuning.md#local-turn-detection-v1).

### V3 — Pluggable transcriber backend (shipped)

Phase 1: `Transcriber` Protocol (`src/familiar_connect/stt/protocol.py`),
backend dispatch in `stt/factory.create_transcriber` keyed on
`[providers.stt].backend`. `DeepgramTranscriber` lives in
`stt/deepgram.py`; the rest of the voice pipeline (`bot.py`,
`sources/voice.py`, `familiar.py`) types against `Transcriber`.

Phase 2: `ParakeetTranscriber` (`stt/parakeet.py`) wraps NeMo
Parakeet-TDT 0.6B v3. Selected via `backend = "parakeet"`; install
the `local-stt-parakeet` extra (NeMo + torch).

Phase 3: `FasterWhisperTranscriber` (`stt/faster_whisper.py`) wraps
`faster-whisper` (CTranslate2). Same buffer-and-finalize shape as
Parakeet but lighter — no torch, ~150 MB for the `small` model.
Selected via `backend = "faster_whisper"`; install the
`local-stt-whisper` extra.

Both local backends share the buffer-and-finalize seam: the local
turn detector (V1) drives `finalize()`, so they require
`[providers.turn_detection].strategy = "ten+smart_turn"`.

### V4 — Pluggable TTS backend & Mimi-codec readiness

Already pluggable via `synthesize() → TTSResult`. Track here so the
lesson isn't lost: Mimi (Kyutai) is becoming the open audio-token
standard. When Sesame CSM stabilises, drop in a `SesameTTSClient`
behind the existing surface. `PiperTTSClient` is a smaller
intermediate step.

### V5 — Full-duplex / S2S as a research branch

Not adopting. Rejection captured in
[Decisions](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm).
Revisit if a Mimi-based S2S model gains an external-LLM-brain seam.

## Architecture / experimentation seams

### A1 — Strategy-swap configuration spine

`[providers.turn_detection]` selector shipped: `strategy = "deepgram"
| "ten+smart_turn"` in `character.toml` selects the endpointer chain;
its tuning lives in `[providers.turn_detection.local]`. See
[Tuning — local turn detection](tuning.md#local-turn-detection-v1).

`[providers.stt].backend` selector wired in V3 phase 1; phases 2 and 3
added `parakeet` and `faster_whisper`. V3 closed.

Env vars are restricted to secrets and deployment identity (API keys,
Discord token, `FAMILIAR_ID`); all behavior knobs live in TOML.

`[providers.memory]` selector wired in M5; built-in names today are
`rolling_summary`, `rich_note`, `people_dossier`, `reflection`.
Third-party projectors register at import time; the selector picks
them up by name.

Remaining selector (not yet wired — implementation doesn't exist):

```toml
[providers.voice_pipeline]
mode = "cascaded"             # | "s2s"                           # (V5)
```

### A2 — Dynamic context budgeter

Initial budgeter shipped: a per-tier `total_tokens` envelope drives
recent-history trimming and per-section caps for RAG, dossiers,
summary, and cross-channel context. Token accounting uses a fast
`len(text) / 4` heuristic (no real tokenizer on the hot path —
sub-microsecond per message).

What landed:

- `[budget.<tier>]` config blocks, one per assembly tier
  (`voice` / `text` / `background`); `total_tokens` is the operator's
  primary knob, sub-caps default to fixed fractions of it.
- :class:`familiar_connect.budget.Budgeter` enforces the total cap
  post-assembly by dropping oldest history turns first.
- Each dynamic layer accepts `max_tokens` and self-truncates while
  building.
- `voice_window_size` / `text_window_size` retained as hard
  upper bounds on history turns (safety net behind the token caps).
- `[channels.<id>].history_window_size` continues to override the
  per-channel turn cap.

What's still open (worth doing, but not blocking voice quality):

- "Natural" silence-gap boundaries in text channels — fold older
  turns up to a low-density boundary so prefixes stabilise for
  prompt caching.
- Model-specific context-degradation curves (today the same envelope
  applies regardless of model; frontier models could absorb more).
- Per-channel `total_tokens` overrides (today only the turn-count
  cap is per-channel).

## Out of scope

- **Letta / MemGPT-style core-memory edit tools** — destructive,
  conflicts with source-of-truth preservation. See
  [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).
- **Hosted memory services (Zep Cloud, mem0)** — fail local-first.
  See [Decisions](decisions.md#third-party-managed-memory-services).
- **Headless-browser SillyTavern bridge** — already rejected. See
  [Decisions](decisions.md#bridging-to-a-running-sillytavern-instance).
