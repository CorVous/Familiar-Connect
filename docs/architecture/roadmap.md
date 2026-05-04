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

### M3 — Reflection layer

Today: summaries (per-channel) and facts (atomic). No higher-order
syntheses.

Change: `reflections` table with `cited_turn_ids` /
`cited_fact_ids`. `ReflectionWorker` ticks slower than the dossier
worker, asks "what high-level questions do recent events raise?",
writes one row per answer. `ReflectionLayer` renders citations as
breadcrumbs. Reflections citing superseded facts are flagged stale
on read, never deleted.

### M4 — Lorebook layer

Today: `core_instructions.md` + `character.md` are always-on. No
keyword-activated canon.

Change: `data/familiars/<id>/lorebook.toml` with entries carrying
`keys`, `content`, `priority`, optional `selective`. `LorebookLayer`
matches recent-history tokens against keys, inserts hits at the
declared priority. No worker — file is source of truth.

### M5 — Pluggable memory-store backend

Today: writers (`SummaryWorker`, `FactExtractor`,
`PeopleDossierWorker`, FTS triggers) collectively implement one
strategy.

Change: lift them behind a `MemoryProjector` Protocol. Defaults stay.
A `GraphitiProjector` or `CogneeProjector` plugs alongside or
replaces. Selected via `[providers.memory].projectors`. See
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

Remaining selectors (not yet wired — implementations don't exist):

```toml
[providers.memory]
projectors = ["rich_note", "people_dossier"]                      # (M5)

[providers.voice_pipeline]
mode = "cascaded"             # | "s2s"                           # (V5)
```

These will be wired when the corresponding backends land (M5).

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
