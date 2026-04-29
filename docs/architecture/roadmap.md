# Roadmap

Forward-looking work, ordered by approximate priority. Each item lists
the lesson driving it, the current status in the codebase, and a sketch
of the change. Cross-references point at the architecture pages that
already cover the underlying machinery.

This page is a living document. When an item ships, move its description
into the relevant architecture page (the *what*) and prune the entry
here (the *why*).

## Background

The 2026 survey of self-hosted AI-character architectures (memory and
voice latency) is summarised in two new pages:

- [Memory strategies](memory-strategies.md) — bi-temporal logs, rich
  notes, reflections, lorebooks, and how to swap them.
- [Voice pipeline](voice-pipeline.md) — cascaded vs full-duplex,
  two-stage turn detection, sentence-level streaming, swap points.

The project's existing instinct (event-sourced `turns` table, side
indices regenerable from source-of-truth, supersession over overwrite)
already aligns with the rigorous end of the field. The roadmap below is
about widening the seams that already exist so individual strategies
can be A/B tested, and about closing the latency gap between the
current voice path and a sub-1 s target.

## Memory

### M1 — Bi-temporal facts (`t_valid_from` / `t_valid_to`)

**Lesson.** Graphiti's bi-temporal model (event time vs ingest time)
is the cleanest way to express "this fact was true at t1, superseded
at t2" without losing the historical record. The current `facts` table
has a single supersession marker; that captures *was-superseded* but
not *when the fact applied in the world*.

**Change.** Add `valid_from` / `valid_to` columns to `facts`, populated
by the extractor from the source turn's wall-clock timestamp by
default and overridable when the LLM extracts an explicit "as of"
phrase. Default reads (`recent_facts`, `search_facts`) keep the
current "current truth" filter; an `as_of` parameter lets a future
audit UI ask "what did we believe on date X?".

### M2 — Importance-weighted retrieval

**Lesson.** Park et al.'s generative-agents memory stream weights
retrieval on `recency × importance × similarity`, not similarity
alone. Importance is LLM-rated at write-time and stored. The cost is
one extra structured field per fact extraction; the upgrade in
"remembers what mattered" is dramatic.

**Change.** Extend the `FactExtractor` LLM prompt to emit a 1–10
importance score per fact. Persist alongside the fact. Update
`RagContextLayer` ranking to combine BM25, recency, and importance
(weights configurable in TOML — see [Tuning](tuning.md)).

### M3 — Reflection layer (with citations to source turns)

**Lesson.** Higher-order syntheses ("recurring tensions between Alice
and Bob", "the band's running joke about pho") are useful in the
prompt and not produced by either summaries (per-channel) or facts
(atomic observations). Generative agents call these *reflections* and
crucially keep them as appended rows that **cite** the underlying
observations, never overwriting them.

**Change.** New `reflections` table keyed on `(familiar_id,
reflection_id)` with `cited_turn_ids` and `cited_fact_ids` JSON arrays.
A `ReflectionWorker` ticks slower than the dossier worker, periodically
asks the LLM "what high-level questions do recent events raise?" and
writes one row per answer. A new `ReflectionLayer` reads them into the
prompt with their citations rendered as breadcrumbs. Reflections that
cite a now-superseded fact are flagged stale on read, not deleted.

### M4 — Lorebook layer (hand-authored canon)

**Lesson.** RisuAI / SillyTavern's keyword-activated lorebook is a
deterministic, inspectable, hand-authored RAG that works *better* than
embedding-based retrieval for stable worldbuilding. It belongs as a
**separate layer from experiential memory** because hand-authored
canon must never be subject to the agent's own evolution.

**Change.** Add `data/familiars/<id>/lorebook.toml` with
keyword-activated entries. Each entry has `keys` (trigger tokens),
`content` (text inserted), `priority` (insertion order weight), and
optional `selective` (require all keys vs any). New `LorebookLayer`
matches recent-history tokens against keys and inserts hits at the
declared priority. No worker; the file is the source of truth, edited
by hand.

### M5 — Pluggable memory-store backend

**Lesson.** Rich-note-with-emergent-links (A-MEM) and bi-temporal
graph (Graphiti) are different *projections* of the same event log.
The choice between them should be a config knob, not a fork.

**Change.** Introduce a `MemoryProjector` Protocol — given a stream of
new turns, write to whichever side-indices the projector maintains.
The current write paths (FTS triggers, `FactExtractor`,
`SummaryWorker`, `PeopleDossierWorker`) are recast as the default
"rich-note + summary" projector. A future `GraphitiProjector` (or
`CogneeProjector`) can run alongside or replace it. See
[Memory strategies — swap points](memory-strategies.md#swap-points).

### M6 — Embeddings for semantic recall

**Lesson.** Pure FTS5 keyword recall misses paraphrase cues ("What
did Aria order?" → "Aria likes pho"). The whole field has converged on
hybrid retrieval (BM25 + vector). Local options (FastEmbed, sqlite-vec,
DuckDB-VSS, ChromaDB) are now mature enough to keep the local-first
constraint.

**Change.** Add an embedding provider behind a Protocol. Default to
local FastEmbed/ONNX. Store vectors in `sqlite-vec` virtual table
alongside the existing FTS index. `RagContextLayer` runs both queries
and fuses scores. Same TOML knob set as M2 controls the fusion weights.

## Voice

### V1 — Local VAD + semantic turn detection

**Lesson.** Every production voice-agent stack (Pipecat, LiveKit,
TEN, RealtimeVoiceChat) has converged on **two-stage turn detection**:
a lightweight VAD detects speech vs silence, then a semantic turn
classifier decides whether the user is *done*. Pure-VAD endpointing
with long silence timeouts is now an anti-pattern. Local VAD beats
remote VAD by 150–200 ms.

**Change.** Add Silero VAD (ONNX) to the voice path, running locally
in the bot process. On VAD speech-end, run Pipecat's open-source
Smart Turn v3 ONNX model (BSD-2, ~12 ms inference, 360 MB) over the
buffered audio; emit `voice.activity.end` only when the turn is
classified `complete`. Deepgram's `endpointing_ms` /
`utterance_end_ms` fall back to today's behaviour when the local
classifier is disabled. See
[Voice pipeline — turn detection](voice-pipeline.md#turn-detection).

### V2 — Sentence-level TTS streaming

**Lesson.** Pipecat's `SentenceAggregator` flushes LLM output to TTS
at the first sentence boundary, not after the full response. This
single optimization is usually 1–3 s of perceived-latency win. Today
the `VoiceResponder` accumulates the whole reply before calling
`TTSPlayer.speak`.

**Change.** Introduce a `SentenceStreamer` between
`LLMClient.chat_stream` and `TTSPlayer.speak`. Buffer deltas, emit on
sentence boundaries (with abbreviation-aware splitting), feed each
sentence to TTS as soon as it's ready. The `<silent>` sentinel
detection moves to the streamer's first-sentence check. The TTS
player keeps its scope-cancellation contract; cancellation flushes
all in-flight sentences.

### V3 — Pluggable transcriber backend

**Lesson.** Faster-Whisper (CTranslate2) and NVIDIA Parakeet-TDT 0.6B
v3 are now competitive with cloud STT on latency and meaningfully
beat it on cost / privacy. Several stacks (Modal, Pipecat) have
benchmarked them favourably.

**Change.** Promote Deepgram's transcriber-as-template pattern into a
proper `Transcriber` Protocol. The `clone()` and queue-output shape
are already there; lift them out of `DeepgramTranscriber` and add a
`FasterWhisperTranscriber` implementation behind the same surface.
`[providers.stt]` in TOML selects which one.

### V4 — Pluggable TTS backend & Mimi-codec readiness

**Lesson.** Mimi (Kyutai) is becoming the lingua franca of next-gen
neural audio codecs; Sesame CSM-1B, Hibiki, and Moshi all use it. A
Mimi-based local TTS will be the natural drop-in for Cartesia once
Sesame's voice-stability story matures.

**Change.** No code change today — already pluggable. The roadmap
item is to add a `SesameTTSClient` (or `PiperTTSClient` as a smaller
intermediate step) when the upstream model is stable, behind the
existing `synthesize() -> TTSResult` interface. Tracked here so the
lesson isn't lost.

### V5 — Full-duplex / S2S as a future research branch

**Lesson.** Moshi (Kyutai) and Sesame CSM are architecturally
fascinating — 200 ms theoretical voice-to-voice latency, true
overlap-talk handling. They give up the LLM-swap-out flexibility we
get from OpenRouter; they pin you to the bundled brain. For a
character bot whose appeal is the LLM persona, that's too steep.

**Change.** **Not adopting today.** Captured in
[Decisions](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm)
so a future contributor doesn't rediscover the trade-off cold. If a
Mimi-based S2S model later supports an external LLM brain, revisit.

## Architecture / experimentation seams

### A1 — Strategy-swap configuration spine

**Lesson.** The codebase has good seams (`EventBus`, `Layer`,
`TTSPlayer`, transcriber-as-template, watermark workers) but the
*selection* of which strategy is wired today lives in Python
(`commands/run.py::_default_assembler` etc.). To make swap
experiments cheap, the wiring needs a config-driven entry point.

**Change.** Two new TOML sections:

```toml
[providers.stt]
backend = "deepgram"      # "deepgram" | "faster_whisper" | "parakeet"

[providers.turn_detection]
strategy = "deepgram"     # "deepgram" | "silero+smart_turn" | "ten"

[providers.memory]
projectors = ["rich_note", "people_dossier"]
# future: ["graphiti"], ["a_mem"]

[providers.voice_pipeline]
mode = "cascaded"         # "cascaded" | "s2s" (future, V5)
sentence_streaming = true # V2 toggle
```

`commands/run.py` reads these and selects the wiring; defaults preserve
today's behaviour. Consolidates with the
[Tuning](tuning.md) page so a single TOML file drives the whole bot.

### A2 — Consolidate STT env vars into TOML

**Lesson.** Today's Deepgram knobs (`DEEPGRAM_ENDPOINTING_MS`,
`DEEPGRAM_KEYTERMS`, etc.) live in env. That's appropriate for
secrets but awkward for tuning knobs that operators tweak per
character. The current model also makes per-channel overrides hard.

**Change.** Move non-secret Deepgram knobs into
`[providers.stt.deepgram]` in `character.toml`. Secrets
(`DEEPGRAM_API_KEY`) stay in env. Env vars become *overrides* for
container deployments, with TOML as the declarative source. See
[Tuning](tuning.md) for the planned schema.

### A3 — Diagnostics → metrics

**Lesson (existing).** The `@span` decorator emits timing logs today;
a metrics collector + `/diagnostics` slash command is already in the
plan (Phase 5). With V1+V2 landing, latency-budget instrumentation
becomes the most useful single signal — a per-turn breakdown
(`vad_end → stt_final → llm_first_token → tts_first_audio →
playback_start`) is what tells an operator whether their tuning is
working.

**Change.** Land the metrics collector. Expose a `/diagnostics
voice-budget` slash command that prints the last-N-turn latency
breakdown.

## Out of scope / not on the roadmap

- **Letta / MemGPT-style core-memory edit tools.** Destructive
  `core_memory_replace` conflicts with the project's source-of-truth
  preservation rule. See
  [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).
- **Hosted memory services (Zep Cloud, mem0).** Fail the local-first
  constraint. See
  [Decisions](decisions.md#third-party-managed-memory-services).
- **Headless-browser SillyTavern bridge.** Already rejected;
  unchanged. See
  [Decisions](decisions.md#bridging-to-a-running-sillytavern-instance).
