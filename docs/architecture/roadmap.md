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

### M1 — Bi-temporal facts (`valid_from` / `valid_to`)

Today: `facts` has `superseded_at` only. Captures *was-superseded*,
not *when the fact applied in the world*.

Change: add `valid_from` / `valid_to`, populated from the source
turn's timestamp by default, overridable when the LLM extracts an
explicit "as of" phrase. Default reads stay "current truth"; an
`as_of` parameter unlocks audit queries.

### M2 — Importance-weighted retrieval

Today: BM25 + recent-window exclusion.

Change: extend `FactExtractor` to emit a 1–10 importance score,
persist alongside the fact, fold into `RagContextLayer` ranking.
Weights configurable in TOML (see [Tuning](tuning.md)). One extra
field per extraction, large UX win.

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

Today: Deepgram's hosted endpointer (silence-based).

Change: Silero VAD (ONNX, MIT) in-process. On VAD silence, run
Pipecat's Smart Turn v3 (BSD-2, ~12 ms, 360 MB) over the buffered
audio; emit `voice.activity.end` on `complete`. Local VAD saves
150–200 ms over remote. See
[Voice pipeline — turn detection](voice-pipeline.md#turn-detection).

### V3 — Pluggable transcriber backend

Today: `DeepgramTranscriber` is concrete; the clone-template pattern
is a Protocol seam in spirit.

Change: formalize as `Transcriber` Protocol. `FasterWhisperTranscriber`
(CTranslate2) and `ParakeetTranscriber` (Parakeet-TDT 0.6B v3) drop
in behind it. Selected via `[providers.stt]`.

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

Today: strategy selection lives in `commands/run.py` Python wiring.

Change: TOML-driven selectors:

```toml
[providers.stt]
backend = "deepgram"          # | "faster_whisper" | "parakeet"

[providers.turn_detection]
strategy = "deepgram"         # | "silero+smart_turn" | "ten"

[providers.memory]
projectors = ["rich_note", "people_dossier"]

[providers.voice_pipeline]
mode               = "cascaded"  # | "s2s"
sentence_streaming = true
```

Defaults preserve today's behaviour. Consolidates with [Tuning](tuning.md)
so a single TOML drives the whole bot.

### A2 — Consolidate STT env vars into TOML

Today: `DEEPGRAM_*` knobs live in env. Awkward for per-character
tuning; per-channel overrides impossible.

Change: move non-secret Deepgram knobs to
`[providers.stt.deepgram]`. Secrets stay in env. Env continues to
override TOML for container deployments. Schema in [Tuning](tuning.md).

## Out of scope

- **Letta / MemGPT-style core-memory edit tools** — destructive,
  conflicts with source-of-truth preservation. See
  [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).
- **Hosted memory services (Zep Cloud, mem0)** — fail local-first.
  See [Decisions](decisions.md#third-party-managed-memory-services).
- **Headless-browser SillyTavern bridge** — already rejected. See
  [Decisions](decisions.md#bridging-to-a-running-sillytavern-instance).
