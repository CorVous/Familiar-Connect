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

### V3 — Pluggable transcriber backend

Phase 1 shipped: `Transcriber` Protocol
(`src/familiar_connect/stt/protocol.py`), backend dispatch in
`stt/factory.create_transcriber`, and the `STT_BACKEND` env override
on top of `[providers.stt].backend`. `DeepgramTranscriber` lives in
`stt/deepgram.py`; the rest of the voice pipeline (`bot.py`,
`sources/voice.py`, `familiar.py`) types against `Transcriber`. Wiring
mirrors the V1 turn-detection pattern so a new backend is purely
additive — drop a module, add an arm to `_KNOWN_BACKENDS`, register
a parallel `[providers.stt.<name>]` config block.

Phase 2 (next): `ParakeetTranscriber` over NeMo Parakeet-TDT 0.6B v3.
Phase 3: `FasterWhisperTranscriber` (CTranslate2). Both are local —
streaming PCM intake, finalize-on-VAD-end semantics, no API key.

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
env var `LOCAL_TURN_DETECTION` continues to override TOML for container
deployments. See [Tuning — local turn detection](tuning.md#local-turn-detection-v1).

`[providers.stt].backend` selector wired in V3 phase 1 — same pattern
(env override `STT_BACKEND` on top of TOML). Today only `deepgram`
satisfies the dispatch; phases 2/3 widen the whitelist.

Remaining selectors (not yet wired — implementations don't exist):

```toml
[providers.memory]
projectors = ["rich_note", "people_dossier"]                      # (M5)

[providers.voice_pipeline]
mode = "cascaded"             # | "s2s"                           # (V5)
```

These will be wired when the corresponding backends land (M5).

## Out of scope

- **Letta / MemGPT-style core-memory edit tools** — destructive,
  conflicts with source-of-truth preservation. See
  [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).
- **Hosted memory services (Zep Cloud, mem0)** — fail local-first.
  See [Decisions](decisions.md#third-party-managed-memory-services).
- **Headless-browser SillyTavern bridge** — already rejected. See
  [Decisions](decisions.md#bridging-to-a-running-sillytavern-instance).
