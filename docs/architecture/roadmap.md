# Roadmap

Forward-looking work, ordered by approximate priority. When an item
ships, fold its description into the matching architecture page and
prune it to a one-line pointer here.

Grounding: the existing instinct (event-sourced `turns`, regenerable
side-indices, supersession over overwrite) aligns with the rigorous end
of the 2026 self-hosted-AI-character field. The roadmap widens seams for
A/B testing strategies and closes the voice-latency gap to sub-1 s. The
survey itself lives in [Memory strategies](memory-strategies.md) and
[Voice pipeline](voice-pipeline.md).

## Shipped

Folded into the architecture pages; the detail lives there.

| Item | Lands in |
|---|---|
| M2 — importance-weighted retrieval | [Memory strategies — retrieval ranking](memory-strategies.md#3-retrieval-ranking-ragcontextlayer) |
| M3 — reflection layer | [Memory strategies — reflections](memory-strategies.md#reflections-m3) |
| M4 — lorebook layer | [Memory strategies — lorebook](memory-strategies.md#lorebook-m4) |
| M6 — embeddings for semantic recall | [Memory strategies — embeddings](memory-strategies.md#embeddings-m6) |
| V3 — pluggable transcriber backend | [Voice pipeline — STT](voice-pipeline.md#stt-transcription) |
| A1 — strategy-swap config spine | [Tuning](tuning.md) |
| A2 — dynamic context budgeter | [Tuning — prompt assembly budget](tuning.md#prompt-assembly-budget) |
| A3 — tool calling (MVP) | [Tool calling](overview.md#tool-calling) |

(M5 and V1 also shipped — kept as headings below for inbound links.)

## Memory

### M5 — Pluggable memory-store backend (shipped)

Watermark-driven writers lifted behind a `MemoryProjector`
Protocol; operators select the active set via `[providers.memory]`.
Third-party projectors (Graphiti, Cognee) register at import time. See
[Memory strategies — swap points](memory-strategies.md#swap-points).

### M6 phase 3 — ANN at scale (deferred)

`sqlite-vec` once fact volumes outgrow brute-force cosine over BM25
candidates. The `Embedder` seam is unchanged; storage swaps a virtual
table for the BLOB column. See
[Memory strategies — embeddings](memory-strategies.md#embeddings-m6).

## Voice

### V1 — Local VAD + semantic turn detection

Shipped. TEN-VAD + Smart Turn v3 own endpointing locally; Deepgram
becomes pure STT. Saves 150–200 ms vs remote endpointing. See
[Voice pipeline — turn detection](voice-pipeline.md#turn-detection).

### V4 — Pluggable TTS backend & Mimi-codec readiness

Already pluggable via `synthesize() → TTSResult`. Tracked so the lesson
isn't lost: Mimi (Kyutai) is becoming the open audio-token standard.
When Sesame CSM stabilises, drop in a `SesameTTSClient` behind the
existing surface. `PiperTTSClient` is a smaller intermediate step.

### V5 — Full-duplex / S2S as a research branch

Not adopting. Rejection captured in
[Decisions](decisions.md#full-duplex-speech-to-speech-pipelines-moshi-sesame-csm).
Revisit if a Mimi-based S2S model exposes an external-LLM-brain seam.
OpenAI's GPT-Live (2026-07) is directional confirmation, not a trigger —
see the dated update in Decisions.

## Architecture / experimentation seams

### Remaining strategy selector

`[providers.voice_pipeline]` is the last unwired selector — no
implementation behind it yet:

```toml
[providers.voice_pipeline]
mode = "cascaded"             # | "s2s"   (V5)
```

### Tool-calling eval suite

Pre-tool eval asserting the configured model produces non-empty content
alongside any tool_call, per supported model. Until then the
`[voice].tool_filler_phrases` backstop is the model-agnostic floor. See
[Tool calling](overview.md#tool-calling).

## Out of scope

- **Letta / MemGPT-style core-memory edit tools** — destructive,
  conflicts with source-of-truth preservation. See
  [Decisions](decisions.md#letta-memgpt-as-the-memory-runtime).
- **Hosted memory services (Zep Cloud, mem0)** — fail local-first.
  See [Decisions](decisions.md#third-party-managed-memory-services).
- **Headless-browser SillyTavern bridge** — already rejected. See
  [Decisions](decisions.md#bridging-to-a-running-sillytavern-instance).
