# Design Decisions Considered and Rejected

Ideas seriously considered during planning and deliberately turned down. Recorded here so future contributors don't rediscover them without the rationale.

## Bridging to a running SillyTavern instance

**The idea:** Run SillyTavern as a side-car and route Familiar-Connect's context assembly or generation through it. SillyTavern's large extension ecosystem would short-circuit a lot of work.

**Why rejected:**

- **SillyTavern is a single-user local web app, not a library.** Extensions are browser-side JavaScript hooked into the chat UI's event bus (`eventSource`, `getContext()`, generation interceptors). None of it is reachable from an external process.
- **SillyTavern's HTTP server is a thin LLM proxy.** It exists so the browser can dodge CORS for the upstream model API. It does not run the extension pipeline.
- **Running SillyTavern extensions outside SillyTavern requires a headless browser** driving a real ST tab (Playwright / CDP), intercepting `generate` calls, marshaling chat state in and out. High-latency, fragile across ST versions, painful to test alongside tokio-scoped structured concurrency.
- **SillyTavern's architecture assumes one user, one active chat.** Familiar-Connect is multi-guild and concurrent. Forcing every guild through one ST session serialises the bot; one ST instance per guild is a deployment nightmare.

## Embedding a SillyTavern extension runtime via headless browser

**The idea:** A more aggressive variant — spin up a headless Chromium with ST inside it, intercept the LLM call, pass results back over CDP.

**Why rejected:** Hits all the latency, fragility, multi-guild, and concurrency-fit problems above, *and* adds Chromium to the runtime. Maintaining the glue layer would dwarf porting the two or three extensions we actually want.

## Adopting a large LLM orchestration framework as runtime

(LangChain, LlamaIndex, Haystack, etc.)

**The idea:** Build context management *inside* an existing framework's abstractions — register our providers as its components, let its runtime drive the event loop, use its memory / retriever / agent classes as primary building blocks.

**Why rejected as a runtime:**

- **These frameworks assume a synchronous request/response chat app and bury the prompt-assembly step we specifically want visible and testable.** Adopting one would mean either fighting its opinions or shoehorning our pipeline inside it.
- **LangChain's abstractions in particular have been rewritten repeatedly.** Production users commonly wrap their own pipelines on top rather than depending on the framework's memory/agent layers.
- **Each wants to own the event loop and request lifecycle.** Familiar-Connect already has opinions about both (single-process, tokio-based structured concurrency, multi-modality). The framework would be fighting us at the layer we care about most.

**What we do allow:** importing a *specific utility* from one of these libraries when it's a net simplification. Rule of thumb: a single function you call once (text splitter, document loader, tokenizer helper) is a utility — fine. Anything wanting to own the event loop, prompt structure, or retrieval flow is a runtime — not fine.

## Third-party managed memory services

(mem0, Zep, etc.)

**The idea:** Outsource long-term memory and user-fact tracking to a managed or self-hosted memory service.

**Why rejected:** The project commits to a **local-first principle**: all context state lives in-process, in the filesystem, or in SQLite on the same host. Sending conversation transcripts to a third-party memory service violates that, and the scale Familiar-Connect targets (one bot, N guilds, one host) doesn't justify the operational or privacy cost.

This also rejects **running a memory MCP server as a sidecar** for the bot's own internal use. MCP is useful when separate agents share a tool surface; when both ends of the wire are inside the same process, in-process function calls are simpler on every axis (latency, debuggability, no socket lifecycle).

The *open-source library* underneath Zep — Graphiti — is a different proposition. Graphiti is a Python package with pluggable graph backends; embedding it (or porting its bi-temporal edge logic onto our SQLite store) keeps every byte of state local. M5 [shipped](roadmap.md#m5-pluggable-memory-store-backend-shipped) the projector swap point Graphiti would plug into.

## Letta / MemGPT as the memory runtime

**The idea:** Adopt Letta (the maintained MemGPT continuation). Give the LLM tools like `core_memory_replace`, `archival_memory_insert`; let it manage its context as virtual memory.

**Why rejected:**

- **`core_memory_replace` is destructive by design.** The LLM mutates source-of-truth in-place. Familiar-Connect commits to the opposite: append-only, supersession over overwrite, bi-temporal records.
- **Recursive summarization compounds the destruction.** Older context becomes a lossy sketch of itself with no "true at t1, superseded at t2" handle. Breaks audit / contradiction-inspection.
- **Letta is an agent runtime, not a memory layer.** It owns the loop, tool execution, persistence. Adopting it forks our architecture into theirs.

**Kept from the design:** the two-tier framing — small always-in-context core block vs large on-demand recall layer. Familiar-Connect implements that split today via `character.md` (persona plus operational essentials) + the recent-history layer on top of an append-only event log.

## Full-duplex speech-to-speech pipelines (Moshi, Sesame CSM)

**The idea:** Replace cascaded STT → LLM → TTS with a full-duplex S2S model (Moshi, Sesame CSM, Ultravox). ~200 ms theoretical voice-to-voice; native overlap-talk and barge-in.

**Why rejected (today):**

- **LLM brain is bundled.** Moshi pins you to Helium 7B; Sesame CSM ships its own LLaMA backbone. We route persona through OpenRouter — the most operator-impactful knob in the system.
- **Tool-calling and prompt knobs degrade.** S2S models don't yet match frontier text LLMs on tool use or prompt engineering.
- **Cascaded latency can mostly be closed.** Two-stage turn detection (TEN-VAD + Smart Turn) plus sentence-streaming TTS lands cascaded at ~700–900 ms — comfortably inside "feels natural".
- **Discord audio is constrained.** A 48 kHz Opus stream we don't fully control suits cascaded; S2S stacks expect to own the transport.

**Revisit when** a Mimi-based S2S model gains an external-LLM-brain seam. Tracked in [roadmap V5](roadmap.md#v5-full-duplex-s2s-as-a-research-branch).

**2026-07 update — OpenAI GPT-Live.** GPT-Live (full-duplex front-end that
delegates reasoning/search to GPT-5.5 in the background) validates the
external-brain-seam thesis directionally, but clears none of the four
rejections: the delegated brain is OpenAI-internal (not OpenRouter-swappable),
weights are closed (not Mimi/self-hostable), it wants to own the transport
(not a Discord Opus stream), and the developer API is a signup form, not
shippable. Stay cascaded; no trigger. Revisit if the API ships *with* a
bring-your-own-brain seam.

## Heavy turn-detection LLM (TEN Turn Detection's 7B classifier)

**The idea:** Replace silence-based endpointing with a fine-tuned 7B LLM (TEN Turn Detection's Qwen 2.5-7B) classifying transcript chunks `finished` / `unfinished` / `wait`.

**Why rejected:** Wildly overkill for "did the user finish saying 'uh, hold on'". Pipecat's Smart Turn v3 (~360 MB ONNX, ~12 ms, BSD-2) does the same job on filler-word-aware audio for a fraction of the cost. TEN's approach fits enterprise SLAs and rare-language coverage; for a Discord familiar the cheaper classifier wins. Tracked in [roadmap V1](roadmap.md#v1-local-vad-semantic-turn-detection).

This rejects the TEN *Turn Detection* 7B model only. Stage-1 VAD does use TEN-framework's separate **TEN-VAD** (small native lib + bundled ONNX, Apache 2.0) — see [Voice pipeline — turn detection](voice-pipeline.md#turn-detection).

## Two-phase tool calling for voice (speak, then tools)

**The idea:** When voice tool calling is enabled, issue two LLM round-trips per turn — first without tools for the spoken reply, then with tools to decide on a tool call. Mechanical "speak before tool" ordering with no reliance on prompt instructions.

**Why rejected:**

- **Doubles the LLM round-trip cost** on every voice turn that doesn't end up calling a tool — i.e. almost all of them. Time-to-first-token already dominates the voice latency budget.
- **Single streaming call already orders content before tool_calls** for the models we route (OpenAI / Anthropic via OpenRouter). Streaming SSE emits content deltas first, then `tool_calls` deltas, then a `finish_reason`. Buffering the tool_call deltas until the stream closes is the same effect with zero added latency.

**What we ship instead** (see [Tool calling](overview.md#tool-calling)): three defenses in depth. (1) Mechanical: the agentic loop is a single streaming call per iteration with tool_call deltas buffered to end-of-stream. (2) Sharpened, end-placed prompt nudging the model to speak before invoking tools — targeting the *empty-content tool_call* failure mode specifically. (3) A filler-phrase backstop that the voice responder injects into TTS when an iteration closes with a tool call and no spoken content. Layers 2 and 3 cover model-compliance variance across slot configurations without paying the round-trip tax of layer (A).
