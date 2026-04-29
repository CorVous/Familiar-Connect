# Design Decisions Considered and Rejected

Ideas seriously considered during planning and deliberately turned down. Recorded here so future contributors don't rediscover them from scratch without the rationale.

## Bridging to a running SillyTavern instance

**The idea:** Run SillyTavern as a side-car process and route Familiar-Connect's context assembly and/or generation through it. SillyTavern has a large, mature extension ecosystem and reusing it directly would short-circuit a great deal of work.

**Why it's rejected:**

- **SillyTavern is a single-user local web app, not a library.** Its extensions are browser-side JavaScript hooked into the chat UI's event bus (`eventSource`, `getContext()`, generation interceptors). None of that is reachable from an external process.
- **SillyTavern's HTTP server is a thin LLM proxy.** It exists so the browser can dodge CORS for the upstream model API. It does not run the extension pipeline.
- **Running SillyTavern extensions outside SillyTavern requires a headless browser** driving a real ST tab (Playwright / CDP), intercepting `generate` calls, and marshaling chat state in and out. High-latency, fragile across ST versions, and painful to test alongside `asyncio.TaskGroup`-scoped concurrency.
- **SillyTavern's architecture assumes one user, one active chat.** Familiar-Connect is multi-guild and concurrent. Forcing every guild through a single ST session serialises the bot; spawning one ST instance per guild is a deployment nightmare.

## Embedding a SillyTavern extension runtime via headless browser

**The idea:** A more aggressive variant of the bridge — actually load each SillyTavern extension we want by spinning up a headless Chromium with ST inside it, intercept the LLM call, and pass results back over CDP.

**Why it's rejected:** It still falls to all the latency, fragility, multi-guild, and concurrency-fit problems above, *and* it adds Chromium to the runtime. The cost of maintaining the glue layer would dwarf the cost of porting the two or three extensions we actually want.

## Adopting a large LLM orchestration framework as runtime

(LangChain, LlamaIndex, Haystack, etc.)

**The idea:** Build context management *inside* an existing framework's abstractions — register our providers as its components, let its runtime drive the bot's event loop, use its memory / retriever / agent classes as our primary building blocks.

**Why it's rejected as a runtime:**

- **These frameworks assume a synchronous request/response chat app and bury the prompt-assembly step we specifically want to be visible and testable.** Adopting one would mean either fighting its opinions or shoehorning our pipeline inside it.
- **LangChain's abstractions in particular have been rewritten repeatedly.** Production users commonly end up wrapping their own pipelines on top rather than depending on the framework's own memory/agent layers.
- **Each of them wants to own the event loop and the request lifecycle.** Familiar-Connect already has opinions about both (single-process, structured concurrency under `asyncio.TaskGroup`, multi-modality). The framework would be working against us at the layer we care about most.

**What we do allow:** importing a *specific utility* from one of these libraries when it's a net simplification. Rule of thumb: if the import gives you a single function you call once (e.g. a text splitter, a document loader, a tokenizer helper), it's a utility — fine. If it wants to own your event loop, your prompt structure, or your retrieval flow, it's a runtime — not fine.

## Third-party managed memory services

(mem0, Zep, etc.)

**The idea:** Outsource long-term memory and user-fact tracking to a managed or self-hosted memory service.

**Why it's rejected:** The project commits to a **local-first principle**: all context state lives in-process, in the filesystem, or in SQLite on the same host. Sending conversation transcripts to a third-party memory service violates that principle, and the scale Familiar-Connect targets (one bot, N guilds, a single host) does not justify the operational or privacy cost.

This rejection also applies to **running a memory MCP server we own as a sidecar** for the bot's own internal use. MCP is useful when multiple separate agents need to share a tool surface; when both ends of the wire are inside the same Python process, in-process function calls are simpler on every axis (latency, debuggability, no socket lifecycle to manage).

The *open-source library* underneath Zep — Graphiti — is a different proposition. Graphiti is a Python package with pluggable graph backends; embedding it (or porting its bi-temporal edge logic onto our SQLite store) keeps every byte of state local. On the [roadmap](roadmap.md#m5-pluggable-memory-store-backend) as M5.

## Letta / MemGPT as the memory runtime

**The idea:** Adopt Letta (the maintained MemGPT continuation). Give the LLM tools like `core_memory_replace`, `archival_memory_insert`; let it manage its context as virtual memory.

**Why it's rejected:**

- **`core_memory_replace` is destructive by design.** The LLM mutates source-of-truth in-place. Familiar-Connect commits to the opposite: append-only, supersession over overwrite, bi-temporal records.
- **Recursive summarization compounds the destruction.** Older context becomes a lossy sketch of itself with no "true at t1, superseded at t2" handle. Breaks audit / contradiction-inspection.
- **Letta is an agent runtime, not a memory layer.** It owns the loop, tool execution, persistence. Adopting it forks our architecture into theirs.

**Kept from the design:** the two-tier framing — small always-in-context core block vs large on-demand recall layer. Familiar-Connect implements that split today via `core_instructions.md` + `character.md` + the recent-history layer on top of an append-only event log.

## Full-duplex speech-to-speech pipelines (Moshi, Sesame CSM)

**The idea:** Replace cascaded STT → LLM → TTS with a full-duplex S2S model (Moshi, Sesame CSM, Ultravox). ~200 ms theoretical voice-to-voice; native overlap-talk and barge-in.

**Why it's rejected (today):**

- **LLM brain is bundled.** Moshi pins you to Helium 7B; Sesame CSM ships its own LLaMA backbone. We route persona through OpenRouter — that's the most operator-impactful knob in the system.
- **Tool-calling and prompt knobs degrade.** S2S models don't yet match frontier text LLMs on tool use or prompt engineering.
- **Cascaded latency can mostly be closed.** Two-stage turn detection (Silero + Smart Turn) plus sentence-streaming TTS lands cascaded at ~700–900 ms — comfortably inside "feels natural".
- **Discord audio is constrained.** A 48 kHz Opus stream we don't fully control suits cascaded; S2S stacks expect to own the transport.

**Revisit when** a Mimi-based S2S model gains an external-LLM-brain seam. Tracked in [roadmap V5](roadmap.md#v5-full-duplex-s2s-as-a-research-branch).

## Heavy turn-detection LLM (TEN-style 7B classifier)

**The idea:** Replace silence-based endpointing with a fine-tuned 7B LLM (TEN Turn Detection's Qwen 2.5-7B) classifying transcript chunks `finished` / `unfinished` / `wait`.

**Why it's rejected:** wildly overkill for "did the user finish saying 'uh, hold on'". Pipecat's Smart Turn v3 (~360 MB ONNX, ~12 ms, BSD-2) does the same job on filler-word-aware audio for a fraction of the cost. TEN's approach fits enterprise SLAs and rare-language coverage; for a Discord familiar the cheaper classifier wins. Tracked in [roadmap V1](roadmap.md#v1-local-vad-semantic-turn-detection).
