# Design Decisions Considered and Rejected

These are ideas that were seriously considered during planning and deliberately turned down. They are recorded here so future contributors (including future maintainers revisiting the codebase) don't rediscover them from scratch without the rationale.

## Bridging to a running SillyTavern instance

**The idea:** Run SillyTavern as a side-car process and route Familiar-Connect's context assembly and/or generation through it. The appeal was obvious — SillyTavern has a large, mature extension ecosystem (World Info, stepped thinking, recast post-processing, TunnelVision RAG, and many more) and reusing it directly would short-circuit a great deal of work.

**Why it's rejected:**

- **SillyTavern is a single-user local web app, not a library.** Its extensions are browser-side JavaScript hooked into the chat UI's event bus (`eventSource`, `getContext()`, generation interceptors). None of that is reachable from an external process.
- **SillyTavern's HTTP server is a thin LLM proxy.** It exists so the browser can dodge CORS for the upstream model API. It does not run the extension pipeline. Calling it from Familiar-Connect would buy us nothing we don't already get by talking to OpenRouter directly.
- **Running SillyTavern extensions outside SillyTavern requires a headless browser** driving a real ST tab (Playwright / CDP), intercepting `generate` calls, and marshaling chat state in and out. This is high-latency (the bot picked Cartesia for sub-100ms TTFB; a Chromium round-trip blows that budget), fragile across ST versions, and painful to test alongside the bot's `asyncio.TaskGroup`-scoped concurrency.
- **SillyTavern's architecture assumes one user, one active chat.** Familiar-Connect is multi-guild and concurrent. Forcing every guild through a single ST session serialises the bot; spawning one ST instance per guild is a deployment nightmare.

**What we take from it instead:** file formats only — Character Card V3, presets, World Info / lorebooks, and the macro vocabulary. These give SillyTavern users a painless on-ramp without making SillyTavern a runtime dependency. See [Context pipeline](context-pipeline.md) for the detailed split.

## Embedding a SillyTavern extension runtime via headless browser

**The idea:** A more aggressive variant of the bridge — actually load each SillyTavern extension we want by spinning up a headless Chromium with ST inside it, intercept the LLM call, and pass results back over CDP.

**Why it's rejected:** Listed separately because it's tempting in its own right. It still falls to all the latency, fragility, multi-guild, and concurrency-fit problems above, *and* it adds Chromium to the runtime. The cost of maintaining the glue layer would dwarf the cost of porting the two or three extensions we actually want.

## Adopting a large LLM orchestration framework as runtime

(LangChain, LlamaIndex, Haystack, etc.)

**The idea:** Build context management *inside* an existing framework's abstractions — register our providers as its components, let its runtime drive the bot's event loop, use its memory / retriever / agent classes as our primary building blocks.

**Why it's rejected as a runtime:**

- **These frameworks assume a synchronous request/response chat app and bury the prompt-assembly step we specifically want to be visible and testable.** Adopting one would mean either fighting its opinions or shoehorning our pipeline inside it.
- **LangChain's abstractions in particular have been rewritten repeatedly.** Production users commonly end up wrapping their own pipelines on top rather than depending on the framework's own memory/agent layers. We'd be writing the same glue and paying for the dependency anyway.
- **Each of them wants to own the event loop and the request lifecycle.** Familiar-Connect already has opinions about both (single-process, structured concurrency under `asyncio.TaskGroup`, per-turn deadlines, multi-modality). The framework would be working against us at the layer we care about most.

**What we do allow:** importing a *specific utility* from one of these libraries when it's a net simplification. Rule of thumb: if the import gives you a single function you call once (e.g. `llama-index`'s text splitters, a document loader for PDFs/HTML, a tokenizer helper), it's a utility — fine. If it wants to own your event loop, your prompt structure, or your retrieval flow, it's a runtime — not fine.

**What we take from them in spirit:** the "provider / retriever / processor" vocabulary these frameworks popularised. We implement it in a few hundred lines of project-local Python against our own protocols.

## Third-party managed memory services

(mem0, Zep, etc.)

**The idea:** Outsource long-term memory and user-fact tracking to a managed or self-hosted memory service.

**Why it's rejected:** The project commits to a **local-first principle**: all context state lives in-process, in the filesystem, or in SQLite on the same host. Sending conversation transcripts to a third-party memory service violates that principle, and the scale Familiar-Connect targets (one bot, N guilds, a single host, a learning project not headed for wide adoption) does not justify the operational or privacy cost. A plain-text memory directory plus a cheap search agent is strictly simpler and has the additional property that a human can `grep` and edit the familiar's memory directly. See [Memory](memory.md).

This rejection also applies to **running a memory MCP server we own as a sidecar** for the bot's own internal use. MCP is useful when multiple separate agents need to share a tool surface; when both ends of the wire are inside the same Python process, in-process function calls are simpler on every axis (latency, debuggability, no socket lifecycle to manage). MCP stays on the table as a way to later *expose* Familiar-Connect's memory to external tools; it is not how Familiar-Connect consumes its own.
