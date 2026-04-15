# Concurrency model

Familiar-Connect runs as a single `asyncio` process. The root
`asyncio.TaskGroup` wraps the Discord gateway task, optional Twitch
listener, optional voice capture, optional monitoring dashboard, and
the periodic memory-writer scheduler. A crash anywhere cancels the
whole reply path; structured concurrency guarantees no orphan tasks.

This page catalogues what runs **in parallel**, what runs
**sequentially**, and where the current bottlenecks are. Use it
alongside [Message flow](message-flow.md) and
[ConversationMonitor state machine](chattiness-fsm.md).

## What runs in parallel

- **Provider fan-out** — every per-turn `ContextProvider` runs
  concurrently under a scoped `asyncio.TaskGroup`, each with its own
  `asyncio.timeout(deadline_s)`. A slow sibling never blocks a fast
  one; a raising sibling never cancels the group (`_run_single_provider`
  catches everything and records an `error` / `timeout`
  `ProviderOutcome`). `src/familiar_connect/context/pipeline.py:144–223`.
- **Root services** — Discord gateway, Twitch EventSub, voice
  capture, TTS playback, the monitoring dashboard (when shipped), and
  the memory writer scheduler all live as siblings in the root
  `TaskGroup`.
- **Lull timers** — per-channel `loop.call_later` handles; when they
  fire they spawn a task tracked in `ConversationMonitor._lull_tasks`
  with a done-callback for cleanup.
- **Voice generation task** — the `main_prose` call for voice is
  wrapped in `asyncio.create_task` so the interruption detector can
  cancel it cleanly (`bot.py:_run_voice_response`, ≈ 132–471).

## What runs sequentially

- **Pre-processors** — run in registration order before provider
  fan-out, inside `ContextPipeline.assemble`. Only
  `SteppedThinkingPreProcessor` is wired today; the loop is prepared
  for more but they stack linearly.
- **Post-processors** — run in **reverse** registration order after
  the LLM reply, so they wrap symmetrically around the main call.
- **Chattiness decision** — `ConversationMonitor._evaluate` runs
  under `buf.lock` **before** the pipeline. The `main_prose` LLM
  call does not start until the decision resolves `YES`.
- **Main reply LLM call** — one call per turn, awaited directly. On
  the text path the `async with channel.typing()` block (`bot.py:949`)
  spans pipeline assembly + LLM + post-proc so the indicator appears
  immediately after `decision=YES`.
- **HistoryStore writes** — user turns then the assistant turn, both
  after the LLM reply so a failed turn never orphans the user's
  message. `bot.py:1026–1036`.

## Deadlines

- **Per-turn pipeline budget** — `ContextRequest.deadline_s`,
  configurable per channel. Not currently enforced as a wall clock
  over the entire pipeline; each provider has its own deadline.
- **Per-provider deadline** — `ContextProvider.deadline_s`,
  enforced via `asyncio.timeout`. Misses are recorded as
  `status="timeout"` outcomes; the contribution list is empty and
  siblings are unaffected.
- **Per-LLM-call timeout** — `LLMClient` has its own 120 s httpx
  timeout. No extra `asyncio.timeout` wraps the `main_prose` call —
  a long wait is preferable to a fallback there.

## Current bottlenecks (April 2026)

Measured on a running instance:

1. **History provider summariser** — up to 8 s inside
   `HistoryProvider.contribute` for cache-miss turns. The recent-history
   fs read (~ms) is serialised with the summariser LLM call. Fix in
   flight: move the summariser to a fire-and-forget background refresh
   so the provider returns cached-or-empty immediately and the fresh
   summary lands on the *next* turn. See
   [context-pipeline.md](context-pipeline.md).
2. **Content search agent loop** — burns up to 5 `memory_search` LLM
   calls per turn when the model never emits `ANSWER:`, returns empty.
   Fix in flight: forced-answer final iteration + lower default
   iteration cap + redundant-call bail-out.
3. **Sequential decision + pipeline** — chattiness evaluation
   completes before any context providers run. Speculative pipeline
   pre-warm (cheap, deterministic providers) during the decision eval
   is a tracked follow-up.

## Single-operator assumption

Familiar-Connect assumes one active familiar per process, one
operator per install. Every component assumes an `asyncio` event loop
and a single instance of state (`HistoryStore`, `MemoryStore`,
`SubscriptionRegistry`, `ChannelConfigStore`). Running multiple
familiars simultaneously requires separate processes with separate
`FAMILIAR_ID` env vars. See
[Configuration model](configuration-model.md).

## Failure isolation boundaries

| Boundary | Behaviour | Reference |
|----------|-----------|-----------|
| Provider raise / timeout | Isolated; outcome recorded, siblings unaffected | `pipeline.py:166–214` |
| Pre-processor `PreProcessorError` | Logged, skipped; next pre-processor runs on last good request | `pipeline.py:90–100` |
| Pre-processor any other exception | **Propagates** — contract violation, meant to be loud | `pipeline.py:87–100` |
| Post-processor raise | Caught, stage skipped, reply text passes through | `pipeline.py:131–141` |
| `main_prose` raise (text) | `(httpx.HTTPError, ValueError, KeyError)` caught; reply abandoned, nothing sent, nothing written | `bot.py:1001–1003` |
| `main_prose` raise (voice) | Same catch set; tracker returned to `IDLE` | `bot.py:249` |
| TTS raise | Logged with `_logger.exception`; text reply still sent if applicable | `bot.py:_run_voice_response` |
| Interjection LLM raise | Caught, `_evaluate` returns `False` (no response this turn) | `chattiness.py:389–394` |
