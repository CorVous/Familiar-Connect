# ADR: in-process streaming bus (no broker, no sidecar)

## Status

Accepted. Implemented in
`src/familiar_connect/bus/` and all processors under
`src/familiar_connect/processors/`.

## Context

Familiar-Connect runs one character per process. Events flow from
several input surfaces — Discord text messages, Deepgram voice
transcripts, Twitch EventSub — into several processing surfaces —
reply generation, TTS playback, rolling summary, fact extraction,
debug logging. The re-architecture needed a structured way to
connect them without collapsing into a hand-wired spaghetti of
`asyncio.Queue`s and direct method calls.

Three approaches were on the table:

1. **External message broker** (Redis Streams, NATS, etc.). Durable,
   cross-process, battle-tested; but a single extra service to run
   and debug, and the project is explicitly local-first per
   [decisions.md](decisions.md).
2. **MCP-style sidecar / subprocess graph.** Fashionable; still adds
   inter-process machinery, and nothing in the project today benefits
   from isolation between components running in one user's Discord
   bot instance.
3. **In-process pub/sub.** A `Protocol`-hidden event bus that runs
   inside the single bot process. No broker, no sidecar. Cross-
   process messaging stays possible later by swapping the
   implementation behind the `EventBus` Protocol.

## Decision

Ship option 3 as the single data-plane. The `EventBus` Protocol
(`src/familiar_connect/bus/protocols.py`) is the seam; the concrete
`InProcessEventBus` is the only implementation we need today. Every
processor, responder, and worker subscribes via the Protocol so a
future `CrossProcessEventBus` can drop in without rewriting them.

### Constraints baked in

- **Topic-keyed fan-out.** Every subscriber gets its own queue; no
  shared-state broadcast.
- **Per-topic backpressure policies** (`BLOCK`, `DROP_OLDEST`,
  `DROP_NEWEST`, `UNBOUNDED`). `voice.audio.raw` is drop-oldest —
  losing a packet is always better than back-pressuring the Discord
  recording thread. Discord text and Twitch events are unbounded
  because their volume is low and dropping them is costly.
- **Lifecycle states** (`starting → running → draining → stopped`)
  with idempotent `start`/`shutdown`.
- **Content-addressed envelopes.** `Event` is a frozen dataclass
  with `event_id`, `turn_id`, `session_id`, `parent_event_ids`,
  `topic`, `timestamp`, `sequence_number`, `payload`. ``parent_event_ids``
  carries lineage for in-memory provenance; derived SQLite rows
  carry `source_turn_ids` for forever provenance.
- **Turn scoping.** `TurnRouter.begin_turn(session_id, turn_id)`
  cancels any active `TurnScope` in the same session before
  installing a new one. Different sessions are independent. This is
  how barge-in is expressed.

## Consequences

### Good

- One dependency surface to debug — standard library asyncio plus
  SQLite. No Redis-is-down failure modes.
- Sub-200 ms barge-in latency (verified by
  `tests/test_voice_responder.py::TestBargeIn`).
- Everything that matters is a pure-Python test away from being
  covered.
- Processor composition is simple enough that `commands/run.py`
  wires them up in ~15 lines.

### Bad

- No durability. If the process crashes mid-turn, the in-flight turn
  is lost. The `turns` table is the source of truth and survives
  restart, so the *durable* state is fine; only the ephemeral
  "currently speaking" state is lost.
- No cross-process fan-out. If we ever want one summary worker
  serving multiple bots, we'll need a new `EventBus` implementation.
  The Protocol was designed for this.
- Backpressure policy is per-topic, per-subscriber. A misconfigured
  subscriber (e.g. `BLOCK` on `voice.audio.raw`) would stall audio
  capture. Defaults live next to the topic constants; the wiring
  code in `commands/run.py` is the one place humans make the choice.

### Neutral

- Events are not persisted; there is no replay. If a processor is
  buggy we re-derive from `turns` — the source-of-truth table — by
  dropping the relevant side-index table and letting the worker
  rebuild. This is why FTS indexes, summaries, and facts all live in
  watermarked tables rather than in the bus itself.

## Alternatives considered

### Broker (Redis Streams / NATS / RabbitMQ)

Durable, supports fan-out across machines, mature clients. But:

- Adds a service a user has to run alongside their single Discord
  bot.
- Solves a problem (cross-process replay) we don't actually have.
- Doesn't help with the problem we *do* have — sub-200 ms barge-in
  cancel across a per-turn scope — which is more about in-process
  task orchestration than inter-process delivery.

### MCP subprocess graph

Each processor as a separate process speaking MCP. Offers isolation
but nothing here benefits from it. The overhead of marshalling every
audio chunk across process boundaries would push us off the barge-in
latency budget.

### Direct method calls (no bus at all)

Tempting given one character per process. Doesn't model the "any step
may be interrupted by new data" constraint cleanly — you end up with
cancellation tokens threaded through every signature, or with
`asyncio.Queue`s that are orphaned because nobody consumes them. The
bus makes cancel-on-new-turn explicit (TurnRouter + TurnScope) and
makes adding a new data stream cheap (new `StreamSource`, new topic,
no edits to existing processors).

## Related

- [Context pipeline](context-pipeline.md) — how side-indices are
  maintained and composed into the system prompt. Everything there
  runs on top of this bus.
- [Decisions](decisions.md) — local-first stance that made the
  broker option a non-starter.
- `docs/architecture/overview.md` — diagrams of the live
  source → bus → processor flow.
