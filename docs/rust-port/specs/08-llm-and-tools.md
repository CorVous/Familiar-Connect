# 08-llm-and-tools — port spec

Source files: `llm.py` (~780 loc), `structured_request.py` (~260 loc),
`tools/` — `registry.py`, `loop.py`, `builtins.py`, `alarm.py`,
`scheduler.py`, `waker.py`, `silent.py`, `shift_focus.py`,
`read_channel.py`, `channel_view.py`, `image.py`, `image_describe.py`,
`image_compress.py`, `start_activity.py`, `__init__.py` (~1,700 loc).
Total ~2,741 loc.

## Role

The model boundary. `LLMClient` is a per-call-site-slot OpenRouter
chat-completions client (blocking `chat`, SSE-streaming
`stream_completion`/`chat_stream`) with a process-wide rate-limit
semaphore whose release timing is the barge-in load-bearing contract.
`structured_request` is the request-side of structured output: declare a
`Schema`, render its contract text, re-ask on shape failures.
The tools package is the agentic machinery: a name-indexed `ToolRegistry`,
the `agentic_loop` (stream → buffer tool calls → execute → re-call, with
leak-stripping guards), and every shipped tool — alarms (with DB-backed
`AlarmScheduler` and the `AlarmWaker` bus processor), `silent`,
`shift_focus`, `read_channel`, `view_image` (fetch → compress → vision
describe), and `start_activity`. `Message` (the OpenAI-dict message type)
defined here is the lingua franca consumed by 04/05/06/07/09/11.

## Public API surface

### `llm.py`

```python
sanitize_name(name: str) -> str | None
    # [^a-zA-Z0-9_-] → "_", truncate to 64, strip leading/trailing "_";
    # empty after cleanup → None. Used by 02 identity + 05 layers for the
    # OpenAI `name` field.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

get_request_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore
    # module-global, lazy-init; max_concurrent only honored on first call.
configure_request_semaphore(max_concurrent: int) -> asyncio.Semaphore
    # REPLACES the global semaphore; must run before any request in flight
    # (called once by create_llm_clients at startup).

@dataclass Message:
    role: str
    content: str | list[dict]        # list = multimodal / tool-result blocks
    name: str | None = None
    tool_calls: list[dict] | None = None   # OpenAI tool_calls dicts
    tool_call_id: str | None = None        # on role="tool" turns
    content_str -> str               # str passthrough; joins "text" blocks
    to_dict() -> dict                # omits None-valued optional fields

@dataclass LLMDelta:
    content: str = ""
    tool_calls: list[dict] = []      # raw streaming fragments (with "index")
    finish_reason: str | None = None

@dataclass SystemPromptLayers            # LEGACY — real assembly is 05
build_system_prompt(layers) -> str       # join non-empty card/rag/summary "\n\n"

class LLMClient:
    __init__(*, api_key, model, base_url=OPENROUTER_BASE_URL,
             temperature=None, top_p=None, top_k=None,
             presence_penalty=None, slot=None,
             provider_order: tuple[str,...]|None = None,
             provider_allow_fallbacks=True,
             reasoning: str|None = None, reasoning_max_tokens: int|None = None,
             tool_calling=False, image_tools=False, multimodal=False,
             no_stream=False, think_prepend=False)
    tool_calling_enabled / image_tools_enabled / multimodal  # attrs read by 06
    build_headers() -> dict          # Authorization: Bearer + Content-Type
    build_payload(messages, *, tools=None) -> dict
    async chat(messages) -> Message              # blocking, 429-retrying
    async stream_completion(messages, *, tools=None) -> AsyncIterator[LLMDelta]
    async chat_stream(messages) -> AsyncIterator[str]   # content-only projection
    async close()                                # shuts httpx pool

create_llm_clients(api_key, character_config) -> dict[str, LLMClient]
    # one client per slot in LLM_SLOT_NAMES {"fast","prose","background"} +
    # reserved "__image_description__" when image_description_model set.
```

`stream_completion`/`chat_stream` are the only streaming seams; tests and
responders substitute the whole client with duck-typed mocks (only
`stream_completion`, `chat`, `.multimodal`, `.slot`, `.tool_calling_enabled`
are ever touched from outside), so the Rust port should define an
`LlmClient` trait covering exactly those.

### `structured_request.py`

```python
DEFAULT_MAX_RETRIES = 1

@dataclass(frozen) Field: name, placeholder, desc="", required=True
@dataclass(frozen) Schema:
    fields: tuple[Field, ...]
    root: Literal["object","array"] = "array"
    container: str | None = None      # only with root="object"; else ValueError
    empty_note: str = ""
    constraints: tuple[str, ...] = ()
@dataclass(frozen) StructuredReply: value=None, ok=False, attempts=0

render_contract(schema) -> str        # deterministic contract text (see Data formats)
async request_structured(llm, *, messages, schema,
                         max_retries=DEFAULT_MAX_RETRIES) -> StructuredReply
```

Consumers: 07 workers (fact extractor, supersede, reflection) and 04
sleep (opinion formation). Parsing delegated to
`structured_output.coerce_json` (owned by 02).

### `tools/registry.py`

```python
@dataclass ImageResult: description: str, jpeg_base64: str,
                        media_type: str = "image/jpeg"
@dataclass ToolContext:
    familiar_id: str; channel_id: int; channel_kind: str  # "text"|"voice"
    turn_id: str; history: AsyncHistoryStore; bus: EventBus
    scheduler: Any|None = None        # AlarmScheduler (Any to break cycle)
    images: dict[str,str] = {}        # img_id → URL
    description_llm: LLMClient|None = None
    focus_manager: FocusManager|None = None
    store: AsyncHistoryStore|None = None
@dataclass Tool: name, description, parameters (JSON Schema dict),
    handler: async (dict, ToolContext) -> str | ImageResult,
    timeout_s: float = 10.0
class ToolRegistry:
    register(tool)        # duplicate name → ValueError
    get(name) -> Tool     # unknown → KeyError
    tools() -> Iterable[Tool]                  # insertion order
    as_openai_tools() -> list[dict]            # [] when empty
```

### `tools/loop.py`

```python
@dataclass AgenticResult: final_content: str, iterations: int,
    tool_calls_made: int, transcript: list[Message], is_silent: bool = False

async agentic_loop(*, llm, messages, registry, ctx,
    on_delta: async (LLMDelta) -> None | None = None,
    on_before_tools: async (Message) -> None | None = None,
    on_iteration_end: async (Message, list[Message]) -> None | None = None,
    max_iterations: int = 5) -> AgenticResult

serialize_image_result(res, *, multimodal) -> str | list[dict]
tool_content_as_text(content) -> str    # history-persistence projection
```

### Tool builders & runtime

```python
builtins.build_voice_registry(scheduler, *, focus_manager=None) -> ToolRegistry
builtins.build_text_registry(scheduler, *, image_tools=False,
    describe_constraints="", focus_manager=None,
    activity_engine=None) -> ToolRegistry

alarm.build_alarm_tool(scheduler) -> Tool          # scheduler arg unused;
alarm.build_cancel_alarm_tool(scheduler) -> Tool   # live one via ctx.scheduler

class scheduler.AlarmScheduler:
    __init__(*, history: AsyncHistoryStore, bus: EventBus, familiar_id: str)
    async start()     # idempotent; reload pending rows, spawn sleep tasks
    async shutdown()  # cancel tasks, DB untouched
    async add(*, channel_id, channel_kind, scheduled_at: datetime, reason,
              originating_turn_id=None) -> str      # alarm id
    async cancel(*, alarm_id) -> bool

class waker.AlarmWaker:              # bus processor (01 processor contract)
    name = "alarm-waker"; topics = (TOPIC_ALARM_FIRED,)
    __init__(*, familiar_id); async handle(event, bus)

silent.SILENT_RESULT = "__SILENT__"
silent.build_silent_tool() -> Tool
shift_focus.build_shift_focus_tool() -> Tool
read_channel.build_read_channel_tool() -> Tool
channel_view.serialize_turns(turns: list[HistoryTurn]) -> list[dict]
image.build_view_image_tool(describe_constraints="") -> Tool  # timeout 30s
image_describe.describe_image(*, llm, jpeg_base64,
    media_type="image/jpeg", constraints="") -> str
image_compress.compress_to_jpeg(raw, *, ceiling=1_000_000) -> bytes
image_compress.compress_for_description(raw) -> bytes          # 4MB ceiling
image_compress.encode_base64_jpeg(raw) -> str
image_compress.ImageTooLargeError(Exception)
start_activity.StartActivityEngine(Protocol):     # structural seam →
    catalog: tuple[ActivityType, ...]; active: object|None
    defer_start(type_id, note=None) -> dict
start_activity.build_start_activity_tool(engine) -> Tool
```

## Behaviors & invariants

### Rate-limit semaphore (the barge-in contract)

1. One process-wide semaphore shared by ALL clients (the bottleneck is the
   OpenRouter key, not a slot). `get_request_semaphore()` lazy-inits at 4;
   `configure_request_semaphore(n)` replaces it and is called exactly once
   at startup from `create_llm_clients` before any request exists.
2. `chat` (`_post`) holds the semaphore ONLY around the HTTP POST — it is
   released before backoff sleeps, so a retrying background call never
   starves live traffic.
3. `stream_completion` acquires the semaphore, opens the SSE request, and
   releases it IMMEDIATELY after `raise_for_status()` passes on the
   response headers — before body iteration begins. Streaming a long reply
   does not occupy a rate-limit slot. On failure/cancel during the open,
   the context manager is closed, the semaphore released, metrics emitted
   with status `cancelled`/`error`, and the exception re-raised. A Rust
   port must reproduce this exact release point (tests pin
   `sem._value == cap` after a consumer abandons the stream).

### `chat` retry policy

4. Retries on HTTP 429 ONLY. `_MAX_RETRIES = 4` (5 attempts total).
   Backoff: `min(1.0 * 2**attempt, 30.0)` seconds; a `Retry-After` header
   overrides with `min(float(header), 30.0)`; an unparseable header falls
   back to exponential. The final attempt returns the 429 response without
   sleeping; the caller's `raise_for_status()` surfaces it. Non-429
   statuses (including 5xx) return immediately — never retried.
5. On any status ≥ 400 the response body is logged at WARNING (truncated
   600 chars, slot-suffixed tag) BEFORE `raise_for_status()` — upstream
   error payloads must not be swallowed. Non-string `.text` (mock objects)
   is coerced to `""` so logging never crashes the request path.
6. Reply normalization: no `choices` → `ValueError("No choices returned
   from the API")`; `content` of None → `""`; `tool_calls` kept only if a
   non-empty list, filtered to dict items, else `None`.

### Payload construction (`build_payload`)

7. Base: `{"model", "messages": [m.to_dict()...]}`. Optional keys added
   only when the knob is not None: `temperature`, `top_p`, `top_k`,
   `presence_penalty`; `provider = {"order": [...], "allow_fallbacks":
   bool}` when `provider_order` set.
8. Reasoning precedence: `reasoning_max_tokens` → `{"max_tokens": n}` wins
   over everything; else `reasoning == "off"` → `{"exclude": true}`; else
   any other non-None string → `{"effort": <str>}`; None → key omitted.
   (Config validates against `REASONING_LEVELS` = off/none/low/medium/
   high/default; "default" is mapped to None by config parsing in 02 —
   the client itself passes whatever string it holds.)
9. `tools` + `"tool_choice": "auto"` added only when `tools` is truthy
   (empty list → omitted).
10. Anthropic prompt caching: iff `model.startswith("anthropic/")`, the
    FIRST system message's content is converted to a list of text blocks
    (string → single block; existing list → shallow-copied blocks) and the
    LAST block gets `"cache_control": {"type": "ephemeral"}`. Trailing
    system messages (per-turn-volatile reminder from 06) are deliberately
    left uncached. No system message → no-op. User/assistant messages
    untouched.
11. `think_prepend` appends `{"role":"assistant","content":
    "<think>\n\n</think>"}` as the LAST payload message (after
    cache-marking) on EVERY call — Qwen3 no-think stabiliser.

### Streaming (`stream_completion`)

12. Streamed payload adds `"stream": true` and `"usage": {"include":
    true}` (buys a trailing usage chunk for telemetry).
13. SSE line parsing (`_parse_sse_event`): only lines starting `data:`;
    `[DONE]` → skip; JSON parse failure or non-dict → skip; a top-level
    `{"error": {...}}` frame → WARNING log (message + code) and skip —
    mid-stream provider errors surface to the operator, not the caller.
14. Per parsed event: content deltas from every choice are joined into one
    string; tool-call fragments collected from every choice's
    `delta.tool_calls`; `finish_reason` from the first choice carrying a
    string. Events contributing none of the three yield nothing. Each
    yield is one `LLMDelta`.
15. NO retry loop on streaming — a 429 raises `httpx.HTTPStatusError`
    directly (retrying would hold the slot through sleeps and starve
    barge-in). Test-pinned.
16. Cancellation semantics: consumer `break`/`aclose()`/task cancel
    surfaces as `GeneratorExit`/`CancelledError` inside the generator →
    metrics status `"cancelled"` (distinct from `"error"` so diagnostics
    show real provider failures); any other exception → `"error"`. The
    `finally` block always stamps `t_end`, closes the httpx stream context
    (exceptions suppressed), and emits metrics exactly once.
17. `chat_stream` wraps `stream_completion`, yields only non-empty
    `.content`, and in its own `finally` explicitly calls the inner
    generator's `aclose()` (if present, exceptions suppressed) so the
    inner cleanup runs even when the outer generator is abandoned.
18. `no_stream=True` (constructor-only; NOT config-wired; workaround for
    models that emit tool calls as text under streaming): delegates to
    `chat()`, then synthesizes deltas — one content delta if content
    non-empty; one delta per tool call with
    `{"index": i, "id": tc.id or f"call_{i}", "type": "function",
    "function": {"name", "arguments" (default "{}")}}`; terminal delta
    with `finish_reason = "tool_calls"` if tool calls else `"stop"`.

### Usage telemetry (`_CallMetrics`)

19. Timings (perf-counter): `t_start` before request, `t_first_byte` after
    headers accepted, `t_first_delta` at first non-empty content delta,
    `t_end` in finally. Emits spans `llm.ttfb.<slot>`, `llm.ttft.<slot>`,
    `llm.total.<slot>` into the process `SpanCollector` (01) — suffix
    omitted when `slot is None`; ttft omitted when no content ever arrived
    (pure tool-call streams); ms values clamped ≥ 0. Collector failures
    suppressed.
20. Every chunk is fed to `absorb()`: `provider` (string) and `usage`
    (`prompt_tokens`, `completion_tokens`, and
    `prompt_tokens_details.cached_tokens`, each only if int) are taken
    from whichever chunk carries them (last wins).
21. One structured INFO line per call: tag `LLM call`, kv slot ("-" when
    None), model, status (ok/cancelled/error), chars (sum of str-content
    lengths of input messages; list content counts 0), then
    ttfb_ms/ttft_ms/total_ms/provider/in_tokens/out_tokens/cached — each
    only when known. `input_chars` is computed BEFORE the request.

### Client construction

22. `create_llm_clients`: calls `configure_request_semaphore(
    character_config.llm_max_concurrent_requests)` first; builds one
    client per slot name in `LLM_SLOT_NAMES` (KeyError propagates if a
    slot is missing — run.py catches and errors out); threads model,
    temperature, top_p, top_k, presence_penalty, slot name, provider
    order/fallbacks, reasoning, tool_calling, image_tools, multimodal,
    think_prepend. `reasoning_max_tokens` and `no_stream` are NOT
    config-threaded. When `image_description_model` is non-empty, adds a
    client under reserved key `"__image_description__"` with
    `slot="image_description"` and all other knobs default. One INFO
    config line per slot.

### Structured requests

23. `Schema.__post_init__` raises `ValueError` when `container` is set
    with `root="array"` — programming error, not data error.
24. `render_contract` output is byte-exact (tests match substrings):
    line 1 `Reply with JSON only, no prose or code fences: <skeleton>`
    where skeleton is `[{item}, ...]` (array), `{"<container>": [{item},
    ...]}` (object+container), or `{item}` (flat object); item is
    `{"name": placeholder, ...}` comma-space separated. If any field has a
    `desc`: header `Each item's fields:` (array or container) or
    `Fields:` (flat), then one bullet per desc-bearing field:
    `- \`name\`{ (optional)}: desc`. Then each `constraints` line, then
    `empty_note` last. Joined with `\n`.
25. `request_structured`: `retries = max(0, max_retries)`; runs
    `retries + 1` attempts. Caller's `messages` list is copied — NEVER
    mutated. Each attempt: `llm.chat(convo)`, `coerce_json(content_str,
    expect=schema.root)` (02); shape problems are: parse failure ("the
    reply was not valid JSON"), or parsed value not dict/list per root.
    Parsing runs once per attempt (result reused on success). On failure
    with attempts remaining, the convo grows by the assistant's raw reply
    plus a correction user turn: `Your previous reply could not be used:
    {problem}. Reply again with ONLY the JSON described below — no prose,
    no code fences, no explanation.\n{render_contract(schema)}`.
26. Success → `StructuredReply(value=parsed_root, ok=True, attempts=n)`.
    Exhaustion → WARNING log (`degraded`, slot via `getattr(llm, "slot")`
    fallback "-", attempts, last problem) and
    `StructuredReply(None, False, attempts)` — caller degrades to its
    empty container. Transport/HTTP exceptions from `chat` PROPAGATE
    unchanged — only shape problems are retried. Fenced JSON is accepted
    (coerce_json strips fences). Domain validation is the caller's job.

### Agentic loop

27. `tools_payload` is computed once before the loop: registry non-empty →
    `as_openai_tools()`, else `None` — passed identically to every
    iteration's `stream_completion`. Empty registry means the first
    iteration is terminal (no tools offered).
28. Per iteration: content deltas append to a buffer; tool-call fragments
    merge into `pending: dict[index → call]` — non-int index skipped;
    bucket seeded `{"id":"", "type":"function", "function":{"name":"",
    "arguments":""}}`; `id`/`type` overwritten when a string fragment
    carries them; `function.name` overwritten only by non-empty strings;
    `function.arguments` string-CONCATENATED across fragments. `on_delta`
    is awaited for every delta AFTER buffering (responders stream to
    TTS/Discord through it).
29. Finalize: calls sorted by index, entries with empty `id` dropped.
    The assistant `Message(role="assistant", content, tool_calls or None)`
    is appended to `messages` — the input list is MUTATED in place and
    also returned as `transcript`.
30. `on_before_tools(assistant_msg)` awaited only when tool calls exist,
    after the assistant message is built, before any handler runs (voice
    filler-phrase hook).
31. Tool execution is SEQUENTIAL in index order (never concurrent). Each
    call: arguments JSON decoded (`""`/whitespace → `{}`); decode failure
    → `{"error": "invalid arguments JSON: <exc>"}`; non-object →
    `{"error": "invalid arguments JSON: not a JSON object"}`; unknown tool
    → `{"error": "unknown tool: <name>"}` — all WITHOUT running a handler.
    Otherwise `asyncio.wait_for(handler(args, ctx), timeout=tool.
    timeout_s)`; `TimeoutError` → `{"error": "timeout after <t>s"}`; any
    other exception → `{"error": "<Type>: <msg>"}`. Result message:
    `role="tool"`, `tool_call_id` = call id (missing → `""`). Each tool
    message appended to `messages` as produced. `tool_calls_made`
    increments per call attempted.
32. `ImageResult` results are serialized per the CLIENT's `multimodal`
    flag (read via duck-typed `getattr(llm, "multimodal", False)`):
    text-only → the description string; multimodal → `[{"type":"text",
    "text": desc}, {"type":"image_url", "image_url": {"url":
    "data:<media_type>;base64,<jpeg>"}}]`.
33. Silent short-circuit: if ANY tool message this iteration has string
    content exactly equal to `SILENT_RESULT` (`"__SILENT__"`), return
    immediately `AgenticResult(final_content="", is_silent=True, ...)`
    — critically BEFORE `on_iteration_end`, so responders never persist
    the silent call or its reasoning to history (persisting would re-seed
    the model's rationale for silence next turn). The sentinel is
    lazy-imported to avoid an import cycle; fallback constant is the same
    string.
34. `on_iteration_end(assistant_msg, tool_msgs)` awaited once per
    non-silent iteration (responders persist intermediate turns here).
35. Termination: no tool calls → break (normal). `iterations >=
    max_iterations` with tool calls still pending → WARNING
    `hit_max_iterations` + break — tool results for the final iteration
    ARE executed and appended, but no re-call happens; `final_content` is
    that iteration's content.
36. Leak guard on the final content (runs on every exit path except the
    silent short-circuit): first strip ONE leading think artifact
    (`^\s*(?:<think>.*?</think>|</think>)\s*`, DOTALL), then leaked
    tool-call text: content leading with `<invoke` (optionally
    namespace-prefixed `<ns:invoke`) → all `<invoke ...>...</invoke>`
    blocks removed (DOTALL), stripped; `silent_leak=True` if any removed
    block's `name="..."` contains "silent". Content leading with
    `silent(` (case-insensitive) → cleaned `""`, silent. Leading
    `read_channel(` or `shift_focus(` (case-insensitive) → cleaned `""`,
    NOT silent. Mid-prose mentions of `<think>`, "invoke", or tool names
    are untouched. If cleaning changed anything, WARNING
    `leaked_tool_call_stripped`. Result: `final_content=cleaned`,
    `is_silent = silent_leak and not cleaned`.

### Registry composition (builtins)

37. Voice registry: `set_alarm`, `cancel_alarm`, `silent`; plus
    `shift_focus` iff a focus manager is provided. `view_image`,
    `read_channel`, `start_activity` NEVER in voice.
38. Text registry: `set_alarm`, `cancel_alarm`, `silent`; plus
    `view_image` iff `image_tools` (constraints bound at build time);
    plus `shift_focus` AND `read_channel` iff focus manager; plus
    `start_activity` iff activity engine. Wiring (10) gates `image_tools`
    on the PROSE slot's flag.

### Alarms

39. `set_alarm` handler: `ctx.scheduler` None → error. `reason` required,
    non-empty string, ≤ 200 chars. Time resolution: `when` (non-empty
    string) takes precedence — must parse ISO-8601 AND carry a timezone
    (naive → error); may be up to 5 s in the past (skew tolerance),
    further past → error `'when' is <n>s in the past`. Otherwise
    `delay_seconds` must be an int (bool explicitly rejected) in
    [1, 31 536 000] (one year). Neither → error naming both options. All
    errors are JSON `{"error": ...}` strings, never exceptions. Success:
    `{"alarm_id", "scheduled_at": <iso>, "ack": "ok"}`. The wake routes
    to the CALLING channel: `ctx.channel_id`/`ctx.channel_kind`;
    `originating_turn_id = ctx.turn_id`.
40. `cancel_alarm` handler: `alarm_id` required non-empty string;
    `scheduler.cancel` returning False → `{"error": "no pending alarm
    with id <id>"}`; True → `{"alarm_id", "ack": "ok"}`.
41. `AlarmScheduler.add`: DB insert FIRST (row is durable before any task
    exists), then spawn one asyncio task named `alarm-<id>`, tracked in
    `self._tasks[id]`.
42. `_sleep_then_fire`: computes delay from now; ≤ 0 → fires immediately
    (past-due reload case). After sleeping, `mark_alarm_fired` is a
    CONDITIONAL update (`fired_at IS NULL AND cancelled_at IS NULL`) —
    returns False if another path fired/cancelled first, in which case NO
    event is published (idempotence against cancel races). On success
    publishes `TOPIC_ALARM_FIRED` with `event_id=uuid4().hex`,
    `turn_id=f"alarm-{id}"`, `session_id=f"alarm:{channel_id}"`,
    `sequence_number=0`, payload per Data formats. `finally` always
    removes the task from the map.
43. `start()` is idempotent (guard flag); loads
    `list_pending_alarms(familiar_id)` (fired_at/cancelled_at both NULL,
    ordered scheduled_at ASC) and spawns a task per row — this is the
    restart-survival path. `shutdown()` cancels and awaits all tasks
    (CancelledError suppressed) but NEVER touches the DB — rows stay
    pending and reload on next boot.
44. `cancel()`: pops + cancels the in-process task (if any) FIRST, then
    stamps `cancelled_at` (conditional update); returns the DB result —
    so cancelling an alarm loaded by a *different* process instance still
    returns True and prevents its fire via the conditional
    `mark_alarm_fired`.
45. `AlarmWaker.handle`: ignores non-`alarm.fired` topics and non-dict
    payloads; `channel_id` must be int (else drop); `channel_kind`
    defaults `"text"`, values outside {text, voice} → WARNING + drop;
    `"voice"` → INFO fallback log but SAME synthetic shape (MVP: voice
    alarms wake as text in the voice channel's id). Publishes a synthetic
    `discord.text` event: `turn_id=f"wake-{alarm_id}"` (falls back to the
    new event id when payload lacks alarm_id), `session_id=
    str(channel_id)`, `parent_event_ids=(source.event_id,)`, payload per
    Data formats. The synthetic payload's `familiar_id` is the WAKER's
    configured id (alarm payloads carry no familiar id); per-familiar
    filtering happens in the TextResponder (06). `alarm: True` marker
    pierces activity absence gating (11).

### silent / shift_focus / read_channel

46. `silent` handler ignores ctx, logs `reasoning` at INFO, returns the
    sentinel string — even with empty/missing reasoning. Schema requires
    `reasoning` (string).
47. `shift_focus` is IMMEDIATE: `focus_manager` None → error;
    `channel_id` must be int → else error; unsubscribed target → error
    JSON that also carries `available_channels: [{channel_id, label}]`
    (from `fm.subscribed_channels()` / `fm.channel_label`) so the model
    can recover in-turn. Subscribed → `await fm.shift_now(channel_id)`
    (05: promotes catch-up window, marks older backlog missed) — applied
    at handler time, so a turn that then goes silent still leaves focus
    at the target with nothing leaked. When `ctx.store` is present the
    handler eagerly fetches `store.recent(familiar_id, channel_id,
    limit=fm.catch_up_limit)` and returns `{"ok": true, "channel_id",
    "messages": serialize_turns(...)}` — the preview size deliberately
    EQUALS the promotion window (what she sees is exactly what gets
    consumed); empty/voice channels → `[]`. Without a store → bare ack.
48. `read_channel` (text-only, read-only — never touches consumed_at):
    requires focus_manager AND store (each missing → its own error);
    focus channel = `fm.get_focus("text")`, None → `{"error": "no text
    focus active"}`. `limit`: int → `min(limit, 50)`, non-int/missing →
    default 20. `before_id` and `around_id` mutually exclusive → error.
    `around_id` → `store.turns_around(turn_id=around_id,
    before=half, after=half)` with `half = max(1, limit // 2)`; else
    `store.recent(limit=limit, before_id=before_id)`. Deliberately
    UNFILTERED by the activities archive watermark (fresh eyes may scroll
    archived past). Result: `json.dumps(serialize_turns(turns))`.
49. `serialize_turns`: drops `role == "tool"` turns AND turns whose
    content is empty/whitespace (tool-call scaffolding husks); survivors
    → `{"id", "role", "author", "content", "timestamp": <iso>}` with
    author = `display_name or username` or None. This exclusion closes a
    recursion vector: a persisted tool-dump preview re-embedded into a
    later preview would compound until context overflow. Idempotent.

### view_image pipeline

50. Handler flow: `image_id` must be a string (else error); resolved via
    `ctx.images` (placeholder map injected per-turn by 06/10; unknown id
    → error). Fetch: fresh `httpx.AsyncClient(timeout=15.0,
    follow_redirects=True)` per call, streaming GET;
    `raise_for_status()`; content-type checked against allow-list
    {jpeg, png, gif, webp, bmp, tiff} after stripping `;` params and
    lowercasing (non-image → ValueError); body streamed in 64 KiB chunks
    with a hard 20 MB cap (exceeded → ValueError). ANY fetch exception →
    WARNING log + `{"error": "fetch failed: ..."}`.
51. Description leg (high quality — result persists in history):
    `ctx.description_llm` None → description
    `"(no description model configured)"`; else
    `compress_for_description` (quality ladder 95→20, 4 MB ceiling) →
    base64 → `describe_image`; any exception → WARNING +
    `"(description unavailable)"` (degrades, never fails the tool).
52. Payload leg (tight — travels in the prompt): `compress_to_jpeg`
    (85→20, 1 MB ceiling); failure → `{"error": "compression failed:
    ..."}` (this one IS fatal to the tool call). Success →
    `ImageResult(description, jpeg_base64)`. Tool `timeout_s=30`
    (vs default 10) covers fetch + describe + compress.
53. `describe_image`: prompt = neutral base (`"Describe this image
    concisely for a chat assistant. Focus on the main subject and any
    notable details."`) + `" " + constraints.strip()` when constraints
    non-blank (no trailing space when blank); single user message with
    `[text block, image_url data-URI block]`; returns `reply.content` if
    str else `""`. Constraints come from
    `[prompt].image_description_constraints`, bound into the tool closure
    at BUILD time (static per familiar), not carried on ToolContext.
54. Compression: `_open_as_rgb` — static images convert to RGB; animated
    GIFs (n_frames > 1) tile up to 4 EVENLY-SPACED frames (indices
    `round(i*(n-1)/(count-1))`) horizontally, each scaled (LANCZOS) to
    the tallest frame's height, so the vision model sees the animation
    arc. Then `thumbnail((1024, 1024))` — downscale only, aspect
    preserved. JPEG encode with `optimize=True`, quality stepping down by
    5 from the start quality to 20 until size ≤ ceiling;
    `ImageTooLargeError` if even q=20 exceeds it.

### start_activity

55. Engine bound at build time (ctx unused). If `engine.active` is not
    None (already out), returns the SILENT sentinel — calling
    start_activity while out signals stay-out intent and must not narrate
    meta-text to the channel (the loop's silent detection then ends the
    turn). Otherwise: `activity` required non-empty string, `note`
    optional but must be string if present (each violation → error JSON);
    `engine.defer_start(activity, note)` result JSON-serialized verbatim
    (engine errors pass through as `{"error": ...}`). Actual departure is
    DEFERRED — applied by `engine.end_turn()` after the reply ships (11).
56. Schema: `activity` has `enum` of catalog type-ids; its description
    enumerates `'id' = label` entries, appending an availability clause
    `[Mon Tue 09:00-17:00]` for entries with `active_days` (sorted,
    Mon=0 abbreviations) and/or `active_hours` (`%H:%M-%H:%M`). The tool
    DESCRIPTION carries the entire when-to-go policy (budget-tested:
    ≤ ~600 chars) — zero character-card growth by design.

## Data formats

### OpenAI-compatible message dict (`Message.to_dict`)

```json
{"role": "user", "content": "hi", "name": "alice"}
{"role": "assistant", "content": "", "tool_calls": [
   {"id": "call_1", "type": "function",
    "function": {"name": "set_alarm", "arguments": "{\"reason\":\"x\"}"}}]}
{"role": "tool", "tool_call_id": "call_1", "content": "{\"ack\":\"ok\"}"}
{"role": "tool", "tool_call_id": "c", "content": [
   {"type": "text", "text": "desc"},
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}
```

`name`/`tool_calls`/`tool_call_id` omitted when None. Cached system
message (anthropic/ models):
`{"role":"system","content":[{"type":"text","text":"...",
"cache_control":{"type":"ephemeral"}}]}`.

### `tools` payload entry (`as_openai_tools`)

```json
{"type": "function", "function":
  {"name": "...", "description": "...", "parameters": {<JSON Schema>}}}
```

### SSE stream (request `stream: true, usage: {include: true}`)

Lines `data: <json>`; terminal `data: [DONE]`. Content chunk:
`{"choices":[{"delta":{"content":"Hel"}}]}`. Tool-call fragment chunk:
`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",
"type":"function","function":{"name":"f","arguments":"{\"a\""}}]}}]}` —
`arguments` accumulates by index across chunks. Finish:
`{"choices":[{"finish_reason":"tool_calls"}]}`. Usage/provider chunk:
`{"provider":"anthropic","usage":{"prompt_tokens":n,
"completion_tokens":n,"prompt_tokens_details":{"cached_tokens":n}}}`.
Error frame: `{"error":{"message":"...","code":404}}` (logged, skipped).

### `alarms` table (in history.db, owned by 03; consumed here)

```sql
CREATE TABLE IF NOT EXISTS alarms (
    id                   TEXT    PRIMARY KEY,   -- uuid4 hex
    familiar_id          TEXT    NOT NULL,
    channel_id           INTEGER NOT NULL,
    channel_kind         TEXT    NOT NULL CHECK(channel_kind IN ('text','voice')),
    scheduled_at         TEXT    NOT NULL,      -- ISO-8601 UTC
    reason               TEXT    NOT NULL,
    originating_turn_id  TEXT,
    fired_at             TEXT,                  -- NULL = not fired
    cancelled_at         TEXT,                  -- NULL = not cancelled
    created_at           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alarms_pending
    ON alarms (familiar_id, fired_at, cancelled_at, scheduled_at);
```

Pending = `fired_at IS NULL AND cancelled_at IS NULL`. Both state
transitions are conditional updates on that predicate (race-safe).

### `alarm.fired` event payload

```json
{"alarm_id": "<hex>", "channel_id": 42, "channel_kind": "text",
 "reason": "...", "scheduled_at": "<iso>", "fired_at": "<iso>",
 "originating_turn_id": "<id or null>"}
```

Envelope: topic `alarm.fired`, `turn_id="alarm-<id>"`,
`session_id="alarm:<channel_id>"`, `sequence_number=0`.

### Synthetic wake `discord.text` payload (AlarmWaker)

```json
{"familiar_id": "<waker's id>", "channel_id": 42,
 "content": "[alarm fired: <reason>]", "author": null, "guild_id": null,
 "message_id": null, "reply_to_message_id": null, "mentions": [],
 "alarm": true}
```

`mentions` is the empty tuple in Python; `turn_id="wake-<alarm_id>"`,
`session_id=str(channel_id)`, `parent_event_ids=(source_event_id,)`.

### Tool result JSON shapes (strings returned by handlers)

- errors, always: `{"error": "<message>"}` (+ `available_channels` for
  shift_focus rejection)
- set_alarm ok: `{"alarm_id": "...", "scheduled_at": "<iso>", "ack": "ok"}`
- cancel_alarm ok: `{"alarm_id": "...", "ack": "ok"}`
- shift_focus ok: `{"ok": true, "channel_id": N, "messages": [...]}`
  (messages key only when store wired)
- read_channel ok: bare array `[{"id", "role", "author", "content",
  "timestamp"}, ...]`
- silent: the raw string `__SILENT__` (not JSON)
- view_image ok: `ImageResult` (serialized by the loop, not the handler)

### Structured contract text

```
Reply with JSON only, no prose or code fences: [{"stance": "<stance>", "ids": [<id>...]}, ...]
Each item's fields:
- `stance`: plain-language gloss
<constraint lines...>
<empty note>
```

### Image constants

Max edge 1024 px; prose JPEG quality 85→20 step 5, ceiling 1,000,000 B;
describe JPEG quality 95→20 step 5, ceiling 4,000,000 B; GIF strip max 4
frames; download cap 20 MiB; fetch timeout 15 s; view_image tool timeout
30 s; default tool timeout 10 s.

## Config knobs

| Key | Default | Consumer |
|---|---|---|
| env `OPENROUTER_API_KEY` | required | run.py → `create_llm_clients` |
| `[llm].max_concurrent_requests` | 4 (positive int, bool rejected) | `configure_request_semaphore` |
| `[llm].image_description_model` | `""` (disabled) | `__image_description__` client |
| `[llm.<slot>].model` | required per slot | client |
| `[llm.<slot>].temperature` / `.top_p` / `.top_k` / `.presence_penalty` | absent → provider default | payload |
| `[llm.<slot>].think_prepend` | `false` | assistant prefill |
| `[llm.<slot>].provider_order` | absent → OpenRouter default | `provider.order` |
| `[llm.<slot>].provider_allow_fallbacks` | `true` | `provider.allow_fallbacks` |
| `[llm.<slot>].reasoning` | absent → model default; one of off/none/low/medium/high/default | reasoning payload |
| `[llm.<slot>].tool_calling` | `false` (shipped: only `background` on) | responders gate agentic loop |
| `[llm.<slot>].image_tools` | `false` | view_image registration (prose slot, independent of tool_calling) |
| `[llm.<slot>].multimodal` | `false` | ImageResult serialization |
| `[tools].loop_max_iterations` | 5 (positive int) | `agentic_loop(max_iterations=...)` via responders |
| `[prompt].image_description_constraints` | `""` | bound into view_image at build |

Slot names fixed: `fast` (voice), `prose` (text), `background` (workers).
Hardcoded: retry (4 retries, 1 s base, 30 s cap), semaphore default 4,
httpx timeout 120 s, `DEFAULT_MAX_RETRIES=1` (structured),
alarm reason ≤ 200 chars, delay 1 s..1 year, past skew 5 s,
read_channel default 20 / max 50, image constants above.
`reasoning_max_tokens` and `no_stream` exist only as constructor args.

## Dependency edges

Imports (this subsystem → others):

| Module | Subsystem | Used for |
|---|---|---|
| `log_style`, `diagnostics.collector` | 01 | log formatting, spans |
| `bus.envelope`, `bus.topics`, `bus.protocols` | 01 | Event, TOPIC_ALARM_FIRED, TOPIC_DISCORD_TEXT, EventBus |
| `config` (LLM_SLOT_NAMES, CharacterConfig) | 02 | client construction |
| `structured_output.coerce_json` | 02 | structured_request parsing |
| `history.async_store`, `history.store` (HistoryTurn, alarm CRUD) | 03 | scheduler persistence, channel previews |
| `focus.FocusManager` (type + runtime via ctx) | 05 | shift_focus/read_channel |
| `activities.config.ActivityType` (type only) | 11 | start_activity schema |
| `httpx`, `PIL` | external | transport, JPEG |

Imported by (others → this subsystem):

| Importer | Subsystem | What |
|---|---|---|
| `identity.py`, `context/layers.py` | 02, 05 | `sanitize_name`, `Message` |
| `context/assembler.py`, `budget.py` | 05 | `Message` |
| `processors/text_responder.py`, `voice_responder.py` | 06 | `LLMClient`, `LLMDelta`, `agentic_loop`, `AgenticResult`, `ToolRegistry`, `ToolContext`, `tool_content_as_text`, `SILENT_RESULT` |
| `processors/{summary,people_dossier,reflection,fact_*}_worker.py`, `projectors.py` | 07 | `Message`, `LLMClient`, `request_structured`, `Schema`, `Field` |
| `sleep/*` | 04 | `Message`, `LLMClient`, `request_structured` |
| `stt/protocol.py`, `sentence_streamer.py` | 09 | `Message` type / stream shapes |
| `twitch.py`, `activities/engine.py` | 11 | `Message`, `LLMClient`, `SILENT_RESULT` interplay |
| `commands/run.py`, `familiar.py` | 10 | `create_llm_clients`, registries, `AlarmScheduler`, `AlarmWaker`, `ToolContext` factory |

Cycle note: `registry.ToolContext.scheduler` is typed `Any` and
`loop._get_silent_result` lazy-imports `silent` — both are Python
import-cycle workarounds, not semantics; Rust can use concrete types.

## Test inventory

| Test file | Behaviors pinned | Portability |
|---|---|---|
| `tests/test_llm.py` (1019 loc) | Message name/to_dict field omission; build_system_prompt ordering + empty-section skip; payload knob inclusion/omission, provider pinning, reasoning off/effort/max_tokens precedence, think_prepend prefill, anthropic cache breakpoint (first-system only, list-content no double-wrap, no-system no-op); chat parse (assistant reply, all-messages sent, HTTP error raise + body log, empty choices); create_llm_clients (per-slot clients, shared key, reserved image slot, flag threading, semaphore cap applied); 429 retry (retry-after honor, cap, give-up, non-429 no-retry, backoff cap); http client reuse/close; semaphore limits concurrency | logic-portable (needs Rust HTTP mock; two `TestOpenRouterLive` tests are live-API, skip) |
| `tests/test_llm_stream.py` | delta-per-chunk; consumer break stops iteration; `stream: true` sent; **semaphore released after consumer abandons stream**; 429 raises with NO retry; 4xx body logged; SSE error frame logged + skipped, yields nothing | logic-portable (fake SSE transport) |
| `tests/test_llm_tool_calls.py` | Message tool fields round-trip; tools+tool_choice payload (omitted when empty/None); chat parses tool_calls (content None→""); stream accumulates fragmented arguments by index; finish_reason yielded; tool_calling flag default/ctor | logic-portable |
| `tests/test_llm_diagnostics.py` | usage.include flag; ttfb/ttft/total spans (slot suffix, unsuffixed fallback, no ttft w/o content); structured call log fields; status ok vs cancelled (consumer break) vs error (4xx); cached_tokens surfaced | logic-portable (needs span-collector seam) |
| `tests/test_structured_request.py` | contract rendering all three roots; container-on-array ValueError; success first try; correction retry convo shape; degrade after exhaustion; wrong root = shape failure; zero retries = 1 attempt; caller messages not mutated; fenced JSON ok; transport errors propagate; DEFAULT_MAX_RETRIES==1 | logic-portable |
| `tests/test_tools_registry.py` | register/get, KeyError, duplicate ValueError, OpenAI shape, empty→[], iteration, default timeout 10s, ImageResult fields, ToolContext defaults | logic-portable |
| `tests/test_agentic_loop.py` (645 loc) | leak guard (XML + namespaced + python-style, silent vs non-silent, think-tag stripping incl. combined, mid-prose untouched); termination w/o tools; empty registry → tools=None; tool exec + re-call transcript order; handler exception/unknown tool/bad-args → error results; max-iterations cap + warning; handler timeout; on_delta per chunk; on_iteration_end args | logic-portable (scripted stream mock) |
| `tests/test_attentional_tools.py` (941 loc) | ToolContext new-field defaults; shift_focus (shift_now called, ok JSON, preview from TARGET channel, preview limit == catch_up_limit, empty→[], missing fm/channel_id errors, unsubscribed rejection + available list, schema); silent sentinel; loop silent detection (is_silent, no on_iteration_end); read_channel (json shape, limit clamp 50, missing store/fm/focus errors, before_id paging, around_id half-split ≥1, mutual exclusion, schema); start_activity (enum from catalog, description budget + policy text, defer_start w/ note, engine error passthrough, already-out → silent, availability window rendering); builtins composition matrix incl. never-in-voice rules + describe_constraints binding | logic-portable (mock FocusManager/store/engine) |
| `tests/test_alarm_scheduler.py` | fires at scheduled time (row stamped, event published); past-due fires on start; cancel prevents fire; cancel unknown → False; payload carries originating_turn_id | logic-portable (real SQLite tmp DB, short sleeps — use tokio::time::pause) |
| `tests/test_alarm_tool.py` | delay/ISO insert paths; past-timestamp reject; missing reason reject; channel routed from ctx; cancel roundtrip; cancel unknown error | logic-portable |
| `tests/test_alarm_waker.py` | republish as discord.text with `[alarm fired: ...]`; `alarm: true` marker; waker stamps its OWN familiar_id (no payload-side filter) | logic-portable (needs in-process bus) |
| `tests/test_channel_view.py` | tool/empty-turn exclusion; verbatim passthrough; display_name > username > None; huge tool payload doesn't inflate; stable dict shape; recursion closed on re-ingest; idempotent | logic-portable |
| `tests/test_image_compress.py` | shrink oversized; RGBA handled; small image not upscaled; unreachable ceiling raises; GIF single-frame no tiling; animated tiles horizontally; fewer-than-max frames; outputs valid JPEG; base64 ascii | logic-portable (swap PIL for `image` crate; exact byte sizes differ — assert ceilings, not bytes) |
| `tests/test_image_describe.py` | vision block shape; custom media type; neutral base prompt; constraints appended; no trailing space when blank | logic-portable |
| `tests/test_image_serialization.py` | text-only vs multimodal serialization; tool_content_as_text projection; loop serializes per llm.multimodal; Message passes list content | logic-portable |
| `tests/test_image_tool.py` | tool schema; unknown id error; ImageResult returned; constraints flow to describe; missing description LLM degrades gracefully | needs-Rust-mock (patches `_fetch_image_bytes` / `describe_image`) |

Responder-side integration (`test_text_responder_tools.py`,
`test_voice_responder_tools.py`, worker tests) belongs to 06/07 but
exercises this subsystem's loop/stream contracts end-to-end.

## Rust port notes

- **Streaming generator → Stream.** `stream_completion` is an async
  generator whose `finally` is load-bearing (semaphore release already
  happened at header time, but metrics emit + body close run in finally,
  and cancelled-vs-error status is decided by the exception class). In
  Rust, model it as a struct implementing `Stream<Item =
  Result<LLMDelta, LlmError>>` whose `Drop` handles abandonment: emit
  metrics with status `cancelled` if the stream was dropped before a
  terminal event, `ok` after clean end, `error` after an Err. Do NOT
  rely on the consumer polling to completion. `chat_stream`'s explicit
  inner-`aclose` dance disappears — Drop composes.
- **Semaphore.** Replace the module global with an
  `Arc<tokio::sync::Semaphore>` injected into every client (built once at
  startup from config). Preserve the two release points exactly: `chat`
  holds a permit only across the POST (never across backoff sleeps);
  streaming drops the permit the moment response headers pass the status
  check. `tokio`'s `OwnedSemaphorePermit` + explicit `drop(permit)` maps
  cleanly. The Python lazy-init-then-replace dance is a global-state wart
  — do not port it; require injection.
- **AlarmScheduler tasks.** One `tokio::spawn` per pending alarm with a
  `JoinHandle` map guarded by a mutex; `tokio::time::sleep_until` for the
  wait. Keep the DB-first insert order and the conditional-update
  idempotence (`mark_alarm_fired` returning false suppresses publish) —
  that is the entire cancel/fire race story. `shutdown` = abort + await
  all handles; DB untouched. Tests should use `tokio::time::pause`.
- **Duck-typed seams.** `agentic_loop` reads `llm.multimodal` via
  `getattr` and `request_structured` reads `llm.slot` via `getattr` so
  mocks work; in Rust put `multimodal()` and `slot()` on the `LlmClient`
  trait. `ToolContext.scheduler: Any` and the lazy `SILENT_RESULT` import
  are cycle workarounds — use a concrete `Arc<AlarmScheduler>`
  (`Option`) and a shared constant.
- **Tool handler type.** `handler: async fn(serde_json::Value /*object*/,
  &ToolContext) -> ToolOutput` where
  `enum ToolOutput { Text(String), Image(ImageResult) }` replaces the
  `str | ImageResult` union. Errors are NOT Rust errors — handlers return
  `Ok(Text(json!({"error": ...})))`; only the loop's timeout/panic guard
  converts failures to error JSON. Keep sequential execution and
  per-tool `timeout_s` (`tokio::time::timeout`).
- **Callbacks.** `on_delta`/`on_before_tools`/`on_iteration_end` are
  async closures over responder state; in Rust prefer a
  `trait AgenticHooks` with default no-op methods over three boxed
  `Fn`s — the responders (06) implement it once.
- **Regexes.** Port the leak-guard patterns verbatim (`regex` crate;
  `(?s)` for DOTALL, `(?i)` for the python-style patterns). They are
  behavior-pinned by ~13 tests including namespaced invoke tags and
  think-block-before-leaked-silent ordering (think-strip FIRST).
- **JSON dynamism.** Streaming fragments, tool arguments, and
  `parameters` schemas are `serde_json::Value` territory — do not force
  static types on SSE chunks (providers vary). `Message.content` becomes
  `enum Content { Text(String), Blocks(Vec<Value>) }` with `content_str`
  as a method; preserve to_dict's omit-None behavior via
  `skip_serializing_if`.
- **Images.** Replace PIL with the `image` crate (`jpeg` encoder quality
  loop; `imageops::thumbnail`-style downscale preserving aspect;
  Lanczos3 filter for GIF frame scaling; `image::codecs::gif::GifDecoder`
  frame iteration). Exact output byte sizes WILL differ from PIL —
  conformance target is the ceilings/dimensions, not bytes. `optimize=
  True` has no direct equivalent; acceptable divergence.
- **HTTP.** `reqwest` with 120 s timeout for chat; streaming via
  `bytes_stream` + an SSE line splitter (or `eventsource-stream`). The
  view_image fetch wants `reqwest` streaming with the 20 MB cap enforced
  chunk-wise and redirect following ON (client default caps at 10).
- **Suggested crates**: `reqwest`, `tokio` (sync + time), `serde`/
  `serde_json`, `regex`, `image`, `base64`, `uuid`, `chrono` (ISO-8601
  with offset; reject naive datetimes in set_alarm), `futures`
  (`Stream`), `tracing` for the structured log lines + spans.
- **Redesign candidates**: (a) the module-global semaphore → injected
  handle (above); (b) `build_system_prompt`/`SystemPromptLayers` are
  legacy shims — port only if 05's assembler spec still references them,
  otherwise drop; (c) `_parse_sse_deltas` is a dead backward-compat
  helper — drop; (d) `no_stream` synthesis and `think_prepend` are
  model-family workarounds — keep behind client options but isolate them
  so removing a workaround is a one-line change; (e) the eager
  per-call `httpx.AsyncClient` in view_image could share a pooled client,
  but keep the 15 s timeout and redirect semantics either way.
