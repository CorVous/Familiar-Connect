# Metrics and profiling

How to measure, compare, and analyze per-turn performance for a familiar.

## Logging vs metrics — when to use which

Familiar-Connect has two observability systems. They answer different questions.

| | `logging` (stdlib) | `metrics` (TraceBuilder + collector) |
|---|---|---|
| **Purpose** | Operational observability: "what happened?" | Performance data: "how long? how much?" |
| **Audience** | Human watching the terminal right now | Developer analyzing trends later |
| **Format** | Free-text, human-readable | Structured `TurnTrace` records |
| **Lifetime** | Ephemeral (stderr/stdout) | Persisted to SQLite |
| **Queryable** | No (grep at best) | Yes (SQL, `familiar-connect metrics`) |
| **Use for** | Errors, warnings, status messages, debug info | Latency, throughput, token counts, A/B tags |

## The bridge — spans log for you

A `TraceBuilder.span` emits a `DEBUG` log line on exit and `TraceBuilder.finalize` emits an `INFO` summary. This means you never dual-write timing data.

At `-v` (INFO), one summary per turn:

```
INFO: metrics turn trace_id=a1b2... channel=12345 modality=text total=4.12s stages=6
```

At `-vv` (DEBUG), per-stage timing as it happens:

```
DEBUG: metrics span pipeline_assembly duration=0.847s [provider_outcomes=[...]]
DEBUG: metrics span llm_call duration=2.134s [model=z-ai/glm-5.1, reply_length=342]
```

## Developer rules of thumb

1. **Errors, warnings, status changes** → use `_logger.warning()` / `_logger.info()` directly.
2. **Timing anything** → wrap in `async with builder.span("name")`. It logs *and* persists.
3. **Never manually log durations** alongside a span. The span handles both — dual-writing is a smell.
4. **Adding a new pipeline stage?** Wrap it in a span. The log line and metrics record appear automatically.
5. **Want to compare variants?** `builder.tag("key", "value")` then `familiar-connect metrics --compare key`.

## Adding a span to new code

The canonical pattern lives in `src/familiar_connect/bot.py::_run_text_response`.
Copy it:

```python
from familiar_connect.metrics import TraceBuilder

builder = TraceBuilder(
    familiar_id=familiar.id,
    channel_id=channel_id,
    guild_id=guild_id,
    modality="text",
)
builder.tag("channel_mode", channel_config.mode.value)

async with builder.span("my_stage") as meta:
    result = await do_work()
    meta["items_processed"] = len(result)  # enrich after work finishes

# ... more spans ...

familiar.metrics_collector.record(builder.finalize())
```

`meta` is a mutable dict — assign keys inside the `with` block to attach
stage-specific data (tokens, model, status, anything). It ends up in
the persisted trace and the DEBUG log line.

## A/B testing with tags

Tags are top-level KV on the trace. Useful for comparing prompt variants,
model changes, or manual quality ratings.

```python
builder.tag("model_variant", "glm-5.1")
builder.tag("prompt_version", "v2.3")
```

Then query:

```console
$ familiar-connect metrics --familiar myfam --compare model_variant
```

Output groups traces by the tag value and shows a p50/p95/mean latency table.

## CLI reference

```console
# Summary + stage breakdown + provider success rates (last 100 traces)
$ familiar-connect metrics --familiar myfam

# Most recent 50 traces only
$ familiar-connect metrics --familiar myfam --last 50

# Filter by tag
$ familiar-connect metrics --familiar myfam --tag channel_mode=full_rp

# Filter to traces containing a specific stage
$ familiar-connect metrics --familiar myfam --stage tts

# A/B comparison
$ familiar-connect metrics --familiar myfam --compare prompt_version

# Histogram plot (requires matplotlib)
$ familiar-connect metrics --familiar myfam --plot
```

The metrics database lives at `data/familiars/<id>/metrics.db`. Deleting
it resets collected history with no loss of any functional state.

## Custom collectors

`MetricsCollector` is a `Protocol` — implement `record(trace)` and `close()`
to send traces anywhere. Point `familiar.metrics_collector` at your
implementation after `Familiar.load_from_disk` returns.

```python
class CSVCollector:
    def __init__(self, path: Path) -> None: ...
    def record(self, trace: TurnTrace) -> None: ...
    def close(self) -> None: ...
```

The built-in `SQLiteCollector` is the reference implementation; see
`src/familiar_connect/metrics/sqlite_collector.py`.

## Overhead

Spans cost ~1μs each (two `loop.time()` calls + a `list.append`). The
SQLite writes are batched (flush at 50 traces) and run in microseconds.
The metrics layer is invisible against the 500 ms – 30 s LLM call that
dominates a turn.
