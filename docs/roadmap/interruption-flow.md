# Voice Interruption Flow

!!! success "Status: Partially shipped — plumbing + detection complete; dispatch pending"
    The state machine, interruption detector, tolerance system, Cartesia
    streaming TTS, and cancellable generation task are all in production
    code. Interruptions are classified and logged but **no dispatch
    happens yet** — the bot's voice behaviour is identical to before
    this work began. The next integration-test stops (Steps 8, 9, 11,
    12) will enable the actual yield / cancel / resume paths one at a
    time. See [Implementation status](#implementation-status) below for
    what is live, what is a stub, and what is still pending.

When a user speaks for at least `min_interruption_s` while the familiar
is either generating a response or speaking, it counts as an
**interruption**. This feature applies only to voice channels — text
conversations are unaffected.

Back-channel sounds (brief "mm-hm", laughter, etc.) are naturally
filtered out by the minimum-duration threshold — no audio classification
is needed.

Multiple people speaking during the same interruption window are merged
into a single event.

---

## Implementation status

### Done

| Step | Component | What it does |
|---|---|---|
| 1 | `config.py` — `InterruptTolerance` + `CharacterConfig` fields | Parses `[voice.interruption]` TOML; exposes `interrupt_tolerance`, `min_interruption_s`, `short_long_boundary_s` at startup; logs loaded values. |
| 2 | `tts.py` — Cartesia streaming WebSocket | `synthesize()` returns `TTSResult(audio, timestamps)` with per-word `WordTimestamp` objects. Timestamps are logged on every synth call. |
| 3 | `voice/interruption.py` — `ResponseTracker` scaffold | Per-guild state machine (`IDLE → GENERATING → SPEAKING → IDLE`). Lifecycle logged on every transition. `is_unsolicited` flag distinguishes direct-address from chattiness-triggered replies. |
| 4 | `voice_lull.py` — voice-activity callbacks | `VoiceLullMonitor.on_voice_activity` hook fires `speech_started` / `speech_ended` per user; `InterruptionDetector` subscribes via this hook. |
| 5 | `voice/interruption.py` — `InterruptionDetector` | Watches voice-activity events while state ≠ `IDLE`. Accumulates burst duration, classifies as *discarded* / *short* / *long*, logs result with state + starter user. No dispatch yet. |
| 6 | `mood.py` — `MoodEvaluator` stub + tolerance roll | `ResponseTracker.compute_effective_tolerance` and `should_keep_talking` methods. Tolerance roll is **logged** at min-threshold crossing during `SPEAKING`. No action on the result yet. |
| 7 | `bot.py` — cancellable LLM task | Voice `main_prose` chat call wrapped as `asyncio.Task` parked on `tracker.generation_task`. `CancelledError` is caught and returns cleanly to `IDLE`. No caller cancels yet. |
| 8 (plumbing) | `context/types.py` + `context/render.py` | `ContextRequest.interruption_context: str \| None` field. Renderer inserts it as a `system` message before the final user turn when non-empty. Nothing populates it yet. |
| 10 | `voice/interruption.py` — `split_at_elapsed` + playback stamp | `transition(SPEAKING)` stamps `playback_start_time`. `split_at_elapsed(timestamps, elapsed_ms) → (delivered, remaining)` partitions word timestamps at a word boundary. Neither is called yet. |

### Stubs — interface live, real logic pending

| Stub | Location | Behaviour today | Replaced in |
|---|---|---|---|
| `MoodEvaluator.evaluate()` | `mood.py` | Always returns `0.0`. Logged as `mood_modifier=0.00 (stub)`. | Step 13 — real side-model LLM call |
| Tolerance-roll dispatch | `voice/interruption.py` `InterruptionDetector._on_min_crossed` | Roll is computed and logged (`toll: base=… → keep_talking/yield`) but the result drives no action. | Step 11 — yield path wired |
| `tracker.generation_task.cancel()` | `bot.py` | Task exists and responds correctly to cancellation, but no code calls `.cancel()` yet. | Step 8 dispatch |
| `ContextRequest.interruption_context` | `context/types.py` | Field exists; renderer handles it. No caller populates it. | Steps 8 + 12 |
| `split_at_elapsed` | `voice/interruption.py` | Helper is implemented and unit-tested. No caller uses it. | Step 11 |

### Pending (behavior-changing)

| Step | What changes |
|---|---|
| 8 dispatch | Long interruption during `GENERATING` → `generation_task.cancel()` + rebuild `ContextRequest` with `interruption_context` + re-generate. **STOP gate — integration test required.** |
| 9 | Short interruption during `GENERATING` → gate delivery on `asyncio.Event`; deliver original response once user is quiet. **STOP gate.** |
| 11 | Short interruption during `SPEAKING` → yield path (`vc.stop()` + re-synth remaining) and push-through path (audio continues). **STOP gate.** |
| 12 | Long interruption during `SPEAKING` → yield path: record `delivered_text`, build new `ContextRequest`, re-generate. **STOP gate.** |
| 13 | `MoodEvaluator` real implementation — short LLM prompt inspecting recent turns; result in `[−0.5, +0.5]`. **STOP gate.** |
| 14 | Docs + final polish. Move this page to `docs/architecture/` once fully shipped. |

---

## Configuration

### `character.toml`

```toml
[voice.interruption]
# Five named tiers or a raw float 0.0–1.0.
# very_meek = 0.10, meek = 0.20, average = 0.30 (default),
# stubborn = 0.45, very_stubborn = 0.60
interrupt_tolerance = "average"

# Minimum seconds of continuous user speech before it counts as an
# interruption. Shorter bursts ("mm-hm") are discarded.
min_interruption_s = 2.0

# Duration threshold separating short from long interruptions.
# Must exceed min_interruption_s.
short_long_boundary_s = 30.0
```

All three values live on `CharacterConfig` and are loaded at startup. They
describe the familiar's personality, not a channel's tuning, so they are
intentionally not per-channel.

### Tolerance tiers

| Tier | Base probability |
|---|---|
| `very_meek` | 0.10 |
| `meek` | 0.20 |
| `average` | 0.30 *(default)* |
| `stubborn` | 0.45 |
| `very_stubborn` | 0.60 |

The probability is the chance the familiar **keeps talking** when interrupted
while speaking. It is combined with the mood modifier and an unsolicited bias
before the roll:

```
effective = clamp(base + mood_modifier + unsolicited_bias, 0, 1)
keep_talking = random() < effective
```

`unsolicited_bias = +0.35` when the response was initiated as a
chattiness/interjection (unsolicited) reply. Unsolicited remarks are more
committed — even `average` tolerance + unsolicited bias yields a 65% push-
through probability.

---

## Response state machine

A per-guild `ResponseTracker` tracks the familiar's current phase:

```
IDLE ──→ GENERATING ──→ SPEAKING ──→ IDLE
              │              │
         (interrupted)  (interrupted)
```

### `ResponseState` enum

| State | Meaning |
|---|---|
| `IDLE` | Not generating or speaking. Interruption detection is disabled. |
| `GENERATING` | LLM call is in-flight. TTS has not started. |
| `SPEAKING` | TTS audio is playing via `vc.play()`. |

The tracker also holds:

- `generation_task` — the cancellable `asyncio.Task` wrapping the LLM call
- `response_text` — the full reply text (set after generation completes)
- `timestamps` — `list[WordTimestamp]` from Cartesia (for word-boundary splits)
- `playback_start_time` — `time.monotonic()` at the `SPEAKING` transition
- `vc` — reference to the Discord `VoiceClient`
- `is_unsolicited` — whether the reply was initiated by the chattiness path
- `mood_modifier` — per-response float cached from `MoodEvaluator` at `GENERATING` entry

Every transition is logged at INFO:
```
tracker guild=999 state: IDLE→GENERATING (unsolicited=False)
tracker guild=999 state: GENERATING→SPEAKING (unsolicited=False)
tracker guild=999 state: SPEAKING→IDLE (unsolicited=False)
```

---

## Interruption detection

### Signal source

Discord audio frames flow through `VoiceLullMonitor`, which already tracks
per-user speaking state. Two observable events are exposed via an
`on_voice_activity` callback hook:

- `speech_started(user_id)` — first audio frame after a quiet period.
- `speech_ended(user_id)` — silence watchdog fired (user quiet for `user_silence_s`).

The `InterruptionDetector` subscribes to this hook. It ignores all events
when the tracker is `IDLE`.

### Burst accumulation

The detector accumulates a **burst** from the first `speech_started` while
state ≠ `IDLE` until all users have been quiet for `lull_timeout_s`. Burst
duration is wall-clock time from first-started to last-ended.

Logging:
```
voice activity user=12345 event=started
interruption: min threshold crossed by user=12345 during GENERATING
voice activity user=12345 event=ended
interruption: short (3.20s) by user=12345 during GENERATING
```

### Classification

| Duration | Class | Action (current) | Action (shipped) |
|---|---|---|---|
| `< min_interruption_s` | *discarded* | Log and drop | No change |
| `≥ min_interruption_s`, `< short_long_boundary_s` | *short* | Log | Steps 9 + 11 |
| `≥ short_long_boundary_s` | *long* | Log | Steps 8 + 12 |

### State-upgrade rule

If a burst starts while state is `GENERATING` and the tracker transitions
to `SPEAKING` before the burst ends, the burst is **upgraded** to `SPEAKING`
at classification time. A burst that began during `IDLE` and carried into
`GENERATING` counts from the first frame that overlaps `GENERATING`.

---

## Scenario 1: interrupted during generation

The LLM call is in-flight. A user starts talking.

### Short interruption (Step 9 — pending)

- Do **not** cancel the generation — let it finish.
- Gate delivery on an `asyncio.Event` set when silence resumes.
- Once quiet, proceed to TTS + playback as normal.
- Effect: the familiar pauses politely, then speaks as if nothing happened.

### Long interruption (Step 8 dispatch — pending)

- Cancel `tracker.generation_task`.
- Capture the interruption transcript.
- Set `ContextRequest.interruption_context`:
  ```
  {speaker} interrupted while you were forming a response.
  They said: "{transcript}"
  ```
- Regenerate. The reply now accounts for the new context.

---

## Scenario 2: interrupted while speaking

TTS audio is playing. A user starts talking.

### Moment 1 — tolerance roll (logged, not dispatched yet)

At `min_interruption_s` after first `speech_started`:

```
effective = clamp(base + mood_modifier + unsolicited_bias, 0, 1)
keep_talking = random() < effective
```

Log line today:
```
toll: base=0.30 mood=+0.00 unsolicited=+0.00 effective=0.30 roll=0.71 → yield
```

- **keep talking** → continue playback (Step 11 push-through path)
- **yield** → `vc.stop()`; record `elapsed_ms` from `playback_start_time` (Step 11)

### Short interruption + yielded (Step 11 — pending)

- Compute `delivered, remaining = split_at_elapsed(timestamps, elapsed_ms)`.
- Once user is quiet, re-synthesize `remaining` and resume playback.
- History records the **full** response text (not just the delivered portion).

### Long interruption + yielded (Step 12)

- Compute `delivered_text` from word timestamps.
- Build a new `ContextRequest`:
  ```
  You were speaking and said: "{delivered_text}"
  {speaker} interrupted you. They said: "{transcript}"
  ```
- Re-generate. History records only `delivered_text` for the interrupted turn.

---

## Word-boundary split

`split_at_elapsed(timestamps, elapsed_ms)` partitions the `list[WordTimestamp]`
returned by Cartesia. A word is treated as **delivered** once its `start_ms`
has passed — even if playback was cut mid-word — so resumption always begins
on a clean boundary with no stutter.

```python
delivered, remaining = split_at_elapsed(tracker.timestamps, elapsed_ms)
delivered_text = " ".join(w.word for w in delivered)
remaining_text = " ".join(w.word for w in remaining)
```

---

## Mood modifier (stub)

`MoodEvaluator.evaluate()` returns `0.0` today. The real implementation
(Step 13) will issue a short LLM prompt inspecting recent history and return
a value in `[−0.5, +0.5]` cached on the tracker at `GENERATING` entry.

Examples of drift:
- Excited about a topic → positive modifier (talks over people more)
- Just got corrected → negative modifier (yields more readily)
- Asking a question → negative modifier (wants the answer)

---

## New components

| Component | Location | Status |
|---|---|---|
| `ResponseState` enum | `voice/interruption.py` | Shipped |
| `ResponseTracker` | `voice/interruption.py` | Shipped |
| `InterruptionDetector` | `voice/interruption.py` | Shipped (classify only) |
| `ResponseTrackerRegistry` | `voice/interruption.py` | Shipped |
| `split_at_elapsed` | `voice/interruption.py` | Shipped (unused pending Step 11) |
| `MoodEvaluator` | `mood.py` | Stub (Step 13) |
| `InterruptTolerance` enum | `config.py` | Shipped |
| `interrupt_tolerance`, `min_interruption_s`, `short_long_boundary_s` | `config.py` | Shipped |
| `WordTimestamp`, `TTSResult` | `tts.py` | Shipped |
| Cancellable generation task | `bot.py` | Shipped (no callers yet) |
| `ContextRequest.interruption_context` | `context/types.py` | Shipped (no callers yet) |
| Interruption-context renderer | `context/render.py` | Shipped (no callers yet) |

---

## Sequence diagrams

### Interrupted during generation (long) — Step 8 dispatch

```
User speaks (>boundary_s) ─── detector ─── long@GENERATING
                                                   │
                                    generation_task.cancel()
                                                   │
User quiet ─── capture transcript ─────────────────►
                                                   │
                                    build ContextRequest
                                    (interruption_context set)
                                                   │
                                           new LLM call
                                                   │
                                             TTS + play
```

### Interrupted while speaking (short, yielded) — Step 11

```
Familiar playing ─── vc.play(audio) ──────────────────────►
                                                           │
User speaks (>min, <boundary) ─── roll: yield ─────────────►
                                                           │
                                              vc.stop()
                                              elapsed_ms captured
                                                           │
User quiet ─────────────────────────────────────────────────►
                                                           │
                                  split_at_elapsed → remaining
                                  re-synthesize + vc.play(remaining)
```

### Interrupted while speaking (long, yielded) — Step 12

```
Familiar playing ─── vc.play(audio) ──────────────────────►
                                                           │
User speaks (>boundary) ─── roll: yield ────────────────────►
                                                           │
                                              vc.stop()
                                              delivered_text captured
                                                           │
User quiet ─── capture transcript ──────────────────────────►
                                                           │
                                    build ContextRequest
                                    (delivered + interruption_context)
                                                           │
                                               LLM + TTS + play
```
