# Voice Interruption Flow

!!! success "Status: Partially shipped — Step 8 (GENERATING dispatch) live; Steps 11–12 pending"
    The state machine, interruption detector, tolerance system, Cartesia
    streaming TTS, and cancellable generation task are all in production
    code. Interruption-during-GENERATING now dispatches via a unified
    pause-then-commit mechanism (Step 8, absorbing former Step 9): short
    interruptions pause delivery until silence resumes, long interruptions
    cancel generation and regenerate carrying the original pending turns
    plus the interrupter's transcript forward. SPEAKING-state dispatch
    (Steps 11, 12) remains pending. See
    [Implementation status](#implementation-status) below.

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
| 8 (plumbing) | `context/types.py` + `context/render.py` | `ContextRequest.interruption_context: str \| None` field. Renderer inserts it as a `system` message before the final user turn when non-empty. Reserved for Step 12; not populated by Step 8 dispatch. |
| 8 (dispatch) | `voice/interruption.py` + `bot.py` | Unified pause-then-commit during `GENERATING`. At `min_interruption_s` crossed, `tracker.delivery_gate` clears so TTS blocks. On burst finalize-as-short, gate is set → original reply delivered. On `short_long_boundary_s` crossed mid-burst, `_commit_long()` cancels `generation_task`, flips `cancel_committed`, and dispatches `dispatch_interruption_regen` which rebuilds a `ContextRequest` carrying the original `pending_turns` plus a trailing `PendingTurn` from the interrupter. History writes are skipped on the cancelled turn. Step 9 absorbed. |
| 10 | `voice/interruption.py` — `split_at_elapsed` + playback stamp | `transition(SPEAKING)` stamps `playback_start_time`. `split_at_elapsed(timestamps, elapsed_ms) → (delivered, remaining)` partitions word timestamps at a word boundary. Neither is called yet. |

### Stubs — interface live, real logic pending

| Stub | Location | Behaviour today | Replaced in |
|---|---|---|---|
| `MoodEvaluator.evaluate()` | `mood.py` | Always returns `0.0`. Logged as `mood_modifier=0.00 (stub)`. | Step 13 — real side-model LLM call |
| Tolerance-roll dispatch | `voice/interruption.py` `InterruptionDetector._on_min_crossed` | Roll is computed and logged (`toll: base=… → keep_talking/yield`) but the result drives no action. | Step 11 — yield path wired |
| `ContextRequest.interruption_context` | `context/types.py` | Field exists; renderer handles it. Reserved for Step 12 (long-during-SPEAKING). Step 8 does not use it — carries interrupter transcript as a `PendingTurn` instead. | Step 12 |
| `split_at_elapsed` | `voice/interruption.py` | Helper is implemented and unit-tested. No caller uses it. | Step 11 |

### Pending (behavior-changing)

| Step | What changes |
|---|---|
| ~~9~~ | Absorbed into Step 8 (pause-then-commit covers both short and long during `GENERATING`). |
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

Step 8 handles both short and long via a unified pause-then-commit gate.

### Short interruption (Step 8 — shipped)

- Do **not** cancel the generation — let it finish.
- At `min_interruption_s`, `tracker.delivery_gate` is cleared so the
  post-generation `await tracker.delivery_gate.wait()` blocks TTS.
- On burst finalize-as-short, the gate is set → the originally-generated
  reply is delivered unchanged.
- Effect: the familiar pauses politely, then speaks as if nothing happened.

### Long interruption (Step 8 — shipped)

- At `short_long_boundary_s` crossed mid-burst, `_commit_long()` fires:
  - Cancels `tracker.generation_task`.
  - Sets `tracker.cancel_committed = True`.
  - Sets the delivery gate (awaiter unblocks, sees the flag, returns
    without TTS or history writes).
  - Dispatches `dispatch_interruption_regen(...)` on the event loop.
- The dispatcher:
  - Reads `tracker.pending_buffer` (original user turns stashed at
    `GENERATING` entry).
  - Builds a new `ContextRequest` with `pending_turns` equal to the
    original buffer **plus** a trailing `PendingTurn` from the
    interrupter carrying the accumulated transcript.
  - Calls the shared `_run_voice_response_with_request` helper which
    generates, synthesizes, and plays the replacement reply.
- Effect: no tokens lost — original user messages plus the interruption
  appear in the regen request as contiguous user speech. The LLM sees a
  uniform conversation view; `interruption_context` is **not** used here
  (it is reserved for Step 12's long-during-SPEAKING case, where framing
  about a cut-off assistant reply is actually needed).

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

### Long interruption + yielded (Step 12 — pending)

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
User speaks (>boundary_s) ─── _commit_long fires mid-burst
                                                   │
                                    generation_task.cancel()
                                    cancel_committed = True
                                    delivery_gate.set()
                                                   │
                        dispatch_interruption_regen()
                                                   │
                          build ContextRequest with:
                           • original pending_turns (preserved)
                           • trailing PendingTurn from interrupter
                                                   │
                                           new LLM call
                                                   │
                                             TTS + play
```

### Interrupted during generation (short) — Step 8 pause

```
User speaks (≥min, <boundary) ─── _maybe_log_min_crossed
                                        delivery_gate.clear()
                                                   │
LLM completes ─── await delivery_gate.wait() (blocks)
                                                   │
User quiet ─── _finalize_burst classifies short
                                        delivery_gate.set()
                                                   │
                                  await returns → TTS + play
                                  (original reply, unchanged)
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
