# Voice interruption

When a user speaks for at least `min_interruption_s` while the familiar is
generating a response or playing one, it counts as an **interruption**. This
applies only to voice channels — text conversations are unaffected.

Back-channel sounds (brief "mm-hm", laughter) are filtered by the minimum-
duration threshold — no audio classification is needed.

Multiple people speaking in the same window are merged into one event.

---

## Configuration

### `character.toml`

```toml
[voice.interruption]
# Five named tiers or a raw float 0.0–1.0.
# very_meek = 0.10, meek = 0.20, average = 0.30 (default),
# stubborn = 0.45, very_stubborn = 0.60
interrupt_tolerance = "average"

# Minimum seconds of continuous user speech before it counts as an interruption.
# Shorter bursts ("mm-hm") are discarded.
min_interruption_s = 2.0

# Duration threshold separating short from long interruptions.
# Must exceed min_interruption_s.
short_long_boundary_s = 30.0
```

All three values live on `CharacterConfig` and are loaded at startup.
They describe the familiar's personality, not a channel's tuning, so they
are intentionally not per-channel.

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

`unsolicited_bias = +0.35` when the response was initiated as an
interjection (the familiar barged in without being addressed). Lulls and
direct addresses carry no bias. `average` tolerance + unsolicited bias
yields a 65% push-through probability.

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
| `IDLE` | Not generating or speaking. Interruption detection is off. |
| `GENERATING` | LLM call is in-flight. TTS has not started. |
| `SPEAKING` | TTS audio is playing via `vc.play()`. |

The tracker holds:

- `generation_task` — cancellable `asyncio.Task` wrapping the LLM call
- `response_text` — full reply text (set after generation completes)
- `timestamps` — `list[WordTimestamp]` from Cartesia (for word-boundary splits)
- `playback_start_time` — `time.monotonic()` at the `SPEAKING` transition
- `vc` — reference to the Discord `VoiceClient`
- `is_unsolicited` — whether the reply was initiated by the chattiness path
- `mood_modifier` — per-response float cached from `MoodEvaluator` at `GENERATING` entry
- `interruption_elapsed_ms` — ms since playback started when the familiar yielded
- `interrupt_event` — `asyncio.Event` set at burst finalization on a SPEAKING yield
- `interrupt_classification` — `InterruptionClass` populated at finalization on yield
- `interrupt_transcript` — interrupter's words (from burst transcript)
- `interrupt_starter_name` — display name of the user whose burst triggered the yield
- `pending_interrupter_turns` — `(name, text)` pairs from short@GENERATING bursts,
  flushed to history after the original buffer write
- `short_yield_pending` — True while a short@SPEAKING resume task is in flight;
  suppresses the concurrent voice endpointing lull

Every transition is logged at INFO:
```
tracker guild=999 state: IDLE→GENERATING (unsolicited=False)
tracker guild=999 state: GENERATING→SPEAKING (unsolicited=False)
tracker guild=999 state: SPEAKING→IDLE (unsolicited=False)
```

---

## Interruption detection

### Signal source

Deepgram interim and final `Results` messages feed
`DeepgramVoiceActivityDetector` (`src/familiar_connect/voice/deepgram_vad.py`),
which derives per-speaker speech-start / speech-end edges from recognized-word
arrivals. These callbacks drive `VoiceLullMonitor`, which re-exposes them via
`on_voice_activity`:

- `speech_started(user_id)` — first non-empty Deepgram interim or final for
  this speaker (DGVAD speech-start edge).
- `speech_ended(user_id)` — 700 ms silence watchdog expiry with no new
  Deepgram results (DGVAD watchdog), or Deepgram endpointed final for a
  speaking user.

`InterruptionDetector` subscribes to this hook. It ignores all events when
the tracker is `IDLE`.

### Burst accumulation

The detector accumulates a **burst** from the first `speech_started` while
state ≠ `IDLE` until all users have been quiet for `lull_timeout_s`. Burst
duration is wall-clock time from first-started to last-ended.

```
voice activity user=12345 event=started
interruption: min threshold crossed by user=12345 during GENERATING
voice activity user=12345 event=ended
interruption: short (3.20s) by user=12345 during GENERATING
```

The burst transcript is accumulated from Deepgram finals that arrive during
the active burst. `InterruptionDetector.on_transcript` is wired from
`_route_transcript_to_monitor` alongside the existing `VoiceLullMonitor` call.

### Classification

| Duration | Class | Behaviour |
|---|---|---|
| `< min_interruption_s` | *discarded* | Drop. No dispatch. |
| `≥ min_interruption_s`, `< short_long_boundary_s` | *short* | See dispatch matrix below |
| `≥ short_long_boundary_s` | *long* | See dispatch matrix below |

### State-upgrade rule

If a burst starts while state is `GENERATING` and the tracker transitions to
`SPEAKING` before the burst ends, the burst is upgraded to `SPEAKING` at
classification time.

---

## Dispatch matrix

Five paths, keyed on (classification × state-at-burst-start):

| Classification | State | Path | Log |
|---|---|---|---|
| *discarded* | any | Drop | `interruption: discarded (…s)` |
| *short* | `GENERATING` | Polite wait (see below) | `dispatch: short@GENERATING → polite-wait speaker=…` |
| *long* | `GENERATING` | Cancel + regen (see below) | `dispatch: long@GENERATING → cancel+regen user=…` |
| *short* | `SPEAKING` | Yield+resume or push-through (see below) | `dispatch: short@SPEAKING → yield+resume speaker=…` |
| *long* | `SPEAKING` | Yield + regen with context (see below) | `dispatch: long@SPEAKING → regen speaker=…` |

---

## Short@GENERATING — polite wait

The familiar keeps generating. A **delivery gate** (`asyncio.Event`) is
cleared when any burst starts and opened at finalize. `_run_voice_response`
awaits `detector.wait_for_lull()` between TTS synthesis and playback; it
returns `short` and playback proceeds normally.

The interrupter's transcript is stashed on
`tracker.pending_interrupter_turns` at finalize time. `_run_voice_response`
flushes them to history **after** the original buffer write so chronology
reads: original user turn → interrupter turn → assistant reply.

An early long-boundary timer also fires if the burst crosses
`short_long_boundary_s` during `GENERATING` before the lull — see the next
section.

---

## Long@GENERATING — cancel + regen

The long-boundary timer fires as soon as accumulated burst duration crosses
`short_long_boundary_s`. `_on_long_boundary_crossed` cancels
`tracker.generation_task` immediately — before the lull — so no tokens are
wasted on a reply that will be discarded.

At lull finalization, `on_long_during_generating` fires. It:

1. Schedules `dispatch_interruption_regen` as an `asyncio.Task`.
2. Sets `extras["_regen_pending"]` synchronously so the concurrent voice
   endpointing lull's `_deliver_to_monitor` call is suppressed.

`dispatch_interruption_regen` calls `_run_voice_response` with an
`interruption_context` note:

```
{speaker} interrupted while you were forming a response.
They said: "{transcript}"
```

History is not written for the cancelled generation.

---

## Moment 1 — tolerance roll (SPEAKING)

When a burst crosses `min_interruption_s` while the familiar is `SPEAKING`:

```
effective = clamp(base + mood_modifier + unsolicited_bias, 0, 1)
keep_talking = random() < effective
```

- **keep talking** → audio continues (push-through path)
- **yield** → `vc.stop()` called immediately; `interruption_elapsed_ms` and
  `_remaining_timestamps` captured; `interrupt_event` armed for post-playback
  coordination

```
toll: base=0.30 mood=+0.00 unsolicited=+0.00 effective=0.30 roll=0.71 → yield
```

---

## Short@SPEAKING — yield+resume or push-through

**Yield path** (keep_talking = False):

1. `vc.stop()` at Moment 1; `interrupt_event` set.
2. At finalize: `_on_short_yield_resume(remaining)` created as an
   `asyncio.Task`. `tracker.short_yield_pending = True` is set first so the
   concurrent voice endpointing lull is suppressed while the resume is pending.
3. Resume task: re-synthesizes `remaining` words and plays.
4. History records the **full** `tracker.response_text` (not just the
   delivered portion).

**Push-through path** (keep_talking = True):

1. Audio continues uninterrupted.
2. At finalize: `_on_push_through_transcript` writes the interrupter's
   transcript to history immediately so it is not lost.

---

## Long@SPEAKING — yield + regen with context

1. `vc.stop()` at Moment 1.
2. `_run_voice_response` awaits `tracker.interrupt_event` post-playback.
3. `tracker.interrupt_classification` is `long` → compute `delivered_text`
   from `split_at_elapsed(tracker.timestamps, elapsed_ms)`.
4. Write `delivered_text` to history as the interrupted assistant turn.
5. Re-generate with `interruption_context`:

```
You were speaking and said: "{delivered_text}". {name} interrupted you. They said: "{transcript}"
```

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

## Mood modifier

`MoodEvaluator.evaluate()` issues a short LLM prompt inspecting the last
six history turns and returns a float in `[−0.5, +0.5]`, cached on
`tracker.mood_modifier` at `GENERATING` entry.

Examples:
- Excited about a topic → positive modifier (talks over people more)
- Just got corrected → negative modifier (yields more readily)
- Asking a question → negative modifier (wants the answer)

Uses the `mood_eval` LLM slot (configured under `[llm.mood_eval]` in
`character.toml`).

---

## Key components

| Component | Location |
|---|---|
| `ResponseState` enum | `voice/interruption.py` |
| `ResponseTracker` | `voice/interruption.py` |
| `ResponseTrackerRegistry` | `voice/interruption.py` |
| `InterruptionDetector` | `voice/interruption.py` |
| `InterruptionClass` enum | `voice/interruption.py` |
| `split_at_elapsed` | `voice/interruption.py` |
| `MoodEvaluator` | `mood.py` |
| `InterruptTolerance` enum | `config.py` |
| Cancellable generation task | `bot.py` — `tracker.generation_task` |
| Delivery gate | `InterruptionDetector._delivery_gate` |
| `ContextRequest.interruption_context` | `context/types.py` |
| Interruption-context renderer | `context/render.py` |
| `WordTimestamp`, `TTSResult` | `tts.py` |

---

## Sequence diagrams

### Short@GENERATING — polite wait

```
User speaks (<boundary) ── detector ── short@GENERATING
                                             │
                              delivery gate cleared on burst start
                              (bot.py awaits gate before playback)
                                             │
User quiet ─────────────────────────────────►
                                             │
                              gate opens → interrupter turns stashed
                              playback proceeds as normal
```

### Long@GENERATING — cancel + regen

```
User speaks (>boundary) ── long-boundary timer fires
                                             │
                              generation_task.cancel()
                                             │
User quiet ─── capture transcript ──────────►
                                             │
                              build ContextRequest (interruption_context set)
                                             │
                                   new LLM call → TTS + play
```

### Short@SPEAKING — yield + resume

```
Familiar playing ─── vc.play(audio) ──────────────────────────►
                                                               │
User speaks (>min, <boundary) ─── roll: yield ─────────────────►
                                                               │
                                               vc.stop()
                                               elapsed_ms captured
                                               short_yield_pending = True
                                                               │
User quiet ─────────────────────────────────────────────────────►
                                                               │
                               split_at_elapsed → remaining
                               re-synthesize + vc.play(remaining)
                               full response_text → history
```

### Long@SPEAKING — yield + regen

```
Familiar playing ─── vc.play(audio) ──────────────────────────►
                                                               │
User speaks (>boundary) ─── roll: yield ──────────────────────►
                                                               │
                                               vc.stop()
                                               interrupt_event armed
                                                               │
User quiet ─── capture transcript ──────────────────────────────►
                                                               │
                               delivered_text → history
                               build ContextRequest (delivered + transcript)
                                                               │
                                               LLM + TTS + play
```
