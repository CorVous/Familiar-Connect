# Message flow

End-to-end path from a Discord message arriving to a reply being sent.
Covers the **text** path; the voice path differs at the edges (audio
capture, TTS playback, interruption-aware generation task) but uses
the same context pipeline and `ConversationMonitor` gate.

## Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Discord
    participant Bot as bot.py<br/>on_message
    participant Monitor as ConversationMonitor<br/>chattiness.py
    participant Buf as ChannelBuffer
    participant Pipeline as ContextPipeline
    participant LLM as main_prose<br/>LLMClient
    participant Post as Post-processors
    participant Store as HistoryStore

    Discord->>Bot: message event
    Bot->>Monitor: on_message(channel, author, text, is_mention)
    Monitor->>Buf: append BufferedMessage
    Monitor->>Monitor: cancel existing lull timer

    alt direct address (name / alias / @mention)
        Monitor->>Buf: acquire lock
        Monitor->>LLM: _evaluate(direct_address_prompt)
        Note right of LLM: logged as<br/>interjection channel=X trigger=direct_address decision=YES|NO
        Monitor->>Monitor: reset buffer (always, regardless of result)
    else message_counter >= next_interjection_at
        Monitor->>Buf: acquire lock
        Monitor->>LLM: _evaluate(interjection_prompt)
        alt decision=NO
            Monitor->>Buf: check_count++; next_interjection_at += shrunk interval
        end
    else text path (no direct address, below threshold)
        Monitor->>Monitor: start lull timer (call_later)
        Note right of Monitor: timer expiry ⇒ evaluate with lull_prompt
    end

    alt decision=YES
        Monitor->>Monitor: _fire_respond: snapshot buffer, reset state
        Monitor->>Bot: on_respond(channel_id, snapshot, trigger)
        Bot->>Discord: async with channel.typing()
        Bot->>Pipeline: assemble(request, budget_by_layer)
        Note right of Pipeline: providers fan out via TaskGroup<br/>(character, history, content_search,<br/>mode_instructions)
        Pipeline-->>Bot: PipelineOutput
        Bot->>LLM: chat(messages)
        LLM-->>Bot: reply
        Bot->>Post: run_post_processors(reply, request)
        Post-->>Bot: reply_text (typing scope ends here)
        Bot->>Store: append_turn (user buffer then assistant reply)
        Bot->>Discord: channel.send(reply_text)
    end
```

## Key locations

| Step | File | Lines |
|------|------|-------|
| `on_message` entry | `src/familiar_connect/bot.py` | 1078–1096 |
| Buffer append + lull timer cancel | `src/familiar_connect/chattiness.py` | 226–235 |
| Direct-address evaluation | `src/familiar_connect/chattiness.py` | 237–256 |
| Interjection evaluation | `src/familiar_connect/chattiness.py` | 258–275 |
| Lull timer start | `src/familiar_connect/chattiness.py` | 285–318 |
| Decision log line (`interjection channel=… trigger=… decision=…`) | `src/familiar_connect/chattiness.py` | 402–409 |
| `_fire_respond` (snapshot + reset) | `src/familiar_connect/chattiness.py` | 412–421 |
| Pipeline assemble | `src/familiar_connect/context/pipeline.py` | 80–120 |
| Provider fan-out | `src/familiar_connect/context/pipeline.py` | 144–165 |
| `channel.typing()` scope (text) | `src/familiar_connect/bot.py` | 949 |
| `channel.send` | `src/familiar_connect/bot.py` | 1049 |

## Notes on the current shape

- `buf.lock` is held only during evaluation and `_fire_respond`. The
  LLM call for `main_prose` runs **outside** the lock, so a second
  message arriving during generation lands on the fresh buffer and is
  evaluated next turn.
- Typing indicator spans pipeline assembly + LLM + post-proc
  (`async with channel.typing()` at `bot.py:949`), so the indicator
  appears immediately after `decision=YES` instead of several seconds
  later when the LLM call finally starts.
- `channel.send` runs **outside** the typing scope — Discord's
  indicator clears as the reply is handed off.
- Voice path replaces `channel.send` / `channel.typing()` with a TTS
  synth + playback sequence, and wraps the `main_prose` call in an
  interruption-aware `asyncio.Task` parked on a per-guild tracker.
  See `bot.py:_run_voice_response` (≈ 132–471) and
  [interruption flow](interruption.md).
- Every span is wrapped by a `TraceBuilder` from
  `src/familiar_connect/metrics/timing.py`, so each boundary emits a
  DEBUG `stage=… duration_ms=…` line and contributes to a per-turn
  `TurnTrace` persisted under `data/familiars/<id>/metrics.db`. See
  [metrics guide](../guides/metrics.md).
