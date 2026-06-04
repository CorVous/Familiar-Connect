"""Smoke test for attentional stream (issue #107).

Uses in-memory stores + real OpenRouter API. Does NOT touch any
existing familiars on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime

# ── bootstrap path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))

from familiar_connect.history.store import HistoryStore
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.subscriptions import SubscriptionRegistry, SubscriptionKind
from familiar_connect.focus import FocusManager
from familiar_connect.tools.shift_focus import build_shift_focus_tool
from familiar_connect.tools.silent import build_silent_tool, SILENT_RESULT
from familiar_connect.tools.read_channel import build_read_channel_tool
from familiar_connect.tools.registry import ToolRegistry, ToolContext
from familiar_connect.llm import LLMClient, Message
from familiar_connect.tools.loop import agentic_loop

FAMILIAR_ID = "smoke-test"
TEXT_CHANNEL = 111
OTHER_CHANNEL = 222
VOICE_CHANNEL = 333

results: list[dict] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "status": status, "detail": detail})
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}", f"— {detail}" if detail else "")


async def run_smoke_tests() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print("\n=== Attentional Stream Smoke Tests ===\n")

    # ── 1. In-memory store + schema ─────────────────────────────────────────
    print("1. Store layer")
    store = HistoryStore(":memory:")
    astore = AsyncHistoryStore(store)

    t1 = store.append_turn(
        familiar_id=FAMILIAR_ID, channel_id=TEXT_CHANNEL,
        role="user", content="hello from focused", consumed=True,
    )
    check("append_turn (consumed)", t1.consumed_at is not None,
          f"consumed_at={t1.consumed_at}")

    t2 = store.append_turn(
        familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL,
        role="user", content="hello from other channel", consumed=False,
    )
    check("append_turn (staged)", t2.consumed_at is None,
          f"consumed_at={t2.consumed_at}")

    cross = store.recent_cross_channel(familiar_id=FAMILIAR_ID, limit=10)
    check("recent_cross_channel only consumed", len(cross) == 1 and cross[0].id == t1.id,
          f"got {len(cross)} turn(s)")

    staged = store.staged_channels(familiar_id=FAMILIAR_ID)
    check("staged_channels reports OTHER_CHANNEL", OTHER_CHANNEL in staged,
          f"staged={staged}")

    promoted = store.promote_staged_turns(familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL)
    check("promote_staged_turns", promoted == 1, f"promoted={promoted}")

    cross2 = store.recent_cross_channel(familiar_id=FAMILIAR_ID, limit=10)
    check("cross-channel after promote has 2 turns", len(cross2) == 2,
          f"got {len(cross2)} turn(s)")

    store.set_focus_pointers(
        FAMILIAR_ID, text_channel_id=TEXT_CHANNEL, voice_channel_id=VOICE_CHANNEL
    )
    fp = store.get_focus_pointers(FAMILIAR_ID)
    check("focus_pointers round-trip",
          fp is not None and fp.text_channel_id == TEXT_CHANNEL,
          f"text_channel_id={fp and fp.text_channel_id}")

    # ── 2. FocusManager ─────────────────────────────────────────────────────
    print("\n2. FocusManager")
    import tempfile, pathlib, tomllib
    with tempfile.TemporaryDirectory() as tmp:
        subs_path = pathlib.Path(tmp) / "subs.toml"
        subs_path.write_text(
            "[[subscription]]\nchannel_id = 111\nkind = \"text\"\n\n"
            "[[subscription]]\nchannel_id = 333\nkind = \"voice\"\n"
        )
        subs = SubscriptionRegistry(subs_path)
        check("kind_for text channel", subs.kind_for(TEXT_CHANNEL) is SubscriptionKind.text)
        check("kind_for voice channel", subs.kind_for(VOICE_CHANNEL) is SubscriptionKind.voice)
        check("kind_for unknown", subs.kind_for(999) is None)

        store2 = HistoryStore(":memory:")
        astore2 = AsyncHistoryStore(store2)
        fm = FocusManager(familiar_id=FAMILIAR_ID, store=astore2, subscriptions=subs)
        await fm.initialize()

        fm.set_focus_immediately(TEXT_CHANNEL, "text")
        check("set_focus_immediately", fm.get_focus("text") == TEXT_CHANNEL)
        check("is_focused(TEXT_CHANNEL)", fm.is_focused(TEXT_CHANNEL))
        check("is_focused(OTHER_CHANNEL) = False", not fm.is_focused(OTHER_CHANNEL))

        # stage a turn, then defer shift + end_turn
        store2.append_turn(
            familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL,
            role="user", content="staged msg", consumed=False,
        )
        fm.defer_shift(OTHER_CHANNEL)
        await fm.end_turn()
        check("end_turn promotes staged turns",
              store2.count_staged(familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL) == 0)
        check("focus pointer updated after end_turn", fm.get_focus("text") == OTHER_CHANNEL)

    # ── 3. Tools ────────────────────────────────────────────────────────────
    print("\n3. Tools")
    check("SILENT_RESULT constant", SILENT_RESULT == "__SILENT__")

    silent_tool = build_silent_tool()
    ctx_bare = ToolContext(
        familiar_id=FAMILIAR_ID, channel_id=TEXT_CHANNEL,
        channel_kind="text", turn_id="t1",
        history=astore, bus=None, scheduler=None,
    )
    result = await silent_tool.handler({"reasoning": "testing silence"}, ctx_bare)
    check("silent tool returns SILENT_RESULT", result == SILENT_RESULT)

    # ── 4. OpenRouter API — silent tool via agentic loop ────────────────────
    print("\n4. OpenRouter API (live)")
    llm = LLMClient(
        api_key=api_key,
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        tool_calling=True,
    )
    registry = ToolRegistry()
    registry.register(build_silent_tool())

    messages: list[Message] = [
        Message(role="system", content=(
            "You are a helpful assistant. "
            "When asked to stay silent, call the silent() tool."
        )),
        Message(role="user", content=(
            "Please stay silent — call the silent tool with reasoning='user asked me to'."
        )),
    ]
    t0 = time.perf_counter()
    try:
        result = await agentic_loop(llm=llm, messages=messages, registry=registry, ctx=ctx_bare)
        elapsed = time.perf_counter() - t0
        check("agentic_loop detects silent tool", result.is_silent,
              f"is_silent={result.is_silent}, elapsed={elapsed:.2f}s")
    except Exception as e:
        check("agentic_loop silent tool", False, f"exception: {e}")

    # ── 5. OpenRouter API — shift_focus tool ────────────────────────────────
    print("\n5. OpenRouter API — shift_focus")
    with tempfile.TemporaryDirectory() as tmp2:
        subs_path2 = pathlib.Path(tmp2) / "subs.toml"
        subs_path2.write_text(
            "[[subscription]]\nchannel_id = 222\nkind = \"text\"\n"
        )
        subs2 = SubscriptionRegistry(subs_path2)
        store3 = HistoryStore(":memory:")
        astore3 = AsyncHistoryStore(store3)
        fm2 = FocusManager(familiar_id=FAMILIAR_ID, store=astore3, subscriptions=subs2)
        fm2.set_focus_immediately(TEXT_CHANNEL, "text")

        registry2 = ToolRegistry()
        registry2.register(build_shift_focus_tool())
        registry2.register(build_silent_tool())

        ctx2 = ToolContext(
            familiar_id=FAMILIAR_ID, channel_id=TEXT_CHANNEL,
            channel_kind="text", turn_id="t2",
            history=astore3, bus=None, scheduler=None,
            focus_manager=fm2,
        )
        messages2: list[Message] = [
            Message(role="system", content=(
                "You have a shift_focus tool. When asked to switch channels, call it. "
                "After calling shift_focus, reply with a brief acknowledgement."
            )),
            Message(role="user", content="Please shift focus to channel 222."),
        ]
        t0 = time.perf_counter()
        try:
            result2 = await agentic_loop(llm=llm, messages=messages2, registry=registry2, ctx=ctx2)
            elapsed = time.perf_counter() - t0
            check("shift_focus tool called (deferred shift registered)",
                  fm2._pending_shift.get("text") == OTHER_CHANNEL or fm2.get_focus("text") == OTHER_CHANNEL,
                  f"pending={fm2._pending_shift}, focus={fm2.get_focus('text')}, elapsed={elapsed:.2f}s")
        except Exception as e:
            check("shift_focus tool", False, f"exception: {e}")

    await llm.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    return results


if __name__ == "__main__":
    loop_results = asyncio.run(run_smoke_tests())
    failed = [r for r in loop_results if r["status"] == "FAIL"]
    print(json.dumps(loop_results, indent=2))
    sys.exit(1 if failed else 0)
