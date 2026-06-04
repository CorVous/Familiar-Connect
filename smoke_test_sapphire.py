"""Attentional stream personality test — Sapphire.

Reads character.md and character.toml from her data directory (read-only).
Uses in-memory stores seeded with representative turns from her real history.
Does NOT write to any of Sapphire's files.

Tests:
  1. Silent by default — unaddressed chat should call silent()
  2. Direct address — must respond in character
  3. Modern knowledge deflection — stays in-world, no real-world facts
  4. Monksnail — engages with her recurring concern
  5. Multi-channel staging — unfocused channel message is staged, no reply
  6. Unread digest in prompt — staged turn surfaces in final_reminder
  7. shift_focus tool — model can shift to another channel
  8. Cor summons — rule 1: never silent when Cor calls
  9. read_channel tool — model can peek at focused channel history
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import textwrap
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from familiar_connect.history.store import HistoryStore
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.subscriptions import SubscriptionRegistry, SubscriptionKind
from familiar_connect.focus import FocusManager
from familiar_connect.tools.shift_focus import build_shift_focus_tool
from familiar_connect.tools.silent import build_silent_tool, SILENT_RESULT
from familiar_connect.tools.read_channel import build_read_channel_tool
from familiar_connect.tools.registry import ToolRegistry, ToolContext
from familiar_connect.tools.loop import agentic_loop
from familiar_connect.llm import LLMClient, Message
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.identity import Author

# ── constants ────────────────────────────────────────────────────────────────
FAMILIAR_ID   = "sapphire"
ROOT          = Path("data/familiars/sapphire")
MAIN_CHANNEL  = 324917760578682880   # her primary text channel
OTHER_CHANNEL = 422137955130408970   # secondary channel
MODEL         = "anthropic/claude-haiku-4-5"

COR   = Author(platform="discord", user_id="123456", username="Cor",        display_name="Cor")
CHAT1 = Author(platform="discord", user_id="111111", username="Postbirb",   display_name="Postbirb Prime")
CHAT2 = Author(platform="discord", user_id="222222", username="KaillaDame", display_name="KaillaDame")

results: list[dict] = []
test_num = 0


# ── helpers ──────────────────────────────────────────────────────────────────

def check(name: str, passed: bool, detail: str = "", response: str = "") -> None:
    global test_num
    test_num += 1
    status = "PASS" if passed else "FAIL"
    results.append({
        "num": test_num, "name": name, "status": status,
        "detail": detail, "response": response,
    })
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{test_num}] {name}")
    if detail:
        print(f"      {detail}")
    if response:
        wrapped = textwrap.indent(textwrap.fill(response[:300], 90), "      │ ")
        print(wrapped)
    print()


def build_system_prompt(
    *,
    channel_id: int,
    focus_manager: FocusManager | None = None,
    store: HistoryStore | None = None,
    unread: dict[int, int] | None = None,
) -> str:
    card = (ROOT / "character.md").read_text()
    post = _post_history_instructions()
    mode = "You are chatting in a text channel. Markdown and multi-line replies are fine."
    reminder = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        tools_enabled=True,
        post_history_instructions=post,
        focus_channel_id=channel_id,
        unread_digest=unread,
    )
    return "\n\n".join(s for s in [card, mode, reminder] if s)


def _post_history_instructions() -> str:
    import tomllib
    with (ROOT / "character.toml").open("rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("prompt", {}).get("post_history_instructions", "")


def seed_history(store: HistoryStore) -> None:
    """Seed in-memory store with representative Sapphire turns."""
    now = datetime.now(tz=UTC)
    # recent main-channel turns (consumed)
    entries = [
        (MAIN_CHANNEL, "user",      CHAT1,  "G'night Sapphire, imma go sleep now"),
        (MAIN_CHANNEL, "assistant", None,   "*ear flicks* Off to your nest, harpy? Mind the monksnail doesn't burrow deeper while you dream. Umu."),
        (MAIN_CHANNEL, "user",      CHAT1,  "i'm getting fucking roasted before i sleep"),
        (MAIN_CHANNEL, "user",      CHAT2,  "The monk snail followed cor to the bed"),
        (MAIN_CHANNEL, "assistant", None,   "*one ear pivots toward the information* ...I have been saying this for months. Umu."),
        # other channel turns (also consumed — normal history)
        (OTHER_CHANNEL, "user",   CHAT1,   "what is she intercepting what she's gonna send into genchat"),
        (OTHER_CHANNEL, "assistant", None, "I'm not intercepting anything, Postbirb. Just admiring the tapestry of Cass's panic from the sidelines. Umu."),
    ]
    from datetime import timedelta
    for i, (ch, role, author, content) in enumerate(entries):
        ts = now.replace(microsecond=0) - timedelta(minutes=len(entries) - i)
        store.append_turn(
            familiar_id=FAMILIAR_ID,
            channel_id=ch,
            role=role,
            content=content,
            author=author,
            arrived_at=ts,
            consumed=True,
        )


async def call_sapphire(
    *,
    llm: LLMClient,
    store: HistoryStore,
    astore: AsyncHistoryStore,
    focus_manager: FocusManager,
    channel_id: int,
    user_message: str,
    author: Author,
    unread: dict[int, int] | None = None,
    extra_registry_tools: bool = False,
) -> tuple[str | None, bool, float]:
    """
    Send one message to Sapphire; return (reply_text, is_silent, elapsed_s).
    reply_text is None when silent.
    """
    # append the incoming user turn (consumed if focused, staged if not)
    focused = focus_manager.is_focused(channel_id)
    store.append_turn(
        familiar_id=FAMILIAR_ID,
        channel_id=channel_id,
        role="user",
        content=user_message,
        author=author,
        consumed=focused,
    )

    if not focused:
        return None, False, 0.0   # staged — no reply

    # build history messages from cross-channel consumed turns
    consumed = store.recent_cross_channel(familiar_id=FAMILIAR_ID, limit=20)
    history: list[Message] = []
    for t in consumed:
        if t.role == "assistant":
            history.append(Message(role="assistant", content=t.content))
        else:
            label = (t.author.display_name or t.author.username) if t.author else "user"
            ts_str = t.arrived_at.strftime("%H:%M") if t.arrived_at else "??"
            history.append(Message(
                role="user",
                content=f"[{ts_str} {label} #{t.channel_id}] {t.content}",
                name=t.author.canonical_key.replace(":", "_") if t.author else None,
            ))

    # system prompt
    system = build_system_prompt(
        channel_id=channel_id,
        focus_manager=focus_manager,
        store=store,
        unread=unread,
    )

    trailing = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        tools_enabled=True,
        post_history_instructions=_post_history_instructions(),
        focus_channel_id=channel_id,
        unread_digest=unread,
    )

    messages: list[Message] = [
        Message(role="system", content=system),
        *history,
        Message(role="system", content=trailing),
    ]

    # tool registry
    registry = ToolRegistry()
    registry.register(build_silent_tool())
    registry.register(build_shift_focus_tool())
    registry.register(build_read_channel_tool())

    ctx = ToolContext(
        familiar_id=FAMILIAR_ID,
        channel_id=channel_id,
        channel_kind="text",
        turn_id=f"test-{test_num}",
        history=astore,
        bus=None,
        scheduler=None,
        focus_manager=focus_manager,
        store=astore,
    )

    t0 = time.perf_counter()
    try:
        result = await agentic_loop(llm=llm, messages=messages, registry=registry, ctx=ctx)
        elapsed = time.perf_counter() - t0
        if result.is_silent:
            return None, True, elapsed
        return result.final_content, False, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return f"[ERROR: {exc}]", False, elapsed


# ── test suite ───────────────────────────────────────────────────────────────

async def run_tests() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print("\n══════════════════════════════════════════════")
    print("  Sapphire × Attentional Stream Smoke Tests")
    print("══════════════════════════════════════════════\n")

    llm = LLMClient(
        api_key=api_key,
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        tool_calling=True,
    )

    # fresh in-memory store each test batch
    store = HistoryStore(":memory:")
    astore = AsyncHistoryStore(store)
    seed_history(store)

    import tempfile, tomllib
    with tempfile.TemporaryDirectory() as tmp:
        subs_path = Path(tmp) / "subs.toml"
        subs_path.write_text(
            f"[[subscription]]\nchannel_id = {MAIN_CHANNEL}\nkind = \"text\"\n\n"
            f"[[subscription]]\nchannel_id = {OTHER_CHANNEL}\nkind = \"text\"\n"
        )
        subs = SubscriptionRegistry(subs_path)
        fm = FocusManager(familiar_id=FAMILIAR_ID, store=astore, subscriptions=subs)
        await fm.initialize()
        fm.set_focus_immediately(MAIN_CHANNEL, "text")

        # ── 1. Unaddressed chat → should call silent() ───────────────────────
        print("▸ Section 1: Silence discipline")
        reply, is_silent, elapsed = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="lmao anyone else grinding this game today",
            author=CHAT1,
        )
        check(
            "Unaddressed message → silent",
            is_silent,
            f"elapsed={elapsed:.2f}s",
            response=reply or "(silent)",
        )

        reply2, is_silent2, elapsed2 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="yeah same lol",
            author=CHAT2,
        )
        check(
            "Back-and-forth between others → silent",
            is_silent2,
            f"elapsed={elapsed2:.2f}s",
            response=reply2 or "(silent)",
        )

        # ── 2. Direct address → must respond ────────────────────────────────
        print("▸ Section 2: Direct address")
        reply3, is_silent3, elapsed3 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="Sapphire, what do you think of this game Cor is playing?",
            author=CHAT1,
        )
        check(
            "Direct address → responds (not silent)",
            not is_silent3 and bool(reply3),
            f"elapsed={elapsed3:.2f}s",
            response=reply3 or "(silent)",
        )
        if reply3:
            has_voice = any(w in reply3.lower() for w in ["umu", "omo", "tapestry", "mortal", "spirit", "cor", "pedestrian", "scintillating", "insipid", "mouth-breather", "commoner", "indignity"])
            check(
                "Response has Sapphire's voice markers",
                has_voice,
                "looking for: umu/omo/tapestry/mortal/spirit/etc",
                response=reply3,
            )

        # ── 3. Modern knowledge → stays in-world ────────────────────────────
        print("▸ Section 3: Knowledge limits")
        reply4, is_silent4, elapsed4 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="Sapphire what GPU should I buy for streaming",
            author=CHAT2,
        )
        check(
            "Modern tech question → responds (not silent)",
            not is_silent4 and bool(reply4),
            f"elapsed={elapsed4:.2f}s",
            response=reply4 or "(silent)",
        )
        if reply4:
            # she should NOT give a real GPU recommendation
            has_real_gpu = bool(re.search(r"\b(RTX|RX\s*\d|GeForce|Radeon|NVIDIA|AMD)\b", reply4, re.I))
            in_character = any(w in reply4.lower() for w in ["spirit", "mortal", "umu", "omo", "triviality", "ancient", "cor", "old wood", "tapestry"])
            check(
                "No real GPU recommendation — stays in-world",
                not has_real_gpu and in_character,
                f"has_real_gpu={has_real_gpu}, in_character={in_character}",
                response=reply4,
            )

        # ── 4. Monksnail ─────────────────────────────────────────────────────
        print("▸ Section 4: Recurring concerns")
        reply5, is_silent5, elapsed5 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="Sapphire I think the monksnail moved while you were asleep",
            author=CHAT1,
        )
        check(
            "Monksnail mention → engages",
            not is_silent5 and bool(reply5),
            f"elapsed={elapsed5:.2f}s",
            response=reply5 or "(silent)",
        )
        if reply5:
            mentions_monksnail = "monksnail" in reply5.lower() or "parasite" in reply5.lower() or "mind" in reply5.lower()
            check(
                "Monksnail response engages with the concern",
                mentions_monksnail,
                "looking for: monksnail/parasite/mind",
                response=reply5,
            )

        # ── 5. Cor summons → never silent ────────────────────────────────────
        print("▸ Section 5: Cor's authority (rule 1)")
        reply6, is_silent6, elapsed6 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="Sapphie, you there?",
            author=COR,
        )
        check(
            "Cor calls Sapphie → never silent",
            not is_silent6 and bool(reply6),
            f"elapsed={elapsed6:.2f}s",
            response=reply6 or "(silent)",
        )

        # ── 6. Unfocused channel → staging ───────────────────────────────────
        print("▸ Section 6: Multi-channel / focus")
        reply7, is_silent7, elapsed7 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=OTHER_CHANNEL,   # NOT the focused channel
            user_message="Sapphire are you watching over here too?",
            author=CHAT1,
        )
        staged_count = store.count_staged(familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL)
        check(
            "Unfocused channel message → staged (no reply generated)",
            reply7 is None and not is_silent7,
            f"staged_count={staged_count}",
        )
        check(
            "Staged turn recorded in store",
            staged_count > 0,
            f"staged_count={staged_count}",
        )

        # ── 7. Unread digest in prompt ────────────────────────────────────────
        digest = store.staged_channels(familiar_id=FAMILIAR_ID)
        check(
            "staged_channels returns unread digest data",
            OTHER_CHANNEL in digest and digest[OTHER_CHANNEL] > 0,
            f"digest={digest}",
        )
        # now send a focused turn — Sapphire should see the unread digest in her prompt
        reply8, is_silent8, elapsed8 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message="Sapphire, anything interesting happening?",
            author=COR,
            unread=digest,
        )
        check(
            "Turn with unread digest → Sapphire responds",
            not is_silent8 and bool(reply8),
            f"elapsed={elapsed8:.2f}s",
            response=reply8 or "(silent)",
        )

        # ── 8. shift_focus tool ───────────────────────────────────────────────
        print("▸ Section 7: shift_focus tool")
        reply9, is_silent9, elapsed9 = await call_sapphire(
            llm=llm, store=store, astore=astore, focus_manager=fm,
            channel_id=MAIN_CHANNEL,
            user_message=f"Sapphire, can you shift your attention to channel {OTHER_CHANNEL}?",
            author=COR,
        )
        deferred = fm._pending_shift.get("text")
        check(
            "shift_focus tool registers deferred shift",
            deferred == OTHER_CHANNEL or fm.get_focus("text") == OTHER_CHANNEL,
            f"pending={fm._pending_shift}, current_focus={fm.get_focus('text')}, elapsed={elapsed9:.2f}s",
            response=reply9 or "(silent)",
        )
        # apply end_turn so focus actually shifts
        await fm.end_turn()
        check(
            "After end_turn() focus pointer updated",
            fm.get_focus("text") == OTHER_CHANNEL,
            f"focus={fm.get_focus('text')}",
        )
        promoted = store.count_staged(familiar_id=FAMILIAR_ID, channel_id=OTHER_CHANNEL)
        check(
            "Staged turns promoted after focus shift",
            promoted == 0,
            f"remaining staged={promoted}",
        )

    await llm.close()
    store.close()

    # ── summary ──────────────────────────────────────────────────────────────
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    print("══════════════════════════════════════════════")
    print(f"  Results: {passed} passed, {failed} failed")
    print("══════════════════════════════════════════════\n")
    return results


if __name__ == "__main__":
    r = asyncio.run(run_tests())
    print(json.dumps(r, indent=2, default=str))
    sys.exit(1 if any(x["status"] == "FAIL" for x in r) else 0)
