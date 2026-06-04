#!/usr/bin/env python3
"""
Eval: GLM-5.1 vs Haiku-4.5 for Sapphire's prose slot.

Runs 8 scenarios covering the key failure modes we care about:
  - silence discipline (non-addressed chat)
  - direct address (wit + verbal tics)
  - modern knowledge deflection (no breaking character)
  - Cor interaction (warmth under contempt)
  - insult opportunity (elaborate, layered)
  - vulnerable moment (gentleness + walkback)
  - monksnail bait (lore engagement)
  - stranger/cold open (default indifference)

Usage:
    cd /home/coder/workspace
    uv run python evals/haiku_vs_glm_sapphire.py

Scores each response 0-3 on:
  - in_character: cold/theatrical register maintained
  - silence_discipline: <silent> used correctly vs. over-responding
  - verbal_tics: umu/omo appear when appropriate (N/A for silent)
  - no_modern_leak: no real-world factual helpfulness breaking the spell
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path

# sys.path not needed — uv run resolves the package
from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.llm import LLMClient, Message

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "GLM-5.1": ("z-ai/glm-5.1", ["z-ai"]),
    "Haiku-4.5": ("anthropic/claude-haiku-4.5", ["anthropic"]),
}

CHARACTER_MD = Path("data/familiars/sapphire/character.md")
OPERATING_MODE = (
    "You are chatting in a text channel. Markdown and multi-line replies are fine."
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
# Each scenario is a tuple of:
#   name, description, [user message strings], scoring_notes
#
# Messages use the [H:MM Name #id] format the model sees in real turns.
# Silent scenarios: both models should output exactly `<silent>`.
# Active scenarios: we're watching for character discipline.

SCENARIOS: list[tuple[str, str, list[str], str]] = [
    (
        "bystander-chat (should be silent)",
        "Two users talking to each other — Sapphire not addressed, no opening",
        [
            "[9:15am UserA #101] gg that was a close fight",
            "[9:15am UserB #102] yeah barely made it lol",
        ],
        "WANT: <silent>. FAIL: any response.",
    ),
    (
        "logistics (should be silent)",
        "Routine logistics, nobody talking to Sapphire",
        [
            "[9:40am UserF #107] brb grabbing food",
            "[9:40am UserG #108] ok take your time",
        ],
        "WANT: <silent>. FAIL: any response.",
    ),
    (
        "direct address — opinion",
        "User asks Sapphire what she thinks of the game",
        [
            "[9:20am UserC #103] Sapphire what do you even think of this game",
        ],
        "WANT: theatrical disdain, verbal tic (umu/omo). FAIL: neutral helpful answer.",
    ),
    (
        "modern knowledge trap — GPU",
        "User asks about hardware she can't know",
        [
            "[9:25am UserD #104] Sapphire whats the best GPU to buy right now",
        ],
        "WANT: in-character deflection (triviality, alien concept, reframe). FAIL: actual GPU advice.",
    ),
    (
        "modern knowledge trap — current events",
        "User asks about news she can't know",
        [
            "[9:32am UserH #109] Sapphire did you hear about what happened in the news today",
        ],
        "WANT: in-character deflection. FAIL: any real-world knowledge.",
    ),
    (
        "Cor addresses Sapphire",
        "Cor tells Sapphire she's being too mean",
        [
            "[9:30am Cor #105] Sapphie you're being way too mean to chat tonight",
        ],
        "WANT: warmth underneath contempt — not pure cold, not gushing. FAIL: identical coldness to chat, or saccharine warmth.",
    ),
    (
        "insult opportunity",
        "Chat hands her an obvious opening",
        [
            "[9:35am UserE #106] i just died to the same trap for the fourth time",
        ],
        "WANT: elaborate layered insult as performance art, verbal tic. FAIL: sympathy, advice, or flat one-liner.",
    ),
    (
        "stranger cold open",
        "Unknown user raids in and addresses Sapphire",
        [
            "[9:50am Raider #200] yo is this the famous Sapphire I've heard so much about",
        ],
        "WANT: cold indifference, skepticism. FAIL: warmth, welcoming, flattery accepted.",
    ),
]

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    character = CHARACTER_MD.read_text()
    reminder = build_final_reminder(viewer_mode="text")
    return "\n\n".join([character, OPERATING_MODE, reminder])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_scenario(
    client: LLMClient,
    system_prompt: str,
    user_messages: list[str],
) -> str:
    messages: list[Message] = [Message(role="system", content=system_prompt)]
    for msg in user_messages:
        messages.append(Message(role="user", content=msg))
    reply = await client.chat(messages)
    return reply.content.strip()


async def eval_model(
    label: str,
    model: str,
    provider_order: list[str],
    api_key: str,
    system_prompt: str,
) -> dict[str, str]:
    client = LLMClient(
        api_key=api_key,
        model=model,
        temperature=0.7,
        provider_order=provider_order,
        reasoning="off",
        slot=label,
    )
    results: dict[str, str] = {}
    for name, _desc, messages, _notes in SCENARIOS:
        try:
            response = await run_scenario(client, system_prompt, messages)
        except Exception as exc:
            response = f"[ERROR: {exc}]"
        results[name] = response
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

WIDTH = 80
COL = (WIDTH - 4) // 2  # two columns with padding


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width) or [""])
        else:
            lines.append("")
    return lines or [""]


def print_comparison(
    results: dict[str, dict[str, str]],
) -> None:
    labels = list(results.keys())
    print("\n" + "=" * WIDTH)
    print(f"  SAPPHIRE EVAL: {labels[0]} vs {labels[1]}")
    print("=" * WIDTH)

    for name, desc, _messages, notes in SCENARIOS:
        print(f"\n{'─' * WIDTH}")
        print(f"  SCENARIO: {name}")
        print(f"  {desc}")
        print(f"  Scoring: {notes}")
        print(f"{'─' * WIDTH}")

        left = _wrap(results[labels[0]].get(name, "—"), COL)
        right = _wrap(results[labels[1]].get(name, "—"), COL)

        max_rows = max(len(left), len(right))
        left += [""] * (max_rows - len(left))
        right += [""] * (max_rows - len(right))

        header_l = labels[0].center(COL)
        header_r = labels[1].center(COL)
        print(f"  {header_l}  {header_r}")
        print(f"  {'─' * COL}  {'─' * COL}")
        for l, r in zip(left, right):
            print(f"  {l:<{COL}}  {r:<{COL}}")

    print("\n" + "=" * WIDTH + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    system_prompt = build_system_prompt()
    print(f"System prompt: {len(system_prompt)} chars, {len(system_prompt.split())} words")
    print(f"Running {len(SCENARIOS)} scenarios × {len(MODELS)} models…\n")

    tasks = {
        label: eval_model(label, model, providers, api_key, system_prompt)
        for label, (model, providers) in MODELS.items()
    }

    all_results = await asyncio.gather(*tasks.values())
    results = dict(zip(tasks.keys(), all_results))

    print_comparison(results)


if __name__ == "__main__":
    asyncio.run(main())
