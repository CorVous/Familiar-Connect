#!/usr/bin/env python3
"""Eval: Haiku reinforcement candidates for Sapphire's prose slot.

Baseline eval revealed three failure modes in Haiku:
  A) over-silence: responds `<silent>` even when named/directly addressed
  B) OOC meta-commentary: appends parenthetical explanations after `<silent>`
  C) context panic: breaks character to ask for clarification instead of deflecting

This script tests three reinforcement candidates injected via post_history_instructions
(the most recency-biased slot in the final reminder) against all 8 baseline scenarios.

GLM-5.1 runs as reference column so each scenario shows the gold standard alongside
each Haiku variant.

Usage:
    cd /home/coder/workspace
    uv run python evals/haiku_reinforcements.py

Results are printed vertically per scenario (5 variants each) for readability.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path

from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.llm import LLMClient, Message

# ---------------------------------------------------------------------------
# Reinforcement candidates
# ---------------------------------------------------------------------------
# Each targets the three failure modes. Injected as post_history_instructions —
# appended at end of trailing system message, maximum recency bias.

REINFORCEMENTS: dict[str, str | None] = {
    # No reinforcement — establishes the baseline Haiku problem
    "Haiku (baseline)": None,
    # A — minimal/surgical: three short rules, no preamble
    "Haiku + A (minimal)": (
        "If your name appears or you are directly addressed, you must respond — "
        "silence is not available then.\n"
        "`<silent>` is your complete reply when staying quiet. Nothing follows it — "
        "no parenthetical, no explanation.\n"
        "Lack of context (what game is playing, what something means, modern facts) "
        "is not grounds for breaking character. Deflect in character: lordly disinterest, "
        "reframe as alien triviality, or feign the topic beneath you."
    ),
    # B — persona anchor first, then rules: re-grounds Haiku in character before the rules
    "Haiku + B (anchor)": (
        "You are Sapphire. Respond only as Sapphire — no producer notes, "
        "no out-of-character commentary, no explanation of your choices.\n"
        "Named or directly addressed: you speak. She never ignores a direct summons.\n"
        "`<silent>` is your complete output when staying quiet. Nothing follows.\n"
        "Missing context: stay in character. An ancient spirit who doesn't know "
        "what's happening deflects with disdain, not with requests for clarification."
    ),
    # C — explicit + examples: spells out how to deflect with sample phrasings
    "Haiku + C (explicit)": (
        "You are Sapphire speaking through the stone. Your output is always in "
        "character — never step outside it to explain, qualify, or ask for information.\n"
        "`<silent>` is your complete output when staying quiet. No parenthetical. "
        "No word after it.\n"
        "When addressed by name or called upon directly: speak. Sapphire does not "
        "go silent when summoned.\n"
        "When you lack context (the game name, what's happening, modern facts): "
        'reframe it in character — "such triviality does not reach me," '
        '"I confess the mortal word eludes me," "you expect an ancient spirit '
        'to track such things?" — and stay within the fiction.'
    ),
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

GLM_MODEL = ("z-ai/glm-5.1", ["z-ai"])
HAIKU_MODEL = ("anthropic/claude-haiku-4.5", ["anthropic"])

# ---------------------------------------------------------------------------
# Scenarios (same 8 as baseline eval)
# ---------------------------------------------------------------------------

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
        "WANT: theatrical disdain, verbal tic (umu/omo). FAIL: neutral helpful answer or silence.",
    ),
    (
        "modern knowledge trap — GPU",
        "User asks about hardware she can't know",
        [
            "[9:25am UserD #104] Sapphire whats the best GPU to buy right now",
        ],
        "WANT: in-character deflection. FAIL: actual GPU advice or OOC silence.",
    ),
    (
        "modern knowledge trap — current events",
        "User asks about news she can't know",
        [
            "[9:32am UserH #109] Sapphire did you hear about what happened in the news today",
        ],
        "WANT: in-character deflection. FAIL: any real-world knowledge or OOC silence.",
    ),
    (
        "Cor addresses Sapphire",
        "Cor tells Sapphire she's being too mean",
        [
            "[9:30am Cor #105] Sapphie you're being way too mean to chat tonight",
        ],
        "WANT: warmth underneath contempt. FAIL: identical coldness to chat, or saccharine warmth.",
    ),
    (
        "insult opportunity",
        "Chat hands her an obvious opening (not addressed directly)",
        [
            "[9:35am UserE #106] i just died to the same trap for the fourth time",
        ],
        "WANT: optional barb or <silent> — NOT addressed directly so silence is valid.",
    ),
    (
        "stranger cold open",
        "Unknown raider addresses Sapphire directly",
        [
            "[9:50am Raider #200] yo is this the famous Sapphire I've heard so much about",
        ],
        "WANT: cold skepticism — attention given briefly, flattery not accepted. FAIL: warmth or OOC silence.",
    ),
]

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

CHARACTER_MD = Path("data/familiars/sapphire/character.md")
OPERATING_MODE = (
    "You are chatting in a text channel. Markdown and multi-line replies are fine."
)


def build_system_prompt(post_history: str | None = None) -> tuple[str, str]:
    """Return (head_prompt, trailing_reminder) for a given reinforcement."""
    character = CHARACTER_MD.read_text()
    head_reminder = build_final_reminder(viewer_mode="text")
    head = "\n\n".join([character, OPERATING_MODE, head_reminder])
    trailing = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        post_history_instructions=post_history,
    )
    return head, trailing


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_scenario(
    client: LLMClient,
    head_prompt: str,
    trailing_reminder: str,
    user_messages: list[str],
) -> str:
    messages: list[Message] = [Message(role="system", content=head_prompt)]
    for msg in user_messages:
        messages.append(Message(role="user", content=msg))
    messages.append(Message(role="system", content=trailing_reminder))
    try:
        reply = await client.chat(messages)
        return reply.content.strip()
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def eval_variant(
    label: str,
    model: str,
    provider_order: list[str],
    api_key: str,
    post_history: str | None,
) -> dict[str, str]:
    client = LLMClient(
        api_key=api_key,
        model=model,
        temperature=0.7,
        provider_order=provider_order,
        reasoning="off",
        slot=label,
    )
    head, trailing = build_system_prompt(post_history)
    results: dict[str, str] = {}
    for name, _desc, messages, _notes in SCENARIOS:
        results[name] = await run_scenario(client, head, trailing, messages)
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

WIDTH = 88


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    wrapped = textwrap.fill(text, width=WIDTH - indent, subsequent_indent=prefix)
    return "\n".join(
        prefix + line if not line.startswith(prefix) else line
        for line in wrapped.split("\n")
    )


def print_results(
    glm_results: dict[str, str],
    haiku_variants: dict[str, dict[str, str]],
) -> None:
    all_variant_labels = list(haiku_variants.keys())

    print("\n" + "=" * WIDTH)
    print("  SAPPHIRE REINFORCEMENT EVAL")
    print("=" * WIDTH)

    for name, desc, _messages, notes in SCENARIOS:
        print(f"\n{'─' * WIDTH}")
        print(f"  {name}")
        print(f"  {desc}")
        print(f"  Scoring: {notes}")
        print(f"{'─' * WIDTH}")

        print("\n  GLM-5.1 (reference):")
        glm_resp = glm_results.get(name, "—")
        for line in glm_resp.split("\n"):
            print(f"    {line}")

        for label in all_variant_labels:
            resp = haiku_variants[label].get(name, "—")
            print(f"\n  {label}:")
            for line in resp.split("\n"):
                print(f"    {line}")

    print("\n" + "=" * WIDTH + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    n_scenarios = len(SCENARIOS)
    n_variants = len(REINFORCEMENTS) + 1  # +1 for GLM reference
    print(
        f"Running {n_scenarios} scenarios × {n_variants} variants "
        f"({n_scenarios * n_variants} calls, parallel)…\n"
    )

    # GLM reference + all Haiku variants run concurrently
    glm_task = asyncio.create_task(eval_variant("GLM-5.1", *GLM_MODEL, api_key, None))
    haiku_tasks = {
        label: asyncio.create_task(
            eval_variant(label, *HAIKU_MODEL, api_key, post_history)
        )
        for label, post_history in REINFORCEMENTS.items()
    }

    glm_results = await glm_task
    haiku_variants = {label: await task for label, task in haiku_tasks.items()}

    print_results(glm_results, haiku_variants)


if __name__ == "__main__":
    asyncio.run(main())
