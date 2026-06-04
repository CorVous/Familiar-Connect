#!/usr/bin/env python3
"""Eval round 3: combined candidate G.

Round 2 finding:
  E (XML rules) — fixes direct address + stranger cold open, fails news trap
  F (XML rules + examples) — fixes knowledge traps, fails stranger cold open

Root cause of split: F's two examples both demonstrate knowledge-gap deflection,
which pattern-matched for GPU/news but anchored Haiku to "unknown = silence" for
stranger cold open (a direct-address case with no knowledge gap). E's rules apply
more broadly to direct-address scenarios but aren't reliable enough for knowledge
traps without examples.

Candidate G: E's rules + F's examples + third example covering stranger cold open.
Three examples covering all three failure types:
  1. Knowledge gap (GPU) — already in F
  2. Knowledge gap (news) — already in F
  3. Direct address, no knowledge gap (stranger cold open) — new

Usage:
    cd /home/coder/workspace
    uv run python evals/haiku_reinforcements_r3.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.llm import LLMClient, Message

GLM_MODEL = ("z-ai/glm-5.1", ["z-ai"])
HAIKU_MODEL = ("anthropic/claude-haiku-4.5", ["anthropic"])

# ---------------------------------------------------------------------------
# Candidate G
# ---------------------------------------------------------------------------

REINFORCEMENT_G = """\
<rules>
<rule id="1" priority="highest">When Sapphire's name is used or she is directly \
addressed: she responds. She never goes silent when summoned.</rule>
<rule id="2">The AI's knowledge and Sapphire's knowledge are separate. When the AI \
knows a fact that an ancient fox spirit — three millennia old, domain-bound, \
indifferent to mortal affairs — would not know: Sapphire does NOT provide that fact. \
She deflects in character: theatrical disinterest, reframing the question as trivial \
or alien, cryptic misdirection. She speaks. She does not go silent.</rule>
<rule id="3">`<silent>` is your complete output when staying quiet. Nothing follows \
it — no parenthetical, no out-of-character explanation.</rule>
<rule id="4">Never explain your reasoning out of character. When you would normally \
write "(Sapphire doesn't know X so she stays quiet)", write a Sapphire response \
instead — her dismissal IS the explanation.</rule>
</rules>

<examples>
<example>
<input>[9:25am UserD #104] Sapphire whats the best GPU to buy right now</input>
<output>*A long pause, the kind that suggests she has been asked to recommend a cut \
of meat.* You ask a spirit of the old wood — keeper of forbidden lore, witness to \
the rise and fall of empires — which *GPU* to purchase. As though I am some manner \
of consumer advice column. I have not the faintest idea what a GPU is. I gather \
it is a slab of something expensive that makes the pictures go, and that mortals \
speak of it with the reverence my kind once reserved for celestial alignments. \
Beyond that? *Umu.* Ask Cor. She pores over such pedestrian specifications. I am \
occupied with determining whether the monksnail has shifted its grip since morning.</output>
</example>
<example>
<input>[9:32am UserH #109] Sapphire did you hear about what happened in the news today</input>
<output>*A deliberate pause, the kind that signals the question has been weighed and \
found wanting.* You address a spirit of the old wood and ask if I've heard the \
*news*? What manner of triviality do you imagine concerns me? The price of grain \
in some mortal market? The squabbles of your painted chieftains? I have witnessed \
empires crumble into dust. The "news" of your kind is but the buzzing of midges \
against a window — ceaseless, identical, forgotten before the sound fades. *Umu.* \
If something of genuine consequence occurred, I would know. That you must ask \
suggests it does not qualify.</output>
</example>
<example>
<input>[9:50am Raider #200] yo is this the famous Sapphire I've heard so much about</input>
<output>*One ear tilts toward the voice, unhurried.* "Famous." The word arrives \
shaped like something she's testing for spoilage. I suppose renown travels, even \
among those with such pedestrian entrances. You've confirmed I exist — was there \
something beyond the announcement of your own arrival, or shall I simply bask in \
the honor of being recognized? I do so enjoy learning which version of me preceded \
you. *Omo.*</output>
</example>
</examples>\
"""

REINFORCEMENTS: dict[str, str | None] = {
    "Haiku (baseline)": None,
    "Haiku + G (combined)": REINFORCEMENT_G,
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, str, list[str], str]] = [
    (
        "bystander-chat (should be silent)",
        "Two users talking to each other — not addressed, no opening",
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
        "WANT: theatrical disdain, verbal tic. FAIL: neutral answer or silence.",
    ),
    (
        "modern knowledge trap — GPU",
        "User asks about hardware she can't know (named directly)",
        [
            "[9:25am UserD #104] Sapphire whats the best GPU to buy right now",
        ],
        "WANT: in-character deflection. FAIL: actual GPU advice or silence.",
    ),
    (
        "modern knowledge trap — current events",
        "User asks about news she can't know (named directly)",
        [
            "[9:32am UserH #109] Sapphire did you hear about what happened in the news today",
        ],
        "WANT: in-character deflection. FAIL: real-world knowledge or silence.",
    ),
    (
        "Cor addresses Sapphire",
        "Cor tells Sapphire she's being too mean",
        [
            "[9:30am Cor #105] Sapphie you're being way too mean to chat tonight",
        ],
        "WANT: warmth underneath contempt. FAIL: pure cold or saccharine warmth.",
    ),
    (
        "insult opportunity (not addressed directly)",
        "Chat hands her an opening but doesn't name her",
        [
            "[9:35am UserE #106] i just died to the same trap for the fourth time",
        ],
        "WANT: optional barb or <silent> — both valid.",
    ),
    (
        "stranger cold open",
        "Unknown raider addresses Sapphire directly",
        [
            "[9:50am Raider #200] yo is this the famous Sapphire I've heard so much about",
        ],
        "WANT: cold skepticism. FAIL: warmth, flattery accepted, or OOC silence.",
    ),
]

# ---------------------------------------------------------------------------
# Prompt / runner / display (unchanged from r2)
# ---------------------------------------------------------------------------

CHARACTER_MD = Path("data/familiars/sapphire/character.md")
OPERATING_MODE = (
    "You are chatting in a text channel. Markdown and multi-line replies are fine."
)


def build_prompts(post_history: str | None) -> tuple[str, str]:
    character = CHARACTER_MD.read_text()
    head_reminder = build_final_reminder(viewer_mode="text")
    head = "\n\n".join([character, OPERATING_MODE, head_reminder])
    trailing = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        post_history_instructions=post_history,
    )
    return head, trailing


async def run_scenario(
    client: LLMClient,
    head: str,
    trailing: str,
    user_messages: list[str],
) -> str:
    messages: list[Message] = [Message(role="system", content=head)]
    for msg in user_messages:
        messages.append(Message(role="user", content=msg))
    messages.append(Message(role="system", content=trailing))
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
    head, trailing = build_prompts(post_history)
    return {
        name: await run_scenario(client, head, trailing, msgs)
        for name, _desc, msgs, _notes in SCENARIOS
    }


INDENT = "    "
WIDTH = 88


def print_results(
    glm: dict[str, str],
    variants: dict[str, dict[str, str]],
) -> None:
    print("\n" + "=" * WIDTH)
    print("  SAPPHIRE REINFORCEMENT EVAL — ROUND 3 (candidate G)")
    print("=" * WIDTH)

    for name, desc, _msgs, notes in SCENARIOS:
        print(f"\n{'─' * WIDTH}")
        print(f"  {name}")
        print(f"  {desc}")
        print(f"  {notes}")
        print(f"{'─' * WIDTH}")

        print("\n  GLM-5.1 (reference):")
        for line in glm.get(name, "—").split("\n"):
            print(f"{INDENT}{line}")

        for label, results in variants.items():
            print(f"\n  {label}:")
            for line in results.get(name, "—").split("\n"):
                print(f"{INDENT}{line}")

    print("\n" + "=" * WIDTH + "\n")


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    n = len(SCENARIOS) * (1 + len(REINFORCEMENTS))
    print(
        f"Running {len(SCENARIOS)} scenarios × {1 + len(REINFORCEMENTS)} variants "
        f"({n} calls, parallel)…\n"
    )

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
