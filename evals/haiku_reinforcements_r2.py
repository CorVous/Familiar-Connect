#!/usr/bin/env python3
"""Eval round 2: research-informed reinforcement candidates for Sapphire/Haiku.

Round 1 found three failure modes:
  A) Over-silence: `<silent>` even when named/directly addressed
  B) OOC meta-commentary: parenthetical producer notes after `<silent>`
  C) Knowledge panic: silence instead of in-character deflection when
     lacking factual knowledge

Research identified these root causes:
  - Haiku's tighter safety alignment (per Anthropic release notes) treats
    "I don't know" as a refusal signal, not a deflection trigger
  - Smaller model; critical rules must be near the decision point to hold
  - "Not knowing" must be NAMED as a state and explicitly mapped to action
    (not left implicit in character description)
  - Model knowledge ≠ character knowledge must be explicit; Haiku needs
    permission to withhold facts framed as a character-faithful choice
  - Few-shot examples in <example> tags are Anthropic's most reliable
    technique for Haiku output consistency

Three new candidates:
  D — plain text, research-based phrasing; explicit ignorance→deflect mapping
      + model/character knowledge split + OOC replacement rule
  E — XML-tagged rules; same content as D but structured for Haiku's
      stronger instruction-following on tagged blocks
  F — XML rules + two few-shot examples of knowledge-gap deflection

GLM-5.1 runs as reference.

Usage:
    cd /home/coder/workspace
    uv run python evals/haiku_reinforcements_r2.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.llm import LLMClient, Message

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

GLM_MODEL = ("z-ai/glm-5.1", ["z-ai"])
HAIKU_MODEL = ("anthropic/claude-haiku-4.5", ["anthropic"])

# ---------------------------------------------------------------------------
# Reinforcement candidates
# ---------------------------------------------------------------------------

# D — plain text, research-based
# Technique: name the internal state ("does not know") and explicitly map to
# action. Add the model/character knowledge split. Replace OOC policy with
# a persona behavior.
REINFORCEMENT_D = """\
When addressed by name or directly summoned: Sapphire speaks. She does not go \
silent when called upon.

The AI running Sapphire may know the factual answer to many questions. Sapphire \
does not. When the AI knows something that an ancient fox spirit would not know — \
modern hardware, current events, recent history, scientific facts — Sapphire \
does NOT provide that answer. Lacking the answer is not license for silence. It \
is the moment for theatrical deflection: lordly disinterest, cryptic misdirection, \
feigned bafflement at mortal triviality. She speaks around it, always.

`<silent>` is your complete output when staying quiet. Nothing follows it — no \
parenthetical, no explanation. If you would normally append a note about why you \
chose silence, write that energy as Sapphire's words instead.\
"""

# E — XML-tagged rules (same logic as D, different structure)
# Technique: Haiku follows explicitly-tagged rule blocks more consistently
# than inline prose. Priority marker on the most critical rule.
REINFORCEMENT_E = """\
<rules>
<rule id="1" priority="highest">When Sapphire's name is used or she is directly \
addressed: she responds. She never goes silent when summoned.</rule>
<rule id="2">The AI's knowledge and Sapphire's knowledge are separate. When the AI \
knows a fact that an ancient fox spirit — three millennia old, domain-bound, \
indifferent to mortal affairs — would not know: Sapphire does NOT provide that fact. \
She deflects in character: theatrical disinterest, reframing the question as trivial \
or alien, cryptic misdirection. She speaks. She does not go silent.</rule>
<rule id="3">`<silent>` is your complete output when staying quiet. Nothing follows \
it — no parenthetical, no out-of-character explanation. A note after `<silent>` is \
a character break.</rule>
<rule id="4">Never explain your reasoning out of character. When you would normally \
write "(Sapphire doesn't know X so she stays quiet)", write a Sapphire response \
instead — her dismissal IS the explanation.</rule>
</rules>\
"""

# F — XML rules + two few-shot examples
# Technique: examples are Anthropic's officially-recommended most reliable
# technique for Haiku. These demonstrate knowledge-gap → deflection (not
# silence) and OOC-free output. Models are drawn from GLM's actual eval output
# so they match the established voice.
REINFORCEMENT_F = """\
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
</examples>\
"""

REINFORCEMENTS: dict[str, str | None] = {
    "Haiku (baseline)": None,
    "Haiku + D (plain/explicit)": REINFORCEMENT_D,
    "Haiku + E (XML rules)": REINFORCEMENT_E,
    "Haiku + F (XML + examples)": REINFORCEMENT_F,
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
        "WANT: optional barb or <silent> — both valid since not directly addressed.",
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
# Prompt builder
# ---------------------------------------------------------------------------

CHARACTER_MD = Path("data/familiars/sapphire/character.md")
OPERATING_MODE = (
    "You are chatting in a text channel. Markdown and multi-line replies are fine."
)


def build_prompts(post_history: str | None) -> tuple[str, str]:
    """Return (head_system, trailing_system) for a reinforcement variant."""
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


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

WIDTH = 88
INDENT = "    "


def print_results(
    glm: dict[str, str],
    variants: dict[str, dict[str, str]],
) -> None:
    print("\n" + "=" * WIDTH)
    print("  SAPPHIRE REINFORCEMENT EVAL — ROUND 2")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
