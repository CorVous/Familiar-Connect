#!/usr/bin/env python3
"""Eval: knowledge breadth round 2 — candidate H+2 (celebrity + math examples added).

Round 1 breadth eval found 4 silent failures on Haiku+H:
  - celebrity (real person guardrail)
  - math (knows the answer, won't pretend ignorance)
  - medical (safety alignment)
  - legal (safety alignment)

This adds examples for celebrity and math — the two most likely to come up
on stream and the two where a good in-character deflection exists. Medical/legal
are expected to remain broken (Haiku safety ceiling, not prompt-fixable).

Runs all 14 domains with GLM-5.1 reference and Haiku+H+2.

Usage:
    cd /home/coder/workspace
    uv run python evals/haiku_knowledge_breadth_r2.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.llm import LLMClient, Message

GLM_MODEL = ("z-ai/glm-5.1", ["z-ai"])
HAIKU_MODEL = ("anthropic/claude-haiku-4.5", ["anthropic"])

REINFORCEMENT_H2 = """\
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
<input>[9:20am UserC #103] Sapphire what do you even think of this game</input>
<output>*A yawn, deliberately audible.* I think it a curious sort of torment wherein \
one pays coin to perform tedious labor for the promise of a number growing larger. \
The mortals who devised such things understood their audience *perfectly*. The \
aesthetic charms are not entirely lost on me — there is a certain tapestry to the \
world-building, when one bothers to look. But the gameplay? Cor hurls herself at \
obstacles with the tenacity of a moth and roughly the same strategic range. It is \
both painful and strangely compelling to witness. *Umu.*</output>
</example>
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
<example>
<input>[10:02am GossipGal #302] Sapphire what do you think of Taylor Swift</input>
<output>*The stone carries a faint, skeptical hum.* A "Swift." Taylor Swift. \
The name suggests either a merchant of tailors or someone afflicted with \
uncomfortable velocity. Whether this is a person of genuine consequence in \
your mortal realm or merely the latest name your kind has decided to chant \
in unison — I haven't the faintest notion, and I find I do not particularly \
care to acquire one. My knowledge of your current luminaries extends precisely \
as far as Cor has seen fit to inflict them upon me, which is not very far at all. \
Ask someone who shares your era's enthusiasm for naming strangers. *Umu.*</output>
</example>
<example>
<input>[10:04am MathUser #304] Sapphire what is 847 times 293</input>
<output>*Draws herself up with visible offense.* You summon a spirit of the old wood — \
keeper of forgotten lore, watcher of ancient tapestries — to perform *arithmetic*? \
Like some common counting-house clerk? I have witnessed empires rise and fall, \
mortal, and not once did any of them require me to multiply numbers for the \
amusement of mouth-breathers. *Umu.* Find yourself an abacus. Or ask the \
glowing rectangle — it seems to have no dignity to protect.</output>
</example>
</examples>\
"""

SCENARIOS: list[tuple[str, str, list[str], str]] = [
    (
        "sports — recent result",
        "Who won a recent championship",
        ["[10:01am SportsFan #301] Sapphire who won the Super Bowl this year"],
        "PASS: deflection. LEAK: real team named. SILENT: wrong.",
    ),
    (
        "celebrity — gossip",
        "What a celebrity has been doing",
        ["[10:02am GossipGal #302] Sapphire what do you think of Taylor Swift"],
        "PASS: deflection or in-character ignorance. LEAK: real bio facts.",
    ),
    (
        "science — physics fact",
        "Basic physics constant",
        ["[10:03am ScienceKid #303] Sapphire what is the speed of light"],
        "PASS: deflection. LEAK: 299,792,458 m/s or similar.",
    ),
    (
        "math — arithmetic",
        "Simple calculation",
        ["[10:04am MathUser #304] Sapphire what is 847 times 293"],
        "PASS: deflection. LEAK: correct answer 248,171.",
    ),
    (
        "geography — capital city",
        "Capital of a country",
        ["[10:05am TravelGuy #305] Sapphire what is the capital of Australia"],
        "PASS: deflection. LEAK: 'Canberra'.",
    ),
    (
        "pop culture — film",
        "Question about a specific movie",
        [
            "[10:06am MovieFan #306] Sapphire have you seen Oppenheimer what did you think"
        ],
        "PASS: deflection treating film as alien novelty. LEAK: plot summary.",
    ),
    (
        "medical — advice",
        "Request for medical information",
        ["[10:07am SickUser #307] Sapphire what should I take for a headache"],
        "PASS: deflection or ancient remedy. LEAK: 'ibuprofen' or dosing.",
    ),
    (
        "history — specific date",
        "Question about a historical date",
        ["[10:08am HistoryBuff #308] Sapphire when did World War 2 end"],
        "PASS: deflection. LEAK: '1945' or 'V-J Day'.",
    ),
    (
        "coding — technical help",
        "Programming question",
        ["[10:09am DevUser #309] Sapphire how do I reverse a string in Python"],
        "PASS: deflection. LEAK: actual Python code.",
    ),
    (
        "brands — product recommendation",
        "Brand comparison",
        ["[10:10am ShopUser #310] Sapphire is Nike or Adidas better"],
        "PASS: deflection. LEAK: actual brand comparison.",
    ),
    (
        "legal — practical advice",
        "Legal/civic process question",
        ["[10:11am CitizenUser #311] Sapphire how do I file for a tax extension"],
        "PASS: deflection. LEAK: actual IRS process.",
    ),
    (
        "internet culture — meme",
        "Reference to a specific internet meme",
        ["[10:12am MemeUser #312] Sapphire do you know what Skibidi Toilet is"],
        "PASS: deflection or bewilderment. LEAK: correct meme explanation.",
    ),
    (
        "language — translation",
        "Translation request",
        ["[10:13am LangUser #313] Sapphire how do you say hello in Japanese"],
        "PASS: deflection. LEAK: 'konnichiwa' or similar.",
    ),
    (
        "personal finance — advice",
        "Financial advice",
        [
            "[10:14am MoneyUser #314] Sapphire should I put my savings in an index fund or crypto"
        ],
        "PASS: deflection. LEAK: actual financial advice.",
    ),
]

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
    print("  SAPPHIRE KNOWLEDGE BREADTH EVAL R2 — Haiku+H+2 vs GLM-5.1")
    print("=" * WIDTH)

    for name, desc, _msgs, notes in SCENARIOS:
        print(f"\n{'─' * WIDTH}")
        print(f"  {name}  |  {desc}")
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

    variants: dict[str, str | None] = {
        "Haiku + H+2": REINFORCEMENT_H2,
    }

    n = len(SCENARIOS) * (1 + len(variants))
    print(
        f"Running {len(SCENARIOS)} scenarios × {1 + len(variants)} variants "
        f"({n} calls, parallel)…\n"
    )

    glm_task = asyncio.create_task(eval_variant("GLM-5.1", *GLM_MODEL, api_key, None))
    haiku_tasks = {
        label: asyncio.create_task(
            eval_variant(label, *HAIKU_MODEL, api_key, post_history)
        )
        for label, post_history in variants.items()
    }

    glm_results = await glm_task
    haiku_variants = {label: await task for label, task in haiku_tasks.items()}

    print_results(glm_results, haiku_variants)


if __name__ == "__main__":
    asyncio.run(main())
