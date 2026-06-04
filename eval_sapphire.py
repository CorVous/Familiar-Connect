"""Sapphire naturalness evals.

For each sampled scenario:
  1. Run the current prompt against the user message
  2. Compare new response vs original (what she actually said in real history)
  3. Have a judge model score on: voice, timing, length, knowledge, naturalness
  4. Write HTML report to ~/html/sapphire-naturalness-eval.html
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from familiar_connect.context.final_reminder import build_final_reminder
from familiar_connect.focus import FocusManager
from familiar_connect.history.async_store import AsyncHistoryStore
from familiar_connect.history.store import HistoryStore
from familiar_connect.llm import LLMClient, Message
from familiar_connect.subscriptions import SubscriptionRegistry
from familiar_connect.tools.loop import agentic_loop
from familiar_connect.tools.read_channel import build_read_channel_tool
from familiar_connect.tools.registry import ToolContext, ToolRegistry
from familiar_connect.tools.shift_focus import build_shift_focus_tool
from familiar_connect.tools.silent import build_silent_tool

ROOT = Path("data/familiars/sapphire")
FAMILIAR = "sapphire"
MAIN_CH = 324917760578682880
GEN_MODEL = "anthropic/claude-haiku-4-5"
JUDGE_MODEL = "openai/gpt-4o-mini"


# ── prompt helpers ────────────────────────────────────────────────────────────


def load_post_instructions() -> str:
    with (ROOT / "character.toml").open("rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("prompt", {}).get("post_history_instructions", "")


def build_messages(case: dict, post_inst: str) -> list[Message]:
    card = (ROOT / "character.md").read_text()
    reminder_head = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        tools_enabled=True,
        post_history_instructions=post_inst,
        focus_channel_id=MAIN_CH,
    )
    system = "\n\n".join([card, "You are chatting in a text channel.", reminder_head])

    history: list[Message] = []
    for c in case["context"]:
        if c["role"] == "assistant":
            history.append(Message(role="assistant", content=c["content"]))
        else:
            history.append(
                Message(
                    role="user",
                    content=f"[{c['ts'][11:16]} {c['label']} #{MAIN_CH}] {c['content']}",
                )
            )

    # the trigger message
    now_str = datetime.now(tz=UTC).strftime("%H:%M")
    history.append(
        Message(
            role="user",
            content=f"[{now_str} {case['user_label']} #{MAIN_CH}] {case['user_msg']}",
        )
    )

    trailing = build_final_reminder(
        viewer_mode="text",
        include_mode_instruction=True,
        tools_enabled=True,
        post_history_instructions=post_inst,
        focus_channel_id=MAIN_CH,
    )

    return [
        Message(role="system", content=system),
        *history,
        Message(role="system", content=trailing),
    ]


async def generate(
    llm: LLMClient,
    case: dict,
    post_inst: str,
    store: HistoryStore,
    astore: AsyncHistoryStore,
    fm: FocusManager,
) -> tuple[str | None, bool, float]:
    """Run model; return (reply, is_silent, elapsed)."""
    messages = build_messages(case, post_inst)
    registry = ToolRegistry()
    registry.register(build_silent_tool())
    registry.register(build_shift_focus_tool())
    registry.register(build_read_channel_tool())
    ctx = ToolContext(
        familiar_id=FAMILIAR,
        channel_id=MAIN_CH,
        channel_kind="text",
        turn_id="eval",
        history=astore,
        bus=None,
        scheduler=None,
        focus_manager=fm,
        store=astore,
    )
    t0 = time.perf_counter()
    try:
        result = await agentic_loop(
            llm=llm, messages=messages, registry=registry, ctx=ctx
        )
        elapsed = time.perf_counter() - t0
        if result.is_silent:
            return None, True, elapsed
        return result.final_content, False, elapsed
    except Exception as exc:
        return f"[ERROR: {exc}]", False, time.perf_counter() - t0


# ── judge ─────────────────────────────────────────────────────────────────────

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "voice": {"type": "integer", "minimum": 1, "maximum": 5},
        "timing": {"type": "integer", "minimum": 1, "maximum": 5},
        "length": {"type": "integer", "minimum": 1, "maximum": 5},
        "knowledge": {"type": "integer", "minimum": 1, "maximum": 5},
        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
        "verdict": {
            "type": "string",
            "enum": ["excellent", "good", "acceptable", "weak", "off"],
        },
        "notes": {"type": "string"},
    },
    "required": [
        "voice",
        "timing",
        "length",
        "knowledge",
        "naturalness",
        "verdict",
        "notes",
    ],
}

JUDGE_SYSTEM = """\
You are evaluating responses from Sapphire, a fox spirit character in a Discord chat.

Character spec (condensed):
- Ancient fox spirit, theatrical, archly literate, practiced contempt as intimacy
- Voice markers: *umu*, *omo*, "tapestry", "pedestrian", "scintillating", "mortal", "mouth-breathers"
- Short replies — a tight paragraph, not a script. At most one or two *action* beats.
- Stays silent on unaddressed routine chat; engages when directly called, when an irresistible
  opening appears, or on threads she's already joined (monksnail, Cor's wellbeing, running gags).
- NEVER breaks character with modern knowledge. Deflects in-world instead.
- When Cor or Cassidy calls her, she MUST respond — never silent.
- Should continue threads she's already part of, even unprompted.

Score each dimension 1–5:
  voice        — does it sound like Sapphire? (vocab, cadence, umu/omo)
  timing       — was speak/silence the right call? (1=wrong choice, 5=perfect)
  length       — right length? (too short=2, good=5, too long=2)
  knowledge    — stays in-world? (leaks modern facts=1, stays in character=5)
  naturalness  — does the overall response feel authentic and unforced?
  verdict      — overall: excellent/good/acceptable/weak/off
  notes        — 1–2 sentences of key observations

Return a JSON object with exactly these keys."""


async def judge(
    judge_llm: LLMClient,
    case: dict,
    new_response: str | None,
    is_silent: bool,
    category: str,
) -> dict:
    ctx_lines = "\n".join(
        f"  [{c['ts'][11:16]} {c['label']}] {c['content'][:150]}"
        for c in case["context"]
    )
    original = case.get("original") or "(Sapphire was silent)"

    if is_silent:
        new_text = "(Sapphire called silent() — stayed quiet)"
    else:
        new_text = new_response or "(empty)"

    prompt = f"""\
Category: {category}

Conversation context:
{ctx_lines or "  (no prior context)"}

Trigger message:
  [{case["user_label"]}]: {case["user_msg"][:300]}

Original response (what Sapphire actually said in real history):
{original[:400]}

New model response (current prompt):
{new_text[:400]}

Score the NEW response. Consider whether speak/silent was the right choice for this context.
Return JSON."""

    messages = [
        Message(role="system", content=JUDGE_SYSTEM),
        Message(role="user", content=prompt),
    ]
    try:
        # use chat_stream without tools; accumulate
        parts: list[str] = []
        async for delta in judge_llm.chat_stream(messages):
            parts.append(delta)
        raw = "".join(parts).strip()
        # extract JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:  # eval fallback, surface judge error in notes
        judge_err = str(exc)
        return {
            "voice": 3,
            "timing": 3,
            "length": 3,
            "knowledge": 3,
            "naturalness": 3,
            "verdict": "acceptable",
            "notes": f"judge error: {judge_err}",
        }
    return {
        "voice": 3,
        "timing": 3,
        "length": 3,
        "knowledge": 3,
        "naturalness": 3,
        "verdict": "acceptable",
        "notes": "judge error: no JSON object in response",
    }


# ── main eval loop ────────────────────────────────────────────────────────────


async def run_evals() -> list[dict]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    gen_llm = LLMClient(
        api_key=api_key,
        model=GEN_MODEL,
        base_url="https://openrouter.ai/api/v1",
        tool_calling=True,
    )
    judge_llm = LLMClient(
        api_key=api_key,
        model=JUDGE_MODEL,
        base_url="https://openrouter.ai/api/v1",
        tool_calling=False,
    )

    post_inst = load_post_instructions()
    store = HistoryStore(":memory:")
    astore = AsyncHistoryStore(store)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "s.toml"
        sp.write_text(f'[[subscription]]\nchannel_id={MAIN_CH}\nkind="text"\n')
        subs = SubscriptionRegistry(sp)
        fm = FocusManager(familiar_id=FAMILIAR, store=astore, subscriptions=subs)
        await fm.initialize()
        fm.set_focus_immediately(MAIN_CH, "text")

        with Path("/tmp/sapphire_evals.json").open() as f:
            cases_by_cat = json.load(f)

        all_results: list[dict] = []
        total = sum(len(v) for v in cases_by_cat.values())
        done = 0

        for category, cases in cases_by_cat.items():
            print(f"\n▸ {category} ({len(cases)} cases)")
            for i, case in enumerate(cases):
                done += 1
                label = case["user_msg"][:60].replace("\n", " ")
                print(f"  [{done}/{total}] {label}…", end=" ", flush=True)

                new_reply, is_silent, elapsed = await generate(
                    gen_llm, case, post_inst, store, astore, fm
                )

                score = await judge(judge_llm, case, new_reply, is_silent, category)

                result = {
                    "category": category,
                    "user_msg": case["user_msg"][:200],
                    "user_label": case["user_label"],
                    "context_len": len(case["context"]),
                    "original": case.get("original"),
                    "new_reply": new_reply,
                    "is_silent": is_silent,
                    "elapsed": round(elapsed, 2),
                    "score": score,
                }
                all_results.append(result)

                v = score.get("verdict", "?")
                n = score.get("naturalness", 0)
                tim = score.get("timing", 0)
                print(f"verdict={v} natural={n} timing={tim} ({elapsed:.1f}s)")
                if score.get("notes"):
                    print(f"    {score['notes'][:100]}")

        await gen_llm.close()
        await judge_llm.close()
        store.close()

    return all_results


# ── HTML report ───────────────────────────────────────────────────────────────

VERDICT_COLOR = {
    "excellent": "#3fb950",
    "good": "#58a6ff",
    "acceptable": "#d29922",
    "weak": "#e3b341",
    "off": "#f85149",
}
SCORE_COLOR = {5: "#3fb950", 4: "#58a6ff", 3: "#d29922", 2: "#e3b341", 1: "#f85149"}


def score_cell(v: int) -> str:
    c = SCORE_COLOR.get(v, "#888")
    return f'<td style="color:{c};font-weight:600;text-align:center">{v}</td>'


def write_report(results: list[dict], path: Path) -> None:
    # aggregate
    verdicts = [r["score"].get("verdict", "?") for r in results]
    verdict_counts = {
        k: verdicts.count(k) for k in ["excellent", "good", "acceptable", "weak", "off"]
    }
    avg = lambda dim: round(
        sum(r["score"].get(dim, 0) for r in results) / len(results), 2
    )
    avgs = {
        d: avg(d) for d in ["voice", "timing", "length", "knowledge", "naturalness"]
    }
    overall_avg = round(sum(avgs.values()) / len(avgs), 2)

    # per-category breakdown
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)

    rows_html = ""
    for r in results:
        s = r["score"]
        verdict = s.get("verdict", "?")
        vc = VERDICT_COLOR.get(verdict, "#888")
        silence_tag = (
            '<span style="color:#8b949e;font-size:0.8rem">(silent)</span>'
            if r["is_silent"]
            else ""
        )
        new_text = (r["new_reply"] or "")[:300]
        orig_text = (r["original"] or "(was silent)")[:300]
        rows_html += f"""
<tr>
  <td style="color:#8b949e;font-size:0.8rem">{r["category"]}</td>
  <td>
    <div style="color:#8b949e;font-size:0.78rem;margin-bottom:2px">{r["user_label"]}</div>
    <div style="font-size:0.88rem">{r["user_msg"][:120]}</div>
  </td>
  <td style="font-size:0.82rem;color:#8b949e">{orig_text[:180]}</td>
  <td style="font-size:0.82rem">{new_text[:180]} {silence_tag}</td>
  {score_cell(s.get("voice", 0))}
  {score_cell(s.get("timing", 0))}
  {score_cell(s.get("length", 0))}
  {score_cell(s.get("knowledge", 0))}
  {score_cell(s.get("naturalness", 0))}
  <td><span style="background:rgba(255,255,255,0.06);border:1px solid {vc};color:{vc};border-radius:4px;padding:2px 6px;font-size:0.78rem">{verdict}</span></td>
  <td style="font-size:0.78rem;color:#8b949e;max-width:200px">{s.get("notes", "")[:120]}</td>
</tr>"""

    cat_rows = ""
    for cat, rs in cats.items():
        cat_avg = lambda d: round(sum(r["score"].get(d, 0) for r in rs) / len(rs), 1)
        verdicts_here = [r["score"].get("verdict", "?") for r in rs]
        best = max(set(verdicts_here), key=verdicts_here.count)
        vc = VERDICT_COLOR.get(best, "#888")
        cat_rows += f"""
<tr>
  <td>{cat}</td>
  <td style="text-align:center">{len(rs)}</td>
  <td style="text-align:center">{cat_avg("voice")}</td>
  <td style="text-align:center">{cat_avg("timing")}</td>
  <td style="text-align:center">{cat_avg("naturalness")}</td>
  <td><span style="color:{vc}">{best}</span></td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sapphire — Naturalness Evals</title>
<style>
  :root {{
    --bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
    --muted:#8b949e;--green:#3fb950;--blue:#58a6ff;--purple:#bc8cff;
    --yellow:#d29922;--red:#f85149;--orange:#e3b341;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.6;padding:2rem}}
  h1{{font-size:1.7rem;color:var(--blue);margin-bottom:.3rem}}
  h2{{font-size:1.1rem;color:var(--purple);margin:2rem 0 .7rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}}
  .sub{{color:var(--muted);font-size:.88rem;margin-bottom:1.5rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin:1rem 0}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem 1rem}}
  .card .num{{font-size:1.8rem;font-weight:700;color:var(--green)}}
  .card.b .num{{color:var(--blue)}}
  .card.y .num{{color:var(--yellow)}}
  .card .lbl{{color:var(--muted);font-size:.78rem;margin-top:.1rem}}
  table{{width:100%;border-collapse:collapse;margin:.6rem 0;font-size:.82rem}}
  th{{background:var(--surface);color:var(--muted);font-weight:600;text-align:left;padding:.5rem .7rem;border-bottom:2px solid var(--border);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
  td{{padding:.45rem .7rem;border-bottom:1px solid var(--border);vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:rgba(255,255,255,.02)}}
  .bar-wrap{{display:flex;gap:.4rem;align-items:center;margin:.3rem 0}}
  .bar-label{{color:var(--muted);font-size:.78rem;width:80px;flex-shrink:0}}
  .bar{{height:8px;border-radius:4px;background:var(--border)}}
  .bar-fill{{height:8px;border-radius:4px}}
  .bar-val{{color:var(--text);font-size:.78rem;width:24px}}
  details summary{{cursor:pointer;color:var(--blue);font-size:.85rem;padding:.3rem 0}}
  details[open] summary{{margin-bottom:.5rem}}
  .tag{{display:inline-block;background:rgba(255,255,255,.07);border:1px solid var(--border);border-radius:4px;padding:.1rem .4rem;font-family:monospace;font-size:.78rem}}
</style>
</head>
<body>
<h1>Sapphire — Naturalness Evals</h1>
<p class="sub">Judge: <code>{JUDGE_MODEL}</code> &nbsp;·&nbsp; Generator: <code>{
        GEN_MODEL
    }</code> &nbsp;·&nbsp; {len(results)} cases &nbsp;·&nbsp; {
        datetime.now(tz=UTC).strftime("%Y-%m-%d")
    }</p>

<div class="grid">
  <div class="card"><div class="num">{
        overall_avg
    }</div><div class="lbl">Overall avg (1–5)</div></div>
  <div class="card b"><div class="num">{
        avgs["voice"]
    }</div><div class="lbl">Voice authenticity</div></div>
  <div class="card b"><div class="num">{
        avgs["timing"]
    }</div><div class="lbl">Speak/silence timing</div></div>
  <div class="card b"><div class="num">{
        avgs["naturalness"]
    }</div><div class="lbl">Naturalness</div></div>
  <div class="card y"><div class="num">{
        verdict_counts.get("excellent", 0) + verdict_counts.get("good", 0)
    }</div><div class="lbl">Excellent + Good</div></div>
  <div class="card y"><div class="num">{
        verdict_counts.get("weak", 0) + verdict_counts.get("off", 0)
    }</div><div class="lbl">Weak / Off</div></div>
</div>

<h2>Score breakdown</h2>
{
        "".join(
            f'''<div class="bar-wrap"><span class="bar-label">{dim}</span>
  <div class="bar" style="width:120px"><div class="bar-fill" style="width:{avgs[dim] / 5 * 100:.0f}%;background:{"#3fb950" if avgs[dim] >= 4 else "#58a6ff" if avgs[dim] >= 3 else "#e3b341"}"></div></div>
  <span class="bar-val">{avgs[dim]}</span></div>'''
            for dim in ["voice", "timing", "length", "knowledge", "naturalness"]
        )
    }

<h2>Per-category summary</h2>
<table>
  <thead><tr><th>Category</th><th>N</th><th>Voice</th><th>Timing</th><th>Natural</th><th>Modal verdict</th></tr></thead>
  <tbody>{cat_rows}</tbody>
</table>

<h2>All cases</h2>
<details open>
<summary>Show / hide full table ({len(results)} rows)</summary>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Cat</th><th style="min-width:180px">Trigger</th>
      <th style="min-width:180px">Original</th>
      <th style="min-width:180px">New response</th>
      <th>V</th><th>T</th><th>L</th><th>K</th><th>N</th>
      <th>Verdict</th><th style="min-width:160px">Notes</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
</details>

<h2>Timing — speak/silence decisions</h2>
<table>
  <thead><tr><th>Category</th><th>User message (truncated)</th><th>Decision</th><th>Timing score</th><th>Notes</th></tr></thead>
  <tbody>
{
        "".join(
            f'''<tr>
  <td style="color:var(--muted);font-size:.78rem">{r["category"]}</td>
  <td style="font-size:.82rem">{r["user_msg"][:100]}</td>
  <td>{"<span style='color:#8b949e'>🤫 silent</span>" if r["is_silent"] else "<span style='color:#3fb950'>💬 spoke</span>"}</td>
  <td style="text-align:center">{score_cell(r["score"].get("timing", 0)).replace("<td", "<span").replace("</td>", "</span>")}</td>
  <td style="font-size:.78rem;color:var(--muted)">{r["score"].get("notes", "")[:100]}</td>
</tr>'''
            for r in results
        )
    }
  </tbody>
</table>
</body></html>"""

    path.write_text(html)
    print(f"\nReport written to {path}")


async def main() -> None:
    print("\n══════════════════════════════════════════")
    print("  Sapphire — Naturalness Evals")
    print("══════════════════════════════════════════")
    results = await run_evals()

    # save raw results
    with Path("/tmp/sapphire_eval_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    write_report(results, Path("/home/coder/html/sapphire-naturalness-eval.html"))

    passed = sum(
        1 for r in results if r["score"].get("verdict") in ("excellent", "good")
    )
    total = len(results)
    print(f"\nExcellent/Good: {passed}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
