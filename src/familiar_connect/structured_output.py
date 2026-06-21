"""Structured-output coercion — parse possibly-fenced LLM JSON, tolerate garbage.

Chat-tuned models wrap JSON in ``` / ```json fences and pad it with
prose. Three call sites already each reimplement the same "strip the
fence, pull the first balanced object/array, ``json.loads``, and
degrade to empty on any failure" idiom:

  * :func:`familiar_connect.sleep.hygiene.parse_actions`
  * :func:`familiar_connect.sleep.dream._extract_object`
  * :func:`familiar_connect.processors.fact_extractor._parse_facts`

This module is the single authoritative version of that idiom (the
Rule-of-Three case), plus the two id/key coercers those sites share.
Every function here DEGRADES on bad input and NEVER raises — a model
that fumbles its JSON must not crash the worker reading it.

Stdlib only by design: no LLM-client coupling, just the raw reply str.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Greedy + DOTALL: span from the first ``{``/``[`` to the last matching
# bracket, so a multi-line object survives. Mirrors the per-site regexes
# (``\{.*\}`` / ``\[.*\]``) the three call sites use today.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass(frozen=True)
class JsonResult:
    """Outcome of a tolerant parse.

    ``parsed_ok`` is the success signal; on failure ``value`` is
    ``None``. A caller wanting an empty container on failure reads e.g.
    ``result.value or []`` — the boolean keeps "model fumbled the JSON"
    distinguishable from "model returned an empty object", which the
    hygiene pass surfaces as an audit note.
    """

    value: Any = None
    parsed_ok: bool = False


def coerce_json(reply: str) -> JsonResult:
    """Parse a possibly-fenced LLM reply into JSON; never raise.

    Strips ``` / ```json fences, extracts the first balanced ``{...}``
    object or ``[...]`` array (whichever appears first), and
    ``json.loads`` it. Any failure — empty reply, no JSON, malformed
    JSON — degrades to ``JsonResult(value=None, parsed_ok=False)``.
    """
    if not reply or not reply.strip():
        return JsonResult()
    cleaned = _FENCE_RE.sub("", reply).strip()
    blob = _first_json_blob(cleaned)
    try:
        value = json.loads(blob)
    except ValueError:
        return JsonResult()
    return JsonResult(value=value, parsed_ok=True)


def _first_json_blob(cleaned: str) -> str:
    """Whichever of the first object / first array starts earlier.

    A reply that is bare JSON (no surrounding prose) has no match and is
    handed through verbatim, letting ``json.loads`` accept scalars and
    pre-trimmed payloads exactly as the per-site code does.
    """
    obj = _JSON_OBJECT_RE.search(cleaned)
    arr = _JSON_ARRAY_RE.search(cleaned)
    if obj and arr:
        return (obj if obj.start() <= arr.start() else arr).group(0)
    match = obj or arr
    return match.group(0) if match else cleaned


def coerce_positive_int_list(raw: Any) -> list[int]:  # noqa: ANN401 — arbitrary parsed JSON
    """Distinct positive ints from a JSON value, order-preserving.

    Bools are rejected outright (``True == 1`` must not slip through as
    the int 1); numeric strings are accepted; non-positive values and
    duplicates are dropped. Anything not a list degrades to ``[]``.
    """
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            val = item
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            val = int(item.strip())
        else:
            continue
        if val > 0 and val not in out:
            out.append(val)
    return out


def coerce_str_list(raw: Any) -> list[str]:  # noqa: ANN401 — arbitrary parsed JSON
    """Distinct non-empty strings from a JSON value, order-preserving.

    Blank / whitespace-only strings and non-string items are dropped;
    duplicates collapse to the first occurrence. Anything not a list
    degrades to ``[]``.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() and item not in out:
            out.append(item)
    return out
